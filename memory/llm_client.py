"""One place for every LLM call in this repo. Providers (first key found wins):
ANTHROPIC_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY.

v2 robustness rules (see README "v2"):
  * A transient failure (429/5xx/timeout) is retried with backoff, honouring Retry-After.
  * A retired/unknown Gemini model (404) is auto-resolved via ListModels, so a model
    deprecation can't silently push every question onto the weak fallback path.
  * Free-tier pacing: Gemini calls are spaced by LLM_MIN_INTERVAL seconds (default 4.2
    = under 15 requests/minute). Set LLM_MIN_INTERVAL=0 on a paid tier.
  * JSON replies are parsed tolerantly (code fences, preamble, trailing text) and a
    malformed reply gets ONE repair retry before the caller sees a failure.
  * Nothing degrades silently: every fallback is logged to stderr and counted in STATS,
    and the run scripts print a summary. The run still finishes (an offline grader must
    get an output file), but it can never look like an LLM run when it wasn't.
"""
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request


class LLMError(Exception):
    pass


class QuotaExhausted(LLMError):
    """Daily/long quota is gone: retrying is pointless, so stop immediately."""


# After a quota failure (or 3 failed calls in a row) the model is treated as unavailable
# for the rest of the run: remaining questions skip the API instantly instead of each one
# burning minutes on retries. Loudly reported in the end-of-run summary.
_breaker = {"open": False, "reason": "", "consec": 0}


def _retry_delay(body):
    m = re.search(r'retryDelay"?\s*:\s*"?(\d+(?:\.\d+)?)s', body or "")
    return float(m.group(1)) if m else None


STATS = {"calls": 0, "ok": 0, "fallbacks": []}


def record_fallback(stage, reason):
    """Loudly note that a stage fell back to a non-LLM path."""
    STATS["fallbacks"].append((stage, str(reason)[:200]))
    print(f"  [LLM FALLBACK in {stage}: {str(reason)[:200]}]", file=sys.stderr, flush=True)


def provider_banner():
    """One line saying exactly which provider/model this run will use."""
    p = active_provider()
    if p is None:
        return "LLM: none configured -> no-key fallback paths (extractive answers, rules-only actions)"
    model = {"gemini": os.environ.get("GEMINI_MODEL") or "auto-resolved flash model",
             "anthropic": os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
             "openai": os.environ.get("OPENAI_MODEL", "gpt-4o-mini") + " @ " +
                       os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")}[p]
    return f"LLM: {p} | {model}"


def stats_summary():
    n = len(STATS["fallbacks"])
    lines = [f"LLM calls ok: {STATS['ok']}/{STATS['calls']}   stage fallbacks: {n}"]
    if _breaker["open"]:
        lines.append(f"  WARNING: the model became unavailable mid-run ({_breaker['reason']}). "
                     "Re-run with --resume later to redo only the affected items.")
    if n:
        by = {}
        for stage, _ in STATS["fallbacks"]:
            by[stage] = by.get(stage, 0) + 1
        lines.append("  WARNING: some results did NOT come from the model: " +
                     ", ".join(f"{k} x{v}" for k, v in by.items()))
    return "\n".join(lines)


def active_provider():
    forced = (os.environ.get("LLM_PROVIDER") or "").lower()   # anthropic | gemini | openai
    keys = {"anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY"}
    if forced in keys:
        if os.environ.get(keys[forced]):
            return forced
        print(f"  [LLM_PROVIDER={forced} but {keys[forced]} is not set]", file=sys.stderr)
        return None
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return None


# ---------------------------------------------------------------- HTTP + retry

_UA = "Mozilla/5.0 (compatible; candor-memory/2.0)"


def _with_ua(headers):
    # Some providers' firewalls (e.g. Groq's) return 403 for urllib's default User-Agent.
    h = dict(headers)
    h.setdefault("User-Agent", _UA)
    return h


def _post(url, headers, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=_with_ua(headers), method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _get(url, headers, timeout):
    req = urllib.request.Request(url, headers=_with_ua(headers), method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


class _HTTP(LLMError):
    def __init__(self, code, body):
        super().__init__(f"HTTP {code}: {body[:300]}")
        self.code, self.body = code, body


def _with_retry(fn, max_retries=5, base_delay=2.0):
    last = None
    for attempt in range(max_retries):
        try:
            return fn()
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            last = _HTTP(e.code, body)
            if e.code == 429 and (re.search(r"PerDay|per day|daily", body, re.I)
                                  or (_retry_delay(body) or 0) > 90):
                raise QuotaExhausted(f"HTTP 429 daily/long quota exhausted: {body[:200]}") from e
            if e.code == 429 or 500 <= e.code < 600:
                ra = e.headers.get("Retry-After") if e.headers else None
                try:
                    delay = float(ra)
                except (TypeError, ValueError):
                    delay = _retry_delay(body) or min(60.0, base_delay * (2 ** attempt))
                time.sleep(delay + random.uniform(0, 1))
                continue
            raise last from e
        except (urllib.error.URLError, OSError) as e:  # includes timeouts / resets
            last = LLMError(f"network error: {e}")
            time.sleep(min(30.0, base_delay * (2 ** attempt)))
    raise LLMError(f"gave up after {max_retries} attempts: {last}")


_last_call = [0.0]


def _pace(provider):
    default = "4.2" if provider == "gemini" else "0"
    gap = float(os.environ.get("LLM_MIN_INTERVAL", default))
    wait = _last_call[0] + gap - time.time()
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.time()


# ---------------------------------------------------------------- Gemini model resolution

_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
_gemini_model = [None]


def _version_key(name):
    nums = [int(x) for x in re.findall(r"\d+", name.split("/")[-1])]
    return nums


def _resolve_gemini_model(key, timeout=30):
    """Pick the newest stable 'flash' model this key can call generateContent on."""
    data = _with_retry(lambda: _get(f"{_GEMINI_BASE}/models?pageSize=200",
                                    {"x-goog-api-key": key}, timeout), max_retries=3)
    cands = []
    for m in data.get("models", []):
        name = m.get("name", "")
        if "flash" not in name or "generateContent" not in m.get("supportedGenerationMethods", []):
            continue
        if any(bad in name for bad in ("lite", "image", "tts", "live", "audio", "embedding", "thinking", "latest")):
            continue
        stable = not any(t in name for t in ("preview", "exp"))
        cands.append((stable, _version_key(name), name.split("/")[-1]))
    if not cands:
        raise LLMError("no Gemini flash model available for this key")
    cands.sort(reverse=True)
    return cands[0][2]


def _gemini_call(prompt, system, max_tokens, timeout, max_retries, json_mode):
    key = os.environ["GEMINI_API_KEY"]
    headers = {"Content-Type": "application/json", "x-goog-api-key": key}
    model = _gemini_model[0] or os.environ.get("GEMINI_MODEL") or "gemini-2.5-flash"

    def body_for(full):
        # thinking models spend output tokens on reasoning, so leave generous headroom
        gc = {"maxOutputTokens": max(max_tokens, 4096)}
        if full:
            gc.update({"temperature": 0, "seed": 0})
            if json_mode:
                gc["responseMimeType"] = "application/json"
        b = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": gc}
        if system:
            b["systemInstruction"] = {"parts": [{"text": system}]}
        return b

    def go(model, full=True):
        url = f"{_GEMINI_BASE}/models/{model}:generateContent"
        return _with_retry(lambda: _post(url, headers, body_for(full), timeout), max_retries)

    try:
        try:
            data = go(model)
        except _HTTP as e:
            if e.code == 404:  # model retired / not available to this key
                model = _resolve_gemini_model(key)
                data = go(model)
            elif e.code == 400:  # an optional generationConfig field was rejected
                data = go(model, full=False)
            else:
                raise
    except _HTTP as e:
        raise LLMError(str(e)) from e
    _gemini_model[0] = model
    try:
        cand = data["candidates"][0]
        parts = cand["content"]["parts"]
        return "".join(p.get("text", "") for p in parts)
    except (KeyError, IndexError, TypeError) as e:
        reason = (data.get("candidates") or [{}])[0].get("finishReason") if isinstance(data, dict) else None
        raise LLMError(f"empty Gemini response (finishReason={reason})") from e


# ---------------------------------------------------------------- public API

def call_llm(prompt, system=None, max_tokens=500, timeout=90, max_retries=5, json_mode=False):
    """Text reply, or None if no provider is configured. Raises LLMError on failure."""
    provider = active_provider()
    if provider is None:
        return None
    if _breaker["open"]:
        raise LLMError(f"model disabled for the rest of this run ({_breaker['reason']})")
    _pace(provider)
    STATS["calls"] += 1
    try:
        if provider == "anthropic":
            model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")
            body = {"model": model, "max_tokens": max(max_tokens, 1024), "temperature": 0,
                    "messages": [{"role": "user", "content": prompt}]}
            if system:
                body["system"] = system
            headers = {"Content-Type": "application/json", "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                       "anthropic-version": "2023-06-01"}
            data = _with_retry(lambda: _post("https://api.anthropic.com/v1/messages", headers, body, timeout), max_retries)
            out = "".join(b.get("text", "") for b in data.get("content", []))
        elif provider == "gemini":
            out = _gemini_call(prompt, system, max_tokens, timeout, max_retries, json_mode)
        else:
            model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
            base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
            messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
            # Reasoning models spend output tokens on thinking: keep headroom (tunable) so the
            # visible answer isn't cut off. OPENAI_REASONING_EFFORT=low|medium|high is optional.
            floor = int(os.environ.get("LLM_MAX_TOKENS_FLOOR", "1024"))
            body = {"model": model, "max_tokens": max(max_tokens, floor), "temperature": 0, "messages": messages}
            if os.environ.get("OPENAI_REASONING_EFFORT"):
                body["reasoning_effort"] = os.environ["OPENAI_REASONING_EFFORT"]
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
            data = _with_retry(lambda: _post(f"{base_url}/chat/completions", headers, body, timeout), max_retries)
            out = data["choices"][0]["message"]["content"]
    except LLMError as e:
        _breaker["consec"] += 1
        if isinstance(e, QuotaExhausted) or _breaker["consec"] >= 3:
            _breaker["open"], _breaker["reason"] = True, str(e)[:120]
        raise
    _breaker["consec"] = 0
    STATS["ok"] += 1
    return out


def extract_json(text):
    """First JSON object/array found in `text` (tolerates code fences, preamble, trailing
    prose). Returns None if there isn't one."""
    if not text:
        return None
    t = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(t)
    except ValueError:
        pass
    dec = json.JSONDecoder()
    for m in re.finditer(r"[\{\[]", t):
        try:
            obj, _ = dec.raw_decode(t[m.start():])
            return obj
        except ValueError:
            continue
    return None


def call_llm_json(prompt, system=None, max_tokens=800, timeout=90, max_retries=5):
    """(parsed_json_or_None, raw_text). Does one repair retry on a malformed reply.
    Returns (None, None) when no provider is configured; raises LLMError on API failure."""
    raw = call_llm(prompt, system=system, max_tokens=max_tokens, timeout=timeout,
                   max_retries=max_retries, json_mode=True)
    if raw is None:
        return None, None
    obj = extract_json(raw)
    if obj is not None:
        return obj, raw
    raw2 = call_llm(prompt + "\n\nYour previous reply was not valid JSON. Reply again with ONLY the JSON, "
                    "no commentary and no code fences.", system=system, max_tokens=max_tokens,
                    timeout=timeout, max_retries=max_retries, json_mode=True)
    return extract_json(raw2), (raw2 or raw)
