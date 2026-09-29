"""One place for every LLM call in this repo. Supports three providers, tried in
this order based on which key is set: ANTHROPIC_API_KEY, GEMINI_API_KEY,
OPENAI_API_KEY (order doesn't matter in practice -- set exactly one). Gemini is
listed in .env.example as the recommended free option.

Retries with exponential backoff on 429 (rate limit) and 5xx responses, since a
single transient rate-limit shouldn't knock every remaining question down to the
extractive/rule-based fallback -- that's what actually happened in early runs
(see README "what didn't work").
"""
import json
import os
import random
import time
import urllib.error
import urllib.request


class LLMError(Exception):
    pass


def _post(url, headers, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _with_retry(fn, max_retries=4, base_delay=2.0):
    last_err = None
    for attempt in range(max_retries):
        try:
            return fn()
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code == 429 or 500 <= e.code < 600:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                time.sleep(delay)
                continue
            raise LLMError(f"HTTP {e.code}: {e.read()[:300]}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = e
            time.sleep(base_delay * (2 ** attempt))
    raise LLMError(f"gave up after {max_retries} attempts: {last_err}")


def active_provider():
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return None


def call_llm(prompt, system=None, max_tokens=500, timeout=60, max_retries=4):
    """Returns the model's text response, or None if no provider is configured.
    Raises LLMError on a non-retryable failure or after exhausting retries --
    callers should catch this and fall back to the rule-based/extractive path."""
    provider = active_provider()
    if provider is None:
        return None

    if provider == "anthropic":
        model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")
        body = {"model": model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]}
        if system:
            body["system"] = system
        headers = {"Content-Type": "application/json", "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                   "anthropic-version": "2023-06-01"}
        data = _with_retry(lambda: _post("https://api.anthropic.com/v1/messages", headers, body, timeout), max_retries)
        return "".join(b.get("text", "") for b in data.get("content", []))

    if provider == "gemini":
        model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
        key = os.environ["GEMINI_API_KEY"]
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
        contents = [{"role": "user", "parts": [{"text": prompt}]}]
        body = {"contents": contents, "generationConfig": {"maxOutputTokens": max_tokens}}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        headers = {"Content-Type": "application/json"}
        data = _with_retry(lambda: _post(url, headers, body, timeout), max_retries)
        try:
            parts = data["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts)
        except (KeyError, IndexError) as e:
            raise LLMError(f"unexpected Gemini response shape: {data}") from e

    if provider == "openai":
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        body = {"model": model, "max_tokens": max_tokens, "messages": messages}
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
        data = _with_retry(lambda: _post(f"{base_url}/chat/completions", headers, body, timeout), max_retries)
        return data["choices"][0]["message"]["content"]

    return None
