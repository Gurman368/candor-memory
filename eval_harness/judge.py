"""LLM judge used by score_memory.py. It only runs after the rules pass, and it can confirm or
downgrade the rules' verdict, never upgrade it. Standard library only.

Providers (pick with --judge):
  claude-cli  the local Claude Code CLI, run isolated (see llm.py). No API key needed.
  anthropic   needs ANTHROPIC_API_KEY
  openai      any OpenAI-compatible endpoint: OPENAI_API_KEY, optionally OPENAI_BASE_URL
              (e.g. Gemini's free tier: OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
               with a Google AI Studio key and --model gemini-2.5-flash)
"""
import json
import os
import re
import time
import urllib.request

import llm

PROMPT = """You are grading one answer from a personal memory assistant against a reference.
Use only the reference and the checklist below. Do not use outside knowledge.

Question (asked as of {as_of}): {question}

Reference answer: {gold}

Checklist (each point is yes/no; points marked "Optional" never make an answer wrong):
{rubric}

Answer to grade:
{answer}

For each checklist point, decide yes or no. Then give a verdict:
- "correct": every non-optional point is yes, and nothing in the answer contradicts the reference.
- "partial": the main (first) point is yes, but another non-optional point is no.
- "incorrect": the main point is no, or the answer states outdated information as current, invents
  facts, or answers when the reference says the information isn't in memory.

Reply with JSON only:
{{"points": ["yes"|"no", ...], "verdict": "correct"|"partial"|"incorrect", "reason": "<one sentence>"}}"""


def _post(url, headers, body, retries=3):
    data = json.dumps(body).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def _complete(provider, model, prompt):
    if provider == "claude-cli":
        return llm.ask(prompt, model=model)
    if provider == "anthropic":
        out = _post("https://api.anthropic.com/v1/messages",
                    {"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
                    {"model": model, "max_tokens": 400, "temperature": 0,
                     "messages": [{"role": "user", "content": prompt}]})
        return out["content"][0]["text"]
    if provider == "openai":
        base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        out = _post(f"{base}/chat/completions",
                    {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}", "content-type": "application/json"},
                    {"model": model, "temperature": 0, "messages": [{"role": "user", "content": prompt}]})
        return out["choices"][0]["message"]["content"]
    raise ValueError(f"unknown judge provider: {provider}")


def judge(provider, model, item, answer):
    prompt = PROMPT.format(as_of=item["as_of"], question=item["question"], gold=item["gold_answer"],
                           rubric="\n".join(f"{i + 1}. {r}" for i, r in enumerate(item["rubric"])),
                           answer=answer or "(no answer)")
    text = _complete(provider, model, prompt)
    match = re.search(r"\{.*\}", text, re.S)
    try:
        parsed = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        parsed = {}
    verdict = parsed.get("verdict")
    if verdict not in ("correct", "partial", "incorrect"):
        return "incorrect", f"judge output unreadable: {text[:120]}"
    return verdict, parsed.get("reason", "")
