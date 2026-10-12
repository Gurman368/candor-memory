#!/usr/bin/env python3
"""Make ONE tiny model call and print the provider's raw answer, so a rate-limit error shows
exactly which limit was hit and when it resets (e.g. "Please try again in 3h12m").
Uses whatever provider/env the run scripts use (LLM_PROVIDER / OPENAI_* / GEMINI_API_KEY ...)."""
import json, os, sys, urllib.request, urllib.error
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import memory.llm_client as lc

print(lc.provider_banner())
p = lc.active_provider()
if p != "openai":
    print("This helper prints raw errors for OpenAI-compatible providers (Groq etc.). For others run:\n"
          '  python -c "import memory.llm_client as l; print(l.call_llm(\'ok\', max_tokens=5))"')
    sys.exit(0)
req = urllib.request.Request(
    os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1") + "/chat/completions",
    data=json.dumps({"model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"), "max_tokens": 20,
                     "messages": [{"role": "user", "content": "Say ok"}]}).encode(),
    headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"], "Content-Type": "application/json",
             "User-Agent": "Mozilla/5.0 (compatible; candor-memory/2.0)"})
try:
    r = urllib.request.urlopen(req, timeout=60)
    print("OK -- the model is available:", json.loads(r.read())["choices"][0]["message"]["content"][:60])
except urllib.error.HTTPError as e:
    print("HTTP", e.code, "\n" + e.read().decode("utf-8", "replace")[:900])
    print("\nRetry-After header:", e.headers.get("Retry-After"))
