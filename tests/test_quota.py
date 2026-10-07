"""Quota handling: a daily-quota 429 must stop retries at once and trip the breaker."""
import io, os, sys, urllib.error
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.update(GEMINI_API_KEY="fake", LLM_MIN_INTERVAL="0")
import memory.llm_client as lc
def boom(*a, **k):
    raise urllib.error.HTTPError("u", 429, "x", {}, io.BytesIO(b'{"error":{"message":"quota","details":"GenerateRequestsPerDayPerProjectPerModel-FreeTier"}}'))
lc._post = boom
tries = []
try: lc.call_llm("hi")
except lc.QuotaExhausted as e: tries.append(str(e)[:40])
assert tries, "daily quota should raise QuotaExhausted immediately"
try: lc.call_llm("hi"); raise SystemExit("breaker did not open")
except lc.LLMError as e: assert "disabled for the rest of this run" in str(e)
print("PASS quota -> immediate stop + breaker open"); print(lc.stats_summary())
