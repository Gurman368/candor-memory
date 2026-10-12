"""Plumbing tests for v3 retrieval with a fake model / fake embedder (no network).
They check that the pieces connect and fail safely, NOT how good a real model is.
Run: python3 tests/test_retrieval_v3.py"""
import json, os, sys
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.update(GEMINI_API_KEY="fake", LLM_MIN_INTERVAL="0", DISABLE_LLM_RERANK="1")
import memory.llm_client as lc
from memory import candidates as C, retrieve as R
import run_memory

def check(n, c): print(("PASS " if c else "FAIL ") + n); assert c, n

# 1. relative dates resolve against the RECORD's own timestamp (nothing from later)
wed = datetime.fromisoformat("2026-09-16T10:00:00-07:00")      # a Wednesday
check("'Friday' written on Wed Sep 16 -> Sep 18", (9, 18) in C.relative_dates("let's sync Friday", wed))
check("'tomorrow' -> Sep 17", (9, 17) in C.relative_dates("see you tomorrow", wed))
check("'last Monday' -> Sep 14", (9, 14) in C.relative_dates("as I said last Monday", wed))

# 2. rank fusion keeps items that several retrievers agree on
check("rrf prefers agreement", C.rrf([(["a", "b", "c"], 1), (["b", "a"], 1)])[:2] in (["a", "b"], ["b", "a"]))

# 3. n-gram retriever matches word forms BM25's exact tokens miss
from memory.ingest import load_visible
units = load_visible("data", "2026-09-18T18:00:00-07:00")
ids = C.ngram_rank(units, "launching schedule for the planner", 5)
check("ngram returns candidates", len(ids) == 5)

# 4. embedder interface (fake model): vectors are cached and used
class Fake:
    calls = 0
    def encode(self, texts):
        Fake.calls += 1
        return [[float("flight" in t.lower()), float("denver" in t.lower()), 1.0] for t in texts]
got = C.embed_rank(units, "denver flight", 5, model=Fake())
check("embed_rank ranks by the model's vectors", got and any("FLIGHT" in i or "denver" in by.lower() for i, by in
      [(i, next(u.text for u in units if u.id == i)) for i in got]))

# 5. planner: model rewrites the question; its queries pull in a record the question's words never reach
def fake_llm(prompt, system=None, max_tokens=500, timeout=90, max_retries=5, json_mode=False):
    lc.STATS["calls"] += 1; lc.STATS["ok"] += 1
    if system.startswith("You plan searches"):
        return json.dumps({"queries": ["United Airlines trip confirmation Denver departure"], "dates": [], "names": []})
    if system.startswith("You are the second-chance"):
        return json.dumps({"queries": ["United Airlines trip confirmation Denver departure"]})
    if system.startswith("You answer questions"):
        if "EM-0912-FLIGHT" in prompt:
            return json.dumps({"evidence": "EM-0912-FLIGHT", "answer": "Found it.", "sources": ["EM-0912-FLIGHT"], "abstained": False, "missing": []})
        return json.dumps({"evidence": "nothing relevant", "answer": "I don't know", "sources": [], "abstained": True, "missing": ["flight departure time"]})
    return json.dumps({"ranked": []})
lc.call_llm = fake_llm
Q, AS = "What time does my plane to Colorado take off?", "2026-09-18T18:00:00-07:00"
without, _ = R.retrieve(Q, AS, "data", top_k=10, rerank=False, use_plan=False)
with_plan, dbg = R.retrieve(Q, AS, "data", top_k=10, rerank=False, use_plan=True)
check("record unreachable by the question's words ...", "EM-0912-FLIGHT" not in without)
check("... is in the pool the reranker sees once the model plans the search", "EM-0912-FLIGHT" in dbg["pool_ids"] and dbg["plan"]["queries"])

# 6. search again before giving up (planner off so the FIRST pass misses it)
os.environ["DISABLE_LLM_PLAN"] = "1"
res = run_memory.run_one(Q, AS, "data")
check("first answer abstained, second search found it, answer rewritten", not res["abstained"] and res["answer"] == "Found it.")
check("the record the answer relies on is ranked first", res["retrieved"][0] == "EM-0912-FLIGHT")

# 7. truly absent topic: still abstains after the second search (no forced answer)
def fake2(prompt, system=None, **k):
    lc.STATS["calls"] += 1; lc.STATS["ok"] += 1
    if system.startswith("You are the second-chance"): return json.dumps({"queries": ["zebra"]})
    return json.dumps({"evidence": "none", "answer": "I don't know", "sources": [], "abstained": True, "missing": ["x"]})
lc.call_llm = fake2
res = run_memory.run_one("What is Alex's favourite colour?", AS, "data")
check("unanswerable question still abstains", res["abstained"])

# 8. planner failure is logged, never fatal
def boom(*a, **k): raise lc.LLMError("down")
lc.call_llm = boom; os.environ.pop("DISABLE_LLM_PLAN")
n = len(lc.STATS["fallbacks"])
ids, _ = R.retrieve(Q, AS, "data", top_k=10, rerank=False, use_plan=True)
check("planner failure falls back to the other retrievers and is logged", ids and len(lc.STATS["fallbacks"]) == n + 1)
# 9. local cross-encoder (fake): jointly scores (query, record) pairs and reorders candidates
class FakeCross:
    def predict(self, pairs):
        return [float("denver" in text.lower()) for _, text in pairs]
by = {u.id: u for u in units}
ids_in = [u.id for u in units][:60]
ranked = C.cross_rank(by, ids_in, ["anything"], model=FakeCross())
top_has = ["denver" in by[i].text.lower() for i in ranked]
check("cross_rank puts the records the cross-encoder likes first", top_has[0] or not any(top_has))
C._CROSS.update(model=FakeCross(), tried=True, name="fake")
os.environ["DISABLE_LLM_PLAN"] = "1"
ids, dbg = R.retrieve("anything about travel", AS, "data", top_k=20, rerank=False, use_plan=False)
check("retrieve() uses the local reranker when present", "cross" in dbg["retrievers"])
C._CROSS.update(model=None, tried=True)
print("\nall v3 checks passed")
