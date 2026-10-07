"""Offline plumbing tests: a fake model stands in for the API so the v2 code paths
(rerank, JSON repair, answer parsing, loud fallbacks) run without network.
Run: python3 tests/test_offline.py   (these test the plumbing, NOT model quality)"""
import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.update(GEMINI_API_KEY="fake", DISABLE_LLM_QUERY_EXPANSION="1", LLM_MIN_INTERVAL="0")

import memory.llm_client as lc
from memory import answer as ans, retrieve as rt

Q, ASOF = "When is Route Planner v2 launching?", "2026-09-18T18:00:00-07:00"

def fake(script):
    it = iter(script)
    def _call(prompt, system=None, max_tokens=500, timeout=90, max_retries=5, json_mode=False):
        lc.STATS["calls"] += 1; lc.STATS["ok"] += 1
        r = next(it)
        if isinstance(r, Exception): raise r
        return r
    return _call

def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name); assert cond, name

# extract_json tolerance
check("fenced json", lc.extract_json('```json\n{"a": 1}\n```') == {"a": 1})
check("preamble+trailing", lc.extract_json('Sure! {"a": [1,2]} hope that helps') == {"a": [1, 2]})
check("garbage -> None", lc.extract_json("no json here") is None)

# rerank: model picks go first, only valid ids kept, rest keep lexical order
base, _ = rt.retrieve(Q, ASOF, "data", rerank=False)
pick = base[7]
lc.call_llm = fake([json.dumps({"ranked": [pick, "NOT-A-REAL-ID", base[3]]})]); rt.call_llm_json.__globals__["call_llm"] = lc.call_llm
ids, dbg = rt.retrieve(Q, ASOF, "data")
check("rerank promotes model picks", ids[:2] == [pick, base[3]] and dbg["reranked"])
check("rerank drops invented ids", "NOT-A-REAL-ID" not in ids)
check("no duplicates", len(ids) == len(set(ids)))

# malformed reply -> repair retry succeeds
lc.call_llm = fake(["not json at all", json.dumps({"ranked": [base[1]]})]); rt.call_llm_json.__globals__["call_llm"] = lc.call_llm
ids, dbg = rt.retrieve(Q, ASOF, "data")
check("repair retry works", ids[0] == base[1])

# API failure in rerank -> lexical order, logged loudly, run continues
n0 = len(lc.STATS["fallbacks"])
lc.call_llm = fake([lc.LLMError("boom")]); rt.call_llm_json.__globals__["call_llm"] = lc.call_llm
ids, dbg = rt.retrieve(Q, ASOF, "data")
check("rerank failure keeps lexical order", ids == base and not dbg["reranked"])
check("rerank failure recorded", len(lc.STATS["fallbacks"]) == n0 + 1)

# answer: good JSON, abstain flag, raw-text salvage, API failure -> extractive (logged)
units = [u for u in rt.load_visible("data", ASOF)][:5]
good = json.dumps({"evidence": "x", "answer": "October 21.", "sources": [units[0].id, "bogus"], "abstained": False})
lc.call_llm = fake([good]); ans.call_llm_json.__globals__["call_llm"] = lc.call_llm
t, s, a = ans.answer(Q, ASOF, units)
check("answer parsed, sources filtered", t == "October 21." and s == [units[0].id] and not a)
lc.call_llm = fake([json.dumps({"answer": "I don't know -- not covered.", "sources": [units[0].id], "abstained": False})])
ans.call_llm_json.__globals__["call_llm"] = lc.call_llm
t, s, a = ans.answer(Q, ASOF, units)
check("'I don't know' text counts as abstained, no sources", a and s == [])
lc.call_llm = fake(["totally not json", "still not json"]); ans.call_llm_json.__globals__["call_llm"] = lc.call_llm
t, s, a = ans.answer(Q, ASOF, units)
check("unparseable reply keeps model text", t == "still not json")
lc.call_llm = fake([lc.LLMError("down")]); ans.call_llm_json.__globals__["call_llm"] = lc.call_llm
t, s, a = ans.answer(Q, ASOF, units)
check("API failure falls back to extractive (not a crash)", isinstance(t, str) and len(t) > 0)
print("\nall offline checks passed"); print(lc.stats_summary())
