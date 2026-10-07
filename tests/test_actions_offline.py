"""Offline plumbing tests for the v2 action planner (fake model; tests policy + validation,
NOT model quality). Run: python3 tests/test_actions_offline.py"""
import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.update(GEMINI_API_KEY="fake", LLM_MIN_INTERVAL="0")
import memory.llm_client as lc
import actions.llm_parse as lp
from actions.directory import load_directory

D = load_directory("data"); AS = "2026-09-18T09:00:00-07:00"
seen = {}
def fake(reply):
    def _c(prompt, system=None, max_tokens=500, timeout=90, max_retries=5, json_mode=False):
        seen["prompt"] = prompt; seen["n"] = seen.get("n", 0) + 1
        lc.STATS["calls"] += 1; lc.STATS["ok"] += 1
        return reply if isinstance(reply, str) else json.dumps(reply)
    lc.call_llm = _c; lp.call_llm_json.__globals__["call_llm"] = _c
def check(n, c): print(("PASS " if c else "FAIL ") + n); assert c, n

fake({"actions": [{"type": "slack.send_message", "args": {"to": "U03SARAHK", "text": "x"}}]}); seen.clear()
r = lp.llm_parse("Message Sarah about the pricing proposal", AS, D, "data")
check("same-first-name ambiguity -> clarify without any model call", r[0]["type"] == "clarify" and "n" not in seen)
r = lp.llm_parse("Delete all my emails from Marcus", AS, D, "data")
check("destructive -> confirm without any model call", r[0]["type"] == "confirm" and "n" not in seen)

ok = {"actions": [{"type": "gmail.send", "args": {"to": ["sarah.patel@acmefreight.example.com"], "subject": "Proposal", "body": "Did you review the proposal?"}}]}
fake(ok); r = lp.llm_parse("Email Sarah Patel and ask if she's had a chance to look at the proposal", AS, D, "data")
check("valid model actions accepted (cc defaulted)", r and r[0]["args"]["cc"] == [])
check("prompt carries memory records + rules draft + directory", all(k in seen["prompt"] for k in ("Relevant memory records", "Rules-based draft", "slack_users")))

fake({"actions": [{"type": "gmail.send", "args": {"to": ["ghost@nowhere.example"], "subject": "", "body": "hi"}}]})
check("invented email rejected -> None (rules fallback)", lp.llm_parse("Email Priya the plan", AS, D, "data") is None)
fake({"actions": [{"type": "calendar.update_event", "args": {"event_id": "CAL-NOPE", "start": "2026-09-18T15:00:00-07:00"}}]})
check("unknown event id rejected", lp.llm_parse("Move board deck prep to 3pm", AS, D, "data") is None)
fake("not json"); n = len(lc.STATS["fallbacks"])
check("garbage reply -> None and still no crash", lp.llm_parse("Open Figma", AS, D, "data") is None)
fake({"actions": [{"type": "app.open", "args": {"app": "Figma"}}]})
check("app.open accepted", lp.llm_parse("Open Figma", AS, D, "data")[0]["args"]["app"] == "Figma")
print("\nall action checks passed")
