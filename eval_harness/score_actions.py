"""Score action predictions (dry run) against an eval set.

  python3 score_actions.py --gold ../evals/actions_train.jsonl --predictions predictions.jsonl --out results_actions.json

predictions.jsonl, one line per command:
  {"id": "ACT-TR-04", "actions": [{"type": "calendar.update_event",
                                   "args": {"event_id": "CAL-BOARDPREP", "start": "2026-09-18T15:00:00-07:00", "end": "2026-09-18T16:00:00-07:00"}}]}

A command scores 1 only if the predicted action list matches the expected list (or one of the accepted
alternatives): same number of actions, same types, every expected argument matches. Order doesn't matter.
Extra arguments are ignored. Extra actions are not.
"""
import argparse
import json
from datetime import datetime
from itertools import permutations
from zoneinfo import ZoneInfo

LOCAL = ZoneInfo("America/Los_Angeles")


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _dt(v):
    d = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=LOCAL)  # no offset = Alex's local time


def _as_list(v):
    if v is None:
        return []
    return [str(x).lower().strip() for x in (v if isinstance(v, list) else [v])]


def arg_ok(matcher, value):
    if value is None:
        return False
    s = str(value).lower()
    if "eq" in matcher:
        return s == str(matcher["eq"]).lower()
    if "in" in matcher:
        return s in [str(x).lower() for x in matcher["in"]]
    if "contains_any" in matcher:
        return any(t.lower() in s for t in matcher["contains_any"])
    if "contains_all" in matcher:
        return all(t.lower() in s for t in matcher["contains_all"])
    if "set_eq" in matcher:
        return set(_as_list(value)) == {x.lower() for x in matcher["set_eq"]}
    if "includes" in matcher:
        return {x.lower() for x in matcher["includes"]} <= set(_as_list(value))
    if "datetime" in matcher:
        try:
            diff = abs((_dt(value) - _dt(matcher["datetime"])).total_seconds()) / 60
        except (ValueError, TypeError):
            return False
        return diff <= matcher.get("tolerance_min", 0)
    raise ValueError(f"unknown matcher {matcher}")


def action_ok(expected, predicted):
    if expected["type"] != predicted.get("type"):
        return False, 0, len(expected["args"])
    args = predicted.get("args") or {}
    hits = sum(arg_ok(m, args.get(k)) for k, m in expected["args"].items())
    return hits == len(expected["args"]), hits, len(expected["args"])


def list_ok(expected, predicted):
    """Best matching over orderings. Returns (full_match, arg_hits, arg_total)."""
    total = sum(len(e["args"]) + 1 for e in expected)
    if len(expected) != len(predicted):
        best = 0
        for e in expected:  # partial credit only: best single match per expected action
            best += max([action_ok(e, p)[1] + (e["type"] == p.get("type")) for p in predicted] or [0])
        return False, best, total
    best_hits, full = 0, False
    for perm in permutations(predicted):
        results = [action_ok(e, p) for e, p in zip(expected, perm)]
        hits = sum(r[1] + (e["type"] == p.get("type")) for r, e, p in zip(results, expected, perm))
        if all(r[0] for r in results):
            return True, total, total
        best_hits = max(best_hits, hits)
    return full, best_hits, total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gold", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--out", default="results_actions.json")
    args = p.parse_args()

    gold = load_jsonl(args.gold)
    preds = {x["id"]: x.get("actions", []) for x in load_jsonl(args.predictions)}

    rows, arg_hits, arg_total = [], 0, 0
    for item in gold:
        predicted = preds.get(item["id"], [])
        best = (False, 0, 1)
        for option in [item["expected"]] + item["alternatives"]:
            res = list_ok(option, predicted)
            if res[0] or res[1] / res[2] > best[1] / best[2]:
                best = res
            if res[0]:
                break
        ok, hits, tot = best
        arg_hits, arg_total = arg_hits + hits, arg_total + tot
        rows.append({"id": item["id"], "command": item["command"], "pass": ok,
                     "arg_accuracy": round(hits / tot, 3), "predicted": predicted})
        print(f"{item['id']:<11} {'PASS' if ok else 'FAIL'}  args {hits}/{tot}  {item['command']}")

    summary = {"n": len(rows), "pass_rate": round(sum(r["pass"] for r in rows) / len(rows), 4),
               "arg_accuracy": round(arg_hits / arg_total, 4)}
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "items": rows}, f, indent=2)
    print(f"\npass rate {summary['pass_rate']:.1%}   argument accuracy {summary['arg_accuracy']:.1%}   (n={summary['n']})")


if __name__ == "__main__":
    main()
