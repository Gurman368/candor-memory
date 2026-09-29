"""Score retrieval: did the system fetch the records each question needs? This is the main memory score.
No judge needed.

  python3 score_retrieval.py --gold ../evals/memory_train.jsonl --answers answers.jsonl --out results_retrieval.json

Uses `retrieved` from each answer: a RANKED list (best first) of up to 20 record or segment ids,
i.e. what the system would hand to its answer writer. If `retrieved` is missing, `sources` is used
and the report says so.

Gold `needed` is a list of groups: every group must be retrieved, and any one member of a group
satisfies it. A member that is a whole record (e.g. a meeting id) is satisfied by any of its segments.

Per question (primary): everything needed is in the top 10 AND nothing forbidden is in the top 10.
Forbidden = a record that didn't exist yet at `as_of`, or was deleted by `as_of`. Retrieving one is
a hard failure. Questions about deleted content (marked `harm_check`) have nothing to find: they pass
only if nothing forbidden is in the top 20. Questions whose answer is "not in memory" aren't part of the
retrieval score (their forbidden-record check is still reported).

Also reported: coverage in the top 5 / 10 / 20 (exact passage and whole record), questions where
nothing needed was found, MRR, and how many forbidden records were retrieved.
"""
import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import records
from stats import cluster_bootstrap

KS = (5, 10, 20)
PRIMARY_K = 10


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def needed_groups(item):
    return item.get("needed") or [[e] for e in item.get("evidence", [])]


def ranked(ans):
    raw = ans.get("retrieved")
    fallback = raw is None
    seen = {}
    for x in (ans.get("sources") or []) if fallback else raw:
        cid = x.get("id") if isinstance(x, dict) else x
        if isinstance(cid, str):
            seen.setdefault(cid, None)
    return list(seen)[:max(KS)], fallback


def score(gold, answers, ctx):
    record_of, avail, deleted = ctx["record_of"], ctx["avail"], ctx["deleted"]
    units_of = defaultdict(list)
    for u, r in record_of.items():
        units_of[r].append(u)

    def rec(cid):
        return record_of.get(cid, cid)

    rows = []
    for item in gold:
        ans = answers.get(item["id"]) or {}
        top20, fallback = ranked(ans)
        as_of = datetime.fromisoformat(item["as_of"])

        def forbidden(cid):
            t = avail.get(cid)
            if t and t > as_of:
                return "not yet delivered"
            if cid in deleted and deleted[cid] <= as_of:
                return "deleted"
            return None

        harm = [(i + 1, cid, why) for i, cid in enumerate(top20) if (why := forbidden(cid))]
        groups = needed_groups(item) if item.get("retrieval_scored", item["answerable"]) else []
        scored = bool(groups)
        row = {"id": item["id"], "storyline": item.get("storyline"), "category": item["category"],
               "scored": scored, "harm_check": bool(item.get("harm_check")), "fallback": fallback, "retrieved": len(top20),
               "harm": [{"rank": r, "id": c, "why": w} for r, c, w in harm]}
        for k in KS:
            top = top20[:k]
            top_set, top_recs = set(top), {rec(c) for c in top}
            # an exact id counts; a whole-record member (e.g. "MTG-0909-ACME") is met by any of its segments
            unit_ok = [any(m in top_set or (m not in record_of and m in top_recs) for m in g) for g in groups]
            rec_ok = [any(rec(m) in top_recs for m in g) for g in groups]
            if scored:
                row[f"complete_unit@{k}"] = float(all(unit_ok))
                row[f"complete_record@{k}"] = float(all(rec_ok))
                row[f"recall_unit@{k}"] = sum(unit_ok) / len(groups)
            row[f"clean@{k}"] = float(not any(r <= k for r, _, _ in harm))
        if scored:
            first = None
            for i, cid in enumerate(top20):
                if any(cid in g or (rec(cid) in g and rec(cid) not in record_of) for g in groups):
                    first = i + 1
                    break
            row["first_rank"] = first
            row["score"] = row[f"complete_unit@{PRIMARY_K}"] * row[f"clean@{PRIMARY_K}"]
        else:
            row["score"] = row[f"clean@{max(KS)}"]
        rows.append(row)

    s_rows = [r for r in rows if r["scored"]]

    def mean(key, rs):
        vals = [r[key] for r in rs if r.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    summary = {
        "n": len(rows), "scored_on_coverage": len(s_rows), "scored_on_harm_only": len(rows) - len(s_rows),
        "primary": "everything needed in top 10 and nothing forbidden in top 10",
        "score": cluster_bootstrap([r for r in rows if r["scored"] or r["harm_check"]]),
        "harm_only_clean@20": mean("clean@20", [r for r in rows if not r["scored"]]),
        **{f"{m}@{k}": mean(f"{m}@{k}", s_rows) for m in ("complete_unit", "complete_record") for k in KS},
        "found_none@20": round(sum(r["recall_unit@20"] == 0 for r in s_rows) / len(s_rows), 4) if s_rows else None,
        "mrr": round(sum(1 / r["first_rank"] for r in s_rows if r["first_rank"]) / len(s_rows), 4) if s_rows else None,
        "questions_with_forbidden@10": sum(not r[f"clean@{PRIMARY_K}"] for r in rows),
        "forbidden_retrieved@20": sum(len(r["harm"]) for r in rows),
        "used_sources_fallback": sum(r["fallback"] for r in rows),
        "by_category": {c: round(sum(r["score"] for r in v) / len(v), 4) for c, v in sorted(by_cat.items())},
    }
    return {"summary": summary, "items": rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gold", required=True)
    p.add_argument("--answers", required=True)
    p.add_argument("--data", default=str(Path(__file__).resolve().parent.parent / "data"))
    p.add_argument("--out", default="results_retrieval.json")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    gold = load_jsonl(args.gold)
    answers = {a["id"]: a for a in load_jsonl(args.answers)}
    result = score(gold, answers, records.context(args.data))
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, default=str)

    if not args.quiet:
        for r in result["items"]:
            status = "PASS" if r["score"] == 1 else "FAIL"
            detail = (f"found {r['recall_unit@10']:.0%} of needed in top 10" if r["scored"] else "harm check only")
            harm = f"  FORBIDDEN: {', '.join(h['id'] + ' (' + h['why'] + ')' for h in r['harm'][:3])}" if r["harm"] else ""
            print(f"{r['id']:<11} {status}  {detail}{harm}")
    s = result["summary"]
    ci = s["score"]["ci95"]
    print(f"\nretrieval score {s['score']['mean']:.1%}  (95% CI over storylines {ci[0]:.0%}–{ci[1]:.0%}, n={s['n']})" if ci
          else f"\nretrieval score {s['score']['mean']:.1%}  (n={s['n']})")
    print("found everything needed, top 5 / 10 / 20:  exact passage " +
          " / ".join(f"{s[f'complete_unit@{k}']:.0%}" for k in KS) +
          "   whole record " + " / ".join(f"{s[f'complete_record@{k}']:.0%}" for k in KS))
    print(f"found nothing needed in top 20: {s['found_none@20']:.0%}   MRR {s['mrr']}")
    print(f"questions with a forbidden record in top 10: {s['questions_with_forbidden@10']}   "
          f"forbidden records retrieved (top 20): {s['forbidden_retrieved@20']}")
    if s["used_sources_fallback"]:
        print(f"WARNING: {s['used_sources_fallback']} answers had no `retrieved`; used `sources` instead")


if __name__ == "__main__":
    main()
