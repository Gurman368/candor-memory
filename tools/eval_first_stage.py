#!/usr/bin/env python3
"""First-stage candidate recall, no model involved.

For each answerable question: is EVERYTHING needed (every gold group, any member) inside the
first N ranked candidates? Reported at N = 10 / 20 / 45 / 150. Use it to test each retrieval
change on its own (toggle with the CAND_* env vars) on the train questions AND on the
paraphrased dev set (evals/memory_dev_paraphrase.jsonl: same gold records, different wording,
which is what the hidden set looks like).

  python3 tools/eval_first_stage.py [--sets evals/memory_train.jsonl evals/memory_dev_paraphrase.jsonl]
"""
import argparse, json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("DISABLE_LLM_RERANK", "1")
os.environ.setdefault("PLAN_CACHE", ".cache/plans.json")   # model plans are cached: tuning needs no new calls
from memory.retrieve import retrieve

def sat(ids, grp):
    s = set(ids); base = {i.split("#")[0] for i in ids}
    return any(m in s or m in base for m in grp)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["evals/memory_train.jsonl", "evals/memory_dev_paraphrase.jsonl"])
    ap.add_argument("--data", default="data")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--plan", action="store_true", help="also run the model query planner (needs an API key)")
    a = ap.parse_args()
    from memory.candidates import retrieval_banner
    print("\n".join(retrieval_banner()))
    cuts = (10, 20, 45, 150)
    for path in a.sets:
        items = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        items = [g for g in items if g.get("needed")]
        hit = {n: 0 for n in cuts}; misses = {n: [] for n in cuts}; pool_hit = 0; pool_miss = []
        for g in items:
            ids, dbg = retrieve(g["question"], g["as_of"], a.data, top_k=max(cuts), rerank=False, use_plan=a.plan)
            pids = dbg.get("pool_ids") or ids[:45]
            if all(sat(pids, grp) for grp in g["needed"]):
                pool_hit += 1
            else:
                pool_miss.append(g["id"])
            if a.verbose and dbg.get("plan"):
                print("   plan", g["id"], dbg["plan"]["queries"], dbg["plan"]["dates"])
            for n in cuts:
                if all(sat(ids[:n], grp) for grp in g["needed"]):
                    hit[n] += 1
                else:
                    misses[n].append(g["id"])
        row = "  ".join(f"R@{n}={hit[n]}/{len(items)}" for n in cuts) + f"  POOL(what the model sees)={pool_hit}/{len(items)}"
        print(f"{Path(path).name:<32} {row}")
        if a.verbose:
            print("   misses@10:", misses[10]); print("   pool misses:", pool_miss)

if __name__ == "__main__":
    main()
