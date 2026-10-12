#!/usr/bin/env python3
"""Tune how much the model's planned searches count, WITHOUT new model calls.

Plans are cached by tools/eval_first_stage.py (PLAN_CACHE), so run that once with --plan, then:

  python3 tools/sweep_plan.py

prints, for each setting of PLAN_WEIGHT (how much the planned searches count when fused) and
PLAN_SLOTS (how many pool places are reserved for what the plan adds), the top-10 and POOL recall
on the train questions and on the paraphrased dev set. Pick a setting that helps dev without
costing train; do not pick per question.
"""
import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("DISABLE_LLM_RERANK", "1")
os.environ.setdefault("PLAN_CACHE", ".cache/plans.json")
from memory.retrieve import retrieve

def sat(ids, grp):
    s = set(ids); base = {i.split("#")[0] for i in ids}
    return any(m in s or m in base for m in grp)

SETS = ["evals/memory_train.jsonl", "evals/memory_dev_paraphrase.jsonl"]
items = {p: [g for g in (json.loads(l) for l in open(p, encoding="utf-8") if l.strip()) if g.get("needed")] for p in SETS}
CONFIGS = [  # (PLAN_WEIGHT, PLAN_SLOTS, PLAN_POOL)  -- weight "0" = no planning
    ("0", "0", "-"),
    ("1.0", "15", "legacy"), ("1.0", "15", "additions"), ("1.0", "25", "additions"),
    ("2.0", "15", "legacy"), ("2.0", "15", "additions"),
]
print(f"{'WEIGHT':>6} {'SLOTS':>5} {'POOL-MODE':>10} | " + " | ".join(f"{Path(p).stem[7:13]:<6} top10  pool" for p in SETS))
for w, slots, mode in CONFIGS:
    os.environ["PLAN_WEIGHT"], os.environ["PLAN_SLOTS"], os.environ["PLAN_POOL"] = w, slots, (mode if mode != "-" else "additions")
    cells = []
    for p in SETS:
        t10 = pool = 0
        for g in items[p]:
            ids, dbg = retrieve(g["question"], g["as_of"], "data", top_k=45, rerank=False, use_plan=(w != "0"))
            pids = dbg.get("pool_ids") or ids[:45]
            t10 += all(sat(ids[:10], grp) for grp in g["needed"])
            pool += all(sat(pids, grp) for grp in g["needed"])
        cells.append(f"{t10:>2}/{len(items[p])}   {pool:>2}/{len(items[p])}")
    print(f"{w:>6} {slots:>5} {mode:>10} | " + " | ".join(f"{c:<17}" for c in cells) + (" (no plan)" if w == "0" else ""), flush=True)
