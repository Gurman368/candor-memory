#!/usr/bin/env python3
"""Run the memory system on a questions file.

  python3 run_memory.py --questions evals/memory_train.jsonl --out out/memory_train_answers.jsonl

Input line:  {"id": "...", "question": "...", "as_of": "..."}
Output line: {"id": "...", "answer": "...", "sources": [...], "retrieved": [...], "abstained": bool}
"""
# Windows' default file encoding is cp1252, not UTF-8 -- this data (and this code)
# is UTF-8. Force it here so every open() call in this process decodes correctly
# regardless of OS/locale. No effect on Mac/Linux, where UTF-8 is already the default.
import builtins as _builtins
_orig_open = _builtins.open
def _utf8_open(file, mode="r", *args, **kwargs):
    if "b" not in mode and "encoding" not in kwargs:
        kwargs["encoding"] = "utf-8"
    return _orig_open(file, mode, *args, **kwargs)
_builtins.open = _utf8_open

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Auto-load a .env file if present (no dependency on python-dotenv)
_env_file = Path(__file__).resolve().parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip()
            if val and key not in os.environ:
                os.environ[key] = val

from memory.answer import answer
from memory.llm_client import STATS, stats_summary
from memory.ingest import load_visible
from memory.retrieve import retrieve


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_one(question, as_of, data_dir, context_k=12):
    ids, dbg = retrieve(question, as_of, data_dir, top_k=20)
    if not ids:
        return {"answer": "I don't know -- nothing in memory covers this.",
                "sources": [], "retrieved": [], "abstained": True}
    units = load_visible(data_dir, as_of)
    by_id = {u.id: u for u in units}
    context_units = [by_id[i] for i in ids[:context_k] if i in by_id]
    text, sources, abstained = answer(question, as_of, context_units)
    return {"answer": text, "sources": sources, "retrieved": ids, "abstained": abstained}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--questions", required=True)
    p.add_argument("--data", default="data")
    p.add_argument("--out", required=True)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="keep answers from an earlier run that came fully from the model; redo the rest")
    args = p.parse_args()

    items = load_jsonl(args.questions)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    status_path = Path(args.out + ".status.json")
    keep = {}
    if args.resume and Path(args.out).exists() and status_path.exists():
        old = {r["id"]: r for r in load_jsonl(args.out)}
        st = json.loads(status_path.read_text())
        keep = {i: r for i, r in old.items() if st.get(i) == "ok"}
        print(f"resuming: keeping {len(keep)} fully model-generated answers")
    status = {}
    with open(args.out, "w") as f:
        for item in items:
            if item["id"] in keep:
                result, status[item["id"]] = keep[item["id"]], "ok"
            else:
                before, ok_before = len(STATS["fallbacks"]), STATS["ok"]
                result = run_one(item["question"], item["as_of"], args.data)
                result["id"] = item["id"]
                # "ok" only if the model really answered (a no-key run is NOT "ok")
                status[item["id"]] = ("ok" if STATS["ok"] > ok_before and len(STATS["fallbacks"]) == before
                                      else "degraded")
            f.write(json.dumps(result) + "\n")
            f.flush()
            status_path.write_text(json.dumps(status))
            if not args.quiet:
                flag = "ABSTAIN" if result["abstained"] else "answer "
                print(f"{item['id']:<11} {flag}  {result['answer'][:90]}")
    print(f"\nwrote {len(items)} answers to {args.out}")
    print(stats_summary())


if __name__ == "__main__":
    main()
