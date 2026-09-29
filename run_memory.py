#!/usr/bin/env python3
"""Run the memory system on a questions file.

  python3 run_memory.py --questions evals/memory_train.jsonl --out out/memory_train_answers.jsonl

Input line:  {"id": "...", "question": "...", "as_of": "..."}
Output line: {"id": "...", "answer": "...", "sources": [...], "retrieved": [...], "abstained": bool}
"""
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
from memory.ingest import load_visible
from memory.retrieve import retrieve


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_one(question, as_of, data_dir, context_k=8):
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
    args = p.parse_args()

    items = load_jsonl(args.questions)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for item in items:
            result = run_one(item["question"], item["as_of"], args.data)
            result["id"] = item["id"]
            f.write(json.dumps(result) + "\n")
            f.flush()
            if not args.quiet:
                flag = "ABSTAIN" if result["abstained"] else "answer "
                print(f"{item['id']:<11} {flag}  {result['answer'][:90]}")
    print(f"\nwrote {len(items)} answers to {args.out}")


if __name__ == "__main__":
    main()
