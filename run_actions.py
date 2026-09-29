#!/usr/bin/env python3
"""Run the action parser on a commands file.

  python3 run_actions.py --commands evals/actions_train.jsonl --out out/actions_train_predictions.jsonl

Input line:  {"id": "...", "command": "...", "as_of": "..."}
Output line: {"id": "...", "actions": [{"type": "...", "args": {...}}, ...]}
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

_env_file = Path(__file__).resolve().parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip()
            if val and key not in os.environ:
                os.environ[key] = val

from actions.directory import load_directory
from actions.llm_parse import llm_parse
from actions.rules import parse as rules_parse


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_one(command, as_of, directory, data_dir):
    actions = llm_parse(command, as_of, directory)
    if actions is not None:
        return actions
    return rules_parse(command, as_of, directory, data_dir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--commands", required=True)
    p.add_argument("--data", default="data")
    p.add_argument("--out", required=True)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    directory = load_directory(args.data)
    items = load_jsonl(args.commands)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for item in items:
            actions = run_one(item["command"], item["as_of"], directory, args.data)
            f.write(json.dumps({"id": item["id"], "actions": actions}) + "\n")
            f.flush()
            if not args.quiet:
                print(f"{item['id']:<11} {actions}")
    print(f"\nwrote {len(items)} predictions to {args.out}")


if __name__ == "__main__":
    main()
