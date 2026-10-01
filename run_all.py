#!/usr/bin/env python3
"""Run everything with one command.

  python3 run_all.py

Runs the memory system on evals/memory_train.jsonl and the actions bonus on
evals/actions_train.jsonl, writing both output files to out/. Equivalent to
running run_memory.py and run_actions.py separately (see their --help for
custom paths) -- this just wires up the default, expected-by-the-brief case
so the whole project runs with a single command.
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data")
    p.add_argument("--memory-questions", default="evals/memory_train.jsonl")
    p.add_argument("--action-commands", default="evals/actions_train.jsonl")
    p.add_argument("--memory-out", default="out/memory_train_answers.jsonl")
    p.add_argument("--actions-out", default="out/actions_train_predictions.jsonl")
    p.add_argument("--skip-actions", action="store_true", help="run the memory system only")
    args = p.parse_args()

    print("=" * 60)
    print("1/2  Memory system")
    print("=" * 60)
    subprocess.run([sys.executable, str(HERE / "run_memory.py"),
                     "--questions", args.memory_questions,
                     "--data", args.data,
                     "--out", args.memory_out], check=True)

    if not args.skip_actions:
        print("\n" + "=" * 60)
        print("2/2  Actions bonus")
        print("=" * 60)
        subprocess.run([sys.executable, str(HERE / "run_actions.py"),
                         "--commands", args.action_commands,
                         "--data", args.data,
                         "--out", args.actions_out], check=True)

    print("\nDone.")
    print(f"  memory answers      -> {args.memory_out}")
    if not args.skip_actions:
        print(f"  action predictions  -> {args.actions_out}")
    print("\nScore them with the harness, e.g.:")
    print("  cd eval_harness")
    print(f"  python3 score_retrieval.py --gold ../evals/memory_train.jsonl --answers ../{args.memory_out}")
    print(f"  python3 score_memory.py    --gold ../evals/memory_train.jsonl --answers ../{args.memory_out} --judge none")
    if not args.skip_actions:
        print(f"  python3 score_actions.py   --gold ../evals/actions_train.jsonl --predictions ../{args.actions_out}")


if __name__ == "__main__":
    main()
