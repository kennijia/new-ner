#!/usr/bin/env python3
"""Grid search for negative sampling ratio (BERT-RE).

Runs `BERT-RE/run.py` with different `neg_pos_ratio` and seeds.

This script relies on env overrides supported by `BERT-RE/train.py`:
  - BERT_RE_NEG_POS_RATIO
  - BERT_RE_SEED
  - BERT_RE_SPLIT_LEVEL
  - BERT_RE_EXP_DIR

Example:
  cd /root/msy/ner/BERT-RE
  python grid_neg_ratio.py --ratios 1 3 5 --seeds 42 43 44
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime


def _format_r(r: float) -> str:
    s = (f"{r:.6f}").rstrip("0").rstrip(".")
    return s.replace(".", "p")


def _is_finished(exp_dir: str) -> bool:
    log_path = os.path.join(exp_dir, "train.log")
    if not os.path.exists(log_path):
        return False
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "Training done." in line:
                return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratios", nargs="+", type=float, default=[1.0, 3.0, 5.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument(
        "--split-level",
        choices=["pair", "sentence"],
        default="sentence",
        help="Train/dev split level (recommended: sentence).",
    )
    parser.add_argument("--no-skip", action="store_true")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=== BERT-RE neg_pos_ratio grid ===")
    print("Ratios:", args.ratios)
    print("Seeds:", args.seeds)
    print("split_level:", args.split_level)
    print("Timestamp:", ts)

    total = len(args.ratios) * len(args.seeds)
    done = 0

    for r in args.ratios:
        for seed in args.seeds:
            exp_name = f"grid_negpos_r{_format_r(r)}_seed{seed}_split{args.split_level}_{ts}"
            exp_dir = os.path.join(base_dir, "experiments", exp_name)
            os.makedirs(exp_dir, exist_ok=True)

            if (not args.no_skip) and _is_finished(exp_dir):
                done += 1
                print(f"SKIP finished: neg_pos_ratio={r}, seed={seed} -> {exp_name}")
                continue

            env = os.environ.copy()
            env.setdefault("OMP_NUM_THREADS", "1")
            env["BERT_RE_NEG_POS_RATIO"] = str(r)
            env["BERT_RE_SEED"] = str(seed)
            env["BERT_RE_SPLIT_LEVEL"] = args.split_level
            env["BERT_RE_EXP_DIR"] = exp_dir

            done += 1
            print(f"\n--- RUN ({done}/{total}) neg_pos_ratio={r}, seed={seed} -> {exp_dir} ---")
            cmd = [sys.executable, os.path.join(base_dir, "run.py")]
            subprocess.run(cmd, cwd=base_dir, env=env, check=True)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

