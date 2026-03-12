#!/usr/bin/env python3
"""Grid search for Dice loss weight (BERT-CRF).

Runs `BERT-CRF/run.py` with different `dice_loss_weight` values.
Fixed during runs:
  - use_dice_loss=True
  - dice_exclude_o=True

Supports multiple seeds per weight for stability.

Examples:
  python grid_dice_weight.py
  python grid_dice_weight.py --weights 0.1 0.3 0.5 1.0 --seeds 42 43 44
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime


def _format_w(w: float) -> str:
    s = (f"{w:.6f}").rstrip("0").rstrip(".")
    return s.replace(".", "p")


def _is_finished(exp_dir: str) -> bool:
    log_path = os.path.join(exp_dir, "train.log")
    if not os.path.exists(log_path):
        return False
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "test loss:" in line and "f1 score:" in line:
                return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=[0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0],
        help="List of dice_loss_weight values to try.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44],
        help="Random seeds to repeat each weight (for stability).",
    )
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Any additional args to pass to run.py (if your run.py supports CLI args).",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Do not skip runs that appear finished (have test f1 in train.log).",
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=== Dice loss weight grid ===")
    print("Fixed: use_dice_loss=True, dice_exclude_o=True")
    print("Weights:", args.weights)
    print("Seeds:", args.seeds)
    print("Timestamp:", ts)

    total = len(args.weights) * len(args.seeds)
    done = 0

    for w in args.weights:
        for seed in args.seeds:
            exp_name = f"grid_dice_w{_format_w(w)}_seed{seed}_{ts}"
            exp_dir = os.path.join(base_dir, "experiments", exp_name)
            os.makedirs(exp_dir, exist_ok=True)

            if (not args.no_skip) and _is_finished(exp_dir):
                done += 1
                print(f"SKIP finished: dice_loss_weight={w}, seed={seed} -> {exp_name}")
                continue

            env = os.environ.copy()
            env["BERT_CRF_USE_DICE_LOSS"] = "1"
            env["BERT_CRF_DICE_EXCLUDE_O"] = "1"
            env["BERT_CRF_DICE_LOSS_WEIGHT"] = str(w)
            env["BERT_CRF_EXP_DIR"] = exp_dir
            env["BERT_CRF_SEED"] = str(seed)

            done += 1
            print(f"\n--- RUN ({done}/{total}) dice_loss_weight={w}, seed={seed} -> {exp_dir} ---")
            cmd = [sys.executable, os.path.join(base_dir, "run.py"), *args.extra_args]
            subprocess.run(cmd, cwd=base_dir, env=env, check=True)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
