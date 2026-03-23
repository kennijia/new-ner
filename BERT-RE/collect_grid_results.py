#!/usr/bin/env python3
"""Collect BERT-RE grid results.

Parses `train.log` files to extract the best dev micro-F1 per experiment.

It looks for lines like:
  Saved best checkpoint: ... (best_f1=0.2261)

Usage:
  cd /root/msy/ner/BERT-RE
  python collect_grid_results.py --root experiments
  python collect_grid_results.py --root experiments --summary
"""

import argparse
import os
import re
from collections import defaultdict
from statistics import mean, pstdev


BEST_RE = re.compile(r"best_f1=([0-9.]+)")


def _parse_best_f1(log_path: str):
    best = None
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = BEST_RE.search(line)
            if m:
                best = float(m.group(1))
    return best


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="experiments")
    p.add_argument("--summary", action="store_true", help="Group by neg_pos_ratio and print mean/std.")
    args = p.parse_args()

    root = os.path.abspath(args.root)
    rows = []

    for dirpath, dirnames, filenames in os.walk(root):
        if "train.log" not in filenames:
            continue
        log_path = os.path.join(dirpath, "train.log")
        best = _parse_best_f1(log_path)
        if best is None:
            continue

        exp = os.path.relpath(dirpath, root)
        # heuristics: parse ratio/seed from folder name
        ratio = None
        seed = None
        m = re.search(r"_r([0-9p]+)_", exp)
        if m:
            ratio = float(m.group(1).replace("p", "."))
        m = re.search(r"seed(\d+)", exp)
        if m:
            seed = int(m.group(1))
        rows.append((exp, ratio, seed, best))

    rows.sort(key=lambda x: (x[1] is None, x[1], x[2] is None, x[2], x[0]))

    print("exp\tratio\tseed\tbest_dev_micro_f1")
    for exp, ratio, seed, best in rows:
        print(f"{exp}\t{'' if ratio is None else ratio}\t{'' if seed is None else seed}\t{best:.6f}")

    if args.summary:
        grouped = defaultdict(list)
        for exp, ratio, seed, best in rows:
            if ratio is None:
                continue
            grouped[ratio].append(best)
        print("\n# Summary (mean±std across seeds)")
        for ratio in sorted(grouped.keys()):
            vals = grouped[ratio]
            mu = mean(vals)
            sd = pstdev(vals) if len(vals) > 1 else 0.0
            print(f"ratio={ratio}: mean={mu:.6f} std={sd:.6f} (n={len(vals)})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

