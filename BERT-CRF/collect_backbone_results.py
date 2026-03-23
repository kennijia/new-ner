#!/usr/bin/env python3
"""Collect backbone grid results from BERT-CRF/experiments.

Scans experiment folders like:
  experiments/grid_backbone_chinese-macbert-base_seed42_YYYYMMDD_HHMMSS/

Extracts:
  - backbone
  - seed
  - best dev f1
  - test f1

Outputs TSV to stdout (or --output).
Optional --summary prints grouped mean/std across seeds per backbone.
"""

import argparse
import glob
import os
import re
from typing import Dict, List, Optional, Tuple


def parse_backbone_seed_from_dirname(name: str) -> Tuple[Optional[str], Optional[int]]:
    m = re.search(r"grid_backbone_(.+?)_seed(\d+)_", name)
    if not m:
        return None, None
    backbone = m.group(1)
    seed = int(m.group(2))
    return backbone, seed


def parse_log(log_path: str) -> Tuple[Optional[float], Optional[float]]:
    best = None
    test = None
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "Best val f1:" in line:
                try:
                    best = float(line.strip().split("Best val f1:")[-1])
                except ValueError:
                    pass
            if "test loss:" in line and "f1 score:" in line:
                try:
                    test = float(line.strip().split("f1 score:")[-1])
                except ValueError:
                    pass
    return best, test


def _mean_std(xs: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not xs:
        return None, None
    if len(xs) == 1:
        return xs[0], 0.0
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
    return mean, var ** 0.5


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiments-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments"),
    )
    parser.add_argument(
        "--pattern",
        default="grid_backbone_*_seed*_*",
        help="Glob pattern under experiments-dir",
    )
    parser.add_argument("--output", default="", help="Write TSV to this file; default stdout")
    parser.add_argument("--summary", action="store_true", help="Print grouped mean/std by backbone")
    args = parser.parse_args()

    rows = []
    for exp_dir in glob.glob(os.path.join(args.experiments_dir, args.pattern)):
        log_path = os.path.join(exp_dir, "train.log")
        if not os.path.exists(log_path):
            continue

        backbone, seed = parse_backbone_seed_from_dirname(os.path.basename(exp_dir))
        best, test = parse_log(log_path)
        rows.append((backbone, seed, best, test, os.path.basename(exp_dir)))

    rows.sort(
        key=lambda x: (
            x[0] is None,
            x[0] if x[0] is not None else "",
            x[1] if x[1] is not None else -1,
        )
    )

    lines = ["backbone\tseed\tbest_dev_f1\ttest_f1\texp_dir"]
    for backbone, seed, best, test, name in rows:
        lines.append(f"{backbone}\t{seed}\t{best}\t{test}\t{name}")

    out = "\n".join(lines) + "\n"
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out)
    else:
        print(out, end="")

    if args.summary:
        grouped_best: Dict[str, List[float]] = {}
        grouped_test: Dict[str, List[float]] = {}
        for backbone, _seed, best, test, _name in rows:
            if backbone is None:
                continue
            if best is not None:
                grouped_best.setdefault(backbone, []).append(best)
            if test is not None:
                grouped_test.setdefault(backbone, []).append(test)

        keys = sorted(set(grouped_best.keys()) | set(grouped_test.keys()))
        print("\n# Summary (grouped by backbone; mean±std across seeds)")
        print("backbone\tn_best\tbest_dev_f1_mean\tbest_dev_f1_std\tn_test\ttest_f1_mean\ttest_f1_std")
        for k in keys:
            best_mean, best_std = _mean_std(grouped_best.get(k, []))
            test_mean, test_std = _mean_std(grouped_test.get(k, []))
            print(
                f"{k}\t{len(grouped_best.get(k, []))}\t{best_mean}\t{best_std}"
                f"\t{len(grouped_test.get(k, []))}\t{test_mean}\t{test_std}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
