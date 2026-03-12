#!/usr/bin/env python3
"""Collect grid search results from BERT-CRF/experiments.

Scans experiment folders like:
  experiments/grid_dice_w0p3_YYYYMMDD_HHMMSS/
  experiments/grid_dice_w0p3_seed42_YYYYMMDD_HHMMSS/

and extracts:
  - weight
  - seed (if present)
  - Best val f1
  - test f1

Outputs a TSV to stdout (or to --output).

Optionally, can also print a grouped-by-weight summary across seeds (mean/std).
"""

import argparse
import glob
import os
import re
from typing import Dict, List, Optional, Tuple


def parse_weight_and_seed_from_dirname(name: str) -> Tuple[Optional[float], Optional[int]]:
    """Parse weight/seed from experiment directory name.

    Supported:
      - grid_dice_w0p3_YYYY...
      - grid_dice_w0p3_seed42_YYYY...
    """
    m = re.search(r"grid_dice_w([^_]+)(?:_seed(\d+))?_", name)
    if not m:
        return None, None
    w = float(m.group(1).replace("p", "."))
    seed = int(m.group(2)) if m.group(2) is not None else None
    return w, seed


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
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, var ** 0.5


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiments-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments"),
    )
    parser.add_argument(
        "--pattern",
        default="grid_dice_w*_seed*_*",
        help="Glob pattern under experiments-dir",
    )
    parser.add_argument("--output", default="", help="Write TSV to this file; default stdout")
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Also print a grouped-by-weight summary (mean/std across seeds) to stdout.",
    )
    args = parser.parse_args()

    rows = []
    for exp_dir in glob.glob(os.path.join(args.experiments_dir, args.pattern)):
        log_path = os.path.join(exp_dir, "train.log")
        if not os.path.exists(log_path):
            continue

        w, seed = parse_weight_and_seed_from_dirname(os.path.basename(exp_dir))
        best, test = parse_log(log_path)
        rows.append((w, seed, best, test, os.path.basename(exp_dir)))

    rows.sort(
        key=lambda x: (
            x[0] is None,
            x[0] if x[0] is not None else 0.0,
            x[1] is None,
            x[1] if x[1] is not None else 0,
        )
    )

    lines = ["weight\tseed\tbest_dev_f1\ttest_f1\texp_dir"]
    for w, seed, best, test, name in rows:
        lines.append(f"{w}\t{seed}\t{best}\t{test}\t{name}")

    out = "\n".join(lines) + "\n"
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out)
    else:
        print(out, end="")

    if args.summary:
        grouped_best: Dict[float, List[float]] = {}
        grouped_test: Dict[float, List[float]] = {}
        for w, seed, best, test, _name in rows:
            if w is None:
                continue
            if best is not None:
                grouped_best.setdefault(w, []).append(best)
            if test is not None:
                grouped_test.setdefault(w, []).append(test)

        print("\n# Summary (grouped by weight; mean±std across seeds)")
        print("weight\tn_best\tbest_dev_f1_mean\tbest_dev_f1_std\tn_test\ttest_f1_mean\ttest_f1_std")
        for w in sorted(set(list(grouped_best.keys()) + list(grouped_test.keys()))):
            best_mean, best_std = _mean_std(grouped_best.get(w, []))
            test_mean, test_std = _mean_std(grouped_test.get(w, []))
            print(
                f"{w}\t{len(grouped_best.get(w, []))}\t{best_mean}\t{best_std}"
                f"\t{len(grouped_test.get(w, []))}\t{test_mean}\t{test_std}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
