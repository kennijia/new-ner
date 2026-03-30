#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parse confusion snippets from BERT-RE experiment stdout.log files.

This repo's BERT-RE training script (`BERT-RE/train.py`) prints only a *partial*
confusion summary by default, e.g.:

  Dev confusion (gold=Attribute_of, n=10): NoRelation=10

So from logs we can reliably recover only:
  - gold label name
  - gold support n
  - a (possibly partial) predicted-label->count mapping that was printed

This script extracts those rows from all matching logs and writes a CSV.

Usage (PowerShell):
  python .\BERT-RE\tools\parse_confusion_from_logs.py --out .\BERT-RE\tools\confusion_from_logs.csv

Notes
-----
- The output is not a full confusion matrix unless the logs contain full rows.
- The same run prints this line every epoch; keep_all will keep all rows.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


CONF_RE = re.compile(
    r"Dev confusion \(gold=(?P<gold>[^,]+), n=(?P<n>\d+)\): (?P<rest>.*)$"
)
PAIR_RE = re.compile(r"(?P<label>[^=,\s]+)=(?P<count>\d+)")


@dataclass
class ConfRow:
    log_path: str
    seed: Optional[int]
    mode: str
    epoch: Optional[int]
    gold: str
    n: int
    pred_counts: Dict[str, int]


def infer_mode_from_path(path: str) -> str:
    p = path.replace("\\", "/").lower()
    if "/dynfilter_" in p:
        return "DynFilter"
    if "/nofilter_" in p:
        return "NoFilter"
    return "Unknown"


def infer_seed_from_path(path: str) -> Optional[int]:
    m = re.search(r"seed(\d+)", os.path.basename(os.path.dirname(path)))
    if not m:
        m = re.search(r"seed(\d+)", path)
    return int(m.group(1)) if m else None


def parse_epoch_from_prev_lines(prev_lines: List[str]) -> Optional[int]:
    # Look backward for: "Epoch 9/10 | ..."
    for line in reversed(prev_lines[-5:]):
        m = re.search(r"Epoch\s+(\d+)/\d+\b", line)
        if m:
            return int(m.group(1))
    return None


def parse_pred_counts(rest: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for m in PAIR_RE.finditer(rest):
        out[m.group("label")] = int(m.group("count"))
    return out


def parse_log(log_path: str) -> List[ConfRow]:
    rows: List[ConfRow] = []
    seed = infer_seed_from_path(log_path)
    mode = infer_mode_from_path(log_path)

    prev: List[str] = []
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.rstrip("\n")
            m = CONF_RE.search(line)
            if m:
                gold = m.group("gold").strip()
                n = int(m.group("n"))
                rest = (m.group("rest") or "").strip()
                pred_counts = parse_pred_counts(rest)
                epoch = parse_epoch_from_prev_lines(prev)
                rows.append(
                    ConfRow(
                        log_path=log_path,
                        seed=seed,
                        mode=mode,
                        epoch=epoch,
                        gold=gold,
                        n=n,
                        pred_counts=pred_counts,
                    )
                )
            prev.append(line)
    return rows


def normalise_relpath(path: str, base: str) -> str:
    try:
        return os.path.relpath(path, base)
    except Exception:
        return path


def select_best_epoch(rows: List[ConfRow]) -> List[ConfRow]:
    """Best-effort: keep the last confusion row (often best model saved elsewhere).

    Since logs don't mark 'best epoch' for confusion, we default to the last
    occurrence per (log, gold).
    """
    key2row: Dict[Tuple[str, str], ConfRow] = {}
    for r in rows:
        key2row[(r.log_path, r.gold)] = r
    return list(key2row.values())


def main() -> int:
    ap = argparse.ArgumentParser(description="Parse Dev confusion snippets from BERT-RE stdout logs")
    ap.add_argument(
        "--root",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments"),
        help="Root dir to search (default: BERT-RE/experiments)",
    )
    ap.add_argument(
        "--pattern",
        default="stdout.log",
        help="Filename to match (default: stdout.log)",
    )
    ap.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "confusion_from_logs.csv"),
        help="Output CSV path",
    )
    ap.add_argument(
        "--keep_all",
        action="store_true",
        help="Keep all epochs (default: only keep last row per (log,gold))",
    )
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    found_logs: List[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn == args.pattern:
                found_logs.append(os.path.join(dirpath, fn))

    all_rows: List[ConfRow] = []
    for lp in sorted(found_logs):
        all_rows.extend(parse_log(lp))

    if not args.keep_all:
        all_rows = select_best_epoch(all_rows)

    # Collect all predicted labels that ever appeared
    pred_labels = sorted({lbl for r in all_rows for lbl in r.pred_counts.keys()})

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "log_path",
            "seed",
            "mode",
            "epoch",
            "gold",
            "gold_support",
            *[f"pred_{p}" for p in pred_labels],
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in sorted(all_rows, key=lambda x: (x.seed is None, x.seed or -1, x.mode, x.gold, x.epoch or -1)):
            row = {
                "log_path": normalise_relpath(r.log_path, base=os.path.abspath(os.path.join(root, os.pardir))),
                "seed": "" if r.seed is None else r.seed,
                "mode": r.mode,
                "epoch": "" if r.epoch is None else r.epoch,
                "gold": r.gold,
                "gold_support": r.n,
            }
            for p in pred_labels:
                row[f"pred_{p}"] = r.pred_counts.get(p, 0)
            w.writerow(row)

    print(f"logs={len(found_logs)} confusion_rows={len(all_rows)} out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
