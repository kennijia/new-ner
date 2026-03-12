#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import random
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample triples for manual audit")
    parser.add_argument("--input", default="data/my/kg/processed/triples_clean.jsonl", help="Input triples JSONL")
    parser.add_argument("--output", default="data/my/kg/processed/sample_audit_100.csv", help="Output audit CSV")
    parser.add_argument("--total", type=int, default=100, help="Total sample size")
    parser.add_argument("--trigger", type=int, default=80, help="Number of relation=触发 samples")
    parser.add_argument("--cause", type=int, default=20, help="Number of relation=导致 samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def sample_rows(rows: List[Dict], relation: str, n: int, rng: random.Random) -> List[Dict]:
    pool = [x for x in rows if str(x.get("relation", "")).strip() == relation]
    if n <= 0:
        return []
    if len(pool) <= n:
        rng.shuffle(pool)
        return pool
    return rng.sample(pool, n)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    rows = load_jsonl(args.input)

    if args.total != args.trigger + args.cause:
        raise ValueError("--total must equal --trigger + --cause")

    picked = []
    picked.extend(sample_rows(rows, "触发", args.trigger, rng))
    picked.extend(sample_rows(rows, "导致", args.cause, rng))

    if len(picked) < args.total:
        rest = [x for x in rows if x not in picked]
        rng.shuffle(rest)
        picked.extend(rest[: args.total - len(picked)])

    rng.shuffle(picked)

    fieldnames = [
        "sample_id",
        "source_index",
        "head",
        "relation",
        "tail",
        "evidence",
        "check_relation",
        "check_direction",
        "check_traceable",
        "check_complete",
        "check_non_causal_noise",
        "error_type",
        "fix_suggestion",
        "reviewer",
    ]

    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(picked[: args.total], 1):
            writer.writerow(
                {
                    "sample_id": i,
                    "source_index": row.get("source_index", ""),
                    "head": row.get("head", ""),
                    "relation": row.get("relation", ""),
                    "tail": row.get("tail", ""),
                    "evidence": row.get("evidence", ""),
                    "check_relation": "",
                    "check_direction": "",
                    "check_traceable": "",
                    "check_complete": "",
                    "check_non_causal_noise": "",
                    "error_type": "",
                    "fix_suggestion": "",
                    "reviewer": "",
                }
            )

    print(f"input_rows={len(rows)}")
    print(f"sample_written={min(args.total, len(picked))}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
