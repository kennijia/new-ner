#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split RE labeled JSONL into train/dev/test with stratified sampling by label.
Input format: {"text": ..., "label": "CAUSES"} or {"text": ..., "label": ["CAUSES"]}
Output files: train.jsonl, dev.jsonl, test.jsonl
"""

import argparse
import json
import random
from collections import defaultdict, Counter
from typing import Dict, List


def parse_args():
    parser = argparse.ArgumentParser(description="Split RE JSONL into train/dev/test")
    parser.add_argument("--input", required=True, help="Input labeled JSONL")
    parser.add_argument("--train_out", required=True, help="Output train JSONL")
    parser.add_argument("--dev_out", required=True, help="Output dev JSONL")
    parser.add_argument("--test_out", required=True, help="Output test JSONL")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train ratio, default 0.8")
    parser.add_argument("--dev_ratio", type=float, default=0.1, help="Dev ratio, default 0.1")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test ratio, default 0.1")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def normalize_label(label):
    if isinstance(label, list):
        if len(label) == 0:
            return "NA"
        return str(label[0])
    if label is None:
        return "NA"
    return str(label)


def read_data(path: str) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text", "")
            if not text:
                continue
            label = normalize_label(obj.get("label"))
            records.append({"text": text, "label": label})
    return records


def split_stratified(records: List[Dict], train_ratio: float, dev_ratio: float, test_ratio: float, seed: int):
    if abs((train_ratio + dev_ratio + test_ratio) - 1.0) > 1e-8:
        raise ValueError("train_ratio + dev_ratio + test_ratio must equal 1.0")

    rng = random.Random(seed)
    by_label = defaultdict(list)
    for r in records:
        by_label[r["label"]].append(r)

    train, dev, test = [], [], []

    for label, items in by_label.items():
        rng.shuffle(items)
        n = len(items)

        n_train = int(n * train_ratio)
        n_dev = int(n * dev_ratio)
        n_test = n - n_train - n_dev

        if n >= 3:
            if n_train == 0:
                n_train = 1
                n_test = max(0, n_test - 1)
            if n_dev == 0:
                n_dev = 1
                n_test = max(0, n_test - 1)
            if n_test == 0:
                n_test = 1
                if n_train > n_dev and n_train > 1:
                    n_train -= 1
                elif n_dev > 1:
                    n_dev -= 1

        train.extend(items[:n_train])
        dev.extend(items[n_train:n_train + n_dev])
        test.extend(items[n_train + n_dev:])

    rng.shuffle(train)
    rng.shuffle(dev)
    rng.shuffle(test)
    return train, dev, test


def write_jsonl(path: str, records: List[Dict]):
    with open(path, "w", encoding="utf-8") as fout:
        for r in records:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    records = read_data(args.input)
    train, dev, test = split_stratified(
        records,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    write_jsonl(args.train_out, train)
    write_jsonl(args.dev_out, dev)
    write_jsonl(args.test_out, test)

    print("Total:", len(records))
    print("Train:", len(train), Counter([r["label"] for r in train]))
    print("Dev:", len(dev), Counter([r["label"] for r in dev]))
    print("Test:", len(test), Counter([r["label"] for r in test]))


if __name__ == "__main__":
    main()
