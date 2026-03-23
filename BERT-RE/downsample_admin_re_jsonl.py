from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def is_positive(ex: Dict[str, Any]) -> bool:
    """A sentence is positive if it contains at least one relation."""
    return bool(ex.get("relations"))


def rel_type_counter(items: List[Dict[str, Any]]) -> Counter:
    c = Counter()
    for ex in items:
        if is_positive(ex):
            for r in ex.get("relations", []):
                c[r.get("type", "<MISSING>")] += 1
        else:
            c["NoRelation_sentence"] += 1
    return c


def main() -> int:
    ap = argparse.ArgumentParser(description="Downsample NoRelation sentences in admin-re.jsonl")
    ap.add_argument("--input", required=True, help="Input JSONL")
    ap.add_argument("--output", required=True, help="Output JSONL")
    ap.add_argument("--seed", type=int, default=42)

    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--sentence_keep_ratio",
        type=float,
        default=None,
        help="Keep this fraction of negative sentences (0~1). e.g. 0.2",
    )
    g.add_argument(
        "--target_neg_pos_ratio",
        type=float,
        default=None,
        help="Keep negatives so that neg_sent:pos_sent ~= ratio. e.g. 1.0. Use -1 to keep all negatives.",
    )

    args = ap.parse_args()
    rnd = random.Random(args.seed)

    in_path = Path(args.input)
    out_path = Path(args.output)

    examples: List[Dict[str, Any]] = []
    with in_path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {ln}: {e}") from e

    pos = [ex for ex in examples if is_positive(ex)]
    neg = [ex for ex in examples if not is_positive(ex)]

    if args.sentence_keep_ratio is not None:
        r = float(args.sentence_keep_ratio)
        if not (0.0 <= r <= 1.0):
            raise ValueError("sentence_keep_ratio must be in [0,1]")
        rnd.shuffle(neg)
        keep_neg = neg[: int(round(len(neg) * r))]
    else:
        ratio = float(args.target_neg_pos_ratio)
        if ratio < 0:
            keep_neg = neg
        else:
            target_neg = int(round(len(pos) * ratio))
            rnd.shuffle(neg)
            keep_neg = neg[: min(len(neg), target_neg)]

    out = pos + keep_neg
    rnd.shuffle(out)

    report = {
        "input": str(in_path),
        "output": str(out_path),
        "seed": args.seed,
        "input_total": len(examples),
        "input_pos_sent": len(pos),
        "input_neg_sent": len(neg),
        "output_total": len(out),
        "output_pos_sent": sum(1 for x in out if is_positive(x)),
        "output_neg_sent": sum(1 for x in out if not is_positive(x)),
        "stats_input": dict(rel_type_counter(examples)),
        "stats_output": dict(rel_type_counter(out)),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for ex in out:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with out_path.with_suffix(out_path.suffix + ".report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
