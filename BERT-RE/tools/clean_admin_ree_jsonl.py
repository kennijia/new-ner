#!/usr/bin/env python3
"""Clean and validate admin-ree.jsonl.

Reads JSONL and keeps only well-formed JSON objects that contain the keys
{id, text, entities, relations}. Writes a cleaned JSONL plus a report JSON.

This is intentionally strict to make experiments reproducible for a thesis.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any


REQUIRED_KEYS = ("id", "text", "entities", "relations")


def _is_valid_example(obj: Any) -> tuple[bool, str]:
    if not isinstance(obj, dict):
        return False, "not_a_dict"
    for k in REQUIRED_KEYS:
        if k not in obj:
            return False, f"missing_key:{k}"
    if not isinstance(obj.get("entities"), list):
        return False, "entities_not_list"
    if not isinstance(obj.get("relations"), list):
        return False, "relations_not_list"
    return True, "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="admin-ree.jsonl")
    ap.add_argument("--output", default="admin-ree.cleaned.jsonl")
    ap.add_argument("--report", default="admin-ree.cleaned.report.json")
    args = ap.parse_args()

    total_nonempty = 0
    kept = 0
    bad_json = 0
    bad_schema = 0

    rel_counter: Counter[str] = Counter()
    ent_counter: Counter[str] = Counter()
    empty_rel_sent = 0
    bad_lines_first50: list[dict[str, Any]] = []

    with open(args.input, "r", encoding="utf-8") as fin, open(
        args.output, "w", encoding="utf-8"
    ) as fout:
        for line_no, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            total_nonempty += 1

            try:
                obj = json.loads(line)
            except Exception as e:
                bad_json += 1
                if len(bad_lines_first50) < 50:
                    bad_lines_first50.append(
                        {
                            "line": line_no,
                            "reason": "json_error",
                            "detail": str(e)[:160],
                            "snippet": line[:200],
                        }
                    )
                continue

            ok, reason = _is_valid_example(obj)
            if not ok:
                bad_schema += 1
                if len(bad_lines_first50) < 50:
                    bad_lines_first50.append(
                        {
                            "line": line_no,
                            "reason": reason,
                            "snippet": line[:200],
                        }
                    )
                continue

            # stats
            for ent in obj["entities"]:
                if isinstance(ent, dict) and "label" in ent:
                    ent_counter[str(ent["label"])] += 1
            if len(obj["relations"]) == 0:
                empty_rel_sent += 1
            for rel in obj["relations"]:
                if isinstance(rel, dict) and "type" in rel:
                    rel_counter[str(rel["type"])] += 1

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1

    report = {
        "input": args.input,
        "output": args.output,
        "total_nonempty_lines": total_nonempty,
        "kept": kept,
        "bad_json": bad_json,
        "bad_schema": bad_schema,
        "empty_rel_sentences": empty_rel_sent,
        "entity_label_counts": dict(ent_counter.most_common()),
        "relation_type_counts": dict(rel_counter.most_common()),
        "bad_lines_first50": bad_lines_first50,
    }

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
