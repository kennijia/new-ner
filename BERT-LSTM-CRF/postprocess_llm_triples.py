#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Post-process LLM extracted triples for KG construction.

Input JSONL fields (expected):
- head, relation, tail, confidence, evidence, source_index
- optional: head_anchors, tail_anchors

Outputs (default in output_dir):
- triples_clean.jsonl
- neo4j_nodes.csv
- neo4j_rels.csv
- quality_report.json
"""

import argparse
import csv
import json
import os
from collections import Counter
from typing import Dict, List, Tuple


CAUSAL_RELATIONS = {"触发", "导致"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-process LLM triples for KG")
    parser.add_argument("--input", default="data/my/kg/llm_triples.jsonl", help="Input triples JSONL")
    parser.add_argument("--output_dir", default="data/my/kg/processed", help="Output directory")
    parser.add_argument("--min_conf", type=float, default=0.85, help="Minimum confidence threshold")
    parser.add_argument(
        "--relation_scope",
        choices=["causal", "all"],
        default="causal",
        help="Relation filtering scope",
    )
    parser.add_argument("--min_phrase_len", type=int, default=3, help="Min character length for head/tail")
    parser.add_argument("--dedup_mode", choices=["ht", "htr"], default="htr", help="Dedup key mode")
    parser.add_argument("--keep_topk_per_pair", type=int, default=1, help="Keep top-K by confidence per dedup key")
    parser.add_argument("--sample_bad", type=int, default=20, help="Max bad-case samples in report")
    return parser.parse_args()


def load_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                rows.append({"_parse_error": f"line={line_no}, err={exc}", "_raw": line})
    return rows


def as_conf(v) -> float:
    try:
        c = float(v)
    except Exception:
        c = 0.0
    if c < 0:
        return 0.0
    if c > 1:
        return 1.0
    return round(c, 4)


def normalize_row(row: Dict) -> Dict:
    head = str(row.get("head", "")).strip()
    relation = str(row.get("relation", "")).strip()
    tail = str(row.get("tail", "")).strip()
    confidence = as_conf(row.get("confidence", 0.0))

    head_anchors = row.get("head_anchors", [])
    if not isinstance(head_anchors, list):
        head_anchors = [str(head_anchors)] if str(head_anchors).strip() else []
    head_anchors = [str(x).strip() for x in head_anchors if str(x).strip()]

    tail_anchors = row.get("tail_anchors", [])
    if not isinstance(tail_anchors, list):
        tail_anchors = [str(tail_anchors)] if str(tail_anchors).strip() else []
    tail_anchors = [str(x).strip() for x in tail_anchors if str(x).strip()]

    evidence = str(row.get("evidence", "")).strip()

    source_index = row.get("source_index")
    if isinstance(source_index, bool):
        source_index = None
    if not isinstance(source_index, int):
        source_index = None

    return {
        "head": head,
        "relation": relation,
        "tail": tail,
        "head_anchors": head_anchors,
        "tail_anchors": tail_anchors,
        "evidence": evidence,
        "confidence": confidence,
        "source_index": source_index,
    }


def validate_row(row: Dict, args: argparse.Namespace) -> Tuple[bool, str]:
    if not row.get("head") or not row.get("tail") or not row.get("relation"):
        return False, "missing_head_tail_relation"
    if row["head"] == row["tail"]:
        return False, "self_loop"
    if len(row["head"]) < args.min_phrase_len or len(row["tail"]) < args.min_phrase_len:
        return False, "too_short_phrase"
    if row["confidence"] < args.min_conf:
        return False, "low_confidence"
    if args.relation_scope == "causal" and row["relation"] not in CAUSAL_RELATIONS:
        return False, "non_causal_relation"
    if row.get("evidence"):
        if row["head"] not in row["evidence"] or row["tail"] not in row["evidence"]:
            return False, "not_in_evidence"
    return True, "ok"


def dedup_rows(rows: List[Dict], args: argparse.Namespace) -> List[Dict]:
    grouped: Dict[Tuple[str, ...], List[Dict]] = {}
    for row in rows:
        if args.dedup_mode == "ht":
            key = (row["head"], row["tail"])
        else:
            key = (row["head"], row["relation"], row["tail"])
        grouped.setdefault(key, []).append(row)

    kept: List[Dict] = []
    for _, group in grouped.items():
        group.sort(key=lambda x: (x["confidence"], len(x.get("evidence", ""))), reverse=True)
        kept.extend(group[: max(1, args.keep_topk_per_pair)])
    return kept


def write_jsonl(path: str, rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_neo4j_csv(out_dir: str, rows: List[Dict]) -> Tuple[int, int]:
    node_map: Dict[str, str] = {}
    nodes: List[Dict] = []

    def node_id(name: str) -> str:
        if name not in node_map:
            node_map[name] = f"n{len(node_map) + 1}"
            nodes.append({":ID": node_map[name], "name": name, ":LABEL": "Event"})
        return node_map[name]

    rels: List[Dict] = []
    for row in rows:
        sid = node_id(row["head"])
        tid = node_id(row["tail"])
        rels.append(
            {
                ":START_ID": sid,
                ":END_ID": tid,
                ":TYPE": row["relation"],
                "confidence": row["confidence"],
                "source_index": "" if row["source_index"] is None else row["source_index"],
                "evidence": row.get("evidence", ""),
            }
        )

    nodes_path = os.path.join(out_dir, "neo4j_nodes.csv")
    rels_path = os.path.join(out_dir, "neo4j_rels.csv")

    with open(nodes_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[":ID", "name", ":LABEL"])
        writer.writeheader()
        writer.writerows(nodes)

    with open(rels_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[":START_ID", ":END_ID", ":TYPE", "confidence", "source_index", "evidence"],
        )
        writer.writeheader()
        writer.writerows(rels)

    return len(nodes), len(rels)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    raw = load_jsonl(args.input)
    parse_errors = [x for x in raw if x.get("_parse_error")]
    parsed = [normalize_row(x) for x in raw if not x.get("_parse_error")]

    bad_samples: List[Dict] = []
    reason_counter: Counter = Counter()
    valid_rows: List[Dict] = []

    for row in parsed:
        ok, reason = validate_row(row, args)
        if ok:
            valid_rows.append(row)
        else:
            reason_counter[reason] += 1
            if len(bad_samples) < args.sample_bad:
                bad_samples.append(
                    {
                        "reason": reason,
                        "head": row.get("head", ""),
                        "relation": row.get("relation", ""),
                        "tail": row.get("tail", ""),
                        "confidence": row.get("confidence", 0.0),
                        "source_index": row.get("source_index"),
                    }
                )

    deduped = dedup_rows(valid_rows, args)

    triples_out = os.path.join(args.output_dir, "triples_clean.jsonl")
    write_jsonl(triples_out, deduped)

    node_count, rel_count = write_neo4j_csv(args.output_dir, deduped)

    relation_dist = Counter([x["relation"] for x in deduped])
    conf_bins = {"<0.85": 0, "0.85-0.9": 0, "0.9-0.95": 0, ">=0.95": 0}
    for x in deduped:
        c = x["confidence"]
        if c < 0.85:
            conf_bins["<0.85"] += 1
        elif c < 0.9:
            conf_bins["0.85-0.9"] += 1
        elif c < 0.95:
            conf_bins["0.9-0.95"] += 1
        else:
            conf_bins[">=0.95"] += 1

    report = {
        "input": args.input,
        "output_dir": args.output_dir,
        "config": {
            "min_conf": args.min_conf,
            "relation_scope": args.relation_scope,
            "min_phrase_len": args.min_phrase_len,
            "dedup_mode": args.dedup_mode,
            "keep_topk_per_pair": args.keep_topk_per_pair,
        },
        "counts": {
            "raw_rows": len(raw),
            "parse_errors": len(parse_errors),
            "parsed_rows": len(parsed),
            "valid_rows_before_dedup": len(valid_rows),
            "rows_after_dedup": len(deduped),
            "neo4j_nodes": node_count,
            "neo4j_rels": rel_count,
        },
        "drop_reason_distribution": dict(reason_counter),
        "relation_distribution": dict(relation_dist),
        "confidence_bins": conf_bins,
        "bad_samples": bad_samples,
    }

    report_path = os.path.join(args.output_dir, "quality_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=== Postprocess Done ===")
    print(f"input_rows={len(raw)} parsed={len(parsed)} parse_errors={len(parse_errors)}")
    print(f"valid_before_dedup={len(valid_rows)} after_dedup={len(deduped)}")
    print(f"nodes={node_count} rels={rel_count}")
    print(f"triples_clean={triples_out}")
    print(f"quality_report={report_path}")


if __name__ == "__main__":
    main()
