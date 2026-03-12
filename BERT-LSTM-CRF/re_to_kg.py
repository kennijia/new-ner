#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert RE outputs to KG triples and Neo4j CSV files.

Input JSONL example:
{"text": "[HEAD]水位超过警戒线[/HEAD]时，需立即[TAIL]启动应急响应[/TAIL]。", "label": "CAUSES"}
or
{"text": "...", "pred_label": "CAUSES", "confidence": 0.92}

Outputs:
- triples.jsonl: {head, relation, tail, evidence, confidence}
- neo4j_nodes.csv: :ID,name,:LABEL
- neo4j_rels.csv: :START_ID,:END_ID,:TYPE,evidence,confidence
"""

import argparse
import csv
import json
import os
import re
from typing import Dict, List, Optional, Tuple


HEAD_RE = re.compile(r"\[HEAD\](.*?)\[/HEAD\]")
TAIL_RE = re.compile(r"\[TAIL\](.*?)\[/TAIL\]")
TAG_RE = re.compile(r"\[/?(?:HEAD|TAIL)\]")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert RE JSONL to KG triples")
    parser.add_argument("--input", required=True, help="Input RE JSONL")
    parser.add_argument("--output_dir", default="data/my/kg", help="Output directory")
    parser.add_argument("--label_field", default="label", help="Label field name (e.g., label/pred_label)")
    parser.add_argument("--confidence_field", default="confidence", help="Confidence field name")
    parser.add_argument("--skip_na", action="store_true", default=True, help="Skip NA relations")
    return parser.parse_args()


def extract_head_tail(text: str) -> Tuple[Optional[str], Optional[str], str]:
    head_m = HEAD_RE.search(text)
    tail_m = TAIL_RE.search(text)
    head = head_m.group(1).strip() if head_m else None
    tail = tail_m.group(1).strip() if tail_m else None
    evidence = TAG_RE.sub("", text).strip()
    return head, tail, evidence


def normalize_relation(label: str) -> str:
    lb = (label or "").strip().upper()
    mapping = {
        "CAUSES": "CAUSES",
        "INCREASES": "INCREASES",
        "DECREASES": "DECREASES",
        "PREVENTS": "PREVENTS",
        "NA": "NA",
    }
    return mapping.get(lb, lb)


def relation_to_edge_type(rel: str) -> str:
    return rel if rel else "RELATED_TO"


def load_records(path: str) -> List[Dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    records = load_records(args.input)
    triples = []

    for rec in records:
        text = rec.get("text", "")
        if not text:
            continue

        raw_label = rec.get(args.label_field)
        if isinstance(raw_label, list):
            raw_label = raw_label[0] if raw_label else "NA"
        label = normalize_relation(str(raw_label) if raw_label is not None else "NA")

        if args.skip_na and label == "NA":
            continue

        confidence = rec.get(args.confidence_field, 1.0)
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 1.0

        head, tail, evidence = extract_head_tail(text)
        if not head or not tail:
            continue

        triples.append(
            {
                "head": head,
                "relation": label,
                "tail": tail,
                "evidence": evidence,
                "confidence": confidence,
            }
        )

    triples_path = os.path.join(args.output_dir, "triples.jsonl")
    with open(triples_path, "w", encoding="utf-8") as f:
        for t in triples:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    # Build Neo4j CSV
    node_ids = {}
    nodes = []

    def get_node_id(name: str) -> str:
        if name not in node_ids:
            node_id = f"n{len(node_ids) + 1}"
            node_ids[name] = node_id
            nodes.append({":ID": node_id, "name": name, ":LABEL": "Entity"})
        return node_ids[name]

    rel_rows = []
    for t in triples:
        sid = get_node_id(t["head"])
        tid = get_node_id(t["tail"])
        rel_rows.append(
            {
                ":START_ID": sid,
                ":END_ID": tid,
                ":TYPE": relation_to_edge_type(t["relation"]),
                "evidence": t["evidence"],
                "confidence": t["confidence"],
            }
        )

    nodes_csv = os.path.join(args.output_dir, "neo4j_nodes.csv")
    rels_csv = os.path.join(args.output_dir, "neo4j_rels.csv")

    with open(nodes_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[":ID", "name", ":LABEL"])
        writer.writeheader()
        writer.writerows(nodes)

    with open(rels_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[":START_ID", ":END_ID", ":TYPE", "evidence", "confidence"],
        )
        writer.writeheader()
        writer.writerows(rel_rows)

    print(f"Input records: {len(records)}")
    print(f"Triples (non-NA): {len(triples)}")
    print(f"Nodes: {len(nodes)}")
    print(f"Relations: {len(rel_rows)}")
    print(f"Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
