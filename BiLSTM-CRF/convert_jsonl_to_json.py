#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert JSONL with span triples [start,end,tag] into the json format
expected by BiLSTM-CRF `data_process.py`:

{ "text": "...", "label": {"TAG": {"entity_text": [[start, end], ...]}}}

Note: this script assumes input spans are [start, end) (end exclusive).
It converts them to inclusive end indices used by the project.

Usage:
  python convert_jsonl_to_json.py --input ../admin.jsonl --output data/my/admin.json
"""

import argparse
import json
import os
from collections import defaultdict


def convert_line(entry):
    text = entry["text"]
    labels = entry.get("label", [])
    # Build the nested dict: {tag: {entity_text: [[s,e], ...]}}
    out_label = defaultdict(lambda: defaultdict(list))

    # labels expected to be a list of [start, end, tag]
    for triple in labels:
        if len(triple) < 3:
            continue
        s, e, tag = triple
        # input is assumed end-exclusive; convert to inclusive
        # validate indices
        if not (0 <= s < len(text) and 0 < e <= len(text) and s < e):
            # skip invalid spans
            continue
        e_incl = e - 1
        entity_text = text[s:e]
        # store
        out_label[tag][entity_text].append([s, e_incl])

    return {"text": text, "label": out_label}


def main(input_path, output_path, force=False):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if os.path.exists(output_path) and not force:
        print(f"Output file {output_path} already exists. Use --force to overwrite.")
        return

    with open(input_path, 'r', encoding='utf-8') as fin, open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            converted = convert_line(entry)

            # convert defaultdicts to normal dicts for JSON
            converted['label'] = {k: dict(v) for k, v in converted['label'].items()}
            fout.write(json.dumps(converted, ensure_ascii=False) + "\n")

    print(f"Converted {input_path} -> {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default='./admin.jsonl', help='Input jsonl path')
    parser.add_argument('--output', '-o', default='data/my/admin.json', help='Output json path')
    parser.add_argument('--force', '-f', action='store_true', help='Overwrite output if exists')
    args = parser.parse_args()

    main(args.input, args.output, force=args.force)
