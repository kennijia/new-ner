#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate RE candidate pairs for doccano Text Classification.
Input: JSONL with {"text": ..., "label": {...}} (NER format).
Output: JSONL with {"text": "... [HEAD]...[/HEAD] ... [TAIL]...[/TAIL]"}
"""
import argparse
import json
import random
from typing import Dict, List, Tuple


def parse_args():
    parser = argparse.ArgumentParser(description="Generate RE pairs from NER JSONL")
    parser.add_argument("--input", required=True, help="Input JSONL (NER format)")
    parser.add_argument("--output", required=True, help="Output JSONL for doccano Text Classification")
    parser.add_argument("--limit", type=int, default=0, help="Limit records (0 = no limit)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--head_types",
        default="ACTION",
        help="Comma-separated head entity types (default: ACTION)",
    )
    parser.add_argument(
        "--tail_types",
        default="LEVEL_KEY,OBJ,VALUE",
        help="Comma-separated tail entity types (default: LEVEL_KEY,OBJ,VALUE)",
    )
    parser.add_argument(
        "--max_pairs_per_record",
        type=int,
        default=10,
        help="Max pairs per record (0 = no limit)",
    )
    parser.add_argument(
        "--fallback_all_pairs",
        action="store_true",
        help="If no head entity, generate all type pairs among entities",
    )
    parser.add_argument(
        "--mode",
        choices=["pairs", "condition_action"],
        default="condition_action",
        help="pairs: generate all head/tail pairs; condition_action: build condition->action pairs",
    )
    parser.add_argument(
        "--expand_action_span",
        action="store_true",
        help="Expand ACTION span to include following object phrase",
    )
    return parser.parse_args()


def collect_entities(label: Dict, text: str) -> List[Dict]:
    entities = []
    if not label:
        return entities
    for ent_type, ent_dict in label.items():
        if not isinstance(ent_dict, dict):
            continue
        for ent_text, spans in ent_dict.items():
            if not ent_text or not spans:
                continue
            for span in spans:
                if not isinstance(span, list) or len(span) != 2:
                    continue
                start, end = span
                if start < 0 or end < start or end >= len(text):
                    continue
                entities.append(
                    {
                        "type": ent_type,
                        "start": start,
                        "end": end,
                        "text": text[start : end + 1],
                    }
                )
    return entities


def insert_tags(text: str, head: Dict, tail: Dict) -> str:
    # Insert tags from right to left to keep indices valid.
    spans = [
        (head["start"], head["end"], "HEAD"),
        (tail["start"], tail["end"], "TAIL"),
    ]
    spans.sort(key=lambda x: (x[0], x[1]), reverse=True)
    out = text
    for start, end, tag in spans:
        out = out[: end + 1] + f"[/{tag}]" + out[end + 1 :]
        out = out[:start] + f"[{tag}]" + out[start:]
    return out


def _find_left_keyword(text: str, start: int) -> int:
    keywords = [
        "水位上涨",
        "水位上升",
        "水位达到",
        "水位超过",
        "上涨",
        "上升",
        "回落",
        "降至",
        "升至",
        "达到",
        "超过",
        "高于",
        "低于",
        "未超过",
        "未达",
        "超",
        "不足",
        "小于",
        "大于",
        "不超过",
    ]
    left_window = max(0, start - 20)
    window_text = text[left_window:start]
    idx = -1
    for kw in keywords:
        pos = window_text.rfind(kw)
        if pos != -1:
            idx = left_window + pos
            break
    if idx == -1:
        return start

    # extend left to include subject phrase up to punctuation
    extend_left = idx
    max_left = 12
    while extend_left > 0 and (idx - extend_left) < max_left:
        ch = text[extend_left - 1]
        if ch in " ，,。；;\t\n":
            break
        extend_left -= 1
    return extend_left


def expand_condition_span(text: str, level_entity: Dict) -> Tuple[int, int]:
    start = level_entity["start"]
    end = level_entity["end"]

    # fix common truncation like "警戒水" -> "警戒水位"
    if end + 1 < len(text) and text[end + 1] in "位线值":
        end += 1

    # expand left with trigger keyword if any
    start = _find_left_keyword(text, start)

    # expand right until a boundary or a max window
    boundaries = ["时", "则", "，", ",", "。", "；", ";", "后", "前", "内"]
    max_right = 18
    right_limit = min(len(text) - 1, end + max_right)
    segment = text[end + 1 : right_limit + 1]
    stop = None
    for i, ch in enumerate(segment):
        if ch in boundaries:
            stop = end + 1 + i
            break
    if stop is not None and stop >= start:
        end = stop - 1
    else:
        end = right_limit

    if end < start:
        end = level_entity["end"]
        start = level_entity["start"]

    return start, end


def expand_action_span(text: str, action_entity: Dict) -> Dict:
    start = action_entity["start"]
    end = action_entity["end"]

    # expand right to include object phrase if short
    boundaries = ["，", ",", "。", "；", ";", "、", "和", "或", "及"]
    max_right = 10
    right_limit = min(len(text) - 1, end + max_right)
    segment = text[end + 1 : right_limit + 1]
    stop = None
    for i, ch in enumerate(segment):
        if ch in boundaries:
            stop = end + 1 + i
            break
    if stop is not None and stop > end:
        end = stop - 1
    else:
        end = right_limit

    return {
        "type": action_entity["type"],
        "start": start,
        "end": end,
        "text": text[start : end + 1],
    }


def build_condition_action_pairs(
    entities: List[Dict], text: str, action_type: str = "ACTION", level_type: str = "LEVEL_KEY"
) -> List[Tuple[Dict, Dict]]:
    actions = [e for e in entities if e["type"] == action_type]
    levels = [e for e in entities if e["type"] == level_type]
    pairs = []
    if not actions or not levels:
        return pairs

    for action in actions:
        level = min(levels, key=lambda e: abs(e["start"] - action["start"]))
        cond_start, cond_end = expand_condition_span(text, level)
        if not (0 <= cond_start <= cond_end < len(text)):
            continue
        action_expanded = expand_action_span(text, action)
        # avoid overlap
        if not (cond_end < action_expanded["start"] or cond_start > action_expanded["end"]):
            continue
        head = {
            "type": "COND",
            "start": cond_start,
            "end": cond_end,
            "text": text[cond_start : cond_end + 1],
        }
        pairs.append((head, action_expanded))
    return pairs


def build_pairs(
    entities: List[Dict],
    head_types: Tuple[str, ...],
    tail_types: Tuple[str, ...],
    fallback_all_pairs: bool,
) -> List[Tuple[Dict, Dict]]:
    heads = [e for e in entities if e["type"] in head_types]
    tails = [e for e in entities if e["type"] in tail_types]

    pairs = []
    if heads and tails:
        for h in heads:
            for t in tails:
                if h["start"] == t["start"] and h["end"] == t["end"]:
                    continue
                pairs.append((h, t))
    elif fallback_all_pairs:
        # All ordered pairs (excluding self)
        for i, h in enumerate(entities):
            for j, t in enumerate(entities):
                if i == j:
                    continue
                pairs.append((h, t))
    return pairs


def main():
    args = parse_args()
    random.seed(args.seed)

    head_types = tuple([t.strip().upper() for t in args.head_types.split(",") if t.strip()])
    tail_types = tuple([t.strip().upper() for t in args.tail_types.split(",") if t.strip()])

    count = 0
    out_count = 0
    with open(args.input, "r", encoding="utf-8") as fin, open(
        args.output, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = record.get("text", "")
            label = record.get("label", None)
            if not text or not label:
                continue

            entities = collect_entities(label, text)
            if not entities:
                continue

            if args.mode == "condition_action":
                pairs = build_condition_action_pairs(entities, text)
                if not pairs:
                    pairs = build_pairs(entities, head_types, tail_types, args.fallback_all_pairs)
            else:
                pairs = build_pairs(entities, head_types, tail_types, args.fallback_all_pairs)
            if not pairs:
                continue

            if args.max_pairs_per_record and len(pairs) > args.max_pairs_per_record:
                pairs = random.sample(pairs, args.max_pairs_per_record)

            seen = set()
            for head, tail in pairs:
                key = (head["start"], head["end"], tail["start"], tail["end"])
                if key in seen:
                    continue
                seen.add(key)
                marked = insert_tags(text, head, tail)
                fout.write(json.dumps({"text": marked}, ensure_ascii=False) + "\n")
                out_count += 1

            count += 1
            if args.limit and count >= args.limit:
                break

    print(f"Processed records: {count}")
    print(f"Output pairs: {out_count}")


if __name__ == "__main__":
    main()
