#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 NER JSONL 中筛选高价值关系抽取语料，并对 ACTION 实体做语义补全。

输入格式（每行）：
    {
      "text": "...",
      "label": [[start, end, "TYPE"], ...]
    }

其中 label 的 span 按 [start, end)（右开区间）处理。

默认输出格式（便于 Doccano 导入）：
    {
      "text": "...",
      "label": [[start, end, "TYPE"], ...]
    }
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


ACTION_SUFFIX_PATTERN = re.compile(
    r"(防汛|应急|日常|调度|处置|执行|管理|抗旱)?(预案|方案|指令|程序|措施|计划|要求|规定)"
)

CAUSAL_KEYWORDS = [
    "导致",
    "由于",
    "引起",
    "从而",
    "若",
    "当",
    "则",
    "诱发",
    "触发",
    "限制",
    "上涨至",
    "超过",
]

CLAUSE_BOUNDARIES = "，,。；;！!？?\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine ACTION spans and filter high-value RE samples")
    parser.add_argument(
        "--input",
        default="data/my/admin.jsonl",
        help="Input NER JSONL, default: data/my/admin.jsonl",
    )
    parser.add_argument(
        "--output",
        default="data/my/admin_re_high_value.jsonl",
        help="Filtered output JSONL, default: data/my/admin_re_high_value.jsonl",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=15,
        help="Max chars after ACTION end for suffix matching, default: 15",
    )
    parser.add_argument(
        "--max_gap",
        type=int,
        default=2,
        help="Max allowed unmatched chars before suffix match starts, default: 2",
    )
    parser.add_argument(
        "--keep_fields",
        action="store_true",
        help="Keep original fields (e.g., id/Comments), only replace label",
    )
    return parser.parse_args()


def _cut_same_clause(text: str, start: int, window: int) -> str:
    seg = text[start : start + max(0, window)]
    if not seg:
        return ""
    for idx, ch in enumerate(seg):
        if ch in CLAUSE_BOUNDARIES:
            return seg[:idx]
    return seg


def extend_action_end(text: str, start: int, end: int, window: int, max_gap: int) -> int:
    if not (0 <= start <= end <= len(text)):
        return end
    following = _cut_same_clause(text, end, window)
    if not following:
        return end
    match = ACTION_SUFFIX_PATTERN.search(following)
    if not match:
        return end
    if match.start() > max_gap:
        return end
    return min(len(text), end + match.end())


def normalize_labels(raw_labels, text: str) -> Tuple[List[List], int]:
    """Return (valid_labels, dropped_count)."""
    valid_labels: List[List] = []
    dropped = 0
    if not isinstance(raw_labels, list):
        return valid_labels, 0

    for ent in raw_labels:
        if not isinstance(ent, list) or len(ent) < 3:
            dropped += 1
            continue
        start, end, label = ent[0], ent[1], ent[2]
        if not isinstance(start, int) or not isinstance(end, int) or not isinstance(label, str):
            dropped += 1
            continue
        if not (0 <= start <= end <= len(text)):
            dropped += 1
            continue
        valid_labels.append([start, end, label])
    return valid_labels, dropped


def refine_labels(text: str, labels: List[List], window: int, max_gap: int) -> Tuple[List[List], int]:
    extended_action_count = 0
    new_labels: List[List] = []
    for start, end, label in labels:
        new_end = end
        if label == "ACTION":
            new_end = extend_action_end(text, start, end, window=window, max_gap=max_gap)
            if new_end > end:
                extended_action_count += 1
        new_labels.append([start, new_end, label])
    return new_labels, extended_action_count


def is_high_value(text: str, labels: List[List]) -> bool:
    types = [ent[2] for ent in labels]
    has_causal_word = any(keyword in text for keyword in CAUSAL_KEYWORDS)
    has_core_pair = ("LEVEL_KEY" in types and "ACTION" in types) or (types.count("LEVEL_KEY") >= 2)
    return has_causal_word or has_core_pair


def build_output_record(record: Dict, labels: List[List], keep_fields: bool) -> Dict:
    if keep_fields:
        out = dict(record)
        out["label"] = labels
        return out
    return {
        "text": record.get("text", ""),
        "label": labels,
    }


def refine_and_filter_data(
    input_file: str,
    output_file: str,
    window: int = 15,
    max_gap: int = 2,
    keep_fields: bool = False,
) -> None:
    total = 0
    kept = 0
    bad_json = 0
    empty_text = 0
    dropped_labels = 0
    changed_records = 0
    extended_actions = 0
    results: List[Dict] = []

    with open(input_file, "r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                bad_json += 1
                continue

            text = data.get("text", "")
            if not isinstance(text, str) or not text:
                empty_text += 1
                continue

            labels, dropped = normalize_labels(data.get("label", []), text)
            dropped_labels += dropped

            new_labels, ext_cnt = refine_labels(text, labels, window=window, max_gap=max_gap)
            if new_labels != labels:
                changed_records += 1
            extended_actions += ext_cnt

            if is_high_value(text, new_labels):
                results.append(build_output_record(data, new_labels, keep_fields=keep_fields))
                kept += 1

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fout:
        for item in results:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"处理完成：{input_file}")
    print(f"总样本: {total}")
    print(f"筛选保留: {kept}")
    print(f"ACTION 被扩展次数: {extended_actions}")
    print(f"发生实体变更的样本数: {changed_records}")
    print(f"非法标签片段丢弃数: {dropped_labels}")
    print(f"坏行(JSON解析失败): {bad_json}")
    print(f"空文本跳过: {empty_text}")
    print(f"输出文件: {output_file}")


def main() -> None:
    args = parse_args()
    refine_and_filter_data(
        input_file=args.input,
        output_file=args.output,
        window=args.window,
        max_gap=args.max_gap,
        keep_fields=args.keep_fields,
    )


if __name__ == "__main__":
    main()
