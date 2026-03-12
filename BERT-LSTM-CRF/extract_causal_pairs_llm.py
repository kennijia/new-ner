#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Use an OpenAI-compatible LLM API to extract causal relation triples from NER dataset.

Input format (JSONL):
    {"text": "...", "label": {"ORG": {"水文局": [[0,2]]}, ...}}

Output record format (JSONL):
    {
      "source_index": 12,
      "text": "...",
      "entities": {...},
      "triples": [
        {"cause": "...", "relation": "导致", "effect": "...", "confidence": 0.91}
      ],
      "raw_response": "..."
    }

Flatten triples output (optional):
    {"head": "...", "relation": "导致", "tail": "...", "evidence": "...", "confidence": 0.91, "source_index": 12}
"""

import argparse
import json
import os
import re
import time
from typing import Dict, List, Optional, Set, Tuple

from openai import OpenAI


ALLOWED_RELATIONS = {"触发", "导致"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract causal relation triples via LLM")
    parser.add_argument("--env_file", default=".env", help="Path to .env file")
    parser.add_argument("--input", default="data/my/admin.json", help="Input JSONL file")
    parser.add_argument(
        "--output",
        default="data/my/re_pairs_condition_llm.jsonl",
        help="Per-record extraction output JSONL",
    )
    parser.add_argument(
        "--triples_out",
        default="data/my/kg/llm_triples.jsonl",
        help="Flattened triples output JSONL",
    )
    parser.add_argument("--model", default="", help="Model name (default from env MODEL_NAME or gpt-4o-mini)")
    parser.add_argument("--base_url", default="", help="OpenAI-compatible base URL (default from env OPENAI_BASE_URL)")
    parser.add_argument("--api_key", default="", help="API key (prefer env OPENAI_API_KEY)")
    parser.add_argument("--max_records", type=int, default=0, help="Max records to process (0 means all)")
    parser.add_argument("--start_index", type=int, default=0, help="Start line index (0-based)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--max_retries", type=int, default=3, help="Max retries per request")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between requests")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output instead of resume")
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
            except json.JSONDecodeError as exc:
                print(f"[WARN] skip invalid json at line {line_no}: {exc}")
    return rows


def _collect_entity_names(entities: Dict) -> List[str]:
    names: List[str] = []
    if not isinstance(entities, dict):
        return names
    for label, items in entities.items():
        if not isinstance(items, dict):
            continue
        for name in items:
            s = str(name).strip()
            if s:
                names.append(s)
    seen: Set[str] = set()
    uniq: List[str] = []
    for n in names:
        if n in seen:
            continue
        seen.add(n)
        uniq.append(n)
    return uniq


def generate_re_prompt(text: str, entities: Dict) -> str:
    entity_str = ", ".join(
        [f"{name}({label})" for label, items in entities.items() for name in items]
    )
    if not entity_str:
        entity_str = "无"

    prompt = f"""
你是一个水利调度专家。给定一段规章文本和其中已经识别出的实体，请提取“事件级”因果关系。
文本：{text}
已知实体：{entity_str}

请输出因果三元组 (原因事件短语, 逻辑关系, 结果事件短语)，并给出 0-1 之间的置信度。
关系类型仅限：[触发, 导致]。如果没有关系，请输出 NA。

关系判定规则（务必遵守）：
1) 出现“当/若/一旦/超过…时/达到…时”等条件触发结构，且后面是“应/需/立即/必须”等处置动作，优先标注为“触发”。
2) 描述一般原因造成结果（机理、趋势、后果）的，标注为“导致”。
3) 不要输出管理关系或语义从属关系（如“下达”“归属”）。
4) 同一条关系只选一个标签；若更像“条件满足后启动动作”，优先用“触发”。

注意：
1) 事件短语应尽量直接截取原文连续片段，保留语义完整，不要只输出单个动词（如“启动”）。
2) 每个事件短语都要给出其对应的实体锚点列表，锚点必须来自“已知实体”。
3) 允许事件短语大于实体边界，例如“水位超过警戒水位” 或 “启动应急响应措施”。
4) 事件短语必须能在原文中找到对应片段，禁止改写、扩写或凭空补充。
输出格式为 JSON 列表。

严格按以下格式输出：
1) 有关系时：
[
    {{
        "cause": "水位超过警戒水位",
        "relation": "导致",
        "effect": "启动应急响应措施",
        "cause_anchors": ["水位", "警戒水位"],
        "effect_anchors": ["启动"],
        "confidence": 0.95
    }}
]
2) 无关系时：
NA
"""
    return prompt.strip()


def _strip_code_fence(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s)
    return s.strip()


def _extract_json_array(text: str) -> Optional[List]:
    s = _strip_code_fence(text)
    if s.upper() == "NA":
        return []

    try:
        data = json.loads(s)
        if isinstance(data, list):
            return data
    except Exception:
        pass

    left = s.find("[")
    right = s.rfind("]")
    if left != -1 and right != -1 and right > left:
        chunk = s[left : right + 1]
        try:
            data = json.loads(chunk)
            if isinstance(data, list):
                return data
        except Exception:
            return None

    return None


def _to_confidence(value) -> float:
    try:
        score = float(value)
    except Exception:
        score = 0.0
    if score < 0:
        return 0.0
    if score > 1:
        return 1.0
    return round(score, 4)


def _normalize_anchor_list(value) -> List[str]:
    if isinstance(value, list):
        arr = value
    elif isinstance(value, str) and value.strip():
        arr = [value.strip()]
    else:
        arr = []
    out: List[str] = []
    seen: Set[str] = set()
    for x in arr:
        s = str(x).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _infer_anchors(phrase: str, known_entities: List[str]) -> List[str]:
    matched = [e for e in known_entities if e and e in phrase]
    matched.sort(key=len, reverse=True)
    picked: List[str] = []
    for m in matched:
        if any(m in p for p in picked):
            continue
        picked.append(m)
    return picked


def normalize_triples(items: Optional[List], text: str, entities: Dict) -> List[Dict]:
    if not items:
        return []

    known_entities = _collect_entity_names(entities)
    known_set = set(known_entities)

    triples: List[Dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue

        cause = str(item.get("cause", "")).strip()
        relation = str(item.get("relation", "")).strip()
        effect = str(item.get("effect", "")).strip()
        confidence = _to_confidence(item.get("confidence", 0.0))
        cause_anchors = _normalize_anchor_list(item.get("cause_anchors", []))
        effect_anchors = _normalize_anchor_list(item.get("effect_anchors", []))

        if not cause_anchors:
            cause_anchors = _infer_anchors(cause, known_entities)
        if not effect_anchors:
            effect_anchors = _infer_anchors(effect, known_entities)

        if not cause or not effect or relation not in ALLOWED_RELATIONS:
            continue
        if cause != text and cause not in text:
            continue
        if effect != text and effect not in text:
            continue
        if len(cause) <= 2 or len(effect) <= 2:
            continue
        if not cause_anchors or not effect_anchors:
            continue
        if any(a not in known_set for a in cause_anchors):
            continue
        if any(a not in known_set for a in effect_anchors):
            continue

        triples.append(
            {
                "cause": cause,
                "relation": relation,
                "effect": effect,
                "cause_anchors": cause_anchors,
                "effect_anchors": effect_anchors,
                "confidence": confidence,
            }
        )

    dedup_key: Set[Tuple[str, str, str]] = set()
    uniq: List[Dict] = []
    for t in triples:
        key = (t["cause"], t["relation"], t["effect"])
        if key in dedup_key:
            continue
        dedup_key.add(key)
        uniq.append(t)
    return uniq


def call_llm(client: OpenAI, model: str, prompt: str, temperature: float, max_retries: int) -> str:
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {
                        "role": "system",
                        "content": "你是严谨的信息抽取助手，只输出用户要求的结果，不添加解释。",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            last_err = exc
            if attempt < max_retries:
                time.sleep(min(2 * attempt, 6))
                continue
    raise RuntimeError(f"LLM request failed after retries: {last_err}")


def load_processed_indices(path: str) -> Set[int]:
    done: Set[int] = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                idx = rec.get("source_index")
                if isinstance(idx, int):
                    done.add(idx)
            except Exception:
                continue
    return done


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_env_file(env_path: str) -> None:
    if not env_path or not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            if (value.startswith('"') and value.endswith('"')) or (
                value.startswith("'") and value.endswith("'")
            ):
                value = value[1:-1]
            os.environ.setdefault(key, value)


def main() -> None:
    args = parse_args()

    load_env_file(args.env_file)

    api_key = args.api_key or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("Missing API key. Please set --api_key or OPENAI_API_KEY.")

    base_url = args.base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai-proxy.org/v1")
    model = args.model or os.getenv("MODEL_NAME", "gpt-4o-mini")

    rows = load_jsonl(args.input)
    if not rows:
        print("[WARN] no valid input rows")
        return

    if args.max_records > 0:
        end_idx = min(len(rows), args.start_index + args.max_records)
    else:
        end_idx = len(rows)

    client = OpenAI(base_url=base_url, api_key=api_key)

    if args.overwrite:
        processed_indices: Set[int] = set()
        out_mode = "w"
    else:
        processed_indices = load_processed_indices(args.output)
        out_mode = "a"

    ensure_parent(args.output)
    ensure_parent(args.triples_out)

    total = 0
    success = 0
    na_count = 0
    parsed_fail = 0
    total_triples = 0

    flat_mode = "w" if args.overwrite else "a"
    with open(args.output, out_mode, encoding="utf-8") as fout, open(
        args.triples_out, flat_mode, encoding="utf-8"
    ) as tfout:
        for i in range(args.start_index, end_idx):
            if i in processed_indices:
                continue

            total += 1
            rec = rows[i]
            text = str(rec.get("text", "")).strip()
            entities = rec.get("label", {})
            if not text:
                continue
            if not isinstance(entities, dict):
                entities = {}

            prompt = generate_re_prompt(text, entities)

            try:
                raw = call_llm(
                    client=client,
                    model=model,
                    prompt=prompt,
                    temperature=args.temperature,
                    max_retries=args.max_retries,
                )
            except Exception as exc:
                raw = f"ERROR: {exc}"

            parsed = _extract_json_array(raw)
            triples = normalize_triples(parsed, text=text, entities=entities)

            if parsed is None and not raw.startswith("ERROR:"):
                parsed_fail += 1

            if not triples:
                na_count += 1
            else:
                success += 1

            out_obj = {
                "source_index": i,
                "text": text,
                "entities": entities,
                "triples": triples,
                "raw_response": raw,
            }
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

            for t in triples:
                flat = {
                    "head": t["cause"],
                    "relation": t["relation"],
                    "tail": t["effect"],
                    "head_anchors": t.get("cause_anchors", []),
                    "tail_anchors": t.get("effect_anchors", []),
                    "evidence": text,
                    "confidence": t["confidence"],
                    "source_index": i,
                }
                tfout.write(json.dumps(flat, ensure_ascii=False) + "\n")
                total_triples += 1

            if args.sleep > 0:
                time.sleep(args.sleep)

            if total % 20 == 0:
                print(
                    f"[Progress] processed={total}, with_rel={success}, no_rel={na_count}, parse_fail={parsed_fail}, triples={total_triples}"
                )

    print("=== Done ===")
    print(f"Input rows: {len(rows)}")
    print(f"Processed rows (this run): {total}")
    print(f"Rows with relations: {success}")
    print(f"Rows with NA/no relation: {na_count}")
    print(f"Response parse failed: {parsed_fail}")
    print(f"Total triples written: {total_triples}")
    print(f"Record output: {args.output}")
    print(f"Triples output: {args.triples_out}")


if __name__ == "__main__":
    main()
