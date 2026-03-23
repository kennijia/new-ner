#!/usr/bin/env python3
"""LLM zero-shot / few-shot NER baseline for BERT-LSTM-CRF dataset.

Purpose
-------
Add a modern LLM-style baseline that can be compared with existing NER models
using the SAME metric implementation in this repo (`metrics.py`).

Input
-----
NPZ file with:
- words: object array of list[str] (character tokens)
- labels: object array of list[str] (BIO/S tags)

Output
------
- JSON summary with micro F1 and per-label F1
- JSONL prediction details per sample (optional)

Example
-------
cd /root/msy/ner/BERT-LSTM-CRF
python llm_ner_baseline.py \
  --input data/my/admin_test.npz \
  --model qwen-max \
  --base_url http://127.0.0.1:8000/v1 \
  --api_key sk-xxx \
  --output experiments/llm_ner_baseline/predictions.jsonl \
  --summary experiments/llm_ner_baseline/metrics.json
"""

import argparse
import json
import os
import re
import time
from typing import Dict, List, Tuple

import numpy as np
from openai import OpenAI

import config
from metrics import f1_score, get_entities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM zero-shot / few-shot NER baseline")
    parser.add_argument("--input", default="data/my/admin_test.npz", help="Path to NPZ test set")
    parser.add_argument("--model", default="qwen-max", help="OpenAI-compatible chat model name")
    parser.add_argument("--base_url", default="", help="OpenAI-compatible base URL (or env OPENAI_BASE_URL)")
    parser.add_argument("--api_key", default="", help="API key (or env OPENAI_API_KEY)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--max_retries", type=int, default=3, help="Max retries per sample")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between requests")
    parser.add_argument("--limit", type=int, default=0, help="Only evaluate first N samples (0 = all)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output JSONL")
    parser.add_argument("--few_shot_k", type=int, default=0, help="Number of few-shot examples (0 = zero-shot)")
    parser.add_argument(
        "--few_shot_file",
        default="data/my/admin_train.npz",
        help="NPZ file to source few-shot examples from",
    )
    parser.add_argument(
        "--few_shot_strategy",
        choices=["first", "random"],
        default="first",
        help="How to select few-shot examples from few_shot_file",
    )
    parser.add_argument("--few_shot_seed", type=int, default=42, help="Random seed for few-shot selection")
    parser.add_argument(
        "--few_shot_max_chars",
        type=int,
        default=120,
        help="Max text length for a few-shot example (0 = no limit)",
    )
    parser.add_argument(
        "--output",
        default="experiments/llm_ner_baseline/predictions.jsonl",
        help="Prediction JSONL path",
    )
    parser.add_argument(
        "--summary",
        default="experiments/llm_ner_baseline/metrics.json",
        help="Summary JSON path",
    )
    return parser.parse_args()


def tags_to_entities(tags: List[str], text: str, allowed_types: set) -> List[Dict]:
    entities: List[Dict] = []
    for etype, start, end in get_entities(tags):
        if etype not in allowed_types:
            continue
        if start < 0 or end < 0 or start > end or end >= len(text):
            continue
        entities.append(
            {
                "type": etype,
                "start": int(start),
                "end": int(end),
                "text": text[start: end + 1],
            }
        )
    return entities


def build_few_shot_examples(
    npz_path: str,
    k: int,
    allowed_types: set,
    strategy: str,
    seed: int,
    max_chars: int,
) -> List[Dict]:
    if k <= 0:
        return []

    data = np.load(npz_path, allow_pickle=True)
    words = data["words"]
    labels = data["labels"]

    indices = list(range(len(words)))
    if strategy == "random":
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    examples: List[Dict] = []
    used = set()

    def _collect(require_entity: bool) -> None:
        nonlocal examples
        for idx in indices:
            if idx in used:
                continue
            text = "".join(list(words[idx]))
            if max_chars > 0 and len(text) > max_chars:
                continue

            tags = list(labels[idx])
            ents = tags_to_entities(tags, text, allowed_types)
            if require_entity and len(ents) == 0:
                continue

            examples.append({"text": text, "entities": ents})
            used.add(idx)
            if len(examples) >= k:
                break

    # Prefer informative examples (contain at least one entity), then backfill.
    _collect(require_entity=True)
    if len(examples) < k:
        _collect(require_entity=False)

    return examples[:k]


def build_prompt(text: str, labels: List[str], few_shot_examples: List[Dict]) -> str:
    labels_str = ", ".join(labels)
    parts = [f"""你是中文命名实体识别助手。
请从给定文本中抽取实体，实体类型只能从以下集合中选择：[{labels_str}]。

要求：
1) 使用字符级下标，start/end 都是闭区间，且从 0 开始。
2) 只输出 JSON，不要输出任何额外解释。
3) 格式严格如下：
{{
  "entities": [
    {{"type": "ACTION", "start": 0, "end": 1, "text": "xx"}}
  ]
}}
4) 没有实体时返回 {{"entities": []}}。
""".strip()]

    if few_shot_examples:
        example_lines = ["下面是标注示例（few-shot）："]
        for i, ex in enumerate(few_shot_examples, start=1):
            ex_out = {"entities": ex["entities"]}
            example_lines.append(
                f"示例{i}：\n"
                f"文本：{ex['text']}\n"
                f"输出：\n{json.dumps(ex_out, ensure_ascii=False)}"
            )
        parts.append("\n\n".join(example_lines))

    parts.append(f"现在请处理以下文本：\n{text}")
    return "\n\n".join(parts)


def call_llm(client: OpenAI, model: str, prompt: str, temperature: float, max_retries: int) -> str:
    last_err = None
    for i in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": "你是严格输出 JSON 的信息抽取助手。"},
                    {"role": "user", "content": prompt},
                ],
            )
            content = resp.choices[0].message.content
            return content if content is not None else ""
        except Exception as e:  # noqa: BLE001
            last_err = e
            if i + 1 < max_retries:
                time.sleep(1.0 * (i + 1))
    raise RuntimeError(f"LLM request failed after retries: {last_err}")


def extract_first_json(text: str):
    text = text.strip()
    if not text:
        return None

    # direct parse
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001
        pass

    # fenced code block
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*```", text, flags=re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:  # noqa: BLE001
            pass

    # first JSON object/array substring (greedy-safe enough for our use)
    m_obj = re.search(r"\{[\s\S]*\}", text)
    if m_obj:
        try:
            return json.loads(m_obj.group(0))
        except Exception:  # noqa: BLE001
            pass

    m_arr = re.search(r"\[[\s\S]*\]", text)
    if m_arr:
        try:
            return json.loads(m_arr.group(0))
        except Exception:  # noqa: BLE001
            pass

    return None


def normalize_entities(parsed, text: str, allowed_types: set) -> List[Dict]:
    if parsed is None:
        return []

    if isinstance(parsed, dict):
        items = parsed.get("entities", [])
    elif isinstance(parsed, list):
        items = parsed
    else:
        items = []

    cleaned = []
    n = len(text)
    for item in items:
        if not isinstance(item, dict):
            continue

        t = str(item.get("type", "")).strip()
        if t not in allowed_types:
            continue

        try:
            s = int(item.get("start"))
            e = int(item.get("end"))
        except Exception:  # noqa: BLE001
            continue

        if s < 0 or e < 0 or s > e or e >= n:
            continue

        cleaned.append({"type": t, "start": s, "end": e})

    # sort and remove overlaps greedily
    cleaned.sort(key=lambda x: (x["start"], x["end"]))
    selected = []
    last_end = -1
    for ent in cleaned:
        if ent["start"] <= last_end:
            continue
        selected.append(ent)
        last_end = ent["end"]

    # add text span (for debug/output)
    for ent in selected:
        ent["text"] = text[ent["start"]: ent["end"] + 1]

    return selected


def entities_to_tags(text_len: int, entities: List[Dict], label_set: set) -> List[str]:
    tags = ["O"] * text_len
    for ent in entities:
        t = ent["type"]
        s = ent["start"]
        e = ent["end"]

        if s == e:
            s_tag = f"S-{t}"
            b_tag = f"B-{t}"
            tags[s] = s_tag if s_tag in label_set else b_tag
            continue

        b = f"B-{t}"
        i = f"I-{t}"
        if b not in label_set or i not in label_set:
            continue
        tags[s] = b
        for pos in range(s + 1, e + 1):
            tags[pos] = i

    return tags


def load_resume(path: str) -> Dict[int, Dict]:
    cache = {}
    if not os.path.exists(path):
        return cache
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                idx = int(obj.get("idx"))
                cache[idx] = obj
            except Exception:  # noqa: BLE001
                continue
    return cache


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def main() -> int:
    args = parse_args()

    api_key = args.api_key or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("Missing API key. Please set --api_key or OPENAI_API_KEY.")

    base_url = args.base_url or os.getenv("OPENAI_BASE_URL", "")
    client = OpenAI(api_key=api_key, base_url=base_url or None)

    data = np.load(args.input, allow_pickle=True)
    words = data["words"]
    labels = data["labels"]

    total = len(words)
    if args.limit and args.limit > 0:
        total = min(total, args.limit)

    label_types = list(config.labels)
    label_set = set(config.label2id.keys())
    allowed_types = set(label_types)

    few_shot_examples: List[Dict] = []
    if args.few_shot_k > 0:
        if not os.path.exists(args.few_shot_file):
            raise FileNotFoundError(f"few_shot_file not found: {args.few_shot_file}")
        if os.path.abspath(args.few_shot_file) == os.path.abspath(args.input):
            raise ValueError("few_shot_file should be different from --input to avoid data leakage.")

        few_shot_examples = build_few_shot_examples(
            npz_path=args.few_shot_file,
            k=args.few_shot_k,
            allowed_types=allowed_types,
            strategy=args.few_shot_strategy,
            seed=args.few_shot_seed,
            max_chars=args.few_shot_max_chars,
        )
        print(
            f"few-shot enabled: k={args.few_shot_k}, selected={len(few_shot_examples)}, "
            f"source={args.few_shot_file}, strategy={args.few_shot_strategy}"
        )

    ensure_parent(args.output)
    ensure_parent(args.summary)

    resume_cache = load_resume(args.output) if args.resume else {}
    write_mode = "a" if (args.resume and os.path.exists(args.output)) else "w"

    y_true: List[List[str]] = []
    y_pred: List[List[str]] = []

    parsed_ok = 0
    parsed_fail = 0
    request_fail = 0

    with open(args.output, write_mode, encoding="utf-8") as fout:
        for idx in range(total):
            text_chars = list(words[idx])
            text = "".join(text_chars)
            gold = list(labels[idx])

            if idx in resume_cache:
                cached = resume_cache[idx]
                pred = cached.get("pred_tags", [])
                if isinstance(pred, list) and len(pred) == len(gold):
                    y_true.append(gold)
                    y_pred.append(pred)
                    if bool(cached.get("parsed_ok", False)):
                        parsed_ok += 1
                    else:
                        parsed_fail += 1
                    continue

            prompt = build_prompt(text, label_types, few_shot_examples)
            raw = ""
            pred_tags = ["O"] * len(gold)
            entities = []
            ok = False

            try:
                raw = call_llm(
                    client=client,
                    model=args.model,
                    prompt=prompt,
                    temperature=args.temperature,
                    max_retries=args.max_retries,
                )
                parsed = extract_first_json(raw)
                entities = normalize_entities(parsed, text=text, allowed_types=allowed_types)
                pred_tags = entities_to_tags(len(text), entities, label_set)
                ok = parsed is not None
            except Exception as e:  # noqa: BLE001
                request_fail += 1
                raw = f"__ERROR__: {e}"
                pred_tags = ["O"] * len(gold)
                ok = False

            # align for safety
            if len(pred_tags) != len(gold):
                pred_tags = (pred_tags + ["O"] * len(gold))[: len(gold)]

            rec = {
                "idx": idx,
                "text": text,
                "gold_tags": gold,
                "pred_tags": pred_tags,
                "entities": entities,
                "parsed_ok": ok,
                "raw": raw,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

            y_true.append(gold)
            y_pred.append(pred_tags)

            if ok:
                parsed_ok += 1
            else:
                parsed_fail += 1

            if args.sleep > 0:
                time.sleep(args.sleep)

            if (idx + 1) % 10 == 0 or (idx + 1) == total:
                print(f"[{idx + 1}/{total}] parsed_ok={parsed_ok} parsed_fail={parsed_fail} request_fail={request_fail}")

    f1_labels, micro_f1 = f1_score(y_true, y_pred, mode="test")

    summary = {
        "input": args.input,
        "model": args.model,
        "base_url": base_url,
        "few_shot_k": args.few_shot_k,
        "few_shot_file": args.few_shot_file if args.few_shot_k > 0 else "",
        "few_shot_strategy": args.few_shot_strategy if args.few_shot_k > 0 else "",
        "few_shot_seed": args.few_shot_seed if args.few_shot_k > 0 else None,
        "total": total,
        "parsed_ok": parsed_ok,
        "parsed_fail": parsed_fail,
        "request_fail": request_fail,
        "micro_f1": micro_f1,
        "f1_labels": f1_labels,
        "output": args.output,
    }

    with open(args.summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=== LLM NER baseline summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
