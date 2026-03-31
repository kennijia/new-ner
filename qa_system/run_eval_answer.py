#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""批量评测（含 LLM 输出答案），用于第五章“回答准确率”主指标。

功能：
- 读取 `eval_questions.jsonl`
- 对每个问题按指定 mode 跑一遍 pipeline + LLM
- 输出 JSONL（每行一个问题的完整记录：证据、prompt、结构化答案等）

说明：
- LLM 配置复用 `qa_system.run_qa_answer`（OpenAI-compatible，依赖 .env）
- 为可复现起见，建议在 .env 中固定 MODEL_NAME，并设置 temperature=0

输出字段（简化）：
- qid, type, query, mode
- text_channel / graph_channel / fused_evidence / prompt
- llm_answer_raw / llm_answer_json（若可解析）
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

from qa_system.answering import build_answer_prompt, parse_structured_answer
from qa_system.config import DEFAULT_CONFIG, QAConfig
from qa_system.llm_client import LLMClient, LLMConfig
from qa_system.pipeline import QAPipeline

def _as_text_only(debug: Dict[str, Any]) -> Dict[str, Any]:
    debug = dict(debug)
    debug["graph_channel"] = {"query": debug.get("query"), "entities": [], "entity_mapping": {}, "paths": []}
    debug["fused_evidence"] = [e for e in debug.get("fused_evidence", []) if e.get("channel") == "text"]
    return debug


def _as_graph_only(debug: Dict[str, Any]) -> Dict[str, Any]:
    debug = dict(debug)
    debug["text_channel"] = []
    debug["fused_evidence"] = [e for e in debug.get("fused_evidence", []) if e.get("channel") == "graph"]
    return debug


def _as_dual_no_fusion(debug: Dict[str, Any]) -> Dict[str, Any]:
    """dual_concat：不做融合打分/排序，仅拼接两路证据供 prompt 使用。"""
    debug = dict(debug)

    fused = []

    # text first (keep original text rank order)
    for t in debug.get("text_channel", []) or []:
        fused.append({"channel": "text", "id": t.get("id"), "text": t.get("text"), "score": t.get("score")})

    # graph second (keep original path order)
    for p in debug.get("graph_channel", {}).get("paths", []) or []:
        fused.append(
            {
                "channel": "graph",
                "path_id": p.get("path_id"),
                "logic": p.get("logic"),
                "confidence": p.get("confidence"),
                "score": p.get("score"),
            }
        )

    debug["fused_evidence"] = fused
    return debug


def _rebuild_prompt(pipe: QAPipeline, query: str, debug: Dict[str, Any]) -> str:
    """根据 debug 中的 text_channel / graph_channel.paths 重建与模式一致的 prompt。"""
    dual = pipe._import_dual_retrieval()
    return dual.build_causal_prompt(
        question=query,
        text_items=debug.get("text_channel", []) or [],
        path_items=(debug.get("graph_channel", {}) or {}).get("paths", []) or [],
        path_force_threshold=float(pipe.config.path_force_threshold),
    )

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch eval with LLM answering (Chapter 5)")
    p.add_argument("--in", dest="in_path", required=True, help="Path to eval_questions.jsonl")
    p.add_argument("--out", dest="out_path", required=True, help="Output JSONL path")
    p.add_argument(
        "--mode",
        required=True,
        choices=["text", "graph", "dual_concat", "fusion"],
        help="Retrieval mode",
    )

    # allow overriding retrieval / fusion params if needed
    p.add_argument("--top_text_k", type=int, default=DEFAULT_CONFIG.top_text_k)
    p.add_argument("--top_path_k", type=int, default=DEFAULT_CONFIG.top_path_k)
    p.add_argument("--max_hops", type=int, default=DEFAULT_CONFIG.max_hops)
    p.add_argument("--alpha", type=float, default=DEFAULT_CONFIG.alpha)
    p.add_argument("--beta", type=float, default=DEFAULT_CONFIG.beta)

    # LLM params
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_tokens", type=int, default=512)

    return p


def load_questions(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            yield json.loads(line)


def cfg_with_overrides(args: argparse.Namespace) -> QAConfig:
    cfg = DEFAULT_CONFIG
    cfg.top_text_k = int(args.top_text_k)
    cfg.top_path_k = int(args.top_path_k)
    cfg.max_hops = int(args.max_hops)
    cfg.alpha = float(args.alpha)
    cfg.beta = float(args.beta)
    return cfg


def main() -> int:
    args = build_arg_parser().parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = cfg_with_overrides(args)
    pipe = QAPipeline(cfg)

    # LLM client from env
    llm_cfg = LLMConfig(temperature=float(args.temperature))
    llm = LLMClient(llm_cfg)

    with out_path.open("w", encoding="utf-8") as w:
        for q in load_questions(in_path):
            qid = q.get("qid")
            qtype = q.get("type")
            query = q.get("query", "")

            rb = pipe.retrieve(query=query)
            bundle: Dict[str, Any] = pipe.to_debug_json(rb)

            mode = str(args.mode)
            if mode == "text":
                bundle = _as_text_only(bundle)
            elif mode == "graph":
                bundle = _as_graph_only(bundle)
            elif mode == "dual_concat":
                bundle = _as_dual_no_fusion(bundle)
            elif mode == "fusion":
                pass
            else:
                raise ValueError(f"Unknown mode: {mode}")

            prompt_with_evidence = _rebuild_prompt(pipe, query=query, debug=bundle)
            answer_prompt = build_answer_prompt(prompt_with_evidence)

            llm_raw = llm.chat(answer_prompt)
            parsed, err = parse_structured_answer(llm_raw)

            rec: Dict[str, Any] = {
                "qid": qid,
                "type": qtype,
                "query": query,
                "mode": mode,
                "config": {
                    "top_text_k": cfg.top_text_k,
                    "top_path_k": cfg.top_path_k,
                    "max_hops": cfg.max_hops,
                    "alpha": cfg.alpha,
                    "beta": cfg.beta,
                },
                "text_channel": bundle.get("text_channel", []),
                "graph_channel": bundle.get("graph_channel", {}),
                "fused_evidence": bundle.get("fused_evidence", []),
                "prompt": prompt_with_evidence,
                "llm_answer_raw": llm_raw,
                "llm_answer_json": asdict(parsed) if parsed else None,
                "llm_parse_error": err,
            }

            w.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
