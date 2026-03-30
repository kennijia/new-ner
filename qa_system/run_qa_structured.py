#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""问答系统 CLI（结构化答案输出版本）。

流程：
1) 双通道检索 + 证据融合 + Prompt 构造（复用 `qa_system.pipeline.QAPipeline`）
2) 在 Prompt 后追加“只输出 JSON”的强约束（`qa_system.answering.build_answer_prompt`）
3) 调用 OpenAI-compatible LLM（复用 `.env` / OPENAI_API_KEY / OPENAI_BASE_URL / MODEL_NAME）
4) 解析并校验 LLM 输出（evidence_ids 必须来自 [T#..]/[P#..]）

输出：
- 默认打印一个 JSON，包含检索证据、prompt、raw answer、structured_answer、valid/错误信息。
- 可通过 --out 保存到文件。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qa_system.answering import build_answer_prompt, parse_structured_answer, validate_evidence_ids
from qa_system.config import QAConfig, DEFAULT_CONFIG
from qa_system.llm_client import LLMClient, LLMConfig
from qa_system.pipeline import QAPipeline


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Causal-KG enhanced QA (Chapter 5) - structured JSON answer")
    p.add_argument("--query", required=True, help="User question")

    # paths
    p.add_argument("--graph", default="", help="Path to nx_graph.gpickle")
    p.add_argument("--index", default="", help="Path to faiss index")
    p.add_argument("--meta", default="", help="Path to FAISS meta JSON")
    p.add_argument("--emb_model", default="", help="Embedding model name/path")

    # retrieval params
    p.add_argument("--top_text_k", type=int, default=DEFAULT_CONFIG.top_text_k)
    p.add_argument("--top_path_k", type=int, default=DEFAULT_CONFIG.top_path_k)
    p.add_argument("--max_hops", type=int, default=DEFAULT_CONFIG.max_hops)
    p.add_argument("--allowed_relations", default=",".join(DEFAULT_CONFIG.allowed_relations))

    # fusion params
    p.add_argument("--alpha", type=float, default=DEFAULT_CONFIG.alpha)
    p.add_argument("--beta", type=float, default=DEFAULT_CONFIG.beta)

    # prompt params
    p.add_argument("--path_force_threshold", type=float, default=DEFAULT_CONFIG.path_force_threshold)

    # NER
    p.add_argument("--use_ner", action="store_true", help="Use trained NER model to extract entities")
    p.add_argument("--ner_model_dir", default="", help="NER model dir for BERT-LSTM-CRF")

    # LLM (OpenAI-compatible)
    p.add_argument("--env_file", default=".env", help=".env path (OPENAI_API_KEY/OPENAI_BASE_URL/MODEL_NAME)")
    p.add_argument("--model", default="", help="Override model name")
    p.add_argument("--base_url", default="", help="Override base_url")
    p.add_argument("--api_key", default="", help="Override api_key")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max_retries", type=int, default=3)

    # output
    p.add_argument("--out", default="", help="Write output JSON to path")
    return p


def apply_overrides(cfg: QAConfig, args: argparse.Namespace) -> QAConfig:
    if args.graph:
        cfg.nx_graph = Path(args.graph)
    if args.index:
        cfg.faiss_index = Path(args.index)
    if args.meta:
        cfg.faiss_meta = Path(args.meta)
    if args.emb_model:
        cfg.embedding_model = args.emb_model

    cfg.top_text_k = int(args.top_text_k)
    cfg.top_path_k = int(args.top_path_k)
    cfg.max_hops = int(args.max_hops)
    cfg.allowed_relations = tuple([x.strip() for x in args.allowed_relations.split(",") if x.strip()])

    cfg.alpha = float(args.alpha)
    cfg.beta = float(args.beta)
    cfg.path_force_threshold = float(args.path_force_threshold)

    cfg.use_ner = bool(args.use_ner)
    cfg.ner_model_dir = Path(args.ner_model_dir) if args.ner_model_dir else None

    return cfg


def _collect_allowed_evidence_ids(bundle) -> list[str]:
    allowed: list[str] = []
    for t in bundle.text_items:
        tid = t.get("id")
        if tid is not None:
            allowed.append(f"T#{tid}")
    for p in bundle.graph_pack.get("paths", []):
        pid = p.get("path_id")
        if pid is not None:
            allowed.append(f"P#{pid}")
    return allowed


def main() -> int:
    args = build_arg_parser().parse_args()

    cfg = apply_overrides(QAConfig(), args)
    pipeline = QAPipeline(cfg)

    bundle = pipeline.retrieve(args.query)
    output = pipeline.to_debug_json(bundle)

    # LLM answer (structured)
    llm_cfg = LLMConfig(
        env_file=args.env_file,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=float(args.temperature),
        max_retries=int(args.max_retries),
    )
    client = LLMClient(llm_cfg)

    structured_prompt = build_answer_prompt(bundle.prompt)
    raw = client.chat(structured_prompt)
    output["llm_answer_raw"] = raw

    ans, err = parse_structured_answer(raw)
    if ans is None:
        output["structured_answer"] = None
        output["structured_answer_error"] = err
        output["structured_answer_valid"] = False
        output["structured_answer_validation_error"] = "parse_failed"
    else:
        allowed_ids = _collect_allowed_evidence_ids(bundle)
        ok, verr = validate_evidence_ids(ans, allowed_ids)
        output["structured_answer"] = {
            "conclusion": ans.conclusion,
            "evidence_ids": ans.evidence_ids,
            "causal_chain": ans.causal_chain,
            "uncertainty": ans.uncertainty,
        }
        output["structured_answer_valid"] = ok
        output["structured_answer_validation_error"] = verr if not ok else ""
        output["structured_answer_allowed_ids"] = allowed_ids

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved={args.out}")

    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
