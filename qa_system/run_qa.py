#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""问答系统 CLI（先完成“检索→融合→Prompt 生成”闭环）。

当前阶段：
- 输出双通道检索结果与生成 Prompt（用于论文展示与后续接 LLM 生成）

后续阶段（下一步实现）：
- 增加 llm_client + answer 解析，生成最终可解释答案 JSON。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qa_system.config import QAConfig, DEFAULT_CONFIG
from qa_system.pipeline import QAPipeline


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Causal-KG enhanced QA (Chapter 5)")
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

    # output
    p.add_argument("--out", default="", help="Write debug bundle JSON to path")
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


def main() -> int:
    args = build_arg_parser().parse_args()

    cfg = apply_overrides(QAConfig(), args)
    pipeline = QAPipeline(cfg)

    bundle = pipeline.retrieve(args.query)

    if args.out:
        pipeline.save_debug(bundle, args.out)
        print(f"saved={args.out}")

    print(json.dumps(pipeline.to_debug_json(bundle), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
