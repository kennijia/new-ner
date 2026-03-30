"""问答系统端到端流水线（第五章核心实现）。

职责：
- 加载 NetworkX 因果图谱
- 调用双通道检索（向量语义 + 因果路径）
- 统一评分与融合（alpha/beta）
- 构造因果约束 Prompt，供生成模型回答

说明：
- 底层检索实现复用：`BERT-LSTM-CRF/causal_dual_retrieval.py`
- 本文件提供更“系统工程化”的封装，便于 CLI/实验调用。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class RetrievalBundle:
    query: str
    text_items: List[Dict[str, Any]]
    graph_pack: Dict[str, Any]
    fused: List[Dict[str, Any]]
    prompt: str


class QAPipeline:
    def __init__(self, config) -> None:
        self.config = config

    def _import_dual_retrieval(self):
        """延迟导入，避免在未安装 faiss/sentence-transformers 时影响其他模块。"""

        import importlib.util

        # Resolve path to existing module
        root = Path(__file__).resolve().parents[1]
        mod_path = root / "BERT-LSTM-CRF" / "causal_dual_retrieval.py"
        if not mod_path.exists():
            raise FileNotFoundError(f"causal_dual_retrieval.py not found: {mod_path}")

        spec = importlib.util.spec_from_file_location("_causal_dual_retrieval", str(mod_path))
        if spec is None or spec.loader is None:
            raise RuntimeError("Failed to load causal_dual_retrieval module spec")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def retrieve(self, query: str) -> RetrievalBundle:
        c = self.config
        dual = self._import_dual_retrieval()

        g = dual.load_graph(str(c.abs_path(c.nx_graph)))

        text_items = dual.search_faiss(
            query,
            index_path=str(c.abs_path(c.faiss_index)),
            meta_path=str(c.abs_path(c.faiss_meta)),
            model_name=c.embedding_model,
            top_k=int(c.top_text_k),
        )

        graph_pack = dual.top_k_paths_for_query(
            g,
            query=query,
            k=int(c.top_path_k),
            max_hops=int(c.max_hops),
            allowed_relations=tuple(c.allowed_relations),
            ner_model_dir=None if not c.ner_model_dir else str(c.abs_path(c.ner_model_dir)),
            use_ner=bool(c.use_ner),
        )

        fused = dual.fuse_evidence(
            text_items,
            graph_pack.get("paths", []),
            alpha=float(c.alpha),
            beta=float(c.beta),
        )

        prompt = dual.build_causal_prompt(
            question=query,
            text_items=text_items,
            path_items=graph_pack.get("paths", []),
            path_force_threshold=float(c.path_force_threshold),
        )

        return RetrievalBundle(
            query=query,
            text_items=text_items,
            graph_pack=graph_pack,
            fused=fused,
            prompt=prompt,
        )

    def to_debug_json(self, bundle: RetrievalBundle) -> Dict[str, Any]:
        return {
            "query": bundle.query,
            "text_channel": bundle.text_items,
            "graph_channel": bundle.graph_pack,
            "fused_evidence": bundle.fused,
            "prompt": bundle.prompt,
        }

    def save_debug(self, bundle: RetrievalBundle, out_path: str | Path) -> None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(self.to_debug_json(bundle), f, ensure_ascii=False, indent=2)
