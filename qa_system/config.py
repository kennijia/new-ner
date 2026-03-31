"""qa_system 配置项。

尽量把路径、阈值和 TopK 参数集中管理，便于论文复现实验与消融。

说明：
- 图谱与双通道检索的底层能力复用 `BERT-LSTM-CRF/causal_dual_retrieval.py`。
- 这里的配置只负责“系统层”的默认值，不强绑定具体数据集。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class QAConfig:
    """问答系统默认配置。"""

    # --- Workspace paths ---
    workspace_root: Path = Path(__file__).resolve().parents[1]

    artifacts_dir: Path = Path(__file__).resolve().parent / "artifacts"

    # --- Data artifacts (defaults follow the existing project conventions) ---
    admin_json: Path = Path("data/my/admin.json")

    # cleaned triples & networkx graph
    triples_clean: Path = artifacts_dir / "triples_clean.jsonl"
    nx_graph: Path = artifacts_dir / "nx_graph.gpickle"
    # vector index
    faiss_index: Path = artifacts_dir / "faiss_index_qwen_sem.bin"
    faiss_meta: Path = artifacts_dir / "faiss_meta_qwen_sem.json"
    embedding_model: str = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-Embedding-0___6B"

    # optional: third-chapter NER model dir (BERT-LSTM-CRF trained artifacts)
    ner_model_dir: Path | None = None
    use_ner: bool = False

    # --- Retrieval params ---
    top_text_k: int = 5
    top_path_k: int = 5
    max_hops: int = 3
    allowed_relations: tuple[str, ...] = ("触发", "导致")

    # --- Evidence fusion params ---
    # fused_score = alpha * text_score_norm + beta * path_conf
    alpha: float = 0.45
    beta: float = 0.55

    # --- Prompt control ---
    path_force_threshold: float = 0.80

    def abs_path(self, p: Path) -> Path:
        """把相对路径转为 workspace 下的绝对路径。"""

        if p.is_absolute():
            return p
        return (self.workspace_root / p).resolve()


DEFAULT_CONFIG = QAConfig()
