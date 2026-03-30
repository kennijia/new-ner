"""qa_system 数据结构定义。

这里的 schema 用于把“文本证据”“图路径证据”“融合结果”“最终答案”统一成可序列化对象，
方便：
- 输出 debug JSON（论文复现实验）
- 证据排序与消融（alpha/beta、阈值等）
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass
class TextEvidence:
    id: str
    score: float
    text: str
    meta: Dict[str, Any] | None = None


@dataclass
class PathEvidence:
    path_id: str
    confidence: float
    logic: str
    nodes: List[str]
    relations: List[str]
    edge_confidences: List[float]


@dataclass
class FusedEvidence:
    evidence_type: str  # 'text' | 'path'
    evidence_id: str
    score: float
    payload: Dict[str, Any]


@dataclass
class Answer:
    conclusion: str
    evidence_ids: List[str]
    causal_chain: str = ""
    uncertainty: str = ""


def to_jsonable(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, list):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    return obj
