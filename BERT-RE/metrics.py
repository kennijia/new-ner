from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class ClsMetrics:
    precision: float
    recall: float
    f1: float
    support: int


def micro_prf(y_true: List[int], y_pred: List[int], positive_ids: List[int]) -> ClsMetrics:
    """Micro P/R/F1 over a subset of labels (typically excluding NoRelation)."""
    pos_set = set(positive_ids)
    tp = fp = fn = 0
    for t, p in zip(y_true, y_pred):
        t_pos = t in pos_set
        p_pos = p in pos_set
        if p_pos and t_pos and p == t:
            tp += 1
        elif p_pos and (not t_pos or p != t):
            fp += 1
        elif (not p_pos) and t_pos:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return ClsMetrics(precision=precision, recall=recall, f1=f1, support=tp + fn)

