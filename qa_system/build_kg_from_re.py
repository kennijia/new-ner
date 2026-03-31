#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""从关系抽取标注数据（admin-ree.jsonl）构建第五章问答系统所需的因果知识图谱工件。

输入（RE 标注格式，来自 label-studio/自定义标注）：
- 每行 JSON：包含 text/entities/relations

输出（写入 qa_system/artifacts 目录，便于 QA 直接引用）：
- triples_clean.jsonl: {head, relation, tail, evidence, confidence}
- nx_graph.gpickle: NetworkX 有向图（边属性包含 relation/confidence/evidence）

说明：
- 本脚本优先保证“可复现 + 与现有检索模块兼容”。
- relation 映射：
  - Trigger -> 触发
  - Causes -> 导致（若数据集中存在）
  - Condition/Attribute_of 等非因果关系默认跳过（可用 --keep_non_causal 保留）
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class Triple:
    head: str
    relation: str
    tail: str
    evidence: str
    confidence: float


def _span_text(text: str, ent: Dict[str, Any]) -> str:
    try:
        s = int(ent.get("start_offset"))
        e = int(ent.get("end_offset"))
        if 0 <= s < e <= len(text):
            return text[s:e].strip()
    except Exception:
        pass
    return ""


def _build_entity_map(text: str, entities: List[Dict[str, Any]]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for ent in entities or []:
        eid = ent.get("id")
        if eid is None:
            continue
        name = _span_text(text, ent)
        if not name:
            continue
        out[int(eid)] = name
    return out


def _normalize_relation(rel_type: str) -> str:
    t = (rel_type or "").strip()
    mapping = {
        "Trigger": "触发",
        "Causes": "导致",
        "Cause": "导致",
        "CAUSES": "导致",
        "导致": "导致",
        "触发": "触发",
    }
    return mapping.get(t, t)


def iter_re_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build_triples(
    records: Iterable[Dict[str, Any]],
    *,
    keep_non_causal: bool = False,
    default_confidence: float = 0.98,
) -> List[Triple]:
    triples: List[Triple] = []
    for rec in records:
        text = str(rec.get("text", "")).strip()
        if not text:
            continue

        ent_map = _build_entity_map(text, rec.get("entities") or [])

        for r in rec.get("relations") or []:
            r_type = _normalize_relation(str(r.get("type", "")))

            # 仅保留因果边（默认）
            if not keep_non_causal and r_type not in ("触发", "导致"):
                continue

            fid = r.get("from_id")
            tid = r.get("to_id")
            if fid is None or tid is None:
                continue

            head = ent_map.get(int(fid), "").strip()
            tail = ent_map.get(int(tid), "").strip()
            if not head or not tail:
                continue

            # admin-ree.jsonl 里通常没有 confidence，这里用默认值
            confidence = float(r.get("confidence", default_confidence))

            triples.append(
                Triple(
                    head=head,
                    relation=r_type,
                    tail=tail,
                    evidence=text,
                    confidence=confidence,
                )
            )

    return triples


def dedup_triples(triples: List[Triple]) -> List[Triple]:
    # 对 (head, relation, tail) 去重；合并 evidence；confidence 取 max
    merged: Dict[Tuple[str, str, str], Triple] = {}
    for t in triples:
        key = (t.head, t.relation, t.tail)
        if key not in merged:
            merged[key] = t
        else:
            prev = merged[key]
            conf = max(prev.confidence, t.confidence)
            evidence = prev.evidence
            if t.evidence and t.evidence not in evidence:
                evidence = (evidence + " | " + t.evidence).strip(" |")
            merged[key] = Triple(
                head=prev.head,
                relation=prev.relation,
                tail=prev.tail,
                evidence=evidence,
                confidence=conf,
            )
    return list(merged.values())


def save_triples_jsonl(triples: List[Triple], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for t in triples:
            f.write(
                json.dumps(
                    {
                        "head": t.head,
                        "relation": t.relation,
                        "tail": t.tail,
                        "evidence": t.evidence,
                        "confidence": t.confidence,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def build_nx_graph(triples: List[Triple], out_path: Path) -> None:
    import networkx as nx

    # causal_dual_retrieval.py 里按 MultiDiGraph 遍历边：
    # edge_dict = g.get_edge_data(u, v) -> {key: {attr...}}
    # for _, ed in edge_dict.items(): ed 应为 dict
    # 若这里用 DiGraph，则 edge_dict.items() 会遍历 attr/value，导致 ed 变成 str 而报错。
    g = nx.MultiDiGraph()

    for t in triples:
        g.add_node(t.head)
        g.add_node(t.tail)

        # 若同一 (head, tail, relation) 已存在，保留置信度更高的那条边
        edge_dict = g.get_edge_data(t.head, t.tail, default={}) or {}
        existing_key = None
        existing_conf = -1.0

        for k, ed in edge_dict.items():
            if isinstance(ed, dict) and str(ed.get("relation", "")).strip() == str(t.relation).strip():
                existing_key = k
                existing_conf = float(ed.get("confidence", 0.0))
                break

        if existing_key is not None and existing_conf >= float(t.confidence):
            continue
        if existing_key is not None:
            g.remove_edge(t.head, t.tail, key=existing_key)

        g.add_edge(
            t.head,
            t.tail,
            relation=t.relation,
            confidence=float(t.confidence),
            weight=float(t.confidence),   # 兼容检索代码读 weight/confidence
            evidence=t.evidence,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # networkx 3.x 移除了顶层 write_gpickle
    try:
        from networkx.readwrite.gpickle import write_gpickle  # type: ignore
        write_gpickle(g, out_path)
    except Exception:
        import pickle
        with out_path.open("wb") as f:
            pickle.dump(g, f, protocol=pickle.HIGHEST_PROTOCOL)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build causal KG artifacts from admin-ree.jsonl")
    p.add_argument(
        "--input",
        default="/root/msy/ner/BERT-RE/admin-ree.jsonl",
        help="Path to RE dataset jsonl (admin-ree.jsonl)",
    )
    p.add_argument(
        "--artifacts_dir",
        default="/root/msy/ner/qa_system/artifacts",
        help="Output artifacts directory",
    )
    p.add_argument("--keep_non_causal", action="store_true", help="Keep non-causal relations too")
    p.add_argument("--default_conf", type=float, default=0.98, help="Default confidence for relations")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    inp = Path(args.input)
    out_dir = Path(args.artifacts_dir)

    triples_path = out_dir / "triples_clean.jsonl"
    graph_path = out_dir / "nx_graph.gpickle"

    records = list(iter_re_jsonl(inp))
    triples = build_triples(records, keep_non_causal=bool(args.keep_non_causal), default_confidence=float(args.default_conf))
    triples = dedup_triples(triples)

    save_triples_jsonl(triples, triples_path)
    build_nx_graph(triples, graph_path)

    print(json.dumps({"input": str(inp), "records": len(records), "triples": len(triples), "out_triples": str(triples_path), "out_graph": str(graph_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
