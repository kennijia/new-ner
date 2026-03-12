#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import networkx as nx
import numpy as np

try:
    import faiss
except Exception:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from predict import predict_texts
except Exception:
    predict_texts = None


@dataclass
class PathResult:
    path_id: str
    nodes: List[str]
    relations: List[str]
    edge_confidences: List[float]
    confidence: float
    logic: str


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_graph(path: str) -> nx.MultiDiGraph:
    with open(path, "rb") as f:
        return pickle.load(f)


def build_chunks_from_admin(admin_path: str, chunk_size: int = 120, stride: int = 80) -> List[Dict[str, Any]]:
    rows = load_jsonl(admin_path)
    chunks: List[Dict[str, Any]] = []
    cid = 1
    for row in rows:
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        if len(text) <= chunk_size:
            chunks.append({"id": cid, "text": text})
            cid += 1
            continue
        start = 0
        while start < len(text):
            chunk = text[start : start + chunk_size]
            if chunk:
                chunks.append({"id": cid, "text": chunk})
                cid += 1
            if start + chunk_size >= len(text):
                break
            start += stride
    return chunks


def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def l2_normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms


def build_faiss_index(chunks: List[Dict[str, Any]],
                      model_name: str,
                      index_path: str,
                      meta_path: str,
                      batch_size: int = 64) -> None:
    if faiss is None:
        raise RuntimeError("faiss 未安装，请先安装: pip install faiss-cpu")

    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers 未安装，无法执行语义检索。请先安装: pip install sentence-transformers")

    texts = [x["text"] for x in chunks]
    try:
        model = SentenceTransformer(model_name, local_files_only=True)
        embs = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True)
        embs = l2_normalize(embs.astype("float32"))
    except Exception as exc:
        raise RuntimeError(f"语义模型加载或编码失败，未执行回退。请检查模型与网络/缓存: {exc}") from exc

    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)
    save_json(
        meta_path,
        {
            "chunks": chunks,
            "embedding_backend": "semantic",
            "model_name": model_name,
        },
    )


def search_faiss(query: str,
                 index_path: str,
                 meta_path: str,
                 model_name: str,
                 top_k: int = 5) -> List[Dict[str, Any]]:
    if faiss is None:
        return []

    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        return []

    index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_obj = json.load(f)

    if isinstance(meta_obj, dict) and "chunks" in meta_obj:
        meta = meta_obj.get("chunks", [])
        backend = meta_obj.get("embedding_backend", "semantic")
    else:
        meta = meta_obj
        backend = "semantic"

    if backend != "semantic":
        raise RuntimeError("检测到非语义索引（如 TF-IDF）元数据。当前模式仅允许语义检索，请重建 FAISS 索引。")

    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers 未安装，无法执行语义检索。")

    try:
        model = SentenceTransformer(model_name, local_files_only=True)
        q = model.encode([query], convert_to_numpy=True).astype("float32")
        q = l2_normalize(q)
    except Exception as exc:
        raise RuntimeError(f"语义查询编码失败，未执行回退: {exc}") from exc

    scores, ids = index.search(q, top_k)
    result: List[Dict[str, Any]] = []
    for rank, (sid, sc) in enumerate(zip(ids[0], scores[0]), 1):
        if sid < 0 or sid >= len(meta):
            continue
        result.append(
            {
                "id": meta[sid]["id"],
                "text": meta[sid]["text"],
                "score": float(round(sc, 6)),
                "rank": rank,
            }
        )
    return result


def extract_entities_from_query(query: str, model_dir: str = None) -> List[str]:
    if predict_texts is None:
        return []
    try:
        pred = predict_texts([query], model_dir=model_dir)[0]
        ents = [x.get("text", "").strip() for x in pred.get("entities", []) if x.get("text")]
        uniq = []
        seen = set()
        for e in ents:
            if e and e not in seen:
                seen.add(e)
                uniq.append(e)
        return uniq
    except Exception:
        return []


def map_entities_to_graph_nodes(g: nx.MultiDiGraph, entities: List[str], top_n: int = 5) -> Dict[str, List[str]]:
    nodes = list(g.nodes)
    mapping: Dict[str, List[str]] = {}
    for ent in entities:
        scored: List[Tuple[int, str]] = []
        for n in nodes:
            if ent == n:
                scored.append((1000, n))
            elif ent in n:
                scored.append((500 - abs(len(n) - len(ent)), n))
            elif n in ent and len(n) >= 2:
                scored.append((300 - abs(len(n) - len(ent)), n))
        scored.sort(key=lambda x: x[0], reverse=True)
        mapping[ent] = [x[1] for x in scored[:top_n]]
    return mapping


def edge_type_factor(rel: str) -> float:
    if rel == "导致":
        return 1.0
    if rel == "触发":
        return 0.92
    return 0.8


def aggregate_path_conf(conf_list: List[float], rel_list: List[str]) -> float:
    if not conf_list:
        return 0.0
    score = 1.0
    for conf, rel in zip(conf_list, rel_list):
        c = max(0.0, min(1.0, float(conf)))
        score *= c * edge_type_factor(rel)
    hop_penalty = 0.96 ** max(0, len(conf_list) - 1)
    score *= hop_penalty
    return round(float(score), 6)


def enumerate_paths_with_constraints(g: nx.MultiDiGraph,
                                     src: str,
                                     max_hops: int = 3,
                                     allowed_relations: Tuple[str, ...] = ("触发", "导致"),
                                     max_paths: int = 300) -> List[PathResult]:
    if src not in g:
        return []

    allowed = set(allowed_relations)
    results: List[PathResult] = []

    _ = nx.single_source_shortest_path(g.to_undirected(), source=src, cutoff=max_hops)
    successors = nx.dfs_successors(g, source=src, depth_limit=max_hops)

    def dfs(curr: str, nodes: List[str], rels: List[str], confs: List[float], visited: set) -> None:
        if len(rels) > 0:
            conf = aggregate_path_conf(confs, rels)
            logic = " ".join([nodes[0]] + [f"-({r},{c:.2f})-> {n}" for r, c, n in zip(rels, confs, nodes[1:])])
            results.append(
                PathResult(
                    path_id=f"p{len(results)+1}",
                    nodes=list(nodes),
                    relations=list(rels),
                    edge_confidences=list(confs),
                    confidence=conf,
                    logic=logic,
                )
            )
            if len(results) >= max_paths:
                return

        if len(rels) >= max_hops:
            return

        next_candidates = successors.get(curr, [])
        for nxt in next_candidates:
            if nxt in visited:
                continue
            edge_dict = g.get_edge_data(curr, nxt, default={})
            best_rel = None
            best_conf = -1.0
            for _, ed in edge_dict.items():
                rel = str(ed.get("relation", "")).strip()
                if rel not in allowed:
                    continue
                conf = float(ed.get("weight", ed.get("confidence", 1.0)))
                if conf > best_conf:
                    best_conf = conf
                    best_rel = rel
            if best_rel is None:
                continue
            visited.add(nxt)
            dfs(nxt, nodes + [nxt], rels + [best_rel], confs + [best_conf], visited)
            visited.remove(nxt)
            if len(results) >= max_paths:
                return

    dfs(src, [src], [], [], {src})
    results.sort(key=lambda x: x.confidence, reverse=True)
    return results


def top_k_paths_for_query(g: nx.MultiDiGraph,
                          query: str,
                          k: int,
                          max_hops: int,
                          allowed_relations: Tuple[str, ...],
                          ner_model_dir: str = None,
                          use_ner: bool = False) -> Dict[str, Any]:
    entities: List[str] = []
    if use_ner:
        entities = extract_entities_from_query(query, model_dir=ner_model_dir)
    if not entities:
        node_hits = [n for n in g.nodes if isinstance(n, str) and n and n in query]
        node_hits = sorted(node_hits, key=lambda x: len(x), reverse=True)
        entities = node_hits[:5]
    mapping = map_entities_to_graph_nodes(g, entities)

    all_paths: List[PathResult] = []
    for ent, nodes in mapping.items():
        for n in nodes:
            all_paths.extend(
                enumerate_paths_with_constraints(
                    g,
                    src=n,
                    max_hops=max_hops,
                    allowed_relations=allowed_relations,
                    max_paths=120,
                )
            )

    uniq: Dict[Tuple[str, ...], PathResult] = {}
    for p in all_paths:
        key = tuple(p.nodes)
        if key not in uniq or p.confidence > uniq[key].confidence:
            uniq[key] = p

    final_paths = sorted(uniq.values(), key=lambda x: x.confidence, reverse=True)[:k]
    return {
        "query": query,
        "entities": entities,
        "entity_mapping": mapping,
        "paths": [
            {
                "path_id": p.path_id,
                "nodes": p.nodes,
                "relations": p.relations,
                "edge_confidences": p.edge_confidences,
                "logic": p.logic,
                "confidence": p.confidence,
                "score": p.confidence,
            }
            for p in final_paths
        ],
    }


def top_k_paths_from_entity(g: nx.MultiDiGraph,
                            entity: str,
                            k: int = 5,
                            max_hops: int = 3,
                            allowed_relations: Tuple[str, ...] = ("触发", "导致")) -> List[Dict[str, Any]]:
    mapping = map_entities_to_graph_nodes(g, [entity], top_n=5)
    all_paths: List[PathResult] = []
    for n in mapping.get(entity, []):
        all_paths.extend(
            enumerate_paths_with_constraints(
                g,
                src=n,
                max_hops=max_hops,
                allowed_relations=allowed_relations,
                max_paths=200,
            )
        )

    uniq: Dict[Tuple[str, ...], PathResult] = {}
    for p in all_paths:
        key = tuple(p.nodes)
        if key not in uniq or p.confidence > uniq[key].confidence:
            uniq[key] = p

    final_paths = sorted(uniq.values(), key=lambda x: x.confidence, reverse=True)[:k]
    return [
        {
            "path_id": p.path_id,
            "logic": p.logic,
            "nodes": p.nodes,
            "relations": p.relations,
            "edge_confidences": p.edge_confidences,
            "score": p.confidence,
            "confidence": p.confidence,
        }
        for p in final_paths
    ]


def fuse_evidence(text_items: List[Dict[str, Any]],
                  path_items: List[Dict[str, Any]],
                  alpha: float = 0.45,
                  beta: float = 0.55) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for t in text_items:
        merged.append(
            {
                "channel": "text",
                "id": t.get("id"),
                "text": t.get("text", ""),
                "score": float(t.get("score", 0.0)),
                "final_score": round(alpha * float(t.get("score", 0.0)), 6),
            }
        )

    for p in path_items:
        merged.append(
            {
                "channel": "graph",
                "path_id": p.get("path_id"),
                "logic": p.get("logic", ""),
                "confidence": float(p.get("confidence", 0.0)),
                "score": float(p.get("score", p.get("confidence", 0.0))),
                "final_score": round(beta * float(p.get("score", p.get("confidence", 0.0))), 6),
            }
        )

    merged.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
    return merged


def build_causal_prompt(question: str,
                        text_items: List[Dict[str, Any]],
                        path_items: List[Dict[str, Any]],
                        path_force_threshold: float = 0.8) -> str:
    text_lines = []
    for t in text_items:
        text_lines.append(f"- [T#{t.get('id')}] score={t.get('score', 0.0):.3f} | {t.get('text', '')}")

    graph_lines = []
    for p in path_items:
        graph_lines.append(
            f"- [P#{p.get('path_id')}] conf={p.get('confidence', 0.0):.3f} | {p.get('logic', '')}"
        )

    prompt = f"""你是法规防汛问答助手。请仅基于给定证据回答问题。

问题：{question}

文本证据：
{os.linesep.join(text_lines) if text_lines else '- (无)'}

图谱因果路径：
{os.linesep.join(graph_lines) if graph_lines else '- (无)'}

强约束规则（必须遵守）：
1) 若存在图谱路径 confidence > {path_force_threshold:.2f}，答案必须优先遵循该路径的因果逻辑；
2) 严禁仅凭语义相近进行推测，证据不支持的结论必须明确回答“证据不足”；
3) 若多条高置信路径冲突，按 confidence 从高到低解释并给出不确定性；
4) 回答需包含“结论 + 证据编号 + 因果链路”。

输出JSON：
{{
  "answer": "...",
  "used_text_ids": [1,2],
  "used_path_ids": ["p1"],
  "certainty": 0.0
}}
"""
    return prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual-channel retrieval with causal paths")
    sub = parser.add_subparsers(dest="action", required=True)

    p_index = sub.add_parser("build_index")
    p_index.add_argument("--admin", default="data/my/admin.json")
    p_index.add_argument("--index", default="data/my/kg/processed/faiss_index_qwen_sem.bin")
    p_index.add_argument("--meta", default="data/my/kg/processed/faiss_meta_qwen_sem.json")
    p_index.add_argument("--model", default="/root/.cache/modelscope/hub/models/Qwen/Qwen3-Embedding-0___6B")
    p_index.add_argument("--chunk_size", type=int, default=120)
    p_index.add_argument("--stride", type=int, default=80)

    p_query = sub.add_parser("retrieve")
    p_query.add_argument("--query", required=True)
    p_query.add_argument("--graph", default="data/my/kg/processed/nx_graph.gpickle")
    p_query.add_argument("--index", default="data/my/kg/processed/faiss_index_qwen_sem.bin")
    p_query.add_argument("--meta", default="data/my/kg/processed/faiss_meta_qwen_sem.json")
    p_query.add_argument("--model", default="/root/.cache/modelscope/hub/models/Qwen/Qwen3-Embedding-0___6B")
    p_query.add_argument("--ner_model_dir", default=None)
    p_query.add_argument("--use_ner", action="store_true", help="Use trained NER model for entity extraction")
    p_query.add_argument("--top_text_k", type=int, default=5)
    p_query.add_argument("--top_path_k", type=int, default=5)
    p_query.add_argument("--max_hops", type=int, default=3)
    p_query.add_argument("--allowed_relations", default="触发,导致")
    p_query.add_argument("--out", default="")

    p_entity = sub.add_parser("entity_paths")
    p_entity.add_argument("--entity", required=True)
    p_entity.add_argument("--graph", default="data/my/kg/processed/nx_graph.gpickle")
    p_entity.add_argument("--top_k", type=int, default=5)
    p_entity.add_argument("--max_hops", type=int, default=3)
    p_entity.add_argument("--allowed_relations", default="触发,导致")
    p_entity.add_argument("--out", default="")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.action == "build_index":
        chunks = build_chunks_from_admin(args.admin, chunk_size=args.chunk_size, stride=args.stride)
        build_faiss_index(chunks, args.model, args.index, args.meta)
        print(f"chunks={len(chunks)}")
        print(f"index={args.index}")
        print(f"meta={args.meta}")
        return

    if args.action == "entity_paths":
        g = load_graph(args.graph)
        allowed_relations = tuple([x.strip() for x in args.allowed_relations.split(",") if x.strip()])
        paths = top_k_paths_from_entity(
            g,
            entity=args.entity,
            k=args.top_k,
            max_hops=args.max_hops,
            allowed_relations=allowed_relations,
        )
        output = {
            "entity": args.entity,
            "paths": paths,
        }
        if args.out:
            save_json(args.out, output)
            print(f"saved={args.out}")
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return

    g = load_graph(args.graph)
    allowed_relations = tuple([x.strip() for x in args.allowed_relations.split(",") if x.strip()])

    text_items = search_faiss(args.query, args.index, args.meta, args.model, top_k=args.top_text_k)
    graph_pack = top_k_paths_for_query(
        g,
        query=args.query,
        k=args.top_path_k,
        max_hops=args.max_hops,
        allowed_relations=allowed_relations,
        ner_model_dir=args.ner_model_dir,
        use_ner=args.use_ner,
    )

    fused = fuse_evidence(text_items, graph_pack["paths"])
    prompt = build_causal_prompt(args.query, text_items, graph_pack["paths"])

    output = {
        "query": args.query,
        "text_channel": text_items,
        "graph_channel": graph_pack,
        "fused_evidence": fused,
        "prompt": prompt,
    }

    if args.out:
        save_json(args.out, output)
        print(f"saved={args.out}")

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
