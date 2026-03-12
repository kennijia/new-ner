#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
import random
from typing import Dict, List, Any, Tuple

from causal_dual_retrieval import (
    load_jsonl,
    load_graph,
    search_faiss,
    top_k_paths_for_query,
    fuse_evidence,
)


def save_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def normalize_text(x: str) -> str:
    return str(x).strip().replace(" ", "")


def extract_text_matched_candidates(retrieved_texts: List[Dict[str, Any]],
                                   candidates: List[str],
                                   max_matches: int = 10) -> List[str]:
    if not retrieved_texts:
        return []
    joined = "\n".join([str(x.get("text", "")) for x in retrieved_texts])
    out: List[str] = []
    for c in candidates:
        if c and c in joined:
            out.append(c)
            if len(out) >= max_matches:
                break
    return out


def evidence_supports_prediction(pred: str,
                                 retrieved_texts: List[Dict[str, Any]],
                                 paths: List[Dict[str, Any]]) -> bool:
    pred_n = normalize_text(pred)
    if not pred_n:
        return False

    for t in retrieved_texts:
        t_n = normalize_text(t.get("text", ""))
        if pred_n and pred_n in t_n:
            return True

    for p in paths:
        nodes = p.get("nodes", []) or []
        if nodes:
            tail_n = normalize_text(nodes[-1])
            if tail_n == pred_n:
                return True
        logic_n = normalize_text(p.get("logic", ""))
        if pred_n and pred_n in logic_n:
            return True

    return False


def prepare_questions(triples_path: str, out_path: str, n_questions: int = 1000, seed: int = 42) -> List[Dict[str, Any]]:
    triples = load_jsonl(triples_path)
    rng = random.Random(seed)
    rng.shuffle(triples)

    templates = [
        "当{head}时，会触发什么？",
        "出现{head}后，通常导致什么后果？",
        "针对{head}，应采取什么响应措施？",
        "如果发生{head}，下一步可能是什么？",
    ]

    rows: List[Dict[str, Any]] = []
    if not triples:
        save_jsonl(out_path, rows)
        return rows

    i = 0
    while len(rows) < n_questions:
        t = triples[i % len(triples)]
        head = str(t.get("head", "")).strip()
        tail = str(t.get("tail", "")).strip()
        relation = str(t.get("relation", "")).strip()
        if not head or not tail:
            i += 1
            continue

        tpl = templates[len(rows) % len(templates)]
        rows.append(
            {
                "qid": len(rows) + 1,
                "question": tpl.format(head=head),
                "query_entity_hint": head,
                "gold_answer": tail,
                "gold_relation": relation,
                "source_index": t.get("source_index"),
            }
        )
        i += 1

    save_jsonl(out_path, rows)
    return rows


def pick_answer_from_text(retrieved_texts: List[Dict[str, Any]], candidates: List[str]) -> str:
    matches = extract_text_matched_candidates(retrieved_texts, candidates, max_matches=1)
    return matches[0] if matches else ""


def run_ablation(
    question_path: str,
    triples_path: str,
    graph_path: str,
    faiss_index: str,
    faiss_meta: str,
    emb_model: str,
    out_csv: str,
    top_text_k: int = 5,
    top_path_k: int = 5,
    max_hops: int = 3,
    max_questions: int = 0,
    graph_force_threshold: float = 0.88,
    graph_text_margin: float = 0.08,
) -> None:
    questions = load_jsonl(question_path)
    if max_questions and max_questions > 0:
        questions = questions[:max_questions]
    triples = load_jsonl(triples_path)
    g = load_graph(graph_path)

    all_tails = list({str(x.get("tail", "")).strip() for x in triples if str(x.get("tail", "")).strip()})
    sorted_tails = sorted(all_tails, key=lambda x: len(x), reverse=True)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "qid",
                "experiment",
                "question",
                "gold_answer",
                "pred_answer",
                "accuracy",
                "hallucination",
                "top_text_ids",
                "top_path_ids",
            ],
        )
        writer.writeheader()

        for q in questions:
            qid = q["qid"]
            question = q["question"]
            gold = str(q.get("gold_answer", "")).strip()

            texts = search_faiss(question, faiss_index, faiss_meta, emb_model, top_k=top_text_k)
            paths_pack = top_k_paths_for_query(
                g,
                query=question,
                k=top_path_k,
                max_hops=max_hops,
                allowed_relations=("触发", "导致"),
                ner_model_dir=None,
            )
            paths = paths_pack.get("paths", [])

            pred_1 = pick_answer_from_text(texts, sorted_tails)

            pred_2 = ""
            if paths:
                first_path = paths[0]
                nodes = first_path.get("nodes", [])
                if len(nodes) >= 2:
                    pred_2 = nodes[-1]

            fused = fuse_evidence(texts, paths)
            pred_3 = ""
            top_graph_score = float(paths[0].get("score", 0.0)) if paths else 0.0
            top_text_score = float(texts[0].get("score", 0.0)) if texts else 0.0

            use_graph = (
                bool(paths)
                and (
                    top_graph_score >= graph_force_threshold
                    or (top_graph_score - top_text_score) >= graph_text_margin
                )
            )

            candidate_scores: Dict[str, float] = {}

            for p in paths:
                nodes = p.get("nodes", [])
                if len(nodes) < 2:
                    continue
                tail = str(nodes[-1]).strip()
                if not tail:
                    continue
                score = float(p.get("score", p.get("confidence", 0.0)))
                candidate_scores[tail] = candidate_scores.get(tail, 0.0) + 0.70 * score

            text_candidates = extract_text_matched_candidates(texts, sorted_tails, max_matches=8)
            for rank, cand in enumerate(text_candidates, 1):
                rank_discount = 1.0 / rank
                candidate_scores[cand] = candidate_scores.get(cand, 0.0) + 0.30 * top_text_score * rank_discount

            if candidate_scores:
                if use_graph:
                    pred_3 = max(candidate_scores.items(), key=lambda x: x[1])[0]
                else:
                    if pred_1 and pred_1 in candidate_scores:
                        pred_3 = pred_1
                    else:
                        pred_3 = max(candidate_scores.items(), key=lambda x: x[1])[0]

            if not pred_3:
                pred_3 = pred_1

            if not pred_3 and paths:
                nodes = paths[0].get("nodes", [])
                if len(nodes) >= 2:
                    pred_3 = nodes[-1]

            rows = [
                ("exp1_pure_rag", pred_1),
                ("exp2_rag_plus_graph_no_score", pred_2),
                ("exp3_full_dual_scored", pred_3),
            ]

            text_ids = [str(x.get("id")) for x in texts]
            path_ids = [str(x.get("path_id")) for x in paths]

            for exp_name, pred in rows:
                pred_n = normalize_text(pred)
                gold_n = normalize_text(gold)
                acc = 1 if pred_n and gold_n and (pred_n == gold_n or gold_n in pred_n or pred_n in gold_n) else 0
                hallucination = 0 if evidence_supports_prediction(pred, texts, paths) else 1
                writer.writerow(
                    {
                        "qid": qid,
                        "experiment": exp_name,
                        "question": question,
                        "gold_answer": gold,
                        "pred_answer": pred,
                        "accuracy": acc,
                        "hallucination": hallucination,
                        "top_text_ids": "|".join(text_ids),
                        "top_path_ids": "|".join(path_ids),
                    }
                )


def summarize_ablation(result_csv: str, summary_json: str) -> None:
    summary: Dict[str, Dict[str, float]] = {}
    count: Dict[str, int] = {}

    with open(result_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            exp = row["experiment"]
            acc = float(row.get("accuracy", 0))
            hal = float(row.get("hallucination", 0))
            if exp not in summary:
                summary[exp] = {"accuracy_sum": 0.0, "hallucination_sum": 0.0}
                count[exp] = 0
            summary[exp]["accuracy_sum"] += acc
            summary[exp]["hallucination_sum"] += hal
            count[exp] += 1

    out = {}
    for exp, vals in summary.items():
        n = max(1, count[exp])
        out[exp] = {
            "samples": count[exp],
            "accuracy": round(vals["accuracy_sum"] / n, 6),
            "hallucination_rate": round(vals["hallucination_sum"] / n, 6),
        }

    os.makedirs(os.path.dirname(summary_json), exist_ok=True)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablation study toolkit")
    sub = parser.add_subparsers(dest="action", required=True)

    p1 = sub.add_parser("prepare")
    p1.add_argument("--triples", default="data/my/kg/processed/triples_clean.jsonl")
    p1.add_argument("--out", default="data/my/kg/processed/ablation_questions_1000.jsonl")
    p1.add_argument("--n_questions", type=int, default=1000)
    p1.add_argument("--seed", type=int, default=42)

    p2 = sub.add_parser("run")
    p2.add_argument("--questions", default="data/my/kg/processed/ablation_questions_1000.jsonl")
    p2.add_argument("--triples", default="data/my/kg/processed/triples_clean.jsonl")
    p2.add_argument("--graph", default="data/my/kg/processed/nx_graph.gpickle")
    p2.add_argument("--index", default="data/my/kg/processed/faiss_index_qwen_sem.bin")
    p2.add_argument("--meta", default="data/my/kg/processed/faiss_meta_qwen_sem.json")
    p2.add_argument("--emb_model", default="/root/.cache/modelscope/hub/models/Qwen/Qwen3-Embedding-0___6B")
    p2.add_argument("--out_csv", default="data/my/kg/processed/ablation_results.csv")
    p2.add_argument("--max_questions", type=int, default=0)
    p2.add_argument("--graph_force_threshold", type=float, default=0.88)
    p2.add_argument("--graph_text_margin", type=float, default=0.08)

    p3 = sub.add_parser("summary")
    p3.add_argument("--result_csv", default="data/my/kg/processed/ablation_results.csv")
    p3.add_argument("--out_json", default="data/my/kg/processed/ablation_summary.json")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action == "prepare":
        rows = prepare_questions(args.triples, args.out, n_questions=args.n_questions, seed=args.seed)
        print(f"prepared_questions={len(rows)}")
        print(f"output={args.out}")
        return

    if args.action == "run":
        run_ablation(
            question_path=args.questions,
            triples_path=args.triples,
            graph_path=args.graph,
            faiss_index=args.index,
            faiss_meta=args.meta,
            emb_model=args.emb_model,
            out_csv=args.out_csv,
            max_questions=args.max_questions,
            graph_force_threshold=args.graph_force_threshold,
            graph_text_margin=args.graph_text_margin,
        )
        print(f"results={args.out_csv}")
        return

    summarize_ablation(args.result_csv, args.out_json)
    print(f"summary={args.out_json}")


if __name__ == "__main__":
    main()
