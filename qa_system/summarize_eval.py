"""Summarize evaluation outputs from qa_system.run_eval.

This script is intentionally lightweight (no pandas dependency) and produces:
- outputs/metrics_summary.csv: overall metrics per run file
- outputs/metrics_by_type.csv: metrics per (run file, question type)

Expected input files are JSONL (one JSON object per line) produced by `python -m qa_system.run_eval`.
Each JSON object should contain at least:
- qid, type
- text_channel: list[{'score': float, ...}]
- graph_channel: {'entities': list[str], 'paths': list[{'confidence': float, ...}], ...}
- fused_evidence: list[{'channel': 'text'|'graph', ...}] (optional for non-fusion modes)

Usage:
  python -m qa_system.summarize_eval \
    --questions /root/msy/ner/qa_system/eval_questions.jsonl \
    --runs /root/msy/ner/qa_system/outputs/eval_text.jsonl \
           /root/msy/ner/qa_system/outputs/eval_graph.jsonl \
           /root/msy/ner/qa_system/outputs/eval_dual.jsonl \
           /root/msy/ner/qa_system/outputs/eval_fusion.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass
class Agg:
    n: int = 0
    entity_hit: int = 0
    path_hit: int = 0
    num_paths_sum: int = 0
    path_conf_sum: float = 0.0
    path_conf_n: int = 0
    text_top1_sum: float = 0.0
    text_top1_n: int = 0
    fusion_top1_graph_hit: int = 0
    fusion_top1_n: int = 0

    # topK graph share (computed from fused_evidence)
    topk_graph_share_sum: float = 0.0
    topk_graph_share_n: int = 0

    def add(self, row: dict[str, Any], *, is_fusion: bool) -> None:
        self.n += 1

        graph = row.get("graph_channel") or {}
        entities = graph.get("entities") or []
        paths = graph.get("paths") or []

        if len(entities) > 0:
            self.entity_hit += 1
        if len(paths) > 0:
            self.path_hit += 1

        self.num_paths_sum += len(paths)
        for p in paths:
            conf = p.get("confidence")
            if isinstance(conf, (int, float)):
                self.path_conf_sum += float(conf)
                self.path_conf_n += 1

        text = row.get("text_channel") or []
        if isinstance(text, list) and len(text) > 0:
            score = text[0].get("score") if isinstance(text[0], dict) else None
            if isinstance(score, (int, float)):
                self.text_top1_sum += float(score)
                self.text_top1_n += 1

        if is_fusion:
            fused = row.get("fused_evidence") or []
            if isinstance(fused, list) and len(fused) > 0 and isinstance(fused[0], dict):
                self.fusion_top1_n += 1
                if fused[0].get("channel") == "graph":
                    self.fusion_top1_graph_hit += 1


        # topK graph share for any run that has fused_evidence (fusion/dual/text/graph)
        fused_any = row.get("fused_evidence") or []
        if isinstance(fused_any, list) and len(fused_any) > 0:
            k = 5
            head = fused_any[:k]
            graph_cnt = 0
            for ev in head:
                if isinstance(ev, dict) and ev.get("channel") == "graph":
                    graph_cnt += 1
            self.topk_graph_share_sum += graph_cnt / float(len(head))
            self.topk_graph_share_n += 1

    def to_metrics(self) -> dict[str, Any]:
        def safe_div(a: float, b: float) -> float:
            return float(a) / float(b) if b else 0.0

        return {
            "n": self.n,
            "entity_hit_rate": safe_div(self.entity_hit, self.n),
            "path_hit_rate": safe_div(self.path_hit, self.n),
            "avg_num_paths": safe_div(self.num_paths_sum, self.n),
            "avg_path_conf": safe_div(self.path_conf_sum, self.path_conf_n),
            "avg_text_top1_score": safe_div(self.text_top1_sum, self.text_top1_n),
            "fusion_top1_graph_rate": safe_div(self.fusion_top1_graph_hit, self.fusion_top1_n),
            "top5_graph_share": safe_div(self.topk_graph_share_sum, self.topk_graph_share_n),
        }


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception as e:  # noqa: BLE001
                raise ValueError(f"Invalid JSONL at {path}:{ln}: {e}\nLINE={s[:200]}") from e
            if not isinstance(obj, dict):
                continue
            yield obj


def load_question_types(questions_path: Path) -> dict[str, str]:
    """Return qid -> type mapping (fallback: {})."""

    mapping: dict[str, str] = {}
    for row in iter_jsonl(questions_path):
        qid = row.get("qid")
        qtype = row.get("type")
        if isinstance(qid, str) and isinstance(qtype, str):
            mapping[qid] = qtype
    return mapping


def detect_is_fusion(run_path: Path) -> bool:
    name = run_path.name.lower()
    if "fusion" in name:
        return True
    # fallback: check one sample line for explicit mode
    for row in iter_jsonl(run_path):
        if row.get("mode") == "fusion":
            return True
        break
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=Path, required=True, help="eval_questions.jsonl")
    ap.add_argument("--runs", type=Path, nargs="+", required=True, help="one or more eval_*.jsonl outputs")
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="where to write csv files (default: qa_system/outputs)",
    )
    args = ap.parse_args()

    qid2type = load_question_types(args.questions)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    overall_rows: list[dict[str, Any]] = []
    by_type_rows: list[dict[str, Any]] = []

    for run_path in args.runs:
        run_path = run_path.resolve()
        is_fusion = detect_is_fusion(run_path)

        overall = Agg()
        by_type: dict[str, Agg] = defaultdict(Agg)

        for row in iter_jsonl(run_path):
            qid = row.get("qid")
            qtype = row.get("type")
            if not isinstance(qtype, str):
                if isinstance(qid, str) and qid in qid2type:
                    qtype = qid2type[qid]
                else:
                    qtype = "unknown"

            overall.add(row, is_fusion=is_fusion)
            by_type[qtype].add(row, is_fusion=is_fusion)

        m = overall.to_metrics()
        m.update({"run": run_path.name})
        overall_rows.append(m)

        for qtype, agg in sorted(by_type.items(), key=lambda x: x[0]):
            mm = agg.to_metrics()
            mm.update({"run": run_path.name, "type": qtype})
            by_type_rows.append(mm)

    # Write CSVs
    overall_csv = args.out_dir / "metrics_summary.csv"
    by_type_csv = args.out_dir / "metrics_by_type.csv"

    overall_fields = [
        "run",
        "n",
        "entity_hit_rate",
        "path_hit_rate",
        "avg_num_paths",
        "avg_path_conf",
        "avg_text_top1_score",
        "fusion_top1_graph_rate",
        "top5_graph_share",
    ]
    by_type_fields = ["run", "type"] + [f for f in overall_fields if f != "run"]

    with overall_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=overall_fields)
        w.writeheader()
        for r in overall_rows:
            w.writerow(r)

    with by_type_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=by_type_fields)
        w.writeheader()
        for r in by_type_rows:
            w.writerow(r)

    print(f"Wrote: {overall_csv}")
    print(f"Wrote: {by_type_csv}")


if __name__ == "__main__":
    main()
