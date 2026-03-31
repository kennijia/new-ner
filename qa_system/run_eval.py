"""批量评测入口（用于第五章对比/消融实验的可复现批跑）。

读取 JSONL 问题集，对每条 query 执行检索/融合流水线，并逐行写出 JSONL 结果。

建议用法：
    python -m qa_system.run_eval --in qa_system/eval_questions.jsonl --out qa_system/outputs/eval_fusion.jsonl --mode fusion

mode 说明：
- text: 仅保留文本证据（text_channel + prompt 中不含图谱路径）
- graph: 仅保留图谱路径证据
- dual: 同时输出 text_channel 与 graph_channel，但 fused_evidence 不做融合（简单拼接）
- fusion: 默认融合（alpha/beta），与 run_qa.py 行为一致
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Literal

from qa_system.config import DEFAULT_CONFIG
from qa_system.pipeline import QAPipeline

Mode = Literal["text", "graph", "dual_concat", "fusion"]


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception as e:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {e}") from e


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _as_text_only(debug: Dict[str, Any]) -> Dict[str, Any]:
    debug = dict(debug)
    debug["graph_channel"] = {"query": debug.get("query"), "entities": [], "entity_mapping": {}, "paths": []}
    debug["fused_evidence"] = [e for e in debug.get("fused_evidence", []) if e.get("channel") == "text"]
    return debug


def _as_graph_only(debug: Dict[str, Any]) -> Dict[str, Any]:
    debug = dict(debug)
    debug["text_channel"] = []
    debug["fused_evidence"] = [e for e in debug.get("fused_evidence", []) if e.get("channel") == "graph"]
    return debug


def _as_dual_no_fusion(debug: Dict[str, Any]) -> Dict[str, Any]:
    """dual_concat：不做融合打分/排序，仅拼接两路证据供 prompt 使用。"""

    debug = dict(debug)

    fused: list[dict[str, Any]] = []

    # text first (keep original text rank order)
    for t in debug.get("text_channel", []) or []:
        fused.append(
            {
                "channel": "text",
                "id": t.get("id"),
                "text": t.get("text"),
                "score": t.get("score"),
            }
        )

    # graph second (keep original path order)
    for p in debug.get("graph_channel", {}).get("paths", []) or []:
        fused.append(
            {
                "channel": "graph",
                "path_id": p.get("path_id"),
                "logic": p.get("logic"),
                "confidence": p.get("confidence"),
                "score": p.get("score"),
            }
        )

    debug["fused_evidence"] = fused
    return debug


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="JSONL 问题集路径")
    ap.add_argument("--out", dest="out_path", required=True, help="输出 JSONL 结果路径")
    ap.add_argument(
        "--mode",
        choices=["text", "graph", "dual_concat", "fusion"],
        default="fusion",
        help="评测模式：text/graph/dual_concat/fusion",
    )
    ap.add_argument("--limit", type=int, default=0, help="仅跑前 N 条，0 表示不限制")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    mode: Mode = args.mode
    limit = int(args.limit)

    pipe = QAPipeline(DEFAULT_CONFIG)

    rows = []
    for i, item in enumerate(_iter_jsonl(in_path), start=1):
        if limit and i > limit:
            break

        qid = item.get("qid")
        qtype = item.get("type")
        query = item.get("query")
        if not query or not isinstance(query, str):
            raise ValueError(f"Missing field 'query' in item #{i}: {item}")

        bundle = pipe.retrieve(query)
        debug = pipe.to_debug_json(bundle)
        debug["qid"] = qid
        debug["type"] = qtype

        if mode == "text":
            debug = _as_text_only(debug)
        elif mode == "graph":
            debug = _as_graph_only(debug)
        elif mode == "dual_concat":
            debug = _as_dual_no_fusion(debug)
        else:
            pass

        rows.append(debug)
        print(f"[{i}] {qid or ''} mode={mode} query={query}")

    _write_jsonl(out_path, rows)
    print(f"Saved: {out_path} (n={len(rows)})")


if __name__ == "__main__":
    main()
