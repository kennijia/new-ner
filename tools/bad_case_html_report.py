#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert bad_case.txt into an easy-to-read HTML report.

Supports the common format used in this repo:

bad case k:
sentence: [...]
golden label: [...]
model pred: [...]

Output:
- token-level table highlighting mismatches
- entity spans extracted from BIO/BIOES-like tags (B-/I-/S-, O)

Usage:
  python /root/msy/ner/tools/bad_case_html_report.py \
    --input /root/msy/ner/BERT-CRF/case/bad_case.txt \
    --output /root/msy/ner/BERT-CRF/case/bad_case_report.html

Optionally limit cases:
  --max_cases 50
"""

from __future__ import annotations

import argparse
import ast
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any


CASE_RE = re.compile(r"^bad case\s+(\d+)\s*:\s*$", re.IGNORECASE)
SENT_RE = re.compile(r"^sentence:\s*(\[.*\])\s*$")
GOLD_RE = re.compile(r"^golden label:\s*(\[.*\])\s*$")
PRED_RE = re.compile(r"^model pred:\s*(\[.*\])\s*$")


@dataclass
class Case:
    idx: int
    tokens: List[str]
    gold: List[str]
    pred: List[str]


def _safe_list_literal(s: str) -> List[Any]:
    """Parse a Python-like list literal from the text file."""
    try:
        val = ast.literal_eval(s)
    except Exception as e:
        raise ValueError(f"Failed to parse list literal: {s[:120]}...") from e
    if not isinstance(val, list):
        raise ValueError("Parsed value is not a list")
    return val


def parse_bad_case_file(path: Path, max_cases: Optional[int] = None) -> List[Case]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    cases: List[Case] = []
    cur_idx: Optional[int] = None
    cur_tokens: Optional[List[str]] = None
    cur_gold: Optional[List[str]] = None
    cur_pred: Optional[List[str]] = None

    def flush():
        nonlocal cur_idx, cur_tokens, cur_gold, cur_pred
        if cur_idx is None:
            return
        if cur_tokens is None or cur_gold is None or cur_pred is None:
            # incomplete block; skip
            cur_idx = None
            cur_tokens = None
            cur_gold = None
            cur_pred = None
            return
        cases.append(Case(idx=cur_idx, tokens=cur_tokens, gold=cur_gold, pred=cur_pred))
        cur_idx = None
        cur_tokens = None
        cur_gold = None
        cur_pred = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        m = CASE_RE.match(line)
        if m:
            if max_cases is not None and len(cases) >= max_cases:
                break
            flush()
            cur_idx = int(m.group(1))
            continue

        m = SENT_RE.match(line)
        if m:
            cur_tokens = [str(x) for x in _safe_list_literal(m.group(1))]
            continue

        m = GOLD_RE.match(line)
        if m:
            cur_gold = [str(x) for x in _safe_list_literal(m.group(1))]
            continue

        m = PRED_RE.match(line)
        if m:
            cur_pred = [str(x) for x in _safe_list_literal(m.group(1))]
            continue

    flush()
    return cases


def _tag_type(tag: str) -> Tuple[str, Optional[str]]:
    """Return (prefix, type) for tags like B-ORG / I-ORG / S-ORG / O."""
    tag = tag.strip()
    if tag == "O" or tag == "-1" or tag == "":
        return "O", None
    if "-" not in tag:
        return tag, None
    p, t = tag.split("-", 1)
    return p, t


def extract_entities(tokens: List[str], tags: List[str]) -> List[Dict[str, Any]]:
    """Extract entity spans from BIO/BIOES-ish tags.

    Output list of {type, start, end, text}
    where end is inclusive.
    """
    ents: List[Dict[str, Any]] = []
    start: Optional[int] = None
    cur_type: Optional[str] = None

    def close(i: int):
        nonlocal start, cur_type
        if start is None or cur_type is None:
            start = None
            cur_type = None
            return
        end = i - 1
        text = "".join(tokens[start : end + 1])
        ents.append({"type": cur_type, "start": start, "end": end, "text": text})
        start = None
        cur_type = None

    for i, tag in enumerate(tags):
        p, t = _tag_type(tag)
        if p == "O" or t is None:
            close(i)
            continue

        if p == "S":
            close(i)
            ents.append({"type": t, "start": i, "end": i, "text": tokens[i]})
            continue

        if p == "B":
            close(i)
            start = i
            cur_type = t
            continue

        if p == "I":
            # continue only if same type; otherwise start a new span
            if cur_type != t or start is None:
                close(i)
                start = i
                cur_type = t
            continue

        # unknown prefix
        close(i)

    close(len(tags))
    return ents


def _escape(s: str) -> str:
    return html.escape(s, quote=True)


def build_html(cases: List[Case], title: str) -> str:
    # Basic self-contained HTML
    css = """
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans', 'PingFang SC', 'Microsoft YaHei', sans-serif; margin: 24px; }
    h1 { font-size: 20px; margin-bottom: 16px; }
    .meta { color: #666; font-size: 12px; margin-bottom: 16px; }
    .case { border: 1px solid #e5e7eb; border-radius: 10px; padding: 14px; margin: 14px 0; }
    .case h2 { font-size: 16px; margin: 0 0 10px 0; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .panel { background: #fafafa; border: 1px solid #eee; border-radius: 8px; padding: 10px; }
    .panel h3 { font-size: 13px; margin: 0 0 6px 0; }
    .panel pre { margin: 0; white-space: pre-wrap; word-break: break-all; font-size: 12px; }
    table { width: 100%; border-collapse: collapse; margin-top: 10px; }
    th, td { border: 1px solid #e5e7eb; padding: 6px 8px; font-size: 12px; text-align: left; }
    th { background: #f3f4f6; position: sticky; top: 0; }
    tr.mis { background: #fff1f2; }
    .tag { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
    .pill { display: inline-block; padding: 2px 8px; border-radius: 999px; background: #eef2ff; border: 1px solid #e0e7ff; margin: 2px 4px 2px 0; font-size: 12px; }
    .pill.bad { background: #fff1f2; border-color: #fecdd3; }
    .toolbar { display: flex; gap: 10px; align-items: center; margin-bottom: 10px; }
    .toolbar input { padding: 6px 10px; border: 1px solid #e5e7eb; border-radius: 8px; }
    .toolbar label { font-size: 12px; color: #333; }
    """

    js = """
    function applyFilter() {
      const q = (document.getElementById('q').value || '').trim();
      const onlyMis = document.getElementById('onlyMis').checked;
      const cases = document.querySelectorAll('.case');
      for (const c of cases) {
        const text = c.getAttribute('data-text');
        const hasMis = c.getAttribute('data-has-mis') === '1';
        const okQ = q === '' || (text && text.indexOf(q) !== -1);
        const okMis = !onlyMis || hasMis;
        c.style.display = (okQ && okMis) ? '' : 'none';
      }
    }
    """

    parts: List[str] = []
    parts.append("<!doctype html>")
    parts.append("<html><head><meta charset='utf-8'>")
    parts.append(f"<title>{_escape(title)}</title>")
    parts.append(f"<style>{css}</style>")
    parts.append(f"<script>{js}</script>")
    parts.append("</head><body>")
    parts.append(f"<h1>{_escape(title)}</h1>")
    parts.append(f"<div class='meta'>cases: {len(cases)}</div>")

    parts.append(
        "<div class='toolbar'>"
        "<label>Search:</label>"
        "<input id='q' placeholder='输入任意 token/短语...' oninput='applyFilter()' />"
        "<label><input type='checkbox' id='onlyMis' onchange='applyFilter()' /> 只看有错误的 case</label>"
        "</div>"
    )

    for c in cases:
        n = min(len(c.tokens), len(c.gold), len(c.pred))
        has_mis = any(c.gold[i] != c.pred[i] for i in range(n)) or (len(c.gold) != len(c.pred))
        text_preview = "".join(c.tokens)

        gold_ents = extract_entities(c.tokens[:n], c.gold[:n])
        pred_ents = extract_entities(c.tokens[:n], c.pred[:n])

        parts.append(
            f"<div class='case' id='case-{c.idx}' data-has-mis='{1 if has_mis else 0}' "
            f"data-text='{_escape(text_preview)}'>"
        )
        parts.append(f"<h2>bad case {c.idx}</h2>")

        parts.append("<div class='grid'>")
        parts.append("<div class='panel'><h3>Gold entities</h3><div>")
        if gold_ents:
            for e in gold_ents:
                parts.append(f"<span class='pill'>{_escape(e['type'])}: {_escape(e['text'])} [{e['start']},{e['end']}]</span>")
        else:
            parts.append("<span class='pill'>None</span>")
        parts.append("</div></div>")

        parts.append("<div class='panel'><h3>Pred entities</h3><div>")
        if pred_ents:
            for e in pred_ents:
                cls = "pill"
                parts.append(f"<span class='{cls}'>{_escape(e['type'])}: {_escape(e['text'])} [{e['start']},{e['end']}]</span>")
        else:
            parts.append("<span class='pill'>None</span>")
        parts.append("</div></div>")
        parts.append("</div>")

        parts.append("<table>")
        parts.append("<thead><tr><th>#</th><th>token</th><th>gold</th><th>pred</th></tr></thead><tbody>")
        for i in range(n):
            mis = c.gold[i] != c.pred[i]
            tr_cls = " class='mis'" if mis else ""
            parts.append(
                f"<tr{tr_cls}><td>{i}</td><td>{_escape(c.tokens[i])}</td>"
                f"<td class='tag'>{_escape(c.gold[i])}</td><td class='tag'>{_escape(c.pred[i])}</td></tr>"
            )
        parts.append("</tbody></table>")

        if len(c.tokens) != len(c.gold) or len(c.tokens) != len(c.pred):
            parts.append("<div class='panel' style='margin-top:10px;'><h3>Length warning</h3><pre>")
            parts.append(_escape(json.dumps({"tokens": len(c.tokens), "gold": len(c.gold), "pred": len(c.pred)}, ensure_ascii=False)))
            parts.append("</pre></div>")

        parts.append("</div>")

    parts.append("</body></html>")
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to bad_case.txt")
    ap.add_argument("--output", required=True, help="Path to output HTML")
    ap.add_argument("--max_cases", type=int, default=None)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cases = parse_bad_case_file(in_path, max_cases=args.max_cases)
    title = args.title or f"Bad case report: {in_path}"

    html_text = build_html(cases, title=title)
    out_path.write_text(html_text, encoding="utf-8")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
