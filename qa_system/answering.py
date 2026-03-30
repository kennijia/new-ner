"""LLM 输出约束与答案解析。

目标：
- 让 LLM 按严格 JSON 输出：结论、证据引用、因果链路、（可选）不确定性说明
- 对输出做健壮解析：支持 code fence、支持从混合文本中截取 JSON

该模块用于第五章“分层约束生成控制机制”的实现落地。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.S)


@dataclass
class StructuredAnswer:
    conclusion: str
    evidence_ids: List[str]
    causal_chain: str = ""
    uncertainty: str = ""


def build_answer_prompt(prompt_with_evidence: str) -> str:
    """在已包含证据的 prompt 基础上，追加结构化输出约束。"""

    schema = (
        "\n\n你必须仅输出一个 JSON 对象，严格遵循以下 schema：\n"
        "{\n"
        "  \"conclusion\": string,\n"
        "  \"evidence_ids\": string[],\n"
        "  \"causal_chain\": string,\n"
        "  \"uncertainty\": string\n"
        "}\n\n"
        "要求：\n"
        "1) conclusion：简洁结论/答案；若证据不足，写 '证据不足'。\n"
        "2) evidence_ids：必须引用证据编号，来自给定证据中的 [T#..] 或 [P#..]，例如 ['T#12','P#3']；无则空数组。\n"
        "3) causal_chain：若使用了图谱路径，用自然语言复述关键因果链路；否则可为空字符串。\n"
        "4) uncertainty：若存在冲突或无法确定，请说明不确定性来源；否则为空字符串。\n"
        "5) 禁止输出除 JSON 以外的任何文本。\n"
    )
    return prompt_with_evidence.rstrip() + schema


def _strip_code_fence(text: str) -> str:
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s)
    return s.strip()


def extract_json_obj(text: str) -> Optional[Dict[str, Any]]:
    """从 LLM 输出中提取 JSON 对象。"""

    s = _strip_code_fence(text)
    if not s:
        return None

    # 1) 直接解析
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 2) 从混合文本中截取 { ... }
    m = _JSON_OBJ_RE.search(s)
    if not m:
        return None

    chunk = m.group(0)
    try:
        obj = json.loads(chunk)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None

    return None


def parse_structured_answer(raw: str) -> Tuple[Optional[StructuredAnswer], str]:
    """解析结构化答案。

    Returns:
        (answer, error_message)
    """

    obj = extract_json_obj(raw)
    if obj is None:
        return None, "failed_to_extract_json"

    conclusion = str(obj.get("conclusion", "")).strip()
    if not conclusion:
        return None, "missing_conclusion"

    ev = obj.get("evidence_ids", [])
    if isinstance(ev, list):
        evidence_ids = [str(x).strip() for x in ev if str(x).strip()]
    elif isinstance(ev, str) and ev.strip():
        evidence_ids = [ev.strip()]
    else:
        evidence_ids = []

    causal_chain = str(obj.get("causal_chain", "")).strip()
    uncertainty = str(obj.get("uncertainty", "")).strip()

    return (
        StructuredAnswer(
            conclusion=conclusion,
            evidence_ids=evidence_ids,
            causal_chain=causal_chain,
            uncertainty=uncertainty,
        ),
        "",
    )


def validate_evidence_ids(answer: StructuredAnswer, allowed_ids: List[str]) -> Tuple[bool, str]:
    allowed = set([x.strip() for x in allowed_ids if x and x.strip()])
    for x in answer.evidence_ids:
        if x not in allowed:
            return False, f"invalid_evidence_id:{x}"
    return True, ""
