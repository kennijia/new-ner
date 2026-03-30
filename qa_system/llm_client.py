"""OpenAI-compatible LLM 客户端封装（复用你现有项目的调用方式）。

目标：
- 和 `BERT-LSTM-CRF/extract_causal_pairs_llm.py` 的配置方式保持一致（.env / base_url / api_key / model）
- 给 QA 系统提供最小能力：输入 prompt，返回文本

备注：
- 这里只封装“调用”，不做具体 Prompt 设计。
- Windows 下也可用（OpenAI SDK 纯 Python）。
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from openai import OpenAI


def load_env_file(env_path: str | Path) -> None:
    """加载 .env（与 extract_causal_pairs_llm.py 类似，但更健壮一些）。"""

    p = Path(env_path)
    if not p.exists():
        return

    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


@dataclass
class LLMConfig:
    env_file: str = ".env"
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    temperature: float = 0.2
    max_retries: int = 3

    def resolve(self) -> "LLMConfig":
        load_env_file(self.env_file)
        model = self.model or os.getenv("MODEL_NAME", "gpt-4o-mini")
        base_url = self.base_url or os.getenv("OPENAI_BASE_URL", "")
        api_key = self.api_key or os.getenv("OPENAI_API_KEY", "")
        return LLMConfig(
            env_file=self.env_file,
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=self.temperature,
            max_retries=self.max_retries,
        )


class LLMClient:
    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg.resolve()
        if not self.cfg.api_key:
            raise ValueError("OPENAI_API_KEY is empty (pass api_key or set env)")
        self.client = OpenAI(api_key=self.cfg.api_key, base_url=self.cfg.base_url or None)

    def chat(self, prompt: str, system: str = "你是严谨的问答助手，必须严格依据证据回答。") -> str:
        last_err: Optional[Exception] = None
        for attempt in range(1, int(self.cfg.max_retries) + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.cfg.model,
                    temperature=float(self.cfg.temperature),
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as exc:
                last_err = exc
                if attempt < int(self.cfg.max_retries):
                    time.sleep(min(2 * attempt, 6))
                    continue
        raise RuntimeError(f"LLM request failed after retries: {last_err}")
