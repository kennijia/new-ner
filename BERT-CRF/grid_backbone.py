#!/usr/bin/env python3
"""Grid comparison for different backbones on BERT-CRF.

This script repeatedly runs `run.py` while overriding backbone path/name by env var:
  - BERT_CRF_BACKBONE
  - BERT_CRF_EXP_DIR
  - BERT_CRF_SEED

Optional switches are also supported via env override:
  - BERT_CRF_USE_BILSTM
  - BERT_CRF_USE_FGM
  - BERT_CRF_USE_DICE_LOSS
  - BERT_CRF_DICE_LOSS_WEIGHT
  - BERT_CRF_DICE_EXCLUDE_O

Examples
--------
python grid_backbone.py

python grid_backbone.py \
  --backbones \
    /root/msy/ner/BERT-CRF/pretrained_bert_models/bert-base-chinese \
    /root/msy/ner/BERT-CRF/pretrained_bert_models/chinese_roberta_wwm_large_ext \
    /models/chinese-macbert-base \
  --seeds 42 43 44

python grid_backbone.py \
  --backbones /models/chinese-macbert-base /models/chinese-macbert-large \
  --use-dice --dice-weight 0.05
"""

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List


def _bool_to_env(v: bool) -> str:
    return "1" if v else "0"


def _is_finished(exp_dir: str) -> bool:
    log_path = os.path.join(exp_dir, "train.log")
    if not os.path.exists(log_path):
        return False
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "test loss:" in line and "f1 score:" in line:
                return True
    return False


def _safe_name(backbone: str) -> str:
    # For local path: use folder name.
    # For HF repo id like hfl/chinese-macbert-base: keep readable form.
    backbone = backbone.strip().rstrip("/")
    if os.path.exists(backbone):
        name = os.path.basename(backbone)
    else:
        name = backbone.replace("/", "__")
    name = re.sub(r"[^0-9A-Za-z_.-]+", "-", name)
    return name.strip("-") or "backbone"


def _default_backbones(base_dir: str) -> List[str]:
    root = Path(base_dir) / "pretrained_bert_models"
    return [
        str(root / "bert-base-chinese"),
        str(root / "chinese_roberta_wwm_large_ext"),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backbones",
        nargs="+",
        default=None,
        help="Backbone local paths or HF names. If omitted, use local bert-base + roberta-wwm.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44],
        help="Random seeds to repeat each backbone.",
    )
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Any extra args to pass to run.py (if your run.py supports CLI args).",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Do not skip finished runs (train.log already has test f1).",
    )

    parser.add_argument(
        "--use-bilstm",
        action="store_true",
        help="Enable BiLSTM head (BERT-BiLSTM-CRF). Default keeps current config.py value.",
    )
    parser.add_argument(
        "--disable-fgm",
        action="store_true",
        help="Force disable FGM via env override.",
    )

    parser.add_argument(
        "--use-dice",
        action="store_true",
        help="Enable Dice loss via env override.",
    )
    parser.add_argument(
        "--dice-weight",
        type=float,
        default=0.05,
        help="Dice loss weight when --use-dice is enabled.",
    )
    parser.add_argument(
        "--include-o",
        action="store_true",
        help="Include O/background class in Dice averaging (default excludes O).",
    )

    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    backbones = args.backbones if args.backbones else _default_backbones(base_dir)

    print("=== Backbone grid (BERT-CRF) ===")
    print("Backbones:")
    for b in backbones:
        print(" -", b)
    print("Seeds:", args.seeds)
    print("Timestamp:", ts)

    total = len(backbones) * len(args.seeds)
    done = 0

    for backbone in backbones:
        backbone_name = _safe_name(backbone)
        for seed in args.seeds:
            exp_name = f"grid_backbone_{backbone_name}_seed{seed}_{ts}"
            exp_dir = os.path.join(base_dir, "experiments", exp_name)
            os.makedirs(exp_dir, exist_ok=True)

            if (not args.no_skip) and _is_finished(exp_dir):
                done += 1
                print(f"SKIP finished: backbone={backbone}, seed={seed} -> {exp_name}")
                continue

            env = os.environ.copy()
            env["BERT_CRF_BACKBONE"] = backbone
            env["BERT_CRF_EXP_DIR"] = exp_dir
            env["BERT_CRF_SEED"] = str(seed)

            if args.use_bilstm:
                env["BERT_CRF_USE_BILSTM"] = "1"

            if args.disable_fgm:
                env["BERT_CRF_USE_FGM"] = "0"

            if args.use_dice:
                env["BERT_CRF_USE_DICE_LOSS"] = "1"
                env["BERT_CRF_DICE_LOSS_WEIGHT"] = str(args.dice_weight)
                env["BERT_CRF_DICE_EXCLUDE_O"] = _bool_to_env(not args.include_o)

            done += 1
            print(f"\n--- RUN ({done}/{total}) backbone={backbone_name}, seed={seed} ---")
            cmd = [sys.executable, os.path.join(base_dir, "run.py"), *args.extra_args]
            subprocess.run(cmd, cwd=base_dir, env=env, check=True)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
