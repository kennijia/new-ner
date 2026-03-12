from __future__ import annotations

import os
import torch
from transformers import BertConfig, BertForSequenceClassification


def build_model(model_name_or_path: str, num_labels: int):
    """Build a BERT-based relation classifier.

    The local `chinese_roberta_wwm_large_ext` checkpoint in this workspace ships a
    non-standard `config.json` (no `model_type`). HF AutoConfig will fail, so we
    instantiate a compatible BertConfig and load weights from `pytorch_model.bin`.
    """

    state_path = os.path.join(model_name_or_path, "pytorch_model.bin")
    if os.path.exists(state_path):
        cfg = BertConfig(
            vocab_size=21128,
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=4096,
            max_position_embeddings=512,
            type_vocab_size=2,
        )
        cfg.num_labels = num_labels
        model = BertForSequenceClassification(cfg)
        sd = torch.load(state_path, map_location="cpu", weights_only=False)
        missing, unexpected = model.bert.load_state_dict(sd, strict=False)
        # classifier head is randomly initialized (expected)
        _ = (missing, unexpected)
        return model

    # Fallback for standard HF checkpoints
    from transformers import AutoModelForSequenceClassification

    return AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
    )

