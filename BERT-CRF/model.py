from transformers import BertPreTrainedModel, BertModel, BertConfig
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torchcrf import CRF
import torch

import config
from dice_loss import DiceLoss


class BertNER(BertPreTrainedModel):
    base_model_prefix = "bert"

    def __init__(self, config):
        super(BertNER, self).__init__(config)
        # ensure BERT will return hidden states (for layer fusion) on older transformers versions
        config.output_hidden_states = True
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.use_bilstm = bool(getattr(config, "use_bilstm", True))
        if self.use_bilstm:
            self.bilstm = nn.LSTM(
                config.hidden_size,
                config.hidden_size // 2,
                num_layers=1,
                bidirectional=True,
                batch_first=True,
            )
        else:
            self.bilstm = None

        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(config.num_labels, batch_first=True)

        # Optional auxiliary Dice loss to mitigate label imbalance.
        # We keep it inside the model so training code doesn't have to be rewritten.
        self.use_dice_loss = bool(getattr(config, "use_dice_loss", False))
        self.dice_loss_weight = float(getattr(config, "dice_loss_weight", 0.5))

        # Exclude background ('O', id=0) from Dice average if configured.
        # Priority: HF config attribute -> project config.py attribute.
        _exclude_o = getattr(config, "dice_exclude_o", getattr(__import__("config"), "dice_exclude_o", True))
        self.dice_exclude_o = bool(_exclude_o)

        self.dice_loss_fn = DiceLoss(
            ignore_index=-1,
            include_background=not self.dice_exclude_o,
            background_index=0,
        )
        self.init_weights()

    def tie_weights(self, *args, **kwargs):
        # Transformers v5 may call tie_weights(recompute_mapping=...) during init/finalize.
        # This model has no tied weights.
        return

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Custom loader to avoid Transformers v5 finalize crash (all_tied_weights_keys).

        - If a local directory contains pytorch_model.bin / model.safetensors, treat it as a full
          fine-tuned checkpoint and load the full state dict into the custom model.
        - Otherwise, fall back to initializing from a base BERT directory by loading only the
          underlying BertModel weights.
        """
        import os
        from pathlib import Path

        num_labels = kwargs.pop("num_labels", None)
        hf_config = kwargs.pop("config", None)

        # If caller gives a HF config, keep it; otherwise load.
        model_dir = Path(str(pretrained_model_name_or_path))
        is_local_dir = model_dir.exists() and model_dir.is_dir()

        weight_bin = model_dir / "pytorch_model.bin" if is_local_dir else None
        weight_safe = model_dir / "model.safetensors" if is_local_dir else None

        if hf_config is None:
            if is_local_dir and (model_dir / "config.json").exists():
                hf_config = BertConfig.from_pretrained(str(model_dir))
            else:
                hf_config = BertConfig.from_pretrained(pretrained_model_name_or_path)

        if num_labels is not None:
            hf_config.num_labels = num_labels

        # propagate our project-level switch into HF config so the model can see it
        hf_config.use_bilstm = bool(getattr(config, "use_bilstm", True))

        model = cls(hf_config)

        def _load_state_dict(path: Path) -> dict:
            if path.suffix == ".safetensors":
                from safetensors.torch import load_file
                return load_file(str(path))
            return torch.load(str(path), map_location="cpu")

        if is_local_dir and weight_bin is not None and weight_bin.exists():
            state = _load_state_dict(weight_bin)
            model.load_state_dict(state, strict=False)
            return model

        if is_local_dir and weight_safe is not None and weight_safe.exists():
            state = _load_state_dict(weight_safe)
            model.load_state_dict(state, strict=False)
            return model

        # Base encoder init: load only bert.* weights
        base = BertModel.from_pretrained(pretrained_model_name_or_path)
        model.bert.load_state_dict(base.state_dict(), strict=False)
        return model

    def forward(self, input_data, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, inputs_embeds=None, head_mask=None):
        input_ids, input_token_starts = input_data
        
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)
        
        # Scalar Mix: combine all hidden states
        # support both object and tuple outputs
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states
        elif isinstance(outputs, (tuple, list)) and len(outputs) > 2:
            hidden_states = outputs[2]
        else:
            raise RuntimeError("BERT did not return hidden states. Ensure config.output_hidden_states=True when initializing the model.")
        
        # Calculate Scalar Mix (Disabled)
        # weights = torch.softmax(self.weights, dim=0)
        # sequence_output = torch.stack(hidden_states, dim=0) # (num_layers, batch, seq_len, hidden)
        # sequence_output = (weights.view(-1, 1, 1, 1) * sequence_output).sum(dim=0)
        # sequence_output = sequence_output * self.gamma

        # 简化模型：直接使用最后一层 Hidden State
        sequence_output = hidden_states[-1]

        # 序列对齐
        origin_sequence_output = [layer[starts.nonzero().squeeze(1)]
                                  if starts.nonzero().size(0) > 0 else layer[:0]
                                  for layer, starts in zip(sequence_output, input_token_starts)]
        padded_sequence_output = pad_sequence(origin_sequence_output, batch_first=True)
        
        # optional BiLSTM
        if self.bilstm is not None:
            padded_sequence_output, _ = self.bilstm(padded_sequence_output)
        
        padded_sequence_output = self.dropout(padded_sequence_output)
        logits = self.classifier(padded_sequence_output)
        
        # 调试信息：检查 logits 分布
        # if not self.training:
        #     print(f"Logits mean: {logits.mean().item()}, max: {logits.max().item()}, min: {logits.min().item()}")
        
        outputs = (logits,)
        if labels is not None:
            loss_mask = labels.gt(-1)
            loss_mask[:, 0] = True
            # CRF Loss
            crf_loss = self.crf(logits, labels, loss_mask) * (-1)

            # Dice Loss (token-level, on softmax probabilities)
            if self.use_dice_loss and self.dice_loss_weight > 0:
                dice_loss = self.dice_loss_fn(logits, labels)
                loss = crf_loss + self.dice_loss_weight * dice_loss
            else:
                loss = crf_loss
            outputs = (loss,) + outputs
        return outputs
