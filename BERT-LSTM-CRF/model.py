import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, BertConfig
from torch.nn.utils.rnn import pad_sequence
from torchcrf import CRF
from pathlib import Path


class BertNER(BertPreTrainedModel):
    # Transformers uses this to find/load the base model.
    base_model_prefix = "bert"

    def __init__(self, config):
        super(BertNER, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.bilstm = nn.LSTM(
            input_size=config.lstm_embedding_size,
            hidden_size=config.hidden_size // 2,
            batch_first=True,
            num_layers=2,
            dropout=config.lstm_dropout_prob,
            bidirectional=True,
        )
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(config.num_labels, batch_first=True)

        self.init_weights()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Load either a base encoder checkpoint or a fine-tuned BertNER checkpoint.

        - If `pretrained_model_name_or_path` points to a directory produced by
          `BertNER.save_pretrained(...)`, we load the FULL model weights (bert + head).
        - Otherwise, we treat it as a base encoder checkpoint (MLM/NSP heads etc.) and
          only load `BertModel` weights, leaving task head randomly initialized.

        This keeps training init behavior unchanged while making `test()`/inference load correct.
        """
        path = Path(pretrained_model_name_or_path)

        # If it's a fine-tuned directory, it should contain a HF model weight file.
        weight_file = None
        if path.exists() and path.is_dir():
            if (path / "model.safetensors").exists():
                weight_file = path / "model.safetensors"
            elif (path / "pytorch_model.bin").exists():
                weight_file = path / "pytorch_model.bin"

        # ---- case 1: load full fine-tuned BertNER (manual load; avoids HF finalize/tie logic) ----
        if weight_file is not None:
            num_labels = kwargs.pop("num_labels", None)
            config = kwargs.pop("config", None)
            if config is None:
                config = BertConfig.from_pretrained(str(path), **kwargs)
            if num_labels is not None:
                config.num_labels = num_labels

            model = cls(config)

            if weight_file.suffix == ".safetensors":
                try:
                    from safetensors.torch import load_file as safe_load_file
                except Exception as e:
                    raise RuntimeError(
                        "Found model.safetensors but safetensors is not available. "
                        "Install safetensors or re-save the checkpoint as pytorch_model.bin."
                    ) from e
                state_dict = safe_load_file(str(weight_file))
            else:
                state_dict = torch.load(str(weight_file), map_location="cpu")

            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if len(unexpected) > 0:
                # keep non-fatal: some checkpoints may contain optimizer keys etc.
                pass
            model.eval()
            return model

        # ---- case 2: load base encoder only ----
        num_labels = kwargs.pop("num_labels", None)

        config = kwargs.pop("config", None)
        if config is None:
            config = BertConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        if num_labels is not None:
            config.num_labels = num_labels

        model = cls(config)
        model.bert = BertModel.from_pretrained(pretrained_model_name_or_path, config=config)
        return model

    # Disable tied-weights logic in Transformers v5 (we don't tie any weights).
    def tie_weights(self, *args, **kwargs):
        return

    def forward(self, input_data, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, inputs_embeds=None, head_mask=None):
        input_ids, input_token_starts = input_data
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)
        sequence_output = outputs[0]

        # 去除[CLS]标签等位置，获得与label对齐的pre_label表示
        origin_sequence_output = [layer[starts.nonzero().squeeze(1)]
                                  for layer, starts in zip(sequence_output, input_token_starts)]
        # 将sequence_output的pred_label维度padding到最大长度
        padded_sequence_output = pad_sequence(origin_sequence_output, batch_first=True)
        # dropout pred_label的一部分feature
        padded_sequence_output = self.dropout(padded_sequence_output)
        lstm_output, _ = self.bilstm(padded_sequence_output)
        # 得到判别值
        logits = self.classifier(lstm_output)
        outputs = (logits,)
        if labels is not None:
            loss_mask = labels.gt(-1)
            loss = self.crf(logits, labels, loss_mask) * (-1)
            outputs = (loss,) + outputs

        # contain: (loss), scores
        return outputs
