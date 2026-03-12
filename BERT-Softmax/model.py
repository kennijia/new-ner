import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, BertConfig
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path


class BertNER(BertPreTrainedModel):
    base_model_prefix = "bert"

    def __init__(self, config):
        super(BertNER, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        path = Path(pretrained_model_name_or_path)

        weight_file = None
        if path.exists() and path.is_dir():
            if (path / "model.safetensors").exists():
                weight_file = path / "model.safetensors"
            elif (path / "pytorch_model.bin").exists():
                weight_file = path / "pytorch_model.bin"

        # full fine-tuned checkpoint
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

            model.load_state_dict(state_dict, strict=False)
            model.eval()
            return model

        # base encoder init
        num_labels = kwargs.pop("num_labels", None)
        config = kwargs.pop("config", None)
        if config is None:
            config = BertConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        if num_labels is not None:
            config.num_labels = num_labels

        model = cls(config)
        model.bert = BertModel.from_pretrained(pretrained_model_name_or_path, config=config)
        return model

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
        # 得到判别值
        logits = self.classifier(padded_sequence_output)

        outputs = (logits,)
        if labels is not None:
            loss_mask = labels.gt(-1)
            loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            if loss_mask is not None:
                # 只留下label存在的位置计算loss
                active_loss = loss_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        # contain: (loss), scores
        return outputs
