from __future__ import annotations

import os
import torch
from torch import nn
from transformers import BertConfig, BertModel

class BertForRelationExtraction(nn.Module):
    def __init__(self, config, head_token_id: int, tail_token_id: int):
        super().__init__()
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # We concatenate [HEAD] and [TAIL]
        self.classifier = nn.Linear(config.hidden_size * 2, config.num_labels)
        self.head_token_id = head_token_id
        self.tail_token_id = tail_token_id
        
        # Loss
        self.loss_fct = nn.CrossEntropyLoss()

    def resize_token_embeddings(self, new_num_tokens: int):
        # Delegate to bert
        self.bert.resize_token_embeddings(new_num_tokens)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        
        sequence_output = outputs[0]  # (batch_size, seq_len, hidden_size)
        
        batch_size = input_ids.size(0)
        
        # Find positions of [HEAD] and [TAIL]
        head_mask = (input_ids == self.head_token_id)
        tail_mask = (input_ids == self.tail_token_id)
        
        # In case a marker is truncated, fallback to [CLS] at position 0
        cls_output = sequence_output[:, 0, :]
        
        head_reps = []
        tail_reps = []
        
        for i in range(batch_size):
            h_indices = head_mask[i].nonzero(as_tuple=True)[0]
            if len(h_indices) > 0:
                head_reps.append(sequence_output[i, h_indices[0], :])
            else:
                head_reps.append(cls_output[i])
                
            t_indices = tail_mask[i].nonzero(as_tuple=True)[0]
            if len(t_indices) > 0:
                tail_reps.append(sequence_output[i, t_indices[0], :])
            else:
                tail_reps.append(cls_output[i])
                
        head_rep = torch.stack(head_reps, dim=0)
        tail_rep = torch.stack(tail_reps, dim=0)
        
        # Concatenate head and tail token representations
        combined_rep = torch.cat([head_rep, tail_rep], dim=-1)
        combined_rep = self.dropout(combined_rep)
        
        logits = self.classifier(combined_rep)
        
        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
        from transformers.modeling_outputs import SequenceClassifierOutput
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

def build_model(model_name_or_path: str, num_labels: int, head_token_id: int, tail_token_id: int):
    """Build a BERT-based relation classifier.

    Extracts features from the [HEAD] and [TAIL] markers instead of just using [CLS].
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
        
        model = BertForRelationExtraction(cfg, head_token_id, tail_token_id)

        sd = torch.load(state_path, map_location="cpu", weights_only=False)
        # allow missing keys from classification head
        model.bert.load_state_dict(sd, strict=False)
        return model

    # Fallback for standard HF checkpoints
    from transformers import AutoConfig, AutoModel
    cfg = AutoConfig.from_pretrained(model_name_or_path)
    cfg.num_labels = num_labels
    
    model = BertForRelationExtraction(cfg, head_token_id, tail_token_id)
    # optionally load weights here if not a from-scratch run
    return model
