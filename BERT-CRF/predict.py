import torch
import numpy as np
from transformers import BertTokenizer

from model import BertNER
import config


def load_model_and_tokenizer():
    """
    加载分词器和模型
    """
    tokenizer = BertTokenizer.from_pretrained(
        config.bert_model,
        do_lower_case=True
    )

    model = BertNER.from_pretrained(config.model_dir)
    model.to(config.device)
    model.eval()

    return model, tokenizer


def predict(text, model, tokenizer):
    """
    对单条中文文本进行实体识别
    """
    # ===== 1. 与 data_loader 保持一致的分词逻辑 =====
    words = []
    word_lens = []
    for char in text:
        tokens = tokenizer.tokenize(char)
        if not tokens:
            tokens = ['[UNK]'] # 确保每个字符至少对应一个 token
        words.append(tokens)
        word_lens.append(len(tokens))
    
    # 构造 BERT 输入格式
    bert_tokens = ['[CLS]'] + [item for sublist in words for item in sublist] + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(bert_tokens)
    
    # 找到每个字符对应的第一个 token 的位置（跳过 index 0 的 [CLS]）
    token_start_idxs = 1 + np.cumsum([0] + word_lens[:-1])
    
    # 构造 mask
    input_token_starts = np.zeros(len(input_ids))
    input_token_starts[token_start_idxs] = 1
    
    # 转为 tensor
    input_ids_tensor = torch.tensor([input_ids], dtype=torch.long).to(config.device)
    input_token_starts_tensor = torch.tensor([input_token_starts], dtype=torch.long).to(config.device)
    attention_mask = torch.ones_like(input_ids_tensor).to(config.device)

    # ===== 2. 前向推理 =====
    model.eval()
    with torch.no_grad():
        outputs = model(
            (input_ids_tensor, input_token_starts_tensor),
            attention_mask=attention_mask
        )
        logits = outputs[0]
        
        # CRF 解码
        # 注意：model 内部 forward 已经根据 input_token_starts 过滤了 tokens
        # 所以 logits 的长度已经等于 len(text)
        decode_mask = torch.ones(1, logits.size(1), dtype=torch.uint8).to(config.device)
        pred_ids = model.crf.decode(logits, mask=decode_mask)[0]

    # ===== 3. id → label =====
    # 建立一个干净的映射，优先映射回 B- 或 I-
    id2label = {}
    for k, v in config.label2id.items():
        if v not in id2label or k.startswith('B-'):
            id2label[v] = k

    pred_labels = [id2label.get(i, 'O') for i in pred_ids]

    # ===== 4. 实体抽取 (支持 B-, I-, S-) =====
    entities = []
    entity = None

    for idx, (char, label) in enumerate(zip(text, pred_labels)):
        # B- 或者 S- 都代表实体的开始
        if label.startswith('B-') or label.startswith('S-'):
            if entity:
                entities.append(entity)
            entity = {
                'type': label[2:],
                'start': idx,
                'end': idx,
                'text': char
            }
            # 如果是 S-，抽取完直接结束当前实体
            if label.startswith('S-'):
                entities.append(entity)
                entity = None

        elif label.startswith('I-') and entity:
            if label[2:] == entity['type']:
                entity['text'] += char
                entity['end'] = idx
            else:
                # 标签类型不一致，先结束旧实体
                entities.append(entity)
                # 如果这个 I- 标签被视为新实体的开始（虽然不符合 BIO 规范，但为了鲁棒性）
                entity = {
                    'type': label[2:],
                    'start': idx,
                    'end': idx,
                    'text': char
                }
        else:
            if entity:
                entities.append(entity)
                entity = None

    if entity:
        entities.append(entity)

    return entities


if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer()
    test_texts = [
        "斗岭水库最高水位165.82m",
        "葛岙水库位于奉化区，正常蓄水位62.00m",
        "采取错峰调度，开启取水闸",
    ]
    for text in test_texts:
        res = predict(text, model, tokenizer)
        print(f"Text: {text}")
        print(f"Entities: {res}")
        print("-" * 30)
