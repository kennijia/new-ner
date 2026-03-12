import argparse
import os
import torch
import numpy as np
from typing import List, Dict, Any

import config as cfg
from transformers import BertTokenizer
from model import BertNER


def load_model(model_dir: str = None):
    model_dir = model_dir or cfg.model_dir
    model = BertNER.from_pretrained(model_dir)
    model.to(cfg.device)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(cfg.bert_model, do_lower_case=True)
    return model, tokenizer


def decode_entities(labels: List[str], text: str) -> List[Dict[str, Any]]:
    entities = []
    i = 0
    while i < len(labels):
        tag = labels[i]
        if tag is None or tag == 'O':
            i += 1
            continue
        if tag.startswith('S-'):
            etype = tag[2:]
            entities.append({'type': etype, 'span': [i, i], 'text': text[i:i+1]})
            i += 1
            continue
        if tag.startswith('B-'):
            etype = tag[2:]
            start = i
            i += 1
            while i < len(labels) and labels[i] == f'I-{etype}':
                i += 1
            end = i - 1
            entities.append({'type': etype, 'span': [start, end], 'text': text[start:end+1]})
            continue
        # fallthrough for unexpected tag
        i += 1
    return entities


def predict_texts(texts: List[str], model_dir: str = None) -> List[Dict[str, Any]]:
    model, tokenizer = load_model(model_dir)
    results = []
    id2label = cfg.id2label

    for text in texts:
        # char-level tokens; expand to wordpieces per char
        chars = list(text)
        pieces_per_char = []
        for ch in chars:
            pieces = tokenizer.tokenize(ch)
            if len(pieces) == 0:
                pieces = [tokenizer.unk_token]
            pieces_per_char.append(pieces)
        # flatten and add [CLS]
        words = ['[CLS]'] + [p for pieces in pieces_per_char for p in pieces]
        input_ids = tokenizer.convert_tokens_to_ids(words)
        # build start mask aligned to original chars
        word_lens = [len(p) for p in pieces_per_char]
        seq_len = len(input_ids)
        starts_mask = np.zeros(seq_len)
        # positions after [CLS]
        pos = 1
        for l in word_lens:
            starts_mask[pos] = 1
            pos += l

        input_ids_t = torch.tensor([input_ids], dtype=torch.long).to(cfg.device)
        starts_mask_t = torch.tensor([starts_mask], dtype=torch.long).to(cfg.device)
        attn_mask = input_ids_t.gt(0)

        with torch.no_grad():
            logits = model((input_ids_t, starts_mask_t), token_type_ids=None, attention_mask=attn_mask)[0]
            # CRF decode requires mask of label length (positions selected by starts)
            label_len = logits.size(1)
            label_mask = torch.ones((1, label_len), dtype=torch.bool).to(cfg.device)
            pred_ids = model.crf.decode(logits, mask=label_mask)[0]

        pred_labels = [id2label.get(i) for i in pred_ids]
        entities = decode_entities(pred_labels, text)
        results.append({
            'text': text,
            'labels': pred_labels,
            'entities': entities
        })

    return results


def main():
    parser = argparse.ArgumentParser(description='BERT-LSTM-CRF NER Prediction')
    parser.add_argument('--model_dir', type=str, default=None, help='Path to saved model dir')
    parser.add_argument('--text', type=str, default=None, help='Single input text')
    parser.add_argument('--file', type=str, default=None, help='Path to a file with one text per line')
    args = parser.parse_args()

    inputs = []
    if args.text:
        inputs.append(args.text)
    if args.file:
        with open(args.file, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    inputs.append(line)
    if not inputs:
        print('No input provided. Use --text or --file.')
        return

    results = predict_texts(inputs, args.model_dir)
    for r in results:
        print('TEXT:', r['text'])
        print('LABELS:', ' '.join(map(str, r['labels'])))
        print('ENTITIES:', r['entities'])
        print('-'*50)


if __name__ == '__main__':
    main()
