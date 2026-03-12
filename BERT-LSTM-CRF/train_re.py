#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train RE text classification model on JSONL files.

Input JSONL format (one record per line):
{"text": "... [HEAD]...[/HEAD] ... [TAIL]...[/TAIL] ...", "label": "CAUSES"}

Example:
python train_re.py \
  --train_file data/my/re_train.jsonl \
  --dev_file data/my/re_dev.jsonl \
  --test_file data/my/re_test.jsonl \
  --model_name_or_path pretrained_bert_models/bert-base-chinese \
  --output_dir experiments/re_cls
"""

import argparse
import json
import os
import random
import importlib.util
import copy
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW


@dataclass
class Sample:
    text: str
    label: str


class REDataset(Dataset):
    def __init__(self, samples: List[Sample], tokenizer, label2id: Dict[str, int], max_length: int = 256):
        self.samples = samples
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        enc = self.tokenizer.encode_plus(
            s.text,
            add_special_tokens=True,
            max_length=self.max_length,
            pad_to_max_length=False,
            return_token_type_ids=True,
            return_attention_mask=True,
        )
        item = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": self.label2id[s.label],
        }
        if "token_type_ids" in enc:
            item["token_type_ids"] = enc["token_type_ids"]
        return item


def collate_fn(batch, pad_token_id=0):
    max_len = max(len(x["input_ids"]) for x in batch)

    input_ids, attention_mask, labels = [], [], []
    token_type_ids = []
    has_token_type = "token_type_ids" in batch[0]

    for x in batch:
        cur_len = len(x["input_ids"])
        pad_len = max_len - cur_len

        input_ids.append(x["input_ids"] + [pad_token_id] * pad_len)
        attention_mask.append(x["attention_mask"] + [0] * pad_len)
        labels.append(x["labels"])

        if has_token_type:
            token_type_ids.append(x["token_type_ids"] + [0] * pad_len)

    out = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
    if has_token_type:
        out["token_type_ids"] = torch.tensor(token_type_ids, dtype=torch.long)
    return out


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_label(label):
    if isinstance(label, list):
        if len(label) == 0:
            return "NA"
        return str(label[0])
    if label is None:
        return "NA"
    return str(label)


def load_jsonl(path: str) -> List[Sample]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text", "")
            if not text:
                continue
            label = normalize_label(obj.get("label"))
            samples.append(Sample(text=text, label=label))
    return samples


def compute_binary_metrics(y_true: List[int], y_pred: List[int], positive_id: int):
    tp = fp = fn = tn = 0
    for t, p in zip(y_true, y_pred):
        if t == positive_id and p == positive_id:
            tp += 1
        elif t != positive_id and p == positive_id:
            fp += 1
        elif t == positive_id and p != positive_id:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-12)

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def evaluate(model, data_loader, device, positive_id):
    model.eval()
    all_true, all_pred = [], []
    total_loss = 0.0
    n_steps = 0

    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            if isinstance(outputs, tuple):
                loss = outputs[0]
                logits = outputs[1]
            else:
                loss = outputs.loss
                logits = outputs.logits
            pred = torch.argmax(logits, dim=-1)

            total_loss += loss.item()
            n_steps += 1
            all_true.extend(batch["labels"].tolist())
            all_pred.extend(pred.tolist())

    m = compute_binary_metrics(all_true, all_pred, positive_id=positive_id)
    m["loss"] = total_loss / max(1, n_steps)
    return m


def load_re_config(config_path: str) -> Dict:
    defaults = {
        "train_file": "data/my/re_train.jsonl",
        "dev_file": "data/my/re_dev.jsonl",
        "test_file": "data/my/re_test.jsonl",
        "model_name_or_path": "pretrained_bert_models/bert-base-chinese",
        "output_dir": "experiments/re_cls",
        "max_length": 256,
        "batch_size": 16,
        "lr": 2e-5,
        "weight_decay": 0.01,
        "epochs": 8,
        "warmup_ratio": 0.1,
        "patience": 3,
        "seed": 42,
        "use_class_weight": True,
    }

    if not config_path or not os.path.exists(config_path):
        return defaults

    spec = importlib.util.spec_from_file_location("re_config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for k in list(defaults.keys()):
        if hasattr(module, k):
            defaults[k] = getattr(module, k)
    return defaults


def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default="re_config.py")
    pre_args, _ = pre_parser.parse_known_args()
    cfg = load_re_config(pre_args.config)

    parser = argparse.ArgumentParser(description="Train RE classifier")
    parser.add_argument("--config", default=pre_args.config, help="Path to re_config.py")
    parser.add_argument("--train_file", default=cfg["train_file"])
    parser.add_argument("--dev_file", default=cfg["dev_file"])
    parser.add_argument("--test_file", default=cfg["test_file"])
    parser.add_argument("--model_name_or_path", default=cfg["model_name_or_path"])
    parser.add_argument("--output_dir", default=cfg["output_dir"])
    parser.add_argument("--max_length", type=int, default=cfg["max_length"])
    parser.add_argument("--batch_size", type=int, default=cfg["batch_size"])
    parser.add_argument("--lr", type=float, default=cfg["lr"])
    parser.add_argument("--weight_decay", type=float, default=cfg["weight_decay"])
    parser.add_argument("--epochs", type=int, default=cfg["epochs"])
    parser.add_argument("--warmup_ratio", type=float, default=cfg["warmup_ratio"])
    parser.add_argument("--patience", type=int, default=cfg["patience"], help="Early stop patience on dev F1")
    parser.add_argument("--seed", type=int, default=cfg["seed"])
    parser.add_argument("--no_class_weight", action="store_true", help="Disable class-weighted CE loss")
    args = parser.parse_args()

    if cfg.get("use_class_weight", True) is False:
        args.no_class_weight = True

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    train_samples = load_jsonl(args.train_file)
    dev_samples = load_jsonl(args.dev_file)
    test_samples = load_jsonl(args.test_file)

    labels = sorted(set([s.label for s in train_samples + dev_samples + test_samples]))
    label2id = {lb: i for i, lb in enumerate(labels)}
    id2label = {i: lb for lb, i in label2id.items()}

    if "CAUSES" in label2id:
        positive_id = label2id["CAUSES"]
    else:
        # fallback to first non-NA label or last label
        non_na = [lb for lb in labels if lb != "NA"]
        positive_id = label2id[non_na[0]] if non_na else (len(labels) - 1)

    with open(os.path.join(args.output_dir, "label2id.json"), "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=len(labels),
    )
    model.config.id2label = id2label
    model.config.label2id = label2id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_ds = REDataset(train_samples, tokenizer, label2id, max_length=args.max_length)
    dev_ds = REDataset(dev_samples, tokenizer, label2id, max_length=args.max_length)
    test_ds = REDataset(test_samples, tokenizer, label2id, max_length=args.max_length)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_token_id=tokenizer.pad_token_id or 0),
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_token_id=tokenizer.pad_token_id or 0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_token_id=tokenizer.pad_token_id or 0),
    )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * max(1, len(train_loader))
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    class_weights = None
    if not args.no_class_weight:
        label_counter = Counter([label2id[s.label] for s in train_samples])
        n_total = sum(label_counter.values())
        n_cls = len(labels)
        weights = []
        for i in range(n_cls):
            cnt = label_counter.get(i, 1)
            weights.append(n_total / (n_cls * cnt))
        class_weights = torch.tensor(weights, dtype=torch.float, device=device)

    best_dev_f1 = -1.0
    best_epoch = -1
    bad_epochs = 0
    best_state_dict = None

    print("Config file:", args.config)
    print("Label distribution (train):", Counter([s.label for s in train_samples]))
    print("Labels:", label2id)
    print("Device:", device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels_tensor = batch.pop("labels")

            outputs = model(**batch)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs.logits

            if class_weights is not None:
                loss = torch.nn.functional.cross_entropy(logits, labels_tensor, weight=class_weights)
            else:
                loss = torch.nn.functional.cross_entropy(logits, labels_tensor)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        train_loss = running_loss / max(1, len(train_loader))
        dev_metrics = evaluate(model, dev_loader, device, positive_id=positive_id)

        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | "
            f"dev_loss={dev_metrics['loss']:.4f} | dev_f1={dev_metrics['f1']:.4f} | "
            f"dev_p={dev_metrics['precision']:.4f} | dev_r={dev_metrics['recall']:.4f}"
        )

        if dev_metrics["f1"] > best_dev_f1:
            best_dev_f1 = dev_metrics["f1"]
            best_epoch = epoch
            bad_epochs = 0
            best_state_dict = copy.deepcopy(model.state_dict())
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}, best epoch={best_epoch}, best_dev_f1={best_dev_f1:.4f}")
                break

    # load best state and test
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    model.to(device)
    test_metrics = evaluate(model, test_loader, device, positive_id=positive_id)

    print("=" * 60)
    print(f"Best epoch: {best_epoch}, best dev F1: {best_dev_f1:.4f}")
    print(
        f"Test | loss={test_metrics['loss']:.4f} | acc={test_metrics['acc']:.4f} | "
        f"p={test_metrics['precision']:.4f} | r={test_metrics['recall']:.4f} | f1={test_metrics['f1']:.4f}"
    )

    with open(os.path.join(args.output_dir, "metrics_test.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
