from __future__ import annotations

import os
from collections import Counter
from typing import Dict, List

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertTokenizer, get_linear_schedule_with_warmup

import config
from data_loader import PairDataset, build_pair_samples, collate_fn, downsample_negatives, split_by_sentence_id, split_train_dev
from metrics import micro_prf
from model import build_model
from utils import set_logger, set_seed


def _build_label_space(samples) -> Dict[str, int]:
    labels = sorted({s.label for s in samples})
    # ensure NoRelation at 0 for convenience
    if "NoRelation" in labels:
        labels.remove("NoRelation")
        labels = ["NoRelation", *labels]
    return {l: i for i, l in enumerate(labels)}


@torch.no_grad()
def evaluate(model, dl, device, positive_ids: List[int]):
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    total_loss = 0.0
    n = 0

    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss
        logits = out.logits
        pred = logits.argmax(dim=-1)

        total_loss += float(loss.item()) * batch["input_ids"].shape[0]
        n += batch["input_ids"].shape[0]
        y_true.extend(batch["labels"].detach().cpu().tolist())
        y_pred.extend(pred.detach().cpu().tolist())

    m = micro_prf(y_true, y_pred, positive_ids=positive_ids)
    return total_loss / max(1, n), m


def main() -> int:
    os.makedirs(config.exp_dir, exist_ok=True)
    logger = set_logger(config.log_path)

    set_seed(config.seed)

    logger.info("=== BERT-RE pair classification ===")
    logger.info("train_file=%s", config.train_file)
    logger.info("bert_model=%s", config.bert_model)
    logger.info("exp_dir=%s", config.exp_dir)

    samples = build_pair_samples(config.train_file, negative_label="NoRelation")
    logger.info("Built pair samples: %d", len(samples))
    logger.info("Label distribution (raw): %s", dict(Counter([s.label for s in samples])))

    # Split strategy
    if getattr(config, "split_level", "pair") == "sentence":
        train_samples, dev_samples = split_by_sentence_id(samples, config.dev_split_size, seed=config.seed)
    else:
        train_samples, dev_samples = split_train_dev(samples, config.dev_split_size, seed=config.seed)

    # Negative downsampling (train only)
    neg_pos_ratio = getattr(config, "neg_pos_ratio", None)
    train_samples = downsample_negatives(
        train_samples,
        negative_label="NoRelation",
        neg_pos_ratio=neg_pos_ratio,
        seed=config.seed,
    )

    logger.info("Split: train=%d dev=%d | split_level=%s", len(train_samples), len(dev_samples), getattr(config, "split_level", "pair"))
    logger.info("Label distribution (train after neg-sampling): %s", dict(Counter([s.label for s in train_samples])))
    logger.info("Label distribution (dev): %s", dict(Counter([s.label for s in dev_samples])))

    label2id = _build_label_space(samples)
    id2label = {v: k for k, v in label2id.items()}
    positive_ids = [i for l, i in label2id.items() if l != "NoRelation"]
    logger.info("Labels: %s", label2id)

    # Robust tokenizer loading for local Chinese RoBERTa checkpoints.
    # Some checkpoints in this repo have a non-standard `config.json` that can
    # confuse AutoTokenizer; they still ship a standard `vocab.txt`.
    vocab_path = os.path.join(config.bert_model, "vocab.txt")
    if os.path.exists(vocab_path):
        tokenizer = BertTokenizer.from_pretrained(vocab_path, do_lower_case=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.bert_model, use_fast=False)

    # add marker tokens so they are single tokens
    special = {"additional_special_tokens": ["[HEAD]", "[/HEAD]", "[TAIL]", "[/TAIL]"]}
    tokenizer.add_special_tokens(special)

    train_ds = PairDataset(train_samples, tokenizer, label2id, max_length=config.max_length)
    dev_ds = PairDataset(dev_samples, tokenizer, label2id, max_length=config.max_length)

    train_dl = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_token_id=tokenizer.pad_token_id or 0),
    )
    dev_dl = DataLoader(
        dev_ds,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_token_id=tokenizer.pad_token_id or 0),
    )

    model = build_model(config.bert_model, num_labels=len(label2id))
    model.resize_token_embeddings(len(tokenizer))
    model.to(config.device)

    optim = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    total_steps = len(train_dl) * config.epoch_num
    warmup_steps = int(total_steps * config.warmup_ratio)
    sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    best_f1 = -1.0
    best_path = os.path.join(config.exp_dir, "best.pt")

    for epoch in range(1, config.epoch_num + 1):
        model.train()
        total_loss = 0.0
        n = 0
        for batch in train_dl:
            batch = {k: v.to(config.device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)

            total_loss += float(loss.item()) * batch["input_ids"].shape[0]
            n += batch["input_ids"].shape[0]

        train_loss = total_loss / max(1, n)
        dev_loss, dev_m = evaluate(model, dev_dl, config.device, positive_ids=positive_ids)

        logger.info(
            "Epoch %d/%d | train_loss=%.6f | dev_loss=%.6f | dev_micro_p=%.4f dev_micro_r=%.4f dev_micro_f1=%.4f (pos_support=%d)",
            epoch,
            config.epoch_num,
            train_loss,
            dev_loss,
            dev_m.precision,
            dev_m.recall,
            dev_m.f1,
            dev_m.support,
        )

        if dev_m.f1 > best_f1:
            best_f1 = dev_m.f1
            ckpt = {
                "model": model.state_dict(),
                "label2id": label2id,
                "id2label": id2label,
                "tokenizer": config.bert_model,
            }
            torch.save(ckpt, best_path)
            logger.info("Saved best checkpoint: %s (best_f1=%.4f)", best_path, best_f1)

    logger.info("Training done. Best dev micro-F1=%.4f", best_f1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

