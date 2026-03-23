from __future__ import annotations

import os
from collections import Counter
from typing import Dict, List, Tuple

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertTokenizer, get_linear_schedule_with_warmup

import config
from data_loader import PairDataset, build_pair_samples, collate_fn, downsample_negatives, split_by_sentence_id, split_train_dev, split_by_sentence_id_train_dev_test, split_train_dev_test
from metrics import micro_prf
from model import build_model
from utils import set_logger, set_seed


def _env_override_str(key: str, default: str) -> str:
    v = os.environ.get(key)
    return default if v is None or v == "" else v


def _env_override_int(key: str, default: int) -> int:
    v = os.environ.get(key)
    if v is None or v == "":
        return default
    return int(v)


def _env_override_float(key: str, default: float) -> float:
    v = os.environ.get(key)
    if v is None or v == "":
        return default
    return float(v)


def _env_override_bool(key: str, default: bool) -> bool:
    v = os.environ.get(key)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y"}


def _resolve_run_config():
    """Resolve run config from `config.py` + env overrides.

    This keeps the code CLI-free while enabling grid search scripts.
    """
    exp_dir = _env_override_str("BERT_RE_EXP_DIR", getattr(config, "exp_dir"))
    seed = _env_override_int("BERT_RE_SEED", getattr(config, "seed"))
    split_level = _env_override_str("BERT_RE_SPLIT_LEVEL", getattr(config, "split_level", "pair"))

    # neg_pos_ratio can be disabled by setting it to -1
    neg_pos_ratio = _env_override_float("BERT_RE_NEG_POS_RATIO", getattr(config, "neg_pos_ratio", -1.0))

    use_fgm = _env_override_bool("BERT_RE_USE_FGM", getattr(config, "use_fgm", False))
    fgm_epsilon = _env_override_float("BERT_RE_FGM_EPSILON", getattr(config, "fgm_epsilon", 1.0))

    return {
        "exp_dir": exp_dir,
        "log_path": os.path.join(exp_dir, "train.log"),
        "seed": seed,
        "split_level": split_level,
        "neg_pos_ratio": neg_pos_ratio,
        "use_fgm": use_fgm,
        "fgm_epsilon": fgm_epsilon,
    }


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

    # Fine-grained per-class metrics (typically for positive labels only)
    def _per_label_prf(y_t: List[int], y_p: List[int], label_ids: List[int]) -> Dict[int, Dict[str, float]]:
        out: Dict[int, Dict[str, float]] = {}
        for lid in label_ids:
            tp = fp = fn = 0
            for t, p in zip(y_t, y_p):
                if p == lid and t == lid:
                    tp += 1
                elif p == lid and t != lid:
                    fp += 1
                elif p != lid and t == lid:
                    fn += 1
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
            out[lid] = {"precision": prec, "recall": rec, "f1": f1, "support": (tp + fn)}
        return out

    per_cls = _per_label_prf(y_true, y_pred, positive_ids)
    return total_loss / max(1, n), m, per_cls


def _atomic_torch_save(obj, path: str, *, retries: int = 1) -> None:
    """Atomically save a checkpoint to avoid corrupted partial writes."""
    tmp_path = path + ".tmp"
    err: Exception | None = None
    for _ in range(max(1, retries + 1)):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            # Use legacy (non-zip) format to avoid PytorchStreamWriter zip failures.
            torch.save(obj, tmp_path, _use_new_zipfile_serialization=False)
            os.replace(tmp_path, path)
            return
        except Exception as e:
            err = e
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
    assert err is not None
    raise err


def main() -> int:
    run_cfg = _resolve_run_config()
    os.makedirs(run_cfg["exp_dir"], exist_ok=True)
    logger = set_logger(run_cfg["log_path"])

    set_seed(run_cfg["seed"])

    # Persist effective config for paper-friendly reproducibility
    try:
        import json

        dump = {
            "train_file": config.train_file,
            "bert_model": config.bert_model,
            "max_length": config.max_length,
            "batch_size": config.batch_size,
            "epoch_num": config.epoch_num,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "warmup_ratio": config.warmup_ratio,
            "dev_split_size": config.dev_split_size,
            "test_split_size": getattr(config, "test_split_size", 0.0),
            **run_cfg,
        }
        with open(os.path.join(run_cfg["exp_dir"], "config.json"), "w", encoding="utf-8") as f:
            json.dump(dump, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    logger.info("=== BERT-RE pair classification ===")
    logger.info("train_file=%s", config.train_file)
    logger.info("bert_model=%s", config.bert_model)
    logger.info("exp_dir=%s", run_cfg["exp_dir"])
    logger.info("seed=%s split_level=%s neg_pos_ratio=%s", run_cfg["seed"], run_cfg["split_level"], run_cfg["neg_pos_ratio"])

    samples = build_pair_samples(config.train_file, negative_label="NoRelation")
    logger.info("Built pair samples: %d", len(samples))
    logger.info("Label distribution (raw): %s", dict(Counter([s.label for s in samples])))

    # Split strategy: train/dev/test
    dev_ratio = float(getattr(config, "dev_split_size", 0.1))
    test_ratio = float(getattr(config, "test_split_size", 0.1))

    if (dev_ratio + test_ratio) >= 1.0:
        raise ValueError(f"Invalid split ratios: dev_split_size={dev_ratio}, test_split_size={test_ratio} (sum must be < 1.0)")

    if run_cfg["split_level"] == "sentence":
        train_samples, dev_samples, test_samples = split_by_sentence_id_train_dev_test(
            samples, dev_ratio=dev_ratio, test_ratio=test_ratio, seed=run_cfg["seed"]
        )
    else:
        train_samples, dev_samples, test_samples = split_train_dev_test(samples, dev_ratio=dev_ratio, test_ratio=test_ratio, seed=run_cfg["seed"])

    # Negative downsampling (train only)
    neg_pos_ratio = run_cfg["neg_pos_ratio"]

    # If neg_pos_ratio <= 0, disable negative downsampling.
    # Otherwise keep at most `neg_pos_ratio * #positive` negatives to mitigate class imbalance.
    before_counts = Counter([s.label for s in train_samples])
    train_samples = downsample_negatives(
        train_samples,
        negative_label="NoRelation",
        neg_pos_ratio=neg_pos_ratio,
        seed=run_cfg["seed"],
    )
    after_counts = Counter([s.label for s in train_samples])
    try:
        logger.info(
            "Neg downsampling applied: neg_pos_ratio=%s | train size %d -> %d | before=%s | after=%s",
            neg_pos_ratio,
            sum(before_counts.values()),
            len(train_samples),
            dict(before_counts),
            dict(after_counts),
        )
    except Exception:
        pass

    logger.info(
        "Split: train=%d dev=%d test=%d | split_level=%s",
        len(train_samples),
        len(dev_samples),
        len(test_samples),
        run_cfg["split_level"],
    )
    logger.info("Label distribution (train after neg-sampling): %s", dict(Counter([s.label for s in train_samples])))
    logger.info("Label distribution (dev): %s", dict(Counter([s.label for s in dev_samples])))
    logger.info("Label distribution (test): %s", dict(Counter([s.label for s in test_samples])))

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
    test_ds = PairDataset(test_samples, tokenizer, label2id, max_length=config.max_length)

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
    test_dl = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_token_id=tokenizer.pad_token_id or 0),
    )

    head_token = "[HEAD]"
    tail_token = "[TAIL]"
    head_token_id = tokenizer.convert_tokens_to_ids(head_token)
    tail_token_id = tokenizer.convert_tokens_to_ids(tail_token)

    # Log resolved ids for debugging/reproducibility.
    try:
        logger.info("Marker token ids: %s=%s, %s=%s", head_token, head_token_id, tail_token, tail_token_id)
    except Exception:
        pass

    # Fail fast if markers are not actually present as single tokens.
    # Different tokenizers may return `unk_token_id`, -1, or None when token is missing.
    unk_id = getattr(tokenizer, "unk_token_id", None)
    missing = (
        head_token_id is None
        or tail_token_id is None
        or head_token_id == -1
        or tail_token_id == -1
        or (unk_id is not None and (head_token_id == unk_id or tail_token_id == unk_id))
    )
    if missing:
        raise ValueError(
            "Marker token id lookup failed. "
            f"{head_token} -> {head_token_id}, {tail_token} -> {tail_token_id}, unk_token_id={unk_id}. "
            "Check that `PairDataset` inserts these tokens (exact string match) and that they were added via "
            "`tokenizer.add_special_tokens({'additional_special_tokens': [...]})`."
        )

    model = build_model(
        config.bert_model,
        num_labels=len(label2id),
        head_token_id=head_token_id,
        tail_token_id=tail_token_id,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.to(config.device)

    optim = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    total_steps = len(train_dl) * config.epoch_num
    warmup_steps = int(total_steps * config.warmup_ratio)
    sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    best_f1 = -1.0
    best_path = os.path.join(run_cfg["exp_dir"], "best.pt")
    best_test_m = None

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
        dev_loss, dev_m, dev_per_cls = evaluate(model, dev_dl, config.device, positive_ids=positive_ids)
        test_loss, test_m, test_per_cls = evaluate(model, test_dl, config.device, positive_ids=positive_ids)

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

        # Per-class dev metrics (positive labels)
        try:
            parts = []
            for lid in sorted(dev_per_cls.keys()):
                name = id2label.get(lid, str(lid))
                mm = dev_per_cls[lid]
                parts.append(
                    f"{name}:P={mm['precision']:.3f} R={mm['recall']:.3f} F1={mm['f1']:.3f} (sup={int(mm['support'])})"
                )
            logger.info("Dev per-class (pos only): %s", " | ".join(parts))
        except Exception:
            pass

        # Also log test for paper reporting (do NOT select best by test)
        try:
            logger.info(
                "Test: loss=%.6f | micro_p=%.4f micro_r=%.4f micro_f1=%.4f (pos_support=%d)",
                test_loss,
                test_m.precision,
                test_m.recall,
                test_m.f1,
                test_m.support,
            )
        except Exception:
            pass

        if dev_m.f1 > best_f1:
            best_f1 = dev_m.f1
            best_test_m = test_m
            ckpt = {
                # Save only the model weights (includes classifier) to minimize disk usage.
                "model_state_dict": model.state_dict(),
                "label2id": label2id,
                "id2label": id2label,
                "bert_model": config.bert_model,
                "max_length": config.max_length,
            }
            try:
                _atomic_torch_save(ckpt, best_path, retries=1)
                logger.info("Saved best checkpoint: %s (best_f1=%.4f)", best_path, best_f1)
            except Exception as e:
                # Fallback: save only the classifier head
                try:
                    head_only = {
                        "classifier": model.classifier.state_dict() if hasattr(model, "classifier") else None,
                        "label2id": label2id,
                        "id2label": id2label,
                        "note": f"full checkpoint save failed: {type(e).__name__}: {e}",
                    }
                    _atomic_torch_save(head_only, best_path + ".head_only.pt", retries=1)
                    logger.exception("Failed to save full checkpoint to %s; saved head-only fallback.", best_path)
                except Exception:
                    logger.exception("Failed to save checkpoint artifacts.")

    if best_test_m is not None:
        try:
            logger.info(
                "Training done. Best dev micro-F1=%.4f | corresponding test micro-F1=%.4f (P=%.4f R=%.4f, pos_support=%d)",
                best_f1,
                best_test_m.f1,
                best_test_m.precision,
                best_test_m.recall,
                best_test_m.support,
            )
        except Exception:
            pass
    else:
        logger.info("Training done. Best dev micro-F1=%.4f", best_f1)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

