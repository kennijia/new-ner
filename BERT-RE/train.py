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

    # Optional: cost-sensitive learning via class weights
    use_class_weights = _env_override_bool("BERT_RE_USE_CLASS_WEIGHTS", getattr(config, "use_class_weights", False))
    class_weight_mode = _env_override_str("BERT_RE_CLASS_WEIGHT_MODE", getattr(config, "class_weight_mode", "inv_sqrt"))  # inv_sqrt | inv
    class_weight_max = _env_override_float("BERT_RE_CLASS_WEIGHT_MAX", getattr(config, "class_weight_max", 10.0))

    # Optional: dynamic confidence + forbidden-relation filtering (post-processing)
    use_dyn_filter = _env_override_bool("BERT_RE_USE_DYN_FILTER", getattr(config, "use_dyn_filter", False))
    dyn_filter_target_precision = _env_override_float(
        "BERT_RE_DYN_FILTER_TARGET_PREC", getattr(config, "dyn_filter_target_precision", 0.85)
    )
    dyn_filter_min_threshold = _env_override_float(
        "BERT_RE_DYN_FILTER_MIN_THR", getattr(config, "dyn_filter_min_threshold", 0.0)
    )
    dyn_filter_max_threshold = _env_override_float(
        "BERT_RE_DYN_FILTER_MAX_THR", getattr(config, "dyn_filter_max_threshold", 0.99)
    )
    dyn_filter_num_steps = _env_override_int(
        "BERT_RE_DYN_FILTER_STEPS", getattr(config, "dyn_filter_num_steps", 50)
    )
    forbidden_relations_csv = _env_override_str(
        "BERT_RE_FORBIDDEN_RELATIONS", getattr(config, "forbidden_relations_csv", "")
    )

    return {
        "exp_dir": exp_dir,
        "log_path": os.path.join(exp_dir, "train.log"),
        "seed": seed,
        "split_level": split_level,
        "neg_pos_ratio": neg_pos_ratio,
        "use_fgm": use_fgm,
        "fgm_epsilon": fgm_epsilon,
        "use_class_weights": use_class_weights,
        "class_weight_mode": class_weight_mode,
        "class_weight_max": class_weight_max,
        "use_dyn_filter": use_dyn_filter,
        "dyn_filter_target_precision": dyn_filter_target_precision,
        "dyn_filter_min_threshold": dyn_filter_min_threshold,
        "dyn_filter_max_threshold": dyn_filter_max_threshold,
        "dyn_filter_num_steps": dyn_filter_num_steps,
        "forbidden_relations_csv": forbidden_relations_csv,
    }


def _parse_forbidden_relations(csv_str: str) -> set[str]:
    if not csv_str:
        return set()
    parts = [p.strip() for p in csv_str.split(",")]
    return {p for p in parts if p}


def _compute_class_weights(
    train_samples,
    *,
    label2id: Dict[str, int],
    mode: str = "inv_sqrt",
    max_w: float = 10.0,
) -> torch.Tensor:
    """Compute per-class weights for CrossEntropyLoss from training label frequencies.

    Parameters
    ----------
    mode:
        - inv_sqrt: w_c = 1 / sqrt(freq_c)
        - inv:      w_c = 1 / freq_c
    max_w:
        Clamp max weight to avoid exploding gradients.
    """
    counts = Counter([s.label for s in train_samples])
    freqs = []
    for lbl, lid in sorted(label2id.items(), key=lambda x: x[1]):
        freqs.append(max(1, int(counts.get(lbl, 0))))

    w = []
    for f in freqs:
        if mode == "inv":
            ww = 1.0 / float(f)
        else:
            ww = 1.0 / (float(f) ** 0.5)
        w.append(ww)

    mean_w = sum(w) / max(1, len(w))
    w = [x / mean_w for x in w]

    if max_w is not None and max_w > 0:
        w = [min(float(max_w), float(x)) for x in w]

    return torch.tensor(w, dtype=torch.float32)


def _build_label_space(samples) -> Dict[str, int]:
    labels = sorted({s.label for s in samples})
    # ensure NoRelation at 0 for convenience
    if "NoRelation" in labels:
        labels.remove("NoRelation")
        labels = ["NoRelation", *labels]
    return {l: i for i, l in enumerate(labels)}


def _compute_precision_for_threshold(
    y_true: List[int],
    probs: torch.Tensor,
    *,
    no_relation_id: int,
    threshold: float,
    forbidden_pos_ids: set[int] | None = None,
) -> float:
    """Precision over positive predictions after threshold + forbidden filtering."""
    if forbidden_pos_ids is None:
        forbidden_pos_ids = set()

    # predicted label by argmax
    pred = probs.argmax(dim=-1)
    conf = probs.gather(1, pred.view(-1, 1)).squeeze(1)

    # filter low confidence positives -> NoRelation
    out_pred = pred.clone()
    low_conf = (pred != no_relation_id) & (conf < threshold)
    out_pred[low_conf] = no_relation_id

    # filter forbidden relations -> NoRelation
    if forbidden_pos_ids:
        forb = torch.zeros_like(out_pred, dtype=torch.bool)
        for pid in forbidden_pos_ids:
            forb |= out_pred == pid
        out_pred[forb] = no_relation_id

    # compute precision on positive predictions
    tp = fp = 0
    for t, p in zip(y_true, out_pred.tolist()):
        if p != no_relation_id:
            if p == t and t != no_relation_id:
                tp += 1
            else:
                fp += 1
    return tp / (tp + fp) if (tp + fp) else 0.0


@torch.no_grad()
def _apply_dyn_filter(
    y_true: List[int],
    logits: torch.Tensor,
    *,
    no_relation_id: int,
    threshold: float,
    forbidden_pos_ids: set[int] | None = None,
) -> Tuple[List[int], Dict[str, int]]:
    """Apply threshold + forbidden relation filtering and return filtered predictions."""
    if forbidden_pos_ids is None:
        forbidden_pos_ids = set()

    probs = torch.softmax(logits, dim=-1)
    pred = probs.argmax(dim=-1)
    conf = probs.gather(1, pred.view(-1, 1)).squeeze(1)

    out_pred = pred.clone()

    low_conf = (pred != no_relation_id) & (conf < threshold)
    out_pred[low_conf] = no_relation_id

    forb_cnt = 0
    if forbidden_pos_ids:
        forb_mask = torch.zeros_like(out_pred, dtype=torch.bool)
        for pid in forbidden_pos_ids:
            forb_mask |= out_pred == pid
        forb_cnt = int(forb_mask.sum().item())
        out_pred[forb_mask] = no_relation_id

    stats = {
        "filtered_low_conf": int(low_conf.sum().item()),
        "filtered_forbidden": forb_cnt,
    }
    return out_pred.detach().cpu().tolist(), stats


@torch.no_grad()
def evaluate(
    model,
    dl,
    device,
    positive_ids: List[int],
    *,
    confusion_label_ids: List[int] | None = None,
    dyn_filter: Dict[str, object] | None = None,
):
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    total_loss = 0.0
    n = 0

    # Store logits for optional post-processing
    all_logits: List[torch.Tensor] = []

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

        if dyn_filter is not None:
            all_logits.append(logits.detach().cpu())

    # Optional dynamic filter (threshold + forbidden relations)
    filter_stats = None
    if dyn_filter is not None:
        no_relation_id = int(dyn_filter["no_relation_id"])
        threshold = float(dyn_filter["threshold"])
        forbidden_pos_ids = set(dyn_filter.get("forbidden_pos_ids") or [])

        logits_cat = torch.cat(all_logits, dim=0) if all_logits else torch.empty((0,))
        y_pred, filter_stats = _apply_dyn_filter(
            y_true,
            logits_cat,
            no_relation_id=no_relation_id,
            threshold=threshold,
            forbidden_pos_ids=forbidden_pos_ids,
        )

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

    # Optional: prediction distribution given gold label (for error analysis)
    # confusion[gold][pred] = count
    confusion = None
    if confusion_label_ids:
        confusion = {gid: Counter() for gid in confusion_label_ids}
        for t, p in zip(y_true, y_pred):
            if t in confusion:
                confusion[t][p] += 1

    return total_loss / max(1, n), m, per_cls, confusion, filter_stats


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


def _should_save_full_ckpt() -> bool:
    """Whether to save full model weights.

    Default: False to avoid huge checkpoints (BERT weights) exhausting disk.
    Set env `BERT_RE_SAVE_FULL_CKPT=1` to enable.
    """
    v = os.environ.get("BERT_RE_SAVE_FULL_CKPT", "")
    return v.strip().lower() in {"1", "true", "yes", "y"}


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

    # Dynamic filter config (threshold is tuned on dev each epoch if enabled)
    forbidden_rel_names = _parse_forbidden_relations(str(run_cfg.get("forbidden_relations_csv") or ""))
    forbidden_pos_ids: set[int] = set()
    for name in forbidden_rel_names:
        if name in label2id and name != "NoRelation":
            forbidden_pos_ids.add(int(label2id[name]))

    dyn_cfg = None
    dyn_threshold = 0.0
    if run_cfg.get("use_dyn_filter"):
        dyn_cfg = {
            "no_relation_id": int(label2id.get("NoRelation", 0)),
            "threshold": 0.0,  # filled in each epoch
            "forbidden_pos_ids": forbidden_pos_ids,
        }
        logger.info(
            "Dynamic filter enabled: target_prec=%.3f steps=%d thr_range=[%.2f, %.2f], forbidden=%s",
            float(run_cfg.get("dyn_filter_target_precision") or 0.85),
            int(run_cfg.get("dyn_filter_num_steps") or 50),
            float(run_cfg.get("dyn_filter_min_threshold") or 0.0),
            float(run_cfg.get("dyn_filter_max_threshold") or 0.99),
            sorted(list(forbidden_rel_names)),
        )

    # Optional: class-weighted loss for imbalanced classes
    ce_loss_fn = None
    try:
        if run_cfg.get("use_class_weights"):
            w = _compute_class_weights(
                train_samples,
                label2id=label2id,
                mode=str(run_cfg.get("class_weight_mode") or "inv_sqrt"),
                max_w=float(run_cfg.get("class_weight_max") or 10.0),
            )
            ce_loss_fn = torch.nn.CrossEntropyLoss(weight=w.to(config.device))
            # Log weights with label names for analysis
            parts = []
            for lbl, lid in sorted(label2id.items(), key=lambda x: x[1]):
                parts.append(f"{lbl}={float(w[lid]):.4f}")
            logger.info(
                "Using class-weighted CrossEntropyLoss (mode=%s, max=%.2f): %s",
                run_cfg.get("class_weight_mode"),
                float(run_cfg.get("class_weight_max") or 10.0),
                ", ".join(parts),
            )
    except Exception:
        ce_loss_fn = None

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

            # If enabled, override model's internal loss with class-weighted CE.
            if ce_loss_fn is not None:
                loss = ce_loss_fn(out.logits, batch["labels"])
            else:
                loss = out.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)

            total_loss += float(loss.item()) * batch["input_ids"].shape[0]
            n += batch["input_ids"].shape[0]

        train_loss = total_loss / max(1, n)

        # If enabled, tune threshold on dev to reach target precision (among positives)
        if dyn_cfg is not None:
            # Run a forward pass on dev to collect logits & labels
            y_true_tmp: List[int] = []
            logits_tmp: List[torch.Tensor] = []
            model.eval()
            for batch in dev_dl:
                batch = {k: v.to(config.device) for k, v in batch.items()}
                out = model(**batch)
                logits_tmp.append(out.logits.detach().cpu())
                y_true_tmp.extend(batch["labels"].detach().cpu().tolist())

            logits_cat = torch.cat(logits_tmp, dim=0)
            probs = torch.softmax(logits_cat, dim=-1)
            no_id = int(label2id.get("NoRelation", 0))

            target_p = float(run_cfg.get("dyn_filter_target_precision") or 0.85)
            min_thr = float(run_cfg.get("dyn_filter_min_threshold") or 0.0)
            max_thr = float(run_cfg.get("dyn_filter_max_threshold") or 0.99)
            steps = int(run_cfg.get("dyn_filter_num_steps") or 50)

            best_thr = min_thr
            # Search increasing threshold until reaching target precision; otherwise take max_thr.
            for i in range(steps + 1):
                thr = min_thr + (max_thr - min_thr) * (i / max(1, steps))
                prec = _compute_precision_for_threshold(
                    y_true_tmp,
                    probs,
                    no_relation_id=no_id,
                    threshold=thr,
                    forbidden_pos_ids=forbidden_pos_ids,
                )
                if prec >= target_p:
                    best_thr = thr
                    break
                best_thr = thr

            dyn_threshold = float(best_thr)
            dyn_cfg["threshold"] = dyn_threshold
            logger.info("Dynamic threshold tuned on dev: thr=%.4f (target_prec=%.3f)", dyn_threshold, target_p)

        dev_loss, dev_m, dev_per_cls, dev_conf, dev_filter_stats = evaluate(
            model,
            dev_dl,
            config.device,
            positive_ids=positive_ids,
            confusion_label_ids=positive_ids,
            dyn_filter=dyn_cfg,
        )
        test_loss, test_m, test_per_cls, _, test_filter_stats = evaluate(
            model,
            test_dl,
            config.device,
            positive_ids=positive_ids,
            dyn_filter=dyn_cfg,
        )

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

        # Confusion summary for gold Attribute_of (why F1 can be 0)
        try:
            if dev_conf is not None:
                # Find label id for Attribute_of if present
                attr_id = label2id.get("Attribute_of")
                if attr_id is not None and attr_id in dev_conf:
                    total = sum(dev_conf[attr_id].values())
                    if total > 0:
                        # show top predictions
                        pairs = sorted(dev_conf[attr_id].items(), key=lambda x: (-x[1], x[0]))
                        top = []
                        for pid, cnt in pairs[:10]:
                            top.append(f"{id2label.get(pid, str(pid))}={cnt}")
                        logger.info(
                            "Dev confusion (gold=Attribute_of, n=%d): %s",
                            total,
                            ", ".join(top),
                        )
        except Exception:
            pass

        # Also log test for paper reporting (do NOT select best by test)
        try:
            if test_filter_stats is not None:
                logger.info(
                    "Test filter stats: thr=%.4f low_conf=%d forbidden=%d",
                    float(dyn_threshold),
                    int(test_filter_stats.get("filtered_low_conf", 0)),
                    int(test_filter_stats.get("filtered_forbidden", 0)),
                )
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

            # To keep disk usage low, by default only save lightweight artifacts.
            # Full BERT checkpoints can be hundreds of MB per run.
            if _should_save_full_ckpt():
                ckpt = {
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
            else:
                # Lightweight default: save only classification head + metadata.
                # This is sufficient for paper reporting and comparison experiments.
                try:
                    head_only = {
                        "classifier": model.classifier.state_dict() if hasattr(model, "classifier") else None,
                        "label2id": label2id,
                        "id2label": id2label,
                        "bert_model": config.bert_model,
                        "max_length": config.max_length,
                        "note": "lightweight checkpoint (classifier head only). Set BERT_RE_SAVE_FULL_CKPT=1 to save full model.",
                    }
                    _atomic_torch_save(head_only, best_path + ".head_only.pt", retries=1)
                    logger.info(
                        "Saved best lightweight checkpoint: %s (best_f1=%.4f)",
                        best_path + ".head_only.pt",
                        best_f1,
                    )
                except Exception:
                    logger.exception("Failed to save lightweight checkpoint artifacts.")

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

