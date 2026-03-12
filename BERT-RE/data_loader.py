from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from utils import read_jsonl


@dataclass
class PairSample:
    sent_id: int
    text: str  # with marker tokens
    label: str


def _insert_markers(text: str, h_span: Tuple[int, int], t_span: Tuple[int, int]) -> str:
    """Insert entity markers into raw text by character offsets.

    Offsets in the dataset appear to be character offsets.
    We insert: [HEAD]...[/HEAD] and [TAIL]...[/TAIL].
    """

    (h_s, h_e) = h_span
    (t_s, t_e) = t_span

    # ensure head is before tail for correct insertion order
    if h_s <= t_s:
        first = (h_s, h_e, "[HEAD]", "[/HEAD]")
        second = (t_s, t_e, "[TAIL]", "[/TAIL]")
    else:
        first = (t_s, t_e, "[TAIL]", "[/TAIL]")
        second = (h_s, h_e, "[HEAD]", "[/HEAD]")

    def _apply(x: str, s: int, e: int, l: str, r: str) -> str:
        return x[:s] + l + x[s:e] + r + x[e:]

    out = _apply(text, first[0], first[1], first[2], first[3])
    # second span needs to be shifted by added marker length if inserted after first
    shift = len(first[2]) + len(first[3])
    if second[0] >= first[1]:
        out = _apply(out, second[0] + shift, second[1] + shift, second[2], second[3])
    else:
        out = _apply(out, second[0], second[1], second[2], second[3])
    return out


def build_pair_samples(jsonl_path: str, negative_label: str = "NoRelation") -> List[PairSample]:
    """Convert doc-level JSONL (entities + relations) to pair classification samples."""
    data = read_jsonl(jsonl_path)
    samples: List[PairSample] = []

    for item in data:
        sent_id = int(item.get("id", -1))
        text = item["text"]
        entities = item.get("entities", [])
        relations = item.get("relations", [])

        ent_by_id = {e["id"]: e for e in entities}

        # positive map: (from_id, to_id) -> type
        pos: Dict[Tuple[int, int], str] = {}
        for r in relations:
            pos[(r["from_id"], r["to_id"])] = r["type"]

        ent_ids = [e["id"] for e in entities]
        # generate ordered pairs (i != j)
        for h_id in ent_ids:
            for t_id in ent_ids:
                if h_id == t_id:
                    continue
                label = pos.get((h_id, t_id), negative_label)
                h = ent_by_id[h_id]
                t = ent_by_id[t_id]
                marked = _insert_markers(
                    text,
                    (int(h["start_offset"]), int(h["end_offset"])),
                    (int(t["start_offset"]), int(t["end_offset"])),
                )
                samples.append(PairSample(sent_id=sent_id, text=marked, label=label))

    return samples


def downsample_negatives(
    samples: List[PairSample],
    negative_label: str = "NoRelation",
    neg_pos_ratio: Optional[float] = 3.0,
    seed: int = 42,
) -> List[PairSample]:
    """Downsample negatives to ~neg_pos_ratio * positives.

    If neg_pos_ratio is None or < 0: keep all negatives.
    """
    if neg_pos_ratio is None or neg_pos_ratio < 0:
        return samples

    pos = [s for s in samples if s.label != negative_label]
    neg = [s for s in samples if s.label == negative_label]

    if not pos:
        return samples

    target_neg = int(len(pos) * float(neg_pos_ratio))
    if len(neg) <= target_neg:
        return samples

    rnd = random.Random(seed)
    neg = list(neg)
    rnd.shuffle(neg)
    neg = neg[:target_neg]
    return pos + neg


def split_by_sentence_id(
    samples: List[PairSample],
    dev_ratio: float,
    seed: int,
) -> Tuple[List[PairSample], List[PairSample]]:
    """Split by sentence id (recommended to avoid leakage)."""
    rnd = random.Random(seed)
    sent_ids = sorted({s.sent_id for s in samples})
    rnd.shuffle(sent_ids)
    n_dev = max(1, int(len(sent_ids) * dev_ratio))
    dev_ids = set(sent_ids[:n_dev])
    train = [s for s in samples if s.sent_id not in dev_ids]
    dev = [s for s in samples if s.sent_id in dev_ids]
    return train, dev


def split_train_dev(samples: List[PairSample], dev_ratio: float, seed: int) -> Tuple[List[PairSample], List[PairSample]]:
    # Backward compatible: default is pair-level shuffle split.
    rnd = random.Random(seed)
    samples = list(samples)
    rnd.shuffle(samples)
    n_dev = max(1, int(len(samples) * dev_ratio))
    return samples[n_dev:], samples[:n_dev]


class PairDataset(Dataset):
    def __init__(self, samples: List[PairSample], tokenizer: AutoTokenizer, label2id: Dict[str, int], max_length: int):
        self.samples = samples
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        enc = self.tokenizer(
            s.text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_attention_mask=True,
        )
        item = {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(self.label2id[s.label], dtype=torch.long),
        }
        if "token_type_ids" in enc:
            item["token_type_ids"] = torch.tensor(enc["token_type_ids"], dtype=torch.long)
        return item


def collate_fn(batch, pad_token_id: int = 0):
    max_len = max(x["input_ids"].shape[0] for x in batch)
    input_ids = []
    attention_mask = []
    token_type_ids = []
    labels = []
    has_tt = "token_type_ids" in batch[0]

    for x in batch:
        ids = x["input_ids"]
        mask = x["attention_mask"]
        pad = max_len - ids.shape[0]
        input_ids.append(torch.nn.functional.pad(ids, (0, pad), value=pad_token_id))
        attention_mask.append(torch.nn.functional.pad(mask, (0, pad), value=0))
        labels.append(x["labels"])
        if has_tt:
            tt = x["token_type_ids"]
            token_type_ids.append(torch.nn.functional.pad(tt, (0, pad), value=0))

    batch_out = {
        "input_ids": torch.stack(input_ids, dim=0),
        "attention_mask": torch.stack(attention_mask, dim=0),
        "labels": torch.stack(labels, dim=0),
    }
    if has_tt:
        batch_out["token_type_ids"] = torch.stack(token_type_ids, dim=0)
    return batch_out

