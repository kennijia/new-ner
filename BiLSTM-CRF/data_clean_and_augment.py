#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple data cleaning and augmentation for admin dataset.
- Validate span -> text consistency; if mismatch try to locate the entity text in the sentence and fix span
- Normalize VALUE/LEVEL_KEY entity texts (strip spaces, normalize number format like '163m' or '165.82m')
- Provide an optional small augmentation: replace ORG with other ORG strings; perturb numeric VALUES

Outputs:
- data/my/admin_clean.json (cleaned)
- data/my/admin_aug.json (cleaned + augmented)

Usage: python data_clean_and_augment.py --input data/my/admin.json --out_dir data/my/ --augment 100
"""
import argparse
import json
import re
import os
import random
from collections import defaultdict

NUM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[mM]?")


def normalize_value(text):
    # remove spaces and unify 'm' lowercase
    t = text.replace(' ', '')
    t = t.replace('Ｍ', 'm').replace('M', 'm')
    return t


def find_subtext_offsets(text, subtext):
    # return list of (start, end_inclusive)
    res = []
    start = 0
    while True:
        idx = text.find(subtext, start)
        if idx == -1:
            break
        res.append((idx, idx + len(subtext) - 1))
        start = idx + 1
    return res


def repair_entry(entry):
    text = entry['text']
    labels = entry.get('label', {})
    repaired = defaultdict(lambda: defaultdict(list))
    for tag, entmap in labels.items():
        for ent_text, spans in entmap.items():
            norm_ent_text = ent_text.strip()
            if tag in ('VALUE', 'LEVEL_KEY'):
                norm_ent_text = normalize_value(norm_ent_text)
            for s, e in spans:
                # verify
                if s <= e and ''.join(list(text)[s:e+1]) == norm_ent_text:
                    repaired[tag][norm_ent_text].append([s, e])
                else:
                    # try direct match
                    offsets = find_subtext_offsets(text, norm_ent_text)
                    if offsets:
                        repaired[tag][norm_ent_text].append([offsets[0][0], offsets[0][1]])
                    else:
                        # fallback: if numeric, try normalizing text (strip spaces) and search for digits
                        if tag in ('VALUE', 'LEVEL_KEY'):
                            # extract number-like string from ent_text
                            m = NUM_RE.search(ent_text)
                            if m:
                                candidate = m.group(0).replace(' ', '')
                                idx = text.find(candidate)
                                if idx != -1:
                                    repaired[tag][candidate].append([idx, idx+len(candidate)-1])
                                else:
                                    # give up: keep original span if in bounds
                                    if 0 <= s < len(text) and 0 <= e < len(text):
                                        repaired[tag][norm_ent_text].append([s, e])
                            else:
                                # keep original span if in bounds
                                if 0 <= s < len(text) and 0 <= e < len(text):
                                    repaired[tag][norm_ent_text].append([s, e])
                        else:
                            # non-numeric mismatch: try to search for ent_text stripped of punctuation
                            candidate = re.sub(r"[\W_]+", '', norm_ent_text)
                            candidate_offsets = find_subtext_offsets(re.sub(r"[\W_]+", '', text), candidate)
                            if candidate_offsets:
                                # map candidate offsets back to original text (approximate)
                                start = candidate_offsets[0][0]
                                repaired[tag][norm_ent_text].append([start, start+len(norm_ent_text)-1])
                            else:
                                # as last resort, keep original span if within bounds
                                if 0 <= s < len(text) and 0 <= e < len(text):
                                    repaired[tag][norm_ent_text].append([s, e])
    return {'text': text, 'label': dict(repaired)}


def augment_dataset(entries, n_aug=100, random_seed=42):
    random.seed(random_seed)
    # collect ORG candidates and numeric values
    orgs = []
    values = []
    for ent in entries:
        for tag, entmap in ent.get('label', {}).items():
            if tag == 'ORG':
                for k in entmap.keys():
                    orgs.append(k)
            if tag in ('VALUE', 'LEVEL_KEY'):
                for k in entmap.keys():
                    values.append(k)
    orgs = list(set([o for o in orgs if len(o) > 1]))
    values = list(set(values))

    out = entries.copy()
    for i in range(n_aug):
        src = random.choice(entries)
        text = src['text']
        label = src['label']
        # pick augmentation type randomly
        aug = dict(label)
        aug_text = text
        modified = False
        # 50% replace an ORG
        if orgs and random.random() < 0.5:
            # pick a random org present in this sample
            orgs_in_sample = [k for k in label.get('ORG', {}).keys()]
            if orgs_in_sample:
                old = random.choice(orgs_in_sample)
                new = random.choice(orgs)
                if old != new:
                    # replace first occurrence in text (we keep spans consistent by searching)
                    idx = aug_text.find(old)
                    if idx != -1:
                        aug_text = aug_text[:idx] + new + aug_text[idx+len(old):]
                        # recompute labels: remove old entry and add new one
                        aug_label = dict(label)
                        aug_label = json.loads(json.dumps(aug_label))
                        spans = aug_label['ORG'].pop(old)
                        aug_label.setdefault('ORG', {})
                        aug_label['ORG'].setdefault(new, []).extend([[idx, idx+len(new)-1] for _ in spans])
                        modified = True
                        out.append({'text': aug_text, 'label': aug_label})
        # else, numeric perturbation
        if not modified and values and random.random() < 0.5:
            val_in_sample = [k for k in label.keys() if k in ('VALUE', 'LEVEL_KEY')]
            if val_in_sample:
                # choose first numeric entity type and perturb a numeric token
                t = val_in_sample[0]
                keys = list(label[t].keys())
                if keys:
                    old = random.choice(keys)
                    m = re.search(r"(\d+(?:\.\d+)?)", old)
                    if m:
                        num = float(m.group(1))
                        # perturb by +/- (1..5)
                        delta = random.choice([-5, -1, 1, 5])
                        newnum = num + delta
                        # preserve unit if present
                        unit = old.replace(m.group(1), '')
                        new = (str(int(newnum)) if newnum.is_integer() else f"{newnum:.2f}") + unit
                        idx = aug_text.find(old)
                        if idx != -1:
                            aug_text = aug_text[:idx] + new + aug_text[idx+len(old):]
                            aug_label = json.loads(json.dumps(label))
                            spans = aug_label[t].pop(old)
                            aug_label.setdefault(t, {})
                            aug_label[t].setdefault(new, []).extend([[idx, idx+len(new)-1] for _ in spans])
                            out.append({'text': aug_text, 'label': aug_label})
        # otherwise skip
    return out


def main(input_path, out_dir, augment_count=0):
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    entries = []
    with open(input_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            if not line.strip():
                continue
            entries.append(json.loads(line))

    cleaned = [repair_entry(e) for e in entries]

    clean_path = os.path.join(out_dir, 'admin_clean.json')
    with open(clean_path, 'w', encoding='utf-8') as fout:
        for e in cleaned:
            fout.write(json.dumps(e, ensure_ascii=False) + '\n')

    if augment_count and augment_count > 0:
        augmented = augment_dataset(cleaned, n_aug=augment_count)
        # ensure augmented samples are repaired (span/text consistency)
        repaired_aug = [repair_entry(e) for e in augmented]
        aug_path = os.path.join(out_dir, 'admin_aug.json')
        with open(aug_path, 'w', encoding='utf-8') as fout:
            for e in repaired_aug:
                fout.write(json.dumps(e, ensure_ascii=False) + '\n')
        print(f'Wrote augmented dataset to {aug_path} (total {len(repaired_aug)} examples)')
        augmented = repaired_aug
    else:
        augmented = cleaned

    print(f'Wrote cleaned dataset to {clean_path} (total {len(cleaned)} examples)')

    # Convert to npz via data_process by temporarily pointing config.data_dir
    # We'll also create admin_clean.npz and admin_aug.npz using the project's data_process
    from data_process import Processor
    import config
    old_data_dir = config.data_dir
    old_files = config.files
    # write temporary files (processor expects data_dir + <file>.json)
    tmp_dir = out_dir
    # ensure processor will use this folder
    config.data_dir = tmp_dir + '/'
    config.files = ['admin_clean']
    # we already have admin_clean.json
    p = Processor(config)
    p.get_examples('admin_clean')

    if augment_count and augment_count > 0:
        config.files = ['admin_aug']
        p.get_examples('admin_aug')

    # restore
    config.data_dir = old_data_dir
    config.files = old_files
    print('Converted to .npz files via Processor')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default='data/my/admin.json')
    parser.add_argument('--out_dir', '-o', default='data/my')
    parser.add_argument('--augment', '-a', type=int, default=0)
    args = parser.parse_args()
    main(args.input, args.out_dir, args.augment)
