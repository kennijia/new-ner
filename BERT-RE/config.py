import os
import torch

# ===== Paths =====
# Default to the same pretrained model directory used across the repo.
base_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(base_dir, os.pardir))

data_dir = os.path.join(project_dir, "BERT-CRF", "data", "my")
train_file = os.path.join(data_dir, "admin-re.jsonl")
dev_file = ""  # optional; if empty, will split from train
test_file = ""  # optional

bert_model = os.path.join(project_dir, "BERT-CRF", "pretrained_bert_models", "chinese_roberta_wwm_large_ext")

exp_root = os.path.join(base_dir, "experiments")
exp_name = "re_pair_cls"
exp_dir = os.path.join(exp_root, exp_name)
log_path = os.path.join(exp_dir, "train.log")

# ===== Training =====
max_length = 256
batch_size = 32
epoch_num = 10
learning_rate = 2e-5
weight_decay = 0.01
warmup_ratio = 0.1

seed = 42

# Split strategy:
# - "pair": split after building entity pairs (fast, but can leak sentence info)
# - "sentence": split by sentence id, then build pairs (recommended for papers)
split_level = "sentence"

dev_split_size = 0.1

# Negative sampling:
# With all ordered entity pairs, NoRelation is dominant. To stabilize training,
# we downsample negative pairs to approximately neg_pos_ratio * num_positive.
# - None / -1: keep all negatives
# - 1.0: roughly 1:1 negatives to positives
neg_pos_ratio = 3.0

# FGM adversarial training (optional)
use_fgm = False
fgm_epsilon = 1.0

gpu = "0"
if gpu != "":
    device = torch.device(f"cuda:{gpu}")
else:
    device = torch.device("cpu")

