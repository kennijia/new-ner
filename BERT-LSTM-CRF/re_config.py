# RE classification training config

# data
train_file = "data/my/re_train.jsonl"
dev_file = "data/my/re_dev.jsonl"
test_file = "data/my/re_test.jsonl"

# model
model_name_or_path = "pretrained_bert_models/bert-base-chinese"
output_dir = "experiments/re_cls"

# optimization
max_length = 256
batch_size = 16
lr = 2e-5
weight_decay = 0.01
epochs = 8
warmup_ratio = 0.1
patience = 3
seed = 42

# imbalance handling
use_class_weight = True
