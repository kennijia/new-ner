import os
import torch

# 数据目录与模型输出目录
data_dir = os.getcwd() + '/data/my/'
train_dir = data_dir + 'admin_train.npz'
test_dir = data_dir + 'admin_test.npz'
files = ['admin_train', 'admin_test']
bert_model = 'pretrained_bert_models/chinese_roberta_wwm_large_ext/'
roberta_model = 'pretrained_bert_models/chinese_roberta_wwm_large_ext/'
model_dir = os.getcwd() + '/experiments/my/'
log_dir = model_dir + 'train.log'
case_dir = os.getcwd() + '/case/bad_case.txt'

# 训练集、验证集划分比例
dev_split_size = 0.1

# 是否加载训练好的NER模型
load_before = False

# 是否对整个BERT进行fine tuning
full_fine_tuning = True

# hyper-parameter
learning_rate = 3e-5
weight_decay = 0.01
clip_grad = 5

batch_size = 32
epoch_num = 50
min_epoch_num = 5
patience = 0.0002
patience_num = 10

# FGM adversarial training
use_fgm = True
fgm_epsilon = 1.0

gpu = '0'

if gpu != '':
    device = torch.device(f"cuda:{gpu}")
else:
    device = torch.device("cpu")

labels = ['ACTION', 'LEVEL_KEY', 'OBJ', 'ORG', 'VALUE']

label2id = {
    'O': 0,
    'B-ACTION': 1,
    'B-LEVEL_KEY': 2,
    'B-OBJ': 3,
    'B-ORG': 4,
    'B-VALUE': 5,
    'I-ACTION': 6,
    'I-LEVEL_KEY': 7,
    'I-OBJ': 8,
    'I-ORG': 9,
    'I-VALUE': 10,
    'S-ACTION': 11,
    'S-LEVEL_KEY': 12,
    'S-OBJ': 13,
    'S-ORG': 14,
    'S-VALUE': 15,
}

id2label = {_id: _label for _label, _id in list(label2id.items())}
