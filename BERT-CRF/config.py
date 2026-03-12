import os
import torch

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, 'data', 'my') + '/'
train_dir = data_dir + 'admin_train.npz'
test_dir = data_dir + 'admin_test.npz'
files = ['admin_train', 'admin_test']

# Use absolute local paths so HF Transformers won't treat them as Hub repo IDs.
bert_model = os.path.join(base_dir, 'pretrained_bert_models', 'bert-base-chinese')
roberta_model = os.path.join(base_dir, 'pretrained_bert_models', 'chinese_roberta_wwm_large_ext')

model_dir = os.path.join(base_dir, 'experiments', 'admin_split') + '/'
log_dir = os.path.join(model_dir, 'train.log')
case_dir = os.path.join(base_dir, 'case', 'bad_case.txt')

# 训练集、验证集划分比例
dev_split_size = 0.1

# 是否加载训练好的NER模型
load_before = False  # 改为 False，确保从头微调

# 是否对整个BERT进行fine tuning
full_fine_tuning = True

# 模型选择开关
model_type = 'bert'  # 可选 'bert' 或 'roberta'

if model_type == 'bert':
    bert_model = bert_model
    hidden_size = 768
elif model_type == 'roberta':
    bert_model = roberta_model
    hidden_size = 1024
else:
    raise ValueError(f"Unsupported model_type: {model_type}")

# 是否使用 BiLSTM（True: BERT-BiLSTM-CRF；False: 纯 BERT-CRF）
use_bilstm = False

# hyper-parameter
learning_rate = 5e-5 # 提高整体学习率
weight_decay = 0.01
clip_grad = 1.0 

batch_size = 32 
gradient_accumulation_steps = 1 # 减少累积步数，增加更新频率

epoch_num = 100 
min_epoch_num = 5 # 降低最小轮数，允许更早停止
patience = 0.00002
patience_num = 10 # 降低耐心值，开启有效的 Early Stopping (原 50 太大)

# R-Drop 超参数
use_rdrop = False # 暂时关闭，排查性能下降原因
rdrop_alpha = 4.0 # R-Drop 典型值在 1-5 之间

# FGM 超参数
use_fgm = True # 暂时关闭
fgm_epsilon = 1.0

# Dice Loss (mitigate label imbalance, especially the 'O' class)
# When enabled, Dice loss will be combined with the CRF negative log-likelihood.
use_dice_loss = False
# total_loss = crf_loss + dice_loss_weight * dice_loss
dice_loss_weight = 0.5
# 不把“O”(background)计入 Dice 的平均（更关注实体类）
dice_exclude_o = True

# EMA 平滑系数
# 设为 0 以关闭 EMA (shadow = current)，回归原始简单框架
ema_decay = 0.0 

# 辅助损失权重
aux_loss_alpha = 1.0 

# 为不同层设置不同的学习率
bert_lr = 5e-5 # 提高到标准微调学习率
head_lr = 5e-5 # 与 BERT 保持一致，标准配置

gpu = '0'

if gpu != '':
    device = torch.device(f"cuda:{gpu}")
else:
    device = torch.device("cpu")

# 这里的 labels 需要替换为你最新的 5 个标签
labels = ['ORG', 'ACTION', 'OBJ', 'LEVEL_KEY', 'VALUE']

# 类别权重（根据你的训练表现微调）
# ACTION 和 LEVEL_KEY 表现较差，这里赋予更高的权重（用于加权 Loss 或者作为论文分析点）
class_weights = {
    'ORG': 1.0,
    'ACTION': 1.0, # 恢复为 1.0，强制加权会导致 Precision 暴跌，拖累 F1
    'OBJ': 1.0,
    'LEVEL_KEY': 1.0, # 恢复为 1.0
    'VALUE': 1.0
}

# 自动构建 BIOS 标签映射，确保每个标签都有唯一 ID，避免 predict 出 None
label2id = {'O': 0}
for i, label in enumerate(labels):
    label2id[f'B-{label}'] = i * 3 + 1
    label2id[f'I-{label}'] = i * 3 + 2
    label2id[f'S-{label}'] = i * 3 + 3

id2label = {i: label for label, i in label2id.items()}
num_labels = len(id2label)

# 确保训练输出路径正确
exp_dir = model_dir
