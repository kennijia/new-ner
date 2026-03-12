import os

base_dir = os.path.dirname(os.path.abspath(__file__))

# Use unified dataset (same as other projects)
data_dir = os.path.join(base_dir, 'data', 'my') + '/'
train_dir = data_dir + 'admin_train.npz'
test_dir = data_dir + 'admin_test.npz'
files = ['admin_train', 'admin_test']

vocab_path = data_dir + 'vocab.npz'

exp_dir = os.path.join(base_dir, 'experiments', 'my') + '/'
model_dir = exp_dir + 'model.pth'
log_dir = exp_dir + 'train.log'
case_dir = os.path.join(base_dir, 'case', 'bad_case.txt')

max_vocab_size = 1000000

n_split = 5
dev_split_size = 0.1
embedding_size = 128
hidden_size = 384
drop_out = 0.5
lr = 0.001
betas = (0.9, 0.999)
lr_step = 5
lr_gamma = 0.8

batch_size = 32
epoch_num = 50
# batch_size = 8
# epoch_num = 1
min_epoch_num = 5
# patience = 0.0002
#不早停
patience = 0
patience_num = 5

gpu = '0'

# FGM adversarial training (applied on embedding layer)
use_fgm = True
fgm_epsilon = 1.0

labels = [
        #   'address', 'book', 'company', 'game', 'government',
        #   'movie', 'name', 'organization', 'position', 'scene',
          # new dataset tags
          'ORG', 'ACTION', 'OBJ', 'LEVEL_KEY', 'VALUE']

# 自动生成 label2id，包含 O / B- / I- / S-
label2id = {}
idx = 0
label2id['O'] = idx
idx += 1
# B-*
for tag in labels:
    label2id[f'B-{tag}'] = idx
    idx += 1
# I-*
for tag in labels:
    label2id[f'I-{tag}'] = idx
    idx += 1
# S-*
for tag in labels:
    label2id[f'S-{tag}'] = idx
    idx += 1

id2label = {_id: _label for _label, _id in list(label2id.items())}
