#!/usr/bin/env python3
import os
import config
# 临时覆盖配置用于快速验证
base_dir = os.path.dirname(os.path.abspath(__file__))
config.data_dir = os.path.join(base_dir, 'data', 'my') + '/'
config.files = ['admin']
config.train_dir = config.data_dir + 'admin.npz'
config.test_dir = config.data_dir + 'admin.npz'
base_dir = os.path.dirname(os.path.abspath(__file__))
config.exp_dir = os.path.join(base_dir, 'experiments', 'my') + '/'
# 小样本训练参数
config.epoch_num = 3
config.batch_size = 8
# 使用cpu以兼容没有GPU的环境
config.gpu = ''

# 运行训练（simple_run）
from run import simple_run

if not os.path.exists(config.exp_dir):
    os.makedirs(config.exp_dir, exist_ok=True)

simple_run()
print('Quick run finished')
