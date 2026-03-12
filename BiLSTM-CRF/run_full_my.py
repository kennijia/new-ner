#!/usr/bin/env python3
import os

# Silence libgomp warnings (must be an integer)
os.environ.setdefault("OMP_NUM_THREADS", "1")

import config

# Full training on data/my using GPU
base_dir = os.path.dirname(os.path.abspath(__file__))
config.data_dir = os.path.join(base_dir, 'data', 'my') + '/'

# Use the unified split files produced by the processor
config.files = ['admin_train', 'admin_test']
config.train_dir = os.path.join(config.data_dir, 'admin_train.npz')
config.test_dir = os.path.join(config.data_dir, 'admin_test.npz')

config.exp_dir = os.path.join(base_dir, 'experiments', 'my_full') + '/'
config.model_dir = os.path.join(config.exp_dir, 'model.pth')
config.log_dir = os.path.join(config.exp_dir, 'train.log')
config.case_dir = os.path.join(base_dir, 'case', 'bad_case.txt')

# training params
config.epoch_num = 100
config.batch_size = 32
config.patience_num = 5

# use GPU 0 by default; change if needed
config.gpu = '0'

os.makedirs(config.exp_dir, exist_ok=True)

import run as run_module

run_module.simple_run()
print('Full run finished')
