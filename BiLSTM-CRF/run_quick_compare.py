#!/usr/bin/env python3
import os
import config
import shutil

base_dir = os.path.dirname(os.path.abspath(__file__))
clean_dir = os.path.join(base_dir, 'data', 'my') + '/'

# Quick parameters
config.epoch_num = 5
config.batch_size = 8
# use GPU if available
config.gpu = '0'

from run import simple_run

results = {}

# Helper to run one experiment with a given dataset (admin_clean or admin_aug)
def run_exp(name, file_stem):
    print(f"\n=== RUN {name} ===")
    config.data_dir = clean_dir
    config.files = [file_stem]
    config.train_dir = config.data_dir + file_stem + '.npz'
    config.test_dir = config.data_dir + file_stem + '.npz'
    # set experiment dir
    config.exp_dir = os.path.join(base_dir, 'experiments', f'cmp_{file_stem}') + '/'
    if os.path.exists(config.exp_dir):
        shutil.rmtree(config.exp_dir)
    os.makedirs(config.exp_dir, exist_ok=True)
    simple_run()
    # read last saved f1 from run logs isn't trivial; we will return path to saved model and location of case file
    return {'exp_dir': config.exp_dir, 'model_dir': getattr(config, 'model_dir', None), 'case': config.case_dir}

results['clean'] = run_exp('clean-only', 'admin_clean')
results['clean_aug'] = run_exp('clean+aug', 'admin_aug')

print('\n=== SUMMARY ===')
for k, v in results.items():
    print(k, v)

print('Done')
