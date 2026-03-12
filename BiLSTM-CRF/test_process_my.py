#!/usr/bin/env python3
import os
import config
from data_process import Processor

# 临时处理 data/my/admin.json
config.data_dir = os.path.abspath(os.path.dirname(__file__)) + '/data/my/'
config.files = ['admin']
processor = Processor(config)
processor.data_process()
print('Done processing data/my/admin.json -> data/my/admin.npz')
