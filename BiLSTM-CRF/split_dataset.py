import numpy as np
from sklearn.model_selection import train_test_split
import os

# 1/7 划分数据集用

# 原始数据文件
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, 'data', 'my')
input_file = os.path.join(data_dir, 'admin.npz')
train_file = os.path.join(data_dir, 'admin_train.npz')
test_file = os.path.join(data_dir, 'admin_test.npz')

# 划分比例
split_ratio = 0.2  # 20% 作为测试集

data = np.load(input_file, allow_pickle=True)
words = data['words']
labels = data['labels']

x_train, x_test, y_train, y_test = train_test_split(words, labels, test_size=split_ratio, random_state=42)

np.savez_compressed(train_file, words=x_train, labels=y_train)
np.savez_compressed(test_file, words=x_test, labels=y_test)

print(f"数据集已划分：\n训练集: {train_file} ({len(x_train)})\n测试集: {test_file} ({len(x_test)})")
