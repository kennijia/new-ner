import json
import random

def split_json(input_file, train_file, test_file, split_ratio=0.8):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    random.seed(42)
    random.shuffle(lines)
    
    split_idx = int(len(lines) * split_ratio)
    train_lines = lines[:split_idx]
    test_lines = lines[split_idx:]
    
    with open(train_file, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
    with open(test_file, 'w', encoding='utf-8') as f:
        f.writelines(test_lines)
    print(f"Split done: {len(train_lines)} train, {len(test_lines)} test samples.")

if __name__ == "__main__":
    split_json('/home/c403/msy/CLUENER2020/BERT-CRF/data/my/admin.json', 
               '/home/c403/msy/CLUENER2020/BERT-CRF/data/my/admin_train.json', 
               '/home/c403/msy/CLUENER2020/BERT-CRF/data/my/admin_test.json')
