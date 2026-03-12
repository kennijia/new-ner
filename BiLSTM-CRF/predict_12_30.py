import torch
import numpy as np
import warnings
from model import BiLSTM_CRF
from Vocabulary import Vocabulary
import config

# 过滤 uint8 condition tensor 警告
warnings.filterwarnings('ignore', message='where received a uint8 condition tensor')

# 加载词表和标签映射
vocab = Vocabulary(config)
vocab.get_vocab()
id2label = vocab.id2label
word2id = vocab.word2id

# 加载模型
model = torch.load("experiments/clue/model.pth", map_location="cpu")
model.eval()

def text_to_ids(text):
    # 按字符切分
    tokens = list(text)
    ids = [word2id.get(token, word2id.get('<UNK>', 0)) for token in tokens]
    return ids, tokens

def predict(text):
    ids, tokens = text_to_ids(text)
    input_tensor = torch.tensor([ids], dtype=torch.long)  # batch=1
    with torch.no_grad():
        tag_scores = model.forward(input_tensor)
        pred_label_ids = model.crf.decode(tag_scores)[0]
    pred_labels = [id2label[i] for i in pred_label_ids]
    # 输出实体及其区间
    entities = []
    current_entity = None
    for idx, label in enumerate(pred_labels):
        if label.startswith('B-'):
            if current_entity:
                entities.append(current_entity)
            current_entity = {'type': label[2:], 'start': idx, 'text': tokens[idx]}
        elif label.startswith('I-') and current_entity and label[2:] == current_entity['type']:
            current_entity['text'] += tokens[idx]
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None
    if current_entity:
        entities.append(current_entity)
    return entities

if __name__ == "__main__":
    text = input("请输入一句话：")
    entities = predict(text)
    print("识别到的实体：")
    for ent in entities:
        print(f"{ent['type']}: {ent['text']} (位置: {ent['start']})")