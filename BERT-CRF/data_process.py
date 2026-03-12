import os
import json
import logging
import numpy as np


class Processor:
    def __init__(self, config):
        self.data_dir = config.data_dir
        self.config = config

    def process(self):
        """
        process train and test data
        """
        for file_name in self.config.files:
            self.preprocess(file_name)

    def preprocess(self, mode):
        """
        params:
            words：将json文件每一行中的文本分离出来，存储为words列表
            labels：标记文本对应的标签，存储为labels
        examples:
            words示例：['生', '生', '不', '息', 'C', 'S', 'O', 'L']
            labels示例：['O', 'O', 'O', 'O', 'B-game', 'I-game', 'I-game', 'I-game']
        """
        input_dir = self.data_dir + str(mode) + '.json'
        output_dir = self.data_dir + str(mode) + '.npz'
        
        # 移除存在的检查或更改逻辑，确保修改生效 (或者直接把老的 npz 删掉)
        # if os.path.exists(output_dir) is True:
        #     return
        word_list = []
        label_list = []
        with open(input_dir, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f.readlines()):
                line = line.strip()
                if not line:
                    continue
                try:
                    json_line = json.loads(line)
                except json.decoder.JSONDecodeError as e:
                    logging.error(f"Error parsing JSON on line {i+1} of {input_dir}: {e}")
                    raise e
                text = json_line['text']
                words = list(text)
                if not words:
                    continue
                label_entities = json_line.get('label', None)
                labels = ['O'] * len(words)

                # 兼容新的三元组格式（[start,end,tag] 列表）
                if isinstance(label_entities, list):
                    tmp = {}
                    for triple in label_entities:
                        if len(triple) < 3: continue
                        s, e, tag = triple
                        if not (0 <= s < len(words) and 0 < e <= len(words) and s < e):
                            continue
                        e_incl = e - 1
                        ent_text = ''.join(words[s:e])
                        tmp.setdefault(tag, {}).setdefault(ent_text, []).append([s, e_incl])
                    label_entities = tmp

                if label_entities is not None:
                    for key, value in label_entities.items():
                        # 只处理 config 中定义的标签，过滤掉其他干扰标签
                        if key not in self.config.labels:
                            continue
                        for sub_name, sub_index in value.items():
                            for start_index, end_index in sub_index:
                                span_text = ''.join(words[start_index:end_index + 1])
                                if span_text != sub_name:
                                    full_text = ''.join(words)
                                    found = full_text.find(sub_name)
                                    if found != -1:
                                        start_index, end_index = found, found + len(sub_name) - 1
                                        logging.warning(f"Fixed: {sub_name}")
                                    else:
                                        trimmed = sub_name.strip()
                                        found2 = full_text.find(trimmed)
                                        if found2 != -1:
                                            start_index, end_index = found2, found2 + len(trimmed) - 1
                                        else:
                                            continue
                                
                                # 标注 BIO
                                if 0 <= start_index < len(labels) and 0 <= end_index < len(labels):
                                    if start_index == end_index:
                                        labels[start_index] = 'S-' + key
                                    else:
                                        labels[start_index] = 'B-' + key
                                        for i in range(start_index + 1, end_index + 1):
                                            labels[i] = 'I-' + key
                word_list.append(words)
                label_list.append(labels)

        # NumPy 2.x is stricter about ragged nested sequences; explicitly store as object arrays.
        words_arr = np.array(word_list, dtype=object)
        labels_arr = np.array(label_list, dtype=object)
        np.savez_compressed(output_dir, words=words_arr, labels=labels_arr)
        logging.info("--------{} data process DONE!--------".format(mode))