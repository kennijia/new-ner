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
        if os.path.exists(output_dir) is True:
            return
        word_list = []
        label_list = []
        with open(input_dir, 'r', encoding='utf-8') as f:
            # 先读取到内存中，然后逐行处理
            for line in f.readlines():
                # loads()：用于处理内存中的json对象，strip去除可能存在的空格
                json_line = json.loads(line.strip())

                text = json_line['text']
                words = list(text)
                # 如果没有label，则返回None
                label_entities = json_line.get('label', None)
                labels = ['O'] * len(words)

                if label_entities is not None:
                    joined = ''.join(words)
                    for key, value in label_entities.items():
                        for sub_name, sub_index in value.items():
                            if not sub_name:
                                continue
                            sub_len = len(sub_name)
                            for start_index, end_index in sub_index:
                                # 部分标注存在尾部多 1 的情况，尝试自动纠正
                                if ''.join(words[start_index:start_index + sub_len]) == sub_name:
                                    end_index = start_index + sub_len - 1
                                else:
                                    locate_from = start_index if start_index < len(words) else 0
                                    new_start = joined.find(sub_name, locate_from)
                                    if new_start == -1:
                                        new_start = joined.find(sub_name)
                                    if new_start == -1:
                                        continue
                                    start_index = new_start
                                    end_index = start_index + sub_len - 1

                                if start_index < 0 or end_index >= len(words):
                                    continue

                                if start_index == end_index:
                                    labels[start_index] = 'S-' + key
                                else:
                                    labels[start_index] = 'B-' + key
                                    labels[start_index + 1:end_index + 1] = ['I-' + key] * (sub_len - 1)
                word_list.append(words)
                label_list.append(labels)
                # 保存成二进制文件
            word_array = np.array(word_list, dtype=object)
            label_array = np.array(label_list, dtype=object)
            np.savez_compressed(output_dir, words=word_array, labels=label_array)
            logging.info("--------{} data process DONE!--------".format(mode))