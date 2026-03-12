import os
import json
import logging
import numpy as np


class Processor:
    def __init__(self, config):
        self.data_dir = config.data_dir
        self.files = config.files

    def data_process(self):
        for file_name in self.files:
            self.get_examples(file_name)

    def get_examples(self, mode):
        """
        将json文件每一行中的文本分离出来，存储为words列表
        标记文本对应的标签，存储为labels
        words示例：['生', '生', '不', '息', 'C', 'S', 'O', 'L']
        labels示例：['O', 'O', 'O', 'O', 'B-game', 'I-game', 'I-game', 'I-game']

        支持两种label格式：
         - 嵌套 dict（项目原有格式）：{"tag": {"entity_text": [[s,e], ...]}}
         - 三元组列表（新格式，每项 [start,end,tag]，其中 end 为 **独占** 索引）
        """
        input_dir = self.data_dir + str(mode) + '.json'
        output_dir = self.data_dir + str(mode) + '.npz'
        if os.path.exists(output_dir) is True:
            return
        with open(input_dir, 'r', encoding='utf-8') as f:
            word_list = []
            label_list = []
            # 先读取到内存中，然后逐行处理
            for line in f.readlines():
                # loads()：用于处理内存中的json对象，strip去除可能存在的空格
                json_line = json.loads(line.strip())

                text = json_line['text']
                words = list(text)
                # 如果没有label，则返回None
                label_entities = json_line.get('label', None)
                labels = ['O'] * len(words)

                # 兼容新的三元组格式（[start,end,tag] 列表）
                if isinstance(label_entities, list):
                    # 将其转换成项目原先使用的嵌套字典格式
                    tmp = {}
                    for triple in label_entities:
                        if len(triple) < 3:
                            continue
                        s, e, tag = triple
                        # 假设输入为 [start, end) (end exclusive)，转换为 inclusive
                        if not (0 <= s < len(words) and 0 < e <= len(words) and s < e):
                            continue
                        e_incl = e - 1
                        ent_text = ''.join(words[s:e])
                        tmp.setdefault(tag, {}).setdefault(ent_text, []).append([s, e_incl])
                    label_entities = tmp

                if label_entities is not None:
                    for key, value in label_entities.items():
                        for sub_name, sub_index in value.items():
                            for start_index, end_index in sub_index:
                                # 校验文本片段一致性，若不一致尝试自动修正
                                span_text = ''.join(words[start_index:end_index + 1])
                                if span_text != sub_name:
                                    full_text = ''.join(words)
                                    found = full_text.find(sub_name)
                                    if found != -1:
                                        new_start = found
                                        new_end = found + len(sub_name) - 1
                                        logging.warning(f"Span mismatch fixed for '{sub_name}': {start_index}-{end_index} -> {new_start}-{new_end}")
                                        start_index, end_index = new_start, new_end
                                    else:
                                        # 尝试去除首尾空白再定位
                                        trimmed = sub_name.strip()
                                        found2 = full_text.find(trimmed)
                                        if found2 != -1:
                                            new_start = found2
                                            new_end = found2 + len(trimmed) - 1
                                            logging.warning(f"Span mismatch fixed (trimmed) for '{sub_name}': {start_index}-{end_index} -> {new_start}-{new_end}")
                                            start_index, end_index = new_start, new_end
                                        else:
                                            logging.warning(f"Span text mismatch and cannot fix, skipping entity: {span_text} != {sub_name}")
                                            continue
                                # 经过修正或原本一致后标注BIO
                                # 增加索引边界校验，避免出现越界赋值导致 IndexError
                                if not (0 <= start_index <= end_index < len(labels)):
                                    logging.warning(f"Entity indices out of range after fix, skipping: {start_index}-{end_index} for '{sub_name}' in text length {len(labels)}")
                                    continue
                                if start_index == end_index:
                                    labels[start_index] = 'S-' + key
                                else:
                                    labels[start_index] = 'B-' + key
                                    labels[start_index + 1:end_index + 1] = ['I-' + key] * (end_index - start_index)
                word_list.append(words)
                label_list.append(labels)
            # 保存成二进制文件
            np.savez_compressed(output_dir, words=word_list, labels=label_list)
            logging.info("-------- {} data process DONE!--------".format(mode))
