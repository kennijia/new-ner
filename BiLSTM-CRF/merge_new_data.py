import json
from collections import defaultdict

# 将新旧数据合并

# 路径配置
new_path = '/home/c403/msy/CLUENER2020/BiLSTM-CRF/data/my/121.jsonl'
old_path = '/home/c403/msy/CLUENER2020/BiLSTM-CRF/data/my/admin_aug.json'
out_path = '/home/c403/msy/CLUENER2020/BiLSTM-CRF/data/my/admin_merged.json'

def load_old(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def load_new(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                # 转换label为嵌套字典
                label_dict = defaultdict(dict)
                for start, end, typ in item['label']:
                    if typ not in label_dict:
                        label_dict[typ] = {}
                    entity = item['text'][start:end]
                    if entity not in label_dict[typ]:
                        label_dict[typ][entity] = []
                    label_dict[typ][entity].append([start, end])
                data.append({'text': item['text'], 'label': label_dict})
    return data

def merge_and_dedup(old, new):
    merged = {}
    for item in old + new:
        merged[item['text']] = item  # 后出现的会覆盖前面的
    return list(merged.values())

def main():
    old_data = load_old(old_path)
    new_data = load_new(new_path)
    merged = merge_and_dedup(old_data, new_data)
    with open(out_path, 'w', encoding='utf-8') as f:
        for item in merged:
            # defaultdict转dict
            if not isinstance(item['label'], dict):
                item['label'] = dict(item['label'])
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f'Merged {len(old_data)} old + {len(new_data)} new -> {len(merged)} unique samples.')

if __name__ == '__main__':
    main()
