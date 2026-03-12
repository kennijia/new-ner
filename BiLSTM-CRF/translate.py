import json

def clean_and_convert(jsonl_file, output_file):
    valid_count = 0
    skipped_count = 0
    
    with open(jsonl_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            try:
                data = json.loads(line)
            except:
                continue
                
            text = data.get('text', '').strip()
            labels = data.get('label', [])
            
            # === 清洗规则 1: 扔掉过短的文本 (过滤标题) ===
            if len(text) < 5:
                skipped_count += 1
                continue
                
            # === 清洗规则 2: 扔掉完全没标注的行 ===
            # (非常重要！防止漏标的数据变成负样本误导模型)
            if not labels:
                skipped_count += 1
                continue

            # === 开始转换 BIO ===
            # 初始化全为 'O'
            tags = ['O'] * len(text)
            
            try:
                for start, end, label_type in labels:
                    # 修正 Doccano 有时候导出的 end 越界问题
                    end = min(end, len(text))
                    if start >= end: continue
                    
                    # 标记 B (Begin)
                    tags[start] = f"B-{label_type}"
                    
                    # 标记 I (Inside)
                    for i in range(start + 1, end):
                        tags[i] = f"I-{label_type}"
                        
                # 写入文件
                # BERT 输入格式：字 + 空格 + 标签
                for char, tag in zip(text, tags):
                    # 去掉不可见字符，防止报错
                    if char.strip():
                        f_out.write(f"{char} {tag}\n")
                
                # 句尾加空行
                f_out.write("\n")
                valid_count += 1
                
            except Exception as e:
                print(f"处理出错，跳过该行: {text[:10]}... 错误: {e}")
                skipped_count += 1

    print(f"处理完成！")
    print(f"✅ 保留有效数据: {valid_count} 条")
    print(f"🗑️ 过滤无效/未标注数据: {skipped_count} 条")
    print(f"文件已保存为: {output_file}")

# 使用方法：把你的导出文件名填在第一个参数
clean_and_convert('admin.jsonl', 'train.txt')