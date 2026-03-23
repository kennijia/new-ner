# NER 对比实验升级清单（2026-03-16）

## 1. 目标

在保留经典基线（便于横向对照）的同时，补充更近年的强基线，解决“对比模型过老”的审稿/答辩风险。

---

## 2. 对比组设计（可直接写进论文）

### A. 经典基线（保留）

- BiLSTM-CRF
- BERT-Softmax
- BERT-CRF
- BERT-LSTM-CRF

> 对应总控脚本：`run_ablation_all.py`

### B. 现代强基线（新增，优先）

在**同一训练框架（BERT-CRF）**下，仅替换 backbone：

1. `bert-base-chinese`（旧强基线，参考点）
2. `chinese_roberta_wwm_large_ext`
3. `chinese-macbert-base`（建议）
4. `chinese-macbert-large`（建议）

> 说明：当前代码以 `BertModel/BertTokenizer` 为核心，以上 backbone 兼容性最好、改动最小。

### C. 可选“结构新基线”（有时间再加）

- GlobalPointer / W2NER（比 CRF 结构更新）

> 这部分需要新增模型代码，不是“只换 backbone”。

### D. LLM 范式基线（建议至少加 1 组）

- LLM Zero-shot NER（OpenAI-compatible API）

> 已提供脚本：`BERT-LSTM-CRF/llm_ner_baseline.py`，可直接在同一测试集上计算 Micro-F1 和分类别 F1。

---

## 3. 公平对比设置（建议固定）

- 数据划分：保持现有 `admin_train/admin_test` 不变
- 随机种子：`42, 43, 44`（报告 mean±std）
- 训练策略：
  - 先做 backbone 对比时固定 `use_bilstm=False`
  - 固定 `FGM=False`（避免训练技巧干扰 backbone 结论）
  - Dice loss 可先关闭；若开启，所有 backbone 统一设置
- 评价指标：总体 F1 + 各标签 F1（ORG/ACTION/OBJ/LEVEL_KEY/VALUE）

---

## 4. 直接可跑命令

在 `BERT-CRF` 目录运行：

```bash
cd /root/msy/ner/BERT-CRF

python grid_backbone.py \
  --backbones \
    /root/msy/ner/BERT-CRF/pretrained_bert_models/bert-base-chinese \
    /root/msy/ner/BERT-CRF/pretrained_bert_models/chinese_roberta_wwm_large_ext \
    /path/to/chinese-macbert-base \
    /path/to/chinese-macbert-large \
  --seeds 42 43 44 \
  --disable-fgm
```

汇总结果：

```bash
python collect_backbone_results.py --summary > experiments/backbone_summary.tsv
```

LLM 对比（在 `BERT-LSTM-CRF` 目录运行）：

```bash
cd /root/msy/ner/BERT-LSTM-CRF

python llm_ner_baseline.py \
  --input data/my/admin_test.npz \
  --model qwen-max \
  --base_url http://127.0.0.1:8000/v1 \
  --api_key sk-xxx \
  --output experiments/llm_ner_baseline/predictions.jsonl \
  --summary experiments/llm_ner_baseline/metrics.json
```

---

## 5. 论文表格建议（最小可用）

建议主表分两块：

- 块 1（经典基线）：BiLSTM-CRF / BERT-Softmax / BERT-CRF / BERT-LSTM-CRF
- 块 2（现代基线）：BERT-CRF + 不同 backbone（含 MacBERT）

每行报告：

- Test F1 (mean±std, 3 seeds)
- ACTION / LEVEL_KEY 单类 F1（你任务中更难的类别）

可附加一行：

- LLM Zero-shot（同测试集，使用相同 Micro-F1 与分类别 F1 口径）

---

## 6. 与导师沟通可用话术

> 我保留了经典基线用于可比性，同时补充了近年来更强的中文预训练 backbone（如 MacBERT），并采用多随机种子报告 mean±std，确保结论不依赖过时模型或单次随机结果。
