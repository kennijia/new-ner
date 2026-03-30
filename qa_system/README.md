# qa_system

本目录用于实现论文第五章：**因果知识增强的水资源调度问答系统**。

## 目标
- 复用第三章 NER（`BERT-LSTM-CRF/predict.py`）完成查询解析与实体抽取/链接。
- 复用第四章 RE/因果三元组与 NetworkX 内存图谱（`BERT-LSTM-CRF/nx_kg.py`、`BERT-LSTM-CRF/causal_dual_retrieval.py`）完成多跳因果路径检索。
- 引入向量语义检索（FAISS + embedding 模型）实现文本证据召回。
- 实现异构证据统一评分与融合，构造因果约束 Prompt，并生成可解释答案（结论 + 证据编号 + 因果链路）。

## 目录规划（建议）
- `config.py`：路径与阈值配置（index/meta/graph、topK、max_hops、alpha/beta 等）
- `pipeline.py`：端到端问答流水线（NER → 双通道检索 → 融合 → Prompt 组装）
- `llm_client.py`：OpenAI-compatible 客户端封装（可换本地/在线模型）
- `run_qa.py`：命令行入口（支持保存 debug 输出 JSON）
- `schemas.py`：统一数据结构（Evidence、Path、Answer 等）

## 与现有代码的衔接
- 双通道检索核心逻辑当前在：`BERT-LSTM-CRF/causal_dual_retrieval.py`
- 图谱构建与查询工具在：`BERT-LSTM-CRF/nx_kg.py`
- 因果三元组抽取与清洗在：`BERT-LSTM-CRF/extract_causal_pairs_llm.py`、`BERT-LSTM-CRF/postprocess_llm_triples.py`

后续我们会在 `qa_system` 内对这些能力做一层“系统化封装”，便于写论文与做实验复现。
