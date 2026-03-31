# qa_system 运行说明（论文第 5 章实验）

本模块用于实现“因果知识增强的水资源调度问答系统”。

## 1. 必要数据工件（推荐放到 qa_system/artifacts 下）

你当前的工件在：`/root/msy/ner/BERT-LSTM-CRF/data/my/kg/processed/`。

建议在 `qa_system` 下建立软链接（不复制大文件，结构更清晰）：

- `qa_system/artifacts/nx_graph.gpickle`
- `qa_system/artifacts/faiss_index_qwen_sem.bin`
- `qa_system/artifacts/faiss_meta_qwen_sem.json`

（也可替换为 `faiss_index.bin/faiss_meta.json` 那套。）

## 2. 运行（不调用 LLM，只看检索/融合/Prompt）

在仓库根目录执行：

- `python -m qa_system.run_qa --query "..." --graph qa_system/artifacts/nx_graph.gpickle --index qa_system/artifacts/faiss_index_qwen_sem.bin --meta qa_system/artifacts/faiss_meta_qwen_sem.json --out qa_system/outputs/debug.json`

输出 JSON 包含：`text_channel / graph_channel / fused_evidence / prompt`。

## 3. 运行（检索 + LLM 生成答案）

需要 `.env`（OPENAI_API_KEY/OPENAI_BASE_URL/MODEL_NAME）：

- `python -m qa_system.run_qa_answer --query "..." --graph qa_system/artifacts/nx_graph.gpickle --index qa_system/artifacts/faiss_index_qwen_sem.bin --meta qa_system/artifacts/faiss_meta_qwen_sem.json --env_file .env --out qa_system/outputs/with_llm.json`

## 4. 常见问题

- 如果 `graph_channel.paths` 为空：通常是 NER 未启用或 query 中缺少可链接实体；可尝试加 `--use_ner --ner_model_dir BERT-LSTM-CRF/experiments/...`。
- 第一次运行 embedding 模型会加载较慢，这是正常现象。
