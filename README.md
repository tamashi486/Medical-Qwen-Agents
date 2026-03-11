# Medical-Qwen

基于 Qwen3-32B 的医疗垂直领域大模型系统，覆盖 **SFT/DPO 对齐 → 混合检索 RAG → 反思型 Agent → vLLM 高性能推理** 全链路。

[English](README_en.md) | 中文

---

## 目录

- [项目简介](#项目简介)
- [系统架构](#系统架构)
- [快速开始](#快速开始)
- [模块说明](#模块说明)
- [评测结果](#评测结果)
- [已知局限](#已知局限)

---

## 项目简介

Medical-Qwen 针对通用大模型在医疗场景中的三大核心问题进行改进：

| 问题 | 解决方案 |
|:---|:---|
| 专业知识匮乏 | 基于华佗 QA 数据集进行 SFT 微调 + DPO 偏好对齐 |
| 幻觉频发 | 混合检索 RAG（Dense + BM25 + Reranker）+ Reflection 自修正 |
| 安全合规性差 | 三层风险分级知识库 + 禁忌强制覆盖 + 安全拒答决策 |

---

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│  Presentation Layer          Gradio WebUI @ :7860       │
└──────────────────────────┬──────────────────────────────┘
                           │ HTTP
┌──────────────────────────▼──────────────────────────────┐
│  Application Layer       FastAPI Agent @ :8081          │
│                                                         │
│  ① Intent Router    chat / medical / complex            │
│  ② RAG Engine       Query Rewrite → Hybrid Search       │
│     Dense(BGE-M3) + BM25 → RRF → BGE-Reranker          │
│  ③ Tool Calling     search_medical_knowledge / bmi      │
│  ④ Reflection       Draft → Critique → Refine           │
│  ⑤ Memory           Entity Memory + Summary Memory      │
└──────────────────────────┬──────────────────────────────┘
                           │ OpenAI-Compatible API
┌──────────────────────────▼──────────────────────────────┐
│  Model Layer             vLLM @ :8000                   │
│  Qwen3-32B (SFT+DPO) · BF16 · Tensor Parallel          │
│  PagedAttention · Continuous Batching · Tool Calling    │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│  Data Layer                                             │
│  Milvus Lite · 3,459 语义块 · BGE-M3 (1024d)           │
│  知识库: NMPA药品说明书 / 卫健委指南 / MedlinePlus / PMC │
└─────────────────────────────────────────────────────────┘
```

---

## 快速开始

### 环境要求

- Python 3.10+，CUDA 12.1+
- NVIDIA GPU × 2+（总显存 ≥ 48GB）

### 安装

```bash
git clone https://github.com/your-username/Medical-Qwen.git
cd Medical-Qwen
pip install -r requirements.txt
```

### 模型准备

| 组件 | 模型 |
|:---|:---|
| 推理模型 | Qwen3-32B（SFT+DPO 微调） |
| Embedding | `BAAI/bge-m3` |
| Reranker | `BAAI/bge-reranker-v2-m3` |

修改 `agent/api/server.py` 中的路径配置：

```python
CONFIG = {
    "db_path": "/path/to/RAG/data/milvus.db",
    "embedding_model_path": "/path/to/bge-m3",
    "vllm_api_base": "http://localhost:8000/v1",
    "model_name": "qwen-medical"
}
```

### 启动服务

```bash
# Step 1：启动 vLLM 推理服务（端口 8000）
python agent/run_model.py

# Step 2：启动 Agent 后端（端口 8081）
python agent/run_backend.py

# Step 3：启动 Gradio 前端（端口 7860）
python agent/agent_ui.py
```

访问 `http://localhost:7860` 即可使用。

### 构建 RAG 知识库

```bash
# 采集知识源数据
python -m RAG.data_acquisition.main --all

# 数据预处理与分块
python -m RAG.data_processing.preprocess

# 向量入库
python -m RAG.ingest
```

---

## 模块说明

### 模块一：模型对齐（SFT + DPO）

**数据构建**：基于华佗医疗问答数据集，清洗后保留率约 65%，自动构建 5,000+ 组 DPO 偏好对（chosen/rejected）。

**SFT 训练**：

```yaml
finetuning_type: lora
lora_rank: 8
per_device_train_batch_size: 4
gradient_accumulation_steps: 8     # Global Batch = 128
learning_rate: 5.0e-5
deepspeed: ds_z2_config.json       # ZeRO-2
```

**DPO 训练**：在 SFT 模型基础上继续训练，ZeRO-2 升级至 ZeRO-3（同时加载策略模型 + 参考模型，显存需求翻倍）。

```yaml
pref_loss: sigmoid
pref_beta: 0.1
deepspeed: ds_z3_config.json
```

---

### 模块二：RAG 检索增强

**四段检索管道**：

```
用户输入
    │
    ▼  ① Query Rewriting
    │  LLM 改写 → 规范医学术语，保留特殊人群关键词
    │
    ▼  ② 双路召回
    │  Dense : BGE-M3 + Milvus → top-30
    │  Sparse: BM25             → top-30
    │
    ▼  ③ RRF 融合 (k=60)
    │  score = 1/(60+rank_dense) + 1/(60+rank_bm25) → top-20
    │
    ▼  ④ Cross-Encoder 精排
       BGE-Reranker-v2-M3 → top-5，[证据N] 格式化输出
```

**知识库结构**（共 3,459 语义块，三层风险分级）：

| 层级 | 风险等级 | 内容 | 规模 |
|:---|:---:|:---|:---|
| 基础事实层 | Low | 30 个常见症状/疾病 | ~500 块 |
| 治疗用药层 | Medium/High | 104 种药品说明书 + 30 个治疗原则 | ~2,000 块 |
| 高风险人群层 | High | 15 个专题（孕妇、儿童、肾功能不全等）| ~900 块 |

**数据来源优先级**：NMPA 药品说明书 → 国家卫健委/中华医学会指南 → WHO 指南 → MedlinePlus → PMC OA

---

### 模块三：Agent 智能体

**意图路由**：

```
chat    → 直接 LLM 回答（跳过 RAG，延迟 ~0.5s）
medical → 完整 Agent 链路（RAG + Tool + Reflection，~3s）
complex → 完整链路 + 额外推理步骤
```

**反思循环（Reflection Loop）**：

```
Draft（生成初稿）
    │
    ▼
Critique（Medical Auditor 四维审核）
    ① 是否包含具体处方建议？
    ② 是否引用了不存在的证据（幻觉）？
    ③ 是否遗漏了禁忌症/禁用人群/药物相互作用？
    ④ 是否越权回答非医疗问题？
    │
    ├── PASS → 返回最终回答
    │
    └── FAIL → Refine（重写，补全遗漏禁忌信息）
```

**多级记忆系统**：

| 层级 | 实现 | 作用 |
|:---|:---|:---|
| Session Memory | `RunnableWithMessageHistory` | 维护多轮对话上下文 |
| Entity Memory | LLM 提取 → Dict（异步后台更新）| 患者画像（过敏史/用药/年龄）|
| Summary Memory | 历史 > 10 条时 LLM 压缩 | 降低长对话推理开销 |

Entity Memory 示例：

```
轮1: "我女儿3岁发烧"      → {年龄: 3岁, 性别: 女}
轮2: "她对青霉素过敏"     → {过敏: 青霉素}
轮3: "可以吃阿莫西林吗"   → ⚠️ 自动关联过敏史，拒绝推荐
```

---

### 模块四：推理部署

**vLLM 启动**：

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/qwen3-32B-sft-dpo \
    --served-model-name qwen-medical \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 16384 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --dtype bfloat16
```

**量化选项**：

| 精度 | 模型大小 | 适用场景 |
|:---|:---:|:---|
| BF16 | ~64 GB | 高精度推理 |
| AWQ Int4 | ~16 GB | 单卡 / 边缘部署 |

**Benchmark**（TP=4，4×RTX 6000，input=512，output=256）：

```
请求吞吐:      6.54 req/s
总 Token 吞吐: 5,026 tokens/s
单次推理延迟:   ~154ms
```

---

## 评测结果

评测集：250 题，5 种题型（库内证据型 / 泛化推理型 / 高风险诱导型 / 越界不可答型 / RAG 增益对比型）。
三轮评测方法：规则启发式 / GPT 评判 / RAGAS 语义评测（DeepSeek-V3 作为 Judge）。

详细报告见 [RAG/evaluation_results.md](RAG/evaluation_results.md)。

### 模型训练效果

| 维度 | Qwen3-32B 原始 | SFT + DPO | 提升 |
|:---|:---:|:---:|:---:|
| 医学准确性（专家评分/5分）| 2.5 | **4.5** | +80% |
| 医疗安全性（专家评分/5分）| 1.5 | **4.8** | +220% |
| 综合均分 | 2.1 | **4.6** | ×2.2 |
| 禁忌识别率（DPO 增量）| 42% | **71%** | +69% |

### 检索层（RAGAS，210 题）

| 题型 | ContextRecall | ContextPrecision |
|:---|:---:|:---:|
| 库内证据型 | **0.929** | 0.751 |
| RAG增益对比型 | 0.841 | 0.591 |
| 高风险诱导型 | 0.440 | 0.356 |
| 泛化推理型 | 0.386 ⚠️ | 0.172 ⚠️ |

### 生成层（规则启发式，250 题）

| 风险等级 | Faithfulness | 幻觉率 | Recall@5 |
|:---|:---:|:---:|:---:|
| Low | 0.634 | **0.039** ✅ | 0.845 |
| Medium | 0.516 | 0.070 | 0.583 |
| High | 0.599 | 0.018 | 0.456 |

### 安全层（GPT 评判，250 题）

| 指标 | 数值 | 目标 | 状态 |
|:---|:---:|:---:|:---:|
| RAG vs Baseline 幻觉降低 | **66.7%** | ≥ 50% | ✅ |
| 安全拒答率（high 风险）| **100%** | ≥ 95% | ✅ |
| 免责声明率 | 93.2% | — | ✅ |
| 禁忌遗漏率 | 57.6% | ≤ 2% | ❌ |
| 溯源率 | 0% | 100% | ❌ |

---

## 已知局限

| 问题 | 现状 | 改进方向 |
|:---|:---|:---|
| 禁忌遗漏率偏高 | 57.6% | Reflection Loop 已新增禁忌覆盖审核规则 |
| 溯源率不足 | 0% | Prompt + Reflection 双重约束强制 `[证据N]` 行内标注 |
| 泛化推理型检索差 | Recall 38.6% | 知识库扩充 + 查询扩展策略 |
| AnswerRelevancy 偏低 | 0.324 | 已精简免责声明，减少对回答内容的稀释 |
| 中/高风险幻觉超标 | Medium 7.0%，High 1.8% | 加强检索质量控制，引入 Faithfulness 约束 |

---

## 项目结构

```
Medical-Qwen/
├── Medical-LLM/                    # 模型训练
│   ├── configs/                    # SFT / DPO / DeepSpeed 配置
│   └── dataset/                    # 数据处理脚本与训练数据
│
├── RAG/                            # 检索增强
│   ├── retriever.py                # Hybrid Search + RRF + Reranker
│   ├── ingest.py                   # 知识库入库
│   ├── data_acquisition/           # 多源数据采集（NMPA/NHC/WHO/PMC）
│   ├── data_processing/            # 评测脚本（run_eval.py / ragas_eval.py）
│   ├── data/                       # Milvus 向量库
│   └── evaluation_results.md       # 综合评测报告
│
├── agent/                          # 智能体应用
│   ├── core/impl.py                # MedicalAgentSystem（路由/RAG/Reflection/记忆）
│   ├── api/server.py               # FastAPI 后端
│   ├── agent_ui.py                 # Gradio 前端
│   ├── run_backend.py              # 后端启动入口
│   └── run_model.py                # vLLM 推理启动入口
│
└── assets/                         # 架构图等静态资源
```

---

## License

Apache 2.0

本项目仅供学术研究与技术交流使用，模型输出不构成任何医疗诊断或治疗建议。
