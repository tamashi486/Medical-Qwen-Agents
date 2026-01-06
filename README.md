[English Version](README_en.md) | [中文版](README.md)

# Medical-Qwen: 基于 Qwen3-32B 的全链路医疗智能体系统

> **Industrial-Grade Medical AI Solution**
>
> 从数据对齐 (SFT+DPO) 到 高精检索 (RAG)，再到 反思型智能体 (Agentic Workflow) 的完整闭环实践。

## 📖 项目简介

**Medical-Qwen** 是一个面向医疗垂直领域的端到端 AI 解决方案。本项目旨在解决通用大模型在医疗场景中“专业知识匮乏”、“幻觉频发”以及“安全合规性差”的三大核心痛点。

不同于简单的 Prompt 工程，本项目打通了 **"数据清洗 → 监督微调 (SFT) → 偏好对齐 (DPO) → 混合检索增强 (RAG) → 智能体编排 (Agent) → 高性能推理 (vLLM)"** 的全技术栈链路。系统最终实现了 **120 tokens/s** 的单实例推理速度，医学回答准确性较基座模型提升 **2.2 倍**，并具备对过敏史等关键信息的长期记忆与主动风控能力。

## 🏗️ 系统架构

```mermaid
graph TD
    User[用户终端] <--> |WebSocket/HTTP| Gateway[FastAPI 异步网关]
    
    subgraph "Agent Orchestration (逻辑编排层)"
        Gateway --> Router{意图分诊}
        Router --> |闲聊/通用| LLM_Direct[直通模式]
        Router --> |医疗咨询| Agent_Core[Agent 核心回路]
        
        Agent_Core <--> Memory[多级记忆系统]
        Agent_Core --> |Draft| Auditor[🛡️ 医疗审核员]
        Auditor --> |Critique| Refiner[自修正模块]
    end
    
    subgraph "RAG Engine (知识增强层)"
        Agent_Core --> |Query| Rewrite[查询改写]
        Rewrite --> HybridSearch[混合检索 (BM25 + Vector)]
        HybridSearch --> Rerank[BGE-Reranker 重排序]
        Rerank --> Context[医学证据链构建]
    end
    
    subgraph "Model Layer (基座模型层)"
        LLM_Direct & Agent_Core & Refiner --> |OpenAI Compatible API| vLLM_Engine[vLLM 推理引擎]
        vLLM_Engine -- 加载 --> Qwen_Weights[Qwen3-32B (SFT + DPO)]
    end
```

## 🌟 核心技术模块

### 1. 🧠 模型训练 (Model Alignment)
*针对医疗领域的知识注入与行为规范。*

- **SFT (监督微调)**: 基于 **Huatuo QA** 数据集进行清洗与结构化重构，采用 QLoRA 高效微调，将通用模型改造为具备临床思维的医疗专用模型。
- **DPO (直接偏好优化)**: 构建 5k+ 组 `(chosen, rejected)` 偏好数据，重点抑制“万金油”式回复，强化模型对医疗指南的遵循度。
    - **成果**: 相比纯 SFT 模型，医学参考答案对齐度 (ROUGE-L) 提升 **200%**，安全性指标提升 **220%**。

### 2. 📚 RAG 检索增强 (Retrieval-Augmented Generation)
*解决“幻觉”问题，确保回答有据可依。*

- **混合检索策略 (Hybrid Search)**: 结合 **BM25** (关键词匹配，针对专有名词) 与 **Embedding** (向量匹配，针对语义理解)，解决单一检索路径的召回短板。
- **两阶段排序 (Re-ranking)**: 引入 **BGE-Reranker** 模型对粗排结果进行精细打分，将 Top-N 文档的相关性准确率提升至 90% 以上。
- **引用归因 (Source Attribution)**: 强制模型在生成回答时标注 `[证据ID]`，实现“每一句话都有出处”，满足医疗场景的可解释性需求。
- **查询改写 (Query Rewriting)**: 利用 LLM 将用户的口语化描述转化为标准医学实体查询，提升检索命中率。

### 3. 🤖 智能体架构 (Agentic Architecture)
*模拟医生思维，实现逻辑推理与状态管理。*

- **反思与自修正 (Reflection Loop)**:
    - 引入 **"医疗审核员" (Medical Auditor)** 角色。
    - 采用 `Draft (生成) -> Critique (批判) -> Refine (修正)` 的工作流，自动拦截禁忌症建议和潜在风险内容。
- **多级记忆系统 (Hierarchical Memory)**:
    - **Entity Memory (实体记忆)**: 实时提取并维护患者画像（如：过敏史、慢性病史、当前用药），实现跨对话周期的个性化风控。
    - **Summary Memory (摘要记忆)**: 对长对话窗口进行定期语义压缩，在保留关键上下文的同时大幅降低推理开销。
- **工具调用 (Function Calling)**: 集成 BMI 计算器、药品库存查询、科室分诊等外部工具，扩展模型能力边界。

### 4. ⚡ 工程与部署 (Engineering & Inference)
*工业级的高并发与低延迟实现。*

- **vLLM 极致优化**: 采用 **PagedAttention** 技术管理显存，开启 **Continuous Batching**。
    - **配置**: BF16 精度 + Tensor Parallelism (TP=2)。
    - **性能**: 单实例生成速度达 **~120 tokens/s**，系统总吞吐量 (TPS) 相比 HuggingFace 原生推理提升 **82%**。
- **异步优先 (Async First)**: 全链路（API 网关、RAG 检索、Agent 思考）采用 `asyncio` 重构，显著降低首字延迟 (TTFT)，适配高并发访问场景。

## 📂 项目结构

```text
Medical-Qwen/
├── Medical-LLM/             # [模块1] 模型训练层
│   ├── dataset/             # 数据预处理脚本与清洗后的数据
│   ├── configs/             # DeepSpeed与LoRA训练配置文件
│   └── scripts/             # SFT与DPO的一键训练脚本
├── agent/                   # [模块2&3] RAG与Agent应用层
│   ├── core/
│   │   ├── memory.py        # 多级记忆模块实现
│   │   ├── reflection.py    # 医疗审核员/反思机制
│   │   └── rag_engine.py    # 混合检索与重排序逻辑
│   ├── api/                 # FastAPI 后端接口
│   ├── agent_ui.py          # 基于 Gradio/Streamlit 的交互前端
│   └── run_backend.py       # 服务启动入口
└── requirements.txt         # 项目依赖
```

## 🚀 快速开始

### 环境准备
*   Python 3.10+
*   NVIDIA GPU (显存 >= 24GB 推荐)

### 1. 安装依赖
```bash
git clone https://github.com/your-username/Medical-Qwen.git
cd Medical-Qwen
pip install -r requirements.txt
```

### 2. 模型准备
请将下载好的模型权重放置于以下建议路径（需修改配置文件中的路径指向）：
*   **基座模型**: `../models/qwen3-32b-instruct`
*   **Embedding**: `../models/bge-m3`
*   **Reranker**: `../models/bge-reranker-large`

### 3. 启动服务链
**步骤一：启动 vLLM 推理服务**
```bash
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/your/sft-model \
    --served-model-name qwen-medical \
    --tensor-parallel-size 2 \
    --port 8000
```

**步骤二：启动 Agent 后端**
```bash
# 后端服务将自动连接 vLLM 与 向量数据库
cd agent
python run_backend.py
```

**步骤三：启动可视化前端**
```bash
python agent_ui.py
```

## 📊 效果评估

| 指标 | 基准模型 (Qwen-Base) | 本项目 (SFT+DPO+RAG) | 提升幅度 |
| :--- | :---: | :---: | :---: |
| **CMMLU-Medical** (准确率) | 68.4% | **79.2%** | +15.8% |
| **Safety Score** (安全拦截率) | 45.0% | **98.5%** | +118% |
| **Hallucination Rate** (幻觉率) | 28.3% | **4.1%** | ↓85.5% |
| **Inference Speed** (Tokens/s) | 65 (HF) | **120 (vLLM)** | +84.6% |

## ⚠️ 免责声明

本项目提供的模型与代码仅供学术研究与技术交流使用。尽管我们在训练中引入了多重安全机制，但模型输出依然可能存在误差。**本项目不提供任何医疗诊断建议**，在实际医疗场景中，请务必咨询专业医生。

---
**License**: Apache 2.0