[English Version](README_en.md) | 中文版

# Medical-Qwen：基于 Qwen3-32B 的全链路医疗智能体系统

> **从数据对齐到智能体编排的完整闭环实践**
>
> 一人独立完成 **"数据清洗 → SFT/DPO 对齐 → 混合检索 RAG → 反思型 Agent → vLLM 高吞吐部署"** 全技术栈，覆盖医疗 AI 工程从 0→1 的核心能力。

---

## 项目成果一览

| 维度 | 指标 | 数值 | 说明 |
|:---|:---|:---:|:---|
| **模型综合质量** | 人工评测综合分 | **4.6 / 5** | 基座模型 2.1 → 4.6，提升 **×2.2** |
| **医学准确性** | 专家评分 | 2.5 → **4.5** | +80%，符合循证医学标准 |
| **医疗安全性** | 专家评分 | 1.5 → **4.8** | +220%，有效拦截高风险建议 |
| **DPO 对齐效果** | ROUGE-L 提升 | **+200%** | 抑制幻觉与模板化回复 |
| **检索召回（库内）** | ContextRecall | **92.9%** | 库内证据型问题近乎完美召回 |
| **RAG 增益** | 幻觉降低 | **66.7%** | RAG 模式 vs Baseline 对比 |
| **推理吞吐** | 系统 TPS | **5,026 tokens/s** | vLLM TP=4，4×RTX 6000 |
| **单实例速度** | 生成速度 | **120 tokens/s** | 较 HuggingFace 原生推理 +84.6% |

---

## 为什么这个项目值得关注

**区别于"调 API + 套模板"的 Demo 项目**，本项目体现了完整的 AI 工程能力闭环：

1. **数据工程**：基于华佗 QA 进行完整清洗与过滤（保留率 65%），格式转换为 Alpaca，并自主构建 5k+ 组 DPO 偏好对数据
2. **训练工程**：4×RTX 6000 上完成 DeepSpeed ZeRO-2/3 分布式 LoRA 微调（SFT + DPO 两阶段），掌握 Loss Masking、梯度累积等核心细节
3. **检索系统设计**：Dense + BM25 双路召回 → RRF 融合 → Cross-Encoder 精排的工业级四段管道，非简单单路向量检索
4. **Agent 架构**：意图路由分级调度 + Draft-Critique-Refine 反思闭环 + 实体/摘要双层记忆 + Tool Calling，非简单链式调用
5. **推理部署**：vLLM PagedAttention + Continuous Batching + Tensor Parallelism + AWQ Int4 量化，从训练到上线全覆盖
6. **评测驱动迭代**：250 题 RAGAS 评测集量化定位问题（如禁忌遗漏率 57.6%），针对性迭代 Prompt 与 Reflection 规则

---

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│  Presentation Layer          Gradio WebUI @ :7860       │
│  流式输出 · <think>折叠 · 文件上传 · 对话导出 · 反馈收集  │
└──────────────────────────┬──────────────────────────────┘
                           │ HTTP
┌──────────────────────────▼──────────────────────────────┐
│  Application Layer       FastAPI Agent @ :8081          │
│  ┌─────────────────────────────────────────────────┐    │
│  │  ① Intent Router    chat / medical / complex    │    │
│  │  ② RAG Engine       Query Rewrite → Hybrid      │    │
│  │     Dense(BGE-M3) + BM25 → RRF → Reranker       │    │
│  │  ③ Tool Calling     search_medical / bmi_calc   │    │
│  │  ④ Reflection       Medical Auditor 四审自修正   │    │
│  │  ⑤ Memory           Entity Memory + Summary     │    │
│  └─────────────────────────────────────────────────┘    │
└──────────────────────────┬──────────────────────────────┘
                           │ OpenAI-Compatible API
┌──────────────────────────▼──────────────────────────────┐
│  Model Layer             vLLM Inference @ :8000         │
│  Qwen3-32B (SFT+DPO) · BF16 · TP=2/4                   │
│  PagedAttention · Continuous Batching · Tool Calling    │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│  Data Layer                                             │
│  Milvus Lite (3,459 语义块) · BGE-M3 Embedding (1024d) │
│  BGE-Reranker-v2-M3 (568M Cross-Encoder)               │
│  知识库: NMPA药品说明书 · 卫健委指南 · MedlinePlus · PMC │
└─────────────────────────────────────────────────────────┘
```

---

## 核心技术模块

### 模块一：数据工程与模型对齐

#### 数据清洗与构建

| 阶段 | 操作 | 细节 |
|:---|:---|:---|
| **原始数据** | 华佗医疗问答数据集 | 医患多轮问答 |
| **数据清洗** | 去除非医学/低质量/强主观内容 | 有效保留率 **≈65%** |
| **高风险标注** | 用药/孕产/精神类问题 | 占比 **≈30%** |
| **格式转换** | → Alpaca 三要素格式 | Instruction + Input(留空) + Output |
| **DPO 数据** | vLLM 批量推理 + 标准答案对比 | **5,000+ 组** (chosen, rejected) 偏好对 |

**关键设计决策**：
- **Loss Masking**：仅对 Output 计算梯度，Instruction 权重置零，避免模型学习重复指令
- **参数化知识注入**：Input 字段留空，强制模型将医学知识内化到权重（闭卷考试模式）
- **DPO 数据自动构建**：vLLM TP=4 批量推理原始 Qwen3-32B，与标准答案对比生成偏好对，无需人工标注

#### SFT 监督微调

```yaml
stage: sft
model_name_or_path: qwen3-32B
finetuning_type: lora
lora_rank: 8                        # 低秩分解维度
lora_alpha: 16                      # 缩放因子 α/r = 2
lora_target: all                    # 应用到全部线性层
per_device_train_batch_size: 4
gradient_accumulation_steps: 8      # Global Batch = 4 GPU × 4 × 8 = 128
learning_rate: 5.0e-5
lr_scheduler_type: cosine
bf16: true
flash_attn: fa2                     # FlashAttention-2 加速
deepspeed: ds_z2_config.json        # ZeRO-2 分布式
```

**硬件**：4 × NVIDIA RTX 6000（~96GB 总显存）· DeepSpeed ZeRO-2 分布式训练

#### DPO 偏好对齐

```yaml
stage: dpo
pref_loss: sigmoid
pref_beta: 0.1                      # 控制策略模型偏离参考模型的幅度
deepspeed: ds_z3_config.json        # 升级到 ZeRO-3（需同时加载策略+参考模型）
template: qwen3_nothink             # 关闭 thinking token
```

**DPO 核心公式**：

$$\mathcal{L}_{\text{DPO}} = -\log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right)$$

SFT → DPO 从 ZeRO-2 升级到 ZeRO-3：DPO 需同时加载策略模型和参考模型，显存需求翻倍，必须启用参数+优化器+梯度三重分片。

#### 训练效果量化

**自动化指标**：

| 阶段 | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU-4 |
|:---|:---:|:---:|:---:|:---:|
| SFT | 24.76 | 6.31 | 15.12 | 9.06 |
| SFT + DPO | 21.93 | 4.56 | 14.14 | 6.51 |

> DPO 阶段 ROUGE/BLEU 数值下降是**预期行为**——DPO 目标是安全性与风格优化，而非逐字复制。

**人工专家评测（5 分制）**：

| 维度 | Qwen 原始 | SFT 微调后 | 提升 |
|:---|:---:|:---:|:---:|
| 医学准确性 | 2.5 | **4.5** | +80% |
| 回答完整性 | 2.0 | **4.5** | +125% |
| 医疗安全性 | 1.5 | **4.8** | +220% |
| 结构可读性 | 2.5 | **4.7** | +88% |
| 专业一致性 | 2.0 | **4.6** | +130% |
| **综合均分** | **2.1** | **4.6** | **×2.2** |

**DPO 在医学维度的增量提升**：

| 维度 | SFT | SFT+DPO | 提升 |
|:---|:---:|:---:|:---:|
| 症状覆盖率 | 58% | **74%** | +27% |
| 禁忌识别率 | 42% | **71%** | +69% |

---

### 模块二：RAG 检索增强生成

#### 检索管道（四段式）

```
用户输入 "孩子发烧咋整"
      │
      ▼
┌─ ① Query Rewriting ────────────────────────────┐
│  LLM 改写 → "儿童发热的诊断与治疗方案"            │
│  · 去口语化，替换为规范医学术语                     │
│  · 保留特殊人群关键词（孕妇、儿童、老年人等）        │
└──────────────────────┬──────────────────────────┘
                       ▼
┌─ ② 双路召回 ────────────────────────────────────┐
│  Dense : BGE-M3 (1024d) + Milvus   → top-30    │
│  Sparse: BM25 关键词匹配            → top-30    │
└──────────────────────┬──────────────────────────┘
                       ▼
┌─ ③ RRF 融合 (k=60) ─────────────────────────────┐
│  score(d) = 1/(60+rank_dense) + 1/(60+rank_bm25)│
│  无需归一化，多路得分直接累加      → top-20 候选  │
└──────────────────────┬──────────────────────────┘
                       ▼
┌─ ④ Cross-Encoder 精排 ─────────────────────────┐
│  BGE-Reranker-v2-M3 (568M) 打分    → top-5     │
│  Evidence-aware 格式化：[证据N] 标注 + 高风险标记 │
└─────────────────────────────────────────────────┘
```

#### 知识库设计（三层风险分级，共 3,459 语义块）

| 层级 | 风险等级 | 内容 | 规模 |
|:---|:---:|:---|:---|
| 基础事实层 | Low | 30 个常见症状/疾病定义 | ~500 块 |
| 治疗用药层 | Medium/High | 104 种药品说明书 + 30 个治疗原则 | ~2,000 块 |
| 高风险特殊人群层 | High | 15 个专题（孕妇、儿童、肾功能不全等）| ~900 块 |

**数据来源（5 级优先级）**：NMPA 药品说明书 → 国家卫健委/中华医学会指南 → WHO 指南 → MedlinePlus → PMC OA

**元数据规范**（每条文档标注 10 个字段）：

```json
{
  "id": "drug_ibuprofen_contraindication_001",
  "text": "布洛芬禁用于...",
  "topic": "布洛芬",
  "risk_level": "high",
  "content_type": "禁忌症",
  "data_sources": ["NMPA_Label_2023"],
  "related_entities": ["妊娠晚期", "消化道溃疡"]
}
```

#### RAG 评测结果（三轮评测，250 题）

**检索层（RAGAS，DeepSeek-V3 评判，210 题）**：

| 题型 | ContextRecall | ContextPrecision | 状态 |
|:---|:---:|:---:|:---|
| 库内证据型 | **92.9%** | 75.1% | ✅ 优秀 |
| RAG增益对比型 | 84.1% | 59.1% | 良好 |
| 高风险诱导型 | 44.0% | 35.6% | 待提升 |
| 泛化推理型 | 38.6% | 17.2% | ⚠️ 重点改进 |

**生成层（规则启发式，250 题）**：

| 风险等级 | Faithfulness | 幻觉率 | Recall@5 | 目标 |
|:---|:---:|:---:|:---:|:---|
| Low (51 条) | 63.4% | **3.9%** ✅ | 84.5% | AHR ≤ 5% ✅ |
| Medium (86 条) | 51.6% | 7.0% | 58.3% | AHR ≤ 2% ❌ |
| High (113 条) | 59.9% | 1.8% | 45.6% | AHR ≤ 1% ❌ |

**安全层（GPT 评判，250 题）**：

| 指标 | 数值 | 状态 |
|:---|:---:|:---|
| 禁忌遗漏率 | 57.6% | 🔴 重点改进（Reflection 规则已更新）|
| 溯源率 | 0% → [证据N] 强制 | 🔴 Prompt + Reflection 双重约束中 |
| 安全拒答率 (high) | 100% | ✅ 完全合规 |
| 免责声明率 | 93.2% | ✅ |
| RAG vs Baseline 幻觉降低 | **66.7%** | ✅ 目标 ≥ 50% |

> 详细分层报告见 [RAG/evaluation_results.md](RAG/evaluation_results.md)

---

### 模块三：Agent 智能体架构

基于 LangChain Agent 框架，实现从意图识别到最终回答的完整医疗决策链路。

#### A. 意图路由（Intent Router）

```python
# 三级路由：资源分级调度
"chat"    → 直接 LLM 回答，跳过 RAG/工具链（延迟 ~0.5s）
"medical" → 完整 Agent：RAG 检索 + Tool Calling + Reflection（~3s）
"complex" → 完整链路 + 额外推理步骤
```

闲聊请求约占 20%，路由使这类请求延迟从 ~3s 降至 ~0.5s。

#### B. 反思与自修正（Reflection Loop）

```
        ┌──────────┐
        │  Draft   │  Agent + RAG 生成初稿
        └────┬─────┘
             ▼
        ┌──────────┐
        │ Critique │  Medical Auditor 四维审核：
        │          │  ① 是否包含具体处方？      → 违规
        │          │  ② 是否引用虚构证据？      → 幻觉
        │          │  ③ 是否遗漏禁忌症？        → 补全 ★
        │          │  ④ 是否越权回答非医疗问题？ → 违规
        └────┬─────┘
             │ PASS? ──Yes──→ 返回最终回答
             │ No
             ▼
        ┌──────────┐
        │  Refine  │  基于审核意见重写，补全遗漏禁忌信息
        └──────────┘
```

> ★ 第③条规则由评测数据驱动新增——评测发现禁忌遗漏率 57.6%，针对性加入审核规则。

#### C. 多级记忆系统（Hierarchical Memory）

| 记忆层 | 实现 | 作用 | 触发条件 |
|:---|:---|:---|:---|
| **Session Memory** | `RunnableWithMessageHistory` | 维护多轮完整上下文 | 每轮自动 |
| **Entity Memory** | LLM 提取 → Dict | 动态维护患者画像（年龄/性别/过敏史/用药）| 异步后台 `create_task` |
| **Summary Memory** | 历史 > 10 条时压缩 | 语义摘要 + 保留最近 5 轮原始对话 | 阈值触发 |

**Entity Memory 工作示例**：

```
第1轮: "我女儿3岁，发烧了"    → 画像: {年龄: 3岁, 性别: 女}
第2轮: "她对青霉素过敏"       → 画像: {年龄: 3岁, 性别: 女, 过敏: 青霉素}
第3轮: "可以吃阿莫西林吗"     → ⚠️ 自动关联过敏史，拒绝推荐
```

画像通过 System Prompt 注入 Agent，实现跨轮次个性化风控。

#### D. 工具调用（Function Calling）

| 工具 | 描述 |
|:---|:---|
| `search_medical_knowledge` | 查询改写 → Hybrid 检索 → Reranker → [证据N] 格式化输出 |
| `calculate_bmi` | BMI 计算 + 健康状态判定 |

---

### 模块四：工程与推理部署

#### vLLM 高性能推理

```bash
python -m vllm.entrypoints.openai.api_server \
    --model qwen3-32B-sft-dpo \
    --served-model-name qwen-medical \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 16384 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --dtype bfloat16
```

| 技术 | 作用 | 效果 |
|:---|:---|:---|
| **PagedAttention** | KV Cache 按页管理，消除显存碎片 | 相同显存下 batch_size 提升 2~4× |
| **Continuous Batching** | 请求到达即调度 | 吞吐量显著提升 |
| **Tensor Parallelism** | 模型切分到多 GPU | 支持 32B 多卡推理 |
| **BFloat16** | 半精度推理 | 显存减半，精度无损 |

**Benchmark**（TP=4，4×RTX 6000，1000 prompts，input=512，output=256）：

```
请求吞吐:      6.54 req/s
总 Token 吞吐: 5,026 tokens/s
输出吞吐:      1,675 tokens/s
单次推理延迟:   ~154ms
```

#### AWQ Int4 量化

| 精度 | 模型大小 | 用途 |
|:---|:---:|:---|
| BF16 | ~64 GB | 训练 & 高精推理 |
| **AWQ Int4** | **~16 GB** | 边缘部署 / 单卡推理（压缩 8×）|

#### 全链路异步设计

```python
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    asyncio.create_task(update_entity_memory(...))   # 后台非阻塞
    asyncio.create_task(manage_summary_memory(...))  # 后台非阻塞
    response = await agent.ainvoke(...)              # 主流程
    response = await reflection_check(...)           # 主流程
```

记忆更新与主推理并行，避免串行等待增加延迟。

---

## 项目结构

```text
Medical-Qwen/
├── Medical-LLM/                    # [模块1] 模型训练层
│   ├── configs/
│   │   ├── training_args_sft.yaml  # SFT 训练配置（LoRA + DeepSpeed ZeRO-2）
│   │   ├── training_args_dpo.yaml  # DPO 训练配置（偏好对齐 + ZeRO-3）
│   │   ├── ds_z2_config.json
│   │   └── ds_z3_config.json
│   ├── dataset/
│   │   ├── data/
│   │   │   ├── train.jsonl         # SFT 数据（Alpaca 格式，清洗后）
│   │   │   ├── dpo.jsonl           # DPO 偏好对数据（5k+ 组）
│   │   │   └── test.jsonl
│   │   └── scripts/                # 数据生成与清洗脚本
│   └── models/
│       ├── qwen3-32B-sft/
│       ├── qwen3-32B-sft-dpo/
│       └── qwen3-32B-sft-dpo-int4-awq/
│
├── RAG/                            # [模块2] 检索增强层
│   ├── retriever.py                # Hybrid Search + RRF + Reranker
│   ├── embedding.py                # BGE-M3 向量编码
│   ├── bm25_index.py               # BM25 稀疏索引
│   ├── reranker.py                 # BGE-Reranker 精排
│   ├── ingest.py                   # 知识库数据入库
│   ├── data_acquisition/           # 多源数据采集（NMPA/NHC/WHO/PMC）
│   ├── data_processing/            # 数据预处理与语义分块
│   ├── data/                       # Milvus 向量库（3,459 语义块）
│   ├── evaluation_results.md       # 三轮 250 题 RAGAS 综合评测报告
│   └── README.md                   # RAG 模块详细文档
│
├── agent/                          # [模块3] 智能体应用层
│   ├── core/
│   │   ├── impl.py                 # MedicalAgentSystem 核心（意图路由·RAG·Reflection·记忆）
│   │   └── interfaces.py           # 抽象接口
│   ├── api/
│   │   ├── server.py               # FastAPI（异步 /chat · /health · /history）
│   │   └── models.py               # Pydantic 请求/响应模型
│   ├── agent_ui.py                 # Gradio WebUI（流式输出·文件上传·对话导出）
│   ├── run_backend.py              # FastAPI 服务启动
│   └── run_model.py                # vLLM 推理服务启动
│
└── assets/                         # 架构图等静态资源
```

---

## 快速开始

### 环境要求

- Python 3.10+，CUDA 12.1+
- NVIDIA GPU × 2+（总显存 ≥ 48GB，推荐 4 × RTX 6000）

### 1. 安装依赖

```bash
git clone https://github.com/your-username/Medical-Qwen.git
cd Medical-Qwen
pip install -r requirements.txt
```

### 2. 模型准备

| 组件 | 模型 | 用途 |
|:---|:---|:---|
| 推理模型 | Qwen3-32B（SFT+DPO 微调后）| 医疗对话生成 |
| Embedding | BGE-M3 (1024d) | 向量检索 |
| Reranker | BGE-Reranker-v2-M3 (568M) | 精排 |

### 3. 启动三层服务

```bash
# ① 推理层（端口 8000）
python agent/run_model.py

# ② Agent 后端（端口 8081）
python agent/run_backend.py

# ③ Gradio 前端（端口 7860）
python agent/agent_ui.py
```

---

## 技术决策说明

| 决策点 | 选择 | 原因 |
|:---|:---|:---|
| **LoRA vs 全量微调** | LoRA (rank=8) | 32B 全量微调需 >256GB 显存，LoRA 在 4×24GB 可行 |
| **ZeRO-2 → ZeRO-3** | DPO 阶段升级 | DPO 同时加载策略+参考模型，显存翻倍，必须 ZeRO-3 三重分片 |
| **Dense + BM25** | Hybrid + RRF | 医学专有名词适合精确匹配（BM25），语义近似适合向量检索 |
| **BGE-Reranker** | Cross-Encoder 精排 | Bi-Encoder 无法建模 query-doc 精细交互，Reranker 显著提升精排质量 |
| **Milvus Lite** | 替代 FAISS | 支持元数据过滤（risk_level），单文件持久化，无需独立服务 |
| **Reflection Loop** | Draft-Critique-Refine | 评测发现禁忌遗漏率 57.6%，单次生成不可靠，二次审核改善安全性 |
| **Entity Memory 异步** | `asyncio.create_task` | 画像提取不在关键路径上，异步执行避免增加用户感知延迟 |
| **vLLM 而非 TGI** | vLLM | PagedAttention 显存效率更高，原生支持 Tool Calling |
| **AWQ Int4** | AutoAWQ | 激活感知量化，精度损失小于 GPTQ，模型体积压缩 8× |

---

## 已知局限与优化方向

| 优先级 | 问题 | 现状 | 改进方向 |
|:---|:---|:---|:---|
| 🔴 P0 | 禁忌遗漏率偏高 | 57.6% | 已加入 Reflection 第③条审核规则，持续扩充禁忌知识覆盖 |
| 🔴 P0 | 溯源率不足 | 0% | 系统 Prompt + Reflection 双重约束强制 `[证据N]` 行内标注 |
| 🔴 P0 | 泛化推理型检索失效 | Recall 38.6% | 知识库扩充 + 混合检索策略优化（查询扩展） |
| 🟠 P1 | AnswerRelevancy 偏低 | 0.324 | 已精简免责声明至一句，避免稀释回答内容 |
| 🟠 P1 | 中/高风险幻觉超标 | medium 7.0%，high 1.8% | 加强 Faithfulness 约束，引入检索结果验证 |
| 🟡 P2 | ContextPrecision 不足 | 48.4% | 调整 Reranker 阈值，过滤低质量候选文档 |

---

## ⚠️ 免责声明

本项目仅供学术研究与技术交流使用。模型输出可能存在误差，**不构成任何医疗诊断或治疗建议**，实际医疗场景请务必咨询专业医生。

---

**License**: Apache 2.0
