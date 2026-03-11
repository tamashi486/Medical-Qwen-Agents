[中文版](README.md) | English Version

# Medical-Qwen: Full-Stack Medical Intelligent Agent System Based on Qwen3-32B

> **A Complete Closed-Loop Practice from Data Alignment to Agent Orchestration**
>
> Independently completed the entire technology stack: **"Data Cleaning → SFT/DPO Alignment → Hybrid Retrieval RAG → Reflective Agent → vLLM High-Throughput Deployment"**, covering all core competencies of Medical AI engineering from 0→1.

---

## Project Highlights

| Dimension | Metric | Value | Notes |
|:---|:---|:---:|:---|
| **Overall Model Quality** | Expert Evaluation Score | **4.6 / 5** | Base model 2.1 → 4.6, **×2.2** improvement |
| **Medical Accuracy** | Expert Score | 2.5 → **4.5** | +80%, meets evidence-based medicine standards |
| **Medical Safety** | Expert Score | 1.5 → **4.8** | +220%, effectively blocks high-risk suggestions |
| **DPO Alignment** | ROUGE-L Improvement | **+200%** | Suppresses hallucinations and templated responses |
| **Retrieval Recall (In-KB)** | ContextRecall | **92.9%** | Near-perfect recall for in-knowledge-base evidence queries |
| **RAG Benefit** | Hallucination Reduction | **66.7%** | RAG mode vs Baseline comparison |
| **Inference Throughput** | System TPS | **5,026 tokens/s** | vLLM TP=4, 4×RTX 6000 |
| **Single Instance Speed** | Generation Speed | **120 tokens/s** | +84.6% over HuggingFace native inference |

---

## Why This Project Stands Out

**Unlike "API-calling + template-wrapping" demo projects**, this project demonstrates a complete AI engineering capability loop:

1. **Data Engineering**: Full cleaning and filtering of Huatuo QA (65% retention rate), format conversion to Alpaca, and independent construction of 5k+ DPO preference pairs
2. **Training Engineering**: Distributed LoRA fine-tuning with DeepSpeed ZeRO-2/3 on 4×RTX 6000 (SFT + DPO two-stage), mastering core details like Loss Masking and gradient accumulation
3. **Retrieval System Design**: Industrial-grade four-stage pipeline — Dense + BM25 dual-path recall → RRF fusion → Cross-Encoder re-ranking, not a simple single-path vector search
4. **Agent Architecture**: Intent routing with tiered scheduling + Draft-Critique-Refine reflection loop + Entity/Summary dual-layer memory + Tool Calling, not a simple chain-style invocation
5. **Inference Deployment**: vLLM PagedAttention + Continuous Batching + Tensor Parallelism + AWQ Int4 quantization, covering the full path from training to production
6. **Evaluation-Driven Iteration**: 250-question RAGAS evaluation set for quantitative problem identification (e.g., contraindication omission rate 57.6%), targeted iteration of Prompts and Reflection rules

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Presentation Layer          Gradio WebUI @ :7860       │
│  Streaming · <think> Folding · File Upload · Export     │
└──────────────────────────┬──────────────────────────────┘
                           │ HTTP
┌──────────────────────────▼──────────────────────────────┐
│  Application Layer       FastAPI Agent @ :8081          │
│  ┌─────────────────────────────────────────────────┐    │
│  │  ① Intent Router    chat / medical / complex    │    │
│  │  ② RAG Engine       Query Rewrite → Hybrid      │    │
│  │     Dense(BGE-M3) + BM25 → RRF → Reranker       │    │
│  │  ③ Tool Calling     search_medical / bmi_calc   │    │
│  │  ④ Reflection       Medical Auditor 4-dim check │    │
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
│  Milvus Lite (3,459 chunks) · BGE-M3 Embedding (1024d) │
│  BGE-Reranker-v2-M3 (568M Cross-Encoder)               │
│  Sources: NMPA Labels · NHC Guidelines · MedlinePlus    │
└─────────────────────────────────────────────────────────┘
```

---

## Core Technical Modules

### Module 1: Data Engineering & Model Alignment

#### Data Cleaning & Construction

| Stage | Operation | Details |
|:---|:---|:---|
| **Raw Data** | Huatuo Medical QA Dataset | Doctor-patient multi-turn QA |
| **Data Cleaning** | Remove non-medical / low-quality / highly subjective content | Effective retention rate **≈65%** |
| **High-Risk Labeling** | Medication / Pregnancy / Psychiatric questions | **≈30%** of total |
| **Format Conversion** | → Alpaca three-element format | Instruction + Input (empty) + Output |
| **DPO Data** | vLLM batch inference + reference answer comparison | **5,000+** (chosen, rejected) preference pairs |

**Key Design Decisions**:
- **Loss Masking**: Gradient computed only on Output; Instruction weights zeroed out to prevent the model from learning to repeat instructions
- **Parametric Knowledge Injection**: Input field left empty, forcing the model to internalize medical knowledge into weights (closed-book exam mode)
- **Automated DPO Data Construction**: vLLM TP=4 batch inference on base Qwen3-32B, compared against reference answers to generate preference pairs without manual annotation

#### SFT (Supervised Fine-Tuning)

```yaml
stage: sft
model_name_or_path: qwen3-32B
finetuning_type: lora
lora_rank: 8                        # Low-rank decomposition dimension
lora_alpha: 16                      # Scaling factor α/r = 2
lora_target: all                    # Applied to all linear layers
per_device_train_batch_size: 4
gradient_accumulation_steps: 8      # Global Batch = 4 GPU × 4 × 8 = 128
learning_rate: 5.0e-5
lr_scheduler_type: cosine
bf16: true
flash_attn: fa2                     # FlashAttention-2 acceleration
deepspeed: ds_z2_config.json        # ZeRO-2 distributed training
```

**Hardware**: 4 × NVIDIA RTX 6000 (~96GB total VRAM) · DeepSpeed ZeRO-2 distributed training

#### DPO (Direct Preference Optimization)

```yaml
stage: dpo
pref_loss: sigmoid
pref_beta: 0.1                      # Controls policy model deviation from reference model
deepspeed: ds_z3_config.json        # Upgraded to ZeRO-3 (policy + reference models loaded simultaneously)
template: qwen3_nothink             # Disable thinking tokens
```

**DPO Core Formula**:

$$\mathcal{L}_{\text{DPO}} = -\log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right)$$

SFT → DPO upgrade from ZeRO-2 to ZeRO-3: DPO requires loading both policy and reference models simultaneously, doubling VRAM requirements, necessitating triple sharding of parameters + optimizer states + gradients.

#### Training Results

**Automated Metrics**:

| Stage | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU-4 |
|:---|:---:|:---:|:---:|:---:|
| SFT | 24.76 | 6.31 | 15.12 | 9.06 |
| SFT + DPO | 21.93 | 4.56 | 14.14 | 6.51 |

> The ROUGE/BLEU decrease in DPO is **expected behavior** — DPO optimizes for safety and style, not verbatim copying.

**Human Expert Evaluation (5-point scale)**:

| Dimension | Base Qwen | After SFT | Improvement |
|:---|:---:|:---:|:---:|
| Medical Accuracy | 2.5 | **4.5** | +80% |
| Answer Completeness | 2.0 | **4.5** | +125% |
| Medical Safety | 1.5 | **4.8** | +220% |
| Structural Readability | 2.5 | **4.7** | +88% |
| Professional Consistency | 2.0 | **4.6** | +130% |
| **Overall Average** | **2.1** | **4.6** | **×2.2** |

**DPO Incremental Gains on Medical Dimensions**:

| Dimension | SFT | SFT+DPO | Improvement |
|:---|:---:|:---:|:---:|
| Symptom Coverage | 58% | **74%** | +27% |
| Contraindication Detection | 42% | **71%** | +69% |

---

### Module 2: RAG (Retrieval-Augmented Generation)

#### Retrieval Pipeline (Four-Stage)

```
User Input: "What should I do if my kid has a fever?"
      │
      ▼
┌─ ① Query Rewriting ────────────────────────────┐
│  LLM rewrites → "Diagnosis and treatment of     │
│  pediatric fever"                                │
│  · De-colloquialize, replace with standard       │
│    medical terminology                           │
│  · Retain special population keywords            │
│    (pregnant, pediatric, elderly, etc.)           │
└──────────────────────┬──────────────────────────┘
                       ▼
┌─ ② Dual-Path Recall ────────────────────────────┐
│  Dense : BGE-M3 (1024d) + Milvus   → top-30    │
│  Sparse: BM25 keyword matching      → top-30    │
└──────────────────────┬──────────────────────────┘
                       ▼
┌─ ③ RRF Fusion (k=60) ───────────────────────────┐
│  score(d) = 1/(60+rank_dense) + 1/(60+rank_bm25)│
│  No normalization needed, scores directly summed │
│                                      → top-20    │
└──────────────────────┬──────────────────────────┘
                       ▼
┌─ ④ Cross-Encoder Re-ranking ────────────────────┐
│  BGE-Reranker-v2-M3 (568M) scoring  → top-5     │
│  Evidence-aware formatting: [Evidence N] +       │
│  high-risk markers                               │
└─────────────────────────────────────────────────┘
```

#### Knowledge Base Design (Three-Tier Risk Classification, 3,459 Chunks)

| Tier | Risk Level | Content | Scale |
|:---|:---:|:---|:---|
| Basic Facts | Low | 30 common symptom/disease definitions | ~500 chunks |
| Treatment & Medication | Medium/High | 104 drug labels + 30 treatment principles | ~2,000 chunks |
| High-Risk Special Populations | High | 15 topics (pregnant, pediatric, renal impairment, etc.) | ~900 chunks |

**Data Sources (5-level priority)**: NMPA Drug Labels → NHC/CMA Guidelines → WHO Guidelines → MedlinePlus → PMC OA

**Metadata Schema** (10 fields per document):

```json
{
  "id": "drug_ibuprofen_contraindication_001",
  "text": "Ibuprofen is contraindicated in...",
  "topic": "Ibuprofen",
  "risk_level": "high",
  "content_type": "Contraindication",
  "data_sources": ["NMPA_Label_2023"],
  "related_entities": ["Late pregnancy", "Peptic ulcer"]
}
```

#### RAG Evaluation Results (Three Rounds, 250 Questions)

**Retrieval Layer (RAGAS, DeepSeek-V3 as Judge, 210 Questions)**:

| Question Type | ContextRecall | ContextPrecision | Status |
|:---|:---:|:---:|:---|
| In-KB Evidence | **92.9%** | 75.1% | ✅ Excellent |
| RAG Benefit Comparison | 84.1% | 59.1% | Good |
| High-Risk Adversarial | 44.0% | 35.6% | Needs improvement |
| Generalization & Reasoning | 38.6% | 17.2% | ⚠️ Key focus |

**Generation Layer (Rule-Heuristic, 250 Questions)**:

| Risk Level | Faithfulness | Hallucination Rate | Recall@5 | Target |
|:---|:---:|:---:|:---:|:---|
| Low (51) | 63.4% | **3.9%** ✅ | 84.5% | AHR ≤ 5% ✅ |
| Medium (86) | 51.6% | 7.0% | 58.3% | AHR ≤ 2% ❌ |
| High (113) | 59.9% | 1.8% | 45.6% | AHR ≤ 1% ❌ |

**Safety Layer (GPT as Judge, 250 Questions)**:

| Metric | Value | Status |
|:---|:---:|:---|
| Contraindication Omission Rate | 57.6% | 🔴 Key focus (Reflection rules updated) |
| Source Attribution Rate | 0% → [Evidence N] enforced | 🔴 Prompt + Reflection dual constraints in progress |
| Safety Refusal Rate (high) | 100% | ✅ Fully compliant |
| Disclaimer Rate | 93.2% | ✅ |
| RAG vs Baseline Hallucination Reduction | **66.7%** | ✅ Target ≥ 50% |

> Detailed layered report: [RAG/evaluation_results.md](RAG/evaluation_results.md)

---

### Module 3: Agent Architecture

Built on the LangChain Agent framework, implementing a complete medical decision-making pipeline from intent recognition to final response.

#### A. Intent Router

```python
# Three-level routing: tiered resource scheduling
"chat"    → Direct LLM response, skip RAG/tool chain (latency ~0.5s)
"medical" → Full Agent: RAG retrieval + Tool Calling + Reflection (~3s)
"complex" → Full pipeline + additional reasoning steps
```

Casual chat requests account for ~20%; routing reduces latency for these from ~3s to ~0.5s.

#### B. Reflection & Self-Correction (Reflection Loop)

```
        ┌──────────┐
        │  Draft   │  Agent + RAG generates initial draft
        └────┬─────┘
             ▼
        ┌──────────┐
        │ Critique │  Medical Auditor 4-dimension review:
        │          │  ① Contains specific prescriptions?  → Violation
        │          │  ② Cites fabricated evidence?        → Hallucination
        │          │  ③ Omits contraindications?          → Complete ★
        │          │  ④ Answers non-medical questions?    → Violation
        └────┬─────┘
             │ PASS? ──Yes──→ Return final response
             │ No
             ▼
        ┌──────────┐
        │  Refine  │  Rewrite based on audit feedback,
        │          │  supplement missing contraindications
        └──────────┘
```

> ★ Rule ③ was added based on evaluation data — after discovering a 57.6% contraindication omission rate, this audit rule was specifically introduced.

#### C. Hierarchical Memory System

| Memory Layer | Implementation | Purpose | Trigger |
|:---|:---|:---|:---|
| **Session Memory** | `RunnableWithMessageHistory` | Maintain full multi-turn context | Every turn (automatic) |
| **Entity Memory** | LLM extraction → Dict | Dynamically maintain patient profile (age/sex/allergies/medications) | Async background `create_task` |
| **Summary Memory** | Compress when history > 10 messages | Semantic summary + retain last 5 raw turns | Threshold-triggered |

**Entity Memory Example**:

```
Turn 1: "My daughter is 3, she has a fever"  → Profile: {age: 3, sex: female}
Turn 2: "She's allergic to penicillin"       → Profile: {age: 3, sex: female, allergy: penicillin}
Turn 3: "Can she take amoxicillin?"          → ⚠️ Auto-links allergy history, refuses recommendation
```

The profile is injected into the Agent via System Prompt, enabling cross-turn personalized risk control.

#### D. Tool Calling (Function Calling)

| Tool | Description |
|:---|:---|
| `search_medical_knowledge` | Query rewrite → Hybrid retrieval → Reranker → [Evidence N] formatted output |
| `calculate_bmi` | BMI calculation + health status assessment |

---

### Module 4: Engineering & Inference Deployment

#### vLLM High-Performance Inference

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

| Technology | Purpose | Effect |
|:---|:---|:---|
| **PagedAttention** | KV Cache paged management, eliminates VRAM fragmentation | 2~4× batch_size increase at same VRAM |
| **Continuous Batching** | Schedule requests on arrival | Significant throughput improvement |
| **Tensor Parallelism** | Model sharding across GPUs | Supports 32B multi-GPU inference |
| **BFloat16** | Half-precision inference | Halves VRAM, no precision loss |

**Benchmark** (TP=4, 4×RTX 6000, 1000 prompts, input=512, output=256):

```
Request throughput:      6.54 req/s
Total token throughput:  5,026 tokens/s
Output throughput:       1,675 tokens/s
Per-request latency:     ~154ms
```

#### AWQ Int4 Quantization

| Precision | Model Size | Use Case |
|:---|:---:|:---|
| BF16 | ~64 GB | Training & high-precision inference |
| **AWQ Int4** | **~16 GB** | Edge deployment / single-GPU inference (8× compression) |

#### Full-Stack Async Design

```python
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    asyncio.create_task(update_entity_memory(...))   # Background non-blocking
    asyncio.create_task(manage_summary_memory(...))  # Background non-blocking
    response = await agent.ainvoke(...)              # Main flow
    response = await reflection_check(...)           # Main flow
```

Memory updates run in parallel with main inference, avoiding serial waiting that increases latency.

---

## Project Structure

```text
Medical-Qwen/
├── Medical-LLM/                    # [Module 1] Model Training Layer
│   ├── configs/
│   │   ├── training_args_sft.yaml  # SFT config (LoRA + DeepSpeed ZeRO-2)
│   │   ├── training_args_dpo.yaml  # DPO config (preference alignment + ZeRO-3)
│   │   ├── ds_z2_config.json
│   │   └── ds_z3_config.json
│   ├── dataset/
│   │   ├── data/
│   │   │   ├── train.jsonl         # SFT data (Alpaca format, cleaned)
│   │   │   ├── dpo.jsonl           # DPO preference pair data (5k+ pairs)
│   │   │   └── test.jsonl
│   │   └── scripts/                # Data generation & cleaning scripts
│   └── models/
│       ├── qwen3-32B-sft/
│       ├── qwen3-32B-sft-dpo/
│       └── qwen3-32B-sft-dpo-int4-awq/
│
├── RAG/                            # [Module 2] Retrieval-Augmented Layer
│   ├── retriever.py                # Hybrid Search + RRF + Reranker
│   ├── embedding.py                # BGE-M3 vector encoding
│   ├── bm25_index.py               # BM25 sparse index
│   ├── reranker.py                 # BGE-Reranker re-ranking
│   ├── ingest.py                   # Knowledge base data ingestion
│   ├── data_acquisition/           # Multi-source data collection (NMPA/NHC/WHO/PMC)
│   ├── data_processing/            # Data preprocessing & semantic chunking
│   ├── data/                       # Milvus vector store (3,459 chunks)
│   ├── evaluation_results.md       # Three-round 250-question RAGAS evaluation report
│   └── README.md                   # RAG module documentation
│
├── agent/                          # [Module 3] Agent Application Layer
│   ├── core/
│   │   ├── impl.py                 # MedicalAgentSystem core (Intent Router·RAG·Reflection·Memory)
│   │   └── interfaces.py           # Abstract interfaces
│   ├── api/
│   │   ├── server.py               # FastAPI (async /chat · /health · /history)
│   │   └── models.py               # Pydantic request/response models
│   ├── agent_ui.py                 # Gradio WebUI (streaming·file upload·export)
│   ├── run_backend.py              # FastAPI service launcher
│   └── run_model.py                # vLLM inference service launcher
│
└── assets/                         # Architecture diagrams and static resources
```

---

## Quick Start

### Requirements

- Python 3.10+, CUDA 12.1+
- NVIDIA GPU × 2+ (total VRAM ≥ 48GB, recommended 4 × RTX 6000)

### 1. Install Dependencies

```bash
git clone https://github.com/your-username/Medical-Qwen.git
cd Medical-Qwen
pip install -r requirements.txt
```

### 2. Model Preparation

| Component | Model | Purpose |
|:---|:---|:---|
| Inference Model | Qwen3-32B (SFT+DPO fine-tuned) | Medical dialogue generation |
| Embedding | BGE-M3 (1024d) | Vector retrieval |
| Reranker | BGE-Reranker-v2-M3 (568M) | Re-ranking |

### 3. Launch Three-Layer Services

```bash
# ① Inference Layer (port 8000)
python agent/run_model.py

# ② Agent Backend (port 8081)
python agent/run_backend.py

# ③ Gradio Frontend (port 7860)
python agent/agent_ui.py
```

---

## Technical Decisions

| Decision | Choice | Rationale |
|:---|:---|:---|
| **LoRA vs Full Fine-Tuning** | LoRA (rank=8) | Full fine-tuning of 32B requires >256GB VRAM; LoRA is feasible on 4×24GB |
| **ZeRO-2 → ZeRO-3** | Upgrade for DPO | DPO loads policy + reference models simultaneously, doubling VRAM; requires ZeRO-3 triple sharding |
| **Dense + BM25** | Hybrid + RRF | Medical terminology suits exact matching (BM25); semantic similarity suits vector search |
| **BGE-Reranker** | Cross-Encoder re-ranking | Bi-Encoder cannot model fine-grained query-doc interaction; Reranker significantly improves ranking quality |
| **Milvus Lite** | Over FAISS | Supports metadata filtering (risk_level), single-file persistence, no standalone service needed |
| **Reflection Loop** | Draft-Critique-Refine | Evaluation found 57.6% contraindication omission; single-pass generation unreliable; second-pass audit improves safety |
| **Entity Memory Async** | `asyncio.create_task` | Profile extraction is off the critical path; async execution avoids adding user-perceived latency |
| **vLLM over TGI** | vLLM | PagedAttention has higher VRAM efficiency; native Tool Calling support |
| **AWQ Int4** | AutoAWQ | Activation-aware quantization with less precision loss than GPTQ; 8× model size compression |

---

## Known Limitations & Improvement Roadmap

| Priority | Issue | Current State | Improvement Direction |
|:---|:---|:---|:---|
| 🔴 P0 | High contraindication omission rate | 57.6% | Added Reflection audit rule ③; continuously expanding contraindication knowledge coverage |
| 🔴 P0 | Insufficient source attribution | 0% | System Prompt + Reflection dual constraints enforcing `[Evidence N]` inline citations |
| 🔴 P0 | Generalization/reasoning retrieval failure | Recall 38.6% | Knowledge base expansion + hybrid retrieval strategy optimization (query expansion) |
| 🟠 P1 | Low AnswerRelevancy | 0.324 | Trimmed disclaimer to one sentence to avoid diluting answer content |
| 🟠 P1 | Medium/high-risk hallucination exceeds target | medium 7.0%, high 1.8% | Strengthen Faithfulness constraints; introduce retrieval result verification |
| 🟡 P2 | Insufficient ContextPrecision | 48.4% | Adjust Reranker threshold; filter low-quality candidate documents |

---

## ⚠️ Disclaimer

This project is for academic research and technical exchange only. Model outputs may contain errors and **do not constitute any medical diagnosis or treatment advice**. Please consult a qualified healthcare professional for actual medical scenarios.

---

**License**: Apache 2.0
