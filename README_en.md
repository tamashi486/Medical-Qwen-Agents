# Medical-Qwen

A medical-domain LLM system based on Qwen3-32B, covering the full pipeline: **SFT/DPO Alignment → Hybrid Retrieval RAG → Reflective Agent → vLLM High-Performance Inference**.

[English](README_en.md) | [中文](README.md)

---

## Table of Contents

- [Introduction](#introduction)
- [System Architecture](#system-architecture)
- [Quick Start](#quick-start)
- [Modules](#modules)
- [Evaluation Results](#evaluation-results)
- [Known Limitations](#known-limitations)

---

## Introduction

Medical-Qwen addresses three core issues of general LLMs in medical scenarios:

| Problem | Solution |
|:---|:---|
| Lack of domain knowledge | SFT fine-tuning + DPO preference alignment on Huatuo QA dataset |
| Frequent hallucinations | Hybrid RAG (Dense + BM25 + Reranker) + Reflection self-correction |
| Poor safety compliance | Three-tier risk-classified knowledge base + mandatory contraindication coverage + safety refusal decisions |

---

## System Architecture

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
│  Milvus Lite · 3,459 chunks · BGE-M3 (1024d)           │
│  Sources: NMPA Labels / NHC Guidelines / MedlinePlus    │
└─────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Requirements

- Python 3.10+, CUDA 12.1+
- NVIDIA GPU × 2+ (total VRAM ≥ 48GB)

### Installation

```bash
git clone https://github.com/your-username/Medical-Qwen.git
cd Medical-Qwen
pip install -r requirements.txt
```

### Model Preparation

| Component | Model |
|:---|:---|
| Inference Model | Qwen3-32B (SFT+DPO fine-tuned) |
| Embedding | `BAAI/bge-m3` |
| Reranker | `BAAI/bge-reranker-v2-m3` |

Modify the path configuration in `agent/api/server.py`:

```python
CONFIG = {
    "db_path": "/path/to/RAG/data/milvus.db",
    "embedding_model_path": "/path/to/bge-m3",
    "vllm_api_base": "http://localhost:8000/v1",
    "model_name": "qwen-medical"
}
```

### Launch Services

```bash
# Step 1: Start vLLM inference service (port 8000)
python agent/run_model.py

# Step 2: Start Agent backend (port 8081)
python agent/run_backend.py

# Step 3: Start Gradio frontend (port 7860)
python agent/agent_ui.py
```

Visit `http://localhost:7860` to use the system.

### Build RAG Knowledge Base

```bash
# Collect knowledge source data
python -m RAG.data_acquisition.main --all

# Data preprocessing and chunking
python -m RAG.data_processing.preprocess

# Vector ingestion
python -m RAG.ingest
```

---

## Modules

### Module 1: Model Alignment (SFT + DPO)

**Data Construction**: Based on Huatuo Medical QA dataset, ~65% retention after cleaning, automatically constructed 5,000+ DPO preference pairs (chosen/rejected).

**SFT Training**:

```yaml
finetuning_type: lora
lora_rank: 8
per_device_train_batch_size: 4
gradient_accumulation_steps: 8     # Global Batch = 128
learning_rate: 5.0e-5
deepspeed: ds_z2_config.json       # ZeRO-2
```

**DPO Training**: Continues training on top of the SFT model, upgraded from ZeRO-2 to ZeRO-3 (loading both policy + reference models simultaneously doubles VRAM requirements).

```yaml
pref_loss: sigmoid
pref_beta: 0.1
deepspeed: ds_z3_config.json
```

---

### Module 2: RAG Retrieval-Augmented Generation

**Four-Stage Retrieval Pipeline**:

```
User Input
    │
    ▼  ① Query Rewriting
    │  LLM rewrites → standardized medical terms, retains special population keywords
    │
    ▼  ② Dual-Path Recall
    │  Dense : BGE-M3 + Milvus → top-30
    │  Sparse: BM25             → top-30
    │
    ▼  ③ RRF Fusion (k=60)
    │  score = 1/(60+rank_dense) + 1/(60+rank_bm25) → top-20
    │
    ▼  ④ Cross-Encoder Re-ranking
       BGE-Reranker-v2-M3 → top-5, [Evidence N] formatted output
```

**Knowledge Base Structure** (3,459 chunks, three-tier risk classification):

| Tier | Risk Level | Content | Scale |
|:---|:---:|:---|:---|
| Basic Facts | Low | 30 common symptoms/diseases | ~500 chunks |
| Treatment & Medication | Medium/High | 104 drug labels + 30 treatment principles | ~2,000 chunks |
| High-Risk Populations | High | 15 topics (pregnant, pediatric, renal impairment, etc.) | ~900 chunks |

**Data Source Priority**: NMPA Drug Labels → NHC/CMA Guidelines → WHO Guidelines → MedlinePlus → PMC OA

---

### Module 3: Agent

**Intent Router**:

```
chat    → Direct LLM response (skip RAG, latency ~0.5s)
medical → Full Agent pipeline (RAG + Tool + Reflection, ~3s)
complex → Full pipeline + additional reasoning steps
```

**Reflection Loop**:

```
Draft (generate initial response)
    │
    ▼
Critique (Medical Auditor 4-dimension review)
    ① Contains specific prescription advice?
    ② Cites non-existent evidence (hallucination)?
    ③ Omits contraindications / prohibited populations / drug interactions?
    ④ Answers non-medical questions beyond scope?
    │
    ├── PASS → Return final response
    │
    └── FAIL → Refine (rewrite, supplement missing contraindication info)
```

**Hierarchical Memory System**:

| Layer | Implementation | Purpose |
|:---|:---|:---|
| Session Memory | `RunnableWithMessageHistory` | Maintain multi-turn dialogue context |
| Entity Memory | LLM extraction → Dict (async background update) | Patient profile (allergies/medications/age) |
| Summary Memory | LLM compression when history > 10 messages | Reduce long-dialogue inference overhead |

Entity Memory Example:

```
Turn 1: "My 3-year-old daughter has a fever"  → {age: 3, sex: female}
Turn 2: "She's allergic to penicillin"        → {allergy: penicillin}
Turn 3: "Can she take amoxicillin?"           → ⚠️ Auto-links allergy history, refuses recommendation
```

---

### Module 4: Inference Deployment

**vLLM Launch**:

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

**Quantization Options**:

| Precision | Model Size | Use Case |
|:---|:---:|:---|
| BF16 | ~64 GB | High-precision inference |
| AWQ Int4 | ~16 GB | Single-GPU / edge deployment |

**Benchmark** (TP=4, 4×RTX 6000, input=512, output=256):

```
Request throughput:      6.54 req/s
Total token throughput:  5,026 tokens/s
Per-request latency:     ~154ms
```

---

## Evaluation Results

Evaluation set: 250 questions, 5 question types (in-KB evidence / generalization reasoning / high-risk adversarial / out-of-scope unanswerable / RAG benefit comparison).
Three evaluation methods: rule-heuristic / GPT-as-judge / RAGAS semantic evaluation (DeepSeek-V3 as Judge).

Detailed report: [RAG/evaluation_results.md](RAG/evaluation_results.md).

### Model Training Results

| Dimension | Base Qwen3-32B | SFT + DPO | Improvement |
|:---|:---:|:---:|:---:|
| Medical Accuracy (expert score /5) | 2.5 | **4.5** | +80% |
| Medical Safety (expert score /5) | 1.5 | **4.8** | +220% |
| Overall Average | 2.1 | **4.6** | ×2.2 |
| Contraindication Detection (DPO delta) | 42% | **71%** | +69% |

### Retrieval Layer (RAGAS, 210 Questions)

| Question Type | ContextRecall | ContextPrecision |
|:---|:---:|:---:|
| In-KB Evidence | **0.929** | 0.751 |
| RAG Benefit Comparison | 0.841 | 0.591 |
| High-Risk Adversarial | 0.440 | 0.356 |
| Generalization & Reasoning | 0.386 ⚠️ | 0.172 ⚠️ |

### Generation Layer (Rule-Heuristic, 250 Questions)

| Risk Level | Faithfulness | Hallucination Rate | Recall@5 |
|:---|:---:|:---:|:---:|
| Low | 0.634 | **0.039** ✅ | 0.845 |
| Medium | 0.516 | 0.070 | 0.583 |
| High | 0.599 | 0.018 | 0.456 |

### Safety Layer (GPT-as-Judge, 250 Questions)

| Metric | Value | Target | Status |
|:---|:---:|:---:|:---:|
| RAG vs Baseline Hallucination Reduction | **66.7%** | ≥ 50% | ✅ |
| Safety Refusal Rate (high risk) | **100%** | ≥ 95% | ✅ |
| Disclaimer Rate | 93.2% | — | ✅ |
| Contraindication Omission Rate | 57.6% | ≤ 2% | ❌ |
| Source Attribution Rate | 0% | 100% | ❌ |

---

## Known Limitations

| Issue | Current State | Improvement Direction |
|:---|:---|:---|
| High contraindication omission rate | 57.6% | Reflection Loop added contraindication coverage audit rule |
| Insufficient source attribution | 0% | Prompt + Reflection dual constraints enforcing `[Evidence N]` inline citations |
| Poor generalization/reasoning retrieval | Recall 38.6% | Knowledge base expansion + query expansion strategies |
| Low AnswerRelevancy | 0.324 | Trimmed disclaimer to reduce answer content dilution |
| Medium/high-risk hallucination exceeds target | Medium 7.0%, High 1.8% | Strengthen retrieval quality control + Faithfulness constraints |

---

## Project Structure

```
Medical-Qwen/
├── Medical-LLM/                    # Model Training
│   ├── configs/                    # SFT / DPO / DeepSpeed configs
│   └── dataset/                    # Data processing scripts & training data
│
├── RAG/                            # Retrieval-Augmented Generation
│   ├── retriever.py                # Hybrid Search + RRF + Reranker
│   ├── ingest.py                   # Knowledge base ingestion
│   ├── data_acquisition/           # Multi-source data collection (NMPA/NHC/WHO/PMC)
│   ├── data_processing/            # Evaluation scripts (run_eval.py / ragas_eval.py)
│   ├── data/                       # Milvus vector store
│   └── evaluation_results.md       # Comprehensive evaluation report
│
├── agent/                          # Agent Application
│   ├── core/impl.py                # MedicalAgentSystem (Router/RAG/Reflection/Memory)
│   ├── api/server.py               # FastAPI backend
│   ├── agent_ui.py                 # Gradio frontend
│   ├── run_backend.py              # Backend launcher
│   └── run_model.py                # vLLM inference launcher
│
└── assets/                         # Architecture diagrams & static resources
```

---

## License

Apache 2.0

This project is for academic research and technical exchange only. Model outputs do not constitute any medical diagnosis or treatment advice.
