**English Version** | [中文版本](README.md)

# Enterprise Medical Agent based on Qwen3-32B

> **Project**: Medical-Qwen
> **Summary**: A full-stack implementation from Data Alignment (DPO) to High-Throughput Inference (vLLM), and an Agentic Architecture with Reflection.

## 🚩 Executive Summary

Based on the **Qwen3-32B** foundation model, this project builds an end-to-end **Enterprise Medical AI Solution**. Going beyond simple Prompt Engineering, we established a complete technical pipeline: **"Data Cleaning → Supervised Fine-Tuning (SFT) → Direct Preference Optimization (DPO) → High-Performance Deployment (vLLM) → Agentic RAG"**.

By introducing a **"Medical Auditor" Reflection Mechanism** and a **Hierarchical Memory System**, we address core pain points like hallucinations and compliance in medical scenarios. The final system supports a single-instance inference speed of **120 tokens/s**, increasing total system throughput by 82% and medical accuracy by 2.2x.

## ⚡ Key Highlights

### 1. Model Alignment: SFT + DPO
*   **SFT (Knowledge Injection)**: Fine-tuned on cleaned Huatuo QA data, transforming a general purpose model into a medical specialist. Medical accuracy improved by **80%**.
*   **DPO (Safety Guardrails)**: Constructed 5k+ preference pairs (Self-Instruct) to suppress safe but useless "generic" responses. Alignment with medical reference answers improved by **200%**.

### 2. Inference Optimization: vLLM
*   **Architecture**: Adopts **BF16 + TP=2** architecture (validated on 4x RTX 6000 environments).
*   **Performance**: Single-instance generation speed **~120 tokens/s**. Total system throughput increased by **82%** compared to baseline TP=4.

### 3. Agent Architecture: RAG & Reflection
*   **🛡️ Reflection & Self-Correction**: Introduces a "Medical Auditor" role that intercepts output with a `Draft -> Critique -> Refine` loop to ensure 100% compliance.
*   **🧠 Hierarchical Memory**: 
    *   **Entity Memory**: Asynchronously extracts patient profiles (age, allergies) for personalized advice.
    *   **Summary Memory**: Automatically summarizes long conversations to reduce context usage.
*   **🔗 Deep RAG Optimization**: Implements Query Rewrite for medical entity mapping and Evidence Grounding to cite sources.

### 4. Engineering: Async First
*   **Full-Link Async**: Core interfaces refactored with `asyncio` to adapt to FastAPI high-concurrency scenarios, significantly reducing Time-to-First-Token (TTFT).

## 📂 Project Structure

The project is organized as follows:

- **`Medical-LLM/`**: Contains resources for model training (SFT & DPO).
  - `dataset/`: 
    - `data/`: Medical datasets (e.g., jsonl files).
    - `scripts/`: Data pre-processing and formatting scripts.
  - `configs/`: Training configurations (DeepSpeed configs, training arguments).
  - `models/`: Directory for saving trained model checkpoints.

- **`agent/`**: Contains the code for the agent, RAG system, and deployment.
  - `api/`: Backend API implementation (FastAPI).
  - `core/`: Core logic for the agent (Reflection, Memory).
  - `run_backend.py`: Script to start the backend service.
  - `run_model.py`: Script to load and run the model (vLLM integration).
  - `agent_ui.py`: Frontend UI for the agent.

- **`LLaMA-Factory/`**: The training library used for efficient fine-tuning.

## 🚀 Setup and Usage

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/Medical-Qwen.git
    cd Medical-Qwen
    ```

2.  Install dependencies:
    ```bash
    pip install -r LLaMA-Factory/requirements.txt
    pip install -r agent/requirements.txt # (If available)
    ```

3.  **Model Weights**:
    - Download the **Qwen3-32B** model weights and place them in a suitable directory (e.g., `../qwen3-32B`).
    - Download the **BGE-M3** embedding model weights and place them in `../bge-m3`.
    - Update the configuration files in `agent/` and `Medical-LLM/scripts/` to point to the correct model paths.

### Training (SFT)

Use **LLaMA-Factory** to start training with the provided configurations.

```bash
# Example: Run SFT using the configuration file
llamafactory-cli train Medical-LLM/configs/training_args_sft.yaml
```

### Running the Agent

Navigate to the `agent` directory:

1.  Start the model service:
    ```bash
    cd agent
    python run_model.py
    ```

2.  Start the backend:
    ```bash
    python run_backend.py
    ```

3.  Start the UI:
    ```bash
    python agent_ui.py
    ```

## Notes

- **Large Files**: Some large dataset files and model weights are excluded from this repository. Please ensure you have the necessary data and pre-trained models.
- **Paths**: Check all configuration files for absolute paths that may need to be adjusted for your environment.
