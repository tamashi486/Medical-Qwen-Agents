# 医疗 Agent 系统架构总结

> **项目概述**: 本项目实现了一个基于 Qwen3-32B 模型的全栈医疗智能体系统。通过深度结合 **RAG (检索增强生成)** 与 **Agent (智能体)** 技术，并引入 **多层记忆** 与 **反思机制**，解决了通用模型在医疗场景下的幻觉严重、安全合规性差等痛点。代码结构清晰，具备工业级落地的工程实践特性。

## 1. 核心架构设计 (Architecture)

系统采用典型的分层架构，实现了从底层推理到上层交互的完整链路：

*   **Model Layer (推理层)**: 基于 `vLLM` 部署 Qwen3-32B (SFT + DPO) 模型，提供高吞吐的 OpenAI 兼容接口，支持 Tool Calling 和 Tensor Parallelism。
*   **Orchestration Layer (编排层)**: 使用 `LangChain` 构建 Agent Loop，集成了检索、工具调用、记忆管理和反思流程。
*   **Serving Layer (服务层)**: `FastAPI` 提供异步标准接口 (`/chat`, `/history`)，支持高并发访问。
*   **Interface Layer (交互层)**: `Gradio` 构建的用户友好 WebUI，支持流式输出和多模态文件上传。

---

## 2. RAG 系统深度优化 (Advanced RAG)

针对医疗场景对准确性的极致要求，RAG 流水线进行了多级优化：

*   **Query Rewrite (查询改写)**:
    *   **痛点**: 用户输入往往口语化、含糊不清 (如 "孩子发烧咋整")。
    *   **方案**: 利用 LLM 将用户 Query 改写为包含核心医学实体的独立查询语句，大幅提升向量检索的 Recall (召回率) 和 Precision (精确率)。
*   **Retrieval & Ranking (检索与排序)**:
    *   使用 `HuggingFaceEmbeddings` + `Chroma` 向量库。
    *   代码中预留了 **Re-rank (重排序)** 逻辑接口，模拟了工业界 "粗排 + 精排" 的两阶段检索流程。
*   **Citation & Grounding (引用与溯源)**:
    *   System Prompt 强制约束模型必须基于检索到的 `[证据]` 回答。
    *   要求输出明确的证据来源 (如《内科学》)，有效抑制幻觉 (Hallucination)。

---

## 3. Agent 智能体核心能力 (Agent Capabilities)

Agent 不仅仅是问答，通过 ReAct 范式实现了自主决策与自我修正：

### A. 意图路由 (Intent Routing)
*   **机制**: 在处理请求前，先进行意图分类 (`chat`, `medical`, `complex`)。
*   **价值**: 
    *   闲聊问题 (Chat) 直接由 LLM 回答，降低延迟。
    *   医疗问题 (Medical) 进入 Agent 复杂链路。
    *   实现了计算资源的分级调度。

### B. 工具调用 (Tool Calling)
*   **Search Tool**: 封装了上述的 RAG 流程，模型根据问题自动判断是否需要查阅知识库。
*   **Calculator Tool**: 如 `calculate_bmi`，处理数值计算类问题，弥补 LLM 数学能力的短板。

### C. 反思与自修正 (Reflection & Self-Correction) ✨ *亮眼特性*
*   **机制**: 引入 **"医疗审核员"** 角色，在响应返回给用户前进行二次检查。
*   **检查维度**:
    1.  是否包含具体的处方建议 (如"每日3次") —— **合规性检查**。
    2.  是否引用了不存在的证据 —— **真实性检查**。
*   **流程**: 若审核未通过 (`critique != PASS`)，Agent 会自动根据审核意见重写回答，确保输出安全可靠。

---

## 4. 多层记忆系统 (Hierarchical Memory)

为了在长对话中保持连贯性并控制 Token 成本，实现了多维度的记忆管理：

1.  **Session Memory (会话记忆)**:
    *   基于 `RunnableWithMessageHistory`，维护当前对话窗口的完整上下文。
2.  **Entity Memory (实体记忆)**:
    *   **动态提取**: 在后台通过 LLM 从对话中自动抽取 **[患者画像]** (年龄、性别、既往史、过敏源)。
    *   **持久化**: 将画像注入 System Context，确保 Agent 始终了解患者背景，提供个性化建议。
3.  **Summary Memory (摘要记忆)**:
    *   **Tiered Handling**: 当对话轮数超过阈值时，自动触发后台任务对旧历史进行语义摘要。
    *   **价值**: 解决了长 Context 遗忘问题，并显著降低推理成本。

---

## 5. 工程化实践 (Engineering Excellence)

代码展现了优秀的工程素养，适合作为企业级应用参考：

*   **Async First (异步优先)**: 
    *   核心接口 `achat` 全异步实现，适配 FastAPI 的高并发特性。
    *   **Background Tasks**: 记忆更新 (`_update_entity_memory`)、摘要生成 (`_manage_summary_memory`) 等耗时操作放入后台线程执行，**不阻塞用户主响应**，极大降低了 TTFT (Time-to-First-Token)。
*   **可观测性 (Observability)**: 
    *   关键节点 (Rewrite, Retrieve, Route, Reflection) 均埋设了详细的 Logging，便于链路追踪与调试。
*   **稳健性设计**:
    *   完善的异常捕获 (Try-Catch) 和降级策略 (Fallback)，确保系统在组件故障时仍能给出友好回复。

---

**总结**: 
该模块不仅仅是一个 Demo，而是一个具备 **"听 (Router)"、"想 (Entity Memory)"、"查 (RAG)"、"省 (Reflection)"** 能力的完整医疗智能体框架，充分挖掘了 Qwen3-32B 在复杂指令遵循和工具调用上的潜力。
