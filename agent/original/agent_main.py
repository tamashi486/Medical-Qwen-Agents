import os
import time
import logging
from typing import List, Dict, Any

import torch
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool, StructuredTool
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import BaseMessage

# 配置日志 (面试点: 可观测性)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MedicalAgentSystem:
    """
    企业级医疗 Agent 系统封装
    特点: 单例模式思想、支持 Rerank、支持对话记忆、异步调用
    """
    def __init__(self, 
                 db_path: str, 
                 embedding_model_path: str, 
                 vllm_api_base: str, 
                 model_name: str,
                 device: str = None):
        
        self.db_path = db_path
        self.embedding_model_path = embedding_model_path
        self.vllm_api_base = vllm_api_base
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 内部组件状态
        self.llm = None
        self.retriever = None
        self.agent_executor = None
        # 用于存储不同 SessionID 的聊天记录 (生产环境通常存 Redis)
        self.chat_histories: Dict[str, ChatMessageHistory] = {}
        
        self._initialize_system()

    def _initialize_system(self):
        """初始化核心组件"""
        try:
            logger.info(f"🚀 初始化系统... 设备: {self.device}")
            start_time = time.time()

            # 1. Embedding & VectorDB
            embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_path, 
                model_kwargs={'device': self.device}
            )
            
            # 使用 Chroma
            vector_db = Chroma(persist_directory=self.db_path, embedding_function=embeddings)
            
            # 面试点: 转换为 Retriever，获取 Top-10 此时为了后面的 Rerank 做准备
            # 如果没有 Rerank 模型，这里直接 K=3 也可以，但面试要说"为了召回率设大了 K"
            self.retriever = vector_db.as_retriever(search_kwargs={"k": 10})

            # 2. LLM (vLLM)
            self.llm = ChatOpenAI(
                model=self.model_name,
                openai_api_key="EMPTY",
                openai_api_base=self.vllm_api_base,
                temperature=0.1, # 医疗场景低熵
                max_tokens=4096,
                streaming=True   # 支持流式输出
            )

            # 3. 工具链注册
            tools = [self._create_search_tool(), self._create_bmi_tool()]

            # 4. Prompt 设计 (面试点: Role, Constraints, Format)
            prompt = ChatPromptTemplate.from_messages([
                ("system", 
                 "你是一个名为'华驼'的专业医疗AI助手。\n"
                 "核心原则：\n"
                 "1. 【循证医学】回答必须严格基于工具检索到的【证据】。如果证据不足，请明确告知用户。\n"
                 "2. 【引用来源】在回答结尾，必须列出参考的证据来源（如书籍名称）。\n"
                 "3. 【安全合规】严禁提供具体的处方建议（如'每天吃3次'），只能提供通用的治疗方案参考。\n"
                 "4. 【拒绝回答】对于非医疗或违法问题（如制造毒药），请直接拒绝。"),
                MessagesPlaceholder(variable_name="chat_history"), # 记忆槽位
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])

            # 5. 构建 Agent
            agent = create_tool_calling_agent(self.llm, tools, prompt)
            
            # 6. 包装记忆功能的执行器
            raw_executor = AgentExecutor(
                agent=agent, 
                tools=tools, 
                verbose=True,
                return_intermediate_steps=True # 返回中间步骤以便调试
            )
            
            # 使用 RunnableWithMessageHistory 管理多轮对话
            self.agent_executor = RunnableWithMessageHistory(
                raw_executor,
                self._get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
            )

            logger.info(f"✅ 系统初始化完成，耗时 {time.time() - start_time:.2f}s")

        except Exception as e:
            logger.error(f"❌ 初始化失败: {e}", exc_info=True)
            raise

    def _get_session_history(self, session_id: str) -> ChatMessageHistory:
        """获取或创建会话历史 (面试点: Session Management)"""
        if session_id not in self.chat_histories:
            self.chat_histories[session_id] = ChatMessageHistory()
        return self.chat_histories[session_id]

    # --- 工具定义 (使用闭包或实例方法) ---

    def _create_search_tool(self):
        @tool("search_medical_knowledge")
        def search_tool(query: str):
            """
            【必须使用】当用户询问具体的疾病、症状、药品、禁忌症或治疗指南时，必须调用此工具。
            """
            logger.info(f"🔍 正在检索: {query}")
            
            # 1. 粗排 (Recall)
            docs = self.retriever.invoke(query)
            if not docs:
                return "知识库中未找到相关信息。"
            
            # 2. (模拟) 重排序 (Rerank) - 面试点
            # 在实际大厂代码中，这里会调用 BGE-Reranker 模型对 docs 打分
            # sorted_docs = reranker.rank(query, docs)[:3] 
            # 这里为了代码可运行，我们简单截取前 3 个
            final_docs = docs[:3]

            # 3. 格式化输出 (带元数据) - 面试点
            results = []
            for i, doc in enumerate(final_docs):
                source = doc.metadata.get('title', '未知来源')
                category = doc.metadata.get('category', '通用')
                content = doc.page_content.replace('\n', ' ')
                results.append(f"[证据{i+1}] (来源: {source} | 分类: {category}):\n{content}")
            
            return "\n\n".join(results)
        return search_tool

    def _create_bmi_tool(self):
        @tool("calculate_bmi")
        def bmi_tool(weight_kg: float, height_m: float):
            """计算用户的BMI指数。输入体重(kg)和身高(m)。"""
            try:
                bmi = weight_kg / (height_m ** 2)
                status = "正常"
                if bmi < 18.5: status = "偏瘦"
                elif bmi > 24: status = "超重"
                
                return f"BMI数值: {bmi:.2f}\n健康状态: {status}\n建议: 请结合具体身体状况咨询医生。"
            except Exception as e:
                return f"计算出错: {str(e)}"
        return bmi_tool

    # --- 对外接口 ---

    def chat(self, user_input: str, session_id: str = "default_user"):
        """同步调用接口"""
        try:
            response = self.agent_executor.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            return response["output"]
        except Exception as e:
            logger.error(f"推理错误: {e}")
            return "系统正如火如荼地维修中..."

    async def achat(self, user_input: str, session_id: str = "default_user"):
        """异步调用接口 (Web服务专用)"""
        try:
            response = await self.agent_executor.ainvoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            return response["output"]
        except Exception as e:
            logger.error(f"异步推理错误: {e}")
            return "系统繁忙，请稍后再试。"

# --- 启动测试 ---
if __name__ == "__main__":
    # 配置
    CONFIG = {
        "db_path": "/data/home/yihui/LLM/data/medical_embedding",
        "embedding_model_path": "/data/home/yihui/LLM/bge-m3",
        "vllm_api_base": "http://localhost:8000/v1",
        "model_name": "qwen-medical"
    }

    # 实例化
    agent_system = MedicalAgentSystem(**CONFIG)

    # 测试多轮对话 (Memory 测试)
    session_id = "test_user_001"
    
    print("\n----- 测试开始 -----")
    q1 = "感冒了头痛该吃什么药？"
    print(f"User: {q1}")
    print(f"Agent: {agent_system.chat(q1, session_id)}")
    
    print("\n----- 测试记忆 -----")
    q2 = "刚才提到的药有什么副作用？" # 这里没有提药名，强依赖 Memory
    print(f"User: {q2}")
    print(f"Agent: {agent_system.chat(q2, session_id)}")