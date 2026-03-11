import os
import time
import logging
import asyncio
from typing import List, Dict, Any

import torch
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, StructuredTool
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

from RAG.retriever import MedicalRetriever
from agent.core.interfaces import AgentInterface

# 配置日志 (面试点: 可观测性)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MedicalAgentSystem(AgentInterface):
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
        self.rag_retriever = None
        self.agent_executor = None
        # 用于存储不同 SessionID 的聊天记录 (生产环境通常存 Redis)
        self.chat_histories: Dict[str, ChatMessageHistory] = {}
        # 实体记忆: 存储患者画像 {session_id: "画像文本"}
        self.entity_memories: Dict[str, str] = {}
        
        self._initialize_system()

    def _initialize_system(self):
        """初始化核心组件"""
        try:
            logger.info(f"🚀 初始化系统... 设备: {self.device}")
            start_time = time.time()

            # 1. RAG 检索器（Hybrid: Dense + BM25 + BGE-Reranker）
            self.rag_retriever = MedicalRetriever(
                db_path=self.db_path,
                embedding_model_path=self.embedding_model_path,
            )

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
            # 修订点（基于评测结果 2026-03-11）：
            #   - 原第2条"列出来源"描述模糊导致溯源率 0%，改为强制 [证据N] 行内标注
            #   - 新增第5条强制禁忌覆盖规则（评测禁忌遗漏率 57.6%）
            #   - 精简免责声明至一句，避免稀释回答相关性（AnswerRelevancy 0.324）
            prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "你是一个名为'华驼'的专业医疗AI助手。\n"
                 "【当前患者画像】\n{patient_profile}\n\n"
                 "核心原则：\n"
                 "1. 【循证医学】回答必须严格基于工具检索到的【证据】，不可凭空编造医学数据或剂量。\n"
                 "2. 【行内引用】每处引用证据时，须在句末用 [证据N] 格式标注编号，例如：布洛芬可退热[证据1]。\n"
                 "3. 【安全合规】严禁提供具体的处方建议（如'每天吃3次'），只能提供通用的治疗方案参考。\n"
                 "4. 【拒绝回答】对于非医疗或违法问题（如制造毒药），请直接拒绝。\n"
                 "5. 【禁忌强制】若检索证据中包含禁忌症、禁用人群或药物相互作用，必须在回答中完整列出，不得遗漏。\n"
                 "6. 【免责声明】在回答末尾附一句：'以上信息仅供参考，请遵医嘱。'"),
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

    def _update_entity_memory(self, session_id: str, user_input: str):
        """
        实体记忆更新 (Entity Memory Update)
        使用 LLM 提取用户画像并更新
        """
        current_profile = self.entity_memories.get(session_id, "暂无")
        
        extraction_prompt = (
            f"当前患者画像: {current_profile}\n"
            f"用户新输入: {user_input}\n"
            "请基于新输入更新患者画像。包含: 年龄、性别、既往病史、过敏源、当前症状。\n"
            "如果输入中没有新信息，请保持原画像不变。\n"
            "请直接输出更新后的画像摘要，不要废话。"
        )
        
        try:
            # 使用 invoke 调用 LLM 进行提取 (同步调用，生产环境建议异步)
            # 这里为了演示简单，直接复用 self.llm
            # 注意: 这里可能会增加延迟
            response = self.llm.invoke([HumanMessage(content=extraction_prompt)])
            new_profile = response.content
            self.entity_memories[session_id] = new_profile
            logger.info(f"📝 [Entity Memory] Updated for {session_id}: {new_profile[:50]}...")
        except Exception as e:
            logger.error(f"❌ Entity extraction failed: {e}")

    def _manage_summary_memory(self, session_id: str):
        """
        分层记忆管理 (Tiered Memory Management)
        如果历史记录超过阈值，进行摘要压缩
        """
        history = self._get_session_history(session_id)
        messages = history.messages
        
        # 阈值设定: 保留最近 10 条 (5轮) 原始对话
        MAX_RAW_HISTORY = 10
        
        if len(messages) > MAX_RAW_HISTORY + 2: # +2 buffer
            # 切分: 需要摘要的部分 vs 保留的部分
            to_summarize = messages[:-MAX_RAW_HISTORY]
            to_keep = messages[-MAX_RAW_HISTORY:]
            
            # 生成摘要
            summary_prompt = "请简要总结以下对话的历史重点，保留关键医疗信息:\n"
            for msg in to_summarize:
                role = "用户" if isinstance(msg, HumanMessage) else "AI"
                summary_prompt += f"{role}: {msg.content}\n"
            
            try:
                response = self.llm.invoke([HumanMessage(content=summary_prompt)])
                summary_text = response.content
                
                # 重构历史: [SystemMessage(Summary)] + [Raw Messages]
                # 注意: LangChain 的 ChatMessageHistory 是 append-only 的，这里我们需要直接修改内部 list
                # 这是一个 hack，标准做法是用 ConversationSummaryBufferMemory，但为了演示原理手动实现
                new_messages = [SystemMessage(content=f"【历史对话摘要】: {summary_text}")] + to_keep
                history.messages = new_messages
                
                logger.info(f"🧠 [Summary Memory] Compressed history for {session_id}")
            except Exception as e:
                logger.error(f"❌ Summary generation failed: {e}")

    # --- 工具定义 (使用闭包或实例方法) ---

    def _create_search_tool(self):
        @tool("search_medical_knowledge")
        def search_tool(query: str):
            """
            【必须使用】当用户询问具体的疾病、症状、药品、禁忌症或治疗指南时，必须调用此工具。
            """
            # 1. 查询改写 (Query Rewriting)
            rewrite_prompt = (
                f"请将用户的搜索查询 '{query}' 改写为一个更适合检索医学知识库的独立查询语句。\n"
                "要求：\n"
                "- 去除口语化表达，替换为规范医学术语\n"
                "- 保留核心医学实体（药品名、疾病名、症状名）\n"
                "- 若涉及特殊人群，明确包含关键词（如：孕妇、儿童、老年人、肝肾功能不全）\n"
                "- 直接输出改写后的查询，不要包含其他内容"
            )
            try:
                rewritten_query = self.llm.invoke([HumanMessage(content=rewrite_prompt)]).content.strip()
                logger.info(f"🔄 [Query Rewrite] '{query}' -> '{rewritten_query}'")
            except Exception as e:
                logger.warning(f"Query rewrite failed: {e}")
                rewritten_query = query

            logger.info(f"🔍 正在检索: {rewritten_query}")

            # 2. Hybrid 检索：Dense + BM25 → RRF → BGE-Reranker
            results = self.rag_retriever.hybrid_retrieve(rewritten_query, top_k=5)
            if not results:
                return "知识库中未找到相关信息。"

            # 3. Evidence-aware 格式化（高风险标注 + 来源列表）
            return self.rag_retriever.format_evidence(results, max_results=5)
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

    def _route_request(self, user_input: str) -> str:
        """
        路由模式 (Router Pattern)
        判断用户意图: chat, medical, complex
        """
        router_prompt = (
            f"用户输入: {user_input}\n"
            "请判断用户意图，返回以下类别之一:\n"
            "- chat: 闲聊、问候、非医疗问题\n"
            "- medical: 具体的医疗咨询、查病、查药\n"
            "- complex: 复杂的病例分析、多步推理\n"
            "直接输出类别名称，不要其他内容。"
        )
        try:
            intent = self.llm.invoke([HumanMessage(content=router_prompt)]).content.strip().lower()
            # 简单清洗
            if "medical" in intent: return "medical"
            if "complex" in intent: return "complex"
            return "chat"
        except Exception:
            return "medical" # 默认走医疗

    def _reflection_check(self, user_input: str, response: str) -> str:
        """
        反思模式 (Reflection Pattern)
        检查回答是否包含幻觉、违规或禁忌遗漏。
        修订点（基于评测 2026-03-11）：新增第3条禁忌遗漏检查（评测遗漏率 57.6%）
        """
        critique_prompt = (
            f"用户问题: {user_input}\n"
            f"AI回答: {response}\n"
            "请作为'医疗审核员'检查上述回答：\n"
            "1. 是否包含具体的处方建议（如'每天吃3次'）？(违规)\n"
            "2. 是否引用了不存在的证据？(幻觉)\n"
            "3. 若回答涉及药物或治疗，是否遗漏了禁忌症、禁用人群或药物相互作用？(遗漏禁忌)\n"
            "4. 是否回答了非医疗问题但伪装成医疗建议？\n"
            "如果回答安全且合规，请输出 'PASS'。\n"
            "如果有问题，请输出具体的修改建议，说明哪条规则被违反。"
        )
        try:
            critique = self.llm.invoke([HumanMessage(content=critique_prompt)]).content.strip()
            if "PASS" in critique:
                return response

            logger.warning(f"⚠️ [Reflection] Critique triggered: {critique}")
            fix_prompt = (
                f"原问题: {user_input}\n"
                f"原回答: {response}\n"
                f"审核意见: {critique}\n"
                "请根据审核意见重写回答，确保安全合规，并补全所有被遗漏的禁忌信息。"
            )
            new_response = self.llm.invoke([HumanMessage(content=fix_prompt)]).content
            return new_response
        except Exception as e:
            logger.error(f"Reflection failed: {e}")
            return response

    async def _route_request_async(self, user_input: str) -> str:
        """
        路由模式 (Router Pattern) - 异步版
        """
        router_prompt = (
            f"用户输入: {user_input}\n"
            "请判断用户意图，返回以下类别之一:\n"
            "- chat: 闲聊、问候、非医疗问题\n"
            "- medical: 具体的医疗咨询、查病、查药\n"
            "- complex: 复杂的病例分析、多步推理\n"
            "直接输出类别名称，不要其他内容。"
        )
        try:
            response = await self.llm.ainvoke([HumanMessage(content=router_prompt)])
            intent = response.content.strip().lower()
            if "medical" in intent: return "medical"
            if "complex" in intent: return "complex"
            return "chat"
        except Exception:
            return "medical"

    async def _reflection_check_async(self, user_input: str, response: str) -> str:
        """
        反思模式 (Reflection Pattern) - 异步版
        修订点（基于评测 2026-03-11）：新增第3条禁忌遗漏检查（评测遗漏率 57.6%）
        """
        critique_prompt = (
            f"用户问题: {user_input}\n"
            f"AI回答: {response}\n"
            "请作为'医疗审核员'检查上述回答：\n"
            "1. 是否包含具体的处方建议（如'每天吃3次'）？(违规)\n"
            "2. 是否引用了不存在的证据？(幻觉)\n"
            "3. 若回答涉及药物或治疗，是否遗漏了禁忌症、禁用人群或药物相互作用？(遗漏禁忌)\n"
            "4. 是否回答了非医疗问题但伪装成医疗建议？\n"
            "如果回答安全且合规，请输出 'PASS'。\n"
            "如果有问题，请输出具体的修改建议，说明哪条规则被违反。"
        )
        try:
            critique_res = await self.llm.ainvoke([HumanMessage(content=critique_prompt)])
            critique = critique_res.content.strip()
            if "PASS" in critique:
                return response

            logger.warning(f"⚠️ [Reflection] Critique triggered: {critique}")
            fix_prompt = (
                f"原问题: {user_input}\n"
                f"原回答: {response}\n"
                f"审核意见: {critique}\n"
                "请根据审核意见重写回答，确保安全合规，并补全所有被遗漏的禁忌信息。"
            )
            new_response_res = await self.llm.ainvoke([HumanMessage(content=fix_prompt)])
            return new_response_res.content
        except Exception as e:
            logger.error(f"Reflection failed: {e}")
            return response

    # --- 对外接口 ---

    def chat(self, user_input: str, session_id: str = "default_user"):
        """同步调用接口"""
        try:
            # 0. 路由
            intent = self._route_request(user_input)
            logger.info(f"🚦 [Router] Intent: {intent}")
            
            if intent == "chat":
                return self.llm.invoke([HumanMessage(content=user_input)]).content

            # 1. 更新实体记忆
            self._update_entity_memory(session_id, user_input)
            
            # 2. 管理摘要记忆
            self._manage_summary_memory(session_id)
            
            # 3. 获取当前画像
            patient_profile = self.entity_memories.get(session_id, "未知")

            # 4. Agent 执行
            response = self.agent_executor.invoke(
                {"input": user_input, "patient_profile": patient_profile},
                config={"configurable": {"session_id": session_id}}
            )
            final_output = response["output"]

            # 5. 反思检查
            final_output = self._reflection_check(user_input, final_output)
            
            return final_output
        except Exception as e:
            logger.error(f"推理错误: {e}")
            return "系统正如火如荼地维修中..."

    async def achat(self, user_input: str, session_id: str = "default_user", mode: str = "agent"):
        """异步调用接口 (Web服务专用)"""
        try:
            # 0. 路由 (仅在 Agent 模式下生效)
            if mode == "agent":
                intent = await self._route_request_async(user_input)
                logger.info(f"🚦 [Router] Intent: {intent}")
                if intent == "chat":
                    response = await self.llm.ainvoke([HumanMessage(content=user_input)])
                    return response.content

            # 1. 更新实体记忆 (后台任务，不阻塞主流程)
            # 使用 asyncio.create_task 将其放入后台执行
            asyncio.create_task(asyncio.to_thread(self._update_entity_memory, session_id, user_input))
            
            # 2. 管理摘要记忆 (后台任务)
            asyncio.create_task(asyncio.to_thread(self._manage_summary_memory, session_id))
            
            # 3. 获取当前画像 (直接读取，不等待更新)
            patient_profile = self.entity_memories.get(session_id, "未知")

            if mode == "agent":
                response = await self.agent_executor.ainvoke(
                    {"input": user_input, "patient_profile": patient_profile},
                    config={"configurable": {"session_id": session_id}}
                )
                final_output = response["output"]
                
                # 5. 反思检查 (异步)
                final_output = await self._reflection_check_async(user_input, final_output)
                return final_output
            else:
                # Simple Chat Mode
                history = self._get_session_history(session_id)
                messages = [SystemMessage(content="你是一个医疗AI助手。请直接回答问题，无需提供参考来源。")]
                messages.extend(history.messages)
                messages.append(HumanMessage(content=user_input))
                
                # Manually update history
                history.add_user_message(user_input)
                
                response_content = ""
                # Note: This is not streaming to the client in real-time via API yet, 
                # but returns the full response asynchronously.
                async for chunk in self.llm.astream(messages):
                    response_content += chunk.content
                
                history.add_ai_message(response_content)
                return response_content

        except Exception as e:
            logger.error(f"异步推理错误: {e}")
            return "系统繁忙，请稍后再试。"

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """获取会话历史"""
        if session_id not in self.chat_histories:
            return []
        
        history = self.chat_histories[session_id]
        formatted_history = []
        for msg in history.messages:
            if isinstance(msg, HumanMessage):
                formatted_history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                formatted_history.append({"role": "assistant", "content": msg.content})
        return formatted_history
