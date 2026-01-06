import gradio as gr
import os
import sys
import uuid
import logging
import asyncio
from typing import List, Generator

# --- 引入后端 (确保 agent_main.py 在同一目录) ---
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)
    from medical_agent_pro import MedicalAgentSystem # 假设你保存的后端文件名是这个
except ImportError:
    print("❌ 未找到后端文件 medical_agent_pro.py，请检查文件名")
    MedicalAgentSystem = None

# --- 配置日志 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("WebUI")

# --- 初始化单例系统 ---
# 真正的生产环境配置应该从环境变量读取 (os.getenv)
CONFIG = {
    "db_path": "/data/home/yihui/LLM/data/medical_embedding",
    "embedding_model_path": "/data/home/yihui/LLM/bge-m3",
    "vllm_api_base": "http://localhost:8000/v1",
    "model_name": "qwen-medical"
}

agent_system = None
if MedicalAgentSystem:
    try:
        agent_system = MedicalAgentSystem(**CONFIG)
        logger.info("✅ 后端系统加载成功")
    except Exception as e:
        logger.error(f"❌ 系统初始化失败: {e}")

# --- 辅助函数 ---

def generate_session_id():
    """为每个 Tab 生成唯一会话 ID"""
    return str(uuid.uuid4())

def format_history_for_gradio(history):
    """Gradio 需要 list of dicts 或 list of lists"""
    return history

def process_thinking_process(text: str) -> str:
    """美化思考过程展示"""
    # 简单的 XML 标签解析，也可以用正则
    if "<think>" in text and "</think>" in text:
        parts = text.split("</think>")
        thought = parts[0].replace("<think>", "").strip()
        answer = parts[1].strip()
        # 使用 HTML details 标签实现折叠效果
        return f"""<details class="thought-bubble">
<summary>🧠 思考过程 (点击展开)</summary>
<div class="thought-content">{thought}</div>
</details>

{answer}"""
    return text

# --- 核心逻辑 (Async) ---

async def chat_stream(
    message: str, 
    history: List[dict], 
    mode: str, 
    session_id: str,
    temperature: float
):
    """
    异步流式响应函数
    """
    if not message:
        yield history
        return

    # 1. 更新用户消息
    history.append({"role": "user", "content": message})
    # 添加一个空的 AI 消息占位符
    history.append({"role": "assistant", "content": "⏳ 正在分析病例并查阅文献..."})
    yield history

    try:
        if not agent_system:
            raise Exception("后台服务未连接")

        response_content = ""

        if mode == "Agent模式":
            # 异步调用 Agent，避免阻塞
            # 注意：如果后端 chat 是同步的，这里会阻塞。建议后端实现 achat
            # 这里演示假设后端返回完整字符串，我们模拟流式打字机效果让用户感觉快
            full_response = await agent_system.achat(message, session_id=session_id)
            
            # 美化思考标签
            display_response = process_thinking_process(full_response)
            
            # 模拟流式更新 (如果后端不支持流式 Agent)
            # 实际大厂项目会使用 LangChain 的 astream_events 来真流式输出
            history[-1]["content"] = display_response
            yield history
            
        else:
            # 普通 LLM 模式 (真流式)
            llm = agent_system.llm
            async for chunk in llm.astream(message):
                response_content += chunk.content
                history[-1]["content"] = response_content
                yield history

    except Exception as e:
        logger.error(f"Error: {e}")
        history[-1]["content"] = f"❌ 系统错误: {str(e)}"
        yield history

# --- 反馈回调 (RLHF 数据收集) ---
def on_like(x: gr.LikeData, session_id: str):
    """
    收集用户点赞/点踩数据
    """
    user_feedback = "Liked" if x.liked else "Disliked"
    logger.info(f"Feedback [{session_id}]: {user_feedback} | Message: {x.value[:50]}...")
    gr.Info(f"感谢反馈！已记录 ({user_feedback})")

# --- UI 布局 ---
custom_css = """
/* 更加现代化的配色 */
body { font-family: 'Helvetica Neue', Arial, sans-serif; }
#chatbot { 
    height: 700px !important; 
    border: 1px solid #e0e0e0;
    border-radius: 12px;
}
/* 思考气泡样式 */
.thought-bubble {
    background-color: #f0f4f8;
    border-left: 4px solid #4a90e2;
    padding: 10px;
    margin-bottom: 10px;
    border-radius: 4px;
    font-size: 0.9em;
    color: #555;
}
.thought-content {
    margin-top: 8px;
    white-space: pre-wrap;
}
/* 引用源样式 (假设后端返回的内容包含 Markdown 链接或特定格式) */
a { color: #4a90e2; text-decoration: none; }
a:hover { text-decoration: underline; }
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), css=custom_css, title="Huatuo-Pro Medical") as demo:
    
    # 状态管理：为每个用户分配独立的 SessionID
    session_state = gr.State(generate_session_id)
    
    with gr.Row():
        # 左侧边栏：控制面板
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("### 🏥 华驼医疗大模型专业版")
            gr.Markdown("Based on Qwen3-32B & RAG")
            
            with gr.Group():
                mode_radio = gr.Radio(
                    ["Agent模式", "纯对话模式"], 
                    label="推理模式", 
                    value="Agent模式",
                    info="Agent模式具备查库和工具调用能力"
                )
                temp_slider = gr.Slider(0.0, 1.0, value=0.1, label="温度 (Temperature)", info="医疗场景建议低迷以保证严谨")
            
            gr.Markdown("#### 💡 提示")
            gr.Markdown("- 询问疾病时请描述清楚症状\n- 涉及药物时请咨询禁忌症\n- 模型回答仅供参考，不作为最终医疗诊断")
            
            clean_btn = gr.Button("🗑️ 清空对话", variant="secondary")

        # 右侧：聊天主窗口
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                label="诊断对话",
                type="messages", # 使用新版消息格式
                avatar_images=("user.png", "doctor.png"), # 建议放两个本地图片文件
                show_copy_button=True,
                elem_id="chatbot"
            )
            
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="请输入您的医疗咨询问题 (例如: 高血压患者能吃香蕉吗？)",
                    show_label=False,
                    scale=9,
                    container=False
                )
                submit_btn = gr.Button("发送", variant="primary", scale=1)

    # --- 事件绑定 ---
    
    # 提交消息
    input_params = [msg_input, chatbot, mode_radio, session_state, temp_slider]
    
    msg_input.submit(
        fn=chat_stream, 
        inputs=input_params, 
        outputs=chatbot,
        show_progress="hidden" # 隐藏顶部进度条，使用流式输出
    ).then(lambda: "", None, msg_input) # 发送完清空输入框

    submit_btn.click(
        fn=chat_stream, 
        inputs=input_params, 
        outputs=chatbot
    ).then(lambda: "", None, msg_input)

    # 点赞事件
    chatbot.like(on_like, [session_state], None)

    # 清空
    clean_btn.click(lambda: [], None, chatbot)

if __name__ == "__main__":
    # 生产环境配置
    demo.queue(max_size=20) # 开启队列，支持多用户并发
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860, 
        share=False,
        favicon_path=None
    )