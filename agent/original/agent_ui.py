import gradio as gr
import os
import sys
import time
import html
import base64
import io
import re
import uuid
import logging
from datetime import datetime
from typing import List, Tuple
from PIL import Image
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# --- Environment Setup ---
os.environ['no_proxy'] = 'localhost,127.0.0.1,0.0.0.0'
os.environ['NO_PROXY'] = 'localhost,127.0.0.1,0.0.0.0'

# --- Configuration ---
MAX_MESSAGE_LENGTH = 16000
MAX_HISTORY_LENGTH = 50
MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_OUTPUT_TOKENS = 8192
DEFAULT_OUTPUT_TOKENS = 2048

SUPPORTED_TEXT_EXTENSIONS = {'.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml', '.csv'}
SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("WebUI")

# --- Import Agent Core ---
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    
    from agent_main import MedicalAgentSystem
    logger.info("✅ Agent Class Imported")
except Exception as e:
    logger.error(f"❌ Agent Class Import Failed: {e}")
    MedicalAgentSystem = None

# --- Initialize System ---
agent_system = None
if MedicalAgentSystem:
    try:
        # Configuration matching agent_main.py defaults or environment
        CONFIG = {
            "db_path": "/data/home/yihui/LLM/data/medical_embedding",
            "embedding_model_path": "/data/home/yihui/LLM/bge-m3",
            "vllm_api_base": "http://localhost:8000/v1",
            "model_name": "qwen-medical"
        }
        logger.info("🚀 Initializing Medical Agent System...")
        agent_system = MedicalAgentSystem(**CONFIG)
        logger.info("✅ Medical Agent System Initialized")
    except Exception as e:
        logger.error(f"❌ Medical Agent System Initialization Failed: {e}")

# --- Helper Functions ---

def generate_session_id():
    """为每个 Tab 生成唯一会话 ID"""
    return str(uuid.uuid4())

def process_uploaded_file(file_path: str) -> Tuple[str, str]:
    if not file_path or not os.path.exists(file_path):
        return "", "文件不存在"
    
    file_size = os.path.getsize(file_path)
    if file_size > MAX_FILE_SIZE:
        return "", f"文件过大 ({file_size / 1024 / 1024:.1f}MB)"
    
    file_name = os.path.basename(file_path)
    file_ext = os.path.splitext(file_name)[1].lower()
    
    try:
        if file_ext in SUPPORTED_TEXT_EXTENSIONS:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            return content, f"📄 文件: {file_name} ({file_size} bytes)"
        
        elif file_ext in SUPPORTED_IMAGE_EXTENSIONS:
            with Image.open(file_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.thumbnail((800, 800))
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=85)
                img_base64 = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/jpeg;base64,{img_base64}", f"🖼️ 图片: {file_name}"
        
        return "", f"不支持的文件类型: {file_ext}"
    except Exception as e:
        return "", f"文件处理错误: {str(e)[:100]}"

def get_timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")

def format_message(content: str, role: str) -> str:
    timestamp = get_timestamp()
    role_name = "您" if role == "user" else "AI"
    return f"**[{timestamp}] {role_name}:** {content}"

def clean_content(content: str) -> str:
    """Remove timestamp and role prefix for processing"""
    return re.sub(r'\*\*\[.*?\] .*?:\*\* ', '', content)

def format_thinking(text: str) -> str:
    """Format <think> tags into collapsible details"""
    # Handle complete think blocks (support multiple blocks)
    pattern = r"<think>(.*?)</think>"
    
    def replace_func(match):
        content = match.group(1).strip()
        return f'''<details class="thought-bubble">
<summary>🧠 思考过程 (点击展开)</summary>
<div class="thought-content">{content}</div>
</details>'''
    
    formatted_text = re.sub(pattern, replace_func, text, flags=re.DOTALL)
    
    # Handle incomplete think block (streaming)
    if "<think>" in formatted_text and "</think>" not in formatted_text:
        parts = formatted_text.split("<think>", 1)
        pre_content = parts[0]
        think_content = parts[1]
        return f'''{pre_content}<details class="thought-bubble" open>
<summary>🧠 正在思考...</summary>
<div class="thought-content">{think_content}</div>
</details>'''
        
    return formatted_text

def export_conversation(history: List[dict]) -> str:
    if not history:
        return "暂无对话记录"
    export_text = f"# 对话记录导出\n导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    for msg in history:
        role = "用户" if msg["role"] == "user" else "AI助手"
        content = clean_content(msg["content"])
        export_text += f"## {role}\n{content}\n\n"
    return export_text

def handle_export(history: List[dict]):
    if not history:
        return None, "⚠️ 暂无对话记录可导出"
    try:
        export_content = export_conversation(history)
        filename = f"conversation_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(export_content)
        return filename, f"✅ 对话已成功导出到 {filename}"
    except Exception as e:
        return None, f"❌ 导出失败: {str(e)[:50]}"

def copy_last_response(history: List[dict]):
    if not history:
        return "⚠️ 暂无对话记录"
    for msg in reversed(history):
        if msg["role"] == "assistant":
            content = clean_content(msg["content"])
            return f"✅ 已准备复制: {content[:50]}..."
    return "⚠️ 未找到AI回复"

def on_like(x: gr.LikeData, session_id: str):
    """收集用户点赞/点踩数据"""
    user_feedback = "Liked" if x.liked else "Disliked"
    logger.info(f"Feedback [{session_id}]: {user_feedback} | Message: {x.value[:50]}...")
    gr.Info(f"感谢反馈！已记录 ({user_feedback})")

# --- Core Logic ---
async def chat_stream(
    message: str, 
    history: List[dict], 
    mode: str, 
    session_id: str,
    uploaded_file=None, 
    temperature=0.7, 
    max_tokens=DEFAULT_OUTPUT_TOKENS
):
    if not message.strip() and not uploaded_file:
        yield history
        return

    # Prepare Input
    file_content, file_info = "", ""
    if uploaded_file:
        file_content, file_info = process_uploaded_file(uploaded_file.name)
    
    display_message = message
    if file_info:
        display_message += f"\n\n[{file_info}]"
    
    agent_input = message
    if file_content:
        agent_input += f"\n\n[文件内容]\n{file_content}"

    # Update History (User)
    new_history = history + [{"role": "user", "content": format_message(display_message, "user")}]
    
    # Placeholder for AI
    loading_msg = "⏳ 正在思考并查询知识库..." if mode == "Agent模式" else "⏳ 正在生成..."
    new_history.append({"role": "assistant", "content": format_message(loading_msg, "assistant")})
    yield new_history

    try:
        if not agent_system:
            raise Exception("系统未初始化，请检查后台日志")

        bot_response = ""
        
        if mode == "Agent模式":
            # Agent Mode (Sync)
            # 使用 session_id 保持多轮对话记忆
            # 注意：如果后端 chat 是同步的，这里会阻塞。建议后端实现 achat
            # 这里演示假设后端返回完整字符串，我们模拟流式打字机效果让用户感觉快
            # 如果 agent_system 有 achat 方法，最好用 await agent_system.achat(...)
            if hasattr(agent_system, 'achat'):
                response = await agent_system.achat(agent_input, session_id=session_id)
            else:
                response = agent_system.chat(agent_input, session_id=session_id)
            
            bot_response = response
            
            # Format and yield final result
            formatted_response = format_thinking(bot_response)
            new_history[-1]["content"] = format_message(formatted_response, "assistant")
            yield new_history
            
        else: # 普通问答模式
            # Ordinary Mode (Streaming)
            llm = agent_system.llm
            if not llm:
                raise Exception("LLM component not initialized")
            
            messages = [SystemMessage(content="你是一个医疗AI助手。请直接回答问题，无需提供参考来源。")]
            # Reconstruct history for LLM
            for msg in history:
                content = clean_content(msg["content"])
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=content))
                else:
                    messages.append(AIMessage(content=content))
            messages.append(HumanMessage(content=agent_input))
            
            # Stream with parameters
            # 使用 bind 动态绑定参数
            runnable = llm.bind(temperature=temperature, max_tokens=max_tokens)
            
            # 使用 astream 进行异步流式输出
            async for chunk in runnable.astream(messages):
                content = chunk.content
                bot_response += content
                formatted_response = format_thinking(bot_response)
                new_history[-1]["content"] = format_message(formatted_response, "assistant")
                yield new_history

            # Final yield
            formatted_response = format_thinking(bot_response)
            new_history[-1]["content"] = format_message(formatted_response, "assistant")
            yield new_history

    except Exception as e:
        logger.error(f"Error: {e}")
        new_history[-1]["content"] = format_message(f"错误: {str(e)}", "assistant")
        yield new_history

# --- UI ---
custom_css = """
/* 更加现代化的配色 */
body { font-family: 'Helvetica Neue', Arial, sans-serif; }
#chatbot { 
    height: 700px !important; 
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
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
.message-timestamp {
    font-size: 0.8em;
    color: #666;
}
/* 引用源样式 */
a { color: #4a90e2; text-decoration: none; }
a:hover { text-decoration: underline; }
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), css=custom_css, title="Huatuo-Pro Medical") as demo:
    
    # 状态管理：为每个用户分配独立的 SessionID
    session_state = gr.State(generate_session_id)
    
    with gr.Row():
        # 左侧边栏：控制面板
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("### 🏥 医疗大模型")
            gr.Markdown("Based on Qwen3-32B")
            
            with gr.Group():
                mode_radio = gr.Radio(
                    ["Agent模式", "普通问答模式"], 
                    label="推理模式", 
                    value="Agent模式",
                    info="Agent模式具备查库和工具调用能力"
                )
                
                with gr.Accordion("🎛️ 参数设置", open=False):
                    temperature_slider = gr.Slider(0.0, 1.0, value=0.1, label="温度 (Temperature)", info="医疗场景建议低迷以保证严谨")
                    max_tokens_slider = gr.Slider(100, MAX_OUTPUT_TOKENS, value=DEFAULT_OUTPUT_TOKENS, label="Max Tokens")
            
            file_upload = gr.File(label="上传文件")
            
            gr.Markdown("#### 💡 提示")
            gr.Markdown("- 询问疾病时请描述清楚症状\n- 涉及药物时请咨询禁忌症\n- 模型回答仅供参考，不作为最终医疗诊断")
            
            with gr.Row():
                export_btn = gr.Button("📥 导出对话")
                copy_btn = gr.Button("📋 复制最后回复")
                clean_btn = gr.Button("🗑️ 清空对话", variant="secondary")
            
            export_file = gr.File(label="导出的文件", visible=False)
            status_display = gr.Textbox(label="状态", interactive=False, max_lines=1)

        # 右侧：聊天主窗口
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                label="诊断对话",
                type="messages", 
                avatar_images=(None, "https://img.alicdn.com/imgextra/i4/O1CN01c26iB51UyR3MKMFma_!!6000000002586-2-tps-124-124.png"),
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
    
    input_params = [msg_input, chatbot, mode_radio, session_state, file_upload, temperature_slider, max_tokens_slider]
    
    # 提交消息
    msg_input.submit(
        fn=chat_stream, 
        inputs=input_params, 
        outputs=chatbot,
        show_progress="hidden"
    ).then(lambda: "", None, msg_input)

    submit_btn.click(
        fn=chat_stream, 
        inputs=input_params, 
        outputs=chatbot
    ).then(lambda: "", None, msg_input)

    # 点赞事件
    chatbot.like(on_like, [session_state], None)

    # 额外功能按钮
    clean_btn.click(lambda: [], None, chatbot)
    export_btn.click(handle_export, [chatbot], [export_file, status_display])
    copy_btn.click(copy_last_response, [chatbot], status_display)

if __name__ == "__main__":
    print(f"正在启动 Web UI...")
    # 生产环境配置
    demo.queue(max_size=20) # 开启队列，支持多用户并发
    try:
        demo.launch(
            server_name="0.0.0.0", 
            server_port=7860, 
            share=False,
            favicon_path=None
        )
    except OSError:
        print("端口 7860 被占用，尝试使用 7861...")
        demo.launch(
            server_name="0.0.0.0", 
            server_port=7861, 
            share=False,
            favicon_path=None
        )
