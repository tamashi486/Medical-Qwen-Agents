请按照以下顺序启动三个服务（建议在三个不同的终端窗口中运行）：

启动模型服务 (vLLM)
python agent/run_model.py
等待模型加载完成，服务启动在 localhost:8000

启动业务后端 (FastAPI)
python agent/run_backend.py
服务将启动在 localhost:8081

启动前端界面 (Gradio)
python agent/agent_ui.py
*访问 http://localhost:7860 使用系统