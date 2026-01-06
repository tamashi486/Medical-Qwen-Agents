import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from contextlib import asynccontextmanager
from agent.core.impl import MedicalAgentSystem
from agent.api.models import ChatRequest, ChatResponse, HistoryResponse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("API")

# Global Agent Instance
agent_system = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI.
    Handles startup and shutdown events.
    """
    global agent_system
    logger.info("🚀 Starting up Medical Agent API...")
    
    # Configuration
    CONFIG = {
        "db_path": "/data/home/yihui/LLM/data/medical_embedding",
        "embedding_model_path": "/data/home/yihui/LLM/bge-m3",
        "vllm_api_base": "http://localhost:8000/v1",
        "model_name": "qwen-medical"
    }
    
    try:
        agent_system = MedicalAgentSystem(**CONFIG)
        logger.info("✅ Medical Agent System Initialized Successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Medical Agent System: {e}")
        # We might want to raise here if the system is critical, 
        # but for now let's allow the app to start so we can see the error endpoint
    
    yield
    
    logger.info("🛑 Shutting down Medical Agent API...")
    # Cleanup if necessary

app = FastAPI(title="Medical Agent API", version="1.0.0", lifespan=lifespan)

@app.get("/health")
async def health_check():
    if agent_system:
        return {"status": "healthy", "agent_initialized": True}
    return {"status": "degraded", "agent_initialized": False}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not agent_system:
        raise HTTPException(status_code=503, detail="Agent system not initialized")
    
    try:
        # Use the async chat method
        response_text = await agent_system.achat(request.message, request.session_id, request.mode)
        return ChatResponse(response=response_text, session_id=request.session_id)
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history/{session_id}", response_model=HistoryResponse)
async def get_history_endpoint(session_id: str):
    if not agent_system:
        raise HTTPException(status_code=503, detail="Agent system not initialized")
    
    try:
        history = agent_system.get_history(session_id)
        return HistoryResponse(history=history)
    except Exception as e:
        logger.error(f"Error retrieving history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)
