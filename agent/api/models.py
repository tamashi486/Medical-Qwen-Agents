from pydantic import BaseModel, Field
from typing import Optional

class ChatRequest(BaseModel):
    message: str = Field(..., description="User's input message")
    session_id: str = Field(default="default_user", description="Session ID for conversation history")
    mode: Optional[str] = Field(default="agent", description="Mode of operation: 'agent' or 'chat'")

class ChatResponse(BaseModel):
    response: str = Field(..., description="Agent's response")
    session_id: str = Field(..., description="Session ID")

class HistoryResponse(BaseModel):
    history: list = Field(..., description="Chat history")
