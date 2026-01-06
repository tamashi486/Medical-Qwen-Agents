from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class AgentInterface(ABC):
    """
    Abstract Interface for Agent Systems.
    Defines the contract for any agent implementation.
    """

    @abstractmethod
    def chat(self, user_input: str, session_id: str = "default_user") -> str:
        """
        Synchronous chat method.
        
        Args:
            user_input: The user's message.
            session_id: Unique identifier for the session (for memory).
            
        Returns:
            The agent's response as a string.
        """
        pass

    @abstractmethod
    async def achat(self, user_input: str, session_id: str = "default_user", mode: str = "agent") -> str:
        """
        Asynchronous chat method.
        
        Args:
            user_input: The user's message.
            session_id: Unique identifier for the session (for memory).
            mode: 'agent' for full agent with tools, 'chat' for simple LLM chat.
            
        Returns:
            The agent's response as a string.
        """
        pass

    @abstractmethod
    def get_history(self, session_id: str) -> list:
        """
        Retrieve chat history for a session.
        
        Args:
            session_id: Unique identifier for the session.
            
        Returns:
            List of messages in the format [{"role": "user", "content": "..."}, ...].
        """
        pass
