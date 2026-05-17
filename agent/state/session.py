"""
backend/state/session.py
In-memory session store for AgentState management.

NOTE: This is for development only. For production, replace with Redis:
  - Use redis-py with async support (aioredis)
  - Store serialized AgentState as JSON with TTL
  - Implement the same interface: create, get, update, delete
"""

import time
import uuid
from typing import Optional

from backend.models.schemas import AgentState
from backend.settings import settings

# Session expiry in seconds (2 hours)
SESSION_TTL_SECONDS = 2 * 60 * 60


class SessionStore:
    """In-memory store for agent conversation sessions."""
    
    def __init__(self):
        self._sessions: dict[str, AgentState] = {}
    
    def create_session(self) -> str:
        """Create a new session and return its ID."""
        session_id = str(uuid.uuid4())
        state = AgentState(
            session_id=session_id,
            conversation=[],
            scene_graph=None,
            last_intent=None,
            step="init",
            iteration=0,
            last_accessed=time.time()
        )
        self._sessions[session_id] = state
        return session_id
    
    def get(self, session_id: str) -> AgentState:
        """Get session state, raising KeyError if not found."""
        if session_id not in self._sessions:
            raise KeyError(f"Session not found: {session_id}")
        
        # Cleanup old sessions on access
        self._cleanup_expired()
        
        # Update last accessed timestamp
        state = self._sessions[session_id]
        state.last_accessed = time.time()
        return state
    
    def update(self, state: AgentState) -> None:
        """Persist updated state."""
        state.last_accessed = time.time()
        self._sessions[state.session_id] = state
    
    def delete(self, session_id: str) -> None:
        """Remove a session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
    
    def append_message(self, session_id: str, role: str, content: str) -> None:
        """Append a message to the conversation history."""
        state = self.get(session_id)
        state.conversation.append({"role": role, "content": content})
        self.update(state)
    
    def _cleanup_expired(self) -> None:
        """Remove sessions older than TTL."""
        now = time.time()
        expired = [
            sid for sid, state in self._sessions.items()
            if now - state.last_accessed > SESSION_TTL_SECONDS
        ]
        for sid in expired:
            logger = __import__("logging").getLogger(__name__)
            logger.debug(f"Cleaning up expired session: {sid}")
            del self._sessions[sid]


# Module-level store instance
store = SessionStore()