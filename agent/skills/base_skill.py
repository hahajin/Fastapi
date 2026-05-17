"""
backend/agent/skills/base_skill.py
Abstract base class for agent skills.
"""

from abc import ABC, abstractmethod

from agent.models.schemas import AgentState


class BaseSkill(ABC):
    """Abstract base class for agent pipeline skills."""
    
    name: str
    description: str
    
    @abstractmethod
    async def run(self, state: AgentState) -> AgentState:
        """Execute the skill and return updated state."""
        pass