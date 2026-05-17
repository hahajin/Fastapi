"""
backend/agent/skills/entity_extractor.py
Skill to extract engineering parameters from user input.
"""

from agent.skills.base_skill import BaseSkill
from llm.client import LLMClient
from agent.models.schemas import (
    AgentState, ExtractedEntity, SceneGraph, IntentType
)

ENTITY_EXTRACTOR_PROMPT = """You are a structural engineering parameter extractor.
Extract all measurable entities from the user's message: dimensions (with units),
material types, load values, span lengths, support conditions, element types.
Return a JSON array of ExtractedEntity objects.
If a value has no explicit unit, infer the most likely unit given engineering context."""


class EntityExtractorSkill(BaseSkill):
    """Extract structured engineering entities from user input."""
    
    name = "entity_extractor"
    description = "Extract dimensions, materials, loads, and other engineering parameters"
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
    
    async def run(self, state: AgentState) -> AgentState:
        # Initialize scene_graph if None
        if state.scene_graph is None:
            state.scene_graph = SceneGraph()
        
        # Get context
        last_message = state.conversation[-1] if state.conversation else {"content": ""}
        user_text = last_message.get("content", "") if last_message.get("role") == "user" else ""
        intent = state.last_intent
        
        # Skip extraction if intent is not CREATE/MODIFY/ANALYZE
        if intent and intent.intent not in [IntentType.CREATE, IntentType.MODIFY, IntentType.ANALYZE]:
            return state
        
        messages = [
            {"role": "system", "content": ENTITY_EXTRACTOR_PROMPT},
            {"role": "user", "content": f"Intent: {intent.intent if intent else 'UNKNOWN'}\nMessage: {user_text}"}
        ]
        
        # Extract entities as list
        entities = await self.llm.complete_structured(
            messages=messages,
            response_schema=list[ExtractedEntity]
        )
        
        # Store in scene_graph metadata
        state.scene_graph.metadata["extracted_entities"] = [
            e.model_dump() for e in entities
        ]
        
        return state