"""
backend/agent/skills/model_builder.py
Skill to construct/update the SceneGraph from extracted entities.
"""

import json

from agent.skills.base_skill import BaseSkill
from llm.client import LLMClient
from models.schemas import AgentState, SceneGraph

MODEL_BUILDER_PROMPT = """You are a structural model builder. Given extracted engineering entities and the current
scene graph (which may be empty for new models), return a complete updated SceneGraph JSON.
Rules:
- Every node must have a unique id (use element_type + sequential number, e.g. 'beam_1').
- Positions are in metres, origin at (0,0,0).
- For beams/columns, add 'length_m', 'width_mm', 'height_mm', 'material' to properties.
- Edges represent physical connections (beam-to-column joint = 'RIGID_CONNECTION').
- Preserve all existing node/edge ids when modifying; only change what was requested.
Respond only with valid JSON matching the SceneGraph schema."""


class ModelBuilderSkill(BaseSkill):
    """Build or modify the structural SceneGraph from extracted entities."""
    
    name = "model_builder"
    description = "Construct or update the structural model SceneGraph"
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
    
    async def run(self, state: AgentState) -> AgentState:
        # Ensure we have a scene_graph
        if state.scene_graph is None:
            state.scene_graph = SceneGraph()
        
        # Prepare context for LLM
        entities = state.scene_graph.metadata.get("extracted_entities", [])
        current_graph_json = state.scene_graph.model_dump_json() if state.scene_graph.nodes or state.scene_graph.edges else "null"
        
        intent = state.last_intent
        intent_type = intent.intent if intent else "UNKNOWN"
        
        messages = [
            {"role": "system", "content": MODEL_BUILDER_PROMPT},
            {"role": "user", "content": f"""Intent: {intent_type}
Extracted entities: {json.dumps(entities, indent=2)}
Current SceneGraph: {current_graph_json}

Build the updated SceneGraph:"""}
        ]
        
        # Generate updated SceneGraph
        new_graph = await self.llm.complete_structured(
            messages=messages,
            response_schema=SceneGraph
        )
        
        state.scene_graph = new_graph
        return state