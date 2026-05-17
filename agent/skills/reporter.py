"""
backend/agent/skills/reporter.py
Skill to generate human-readable summary of agent actions.
"""

import json

from agent.skills.base_skill import BaseSkill
from llm.client import LLMClient
from models.schemas import AgentState

REPORTER_PROMPT = """You are a structural engineering assistant. Write a concise 2–4 sentence summary
of what was done in this step. Mention key elements created or modified, any validation
issues, and what the user should do next. Be factual and professional."""


class ReporterSkill(BaseSkill):
    """Generate human-readable report of agent actions."""
    
    name = "reporter"
    description = "Summarize what was built/modified/analyzed for the user"
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
    
    async def run(self, state: AgentState) -> AgentState:
        # Prepare context for report generation
        graph_json = state.scene_graph.model_dump_json() if state.scene_graph else "{}"
        validation = state.scene_graph.metadata.get("validation") if state.scene_graph else None
        analysis = state.scene_graph.metadata.get("analysis") if state.scene_graph else None
        
        context = {
            "scene_graph": json.loads(graph_json),
            "validation": validation,
            "analysis": analysis
        }
        
        messages = [
            {"role": "system", "content": REPORTER_PROMPT},
            {"role": "user", "content": f"Report on these results:\n{json.dumps(context, indent=2)}"}
        ]
        
        # Generate plain text report
        response = await self.llm.complete(messages)
        report_text = response["content"].strip()
        
        # Store report and mark done
        if state.scene_graph:
            state.scene_graph.metadata["report"] = report_text
        state.step = "done"
        
        return state