"""
backend/agent/skills/intent_parser.py
Skill to classify user intent from natural language input.
"""

from agent.skills.base_skill import BaseSkill
from llm.client import LLMClient
from models.schemas import AgentState, ParsedIntent, IntentType

INTENT_PARSER_PROMPT = """You are an intent classifier for a structural engineering CAD assistant.
Classify the user's message into one of: CREATE, MODIFY, DELETE, QUERY, ANALYZE, UNKNOWN.
Set clarification_needed=true and provide clarification_question if the intent is ambiguous.
Respond only with valid JSON matching the ParsedIntent schema."""


class IntentParserSkill(BaseSkill):
    """Parse user message to determine their intent."""
    
    name = "intent_parser"
    description = "Classify user intent and detect ambiguity"
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
    
    async def run(self, state: AgentState) -> AgentState:
        # Get the last user message
        if not state.conversation:
            state.last_intent = ParsedIntent(
                intent=IntentType.UNKNOWN,
                confidence=0.0,
                raw_text="",
                clarification_needed=True,
                clarification_question="I didn't receive a message. What would you like to do?"
            )
            state.step = "awaiting_clarification"
            return state
        
        last_message = state.conversation[-1]
        if last_message.get("role") != "user":
            state.last_intent = ParsedIntent(
                intent=IntentType.UNKNOWN,
                confidence=0.0,
                raw_text=last_message.get("content", ""),
                clarification_needed=True,
                clarification_question="I'm not sure what you'd like me to do. Could you clarify?"
            )
            state.step = "awaiting_clarification"
            return state
        
        user_text = last_message["content"]
        
        # Call LLM for structured intent parsing
        messages = [
            {"role": "system", "content": INTENT_PARSER_PROMPT},
            {"role": "user", "content": user_text}
        ]
        
        parsed = await self.llm.complete_structured(
            messages=messages,
            response_schema=ParsedIntent
        )
        
        # Update state with parsed intent
        parsed.raw_text = user_text  # Ensure raw text is preserved
        state.last_intent = parsed
        
        if parsed.clarification_needed:
            state.step = "awaiting_clarification"
        
        return state