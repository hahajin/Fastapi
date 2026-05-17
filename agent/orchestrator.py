"""
backend/agent/orchestrator.py
Main agent orchestration loop.
"""

import logging

from agent.skills.entity_extractor import EntityExtractorSkill
from agent.skills.intent_parser import IntentParserSkill
from agent.skills.model_builder import ModelBuilderSkill
from agent.skills.reporter import ReporterSkill
from agent.skills.validator import ValidatorSkill
from llm.client import LLMClient
from models.schemas import AgentResponse, AgentState
from settings import settings
from state.session import store

logger = logging.getLogger(__name__)


# async def run_agent(session_id: str, user_message: str) -> AgentResponse:
#     """
#     Run the full agent pipeline for a user message.
    
#     Args:
#         session_id: Session identifier (creates new if not found)
#         user_message: User's text input
        
#     Returns:
#         AgentResponse with updated state and response data
#     """
#     # Load or create session
#     try:
#         state = store.get(session_id)
#     except KeyError:
#         session_id = store.create_session()
#         state = store.get(session_id)
    
#     # Append user message
#     store.append_message(session_id, "user", user_message)
    
#     # Initialize LLM client
#     llm = LLMClient()
    
#     # Skill pipeline
#     skills = [
#         IntentParserSkill(llm),
#         EntityExtractorSkill(llm),
#         ModelBuilderSkill(llm),
#         ValidatorSkill(),
#         ReporterSkill(llm),
#     ]
    
#     for skill in skills:
#         # Check iteration limit
#         if state.iteration >= settings.max_agent_iterations:
#             logger.warning(f"Max iterations ({settings.max_agent_iterations}) reached for session {session_id}")
#             return AgentResponse(
#                 session_id=session_id,
#                 scene_graph=state.scene_graph,
#                 message="Processing limit reached. Please try a simpler request.",
#                 done=False
#             )
        
#         # Run skill
#         state = await skill.run(state)
#         state.iteration += 1
#         store.update(state)
        
#         # Check for early exit conditions
#         if state.step == "awaiting_clarification":
#             return AgentResponse(
#                 session_id=session_id,
#                 scene_graph=state.scene_graph,
#                 message=state.last_intent.clarification_question or "Please clarify your request.",
#                 done=False
#             )
        
#         if state.step == "validation_failed":
#             # Generate user-friendly error message
#             validation = state.scene_graph.metadata.get("validation", {}) if state.scene_graph else {}
#             errors = validation.get("errors", []) if isinstance(validation, dict) else getattr(validation, "errors", [])
            
#             error_summary = "; ".join(errors[:3])  # Limit to first 3 errors
#             if len(errors) > 3:
#                 error_summary += f" (+{len(errors) - 3} more)"
            
#             return AgentResponse(
#                 session_id=session_id,
#                 scene_graph=state.scene_graph,
#                 message=f"Validation issues found: {error_summary}. Please adjust your model.",
#                 validation=validation if isinstance(validation, dict) else validation.model_dump(),
#                 done=False
#             )
        
#         if state.step == "done":
#             break
    
#     # Build final response
#     validation_data = None
#     if state.scene_graph and "validation" in state.scene_graph.metadata:
#         v = state.scene_graph.metadata["validation"]
#         validation_data = v if isinstance(v, dict) else v.model_dump()
    
#     report = state.scene_graph.metadata.get("report") if state.scene_graph else None
    
#     return AgentResponse(
#         session_id=session_id,
#         scene_graph=state.scene_graph,
#         message=report or "Processing complete.",
#         validation=validation_data,
#         done=True
#     )


# agent/orchestrator.py
from typing import Optional, List
from pydantic import BaseModel, Field
from google import genai
from agent.tools.llm_client import GeminiClient  # 封装免费模型调用
from agent.skills.structural_intent import StructuralIntentSkill
from agent.skills.fea_executor import FEAExecutorSkill
from agent.state import SessionState

class AgentResponse(BaseModel):
    text_response: str
    is_structural: bool = False
    fea_request: Optional[dict] = None
    fea_output: Optional[dict] = None
    error: Optional[str] = None
    skills_trace: List[str] = Field(default_factory=list)
    iteration: int = 1
    confidence: float = 0.0

class AgentOrchestrator:
    """轻量编排器：路由技能 + 管理状态"""
    
    def __init__(self, llm_model: str, llm_api_key: str, fea_api_url: str):
        self.llm = GeminiClient(model=llm_model, api_key=llm_api_key)
        self.fea_executor = FEAExecutorSkill(fea_api_url=fea_api_url)
        self.intent_skill = StructuralIntentSkill(llm_client=self.llm)
    
    async def run(self, user_input: str, session_state: SessionState, context: dict) -> AgentResponse:
        response = AgentResponse(text_response="")
        
        try:
            # 🔹 Step 1: 意图识别（是否结构工程问题？）
            intent_result = await self.intent_skill.detect(user_input, session_state, context)
            response.skills_trace.append("intent")
            response.confidence = intent_result.confidence
            
            if not intent_result.is_structural:
                # 🔹 Step 2a: 普通对话 → 直接调用 LLM
                history = await session_state.get_recent_messages(limit=context.get("history_limit", 12))
                response.text_response = await self.llm.chat_complete(
                    user_input=user_input,
                    history=history,
                    system_prompt="You are a helpful engineering assistant."
                )
                return response
            
            # 🔹 Step 2b: 结构工程流程 → 提取参数 → 调用 FEA
            response.is_structural = True
            response.skills_trace.append("extract")
            
            # 提取 FEA 参数（复用您原有的 Pydantic 模型）
            fea_request = await self.intent_skill.extract_fea_params(
                user_input=user_input,
                history=session_state
            )
            response.fea_request = fea_request.model_dump() if fea_request else None
            
            if not fea_request:
                response.error = "Failed to extract structural parameters"
                response.text_response = "抱歉，未能识别完整的工程参数，请补充更多信息。"
                return response
            
            # 🔹 Step 3: 执行 FEA 分析
            response.skills_trace.append("fea_execute")
            fea_result = await self.fea_executor.execute(fea_request)
            response.fea_output = fea_result
            
            # 🔹 Step 4: 生成自然语言回复
            response.skills_trace.append("summarize")
            response.text_response = await self.llm.chat_complete(
                user_input=f"请用中文总结以下有限元分析结果：{fea_result}",
                history=[],
                system_prompt="你是一名结构工程师，用简洁专业的中文解释分析结果。"
            )
            
            return response
            
        except Exception as e:
            response.error = str(e)
            response.text_response = f"处理过程中出现错误: {str(e)}"
            return response