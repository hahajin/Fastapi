# backend/routes/ollama_routes.py  — FastAPI + google-genai (new SDK)
import json
import os
import re
from datetime import datetime
from typing import Optional

from google import genai
from google.genai import types
import requests
from fastapi import APIRouter, Depends, HTTPException
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel
from sqlalchemy.orm import Session
from dotenv import load_dotenv

from aichat.StructuralEngineeringChain import StructuralEngineeringChain
from aichat.chat import ChatMessage, ChatSession
from aichat.code_generator import AnalysisType, CodeGenerator, FEAAnalysisRequest
from server.auth import get_current_user
from server.database import get_db
from server.user import User

from agent.orchestrator import AgentOrchestrator

import logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

router = APIRouter(tags=["chat"])

# ─── LLM setup (new google-genai SDK) ──────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# Create the GenAI client
client = genai.Client(api_key=GEMINI_API_KEY)

# 在 router 定义后初始化 orchestrator（单例）
orchestrator = AgentOrchestrator(
    llm_model=MODEL_NAME,  # 如 "gemini-2.5-flash"
    llm_api_key=GEMINI_API_KEY,
    fea_api_url=os.environ.get("FEA_API_URL", "http://localhost:8000/analyze")  ,
)


# LangChain output parser for FEAAnalysisRequest
output_parser = PydanticOutputParser(pydantic_object=FEAAnalysisRequest)

STRUCTURAL_ENG_PROMPT = PromptTemplate(
    template="""
你是一名专业的结构工程师AI助手。请分析以下工程问题，并返回结构化数据以调用有限元分析API。

问题描述:
{query}

对话历史:
{history}

请根据问题描述提取以下信息:
1. 分析类型(静力、模态、屈曲、热力、非线性)
2. 材料属性(弹性模量、泊松比、密度、屈服强度等)
3. 几何信息(尺寸、形状、截面属性等)
4. 荷载条件(类型、大小、位置、方向)
5. 约束条件(支撑类型、位置)

请以JSON格式返回数据，确保数据完整且符合API要求。

{format_instructions}
""",
    input_variables=["query", "history"],
    partial_variables={"format_instructions": output_parser.get_format_instructions()},
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def build_prompt(history_messages: list) -> str:
    """从数据库消息列表构建对话上下文字符串"""
    context = ""
    for msg in history_messages[-12:]:  # 最近12条
        sender = getattr(msg, "sender", None) if not isinstance(msg, dict) else msg.get("sender")
        text = getattr(msg, "message", "") if not isinstance(msg, dict) else msg.get("message", "")
        role = "User" if sender == "user" else "Assistant"
        context += f"{role}: {text.replace(chr(13), '')}\n"
    context += "Assistant: "
    return context


def call_gemini(prompt: str) -> dict:
    """使用新版 SDK 调用 Gemini 生成内容"""
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.7)
        )
        return {"response": response.text}
    except Exception as e:
        return {"error": f"Failed to call Gemini API: {e}"}


def call_fea_api(analysis_request: FEAAnalysisRequest):
    """调用外部 FEA API（与原来相同）"""
    fea_api_url = os.environ.get("FEA_API_URL", "http://localhost:8000/analyze")
    try:
        payload = {
            "analysis_type": analysis_request.analysis_type.value,
            "material": analysis_request.material.dict(),
            "geometry": analysis_request.geometry,
            "loads": [l.dict() for l in analysis_request.loads],
            "constraints": analysis_request.constraints,
            "mesh_settings": {"size": analysis_request.mesh_size},
        }
        r = requests.post(fea_api_url, json=payload, timeout=300)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return f"有限元分析调用失败: {e}"


# ─── Schemas (unchanged) ─────────────────────────────────────────────────────
class ParseRequest(BaseModel):
    message: str
    session_id: int

class CreateSessionRequest(BaseModel):
    user_id: int
    title: Optional[str] = None

class UpdateSessionRequest(BaseModel):
    title: Optional[str] = None


# ─── Routes ──────────────────────────────────────────────────────────────────

@router.get("/health")
def health(current_user: User = Depends(get_current_user)):
    """检查 Gemini API 连通性"""
    try:
        # 新版 SDK 中 models.list 返回一个迭代器，我们只需检查能否获取第一个元素
        list(client.models.list())
        return {"ok": True, "gemini": True}
    except Exception as e:
        raise HTTPException(status_code=503, detail={"ok": False, "error": str(e)})


# @router.post("/parse")
# def parse(
#     data: ParseRequest,
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db),
# ):
    
#     # 🔥 新增：记录请求原始数据（调试用）
#     logger.info(f"📥 Parse request received: user={current_user.id}, raw_body={data.model_dump()}")
    
#     # 🔥 新增：记录 session_id 类型（关键！）
#     logger.info(f"🔍 session_id type: {type(data.session_id).__name__}, value: {data.session_id}")

#     if not data.message.strip():
#         logger.warning(f"❌ Empty message from user {current_user.id}")
#         raise HTTPException(status_code=400, detail="message is required")

#     session = db.query(ChatSession).filter_by(
#         id=int(data.session_id), user_id=current_user.id, is_archived=False
#     ).first()
#     if not session:
#         raise HTTPException(status_code=404, detail="Session not found")

#     try:
#         # 保存用户消息
#         user_msg = ChatMessage(session_id=int(data.session_id), message=data.message, sender="user")
#         db.add(user_msg)
#         db.flush()

#         # 初始化结构工程链（传入新的 client 和 model_name）
#         structural_chain = StructuralEngineeringChain(
#             client=client,
#             model_name=MODEL_NAME,
#             prompt_template=STRUCTURAL_ENG_PROMPT,
#             output_parser=output_parser
#         )

#         # 处理分析请求（现在需要传入 db session 以查询历史）
#         analysis_result = structural_chain.process_analysis_request(
#             query=data.message,
#             session_id=int(data.session_id),
#             db=db
#         )

#         if analysis_result.get("is_structural", False):
#             if "error" in analysis_result:
#                 # 解析失败，使用普通对话模式
#                 history = (db.query(ChatMessage)
#                            .filter_by(session_id=int(data.session_id))
#                            .order_by(ChatMessage.timestamp.asc()).all())
#                 raw_response = call_gemini(build_prompt(history)).get("response", "")
#             else:
#                 # 成功得到 FEAAnalysisRequest
#                 analysis_request = analysis_result["response"]
#                 fea_result = call_fea_api(analysis_request)
#                 apdl_code = CodeGenerator.generate_apdl(analysis_request)
#                 raw_response = f"分析完成。结果: {fea_result}\n\n生成的APDL代码:\n```\n{apdl_code}\n```"
#         else:
#             # 非结构工程问题，普通对话
#             history = (db.query(ChatMessage)
#                        .filter_by(session_id=int(data.session_id))
#                        .order_by(ChatMessage.timestamp.asc()).all())
#             raw_response = call_gemini(build_prompt(history)).get("response", "")

#         # 保存 AI 回复
#         ai_msg = ChatMessage(session_id=int(data.session_id), message=raw_response, sender="ai")
#         db.add(ai_msg)
#         session.updated_at = datetime.utcnow()
#         db.commit()

#         return {"response": raw_response, "session_id": data.session_id}

#     except requests.exceptions.RequestException as e:
#         db.rollback()
#         raise HTTPException(status_code=503, detail=f"Failed to contact external service: {e}")
#     except Exception as e:
#         db.rollback()
#         raise HTTPException(status_code=500, detail=str(e))

@router.post("/parse")
async def parse(  # 🔥 改为 async
    data: ParseRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    logger.info(f"📥 Parse request: user={current_user.id}, session={data.session_id}")
    
    if not data.message.strip():
        raise HTTPException(status_code=400, detail="message is required")
    
    # 1️⃣ 验证 session
    session = db.query(ChatSession).filter_by(
        id=int(data.session_id), user_id=current_user.id, is_archived=False
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # 2️⃣ 保存用户消息（保持现有逻辑）
    user_msg = ChatMessage(session_id=int(data.session_id), message=data.message, sender="user")
    db.add(user_msg)
    db.flush()
    
    try:
        # 🔥 核心改动：调用 Agent Orchestrator
        # 适配您的数据库 session 到 Agent 状态管理
        session_adapter = DatabaseSessionAdapter(
            db_session=db,
            session_id=int(data.session_id),
            user_id=current_user.id
        )
        
        # 执行 Agent 流水线（异步）
        agent_response = await orchestrator.run(
            user_input=data.message,
            session_state=session_adapter,
            context={
                "history_limit": 12,  # 保持您原有的最近12条逻辑
                "user_timezone": "UTC"
            }
        )
        
        # 3️⃣ 处理 Agent 返回结果（兼容现有返回格式）
        if agent_response.error:
            # Agent 内部错误，降级为普通对话
            history = db.query(ChatMessage).filter_by(
                session_id=int(data.session_id)
            ).order_by(ChatMessage.timestamp.asc()).limit(12).all()
            raw_response = await call_gemini_fallback(build_prompt(history))
        elif agent_response.is_structural:
            # ✅ 结构工程流程：拼接 FEA 结果 + APDL 代码
            fea_result = agent_response.fea_output
            apdl_code = CodeGenerator.generate_apdl(agent_response.fea_request)
            raw_response = (
                f"分析完成。结果: {fea_result}\n\n"
                f"生成的 APDL 代码:\n```\n{apdl_code}\n```"
            )
        else:
            # 💬 普通对话：直接使用 Agent 的 LLM 回复
            raw_response = agent_response.text_response
        
        # 4️⃣ 保存 AI 回复（保持现有逻辑）
        ai_msg = ChatMessage(
            session_id=int(data.session_id),
            message=raw_response,
            sender="ai"
        )
        db.add(ai_msg)
        session.updated_at = datetime.now(timezone.utc)
        db.commit()
        
        return {
            "response": raw_response,
            "session_id": data.session_id,
            "agent_meta": {  # 🔥 新增：返回 Agent 执行元数据（前端可选展示）
                "skills_used": agent_response.skills_trace,
                "iteration": agent_response.iteration,
                "confidence": agent_response.confidence
            }
        }
        
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        logger.exception(f"💥 Agent pipeline error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")




@router.get("/sessions")
def get_sessions(
    user_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Unauthorized access")
    sessions = (db.query(ChatSession)
                .filter_by(user_id=user_id, is_archived=False)
                .order_by(ChatSession.created_at.desc()).all())
    result = []
    for s in sessions:
        last = (db.query(ChatMessage).filter_by(session_id=s.id)
                .order_by(ChatMessage.timestamp.desc()).first())
        result.append({
            "id": s.id,
            "title": s.title,
            "preview": last.message if last else "",
            "created_at": s.created_at.isoformat(),
            "updated_at": s.updated_at.isoformat(),
        })
    return result


@router.get("/sessions/{session_id}/messages")
def get_session_messages(
    session_id: int,
    user_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Unauthorized access")
    session = db.query(ChatSession).filter_by(
        id=session_id, user_id=current_user.id, is_archived=False
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    messages = (db.query(ChatMessage).filter_by(session_id=session_id)
                .order_by(ChatMessage.timestamp.asc()).all())
    return [
        {"text": m.message, "sender": m.sender, "timestamp": m.timestamp.isoformat()}
        for m in messages
    ]


@router.post("/sessions", status_code=201)
def create_session(
    data: CreateSessionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    title = data.title or f"Chat {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}"
    try:
        s = ChatSession(user_id=data.user_id, title=title)
        db.add(s)
        db.commit()
        db.refresh(s)
        return {"id": s.id, "title": s.title, "created_at": s.created_at.isoformat()}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{session_id}")
def delete_session(
    session_id: int,
    user_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Unauthorized access")
    session = db.query(ChatSession).filter_by(
        id=session_id, user_id=current_user.id
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    session.is_archived = True
    db.commit()
    return {"message": "Session deleted"}


@router.put("/sessions/{session_id}")
def update_session(
    session_id: int,
    data: UpdateSessionRequest,
    user_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Unauthorized access")
    session = db.query(ChatSession).filter_by(
        id=session_id, user_id=current_user.id
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if data.title:
        session.title = data.title
        db.commit()
    return {"id": session.id, "title": session.title}