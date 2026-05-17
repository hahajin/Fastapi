# aichat/StructuralEngineeringChain.py  — 不继承 LLMChain，使用新版 google-genai SDK
import re
from typing import Dict, Any
from sqlalchemy.orm import Session
from google import genai
from google.genai import types
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate

from aichat.chat import ChatMessage


class StructuralEngineeringChain:
    """专门用于结构工程分析的链（基于新版 google-genai SDK）"""

    def __init__(
        self,
        client: genai.Client,
        model_name: str,
        prompt_template: PromptTemplate,
        output_parser: PydanticOutputParser,
        temperature: float = 0.7,
    ):
        self.client = client
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.output_parser = output_parser
        self.temperature = temperature

    def _call_gemini(self, prompt: str) -> str:
        """调用 Gemini 并返回原始文本"""
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=self.temperature),
        )
        return response.text

    def _clean_json_response(self, raw_text: str) -> str:
        """如果模型返回了 markdown 代码块，提取其中的 JSON"""
        if "```json" in raw_text:
            match = re.search(r"```json\n(.*?)\n```", raw_text, re.DOTALL)
            if match:
                return match.group(1)
        elif "```" in raw_text:
            match = re.search(r"```\n(.*?)\n```", raw_text, re.DOTALL)
            if match:
                return match.group(1)
        return raw_text

    def _build_history_context(self, messages, max_messages: int = 12) -> str:
        """从 ChatMessage 对象列表构建对话历史"""
        context = ""
        for msg in messages[-max_messages:]:
            role = "user" if msg.sender == "user" else "AI assistant"
            context += f"{role}: {msg.message}\n"
        return context

    def _is_structural_engineering_query(self, query: str) -> bool:
        """简单关键词判断是否为结构工程问题"""
        structural_keywords = [
            '分析', '结构', '力学', '荷载', '应力', '应变', '有限元',
            '模量', '泊松比', '弹性', '强度', '约束', '网格', 'FEA',
            'analysis', 'structural', 'load', 'stress', 'strain', 'FEA',
            'modulus', 'poisson', 'elastic', 'strength', 'constraint', 'mesh'
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in structural_keywords)

    def process_analysis_request(self, query: str, session_id: str, db: Session) -> Dict[str, Any]:
        """
        处理结构工程分析请求
        :param query: 用户输入的文本
        :param session_id: 会话 ID
        :param db: SQLAlchemy Session (用于查询历史消息)
        :return: 包含 is_structural 和解析结果的字典
        """
        # 1. 快速判断是否为结构工程问题
        if not self._is_structural_engineering_query(query):
            return {"is_structural": False, "response": None}

        # 2. 从数据库获取该会话的历史消息
        history_messages = (
            db.query(ChatMessage)
            .filter_by(session_id=session_id)
            .order_by(ChatMessage.timestamp.asc())
            .all()
        )
        history_context = self._build_history_context(history_messages)

        # 3. 使用 PromptTemplate 构建最终提示
        final_prompt = self.prompt_template.format(
            query=query,
            history=history_context,
            format_instructions=self.output_parser.get_format_instructions()
        )

        # 4. 调用 Gemini
        raw_response = self._call_gemini(final_prompt)
        print("----- LLM raw response:")
        print(raw_response)

        # 5. 清理并解析 JSON
        try:
            cleaned = self._clean_json_response(raw_response)
            parsed = self.output_parser.parse(cleaned)
            return {"is_structural": True, "response": parsed}
        except Exception as e:
            print(f"----- Parsing failed: {str(e)}")
            return {
                "is_structural": True,
                "error": f"解析失败: {str(e)}",
                "raw_response": raw_response,
            }