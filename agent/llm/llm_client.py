# backend/agent/llm/llm_client.py
import os
from google import genai
from google.genai import types
from typing import List, Optional

class GeminiClient:
    """封装 google-genai SDK，兼容免费/付费模型"""
    
    def __init__(self, model: str, api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found. Get one from https://aistudio.google.com/")
        
        self.client = genai.Client(api_key=self.api_key)
    
    async def chat_complete(
        self,
        user_input: str,
        history: List[dict],  # [{"role": "user/assistant", "content": "..."}]
        system_prompt: str = "",
        temperature: float = 0.7
    ) -> str:
        """构建完整 prompt 并调用 Gemini"""
        # 构建消息历史（免费模型无多轮 API，需手动拼接）
        messages = []
        if system_prompt:
            messages.append(f"System: {system_prompt}\n")
        
        for msg in history:
            role = "User" if msg["role"] == "user" else "Assistant"
            messages.append(f"{role}: {msg['content']}")
        
        messages.append(f"User: {user_input}\nAssistant:")
        full_prompt = "\n".join(messages)
        
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=2048  # 免费模型有 token 限制
                )
            )
            return response.text.strip()
        except Exception as e:
            # 免费模型常见错误处理
            if "quota" in str(e).lower():
                return "⚠️ 模型调用配额已用尽，请稍后再试。"
            elif "safety" in str(e).lower():
                return "⚠️ 内容触发安全策略，请调整问题描述。"
            raise