"""
backend/agent/llm/client.py
Async LLM client with provider abstraction and structured output support.
"""

import json
import logging
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from backend.settings import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Async LLM client supporting OpenAI, Anthropic, and Ollama providers.
    Normalizes all responses to a common format.
    """
    
    def __init__(self):
        self.provider = settings.llm_provider
        self.api_key = settings.llm_api_key
        self.model = settings.llm_model
        self.base_url = settings.llm_base_url or self._get_default_base_url()
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens
        self.timeout = httpx.Timeout(60.0)
    
    def _get_default_base_url(self) -> str:
        """Get provider-specific default base URL."""
        if self.provider == "openai":
            return "https://api.openai.com/v1"
        elif self.provider == "anthropic":
            return "https://api.anthropic.com/v1"
        elif self.provider == "ollama":
            return "http://localhost:11434/api"
        raise ValueError(f"Unknown provider: {self.provider}")
    
    async def _make_request(
        self,
        endpoint: str,
        headers: dict,
        json_payload: dict
    ) -> dict:
        """Make async HTTP request with error handling."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/{endpoint}",
                headers=headers,
                json=json_payload
            )
            response.raise_for_status()
            return response.json()
    
    def _normalize_openai_response(self, data: dict) -> dict:
        """Normalize OpenAI chat completion response."""
        choice = data["choices"][0]["message"]
        tool_calls = None
        if "tool_calls" in choice and choice["tool_calls"]:
            tool_calls = [
                {
                    "id": tc["id"],
                    "name": tc["function"]["name"],
                    "arguments": json.loads(tc["function"]["arguments"])
                }
                for tc in choice["tool_calls"]
            ]
        return {
            "content": choice.get("content") or "",
            "tool_calls": tool_calls,
            "usage": data.get("usage", {})
        }
    
    def _normalize_anthropic_response(self, data: dict) -> dict:
        """Normalize Anthropic messages response."""
        content = ""
        tool_calls = None
        
        for block in data.get("content", []):
            if block["type"] == "text":
                content += block["text"]
            elif block["type"] == "tool_use":
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append({
                    "id": block["id"],
                    "name": block["name"],
                    "arguments": block["input"]
                })
        
        return {
            "content": content,
            "tool_calls": tool_calls,
            "usage": {
                "prompt_tokens": data.get("usage", {}).get("input_tokens", 0),
                "completion_tokens": data.get("usage", {}).get("output_tokens", 0),
                "total_tokens": data.get("usage", {}).get("input_tokens", 0) + 
                               data.get("usage", {}).get("output_tokens", 0)
            }
        }
    
    def _normalize_ollama_response(self, data: dict) -> dict:
        """Normalize Ollama chat response."""
        # Ollama doesn't support tool calls natively in the same way
        return {
            "content": data.get("message", {}).get("content", ""),
            "tool_calls": None,
            "usage": {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
            }
        }
    
    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None
    ) -> dict:
        """
        Send messages to LLM and return normalized response.
        
        Args:
            messages: List of {role, content} dicts
            tools: Optional list of tool schemas in OpenAI format
            
        Returns:
            Normalized dict: {content, tool_calls, usage}
        """
        if self.provider == "openai":
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            data = await self._make_request("chat/completions", headers, payload)
            result = self._normalize_openai_response(data)
            
        elif self.provider == "anthropic":
            # Convert messages to Anthropic format
            system_msg = None
            anthropic_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    system_msg = msg["content"]
                else:
                    anthropic_messages.append({
                        "role": "assistant" if msg["role"] == "assistant" else "user",
                        "content": msg["content"]
                    })
            
            payload = {
                "model": self.model,
                "messages": anthropic_messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            if system_msg:
                payload["system"] = system_msg
            if tools:
                payload["tools"] = tools
            
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            }
            data = await self._make_request("messages", headers, payload)
            result = self._normalize_anthropic_response(data)
            
        elif self.provider == "ollama":
            # Ollama uses simpler format
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            }
            headers = {"Content-Type": "application/json"}
            data = await self._make_request("chat", headers, payload)
            result = self._normalize_ollama_response(data)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
        
        # Log usage at DEBUG level
        logger.debug(
            f"LLM call: model={self.model}, tokens={result['usage'].get('total_tokens', 'N/A')}"
        )
        
        return result
    
    async def complete_structured(
        self,
        messages: list[dict],
        response_schema: type[BaseModel],
        max_retries: int = 2
    ) -> BaseModel:
        """
        Call LLM and parse response as structured Pydantic model.
        
        Retries on validation errors by appending the error to messages.
        
        Args:
            messages: Conversation history
            response_schema: Pydantic model to validate against
            max_retries: Number of retry attempts on validation failure
            
        Returns:
            Validated Pydantic model instance
        """
        # Add schema instruction to system prompt
        schema_json = json.dumps(response_schema.model_json_schema(), indent=2)
        instruction = (
            f"\n\nIMPORTANT: Respond ONLY with valid JSON matching this schema:\n{schema_json}\n"
            f"Do not include markdown, explanations, or any other text."
        )
        
        working_messages = messages.copy()
        # Inject instruction into the last user message or system message
        for i in range(len(working_messages) - 1, -1, -1):
            if working_messages[i]["role"] == "user":
                working_messages[i]["content"] += instruction
                break
        
        for attempt in range(max_retries + 1):
            try:
                response = await self.complete(working_messages)
                content = response["content"].strip()
                
                # Extract JSON if wrapped in markdown code blocks
                if content.startswith("```"):
                    content = content.split("```json", 1)[-1].split("```", 1)[0].strip()
                elif content.startswith("{"):
                    # Already JSON
                    pass
                else:
                    # Try to find JSON in the response
                    start = content.find("{")
                    end = content.rfind("}") + 1
                    if start != -1 and end != 0:
                        content = content[start:end]
                
                # Parse and validate
                parsed = json.loads(content)
                return response_schema.model_validate(parsed)
                
            except (json.JSONDecodeError, ValidationError) as e:
                if attempt == max_retries:
                    logger.error(f"Failed to parse structured response after {max_retries + 1} attempts: {e}")
                    raise
                
                # Append error message for retry
                error_msg = (
                    f"Your previous response failed validation: {str(e)}\n"
                    f"Please respond again with valid JSON matching the schema."
                )
                working_messages.append({"role": "user", "content": error_msg})
                logger.warning(f"Structured response validation failed (attempt {attempt + 1}): {e}")
        
        # Should not reach here
        raise RuntimeError("Unexpected error in complete_structured")