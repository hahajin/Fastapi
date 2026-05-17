"""
backend/settings.py
Application configuration using pydantic-settings.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # LLM Configuration
    # GEMINI_API_KEY: str
    GEMINI_MODEL: str = "gemini-2.5-flash"  # 免费模型
    # llm_provider: str = "openai"  # "openai" | "anthropic" | "ollama"
    # llm_api_key: str = ""
    # llm_model: str = "gpt-4o"
    # llm_base_url: str = ""  # Override for Ollama / local endpoints
    # llm_temperature: float = 0.2
    # llm_max_tokens: int = 4096
    
    # CORS & Server
    cors_origins: list[str] = ["http://localhost:5173"]
    
    # Logging
    log_level: str = "INFO"
    
    # Agent limits
    max_agent_iterations: int = 8


# Module-level settings instance
settings = Settings()