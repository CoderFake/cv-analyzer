import os
from dataclasses import field
from typing import List, Optional, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # PostgreSQL
    DATABASE_URL: str
    DB_USER: str
    DB_PASSWORD: str
    DB_HOST: str
    DB_PORT: str
    DB_NAME: str

    # JWT
    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int
    ALGORITHM: str

    # R2 Storage
    R2_ACCOUNT_ID: str
    R2_ACCESS_KEY_ID: str
    R2_SECRET_ACCESS_KEY: str
    R2_BUCKET_NAME: str
    R2_PUBLIC_ENDPOINT: str

    # LLM
    LLM_MODEL_PATH: str
    LLM_MAX_TOKENS: int
    LLM_TEMPERATURE: float

    # Web Search
    SEARCH_RESULTS_PER_QUERY: int = 5
    SEARCH_BACKEND: str = "llm_web_search"
    SEARXNG_URL: str = ""

    # App
    ENVIRONMENT: str
    DEBUG: bool
    DOCS_URL: Optional[str] = "/docs"
    REDOC_URL: Optional[str] = "/redoc"
    OPENAPI_URL: Optional[str] = "/openapi.json"
    CORS_ORIGINS: Union[str, List[str]] = field(default_factory=lambda: ["*"])

    #search
    SEARCH_BACKEND: str = "duckduckgo"
    SEARXNG_URL: str = ""
    SEARXNG_LANGUAGE: str = "vi"

    APP_NAME: Optional[str] = None
    APP_DESCRIPTION: Optional[str] = None
    APP_VERSION: Optional[str] = None
    MODEL_NAME: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=f".env.{os.getenv('ENVIRONMENT', 'dev')}",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    @field_validator("CORS_ORIGINS")
    def assemble_cors_origins(cls, v):
        if isinstance(v, str) and v != "*":
            return [origin.strip() for origin in v.split(",")]
        return v


settings = Settings()