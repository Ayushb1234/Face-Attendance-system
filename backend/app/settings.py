from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # DB (defaults to local SQLite file)
    DATABASE_URL: str = "sqlite:///./dev.db"

    # CORS (allow Next.js dev by default)
    CORS_ORIGINS: List[str] = ["http://localhost:3000"]

    # Vector backend (keep "local"; set QDRANT_URL to enable qdrant)
    MATCHER_BACKEND: str = "local"
    QDRANT_URL: Optional[str] = None
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION: str = "faces"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
