"""Runtime settings, loaded from environment / .env."""
from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="ADA_", extra="ignore")

    # LLM endpoints
    ollama_host: str = "http://localhost:11434"
    planner_model: str = "qwen2.5:7b-instruct"
    router_model: str = "qwen2.5:3b-instruct"
    embed_model: str = "paraphrase-multilingual-MiniLM-L12-v2"

    # Paths
    projects_dir: Path = Path("./projects")
    default_project: str = "_example"

    # Behavior
    auto_approve_low_risk: bool = False
    max_questions_per_run: int = 0     # 0 = unlimited
    log_level: str = "INFO"

    def project_path(self, name: str) -> Path:
        return self.projects_dir / name


settings = Settings()
