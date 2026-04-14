"""Configuration loader — reads API credentials from environment variables or .env file."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AgentConfig:
    api_key: str
    base_url: str
    model_name: str


@dataclass
class Config:
    consultant: AgentConfig
    teacher: AgentConfig
    debug_mode: bool = True
    max_teaching_rounds: int = 8


def load_env_file(path: Path | None = None) -> None:
    """Load a .env file into os.environ. Minimal implementation — no dependency needed."""
    if path is None:
        path = Path(__file__).resolve().parents[2] / ".env"
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            os.environ.setdefault(key, value)


def load_config() -> Config:
    """Build a Config from environment variables (loads .env first)."""
    load_env_file()

    def require(key: str) -> str:
        val = os.environ.get(key)
        if not val:
            raise EnvironmentError(f"Missing required environment variable: {key}")
        return val

    return Config(
        consultant=AgentConfig(
            api_key=require("CONSULTANT_API_KEY"),
            base_url=require("CONSULTANT_BASE_URL"),
            model_name=require("CONSULTANT_MODEL_NAME"),
        ),
        teacher=AgentConfig(
            api_key=os.environ.get("TEACHER_API_KEY", "not-needed"),
            base_url=require("TEACHER_BASE_URL"),
            model_name=require("TEACHER_MODEL_NAME"),
        ),
        debug_mode=os.environ.get("DEBUG_MODE", "true").lower() == "true",
        max_teaching_rounds=int(os.environ.get("MAX_TEACHING_ROUNDS", "8")),
    )
