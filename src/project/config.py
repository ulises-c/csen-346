"""Configuration loader — reads API credentials from environment variables or .env file."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AgentConfig:
    api_key: str
    base_url: str
    model_name: str
    max_tokens: int = 4096
    disable_thinking: bool = False
    num_ctx: int = 0


@dataclass
class Config:
    consultant: AgentConfig
    teacher: AgentConfig
    debug_mode: bool = True
    max_teaching_rounds: int = 8
    teacher_local_path: str = ""


def repo_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parents[2]


def load_env_file(path: Path | None = None) -> None:
    """Load a .env file into os.environ. Minimal implementation — no dependency needed.

    Supports `source configs/...` lines (repo-root-relative) for composing component
    configs. Paths must be relative to the repository root, matching bash behaviour
    when scripts cd to the project root before sourcing.
    """
    root = repo_root()
    if path is None:
        path = root / ".env"
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("source "):
                included = root / line[7:].strip()
                load_env_file(included)
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            os.environ.setdefault(key, value)


def load_config(experiment: str | None = None, root_dir: Path | None = None) -> Config:
    """Build a Config from environment variables.

    If experiment is given, loads configs/<experiment>.env instead of .env.
    Falls back to .env if no experiment is specified.
    """
    if root_dir is None:
        root_dir = repo_root()

    # Precedence: experiment config wins (loaded first via setdefault), .env
    # fills in anything the experiment didn't set (e.g. CONSULTANT_API_KEY =
    # OpenAI secret). This lets .env hold shared secrets while each experiment
    # chooses its own base_url / model.
    if experiment:
        env_path = root_dir / "configs" / f"{experiment}.env"
        if not env_path.exists():
            raise FileNotFoundError(f"Experiment config not found: {env_path}")
        load_env_file(env_path)
    load_env_file(root_dir / ".env")

    def require(key: str) -> str:
        val = os.environ.get(key)
        if not val:
            raise OSError(f"Missing required environment variable: {key}")
        return val

    return Config(
        consultant=AgentConfig(
            api_key=require("CONSULTANT_API_KEY"),
            base_url=require("CONSULTANT_BASE_URL"),
            model_name=require("CONSULTANT_MODEL_NAME"),
            max_tokens=int(os.environ.get("CONSULTANT_MAX_TOKENS", "4096")),
            disable_thinking=os.environ.get("CONSULTANT_DISABLE_THINKING", "false").lower()
            == "true",
            num_ctx=int(os.environ.get("CONSULTANT_NUM_CTX", "0")),
        ),
        teacher=AgentConfig(
            api_key=os.environ.get("TEACHER_API_KEY", "not-needed"),
            base_url=require("TEACHER_BASE_URL"),
            model_name=require("TEACHER_MODEL_NAME"),
        ),
        debug_mode=os.environ.get("DEBUG_MODE", "true").lower() == "true",
        max_teaching_rounds=int(os.environ.get("MAX_TEACHING_ROUNDS", "8")),
        teacher_local_path=os.environ.get(
            "TEACHER_LOCAL_PATH",
            str(Path.home() / "hf_models" / "SocratTeachLLM"),
        ),
    )
