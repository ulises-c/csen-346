"""
Local teacher-model server for RTX 3070 (8 GB VRAM).

Loads SocratTeachLLM (9.4 B GLM4-9B fine-tune) in 4-bit (NF4 via bitsandbytes)
and exposes an OpenAI-compatible /v1/chat/completions endpoint so the existing
KELE code can talk to it without modification.

Memory budget on RTX 3070 (8 GB):
  - Model weights at NF4 4-bit: ~5.0 GB
  - Activations + KV cache (max_new_tokens=512): ~1.5 GB
  - Display / OS overhead: ~1.5 GB
  Total: ~8 GB  (tight but workable)

Usage:
  poetry run python -m src.project.serve_teacher
  # or with a custom model path:
  TEACHER_LOCAL_PATH=~/hf_models/SocratTeachLLM poetry run python -m src.project.serve_teacher
"""

import logging
import os
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.project.config import load_env_file

load_env_file()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "SocratTeachLLM"
    messages: list[Message]
    max_tokens: int = 512
    temperature: float = 0.7
    stream: bool = False


def get_runtime_config() -> dict[str, str | int]:
    """Read runtime server configuration from the environment."""
    return {
        "model_path": os.getenv(
            "TEACHER_LOCAL_PATH",
            str(Path.home() / "hf_models" / "SocratTeachLLM"),
        ),
        "host": os.getenv("TEACHER_HOST", "0.0.0.0"),
        "port": int(os.getenv("TEACHER_PORT", "8001")),
        "max_new_tokens": int(os.getenv("MAX_NEW_TOKENS", "512")),
    }


def load_runtime(model_path: str):
    """Load tokenizer and model lazily when the first request arrives."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    log.info("Loading model from: %s", model_path)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model.eval()
    log.info("Model loaded. Starting server...")
    return tokenizer, model


def create_app() -> FastAPI:
    """Create the FastAPI app with lazy model initialization."""
    config = get_runtime_config()
    app = FastAPI(title="KELE Teacher Model Server")
    app.state.runtime = None
    app.state.runtime_config = config

    def get_runtime():
        if app.state.runtime is None:
            app.state.runtime = load_runtime(app.state.runtime_config["model_path"])
        return app.state.runtime

    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [{"id": "SocratTeachLLM", "object": "model"}],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatRequest):
        if req.stream:
            raise HTTPException(status_code=501, detail="Streaming not supported")

        import torch

        tokenizer, model = get_runtime()

        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        try:
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(model.device)
        except Exception:
            prompt = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in messages)
            prompt += "\nAssistant:"
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

        input_len = input_ids.shape[-1]

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
                do_sample=req.temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][input_len:]
        response_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": input_len,
                "completion_tokens": len(new_tokens),
                "total_tokens": input_len + len(new_tokens),
            },
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    runtime_config = get_runtime_config()
    uvicorn.run(
        app,
        host=runtime_config["host"],
        port=runtime_config["port"],
        log_level="info",
    )
