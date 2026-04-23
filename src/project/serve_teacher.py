"""
Local teacher-model server for SocratTeachLLM (GLM4-9B fine-tune).

Exposes an OpenAI-compatible /v1/chat/completions endpoint so the KELE
evaluation pipeline can call the teacher without modification.

Tested hardware (all have ≥24 GB VRAM — no quantization needed):
  - AMD Radeon AI PRO R9700  32 GB  (ROCm, TEACHER_USE_BNB=false)
  - NVIDIA RTX 5090          32 GB  (CUDA, TEACHER_USE_BNB=false)
  - NVIDIA RTX 3090          24 GB  (CUDA, TEACHER_USE_BNB=false)

For a GPU with <12 GB VRAM, set TEACHER_USE_BNB=true to load in NF4 4-bit
(requires bitsandbytes; not supported on ROCm gfx1201).

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

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from src.project.config import load_env_file

load_env_file()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

import transformers
transformers.logging.set_verbosity_info()
transformers.logging.enable_progress_bar()


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "SocratTeachLLM"
    messages: list[Message]
    max_tokens: int = 2048
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
        "api_key": os.getenv("TEACHER_SERVER_API_KEY", ""),
    }


def _patch_transformers_tied_weights() -> None:
    """Patch two sites in transformers 5.x that access all_tied_weights_keys,
    which ChatGLM's trust_remote_code class doesn't implement.

    Site 1: get_total_byte_count (caching_allocator_warmup, fires with device_map)
    Site 2: _move_missing_keys_from_meta_to_device (_finalize_model_loading, always fires)
    """
    def _ensure(model) -> None:
        if not hasattr(model, "all_tied_weights_keys"):
            tied = getattr(model, "_tied_weights_keys", None) or []
            model.all_tied_weights_keys = {k: None for k in tied}

    try:
        import transformers.modeling_utils as mu

        if hasattr(mu, "get_total_byte_count"):
            orig_gtbc = mu.get_total_byte_count
            def _patched_gtbc(model, device_map):
                _ensure(model)
                return orig_gtbc(model, device_map)
            mu.get_total_byte_count = _patched_gtbc

        from transformers import PreTrainedModel
        if hasattr(PreTrainedModel, "_move_missing_keys_from_meta_to_device"):
            orig_mmk = PreTrainedModel._move_missing_keys_from_meta_to_device
            def _patched_mmk(self, missing_keys, *args, **kwargs):
                _ensure(self)
                return orig_mmk(self, missing_keys, *args, **kwargs)
            PreTrainedModel._move_missing_keys_from_meta_to_device = _patched_mmk

        log.info("Patched transformers for ChatGLM trust_remote_code compatibility")
    except Exception as exc:
        log.warning("Could not patch transformers: %s", exc)


def load_runtime(model_path: str):
    """Load tokenizer and model lazily when the first request arrives."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("Loading model from: %s", model_path)

    use_bnb = os.getenv("TEACHER_USE_BNB", "true").lower() not in ("0", "false", "no")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Patch before from_pretrained: two sites in transformers 5.x access
    # all_tied_weights_keys which ChatGLM trust_remote_code doesn't implement.
    _patch_transformers_tied_weights()

    kwargs: dict = {"trust_remote_code": True}

    if use_bnb:
        from transformers import BitsAndBytesConfig
        log.info("Loading with bitsandbytes 4-bit (NF4) on %s", device)
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        kwargs["dtype"] = torch.float16
        kwargs["device_map"] = {"": 0}
    else:
        dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
        log.info("Loading without bitsandbytes, dtype=%s, device=%s", dtype, device)
        kwargs["dtype"] = dtype
        # No device_map — avoids caching_allocator_warmup which calls
        # model.all_tied_weights_keys (not implemented by ChatGLM). We move
        # the model to the target device manually after loading.

    log.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    log.info("Tokenizer loaded. Loading model weights (this may take 1-2 min)...")
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

    if not use_bnb:
        log.info("Moving model to %s...", device)
        model = model.to(device)

    # ChatGLM's generation_config.json sets max_length=128000; clear it so passing
    # max_new_tokens to generate() doesn't trigger a noisy conflict warning.
    if hasattr(model, "generation_config") and model.generation_config.max_length is not None:
        model.generation_config.max_length = None

    # Patch ChatGLMModel.forward to convert DynamicCache → legacy tuple format.
    # Transformers 5.x passes DynamicCache everywhere; ChatGLM's trust_remote_code
    # was written for the old tuple-of-tuples format and does past_key_values[i][j]
    # subscripting that DynamicCache doesn't support.
    _patch_chatglm_dynamic_cache(model)

    log.info("Weights loaded. Calling model.eval()...")
    model.eval()
    log.info("Model ready on device: %s", next(model.parameters()).device)
    return tokenizer, model


def _patch_chatglm_dynamic_cache(model) -> None:
    """Convert transformers 5.x DynamicCache → legacy tuple-of-tuples at ChatGLMModel.forward entry.

    In transformers 5.x, DynamicCache stores KV per layer as DynamicLayer objects
    (cache.layers[i].keys / .values). ChatGLM's trust_remote_code was written for
    the old format: a tuple of (key, value) tensors per layer. We convert at the
    model boundary so ChatGLM's internal get_masks / encoder code is unchanged.
    """
    try:
        from transformers.cache_utils import DynamicCache

        TransformerClass = type(model.transformer)
        orig_forward = TransformerClass.forward

        def _forward_compat(self, *args, **kwargs):
            pkv = kwargs.get("past_key_values")
            if isinstance(pkv, DynamicCache):
                # get_seq_length() == 0 means layers are pre-allocated but not yet
                # populated (keys/values are None). Pass None so ChatGLM initialises
                # kv_caches = [None]*n as it would on a cold first pass.
                if pkv.get_seq_length() > 0:
                    kwargs["past_key_values"] = tuple(
                        (layer.keys, layer.values) for layer in pkv.layers
                    )
                else:
                    kwargs["past_key_values"] = None
            return orig_forward(self, *args, **kwargs)

        TransformerClass.forward = _forward_compat
        log.info("Patched %s.forward for DynamicCache compatibility", TransformerClass.__name__)
    except Exception as exc:
        log.warning("Could not patch ChatGLM forward for DynamicCache: %s", exc)


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

    def require_api_key(
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> None:
        expected_api_key = str(app.state.runtime_config.get("api_key", "")).strip()
        if not expected_api_key:
            return

        bearer_token = ""
        if authorization:
            scheme, _, token = authorization.partition(" ")
            if scheme.lower() == "bearer":
                bearer_token = token.strip()

        if bearer_token == expected_api_key or x_api_key == expected_api_key:
            return

        raise HTTPException(status_code=401, detail="Unauthorized")

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "model_loaded": app.state.runtime is not None}

    @app.get("/v1/models")
    def list_models(
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ):
        require_api_key(authorization=authorization, x_api_key=x_api_key)
        return {
            "object": "list",
            "data": [{"id": "SocratTeachLLM", "object": "model"}],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(
        req: ChatRequest,
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ):
        require_api_key(authorization=authorization, x_api_key=x_api_key)
        if req.stream:
            raise HTTPException(status_code=501, detail="Streaming not supported")

        import torch

        tokenizer, model = get_runtime()

        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        try:
            result = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            # transformers 5.x returns BatchEncoding; older versions returned a tensor
            if hasattr(result, "input_ids"):
                input_ids = result.input_ids.to(model.device)
            elif isinstance(result, dict):
                input_ids = result["input_ids"].to(model.device)
            else:
                input_ids = result.to(model.device)
        except Exception:
            prompt = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in messages)
            prompt += "\nAssistant:"
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

        input_len = input_ids.shape[-1]

        max_new_tokens = min(req.max_tokens, app.state.runtime_config["max_new_tokens"])
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
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


def main() -> None:
    """CLI entry point for serving the local teacher model."""
    import uvicorn

    runtime_config = get_runtime_config()
    uvicorn.run(
        app,
        host=runtime_config["host"],
        port=runtime_config["port"],
        log_level="info",
    )


if __name__ == "__main__":
    main()
