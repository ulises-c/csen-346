#!/usr/bin/env python3
"""SFT fine-tuning for SocratTeachLLM v2.

Supports LoRA (bf16 full weights) and QLoRA (4-bit NF4 base).
Device handling is fully delegated to accelerate — no AMD/NVIDIA-specific code.
Loss is masked to assistant turns only via SFTConfig(assistant_only_loss=True).

Config is read from environment variables.  Load them via --config or export beforehand:

    uv run python scripts/train_sft.py --config configs/train-sft-qwen25-7b-lora.env
    uv run python scripts/train_sft.py --config configs/train-sft-qwen25-7b-lora.env --dry-run

Or source the file and run directly:

    source configs/train-sft-qwen25-7b-lora.env && uv run python scripts/train_sft.py

Environment variables (see configs/train-sft-qwen25-7b-lora.env for defaults):

    TRAIN_BASE_MODEL        HF model ID (e.g. Qwen/Qwen2.5-7B-Instruct)
    TRAIN_METHOD            lora | qlora
    LORA_RANK               LoRA rank r (default 32)
    LORA_ALPHA              LoRA alpha (default 64)
    LORA_DROPOUT            LoRA dropout (default 0.05)
    LORA_TARGET_MODULES     Comma-separated list of target module names
    TRAIN_EPOCHS            Number of epochs (default 3)
    TRAIN_LR                Learning rate (default 5e-5)
    TRAIN_BATCH_SIZE        Per-device batch size (default 4)
    TRAIN_GRAD_ACCUM        Gradient accumulation steps (default 4)
    TRAIN_MAX_SEQ_LEN       Max token length per record (default 2048)
    TRAIN_BF16              true | false (default true)
    TRAIN_GRAD_CKPT         true | false; enable for 14B or tight VRAM (default false)
    TRAIN_PREQ              true | false; load from pre-quantized NF4 checkpoint (default false)
    TRAIN_SOURCES           Comma-separated source keys (default socrat-zh,socrat-en)
    TRAIN_OUTPUT_DIR        Output directory for checkpoints (default outputs/sft)
    TRAIN_LOGGING_STEPS     Log every N steps (default 10)
    TRAIN_SAVE_STEPS        Save checkpoint every N steps (default 200)
    TRAIN_EVAL_STEPS        Evaluate every N steps (default 200)
    TRAIN_DATA_SEED         Sampler-order seed, independent of TRAIN_SEED/seed. Unset →
                            falls back to args.seed. Vary per resume to break the
                            data-order-deterministic gfx1201 fault (PR #101 run #13).
    TRAIN_LOG_DATA_ORDER    Diagnostic: log token-length (M) + fingerprint of the first
                            N samples in consumption order, to verify a knob actually
                            reshuffles the post-resume order. 0/unset = off.
    WANDB_PROJECT           W&B project name (default: csen346-sft)
    WANDB_RUN_NAME          W&B run name (default: TRAIN_OUTPUT_DIR basename)
    WANDB_API_KEY           W&B API key (or run `wandb login` once)
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trl import SFTConfig

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _require(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        print(f"ERROR: required env var {key!r} is not set", file=sys.stderr)
        sys.exit(1)
    return val


def _get(key: str, default: str) -> str:
    val = os.environ.get(key)
    return val if val is not None else default


def _bool(val: str) -> bool:
    return val.lower() in ("1", "true", "yes")


def _parse_sources() -> list[str]:
    raw = _get("TRAIN_SOURCES", "socrat-zh,socrat-en")
    return [s.strip() for s in raw.split(",") if s.strip()]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def build_hf_datasets():
    """Load train/eval splits and return HF Dataset objects with a `messages` column.

    Uses load_split_pair to download each source only once (train + test in one pass).
    """
    from datasets import Dataset as HFDataset  # noqa: PLC0415

    from src.project.dataset import load_split_pair  # noqa: PLC0415

    sources = _parse_sources()
    print(f"\nLoading training data  sources={sources}")
    train_records, eval_records = load_split_pair(sources=sources)
    print(f"  train: {len(train_records)} records")
    print(f"  eval:  {len(eval_records)} records")

    train_ds = HFDataset.from_list([{"messages": r["messages"]} for r in train_records])
    eval_ds = HFDataset.from_list([{"messages": r["messages"]} for r in eval_records])
    return train_ds, eval_ds


# ---------------------------------------------------------------------------
# Model + tokenizer
# ---------------------------------------------------------------------------

# Gemma 4 training-compatible chat template. For rendered message bodies it is
# byte-identical to the stock google/gemma-4-31b-it template (verified: turn
# markers <|turn>...<turn|> — token id 105/106, NOT <start/end_of_turn> — and a
# native system role), but wraps assistant content in {% generation %} so TRL's
# assistant_only_loss can find the assistant-token boundary. No tool-calling
# macros — not needed for SFT on Socratic dialogues.
#
# Intentional divergence: the stock template's generation prompt appends a
# `<|channel>thought\n<channel|>` reasoning primer (Gemma 4 is a thinking model);
# this template omits it, training the teacher to answer directly with no thought
# channel. Production serving must prime generation the same way (plain
# `<|turn>model\n`), or it reintroduces a train/serve mismatch this template avoids.
#
# The model-role terminator <turn|> MUST sit INSIDE the {% generation %} block:
# assistant_only_loss masks everything outside it to -100, so a terminator left
# outside gets zero gradient and the model never learns to stop (non-terminating
# repetition collapse — see TRL chat_templates/README.md and gemma3_training.jinja,
# which keep <end_of_turn> inside the block for the same reason). The <|turn>model
# header stays outside (it is a prompt cue, not generated).
_GEMMA4_TRAINING_CHAT_TEMPLATE = """{{- bos_token -}}
{%- for message in messages -%}
{%- set role = 'model' if message['role'] == 'assistant' else message['role'] -%}
{{- '<|turn>' + role + '\n' -}}
{%- if role == 'model' -%}
{% generation %}{{ message['content'] | trim }}{{ '<turn|>\n' }}{% endgeneration %}
{%- else -%}
{{- message['content'] | trim -}}
{{- '<turn|>\n' -}}
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
{{- '<|turn>model\n' -}}
{%- endif -%}"""


def build_model_and_tokenizer():
    """Load base model and tokenizer.  Returns (model, tokenizer)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_model = _require("TRAIN_BASE_MODEL")
    method = _get("TRAIN_METHOD", "lora").lower()
    use_bf16 = _bool(_get("TRAIN_BF16", "true"))

    print(f"\nLoading tokenizer  model={base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Gemma 4 ships a 17K-char chat template (tool-calling macros) without
    # {% generation %} markers, so TRL 1.4+'s SFTTrainer rejects it under
    # assistant_only_loss=True. TRL has hardcoded training-compatible templates
    # for Gemma/Gemma2/Gemma3 but NOT yet for Gemma 4. Swap in a minimal
    # training-compatible template that renders byte-identical to the original
    # (verified for system/user/assistant messages) but adds the generation
    # block markers around assistant content. No tool-calling support — not
    # needed for SFT on Socratic dialogues.
    if "gemma-4" in base_model.lower():
        tokenizer.chat_template = _GEMMA4_TRAINING_CHAT_TEMPLATE
        print("  Patched tokenizer.chat_template for Gemma 4 (added {% generation %} markers)")

    preq = _bool(_get("TRAIN_PREQ", "false"))
    bnb_config = None
    if method == "qlora" and not preq:
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        print(f"  QLoRA: 4-bit NF4, double-quant, {'bf16' if use_bf16 else 'fp16'} compute")
    elif method == "qlora":
        print("  QLoRA: loading pre-quantized NF4 checkpoint — skipping BF16 staging")

    torch_dtype: torch.dtype | None = None
    if method == "lora" and use_bf16:
        torch_dtype = torch.bfloat16

    print(f"  Loading weights  method={method}  dtype={torch_dtype or 'auto'}")

    import os as _os

    _offload_dir = "offload_weights"
    _os.makedirs(_offload_dir, exist_ok=True)
    # QLoRA: force all layers onto the single GPU. device_map="auto" reserves
    # headroom and can dispatch modules to CPU, which bitsandbytes rejects — for
    # live quant it offloads from the BF16 estimate, and for a pre-quantized
    # *dynamic* 4-bit checkpoint (e.g. unsloth bnb, which keeps embeddings /
    # lm_head / per-layer gates in 16-bit) it can offload one of those 16-bit
    # modules. The NF4 model fits in 32 GB VRAM (~22-27 GB peak), so {"": 0} is
    # safe and is the canonical single-GPU QLoRA placement.
    _device_map = {"": 0} if method == "qlora" else "auto"
    # caching_allocator_warmup is an NVIDIA-specific optimisation added in recent
    # transformers. It calls torch.cuda.cudart() which has no ROCm equivalent and
    # raises "Found no NVIDIA driver." Patch it out on HIP/ROCm — the warmup is
    # not required for correctness, only for CUDA allocator pre-heating.
    if torch.version.hip is not None:
        import transformers.modeling_utils as _mu

        _mu.caching_allocator_warmup = lambda *_a, **_kw: None
    # Only pass quantization_config when it is not None. When None is passed
    # explicitly, auto_factory forwards it to AutoConfig.from_pretrained which
    # calls from_dict(..., quantization_config=None), overwriting the
    # checkpoint's valid config.json quantization_config with None — causing
    # supports_quant_method to crash on the pre-quantized (TRAIN_PREQ) path.
    # FA2 Triton backward OOMs on gfx1201: kernel needs 128 KB shared memory,
    # hardware cap is 64 KB. SDPA is stable and uses PyTorch's built-in path.
    _attn_impl = "sdpa"
    _load_kwargs: dict = {
        "attn_implementation": _attn_impl,
        "device_map": _device_map,
        "low_cpu_mem_usage": True,
        "offload_folder": _offload_dir,
        "offload_state_dict": True,
        "trust_remote_code": True,
    }
    if bnb_config is not None:
        _load_kwargs["quantization_config"] = bnb_config
    if torch_dtype is not None:
        _load_kwargs["dtype"] = torch_dtype
    model = AutoModelForCausalLM.from_pretrained(base_model, **_load_kwargs)

    for _vision_attr in ("visual", "vision_tower"):
        if hasattr(model, _vision_attr):
            import torch as _torch

            delattr(model, _vision_attr)
            _torch.cuda.empty_cache()
            print(f"  Dropped {_vision_attr!r} vision encoder to free VRAM")

    torch.cuda.empty_cache()

    if method == "qlora":
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=False,
        )

    return model, tokenizer


# ---------------------------------------------------------------------------
# LoRA config
# ---------------------------------------------------------------------------


def build_lora_config():
    """Build PEFT LoraConfig from env vars."""
    from peft import LoraConfig, TaskType

    r = int(_get("LORA_RANK", "32"))
    alpha = int(_get("LORA_ALPHA", "64"))
    dropout = float(_get("LORA_DROPOUT", "0.05"))
    targets_raw = _get(
        "LORA_TARGET_MODULES",
        "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    # PEFT treats target_modules as a regex when it's a single string (anchored).
    # Use this to scope LoRA to text-only paths on multimodal models like
    # google/gemma-4-31b-it, where vision_tower's Gemma4ClippableLinear wrappers
    # crash get_peft_model with "Target module ... is not supported".
    # Convention: if the value starts with "^" treat it as a regex; otherwise
    # split on comma as a suffix-match list.
    if targets_raw.lstrip().startswith("^"):
        target_modules = targets_raw.strip()
    else:
        target_modules = [t.strip() for t in targets_raw.split(",") if t.strip()]

    print(f"\nLoRA config  r={r}  alpha={alpha}  dropout={dropout}\n  targets: {target_modules}")

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
    )


# ---------------------------------------------------------------------------
# SFT training config
# ---------------------------------------------------------------------------


def build_sft_config(output_dir: str | None = None, use_wandb: bool = False) -> SFTConfig:
    """Build TRL SFTConfig from env vars.

    Args:
        output_dir: Override the output directory. Defaults to TRAIN_OUTPUT_DIR env var.
                    Pass a temp dir in dry_run() to avoid creating the real output_dir
                    as a side-effect of the SFTConfig (TrainingArguments) constructor.
        use_wandb: Enable W&B logging via report_to=["wandb"].
    """
    from trl import SFTConfig

    if output_dir is None:
        output_dir = _get("TRAIN_OUTPUT_DIR", "outputs/sft")
    epochs = int(_get("TRAIN_EPOCHS", "3"))
    lr = float(_get("TRAIN_LR", "5e-5"))
    batch = int(_get("TRAIN_BATCH_SIZE", "4"))
    grad_accum = int(_get("TRAIN_GRAD_ACCUM", "4"))
    max_seq_len = int(_get("TRAIN_MAX_SEQ_LEN", "2048"))
    use_bf16 = _bool(_get("TRAIN_BF16", "true"))
    grad_ckpt = _bool(_get("TRAIN_GRAD_CKPT", "false"))
    logging_steps = int(_get("TRAIN_LOGGING_STEPS", "10"))
    save_steps = int(_get("TRAIN_SAVE_STEPS", "200"))
    eval_steps = int(_get("TRAIN_EVAL_STEPS", "200"))
    max_steps = int(_get("TRAIN_MAX_STEPS", "-1"))
    # CPU-side parallelism. dataset_num_proc fans the one-time chat-template +
    # tokenization map across processes (it is single-process by default and
    # pegs one core on the 77k-record split at startup). dataloader workers
    # prefetch/collate batches off the main process during training. Neither
    # touches the GPU-bound ~70s/step — they only cut startup + feed latency.
    dataset_num_proc = int(_get("TRAIN_DATASET_NUM_PROC", "1"))
    dataloader_workers = int(_get("TRAIN_DATALOADER_WORKERS", "0"))
    # In-training eval can OOM on memory-constrained setups: the eval forward
    # casts logits to fp32 for the loss (transformers/loss/loss_utils.py:58),
    # doubling the logits tensor briefly. On Gemma 4 31B QLoRA with 256K vocab
    # and seq_len=1280, that's an 8.6 GB transient — fits in train (where we
    # have grad_ckpt) but not in eval. Set TRAIN_EVAL_STRATEGY=no to skip
    # in-training eval entirely; the downstream eval pipeline produces
    # paper-grade numbers anyway.
    eval_strategy = _get("TRAIN_EVAL_STRATEGY", "steps").lower()
    # Resume replays the exact sampler order (HF restores it from rng_state.pth +
    # the seedable sampler), so a data-order-deterministic fault re-hits the same
    # step on every resume (PR #101 run #13: stuck at step 22/24 from ckpt-20).
    # data_seed drives the seedable sampler's permutation independently of args.seed;
    # changing it per resume reshuffles which samples land at each step while
    # ignore_data_skip stays False — so the skip still advances through fresh data
    # and coverage is preserved. (ignore_data_skip=True would instead restart the
    # epoch at batch 0 every resume and silently train on a tiny prefix — rejected.)
    # Unset → None → HF falls back to args.seed (stock reproducible behaviour).
    data_seed_raw = _get("TRAIN_DATA_SEED", "")
    data_seed = int(data_seed_raw) if data_seed_raw else None

    run_name = (os.environ.get("WANDB_RUN_NAME") or Path(output_dir).name) if use_wandb else None

    effective_batch = batch * grad_accum
    print(
        f"\nSFT config  output={output_dir}\n"
        f"  epochs={epochs}  lr={lr}  batch={batch}×{grad_accum}={effective_batch}"
        f"  max_length={max_seq_len}  bf16={use_bf16}  grad_ckpt={grad_ckpt}"
        f"  eval_strategy={eval_strategy}  dataset_num_proc={dataset_num_proc}"
        f"  dataloader_workers={dataloader_workers}"
        + (
            f"  wandb_project={os.environ.get('WANDB_PROJECT')}  run={run_name}"
            if use_wandb
            else ""
        )
    )

    # load_best_model_at_end requires eval to function — auto-disable it when
    # eval is off, otherwise SFTConfig raises at construction time.
    return SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=lr,
        data_seed=data_seed,
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=grad_accum,
        max_length=max_seq_len,
        bf16=use_bf16,
        gradient_checkpointing=grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False} if grad_ckpt else None,
        logging_steps=logging_steps,
        max_steps=max_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        eval_strategy=eval_strategy,
        save_total_limit=5,
        dataset_num_proc=dataset_num_proc,
        dataloader_num_workers=dataloader_workers,
        load_best_model_at_end=(eval_strategy != "no"),
        report_to=["wandb"] if use_wandb else "none",
        run_name=run_name,
        assistant_only_loss=True,
    )


# ---------------------------------------------------------------------------
# Dry-run validation
# ---------------------------------------------------------------------------


def dry_run() -> None:
    """Validate config and data without loading model weights."""
    print("\n=== DRY RUN — no model weights will be downloaded ===\n")

    base_model = _require("TRAIN_BASE_MODEL")
    method = _get("TRAIN_METHOD", "lora").lower()
    sources = _parse_sources()
    print(f"Base model : {base_model}")
    print(f"Method     : {method}")
    print(f"Sources    : {sources}")
    print(f"Output dir : {_get('TRAIN_OUTPUT_DIR', 'outputs/sft')}")
    build_lora_config()
    # Use a temp dir so the SFTConfig constructor (TrainingArguments) doesn't
    # create the real output_dir on disk as a side-effect of the dry run.
    with tempfile.TemporaryDirectory() as tmp:
        _prev_bf16 = os.environ.get("TRAIN_BF16")
        os.environ["TRAIN_BF16"] = "false"
        try:
            build_sft_config(output_dir=tmp)
        finally:
            if _prev_bf16 is None:
                del os.environ["TRAIN_BF16"]
            else:
                os.environ["TRAIN_BF16"] = _prev_bf16

    print("\nLoading training data (HF download required)...")
    from src.project.dataset import load_split_pair

    train_records, eval_records = load_split_pair(sources=sources)
    print(f"  train: {len(train_records)} records")
    print(f"  eval:  {len(eval_records)} records")

    # Spot-check first record schema
    if train_records:
        r = train_records[0]
        assert "messages" in r, "missing 'messages' key"
        assert isinstance(r["messages"], list), "'messages' must be a list"
        roles = {m["role"] for m in r["messages"]}
        assert roles <= {"system", "user", "assistant"}, f"unexpected roles: {roles}"
        print(f"\nSample record  id={r['id']}  source={r['source']}")
        print(f"  messages: {len(r['messages'])} turns  roles: {sorted(roles)}")
        if r.get("ground_truth_states"):
            print(f"  states: {r['ground_truth_states'][:4]}")

    print("\n=== Dry run complete — all checks passed ===")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT fine-tuning for SocratTeachLLM v2")
    parser.add_argument(
        "--config",
        metavar="PATH",
        help="path to an env config file (e.g. configs/train-sft-qwen25-7b-lora.env)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate config + data without downloading model weights",
    )
    args = parser.parse_args()

    # Suppress verbose INFO/WARNING churn from transformers/trl during a full
    # SFT run.  ERROR-only (level 1) keeps logs readable; unset via env to see
    # the original level-3 noise for debugging.
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"

    if args.config:
        from src.project.config import load_env_file

        load_env_file(Path(args.config))

    if args.dry_run:
        dry_run()
        return

    from src.project.wandb_tracking import SftTracker

    use_wandb = SftTracker().enabled

    # ── Full training run ────────────────────────────────────────────────────
    from peft import get_peft_model
    from trl import SFTTrainer

    train_ds, eval_ds = build_hf_datasets()
    # With eval_strategy=no the eval split is never evaluated, but SFTTrainer still
    # tokenizes whatever eval_dataset it's handed in __init__ — wasted work that also
    # crashed the multiprocessing map on the RAM-constrained NVIDIA box. Drop it.
    if _get("TRAIN_EVAL_STRATEGY", "steps").lower() == "no":
        eval_ds = None
    model, tokenizer = build_model_and_tokenizer()
    lora_cfg = build_lora_config()
    sft_cfg = build_sft_config(use_wandb=use_wandb)

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    import torch
    from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

    from src.project.hf_callback import HFCheckpointCallback

    class VRAMLogCallback(TrainerCallback):
        def on_log(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            logs=None,
            **kwargs,
        ):
            if not torch.cuda.is_available():
                return
            alloc = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(
                f"  VRAM  alloc={alloc:.1f}GB  reserved={reserved:.1f}GB  total={total:.1f}GB",
                flush=True,
            )

    _hf_repo = os.environ.get("TRAIN_HF_REPO", "").strip()
    _hf_push_every = int(os.environ.get("TRAIN_HF_PUSH_EVERY", "50"))

    callbacks: list[TrainerCallback] = [VRAMLogCallback()]
    if _hf_repo:
        callbacks.append(HFCheckpointCallback(_hf_repo, _hf_push_every))
        print(
            f"HF auto-push enabled → {_hf_repo}  (every {_hf_push_every} steps)",
            flush=True,
        )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    # Auto-resume from latest checkpoint if one exists in output_dir.
    # HF Trainer scans for `checkpoint-*` dirs and picks the highest step,
    # so a crashed/killed run can be relaunched with the same command and
    # pick up where it left off. Worst-case rollback = TRAIN_SAVE_STEPS
    # × step_time (default 200 × 19s = ~63 min on Gemma 4 31B QLoRA).
    # To start over from scratch, rm -rf the checkpoint-* dirs first.
    has_checkpoint = any(Path(sft_cfg.output_dir).glob("checkpoint-*"))

    # DIAGNOSTIC (gfx1201 page fault, PR #101) — BNB_DEQUANT_PROBE=1 logs the memory
    # descriptor of all MatMul4Bit operands in both directions:
    #   [dequant_probe]       — dequantized weight (B), via F.dequantize_4bit
    #   [gemm_operandA]       — grad_output (backward A), via MatMul4Bit.backward
    #   [gemm_forward_input]  — activation input (forward A), via MatMul4Bit.forward
    # probe-2 revealed the faulting GEMM is a grad-checkpoint RECOMPUTED FORWARD pass
    # (8 unpaired [dequant_probe] lines before the fault, no [gemm_operandA]) — the
    # backward hook was blind to it. This extends coverage to MatMul4Bit.forward.
    # Under HIP_LAUNCH_BLOCKING=1 the last pair before the fault ([gemm_forward_input]
    # then [dequant_probe]) are the faulting GEMM's two operands. Compare both against
    # the fault address to land in Bucket #1 (freed operand) or #2 (wild address).
    # Remove once the fault is root-caused.
    if os.environ.get("BNB_DEQUANT_PROBE"):
        import bitsandbytes.functional as _bnbF
        from bitsandbytes.autograd._functions import MatMul4Bit as _MatMul4Bit

        def _descr(t):
            ptr = t.data_ptr()
            end = ptr + t.numel() * t.element_size()
            return (
                f"shape={tuple(t.shape)} stride={tuple(t.stride())} "
                f"contig={t.is_contiguous()} off={t.storage_offset()} "
                f"dtype={t.dtype} ptr=0x{ptr:x} end=0x{end:x}"
            )

        _orig_dequant = _bnbF.dequantize_4bit

        def _dequant_probe(*a, **k):
            out = _orig_dequant(*a, **k)
            print(f"[dequant_probe] {_descr(out)}", flush=True)
            return out

        _bnbF.dequantize_4bit = _dequant_probe

        _orig_forward = _MatMul4Bit.forward

        def _forward_probe(ctx, A, B, out=None, bias=None, quant_state=None):
            print(f"[gemm_forward_input] {_descr(A)}", flush=True)
            return _orig_forward(ctx, A, B, out=out, bias=bias, quant_state=quant_state)

        _MatMul4Bit.forward = staticmethod(_forward_probe)

        _orig_backward = _MatMul4Bit.backward

        def _backward_probe(ctx, grad_output):
            print(f"[gemm_operandA] {_descr(grad_output)}", flush=True)
            return _orig_backward(ctx, grad_output)

        _MatMul4Bit.backward = staticmethod(_backward_probe)
        print(
            "BNB_DEQUANT_PROBE active — logging all MatMul4Bit operands "
            "(forward input, backward grad_output, dequant weight).",
            flush=True,
        )

    # DIAGNOSTIC (gfx1201 page fault, PR #101) — BNB_FORCE_B_CONTIGUOUS=1 forces the
    # dequantized weight (B) row-major before it reaches the Tensile GEMM. The confirmed
    # faulting kernel (MT64x64x64 ISA1201) is selected only when B is col-major (DTVB1,
    # stride=(1,21504)). Forcing contiguous changes the descriptor to DTVB0 → Tensile
    # should dispatch a different tile. Two outcomes are informative:
    #   - fault disappears → col-major B is the trigger; workaround exists (slower, extra copy)
    #   - fault persists with row-major B → stride hypothesis is incomplete; upstream filing
    #     needs revision before submission.
    # Stacks cleanly on top of BNB_DEQUANT_PROBE if both are set.
    if os.environ.get("BNB_FORCE_B_CONTIGUOUS"):
        import bitsandbytes.functional as _bnbF  # noqa: F811 (re-import is intentional)

        _prev_dequant = _bnbF.dequantize_4bit

        def _contiguous_dequant(*a, **k):
            out = _prev_dequant(*a, **k)
            return out.contiguous() if not out.is_contiguous() else out

        _bnbF.dequantize_4bit = _contiguous_dequant
        print(
            "BNB_FORCE_B_CONTIGUOUS active — forcing dequantized weight row-major before GEMM "
            "(tests DTVB1 col-major B as fault trigger).",
            flush=True,
        )

    # DIAGNOSTIC (gfx1201 page fault, PR #101) — TRAIN_LOG_DATA_ORDER=N logs the token
    # length (M, the GEMM row dim that selects the faulting MT64x64x64 tile) and a
    # deterministic fingerprint of the first N samples in CONSUMPTION order. Run #13
    # showed the fault is data-order sticky: every resume replays the same sequence and
    # re-faults at the same step. This verifies whether TRAIN_DATA_SEED actually changes
    # that order across resumes — run twice with different seeds and diff the [data_order]
    # lines. Hooked at the collator (not the sampler): accelerate rewraps the sampler in a
    # SeedableRandomSampler, so the collator is the one point that sees the true consumed
    # order. The fingerprint folds token ids (int hashing is unsalted → stable across
    # processes), so identical [data_order] sequences ⇒ identical data order.
    if os.environ.get("TRAIN_LOG_DATA_ORDER"):
        _order_n = int(os.environ["TRAIN_LOG_DATA_ORDER"])
        _orig_collator = trainer.data_collator
        _order_seen = 0

        def _order_collator(features, *a, **k):
            nonlocal _order_seen
            batch = _orig_collator(features, *a, **k)
            for f in features:
                if _order_seen >= _order_n:
                    break
                ids = f.get("input_ids") or []
                fp = 0
                for v in ids[:64]:
                    fp = (fp * 1000003 + int(v)) & 0xFFFFFFFF
                print(f"[data_order] #{_order_seen} M={len(ids)} fp={fp:08x}", flush=True)
                _order_seen += 1
            return batch

        trainer.data_collator = _order_collator
        print(
            f"TRAIN_LOG_DATA_ORDER active — logging first {_order_n} samples' token length "
            f"+ fingerprint in consumption order (data_seed={trainer.args.data_seed}).",
            flush=True,
        )

    print("\nStarting training...")
    if has_checkpoint:
        print(f"  (resuming from latest checkpoint in {sft_cfg.output_dir})")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    output_dir = sft_cfg.output_dir
    print(f"\nSaving final adapter to {output_dir}/final")
    trainer.save_model(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")
    print("Done.")


if __name__ == "__main__":
    sys.exit(main())
