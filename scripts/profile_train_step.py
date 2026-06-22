#!/usr/bin/env python
"""De-risk the FA2 bet: profile where a real Stage 2 training step spends time.

The gfx1201 ~70 s/step is ~9× above the raw-TFLOPS floor. The handoff attributes
the gap to "SDPA attention is the bottleneck, not GEMM" — but that was *inferred*
from hipBLASLt=1 showing no gain, never measured directly. This script settles it.

It does two things:

  1. SDPA-backend probe — at Gemma 4's real attention shape (head_dim=256, which
     ROCm flash/mem-efficient kernels typically cap below at 128), times each SDPA
     backend's fwd+bwd. If flash is unavailable and the math fallback is what runs,
     attention is slow *by construction* and FA2 (if its backward fit) would help.

  2. Real-step profile — runs a few REAL training steps (same model load, LoRA,
     collator, assistant_only_loss, grad-ckpt as scripts/train_sft.py) under
     torch.profiler and reports the attention subtree's share of device time
     (via the scaled_dot_product_attention op's *total* cuda time — its self time
     is ~0, so sorting by self-time hides it), plus the raw top-40 kernel table.

Decision it informs:
  - attention dominant      → patching the flash-attn Triton AMD backward
    (bwd_prefill_split.py block sizes → fit 64 KB LDS) could collapse the step
    time. FA2 is worth the spike.
  - NF4 dequant / GEMM dominant → FA2 buys little even with a fitting backward.
    Don't sink time into the kernel patch.

Run on the R9700 (HIP/ROCm profile; ProfilerActivity.CUDA captures HIP kernels):

    make profile-gemma4-31b
    # or directly:
    TORCH_USE_HIPBLASLT=1 PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8 \
      uv run --no-sync python scripts/profile_train_step.py \
      --config configs/train-sft-stage2-gemma4-31b.env

Profiles 4 optimizer steps (schedule wait=1/warmup=2/active=3) with grad-accum
forced to 1 (the attention-vs-GEMM ratio is invariant to accumulation, but ga=1
cuts recorded events ~16× so post-processing finishes in seconds). Pass --trace
to also write outputs/profile-gemma4-31b/trace.json (slow; the printed table is
the deliverable, and on a headless box there's no browser to open it anyway).
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train_sft import (  # noqa: E402
    build_hf_datasets,
    build_lora_config,
    build_model_and_tokenizer,
    build_sft_config,
)

_PROBE_CODE = """
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

print("\\n" + "=" * 100)
print("SDPA BACKEND PROBE — Gemma 4 31B shape: head_dim=256, seq=1280, 32 q / 16 kv heads, bf16, causal")
print("=" * 100)
print(
    f"  allowed: flash={torch.backends.cuda.flash_sdp_enabled()}  "
    f"mem_efficient={torch.backends.cuda.mem_efficient_sdp_enabled()}  "
    f"math={torch.backends.cuda.math_sdp_enabled()}"
)

b, hq, hkv, s, d = 1, 32, 16, 1280, 256
for name, backend in (
    ("flash", SDPBackend.FLASH_ATTENTION),
    ("mem_efficient", SDPBackend.EFFICIENT_ATTENTION),
    ("math", SDPBackend.MATH),
):
    q = k = v = None
    try:
        q = torch.randn(b, hq, s, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(b, hkv, s, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(b, hkv, s, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)

        def run(q, k, v):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
            out.sum().backward()

        with sdpa_kernel([backend]):
            for _ in range(2):
                run(q, k, v)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            iters = 5
            for _ in range(iters):
                run(q, k, v)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / iters * 1000
        print(f"  {name:<14} OK      fwd+bwd {ms:7.2f} ms/call")
    except Exception as e:
        print(f"  {name:<14} FAILED  {str(e).splitlines()[0][:88]}")
    finally:
        del q, k, v
        torch.cuda.empty_cache()
print(
    "\\n  Read: if flash FAILED (head_dim=256 unsupported) and math is what runs, attention is"
    "\\n  on the slow rocBLAS path — FA2 would help it *if* its backward could fit 64 KB LDS."
)
"""


def sdpa_backend_probe() -> None:
    """Time each SDPA backend at Gemma 4 31B's real attention shape.

    Runs in a subprocess so Triton/HIP allocations (outside PyTorch's caching
    allocator) are fully released before the model loads in the parent process.
    """
    import subprocess

    result = subprocess.run(
        [sys.executable, "-c", _PROBE_CODE],
        env=os.environ.copy(),
    )
    if result.returncode != 0:
        print("  (probe subprocess failed — continuing to real-step profile)")


def _self_device_time(e) -> float:
    # torch 2.11 renamed the per-event attribute to self_device_time_total;
    # self_cuda_time_total returns 0 on recent builds. Try names in order.
    for attr in ("self_device_time_total", "self_cuda_time_total", "self_cpu_time_total"):
        v = getattr(e, attr, 0.0)
        if v:
            return float(v)
    return 0.0


def _classify(name: str) -> str:
    n = name.lower()
    if "dequant" in n:
        return "nf4_dequant"
    if "cijk" in n or "gemm" in n:
        return "gemm"
    if "attn_fwd" in n or "attention" in n or n.startswith("bwd_kernel"):
        return "attention"
    return "other"


def report(prof, torch) -> None:
    has_cuda = torch.cuda.is_available()
    sort_key = "self_cuda_time_total" if has_cuda else "self_cpu_time_total"

    # key_averages() re-aggregates every raw event — call it ONCE and reuse.
    ka = prof.key_averages()

    print("\n" + "=" * 100)
    print(f"TOP 40 OPS BY {sort_key}")
    print("=" * 100)
    print(ka.table(sort_by=sort_key, row_limit=40))

    # Bucket over LEAF GPU kernels only (names without "::") so an aten op's self
    # time isn't double-counted against the device kernels it launches. Skip the
    # ProfilerStep* span marker — it overlaps the whole step and isn't work.
    buckets = {"nf4_dequant": 0.0, "gemm": 0.0, "attention": 0.0, "other": 0.0}
    for e in ka:
        if "::" in e.key or e.key.startswith("ProfilerStep"):
            continue
        buckets[_classify(e.key)] += _self_device_time(e)
    total = sum(buckets.values()) or 1.0

    print("\n" + "=" * 100)
    print("STEP TIME BY KERNEL CLASS (leaf GPU kernels; aten-op rows & ProfilerStep excluded)")
    print("=" * 100)
    for name, val in sorted(buckets.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<14} {val / total * 100:5.1f}%   ({val / 1e3:8.1f} ms)")
    print(
        "\nDecision:\n"
        "  attention dominant         → FA2 backward-kernel patch worth the spike\n"
        "  nf4_dequant / gemm dominant → attention isn't the cost; skip FA2\n"
        "  (nf4_dequant is the inherent 4-bit QLoRA tax: every matmul dequants NF4→bf16.)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile a real Stage 2 training step")
    parser.add_argument("--config", metavar="PATH", required=True, help="env config file")
    parser.add_argument("--steps", type=int, default=4, help="optimizer steps to run")
    parser.add_argument(
        "--trace",
        action="store_true",
        help="also export a chrome trace (slow JSON serialization; the printed table is enough)",
    )
    args = parser.parse_args()

    from src.project.config import load_env_file

    load_env_file(Path(args.config))

    # Cap the run to a few steps and keep eval off, regardless of the config.
    os.environ["TRAIN_MAX_STEPS"] = str(args.steps)
    os.environ["TRAIN_EVAL_STRATEGY"] = "no"
    # Force grad-accum=1 for profiling. The attention-vs-GEMM *ratio* is identical
    # regardless of accumulation, but ga=16 records 16× the kernel events per step —
    # which is what makes key_averages() + chrome-trace export crawl for minutes.
    os.environ["TRAIN_GRAD_ACCUM"] = "1"

    import torch
    from peft import get_peft_model
    from torch.profiler import ProfilerActivity, profile, schedule
    from transformers import TrainerCallback
    from trl import SFTTrainer

    out_dir = "outputs/profile-gemma4-31b"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    sdpa_backend_probe()

    train_ds, _ = build_hf_datasets()
    # SFTTrainer tokenizes the WHOLE train split single-threaded at construction
    # (dataset_num_proc=1) before step 1. Profiling only needs a few steps' worth
    # of records — slice to avoid minutes of pointless full-dataset preprocessing.
    bs = int(os.environ.get("TRAIN_BATCH_SIZE", "1"))
    ga = int(os.environ.get("TRAIN_GRAD_ACCUM", "16"))
    n_records = bs * ga * (args.steps + 2)
    train_ds = train_ds.select(range(min(n_records, len(train_ds))))
    print(
        f"\nProfiling on a {len(train_ds)}-record slice (full split skipped — only {args.steps} steps run)"
    )
    model, tokenizer = build_model_and_tokenizer()
    lora_cfg = build_lora_config()
    with tempfile.TemporaryDirectory() as tmp:
        sft_cfg = build_sft_config(output_dir=tmp, use_wandb=False)

        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

        class ProfStep(TrainerCallback):
            def __init__(self, prof):
                self.prof = prof

            def on_step_end(self, *a, **kw):
                self.prof.step()

        trainer = SFTTrainer(
            model=model,
            args=sft_cfg,
            train_dataset=train_ds,
            processing_class=tokenizer,
        )

        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        print(f"\nProfiling {args.steps} optimizer steps (schedule wait=1/warmup=2/active=3)...")
        with profile(
            activities=activities,
            schedule=schedule(wait=1, warmup=2, active=3),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        ) as prof:
            trainer.add_callback(ProfStep(prof))
            trainer.train()

    report(prof, torch)
    if args.trace:
        trace_path = f"{out_dir}/trace.json"
        print(f"\nExporting chrome trace to {trace_path} (slow — serializes every event)...")
        prof.export_chrome_trace(trace_path)
        print(f"Chrome trace: {trace_path}  (open in chrome://tracing or perfetto.dev)")


if __name__ == "__main__":
    main()
