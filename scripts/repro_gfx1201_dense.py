#!/usr/bin/env python3
"""
Dense-VA reproducer for the gfx1201 ISA1201 Tensile GEMM page fault.

Why the original repro_gfx1201_rocblas.py doesn't reliably crash:
  The faulting kernel computes a wild host-VA roughly 6–7 GB beyond the B
  operand end.  In a small process (sparse VA) that address typically lands
  in completely unregistered memory; the GPU page-walk may silently skip it
  or raise a different error class.  In production (19 GB model loaded) the
  address lands in HMM-registered but page-not-present VA → observable
  "Memory access fault … Page not present" every time.

Fix: pre-allocate a ~20 GB "filler" on the GPU *before* building the layer.
  The filler occupies a large VA band.  The layer is then placed at a higher
  VA; the wild address (B.ptr + ~6–7 GB) is more likely to fall inside a
  filler chunk (or immediately above it), which the HMM page-walk resolves
  as page-not-present → crash.  This matches the production VA density
  without loading a 31B model.

Kernel selection invariant:
  The MT64x64x64_ISA1201_DTVB1 kernel is selected by THREE co-conditions:
  (1) compute_dtype=bfloat16, quant_type="nf4"
  (2) bias=True  — selects the BH_Bias_HA_S_SAV_UserArgs epilogue
  (3) shape M=608, N=5376, K=21504  — Gemma-4-31B MLP gate/up projection
  All three are fixed below; do not change them.

Run (canonical):
  AMD_SERIALIZE_KERNEL=3 HIP_LAUNCH_BLOCKING=1 AMD_LOG_LEVEL=3 \\
  TORCH_USE_HIPBLASLT=0 \\
    uv run --no-sync python scripts/repro_gfx1201_dense.py 2>&1 | tee repro_dense.log

Confirm kernel dispatched (even if no crash):
  grep -o 'MT64x64x64[^[:space:]]*ISA1201[^[:space:]]*' repro_dense.log | head -3

Expected if successful:
  Process is killed with "Memory access fault by GPU node-1 … Page not present"
  followed by "GPU coredump: handler exited with error".

Expected if VA still misses:
  200 iterations complete, no crash.  Check the grep above — if the ShaderName
  appears the kernel was dispatched but the wild address landed in unmapped space
  (no VMA at all), which the GPU handles differently.  In that case the bug is
  confirmed at the kernel level but cannot be triggered without the full model VA.

Hardware: AMD Radeon AI PRO R9700, gfx1201 (RDNA4), 32 GB VRAM
Software: ROCm 7.2, torch 2.11.0+rocm7.2, bitsandbytes 0.49.2
Upstream:  github.com/ROCm/rocm-libraries/issues/7992
"""

import os
import sys

os.environ.setdefault("AMD_SERIALIZE_KERNEL", "3")
os.environ.setdefault("HIP_LAUNCH_BLOCKING", "1")
os.environ.setdefault("AMD_LOG_LEVEL", "3")
os.environ.setdefault("TORCH_USE_HIPBLASLT", "0")

import bitsandbytes as bnb  # noqa: E402
import torch  # noqa: E402

FILLER_GB = 20
FILLER_CHUNK_MB = 512
N_ITERS = 200


def print_env() -> None:
    print("=== repro_gfx1201_dense.py ===")
    print(f"torch:          {torch.__version__}")
    print(f"bitsandbytes:   {bnb.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / 1024**3
        print(f"Device:         {props.name}  ({total_gb:.1f} GB)")
    print(f"AMD_SERIALIZE_KERNEL: {os.environ.get('AMD_SERIALIZE_KERNEL')}")
    print(f"HIP_LAUNCH_BLOCKING:  {os.environ.get('HIP_LAUNCH_BLOCKING')}")
    print(f"AMD_LOG_LEVEL:        {os.environ.get('AMD_LOG_LEVEL')}")
    print(f"TORCH_USE_HIPBLASLT:  {os.environ.get('TORCH_USE_HIPBLASLT')}")
    print()


def preallocate_filler(target_gb: int, chunk_mb: int) -> list[torch.Tensor]:
    """Allocate GPU tensors to fill VA space before placing the layer."""
    chunk_floats = (chunk_mb * 1024 * 1024) // 2  # float16 = 2 bytes
    n_target = (target_gb * 1024) // chunk_mb
    chunks: list[torch.Tensor] = []
    print(f"Pre-allocating VA filler: target {target_gb} GB in {chunk_mb} MB chunks …")
    for i in range(n_target):
        try:
            chunks.append(torch.empty(chunk_floats, dtype=torch.float16, device="cuda"))
        except torch.cuda.OutOfMemoryError:
            allocated_gb = i * chunk_mb / 1024
            print(
                f"  OOM at chunk {i} — {allocated_gb:.1f} GB allocated (headroom needed for layer)"
            )
            break
    allocated_gb = len(chunks) * chunk_mb / 1024
    print(f"  Filler: {len(chunks)} chunks, {allocated_gb:.1f} GB allocated\n")
    return chunks


def build_layer() -> bnb.nn.Linear4bit:
    layer = bnb.nn.Linear4bit(
        input_features=21504,
        output_features=5376,
        bias=True,
        compute_dtype=torch.bfloat16,
        quant_type="nf4",
    )
    return layer.to("cuda")


def main() -> None:
    print_env()

    if not torch.cuda.is_available():
        print("ERROR: no CUDA/ROCm device found", file=sys.stderr)
        sys.exit(1)

    filler = preallocate_filler(FILLER_GB, FILLER_CHUNK_MB)

    print("Building Linear4bit layer  (in=21504, out=5376, bias=True, nf4, bf16) …")
    layer = build_layer()

    x = torch.randn(608, 21504, dtype=torch.bfloat16, device="cuda", requires_grad=True)

    with torch.no_grad():
        _ = layer(x.detach())
    torch.cuda.synchronize()

    allocated_mb = torch.cuda.memory_allocated() // (1024**2)
    reserved_mb = torch.cuda.memory_reserved() // (1024**2)
    print(f"After warmup: allocated={allocated_mb} MB  reserved={reserved_mb} MB")
    print(f"\nRunning {N_ITERS} forward+backward passes under SYNC …")
    print("A GPU page fault kills the process — that is the success condition.\n")

    for i in range(1, N_ITERS + 1):
        try:
            out = layer(x)
            loss = out.sum()
            loss.backward()
            torch.cuda.synchronize()
            if x.grad is not None:
                x.grad.zero_()
            if i % 20 == 0:
                print(f"  iter {i:4d}/{N_ITERS} — OK")
        except Exception as exc:
            print(f"  iter {i:4d}/{N_ITERS} — EXCEPTION: {exc}")
            print("\nDevice is likely dirty — run  make gpu-preflight  before next launch.")
            sys.exit(1)

    # Keep filler alive until the very end so it holds VA during all iterations.
    del filler

    print(f"\nCompleted {N_ITERS} iterations without a catchable exception.")
    print("Check the log for 'MT64x64x64' to confirm the kernel was dispatched.")
    print("If the ShaderName appears but no crash: the wild address landed in")
    print("fully unregistered VA (no VMA). The bug is confirmed at the kernel")
    print("level but requires the full model VA footprint to observe the fault.")


if __name__ == "__main__":
    main()
