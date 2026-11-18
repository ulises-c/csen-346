#!/usr/bin/env python3
"""
Minimal standalone reproducer for the gfx1201 ISA1201 Tensile GEMM page fault.

Root cause (confirmed, all arms exhausted):
  Tensile kernel Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x64x64_...ISA1201..._DTVB1...
  computes a wild host-VA from its column-major B descriptor. The bad address lands
  ~1 GB outside both GEMM operand buffers — a Tensile tile/stride calculation bug.

Operand shape:
  A = (608, 21504)  row-major bf16        (activation batch)
  B = (21504, 5376) col-major NF4→bf16    (dequantized QLoRA weight, DTVB1 flag)
  bias = True  (required — BH_Bias_UserArgs epilogue selects this tile)

Run command:
  AMD_SERIALIZE_KERNEL=3 HIP_LAUNCH_BLOCKING=1 AMD_LOG_LEVEL=3 \\
    uv run --no-sync python scripts/repro_gfx1201_rocblas.py 2>&1 | tee repro_gfx1201.log

Then grep the log for the ShaderName to confirm kernel dispatch:
  grep -o 'MT64x64x64.*ISA1201[^ ]*' repro_gfx1201.log | head -3

Hardware: AMD Radeon AI PRO R9700, gfx1201 (RDNA4), 32 GB VRAM
Software: ROCm 7.2, torch 2.11.0+rocm7.2, bitsandbytes 0.49.2
Upstream: ROCm/rocm-libraries — component: rocBLAS / Tensile

NOTE on crash reproducibility:
  This script dispatches the faulting kernel but may NOT crash in a small process.
  The buggy kernel computes a wild host-VA ~1 GB from the B operand. Whether that
  address faults depends on the process VA layout: in full Gemma 4 31B training
  (~19 GB HMM-mapped model weights), the wild address reliably lands in a registered
  but page-not-present region → GPU page fault every time (deterministic at fixed seed,
  steps 22-24 from step 20). In this minimal script the VA map is sparse and the wild
  address either falls outside GPU-accessible HMM memory or hits a valid page.
  Use rocprof/rocm-compute-sanitizer to confirm ISA1201 kernel dispatch without needing a crash.
"""

import os
import sys

# These must be set before any HIP/torch import to take effect.
# Set here as a safety net; the run-command above is the canonical way.
os.environ.setdefault("AMD_SERIALIZE_KERNEL", "3")
os.environ.setdefault("HIP_LAUNCH_BLOCKING", "1")
os.environ.setdefault("AMD_LOG_LEVEL", "3")
# Keep hipBLASLt off — it has no kernel for col-major B at this tile on gfx1201
# (confirmed run #11: hipBLASLt returns "Cannot find the function", falls back to Tensile)
os.environ.setdefault("TORCH_USE_HIPBLASLT", "0")

import bitsandbytes as bnb
import torch

N_ITERS = 200  # fault is probabilistic in async; SYNC makes it ~deterministic


def print_env() -> None:

    print("=== repro_gfx1201_rocblas.py ===")
    print(f"torch:          {torch.__version__}")
    print(f"bitsandbytes:   {bnb.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device:         {torch.cuda.get_device_name(0)}")
    print(f"AMD_SERIALIZE_KERNEL: {os.environ.get('AMD_SERIALIZE_KERNEL')}")
    print(f"HIP_LAUNCH_BLOCKING:  {os.environ.get('HIP_LAUNCH_BLOCKING')}")
    print(f"AMD_LOG_LEVEL:        {os.environ.get('AMD_LOG_LEVEL')}")
    print(f"TORCH_USE_HIPBLASLT:  {os.environ.get('TORCH_USE_HIPBLASLT')}")
    print()


def build_layer() -> bnb.nn.Linear4bit:
    # bias=True is required — the BH_Bias_HA_S_SAV_UserArgs epilogue is what
    # selects this Tensile tile over a bias-free alternative.
    layer = bnb.nn.Linear4bit(
        input_features=21504,
        output_features=5376,
        bias=True,
        compute_dtype=torch.bfloat16,
        quant_type="nf4",
    )
    layer = layer.to("cuda")
    return layer


def main() -> None:
    print_env()

    if not torch.cuda.is_available():
        print("ERROR: no CUDA/ROCm device found", file=sys.stderr)
        sys.exit(1)

    print("Building Linear4bit layer  (in=21504, out=5376, bias=True, nf4, bf16)...")
    layer = build_layer()

    # Warmup forward: triggers in-place NF4 quantization of the weight tensor.
    x = torch.randn(608, 21504, dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        _ = layer(x)
    torch.cuda.synchronize()
    print("Warmup done — weight quantized to NF4. Starting fault probe loop.\n")
    print(f"Running {N_ITERS} forward passes.  Under SYNC the fault is near-deterministic;")
    print("if this process is killed by a GPU page fault the reproducer succeeded.\n")
    print("To confirm the ShaderName dispatched, grep the log for 'MT64x64x64'.\n")

    crashes = 0
    for i in range(1, N_ITERS + 1):
        try:
            with torch.no_grad():
                out = layer(x)
            torch.cuda.synchronize()
            if i % 20 == 0:
                print(f"  step {i:4d}/{N_ITERS} — OK  (out shape {tuple(out.shape)})")
        except Exception as exc:  # noqa: BLE001
            crashes += 1
            print(f"  step {i:4d}/{N_ITERS} — EXCEPTION: {exc}")
            # After a HIP fault the device is unrecoverable without a driver reset;
            # stop immediately rather than confounding further iterations.
            print("\nFault caught as Python exception — device may be dirty.")
            print("Run  make gpu-preflight  (or driver reset) before next launch.")
            break

    print(f"\nDone: {i} iterations, {crashes} exception(s) caught.")
    if crashes == 0:
        print("No Python-catchable exception — either the fault didn't trigger,")
        print("or it produced an unrecoverable hard kill (check process exit code).")
        print("Check repro_gfx1201.log for 'MT64x64x64' kernel dispatch lines.")


if __name__ == "__main__":
    main()
