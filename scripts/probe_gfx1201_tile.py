#!/usr/bin/env python3
"""
Tile-selection probe for the gfx1201 ISA1201 Tensile GEMM page fault.

Goal: determine which combination of input shape and call path causes Tensile
to dispatch the specific faulting kernel:
  Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x64x64_…_ISA1201_…_DTVB1_…

No VA filler — just a few iterations to capture ShaderName.  Run each config and
grep the log for MT64x64x64.*DTVB1.

WHY THE DEFAULT M IS 607 (not 608): the SYNC-deterministic capture
(diag-sync-probe-step1230, HIP_LAUNCH_BLOCKING=1) logs the *faulting* GEMM's A
operand as (1, 607, 21504) — M=607, K=21504, B col-major stride=(1,21504). M is
the current sample's token length and is data-order-dependent (439 and 607 both
observed in SYNC runs). Every earlier probe round fixed M=608 (the even neighbor,
captured under async logging) and only ever dispatched MT128x128x32 — so the
standalone-vs-production tile divergence may simply be that 608 was the wrong M.
Tensile size predicates can flip MT128x128x32 ↔ MT64x64x64 across an M boundary,
so the first thing to do is sweep M over the real faulting values and neighbors.

Run matrix:
  # PRIMARY: sweep M over the SYNC-confirmed faulting values + neighbors in one run.
  # Find which M dispatches MT64x64x64 — that is the standalone-repro shape.
  AMD_SERIALIZE_KERNEL=3 HIP_LAUNCH_BLOCKING=1 AMD_LOG_LEVEL=3 TORCH_USE_HIPBLASLT=0 \
    uv run --no-sync python scripts/probe_gfx1201_tile.py --shape-3d --ckpt \
    --m-sweep 439,512,576,606,607,608,609,640 2>&1 | tee probe_msweep.log

  # 2D plain (baseline — already confirmed at M=608: dispatches MT128x128x32)
  ... python scripts/probe_gfx1201_tile.py --m 608 2>&1 | tee probe_2d_plain.log

  # 3D + ckpt(reentrant=False)  [most faithful to production], default M=607
  ... python scripts/probe_gfx1201_tile.py --shape-3d --ckpt 2>&1 | tee probe_3d_ckpt_nr.log

  # SECONDARY: drop the bias epilogue (faulting kernel is …_Bias_…_UserArgs_…).
  # If MT64x64x64 appears only with bias, the trigger is the fused epilogue.
  ... python scripts/probe_gfx1201_tile.py --shape-3d --ckpt --no-bias 2>&1 | tee probe_nobias.log

Detect which tile each M dispatched (associate ShaderName with the SWEEP banner):
  grep -E 'SWEEP M=|MT[0-9]+x[0-9]+x[0-9]+.*ISA1201' probe_msweep.log \
    | grep -oE 'SWEEP M=[0-9]+|MT[0-9]+x[0-9]+x[0-9]+'
  # Success == any 'MT64x64x64' line appears under a 'SWEEP M=<value>' banner.

Hardware: AMD Radeon AI PRO R9700, gfx1201 (RDNA4), 32 GB VRAM
Software: ROCm 7.2, torch 2.11.0+rocm7.2, bitsandbytes 0.49.2
Upstream:  github.com/ROCm/rocm-libraries/issues/7992
"""

import argparse
import contextlib
import os
import sys

os.environ.setdefault("AMD_SERIALIZE_KERNEL", "3")
os.environ.setdefault("HIP_LAUNCH_BLOCKING", "1")
os.environ.setdefault("AMD_LOG_LEVEL", "3")
os.environ.setdefault("TORCH_USE_HIPBLASLT", "0")

import bitsandbytes as bnb  # noqa: E402
import torch  # noqa: E402
import torch.utils.checkpoint as ckpt_utils  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tile-selection probe for MT64x64x64_ISA1201_DTVB1")
    p.add_argument(
        "--m",
        type=int,
        default=607,
        help="Sequence-length dim (default: 607, SYNC-confirmed faulting M)",
    )
    p.add_argument(
        "--m-sweep",
        type=str,
        default="",
        help="Comma list of M values to run in one process, e.g. 439,607,608,640",
    )
    p.add_argument(
        "--no-bias",
        action="store_true",
        help="Build Linear4bit with bias=False (test the non-Bias epilogue path)",
    )
    p.add_argument("--shape-3d", action="store_true", help="Use (1, M, K) instead of (M, K)")
    p.add_argument("--ckpt", action="store_true", help="Wrap forward in torch.utils.checkpoint")
    p.add_argument(
        "--reentrant", action="store_true", help="Use use_reentrant=True (default: False)"
    )
    p.add_argument(
        "--autocast",
        action="store_true",
        help="Wrap in torch.autocast (matches production training)",
    )
    p.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip no_grad warmup (first dispatch hits cold cache)",
    )
    p.add_argument(
        "--noncontig-a",
        action="store_true",
        help="Make A non-contiguous via slice of larger tensor",
    )
    p.add_argument("--iters", type=int, default=3, help="Number of fwd+bwd iterations")
    return p.parse_args()


def run_probe(m: int, args: argparse.Namespace) -> None:
    shape_str = f"(1, {m}, 21504)" if args.shape_3d else f"({m}, 21504)"
    print(f"\n##### SWEEP M={m}  shape={shape_str} #####", flush=True)

    layer = bnb.nn.Linear4bit(
        input_features=21504,
        output_features=5376,
        bias=not args.no_bias,
        compute_dtype=torch.bfloat16,
        quant_type="nf4",
    ).to("cuda")

    x_shape = (1, m, 21504) if args.shape_3d else (m, 21504)

    if args.noncontig_a:
        # Slice from a wider tensor so A shares storage with a larger allocation — the
        # activation A in production may come from a packed sequence tensor, giving
        # non-standard strides or offsets in the underlying storage.
        big = torch.randn(*x_shape[:-1], 21504 * 2, dtype=torch.bfloat16, device="cuda")
        x_base = big[..., :21504]  # non-contiguous view (stride in last dim = 2)
        x = x_base.requires_grad_(True)
    else:
        x = torch.randn(*x_shape, dtype=torch.bfloat16, device="cuda", requires_grad=True)

    print(f"Input shape:    {list(x.shape)}  stride={list(x.stride())}  contig={x.is_contiguous()}")

    if not args.no_warmup:
        with torch.no_grad():
            _ = layer(x.detach())
        torch.cuda.synchronize()
        print("Warmup complete.\n")
    else:
        print("Skipping warmup — cold kernel cache.\n")

    def forward_fn(x_in: torch.Tensor) -> torch.Tensor:
        return layer(x_in)

    ctx_autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.autocast
        else contextlib.nullcontext()
    )

    for i in range(1, args.iters + 1):
        with ctx_autocast:
            if args.ckpt:
                out = ckpt_utils.checkpoint(forward_fn, x, use_reentrant=args.reentrant)
            else:
                out = layer(x)
        loss = out.sum()
        loss.backward()
        torch.cuda.synchronize()
        if x.grad is not None:
            x.grad.zero_()
        print(f"  M={m} iter {i}/{args.iters} — OK", flush=True)


def main() -> None:
    args = parse_args()

    ckpt_str = f"ckpt(reentrant={args.reentrant})" if args.ckpt else "plain"
    extras = []
    if args.autocast:
        extras.append("autocast")
    if args.no_warmup:
        extras.append("no-warmup")
    if args.noncontig_a:
        extras.append("noncontig-A")
    if args.no_bias:
        extras.append("no-bias")
    extra_str = "  " + "+".join(extras) if extras else ""
    m_values = [int(v) for v in args.m_sweep.split(",")] if args.m_sweep else [args.m]
    print(
        f"=== probe_gfx1201_tile.py  path={ckpt_str}{extra_str}  M={m_values}  iters={args.iters} ==="
    )
    print(f"torch:          {torch.__version__}")
    print(f"bitsandbytes:   {bnb.__version__}")

    if not torch.cuda.is_available():
        print("ERROR: no CUDA/ROCm device found", file=sys.stderr)
        sys.exit(1)

    props = torch.cuda.get_device_properties(0)
    print(f"Device:         {props.name}  ({props.total_memory / 1024**3:.1f} GB)")
    print(f"AMD_SERIALIZE_KERNEL: {os.environ.get('AMD_SERIALIZE_KERNEL')}")
    print(f"HIP_LAUNCH_BLOCKING:  {os.environ.get('HIP_LAUNCH_BLOCKING')}")
    print(f"AMD_LOG_LEVEL:        {os.environ.get('AMD_LOG_LEVEL')}")
    print(f"TORCH_USE_HIPBLASLT:  {os.environ.get('TORCH_USE_HIPBLASLT')}")
    print("Watch for MT64x64x64.*DTVB1 in ShaderName lines under each SWEEP M= banner.")

    for m in m_values:
        run_probe(m, args)

    print(f"\nCompleted {len(m_values)} shape(s) — no crash.")
    print(
        "Which tile per M:  grep -E 'SWEEP M=|MT[0-9]+x[0-9]+x[0-9]+.*ISA1201' this_log.log"
        " | grep -oE 'SWEEP M=[0-9]+|MT[0-9]+x[0-9]+x[0-9]+'"
    )


if __name__ == "__main__":
    main()
