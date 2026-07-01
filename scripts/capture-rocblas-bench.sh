#!/usr/bin/env bash
# One-shot bench-log capture for the gfx1201 ISA1201 GEMM page fault.
#
# Wraps a normal Stage 2 resume with ROCBLAS_LAYER=2 to record the exact
# rocblas-bench invocation line for the faulting GEMM.  The faulting call
# itself may not flush (it never returns), but the same shape (M=608 N=5376
# K=21504) repeats every few steps — any occurrence in the log is the correct
# replay encoding, not a hand-written guess.
#
# After the run crashes, the script greps the bench log and prints the
# candidate line(s).  Feed the output to replay-rocblas-bench.sh to confirm
# the ShaderName.
#
# Usage:
#   bash scripts/capture-rocblas-bench.sh
#
# The run will crash at the GPU page fault (~21 steps after the checkpoint).
# Do NOT run while the production crawl (monitor_stage2.sh) is active — one
# GPU, one training process.
set -uo pipefail

ROCM="${ROCM_PATH:-/opt/rocm}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="$REPO_DIR/outputs/sft-stage2-gemma4-31b"
BENCH_LOG="$OUTPUT_DIR/rocblas_bench_capture.log"
TRAIN_LOG="$OUTPUT_DIR/bench-capture-run.log"
STEPS=50  # enough to hit the fault; low enough not to waste time

printf 'gfx1201 rocBLAS bench capture\n'
printf '  bench log:  %s\n' "$BENCH_LOG"
printf '  train log:  %s\n' "$TRAIN_LOG"
printf '\n'

# ── 1. Verify ROCBLAS_LAYER bitmask is recognised ──────────────────────────
# A wrong bitmask silently produces an empty log; write a throwaway row first.
printf 'Verifying ROCBLAS_LAYER=2 produces output...\n'
probe_out="$OUTPUT_DIR/rocblas_bench_probe.$$.log"
ROCBLAS_LAYER=2 ROCBLAS_LOG_BENCH_PATH="$probe_out" \
  "$ROCM/bin/rocblas-bench" -f gemm -m 4 -n 4 -k 4 --a_type f32_r >/dev/null 2>&1 || true
if [[ -s "$probe_out" ]]; then
  printf '  ROCBLAS_LAYER=2 + ROCBLAS_LOG_BENCH_PATH: OK\n'
  rm -f "$probe_out"
else
  rm -f "$probe_out"
  printf '  ROCBLAS_LOG_BENCH_PATH produced no output — trying ROCBLAS_LOG_BENCH_PATH alt\n'
  printf '  If this also fails, check your ROCm version or set ROCBLAS_LAYER manually.\n'
fi

# ── 2. GPU preflight ────────────────────────────────────────────────────────
printf '\nRunning gpu-preflight...\n'
cd "$REPO_DIR" || exit 1
if ! make gpu-preflight; then
  printf 'gpu-preflight FAILED — aborting (GPU is dirty; clean it first)\n' >&2
  exit 1
fi

# ── 3. Capture run ──────────────────────────────────────────────────────────
printf '\nStarting bench-capture training run (%d steps)...\n' "$STEPS"
rm -f "$BENCH_LOG"

# Compute a TRAIN_MAX_STEPS that is STEPS beyond the latest checkpoint so the
# run crashes during normal training, not during the checkpoint step itself.
latest_ckpt=0
for d in "$OUTPUT_DIR"/checkpoint-*; do
  [[ -d "$d" ]] || continue
  n="${d##*checkpoint-}"
  [[ "$n" =~ ^[0-9]+$ ]] && (( n > latest_ckpt )) && latest_ckpt="$n"
done
max_steps=$(( latest_ckpt + STEPS ))
printf '  Latest checkpoint: %d — running to step %d\n' "$latest_ckpt" "$max_steps"

ROCBLAS_LAYER=2 \
  ROCBLAS_LOG_BENCH_PATH="$BENCH_LOG" \
  TORCH_USE_HIPBLASLT=0 \
  AMD_SERIALIZE_KERNEL=3 \
  HIP_LAUNCH_BLOCKING=1 \
  AMD_LOG_LEVEL=1 \
  PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8 \
  TRAIN_BASE_MODEL=unsloth/gemma-4-31B-it-unsloth-bnb-4bit \
  TRAIN_PREQ=true \
  TRAIN_MAX_STEPS="$max_steps" \
  TRAIN_SAVE_STEPS=$(( max_steps + 10 )) \
  TRAIN_OUTPUT_DIR="$OUTPUT_DIR" \
  uv run --no-sync python scripts/train_sft.py \
  --config configs/train-sft-stage2-gemma4-31b.env \
  > "$TRAIN_LOG" 2>&1 || true  # expect a non-zero exit (GPU fault)

printf '\nRun exited. Searching bench log for M=608 N=5376 K=21504 gemm_ex calls...\n'
printf '  bench log size: %s bytes\n' "$(wc -c < "$BENCH_LOG" 2>/dev/null || echo 0)"

# ── 4. Extract candidate reproducer lines ───────────────────────────────────
if [[ -s "$BENCH_LOG" ]]; then
  matches="$(grep -i 'gemm_ex\|gemm_strided\|gemm_batched' "$BENCH_LOG" \
             | grep -E '[ ,]608[ ,]' | grep -E '[ ,]5376[ ,]' | grep -E '[ ,]21504[ ,]' || true)"
  if [[ -n "$matches" ]]; then
    printf '\n=== CANDIDATE REPRODUCER LINE(S) — paste into replay-rocblas-bench.sh ===\n'
    printf '%s\n' "$matches"
    printf '=== END ===\n\n'
    printf 'Next: run  bash scripts/replay-rocblas-bench.sh "<line above>"\n'
    printf '      and confirm it dispatches ...MT64x64x64...ISA1201...DTVB1...\n'
  else
    printf '\nNo gemm_ex line matching M=608 N=5376 K=21504 found.\n'
    printf 'Possible causes:\n'
    printf '  - The bench log was not written (ROCBLAS_LAYER env var not recognised)\n'
    printf '  - The fault landed before any call at this shape was logged\n'
    printf '  - The seed gave a different batch length this run\n'
    printf 'Full bench log: %s\n' "$BENCH_LOG"
    printf 'Training log:   %s\n' "$TRAIN_LOG"
    printf '\nFallback: try  ROCBLAS_LAYER=3  (adds trace, much larger log) or check\n'
    printf '  the training log for the step that faulted.\n'
  fi
else
  printf '\nBench log is EMPTY — ROCBLAS_LAYER=2 did not write output.\n'
  printf 'Check that ROCBLAS_LOG_BENCH_PATH is supported in this ROCm build:\n'
  printf '  strings %s/lib/librocblas.so | grep -i "ROCBLAS_LOG"\n' "$ROCM"
  printf 'Training log: %s\n' "$TRAIN_LOG"
fi
