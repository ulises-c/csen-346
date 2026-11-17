#!/usr/bin/env bash
# Replay a captured rocblas-bench line and confirm it dispatches the faulting
# ISA1201 Tensile GEMM kernel.
#
# Success criterion: the replay crashes with a GPU page fault AND
# AMD_LOG_LEVEL=3 shows the ShaderName containing
#   ...MT64x64x64...ISA1201...DTVB1...
# That proves the bug is inside rocBLAS/Tensile, independent of PyTorch and
# bitsandbytes — the form the upstream report needs.
#
# If the replay does NOT fault, note that finding in the upstream report and
# fall back to the HIP reproducer (rocblas_gemm_ex + bias, same entry point).
#
# Usage:
#   bash scripts/replay-rocblas-bench.sh 'rocblas-bench -f gemm_ex ...'
#
# Or capture the line first:
#   bash scripts/capture-rocblas-bench.sh
# then paste the printed line as the first argument here.
set -uo pipefail

ROCM="${ROCM_PATH:-/opt/rocm}"
BENCH_BIN="$ROCM/bin/rocblas-bench"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="$REPO_DIR/outputs/sft-stage2-gemma4-31b"
REPLAY_LOG="$OUTPUT_DIR/bench-replay.log"

if [[ $# -lt 1 ]]; then
  printf 'Usage: %s "<rocblas-bench line>"\n' "$0" >&2
  printf '\nExample:\n' >&2
  printf '  bash scripts/replay-rocblas-bench.sh \\\n' >&2
  printf "    'rocblas-bench -f gemm_ex --transposeA N --transposeB N -m 608 -n 5376 -k 21504 ...'\n" >&2
  exit 1
fi

bench_line="$1"
# Strip a leading "rocblas-bench" token if present so we can prepend the full path.
bench_args="${bench_line#rocblas-bench}"
bench_args="${bench_args#"$BENCH_BIN"}"
# An array is required to safely expand a constructed argument list.
read -ra bench_args_arr <<< "$bench_args"

if [[ ! -x "$BENCH_BIN" ]]; then
  printf 'rocblas-bench not found at %s\n' "$BENCH_BIN" >&2
  printf 'Install the rocblas-test or rocblas-benchmarks package.\n' >&2
  exit 1
fi

printf 'gfx1201 rocBLAS bench replay\n'
printf '  bin:        %s\n' "$BENCH_BIN"
printf '  args:       %s\n' "$bench_args"
printf '  replay log: %s\n' "$REPLAY_LOG"
printf '\n'

# Run the replay under AMD_LOG_LEVEL=3 (names every kernel) + serialization.
# Level 3 is verbose but bench runs are short (microseconds) — log stays small.
AMD_LOG_LEVEL=3 \
  AMD_SERIALIZE_KERNEL=3 \
  TORCH_USE_HIPBLASLT=0 \
  "$BENCH_BIN" "${bench_args_arr[@]}" \
  > "$REPLAY_LOG" 2>&1 || true  # non-zero expected if it faults

printf 'Replay exited. Checking results...\n\n'

# ── Fault? ──────────────────────────────────────────────────────────────────
if grep -qi 'page not present\|page fault\|memory access fault' "$REPLAY_LOG"; then
  printf '[FAULT REPRODUCED] GPU page fault confirmed in standalone rocblas-bench run.\n'
  grep -i 'page not present\|page fault\|memory access fault' "$REPLAY_LOG" | head -3
  fault_confirmed=1
else
  printf '[NO FAULT] Replay completed without a page fault.\n'
  printf '  This means the bug requires the fused/UserArgs path (bias or other UserArgs).\n'
  printf '  Next: try the HIP reproducer with rocblas_gemm_ex + bias extension.\n'
  fault_confirmed=0
fi

printf '\n'

# ── ShaderName match? ────────────────────────────────────────────────────────
target_shader='MT64x64x64.*ISA1201.*DTVB1'
if grep -qE "$target_shader" "$REPLAY_LOG"; then
  printf '[SHADER MATCH] Faulting kernel confirmed:\n'
  grep -oE "Cijk[^ ]*MT64x64x64[^ ]*ISA1201[^ ]*" "$REPLAY_LOG" | head -1
elif grep -qi 'Cijk\|MT64x64' "$REPLAY_LOG"; then
  printf '[DIFFERENT SHADER] A Tensile GEMM was dispatched but not the expected tile:\n'
  grep -oE 'Cijk[^ ]*' "$REPLAY_LOG" | head -3
  printf '  Check transB / ldb encoding — a wrong flag selects a different tile.\n'
else
  printf '[NO SHADER NAME] AMD_LOG_LEVEL=3 did not log a kernel name.\n'
  printf '  The kernel may have not been dispatched, or check AMD_LOG_LEVEL support.\n'
fi

printf '\n'

if [[ $fault_confirmed -eq 1 ]]; then
  printf '=== READY FOR UPSTREAM ===\n'
  printf 'Paste this rocblas-bench line into the ROCm/rocm-libraries report as the\n'
  printf 'standalone reproducer — it faults without PyTorch or bitsandbytes.\n'
else
  printf 'Replay did not fault — see notes above. Full log: %s\n' "$REPLAY_LOG"
fi
