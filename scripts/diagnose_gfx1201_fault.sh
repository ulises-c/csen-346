#!/usr/bin/env bash
# Localize the gfx1201 QLoRA backward page fault — turn the async VM fault into a
# synchronous, attributable one.
#
# The fault (Memory access fault / page not present, PERMISSION_FAULTS:0x3) is
# non-deterministic and surfaces during the backward pass. Because HIP kernel
# launches are async, the kernel "running" when the fault is reported is not
# necessarily the culprit — every prior root-cause theory was inferred from
# correlation, never from the faulting kernel itself. This run forces
# attribution:
#   AMD_SERIALIZE_KERNEL=3  — wait for each kernel to finish before the next
#   HIP_LAUNCH_BLOCKING=1   — synchronous launches → Python traceback points at
#                             the exact failing op
#   AMD_LOG_LEVEL=1         — HIP runtime logs the failing API call (errors only)
# Plus a dmesg tail for the kernel-side VM fault decode (needs read access to the
# kernel ring buffer; if empty, re-run with sudo or set kernel.dmesg_restrict=0).
#
# The synchronous traceback + dmesg are what name the kernel — AMD_LOG_LEVEL=1
# (errors) is enough. Do NOT default to level 3: it logs every HIP API call, so a
# serialized run writes MULTIPLE GB to diag.log before the fault and can fill the
# disk. Bump it only if level 1 doesn't name the failing call: LOG_LEVEL=3 bash …
#
# Reads:  if serialization makes the fault MOVE or VANISH → concurrency/allocator
#         race. If it still faults at the SAME named kernel → kernel bug, and the
#         backtrace names it (bnb NF4 dequant vs grad-ckpt recompute vs allocator).
#
# Usage:
#   bash scripts/diagnose_gfx1201_fault.sh             # ~120 steps, unsloth NF4 path
#   STEPS=200 bash scripts/diagnose_gfx1201_fault.sh   # widen the window
#   LOG_LEVEL=3 bash scripts/diagnose_gfx1201_fault.sh # verbose HIP log (multi-GB!)
#
# Throwaway output dir (never collides with the real run).

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

STEPS="${STEPS:-120}"
LOG_LEVEL="${LOG_LEVEL:-1}"
OUT_DIR="outputs/diag-gfx1201"
DIAG_LOG="$OUT_DIR/diag.log"
DMESG_LOG="$OUT_DIR/dmesg.log"

mkdir -p "$OUT_DIR"
rm -rf "$OUT_DIR"/checkpoint-* 2>/dev/null || true

printf 'gfx1201 fault diagnostic — %s steps, serialized kernels, AMD_LOG_LEVEL=%s\n' "$STEPS" "$LOG_LEVEL"
printf '  output: %s   dmesg: %s\n' "$DIAG_LOG" "$DMESG_LOG"
[[ "$LOG_LEVEL" -ge 3 ]] && printf '  WARNING: AMD_LOG_LEVEL=%s logs every HIP call — diag.log can reach many GB.\n' "$LOG_LEVEL"

DMESG_PID=""
# Start a best-effort kernel-ring tail; tolerate dmesg_restrict (no perms).
( dmesg -w 2>/dev/null | grep --line-buffered -iE \
    'amdgpu|gfxhub|VM_L2|page fault|PERMISSION_FAULTS|WALKER_ERROR|retry page' \
    > "$DMESG_LOG" ) &
DMESG_PID=$!

# shellcheck disable=SC2317,SC2329  # body is invoked indirectly via trap
cleanup() {
    if [[ -n "$DMESG_PID" ]]; then
        kill "$DMESG_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# TRAIN_SAVE_STEPS deliberately > STEPS: this is a diagnostic, not a checkpointed
# run — we want the bare fault, no checkpoint I/O in the window.
env TORCH_USE_HIPBLASLT=0 \
    AMD_SERIALIZE_KERNEL=3 \
    HIP_LAUNCH_BLOCKING=1 \
    AMD_LOG_LEVEL="$LOG_LEVEL" \
    PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8 \
    TRAIN_BASE_MODEL=unsloth/gemma-4-31B-it-unsloth-bnb-4bit \
    TRAIN_PREQ=true \
    TRAIN_MAX_STEPS="$STEPS" \
    TRAIN_SAVE_STEPS=$((STEPS + 10)) \
    TRAIN_OUTPUT_DIR="$OUT_DIR" \
    uv run --no-sync python scripts/train_sft.py \
    --config configs/train-sft-stage2-gemma4-31b.env \
    2>&1 | tee "$DIAG_LOG"
rc=${PIPESTATUS[0]}

printf '\n── diagnostic result ──\n'
if [[ $rc -eq 0 ]]; then
    printf 'Completed %s steps with NO fault under serialization.\n' "$STEPS"
    printf '  → strong signal the fault is a concurrency/async-allocator RACE,\n'
    printf '    not a deterministic kernel bug. Next: ablate the allocator knobs.\n'
else
    printf 'Process exited rc=%s. The faulting op is the LAST kernel in:\n  %s\n' "$rc" "$DIAG_LOG"
    printf '  → grep -iE "fault|backtrace|Traceback|dequant|checkpoint" %s\n' "$DIAG_LOG"
    [[ -s "$DMESG_LOG" ]] && printf '  → kernel VM-fault decode: %s\n' "$DMESG_LOG" \
        || printf '  → dmesg empty (no perms?): re-run with sudo for the kernel decode.\n'
fi
exit "$rc"
