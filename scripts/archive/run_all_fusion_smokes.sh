#!/usr/bin/env bash
# Run all four fusion smoke tests sequentially.
#
# Order:
#   1. 27B fusion think         (reuses any warm 27B server)
#   2. 27B fusion no-think       (reuses warm 27B server from step 1)
#   3. Kill 27B, boot A3B
#   4. A3B fusion think          (reuses warm A3B server from step 3 boot)
#   5. A3B fusion no-think       (reuses warm A3B server from step 4)
#   6. Kill A3B server (cleanup)
#
# Status / progress is mirrored to a single log file so a Monitor can stream
# events. Each per-smoke metrics file lands in its own results/ subdir.

set -uo pipefail   # NOTE: not -e — we want to continue if one smoke fails
cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"

STATUS_LOG="$ROOT/logs/fusion-smokes-$(date -u +%Y-%m-%dT%H-%M-%S).log"
mkdir -p "$(dirname "$STATUS_LOG")"

emit() {
  # Status events get a leading "::" so the Monitor can grep them out
  echo ":: $1" | tee -a "$STATUS_LOG"
}

emit "RUN-START $(date)"

# ── Step 1: 27B fusion think ─────────────────────────────────────────────────
emit "STEP 1/4: qwen27b fusion (think on) — kicking off..."
if bash scripts/eval_qwen27b.sh smoke --unified --keep-server >>"$STATUS_LOG" 2>&1; then
  emit "STEP 1/4 DONE: qwen27b fusion think"
else
  emit "STEP 1/4 FAILED: qwen27b fusion think (continuing)"
fi

# ── Step 2: 27B fusion no-think ──────────────────────────────────────────────
emit "STEP 2/4: qwen27b fusion no-think — kicking off..."
if bash scripts/eval_qwen27b.sh smoke --unified --nothink --keep-server >>"$STATUS_LOG" 2>&1; then
  emit "STEP 2/4 DONE: qwen27b fusion no-think"
else
  emit "STEP 2/4 FAILED: qwen27b fusion no-think (continuing)"
fi

# ── Switch model: kill 27B server ────────────────────────────────────────────
emit "Switching model: stopping 27B server"
pkill -f "llama-server.*Qwen3.6-27B-UD-Q5_K_XL" 2>/dev/null || true
sleep 5
emit "27B stopped, VRAM free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null) MiB"

# ── Step 3: A3B fusion think ─────────────────────────────────────────────────
emit "STEP 3/4: qwen35b-a3b fusion (think on) — kicking off (cold-boots A3B)..."
if bash scripts/eval_qwen35b_a3b.sh smoke --unified --keep-server >>"$STATUS_LOG" 2>&1; then
  emit "STEP 3/4 DONE: qwen35b-a3b fusion think"
else
  emit "STEP 3/4 FAILED: qwen35b-a3b fusion think (continuing)"
fi

# ── Step 4: A3B fusion no-think ──────────────────────────────────────────────
emit "STEP 4/4: qwen35b-a3b fusion no-think — kicking off..."
if bash scripts/eval_qwen35b_a3b.sh smoke --unified --nothink --keep-server >>"$STATUS_LOG" 2>&1; then
  emit "STEP 4/4 DONE: qwen35b-a3b fusion no-think"
else
  emit "STEP 4/4 FAILED: qwen35b-a3b fusion no-think (continuing)"
fi

# ── Cleanup: kill A3B server ─────────────────────────────────────────────────
emit "Cleanup: stopping A3B server"
pkill -f "llama-server.*Qwen3.6-35B-A3B" 2>/dev/null || true
sleep 3

emit "RUN-COMPLETE $(date)"
emit "Results landed:"
for d in \
  results/qwen27b-local-smoke-unified \
  results/qwen27b-local-smoke-unified-nothink \
  results/qwen35b-a3b-local-smoke-unified \
  results/qwen35b-a3b-local-smoke-unified-nothink; do
  if [ -f "$d/metrics_summary.json" ]; then
    rouge1=$(jq -r .rouge1 "$d/metrics_summary.json" 2>/dev/null || echo "?")
    state=$(jq -r .state_accuracy.overall "$d/metrics_summary.json" 2>/dev/null || echo "?")
    fb=$(jq -r ".unified_fallback_count // \"?\"" "$d/run_config.json" 2>/dev/null || echo "?")
    elapsed=$(jq -r .total_elapsed_seconds "$d/run_config.json" 2>/dev/null || echo "?")
    emit "  $d: rouge1=$rouge1 state=$state fallbacks=$fb elapsed=${elapsed}s"
  else
    emit "  $d: NO METRICS (smoke failed or didn't run)"
  fi
done
