#!/usr/bin/env bash
# Wait for manually-launched Phase 3 to finish, then refresh the stage-balanced
# backtest with Phase 3's metrics included. Also re-runs after the LLM-judge
# data lands (if it lands later).
#
# Spawned alongside Phase 3. Polls every 2 minutes.
#
# Launch with:
#   nohup bash scripts/post_phase3_backtest.sh \
#     > results/_orchestrator_logs/post_phase3_$(date -u +%Y-%m-%dT%H-%M-%S).log 2>&1 &
#   disown

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

PHASE3_DIR="${PHASE3_DIR:-results/phase3-t4-bert-qwen27b-think-n200-seed42}"
START_TS=$(date +%s)
TIMEOUT_SEC=$((24 * 3600))

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== Post-Phase-3 backtest orchestrator ==="
log "Watching: $PHASE3_DIR/metrics_summary.json"

while true; do
  if [[ -f "$PHASE3_DIR/metrics_summary.json" ]]; then
    log "Phase 3 complete — metrics_summary.json detected."
    break
  fi
  ELAPSED=$(($(date +%s) - START_TS))
  if [[ "$ELAPSED" -gt "$TIMEOUT_SEC" ]]; then
    log "WARN: 24h timeout reached without Phase 3 metrics — firing backtest with whatever exists."
    break
  fi
  sleep 120
done

log "=== Refreshing backtest (v5, includes Phase 3 + judge if landed) ==="
OUT="results/_orchestrator_logs/backtest_stage_balanced_$(date -u +%Y_%m_%d_post_phase3).md"
.venv/bin/python scripts/backtest_stage_balanced.py --out "$OUT"
EXIT=$?
log "Backtest exit: $EXIT"
log "Output: $OUT"

ln -sf "$(basename "$OUT")" results/_orchestrator_logs/backtest_stage_balanced_latest.md
log "Symlink updated to point at $(basename "$OUT")"

log "=== Post-Phase-3 backtest done ==="
date
