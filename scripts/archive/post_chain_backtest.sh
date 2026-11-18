#!/usr/bin/env bash
# Post-chain step: wait for the overnight Qwen27B chain to finish, then
# refresh the stage-balanced backtest with all new data that landed during
# the chain (4 Qwen27B grid cells + LLM-judge scores + Phase 3 OR Layer-2
# results + bilingual probe).
#
# Spawned alongside the main chain rather than appended to it — editing a
# script while bash is actively executing it is undefined behavior, so we
# run as a separate process that polls for the chain to exit cleanly.
#
# Triggers re-run when ANY of:
#   - The chain PID exits
#   - All expected chain artifacts exist on disk (defensive — covers PID-file loss)
#   - 12 hours elapse (safety timeout)
#
# Launch with:
#   nohup bash scripts/post_chain_backtest.sh \
#     > results/_orchestrator_logs/post_chain_backtest_$(date -u +%Y-%m-%dT%H-%M-%S).log 2>&1 &
#   disown

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

CHAIN_PID_FILE="/tmp/qwen27b_overnight_chain.pid"
START_TS=$(date +%s)
TIMEOUT_SEC=$((12 * 3600))

log() { echo "[$(date '+%H:%M:%S')] $*"; }

CHAIN_PID=""
[[ -f "$CHAIN_PID_FILE" ]] && CHAIN_PID="$(cat "$CHAIN_PID_FILE")"

log "=== Post-chain backtest orchestrator ==="
log "Chain PID: ${CHAIN_PID:-unknown}"
log "Waiting for chain to exit (or 12h timeout)..."

while true; do
  # Primary signal: chain PID gone
  if [[ -n "$CHAIN_PID" ]] && ! kill -0 "$CHAIN_PID" 2>/dev/null; then
    log "Chain PID $CHAIN_PID has exited."
    break
  fi
  # Secondary signal: chain wrote its 'done' marker (last line of overnight_qwen27b_chain.sh)
  # Tail the latest chain log for the final marker
  # shellcheck disable=SC2012  # `ls -t` is the cleanest way to get the most-recent file by mtime; find -printf | sort is verbose for no gain here
  LATEST_CHAIN_LOG="$(ls -t results/_orchestrator_logs/overnight_chain_*.log 2>/dev/null | head -1)"
  if [[ -n "$LATEST_CHAIN_LOG" ]] && grep -q "Overnight chain done" "$LATEST_CHAIN_LOG" 2>/dev/null; then
    log "Detected 'Overnight chain done' in $LATEST_CHAIN_LOG."
    break
  fi
  # Safety timeout
  ELAPSED=$(($(date +%s) - START_TS))
  if [[ "$ELAPSED" -gt "$TIMEOUT_SEC" ]]; then
    log "WARN: 12h timeout reached, firing backtest with whatever data exists."
    break
  fi
  sleep 120
done

log "=== Refreshing stage-balanced backtest (v3, includes all chain artifacts) ==="
OUT="results/_orchestrator_logs/backtest_stage_balanced_$(date -u +%Y_%m_%d_post_chain).md"
.venv/bin/python scripts/backtest_stage_balanced.py --out "$OUT"
EXIT=$?
log "Backtest exit: $EXIT"
log "Output: $OUT"

# Convenience: also drop a 'latest' symlink for tomorrow morning's quick-glance
ln -sf "$(basename "$OUT")" results/_orchestrator_logs/backtest_stage_balanced_latest.md
log "Symlink updated: results/_orchestrator_logs/backtest_stage_balanced_latest.md"

log "=== Post-chain backtest done ==="
date
