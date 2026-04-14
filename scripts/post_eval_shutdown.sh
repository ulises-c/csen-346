#!/usr/bin/env bash
# Watcher: waits for baseline eval to complete, stops servers, shuts down.
# Launched in background at user request after eval kickoff.

set -u
LOG=/home/max/Documents/scu/CSEN-346/csen-346/logs/post_eval_shutdown.log
DONE_MARKER=/home/max/Documents/scu/CSEN-346/csen-346/results/baseline/metrics_summary.json

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "watcher started. waiting for $DONE_MARKER"

# Poll for completion (up to 8 hours as safety upper bound)
for i in $(seq 1 960); do
    if [ -f "$DONE_MARKER" ]; then
        log "eval complete — marker file found"
        break
    fi
    sleep 30
done

if [ ! -f "$DONE_MARKER" ]; then
    log "TIMEOUT after 8h — NOT shutting down. Human intervention required."
    exit 1
fi

# Let filesystem quiesce
sleep 5

log "stopping vLLM servers..."
pkill -f "vllm serve" 2>/dev/null || true
sleep 10
pkill -9 -f "vllm serve" 2>/dev/null || true

log "releasing systemd-inhibit lock..."
pkill -f "claude-kele-eval" 2>/dev/null || true
sleep 2

log "initiating poweroff in 30s (cancel with: sudo shutdown -c)"
sleep 30

log "powering off now"
systemctl poweroff
