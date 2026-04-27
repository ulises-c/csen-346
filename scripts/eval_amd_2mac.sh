#!/usr/bin/env bash
# Run KELE eval: AMD Linux teacher + two Mac Mini llama.cpp consultants (parallel workers).
#
# Worker 1 → MAC1 (even-indexed dialogues)
# Worker 2 → MAC2 (odd-indexed dialogues)
#
# Usage:
#   ./scripts/eval_amd_2mac.sh                   # full run
#   ./scripts/eval_amd_2mac.sh --limit 10        # smoke test (10 total, 5 per worker)
#   ./scripts/eval_amd_2mac.sh --limit 10 --new  # fresh run, archives previous results
#
# Mac Mini URLs (override via env vars if hostnames differ):
#   MAC1_CONSULTANT_URL=http://mac-mini-1.local:8080/v1
#   MAC2_CONSULTANT_URL=http://mac-mini-2.local:8080/v1

set -euo pipefail
cd "$(dirname "$0")/.."

# ── Config ────────────────────────────────────────────────────────────────────
EXPERIMENT="local-mac-m4"
MAC1_URL="${MAC1_CONSULTANT_URL:-http://Ulisess-Mac-mini.local:8080/v1}"
MAC2_URL="${MAC2_CONSULTANT_URL:-http://Ulisess-Mac-mini-2.local:8080/v1}"

set -a
source configs/teachers/socrat-r9700.env
set +a
TEACHER_PORT="${TEACHER_PORT:-8001}"

OUTPUT_DIR="results/$EXPERIMENT"
ORCH_LOG="logs/orchestration.log"
TEACHER_LOG="logs/teacher.log"
WORKER1_LOG="logs/worker1.log"
WORKER2_LOG="logs/worker2.log"
PROGRESS_W1="$OUTPUT_DIR/progress_worker1.log"
PROGRESS_W2="$OUTPUT_DIR/progress_worker2.log"
PROGRESS_LOG="$OUTPUT_DIR/progress.log"

mkdir -p logs "$OUTPUT_DIR/dialogues"

# ── Logging ───────────────────────────────────────────────────────────────────
log() { echo "[$(date -Iseconds)] $*" | tee -a "$ORCH_LOG"; }

log "=== KELE Dual-Mac Evaluation: $EXPERIMENT ==="
log "Worker 1 consultant: $MAC1_URL"
log "Worker 2 consultant: $MAC2_URL"

# ── Parse args ────────────────────────────────────────────────────────────────
EXTRA_ARGS=("$@")
FRESH=false
for arg in "$@"; do
    [[ "$arg" == "--new" ]] && FRESH=true
done

# ── Check both consultants ────────────────────────────────────────────────────
log "Checking Mac Mini 1 at $MAC1_URL ..."
if ! curl -sf "${MAC1_URL}/models" > /dev/null; then
    log "ERROR: Mac Mini 1 unreachable. Run serve_consultant_llamacpp.sh on it first."
    exit 1
fi
log "Mac Mini 1 OK."

log "Checking Mac Mini 2 at $MAC2_URL ..."
if ! curl -sf "${MAC2_URL}/models" > /dev/null; then
    log "ERROR: Mac Mini 2 unreachable. Run serve_consultant_llamacpp.sh on it first."
    exit 1
fi
log "Mac Mini 2 OK."

# ── Archive previous results (host handles --new before workers start) ────────
if $FRESH; then
    log "Archiving previous results..."
    poetry run python - <<'PYEOF'
import json, sys
from datetime import datetime
from pathlib import Path

output_dir = Path("results/local-mac-m4")
prev_config = output_dir / "run_config.json"
prev_metrics = output_dir / "metrics_summary.json"

if prev_config.exists() or prev_metrics.exists():
    existing = sorted(output_dir.glob("run[0-9]*_*/"))
    next_n = len(existing) + 1
    try:
        ts_raw = json.loads(prev_config.read_text())["started_at"] if prev_config.exists() else None
    except Exception:
        ts_raw = None
    ts = (ts_raw or datetime.now().astimezone().isoformat(timespec="seconds")).replace(":", "-")
    archive_dir = output_dir / f"run{next_n}_{ts}"
    archive_dir.mkdir()
    for src in (prev_config, prev_metrics):
        if src.exists():
            src.rename(archive_dir / src.name)
    print(f"Archived to {archive_dir}")

dialogues_dir = output_dir / "dialogues"
cleared = 0
for f in dialogues_dir.glob("*.json"):
    f.unlink()
    cleared += 1
for p in output_dir.glob("progress*.log"):
    p.unlink()
print(f"Cleared {cleared} dialogue files and progress logs.")
PYEOF
    log "Archive complete."
fi

# Remove --new from args passed to workers (host already handled it)
WORKER_ARGS=()
skip_next=false
for arg in "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"; do
    if $skip_next; then skip_next=false; continue; fi
    [[ "$arg" == "--new" ]] && continue
    WORKER_ARGS+=("$arg")
done

# ── Start teacher ─────────────────────────────────────────────────────────────
log "Starting teacher server on port $TEACHER_PORT (log: $TEACHER_LOG)..."
./scripts/serve_teacher_local.sh > "$TEACHER_LOG" 2>&1 &
TEACHER_PID=$!

WORKER1_PID=""
WORKER2_PID=""
AGGREGATOR_PID=""

cleanup() {
    log "Shutting down..."
    [[ -n "$AGGREGATOR_PID" ]] && kill "$AGGREGATOR_PID" 2>/dev/null || true
    [[ -n "$WORKER1_PID" ]] && kill "$WORKER1_PID" 2>/dev/null || true
    [[ -n "$WORKER2_PID" ]] && kill "$WORKER2_PID" 2>/dev/null || true
    kill "$TEACHER_PID" 2>/dev/null; wait "$TEACHER_PID" 2>/dev/null || true
}
trap cleanup EXIT

log -n "Waiting for teacher to be ready..."
for i in $(seq 1 120); do
    if curl -sf "http://localhost:$TEACHER_PORT/v1/models" > /dev/null 2>&1; then
        log "Teacher ready (~${i}s)."
        break
    fi
    sleep 1
done
if ! curl -sf "http://localhost:$TEACHER_PORT/v1/models" > /dev/null 2>&1; then
    log "ERROR: Teacher failed to start after 120s. Check $TEACHER_LOG"
    exit 1
fi

# ── Launch workers ────────────────────────────────────────────────────────────
log "Launching worker 1 (→ $MAC1_URL) ..."
CONSULTANT_BASE_URL="$MAC1_URL" \
    poetry run python -m src.project.kele --experiment "$EXPERIMENT" evaluate \
        --output "$OUTPUT_DIR" \
        --worker-id 0 --num-workers 2 \
        "${WORKER_ARGS[@]+"${WORKER_ARGS[@]}"}" \
        > "$WORKER1_LOG" 2>&1 &
WORKER1_PID=$!

log "Launching worker 2 (→ $MAC2_URL) ..."
CONSULTANT_BASE_URL="$MAC2_URL" \
    poetry run python -m src.project.kele --experiment "$EXPERIMENT" evaluate \
        --output "$OUTPUT_DIR" \
        --worker-id 1 --num-workers 2 \
        "${WORKER_ARGS[@]+"${WORKER_ARGS[@]}"}" \
        > "$WORKER2_LOG" 2>&1 &
WORKER2_PID=$!

log "Workers started (PIDs: $WORKER1_PID, $WORKER2_PID)."
log "Tailing: tail -f $WORKER1_LOG  |  tail -f $WORKER2_LOG"

# ── Aggregator loop (combined progress.log) ───────────────────────────────────
aggregate_progress() {
    while true; do
        local c1=0 t1=0 c2=0 t2=0
        if [[ -f "$PROGRESS_W1" ]]; then
            read -r line < "$PROGRESS_W1" || true
            c1="${line%%/*}"; rest="${line#*/}"; t1="${rest%% *}"
        fi
        if [[ -f "$PROGRESS_W2" ]]; then
            read -r line < "$PROGRESS_W2" || true
            c2="${line%%/*}"; rest="${line#*/}"; t2="${rest%% *}"
        fi
        local completed=$(( c1 + c2 ))
        local total=$(( t1 + t2 ))
        if [[ $total -gt 0 ]]; then
            local pct
            pct=$(awk "BEGIN {printf \"%.1f\", $completed/$total*100}")
            printf "%s/%s %s%%  (worker1: %s/%s  worker2: %s/%s)\n" \
                "$completed" "$total" "$pct" "$c1" "$t1" "$c2" "$t2" \
                > "$PROGRESS_LOG"
        fi
        sleep 5
    done
}
aggregate_progress &
AGGREGATOR_PID=$!

# ── Wait for both workers ─────────────────────────────────────────────────────
log "Waiting for workers to finish..."
wait "$WORKER1_PID" && log "Worker 1 done." || log "Worker 1 exited with error."
wait "$WORKER2_PID" && log "Worker 2 done." || log "Worker 2 exited with error."

kill "$AGGREGATOR_PID" 2>/dev/null || true
AGGREGATOR_PID=""

# ── Merge run configs ─────────────────────────────────────────────────────────
log "Merging run configs..."
poetry run python - <<'PYEOF'
import json
from pathlib import Path

output_dir = Path("results/local-mac-m4")
configs = []
for i in (1, 2):
    p = output_dir / f"run_config_worker{i}.json"
    if p.exists():
        configs.append(json.loads(p.read_text()))

if not configs:
    print("No worker configs found — skipping merge.")
else:
    merged = dict(configs[0])
    merged["total_dialogues"] = sum(c["total_dialogues"] for c in configs)
    merged["completed"] = sum(c["completed"] for c in configs)
    merged["total_elapsed_seconds"] = max(c["total_elapsed_seconds"] for c in configs)
    merged["started_at"] = min(c["started_at"] for c in configs)
    merged["finished_at"] = max(c["finished_at"] for c in configs)
    merged["consultant_base_url"] = [c["consultant_base_url"] for c in configs]
    with open(output_dir / "run_config.json", "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Merged {len(configs)} worker configs → run_config.json")
    print(f"  total: {merged['total_dialogues']}  completed: {merged['completed']}")
PYEOF

# ── Compute metrics ───────────────────────────────────────────────────────────
log "Computing metrics..."
poetry run kele-eval "$OUTPUT_DIR"

log "=== Done. Results: $OUTPUT_DIR ==="
