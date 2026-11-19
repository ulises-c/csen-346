#!/usr/bin/env bash
# Phase 4, run 1: the headline cell on the R9700, then the local judge pass.
# Also runs on an NVIDIA card: set LLAMA_SERVER to a CUDA build of llama-server.
#
#   Qwen3.5-0.8B LoRA classifier (consultant) + Gemma 4 31B Q5 (teacher), 10-shot,
#   681 Chinese test dialogues, seed 42, 4 workers; then Qwen3.8-27B Q5 as judge.
#
# Runs unattended and is safe to restart: the evaluation resumes from the
# dialogues already on disk, and a finished stage is skipped. Each stage has a
# time cap, and the GPU servers are stopped on every exit path.
#
#   bash scripts/rerun_headline_r9700.sh --preflight   # checks only, starts nothing
#   bash scripts/rerun_headline_r9700.sh               # run (about 5.5 hours)
#
# Stop it:   systemctl --user stop csen346-headline   (when started by the launcher)
#            or kill this script; the trap stops the server.
# Progress:  cat ~/csen346-runs/STATUS ; tail results/<run>/progress.log
#
# Env: RUN_NAME, LIMIT (dialogues, default all), EVAL_CAP_S (18000), JUDGE_CAP_S (14400),
#      CKPT (classifier or BERT checkpoint; "none" = Gemma is its own consultant), TEACHER_GGUF, JUDGE_MODEL (path to GGUF), JUDGE_ALIAS, SKIP_JUDGE=1,
#      LLAMA_SERVER (binary; default: the one on PATH), TEACHER_PORT, JUDGE_PORT.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO" || exit 1
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

RUN_NAME="${RUN_NAME:-r1-qwen35clf-gemma31b-fewshot10-n681-seed42}"
LIMIT="${LIMIT:-}"
EVAL_CAP_S="${EVAL_CAP_S:-18000}"
JUDGE_CAP_S="${JUDGE_CAP_S:-14400}"
CKPT="${CKPT:-results/state-clf-qwen3.5-0.8b-lora-wandb/final}"
TEACHER_GGUF="${TEACHER_GGUF:-$HOME/Documents/models/weights/gemma-4-31B-it-UD-Q5_K_XL.gguf}"
JUDGE_MODEL="${JUDGE_MODEL:-$HOME/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q5_K_M.gguf}"
JUDGE_ALIAS="${JUDGE_ALIAS:-Qwen3.8-27B-Q5_K_M}"
TEACHER_PORT="${TEACHER_PORT:-8085}" # 8080 is held by llama-swap on ollie-cachy
JUDGE_PORT="${JUDGE_PORT:-8086}"
OUT="results/$RUN_NAME"
LOGS="$HOME/csen346-runs/$RUN_NAME"
export LLAMA_SERVER="${LLAMA_SERVER:-$(command -v llama-server)}"
# A llama.cpp build outside the system paths keeps its backend libraries next to the binary.
LD_LIBRARY_PATH="$(dirname "$LLAMA_SERVER")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH
STATUS="$HOME/csen346-runs/STATUS"
mkdir -p "$LOGS"

SERVER_PID=""
JOB_PID=""
status() { echo "$(date -Is) $RUN_NAME: $*" | tee -a "$STATUS"; }
stop_server() {
  [[ -n "$SERVER_PID" ]] || return 0
  kill "$SERVER_PID" 2>/dev/null
  for _ in 1 2 3 4 5 6; do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
  kill -9 "$SERVER_PID" 2>/dev/null
  SERVER_PID=""
}
cleanup() {
  [[ -n "$JOB_PID" ]] && pkill -TERM -P "$JOB_PID" 2>/dev/null
  [[ -n "$JOB_PID" ]] && kill "$JOB_PID" 2>/dev/null
  stop_server
}
trap 'cleanup' EXIT
trap 'status "interrupted"; exit 130' INT TERM

# Run a command under a time cap in the background and wait for it, so that a
# signal to this script is handled at once instead of after the command ends.
run_capped() { # seconds, logfile, command...
  local cap=$1 log=$2; shift 2
  timeout "$cap" "$@" >>"$log" 2>&1 &
  JOB_PID=$!
  wait "$JOB_PID"
  local rc=$?
  JOB_PID=""
  return $rc
}

vram_gb() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1 | awk '{printf "%.1f", $1/1024}'
  else
    cat /sys/class/drm/card*/device/mem_info_vram_used | sort -n | tail -1 | awk '{printf "%.1f", $1/1e9}'
  fi
}
gpu_backend() { "$LLAMA_SERVER" --list-devices 2>/dev/null | awk -F: '/^[[:space:]]+[A-Za-z]+[0-9]+:/ { gsub(/[[:space:]]/, "", $1); print $1; exit }'; }
wait_ready() { # url pattern
  for _ in $(seq 1 150); do
    sleep 2
    kill -0 "$SERVER_PID" 2>/dev/null || return 1
    curl -s --max-time 2 "$1" 2>/dev/null | grep -q "$2" && return 0
  done
  return 1
}

# ── Preflight ────────────────────────────────────────────────────────────────
fail=0
check() { # description, command...
  local what=$1; shift
  if "$@" >/dev/null 2>&1; then echo "  ok    $what"; else echo "  FAIL  $what"; fail=1; fi
}
echo "Preflight ($(hostname), $(git rev-parse --short HEAD), $(date -Is))"
check "teacher GGUF present" test -f "$TEACHER_GGUF"
check "judge GGUF present" test -f "$JUDGE_MODEL"
[[ "$CKPT" == "none" ]] || check "classifier weights present" test -f "$CKPT/model.safetensors"
check "llama-server present ($LLAMA_SERVER)" test -x "$LLAMA_SERVER"
check "uv environment" test -x .venv/bin/python
check "tracked files unmodified" test -z "$(git status --porcelain --untracked-files=no)"
check "port $TEACHER_PORT free" bash -c "! ss -ltn | grep -q ':$TEACHER_PORT '"
check "port $JUDGE_PORT free" bash -c "! ss -ltn | grep -q ':$JUDGE_PORT '"
# Gemma 4 31B with four slots needs 30.5 GB (of the R9700's 32); anything else on the card breaks it.
check "GPU idle (VRAM in use: $(vram_gb) GB, limit 2)" awk -v v="$(vram_gb)" 'BEGIN { exit !(v < 2.0) }'
exec 9>"$HOME/csen346-runs/.lock"
check "no other run of this script" flock -n 9
check "disk space (need 5 GB)" bash -c "[ \$(df --output=avail -BG . | tail -1 | tr -dc 0-9) -ge 5 ]"
if [[ $fail -ne 0 ]]; then
  status "preflight FAILED, nothing started"
  exit 2
fi
if [[ "${1:-}" == "--preflight" ]]; then
  echo "Preflight passed. Nothing started."
  exit 0
fi
status "preflight passed, commit $(git rev-parse --short HEAD)"

# ── Stage 1: evaluation ──────────────────────────────────────────────────────
if [[ -f "$OUT/metrics_summary.json" ]]; then
  status "evaluation already complete, skipping"
else
  LLAMA_ARG_REASONING=off bash scripts/serve_gemma4_31b_q5.sh -m "$TEACHER_GGUF" -p "$TEACHER_PORT" \
    >"$LOGS/teacher_server.log" 2>&1 9>&- &
  SERVER_PID=$!
  if ! wait_ready "http://localhost:$TEACHER_PORT/v1/models" '"Gemma 4 31B"'; then
    status "teacher server did not come up; see $LOGS/teacher_server.log"
    exit 3
  fi
  status "teacher server ready, VRAM $(vram_gb) GB; evaluation started (cap ${EVAL_CAP_S}s)"
  limit_args=()
  [[ -n "$LIMIT" ]] && limit_args=(--limit "$LIMIT")
  export TEACHER_BASE_URL="http://localhost:$TEACHER_PORT/v1"
  export CONSULTANT_BASE_URL="http://localhost:$TEACHER_PORT/v1"
  export KELE_BERT_DEVICE=cpu KELE_FEW_SHOT_TEACHER=1 KELE_FEW_SHOT_N=10 KELE_PARALLEL_WORKERS=4
  export WANDB_MODE=disabled
  consultant_args=(--bert-consultant "$CKPT")
  [[ "$CKPT" == "none" ]] && consultant_args=()
  KELE_RUN_NOTES="llama.cpp $("$LLAMA_SERVER" --version 2>&1 | grep -o 'build [0-9]*' | head -1), $(gpu_backend), --reasoning off; teacher $(basename "$TEACHER_GGUF"); classifier on CPU; host $(hostname)"
  export KELE_RUN_NOTES
  run_capped "$EVAL_CAP_S" "$LOGS/eval.log" .venv/bin/python -m src.project.kele \
    --experiment gemma4-31b-local evaluate "${consultant_args[@]}" "${limit_args[@]}" --output "$OUT"
  rc=$?
  stop_server
  done_n=$(find "$OUT/dialogues" -name '*.json' 2>/dev/null | wc -l)
  if [[ $rc -ne 0 || ! -f "$OUT/metrics_summary.json" ]]; then
    status "evaluation stopped (exit $rc, $done_n dialogues on disk). Rerun this script to resume."
    exit 4
  fi
  status "evaluation complete: $done_n dialogues, $(tr -d '\n ' <"$OUT/metrics_summary.json" | head -c 150)"
fi

# ── Stage 2: judge ───────────────────────────────────────────────────────────
if [[ "${SKIP_JUDGE:-0}" == "1" ]]; then
  status "judge skipped (SKIP_JUDGE=1)"
elif [[ -f "$OUT/judge_summary.json" ]] && grep -q '"complete": true' "$OUT/judge_summary.json"; then
  status "judge already complete, skipping"
else
  # The judge script caps replies at 200 tokens, so a thinking model must have reasoning off.
  LLAMA_ARG_REASONING=off "$LLAMA_SERVER" -m "$JUDGE_MODEL" -a "$JUDGE_ALIAS" -ngl 99 -c 32768 -np 4 \
    --host 127.0.0.1 --port "$JUDGE_PORT" >"$LOGS/judge_server.log" 2>&1 9>&- &
  SERVER_PID=$!
  if ! wait_ready "http://127.0.0.1:$JUDGE_PORT/health" '"ok"'; then
    status "judge server did not come up; see $LOGS/judge_server.log"
    exit 5
  fi
  status "judge server ready ($(basename "$JUDGE_MODEL"), sha256 $(sha256sum "$JUDGE_MODEL" | cut -c1-16)); judging started (cap ${JUDGE_CAP_S}s)"
  run_capped "$JUDGE_CAP_S" "$LOGS/judge.log" .venv/bin/python scripts/llm_judge_eval.py "$OUT" \
    --backend api --base-url "http://127.0.0.1:$JUDGE_PORT/v1" --api-key local \
    --model "$JUDGE_ALIAS" --workers 4
  rc=$?
  stop_server
  if [[ $rc -ne 0 ]]; then
    status "judge pass not complete (exit $rc; 3 means some turns were not scored). See $LOGS/judge.log"
  else
    status "judge complete: $(tail -1 "$LOGS/judge.log")"
  fi
fi

# ── Stage 3: local archive (dialogues are not in git) ────────────────────────
tar czf "$HOME/csen346-runs/$RUN_NAME.tgz" -C results "$RUN_NAME" &&
  status "archived to ~/csen346-runs/$RUN_NAME.tgz ($(du -h "$HOME/csen346-runs/$RUN_NAME.tgz" | cut -f1)); VRAM now $(vram_gb) GB"
status "finished"
