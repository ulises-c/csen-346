#!/usr/bin/env bash
# Overnight autonomous chain triggered behind the 4-cell Qwen 27B grid.
#
# Sequence (each step waits for the previous):
#   1. WAIT      — poll for grid completion (all 4 metrics OR grid PID exits)
#   2. JUDGE     — LLM-judge re-eval on every cell with dialogues/ (Claude Sonnet)
#   3. DECIDE    — promote_if_winner.py emits WINNER_* + PROMOTE=true|false
#   4a. PROMOTE  — Phase 3 full n=681 with winning cell
#   4b. DEFER    — Layer-2: T4 × {Gemma, A3B, Qwen27B-winner-mode} at n=400 random
#   5. BILINGUAL — T4 on SocratDataset-EN, n=100 random, Gemma teacher
#   6. STAGE-E   — (stub) weighted-loss retrain for T4; logged TODO, not auto-run
#
# Launch with:
#   nohup bash scripts/overnight_qwen27b_chain.sh \
#     > results/_orchestrator_logs/overnight_chain_$(date -u +%Y-%m-%dT%H-%M-%S).log 2>&1 &
#   disown
#
# Idempotent: skips judge if judge_summary.json exists, skips runs whose
# OUT_DIR already has metrics_summary.json. Safe to re-launch if interrupted.

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
# ROOT/BGE_CKPT preserved for documentation + reuse by sourced sub-scripts even
# though shellcheck doesn't see the cross-script reference.
# shellcheck disable=SC2034
ROOT="$(pwd)"

GRID_CELLS=(
  "results/bge-small-bert-qwen27b-fewshot10-n50-fixed"
  "results/t4-bert-qwen27b-fewshot10-n50-fixed"
  "results/bge-small-bert-qwen27b-nothink-fewshot10-n50-fixed"
  "results/t4-bert-qwen27b-nothink-fewshot10-n50-fixed"
)
GRID_PID_FILE="/tmp/qwen27b_grid_orch.pid"
T4_CKPT="results/state-clf-qwen3.5-0.8b-lora/final"
# shellcheck disable=SC2034
BGE_CKPT="results/state_classifier_v1/final"
SAMPLE_SEED=42
JUDGE_MODEL="claude-sonnet-4-6"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Step 1: wait for grid ─────────────────────────────────────────────────
log "=== Step 1: Wait for Qwen 27B grid ==="
GRID_PID=""
[[ -f "$GRID_PID_FILE" ]] && GRID_PID="$(cat "$GRID_PID_FILE")"
log "Grid PID: ${GRID_PID:-unknown}"

while true; do
  done_count=0
  for cell in "${GRID_CELLS[@]}"; do
    [[ -f "$cell/metrics_summary.json" ]] && done_count=$((done_count + 1))
  done
  if [[ "$done_count" -eq "${#GRID_CELLS[@]}" ]]; then
    log "Grid done: 4/4 cells have metrics_summary.json"
    break
  fi
  if [[ -n "$GRID_PID" ]] && ! kill -0 "$GRID_PID" 2>/dev/null; then
    log "WARN: Grid orchestrator PID $GRID_PID exited with only $done_count/4 cells complete — proceeding with partial data"
    break
  fi
  sleep 60
done

# ── Step 2: LLM-judge ─────────────────────────────────────────────────────
log "=== Step 2: LLM-judge re-evaluation (Claude Sonnet 4.6) ==="
for cell in "${GRID_CELLS[@]}"; do
  if [[ ! -d "$cell/dialogues" ]] || [[ -z "$(ls "$cell/dialogues" 2>/dev/null)" ]]; then
    log "  [skip] $cell — no dialogues/"
    continue
  fi
  if [[ -f "$cell/judge_summary.json" ]]; then
    log "  [skip] $cell — judge_summary.json already exists"
    continue
  fi
  log "  [judge] $cell"
  uv run python scripts/llm_judge_eval.py "$cell" --model "$JUDGE_MODEL" --workers 10 \
    || log "  WARN: judge failed for $cell — continuing"
done

# ── Step 3: promotion decision ────────────────────────────────────────────
log "=== Step 3: Promotion decision ==="
# Script prints human summary to stderr (goes to our log via 2>&1 inheritance)
# and env-var assignments to stdout (captured here for source).
.venv/bin/python scripts/promote_if_winner.py > /tmp/promote_envs.txt
if [[ ! -s /tmp/promote_envs.txt ]]; then
  log "ERROR: promote_if_winner.py produced no decision — aborting chain"
  exit 2
fi
# shellcheck disable=SC1091
source /tmp/promote_envs.txt
log "Winner: $WINNER_CELL (composite=$WINNER_COMPOSITE)"
log "Promote to n=681: $PROMOTE"

# ── Step 4: large-scale run ───────────────────────────────────────────────
log "=== Step 4: Large-scale run ==="

run_if_missing() {
  local out_dir="$1"; shift
  if [[ -f "$out_dir/metrics_summary.json" ]]; then
    log "  [skip] $out_dir — metrics already exist"
    return 0
  fi
  log "  [run] $out_dir"
  "$@" || log "  WARN: $out_dir run failed (exit=$?)"
  sleep 15
}

if [[ "$PROMOTE" == "true" ]]; then
  log "Path: PROMOTE — Phase 3 n=681 with $WINNER_CELL"
  PHASE3_OUT="results/phase3-${WINNER_CONSULTANT}-bert-qwen27b-${WINNER_TEACHER_MODE}-n681"
  run_if_missing "$PHASE3_OUT" \
    env OUT_DIR="$PHASE3_OUT" BERT_CKPT="$WINNER_BERT_CKPT" \
    bash "scripts/$WINNER_SCRIPT"
else
  log "Path: DEFER — Layer-2 T4 × 3 teachers at n=400 (random, seed=$SAMPLE_SEED)"

  run_if_missing "results/layer2-t4-gemma-n400" \
    env OUT_DIR="results/layer2-t4-gemma-n400" BERT_CKPT="$T4_CKPT" \
        LIMIT=400 SAMPLE_SEED="$SAMPLE_SEED" \
    bash scripts/eval_bert_gemma_fewshot10_full.sh

  run_if_missing "results/layer2-t4-a3b-n400" \
    env OUT_DIR="results/layer2-t4-a3b-n400" BERT_CKPT="$T4_CKPT" \
        LIMIT=400 SAMPLE_SEED="$SAMPLE_SEED" \
    bash scripts/eval_bert_a3b_fewshot10_full.sh

  WINNER_QWEN_SCRIPT="eval_bert_qwen27b_fewshot10_full.sh"
  [[ "$WINNER_TEACHER_MODE" == "nothink" ]] && WINNER_QWEN_SCRIPT="eval_bert_qwen27b_nothink_fewshot10_full.sh"
  run_if_missing "results/layer2-t4-qwen27b-${WINNER_TEACHER_MODE}-n400" \
    env OUT_DIR="results/layer2-t4-qwen27b-${WINNER_TEACHER_MODE}-n400" BERT_CKPT="$T4_CKPT" \
        LIMIT=400 SAMPLE_SEED="$SAMPLE_SEED" \
    bash "scripts/$WINNER_QWEN_SCRIPT"
fi

# ── Step 5: bilingual probe ───────────────────────────────────────────────
log "=== Step 5: Bilingual probe (T4 on SocratDataset-EN, n=100 random, Gemma teacher) ==="
run_if_missing "results/bilingual-probe-t4-en-stage1-n100" \
  env OUT_DIR="results/bilingual-probe-t4-en-stage1-n100" BERT_CKPT="$T4_CKPT" \
      LIMIT=100 SAMPLE_SEED="$SAMPLE_SEED" \
      DATASET_PATH="references/KELE-EN/SocratDataset.json" \
  bash scripts/eval_bert_gemma_fewshot10_full.sh

# ── Step 6: stage-e weighted-loss retrain (deferred — needs Trainer subclass) ─
log "=== Step 6: Stage-e weighted-loss retrain ==="
log "  TODO: requires custom Trainer with stage-weighted CrossEntropy"
log "  (~30 lines on top of train_state_classifier_34way.py loss site)."
log "  Skipping auto-launch this run; bilingual probe + Layer-2 fills the budget."

log "=== Overnight chain done ==="
date
