#!/usr/bin/env bash
# Orchestrator for BERT + Claude teacher + 10-shot exemplars (Anthropic API).
#
# Architecture A from docs/CLAUDE_API_TEACHER_PLAN.md:
#   - Consultant: trained Chinese-BERT 34-state classifier (local, free)
#   - Teacher: Claude Sonnet 4.6 or Opus 4.6 via Anthropic's OpenAI-compat
#     endpoint, with 10-shot stage-balanced few-shot exemplars in the system
#     prompt
#
# Cost (cached, full n=681):
#   Sonnet 4.6: ~$5    Opus 4.6: ~$8
# Cost (probe, n=5):
#   Sonnet 4.6: ~$0.05  Opus 4.6: ~$0.08
#
# Usage:
#   ./scripts/eval_bert_claude_fewshot10.sh sonnet --n 5     # probe
#   ./scripts/eval_bert_claude_fewshot10.sh sonnet --n 50    # tournament cell
#   ./scripts/eval_bert_claude_fewshot10.sh opus --n 50      # tournament cell
#   ./scripts/eval_bert_claude_fewshot10.sh sonnet           # full n=681
#
# Output: results/bert-consultant-fewshot10-claude-<model>-n<N>/
#
# Prereqs:
#   - ANTHROPIC_API_KEY exported in env or set in .env
#   - BERT checkpoint at results/state_classifier_v1/final

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"

# ── Args ──────────────────────────────────────────────────────────────────────
MODEL="${1:-}"
if [[ "$MODEL" != "sonnet" && "$MODEL" != "opus" ]]; then
  echo "ERROR: first arg must be 'sonnet' or 'opus' (got: '${MODEL:-<empty>}')" >&2
  echo "Usage: $0 {sonnet|opus} [--n N]" >&2
  exit 2
fi
shift

N_DIALOGUES=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --n) N_DIALOGUES="$2"; shift 2 ;;
    *) echo "ERROR: unknown arg: $1" >&2; exit 2 ;;
  esac
done

EXPERIMENT="claude-${MODEL}-46"
N_LABEL="${N_DIALOGUES:-full}"
OUT_DIR="results/bert-consultant-fewshot10-claude-${MODEL}-n${N_LABEL}"
BERT_CKPT="results/state_classifier_v1/final"

# ── Pre-flight ────────────────────────────────────────────────────────────────
if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  # Try loading from .env
  if [[ -f .env ]] && grep -q "^ANTHROPIC_API_KEY=" .env; then
    ANTHROPIC_API_KEY=$(grep "^ANTHROPIC_API_KEY=" .env | head -1 | cut -d= -f2- | tr -d '"' | tr -d "'")
    export ANTHROPIC_API_KEY
  fi
fi
if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "ERROR: ANTHROPIC_API_KEY not set in env or .env" >&2
  echo "Add to .env:  ANTHROPIC_API_KEY=sk-ant-..." >&2
  exit 1
fi
export TEACHER_API_KEY="$ANTHROPIC_API_KEY"

if [[ ! -d "$BERT_CKPT" ]]; then
  echo "ERROR: BERT checkpoint missing at $BERT_CKPT" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
TS=$(date -u +%Y-%m-%dT%H-%M-%S)
RUN_LOG="$OUT_DIR/run_${TS}.log"
exec > >(tee -a "$RUN_LOG") 2>&1

echo "=== BERT + Claude ${MODEL^^} 4.6 + 10-shot eval ==="
echo "Started:    $(date)"
echo "Output:     $OUT_DIR"
echo "BERT:       $BERT_CKPT"
echo "Experiment: $EXPERIMENT"
echo "N:          ${N_DIALOGUES:-681 (full)}"
echo

# ── Eval ─────────────────────────────────────────────────────────────────────
INHIBIT=()
if command -v systemd-inhibit &>/dev/null; then
  INHIBIT=(systemd-inhibit --what=sleep:idle --who="bert_claude_${MODEL}" --why=KELE-eval)
fi

# Parallel workers — Anthropic API tolerates concurrency; cap at 4 for
# tier-1 rate limits (50 RPM, 40K TPM Sonnet). Override via env.
KELE_PARALLEL_WORKERS="${KELE_PARALLEL_WORKERS:-4}"
echo "Parallel workers: $KELE_PARALLEL_WORKERS"

# Build kele args
KELE_ARGS=(
  --experiment "$EXPERIMENT"
  evaluate
  --bert-consultant "$BERT_CKPT"
  --output "$OUT_DIR"
)
if [[ -n "$N_DIALOGUES" ]]; then
  KELE_ARGS+=(--limit "$N_DIALOGUES")
fi

PATH="$ROOT/.venv/bin:$PATH" \
KELE_FEW_SHOT_TEACHER=1 KELE_FEW_SHOT_N=10 \
KELE_PARALLEL_WORKERS="$KELE_PARALLEL_WORKERS" \
TEACHER_API_KEY="$TEACHER_API_KEY" \
  "${INHIBIT[@]}" uv run python -m src.project.kele "${KELE_ARGS[@]}"

EVAL_EXIT=$?
echo
echo "Eval finished (exit=$EVAL_EXIT): $(date)"

# ── Result summary ───────────────────────────────────────────────────────────
if [[ -f "$OUT_DIR/metrics_summary.json" ]]; then
  echo
  echo "=== Final metrics ==="
  cat "$OUT_DIR/metrics_summary.json"
fi

echo
echo "=== Done ==="
echo "Run log: $RUN_LOG"
echo "Results: $OUT_DIR/"
