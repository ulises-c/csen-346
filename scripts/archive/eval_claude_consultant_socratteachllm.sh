#!/usr/bin/env bash
# Orchestrator for Experiment A: Claude consultant + SocratTeachLLM teacher.
#
# Literal mirror of the original GPT-4o baseline architecture, with Claude
# (Sonnet 4.6 or Opus 4.6) swapped into the consultant slot. SocratTeachLLM
# remains the teacher. No BERT, no few-shot, no prompt engineering — raw
# baseline test of whether frontier consultant + memorized teacher beats
# the paper's reported GPT-4o + SocratTeachLLM numbers.
#
# Tests Max's hypothesis: if the KELE authors overfit/overtrained SocratTeachLLM
# to the dataset, swapping in a frontier consultant shouldn't lift scores much.
#
# Prereq: SocratTeachLLM server running on :8001.
#   make serve-socratteachllm   # in another terminal
#
# Usage:
#   ./scripts/eval_claude_consultant_socratteachllm.sh sonnet --n 50
#   ./scripts/eval_claude_consultant_socratteachllm.sh opus   --n 50
#
# Output: results/claude-{model}-consultant-socratteachllm-n{N}/

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

MODEL="${1:-}"
if [[ "$MODEL" != "sonnet" && "$MODEL" != "opus" ]]; then
  echo "ERROR: first arg must be 'sonnet' or 'opus' (got: '${MODEL:-<empty>}')" >&2
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

EXPERIMENT="claude-${MODEL}-46-as-consultant"
N_LABEL="${N_DIALOGUES:-full}"
OUT_DIR="results/claude-${MODEL}-consultant-socratteachllm-n${N_LABEL}"

# ── API key ──────────────────────────────────────────────────────────────────
if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  if [[ -f .env ]] && grep -q "^ANTHROPIC_API_KEY=" .env; then
    ANTHROPIC_API_KEY=$(grep "^ANTHROPIC_API_KEY=" .env | head -1 | cut -d= -f2- | tr -d '"' | tr -d "'")
    export ANTHROPIC_API_KEY
  fi
fi
if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "ERROR: ANTHROPIC_API_KEY not set" >&2
  exit 1
fi
export CONSULTANT_API_KEY="$ANTHROPIC_API_KEY"

# ── SocratTeachLLM ready-check ───────────────────────────────────────────────
if ! curl -s --max-time 3 http://localhost:8001/v1/models 2>/dev/null | grep -q "SocratTeachLLM"; then
  echo "ERROR: SocratTeachLLM not responding on :8001" >&2
  echo "Boot with: make serve-socratteachllm  (in another terminal)" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
TS=$(date -u +%Y-%m-%dT%H-%M-%S)
RUN_LOG="$OUT_DIR/run_${TS}.log"
exec > >(tee -a "$RUN_LOG") 2>&1

echo "=== Claude ${MODEL^^} 4.6 (consultant) + SocratTeachLLM (teacher) ==="
echo "Started:    $(date)"
echo "Output:     $OUT_DIR"
echo "Experiment: $EXPERIMENT"
echo "N:          ${N_DIALOGUES:-681 (full)}"
echo "Architecture: literal mirror of GPT-4o baseline, Claude swapped in"
echo

INHIBIT=()
if command -v systemd-inhibit &>/dev/null; then
  INHIBIT=(systemd-inhibit --what=sleep:idle --who="claude_${MODEL}_consultant" --why=KELE-eval)
fi

KELE_PARALLEL_WORKERS="${KELE_PARALLEL_WORKERS:-4}"
echo "Parallel workers: $KELE_PARALLEL_WORKERS"

KELE_ARGS=(
  --experiment "$EXPERIMENT"
  evaluate
  --output "$OUT_DIR"
)
if [[ -n "$N_DIALOGUES" ]]; then
  KELE_ARGS+=(--limit "$N_DIALOGUES")
fi

PATH="$PWD/.venv/bin:$PATH" \
KELE_PARALLEL_WORKERS="$KELE_PARALLEL_WORKERS" \
CONSULTANT_API_KEY="$CONSULTANT_API_KEY" \
  "${INHIBIT[@]}" .venv/bin/python -m src.project.kele "${KELE_ARGS[@]}"

EVAL_EXIT=$?
echo
echo "Eval finished (exit=$EVAL_EXIT): $(date)"

if [[ -f "$OUT_DIR/metrics_summary.json" ]]; then
  echo
  echo "=== Final metrics ==="
  cat "$OUT_DIR/metrics_summary.json"
fi
echo
echo "Run log: $RUN_LOG"
