#!/usr/bin/env bash
# Retrain the Qwen3.5-0.8B LoRA state classifier on the R9700, on the
# Chinese-only seed-42 train split (the split the evaluation holds out from).
#
# Uses a venv with a ROCm 10.1 build of torch: training on this card page-faulted
# under ROCm 7.2 (the gfx1201 rocBLAS issue), and 10.1 carries the upstream fix.
# The trainer checkpoints every epoch, so a stopped or crashed run continues
# with the same command.
#
#   bash scripts/train_classifier_r9700.sh --probe    # 40 steps, prints steps/min and a projection
#   bash scripts/train_classifier_r9700.sh            # full run, resumes if a checkpoint exists
#
# Stop:     systemctl --user stop csen346-clftrain   (when started with systemd-run)
# Progress: cat ~/csen346-runs/STATUS ; tail ~/csen346-runs/clf-train/train.log
#
# Env: TRAIN_PYTHON (python of the ROCm 10.1 venv), OUT_DIR, CAP_S (21600),
#      LATEST_START_HOUR (17), EXTRA_ARGS (extra trainer flags), WAIT_LOCK_S (0).
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO" || exit 1
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

TRAIN_PYTHON="${TRAIN_PYTHON:-$HOME/rocm10-probe/.venv/bin/python}"
OUT_DIR="${OUT_DIR:-results/state-clf-qwen3.5-0.8b-lora-zh}"
CAP_S="${CAP_S:-21600}"
# The GPU is loud and in a bedroom: a full run is hours, so do not start one late.
LATEST_START_HOUR="${LATEST_START_HOUR:-17}"
WAIT_LOCK_S="${WAIT_LOCK_S:-0}"
MODE="${1:-run}"
LOGS="$HOME/csen346-runs/clf-train"
STATUS="$HOME/csen346-runs/STATUS"
mkdir -p "$LOGS"

JOB_PID=""
status() { echo "$(date -Is) clf-train: $*" | tee -a "$STATUS"; }
trap '[[ -n "$JOB_PID" ]] && kill "$JOB_PID" 2>/dev/null' EXIT
trap 'status "interrupted"; exit 130' INT TERM
vram_gb() { cat /sys/class/drm/card*/device/mem_info_vram_used | sort -n | tail -1 | awk '{printf "%.1f", $1/1e9}'; }

# One GPU job at a time; optionally wait for the one that is running.
exec 9>"$HOME/csen346-runs/.lock"
if ! flock -w "$WAIT_LOCK_S" 9; then
  status "another GPU run holds the lock; nothing started"
  exit 2
fi

fail=0
check() {
  local what=$1; shift
  if "$@" >/dev/null 2>&1; then echo "  ok    $what"; else echo "  FAIL  $what"; fail=1; fi
}
echo "Preflight ($(hostname), $(git rev-parse --short HEAD), $(date -Is))"
check "training python present" test -x "$TRAIN_PYTHON"
check "torch sees the GPU" "$TRAIN_PYTHON" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"
check "GPU idle (VRAM in use: $(vram_gb) GB, limit 2)" awk -v v="$(vram_gb)" 'BEGIN { exit !(v < 2.0) }'
check "tracked files unmodified" test -z "$(git status --porcelain --untracked-files=no)"
if [[ $fail -ne 0 ]]; then
  status "preflight FAILED, nothing started"
  exit 2
fi

COMMON=(scripts/train_state_classifier_34way.py --model-id Qwen/Qwen3.5-0.8B-Base --lora
  --lora-r 8 --lora-alpha 16 --batch_size 8 --bf16-autocast)
# shellcheck disable=SC2206  # EXTRA_ARGS is a flag list, split on purpose
EXTRA=(${EXTRA_ARGS:-})
export PYTHONPATH="$REPO" PYTHONUNBUFFERED=1 WANDB_MODE="${WANDB_MODE:-disabled}"
torch_version="$("$TRAIN_PYTHON" -c 'import torch; print(torch.__version__)')"

if [[ "$MODE" == "--probe" ]]; then
  status "probe started (40 steps, torch $torch_version)"
  timeout 1800 "$TRAIN_PYTHON" "${COMMON[@]}" "${EXTRA[@]}" --max-steps 40 \
    --out-dir "$OUT_DIR-probe" >"$LOGS/probe.log" 2>&1 9>&- &
  JOB_PID=$!
  wait "$JOB_PID"
  rc=$?
  JOB_PID=""
  status "probe exit $rc; $(grep -a -i -E 'steps/min|projected' "$LOGS/probe.log" | tail -1 | cut -c1-200)"
  rm -rf "$OUT_DIR-probe"
  exit $rc
fi

if (( 10#$(date +%H) > LATEST_START_HOUR )); then
  status "not started: past ${LATEST_START_HOUR}:59"
  exit 1
fi
status "training started (torch $torch_version, out $OUT_DIR, cap ${CAP_S}s, commit $(git rev-parse --short HEAD))"
timeout "$CAP_S" "$TRAIN_PYTHON" "${COMMON[@]}" "${EXTRA[@]}" --resume \
  --out-dir "$OUT_DIR" >>"$LOGS/train.log" 2>&1 9>&- &
JOB_PID=$!
wait "$JOB_PID"
rc=$?
JOB_PID=""
if [[ $rc -eq 0 && -f "$OUT_DIR/final/model.safetensors" ]]; then
  status "training complete: $(grep -a -E 'test split|state acc|accuracy' "$LOGS/train.log" | tail -1 | cut -c1-160)"
else
  status "training stopped (exit $rc). Run the same command again to resume from the last epoch. See $LOGS/train.log"
fi
status "VRAM now $(vram_gb) GB"
exit $rc
