#!/usr/bin/env bash
# This script builds GitHub-markdown PR comments via printf; the format strings
# are single-quoted on purpose (%s positional args + literal backticks/newlines),
# so SC2016 ("expressions don't expand in single quotes") is a false positive.
# shellcheck disable=SC2016
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EOS_LOG="$REPO_DIR/outputs/eos-gate-gemma4-31b/train.log"
FINAL_ADAPTER="$REPO_DIR/outputs/eos-gate-gemma4-31b/final"
STAGE2_LOG="$REPO_DIR/outputs/sft-stage2-gemma4-31b/train.log"
SELF_LOG="$REPO_DIR/outputs/monitor_eos_gate.log"
PR_NUMBER=101
MAX_RETRIES=2
POLL_SECONDS=300

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$SELF_LOG"
}

post_pr() {
    local body="$1"
    gh pr comment "$PR_NUMBER" --body "$body" >> "$SELF_LOG" 2>&1 || \
        log "WARNING: gh pr comment failed (network?)"
}

training_running() {
    pgrep -f "train_sft\.py" > /dev/null 2>&1
}

crash_hint() {
    if [[ -f "$EOS_LOG" ]]; then
        grep -i "page fault\|traceback\|runtimeerror\|out of memory\|oom\|killed" "$EOS_LOG" \
            | tail -3 | sed 's/^/  /' || true
    fi
}

last_step() {
    if [[ -f "$EOS_LOG" ]]; then
        grep -oE "[0-9]+/100" "$EOS_LOG" | tail -1 || true
    fi
}

start_train() {
    log "Starting EOS gate training run"
    if [[ -d "$REPO_DIR/outputs/eos-gate-gemma4-31b" ]]; then
        find "$REPO_DIR/outputs/eos-gate-gemma4-31b" -mindepth 1 -delete
    fi
    cd "$REPO_DIR" && make train-gemma4-31b-eos-gate
}

kill_and_clean() {
    log "Killing stray train_sft.py processes"
    pkill -9 -f "train_sft\.py" 2>/dev/null || true
    local waited=0
    while pgrep -f "train_sft\.py" > /dev/null 2>&1 && [[ $waited -lt 30 ]]; do
        sleep 2
        (( waited += 2 ))
    done
    log "KFD settle wait (20s)"
    sleep 20
}

run_eval() {
    log "Training complete — running EOS gate eval"
    local eval_out
    eval_out=$(cd "$REPO_DIR" && \
        env TORCH_USE_HIPBLASLT=0 \
            PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8 \
            TRAIN_BASE_MODEL=unsloth/gemma-4-31B-it-unsloth-bnb-4bit \
            TRAIN_PREQ=true \
            TRAIN_METHOD=qlora \
            TRAIN_BF16=true \
            uv run --no-sync python scripts/eos_gate.py \
            --config configs/train-sft-stage2-gemma4-31b.env \
            --adapter outputs/eos-gate-gemma4-31b/final 2>&1)
    printf '%s\n' "$eval_out" | tee -a "$SELF_LOG"
    printf '%s\n' "$eval_out"
}

# ── Main ────────────────────────────────────────────────────────────────────

mkdir -p "$REPO_DIR/outputs"
log "Monitor starting (PR #$PR_NUMBER, poll every ${POLL_SECONDS}s, max $MAX_RETRIES crash retries)"

post_pr "$(printf '**EOS gate retrain started** (gfx1201 / R9700) — commit \`%s\`\n\n> Root cause fixed (`dbc61a2`): \`<turn|>\` terminator now inside \`{%% generation %%}\` block. Monitoring autonomously; next update when training completes (~115 min).' "$(cd "$REPO_DIR" && git rev-parse --short HEAD)")"

retries=0

# Give the training process a moment to start
sleep 10

while true; do
    if training_running; then
        step="$(last_step)"
        log "Training running${step:+ — last step: $step}"
        sleep "$POLL_SECONDS"
        continue
    fi

    # Process gone — determine outcome
    if [[ -d "$FINAL_ADAPTER" ]]; then
        log "Training finished — adapter at $FINAL_ADAPTER"
        break
    fi

    # Crashed
    hint="$(crash_hint)"
    log "CRASH DETECTED (retry $retries/$MAX_RETRIES)"
    [[ -n "$hint" ]] && log "Log tail: $hint"

    if (( retries >= MAX_RETRIES )); then
        post_pr "$(printf '**EOS gate training CRASHED** — exceeded %d retries, giving up.\n\nCrash hint:\n```\n%s\n```\n\nManual intervention needed. Monitor log: `outputs/monitor_eos_gate.log`' "$MAX_RETRIES" "$hint")"
        log "Max retries exceeded — exiting"
        exit 1
    fi

    post_pr "$(printf '**EOS gate training CRASHED** (retry %d/%d) — restarting after KFD cleanup.\n\nCrash hint:\n```\n%s\n```' "$((retries+1))" "$MAX_RETRIES" "$hint")"

    retries=$(( retries + 1 ))
    kill_and_clean
    start_train
    sleep 30  # let the new process start before polling
done

# ── Eval ────────────────────────────────────────────────────────────────────

eval_output="$(run_eval)"

if printf '%s' "$eval_output" | grep -q "EOS GATE  PASSED"; then
    log "EOS GATE PASSED"

    # Extract the generation snippet for the PR comment
    gate_generation="$(printf '%s' "$eval_output" | grep -A 20 "Generation:" | head -20 || true)"

    post_pr "$(printf '## EOS Gate: PASSED ✓\n\nCommit: `%s`\n\nSTEP 1 warm-up → PASS  \nSTEP 2 EOS gate → **PASS**\n\nThe `<turn|>` terminator fix (`dbc61a2`) is confirmed: model terminates before `max_new_tokens` on the realistic free-form prompt.\n\n<details><summary>Generation output</summary>\n\n```\n%s\n```\n\n</details>\n\n---\nLaunching full Stage 2 run (`make train-gemma4-31b-stage2-unsloth`).' \
        "$(cd "$REPO_DIR" && git rev-parse --short HEAD)" "$gate_generation")"

    log "Launching full Stage 2 run"
    cd "$REPO_DIR" && make train-gemma4-31b-stage2-unsloth

    log "Stage 2 run launched — monitor: tail -f $STAGE2_LOG"
    post_pr "$(printf '**Stage 2 training started.** Log: `outputs/sft-stage2-gemma4-31b/train.log`\n\nNext manual check: serving generation-primer audit before downstream eval (see handoff §2).')"

    log "Launching Stage 2 crash monitor"
    chmod +x "$REPO_DIR/scripts/monitor_stage2.sh"
    nohup setsid bash "$REPO_DIR/scripts/monitor_stage2.sh" > /dev/null 2>&1 &

else
    log "EOS GATE FAILED"

    gate_tail="$(printf '%s' "$eval_output" | tail -30)"
    post_pr "$(printf '## EOS Gate: FAILED ✗\n\nCommit: `%s`\n\nSTEP 2 EOS gate → **FAIL** — model hit max_new_tokens without terminating.\n\nThe fix did not resolve the collapse, or a new issue was introduced. Manual diagnosis needed.\n\n<details><summary>Gate output tail</summary>\n\n```\n%s\n```\n\n</details>' \
        "$(cd "$REPO_DIR" && git rev-parse --short HEAD)" "$gate_tail")"

    log "Gate failed — NOT launching Stage 2. See PR #$PR_NUMBER for details."
    exit 2
fi

log "Monitor done"
