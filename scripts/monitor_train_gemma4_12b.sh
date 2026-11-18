#!/usr/bin/env bash
# Training-side auto-resume monitor for the Gemma 4 12B SFT-uplift PoC on the
# NVIDIA RTX 4000 Ada box. Adapted from scripts/monitor_stage2.sh (the gfx1201/
# AMD 31B training monitor) — same crawl-forward-by-crashing semantics, but:
#
#   1. NVIDIA box, not gfx1201. Crash signatures + dmesg patterns are the CUDA/NVRM
#      ones (xid, "fallen off the bus", cuda error), not the ROCm/HSA ones. The
#      GPU-clean gate is `nvidia-preflight` — which `make train-gemma4-12b` runs as
#      a hard prerequisite on EVERY (re)launch — plus a VRAM-settle wait after kill,
#      instead of the AMD test_gpu_stack.sh --wait-clean path.
#   2. Adaptive power search (memory: training-host-hardware-fault — this card took a
#      power surge and crashes under load). Training is compute-bound, so it lives in
#      exactly the regime that historically faulted. Start at the card max and step
#      the power limit DOWN on each crash until the run stops faulting; the highest
#      power that crawls to completion is the best stability/speed point. Same
#      mechanism as scripts/monitor_eval_gemma4_12b.sh.
#
# Forward progress is counted in CHECKPOINTS (save_steps=50): a relaunch that lands
# a new checkpoint resets the no-progress retry counter, so a run that keeps inching
# forward across faults never exhausts MAX_RETRIES. COMPLETE means outputs/.../final
# exists.
#
# This script builds GitHub-markdown rows via printf; the single-quoted format
# strings (%s args + literal backticks/pipes) make SC2016 a false positive.
# shellcheck disable=SC2016
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR" || exit 1

OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-$REPO_DIR/outputs/sft-gemma4-12b-qlora}"
MAKE_TARGET="${TRAIN_MAKE_TARGET:-train-gemma4-12b}"
TRAIN_LOG="$OUTPUT_DIR/train.log"
FINAL_DIR="$OUTPUT_DIR/final"
SELF_LOG="$REPO_DIR/outputs/monitor_train_gemma4_12b.log"

# Progress + crash rows append to ONE pinned comment on the SFT PoC issue #130
# (same convention as monitor_stage2.sh / monitor_eval_gemma4_12b.sh). Create a
# fresh placeholder comment and export its numeric id (or paste it here). If empty,
# log_row degrades to local-only logging — the crawl still runs, it just doesn't
# post to GitHub.
ISSUE_NUMBER="${ISSUE_NUMBER:-130}"
LOG_COMMENT_ID="${LOG_COMMENT_ID:-}"
# Post a training-metrics progress row every this many steps (loss/grad_norm/lr).
# Matches save_steps so every checkpoint is reported.
PROGRESS_EVERY="${PROGRESS_EVERY:-50}"
# Bound on CONSECUTIVE no-forward-progress retries — a run that keeps advancing
# never exhausts it; only a genuinely stuck GPU (no new checkpoint across this many
# tries) stops the monitor. Crawl-forward needs a high bound, not 2.
MAX_RETRIES="${MAX_RETRIES:-8}"
POLL_SECONDS="${POLL_SECONDS:-300}"

# Adaptive power search: start high, step DOWN POWER_STEP_W on each crash, floored
# at POWER_FLOOR_W. POWER_START_W defaults to the card's reported max limit.
POWER_START_W="${POWER_START_W:-}"
POWER_STEP_W="${POWER_STEP_W:-10}"
POWER_FLOOR_W="${POWER_FLOOR_W:-70}"   # eval crawls proved stable at 85W; floor below that

# After a kill, wait for VRAM to drain before relaunch so training doesn't boot into
# a dirty allocator state. Bounded; the nvidia-preflight gate is the real health check.
VRAM_SETTLE_TIMEOUT="${VRAM_SETTLE_TIMEOUT:-120}"
VRAM_SETTLE_MIB="${VRAM_SETTLE_MIB:-1500}"

mkdir -p "$OUTPUT_DIR" "$REPO_DIR/outputs"

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$SELF_LOG"
}
now_iso() { date '+%Y-%m-%d %H:%M:%S%:z'; }

# ── Live training-log comment ────────────────────────────────────────────────
# Append one row to the pinned comment. Single writer (this monitor), so the
# read-modify-write has no concurrent-update race. gh resolves {owner}/{repo}
# from the cwd's git remote. No-op (local log only) when LOG_COMMENT_ID is unset.
log_row() {
    [[ -z "$LOG_COMMENT_ID" ]] && { log "row (no GH comment): $1"; return 0; }
    local body
    body="$(gh api "repos/{owner}/{repo}/issues/comments/$LOG_COMMENT_ID" --jq '.body' 2>>"$SELF_LOG")" || {
        log "WARNING: could not fetch log comment $LOG_COMMENT_ID — dropping row: $1"; return 0; }
    printf '%s\n%s\n' "$body" "$1" \
        | gh api -X PATCH "repos/{owner}/{repo}/issues/comments/$LOG_COMMENT_ID" -F body=@- >> "$SELF_LOG" 2>&1 \
        || log "WARNING: gh api PATCH (log comment) failed"
}

# One-line crash signature for the table's Note cell. A literal pipe would break the
# markdown row, so swap it for a fullwidth bar; the full trace is in crashlogs/.
crash_signature() {
    [[ -f "$TRAIN_LOG" ]] || return 0
    grep -iE "cuda error|out of memory|fallen off the bus|xid|nvrm|segmentation fault|sigsegv|aborted|core dumped|runtimeerror" \
        "$TRAIN_LOG" | tail -1 | awk -F'\r' '{print $NF}' | tr -d '`' \
        | sed 's/|/／/g' | cut -c1-140
}

log_progress_from_checkpoint() {
    local step="$1" epoch loss acc gn lr
    local state_file="$OUTPUT_DIR/checkpoint-$step/trainer_state.json"
    [[ -f "$state_file" ]] || return 0
    IFS=$'\t' read -r epoch loss acc gn lr < <(python3 - "$state_file" <<'PY' 2>/dev/null || true
import json, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
gs = d.get("global_step", -1)
hist = d.get("log_history", [])
m = next((e for e in reversed(hist) if e.get("step") == gs), hist[-1] if hist else {})
print("\t".join(str(m.get(k, "")) for k in
    ("epoch", "loss", "mean_token_accuracy", "grad_norm", "learning_rate")))
PY
    ) || true
    local note="epoch ${epoch:-?} @ ${CURRENT_POWER:-?}W"
    [[ -n "$acc" ]] && note="$note, acc $acc"
    log_row "$(printf '| %s | 📈 checkpoint | %s | %s | %s | %s | %s |' \
        "$(now_iso)" "$step" "${loss:-}" "${gn:-}" "${lr:-}" "$note")"
}

training_running() {
    pgrep -f "train_sft\.py" > /dev/null 2>&1
}

# Total training steps (max_steps, e.g. 4826) from the latest checkpoint's
# trainer_state.json — stable for the run and independent of which tqdm bars are in
# the log, so the fallback denominator is never a Map-bar total.
train_total_steps() {
    local step state
    step="$(latest_ckpt_step)"
    if (( step < 0 )); then return 0; fi
    state="$OUTPUT_DIR/checkpoint-$step/trainer_state.json"
    [[ -f "$state" ]] || return 0
    python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("max_steps",""))' \
        "$state" 2>/dev/null || true
}

last_step() {
    [[ -f "$TRAIN_LOG" ]] || return 0
    # train.log holds multiple N/M tqdm bars: the training step bar (N/4826) and the
    # one-time dataset tokenization Map bars. Anchor on the rate unit: at many s/step
    # the training bar is the only one rendered as "s/it" (Map → "examples/s", W&B
    # upload → "MB/s", shard-load → "it/s"). tr '\r'->'\n' splits the \r-joined tqdm
    # updates so the most recent step wins.
    local n ckpt total
    n="$(tr '\r' '\n' < "$TRAIN_LOG" | grep -E 's/it\]' | grep -oE '[0-9]+/[0-9]+' | tail -1 || true)"
    if [[ -n "$n" ]]; then printf '%s' "$n"; return 0; fi
    # No training bar yet (crash during model load / tokenization) — degrade to the
    # latest checkpoint step rather than a Map-bar denominator or a blank cell.
    ckpt="$(latest_ckpt_step)"
    if (( ckpt < 0 )); then return 0; fi
    total="$(train_total_steps)"
    printf '%s%s' "$ckpt" "${total:+/$total}"
}

crash_hint() {
    [[ -f "$TRAIN_LOG" ]] || return 0
    local step
    step="$(last_step)"
    [[ -n "$step" ]] && printf '  fault near step %s\n' "$step"
    grep -iE "cuda error|out of memory|fallen off the bus|xid|nvrm|segmentation fault|sigsegv|aborted|core dumped|traceback|runtimeerror" \
        "$TRAIN_LOG" | tail -4 | sed 's/^/  /' || true
}

# An OOM is DETERMINISTIC — the same config will OOM identically every relaunch, so
# retrying (and stepping power down, which OOM ignores) is futile. Distinguished
# from the probabilistic GPU surge fault this monitor exists to crawl through.
is_oom_crash() {
    [[ -f "$TRAIN_LOG" ]] || return 1
    grep -qiE "outofmemoryerror|cuda out of memory|torch\.cuda\.OutOfMemory" "$TRAIN_LOG"
}

# The make target writes train.log with `>` (truncate), so each relaunch would
# overwrite the crashed run's full traceback. Archive the whole log + an nvidia-smi
# + dmesg snapshot per crash BEFORE relaunch so an overnight crawl leaves a
# reviewable trail of every fault, not just the 3-line hint.
archive_crash_log() {
    local ckpt="$1" ts dest crashdir
    [[ -f "$TRAIN_LOG" ]] || return 0
    ts="$(date '+%Y%m%d-%H%M%S')"
    crashdir="$OUTPUT_DIR/crashlogs"
    mkdir -p "$crashdir"
    dest="$crashdir/crash-step${ckpt}-${ts}.log"
    if cp "$TRAIN_LOG" "$dest" 2>/dev/null; then
        log "Archived full crash log → $dest"
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi > "$crashdir/nvidia-smi-step${ckpt}-${ts}.txt" 2>&1 || true
    fi
    dmesg 2>/dev/null \
        | grep -iE "nvrm|xid|nvidia|gpu has fallen" \
        | tail -40 > "$crashdir/dmesg-step${ckpt}-${ts}.log" 2>/dev/null || true
}

# Wait for VRAM to drain after a kill, so the relaunch doesn't boot into a dirty
# allocator state. Bounded; nvidia-preflight (a make prerequisite) is the real gate.
wait_gpu_idle() {
    command -v nvidia-smi >/dev/null 2>&1 || return 0
    local waited=0 used
    while (( waited < VRAM_SETTLE_TIMEOUT )); do
        used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
        [[ "$used" =~ ^[0-9]+$ ]] || break
        if (( used < VRAM_SETTLE_MIB )); then
            log "GPU VRAM settled (${used} MiB used) after ${waited}s"
            return 0
        fi
        sleep 5; (( waited += 5 ))
    done
    log "WARNING: GPU VRAM still high after ${VRAM_SETTLE_TIMEOUT}s — relaunch anyway (preflight gates it)"
}

kill_and_clean() {
    log "Killing stray train_sft.py processes"
    pkill -9 -f "train_sft\.py" 2>/dev/null || true
    local waited=0
    while pgrep -f "train_sft\.py" > /dev/null 2>&1 && [[ $waited -lt 30 ]]; do
        sleep 2; (( waited += 2 ))
    done
    wait_gpu_idle
}

latest_ckpt_step() {
    local d max=-1 n
    for d in "$OUTPUT_DIR"/checkpoint-*; do
        [[ -d "$d" ]] || continue
        n="${d##*checkpoint-}"
        [[ "$n" =~ ^[0-9]+$ ]] && (( n > max )) && max="$n"
    done
    printf '%s' "$max"
}

# A crash landing during a checkpoint write leaves an incomplete checkpoint-N. HF
# resume always picks the highest-numbered dir, so a partial one makes every resume
# fail to load and loop on the same bad checkpoint. trainer_state.json is written
# last in _save_checkpoint, so a valid one implies the rest is complete; if it's
# missing or unparsable, quarantine the dir so resume falls back to N-1.
quarantine_bad_checkpoint() {
    local step latest
    step="$(latest_ckpt_step)"
    [[ "$step" -lt 0 ]] && { log "No checkpoint yet — resume will start from step 0"; return 0; }
    latest="$OUTPUT_DIR/checkpoint-$step"
    if [[ -f "$latest/trainer_state.json" ]] \
        && python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$latest/trainer_state.json" 2>/dev/null; then
        log "Latest checkpoint OK: checkpoint-$step (resume target)"
        return 0
    fi
    log "checkpoint-$step is INCOMPLETE (crash mid-save) — quarantining; resume falls back to prior"
    mv "$latest" "$OUTPUT_DIR/.broken-checkpoint-$step" 2>/dev/null || rm -rf "$latest"
}

# ── GPU power control (the adaptive stability search) ──────────────────────────
CURRENT_POWER=""

gpu_max_power() {
    command -v nvidia-smi >/dev/null 2>&1 || { printf '130'; return; }
    local m
    m="$(nvidia-smi --query-gpu=power.max_limit --format=csv,noheader,nounits 2>/dev/null | head -1 | cut -d. -f1)"
    [[ "$m" =~ ^[0-9]+$ ]] && printf '%s' "$m" || printf '130'
}

apply_power() {
    command -v nvidia-smi >/dev/null 2>&1 || return 0
    [[ -z "$CURRENT_POWER" ]] && return 0
    # SELF_LOG is the monitor user's own file, not a root-only path — SC2024 N/A.
    # shellcheck disable=SC2024
    if sudo -n nvidia-smi -pl "$CURRENT_POWER" >>"$SELF_LOG" 2>&1; then
        log "GPU power limit set to ${CURRENT_POWER}W"
    else
        log "WARNING: could not set power (sudo -n failed) — run 'sudo nvidia-smi -pl ${CURRENT_POWER}' manually"
    fi
}

step_power_down() {
    [[ -z "$CURRENT_POWER" ]] && return 0
    if (( CURRENT_POWER - POWER_STEP_W >= POWER_FLOOR_W )); then
        CURRENT_POWER=$(( CURRENT_POWER - POWER_STEP_W ))
        log "Stepping power down → ${CURRENT_POWER}W (crash-driven stability search)"
    else
        log "Power already at floor ${CURRENT_POWER}W (>= floor ${POWER_FLOOR_W}W) — not stepping lower"
    fi
}

start_training() {
    # Fresh data_seed each launch so resume presents a different sample order at each
    # step — breaks any data-order-sticky fault (the gfx1201 31B run hit one; cheap
    # insurance here). train_sft.py reads TRAIN_DATA_SEED; make train-gemma4-12b runs
    # nvidia-preflight first (hard GPU gate) then nohups training into train.log.
    local data_seed
    data_seed="$(date +%s)"
    log "Starting training (data_seed=$data_seed, power ${CURRENT_POWER:-default}W, nvidia-preflight gates the launch)"
    apply_power
    cd "$REPO_DIR" || return 0
    TRAIN_DATA_SEED="$data_seed" make "$MAKE_TARGET" >> "$SELF_LOG" 2>&1 \
        || log "launch aborted (nvidia-preflight failed?) — counts as no forward progress"
    sleep 30
}

log "Training monitor starting (issue #$ISSUE_NUMBER comment ${LOG_COMMENT_ID:-<none>}, poll every ${POLL_SECONDS}s, max $MAX_RETRIES retries)"

[[ -z "$POWER_START_W" ]] && POWER_START_W="$(gpu_max_power)"
CURRENT_POWER="$POWER_START_W"
log "Power search: start ${POWER_START_W}W, -${POWER_STEP_W}W/crash, floor ${POWER_FLOOR_W}W"

log_row "$(printf '| %s | ▶ monitor start | %s | | | | watching train.log @ %sW; progress every %s steps |' \
    "$(now_iso)" "$(last_step || true)" "$CURRENT_POWER" "$PROGRESS_EVERY")"

# Launch (or resume) immediately rather than waiting a poll cycle.
start_training

retries=0
last_progress_step="$(latest_ckpt_step)"
last_reported_ckpt="$(latest_ckpt_step)"

while true; do
    if training_running; then
        step="$(last_step)"
        log "Training running${step:+ — step $step}"
        ckpt_now="$(latest_ckpt_step)"
        if (( ckpt_now > 0 )) && (( ckpt_now % PROGRESS_EVERY == 0 )) \
            && (( ckpt_now > last_reported_ckpt )); then
            log_progress_from_checkpoint "$ckpt_now"
            last_reported_ckpt="$ckpt_now"
        fi
        sleep "$POLL_SECONDS"
        continue
    fi

    if [[ -d "$FINAL_DIR" ]]; then
        log "Training complete — adapter at $FINAL_DIR"
        log_row "$(printf '| %s | ✅ COMPLETE | %s | | | | adapter: %s ; stable @ %sW |' \
            "$(now_iso)" "$(last_step || true)" "${FINAL_DIR#"$REPO_DIR"/}" "$CURRENT_POWER")"
        log "Done — monitor exiting"
        exit 0
    fi

    # Forward-progress check: a new checkpoint since the last crash means the crawl
    # is advancing — reset the no-progress retry counter. Only consecutive stalls
    # (no new checkpoint) count toward MAX_RETRIES.
    ckpt_step="$(latest_ckpt_step)"
    if (( ckpt_step > last_progress_step )); then
        log "Progress since last crash: checkpoint $last_progress_step → $ckpt_step — resetting retry counter"
        last_progress_step="$ckpt_step"
        retries=0
    fi

    hint="$(crash_hint)"
    log "CRASH DETECTED (consecutive no-progress retries $retries/$MAX_RETRIES, latest ckpt step $ckpt_step, power ${CURRENT_POWER}W)"
    [[ -n "$hint" ]] && log "Log tail:$hint"
    archive_crash_log "$ckpt_step"

    # OOM before any checkpoint = the config does not fit in VRAM. Deterministic;
    # relaunching/stepping power never helps. Bail immediately with a clear cause
    # instead of burning MAX_RETRIES on the identical failure.
    if (( ckpt_step < 0 )) && is_oom_crash; then
        log_row "$(printf '| %s | ⛔ OOM | %s | | | | config exceeds VRAM at load/step 0 (not a surge fault) — fix config, do not retry; %s |' \
            "$(now_iso)" "$(last_step || true)" "$(crash_signature)")"
        log "OOM before any checkpoint — config does not fit in VRAM; retrying is futile — exiting"
        exit 1
    fi

    if (( retries >= MAX_RETRIES )); then
        log_row "$(printf '| %s | ⛔ STALLED | %s | | | | no progress in %d retries @ %sW — manual intervention; last good ckpt %s — %s |' \
            "$(now_iso)" "$(last_step || true)" "$MAX_RETRIES" "$CURRENT_POWER" "$ckpt_step" "$(crash_signature)")"
        log "Stalled — no forward progress in $MAX_RETRIES retries — exiting"
        exit 1
    fi

    log_row "$(printf '| %s | 🔴 crash | %s | | | | retry %d/%d @ %sW, last good ckpt %s — %s |' \
        "$(now_iso)" "$(last_step || true)" "$((retries+1))" "$MAX_RETRIES" "$CURRENT_POWER" "$ckpt_step" "$(crash_signature)")"

    retries=$(( retries + 1 ))
    kill_and_clean
    quarantine_bad_checkpoint
    # Crash-driven stability search: drop the power limit a notch before relaunch.
    step_power_down
    start_training
done
