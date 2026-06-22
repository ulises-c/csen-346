#!/usr/bin/env bash
# This script builds GitHub-markdown issue comments via printf; the format strings
# are single-quoted on purpose (%s positional args + literal backticks/newlines),
# so SC2016 ("expressions don't expand in single quotes") is a false positive.
# shellcheck disable=SC2016
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="$REPO_DIR/outputs/sft-stage2-gemma4-31b"
STAGE2_LOG="$OUTPUT_DIR/train.log"
FINAL_DIR="$OUTPUT_DIR/final"
SELF_LOG="$REPO_DIR/outputs/monitor_stage2.log"
# Progress + crash reports go to the dedicated Training Log issue, not the PR —
# crash spam was flooding PR #101 (code review). See issue #120.
ISSUE_NUMBER=120
# All events append a row to ONE pinned comment (the "Live training log" table on
# issue #120) instead of posting per-event. Bump this id when starting a brand-new
# crawl — create a fresh placeholder comment and paste its numeric id here (same
# convention as WANDB_RUN_ID). The comment body is the source of truth: log_row
# fetches it, appends a row, and PATCHes it back, so it survives monitor restarts
# and OUTPUT_DIR wipes without minting a second log comment.
LOG_COMMENT_ID=4635099006
# Post a training-metrics progress report every this many steps (loss/grad_norm/lr).
PROGRESS_EVERY=50
# The gfx1201 fault is non-deterministic, so the run advances by crashing and
# resuming from the latest checkpoint (save_steps=10). MAX_RETRIES bounds
# CONSECUTIVE retries that make NO forward progress — a run that keeps advancing
# never exhausts it; only a genuinely stuck GPU (no new checkpoint across this
# many tries) stops the monitor. Crawl-forward needs a high bound, not 2.
MAX_RETRIES=8
POLL_SECONDS=300

# gh resolves the {owner}/{repo} api placeholders from the cwd's git remote.
cd "$REPO_DIR" || exit 1

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >> "$SELF_LOG"
}

# ── Live training-log comment ────────────────────────────────────────────────
# Readable ISO 8601 with numeric offset (not a TZ abbreviation, which isn't ISO
# and is ambiguous across hosts).
now_iso() { date '+%Y-%m-%d %H:%M:%S%:z'; }

# One-line crash signature for the table's Note cell. A literal pipe would break
# the markdown row, so swap it for a fullwidth bar; the full trace is in crashlogs/.
crash_signature() {
    [[ -f "$STAGE2_LOG" ]] || return 0
    grep -iE "memory access fault|page not present|out of memory|hip error|cuda error|runtimeerror|aborted" \
        "$STAGE2_LOG" | tail -1 | awk -F'\r' '{print $NF}' | tr -d '`' \
        | sed 's/^[^%]*%|[^]]*\][[:space:]]*//' \
        | sed 's/|/／/g' | cut -c1-140
}

# Append one row to the pinned comment. The comment body is the source of truth:
# fetch it, append the row, PATCH it back — so restarts/OUTPUT_DIR wipes never
# lose history or mint a second comment. Single writer (this monitor), so the
# read-modify-write has no concurrent-update race.
log_row() {
    local body
    body="$(gh api "repos/{owner}/{repo}/issues/comments/$LOG_COMMENT_ID" --jq '.body' 2>>"$SELF_LOG")" || {
        log "WARNING: could not fetch log comment $LOG_COMMENT_ID — dropping row: $1"; return 0; }
    printf '%s\n%s\n' "$body" "$1" \
        | gh api -X PATCH "repos/{owner}/{repo}/issues/comments/$LOG_COMMENT_ID" -F body=@- >> "$SELF_LOG" 2>&1 \
        || log "WARNING: gh api PATCH (log comment) failed"
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
    local note="epoch ${epoch:-?}"
    [[ -n "$acc" ]] && note="$note, acc $acc"
    log_row "$(printf '| %s | 📈 checkpoint | %s | %s | %s | %s | %s |' \
        "$(now_iso)" "$step" "${loss:-}" "${gn:-}" "${lr:-}" "$note")"
}

training_running() {
    pgrep -f "train_sft\.py" > /dev/null 2>&1
}

last_step() {
    if [[ -f "$STAGE2_LOG" ]]; then
        # Skip N/N patterns (100%-complete bars from eval/map/shard ops like 8578/8578);
        # only training-step tqdm has N < total (e.g. 1380/4826).
        grep -oE "[0-9]+/[0-9]+" "$STAGE2_LOG" \
            | awk -F'/' '$1+0 != $2+0' \
            | tail -1 || true
    fi
}

crash_hint() {
    [[ -f "$STAGE2_LOG" ]] || return 0
    # The gfx1201 fault prints "Memory access fault by GPU node-1 ... Page not present"
    # and aborts with "Aborted (core dumped)" — it is NOT the literal string "page fault"
    # and emits no Python traceback, so the old pattern matched nothing and the PR block
    # came up empty. Match the real ROCm/HSA signature, and lead with the last tqdm step
    # so the crash post shows the fault step (the N we are measuring) directly.
    local step
    step="$(grep -oE "[0-9]+/[0-9]+" "$STAGE2_LOG" \
        | awk -F'/' '$1+0 != $2+0' | tail -1 || true)"
    [[ -n "$step" ]] && printf '  fault near step %s\n' "$step"
    grep -iE "memory access fault|page not present|hsa_status|aborted|core dumped|out of memory|hip error|cuda error|traceback|runtimeerror" \
        "$STAGE2_LOG" | tail -4 | sed 's/^/  /' || true
}

# The make target writes train.log with `>` (truncate), so each relaunch would
# overwrite the crashed run's full traceback. Archive the whole log + a dmesg
# snapshot per crash BEFORE relaunch so an overnight crawl leaves a reviewable
# trail of every fault, not just the 3-line hint.
archive_crash_log() {
    local ckpt="$1" ts dest crashdir
    [[ -f "$STAGE2_LOG" ]] || return 0
    ts="$(date '+%Y%m%d-%H%M%S')"
    crashdir="$OUTPUT_DIR/crashlogs"
    mkdir -p "$crashdir"
    dest="$crashdir/crash-step${ckpt}-${ts}.log"
    if cp "$STAGE2_LOG" "$dest" 2>/dev/null; then
        log "Archived full crash log → $dest"
    fi
    dmesg 2>/dev/null \
        | grep -iE "amdgpu|gfxhub|VM_L2|page fault|PERMISSION_FAULTS|WALKER_ERROR" \
        | tail -40 > "$crashdir/dmesg-step${ckpt}-${ts}.log" 2>/dev/null || true
}

kill_and_clean() {
    log "Killing stray train_sft.py processes"
    pkill -9 -f "train_sft\.py" 2>/dev/null || true
    local waited=0
    while pgrep -f "train_sft\.py" > /dev/null 2>&1 && [[ $waited -lt 30 ]]; do
        sleep 2; (( waited += 2 ))
    done
    # A gfx1201 fault leaves the amdkfd dirty (orphaned HIP context + stale VRAM);
    # relaunching into that state faults early on stale PTEs (the cascade that
    # turned a single crash into a permanent loop). Verify the GPU is ACTUALLY
    # clean before relaunch instead of a blind sleep (GFX1201_RDNA4_TRAINING.md §10).
    if ! bash "$REPO_DIR/scripts/test_gpu_stack.sh" --wait-clean 180 >> "$SELF_LOG" 2>&1; then
        log "WARNING: GPU still dirty after 180s — relaunch will likely fault on stale PTEs"
    fi
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

# A crash landing during a checkpoint write leaves an incomplete checkpoint-N.
# HF resume always picks the highest-numbered dir, so a partial one makes every
# resume fail to load and loop on the same bad checkpoint. trainer_state.json is
# written last in _save_checkpoint, so a valid one implies the rest is complete;
# if it's missing or unparsable, quarantine the dir so resume falls back to N-1.
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

start_stage2() {
    # Use a fresh data_seed each launch so resume presents different samples at each
    # step — breaking the data-order-sticky fault confirmed in run #13 (PR #101).
    # Verified (run #14 verif): TRAIN_DATA_SEED genuinely reshuffles the post-resume
    # order and step 22 (previously the sticky fault step) completed clean under seed 99.
    # Unix timestamp gives a distinct seed every resume (each cycle is ≥10 min apart).
    local data_seed
    data_seed="$(date +%s)"
    log "Starting Stage 2 training (data_seed=$data_seed, gpu-preflight gates the launch)"
    cd "$REPO_DIR" || return 0
    TRAIN_DATA_SEED="$data_seed" WANDB_PROJECT=csen346-sft make train-gemma4-31b-stage2-unsloth \
        || log "launch aborted (gpu-preflight failed?) — counts as no forward progress"
    sleep 30
}

log "Stage 2 monitor starting (issue #$ISSUE_NUMBER comment $LOG_COMMENT_ID, poll every ${POLL_SECONDS}s, max $MAX_RETRIES retries)"
log_row "$(printf '| %s | ▶ monitor start | %s | | | | watching train.log; progress every %s steps |' \
    "$(now_iso)" "$(last_step || true)" "$PROGRESS_EVERY")"

sleep 30  # let it get going before first poll

retries=0
last_progress_step="$(latest_ckpt_step)"
last_reported_ckpt="$(latest_ckpt_step)"

while true; do
    if training_running; then
        ckpt_now="$(latest_ckpt_step)"
        log "Stage 2 running — ckpt step $ckpt_now"
        if (( ckpt_now > 0 )) && (( ckpt_now % PROGRESS_EVERY == 0 )) \
            && (( ckpt_now > last_reported_ckpt )); then
            log_progress_from_checkpoint "$ckpt_now"
            last_reported_ckpt="$ckpt_now"
        fi
        sleep "$POLL_SECONDS"
        continue
    fi

    if [[ -d "$FINAL_DIR" ]]; then
        log "Stage 2 complete — adapter at $FINAL_DIR"
        total_steps="$(last_step || true)"
        log_row "$(printf '| %s | ✅ COMPLETE | %s | | | | adapter: outputs/sft-stage2-gemma4-31b/final |' \
            "$(now_iso)" "${total_steps:-?}")"
        log "Stage 2 done — monitor exiting"
        exit 0
    fi

    # Forward-progress check: did a new checkpoint land since the last crash? If
    # so the crawl is advancing — reset the no-progress retry counter. Only
    # consecutive stalls (no new checkpoint) count toward MAX_RETRIES, so a run
    # that keeps inching forward across faults never gives up.
    ckpt_step="$(latest_ckpt_step)"
    if (( ckpt_step > last_progress_step )); then
        log "Progress since last crash: checkpoint $last_progress_step → $ckpt_step — resetting retry counter"
        last_progress_step="$ckpt_step"
        retries=0
    fi

    hint="$(crash_hint)"
    log "CRASH DETECTED (consecutive no-progress retries $retries/$MAX_RETRIES, latest ckpt step $ckpt_step)"
    [[ -n "$hint" ]] && log "Log tail:$hint"
    # Preserve the full traceback + dmesg BEFORE relaunch overwrites train.log.
    archive_crash_log "$ckpt_step"

    if (( retries >= MAX_RETRIES )); then
        log_row "$(printf '| %s | ⛔ STALLED | %s | | | | no progress in %d retries — manual intervention (make diagnose-gfx1201-fault); last good ckpt %s — %s |' \
            "$(now_iso)" "$(last_step || true)" "$MAX_RETRIES" "$ckpt_step" "$(crash_signature)")"
        log "Stalled — no forward progress in $MAX_RETRIES retries — exiting"
        exit 1
    fi

    log_row "$(printf '| %s | 🔴 crash | %s | | | | retry %d/%d, last good ckpt %s — %s |' \
        "$(now_iso)" "$(last_step || true)" "$((retries+1))" "$MAX_RETRIES" "$ckpt_step" "$(crash_signature)")"

    retries=$(( retries + 1 ))
    kill_and_clean
    quarantine_bad_checkpoint
    start_stage2
done
