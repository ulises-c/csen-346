#!/usr/bin/env bash
# Eval-side auto-resume monitor for the Gemma 4 12B SFT-uplift PoC on the
# NVIDIA RTX 4000 Ada box. Sibling of scripts/monitor_stage2.sh (training), but
# the failure model is different and that difference is the whole point:
#
#   The RTX 4000 Ada took a power surge and crashes UNDER LOAD. A fault kills the
#   llama.cpp SERVER, but the eval CLIENT is pure CPU + HTTP (the classifier runs
#   on CPU), so it survives — run_single_dialogue catches the connection error,
#   stamps {"error": …} into every remaining dialogue at machine speed, and exits
#   0 with metrics_summary.json computed over the surviving subset. Unmonitored, a
#   single fault silently ships a TRUNCATED eval reported as success.
#
# So, unlike the training monitor:
#   1. The crash signal is SERVER health (/v1/models), polled out-of-band — NOT
#      client process death (the client exits 0 on a server fault).
#   2. Relaunch must REPAIR before it resumes: kele.py counts any non-zero file as
#      done (src/project/kele.py:448), so error-stamped/truncated dialogues are
#      skipped forever unless deleted first. repair_dialogues() is the eval
#      analogue of the training monitor's quarantine_bad_checkpoint.
#   3. COMPLETE means valid (non-error) dialogue count == dataset size — NOT
#      "metrics_summary.json exists". Forward progress for MAX_RETRIES is counted
#      in valid dialogues.
#
# Progress/crash rows append to ONE pinned comment on the eval log issue #130
# (same convention as monitor_stage2.sh / issue #120). The comment body is the
# source of truth: fetch → append → PATCH, so restarts never lose history.
#
# This script builds GitHub-markdown rows via printf; the single-quoted format
# strings (%s args + literal backticks/pipes) make SC2016 a false positive.
# shellcheck disable=SC2016
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR" || exit 1

PHASE="${1:-}"
case "$PHASE" in
    base)
        EXPERIMENT="gemma4-12b-local"
        OUT_DIR="$REPO_DIR/results/gemma4-12b-base"
        SERVE_SCRIPT="scripts/serve_gemma4_12b.sh"
        EXPECTED_ALIAS="Gemma 4 12B"
        ;;
    sft)
        EXPERIMENT="gemma4-12b-sft-local"
        OUT_DIR="$REPO_DIR/results/gemma4-12b-sft"
        SERVE_SCRIPT="scripts/serve_gemma4_12b_sft.sh"
        EXPECTED_ALIAS="Gemma 4 12B SFT"
        ;;
    *)
        printf 'usage: [MTP=1] %s {base|sft}\n' "$0" >&2
        printf '  base       → serve %s, eval → results/gemma4-12b-base\n' "Gemma 4 12B" >&2
        printf '  sft        → serve %s, eval → results/gemma4-12b-sft\n' "Gemma 4 12B SFT" >&2
        printf '  MTP=1 …    → attach the speculative drafter; output dir gets a -mtp suffix\n' >&2
        exit 2
        ;;
esac

# MTP on/off is a per-run toggle (stage 1 runs base both ways, MTP off first). The
# served alias is unchanged (the drafter is a side-car), so only the output dir +
# W&B run name differ — keeping the two full runs cleanly separated for comparison.
MTP_ENV=()
PHASE_LABEL="$PHASE"
if [[ "${MTP:-0}" == "1" ]]; then
    OUT_DIR="${OUT_DIR}-mtp"
    MTP_ENV=(MTP=1)
    PHASE_LABEL="${PHASE}+mtp"
fi

# Dataset override: eval a non-default HF set (EN held-out / synthetic OOD) on the
# same served model. EVAL_HF_REPO may be space-separated for concatenation; EVAL_SPLIT
# is "test" (held-out 10%) or "all" (whole set — for the tiny never-trained synthetic
# sets). EVAL_OUT_SUFFIX keeps each set's results + W&B run cleanly separated.
EVAL_DATA_ARGS=()
if [[ -n "${EVAL_HF_REPO:-}" ]]; then
    read -r -a _eval_repos <<< "$EVAL_HF_REPO"
    EVAL_DATA_ARGS+=(--hf-repo "${_eval_repos[@]}")
    PHASE_LABEL="${PHASE_LABEL} ${EVAL_HF_REPO##*/}"
fi
if [[ -n "${EVAL_SPLIT:-}" ]]; then
    EVAL_DATA_ARGS+=(--split "$EVAL_SPLIT")
fi
if [[ -n "${EVAL_OUT_SUFFIX:-}" ]]; then
    OUT_DIR="${OUT_DIR}-${EVAL_OUT_SUFFIX}"
fi
# NO_CONSULTANT=1 drops the external Qwen classifier so the served LLM self-consults
# (dual-role) — the apples-to-apples "no external classifier" ablation (handoff T1.1).
# Suffix the out-dir + W&B run so these stay separate from the --bert-consultant runs.
# CONSULTANT_ARGS is built below, once BERT_CKPT is defined.
if [[ "${NO_CONSULTANT:-0}" == "1" ]]; then
    OUT_DIR="${OUT_DIR}-noconsult"
    PHASE_LABEL="${PHASE_LABEL} noconsult"
fi

# ── Constants ────────────────────────────────────────────────────────────────
ISSUE_NUMBER=130
# All events append a row to ONE pinned comment (the "Live eval log" table on
# issue #130). Bump this id for a brand-new crawl: create a fresh placeholder
# comment and paste its numeric id here (same convention as monitor_stage2.sh).
LOG_COMMENT_ID=4644703104
PROGRESS_EVERY=50               # post a progress row every this many valid dialogues
MAX_RETRIES=8                   # consecutive no-forward-progress relaunches before STALLED
POLL_SECONDS=30                 # tight: faster server-death reaction = fewer error-stamps
SERVER_READY_TIMEOUT=360        # cold GGUF load can take 30-90s; allow margin

# Adaptive power search (memory: training-host-hardware-fault — the RTX 4000 Ada
# took a power surge and crashes under load). Rather than a fixed conservative cap,
# PUSH the limit: start high and step DOWN POWER_STEP_W on each crash until the run
# stops faulting — the highest power that completes a crawl is the best stability/
# speed point. This is ~free: per-dialogue checkpointing means a fault loses only
# the in-flight dialogue, and the only recovery cost is the server cold-reload.
# POWER_START_W defaults to the card's reported max limit (fallback 130).
POWER_START_W="${POWER_START_W:-}"
POWER_STEP_W="${POWER_STEP_W:-10}"
POWER_FLOOR_W="${POWER_FLOOR_W:-70}"   # below ~65-70% perf knee it's not worth going lower

PORT="${PORT:-8080}"
LLAMA_URL="http://localhost:${PORT}"
BERT_CKPT="${BERT_CKPT:-$REPO_DIR/results/state-clf-qwen3.5-0.8b-lora-wandb/final}"
# Empty (self-consult) when NO_CONSULTANT=1, else the external classifier flag.
if [[ "${NO_CONSULTANT:-0}" == "1" ]]; then
    CONSULTANT_ARGS=()
else
    CONSULTANT_ARGS=(--bert-consultant "$BERT_CKPT")
fi
EXPECTED_TOTAL="${EXPECTED_TOTAL:-681}"  # n=681 test split; overridden by progress.log denominator once present

DIALOGUES_DIR="$OUT_DIR/dialogues"
PROGRESS_LOG="$OUT_DIR/progress.log"
EVAL_LOG="$OUT_DIR/eval.log"
SERVER_LOG="$OUT_DIR/server.log"
CRASH_DIR="$OUT_DIR/crashlogs"
SELF_LOG="$REPO_DIR/outputs/monitor_eval_gemma4_12b_${PHASE}.log"
mkdir -p "$OUT_DIR" "$DIALOGUES_DIR" "$CRASH_DIR" "$REPO_DIR/outputs"

SERVER_PID=""

# ── Logging ──────────────────────────────────────────────────────────────────
log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$SELF_LOG"
}
now_iso() { date '+%Y-%m-%d %H:%M:%S%:z'; }

# Append one row to the pinned comment. Single writer (this monitor), so the
# read-modify-write has no concurrent-update race. gh resolves {owner}/{repo}
# from the cwd's git remote.
log_row() {
    local body
    body="$(gh api "repos/{owner}/{repo}/issues/comments/$LOG_COMMENT_ID" --jq '.body' 2>>"$SELF_LOG")" || {
        log "WARNING: could not fetch log comment $LOG_COMMENT_ID — dropping row: $1"; return 0; }
    printf '%s\n%s\n' "$body" "$1" \
        | gh api -X PATCH "repos/{owner}/{repo}/issues/comments/$LOG_COMMENT_ID" -F body=@- >> "$SELF_LOG" 2>&1 \
        || log "WARNING: gh api PATCH (log comment) failed"
}

# ── Dialogue accounting (the load-bearing difference from the training monitor) ─
# A dialogue is DONE only if its JSON parses AND carries no "error" key. Error
# files and truncated-mid-write files are NOT progress — they're what a server
# fault leaves behind, and metrics computation skips them.
count_valid() {
    python3 - "$DIALOGUES_DIR" <<'PY' 2>/dev/null || printf '0'
import json, sys
from pathlib import Path
n = 0
for f in Path(sys.argv[1]).glob("*.json"):
    try:
        d = json.loads(f.read_text())
    except Exception:
        continue
    if isinstance(d, dict) and "error" not in d:
        n += 1
print(n)
PY
}

count_errors() {
    python3 - "$DIALOGUES_DIR" <<'PY' 2>/dev/null || printf '0'
import json, sys
from pathlib import Path
n = 0
for f in Path(sys.argv[1]).glob("*.json"):
    try:
        d = json.loads(f.read_text())
    except Exception:
        n += 1          # truncated/unparsable counts as a corrupt file too
        continue
    if isinstance(d, dict) and "error" in d:
        n += 1
print(n)
PY
}

# Delete every dialogue file that fails to parse (truncated mid json.dump) or
# carries an "error" key, so the next relaunch actually re-runs them. Prints the
# number repaired. This is the eval analogue of quarantine_bad_checkpoint.
repair_dialogues() {
    python3 - "$DIALOGUES_DIR" <<'PY' 2>/dev/null || printf '0'
import json, sys
from pathlib import Path
n = 0
for f in Path(sys.argv[1]).glob("*.json"):
    bad = False
    try:
        d = json.loads(f.read_text())
        bad = isinstance(d, dict) and "error" in d
    except Exception:
        bad = True
    if bad:
        try:
            f.unlink(); n += 1
        except Exception:
            pass
print(n)
PY
}

# Dataset size: prefer the live denominator kele writes to progress.log
# ("{done}/{total} …"); fall back to EXPECTED_TOTAL before the first line lands.
dataset_total() {
    local t=""
    [[ -f "$PROGRESS_LOG" ]] && t="$(grep -oE '^[0-9]+/[0-9]+' "$PROGRESS_LOG" 2>/dev/null | head -1 | cut -d/ -f2)"
    printf '%s' "${t:-$EXPECTED_TOTAL}"
}

progress_rate() {
    [[ -f "$PROGRESS_LOG" ]] || return 0
    grep -oE '[0-9.]+ dlg/hr' "$PROGRESS_LOG" 2>/dev/null | head -1 | awk '{print $1}' || true
}

progress_eta() {
    [[ -f "$PROGRESS_LOG" ]] || return 0
    grep -oE 'ETA [0-9]+m' "$PROGRESS_LOG" 2>/dev/null | head -1 || true
}

# ── Crash diagnostics (NVIDIA signatures, not the gfx1201 HSA ones from #120) ──
crash_signature() {
    local sig=""
    for lf in "$EVAL_LOG" "$SERVER_LOG"; do
        [[ -f "$lf" ]] || continue
        sig="$(grep -iE "cudaerror|cuda error|out of memory|fallen off the bus|xid|segmentation fault|sigsegv|aborted|connection refused|runtimeerror" \
            "$lf" | tail -1 | tr -d '`' | sed 's/|/／/g' | cut -c1-120)"
        [[ -n "$sig" ]] && break
    done
    printf '%s' "$sig"
}

# Archive eval+server logs and GPU state per crash BEFORE relaunch overwrites them.
archive_crash_log() {
    local done="$1" ts dest
    ts="$(date '+%Y%m%d-%H%M%S')"
    dest="$CRASH_DIR/crash-${PHASE}-done${done}-${ts}"
    if [[ -f "$EVAL_LOG" ]]; then cp "$EVAL_LOG" "${dest}.eval.log" 2>/dev/null || true; fi
    if [[ -f "$SERVER_LOG" ]]; then cp "$SERVER_LOG" "${dest}.server.log" 2>/dev/null || true; fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi > "${dest}.nvidia-smi.txt" 2>&1 || true
    fi
    dmesg 2>/dev/null | grep -iE "nvrm|xid|nvidia|gpu has fallen" | tail -40 > "${dest}.dmesg.log" 2>/dev/null || true
    log "Archived crash log → ${dest}.*"
}

# ── Server lifecycle (helpers mirror eval_gemma4_31b.sh) ───────────────────────
probe_server() { curl -s --max-time 3 "$LLAMA_URL/v1/models" 2>/dev/null; }

# True only when the server serves the expected alias AND is past "Loading model".
server_ready() {
    local resp
    resp="$(probe_server)" || return 1
    [[ -z "$resp" ]] && return 1
    echo "$resp" | grep -q '"Loading model"' && return 1
    echo "$resp" | grep -q "\"$EXPECTED_ALIAS\""
}

kill_eval() {
    pkill -9 -f "src\.project\.kele" 2>/dev/null || true
    local waited=0
    while pgrep -f "src\.project\.kele" >/dev/null 2>&1 && (( waited < 20 )); do
        sleep 2; (( waited += 2 ))
    done
}

kill_server() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        local waited=0
        while kill -0 "$SERVER_PID" 2>/dev/null && (( waited < 20 )); do
            sleep 2; (( waited += 2 ))
        done
        kill -9 "$SERVER_PID" 2>/dev/null || true
    fi
    pkill -9 -f "llama-server" 2>/dev/null || true
    SERVER_PID=""
}

cleanup() {
    log "Monitor exiting — tearing down eval client + server"
    kill_eval
    kill_server
}
trap cleanup EXIT INT TERM

# CURRENT_POWER walks down from POWER_START_W (the card max) by POWER_STEP_W on
# each crash, floored at POWER_FLOOR_W. After every apply_power it is re-synced to
# the GPU's ACTUAL enforced limit, so logged wattage reflects reality even when the
# step-down is inert (no passwordless sudo → the card stays at its persisted limit).
CURRENT_POWER=""

gpu_max_power() {
    command -v nvidia-smi >/dev/null 2>&1 || { printf '130'; return; }
    local m
    m="$(nvidia-smi --query-gpu=power.max_limit --format=csv,noheader,nounits 2>/dev/null | head -1 | cut -d. -f1)"
    [[ "$m" =~ ^[0-9]+$ ]] && printf '%s' "$m" || printf '130'
}

# The GPU's currently ENFORCED power limit (not the card max). This is what the
# card actually runs at, which differs from CURRENT_POWER's target when the
# sudo -pl below is denied.
gpu_power_limit() {
    command -v nvidia-smi >/dev/null 2>&1 || return 1
    local m
    m="$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits 2>/dev/null | head -1 | cut -d. -f1)"
    [[ "$m" =~ ^[0-9]+$ ]] && printf '%s' "$m" || return 1
}

# Apply CURRENT_POWER as the GPU power limit. Best-effort + non-interactive: the
# user may have to pre-authorize sudo; -n avoids a password prompt stalling the loop.
# Either way, re-sync CURRENT_POWER to the actual enforced limit so the issue-#130
# log rows never claim a wattage the card isn't running at.
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
    local actual
    actual="$(gpu_power_limit)" && CURRENT_POWER="$actual"
}

# Step the power limit down one notch (called on each crash), not below the floor.
step_power_down() {
    [[ -z "$CURRENT_POWER" ]] && return 0
    if (( CURRENT_POWER - POWER_STEP_W >= POWER_FLOOR_W )); then
        CURRENT_POWER=$(( CURRENT_POWER - POWER_STEP_W ))
        log "Stepping power down → ${CURRENT_POWER}W (crash-driven stability search)"
    else
        log "Power already at floor ${CURRENT_POWER}W (>= floor ${POWER_FLOOR_W}W) — not stepping lower"
    fi
}

boot_server() {
    log "Booting llama-server ($EXPECTED_ALIAS${MTP_ENV:+ +MTP}) via $SERVE_SCRIPT"
    env "${MTP_ENV[@]}" bash "$REPO_DIR/$SERVE_SCRIPT" > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    local waited=0
    while (( waited < SERVER_READY_TIMEOUT )); do
        if server_ready; then
            log "Server ready (~${waited}s)"
            return 0
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            log "Server process exited during load — see $SERVER_LOG"
            return 1
        fi
        sleep 5; (( waited += 5 ))
    done
    log "Server failed to become ready in ${SERVER_READY_TIMEOUT}s"
    return 1
}

start_eval() {
    log "Launching eval (experiment=$EXPERIMENT → $OUT_DIR)"
    # WANDB_EVAL_RUN_NAME = output-dir basename so MTP off (gemma4-12b-base) and
    # MTP on (gemma4-12b-base-mtp) are distinct W&B runs despite one --experiment.
    WANDB_EVAL=1 WANDB_EVAL_RUN_NAME="$(basename "$OUT_DIR")" KELE_BERT_DEVICE=cpu nohup \
        uv run --no-sync python -m src.project.kele \
        --experiment "$EXPERIMENT" evaluate \
        "${CONSULTANT_ARGS[@]}" \
        --output "$OUT_DIR" \
        "${EVAL_DATA_ARGS[@]}" \
        > "$EVAL_LOG" 2>&1 &
    sleep 5
}

eval_running() { pgrep -f "src\.project\.kele" >/dev/null 2>&1; }

# ── Main crawl loop ────────────────────────────────────────────────────────────
log "Eval monitor starting: phase=$PHASE, issue #$ISSUE_NUMBER comment $LOG_COMMENT_ID, poll ${POLL_SECONDS}s, max $MAX_RETRIES retries"

# Ensure the consultant classifier is present (the make eval targets' _classifier-ckpt dep).
# Skipped entirely in self-consult mode — there is no external classifier to fetch.
if [[ "${NO_CONSULTANT:-0}" != "1" && ! -f "$BERT_CKPT/model.safetensors" ]]; then
    log "Classifier checkpoint missing — downloading"
    hf download ulises-c/socrates-state-classifier-qwen3.5-lora --local-dir "$BERT_CKPT" >>"$SELF_LOG" 2>&1 || \
        log "WARNING: classifier download failed — eval will error until $BERT_CKPT exists"
fi

[[ -z "$POWER_START_W" ]] && POWER_START_W="$(gpu_max_power)"
CURRENT_POWER="$POWER_START_W"
log "Power search: start ${POWER_START_W}W, -${POWER_STEP_W}W/crash, floor ${POWER_FLOOR_W}W"
apply_power

TOTAL="$(dataset_total)"
log_row "$(printf '| %s | ▶ monitor start | %s | %s/%s | %s | | serving %s @ %sW; progress every %d dlg |' \
    "$(now_iso)" "$PHASE_LABEL" "$(count_valid)" "$TOTAL" "$(count_errors)" "$EXPECTED_ALIAS" "$CURRENT_POWER" "$PROGRESS_EVERY")"

retries=0
last_progress_valid="$(count_valid)"
last_reported_bucket=$(( last_progress_valid / PROGRESS_EVERY ))

if ! boot_server; then
    archive_crash_log "$(count_valid)"
    log_row "$(printf '| %s | 🔴 crash | %s | %s/%s | %s | | server failed to boot @ %sW — %s |' \
        "$(now_iso)" "$PHASE_LABEL" "$(count_valid)" "$TOTAL" "$(count_errors)" "$CURRENT_POWER" "$(crash_signature)")"
fi
start_eval

while true; do
    valid="$(count_valid)"
    errors="$(count_errors)"
    TOTAL="$(dataset_total)"

    # ── Completion: valid count reached the dataset size (NOT metrics file present)
    if (( valid >= TOTAL )); then
        # Let the in-flight kele finish writing metrics_summary.json.
        local_wait=0
        while [[ ! -f "$OUT_DIR/metrics_summary.json" ]] && (( local_wait < 60 )); do
            sleep 5; (( local_wait += 5 ))
        done
        log "Eval COMPLETE — $valid/$TOTAL valid dialogues at ${CURRENT_POWER}W"
        log_row "$(printf '| %s | ✅ COMPLETE | %s | %s/%s | %s | %s | %s ; stable @ %sW ; metrics_summary.json written |' \
            "$(now_iso)" "$PHASE_LABEL" "$valid" "$TOTAL" "$errors" "$(progress_rate)" "${OUT_DIR#"$REPO_DIR"/}" "$CURRENT_POWER")"
        exit 0
    fi

    # ── Healthy and advancing: post periodic progress, keep watching ───────────
    if eval_running && server_ready; then
        bucket=$(( valid / PROGRESS_EVERY ))
        if (( valid > 0 )) && (( bucket > last_reported_bucket )); then
            log_row "$(printf '| %s | 📈 progress | %s | %s/%s | %s | %s | %s @ %sW |' \
                "$(now_iso)" "$PHASE_LABEL" "$valid" "$TOTAL" "$errors" "$(progress_rate)" "$(progress_eta)" "$CURRENT_POWER")"
            last_reported_bucket="$bucket"
        fi
        if (( valid > last_progress_valid )); then
            last_progress_valid="$valid"
            retries=0
        fi
        log "running — $valid/$TOTAL valid, $errors error/corrupt"
        sleep "$POLL_SECONDS"
        continue
    fi

    # ── Not (healthy + running). Debounce one transient unhealthy poll: a single
    #    slow /v1/models under power-cap shouldn't trigger a kill + cold-reload +
    #    (false) power step-down. Require the server to STILL be down on recheck.
    if eval_running && ! server_ready; then
        sleep 5
        if server_ready; then
            log "Transient server blip — recovered on recheck, continuing"
            continue
        fi
    fi

    # Classify the failure — this drives whether we touch power:
    #   server DOWN  → GPU fault (the real hazard): kill client, step power down, reboot.
    #   server UP    → client exited with residual content-errors (not a power fault):
    #                  repair + relaunch against the same server, leave power alone.
    if server_ready; then
        server_fault=false
        log "Eval client exited, server healthy ($valid/$TOTAL) — residual errors; relaunch, no power change"
    else
        server_fault=true
        log "SERVER UNHEALTHY (confirmed) — GPU fault; killing client before it error-stamps the rest"
    fi

    # Forward-progress check (counted in valid dialogues): a relaunch that landed
    # new valid dialogues resets the no-progress counter, so a crawl that keeps
    # inching forward never exhausts MAX_RETRIES.
    if (( valid > last_progress_valid )); then
        log "Progress since last crash: $last_progress_valid → $valid valid — resetting retry counter"
        last_progress_valid="$valid"
        retries=0
    fi

    archive_crash_log "$valid"
    log "CRASH (server_fault=$server_fault, no-progress retries $retries/$MAX_RETRIES, $valid/$TOTAL valid, $errors corrupt)"

    if (( retries >= MAX_RETRIES )); then
        log_row "$(printf '| %s | ⛔ STALLED | %s | %s/%s | %s | | no progress in %d retries @ %sW — manual intervention; %s |' \
            "$(now_iso)" "$PHASE_LABEL" "$valid" "$TOTAL" "$errors" "$MAX_RETRIES" "$CURRENT_POWER" "$(crash_signature)")"
        log "Stalled — exiting"
        exit 1
    fi

    kill_eval
    repaired="$(repair_dialogues)"
    log "Repaired (deleted) $repaired error/corrupt dialogue files"
    retries=$(( retries + 1 ))

    if $server_fault; then
        log_row "$(printf '| %s | 🔴 crash | %s | %s/%s | %s | | GPU fault; retry %d/%d @ %sW, repaired %s — %s |' \
            "$(now_iso)" "$PHASE_LABEL" "$valid" "$TOTAL" "$(count_errors)" "$((retries))" "$MAX_RETRIES" "$CURRENT_POWER" "$repaired" "$(crash_signature)")"
        kill_server
        # Crash-driven stability search: drop the power limit a notch before rebooting.
        step_power_down
        apply_power
        if boot_server; then
            log_row "$(printf '| %s | 🔁 resume | %s | %s/%s | %s | | server rebooted @ %sW; relaunching eval |' \
                "$(now_iso)" "$PHASE_LABEL" "$(count_valid)" "$TOTAL" "$(count_errors)" "$CURRENT_POWER")"
            start_eval
        else
            log "Server reboot failed — will retry next loop"
            sleep "$POLL_SECONDS"
        fi
    else
        # Server still healthy — relaunch against it, no reboot, no power change.
        log_row "$(printf '| %s | 🔁 resume | %s | %s/%s | %s | | content errors; retry %d/%d @ %sW, repaired %s; relaunch (server healthy) |' \
            "$(now_iso)" "$PHASE_LABEL" "$(count_valid)" "$TOTAL" "$(count_errors)" "$((retries))" "$MAX_RETRIES" "$CURRENT_POWER" "$repaired")"
        start_eval
    fi
done
