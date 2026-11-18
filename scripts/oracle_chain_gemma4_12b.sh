#!/usr/bin/env bash
# Chain for the oracle-consultant ablation (handoff T1.1 follow-on): run both
# oracle arms back-to-back, one model at a time on the 20 GB card. SFT first
# (~3 h), then base (~17 h). Oracle = the GT state is fed each turn (1 teacher
# call/turn, as fast as the classifier path — NOT the 2-call self-consult cost).
#
# state_accuracy is ~perfect by construction here, so compare the arms on
# ROUGE/BLEU — this isolates PURE teacher-turn quality given correct state, with
# classifier accuracy removed as a confound (the confound-free ceiling on "does
# the SFT write better Socratic turns?").
#
# Each arm is the crash-resilient monitor, which OWNS its own llama-server: it
# boots the server, walks the GPU power limit down per fault, auto-resumes, and
# its EXIT trap kills the server on completion — so the GPU is free before the
# next arm boots. Outputs: results/gemma4-12b-{sft,base}-oracle. Monitor
# self-logs per phase to outputs/monitor_eval_gemma4_12b_{sft,base}.log; this
# wrapper logs the chain steps.
set -uo pipefail  # NOT -e: a stalled/failed first arm must not abort the second.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR" || exit 1
mkdir -p "$REPO_DIR/outputs"
LOG="$REPO_DIR/outputs/oracle_chain.log"

clog() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S%:z')" "$*" | tee -a "$LOG"; }

run_arm() {
    local phase="$1"
    clog "=== START ${phase} oracle arm → results/gemma4-12b-${phase}-oracle ==="
    ORACLE_CONSULTANT=1 bash "$REPO_DIR/scripts/monitor_eval_gemma4_12b.sh" "$phase"
    local rc=$?
    clog "=== END ${phase} arm (rc=${rc}) ==="
    return "$rc"
}

clog "##### oracle chain begin (sft → base) #####"

run_arm sft || clog "WARNING: sft arm exited non-zero — proceeding to base anyway."

# Settle so the SFT server is fully reaped before base boots on the shared 20 GB card.
sleep 20

run_arm base || clog "WARNING: base arm exited non-zero."

clog "##### oracle chain done #####"
