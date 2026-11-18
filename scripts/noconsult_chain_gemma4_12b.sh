#!/usr/bin/env bash
# Weekend chain for the consultant ablation (handoff T1.1): run both no-consultant
# (self-consult) arms back-to-back, one model at a time on the 20 GB card. SFT first
# (fast, terminates cleanly), then base (~30 h — self-consult is 2 LLM calls/turn).
#
# Each arm is the crash-resilient monitor, which OWNS its own llama-server: it boots
# the server, walks the GPU power limit down per fault, auto-resumes, and its EXIT
# trap kills the server on completion — so the GPU is free before the next arm boots.
# Outputs: results/gemma4-12b-{sft,base}-noconsult. Monitor self-logs per phase to
# outputs/monitor_eval_gemma4_12b_{sft,base}.log; this wrapper logs the chain steps.
set -uo pipefail  # NOT -e: a stalled/failed first arm must not abort the second.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR" || exit 1
mkdir -p "$REPO_DIR/outputs"
LOG="$REPO_DIR/outputs/noconsult_chain.log"

clog() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S%:z')" "$*" | tee -a "$LOG"; }

run_arm() {
    local phase="$1"
    clog "=== START ${phase} no-consultant arm → results/gemma4-12b-${phase}-noconsult ==="
    NO_CONSULTANT=1 bash "$REPO_DIR/scripts/monitor_eval_gemma4_12b.sh" "$phase"
    local rc=$?
    clog "=== END ${phase} arm (rc=${rc}) ==="
    return "$rc"
}

clog "##### no-consultant chain begin (sft → base) #####"

run_arm sft || clog "WARNING: sft arm exited non-zero — proceeding to base anyway."

# Settle so the SFT server is fully reaped before base boots on the shared 20 GB card.
sleep 20

run_arm base || clog "WARNING: base arm exited non-zero."

clog "##### no-consultant chain done #####"
