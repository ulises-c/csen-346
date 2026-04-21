#!/usr/bin/env bash
# Submit the KELE WAVE evaluation job.
# Pulls latest code, creates the log dir, submits, and prints handy commands.
#
# Usage:
#   bash scripts/slurm/submit_wave.sh
#   make slurm

set -euo pipefail
cd "$(dirname "$0")/../.."

# ── Pull latest ───────────────────────────────────────────────────────────────
echo "Pulling latest..."
git pull

# ── Submit ────────────────────────────────────────────────────────────────────
mkdir -p logs
JOB=$(sbatch scripts/slurm/wave_eval.slurm | awk '{print $NF}')
SUBMITTED_AT="$(date '+%Y-%m-%d %H:%M:%S')"

# ── Status summary ────────────────────────────────────────────────────────────
printf "\n[%s] Job %s submitted\n" "$SUBMITTED_AT" "$JOB"
printf "  Status  : squeue -j %s\n"            "$JOB"
printf "  All jobs: squeue -u \$USER\n"
printf "  Logs    : tail -f logs/slurm-%s.out\n" "$JOB"
printf "  Run dir : tail -f logs/*-%s/run.log  (appears after job starts)\n" "$JOB"
printf "  Cancel  : scancel %s\n\n"            "$JOB"

# ── Persist to file ───────────────────────────────────────────────────────────
LOG="logs/job-${JOB}.submitted"
{
    printf "[%s] Job %s submitted\n" "$SUBMITTED_AT" "$JOB"
    printf "  Status  : squeue -j %s\n"            "$JOB"
    printf "  All jobs: squeue -u \$USER\n"
    printf "  Logs    : tail -f logs/slurm-%s.out\n" "$JOB"
    printf "  Run dir : tail -f logs/*-%s/run.log\n" "$JOB"
    printf "  Cancel  : scancel %s\n"              "$JOB"
} | tee "$LOG" > /dev/null

echo "Saved to $LOG"
