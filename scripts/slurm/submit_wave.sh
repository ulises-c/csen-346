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
# Pre-create logs/<jobid>/ so SLURM can write its output files there
# (#SBATCH --output=logs/%j/slurm.out).  The job will rename the dir to
# logs/<timestamp>-<jobid>/ once it starts; Linux keeps open fds alive
# across renames so the SLURM output file keeps working without interruption.
mkdir -p logs
JOB=$(sbatch scripts/slurm/wave_eval.slurm | awk '{print $NF}')
mkdir -p "logs/$JOB"
SUBMITTED_AT="$(date '+%Y-%m-%d %H:%M:%S')"

# ── Persist submission info into the run dir ──────────────────────────────────
LOG="logs/$JOB/job.submitted"
{
    printf "[%s] Job %s submitted\n" "$SUBMITTED_AT" "$JOB"
    printf "  Status  : squeue -j %s\n"                    "$JOB"
    printf "  All jobs: squeue -u \$USER\n"
    printf "  Logs    : tail -f logs/*%s*/slurm.out\n"     "$JOB"
    printf "  Run log : tail -f logs/*%s*/run.log\n"       "$JOB"
    printf "  Cancel  : scancel %s\n"                      "$JOB"
} > "$LOG"

# ── Print to terminal ─────────────────────────────────────────────────────────
printf "\n[%s] Job %s submitted\n" "$SUBMITTED_AT" "$JOB"
printf "  Status  : squeue -j %s\n"                "$JOB"
printf "  All jobs: squeue -u \$USER\n"
printf "  Logs    : tail -f logs/*%s*/slurm.out\n" "$JOB"
printf "  Run log : tail -f logs/*%s*/run.log\n"   "$JOB"
printf "  Cancel  : scancel %s\n\n"                "$JOB"
printf "Saved to %s\n" "$LOG"
