#!/bin/bash
#SBATCH --job-name=kele-wave
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:2
#SBATCH --time=08:00:00
# Note: --gres=gpu:2 requests any two available GPUs.
# WAVE has V100 32GB nodes (gpu[01-04]) and L40S 48GB nodes (oignat01).
# Both work. If you want to target a specific node type:
#   V100:  add --constraint=v100   (if the cluster supports it)
#   L40S:  add --nodelist=oignat01
# To avoid AMD MI100 nodes (not CUDA-compatible), add --exclude=amd01

set -euo pipefail

# Ensure Poetry (installed to ~/.local/bin by wave_setup.sh) is on PATH.
export PATH="$HOME/.local/bin:$PATH"

# Load Python 3.12 so Poetry can find the right interpreter on the compute node.
module load Python/3.12.3-GCCcore-14.2.0

# Mirror the cache layout from wave_setup.sh so Poetry finds the right venv.
PROJECT_SPACE=/WAVE/projects/CSEN-346-Sp26
SCRATCH_SPACE=/WAVE/scratch/CSEN-346-Sp26
export POETRY_VIRTUALENVS_PATH="$PROJECT_SPACE/.virtualenvs"
export POETRY_CACHE_DIR="$SCRATCH_SPACE/.cache/poetry"
export PIP_CACHE_DIR="$SCRATCH_SPACE/.cache/pip"
export TMPDIR="$SCRATCH_SPACE/.tmp/$USER"
export HF_HOME="${HF_HOME:-$PROJECT_SPACE/hf_models}"

REPO_DIR="${REPO_DIR:-$HOME/csen-346}"
EXPERIMENT="${EXPERIMENT:-wave}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-$HF_HOME/SocratTeachLLM}"
CONSULTANT_LOCAL_MODEL_NAME="${CONSULTANT_LOCAL_MODEL_NAME:-Qwen3.5-9B}"
CONSULTANT_MODEL_PATH="${CONSULTANT_MODEL_PATH:-$HF_HOME/Qwen3.5-9B}"

cd "$REPO_DIR"
mkdir -p logs

echo "=== KELE WAVE job ==="
date
hostname
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "---"

export TEACHER_HOST=127.0.0.1
export TEACHER_PORT=8001
export TEACHER_LOG_FILE="logs/vllm_teacher-${SLURM_JOB_ID}.log"
export TEACHER_MODEL_NAME=SocratTeachLLM
export TEACHER_MODEL_PATH
# 0.85 leaves ~15% headroom for CUDA overhead — safe on V100 32GB and larger.
export TEACHER_GPU_MEMORY_UTILIZATION="${TEACHER_GPU_MEMORY_UTILIZATION:-0.85}"
export TEACHER_MAX_MODEL_LEN="${TEACHER_MAX_MODEL_LEN:-4096}"

export CONSULTANT_HOST=127.0.0.1
export CONSULTANT_PORT=8002
export CONSULTANT_LOG_FILE="logs/vllm_consultant-${SLURM_JOB_ID}.log"
export CONSULTANT_LOCAL_MODEL_NAME
export CONSULTANT_MODEL_PATH
export CONSULTANT_GPU_MEMORY_UTILIZATION="${CONSULTANT_GPU_MEMORY_UTILIZATION:-0.85}"
export CONSULTANT_MAX_MODEL_LEN="${CONSULTANT_MAX_MODEL_LEN:-8192}"

wait_for_port() {
    local port=$1
    echo -n "Waiting for port $port..."
    for i in $(seq 1 240); do
        if curl -fsS "http://127.0.0.1:${port}/v1/models" > /dev/null 2>&1; then
            echo " ready (~$((i*2))s)"
            return 0
        fi
        sleep 2
    done
    echo " timeout"
    return 1
}

cleanup() {
    set +e
    if [ -n "${TEACHER_PID:-}" ]; then kill "$TEACHER_PID" 2>/dev/null || true; fi
    if [ -n "${CONSULTANT_PID:-}" ]; then kill "$CONSULTANT_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT

echo "Starting teacher on GPU 0..."
CUDA_VISIBLE_DEVICES=0 ./scripts/serve_socratteachllm.sh &
TEACHER_PID=$!
wait_for_port "$TEACHER_PORT"

echo "Starting consultant on GPU 1..."
CUDA_VISIBLE_DEVICES=1 ./scripts/serve_consultant.sh &
CONSULTANT_PID=$!
wait_for_port "$CONSULTANT_PORT"

echo "---"
curl -fsS "http://127.0.0.1:${TEACHER_PORT}/v1/models"
echo
curl -fsS "http://127.0.0.1:${CONSULTANT_PORT}/v1/models"
echo
echo "---"

./scripts/run_eval.sh "$EXPERIMENT"

echo "=== Job complete: $(date) ==="
