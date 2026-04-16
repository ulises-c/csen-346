#!/usr/bin/env bash
# One-time WAVE HPC setup for csen-346.
# Run this from the repo root on the LOGIN node (internet access, no GPU needed).
#
# Usage:
#   bash scripts/wave_setup.sh              # Install deps only
#   bash scripts/wave_setup.sh --models     # Install deps + download models

set -euo pipefail
cd "$(dirname "$0")/.."

DOWNLOAD_MODELS=false
for arg in "$@"; do
    [[ "$arg" == "--models" ]] && DOWNLOAD_MODELS=true
done

# ── Logging — tee stdout+stderr to logs/wave_setup.log (repo-relative) ────────
mkdir -p logs
LOG_FILE="logs/wave_setup.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Log: $(pwd)/$LOG_FILE"

# ── Redirect all caches off the home directory ────────────────────────────────
# Home dir quota on WAVE is tiny. Poetry venv + pip cache + cuda-bindings alone
# can exceed it. Point everything at project (persistent) or scratch (ephemeral).
# If the class directories aren't provisioned yet, fall back to ~/csen346-cache
# and warn loudly.
PROJECT_SPACE=/WAVE/projects/CSEN-346-Sp26
SCRATCH_SPACE=/WAVE/scratch/CSEN-346-Sp26
FALLBACK="$HOME/csen346-cache"

# Test actual write access by touching a temp file inside the directory.
_can_write() {
    local dir="$1"
    local probe="$dir/.wave_setup_write_test_$$"
    [[ -d "$dir" ]] && touch "$probe" 2>/dev/null && rm -f "$probe"
}

if _can_write "$PROJECT_SPACE" && _can_write "$SCRATCH_SPACE"; then
    warn_fallback=false
else
    warn_fallback=true
    warn "Cannot write to $PROJECT_SPACE or $SCRATCH_SPACE"
    warn "The class directories may not be provisioned yet."
    warn "Contact your sysadmin to request access, then re-run."
    warn "Falling back to $FALLBACK — watch your home quota!"
    PROJECT_SPACE="$FALLBACK"
    SCRATCH_SPACE="$FALLBACK"
fi

# Virtualenv: project space — survives between sessions
export POETRY_VIRTUALENVS_PATH="$PROJECT_SPACE/.virtualenvs"
# Poetry package cache: scratch — throwaway
export POETRY_CACHE_DIR="$SCRATCH_SPACE/.cache/poetry"
# pip download/wheel cache: scratch — throwaway
export PIP_CACHE_DIR="$SCRATCH_SPACE/.cache/pip"
# pip needs a writable TMPDIR for large wheel unpacking
export TMPDIR="$SCRATCH_SPACE/.tmp/$USER"

mkdir -p "$POETRY_VIRTUALENVS_PATH" "$POETRY_CACHE_DIR" "$PIP_CACHE_DIR" "$TMPDIR"
echo "Virtualenvs : $POETRY_VIRTUALENVS_PATH"
echo "Poetry cache: $POETRY_CACHE_DIR"
echo "pip cache   : $PIP_CACHE_DIR"
echo "TMPDIR      : $TMPDIR"

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; BOLD='\033[1m'; NC='\033[0m'
info() { echo -e "${GREEN}[setup]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn]${NC}  $*"; }
die()  { echo -e "${RED}[error]${NC} $*" >&2; exit 1; }
step() { echo -e "\n${BOLD}── $* ──${NC}"; }

echo ""
echo -e "${BOLD}KELE / WAVE HPC Setup${NC}"
echo "=============================="
date
hostname
echo ""

# ── 1. Python ≥ 3.12 ──────────────────────────────────────────────────────────
step "Checking Python"

# Load Python 3.12 module if not already available.
# 'module' is a shell function sourced from the environment init; it may not
# exist in non-interactive scripts, so guard the call.
_has_python312() {
    for cmd in python3.12 python3 python; do
        command -v "$cmd" &>/dev/null || continue
        ver=$("$cmd" --version 2>&1 | awk '{print $2}')
        major="${ver%%.*}"; rest="${ver#*.}"; minor="${rest%%.*}"
        [[ "$major" -ge 3 && "$minor" -ge 12 ]] && return 0
    done
    return 1
}

if ! _has_python312; then
    if command -v module &>/dev/null; then
        info "Python >=3.12 not in PATH — loading Python/3.12.3-GCCcore-14.2.0 module..."
        module load Python/3.12.3-GCCcore-14.2.0
    else
        die "Python >=3.12 not found and 'module' command unavailable. Source the module init or load Python 3.12 manually."
    fi
fi

PYTHON=""
for cmd in python3.12 python3 python; do
    command -v "$cmd" &>/dev/null || continue
    ver=$("$cmd" --version 2>&1 | awk '{print $2}')
    major="${ver%%.*}"; rest="${ver#*.}"; minor="${rest%%.*}"
    if [[ "$major" -ge 3 && "$minor" -ge 12 ]]; then
        PYTHON="$cmd"; break
    fi
done
[[ -n "$PYTHON" ]] || die "Python >=3.12 not found even after module load. Check available modules with: module spider Python"
info "Using $PYTHON — $("$PYTHON" --version)"

# ── 2. Poetry ─────────────────────────────────────────────────────────────────
step "Checking Poetry"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v poetry &>/dev/null; then
    info "Poetry not found — installing via installer..."
    curl -sSL https://install.python-poetry.org | "$PYTHON" -
    info "Added \$HOME/.local/bin to PATH for this session."
    info "Add the following to your ~/.bashrc or ~/.bash_profile:"
    echo '    export PATH="$HOME/.local/bin:$PATH"'
fi
poetry --version

# ── 3. Project dependencies ───────────────────────────────────────────────────
step "Installing project deps (poetry install)"
poetry env use "$PYTHON"
poetry install --with dev --no-interaction

# ── 4. PyTorch — cu128 wheels (WAVE ships CUDA 12.x; PyTorch module is broken) ─
step "Installing PyTorch (cu128)"
info "Installing torch, torchvision, torchaudio from pytorch.org/whl/cu128..."
poetry run pip install --quiet \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128
info "PyTorch installed."

# ── 5. vLLM ───────────────────────────────────────────────────────────────────
step "Installing vLLM"
poetry run pip install --quiet "vllm>=0.7"
info "vLLM installed: $(poetry run python -c 'import vllm; print(vllm.__version__)')"

# ── 6. Sanity check ───────────────────────────────────────────────────────────
step "Sanity check"
poetry run python - <<'PY'
import torch, sys

print(f"  torch    {torch.__version__}")
print(f"  cuda available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"  CUDA     {torch.version.cuda}")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {p.name}  {p.total_memory // 1024**3} GB")
else:
    # Login nodes have no GPU — this is expected and not an error.
    print("  (no GPU on login node — that is normal; CUDA will be available on compute nodes)")

try:
    import vllm
    print(f"  vllm     {vllm.__version__}")
except Exception as e:
    print(f"  vllm import failed: {e}", file=sys.stderr)
    sys.exit(1)
PY
info "All imports OK."

# ── 7. Model downloads (optional) ─────────────────────────────────────────────
if [[ "$DOWNLOAD_MODELS" == true ]]; then
    step "Downloading models"
    # Default to the shared class project space so the team shares one copy.
    # Override by setting HF_HOME before running this script.
    HF_HOME="${HF_HOME:-/WAVE/projects/CSEN-346-Sp26/hf_models}"
    mkdir -p "$HF_HOME"
    info "Model destination: $HF_HOME"

    info "Downloading SocratTeachLLM → $HF_HOME/SocratTeachLLM"
    poetry run hf download ulises-c/SocratTeachLLM \
        --local-dir "$HF_HOME/SocratTeachLLM"

    info "Downloading Qwen3.5-9B → $HF_HOME/Qwen3.5-9B"
    poetry run hf download Qwen/Qwen3.5-9B \
        --local-dir "$HF_HOME/Qwen3.5-9B"

    info "Models downloaded to $HF_HOME"
    echo ""
    du -sh "$HF_HOME/SocratTeachLLM" "$HF_HOME/Qwen3.5-9B" 2>/dev/null || true
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}Setup complete!${NC}"
echo ""
echo "Next steps:"
if [[ "$DOWNLOAD_MODELS" == false ]]; then
    echo "  1. Download models (from login node, needs internet):"
    echo "       bash scripts/wave_setup.sh --models"
    echo ""
    echo "  2. Submit the SLURM job:"
else
    echo "  1. Submit the SLURM job:"
fi
echo "       sbatch scripts/slurm/wave_eval.slurm"
echo ""
echo "  Tail logs during the run:"
echo "       tail -f logs/slurm-<JOBID>.out"
echo ""
echo "  Interactive GPU session (for debugging):"
echo "       srun --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=96G --gres=gpu:2 --time=02:00:00 --pty bash"
