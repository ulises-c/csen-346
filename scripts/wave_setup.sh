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
    HF_HOME="${HF_HOME:-$HOME/hf_models}"
    mkdir -p "$HF_HOME"

    info "Downloading SocratTeachLLM → $HF_HOME/SocratTeachLLM"
    poetry run huggingface-cli download ulises-c/SocratTeachLLM \
        --local-dir "$HF_HOME/SocratTeachLLM"

    info "Downloading Qwen3.5-9B → $HF_HOME/Qwen3.5-9B"
    poetry run huggingface-cli download Qwen/Qwen3.5-9B \
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
