#!/usr/bin/env bash
# One-time setup for a local dual-GPU machine (tested on 2× NVIDIA L40S 48 GB).
# Works on any Linux box with NVIDIA drivers and Poetry installed.
#
# Usage:
#   bash scripts/l40s_setup.sh              # Install deps only
#   bash scripts/l40s_setup.sh --models     # Install deps + download models (~36 GB)

set -euo pipefail
cd "$(dirname "$0")/.."

# Initialize pyenv shims if pyenv is installed — needed in non-interactive shells
# where ~/.bashrc hasn't run and the shims would otherwise exit 127.
if command -v pyenv &>/dev/null; then
    eval "$(pyenv init -)"
fi

DOWNLOAD_MODELS=false
for arg in "$@"; do
    [[ "$arg" == "--models" ]] && DOWNLOAD_MODELS=true
done

# ── Colours & helpers ────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; BOLD='\033[1m'; NC='\033[0m'
info() { echo -e "${GREEN}[setup]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn]${NC}  $*"; }
die()  { echo -e "${RED}[error]${NC} $*" >&2; exit 1; }
step() { echo -e "\n${BOLD}── $* ──${NC}"; }

# ── Logging ───────────────────────────────────────────────────────────────────
mkdir -p logs
LOG_FILE="logs/l40s_setup.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Log: $(pwd)/$LOG_FILE"

echo ""
echo -e "${BOLD}KELE / L40S Setup${NC}"
echo "=============================="
date
hostname
echo ""

# ── 1. NVIDIA driver check ────────────────────────────────────────────────────
step "Checking NVIDIA driver"
command -v nvidia-smi &>/dev/null || die "nvidia-smi not found — install the NVIDIA driver first."
nvidia-smi --query-gpu=index,name,memory.total,compute_cap \
    --format=csv,noheader 2>/dev/null \
    | while IFS=',' read -r idx name mem cc; do
        info "GPU $idx:$(echo "$name" | xargs)  $(echo "$mem" | xargs)  CC $(echo "$cc" | xargs)"
      done
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
[[ "$GPU_COUNT" -ge 2 ]] || warn "Only $GPU_COUNT GPU(s) detected — serve-dual-gpu expects 2."

# ── 2. PyTorch CUDA wheel selection ──────────────────────────────────────────
# Pick the right +cuXXX index based on the installed CUDA runtime.
# nvidia-smi reports the *driver* CUDA ceiling; nvcc --version gives the
# toolkit version. We prefer nvcc if available, fall back to driver ceiling.
step "Detecting CUDA version for PyTorch wheel"

_cuda_major_minor() {
    if command -v nvcc &>/dev/null; then
        nvcc --version 2>/dev/null \
            | awk '/release/{match($0,/[0-9]+\.[0-9]+/); print substr($0,RSTART,RLENGTH); exit}'
    else
        nvidia-smi 2>/dev/null \
            | awk '/CUDA Version:/{match($0,/[0-9]+\.[0-9]+/); print substr($0,RSTART,RLENGTH); exit}'
    fi
}

CUDA_VER=$(_cuda_major_minor)
CUDA_MAJOR="${CUDA_VER%%.*}"
CUDA_MINOR="${CUDA_VER#*.}"

if   [[ "$CUDA_MAJOR" -ge 13 ]];                             then TORCH_INDEX="cu128"
elif [[ "$CUDA_MAJOR" -eq 12 && "$CUDA_MINOR" -ge 8 ]];     then TORCH_INDEX="cu128"
elif [[ "$CUDA_MAJOR" -eq 12 && "$CUDA_MINOR" -ge 6 ]];     then TORCH_INDEX="cu126"
elif [[ "$CUDA_MAJOR" -eq 12 && "$CUDA_MINOR" -ge 4 ]];     then TORCH_INDEX="cu124"
elif [[ "$CUDA_MAJOR" -eq 12 ]];                             then TORCH_INDEX="cu121"
else
    warn "CUDA $CUDA_VER is older than 12.x — defaulting to cu128 wheel."
    warn "If PyTorch fails to load, override: TORCH_INDEX=cu126 bash scripts/l40s_setup.sh"
    TORCH_INDEX="${TORCH_INDEX:-cu128}"
fi

TORCH_INDEX="${TORCH_INDEX:-cu128}"   # env var override wins
info "CUDA $CUDA_VER detected → using pytorch.org/whl/$TORCH_INDEX"

# ── 3. Python ≥ 3.12 ──────────────────────────────────────────────────────────
step "Checking Python"

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
    # Try pyenv (common on Arch/dev machines)
    if command -v pyenv &>/dev/null; then
        info "Python >=3.12 not in PATH — trying pyenv..."
        pyenv install --skip-existing 3.12
        export PATH="$(pyenv root)/versions/3.12.*/bin:$PATH"
    else
        die "Python >=3.12 not found. Install it via pyenv, your distro's package manager, or from python.org."
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
[[ -n "$PYTHON" ]] || die "Python >=3.12 still not found after pyenv. Check your shell PATH."
info "Using $PYTHON — $("$PYTHON" --version)"

# ── 4. Poetry ─────────────────────────────────────────────────────────────────
step "Checking Poetry"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v poetry &>/dev/null; then
    info "Poetry not found — installing..."
    curl -sSL https://install.python-poetry.org | "$PYTHON" -
    info "Add the following to your ~/.bashrc or ~/.zshrc:"
    echo '    export PATH="$HOME/.local/bin:$PATH"'
fi
poetry --version

# ── 5. Project dependencies ───────────────────────────────────────────────────
step "Installing project deps (poetry install)"
# Resolve the real Python binary path, bypassing pyenv shims (which exit 127
# when pyenv doesn't manage that version, even if the system has it).
_real_python_path() {
    local cmd="$1"
    # Prefer well-known system paths so we never hit a shim.
    for dir in /usr/bin /usr/local/bin /opt/homebrew/bin; do
        [[ -x "$dir/$cmd" ]] && echo "$dir/$cmd" && return
    done
    # Fallback: first non-pyenv entry on PATH.
    which -a "$cmd" 2>/dev/null | grep -v '\.pyenv' | head -1
}
PYTHON_PATH=$(_real_python_path "$PYTHON")
[[ -n "$PYTHON_PATH" ]] || PYTHON_PATH="$PYTHON"
info "Resolved Python path: $PYTHON_PATH"
poetry env use "$PYTHON_PATH"
poetry install --with dev --no-interaction

# ── 6. PyTorch ───────────────────────────────────────────────────────────────
step "Installing PyTorch ($TORCH_INDEX)"
poetry run pip install --quiet \
    torch torchvision torchaudio \
    --index-url "https://download.pytorch.org/whl/$TORCH_INDEX"
info "PyTorch installed."

# ── 7. vLLM ───────────────────────────────────────────────────────────────────
step "Installing vLLM"
poetry run pip install --quiet "vllm>=0.7"
info "vLLM installed: $(poetry run python -c 'import vllm; print(vllm.__version__)')"

# ── 8. Pin vLLM-compatible versions ──────────────────────────────────────────
# poetry install may pull versions of transformers/setuptools that vLLM rejects.
step "Pinning vLLM-compatible package versions"
poetry run pip install --quiet "transformers<5.0.0" "setuptools<81.0.0"
info "Pinned transformers and setuptools."

# ── 9. Sanity check ───────────────────────────────────────────────────────────
step "Sanity check"
poetry run python - <<'PY'
import torch, sys

print(f"  torch  {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"  CUDA   {torch.version.cuda}")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        cc = f"{p.major}.{p.minor}"
        print(f"  GPU {i}: {p.name}  {p.total_memory // 1024**3} GB  CC {cc}")
    if torch.cuda.device_count() < 2:
        print("  WARNING: fewer than 2 GPUs — serve-dual-gpu needs 2", file=sys.stderr)
else:
    print("  ERROR: CUDA not available — check driver and torch wheel", file=sys.stderr)
    sys.exit(1)

try:
    import vllm
    print(f"  vllm   {vllm.__version__}")
except Exception as e:
    print(f"  vllm import failed: {e}", file=sys.stderr)
    sys.exit(1)
PY
info "All imports OK."

# ── 10. .env ──────────────────────────────────────────────────────────────────
step "Environment file"
if [[ ! -f .env ]]; then
    cp .env.example .env
    info "Created .env from .env.example."
    info "For a fully local run (no OpenAI key), edit .env and set:"
    echo "    CONSULTANT_API_KEY=not-needed"
    echo "    CONSULTANT_BASE_URL=http://localhost:8002/v1"
    echo "    CONSULTANT_MODEL_NAME=Qwen3.5-9B"
else
    info ".env already exists — skipping."
fi

# ── 11. Model downloads (optional) ────────────────────────────────────────────
if [[ "$DOWNLOAD_MODELS" == true ]]; then
    step "Downloading models (~36 GB total)"
    HF_HOME="${HF_HOME:-$HOME/hf_models}"
    mkdir -p "$HF_HOME"
    info "Model destination: $HF_HOME"

    # Require HF auth — hf download will fail with a confusing error if not logged in.
    if ! poetry run hf auth whoami &>/dev/null; then
        warn "Not logged in to Hugging Face."
        warn "Run: poetry run hf auth login"
        warn "Then re-run with --models to download."
        exit 1
    fi

    info "Downloading SocratTeachLLM (~19 GB)..."
    poetry run hf download ulises-c/SocratTeachLLM \
        --local-dir "$HF_HOME/SocratTeachLLM"

    info "Downloading Qwen3.5-9B (~17 GB)..."
    poetry run hf download Qwen/Qwen3.5-9B \
        --local-dir "$HF_HOME/Qwen3.5-9B"

    info "Downloads complete."
    echo ""
    du -sh "$HF_HOME/SocratTeachLLM" "$HF_HOME/Qwen3.5-9B" 2>/dev/null || true
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}Setup complete!${NC}"
echo ""
if [[ "$DOWNLOAD_MODELS" == false ]]; then
    echo "Next steps:"
    echo "  1. Download models (needs internet, ~36 GB):"
    echo "       bash scripts/l40s_setup.sh --models"
    echo ""
    echo "  2. Start both model servers:"
    echo "       make serve-dual-gpu"
    echo ""
    echo "  3. Run evaluation (in a second terminal once servers are ready):"
    echo "       poetry run kele-eval"
else
    echo "Next steps:"
    echo "  1. Start both model servers:"
    echo "       make serve-dual-gpu"
    echo ""
    echo "  2. Run evaluation (in a second terminal once servers are ready):"
    echo "       poetry run kele-eval"
fi
echo ""
echo "  Monitor GPU usage:"
echo "       watch -n 2 nvidia-smi"
echo ""
echo "  Log from this run: $LOG_FILE"
