.PHONY: help run install install-rocm install-cuda install-hooks slurm \
        post-eval-shutdown run-eval \
        serve-both serve-dual-gpu serve-consultant serve-gemma4 serve-socratteachllm serve-teacher-online \
        setup-l40s pre-commit

# Default target
help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "  run                   Show how to launch the project via poetry"
	@echo "  install               Auto-detect GPU vendor and install matching torch (ROCm or CUDA)"
	@echo "  install-rocm          Install torch for AMD/ROCm (rocm6.3)"
	@echo "  install-cuda          Install torch for NVIDIA/CUDA (cu126)"
	@echo "  install-hooks         Install git hooks from hooks/ into .git/hooks/"
	@echo ""
	@echo "  Scripts (scripts/):"
	@echo "  post-eval-shutdown    Run scripts/post_eval_shutdown.sh"
	@echo "  run-eval              Run scripts/run_eval.sh  (GPU=<config>, default: baseline)
                          Dual-GPU configs: GPU=l40s  GPU=3090ti
                          Other configs:    GPU=baseline  GPU=gemma4
                          Tested hardware:  RTX 5090, RTX 3090 Ti, AMD R9700, NVIDIA L40S, V100 32GB"
	@echo "  setup-l40s            Run scripts/l40s_setup.sh (one-time setup for dual L40S machine)"
	@echo "  serve-both            Run scripts/serve_both.sh (single GPU, shared VRAM)"
	@echo "  serve-dual-gpu        Run scripts/serve_dual_gpu.sh (2 GPUs, teacher→GPU0 consultant→GPU1)"
	@echo "  serve-consultant      Run scripts/serve_consultant.sh"
	@echo "  serve-gemma4          Run scripts/serve_gemma4.sh"
	@echo "  serve-socratteachllm  Run scripts/serve_socratteachllm.sh"
	@echo "  serve-teacher-online  Run scripts/serve_teacher_online.sh"
	@echo ""
	@echo "  WAVE HPC (SLURM):"
	@echo "  slurm                 git pull + sbatch wave_eval.slurm + print status"

# ── Entry point ──────────────────────────────────────────────────────────────

run:
	@echo "Run the project via poetry:"
	@echo ""
	@echo "  poetry run kele            # main KELE entry point"
	@echo "  poetry run kele-eval       # run evaluation"
	@echo "  poetry run serve-teacher   # start teacher server"
	@echo ""
	@echo "  poetry run test            # run tests (or: make test)"
	@echo "  poetry run lint            # lint source  (or: make lint)"
	@echo ""
	@echo "  make pre-commit            # run format + lint + tests (mirrors git pre-commit hook)"

# ── Code quality ─────────────────────────────────────────────────────────────

pre-commit:
	poetry run ruff format .
	poetry run ruff check --fix .
	poetry run pytest -rs

# ── Torch install ────────────────────────────────────────────────────────────
# torch is not declared in pyproject.toml because Poetry cannot resolve the
# +rocm6.3 / +cu126 local-version identifiers alongside PyPI's CPU wheel.
# These targets install torch after `poetry install --no-root`.

# Auto-detect: prefer ROCm if rocm-smi is present, fall back to CUDA.
install:
	@echo "→ Installing base dependencies …"
	poetry install --no-root
	@if command -v rocm-smi >/dev/null 2>&1 && rocm-smi >/dev/null 2>&1; then \
	  echo "→ AMD/ROCm GPU detected — installing torch+rocm6.3"; \
	  $(MAKE) _install-torch-rocm; \
	elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then \
	  echo "→ NVIDIA GPU detected — installing torch+cu126"; \
	  $(MAKE) _install-torch-cuda; \
	else \
	  echo ""; \
	  echo "  No GPU detected (rocm-smi and nvidia-smi both unavailable)."; \
	  echo "  Re-run with an explicit target:"; \
	  echo "    make install-rocm   # AMD / ROCm"; \
	  echo "    make install-cuda   # NVIDIA / CUDA"; \
	  echo ""; \
	  exit 1; \
	fi

install-rocm:
	poetry install --no-root
	$(MAKE) _install-torch-rocm

install-cuda:
	poetry install --no-root
	$(MAKE) _install-torch-cuda

# Internal targets — call via install-rocm / install-cuda / install
_install-torch-rocm:
	poetry run pip install --force-reinstall --no-deps \
	  --index-url https://download.pytorch.org/whl/rocm6.3 \
	  "torch==2.9.1+rocm6.3"
	@echo "✓ torch 2.9.1+rocm6.3 installed"
	poetry run pip install -e . --no-deps
	@echo "✓ project entry points installed"

_install-torch-cuda:
	poetry run pip install --force-reinstall --no-deps \
	  --index-url https://download.pytorch.org/whl/cu126 \
	  "torch>=2.9.0"
	@echo "✓ torch+cu126 installed"
	poetry run pip install -e . --no-deps
	@echo "✓ project entry points installed"

# ── Developer setup ──────────────────────────────────────────────────────────

install-hooks:
	@echo "Installing git hooks from hooks/ → .git/hooks/ …"
	@for hook in hooks/*; do \
	  name=$$(basename $$hook); \
	  cp "$$hook" ".git/hooks/$$name"; \
	  chmod +x ".git/hooks/$$name"; \
	  echo "  installed $$name"; \
	done
	@echo "Done. Hooks will run automatically on git operations."

# ── scripts/ targets ─────────────────────────────────────────────────────────

setup-l40s:
	bash scripts/l40s_setup.sh

post-eval-shutdown:
	bash scripts/post_eval_shutdown.sh

# TODO: auto-detect GPU config from hardware — query nvidia-smi for compute
# capability and total VRAM per device, then select the appropriate configs/
# file automatically (e.g. 2×24GB CC≥8.6 → 3090ti, 2×48GB CC≥8.9 → l40s,
# single GPU → serve-both, V100/CC<8.0 → float16 + enforce-eager, etc.).
# Planned: make run-eval with no GPU= arg runs detection and picks the config.
GPU ?= baseline

run-eval:
	bash scripts/run_eval.sh $(GPU)

serve-both:
	bash scripts/serve_both.sh

serve-dual-gpu:
	bash scripts/serve_dual_gpu.sh

serve-consultant:
	bash scripts/serve_consultant.sh

serve-gemma4:
	bash scripts/serve_gemma4.sh

serve-socratteachllm:
	bash scripts/serve_socratteachllm.sh

serve-teacher-online:
	bash scripts/serve_teacher_online.sh

# ── WAVE HPC ──────────────────────────────────────────────────────────────────

slurm:
	bash scripts/slurm/submit_wave.sh
