.PHONY: help run install-hooks slurm \
        post-eval-shutdown run-eval \
        serve-both serve-dual-gpu serve-consultant serve-gemma4 serve-socratteachllm serve-teacher-online \
        setup-l40s

# Default target
help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "  run                   Show how to launch the project via poetry"
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
