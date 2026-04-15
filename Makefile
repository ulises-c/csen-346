.PHONY: help run \
        post-eval-shutdown run-eval \
        serve-both serve-consultant serve-gemma4 serve-socratteachllm serve-teacher-online

# Default target
help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "  run                   Show how to launch the project via poetry"
	@echo ""
	@echo "  Scripts (scripts/):"
	@echo "  post-eval-shutdown    Run scripts/post_eval_shutdown.sh"
	@echo "  run-eval              Run scripts/run_eval.sh"
	@echo "  serve-both            Run scripts/serve_both.sh"
	@echo "  serve-consultant      Run scripts/serve_consultant.sh"
	@echo "  serve-gemma4          Run scripts/serve_gemma4.sh"
	@echo "  serve-socratteachllm  Run scripts/serve_socratteachllm.sh"
	@echo "  serve-teacher-online  Run scripts/serve_teacher_online.sh"

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

# ── scripts/ targets ─────────────────────────────────────────────────────────

post-eval-shutdown:
	bash scripts/post_eval_shutdown.sh

run-eval:
	bash scripts/run_eval.sh

serve-both:
	bash scripts/serve_both.sh

serve-consultant:
	bash scripts/serve_consultant.sh

serve-gemma4:
	bash scripts/serve_gemma4.sh

serve-socratteachllm:
	bash scripts/serve_socratteachllm.sh

serve-teacher-online:
	bash scripts/serve_teacher_online.sh
