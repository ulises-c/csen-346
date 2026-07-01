.PHONY: help run install-hooks pre-commit sync-mirror setup setup-repo \
		online-demo local-demo _demo-preflight \
		nvidia-preflight _classifier-ckpt \
		train-gemma4-12b train-gemma4-12b-dry-run monitor-train-gemma4-12b \
		serve-gemma4-12b serve-gemma4-12b-mtp serve-gemma4-12b-sft \
		eval-gemma4-12b-base-smoke eval-gemma4-12b-base-full \
		eval-gemma4-12b-sft-smoke eval-gemma4-12b-sft-full \
		monitor-eval-gemma4-12b-base monitor-eval-gemma4-12b-sft \
		slurm \
        post-eval-shutdown run-eval \
        eval-qwen27b-smoke eval-qwen27b-mini eval-qwen27b-full \
        eval-qwen27b-fusion-smoke eval-qwen27b-fusion-nothink-smoke \
        eval-qwen35b-a3b-smoke eval-qwen35b-a3b-mini eval-qwen35b-a3b-full \
        eval-qwen35b-a3b-fusion-smoke eval-qwen35b-a3b-fusion-nothink-smoke \
        eval-gemma4-31b-smoke eval-gemma4-31b-mini eval-gemma4-31b-full \
        eval-gemma4-31b-fusion-smoke \
        serve-both serve-dual-gpu serve-consultant serve-gemma4 \
        serve-gemma4-31b serve-gemma4-26b-a4b \
        serve-qwen27b serve-qwen27b-q4 serve-qwen35b-a3b \
        serve-glm47-23b serve-qwopus35b-a3b \
        serve-socratteachllm serve-teacher-online \
        serve-demo \
        setup-l40s start-local-tl-server \
        test-gpu-stack test-vllm \
        patch-fla-rocm patch-fla-rocm-restore patch-fla-rocm-dry-run \
        download-gemma4-31b \
        prequant-gemma4-31b-l40s transfer-gemma4-31b-nf4 \
        train-gemma4-31b-dry-run train-gemma4-31b-stage2 train-gemma4-31b-stage2-preq \
        train-gemma4-31b-stage2-unsloth train-gemma4-31b-eos-gate eos-gate-gemma4-31b \
        gpu-preflight diagnose-gfx1201-fault \
        profile-gemma4-31b \
        tournament tournament-think tournament-warmup tournament-warmup-think tournament-status tournament-eliminate \
        tournament-finalize tournament-archive tournament-restore tournament-reset \
        tournament-download tournament-help

# ── Remotes ───────────────────────────────────────────────────────────────────
# ulises-c/csen-346 is the source of truth; SCU-CSEN346/KELE is a 1:1 mirror.
SOURCE_REMOTE := git@github.com:ulises-c/csen-346.git
KELE_REMOTE   := git@github.com:SCU-CSEN346/KELE.git

# Default target
help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "  run                   Show how to launch the project via uv"
	@echo "  install-hooks         Install git hooks from hooks/ into .git/hooks/"
	@echo ""
	@echo "  GPU stack tests:"
	@echo "  test-gpu-stack        Full ML stack: ROCm, torch, bitsandbytes 8/4-bit, transformers, PEFT, TRL, flash-attn"
	@echo "  test-vllm             vLLM ROCm engine probe (no model weights)"
	@echo ""
	@echo "  RDNA4 / gfx1201 training workarounds:"
	@echo "  patch-fla-rocm        Patch flash-linear-attention num_stages>=2 → 1 (Triton 3.6.0 RDNA4 fix)"
	@echo "  patch-fla-rocm-dry-run  Count refs without modifying"
	@echo "  patch-fla-rocm-restore  Roll back from .bak files"
	@echo ""
	@echo "  Scripts (scripts/):"
	@echo "  post-eval-shutdown    Run scripts/post_eval_shutdown.sh"
	@echo "  run-eval              Run scripts/run_eval.sh  (GPU=<config>, default: baseline)"
	@echo "                          Dual-GPU configs: GPU=l40s  GPU=3090ti"
	@echo "                          Other configs:    GPU=baseline  GPU=gemma4"
	@echo "                          Tested hardware:  RTX 5090, RTX 3090 Ti, AMD R9700, NVIDIA L40S, V100 32GB"
	@echo "  setup-l40s            Run scripts/l40s_setup.sh (one-time setup for dual L40S machine)"
	@echo "  serve-both            Run scripts/serve_both.sh (single GPU, shared VRAM)"
	@echo "  serve-dual-gpu        Run scripts/serve_dual_gpu.sh (2 GPUs, teacher→GPU0 consultant→GPU1)"
	@echo "  serve-consultant      Run scripts/serve_consultant.sh"
	@echo "  serve-gemma4          Run scripts/serve_gemma4.sh (vLLM + Gemma-4-31B-IT-NVFP4, multi-server)"
	@echo "  serve-gemma4-31b      Run scripts/serve_gemma4_31b_q5.sh (Gemma 4 31B Q5 GGUF, dual-role on llama.cpp)"
	@echo "  serve-qwen27b         Run scripts/serve_qwen27b_q5.sh (Qwen3.6-27B Q5, dual-role teacher+consultant)"
	@echo "  serve-qwen27b-q4      Run scripts/serve_qwen27b_q4_local.sh (Qwen3.6-27B Q4, ~/models/, AMD R9700)"
	@echo "  serve-qwen35b-a3b     Run scripts/serve_qwen35b_a3b.sh (Qwen3.6-35B-A3B MoE, ~3x faster than 27B)"
	@echo "  serve-glm47-23b       Run scripts/serve_glm47_23b.sh (GLM-4.7-Flash REAP 23B-A3B, 14 GB, AMD R9700)"
	@echo "  serve-qwopus35b-a3b   Run scripts/serve_qwopus35b_a3b.sh (Qwopus 35B-A3B LoRA fine-tune, 21 GB)"
	@echo "  serve-socratteachllm  Run scripts/serve_socratteachllm.sh"
	@echo "  serve-teacher-online  Run scripts/serve_teacher_online.sh"
	@echo "  online-demo           Laptop demo via OpenRouter free tier (downloads classifier if missing)"
	@echo "  local-demo            Local demo via llama.cpp on this machine (downloads classifier if missing)"
	@echo "  serve-demo            Self-host the top-performer stack for a live demo (RTX 5090 + Tailscale)"
	@echo "  start-local-tl-server  Start local llama.cpp server for dataset translation (Qwen3.5-9B)"
	@echo "  eval-qwen27b-smoke    Run scripts/eval_llamacpp.sh qwen27b smoke (n=5,   ~5 min)"
	@echo "  eval-qwen27b-mini     Run scripts/eval_llamacpp.sh qwen27b mini  (n=25,  ~15 min)"
	@echo "  eval-qwen27b-full     Run scripts/eval_llamacpp.sh qwen27b full  (n=681, ~75 h — measured)"
	@echo "  eval-qwen35b-a3b-smoke Run scripts/eval_llamacpp.sh qwen35b-a3b smoke (n=5,   ~2 min projected)"
	@echo "  eval-qwen35b-a3b-mini  Run scripts/eval_llamacpp.sh qwen35b-a3b mini  (n=25,  ~5 min projected)"
	@echo "  eval-qwen35b-a3b-full  Run scripts/eval_llamacpp.sh qwen35b-a3b full  (n=681, ~20-30 h projected)"
	@echo "  download-gemma4-31b         Download google/gemma-4-31b-it weights to HF cache (~60 GB)"
	@echo "  prequant-gemma4-31b-l40s    Print instructions for pre-quantizing to NF4 on L40S"
	@echo "  transfer-gemma4-31b-nf4     rsync NF4 checkpoint from L40S (HOST=user@host)"
	@echo "  train-gemma4-31b-stage2-preq  Train Stage 2b from local pre-quantized NF4 checkpoint"
	@echo "  train-gemma4-31b-stage2-unsloth  Train Stage 2b from unsloth bnb-4bit Gemma 4 31B (no local prequant)"
	@echo "  train-gemma4-31b-eos-gate    100-step checkpoint for EOS gate (unsloth path, ~30 min)"
	@echo "  eos-gate-gemma4-31b          Run EOS gate against outputs/eos-gate-gemma4-31b/final"
	@echo "  gpu-preflight                Fast GPU gate (clean KFD + fwd/bwd) — run before any (re)launch"
	@echo "  diagnose-gfx1201-fault       Serialized-kernel run to localize the backward page fault"
	@echo "  profile-gemma4-31b           Profile a real Stage 2 step (attention vs NF4-dequant; FA2 de-risk)"
	@echo "  eval-gemma4-31b-smoke  Run scripts/eval_llamacpp.sh gemma4-31b smoke  (n=5)"
	@echo "  eval-gemma4-31b-mini   Run scripts/eval_llamacpp.sh gemma4-31b mini   (n=25)"
	@echo "  eval-gemma4-31b-full   Run scripts/eval_llamacpp.sh gemma4-31b full   (n=681)"
	@echo ""
	@echo "  Fusion smoke targets (single-call architecture, see SOCRATIC_FUSION_PLAN.md):"
	@echo "  eval-qwen27b-fusion-smoke           27B + unified (think on)"
	@echo "  eval-qwen27b-fusion-nothink-smoke   27B + unified + no-think"
	@echo "  eval-qwen35b-a3b-fusion-smoke         A3B + unified (think on)"
	@echo "  eval-qwen35b-a3b-fusion-nothink-smoke A3B + unified + no-think"
	@echo "  eval-gemma4-31b-fusion-smoke          Gemma 4 31B + unified (Gemma has no thinking-mode)"
	@echo ""
	@echo "  WAVE HPC (SLURM):"
	@echo "  slurm                 git pull + sbatch wave_eval.slurm + print status"
	@echo ""
	@echo "  Tournament (multi-model elimination):"
	@echo "  tournament-help       Full tournament command reference"
	@echo "  tournament            Run one round (n=50, fusion, no-think)"
	@echo "  tournament-think      Run one round (n=50, fusion, thinking budget=4096)"
	@echo "  tournament-warmup     Warmup (n=5, thinking OFF) — verifies models load; do NOT eliminate after"
	@echo "  tournament-warmup-think  Warmup (n=5, thinking budget=4096) — verifies thinking tokens are generated"
	@echo "  tournament-status     Print leaderboard"
	@echo "  tournament-archive    Save current run to archive/<run_id>/ and reset state"
	@echo "  tournament-restore    List archives  (use ID=<id> to restore one)"
	@echo "  tournament-eliminate  Drop worst model  (N=2 to drop two, etc.)"
	@echo "  tournament-finalize   Run survivors to n=681 (fusion mode)"
	@echo "  tournament-reset      Wipe all tournament state  (add CONFIRM=1 to skip prompt)"
	@echo "  tournament-download   Download all pending model weights via hf CLI"


# ── Setup ─────────────────────────────────────────────────────────────────────

setup: setup-repo install-hooks
	@echo "This project uses uv for dependency management. Install uv from https://docs.astral.sh/uv/ if not already installed."
	@echo "Setting up the project via uv:"
	uv sync --group dev

setup-repo:
	@echo "Configuring dual-push remotes..."
	# Set the fetch URL
	git remote set-url origin $(SOURCE_REMOTE)
	# Replace push URL list (--push without --add resets to a single entry)
	git remote set-url --push origin $(SOURCE_REMOTE)
	# Add the second push URL (now idempotent: list was just reset above)
	git remote set-url --add --push origin $(KELE_REMOTE)
	@echo "Repository setup complete. Verify with 'git remote -v'."

# ── Dual remote synchronization ────────────────────────────────────────────────────

# Reconcile KELE to match the source of truth 1:1. Needed because the dual-push
# remote (see setup-repo) never propagates branch *deletions* and only the branch
# you push gets mirrored — so KELE drifts (stale branches, diverged dependabot).
# This snapshots source's authoritative refs and force-mirrors all branches + tags,
# pruning anything on KELE that no longer exists on source. main goes first so it is
# never left behind even if a later ref fails. WARNING: prunes KELE-only branches.
sync-mirror:
	@echo "Mirroring $(SOURCE_REMOTE) -> $(KELE_REMOTE) (all branches + tags, pruning stale)..."
	@git fetch --prune --no-tags $(SOURCE_REMOTE) \
		'+refs/heads/*:refs/mirror-src/heads/*' \
		'+refs/tags/*:refs/mirror-src/tags/*'
	@git push $(KELE_REMOTE) '+refs/mirror-src/heads/main:refs/heads/main'
	@git push --prune $(KELE_REMOTE) \
		'+refs/mirror-src/heads/*:refs/heads/*' \
		'+refs/mirror-src/tags/*:refs/tags/*'
	@echo "Mirror sync complete. KELE now matches $(SOURCE_REMOTE) 1:1."

# ── Entry point ──────────────────────────────────────────────────────────────

run:
	@echo "Run the project via uv:"
	@echo ""
	@echo "  uv run kele            # main KELE entry point"
	@echo "  uv run kele-eval       # run evaluation"
	@echo "  uv run serve-teacher   # start teacher server"
	@echo ""
	@echo "  uv run test            # run tests (or: make test)"
	@echo "  uv run lint            # lint source  (or: make lint)"
	@echo ""
	@echo "  make pre-commit        # run format + lint + tests (mirrors git pre-commit hook)"

# ── Code quality ─────────────────────────────────────────────────────────────

pre-commit:
	uvx ruff format .
	uvx ruff check --fix .
	uv run --no-sync pytest -rs

# ── Torch install ────────────────────────────────────────────────────────────
# torch is not declared in pyproject.toml because uv cannot resolve the
# +rocm7.2 / +cu126 local-version identifiers alongside PyPI's CPU wheel.
# These targets install torch after `uv sync`.

# Auto-detect: prefer ROCm if rocm-smi is present, fall back to CUDA.
install:
	@echo "→ Installing base dependencies …"
	uv sync
	@if command -v rocm-smi >/dev/null 2>&1 && rocm-smi >/dev/null 2>&1; then \
	  echo "→ AMD/ROCm GPU detected — installing torch+rocm7.2"; \
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
	uv sync
	$(MAKE) _install-torch-rocm

install-cuda:
	uv sync
	$(MAKE) _install-torch-cuda

# Internal targets — call via install-rocm / install-cuda / install
_install-torch-rocm:
	uv pip install --force-reinstall \
	  --index-url https://download.pytorch.org/whl/rocm7.2 \
	  "torch==2.11.0" "torchaudio==2.11.0"
	uv pip uninstall torchvision 2>/dev/null || true
	@echo "✓ torch+rocm7.2 installed (torchvision excluded — ABI mismatch on gfx1201)"

_install-torch-cuda:
	uv pip install --force-reinstall \
	  --index-url https://download.pytorch.org/whl/cu126 \
	  torch torchvision torchaudio
	@echo "✓ torch+cu126 installed"

# ── RDNA4 / gfx1201 FLA Triton workaround ────────────────────────────────────
# Patches flash-linear-attention autotune configs: num_stages>=2→1, num_warps>4→4.
# Addresses Triton tritonamdgpu-pipeline UAF + RDNA4 wave-32 scheduling issues.
# Re-run after every `uv sync` or `make install-rocm`. See PR #79 thread.
patch-fla-rocm:
	bash scripts/patch_fla_rocm.sh

patch-fla-rocm-dry-run:
	bash scripts/patch_fla_rocm.sh --dry-run

patch-fla-rocm-restore:
	bash scripts/patch_fla_rocm.sh --restore

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

serve-gemma4-31b:
	bash scripts/serve_gemma4_31b_q5.sh

serve-gemma4-26b-a4b:
	bash scripts/serve_gemma4_26b_a4b.sh

serve-qwen27b:
	bash scripts/serve_qwen27b_q5.sh

serve-qwen27b-q4:
	bash scripts/serve_qwen27b_q4_local.sh

serve-socratteachllm:
	bash scripts/serve_socratteachllm.sh

serve-socratteachllm-llamacpp:
	bash scripts/serve_socratteachllm_llamacpp.sh

serve-teacher-online:
	bash scripts/serve_teacher_online.sh

BERT_CKPT ?= results/state-clf-qwen3.5-0.8b-lora-wandb/final

_demo-preflight:
	@if [[ ! -f "$(BERT_CKPT)/model.safetensors" ]]; then \
	  echo "Classifier checkpoint not found — downloading from HF…"; \
	  hf download ulises-c/socrates-state-classifier-qwen3.5-lora --local-dir "$(BERT_CKPT)"; \
	fi
	uv sync --extra demo --inexact

online-demo: _demo-preflight
	@if [[ ! -f .env ]] || ! grep -qE '^TEACHER_API_KEY=.+' .env; then \
	  printf 'error: TEACHER_API_KEY not set.\n'; \
	  printf '       Add it to .env:  TEACHER_API_KEY=sk-or-...\n'; \
	  exit 1; \
	fi
	ONLINE=1 KELE_BERT_DEVICE=cpu BERT_CKPT="$(BERT_CKPT)" bash scripts/serve_demo_top_performer.sh

local-demo: _demo-preflight
	WEBUI=1 KELE_BERT_DEVICE=cpu BERT_CKPT="$(BERT_CKPT)" bash scripts/serve_demo_top_performer.sh

serve-demo:
	bash scripts/serve_demo_top_performer.sh

start-local-tl-server:
	bash scripts/start_tl_server.sh

test-gpu-stack:
	bash scripts/test_gpu_stack.sh

test-vllm:
	bash scripts/test_vllm_rocm.sh

eval-qwen27b-smoke:
	bash scripts/eval_llamacpp.sh qwen27b smoke

eval-qwen27b-mini:
	bash scripts/eval_llamacpp.sh qwen27b mini

eval-qwen27b-full:
	bash scripts/eval_llamacpp.sh qwen27b full

serve-qwen35b-a3b:
	bash scripts/serve_qwen35b_a3b.sh

serve-glm47-23b:
	bash scripts/serve_glm47_23b.sh

serve-qwopus35b-a3b:
	bash scripts/serve_qwopus35b_a3b.sh

eval-qwen35b-a3b-smoke:
	bash scripts/eval_llamacpp.sh qwen35b-a3b smoke

eval-qwen35b-a3b-mini:
	bash scripts/eval_llamacpp.sh qwen35b-a3b mini

eval-qwen35b-a3b-full:
	bash scripts/eval_llamacpp.sh qwen35b-a3b full

eval-gemma4-31b-smoke:
	bash scripts/eval_llamacpp.sh gemma4-31b smoke

eval-gemma4-31b-mini:
	bash scripts/eval_llamacpp.sh gemma4-31b mini

eval-gemma4-31b-full:
	bash scripts/eval_llamacpp.sh gemma4-31b full

# ── Fusion smoke targets (single-call architecture) ──────────────────────────
# See docs/SOCRATIC_FUSION_PLAN.md. Each writes to a distinct results/ dir
# so all four can coexist alongside the existing two-call smoke results.

eval-qwen27b-fusion-smoke:
	bash scripts/eval_llamacpp.sh qwen27b smoke --unified

eval-qwen27b-fusion-nothink-smoke:
	bash scripts/eval_llamacpp.sh qwen27b smoke --unified --nothink

eval-qwen35b-a3b-fusion-smoke:
	bash scripts/eval_llamacpp.sh qwen35b-a3b smoke --unified

eval-qwen35b-a3b-fusion-nothink-smoke:
	bash scripts/eval_llamacpp.sh qwen35b-a3b smoke --unified --nothink

# Gemma 4 has no thinking-mode equivalent, so only the --unified variant exists.
eval-gemma4-31b-fusion-smoke:
	bash scripts/eval_llamacpp.sh gemma4-31b smoke --unified

# ── Gemma 4 12B SFT-uplift PoC (NVIDIA RTX 4000 Ada) ─────────────────────────
# Baseline (base 12B teacher) vs 1-epoch Socratic QLoRA SFT, same Qwen3.5-LoRA
# state classifier as consultant. NVIDIA box — none of the gfx1201/ROCm env that
# the 31B targets carry. See the 12B PoC plan + configs/train-sft-gemma4-12b-qlora.env.
# Serving and eval are separate steps: the 20 GB card hosts ONE model at a time, so
# the eval targets assume the matching server (serve-gemma4-12b{,-sft}) is already up.

# CUDA fwd/bwd smoke through the training kernel paths — the NVIDIA analogue of
# gpu-preflight (which is the ROCm/gfx1201 gate). Hard gate before train/serve.
nvidia-preflight:
	uv run --no-sync python scripts/test_training_gpu_nvidia.py

train-gemma4-12b-dry-run:
	uv run python scripts/train_sft.py --config configs/train-sft-gemma4-12b-qlora.env --dry-run

# Live-quantize the bf16 base to NF4 on load (TRAIN_PREQ=false → the code builds
# the bnb 4-bit config). unsloth/gemma-4-12b-it is the FULL bf16 model, not a
# bnb-4bit checkpoint, so TRAIN_PREQ=true loaded it in bf16 (~24 GB) and OOM'd on
# the 20 GB card; live quant is ~7.7 GB resident, leaving ~13 GB for activations.
# expandable_segments avoids allocator fragmentation at the VRAM edge (CUDA box).
train-gemma4-12b: nvidia-preflight
	mkdir -p outputs/sft-gemma4-12b-qlora
	nohup env TRAIN_BASE_MODEL=unsloth/gemma-4-12b-it \
	  TRAIN_PREQ=false \
	  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
	  TRAIN_HF_REPO=ulises-c/SocratesLM-12B-QLoRA \
	  TRAIN_HF_PUSH_EVERY=50 \
	  uv run --no-sync python scripts/train_sft.py \
	  --config configs/train-sft-gemma4-12b-qlora.env \
	  > outputs/sft-gemma4-12b-qlora/train.log 2>&1 &
	@echo "Training started. Monitor: tail -f outputs/sft-gemma4-12b-qlora/train.log"

# Auto-resume training crawl for the known-unstable box: owns the train→crash→
# resume loop, archives each fault, quarantines partial checkpoints, walks the GPU
# power limit DOWN per crash (stability search), and logs to issue #130. Re-invokes
# `make train-gemma4-12b` (nvidia-preflight + auto-resume) on every fault. Set
# LOG_COMMENT_ID=<id> to post rows to a pinned issue comment; unset = local log only.
# See scripts/monitor_train_gemma4_12b.sh.
monitor-train-gemma4-12b:
	bash scripts/monitor_train_gemma4_12b.sh

# Serve (port 8080, one at a time). serve-gemma4-12b-mtp attaches the MTP drafter.
serve-gemma4-12b:
	bash scripts/serve_gemma4_12b.sh

serve-gemma4-12b-mtp:
	MTP=1 bash scripts/serve_gemma4_12b.sh

serve-gemma4-12b-sft:
	bash scripts/serve_gemma4_12b_sft.sh

# Ensure the Qwen3.5-LoRA classifier checkpoint is present (consultant for both evals).
_classifier-ckpt:
	@if [[ ! -f "$(BERT_CKPT)/model.safetensors" ]]; then \
	  echo "Classifier checkpoint not found — downloading from HF…"; \
	  hf download ulises-c/socrates-state-classifier-qwen3.5-lora --local-dir "$(BERT_CKPT)"; \
	fi

# Eval: Qwen3.5-LoRA consultant (on CPU, frees VRAM for the teacher) + Gemma 12B
# teacher. smoke = n=5 sanity gate; full = n=681. Keep base/SFT invocations
# identical except the --experiment config so the delta isolates the SFT adapter.
eval-gemma4-12b-base-smoke: _classifier-ckpt
	KELE_BERT_DEVICE=cpu uv run python -m src.project.kele --experiment gemma4-12b-local \
	  test --n 5 --bert-consultant "$(BERT_CKPT)" --output results/gemma4-12b-base-smoke

eval-gemma4-12b-base-full: _classifier-ckpt
	WANDB_EVAL=1 KELE_BERT_DEVICE=cpu uv run python -m src.project.kele --experiment gemma4-12b-local \
	  evaluate --bert-consultant "$(BERT_CKPT)" --output results/gemma4-12b-base

eval-gemma4-12b-sft-smoke: _classifier-ckpt
	KELE_BERT_DEVICE=cpu uv run python -m src.project.kele --experiment gemma4-12b-sft-local \
	  test --n 5 --bert-consultant "$(BERT_CKPT)" --output results/gemma4-12b-sft-smoke

eval-gemma4-12b-sft-full: _classifier-ckpt
	WANDB_EVAL=1 KELE_BERT_DEVICE=cpu uv run python -m src.project.kele --experiment gemma4-12b-sft-local \
	  evaluate --bert-consultant "$(BERT_CKPT)" --output results/gemma4-12b-sft

# Monitored eval crawl for the known-unstable box: owns serve+eval, polls server
# health out-of-band, repairs error/truncated dialogues before relaunch, walks the
# GPU power limit down per crash, and logs to issue #130. MTP=1 attaches the
# drafter (-mtp output suffix). See scripts/monitor_eval_gemma4_12b.sh.
#   make monitor-eval-gemma4-12b-base            # base, MTP off  → results/gemma4-12b-base
#   MTP=1 make monitor-eval-gemma4-12b-base      # base, MTP on   → results/gemma4-12b-base-mtp
#   make monitor-eval-gemma4-12b-sft             # SFT            → results/gemma4-12b-sft
monitor-eval-gemma4-12b-base: _classifier-ckpt
	bash scripts/monitor_eval_gemma4_12b.sh base

monitor-eval-gemma4-12b-sft: _classifier-ckpt
	bash scripts/monitor_eval_gemma4_12b.sh sft

# Consultant ablation (handoff T1.1): both no-consultant (self-consult) arms back to
# back, SFT then base, one model at a time. No _classifier-ckpt dep — self-consult
# uses no external classifier. Long-running; launch detached (nohup … &).
#   nohup make noconsult-chain-gemma4-12b > outputs/noconsult_chain.nohup 2>&1 &
noconsult-chain-gemma4-12b:
	bash scripts/noconsult_chain_gemma4_12b.sh

# ── Gemma 4 31B SFT training (Stage 2b) ──────────────────────────────────────
# No patch-fla-rocm needed — Gemma 4 uses standard softmax attention (no FLA).
#
# The gfx1201 page fault (Memory access fault / page not present,
# PERMISSION_FAULTS:0x3) during the QLoRA backward is NON-DETERMINISTIC and not
# yet attributed to any config knob: the SAME config (same git SHA) both finishes
# 100 steps and crashes at step 10 across repeated runs (wandb-verified — see PR
# #101 and docs/GFX1201_RDNA4_TRAINING.md §6.1). Four root-cause theories
# (workers, GC threshold, hipBLASLt, LR) were each published then falsified.
# Current knobs are PRECAUTIONARY, not proven fixes:
#   TORCH_USE_HIPBLASLT=0           — rocBLAS fallback (HIPBLASLT=1 still crashed,
#                                     so this is precaution, not the cure)
#   PYTORCH_HIP_ALLOC_CONF=gc:0.8   — trims the backward peak; GC-off also crashed
# The real next step is `make diagnose-gfx1201-fault` (serialized kernels → names
# the faulting kernel) and the ablation matrix in §6.1 — NOT another knob flip.
# Every (re)launch is gated on `make gpu-preflight` (clean KFD + working fwd/bwd):
# a prior fault leaves the GPU dirty and the next run faults early on stale PTEs.

gpu-preflight:
	bash scripts/test_gpu_stack.sh --preflight

# Localize the backward page fault: run ~120 steps with serialized kernel launches
# so the async VM fault becomes synchronous and the traceback/dmesg name the exact
# faulting kernel (bnb NF4 dequant vs grad-ckpt recompute vs allocator).
diagnose-gfx1201-fault: gpu-preflight
	bash scripts/diagnose_gfx1201_fault.sh

download-gemma4-31b:
	uv run hf download google/gemma-4-31b-it

train-gemma4-31b-dry-run:
	uv run python scripts/train_sft.py --config configs/train-sft-gemma4-31b-qlora.env --dry-run

train-gemma4-31b-stage2: gpu-preflight
	mkdir -p outputs/sft-stage2-gemma4-31b
	nohup env TORCH_USE_HIPBLASLT=0 \
	  PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8 \
	  uv run --no-sync python scripts/train_sft.py \
	  --config configs/train-sft-stage2-gemma4-31b.env \
	  > outputs/sft-stage2-gemma4-31b/train.log 2>&1 &
	@echo "Training started. Monitor: tail -f outputs/sft-stage2-gemma4-31b/train.log"

prequant-gemma4-31b-l40s:
	@echo "Run on the L40S machine (needs 96GB+ RAM, CUDA GPU):"
	@echo "  python scripts/prequant_gemma4.py --output gemma-4-31b-nf4"
	@echo ""
	@echo "Then transfer back:"
	@echo "  make transfer-gemma4-31b-nf4 HOST=user@l40s-host"

transfer-gemma4-31b-nf4:
	mkdir -p models/gemma-4-31b-nf4
	rsync -avP "$(HOST):gemma-4-31b-nf4/" models/gemma-4-31b-nf4/

train-gemma4-31b-stage2-preq: gpu-preflight
	mkdir -p outputs/sft-stage2-gemma4-31b
	nohup env TORCH_USE_HIPBLASLT=0 \
	  PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8 \
	  TRAIN_BASE_MODEL=models/gemma-4-31b-nf4 \
	  TRAIN_PREQ=true \
	  uv run --no-sync python scripts/train_sft.py \
	  --config configs/train-sft-stage2-gemma4-31b.env \
	  > outputs/sft-stage2-gemma4-31b/train.log 2>&1 &
	@echo "Training started. Monitor: tail -f outputs/sft-stage2-gemma4-31b/train.log"

# Train from the community pre-quantized unsloth bnb-4bit checkpoint (~19 GB NF4).
# Skips the L40S prequant + rsync step entirely: weights download already 4-bit,
# so there is no ~62 GB BF16 CPU staging at load (the R9700's actual blocker).
# TRAIN_PREQ=true is required — it tells train_sft.py to read the embedded
# quantization_config instead of building a live BitsAndBytesConfig (which would
# double-quantize the already-quantized checkpoint).
train-gemma4-31b-stage2-unsloth: gpu-preflight
	mkdir -p outputs/sft-stage2-gemma4-31b
	nohup env TORCH_USE_HIPBLASLT=0 \
	  PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8 \
	  TRAIN_BASE_MODEL=unsloth/gemma-4-31B-it-unsloth-bnb-4bit \
	  TRAIN_PREQ=true \
	  TRAIN_HF_REPO=ulises-c/SocratesLM-31B-stage2b-QLoRA \
	  TRAIN_HF_PUSH_EVERY=50 \
	  uv run --no-sync python scripts/train_sft.py \
	  --config configs/train-sft-stage2-gemma4-31b.env \
	  > outputs/sft-stage2-gemma4-31b/train.log 2>&1 &
	@echo "Training started. Monitor: tail -f outputs/sft-stage2-gemma4-31b/train.log"

# Train 100 steps on the same unsloth path as the real run to produce an adapter
# for the EOS gate. Uses a separate output dir so it never collides with the real run.
train-gemma4-31b-eos-gate:
	mkdir -p outputs/eos-gate-gemma4-31b
	setsid env TORCH_USE_HIPBLASLT=0 \
	  PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8 \
	  TRAIN_BASE_MODEL=unsloth/gemma-4-31B-it-unsloth-bnb-4bit \
	  TRAIN_PREQ=true \
	  TRAIN_MAX_STEPS=100 \
	  TRAIN_SAVE_STEPS=100 \
	  TRAIN_OUTPUT_DIR=outputs/eos-gate-gemma4-31b \
	  uv run --no-sync python scripts/train_sft.py \
	  --config configs/train-sft-stage2-gemma4-31b.env \
	  > outputs/eos-gate-gemma4-31b/train.log 2>&1 &
	@echo "EOS-gate checkpoint training started (~30 min)."
	@echo "Monitor: tail -f outputs/eos-gate-gemma4-31b/train.log"
	@echo "When done, run: make eos-gate-gemma4-31b"

eos-gate-gemma4-31b:
	env TORCH_USE_HIPBLASLT=0 \
	  PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8 \
	  TRAIN_BASE_MODEL=unsloth/gemma-4-31B-it-unsloth-bnb-4bit \
	  TRAIN_PREQ=true \
	  TRAIN_METHOD=qlora \
	  TRAIN_BF16=true \
	  uv run --no-sync python scripts/eos_gate.py \
	  --config configs/train-sft-stage2-gemma4-31b.env \
	  --adapter outputs/eos-gate-gemma4-31b/final

# De-risk the FA2 bet: profile where the real Stage 2 step spends its ~70s.
# Mirrors the stage2-unsloth env so the profiled step matches the real run.
# Prints a kernel self-time table + attention-vs-gemm/dequant bucket summary.
# If attention dominates → patching the flash-attn Triton backward is worth it;
# if NF4 dequant/GEMM dominates → FA2 buys little. ~6 min (6 steps).
profile-gemma4-31b:
	env TORCH_USE_HIPBLASLT=1 \
	  TRAIN_BASE_MODEL=unsloth/gemma-4-31B-it-unsloth-bnb-4bit \
	  TRAIN_PREQ=true \
	  uv run --no-sync python scripts/profile_train_step.py \
	  --config configs/train-sft-stage2-gemma4-31b.env

# ── Tournament ────────────────────────────────────────────────────────────────

tournament-help:
	@echo ""
	@echo "Tournament — multi-model elimination benchmark"
	@echo "=============================================="
	@echo ""
	@echo "RUNNING"
	@echo "  make tournament                    Run one round (n=50, fusion, thinking OFF)"
	@echo "  make tournament-think              Run one round (n=50, fusion, thinking budget=4096)"
	@echo "  make tournament-warmup             Smoke test — n=5, thinking OFF"
	@echo "  make tournament-warmup-think       Smoke test — n=5, thinking budget=4096 (verify thinking_content in dialogues)"
	@echo "  make tournament N=<n>              Custom dialogue count, e.g. make tournament N=20"
	@echo "  make tournament-finalize           Run the 3 survivors to full n=681"
	@echo ""
	@echo "LEADERBOARD"
	@echo "  make tournament-status             Print leaderboard + detailed metrics table"
	@echo ""
	@echo "ELIMINATION"
	@echo "  make tournament-eliminate          Drop 1 worst-scoring model"
	@echo "  make tournament-eliminate N=2      Drop 2 worst (floor: 3 finalists always remain)"
	@echo ""
	@echo "ARCHIVE / RESTORE"
	@echo "  make tournament-archive            Save current run to archive/<run_id>/ and reset"
	@echo "                                     Re-archiving the same run ID overwrites (appends rounds)"
	@echo "  make tournament-restore            List all archived runs with ID, date, round, TB"
	@echo "  make tournament-restore ID=<id>    Restore run <id>; auto-archives current run first"
	@echo ""
	@echo "SETUP"
	@echo "  make tournament-download           Download any missing model weights via hf CLI"
	@echo "  make tournament-reset CONFIRM=1    Wipe everything (state + round dirs) — no undo"
	@echo ""
	@echo "TYPICAL WORKFLOW"
	@echo "  1. make tournament-warmup          # verify all models boot"
	@echo "  2. make tournament                 # run no-think round 1"
	@echo "  3. make tournament-status          # review results"
	@echo "  4. make tournament-archive         # save no-think run (gets a run_id)"
	@echo "  5. make tournament-think           # run thinking=4096 round 1"
	@echo "  6. make tournament-status"
	@echo "  7. make tournament-archive"
	@echo "  8. make tournament-restore ID=<no-think-id>  # switch back if needed"
	@echo ""

tournament-warmup:
	uv run tournament run --n 5 --unified

tournament-warmup-think:
	uv run tournament run --n 5 --unified --thinking-budget 4096

tournament:
	uv run tournament run --unified $(if $(N),--n $(N),)

tournament-think:
	uv run tournament run --unified --thinking-budget 4096 $(if $(N),--n $(N),)

tournament-status:
	uv run tournament status

tournament-eliminate:
	uv run tournament eliminate $(N)

tournament-finalize:
	uv run tournament finalize --unified

tournament-archive:
	uv run tournament archive

tournament-restore:
	@if [ -z "$(ID)" ]; then \
	  uv run tournament restore; \
	else \
	  uv run tournament restore $(ID); \
	fi

tournament-reset:
	@if [ -z "$(CONFIRM)" ]; then \
	  echo "This wipes results/tournament/ entirely. Re-run with CONFIRM=1."; \
	else \
	  uv run tournament reset --confirm; \
	fi

tournament-download:
	uv run tournament download

# ── WAVE HPC ──────────────────────────────────────────────────────────────────

slurm:
	bash scripts/slurm/submit_wave.sh
