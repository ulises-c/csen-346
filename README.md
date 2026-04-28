# csen-346
Natural Language Processing - CSEN 346 SCU

This project reproduces and extends **KELE**, a multi-agent framework for structured Socratic teaching with LLMs.

- **Paper:** Peng et al., "KELE: A Multi-Agent Framework for Structured Socratic Teaching with Large Language Models", *Findings of EMNLP 2025* — [aclanthology.org/2025.findings-emnlp.888](https://aclanthology.org/2025.findings-emnlp.888/)
- **Original repository:** https://github.com/yuanpan1020/KELE
- **SocratTeachLLM model (our fork):** https://huggingface.co/ulises-c/SocratTeachLLM
- **SocratTeachLLM model (original):** https://huggingface.co/yuanpan/SocratTeachLLM

## Hugging Face

The teacher agent model ([SocratTeachLLM](https://huggingface.co/ulises-c/SocratTeachLLM)) is hosted on Hugging Face. Our fork lives at `ulises-c/SocratTeachLLM` and is the model used by default in this project.

**What we use HF for:**
- Downloading SocratTeachLLM for local serving (`hf download ulises-c/SocratTeachLLM`)
- Free inference via the [Inference Providers API](https://huggingface.co/docs/inference-providers/en/index) (rate-limited, no local GPU needed for quick tests)
- Model versioning — experiment checkpoints and variants can be pushed as separate revisions

**Useful HF CLI commands:**
```bash
# Download the model for local serving
hf download ulises-c/SocratTeachLLM --local-dir ~/hf_models/SocratTeachLLM

# Check your HF auth
hf auth whoami
```

## Repository Structure

| Directory | Description |
| --- | --- |
| `src/` | Our reproduction and extensions |
| `references/` | Immutable reference data — do not modify or import directly (see [`references/README.md`](references/README.md)) |
| `references/KELE/` | KELE baseline — translated source, dataset, and attribution (see [`references/KELE/ATTRIBUTION.md`](references/KELE/ATTRIBUTION.md)) |
| `references/requirements/` | Course guidelines |
| `deliverables/` | Course deliverables |
| `docs/` | Working plans and internal documentation |
## Mirroring to the org repo

[ulises-c/csen-346](https://github.com/ulises-c/csen-346) is the primary development repo. It is mirrored to [SCU-CSEN346/KELE](https://github.com/SCU-CSEN346/KELE) via a dual-push remote — every `git push` publishes to both simultaneously, preserving full git history.

### How it works

`origin` is configured with two push URLs:

```
origin  git@github.com:ulises-c/csen-346.git  (fetch)
origin  git@github.com:ulises-c/csen-346.git  (push)
origin  git@github.com:SCU-CSEN346/KELE.git   (push)
```

A normal `git push` hits both. Fetch and pull still only come from `ulises-c/csen-346`.

### Setup (first time, per machine)

If you cloned from `SCU-CSEN346/KELE`, re-point your fetch remote and add the second push URL:

```bash
git remote set-url origin git@github.com:ulises-c/csen-346.git
git remote set-url --add --push origin git@github.com:ulises-c/csen-346.git
git remote set-url --add --push origin git@github.com:SCU-CSEN346/KELE.git
```

If you cloned from `ulises-c/csen-346`, only the two push URLs are needed:

```bash
git remote set-url --add --push origin git@github.com:ulises-c/csen-346.git
git remote set-url --add --push origin git@github.com:SCU-CSEN346/KELE.git
```

Verify with `git remote -v` — you should see one fetch URL and two push URLs.

## Dependencies

[Poetry](https://python-poetry.org/) - Python package manager

## Python Environment

This repo targets Python `3.12` and uses Poetry for dependency management.

### Initial setup

```bash
poetry env use python3.12
poetry install --with dev
```

If you want to confirm the virtualenv Poetry is using:

```bash
poetry env info
poetry run python -V
poetry run which pytest
```

### Install git hooks

```bash
make install-hooks
```

This copies `hooks/pre-commit` into `.git/hooks/` so that ruff (format + lint) and pytest run automatically before every commit, mirroring the CI pipeline.

### Torch note

`torch` is intentionally not declared in `pyproject.toml` because the CUDA wheel installation is environment-specific. After `poetry install`, install the appropriate PyTorch build manually for your machine.

Example for CUDA 12.6:

```bash
poetry run pip install --index-url https://download.pytorch.org/whl/cu126 "torch>=2.10.0"
```

## Common Poetry Commands

Poetry now exposes the main repo entry points directly:

- `poetry run kele`
- `poetry run kele-eval`
- `poetry run serve-teacher`

These map to the main modules in `src/project/`.

### Run tests

Run the offline/default test suite:

```bash
poetry run pytest
```

Show skip reasons too:

```bash
poetry run pytest -rs
```

Run a single test file:

```bash
poetry run pytest tests/test_metrics.py
```

Some tests are conditional and will skip if the required dependency or runtime is not present:

- `tests/test_consultant.py` needs `openai` and relevant API credentials
- `tests/test_metrics.py` / `tests/test_evaluate.py` need `rouge-score` and `sacrebleu`
- `tests/test_serve_teacher.py` needs `fastapi`

### Run the KELE CLI

Show CLI help:

```bash
poetry run kele --help
```

Quick smoke test on a few dialogues:

```bash
poetry run kele --experiment baseline test --n 3 --output results/test
```

Run a full evaluation:

```bash
poetry run kele --experiment baseline evaluate --output results/baseline
```

Run a partial evaluation (useful for smoke-testing the pipeline):

```bash
poetry run kele --experiment baseline evaluate --limit 5
```

Start a **fresh run** — archives any previous `run_config.json` and `metrics_summary.json`
into a timestamped `run{N}_{timestamp}/` subfolder, then clears `dialogues/` and `progress.log`:

```bash
poetry run kele --experiment baseline evaluate --limit 5 --new
```

After several `--new` runs the output directory looks like:

```
results/baseline/
├── run1_2026-04-14T10-14-09-07-00/   ← first run, archived
│   ├── run_config.json
│   └── metrics_summary.json
├── run2_2026-04-26T09-03-22-07-00/   ← second run, archived
│   └── ...
├── dialogues/                         ← current run (per-dialogue JSONs)
├── progress.log                       ← live progress (updated each dialogue)
├── run_config.json                    ← current run metadata
└── metrics_summary.json               ← current run metrics
```

Resume an interrupted run (skips already-saved dialogues automatically):

```bash
poetry run kele --experiment baseline evaluate
```

### Evaluate saved results

Recompute metrics for one run:

```bash
poetry run kele-eval results/baseline
```

Compare two runs side-by-side:

```bash
poetry run kele-eval --compare results/baseline results/gemma4
```

### Start the local teacher server

```bash
poetry run serve-teacher
```

With a custom local model path:

```bash
TEACHER_LOCAL_PATH=~/hf_models/SocratTeachLLM poetry run serve-teacher
```

### Run online from your own machine

If you want a public HTTPS endpoint backed by your local RTX 3070, use the
online-serving helper plus a tunnel such as Cloudflare Tunnel or ngrok.

```bash
TEACHER_LOCAL_PATH=~/hf_models/SocratTeachLLM \
TEACHER_SERVER_API_KEY=replace-this-with-a-long-random-secret \
./scripts/serve_teacher_online.sh
```

Then point your tunnel at `http://127.0.0.1:8001`.

See [`scripts/ONLINE_SETUP.md`](scripts/ONLINE_SETUP.md) for the full public-serving flow.

### Run helper scripts

The repo also includes shell scripts under `scripts/` for multi-process workflows such as model serving and long evaluation runs.

Examples:

```bash
./scripts/run_eval.sh baseline           # full eval run
./scripts/run_eval.sh baseline --limit 5 # smoke test
./scripts/serve_socratteachllm.sh
./scripts/serve_consultant.sh
./scripts/serve_both.sh
```

#### AMD Linux + Mac Mini setup

To run with a local AMD GPU teacher and a Mac Mini llama.cpp consultant:

```bash
# On the Mac Mini (once per session):
set -a && source configs/local-mac-m4.env && set +a
./scripts/serve_consultant_llamacpp.sh

# On the host PC:
./scripts/eval_amd_mac.sh              # full run (resumes if interrupted)
./scripts/eval_amd_mac.sh --limit 5   # smoke test (resumes previous)
./scripts/eval_amd_mac.sh --limit 5 --new  # fresh smoke test, archives old results
```

See [`scripts/MAC_MINI_SETUP.md`](scripts/MAC_MINI_SETUP.md) for first-time Mac Mini setup.

### Run on SCU WAVE nodes

If you have access to SCU WAVE GPU nodes, use the included cluster config and Slurm job:

```bash
sbatch scripts/slurm/wave_eval.slurm
```

See [`scripts/WAVE_SETUP.md`](scripts/WAVE_SETUP.md) for the full setup and model-path overrides.

## Configuration

Runtime settings are loaded from:

- `configs/<experiment>.env` for experiment-specific values
- `.env` for shared secrets and local overrides

Typical usage pattern:

```bash
poetry run kele --experiment baseline test --n 3 --output results/test
```

This loads `configs/baseline.env` first, then fills in any missing values from `.env`.
