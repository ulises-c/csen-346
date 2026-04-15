# csen-346
Natural Language Processing - CSEN 346 SCU

This project reproduces and extends **KELE**, a multi-agent framework for structured Socratic teaching with LLMs.

- **Paper:** Peng et al., "KELE: A Multi-Agent Framework for Structured Socratic Teaching with Large Language Models", *Findings of EMNLP 2025* — [aclanthology.org/2025.findings-emnlp.888](https://aclanthology.org/2025.findings-emnlp.888/)
- **Original repository:** https://github.com/yuanpan1020/KELE
- **SocratTeachLLM model:** https://huggingface.co/yuanpan/SocratTeachLLM

## Repository Structure

| Directory | Description |
| --- | --- |
| `src/` | Our reproduction and extensions |
| `resources/KELE/` | KELE baseline — translated source, dataset, and attribution (see [`resources/KELE/ATTRIBUTION.md`](resources/KELE/ATTRIBUTION.md)) |
| `resources/requirements/` | Course guidelines |
| `resources/ideation/` | Topic research and project idea notes |
| `deliverables/` | Course deliverables |
| `PLAN.md` | Project plan with phases and deadlines |
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

See [`ONLINE_SETUP.md`](ONLINE_SETUP.md) for the full public-serving flow.

### Run helper scripts

The repo also includes shell scripts under `scripts/` for multi-process workflows such as model serving and long evaluation runs.

Examples:

```bash
./scripts/run_eval.sh baseline
./scripts/serve_socratteachllm.sh
./scripts/serve_consultant.sh
./scripts/serve_both.sh
```

### Run on SCU WAVE nodes

If you have access to SCU WAVE GPU nodes, use the included cluster config and Slurm job:

```bash
sbatch scripts/slurm/wave_eval.slurm
```

See [`WAVE_SETUP.md`](WAVE_SETUP.md) for the full setup and model-path overrides.

## Configuration

Runtime settings are loaded from:

- `configs/<experiment>.env` for experiment-specific values
- `.env` for shared secrets and local overrides

Typical usage pattern:

```bash
poetry run kele --experiment baseline test --n 3 --output results/test
```

This loads `configs/baseline.env` first, then fills in any missing values from `.env`.
