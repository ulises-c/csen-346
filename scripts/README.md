# scripts/

Shell entry points for serving models, running evals, and host setup. Python
utilities also live here, but most belong in `src/project/`. Retired one-off and
per-model eval shells have moved to [`archive/`](archive/) — see below.

Most of these are also exposed as `make` targets; run `make help` for the full list.

## Current run — Gemma 4 12B SFT-uplift PoC (NVIDIA)

The active run drives `src.project.kele` directly from the Makefile — no per-model
eval shell. This is the forward template: a new model is a Makefile target + a
`configs/<experiment>.env`, not a new copy of a 280-line script.

| Entry point | Purpose |
|-------------|---------|
| `make eval-gemma4-12b-base-{smoke,full}` | Eval the base 12B teacher (BERT consultant) |
| `make eval-gemma4-12b-sft-{smoke,full}`  | Eval the 1-epoch Socratic QLoRA SFT teacher |
| `make monitor-eval-gemma4-12b-{base,sft}` → `monitor_eval_gemma4_12b.sh` | Crash-crawl monitor: serving + adaptive power search + HF push |
| `serve_gemma4_12b.sh` / `serve_gemma4_12b_sft.sh` / `serve_gemma4_12b_mtp.sh` | Serve base / SFT / MTP-drafter variants |
| `convert_gemma4_12b_sft_to_gguf.sh` | Convert the SFT checkpoint to GGUF for llama.cpp |

## Local-model eval runner

`eval_llamacpp.sh` is the parameterized replacement for the 20 retired `eval_*.sh`
scripts: a single-server llama.cpp orchestrator (teacher = consultant on port 8080)
that boots the model's serve script, runs KELE, and compares against the
`results/baseline` gpt-4o run.

```
./scripts/eval_llamacpp.sh <model> <smoke|mini|full|n50> [flags]
```

| model | serve (default / `--think`) | alias | experiment |
|-------|------------------------------|-------|------------|
| `qwen27b`        | `serve_qwen27b_q5.sh` / `_think` | Qwen 27B Q5 | `qwen27b-local` |
| `qwen35b-a3b`    | `serve_qwen35b_a3b.sh` / `_think` | Qwen 35B A3B | `qwen35b-a3b-local` |
| `qwopus35b-a3b`  | `serve_qwopus35b_a3b_think.sh`    | Qwopus 35B A3B | `qwopus35b-a3b-local` |
| `gemma4-31b`     | `serve_gemma4_31b_q5.sh`          | Gemma 4 31B | `gemma4-31b-local` |
| `gemma4-26b-a4b` | `serve_gemma4_26b_a4b.sh`         | Gemma 4 26B A4B | `gemma4-26b-a4b-local` |

Modes: `smoke` (n=5) · `mini` (n=25) · `n50` (n=50) · `full` (n=681).
Flags: `--think` (Qwen only), `--nothink` (Qwen only), `--unified` (fusion single-call),
`--suffix NAME`, `--keep-server`, `--no-compare`.

The Makefile wraps the common cells: `eval-qwen27b-*`, `eval-qwen35b-a3b-*`,
`eval-gemma4-31b-*`, and the `*-fusion-*` smoke targets all call this runner.

## Serving (standalone)

`serve_*.sh` boot a single model on llama.cpp or vLLM. Beyond the Gemma 4 12B and
eval-runner entries above, common ones: `serve_consultant*.sh`, `serve_socratteachllm*.sh`,
`serve_teacher_{local,online}.sh`, `serve_both.sh`, `serve_dual_gpu.sh`,
`serve_demo_top_performer.sh`. `make help` lists them all with one-liners.

## Host setup

| Script | Host |
|--------|------|
| `l40s_setup.sh` | Dual L40S box ([`RTX5090_SETUP.md`](RTX5090_SETUP.md), [`WAVE_SETUP.md`](WAVE_SETUP.md)) |
| `amd_r9700_setup.sh` | AMD R9700 / gfx1201 (ROCm) |
| `mac_mini_setup.sh` | Mac Mini llama.cpp ([`MAC_MINI_SETUP.md`](MAC_MINI_SETUP.md), [`MLX_SETUP.md`](MLX_SETUP.md)) |
| `wave_setup.sh` | Wave HPC cluster |

`patch_fla_rocm.sh` and the `*gfx1201*` / `*rocblas*` scripts are RDNA4 fault
diagnostics — see the GFX1201 handoffs in `docs/`.

## Process management

`post_eval_shutdown.sh` (tear down vLLM servers after a run) ·
`run_eval.sh` (`make run-eval GPU=<config>`).

## archive/

[`archive/`](archive/) holds retired per-model eval shells (the 20 `eval_*.sh` that
`eval_llamacpp.sh` replaced) and dated one-off chains/backtests. Kept for provenance
and to reconstruct a past run's exact settings; not wired into the Makefile.
