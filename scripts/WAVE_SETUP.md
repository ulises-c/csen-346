# WAVE Setup

## Class policy

> Follow the WAVE documentation to work on our HPC cluster. You all should
> have accounts with access to:
>
> | Path | Purpose |
> |---|---|
> | `/WAVE/projects/CSEN-346-Sp26` | Persistent project files (code, models) |
> | `/WAVE/scratch/CSEN-346-Sp26` | Fast scratch space (large temp files) |
>
> - **Class quota:** 400 GB total
> - **Per-person budget:** ≤ 30 GB — be mindful of shared space
> - **GPUs:** request 1–2 at a time; don't hold idle allocations
> - **End of quarter:** all WAVE data will be erased — make sure final
>   model weights are on Hugging Face and all code is committed to GitHub
>   before the deadline

Store shared model downloads in the project space so the whole team
shares one copy instead of everyone downloading ~40 GB separately:

```bash
export HF_HOME=/WAVE/projects/CSEN-346-Sp26/hf_models
```

This is the default used by `wave_setup.sh --models` and `wave_eval.slurm`.

---

This repo can run on SCU WAVE GPU nodes. The strategy is:

1. Request a node with **2 GPUs**
2. Serve the teacher (SocratTeachLLM) on GPU 0
3. Serve the consultant (Qwen3.5-9B) on GPU 1
4. Run the KELE evaluation against `localhost`

Both models run in FP16 (`float16`) on V100 nodes — V100s have compute
capability 7.0 and do not support BF16 (requires 8.0+). On newer nodes
(L40S, A100) you can override: `export TEACHER_DTYPE=bfloat16 CONSULTANT_DTYPE=bfloat16`.

## WAVE GPU nodes you might land on

| Node | GPU | VRAM | Notes |
|---|---|---|---|
| gpu[01-04] | Tesla V100 | 32 GB × 2 | Standard GPU nodes; most likely allocation |
| oignat01 | L40S | 48 GB × 4 | Course/condo node; more headroom |
| bio01–03 | A16 | 16 GB × 8 | Biology dept — avoid unless you have access |
| amd01 | MI100 | — | **AMD, not CUDA — do not use** |

The Slurm script requests `--gres=gpu:2` (any type). If you land on a V100 node you're fine;
the defaults are tuned for 32 GB. See "Targeting a specific node" below if you need to pin.

## One-time setup on WAVE

Clone the repo on the **login node**, then run the setup script (it handles
Poetry, PyTorch, vLLM, and model downloads all in one shot):

```bash
git clone <your-repo-url>
cd csen-346
bash scripts/wave_setup.sh --models
```

`--models` downloads SocratTeachLLM and Qwen3.5-9B to `~/hf_models/`.
Omit it if you want to install deps first and download models separately.

> **Python module:** The setup script automatically runs
> `module load Python/3.12.3-GCCcore-14.2.0` if Python 3.12 isn't already
> in your PATH. The SLURM job script does the same on the compute node.
>
> **Broken PyTorch module:** `PyTorch/2.9.1-CUDA-13.0` references
> `CUDA/13.0.0` which does not exist on the cluster — avoid it.
> The setup script installs PyTorch directly via `pip` from the `cu128`
> wheel index (WAVE ships CUDA 12.x), bypassing it entirely.
> (`PyTorch/2.10.0-Python-3.12` and `PyTorch/2.11.0-Python-3.12` are also
> available and working, but we install via pip to control the exact version
> vLLM is paired with.)

<details>
<summary>Manual steps (if you prefer not to use the script)</summary>

```bash
# 1. Load Python 3.12
module load Python/3.12.3-GCCcore-14.2.0

# 2. Install Poetry if missing
export PATH="$HOME/.local/bin:$PATH"
curl -sSL https://install.python-poetry.org | python3.12 -

# 3. Project deps
poetry env use python3.12
poetry install --with dev

# 4. PyTorch — use cu128, NOT cu126 (WAVE CUDA is 12.x, not 12.6)
poetry run pip install \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# 5. vLLM
poetry run pip install "vllm>=0.7"

# 6. Models (login node only — compute nodes may lack internet)
export HF_HOME=/WAVE/projects/CSEN-346-Sp26/hf_models
mkdir -p "$HF_HOME"
poetry run hf download ulises-c/SocratTeachLLM --local-dir "$HF_HOME/SocratTeachLLM"
poetry run hf download Qwen/Qwen3.5-9B --local-dir "$HF_HOME/Qwen3.5-9B"
```

</details>

## Submitting the job

```bash
make slurm        # git pull + sbatch + print status commands
```

Or manually:

```bash
mkdir -p logs
JOB=$(sbatch scripts/slurm/wave_eval.slurm | awk '{print $NF}')
printf "[%s] Job %s submitted\n  Status : squeue -u \$USER\n  Logs   : tail -f logs/*%s*/slurm.out\n  Cancel : scancel %s\n" \
    "$(date '+%Y-%m-%d %H:%M:%S')" "$JOB" "$JOB" "$JOB" | tee logs/job-${JOB}.submitted
```

This returns immediately — no blocking wait. A `logs/job-<JOBID>.submitted`
file is written instantly with submission time and the commands you need.
Once the job starts a timestamped run directory is created:

```
logs/
  2026-04-20T14-30-00-<JOBID>/   ← everything for this run in one place
    slurm.out                    ← SLURM's captured stdout (written by SLURM)
    slurm.err                    ← SLURM stderr (usually empty)
    run.log                      ← tee of all job output (same content as slurm.out)
    vllm_teacher.log             ← teacher server output
    vllm_consultant.log          ← consultant server output
    job.submitted                ← submission metadata (written at sbatch time)
```

> **How it works:** `submit_wave.sh` pre-creates `logs/<JOBID>/` so SLURM can
> write its output files there. When the job starts it renames the dir to add
> the ISO timestamp prefix — Linux keeps open file descriptors alive across
> renames so the log files keep writing without interruption.

`run.log` contains:

- **Start time** — printed by the job at launch
- **GPU info** — `nvidia-smi` output
- **Server readiness** — port health checks
- **Eval progress** — live dialogue output
- **End time** — printed on completion

To tail live (the glob works before and after the timestamp rename):

```bash
tail -f logs/*<JOBID>*/slurm.out                  # full SLURM capture
tail -f logs/*<JOBID>*/run.log                    # same, tee'd copy
tail -f logs/*<JOBID>*/vllm_teacher.log           # teacher server only
tail -f logs/*<JOBID>*/vllm_consultant.log        # consultant server only
```

If you already submitted and missed the job ID: `squeue -u $USER`

That job:

- requests 2 GPUs (any NVIDIA node)
- starts SocratTeachLLM on `127.0.0.1:8001` (GPU 0)
- starts Qwen3.5-9B on `127.0.0.1:8002` (GPU 1)
- waits for both servers to be ready
- runs `./scripts/run_eval.sh wave`

## Monitoring a running job

Replace `<JOBID>` with your actual job ID (e.g. `892952`). Never include the `<>`.

### Queue / status

```bash
squeue -u $USER                        # all your jobs
squeue -j <JOBID>                      # one job
squeue -p gpu                          # everyone on the gpu partition
watch -n 10 squeue -u $USER            # auto-refresh every 10s
```

ST column meanings: `PD` = pending, `R` = running, `CG` = completing, `F` = failed.

### GPU utilization (must run on compute node)

`nvidia-smi` is not available on the login node — use `srun` to reach the compute node your job is already on:

```bash
# One-shot snapshot
srun --jobid=<JOBID> nvidia-smi

# Watch live (refresh every 2s)
srun --jobid=<JOBID> --pty watch -n 2 nvidia-smi
```

### Logs

```bash
tail -f logs/*<JOBID>*/slurm.out          # all job output (servers + eval)
tail -f logs/*<JOBID>*/run.log            # same, tee'd copy
tail -f logs/*<JOBID>*/vllm_teacher.log   # teacher server only
tail -f logs/*<JOBID>*/vllm_consultant.log
```

### Eval progress

```bash
grep "complete" logs/*<JOBID>*/slurm.out  # all progress lines at once
tail -f logs/*<JOBID>*/slurm.out | grep "complete"  # live
```

### Disk usage

```bash
quota -s                                  # your home dir quota
du -sh /WAVE/projects/CSEN-346-Sp26/*     # project space breakdown
du -sh /WAVE/scratch/CSEN-346-Sp26/*      # scratch space breakdown
```

### Cancel / history

```bash
scancel <JOBID>                           # cancel a running or pending job
sacct -j <JOBID> --format=JobID,State,Elapsed,ExitCode   # post-run summary
```

## Targeting a specific node type

```bash
# Pin to oignat01 (L40S)
sbatch --nodelist=oignat01 scripts/slurm/wave_eval.slurm

# Exclude AMD nodes explicitly (good habit if you're not specifying a node)
sbatch --exclude=amd01 scripts/slurm/wave_eval.slurm

# Override model paths if you stored models elsewhere
sbatch \
  --export=ALL,TEACHER_MODEL_PATH=$SCRATCH/models/SocratTeachLLM,CONSULTANT_MODEL_PATH=$SCRATCH/models/Qwen3.5-9B \
  scripts/slurm/wave_eval.slurm
```

## Interactive session (for debugging)

```bash
srun --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=96G --gres=gpu:2 --time=02:00:00 --pty bash
```

Then in that shell:

```bash
# Check what GPU you got
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd ~/csen-346

# Start teacher on GPU 0
export TEACHER_HOST=127.0.0.1 TEACHER_PORT=8001
export TEACHER_MODEL_PATH=~/hf_models/SocratTeachLLM
CUDA_VISIBLE_DEVICES=0 ./scripts/serve_socratteachllm.sh &

# In a second pane / after teacher is ready:
export CONSULTANT_HOST=127.0.0.1 CONSULTANT_PORT=8002
export CONSULTANT_LOCAL_MODEL_NAME=Qwen3.5-9B
export CONSULTANT_MODEL_PATH=~/hf_models/Qwen3.5-9B
CUDA_VISIBLE_DEVICES=1 ./scripts/serve_consultant.sh &

# Then run:
./scripts/run_eval.sh wave
```

## GPU memory budget (per GPU, BF16)

| Model | Role | VRAM | GPU utilization setting |
|---|---|---|---|
| SocratTeachLLM 9.4B | Teacher | ~19 GB | 0.85 (27 GB on V100, 41 GB on L40S) | auto: float16 + --enforce-eager on V100 (CC 7.0); bfloat16 on L40S/A100+ (CC ≥ 8.0) |
| Qwen3.5-9B | Consultant | ~17 GB | 0.85 (27 GB on V100, 41 GB on L40S) | auto: float16 + --enforce-eager on V100 (CC 7.0); bfloat16 on L40S/A100+ (CC ≥ 8.0) |

Each model is on its own GPU, so V100 32GB nodes are fine.

## Notes

- Model servers bind to `127.0.0.1` — traffic stays private to the job.
- The Slurm script kills both servers automatically when the job ends.
- Full eval (681 dialogues) estimated time: **~34 hrs on V100, ~8–12 hrs on L40S/A100**.
- If a single GPU has enough VRAM for both models (e.g. you get an L40S 48GB and
  only need one), you can simplify to `--gres=gpu:1` and remove the `CUDA_VISIBLE_DEVICES`
  pinning from the slurm script.

## V100 (CC 7.0) runtime notes

Observed behavior from a real run on gpu03. Applies to any CC < 8.0 node.

### Eager mode vs graph mode

By default vLLM compiles the model's forward pass into a **CUDA graph** — a pre-recorded
sequence of GPU operations that replays with zero Python overhead. This is "graph mode"
and is significantly faster.

`--enforce-eager` disables that and runs in **eager mode**: every operation executes
one at a time through Python. On V100 (CC 7.0) CUDA graph capture is unreliable, so
eager mode is required. The per-call Python overhead is one of the main reasons V100
inference is slower than newer GPUs.

### What works differently vs newer GPUs

| Behavior | V100 (CC 7.0) | L40S / A100 / RTX 5090 (CC ≥ 8.0) |
|---|---|---|
| dtype | `float16` (auto-selected) | `bfloat16` |
| CUDA graphs | Disabled (`--enforce-eager`) | Enabled |
| FlashAttention 2 | ❌ Not supported — falls back to Triton attention | ✅ |
| Triton attention kernel | ✅ Used for both models | Not needed |

### Expected log messages on V100

These appear at startup and are **normal — not bugs**:

```
ERROR Cannot use FA version 2... FA2 is only supported on devices with compute capability >= 8
INFO  Using TRITON_ATTN attention backend
```
vLLM logs this at ERROR level but it's just informational — FA2 is unavailable so it
falls back to the Triton kernel, which works fine.

```
WARNING Using a slow tokenizer. This might cause a significant slowdown.
```
SocratTeachLLM (ChatGLM) has no fast Rust tokenizer. Python tokenization runs on CPU
before every call. Adds a small but real overhead per turn.

```
WARNING Default vLLM sampling parameters overridden by generation_config.json:
        {'temperature': 0.8, 'top_p': 0.8}
```
SocratTeachLLM's own `generation_config.json` sets sampling params. This is intentional
— the model is not using greedy decoding.

### Qwen3.5-9B is a Mamba hybrid

Qwen3.5-9B uses a mixed attention + state-space (Mamba) architecture, which causes:

- **Slow warmup** — `init engine took 36.92 seconds` vs 3.24s for the teacher. Expected.
- **FLA kernel** — vLLM uses `Triton/FLA GDN prefill kernel` for Mamba layers.
- **Shape warning during warmup** — `UserWarning: seq_len (16) < num_heads (32)` appears
  during profiling. Non-fatal, does not affect inference correctness.

### Why eval is slow on V100

Each dialogue makes ~10 serial LLM calls (5 turns × consultant + teacher). On V100:

- No CUDA graph batching (eager mode means per-call Python overhead)
- ~25 tokens/s generation throughput per request
- Slow tokenizer on teacher adds CPU time per call

Result: **~3 min/dialogue → ~34 hrs for 681 dialogues**. Within the 48h partition limit.
The `gpu` partition default is 48h (`sinfo -p gpu -o "%.P %.l"` confirms `2-00:00:00`).

### Observed eval times by setup

| Hardware | Consultant | Eval time | Notes |
|---|---|---|---|
| V100 32GB × 2 (WAVE gpu[01-04]) | Qwen3.5-9B local (GPU 1) | ~34 hrs | float16, eager mode, serial calls |
| RTX 5090 32GB × 1 | GPT-4o-mini API | ~5 hrs | bfloat16, CUDA graphs; 5090 couldn't fit both models in 32GB VRAM, so consultant runs via API |

The 5090 is ~7× faster despite running the consultant over the network — CUDA graphs +
bfloat16 + faster silicon matter more than the API round-trip latency for short consultant
prompts. Both setups produce the same eval results since the teacher model is identical.
