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

Both models run in BF16. Each gets a dedicated GPU, so a V100 32GB per model is enough.

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
mkdir -p logs
JOB=$(sbatch scripts/slurm/wave_eval.slurm | awk '{print $NF}')
echo "Job $JOB queued — waiting for log..."
until [ -f logs/slurm-${JOB}.out ]; do sleep 10; done
tail -f logs/slurm-${JOB}.out
```

`sbatch` prints the job ID on submit (`Submitted batch job 12345`); the
one-liner captures it, waits for the log file to appear (the job may be
queued for a few minutes before a GPU node is allocated), then tails it.

If you already submitted and missed the job ID: `squeue -u $USER`

To cancel a job: `scancel <JOBID>`

That job:

- requests 2 GPUs (any NVIDIA node)
- starts SocratTeachLLM on `127.0.0.1:8001` (GPU 0)
- starts Qwen3.5-9B on `127.0.0.1:8002` (GPU 1)
- waits for both servers to be ready
- runs `./scripts/run_eval.sh wave`

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
| SocratTeachLLM 9.4B | Teacher | ~19 GB | 0.85 (27 GB on V100, 41 GB on L40S) |
| Qwen3.5-9B | Consultant | ~17 GB | 0.85 (27 GB on V100, 41 GB on L40S) |

Each model is on its own GPU, so V100 32GB nodes are fine.

## Notes

- Model servers bind to `127.0.0.1` — traffic stays private to the job.
- The Slurm script kills both servers automatically when the job ends.
- Full eval (681 dialogues) estimated time: 2–4 hrs on V100, 1.5–2.5 hrs on L40S.
- If a single GPU has enough VRAM for both models (e.g. you get an L40S 48GB and
  only need one), you can simplify to `--gres=gpu:1` and remove the `CUDA_VISIBLE_DEVICES`
  pinning from the slurm script.
