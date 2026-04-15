# WAVE Setup

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

Clone the repo and install dependencies:

```bash
git clone <your-repo-url>
cd csen-346
poetry env use python3.12
poetry install --with dev
```

Install the CUDA PyTorch wheel separately (Poetry can't resolve the +cu126 local version):

```bash
poetry run pip install --index-url https://download.pytorch.org/whl/cu126 "torch>=2.10.0"
```

Install vLLM:

```bash
poetry run pip install "vllm>=0.7"
```

Download models (do this from the login node or a data-transfer node — compute nodes
may not have outbound internet access):

```bash
mkdir -p ~/hf_models
poetry run huggingface-cli download yuanpan/SocratTeachLLM --local-dir ~/hf_models/SocratTeachLLM
poetry run huggingface-cli download Qwen/Qwen3.5-9B --local-dir ~/hf_models/Qwen3.5-9B
```

## Submitting the job

```bash
sbatch scripts/slurm/wave_eval.slurm
```

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
