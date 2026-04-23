# GPU Support Reference

Hardware tested across this project and the constraints each imposes on the serving stack.

## Quick reference

| GPU | VRAM | Arch | CC | Vendor | dtype | enforce-eager | vLLM | Serving path |
|---|---|---|---|---|---|---|---|---|
| AMD R9700 AI PRO | 32 GB | RDNA 4 (gfx1201) | — | AMD/ROCm | bfloat16 | no | ⚠️ broken | FastAPI (HF Transformers) |
| NVIDIA RTX 5090 | 32 GB | Blackwell | 10.0 | NVIDIA/CUDA | bfloat16 | no | ✅ | vLLM |
| NVIDIA RTX 3090 Ti | 24 GB | Ampere | 8.6 | NVIDIA/CUDA | bfloat16 | no | ✅ | vLLM |
| NVIDIA L40S | 48 GB | Ada Lovelace | 8.9 | NVIDIA/CUDA | bfloat16 | no | ✅ | vLLM |
| NVIDIA Tesla V100 | 32 GB | Volta | 7.0 | NVIDIA/CUDA | float16 | **yes** | ✅ | vLLM |
| Apple Mac Mini M4 | unified | Apple Silicon | — | Metal | — | — | n/a | Ollama (consultant only) |
| NVIDIA RTX 3070 | 8 GB | Ampere | 8.6 | NVIDIA/CUDA | — | no | — | retired |

## Serving path decision

```
AMD GPU?
  └─ yes → serve_teacher_local.sh  (FastAPI / HF Transformers)
  └─ no  → serve_socratteachllm.sh (vLLM)

NVIDIA CC < 8.0 (V100)?
  └─ scripts auto-detect and pass --enforce-eager + float16
  └─ no action needed; serve_socratteachllm.sh handles it
```

## Per-GPU notes

### AMD R9700 AI PRO (gfx1201 / RDNA 4)

Use **`serve_teacher_local.sh`** — do not use `serve_socratteachllm.sh` (vLLM).

vLLM support for gfx1201 is incomplete upstream as of April 2026:
- FP8 WMMA silently falls back to FP32, roughly halving throughput
- Container startup fails (amdsmi / HIP / `torch.cuda.device_count()` bugs)
- Community workarounds exist but are not merged upstream

Track: https://github.com/vllm-project/vllm/issues/28649

HF Transformers on ROCm works out of the box. The scripts apply three
compatibility patches for SocratTeachLLM (GLM4-9B) under transformers ≥ 5.x:
- `all_tied_weights_keys` missing at load time
- `DynamicCache` not subscriptable (layer-based format in 5.x)
- `apply_chat_template` returns `BatchEncoding`, not a raw tensor

### NVIDIA RTX 5090 / RTX 3090 Ti / L40S (CC ≥ 8.0)

Use **`serve_socratteachllm.sh`** (vLLM). bfloat16 and CUDA graphs are fully
supported. No flags needed beyond what the script sets automatically.

For single-GPU hosts running both models, use `serve_both.sh`.
For dual-GPU hosts, use `serve_dual_gpu.sh` (teacher → GPU 0, consultant → GPU 1).

### NVIDIA Tesla V100 (CC 7.0 — WAVE HPC)

Use **`serve_socratteachllm.sh`** (vLLM). The script auto-detects CC < 8.0
and adds `--enforce-eager --dtype float16`. No manual override needed.

V100 does not support bfloat16 (requires CC ≥ 8.0). CUDA graphs are also
unreliable on Volta under vLLM, which is why `--enforce-eager` is required.

Expected throughput: ~34 hrs for a full 681-dialogue eval (vs ~8–12 hrs on L40S).

### WAVE HPC — other nodes to avoid

| Node | GPU | Reason |
|---|---|---|
| `bio01–03` | A16 16 GB × 8 | 16 GB per card — too small for 9B models without quantization |
| `amd01` | MI100 | AMD ROCm — not tested, no CUDA, avoid |

### Apple Mac Mini M4 (consultant only)

Runs the **consultant** model via Ollama over LAN. Not used for the teacher.
Current model: `qwen2.5:7b`. See `scripts/MAC_MINI_SETUP.md` for setup.
Config: `configs/R9700_Mac-M4.env`.

## Model VRAM footprint

At bfloat16 / float16 (no quantization):

| Model | Params | VRAM (weights only) | Notes |
|---|---|---|---|
| SocratTeachLLM (GLM4-9B) | 9.4B | ~19 GB | trust_remote_code required |
| Qwen3.5-9B | 9.5B | ~17 GB | consultant on NVIDIA setups |
| Qwen2.5:7b | 7.5B | ~14 GB | consultant via Ollama on Mac Mini |

Both teacher + consultant fit on a single V100 32 GB or L40S 48 GB with
`--gpu-memory-utilization 0.85`.
