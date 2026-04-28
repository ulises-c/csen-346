# Mac Mini MLX Setup

This doc covers running the KELE consultant via **MLX** (`mlx-lm`) on an Apple Silicon
Mac Mini, as an alternative to llama.cpp.

MLX is Apple's native ML framework for Apple Silicon. It uses the M-series unified memory
architecture more efficiently than llama.cpp and is typically **2–3× faster** for inference
on the same hardware.

---

## Why MLX over llama.cpp

| | llama.cpp (GGUF) | MLX (`mlx-lm`) |
|---|---|---|
| Inference speed (M4, ~9B) | ~15–20 tok/s | ~40–55 tok/s |
| Memory overhead | Higher — separate process allocator | Lower — native unified memory |
| KV cache format | FP16 | BF16, more efficient |
| Model format | GGUF | MLX safetensors (HuggingFace) |
| OpenAI-compatible API | `llama-server --port` | `mlx_lm.server --port` |
| Model availability | Excellent (Ollama registry + HF) | Good (`mlx-community/` org on HF) |

---

## Memory budget on M4 Mac Mini 16 GB

macOS takes ~4–5 GB, leaving ~11–12 GB for inference.

KV cache for Qwen3/Qwen2.5-9B (28 layers, 8 GQA heads, 128 head-dim) at various ctx sizes:

| ctx size | KV cache |
|---|---|
| 8 192 | ~0.95 GB |
| 16 384 | ~1.9 GB |

Approximate total memory per quant level:

| Model | Quant | Weights | + KV @ 16k | Total | 16 GB fits? |
|---|---|---|---|---|---|
| Qwen3.5-9B UD-Q6_K_XL | GGUF | ~9.5 GB | ~1.9 GB | ~11.4 GB | Borderline |
| Qwen3.5-9B UD-Q5_K_XL | GGUF | ~7.5 GB | ~1.9 GB | ~9.4 GB | Yes |
| Qwen3-8B | MLX 4-bit | ~4.8 GB | ~1.2 GB | ~6 GB | Comfortable |
| Qwen3-8B | MLX 8-bit | ~8.5 GB | ~1.2 GB | ~9.7 GB | Yes |
| Qwen2.5-7B | MLX 4-bit | ~4.3 GB | ~1.0 GB | ~5.3 GB | Comfortable |
| Qwen2.5-7B | MLX 8-bit | ~7.7 GB | ~1.0 GB | ~8.7 GB | Yes |

If llama.cpp hits swap at Q6 or Q5, the root causes are:
1. Weights + KV cache + macOS overhead approaches 16 GB
2. Other apps competing for memory

Dropping `--ctx-size` to `8192` (if dialogues are typically < 6k tokens) cuts the KV cache
in half and usually eliminates swap without changing the model.

---

## One-time setup (Mac Mini)

### 1. Install mlx-lm

MLX does not need Homebrew or brew-installed binaries — it is a Python package.

```bash
pip install mlx-lm
```

Verify:

```bash
python -m mlx_lm.server --help
```

### 2. Download a model

Models live in the `mlx-community/` organization on Hugging Face. No conversion needed.

```bash
# 4-bit Qwen3-8B (~4.8 GB) — best speed/quality tradeoff for 16 GB M4
huggingface-cli download mlx-community/Qwen3-8B-4bit

# 8-bit Qwen3-8B (~8.5 GB) — higher quality, still fits comfortably
huggingface-cli download mlx-community/Qwen3-8B-8bit

# 4-bit Qwen2.5-7B — if you need an exact Qwen2.5 series model
huggingface-cli download mlx-community/Qwen2.5-7B-Instruct-4bit
```

Models are cached in `~/.cache/huggingface/hub/` automatically.

### 3. Start the server

```bash
python -m mlx_lm.server \
    --model mlx-community/Qwen3-8B-4bit \
    --port 8080 \
    --max-kv-size 16384
```

Flags:

| Flag | Purpose |
|---|---|
| `--model` | HuggingFace repo ID or local path |
| `--port` | Listening port (matches `CONSULTANT_PORT`) |
| `--max-kv-size` | KV cache size in tokens (reduce to `8192` to save ~0.9 GB) |
| `--host 0.0.0.0` | Accept remote connections (LAN access) |

For LAN access from the host PC, add `--host 0.0.0.0`:

```bash
python -m mlx_lm.server \
    --model mlx-community/Qwen3-8B-4bit \
    --host 0.0.0.0 \
    --port 8080 \
    --max-kv-size 16384
```

### 4. macOS firewall

Same as llama.cpp — if the host PC can't reach port `8080`:

1. **System Settings → Network → Firewall → Options**
2. Confirm incoming connections on port `8080` are allowed

---

## Using with KELE eval

The MLX server exposes the same OpenAI-compatible `/v1` API as `llama-server`. No eval code
changes are needed — only the config changes.

Update `configs/consultants/m4-llamacpp.env` (or create a new `m4-mlx.env`):

```bash
CONSULTANT_API_KEY=no-key
CONSULTANT_BASE_URL=http://Ulisess-Mac-mini.local:8080/v1
CONSULTANT_MODEL_NAME=Qwen3-8B-4bit
CONSULTANT_NUM_CTX=16384
CONSULTANT_MAX_TOKENS=8192
CONSULTANT_DISABLE_THINKING=true
CONSULTANT_THINKING_BUDGET=0
```

Test from the host PC:

```bash
curl http://Ulisess-Mac-mini.local:8080/v1/models

curl http://Ulisess-Mac-mini.local:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3-8B-4bit","messages":[{"role":"user","content":"say hi"}],"max_tokens":5}'
```

---

## Swap troubleshooting

If you are still hitting swap with MLX:

1. **Reduce `--max-kv-size`** — try `8192` if your dialogues fit. Halves KV memory.
2. **Use 4-bit instead of 8-bit** — saves ~3–4 GB of weights.
3. **Check `memory_pressure`** in Terminal to see actual memory usage:
   ```bash
   memory_pressure
   ```
4. **Quit other apps** — browsers, IDEs, and other processes compete for the same
   unified memory pool.

---

## Keeping the server alive across sleep

```bash
# Prevent sleep while serving (attach to the server process):
caffeinate -s -w $(pgrep -f "mlx_lm.server") &
```

Or use the `caffeinate` wrapper when launching:

```bash
caffeinate -s python -m mlx_lm.server \
    --model mlx-community/Qwen3-8B-4bit \
    --host 0.0.0.0 \
    --port 8080 \
    --max-kv-size 16384
```
