# Handoff — R9700 gfx1201 EOS Gate Run

**Branch:** `feat/gfx1201-rdna4-qlora-fla-training`  
**Date:** 2026-05-31  
**Last commit:** `dbc61a2`

> **STATUS: root cause FOUND and FIXED (commit `dbc61a2`).** The EOS collapse was
> not undertraining and not the consultant eval-line residual — it was the
> training chat template masking the turn terminator out of the loss. See
> **§ Root cause identified & fixed** below. Next action is a single confirmation
> retrain on the R9700, not a multi-hour bet.

---

## What was attempted

100-step QLoRA EOS gate checkpoint on the R9700 (gfx1201, 32 GB) to verify that
the schema-drift fixes from PR #101 / PR #94 actually produce a model that
terminates on realistic inference prompts before the full Stage 2 run is launched.

Model: `unsloth/gemma-4-31B-it-unsloth-bnb-4bit`  
Config: `configs/train-sft-stage2-gemma4-31b.env`  
Data: `socrat-zh-sft` + `socrat-en-sft` (77 202 train / 8 578 eval)

---

## Environment issues encountered and fixed

### 1 — `make test-vllm` poisoned the venv

Running `make test-vllm` before training auto-installed `vllm==0.22.0`, which
pulled in `torchvision==0.26.0` (CUDA build). The CUDA torchvision registers a
fake `torchvision::nms` op at import time that doesn't exist in the ROCm torch,
breaking `peft` and `transformers` at startup:

```
RuntimeError: operator torchvision::nms does not exist
ModuleNotFoundError: Could not import module 'BloomPreTrainedModel'
```

**Fix:** `uv pip uninstall vllm torchvision`

**Going forward:** Never run `make test-vllm` before or during a training session
on this machine. The test auto-installs into the shared training venv and is
destructive. Either isolate it or add a guard to the Makefile.

### 2 — `expandable_segments:True` unsupported on HIP → GPU page fault at step 16

All training Makefile targets had:
```
PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,expandable_segments:True
```

HIP does not support `expandable_segments`. The runtime warned at startup
(`HIPAllocatorConfig.h:40`) and the training crashed with a GPU page fault
every time at step 16:

```
Memory access fault by GPU node-1 on address 0x7f78cbc00000.
Reason: Page not present or supervisor privilege.
```

**Fix (commit 860fc5f):** Stripped `expandable_segments:True` from all Makefile
targets. Note: the PR #101 comment that suggested this setting used
`PYTORCH_CUDA_ALLOC_CONF` (CUDA variable) — on ROCm the variable is
`PYTORCH_HIP_ALLOC_CONF` and `expandable_segments` is not a valid key.

### 3 — GPU KFD dirty state after each page fault

Every page fault on gfx1201 leaves the KFD in a dirty state; subsequent runs
fault at random steps until all Python processes are killed. Always run before
retrying:

```bash
rocm-smi --showpids | grep python && echo "GPU dirty — kill before retrying" || echo "GPU clean"
```

### 4 — `nohup` → `setsid` for clean process group management

Changed the `train-gemma4-31b-eos-gate` target to use `setsid` so the entire
process tree lives in its own process group. Kill cleanly with:
```bash
kill -9 -- -$PGID
```

### 5 — `eos_gate.py` crashed: `apply_chat_template` returns `BatchEncoding` in transformers 5.9

`tokenizer.apply_chat_template(..., return_tensors="pt")` returns a `BatchEncoding`
dict in transformers 5.9, not a raw tensor. `model.generate` received the dict
and raised `AttributeError` on `.shape[0]`.

**Fix:** Extract `enc["input_ids"]` when the return value is not a tensor.

---

## Training run outcome

After all environment fixes, the 100-step checkpoint trained successfully:

- **Steps:** 100/100 ✓
- **VRAM:** alloc=21.1 GB, reserved=29.1 GB / 31.9 GB total (stable throughout)
- **Step time:** ~70 s/step
- **hipBLASLt:** enabled (`TORCH_USE_HIPBLASLT=1`) — Gemma 4 uses standard
  softmax attention; no linear-attn delta-rule kernel, so hipBLASLt is safe
- **Adapter saved:** `outputs/eos-gate-gemma4-31b/final`

---

## EOS gate result: FAILED (pre-fix) — root cause since found & fixed (`dbc61a2`)

```
STEP 1  warm-up (trivial prompt)    → PASS   (terminates after 1 sentence)
STEP 2  EOS gate (realistic prompt) → FAILED (hits max_new_tokens, repetition loop)
```

> This was the failing run *before* the terminator-masking fix. Jump to
> **§ Root cause identified & fixed** for the resolution; the section below is the
> observed failure that led there.

### Failure output (truncated)

```
很好！你能想象一下，当光从空气进入水中时，它的速度会发生什么变化呢？
这对光的方向有什么影响？这对光的方向有什么影响？这对光的方向有什么
影响？[...×80+ repetitions, never emits EOS]
```

The first sentence is correct Socratic output. Repetition begins at token 2.
This is the same failure mode as the 5090 run documented in PR #94.

### Initial (incorrect) hypothesis — recorded for the trail

The first reading, carried from PR #94 / PR #101, was a *consultant
evaluation-field residual*: training stores a discretized annotation, inference
sends live free-form prose, so the model supposedly never learned to terminate
on the unseen prose shape. This was **wrong** — see below.

---

## Root cause identified & fixed (commit `dbc61a2`)

**The turn terminator `<turn|>` was masked out of the training loss.**

The decisive fact: **the 5090 *full* run also collapsed**, not just the 100-step
gate. That rules out undertraining — more steps cannot teach a token that is
never in the labels.

`scripts/train_sft.py`'s Gemma 4 training chat template rendered the model-turn
terminator `<turn|>\n` **after** `{% endgeneration %}`:

```jinja
{% generation %}{{ message['content'] | trim }}{% endgeneration %}
...
{{- '<turn|>\n' -}}        # OUTSIDE the generation block
```

With `assistant_only_loss=True`, TRL builds the loss mask from the
`{% generation %}…{% endgeneration %}` span (via `return_assistant_tokens_mask`);
everything outside it becomes `-100`. So the terminator received **zero gradient
every step** — the model was trained to produce content but never taught to stop.
This is the exact failure TRL's own `chat_templates/gemma3_training.jinja` avoids
by keeping `<end_of_turn>` *inside* the generation block. TRL's guard only checks
that `{% generation %}` is *present*, not that it is placed correctly, so the
broken template passed silently.

This explains every symptom:
- **Full run collapsed too** → structural, not data-quantity.
- **Warm-up passed, gate failed** → the base `-it` model already knows `<turn|>`;
  on a short trivial prompt that prior survives, on the long SFT-distribution
  prompt the adapter dominates and there is no learned stop.
- **First sentence correct, then loops** → content was trained (inside the span),
  termination never was (outside).

### The fix

Move the terminator inside the generation block for the model role:

```jinja
{% generation %}{{ message['content'] | trim }}{{ '<turn|>\n' }}{% endgeneration %}
```

Rendered **text is byte-identical** (verified for both `add_generation_prompt`
values and multi-turn) — only the loss-mask boundary moves, so the locked
headline / Tables are untouched. Added
`test_model_turn_terminator_is_inside_the_loss_mask` (no model download) that
renders the real template through transformers' jinja machinery and asserts the
assistant span is exactly `content + <turn|>\n`.

### Verified against the real `unsloth/gemma-4-31B-it-unsloth-bnb-4bit` tokenizer

(Tokenizer is public — pulls without HF auth; weights not needed.)

- `<turn|>` is a **single registered special token (id 106)**, distinct from
  `<eos>` (id 1). `eos_gate.py:_stop_ids` picks it up correctly. (`<start_of_turn>`
  / `<end_of_turn>` are *absent* from the vocab — this Gemma 4 genuinely uses
  `<|turn>` / `<turn|>`.)
- Custom training template renders **byte-identical message bodies** to the stock
  template. The *only* divergence is the generation primer: stock appends a
  `<|channel>thought\n<channel|>` reasoning primer (Gemma 4 is a thinking model);
  the training template omits it, training the teacher to answer directly.
- The gate reproduced the collapse using the **custom** template (no primer), so
  the masking bug is the cause of the *reproduced* failure — the fix is necessary
  and the gate is a coherent test of it.

---

## What to do next

### 1 — Confirmation retrain on the R9700 (the only GPU step)

Re-run the same 100-step gate; it uses the fixed training template:

```bash
rocm-smi --showpids | grep python || echo "GPU clean"   # KFD must be clean
make train-gemma4-31b-eos-gate        # ~115 min, now with the terminator trained
make eos-gate-gemma4-31b              # expect STEP 2 to PASS
```

If the gate **passes**, launch the full Stage 2 run:
```bash
make train-gemma4-31b-stage2-unsloth
tail -f outputs/sft-stage2-gemma4-31b/train.log
```

### 2 — Before the downstream eval: check the serving generation primer

Separate, pre-existing concern (not the masking bug): production serving
(llama.cpp / GGUF for the 72.24 eval) must prime generation with plain
`<|turn>model\n`, **not** the stock `<|channel>thought\n<channel|>` thinking
primer — otherwise it reintroduces a train/serve mismatch this template
deliberately avoids. Confirm which template `serve_gemma4_31b*.sh` /
`serve_teacher` use before trusting eval numbers.

---

## Files changed this session

| File | Change |
|------|--------|
| `Makefile` | Strip `expandable_segments:True`; add EOS gate targets; `setsid`; `TORCH_USE_HIPBLASLT=1` for eos-gate + stage2 |
| `scripts/train_sft.py` | **Root-cause fix (`dbc61a2`): move `<turn|>` terminator inside `{% generation %}` so it is trained.** `attn_implementation` hardcoded to `"sdpa"` (see FA2 finding below) |
| `scripts/eos_gate.py` | Fix `apply_chat_template` → extract `input_ids` from `BatchEncoding` |
| `tests/test_sft_inference_format.py` | **`dbc61a2`: `test_model_turn_terminator_is_inside_the_loss_mask` — pins the terminator inside the loss span.** |
| `scripts/monitor_eos_gate.sh` | Autonomous EOS gate crash monitor + eval runner + Stage 2 launcher |
| `scripts/monitor_stage2.sh` | Autonomous Stage 2 crash monitor with PR notifications |
| `scripts/test_gpu_stack.sh` | Step 9: set `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`; update FA2 install note |
| `docs/HANDOFF_SFT_SCHEMA_DRIFT_FIX.md` | Pre-existing handoff from prior session |

---

## Stage 2 status (2026-05-31)

**EOS gate: PASSED** at 02:56 — fix confirmed. Stage 2 running since ~03:45.

**Current training config:**
- `TORCH_USE_HIPBLASLT=1` (hipBLASLt for GEMM — restored after confirming the step-0 page fault was a dirty KFD, not hipBLASLt itself)
- `attn_implementation="sdpa"` (see FA2 finding below)
- `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` (set in Makefile for future use; currently no effect since FA2 is disabled)
- ~14,478 total steps at ~70 s/step → ~281h wall time

**Confirmed step time: ~70 s/step** (measured at steps 1–9 of Stage 2 with SDPA + HIPBLASLT=1). Hardware ratio alone (R9700 ≈ 54% of 5090 TFLOPS) predicts ~8 s/step; actual is ~9× slower.

> **CORRECTION (profiled, `scripts/profile_train_step.py`):** the earlier claim
> "SDPA attention is the bottleneck, not GEMM" was wrong. A real-step `torch.profiler`
> trace shows the GPU self-time split is:
> - **`bitsandbytes::dequantize_4bit` (NF4→bf16) ≈ 55%** — the dominant cost;
> - GEMM (`aten::mm` / rocBLAS `Cijk_*`) ≈ 22%;
> - **attention (aotriton `attn_fwd` + `bwd_kernel_*`) ≈ 5%**, already on a fast
>   kernel (~0.5 ms/call fwd) — *not* the slow math path.
>
> So the ~70 s/step is **NF4-dequant-bound**, the inherent 4-bit QLoRA tax (every
> matmul dequantizes its weight), made worse by the RDNA4 dequant kernel being
> memory-bound. This is also why HIPBLASLT=1 didn't help — GEMM isn't the wall.
> **FA2 is therefore not worth pursuing for speed** (see below); attention is ~5%.
> The only real wall-time levers are fewer tokens (epochs 3→1, done) or dropping
> quantization (needs >32 GB VRAM for a 31B — not available here).

---

## FA2 investigation and gfx1201 shared memory limit

**Findings from this session (commit `e755aa9`, reverted in follow-up):**

`flash_attn 2.8.3` was installed via `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` (Triton JIT path, Wave32-aware). The **forward pass** benchmarked at **0.2 ms/call** for Gemma 4 attention shape (seq=1280, 8 heads, dim=128). However, the **backward pass (training step) immediately OOMed**:

```
triton.runtime.errors.OutOfResources: out of resource: shared memory,
Required: 131072, Hardware limit: 65536.
```

gfx1201 has a 64 KB shared memory hard limit; the Triton backward kernel (`_bwd_kernel_dkdv_causal`) requires 128 KB. This is a fundamental hardware constraint, not a configuration issue.

**Implication:** FA2 is usable for **inference** on gfx1201 but **not for training** with the current flash_attn 2.8.3 Triton AMD backward implementation. SDPA remains the correct choice for training.

**Possible future path (now de-prioritised):** one could patch `flash_attn_triton_amd/bwd_prefill_split.py` to cap block sizes within 64 KB shared memory (halving BLOCK_N/BLOCK_M). **But the profiler above settles it: attention is only ~5% of step time and already runs a fast aotriton kernel, so even a perfectly-patched FA2 backward would shave a few percent at most.** Not worth the spike while the step is NF4-dequant-bound. Revisit only if the dequant cost is ever addressed (fused 4-bit GEMM, or training unquantized on a larger card).

**Note on Vulkan:** Vulkan is available for inference (llama.cpp `GGML_VULKAN=ON`, ~20% TG speedup confirmed). No PyTorch training backend exists for Vulkan on Linux.

---

## Key commands

```bash
# Check GPU state before any run
rocm-smi --showpids | grep python || echo "GPU clean"

# GPU stack smoke test (all 13 steps should pass, including flash_attn 2.8.3)
make test-gpu-stack

# EOS gate full sequence
make train-gemma4-31b-eos-gate        # ~115 min (100 steps × 70 s/step)
tail -f outputs/eos-gate-gemma4-31b/train.log
make eos-gate-gemma4-31b              # ~10 min (loads model + generates)

# Full Stage 2 run (only after EOS gate PASS)
# Uses TORCH_USE_HIPBLASLT=1 + FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE + sdpa
make train-gemma4-31b-stage2-unsloth
tail -f outputs/sft-stage2-gemma4-31b/train.log

# Autonomous monitors (started automatically by monitor_eos_gate.sh on EOS PASS)
tail -f outputs/monitor_eos_gate.log
tail -f outputs/monitor_stage2.log
```
