# Training LLMs on AMD gfx1201 (RDNA 4 / Radeon AI PRO R9700): What Works, What Costs You, and Why

A field report on QLoRA / LoRA fine-tuning of modern transformer and hybrid (Mamba +
attention) LLMs on **AMD gfx1201 (RDNA 4, Wave32)** under **ROCm 7.2**, framed against the
NVIDIA/CUDA baseline where most of these capabilities are free.

> **Source repo:** [github.com/ulises-c/csen-346](https://github.com/ulises-c/csen-346) — the
> canonical home of this report, the companion `docs/GPU_SUPPORT.md`, `scripts/patch_fla_rocm.sh`,
> and the full investigation history ([#100](https://github.com/ulises-c/csen-346/issues/100) ·
> [#79](https://github.com/ulises-c/csen-346/pull/79) ·
> [#109](https://github.com/ulises-c/csen-346/issues/109)).

> **Scope.** This is a *synthesis* of hands-on debugging on a single R9700 box, not a
> re-derivation from upstream. Every claim carries a confidence marker (below) so an external
> reader can tell "we ran this on real hardware" from "theorized but untried." Findings are
> generalizable to gfx1201 training; specific model names (Qwen3.6-27B, Gemma 4 31B,
> Qwen3.5-0.8B) appear only as concrete architecture examples.
>
> Companion doc: [`GPU_SUPPORT.md`](https://github.com/ulises-c/csen-346/blob/main/docs/GPU_SUPPORT.md) covers the *serving* side. This doc is
> *training*.

## Confidence legend

| Marker | Meaning |
|---|---|
| ✅ | **Confirmed on the R9700** — observed directly, reproduced |
| ⏳ | **Untried / theorized** — plausible, not yet validated on this hardware |
| 🔭 | **Upstream watch** — the real fix lives in a dependency; tracked, not landed |
| ↩️ | **Correction** — a widely-repeated claim (or an earlier note of ours) that turned out wrong |

---

## TL;DR — the gfx1201 tax vs. NVIDIA/CUDA

On NVIDIA, the modern QLoRA stack (`transformers` + `peft` + `trl` + `bitsandbytes` + Flash
Attention + Triton) installs from PyPI and trains. On gfx1201 each layer of that stack has a
seam where the CUDA assumption leaks through. The headline:

| Capability | NVIDIA/CUDA baseline | gfx1201 / ROCm 7.2 status | Workaround | Conf. |
|---|---|---|---|---|
| **Flash *Linear* Attention** (Mamba/DeltaNet fast-path) | works (PyPI wheel) | **JIT-deadlocks at step 0** — Triton 3.6.0 `tritonamdgpu-pipeline` use-after-free at `num_stages≥2` | patch FLA `num_stages→1` / `num_warps→4`, **or** pin Triton 3.5.1 | ✅ |
| **Flash Attention 2** (softmax) | works (PyPI wheel) | **CK backend won't compile** — Composable Kernel assumes Wave64 (CDNA); RDNA 4 is Wave32 | `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` (Triton backend) — or just use SDPA | ✅ |
| **bitsandbytes 4-bit NF4 *runtime*** | works | **works** (stock `bitsandbytes==0.49.2` from PyPI; no source build needed) | — (source build `-DBNB_ROCM_ARCH=gfx1201` only if a future version regresses) | ✅ |
| **hipBLASLt / cuBLASLt fast matmul** | cuBLASLt, default | **crashes in RoPE** (`HIPBLAS_STATUS_INVALID_VALUE`) on some attention archs | `TORCH_USE_HIPBLASLT=0` → rocBLAS fallback (slower); test per-arch first | ✅ |
| **QLoRA load (4-bit from a BF16 base)** | needs only post-quant RAM | **stages the *full* BF16 model in CPU RAM before quantizing** — a ~31B model wants ~62 GB transient | load a **pre-quantized** bnb checkpoint (already NF4) | ✅ |
| **`expandable_segments:True`** allocator | works | **contested** — helped in one session; a later run saw it precede GPU page faults at step 16 | latest verdict: **leave it off**; `garbage_collection_threshold:0.8` alone (§6) | ⚠️ ↩️ |
| **FP8 (E4M3) compute** | works (Hopper/Blackwell) | **silently falls back to FP32** — gfx1201 missing from AITER arch table | none (affects serving, not LoRA training) | 🔭 |
| **vLLM** | works | **doesn't recognize gfx1201** upstream | serve via HF Transformers instead | ✅ |
| **`pip install` of CUDA-source extensions** (`causal-conv1d`, `flash-attn`) | works | **"NVCC trap"** — isolated build pulls CUDA torch, injects `-gencode compute_80` | build with `--no-build-isolation` against your ROCm torch (routes through `hipify_torch`) | ✅ |

**Net for QLoRA fine-tuning on a 32 GB R9700:** it works, but two things bite that never bite
on NVIDIA — the **FLA Triton deadlock** (only if your model has linear-attention layers) and
the **load-time CPU-RAM staging wall** (for any large 4-bit load from a BF16 base). Both have
clean workarounds. VRAM itself is not the constraint once the model is loaded.

---

## Reproduction environment

| Component | Spec |
|---|---|
| CPU | AMD Ryzen 9 5900X (12C/24T) |
| GPU | **AMD Radeon AI PRO R9700** — gfx1201 (RDNA 4, Navi 48), 64 CUs, **32 GB VRAM**, Wave32 |
| System RAM | 64 GB |
| OS / kernel | CachyOS (Arch-based) |
| ROCm | **7.2** (HIP 7.2.26015, ROCm Runtime 1.18) — native gfx1201, no HSA override needed |
| PyTorch | **2.11.0+rocm7.2** |
| Triton | **3.6.0** (patched) — see §1 |
| transformers / peft / trl | 5.9.0 / 0.19.1 / 1.4.0 |
| bitsandbytes | 0.49.2 |
| flash-linear-attention | 0.5.0 (patched) |
| causal-conv1d | 1.6.2.post1 (built `--no-build-isolation`) |

**Search terms** (for anyone landing here from a search engine): `gfx1201` · `RDNA4` ·
`R9700` · `ROCm 7.2` · `flash-linear-attention deadlock` · `num_stages Triton gfx12` ·
`tritonamdgpu-pipeline use-after-free` · `caching_allocator_warmup` · `Found no NVIDIA driver` ·
`bitsandbytes CPU dispatch ROCm` · `Gemma QLoRA OOM load` · `HIPBLAS_STATUS_INVALID_VALUE RoPE`.

---

## The central distinction: two kernel stacks, two blockers

The single most important thing to internalize before debugging gfx1201 training is that
"attention" splits into two completely separate kernel stacks with separate failure modes.
Which one bites you depends entirely on your model architecture:

| | **Linear-attention models** (Mamba / gated-DeltaNet hybrids — e.g. Qwen3.6-27B: 48 linear + 16 full layers; Qwen3.5-0.8B hybrid) | **Softmax-attention models** (e.g. Gemma 4 31B: sliding-window + global) |
|---|---|---|
| Fast-path library | **Flash *Linear* Attention** (Triton) + `causal-conv1d` | **Flash Attention 2** / SDPA |
| Real gfx1201 blocker | **FLA Triton deadlock** (§1) | **none for training** — SDPA is stable (§2) |
| Load blocker (4-bit) | CPU-RAM staging (§5) | CPU-RAM staging (§5) |

Most community "Flash Attention on RDNA4" material is about **FA2** (softmax). The FA2 story is
the *related landscape*, but if your model has linear-attention layers, your actual training
blocker is **FLA**, which is a different library and a different bug.

---

## 1. Flash Linear Attention (FLA) — the Triton 3.6.0 deadlock ✅

**Symptom.** `flash-linear-attention==0.5.0` installs and imports fine. On the **first training
step** it JIT-compiles ~120 Triton kernels, then one kernel **hangs forever**: process alive at
0% CPU, VRAM full, no error, progress bar stuck at step 0. Community reports describe the same
signature as `hsa_signal_wait` polling a DMA-completion flag that never flips.

**Root cause.** Triton **3.6.0**'s AMD software-pipelining pass (`tritonamdgpu-pipeline`) has a
**use-after-free when `num_stages ≥ 2` on gfx12xx** — and gfx12xx is **not in Triton 3.6.0's
verified target list**. The same compiler bug has two manifestations:

- **FLA:** silent first-step deadlock (the hang above).
- **SageAttention:** hard crash — `RuntimeError: PassManager::run failed 'tt.load' op operation destroyed but still has uses`.

**Fix A — patch the wheel (confirmed, supported).** Rewrite `num_stages=[2-9]→1` and
`num_warps>4→4` (Wave32) in the installed FLA Triton autotune configs, then clear the Triton
and FLA caches. This mirrors the upstream SageAttention fix
([thu-ml/SageAttention#365](https://github.com/thu-ml/SageAttention/pull/365),
[kijai/ComfyUI-WanVideoWrapper#2007](https://github.com/kijai/ComfyUI-WanVideoWrapper/issues/2007)),
which gates `num_stages=1` on `torch.version.hip is not None`. Confirmed working at seq 768.
**Must be re-applied after every dependency sync** because the package manager reinstalls a
fresh, unpatched wheel.

```bash
# concept — find the installed FLA, cap the autotune configs, nuke caches
FLA=$(python -c "import fla; print(fla.__path__[0])")
find "$FLA/ops" -name '*.py' -exec sed -i -E 's/num_stages=[2-9]/num_stages=1/g; s/num_warps=([5-9]|[12][0-9])/num_warps=4/g' {} +
rm -rf ~/.triton/cache
```

**Fix B — pin Triton 3.5.1 (avoids the bug at the source) ⏳.** The UAF is a 3.6.0 regression;
a prebuilt `triton-3.5.1+rocm7.2.1` wheel exists, and the gfx1201 community reports
**Triton 3.5.1 + PyTorch 2.9.1** as the stable native pairing with FLA working **unpatched**.
⚠️ Verify ABI before adopting on a torch 2.11 stack — the documented pairing is torch 2.9.1.

**Nightly status 🔭.** **No nightly fixes this.** As of late May 2026, gfx12xx is still not in
Triton's verified target list, and every "fix" in the wild is a *downstream* `num_stages=1`
workaround in the calling library — not a Triton compiler fix. Upgrading torch/Triton nightly
does **not** help and risks re-introducing the bug. Pin and patch.

**Cheaper levers to try first ⏳** (before any reinstall):
- `TRITON_DISABLE_AUTOTUNING=1` — the hang is in one *autotuned* config; disabling autotune
  means the broken candidate is never generated. Cheapest, stack-agnostic, reversible.
- `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` — enables AOTriton experimental flash paths on
  RDNA 4; stackable with the above.

**Why it matters (stakes).** Without FLA, linear-attention models fall back to a pure-PyTorch
path that only fits at short sequence length, truncating long dialogues and ballooning ETA
(an observed ~143 h / 3 epochs for a 27B at seq 512). FLA is what unlocks seq ≥ 768 *and* a
sane ETA — it is load-bearing, not a nice-to-have.

---

## 2. Flash Attention 2 (softmax) — Wave64 vs Wave32 ✅

Stock `flash-attn` defaults to the **Composable Kernel (CK)** backend, which assumes
**Wave64 (CDNA)**. RDNA 3/4 are **Wave32**, so CK fails to compile / hits an ISA mismatch
(the PyPI `flash-attn` wheel fails on gfx1201 outright).

- **Fix:** `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` + `pip install flash-attn --no-build-isolation`
  to JIT a gfx1201 **Triton** FA2 kernel at runtime (fwd+bwd, causal, varlen). Build the
  [ROCm/flash-attention](https://github.com/ROCm/flash-attention) fork, Triton backend.
- **Simpler for training: just use SDPA.** `attn_implementation="sdpa"` is stable on gfx1201
  and, on ROCm, may route the softmax layers to the **AOTriton** flash kernel when the torch
  build's AOTriton supports gfx1201 (otherwise it silently falls back to the math /
  mem-efficient path — correct, just slower). For a model whose *only* attention is softmax
  (e.g. Gemma 4), SDPA alone is sufficient — no FA2 build needed.

**Dead ends, logged so nobody re-investigates:**
- **rocWMMA FA2** ([Repeerc/flash-attention-v2-RDNA3-minimal](https://github.com/Repeerc/flash-attention-v2-RDNA3-minimal)) — RDNA 3 (gfx1100) only, no backward pass, weak BF16. Not viable for training.
- **ThunderKittens** — NVIDIA-only (Hopper/Blackwell). Its AMD sibling **[HipKittens](https://github.com/HazyResearch/HipKittens)** supports **CDNA3/CDNA4 (MI300/MI350) only — not gfx1201**.

---

## 3. bitsandbytes 4-bit NF4 *runtime* ✅

This is the one true gate common to **every** 4-bit path (live-quant or pre-quant): the NF4
dequant kernels must run on gfx1201 at train time. **Good news: they do, on stock
`bitsandbytes==0.49.2` from PyPI — no source build required.** Confirmed via a GPU stack
smoke test (8-bit LLM.int8 forward ✅, 4-bit NF4 QLoRA ✅, `BitsAndBytesConfig` NF4 +
double-quant ✅).

gfx1201 is in bitsandbytes' RDNA arch table. If a future version regresses, the escape hatch is
a source build: `cmake -DCOMPUTE_BACKEND=hip -DBNB_ROCM_ARCH="gfx1201"`.

---

## 4. hipBLASLt RoPE crash — and when it's safe ✅

Without mitigation, hipBLASLt can raise `HIPBLAS_STATUS_INVALID_VALUE` during the first forward
pass on gfx1201. The blunt fix is `TORCH_USE_HIPBLASLT=0` (forces rocBLAS) — but that costs the
matmul speedup (measured ~4.1× on gfx1201: 5.2 ms vs 21.5 ms; rocBLAS fallback observed at
~75–90 s/step in one config).

**Nuance worth testing per-architecture:** the page-faults (`rc=134`) were traced to a specific
linear-attention kernel path (`torch_chunk_gated_delta_rule`) — **not** a general hipBLASLt
instability. For a pure-softmax model that never touches the delta-rule op,
**`TORCH_USE_HIPBLASLT=1` may be safe** and worth a 20-step trial before defaulting to the
rocBLAS fallback — the latest hands-on Gemma 4 (softmax) launch adopted exactly this. One
session paired it with `expandable_segments:True` to stabilize past the early crash zone, but
that allocator knob later proved contested (§6) — keep the two changes independent when testing.

---

## 5. The QLoRA load wall — CPU-RAM staging ✅ (the blocker NVIDIA users never see)

This is the one that actually stops large-model QLoRA on a modest-RAM box, and it is **not**
VRAM, **not** FLA, and **not** GPU-specific — it would hit any GPU paired with insufficient
*system* RAM:

**`from_pretrained(..., load_in_4bit=True)` materializes the full BF16 model in CPU RSS
*before* NF4 quantization** — regardless of `low_cpu_mem_usage=True`, `offload_state_dict`,
`offload_folder`, GC threshold, or malloc tuning. bitsandbytes quantizes only after all weights
are resident. So a ~31B model needs **~62 GB of transient CPU RAM at load**, even though the
post-quant resident footprint is ~16 GB. On a 64 GB box this thrashes into zram
(~200 s/shard → ~20 h just to load). 8-bit (`load_in_8bit`) has the **same** wall.

**Fix — load a pre-quantized bnb checkpoint.** Weights arrive already NF4 (~19 GB on disk),
streamed straight to 4-bit with no full-BF16 CPU materialization. Drops load-RAM from ~62 GB
to ~8–16 GB and needs no second high-RAM machine. Verify the embedded `quantization_config`
matches your pipeline (`nf4` + double-quant + `bfloat16` compute) and that the LoRA target
regex still matches the checkpoint's module paths.

> **Two device-map gotchas on the 4-bit load path (gfx1201-specific seams):**
> - **`device_map="auto"` can dispatch layers to CPU**, which bitsandbytes rejects. Force
>   `device_map={"": 0}` to keep the whole quantized model on the single GPU.
> - Recent `transformers` runs a CUDA-only `caching_allocator_warmup` that raises
>   **"Found no NVIDIA driver"** on HIP/ROCm. Patch/guard it for `torch.version.hip`.
> - Passing an explicit `quantization_config=None` to `from_pretrained` can make `auto_factory`
>   overwrite a pre-quantized checkpoint's own config with `None` → `supports_quant_method`
>   crash. Only pass the kwarg when it's non-None.

### Hardware estimate — large (~31B) QLoRA, seq 1280, batch 1×16, grad-ckpt on, eval off

| Resource | Pre-quant checkpoint (recommended) | Naive `load_in_4bit` from BF16 |
|---|---|---|
| GPU VRAM (peak, train) | **~22–27 GB** | ~22–27 GB |
| **System RAM (peak, load)** | **~8–16 GB** | **~62–70 GB** ← the blocker |
| Download | ~19 GB | ~62 GB |
| Disk working set | ~30 GB | ~75 GB |

**Minimum viable:** GPU **≥ 32 GB** (R9700 / RTX 5090) · 24 GB (3090) marginal (short seq only) ·
**20 GB (RTX 4000 Ada) won't fit**. System RAM **~16 GB** with the pre-quant checkpoint.
**Keep eval disabled** — a large-vocab `logits.float()` copy adds an ~8.6 GB spike that OOMs
even 32 GB.

---

## 6. Memory-allocator config — what's true now (one contested knob) ↩️

| `PYTORCH_HIP_ALLOC_CONF` knob | Verdict on gfx1201 |
|---|---|
| `garbage_collection_threshold:0.8` | ✅ use it |
| `expandable_segments:True` | ⚠️ **contested — default off** (see below) |
| `max_split_size_mb:128` | ❌ **remove** — blocks legitimate >128 MB allocations → spurious OOM |

> ⚠️ ↩️ **`expandable_segments:True` flip-flopped three times — current verdict is OFF.**
> This knob has the least stable story of anything here, so treat it as unsettled rather than
> a clean ✅:
> 1. **First pass** — reported "unsupported, silently ignored" on gfx1201 → don't use.
> 2. **Second pass** — re-added in a later session, appeared to help (released freed BF16 shards
>    from VRAM during load) → use it. *(Earlier versions of this doc reported this as the
>    resolved state.)*
> 3. **Latest hands-on (most recent)** — **removed again**: the HIP allocator's
>    "not supported" warning **preceded GPU page faults at step 16**, so it was stripped from
>    all `PYTORCH_HIP_ALLOC_CONF` entries as a precaution.
>
> The page-fault correlation is suggestive, not proven causation — but the **latest empirical
> call is to leave it off** and run `garbage_collection_threshold:0.8` alone. If you try it,
> watch for a step ~10–20 page fault.

---

## 6.1 The non-deterministic backward page fault (PR #101) — what we know, and the ablation to settle it ⚠️

> **Live per-run tracker: [`GFX1201_FAULT_ABLATION_LOG.md`](GFX1201_FAULT_ABLATION_LOG.md)** —
> the canonical, append-one-row-per-run log (commit hash, variables, fault step, kernel, verdict).
> It **supersedes** this section where they disagree: the fault is now localized to a **rocBLAS
> Tensile GEMM (`MT64x64x64 ISA1201`)**, not the bitsandbytes path — so the "bnb-NF4 backward"
> framing below and **Arm A are retired**, and hipBLASLt is open again as a *fix* (`=1`), having
> only been ruled out as a *cause* (`=0`).

Gemma 4 31B QLoRA on the R9700 hits a **non-deterministic GPU page fault during the backward
pass** and has never completed a full run. This section is the durable record so we stop
re-deriving it. **Do not "fix" it with another single knob flip** — four have been tried and
falsified.

```
Memory access fault by GPU node-1 on address 0x7f.......  Reason: Page not present or
supervisor privilege.   (amdgpu gfxhub TCP fault; PERMISSION_FAULTS:0x3, WALKER_ERROR:0x0)
```

**The one fact that reframes everything (wandb-verified):** the fault is *probabilistic, not
config-determined.* The **same config on the same git SHA both finishes and crashes**:

| git SHA | run | outcome |
|---|---|---|
| `c752a2b4` | eos-gate g2df2ifl | **finished 100 steps** |
| `c752a2b4` | eos-gate gv9fbjac | **crashed < step 10** |
| `eb12dbd9` | eos-gate 5xd8qt5w | **finished 100 steps** |
| `eb12dbd9` | eos-gate irgdklt9, jtyyhu4t | **crashed @ step 10** |

Crash steps across all Stage 2 + eos-gate runs: 10, 14, 16, 20, 80, 84, or clears 100 — a
random draw, not a threshold. So a "good config" cannot be inferred from one clean run.

**What the wandb data rules OUT** (don't re-investigate these):
- **Not numerical / data / a bad batch.** Loss descends smoothly (2.6→0.8), grad_norm stays
  1–6, no NaN/Inf; two same-LR runs log *byte-identical* losses per step (deterministic forward).
- **Not LR / optimizer magnitude.** A `lr=5e-6` run crashed *earlier* (step ~10) than the
  `lr=5e-5` runs (step ~80) — lower LR should be safer; this inverts the theory.
- **Not hipBLASLt.** `TORCH_USE_HIPBLASLT=0` runs crash with the identical signature.
- **Not sequence length.** Measured token lengths (`scripts/determine_max_sequence.py`):
  **train max = 909, eval max = 892, p99 = 673** over socrat-zh-sft + socrat-en-sft. Nothing
  exceeds 1024. With `per_device_batch_size=1` + dynamic padding the activation peak is set by
  the *actual* longest sequence (909), **not** by `max_length`, so lowering the 1280 cap frees
  no VRAM. `max_length=1280` is non-binding and safe; leave it.

**Still unsettled (do NOT treat as ruled out):** the GC threshold. The `72db9b4` run with
`garbage_collection_threshold` *removed* also crashed, but its log was lost (wandb 404) so
page-fault-vs-OOM was never confirmed, and it is n=1. GC-on vs GC-off is therefore not yet a
clean variable either way — replicate it under the protocol below (arm D).

**Enabling condition:** the run sits at **93–98 % VRAM (~30–31.4 / 32 GB) sustained**. The fault
is in the backward pass, in the immature **bitsandbytes-NF4 / ROCm 7.2** path — the least-tested
part of the stack (§3 only smoke-tested bnb a few steps, never 100+ under pressure). The fault
address has a host-VA `0x7f…` pattern, consistent with a use-after-unmap / bad-buffer access.

**Two failure modes that COMPOUND (operational, not config):**
1. **Dirty-KFD cascade.** A fault leaves orphaned HIP context + stale VRAM (§10). Relaunching
   into it faults early on stale PTEs. The resume monitor must verify a clean GPU
   (`test_gpu_stack.sh --wait-clean`), not blind-sleep, before each relaunch.
2. **Corrupt checkpoint.** `save_steps=10` (the crawl-forward survival patch) multiplies
   checkpoint-write windows 10×; a fault mid-write leaves an incomplete `checkpoint-N` that
   resume loops on. The monitor quarantines an incomplete latest checkpoint (no valid
   `trainer_state.json`) so resume falls back to `N-1`.

`save_steps=10` + the monitor crawl is a **survival workaround, not a fix** — it makes a faulting
run inch forward (confirmed: a run crashed at step 16 but left a resumable `checkpoint-10`). The
fix still has to come from the ablation below.

### Localize first, then ablate — `not guessing`

**Phase 0 — localize (`make diagnose-gfx1201-fault`).** The fault is async, so the kernel
"running" at fault time is not necessarily the culprit — every theory so far was correlation.
The diagnostic runs ~120 steps under `AMD_SERIALIZE_KERNEL=3 HIP_LAUNCH_BLOCKING=1
AMD_LOG_LEVEL=1` + a `dmesg` tail, so the fault becomes synchronous and the traceback/ring-log
**name the faulting kernel**. (Level 1 = errors only; do NOT use level 3 — it logs every HIP
call and writes multi-GB `diag.log` that can fill the disk. `LOG_LEVEL=3 make …` opts in.)
If serialization makes it vanish → concurrency/allocator race; if
it still faults at the same named kernel → kernel bug (bnb dequant vs grad-ckpt recompute vs
allocator). This decides which arms below matter.

**Phase 1 — ablation matrix.** Because the fault is probabilistic, each cell needs **N=3
replicates** and a fixed **≥150-step budget** (clears the 14–84 window with margin), each behind
`make gpu-preflight` (clean GPU) with a fixed seed. Metric per cell: fraction clearing the
budget + median crash-step. Change exactly **one** factor from baseline (HIPBLASLT=0, GC=0.8,
grad-ckpt `use_reentrant=False`, bnb 0.49.2, sdpa, seq 1280, bs 1×16):

| Arm | One change | Hypothesis | Note |
|---|---|---|---|
| A | bitsandbytes **source build** `-DBNB_ROCM_ARCH=gfx1201` | NF4 backward kernel bug | top suspect if Phase 0 names a bnb kernel |
| B | `TRAIN_GRAD_CKPT=false` | recompute drives the faulting dequant | likely **OOMs** at 98 % VRAM — diagnostic, not a candidate fix |
| C | grad-ckpt `use_reentrant=True` | non-reentrant checkpoint hook race | one-line change in `train_sft.py` |
| D | `PYTORCH_HIP_ALLOC_CONF` unset | allocator GC/unmap race | GC-off already crashed once (n=1) — needs replication |
| E | `HSA_ENABLE_SDMA=0` | DMA-engine page-fault mitigation | cheap, stackable |
| F | bump/pin ROCm or torch build | driver/runtime immaturity | last resort; expensive |

Stop-rule: the first arm that is **3/3 clean over 150 steps** becomes the new baseline; keep
stacking from there. The survival workaround stays on throughout — it is how the run makes
progress while the ablation finds the real fix.

> Tooling for this section lives in: `make diagnose-gfx1201-fault`
> (`scripts/diagnose_gfx1201_fault.sh`), `make gpu-preflight`
> (`scripts/test_gpu_stack.sh --preflight` / `--wait-clean`),
> `scripts/determine_max_sequence.py`, and the hardened `scripts/monitor_stage2.sh`.

---

## 7. `HSA_OVERRIDE_GFX_VERSION` — do not set it ↩️

Half the older guides call `HSA_OVERRIDE_GFX_VERSION=11.0.0` "the critical fix." That value is
the **RDNA 3** identifier — it lies to ROCm that the card is RDNA 3, and on a real gfx1201 it
**hangs the GPU**. **ROCm 7.2 supports gfx1201 natively, so no override is needed at all.** If
some tool ever forces one, the *correct* gfx1201 value is **`12.0.1`** — never `11.0.0`. Most of
the "gfx1201 unsupported / HSA override" material predates ROCm 7.2 and is stale.

---

## 8. The `causal-conv1d` "NVCC trap" — an install-method artifact, not a wall ✅

Two theories circulated about what blocks the linear-attention path's short causal conv on
gfx1201. They reconcile into one fact: **it's the install method.**

| Install method | Outcome |
|---|---|
| `pip install causal-conv1d` (isolated build) | ❌ **NVCC trap** — the isolated build env resolves **CUDA** torch from PyPI, shells out to `nvcc -V`, and PyTorch's extension builder injects `-gencode arch=compute_80`, which `amdclang++` rejects |
| `uv pip install --no-build-isolation causal-conv1d` (+ `wheel`) | ✅ builds against the **ROCm** torch already in your venv → **1.6.2.post1**, fwd+bwd verified |
| [EmbeddedLLM/causal-conv1d-rocm](https://github.com/EmbeddedLLM/causal-conv1d-rocm) | ✅ clean fork — `setup.py` gates on `torch.version.hip` |

`--no-build-isolation` isn't a hack: it routes the build down AMD's official **`hipify_torch`**
path (PyTorch's C++ extension builder translates the CUDA source to HIP at build time — no nvcc,
no Triton — *when the build sees a ROCm torch*). This is exactly the mechanism in AMD's own
[Vision Mamba](https://rocm.blogs.amd.com/artificial-intelligence/vision-mamba/README.html)
recipe. **Never run plain `pip install causal-conv1d` in a ROCm venv.**

> The AMD Vision Mamba blog validates the hipified C++ `causal_conv1d`/`selective_scan` build on
> **CDNA (Wave64), ~2024 stack** — never on gfx1201/Wave32. That's *why* the RDNA 4 community
> routes the linear path through FLA's per-arch-JIT **Triton** kernels instead. The hipified C++
> path is a third candidate ⏳, but expect Wave64-tuning risk.

---

## 9. FP8 and vLLM — the serving-side gfx1201 gaps 🔭

Not training blockers, but part of the gfx1201 picture:

- **FP8 (E4M3) silently falls back to FP32** — gfx1201 is missing from AITER's arch table
  ([ROCm/TransformerEngine#520](https://github.com/ROCm/TransformerEngine/issues/520)); a
  two-line fix exists, unmerged. Throughput ~halves. Affects FP8 serving, not LoRA training.
- **vLLM doesn't recognize gfx1201 upstream**
  ([vllm#28649](https://github.com/vllm-project/vllm/issues/28649)) — serve via HF Transformers
  instead. See [`GPU_SUPPORT.md`](https://github.com/ulises-c/csen-346/blob/main/docs/GPU_SUPPORT.md).

---

## 10. Operational gotchas (cost you hours, not in any NVIDIA guide) ✅

- **Always `uv run --no-sync`.** A bare `uv run` triggers `uv sync`, which reinstalls **CUDA**
  torch from PyPI over your hand-installed `torch+rocm`. Every training/inference command needs
  `--no-sync`. (And re-run the FLA patch from §1 after any sync that does slip through.)
- **GPU state corruption cascades.** A GPU page-fault (`rc=134`) leaves the KFD in a dirty
  state; *subsequent* runs — even with a correct config — fault at random steps until you
  `kill -9` the process and confirm a clean GPU. Pre-flight every run:
  ```bash
  rocm-smi --showpids | grep -q python && echo "GPU dirty — kill before retrying" || echo "GPU clean"
  ```
- **Use `setsid`, not `nohup`, for the training wrapper.** `nohup bash run.sh &` doesn't create
  a new process group, so children survive `kill -9 -- -$PGID`. `setsid bash run.sh &` puts the
  whole tree in its own group → kill it cleanly and avoid overlapping processes on the same GPU.

---

## Confirmed version matrix

| Component | Confirmed-working (this box) | Community alt | Notes |
|---|---|---|---|
| ROCm | 7.2 (HIP 7.2.26015) | 7.1 | 7.2 = native gfx1201, no HSA override |
| PyTorch | **2.11.0+rocm7.2** | 2.9.1 (stable) | nightly **not** needed for FLA |
| Triton | **3.6.0 + `num_stages=1` patch** | **3.5.1** (no patch) ⏳ | 3.6.0 has the UAF |
| flash-linear-attention | 0.5.0 (patched) | bare-metal Triton | hangs unpatched |
| causal-conv1d | 1.6.2.post1 (`--no-build-isolation`) | EmbeddedLLM rocm fork | fwd+bwd verified |
| bitsandbytes | 0.49.2 (stock PyPI) | source `-DBNB_ROCM_ARCH=gfx1201` | 4-bit NF4 runtime ✅ |
| flash-attn (FA2) | Triton backend (`..._TRITON_AMD_ENABLE=TRUE`) or SDPA | — | CK backend won't compile (Wave32) |

---

## "Free on NVIDIA, costs a workaround here" — the one-screen summary

1. **Linear-attention fast-path** → patch Triton `num_stages→1` or pin 3.5.1.
2. **FA2** → flip to the Triton backend, or just use SDPA.
3. **Large 4-bit load** → use a pre-quantized checkpoint (the CPU-RAM staging wall is the real
   blocker, not VRAM).
4. **`device_map`** → force `{"": 0}`; guard the CUDA-only allocator warmup.
5. **hipBLASLt** → off for delta-rule archs; test-then-keep for pure softmax.
6. **Allocator** → `garbage_collection_threshold:0.8`; never `max_split_size_mb`; `expandable_segments:True` is contested — default off (§6).
7. **No HSA override** on ROCm 7.2.
8. **Build CUDA-source extensions with `--no-build-isolation`** (hipify path); never bare `pip install`.
9. **`uv run --no-sync`**, `setsid` wrappers, and a GPU-clean pre-flight.

Once past these, a 32 GB RDNA 4 card trains modern 27–31B-class models at 4-bit QLoRA with VRAM
to spare. The gap to NVIDIA is **ecosystem maturity (Triton/CK/vLLM target coverage), not
silicon capability.**

---

## References

**Source repo:** [github.com/ulises-c/csen-346](https://github.com/ulises-c/csen-346) — canonical
report, companion `docs/GPU_SUPPORT.md`, `scripts/patch_fla_rocm.sh`, and investigation history
([#100](https://github.com/ulises-c/csen-346/issues/100) ·
[#79](https://github.com/ulises-c/csen-346/pull/79) ·
[#109](https://github.com/ulises-c/csen-346/issues/109)).

**Primary (hands-on):** debugging sessions on the R9700 across the issues/PRs that produced this
report; `scripts/patch_fla_rocm.sh`; [`GPU_SUPPORT.md`](https://github.com/ulises-c/csen-346/blob/main/docs/GPU_SUPPORT.md).

**Upstream / community:**
- Triton `tritonamdgpu-pipeline` UAF (manifests via): [thu-ml/SageAttention#365](https://github.com/thu-ml/SageAttention/pull/365) · [kijai/ComfyUI-WanVideoWrapper#2007](https://github.com/kijai/ComfyUI-WanVideoWrapper/issues/2007)
- `causal-conv1d` NVCC trap: [Dao-AILab/causal-conv1d#99](https://github.com/Dao-AILab/causal-conv1d/issues/99) · [EmbeddedLLM/causal-conv1d-rocm](https://github.com/EmbeddedLLM/causal-conv1d-rocm)
- FA2 on RDNA: [ROCm/flash-attention](https://github.com/ROCm/flash-attention) · [Repeerc/flash-attention-v2-RDNA3-minimal](https://github.com/Repeerc/flash-attention-v2-RDNA3-minimal) · [HipKittens](https://github.com/HazyResearch/HipKittens)
- FP8 / serving: [ROCm/TransformerEngine#520](https://github.com/ROCm/TransformerEngine/issues/520) · [vllm#28649](https://github.com/vllm-project/vllm/issues/28649)
- AMD Vision Mamba (hipify mechanism): [rocm.blogs.amd.com/artificial-intelligence/vision-mamba](https://rocm.blogs.amd.com/artificial-intelligence/vision-mamba/README.html)
- gfx1201 community guides (apollo-mg): [Master guide (RDNA4 + ROCm 7.1)](https://gist.github.com/apollo-mg/ecba6a0c29323325a7ac3babf08e53be) · [FLA / RDNA4 config](https://gist.github.com/apollo-mg/d44cb753962fa9f6e1e45a7101f14284) · [The NVCC trap](https://gist.github.com/apollo-mg/e86abd863802bde296892fb1fe7aecae)
