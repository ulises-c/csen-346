# Handoff — gfx1201 QLoRA backward page fault: diagnosis in progress (PR #101)

**Date:** 2026-06-01 · **Branch:** `feat/gfx1201-rdna4-qlora-fla-training` · **Issue/PR:** #101
**For:** a fresh Claude instance picking up the root-cause analysis cold.

---

## TL;DR — where we are

Gemma 4 31B QLoRA on an AMD R9700 (gfx1201 / RDNA4, 32 GB, ROCm 7.2) hits a
**GPU page fault during the QLoRA backward pass** and has never finished a run.
We have just produced the **decisive diagnostic result** but have **not yet read
the faulting kernel out of it.** That is your job.

**Diagnostic result (just landed):** the serialized diagnostic run
(`make diagnose-gfx1201-fault`) **faulted at step 84** under
`AMD_SERIALIZE_KERNEL=3 HIP_LAUNCH_BLOCKING=1`. Two consequences:
1. **Serialization did NOT suppress the fault** → it is **not a pure async/stream
   concurrency race**. Down-weight that flavor of the allocator hypothesis.
2. It faulted at **step 84 — the same step as the clean-start baseline runs**
   (`6c9fouit`, `p9t3yxe5`). Strong evidence the fault is **reproducible at ~84 on
   a clean GPU**, and the earlier "random" crashes at steps 10–20 were the
   **dirty-KFD cascade** (faulting on stale page-table entries left by a prior
   crash), not the true first fault.

## YOUR IMMEDIATE TASK

The diagnostic ran with `HIP_LAUNCH_BLOCKING=1`, so the Python traceback points at
the **exact** failing op. On the **AMD box** (this analysis can't be done from the
NVIDIA dev box — the log is on the R9700), extract it:

```bash
# full synchronous traceback block:
ln=$(grep -n "Traceback (most recent call last)" outputs/diag-gfx1201/diag.log | tail -1 | cut -d: -f1)
sed -n "${ln},+80p" outputs/diag-gfx1201/diag.log

# fault line + HIP error:
grep -niE "memory access fault|page not present|PERMISSION_FAULTS|HIP error|RuntimeError" \
  outputs/diag-gfx1201/diag.log | tail -20

# kernel-side VM-fault decode:
sudo dmesg | grep -iE "amdgpu|gfxhub|VM_L2|page fault|PERMISSION_FAULTS|WALKER_ERROR" | tail -30
```

Note: that `diag.log` was produced by the **old** version of the diagnostic with
`AMD_LOG_LEVEL=3`, so it is **multi-GB** and full of HIP API spam — use `grep`/`sed`,
never open it whole. (The script now defaults to `AMD_LOG_LEVEL=1`; future runs
will be small.)

### Decision tree — the last torch frame in the traceback picks the ablation arm

| Last meaningful frame | Verdict | First ablation arm |
|---|---|---|
| `bitsandbytes` / `dequantize_4bit` / `Linear4bit` / `matmul_4bit` | NF4 backward kernel bug on gfx1201 | **A** — bitsandbytes **source build** `-DBNB_ROCM_ARCH=gfx1201` |
| `torch.utils.checkpoint` / recompute | grad-checkpoint recompute drives the bad kernel | **C** (`use_reentrant=True`), then **B** (`grad_ckpt=false`, expect OOM) |
| allocator / `empty_cache` / no clear op (fault between kernels) | memory **fragmentation** (NOT a race — serialization didn't help) | **D** — `PYTORCH_HIP_ALLOC_CONF` unset; consider `max_split_size` tuning |
| RoPE / attention / SDPA kernel | attention-path kernel on gfx1201 | investigate SDPA backend; secondary |

If the traceback is ambiguous (async artifact despite blocking), re-run the
diagnostic — it's now cheap at level 1: `make diagnose-gfx1201-fault`.

---

## What is already PROVEN (do not re-investigate)

Full evidence record + the ablation matrix live in
**`docs/GFX1201_RDNA4_TRAINING.md` §6.1**. Summary of the rule-outs (all from
wandb project `csen346-sft`):

- **Probabilistic, not config-determined.** Same config on the **same git SHA**
  both finished 100 steps and crashed at 10 (eos-gate `g2df2ifl`/`gv9fbjac` on
  `c752a2b4`; `5xd8qt5w`/`irgdklt9` on `eb12dbd9`). A clean 100-step run is luck,
  not proof of a good config. **(Now refined: reproducible at ~84 on a clean GPU;
  see diagnostic result above.)**
- **Not numerical / data / bad batch.** Loss descends smoothly 2.6→0.8, grad_norm
  1–6, no NaN; same-LR runs log byte-identical losses per step (deterministic fwd).
- **Not LR.** `lr=5e-6` crashed *earlier* (~step 10) than `lr=5e-5` (~84) — inverts
  the LR theory. Mechanistically the fault is in backward; LR acts later at
  `optimizer.step()`, so it can't affect the fault. (Likely the 5e-6 run was a
  dirty-GPU cascade victim.)
- **Not hipBLASLt.** `TORCH_USE_HIPBLASLT=0` runs fault identically.
- **Not sequence length.** `scripts/determine_max_sequence.py`: real token
  **max=909** (train) / 892 (eval), p99=673. Nothing exceeds 1024. With
  `batch_size=1` + dynamic padding the peak is set by the *actual* longest seq, not
  `max_length` — so 1280 is non-binding and lowering it frees no VRAM. **Leave it.**
- **GC threshold: still unsettled** (n=1 GC-off run also crashed but log was lost;
  page-fault-vs-OOM unconfirmed). Replicate via arm D.

**Enabling condition:** run sits at **median 89% / spikes to 98% VRAM** (~28.5 GB
steady, ~31.4 GB peak on the 32 GB card). The fault is in the immature
**bitsandbytes-NF4 / ROCm 7.2 backward** path. The serialized run survived repeated
98% spikes, then faulted at 84 — so pressure alone is not the deterministic trigger;
something that builds over ~84 steps (fragmentation?) or a specific kernel is.

---

## Infrastructure built this session (all on the branch, CI green)

| Thing | Where | Purpose |
|---|---|---|
| GPU clean-state + `--preflight` + `--wait-clean` | `scripts/test_gpu_stack.sh` (unified — do NOT add new GPU scripts) | gate every (re)launch on a clean, working GPU |
| `make gpu-preflight` | Makefile | fast clean-KFD + fwd/bwd gate; all train targets depend on it |
| `make diagnose-gfx1201-fault` | `scripts/diagnose_gfx1201_fault.sh` | serialized run that names the faulting kernel (`LOG_LEVEL=3` opts into verbose) |
| Hardened resume monitor | `scripts/monitor_stage2.sh` | `--wait-clean` before relaunch (kills dirty-KFD cascade), quarantines incomplete checkpoints, consecutive-no-progress retries (8), **archives full traceback + dmesg per crash to `outputs/sft-stage2-gemma4-31b/crashlogs/`** |
| Token-length measurement | `scripts/determine_max_sequence.py` | the 909 finding |
| Evidence record + ablation matrix | `docs/GFX1201_RDNA4_TRAINING.md` §6.1 | durable; don't re-derive |

`save_steps=10` (`configs/train-sft-stage2-gemma4-31b.env`) is a **survival crawl,
not a fix** — it lets a faulting run inch forward via checkpoint+resume. The real
fix comes from the ablation.

---

## The ablation protocol (after you've picked the arm from the traceback)

The fault is reproducible at ~84 on a clean GPU, so single runs are now meaningful,
but keep **N=2–3 replicates** per cell (it was probabilistic earlier). Each cell:
`make gpu-preflight` → fixed seed → run to **≥150 steps** → record cleared/crash-step.
Change **one** factor from baseline (HIPBLASLT=0, GC=0.8, grad-ckpt `use_reentrant=False`,
bnb 0.49.2, sdpa, seq 1280, bs 1×16). Arms A–F in §6.1. **Stop-rule:** first arm
that clears 150 over all replicates becomes the new baseline; keep stacking.

Lead with the arm the traceback indicates. If it's a bnb frame, **arm A** (source
build for gfx1201) is the highest-value shot and matches the prior that bnb on
RDNA4 is the least-mature part of the stack.

---

## Environment / reproduction facts

- HW: AMD Radeon AI PRO R9700, gfx1201 (RDNA4, Wave32), 32 GB; Ryzen 9 5900X; 64 GB RAM; 2 TB disk (not a constraint).
- SW: ROCm 7.2, torch 2.11.0+rocm7.2, bitsandbytes 0.49.2, transformers 5.9.0, peft 0.19.1, trl 1.4.0, Python 3.12.
- Model: `unsloth/gemma-4-31B-it-unsloth-bnb-4bit` (pre-quantized NF4, ~19 GB; cached on box). Attn = SDPA (hardcoded `scripts/train_sft.py:221`).
- **Always `uv run --no-sync`** (a bare `uv run` reinstalls CUDA torch over ROCm torch).
- After ANY fault the GPU is dirty — `make gpu-preflight` (or `bash scripts/test_gpu_stack.sh --wait-clean 180`) before the next launch.
- wandb project `csen346-sft`. Key runs: diagnostic `h9z6ebjd` (faulted @84, serialized); baselines `6c9fouit`/`p9t3yxe5` (@~84), `c6ye7mow` (lr5e-6, @~10); survival-patch `te8pgdbr` (@16, left `checkpoint-10`); eos-gates `g2df2ifl`/`5xd8qt5w` (finished 100), `gv9fbjac`/`irgdklt9`/`jtyyhu4t` (crashed). Training metrics log every 10 steps; system stats stream continuously.

## Open questions for you to resolve

1. **What kernel faults?** (the traceback — top priority).
2. **Is ~84 a fragmentation threshold or a kernel-specific trigger?** If arm D
   (alloc churn / fragmentation) and the traceback shows no clear kernel, test
   whether the fault step moves with `PYTORCH_HIP_ALLOC_CONF` changes.
3. **Does the cascade theory hold?** Confirm that early (10–20) crashes only happen
   on a dirty restart, and clean starts reliably reach ~84.
4. **Arm A viability:** can bitsandbytes be source-built for gfx1201 on this box
   (`cmake -DCOMPUTE_BACKEND=hip -DBNB_ROCM_ARCH=gfx1201`)? That's the likely fix if
   the traceback names a bnb kernel.
