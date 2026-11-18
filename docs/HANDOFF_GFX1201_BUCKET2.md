# Handoff — gfx1201 ISA1201 Tensile GEMM: all fix arms exhausted, crawl phase

> **⚠️ SUPERSEDED (2026-06-01).** This is a point-in-time snapshot from when Bucket #2 was
> first confirmed and its "YOUR TWO TASKS" (forward kernel name + `expandable_segments` test)
> were still pending — **both are now done** (runs #9 and #10), and runs #11/#12 followed.
> For the current state and the live consultant/reviewer brief, see
> **`docs/HANDOFF_GFX1201_CONSULTANT.md`**; the canonical run log is
> **`docs/GFX1201_FAULT_ABLATION_LOG.md`**. Kept below as history.

**Date:** 2026-06-01 · **Branch:** `feat/gfx1201-rdna4-qlora-fla-training` · **PR:** #101
**For:** a fresh Claude instance picking up from a confirmed root cause.

---

## TL;DR — where we are

All software fix arms are exhausted. No userspace workaround is available on gfx1201 / ROCm 7.2.
The only remaining path to the adapter is the **brute-force checkpoint+resume crawl**.

**Before committing to the crawl**, one measurement is needed: **production steps-per-resume**
(no SYNC, no probes). This is the input to the crawl go/no-go arithmetic.

GPU is **clean**. Branch is **clean and pushed**. No training is running.

---

## Root cause — fully closed (do not re-investigate)

ISA1201 Tensile GEMM `MT64x64x64` computes a wild address from its column-major B operand
descriptor. Occurs in both the grad-checkpoint recomputed forward GEMM and the backward GEMM.
The col-major B layout (`DTVB1`, `stride=(1,21504)`) is intrinsic to how PyTorch+bitsandbytes
structures the 4-bit matmul call — it cannot be changed from userspace without modifying
bitsandbytes internals.

**Faulting kernel (full ShaderName):**
```
Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x64x64_MI16x16x1_SN_LDSB0_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB1_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_ISA1201_IU1_K1_LDSTI0_LBSPPA1024_LBSPPB0_LBSPPM0_LPA32_LPB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT2_2_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB2_ONLL0_PGR2_PLR1_PKA0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKXCCM0_TLDS0_ULSGRO0_USL1_UIOFGRO0_USFGROn1_VSn1_VWA2_VWB1_WSGRA0_WSGRB0_WS32_WG32_4_1
```

---

## All fix arms — DONE

| Run | Arm | Result |
|---|---|---|
| #10 | Arm D — `expandable_segments:True` | silently ignored on gfx1201; fault persists |
| #11 | Arm HLT1 — `TORCH_USE_HIPBLASLT=1` | hipBLASLt has no `MT64x64x64 DTVB1` kernel; falls back to Tensile; fault persists |
| #12 | Contiguous-B — `BNB_FORCE_B_CONTIGUOUS=1` | Python `.contiguous()` doesn't reach BLAS descriptor; `DTVB1` intrinsic to call structure; fault persists |

Full evidence in `docs/GFX1201_FAULT_ABLATION_LOG.md`. Do not re-investigate any of these.

---

## YOUR ONE TASK — production steps-per-resume measurement

### Why this is needed

All SYNC-mode runs (AMD_SERIALIZE_KERNEL=3 HIP_LAUNCH_BLOCKING=1) fault deterministically
at step 21 (first step after ckpt-20 resume). **SYNC mode is not representative.** Run #2
showed the fault is probabilistic without serialization — same SHA sometimes clears 100 steps.
The crawl viability depends entirely on how far production mode actually gets.

### Launch command (run #13)

```bash
make gpu-preflight    # MUST pass

CKPT=outputs/sft-stage2-gemma4-31b
nohup env TORCH_USE_HIPBLASLT=0 \
  HSA_ENABLE_SDMA=0 \
  AMD_LOG_LEVEL=1 \
  PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8 \
  TRAIN_BASE_MODEL=unsloth/gemma-4-31B-it-unsloth-bnb-4bit TRAIN_PREQ=true \
  TRAIN_SAVE_STEPS=5 TRAIN_MAX_STEPS=300 TRAIN_OUTPUT_DIR="$CKPT" \
  uv run --no-sync python scripts/train_sft.py --config configs/train-sft-stage2-gemma4-31b.env \
  > "$CKPT/prod-measure.log" 2>&1 &
echo "PID: $!"
```

Key differences from all prior diagnostic runs:
- **No `AMD_SERIALIZE_KERNEL` / `HIP_LAUNCH_BLOCKING`** — production mode, probabilistic fault
- **`HSA_ENABLE_SDMA=0`** — free stackable (Arm E), low risk, may help
- **`TRAIN_SAVE_STEPS=5`** — checkpoint frequently so progress is banked
- **`TRAIN_MAX_STEPS=300`** — run long enough to observe multiple fault cycles if they happen

### What to watch for

Monitor `prod-measure.log` for:
- `{'loss': ...}` lines — each is one step completed
- `Memory access fault` — a fault event; count the step from the last `{'loss'` line
- After each fault: **immediately run `make gpu-preflight`** before restarting (dirty KFD cascades)

If it faults, note the step number, then resume:
```bash
make gpu-preflight
nohup env ... uv run --no-sync python scripts/train_sft.py \
  --config configs/train-sft-stage2-gemma4-31b.env \
  > "$CKPT/prod-measure-resume-1.log" 2>&1 &
```

Collect 3–5 crash cycles. Record: step-at-fault for each crash.

### Go/no-go arithmetic (corrected per consultant review)

- **Total steps needed:** ~4,800 (1 epoch at batch=1×16=16, 12,244 records / 16 ≈ 765 steps/epoch × 3 epochs... wait, check the config. `TRAIN_MAX_STEPS` in `configs/train-sft-stage2-gemma4-31b.env` governs this — verify the actual value.)
- **Per-crash wall-clock cost** = KFD-clean wait (~2–5 min) + 19 GB model reload (~2–3 min) + preflight. Budget ~8–10 min/crash, not just reload time.
- **Checkpoint corruption risk:** at `save_steps=5` near a fault step, a save in progress during a fault may produce a corrupt checkpoint. Guard against this: after each crash, verify the latest checkpoint is loadable before resuming.

**Crawl viable if:** mean steps-per-crash is high enough that `(total_steps / mean_steps) × 10min/crash` is acceptable wall-clock overhead.

**Crawl NOT viable if:** production faults every 5–15 steps consistently — that's 320–960 crash cycles for a full run, 53–160h of pure overhead. In that case an upstream fix is mandatory and the adapter cannot be produced on this hardware without it.

---

## After the measurement — update ablation log + post PR #101

Once you have 3–5 crash datapoints:

1. Update `docs/GFX1201_FAULT_ABLATION_LOG.md` — add run #13 row (or multiple rows for each resume)
2. Post to PR #101 with: observed fault steps, mean steps-per-crash, projected wall-clock, and go/no-go verdict
3. If go: launch the full crawl with `TRAIN_MAX_STEPS` unset (or set to full epoch count), `TRAIN_SAVE_STEPS=5`, `HSA_ENABLE_SDMA=0`
4. If no-go: post to PR and document that on-device training is blocked pending upstream fix

---

## Upstream bug report — status

The rocBLAS/Tensile upstream filing has all necessary info:
- Full ShaderName (above)
- Operand descriptors (A: `shape=(1,608,21504)` contig row-major; B: `shape=(21504,5376)` col-major `stride=(1,21504)`)
- Fault addr: `0x7f6459a00000` (~1 GB below A.ptr, ~1.2 GB above B.end)
- Reproduced with `AMD_SERIALIZE_KERNEL=3 HIP_LAUNCH_BLOCKING=1` (not a race)
- Architecture: gfx1201 (RDNA4/Navi 48), ROCm 7.2, torch 2.11.0+rocm7.2
- Environment report: `docs/diagnostics/gfx1201-report-env-20260601-191024.txt`
- File at: https://github.com/ROCm/rocm-libraries/issues (component: rocBLAS / Tensile)
- Note for filing: col-major B is intrinsic to PyTorch+bitsandbytes 4bit matmul; no userspace fix

If the filing hasn't been submitted yet, do it before or alongside the measurement run — it's
independent and doesn't block anything.

---

## Environment

- HW: AMD Radeon AI PRO R9700, gfx1201 (RDNA4), 32 GB VRAM; Ryzen 9 5900X
- SW: ROCm 7.2, torch 2.11.0+rocm7.2, bitsandbytes 0.49.2, transformers 5.9.0, peft 0.19.1, trl 1.4.0, Python 3.12
- Model: `unsloth/gemma-4-31B-it-unsloth-bnb-4bit` (pre-quantized NF4, ~19 GB, cached)
- **Always `uv run --no-sync`** — bare `uv run` reinstalls CUDA torch over ROCm torch
- After ANY fault: `make gpu-preflight` before next launch
- Latest checkpoint: `outputs/sft-stage2-gemma4-31b/checkpoint-20/` (step 20)
- Training config: `configs/train-sft-stage2-gemma4-31b.env`

## Key files

| File | Purpose |
|---|---|
| `docs/GFX1201_FAULT_ABLATION_LOG.md` | Canonical run log — update after every run |
| `scripts/train_sft.py` | Training script; `BNB_DEQUANT_PROBE` + `BNB_FORCE_B_CONTIGUOUS` blocks at line ~547 — leave in, gated by env var |
| `docs/diagnostics/gfx1201-report-env-20260601-191024.txt` | Env report for upstream filing |
| PR #101 comments | Full diagnostic thread + all operator findings |

## Commit history (this branch, session 3)

```
0c763e0  new gfx1201 report generated
4167d84  chore(diag): env-collector writes to docs/diagnostics/; trim output noise
afa46e0  gfx1201 report generated
78e224d  feat(diag): BNB_FORCE_B_CONTIGUOUS probe + ablation log runs #11/#12
154aa8b  feat(diag): env-collector script for the gfx1201 rocBLAS bug report
```
