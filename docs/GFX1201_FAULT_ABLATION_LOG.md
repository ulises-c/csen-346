# gfx1201 QLoRA Backward Page Fault — Ablation & Run Tracker (PR #101)

Canonical, append-only log of every diagnostic/ablation run for the Gemma‑4‑31B QLoRA
backward page fault on the AMD R9700 (gfx1201 / RDNA4, ROCm 7.2). One row per run.
**Update this file whenever a run completes** — it is the system of record; PR comments
and wandb are the raw sources.

> Companion: `docs/GFX1201_RDNA4_TRAINING.md §6.1` holds the narrative + original ablation
> plan. Where the two disagree, **this file wins** — §6.1 was written before the fault was
> localized and still frames it as bnb‑side (now ruled out, see below).

---

## Current root‑cause state (2026‑06‑01) — CLOSED

The fault is a GPU page fault ("page not present", 2 MB‑aligned host‑VA `0x7f…` address)
during the **QLoRA backward pass**, specifically the **gradient‑checkpoint forward recompute**.

**Root cause confirmed (Bucket #2, probe‑3 + kernel‑name runs):**

- **Faulting kernel (full ShaderName):**
  `Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x64x64_MI16x16x1_SN_LDSB0_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB1_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_ISA1201_IU1_K1_LDSTI0_LBSPPA1024_LBSPPB0_LBSPPM0_LPA32_LPB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT2_2_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB2_ONLL0_PGR2_PLR1_PKA0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKXCCM0_TLDS0_ULSGRO0_USL1_UIOFGRO0_USFGROn1_VSn1_VWA2_VWB1_WSGRA0_WSGRB0_WS32_WG32_4_1`
  Same kernel variant in both forward recompute AND backward pass. Confirmed on clean GPU
  (runs #5, #9). Key discriminating flags: `DTVB1` (B uses different dtype/layout),
  `LBSPPB0` (no B LDS prefetch), `NLCB2` (B double‑unrolled), `VWA2_VWB1`.
- **bitsandbytes is NOT the faulting op.** Dequant kernels succeed every cycle.
- **Bucket #2 — wild address confirmed (probe‑3, run #8):**
  Both GEMM operands logged; neither brackets fault 0x7f6459a00000.
  A=(1,608,21504) contig row‑major `ptr=0x7f655ece8000 end=0x7f65605d8000`;
  B=(21504,5376) col‑major `stride=(1,21504) ptr=0x7f63f01a0000 end=0x7f63fde20000`.
  Fault is ~1 GB below A.ptr and ~1.2 GB above B.end. Full‑log scan: 0 operands bracket fault.
  **The ISA1201 Tensile kernel computes a wild address from the column‑major B descriptor.**

~~Bucket #1 (freed/recycled operand) — eliminated by probe‑3.~~

---

## Run log

Legend: **GPU=clean** means `make gpu-preflight` PASSed immediately before launch (else a
dirty‑KFD cascade can fault early and confound the result). All runs use baseline unless noted:
`TORCH_USE_HIPBLASLT=0`, `PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8`,
`grad_ckpt=use_reentrant:False`, bnb 0.49.2, SDPA, seq 1280, bs 1×16, seed 42.
`SYNC` = `AMD_SERIALIZE_KERNEL=3 HIP_LAUNCH_BLOCKING=1`. wandb IDs marked `?` need confirming on the box.

| # | Run / log | wandb | Code | Start | Key vars | GPU | Fault step | Faulting kernel | What it established |
|---|---|---|---|---|---|---|---|---|---|
| 1 | historical `train.log` | various | pre‑diag | fresh@0 | baseline, lvl1 | unknown | ~84 | not named | baseline fault exists |
| 2 | eos‑gate set | `g2df2ifl` `gv9fbjac` `5xd8qt5w` `irgdklt9` | SHAs `c752a2b4`,`eb12dbd9` | fresh@0 | baseline | unknown | 10 / 84 / clears 100 | not named | **probabilistic** — same SHA both finishes & crashes |
| 3 | `gfx1201_fault_2026-06-01.log` | `h9z6ebjd` | `8580844` | fresh@0 | **SYNC**, lvl1 | unknown | 84 | not named (lvl1) | **async race RULED OUT** (serialized still faults) |
| 4 | `diag-l3-resume.log` | `akfrsb3z` | no probe (≈Review#1 cmd) | ckpt‑10 | SYNC, **lvl3** | **dirty** (no preflight) | 22 | **`MT64x64x64 ISA1201`** | named the kernel (but cascade‑confounded) |
| 5 | `clean-repro-1.log` | `001pnijf` | no probe | ckpt‑20 | SYNC, lvl3 | **clean** | 21 | `MT64x64x64 ISA1201` | fault is **real, not a cascade**; bnb dequant succeeds |
| 6 | `probe-1.log` | `i6m2e3sx`? | probe v1 `81cebb4` | ckpt | SYNC, lvl3, `BNB_DEQUANT_PROBE` | clean | ~mid | `MT64x64x64` | fault **1.5 GB from dequant buffer** → "bnb descriptor too small" **RULED OUT** |
| 7 | `probe-2.log` | ? | probe v2 `1b752c2` | ckpt | SYNC, `BNB_DEQUANT_PROBE` (+grad_output) | clean | ~mid | `MT64x64x64` | 8 dequants, no `gemm_operandA` → fault is in **forward recompute**; A‑operand not logged → bucket inconclusive |
| 8 | `probe-3.log` | ? | `dad8f4a` | ckpt‑20 | SYNC, lvl1, `BNB_DEQUANT_PROBE` (+forward input) | clean | ~21 | not named (lvl1) | **Bucket #2 confirmed** — both operands logged & valid, fault 0x7f6459a00000 ≈1 GB from either buffer; col‑major B descriptor → wild address. Bucket #1 **eliminated**. |
| 9 | `kernel-name.log` | N/A | `b6cb557` | ckpt‑20 | SYNC, **lvl3**, no probe | clean | ~21 | **`MT64x64x64_ISA1201_DTVB1_VWA2_VWB1`** (full name above) | **Forward faulting kernel ShaderName confirmed** — identical to backward (run #5). Both recomputed‑forward and backward GEMMs use same Tensile tile. All info ready for upstream report. |
| 10 | `expandable-seg.log` | N/A | `a70a2d1` | ckpt‑20 | SYNC, lvl1, `expandable_segments:True` (ignored) | clean | ~21 | `MT64x64x64 ISA1201` (lvl1, inferred) | **Arm D — fault persists**, addr `0x7f29eb600000`. `expandable_segments` unsupported on gfx1201 (silently ignored). Placement change cannot mask fault → placement irrelevant, confirms pure kernel stride bug. |
| 11 | `hlt1.log` | `e6srgn90` | `ed92b93` | ckpt‑20 | SYNC, lvl1, **`HIPBLASLT=1`** | clean | ~21 | `MT64x64x64 ISA1201` (Tensile fallback) | **Arm HLT1 — fault persists**. hipBLASLt looked up `MT64x64x64_DTVB1` and returned `Cannot find the function` for 6 modules. Tensile dispatched the same bad ISA1201 kernel, addr `0x7f0ac2e00000`. gfx1201 hipBLASLt has no kernel for col‑major B (`DTVB1`) at this tile. Routing fix unavailable in ROCm 7.2. |
| 12 | `contiguous-b.log` | N/A | `9a2ea38` | ckpt‑20 | SYNC, lvl1, **`BNB_FORCE_B_CONTIGUOUS=1`** | clean | ~21 | `MT64x64x64 ISA1201 DTVB1` | **Contiguous‑B — fault persists**. Hook confirmed active. Forced `.contiguous()` on `dequantize_4bit` output (stride `(1,21504)→(5376,1)` at Python level). `DTVB1` still dispatched — descriptor is set by PyTorch BLAS call for `A @ W.T`, not by the Python tensor's physical stride. Python‑level copy cannot reach the BLAS transpose flag. **Col‑major B is intrinsic to the 4bit matmul structure**; no Python‑level intervention possible. Fault addr `0x7f6f48c00000`. |

---

## Pending ablation arms — each is a candidate **fix**, judged on two axes

**Axis A:** did it *eliminate* the fault? (→ clean ~94 h run, crawl unnecessary).
**Axis B:** did it *improve steps‑per‑resume*? (→ brute‑force crawl becomes cheap).
Protocol: `make gpu-preflight` → fixed seed → run ≥150 steps (or to fault) → record both axes.
Change exactly **one** factor from baseline.

| Arm | One change | Targets | Axis A result | Axis B result | Status |
|---|---|---|---|---|---|
| ~~HLT1~~ | ~~`TORCH_USE_HIPBLASLT=1`~~ | ~~#2 (route GEMM off Tensile → hipBLASLt)~~ | fault persists (run #11) | N/A | **DONE** — hipBLASLt has no `MT64x64x64 DTVB1` kernel for gfx1201; falls back to Tensile, same fault. |
| ~~contiguous‑B~~ | ~~`BNB_FORCE_B_CONTIGUOUS=1`~~ | ~~#2 (change B layout before GEMM)~~ | fault persists (run #12) | N/A | **DONE** — Python `.contiguous()` on dequant output doesn't reach BLAS descriptor; `DTVB1` intrinsic to `A @ W.T` call structure. |
| **C** | grad‑ckpt `use_reentrant=True` | #1 (activation lifetime) | | | pending — one‑line, run after probe‑3 if #1 |
| **D** | `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` | placement | fault persists (run #10) | N/A | **DONE** — `expandable_segments` silently ignored on gfx1201 (PyTorch warns at startup); fault at ~step 21 addr `0x7f29eb600000`. Confirms: placement irrelevant, fault is kernel stride bug. |
| **E** | `HSA_ENABLE_SDMA=0` | DMA page‑fault mitigation | | | **HOLD** — sign unknown (changes the SDMA copy path; could move the fault rate either way). Do **not** stack on the viability baseline. Run as a *separate* arm only if pure‑production is marginal. |
| **B** | `TRAIN_GRAD_CKPT=false` | confirms recompute drives it | | | pending — likely **OOMs** at 98 % VRAM (diagnostic, not a fix) |
| ~~A~~ | ~~bnb source build `-DBNB_ROCM_ARCH=gfx1201`~~ | ~~bnb kernel~~ | — | — | **DROPPED** — probe‑1 ruled out bnb as the faulting op |
| TUN | `PYTORCH_TUNABLEOP_ENABLED=1` | #2 (pick different GEMM) | | | speculative fallback — selects on **speed not correctness**; may re‑pick the bad kernel or crash during tuning |

---

## Ruled out — do not re‑investigate (with the run that settled it)

| Hypothesis | Status | Evidence |
|---|---|---|
| Async / concurrency race | **ruled out** | run #3 — serialized run still faults |
| Numerical / data / bad batch | ruled out | §6.1 — smooth loss, byte‑identical per‑step losses |
| LR / optimizer magnitude | ruled out | §6.1 — lr 5e‑6 crashed *earlier* than 5e‑5 |
| Sequence length / `max_length` | ruled out | §6.1 — real token max 909 < 1024; non‑binding |
| bnb NF4 backward kernel is the faulting op | **ruled out** | runs #5–7 — dequant succeeds; fault is in the Tensile GEMM. **Supersedes §6.1's "fault is in the bitsandbytes‑NF4 path" and removes Arm A.** |
| bnb hands the GEMM a too‑small descriptor | **ruled out** | run #6 — fault 1.5 GB from the dequant buffer, not adjacent to `end` |
| hipBLASLt is the **cause** | ruled out | §6.1 — `HIPBLASLT=0` still faults. **But:** =1 was **never tried as a *fix*** (see arm HLT1) — §6.1's "hipBLASLt ruled out" conflated cause with remedy |
| Dirty‑KFD cascade is the *true* fault | ruled out | run #5 — fault reproduces on a verified‑clean GPU |

---

## How to log a run (keep this file current)

After any run completes, append a row to the **Run log** (or fill an ablation arm's Axis
A/B) with: run/log filename, wandb id, commit hash (`git rev-parse --short HEAD`), resume
point, the one changed variable, GPU‑clean status, fault step, faulting kernel, and verdict.

Standard launch (adjust the one variable under test; keep `AMD_LOG_LEVEL=1` unless you need
to re‑name the kernel — level 3 writes multi‑GB logs):

```bash
make gpu-preflight                                   # MUST pass
COMMIT=$(git rev-parse --short HEAD)
CKPT=outputs/sft-stage2-gemma4-31b
nohup env TORCH_USE_HIPBLASLT=0 \
  AMD_SERIALIZE_KERNEL=3 HIP_LAUNCH_BLOCKING=1 AMD_LOG_LEVEL=1 \
  PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8 \
  TRAIN_BASE_MODEL=unsloth/gemma-4-31B-it-unsloth-bnb-4bit TRAIN_PREQ=true \
  TRAIN_MAX_STEPS=95 TRAIN_OUTPUT_DIR="$CKPT" \
  uv run --no-sync python scripts/train_sft.py --config configs/train-sft-stage2-gemma4-31b.env \
  > "$CKPT/run-$COMMIT.log" 2>&1 &
```

---

## Crawl viability measurement (run #13, DONE — STALLED)

**Result (2026-06-01):** crawl is NOT viable from checkpoint-20. 8/8 production-mode resume runs crashed at optimizer step 22 or 24 (within 4 steps of checkpoint-20 every time). No new checkpoint was banked. Monitor exited STALLED per pre-committed MAX_RETRIES=8 no-progress rule.

**N steps/crash: 2–4 (mean ≈ 3).** Pre-committed NO-GO threshold was N ≲ 10.

**Fault is data-order sticky:** HF Trainer restores `rng_state.pth` on resume, replaying identical batches → same samples at steps 21–23 every time → same GEMM shape → deterministic fault. Fault address varies (ASLR) but step is constant. Contrast with run #2 (fresh-from-0, probabilistic, sometimes clears 100 steps): different data trajectory, fault landed at different positions or not at all in 100 steps.

**Next experiment:** `ignore_data_skip=True` on resume — breaks dataloader-state restoration, converts sticky back to probabilistic. Directly predicted by the stickiness mechanism. If this recovers ~100 steps/crash (run-#2-like), arithmetic flips to GO.

---

## Crawl viability measurement (archived — go/no‑go pre‑committed)

With all fix arms exhausted (#11/#12) and the goal set to *31B QLoRA end‑to‑end on
gfx1201/RDNA4*, the only on‑box path is the brute‑force crawl. Before committing
wall‑clock, measure **production steps‑per‑crash** and **sticky‑vs‑advancing** under
the real crawl loop.

**Protocol — launch the pure‑production default** (no SYNC, no probes,
`TORCH_USE_HIPBLASLT=0`, `TRAIN_SAVE_STEPS=10`, **Arm E held back** — clean baseline):

```bash
make gpu-preflight                                   # MUST pass
nohup bash scripts/monitor_stage2.sh > outputs/monitor_stage2.log 2>&1 &
```

`monitor_stage2.sh` **is** the crawl harness: relaunch‑on‑crash (re‑runs gpu‑preflight),
KFD‑clean‑before‑relaunch (`test_gpu_stack.sh --wait-clean 180`), `quarantine_bad_checkpoint`
(partial‑save guard — resume falls back to N‑1), per‑crash log+dmesg archive.

**Two questions, in order:**

1. **Sticky vs. advancing (gating).** Does resume get *past* the fault step, or re‑fault at
   the same step forever (the M=608 data‑order risk)? Already operationalized: the monitor
   counts only **consecutive no‑progress** retries — a new checkpoint resets the counter;
   `MAX_RETRIES=8` with no new checkpoint → posts `STALLED` and exits = **crawl is dead**,
   upstream fix becomes mandatory even on the RDNA4 path.
2. **Wall‑clock viability (the number).** ~4,800 steps/epoch × ~70 s/step ≈ **93 h** pure
   compute. Each crash cycle costs the KFD‑clean wait + ~19 GB reload + preflight (~4–6 min).
   At a mean of **N steps/crash**, cycles ≈ 4,800 / N:
   - **GO** if N ≳ 25–30 (~160–190 cycles, ~10–19 h overhead on top of 93 h).
   - **NO‑GO** if N ≲ 10 consistently (~480+ cycles, 30 h+ overhead) or the monitor `STALL`s.

This threshold is pre‑committed (posted to PR #101) before the first crash so the
decision is a number, not a vibe.
