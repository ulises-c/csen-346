# Handoff — gfx1201 upstream AMD bug triage (ROCm/rocm-libraries#7992)

**Date:** 2026-06-15 · **Branch:** `feat/gfx1201-rdna4-qlora-fla-training`  
**For:** a fresh Claude instance picking up the AMD upstream dialogue cold.

---

## TL;DR

We have a confirmed rocBLAS/Tensile bug on gfx1201 (RDNA4) that crashes QLoRA training via a GPU page fault. The upstream issue is filed and an AMD engineer (Nick) is assigned.

**Two findings (2026-06-15, this session) — read both, they point in different directions:**

**(1) Why our standalone probe never faults.** Comparing the **full** kernel name (not just the tile), the probe dispatches a *different contraction* than production:
- **Production faulting kernel:** `Cijk_Ailk_`**`Bjlk`**`_…_DTVB1_…_MT64x64x64`
- **Every standalone probe ever run:** only `Cijk_Ailk_`**`Bljk`**`_…_DTVB0_…_MT128x128x32` (fwd) and `Cijk_`**`Alik`**`_Bljk_…_MT128x128x32` (bwd)

`Bjlk` vs `Bljk` is a different ProblemType → different solution → different tile. Three constructions (bnb `Linear4bit`, plain bf16 `nn.Linear`, raw col-major-`[K,N]` matmul) all normalize to `Bljk`; PyTorch won't emit `Ailk_Bjlk` from a small script. **Takeaway: stop running PyTorch probe-variant ablations — they've all been on the wrong kernel.** This also corrects the prior AMD comment's "structurally identical BLAS calls" claim (the B contraction differs). NB: production is **stock HF transformers + PEFT-LoRA + bitsandbytes 4-bit + grad-ckpt** (no Unsloth *runtime*; only the pre-quantized `unsloth/…bnb-4bit` checkpoint loaded via `AutoModelForCausalLM`). The faulting shape in=21504/out=5376 is the MLP **`down_proj`**.

**(2) The contraction is NOT the bug.** Nick's force-dispatched kernel (quoted in his comment) is **byte-identical** to production's faulting kernel — `Ailk_Bjlk … DTVB1 … MT64x64x64 … VWA2_VWB1` (verified by diff). He runs the exact compiled binary with randomized inputs and it does **not** fault. Same kernel, fault vs. no-fault → **the discriminator is the invocation arguments (M / leading dims / pointers), not the kernel.** A wild ~1 GB-off address is dimension/stride arithmetic, not data-dependent.
- **M is the leading suspect, NOT ruled out** (but it's n=2: faults at M=607 and M=439; "odd" is plausibly coincidence). The earlier "M ruled out" sweep measured tile-selection on the *wrong* (`Bljk` MT128) kernel — we have **zero** clean M-sweeps on the faulting (`Bjlk` MT64) kernel. **Base pointer / allocation state co-varies** with M in every run, so it's not isolated either.
- **Fastest path to a real repro:** ask Nick for his force-dispatch script and feed it production's exact dims, sweeping M *and* varying allocation/base-address — bypasses the PyTorch-layout problem entirely.

**Training is intentionally stopped** — the GPU is idle and clean, available for experiments.

---

## NEXT-SESSION PLAN (2026-06-15) — what to run on the GPU before leaning on AMD

**Read this first if you're picking the work back up.** The lever: the **real production path emits the faulting kernel readily** (`Cijk_Ailk_Bjlk … MT64x64x64 DTVB1`, 1243×/3 steps via `scripts/profile_train_step.py`); every standalone `probe_gfx1201_tile.py` variant normalizes to `Bljk`/MT128 and **cannot** hit it. So **stop running probe-variant ablations** — drive the *real, loaded model* instead.

AMD already knows the kernel binary is byte-identical to one that doesn't fault for Nick. Only two things are left to narrow, and both are doable here:
1. **Which invocation arg discriminates fault vs no-fault** — M, or operand base-address/alignment (they co-vary in every capture so far).
2. **The exact arithmetic that computes the wild ~1.4 GB-off address** — so AMD can find the buggy index line.

### Ranked experiments

| # | Experiment | Why | Effort |
|---|---|---|---|
| **1** | **Clean M-sweep through the real model.** Load the production model once; drive `down_proj` at controlled M by feeding fixed-length synthetic token batches (M = seq len at bs=1) through fwd+bwd, looping M ∈ {438,439,440, 606,607,608,609,610}. | The experiment **never run on the right kernel** (handoff confirms "zero clean M-sweeps on the faulting kernel"). Gives the **right kernel + right ~19 GB VA footprint** at once — what the dense-VA repro lacked. A faulting M = near-standalone repro; no faulting M = implicates allocation state, also informative. | GPU, build harness |
| **2** | **Allocation-churn at fixed M=607.** Churn the allocator (alloc/free junk to shift operand base pointers) between calls at constant M. | The one knob that co-varies in every SYNC capture and has **never been isolated**. Fault toggling with base address at constant M → discriminator is alignment, not M. Pairs with #1 (same harness, second knob). | GPU, same harness |
| **3** | **Reverse-engineer the OOB arithmetic from the existing SYNC log.** Extract operand base ptrs + fault addr; factor the overshoot against lda/ldb/ldc/M·K and int32-overflow candidates. | **No GPU, free, points straight at the bug.** Seeds already computed (below). | Offline analysis |
| **4** | **Capture the literal kernarg of the faulting dispatch** (`AMD_LOG_LEVEL=4` or a kernarg dump of the UserArgs at the launch host-ptr). | Hands AMD the *exact* arguments (pointers, M, lds, alpha/beta, bias ptr, workspace) to replay byte-for-byte. Converts #3 from guesswork-on-tensor-ptrs into arithmetic on the kernel's real inputs. | GPU, ~25 GB log |
| **5** | **rocblas-bench retry to dispatch the kernel standalone.** `rocblas-bench` is installed (`/opt/rocm/bin`); retry with exact `transA=N transB=T`, bf16, production dims, checking the dispatched kernel. | If *any* invocation selects `ISA1201 MT64x64x64`, an M-sweep becomes a one-line standalone repro. Blocker: the fused `Bias…UserArgs` epilogue is hipBLASLt territory and `hipblaslt-bench` is **missing** (check if it can be built — likely Nick's exact tool). | GPU, low effort |
| **6** | **ROCm 7.3 / nightly Tensile test.** Does a newer build fix the `MT64x64x64_ISA1201` path? | Informs "just wait" vs not. Do **last** — risks the torch-env regression (guard with `make install-rocm` + `torch.version.hip` check). | GPU, heavy |

**Recommendation:** #1 + #2 in one harness (primary GPU experiment) and #3 + #4 as the cheap/offline deliverable — those four move AMD from "can't reproduce" to either a repro or a one-line arithmetic bug report. #5 is a cheap lottery ticket; #6 is the "should I wait" check.

### Tooling inventory (verified 2026-06-15)
- ROCm **7.2.3**.
- Present: `rocblas-bench` (`/opt/rocm/bin/rocblas-bench`).
- **Missing**: `rocprof`/`rocprofv2/v3`, `rocgdb`, `hipblaslt-bench`, `rocm-compute-sanitizer`, `roctracer`. (So #5's hipBLASLt-bench route needs a build/install first.)

### NEW FINDING — the SYNC log carries instrumented operand pointers up to the fault (seeds for #3/#4)

`docs/diagnostics/diag-sync-probe-step1230-20260605-124008.log` (25 GB) has custom `[gemm_forward_input]` / `[dequant_probe]` lines printing tensor ptr/shape/stride right before the faulting `ShaderName`. Last `down_proj` operands before the fault:

```
A (1,607,21504) bf16 row-major   base 0x7efad9200000  end 0x7efadaae5800   (M=607, K=21504)
B (21504,5376)  bf16 col-major   base 0x7ef978800000  end 0x7ef986480000   stride (1,21504)
fault address                    0x7efb33000000   (2 MB-aligned)
```

Arithmetic already run (`fault − base`):
- **fault − A = 0x59E00000 = +1.404 GiB = exactly 719 × 2 MB** above A (fault is *above* A's region, not inside it).
- **fault − B = 0x1BA800000 = +6.914 GiB** above B.
- Overshoot does **not** factor cleanly against lda/ldb/ldc/M·K/K·N from the **PyTorch tensor** ptrs (e.g. (fault−A)/es ÷ K = 35059 rem 17408). The exact-round `0x59E00000` (719 pages) smells like a miscomputed offset, not a data-dependent walk — **but the clean factoring likely needs the kernarg's *real* operand bases** (workspace/swizzle may differ from the tensor ptrs), which is exactly what #4 captures. To extract: `tail -c 6000000 <synclog> | grep -aE 'MT64x64x64|gemm_forward_input|dequant_probe|Memory access fault'`.

---

## The bug in one paragraph

On gfx1201 under ROCm 7.2, the Tensile GEMM kernel `MT64x64x64_ISA1201_DTVB1` (col-major B path, fused bias epilogue) computes a wild address ~1 GB outside both GEMM operands and page-faults. The fault is reproducible within a few steps of resuming Gemma-4-31B QLoRA training, deterministic when the data order is fixed (same `data_seed` → same fault step), and persists under full kernel serialization (`AMD_SERIALIZE_KERNEL=3 HIP_LAUNCH_BLOCKING=1`). No userspace workaround eliminates it. The standalone probe always dispatches `MT128x128x32` instead of `MT64x64x64`, so we cannot reproduce the fault outside the full training loop.

---

## Upstream issue

**URL:** https://github.com/ROCm/rocm-libraries/issues/7992  
**Assignee:** Nick Kuo (`amd-nicknick`)  
**Status:** Open, under triage

AMD's latest comment (Nick): He force-dispatches the exact `MT64x64x64_ISA1201_DTVB1` kernel via a hipBLASLt test script and runs it with randomized inputs — **no fault**. Concludes the kernel itself may be fine; the invocation context matters. Asks for a "reliable reproducible workload."

Corroborating reporter: `lainlives` — same crash (`MT64x64x64` on gfx1201), different hardware (9070 not R9700), different stack (pure PyTorch bf16, voice synthesis training, **no bitsandbytes**). This decouples the bug from our QLoRA/bitsandbytes stack entirely and makes it a broader gfx1201 rocBLAS issue.

---

## Complete experiment log (all sessions)

### Root cause identification (sessions 1–9)

| What | Result |
|---|---|
| AMD_SERIALIZE_KERNEL=3 HIP_LAUNCH_BLOCKING=1 | Fault persists → not async race |
| TORCH_USE_HIPBLASLT=1 | hipBLASLt has no kernel for this shape → falls back to same Tensile kernel → fault persists |
| PYTORCH_HIP_ALLOC_CONF tweaks (GC threshold, expandable_segments) | No effect (`expandable_segments` silently ignored on gfx1201) |
| Contiguous B (Python-level .contiguous()) | Ineffective — bitsandbytes sets transB flag at BLAS API level, .contiguous() doesn't reach the descriptor |
| System rocBLAS library vs torch-bundled | Both dispatch `MT128x128x32` for standalone probe; identical behavior → library path ruled out |
| 12 full ablation arms | All exhausted; zero userspace workarounds found |

### SYNC diagnostic run (session 8, 2026-06-05)

Ran `AMD_SERIALIZE_KERNEL=3 HIP_LAUNCH_BLOCKING=1 AMD_LOG_LEVEL=3` from checkpoint-1230, 26 GB / 230M-line log.

- Fault address: `0x7efb33000000` — 2 MB-aligned (consistent with all 12 prior crashes)
- Full ShaderName confirmed: `Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x64x64_MI16x16x1_SN_LDSB0_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB1_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_ISA1201_IU1_K1_LDSTI0_LBSPPA1024_LBSPPB0_LBSPPM0_LPA32_LPB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT2_2_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB2_ONLL0_PGR2_PLR1_PKA0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKXCCM0_TLDS0_ULSGRO0_USL1_UIOFGRO0_USFGROn1_VSn1_VWA2_VWB1_WSGRA0_WSGRB0_WS32_WG32_4_1`
- Last B-operand probe: shape=(21504,5376) col-major, M=**607** (not 608 as earlier async capture suggested)
- Second SYNC run: faulted at M=**439** (different data seed → different fault step → different M)

### Tile-selection probe grid (session 10, 2026-06-06)

9 variants at M=608 (pre-SYNC value), all using AMD_LOG_LEVEL=3:

| Config | Tile |
|---|---|
| 2D plain | MT128x128x32 |
| 3D plain | MT128x128x32 |
| 3D + ckpt(reentrant=False) | MT128x128x32 |
| 2D + ckpt(reentrant=False) | MT128x128x32 |
| 3D + ckpt(reentrant=True) | MT128x128x32 |
| 2D + autocast | MT128x128x32 |
| 2D + no warmup | MT128x128x32 |
| 2D + non-contiguous A | MT128x128x32 |
| 3D + system library (ROCBLAS_TENSILE_LIBPATH=/opt/rocm/...) | MT128x128x32 |

All 9 dispatch MT128x128x32. System library and bundled library are identical.

### Dense-VA reproducer (session 10, 2026-06-06)

Allocated 20 GB VRAM filler to match production process VA footprint; ran 200 fwd+bwd of `Linear4bit` at M=608, in=21504, out=5376, bias=True, nf4, bf16. DTVB1 dispatched 401 times — no crash, tile was MT128x128x32 each time. VA footprint is ruled out as the differentiator.

### rocblas-bench ROCBLAS_LAYER capture (session 10)

PyTorch routes through `libhipblas.so → librocblas.so` via internal dispatch, which **bypasses ROCBLAS_LAYER hooks**. `rocblas-bench gemm_ex` standalone dispatches `ISA000` (MLIR generic), not `ISA1201`. Dead end — no standalone rocblas-bench reproducer possible through this path.

### M-sweep (2026-06-15, this session)

Theory to test: SYNC logs show M=607 and M=439 as faulting values; prior probes all used M=608. Hypothesis: Tensile size predicate crosses MT128x128x32→MT64x64x64 at an M boundary near the faulting values.

Command run:
```bash
AMD_SERIALIZE_KERNEL=3 HIP_LAUNCH_BLOCKING=1 AMD_LOG_LEVEL=3 TORCH_USE_HIPBLASLT=0 \
  uv run --no-sync python scripts/probe_gfx1201_tile.py --shape-3d --ckpt \
  --m-sweep 439,512,576,606,607,608,609,640 2>&1 | tee probe_msweep.log
```

| M | Tile |
|---|---|
| 439 | MT128x128x32 |
| 512 | MT128x128x32 |
| 576 | MT128x128x32 |
| 606 | MT128x128x32 |
| 607 | MT128x128x32 |
| 608 | MT128x128x32 |
| 609 | MT128x128x32 |
| 640 | MT128x128x32 |

**Result: M is ruled out.** No M value in 439–640 dispatches MT64x64x64. The off-by-one hypothesis (M=607 vs M=608 crossing a tile boundary) is wrong.

### No-bias probe (2026-06-15, this session)

Theory to test: the faulting kernel includes `BH_Bias_UserArgs` epilogue flags. Without bias, a different (non-faulting) tile might be dispatched — or conversely, the bias epilogue might be the gating condition for MT64x64x64.

Command run:
```bash
AMD_SERIALIZE_KERNEL=3 HIP_LAUNCH_BLOCKING=1 AMD_LOG_LEVEL=3 TORCH_USE_HIPBLASLT=0 \
  uv run --no-sync python scripts/probe_gfx1201_tile.py --shape-3d --ckpt --m 607 --no-bias \
  2>&1 | tee probe_nobias.log
```

| Config | Tile |
|---|---|
| M=607, bias=True (production match) | MT128x128x32 |
| M=607, bias=False | MT128x128x32 |

**Result: bias epilogue is not the discriminator.** Both dispatch MT128x128x32 regardless of bias.

### Full-kernel-name diff + NT/col-major-B probes (2026-06-15, this session — the breakthrough)

Instead of ruling out one variable per run, compared the **complete** `Cijk_…` kernel name (every epilogue + contraction flag) the probe dispatched against production's. Source: `probe_msweep.log`, `probe_nobias.log` (existing), plus two new probe modes.

Contraction patterns across **all** probe logs ever produced:

| Source | Contraction patterns dispatched | Tile |
|---|---|---|
| Production fault (SYNC log + issue body) | `Cijk_Ailk_Bjlk … DTVB1` | **MT64x64x64** |
| All probe fwd (every variant) | `Cijk_Ailk_Bljk … DTVB0` | MT128x128x32 |
| All probe bwd (every variant) | `Cijk_Alik_Bljk … DTVB1` | MT128x128x32 |

`grep -hoE 'Cijk_A[a-z]+_B[a-z]+' probe_*.log | sort | uniq -c` → only `Ailk_Bljk` and `Alik_Bljk`. **Zero `Bjlk` lines in any probe log.**

Two new probe modes added to `scripts/probe_gfx1201_tile.py` to try to force `Bjlk`:

| Mode | Construction | Result |
|---|---|---|
| `--nt-linear` | plain bf16 `torch.nn.Linear` (no bitsandbytes; matches `lainlives`' stack) | still `Ailk_Bljk` / `Alik_Bljk`, MT128 |
| `--colmajor-b` | `x @ W` with `W` a col-major `[K,N]` view (stride `(1,K)`, non-contig — the NF4 dequant layout) | still `Ailk_Bljk` / `Alik_Bljk`, MT128 |

```bash
AMD_SERIALIZE_KERNEL=3 HIP_LAUNCH_BLOCKING=1 AMD_LOG_LEVEL=3 TORCH_USE_HIPBLASLT=0 \
  uv run --no-sync python scripts/probe_gfx1201_tile.py --colmajor-b --shape-3d --ckpt \
  --m-sweep 439,607,608 --iters 5 2>&1 | tee probe_colmajor_b.log
```

**Result:** torch normalizes both the NT-linear and col-major-B operands to the `Bljk` (standard `A·B`) contraction at the BLAS layer — a Python-level transpose/stride does not reach the descriptor as `Bjlk`. **The faulting `Ailk_Bjlk` contraction is not reproducible by any vanilla torch/bnb linear at this shape.** Producing it requires whatever production-specific code forms the matmul differently → **Unsloth-patched forward is the prime remaining suspect** (untested), and is now a sharply targeted experiment rather than a shot in the dark.

**Interpretation of the contraction labels (flag as interpretation, not measured):** `Bljk` = B logically `[K,N]` (the standard `A·B` / "NN" contraction); `Bjlk` = B logically `[N,K]` with the free-N index before the sum-K index (the `A·Bᵀ` / "NT" / transB path). Production faults on the NT contraction with the fused `Bias…UserArgs` epilogue; the probe only ever hits NN (fwd) and TN (bwd).

---

## Summary of what's confirmed vs unknown

**Confirmed:**
- `MT64x64x64_ISA1201_DTVB1_VWA2_VWB1` is the faulting kernel, data-order-deterministic, persists under full serialization
- Fault address is 2 MB-aligned, ~1–2 GB from both operands — consistent with wild computed address, not buffer overrun
- `MT128x128x32_ISA1201_DTVB1` runs 400+ iterations without fault — only MT64x64x64 is broken
- DTVB1 (col-major B, stride=(1,K)) is required; no userspace workaround can change this
- The standalone probe dispatches MT128x128x32 regardless of: M value (439–640), bias, shape (2D/3D), ckpt mode, autocast, warmup, non-contiguous A, VA footprint, library path

**Probe-vs-production tile divergence: explained (it was a red herring for the bug).**
The probe runs a **different contraction** than production (`Bljk` vs `Bjlk`), a different ProblemType → different tile. That explains why the probe never faults, and it means the whole probe-ablation track was on the wrong kernel. It does **not** explain the bug.

**The actual bug: same kernel, fault vs no-fault between production and Nick → invocation args.**
- `Ailk_Bjlk … MT64x64x64` faults in production but not in Nick's randomized-input force-dispatch of the *identical* binary.
- The only freedom left is M / lda / ldb / ldc / pointer alignment / alpha-beta.
- **M=607 and M=439** are the leading suspects (n=2; "odd" likely coincidence). Base pointers co-vary. The wild ~1 GB address is arithmetic on dims/strides/pointers, not data.

**Possibly relevant but secondary — what production code emits `Ailk_Bjlk`:** the MLP `down_proj` (in=21504, out=5376) via stock HF + PEFT-LoRA + bnb-4bit + grad-ckpt (NOT Unsloth runtime), and for `lainlives` some plain bf16 op in urvc. Useful only as a route to a PyTorch-level repro; the rocBLAS-level force-dispatch (Nick's script + production dims) is the better route.

---

## What to do next

### Option A — Post the corrected handover to AMD (#7992) [do this]

Draft ready: `docs/diagnostics/amd-comment-draft-contraction-20260615.md` — confirm with the human before posting (outward-facing). Core of it:
- Nick's force-dispatched kernel is **byte-identical** to production's (verified by diff) → he runs the faulting binary and it doesn't fault → the discriminator is **invocation args, not the kernel**.
- The wild ~1 GB address is dimension/stride arithmetic, not data → randomizing data at a fixed shape can't hit it.
- M=607/439 (both odd) are the variable that changes between faults → **ask Nick to sweep M (439/607/608/neighbors) rather than randomize data**, and to **share his force-dispatch script** so we can plug in production's exact dims/lds.
- Keep the `lainlives` `rocprof --kernel-trace` ask (independent no-bnb corroboration).

### Option B — Drive Nick's force-dispatch (or hipBLASLt/rocblas) at production dims, sweeping odd M [top experiment]

The bug lives at the rocBLAS layer with the right kernel + the right args. The PyTorch probe can't emit `Ailk_Bjlk` (framework normalizes layout), so don't keep fighting it. Instead force-dispatch the `Ailk_Bjlk … DTVB1 … MT64x64x64` kernel directly (Nick's script, or a hipBLASLt/rocBLAS call that selects it) with:
`transB, bf16, N=5376, K=21504, bias=True, alpha=1, beta=0`, **sweeping M ∈ {438,439,440, 606,607,608,609,610}** and, at a faulting M, **varying the operand base addresses** (churn the allocator). If a specific M or base offset produces the OOB address, that is the standalone repro.

### Option C — Trace which production op emits `Ailk_Bjlk` [DONE this session]

Ran 3 real steps of the production model (`unsloth/…bnb-4bit` via stock HF, `scripts/profile_train_step.py --steps 3`) under `AMD_LOG_LEVEL=3`, filtered to `probe_production_kernels.log`. Results:

| Contraction | Tile | Count (3 steps) |
|---|---|---|
| `Cijk_Ailk_`**`Bjlk`** | **MT64x64x64 DTVB1** (the faulting kernel) | **1243** |
| `Cijk_Ailk_Bjlk` | MT16x16x32 DTVB0 (small — LoRA adapter GEMMs) | 1243 |
| `Cijk_Ailk_Bljk` | MT128x128x32 | 21 |
| `Cijk_Alik_Bljk` | MT128x128x32 | 18 |

Two confirmations:
1. **The production path emits `Ailk_Bjlk` readily** (1243×/3 steps) — the contraction my standalone probes never produce. Producer = stock HF Gemma4 forward (MLP `down_proj`, in=21504/out=5376) + PEFT-LoRA, NOT a small isolated linear. PyTorch normalizes isolated linears to `Bljk`; the in-model dequant-weight call path keeps the `Bjlk`/DTVB1 layout.
2. **The faulting kernel ran 1243× WITHOUT faulting** in this slice → corroborates invocation-arg dependence, but **weakly**: at batch_size=1 all 62 layers share the same M, so 3 steps ≈ only ~3 distinct token-counts (repeated), none near 607/439. The *strong* kernel-exoneration evidence is Nick's randomized stress test + 2350 production steps (thousands of distinct M, mostly fine), not this run. Also, this fresh-tempdir run varied M **and** allocation state vs. the resumed-checkpoint production fault — so it can't isolate M from base-address.

(Did not re-capture operand strides — AMD_LOG_LEVEL=3 without SYNC doesn't emit them in the stream, and the issue body already lists them: lda=ldb=21504, ldc=5376, N=5376, K=21504, transB, bf16, bias.)

### Option C — ROCm nightly / newer build

ROCm 7.3 or a nightly might contain a Tensile fix for the MT64x64x64_ISA1201 path. Check the AMD ROCm release notes / changelog for anything related to gfx1201 GEMM.

```bash
# Check what's available
pip index versions pytorch-rocm 2>/dev/null  # or check PyTorch nightly
```

### Option D — Wait for AMD

Nick has the kernel. lainlives may provide a self-contained repro. AMD's in-house fix path does not require us to produce a standalone reproducer. The downstream workaround (rotating TRAIN_DATA_SEED per resume, crawl monitor) keeps training alive. Resume training and let AMD work the upstream.

---

## Training state

- **Latest checkpoint:** `outputs/sft-stage2-gemma4-31b/checkpoint-2350` (2026-06-08 01:56)
- **Progress:** 2350/4826 steps ≈ **48.7%** of epoch 1
- **Monitor:** NOT running. Last active 2026-06-08 02:18. Stopped after final crash at step 2350.
- **GPU:** Clean. Preflight passed at session start (VRAM 57 MB used, fwd+bwd OK).
- **HF backup:** `ulises-c/SocratesLM-31B-stage2b-QLoRA` — last push was at step 2350.
- **To resume crawl:** `nohup bash scripts/monitor_stage2.sh > outputs/monitor_stage2.log 2>&1 &`

---

## Relevant files

| Path | Purpose |
|---|---|
| `scripts/probe_gfx1201_tile.py` | Tile-selection probe (all variants) |
| `scripts/repro_gfx1201_dense.py` | Dense-VA standalone reproducer (no crash) |
| `probe_msweep.log` | M-sweep results (2026-06-15) |
| `probe_nobias.log` | No-bias probe results (2026-06-15) |
| `probe_nt_linear.log` | Plain bf16 nn.Linear probe — still `Bljk`, not `Bjlk` (2026-06-15) |
| `probe_colmajor_b.log` | Col-major `[K,N]` matmul probe — still `Bljk`, not `Bjlk` (2026-06-15) |
| `probe_2d_*.log`, `probe_3d_*.log` | Session 10 probe grid |
| `repro_dense.log` | Dense-VA run log |
| `docs/GFX1201_FAULT_ABLATION_LOG.md` | Full ablation log (all 12 arms) |
| `scripts/monitor_stage2.sh` | Crawl monitor (auto-resume with rotating seed) |
| `docs/HANDOFF_STAGE2_CRAWL.md` | Crawl operator guide |

---

## Key invariants — don't break these

1. `torch.version.hip` must not be None. If it is, run `make install-rocm`. This happened in sessions 6 and 7 (uv sync silently overwrites ROCm torch with CUDA torch).
2. `make gpu-preflight` before any GPU work to verify the KFD is clean.
3. `TORCH_USE_HIPBLASLT=0` in all probe runs — hipBLASLt falls back to Tensile for this shape anyway, and keeping it off makes dispatch transparent.
4. Never use `HSA_OVERRIDE_GFX_VERSION=11.0.0` — hangs the GPU on ROCm 7.2 (confirmed).
