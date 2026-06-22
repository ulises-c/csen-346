# Handoff — gfx1201 page-fault investigation: consultant / reviewer brief

**Date:** 2026-06-01 · **Branch:** `feat/gfx1201-rdna4-qlora-fla-training` · **PR:** #101
**For:** a new Claude session acting as the **consultant / reviewer** (not the operator).

---

## Your role

The diagnostic phase is **closed** — root cause is confirmed and all software mitigations are
exhausted. You are the reviewer/consultant, continuing the "Consultant Review" thread on PR
#101. Your job is judgment, not re-running experiments:

- **Review before outward-facing actions.** The upstream bug filing is the big one — sanity-check
  the draft for over-claims before it leaves the building.
- **Weigh in on the open strategic fork** (crawl on-box vs. relocate to NVIDIA).
- **Catch errors / stale claims.** Hold the line between *observed* and *proven*.
- **Do NOT** re-open ruled-out hypotheses or re-run probes. Do NOT silently switch a conclusion
  the operator reached with evidence — surface the conflict instead.

## One-paragraph state

A reproducible GPU page fault ("page not present", host-VA ~1 GB from any operand) is caused by a
**rocBLAS Tensile GEMM kernel `…_MT64x64x64_…_DTVB1_…_ISA1201`** computing an out-of-bounds
address on a **column-major B** operand, on **gfx1201 / RDNA4 (R9700), ROCm 7.2.3**. It faults in
both the backward pass and the grad-checkpoint forward recompute (same tile), survives full
serialization (not a race), and reproduces on a clean GPU. **No userspace workaround exists** in
ROCm 7.2 — hipBLASLt has no kernel for this shape (falls back to Tensile), allocator/placement
knobs are no-ops, and the column-major B is intrinsic to the `A @ Wᵀ` call structure (can't be
changed from Python). The only fix is upstream.

## System of record — read these, in order

| File / location | What it is |
|---|---|
| `docs/GFX1201_FAULT_ABLATION_LOG.md` | **Canonical** run log (runs #1–#12), ruled-out table, pending arms. Source of truth — if anything disagrees with it, it wins. |
| Issue **#113** (latest comment) | Ready-to-paste **upstream draft** for `ROCm/rocm-libraries`. The issue body is the internal system-of-record summary. |
| `docs/SFT_NVIDIA_MIGRATION.md` | The conditional recommendation to relocate the SFT run to NVIDIA. |
| PR **#101** comments | Full diagnostic thread + Consultant Reviews #1–#N (your predecessors). |
| `docs/diagnostics/gfx1201-report-env-*.txt` | Captured OS/driver/ROCm/stack env for the report. Regenerate with `scripts/collect-gfx1201-report-env.sh`. |
| `docs/HANDOFF_GFX1201_BUCKET2.md` | Superseded snapshot — history only. |

## Settled — do NOT reopen (with the run that settled it)

- Async race (run #3, serialized still faults) · OOM (within VRAM headroom) · bnb NF4 dequant
  (succeeds every cycle) · undersized descriptor (fault ~1 GB from any boundary) · dirty-KFD
  cascade (reproduces clean) · allocator placement (run #10) · **hipBLASLt routing** (run #11 —
  no kernel for the shape) · **Python-level B layout** (run #12 — `.contiguous()` can't reach the
  BLAS `transB` flag) · **Bucket #1 / activation-lifetime** (probe-3 — both operands valid).
- The `Cannot find the function ×6` log lines are **benign fallback-probe spam** (rocm-systems#6624),
  not the bug.

## Open items needing judgment

1. **The goal fork (gating, unresolved — needs the human).**
   - *Adapter is the deliverable* (feeds the locked 72.24 / Tables 6/14) → relocate SFT to NVIDIA
     for a clean run; gfx1201 stays a documented contribution (#113 + #109).
   - *"31B QLoRA end-to-end on RDNA4" is the claim* → finish on-box via brute-force crawl.
   - Likely-non-binding caveat: data-residency (medical org) — but the data is HF-hosted Socratic
     dialogues, no PHI, so cloud is probably fine. **Confirm before committing either way.**
2. **Brute-force crawl viability is UNMEASURED.** The deterministic ~step-21 faults are a
   serialization artifact (SYNC + same ckpt-20); run #2 (no SYNC) was *probabilistic* (sometimes
   cleared 100 steps). Before committing to a crawl, measure **production steps-per-resume**
   (no SYNC, no probes) and price in the per-crash KFD-clean wait + lost-save risk. See PR #101
   consultant reviews for the corrected arithmetic (~4,800 steps/epoch, not the stale 2298).
3. **Upstream filing — one artifact left.** The `ROCBLAS_LAYER=2` `rocblas-bench` capture (the
   standalone reproducer) — also the *only* remaining way to confirm the column-major-B trigger
   (`--transposeB T` reproduces vs. `N`). Until captured, the report must say the trigger is
   **observed, not proven**. File at **`ROCm/rocm-libraries`** (rocBLAS = `projects/rocblas/`,
   Tensile = `shared/tensile/`); the standalone `ROCm/rocBLAS` & `ROCm/Tensile` repos are retired.
   Related upstream issues to cross-reference: **#6166** (identical fault in rocBLAS unit tests —
   strongest, AMD-runnable), #4097 (same kernel family, Windows), #7192 (same GPU, ruled out as
   our mechanism).

## Pending ablation arms (candidate mitigations, none run)

`HSA_ENABLE_SDMA=0` (**held back** — sign unknown; run as a *separate* arm only if the pure‑production baseline is marginal, not stacked on it) · `TRAIN_GRAD_CKPT=false`
(diagnostic, likely OOMs) · `PYTORCH_TUNABLEOP_ENABLED=1` (speculative — selects on speed not
correctness). `use_reentrant=True` is **de-prioritized** (targets the refuted Bucket #1).

## Operational gotchas

- **`uv run --no-sync`** always — bare `uv run` reinstalls CUDA torch over the ROCm build.
- **`make gpu-preflight`** before every launch — a prior rc=134 leaves the KFD dirty and cascades.
- Use **`python -m pytest`**, not `pytest` (not a console entry point; the post-test hook breaks on it).
- Confirm before outward-facing actions (the upstream filing, posting comments).

## What to do next

Probe work is done; the next move is the **human's call on the goal fork**. If asked to consult:
(a) review the #113 upstream draft for over-claims before it's filed; (b) advise crawl-vs-NVIDIA
against the fork above; (c) if staying on-box, make sure the `ROCBLAS_LAYER=2` capture runs — it
completes the report *and* confirms the trigger in one shot.
