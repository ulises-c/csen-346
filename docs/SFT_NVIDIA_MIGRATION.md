# Recommendation — Move Stage-2 SFT to NVIDIA (conditional)

**Date:** 2026-06-01 · **Branch:** `feat/gfx1201-rdna4-qlora-fla-training` · **PR:** #101
**Context:** Gemma-4-31B QLoRA SFT is blocked on the R9700 (gfx1201/RDNA4) by a confirmed
upstream rocBLAS/Tensile ISA1201 GEMM bug (wild address from a column-major B descriptor —
see `GFX1201_FAULT_ABLATION_LOG.md`, issue #113). All userspace routing/placement arms are
exhausted (hipBLASLt has no kernel for the faulting shape; allocator knobs are no-ops on
gfx1201). The remaining on-box path is a brute-force checkpoint crawl of unproven viability.

---

## Bottom line

If the deliverable is the **trained adapter**, move the SFT run to an NVIDIA GPU now. It is
the cheapest, fastest, lowest-risk path to a clean checkpoint, and it does **not** discard the
gfx1201 work — that survives as a documented contribution (#113 upstream report, #109 field
report). This recommendation flips only if "31B QLoRA end-to-end on RDNA4" is itself a paper
claim — see the gate below.

---

## 1. The decision is gated on two things, both resolvable in minutes

### Goal fork
Move only under **"the adapter is the deliverable"** (it feeds the SFT eval / the locked 72.24
baseline, Tables 6/14). If **RDNA4 viability** is the claim, do not move — but note you
**already hold** that contribution: the characterization in #113 + #109 substantiates "we ran
31B QLoRA on RDNA4 and root-caused the Tensile bug." A *completed* 94h gfx1201 run is not
required to make the field-report claim; the characterization is.

### Compliance / data-residency
Flagged because the org is medical, but it very likely **does not bind here**:
- Training inputs are the `socrat-zh-sft` / `socrat-en-sft` Socratic teaching dialogues loaded
  from HF; the base model is the HF-hosted `unsloth/gemma-4-31B-it-unsloth-bnb-4bit`.
- No PHI, nothing org-private leaves the box. Both are already public/HF-hosted, so a cloud GPU
  re-downloads them directly with **zero sensitive-data transfer**.

Confirm there is no private data in the pipeline. If confirmed, the residency objection
collapses and cloud is fully on the table.

---

## 2. Hardware options

| Option | VRAM | Cost for one ~94h run | Notes |
|---|---|---|---|
| **Own RTX 5090** | 32 GB | **$0** | CUDA → no Tensile bug. Fits the *same* config that fit R9700's 32 GB (keep grad-ckpt). Fastest to start. Caveat: prior training-host power-fault — verify box health first. |
| **Cloud A100 80 GB / L40S 48 GB** (recommended) | 48–80 GB | **~$140–235** (spot ~$1.5/hr → on-demand ~$2.5/hr × 94h) | Headroom to **disable grad-checkpointing** and raise batch → likely meaningfully <94h. Lambda / RunPod / Vast. |
| **Cloud H100 80 GB** | 80 GB | **~$280** (~$3/hr) | Fastest wall-clock; overkill but absolute cost is trivial for so short a run. |
| RTX 4090 / any 24 GB | 24 GB | — | **Too small** — run uses ~21 GB alloc / ~28 GB reserved. Don't. |

The full clean run costs **~$150–300 on cloud, or $0 on the 5090** — against ~48h of
engineering already spent on crash diagnosis and a 150h+ *uncertain* crawl. The economics are
not close.

> Note: 31B in bf16 (~62 GB weights) does not fit training even on 80 GB once optimizer state
> is added — **keep QLoRA/NF4** regardless of card. The win from a bigger card is headroom to
> disable grad-ckpt recompute and raise batch, not dropping quantization.

---

## 3. Migration effort: ~half a day, mostly *deleting* ROCm workarounds

The training code (`peft` / `trl` / `transformers` / `SFTConfig`, `scripts/train_sft.py`) is
hardware-agnostic — no logic change. The move is subtractive:

- **Drop** every ROCm env workaround: `TORCH_USE_HIPBLASLT`, `PYTORCH_HIP_ALLOC_CONF`,
  `AMD_SERIALIZE_KERNEL`, `HIP_LAUNCH_BLOCKING`, `HSA_ENABLE_SDMA`, and the `make gpu-preflight`
  / KFD-clean dance.
- **Remove** the `BNB_DEQUANT_PROBE` instrumentation (`scripts/train_sft.py:~547`) — its job is
  done.
- **Swap** torch+rocm → torch+cu12, install the **CUDA `bitsandbytes`** (the reference build —
  none of the gfx1201 bnb grief). The `uv run --no-sync` caveat *flips in your favor*: CUDA
  torch is now the default, not the thing you fight to keep.
- **Optionally enable** `attn_implementation="flash_attention_2"` (FA2 is first-class on
  NVIDIA; the R9700 path used SDPA).

---

## 4. What you gain beyond "it doesn't crash"

- **One clean run** — eliminates the entire crawl apparatus *and* the LR-schedule-consistency
  risk that multi-resume introduced (a single run has one continuous schedule by construction).
- **Likely faster wall-clock** — with 48–80 GB you can disable grad-checkpoint recompute and/or
  raise batch; QLoRA on H100/A100 should beat the R9700's ~70s/step. **Verify with a 20-step
  smoke run** before committing the full run — do not take this on faith.
- **Scientific validity intact** — a different GPU yields bitwise-different weights, but the
  adapter is *evaluated* downstream; as long as the eval harness is held constant, hardware
  choice does not compromise the result.

---

## 5. What you must preserve when leaving gfx1201

Moving the *run* does not abandon the *findings*:
- Keep #113 (upstream rocBLAS report) and #109 (field report) as the gfx1201 deliverable.
- Add a one-line note to `GFX1201_FAULT_ABLATION_LOG.md` that the SFT run was relocated to
  NVIDIA for the adapter while the Tensile bug is upstream — so the log closes cleanly rather
  than looking abandoned.

---

## 6. Recommended sequence

1. Confirm the goal fork + no-private-data (minutes).
2. Provision the box (5090 if healthy, else cloud A100 / L40S), install CUDA torch + bnb, pull
   model + datasets from HF.
3. **20-step smoke run** — confirm no fault, record s/step, confirm loss in-family.
4. **EOS/generation gate** on a short checkpoint (the #94 trust gate) — *still mandatory
   regardless of hardware*.
5. Launch the single clean run; standard checkpointing as insurance, not as a crawl.
