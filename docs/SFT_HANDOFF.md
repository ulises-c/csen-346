# SFT Handoff — Gemma 4 12B Socratic QLoRA (NVIDIA PoC, eval phase)

**Updated 2026-06-22.** For a fresh session picking up the eval/compare work.
**Training is DONE (recovered — see below); the merge has run; eval has NOT started.**
This doc is the single source of truth for what to run next and the hazards. Cross-check
against `docs/EXPERIMENT_LOG.md` (newest at top) and the live logs on GitHub issue #130.

Branch: `feat/gemma4-12b-sft-poc-nvidia`. All code fixes from this phase are committed and
pushed (HEAD `66becca` + monitor `730906a` + live-quant `73c72e0`).

---

## The one-sentence goal

Measure whether Socratic QLoRA SFT on Gemma 4 12B improves the KELE eval over the **base**
Gemma 4 12B teacher, everything else fixed. Uplift = SFT − base on **state accuracy**.

## The number to beat

**State accuracy 50.30%** — base teacher, MTP-on, full Chinese test set (n=681)
(`results/gemma4-12b-base-mtp`). Run-to-run σ ≈ 0.7 pp, so an uplift is only real if it
clears **~1.5 pp**. Per-stage base (MTP-on): a 100.0 · b 44.55 · c 32.85 · d 35.98 · e 60.21.

---

## Tracking — where eval progress is recorded

- **GitHub issue #130** is the live tracker. The eval monitor (`monitor_eval_gemma4_12b.sh`)
  **auto-appends** progress/crash/complete rows to its pinned eval-log comment (`4644703104`).
  Do **not** post separate handoff/status comments on #130 — the only comments there are the
  monitor's train/eval log tables. This markdown doc is the handoff; #130 is the live log.
- **W&B** is the metrics tracker. Eval → project **`csen346-eval`** (org
  `uchavarria-santa-clara-university`), run auto-named after the output-dir basename
  (`gemma4-12b-sft-mtp`); per-dialogue metric curves log every 10 dialogues (`WANDB_EVAL_LOG_EVERY`).
  `WANDB_EVAL=1` is set by the monitored eval target. Training W&B (reference): `csen346-sft`,
  run `gemma4-12b-qlora-poc` (shows the NaN spike).

---

## ⚠️ What the SFT adapter actually is (read this — provenance matters)

The training run **diverged to NaN at step ~4260 / 4826 (epoch 0.88)** — a single bad batch
blew the loss to 1.5e8, grad → NaN, weights corrupted. `save_total_limit=5` then evicted every
clean **local** checkpoint before the NaN was noticed. **It was recovered from HuggingFace
commit history** (HF keeps all commits; local eviction is not mirrored). The recovered adapter:

- **`checkpoint-4250`** — step 4250, **epoch 0.881**, loss 0.6041, mean_token_accuracy 0.8052,
  verified **0/656 tensors NaN**. This is a well-converged **0.88-epoch** adapter (the missing
  12% was at LR decaying 6e-6→0, so negligible — but **report it as ~0.88 epoch, not "1 epoch",**
  in the experiment log so the result isn't overstated).
- Staged locally at **`outputs/sft-gemma4-12b-qlora/recovered-4250/`** (adapter_config.json +
  adapter_model.safetensors + tokenizer).
- Also on HF `ulises-c/SocratesLM-12B-QLoRA` at HEAD (NaN dirs 4300–4750 were deleted; HEAD is
  now clean `checkpoint-3200..4250`). Commit `38cdcc8299` is a known-good snapshot.

See memory `sft-nan-divergence-checkpoint-eviction` for the full story + the lesson
(check `list_repo_commits` before declaring a diverged run lost).

### Why training was stopped (not resumed/restarted)

Training was **manually stopped at step ~4800**, deep into a NaN run that began at ~4260. We did
**not** resume or restart because:
- checkpoint-4250 (the last clean step) is a **well-converged adapter** — loss had plateaued at
  ~0.60 since step ~3000 (epoch ~0.62), and the remaining ~12% of the epoch was at LR decaying
  6e-6→0, i.e. near-zero additional learning. So 0.88 epoch ≈ a full epoch for practical purposes.
- Resuming would have re-loaded a NaN local checkpoint (all surviving local checkpoints were NaN);
  resuming from the HF-recovered 4250 to finish 576 steps risked re-diverging for negligible gain.
- The PoC's question (does SFT beat the 50.30 base?) is answerable with checkpoint-4250 **now**;
  a cleaner full run can follow if the uplift is promising.

---

## Recommendations for a future / better SFT run

If the eval shows promise and a cleaner or stronger run is wanted, prioritize **stability first**
(the NaN divergence, not undertraining, was the failure):

**Stability (do these — this is what bit us):**
1. **NaN/inf-abort callback** — stop training the instant `grad_norm` or `loss` is non-finite
   (detect at `logging_steps=10`, which beats `save_steps=50`). Turns a silent 500-step NaN bleed
   into an immediate, recoverable stop. If it exits non-zero, `monitor_train_gemma4_12b.sh`
   relaunches from the latest checkpoint with a fresh `TRAIN_DATA_SEED` → steps over the bad batch.
2. **Protect checkpoints** — raise `save_total_limit` (currently **5** in `scripts/train_sft.py`,
   which evicted the last-good checkpoint before the NaN was noticed) to ~20, or keep a permanent
   checkpoint every N steps. HF history saved us this time; don't rely on it.
3. **Tighten `max_grad_norm`** to ~0.3 (HF default 1.0). The step before divergence had grad_norm
   2.247; tighter clipping buys margin against a grad-driven spike.
4. **Compute the loss/logits in fp32.** The likely root cause is an inf-logit *forward* spike over
   Gemma's 262k vocab in bf16 (divergence happened at LR≈6e-6, so it's not an LR-magnitude blow-up).
   An fp32 final-logits / cross-entropy path is the most direct fix; consider also a
   skip-non-finite-grad optimizer step for true single-bad-batch tolerance.
5. Optionally identify/quarantine the offending record (the batch near step 4260 under
   `data_seed=1782102499`) if divergence recurs across seeds.

**Learning rate / schedule:** current = `5e-5` linear decay, **no warmup**. LR magnitude probably
wasn't the trigger (divergence was late, at low LR), but adding `warmup_ratio≈0.03` and/or lowering
peak LR to **2–3e-5** is standard for 12B QLoRA and adds margin.

**Epochs:** this run was ~0.88 epoch and loss had already plateaued (~0.60 from step ~3000). More
epochs (2–3) are the natural lever for *more* SFT, but watch for overfitting on the per-turn set and
note each epoch is **~30 h** at 85 W (~29 s/step) on this box — let the eval uplift decide whether
additional epochs are worth the wallclock + instability exposure.

**Throughput / box:** ~29 s/step at the stable 85 W → ~30 h/epoch; the 20 GB card + power surge are
the bottleneck and instability risk grows with wallclock. A bigger/stable GPU would allow higher
power, a larger batch (currently 1×16), and shorter, safer runs.

---

## What is FIXED between base and SFT (do not change — it isolates the delta)

- **Consultant = Qwen3.5-0.8B LoRA state classifier** (`results/state-clf-qwen3.5-0.8b-lora-wandb/final`,
  HF `ulises-c/socrates-state-classifier-qwen3.5-lora`), passed via `--bert-consultant`, runs on
  **CPU** (`KELE_BERT_DEVICE=cpu`). Despite the "bert" naming, it's the Qwen3.5 classifier. The
  teacher is the ONLY thing that changes between base and SFT.
- **Bare teacher prompt** — no fewshot. Both base and SFT run bare.
- **Dataset**: Chinese-only test split, n=681 (`ulises-c/SocratDataset` default).
- **MTP = OFF for the SFT eval, run concurrent** (`KELE_PARALLEL_WORKERS≥4`). MTP is lossless but
  a single-stream latency win that loses on wall-clock to concurrency on this box — see Step 5's
  rationale. Sampling is stochastic (no temp/seed) so neither MTP nor batching biases the metric.
- **Base lineage = `unsloth/gemma-4-12b-it`** — NOT google. The base-teacher GGUF was
  `unsloth/gemma-4-12b-it-GGUF` and the adapter was trained on `unsloth/gemma-4-12b-it`, so the
  merge MUST use the unsloth base (the handoff's old `google/...` ref was wrong).

---

## The pipeline — what's done and what remains

### Step 2 — Train ✅ DONE (recovered, see above). Adapter = `recovered-4250/`.

### Step 3 — Merge LoRA → HF BF16 ✅ DONE (or finishing)

```
uv run --no-sync python scripts/merge_lora_gemma4_sft.py \
  --base unsloth/gemma-4-12b-it \
  --adapter outputs/sft-gemma4-12b-qlora/recovered-4250 \
  --out outputs/sft-gemma4-12b-qlora/merged
```

Output: `outputs/sft-gemma4-12b-qlora/merged/` (BF16, ~24 GB, CPU merge, ~10-15 min).
**Verify it finished**: `merged/` should hold `model-*.safetensors` totaling ~24 GB + config.
If `merge.log` shows "Done"/all shards written, it's complete; if interrupted, just re-run the
command (idempotent overwrite).

### Step 4 — Convert merged → Q8_0 GGUF (NOT YET RUN)

```
bash scripts/convert_gemma4_12b_sft_to_gguf.sh
```

Writes `gemma-4-12B-kele-socratic-sft-Q8_0.gguf` and stages it where the serve wrapper looks.
Q8_0 matches the base teacher's bit budget so the quant delta is noise. The script pre-flights
its llama.cpp deps (`~/Documents/models/llama.cpp`: `convert_hf_to_gguf.py` + `build/bin/llama-quantize`)
and fails loud if missing. **This step has not been exercised this phase — watch for missing
llama.cpp binaries** (memory `gemma4-12b-nvidia-poc-stack` flagged a possible build gap).

### Step 5 — Eval (MTP **off**, concurrent) + compare (NOT YET RUN)

```
make serve-gemma4-12b-sft                                   # NO MTP=1 → -np 4, q4_0 KV
KELE_PARALLEL_WORKERS=4 make monitor-eval-gemma4-12b-sft    # → results/gemma4-12b-sft
python -m src.project.evaluate --compare results/gemma4-12b-base results/gemma4-12b-sft
```

Sanity-gate first: `make eval-gemma4-12b-sft-smoke` (n=5, no monitor).

**Why MTP-off + concurrent (decided 2026-06-22 — supersedes the old "MTP on, serial" plan).**
The two base runs already on disk are a natural A/B:

| run | MTP | workers | throughput | wall-clock | state acc |
|---|---|---|---|---|---|
| `gemma4-12b-base-mtp` | on | 1 | 23.6 dlg/hr | 28.9 h | 50.30 |
| `gemma4-12b-base`     | off | 4 | 39.1 dlg/hr | **17.4 h** | 49.62 |

- **MTP and concurrency fight on this 20 GB box, so you pick one.** MTP forces f16 KV
  (`-ctk/-ctv f16`, 2× the q4_0 footprint) → OOM/checkpoint-bloat risk with >1 slot (the
  documented 2026-05-26 crash). Speculative decoding is a *single-stream latency* win; under
  continuous batching the GPU is already saturated so the draft+verify overhead competes and
  the gain collapses — worse here because the base-derived drafter has low acceptance on the
  SFT'd distribution. That's why the MTP run was (correctly) serial — and serial made it the
  *slower* wall-clock despite the "~2× faster" single-stream claim.
- **Neither MTP nor batching biases the metric.** The teacher call sets no temperature/seed
  (`tournament_utilizations.py:440` → server default ≈0.8, stochastic); both MTP and continuous
  batching are distribution-preserving. So the 0.68 pp gap between the two base runs is pure
  sampling noise (this is the σ≈0.7 pp estimate), and the SFT accuracy is comparable to **either**
  base regardless of speed config. We optimize purely for **min wall-clock at the stable 85 W
  point**, because instability exposure scales with wallclock on this box.
- **Concurrency is the live lever, not MTP** — but it plateaus fast. Smoke A/B on the Q8_0 SFT
  GGUF (n=18, same dialogues, `-np 6` server, 2026-06-23): **workers=4 → 226.0 dlg/hr**,
  workers=6 → 245.3 dlg/hr (+8.5% for +50% slots, 0 errors). The GPU is compute-bound at 4
  concurrent decodes, so the extra slots barely help and `-np 6` carries the documented
  KV-checkpoint-bloat crash risk over a 681-run. **Chosen: workers=4 / `-np 4`** (serve default).
  Clean-burst 226 dlg/hr ⇒ ~3 h of compute for the full 681 (the base's 17 h effective rate was
  inflated by crash/resume overhead, not raw speed). q4_0 KV gives headroom MTP's f16 KV wouldn't.

**Compare against BOTH bases.** The honest baseline band is **49.6–50.3**; gate the uplift on the
higher (50.30), so SFT must clear **~51.8** (>1.5 pp ≈ 2σ) to be real. A partial eval at n≈300 is a
valid early signal (convergence stable from n≈200-300).

**Optional MTP confirm run afterward.** If the concurrent SFT eval lands close to the gate and the
result hinges on it, a follow-up `MTP=1 … KELE_PARALLEL_WORKERS=1 make monitor-eval-gemma4-12b-sft`
(→ `results/gemma4-12b-sft-mtp`, serial, ~29 h) re-measures under the exact MTP-on config the
50.30 base used — removing any lingering "but the base was MTP-on" objection. Skip it if the
concurrent run is a clear pass or clear fail.

---

## Hazards & gotchas (read before launching)

1. **Box is KNOWN-UNSTABLE** (RTX 4000 Ada 20 GB, power surge → crashes under load). Eval crawls
   proved stable at **85 W**; the monitor (`scripts/monitor_eval_gemma4_12b.sh`) auto-resumes the
   eval, repairs error/truncated dialogues, polls server health out-of-band, logs to issue #130.
2. **GPU power step-down is INERT** — passwordless sudo is unavailable, so the monitors' power
   search can't change the limit; the card stays at whatever's persisted (85 W). Logged wattage in
   issue rows is cosmetic. Safe (85 W is the stable point). See memory `power-step-down-needs-passwordless-sudo`.
3. **One model fits at a time** on the 20 GB card. Stop the base server before serving SFT (both
   bind 8080). SFT serve uses alias "Gemma 4 12B SFT" so eval fails fast if it hits base weights.
4. **GGUF conversion (step 4) is the untested link this phase.** If it errors on a missing binary,
   build llama.cpp (cmake/gcc-vs-nvcc friction noted in memory). The base teacher is already served
   via llama-server, so llama.cpp is present — but the quantize/convert binaries may need building.
5. **SFT serve script** (`scripts/serve_gemma4_12b_sft.sh`) lacks the `GEMMA4_12B_KV` passthrough
   the base one has — irrelevant for the MTP=1 eval (MTP forces f16 KV anyway).
6. **The training monitor now fast-fails on OOM and self-heals on divergence-abort** — not needed
   for eval, but if you ever retrain: it relaunches with a fresh data_seed from the latest local
   checkpoint. To retrain cleanly, add a NaN-abort callback + raise `save_total_limit` (currently 5,
   which caused the eviction). The training-side OOM/eval-skip/live-quant fixes are committed.

---

## Where things live

- **Plan + live status**: GitHub issue **#130** (eval log comment `4644703104`; the training log
  comment `4761132326` holds the NaN-run history).
- **Recovered adapter**: `outputs/sft-gemma4-12b-qlora/recovered-4250/`; HF `ulises-c/SocratesLM-12B-QLoRA`.
- **Merged model**: `outputs/sft-gemma4-12b-qlora/merged/` (step 3 output, BF16 24 GB);
  HF `ulises-c/SocratesLM-12B` (private).
- **Q8_0 GGUF**: `outputs/sft-gemma4-12b-qlora/gemma-4-12B-kele-socratic-sft-Q8_0.gguf` (12 GB,
  step 4 output, staged to the serve weights dir); HF `ulises-c/SocratesLM-12B-GGUF` (private).
- **Configs**: `configs/gemma4-12b-sft-local.env` (SFT eval), `configs/gemma4-12b-local.env` (base eval),
  `configs/train-sft-gemma4-12b-qlora.env` (training).
- **Scripts**: `scripts/merge_lora_gemma4_sft.py`, `scripts/convert_gemma4_12b_sft_to_gguf.sh`,
  `scripts/serve_gemma4_12b_sft.sh`, `scripts/monitor_eval_gemma4_12b.sh`,
  `scripts/monitor_train_gemma4_12b.sh` (training babysitter, new this phase).
- **W&B**: training → `csen346-sft` (run `gemma4-12b-qlora-poc`); eval → `csen346-eval`.
- **Tests**: pre-commit runs ruff + pyright + codespell + shellcheck + full pytest (~9 s). Use
  `uv run --no-sync` for Python.
- **Relevant memories**: `sft-nan-divergence-checkpoint-eviction`, `power-step-down-needs-passwordless-sudo`,
  `gemma4-12b-nvidia-poc-stack`, `training-host-hardware-fault`, `use-uv-run-for-tests`.

## Definition of done

`results/gemma4-12b-sft/metrics_summary.json` exists with 681/681 valid dialogues; the
`--compare` output quantifies SFT − base state accuracy (real only if > ~1.5 pp); issue #130's
checklist is updated; and an `EXPERIMENT_LOG.md` entry records the uplift with per-stage breakdown
**and notes the adapter is a ~0.88-epoch checkpoint recovered from HF after a NaN divergence.**
