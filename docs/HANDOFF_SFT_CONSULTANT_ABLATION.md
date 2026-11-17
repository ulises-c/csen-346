# Handoff — SFT vs Base: Consultant Ablation (no-consultant + consultant-as-variable)

**Created 2026-06-25.** For a fresh Claude session. Branch `feat/gemma4-12b-sft-poc-nvidia`.
Read alongside `docs/SFT_RESULTS_REPORT.md` (results so far), `docs/SFT_VS_BASE_ANALYSIS_PLAN.md`
(full ablation menu — this handoff is plan item **T1.1**), and `docs/SFT_HANDOFF.md` (pipeline +
serve/box hazards). Live tracker: issue #130. Use `uv run --no-sync` for Python.

## The one-sentence goal

Every result so far gave **both** base and SFT the same external state classifier (the Qwen3.5
"consultant"), so we don't yet know whether the SFT's +7.7–10.3 pp gain is its **own** internalized
Socratic state-tracking or just better use of the classifier. **Run base vs SFT with the consultant
removed** (the model self-tracks state), and more broadly treat **the consultant as a variable**.

## Background you need (consultant modes)

`create_system` (`src/project/kele.py:30`) picks the teaching system by flag:

| mode | flag | who predicts the scored `state` | LLM calls/turn |
|---|---|---|---|
| **External classifier** (all runs so far) | `--bert-consultant <ckpt>` | Qwen3.5 LoRA classifier on **CPU** | 1 (teacher only) |
| **Self-consult / dual-role** ← *"no consultant"* | *(omit `--bert-consultant`)* | the **served LLM itself** (separate consultant call) | **2** (consultant + teacher) |
| **Unified** | `--unified` | one fused structured-output call | 1 |

- The scored `state_accuracy` compares `system.current_state` to ground truth. **Who sets
  `current_state` changes with the mode** — so the no-consultant runs measure the *teacher model's
  own* state classification, which is the whole point of the ablation.
- "No consultant" here means **no external classifier**, i.e. the dual-role `SocraticTeachingSystem`
  where the LLM self-classifies. There is no "no state-tracking at all" mode; the self-consult call
  is the apples-to-apples baseline (identical mechanism for base and SFT).

## ⚠️ Gotchas — read before launching

1. **~2× slower (GPU).** Self-consult does **two** served-LLM calls per turn (consultant + teacher)
   vs one with the CPU classifier. The base model already rambles to the 2048-token cap (≈38 dlg/hr
   with one call → ZH test took 17.4 h). **Base no-consultant could be ~30+ h.** Budget for it, or
   start with a partial n (e.g. `--limit 200`, which is a valid early signal — convergence is stable
   from n≈200–300) and only commit to full 681 if needed. SFT is ~6× faster (terminates cleanly).
2. **The monitor hardcodes `--bert-consultant`.** `scripts/monitor_eval_gemma4_12b.sh` `start_eval`
   always passes `--bert-consultant "$BERT_CKPT"`. To run the crash-resilient monitored path
   without it, add an env toggle (e.g. `NO_CONSULTANT=1` → drop the flag + suffix the out-dir), the
   same pattern as the `EVAL_HF_REPO`/`EVAL_SPLIT`/`EVAL_OUT_SUFFIX` block already there. Otherwise
   run `kele` directly (loses auto-resume; risky on this unstable box for a 30 h base run).
3. **The SFT CONSUMES the consultant, it does not emit state** (verified in `dataset.py:608–647`,
   `socrat-zh-sft`/`socrat-en-sft`): its training prompt = system rules + history + student input +
   `苏格拉底教学顾问评估结果: {evaluation}` + `苏格拉底教学顾问建议的操作: {action}`, and the target is a
   **clean teacher turn** (no inline state). Implications:
   - The scored `state` comes from the **consultant**, never parsed from teacher output — so
     "parse state from the SFT's output" is **not viable** (output carries no state).
   - **Fully dropping the consultant** (no assessment line at all) runs the SFT **off its training
     distribution** AND leaves nothing to score `state_accuracy` → ROUGE/BLEU only. It's a
     robustness probe, not the SFT's normal operating mode.
   - **Self-consult keeps the SFT in-format**: the LLM produces the assessment itself, then consumes
     it — the assessment line is still present, just self-generated. This is the recommended "no
     external classifier" run.
4. **Everything else stays pinned** (so only the consultant + model vary): Q8_0 GGUF, `-np 4`,
   q4_0 KV, MTP off, workers=4, 8 rounds, stochastic server-default sampling, ZH `SocratDataset`
   test split. Serve base via `make serve-gemma4-12b` (alias "Gemma 4 12B"), SFT via
   `make serve-gemma4-12b-sft` (alias "Gemma 4 12B SFT") — one model at a time on the 20 GB card.

## The minimum: 2 runs (the ablation the user asked for)

Base vs SFT, **no consultant**, ZH test, same settings. Direct (no monitor) form:

```
# serve base first (make serve-gemma4-12b), then:
KELE_BERT_DEVICE=cpu KELE_PARALLEL_WORKERS=4 uv run --no-sync python -m src.project.kele \
  --experiment gemma4-12b-local evaluate --output results/gemma4-12b-base-noconsult
# stop base server, serve SFT (make serve-gemma4-12b-sft), then:
KELE_BERT_DEVICE=cpu KELE_PARALLEL_WORKERS=4 uv run --no-sync python -m src.project.kele \
  --experiment gemma4-12b-sft-local evaluate --output results/gemma4-12b-sft-noconsult
# compare, and cross-compare against the WITH-consultant runs:
python -m src.project.evaluate --compare results/gemma4-12b-base-noconsult results/gemma4-12b-sft-noconsult
```
**Recommended instead:** add `NO_CONSULTANT=1` to the monitor (gotcha #2) and run via
`make monitor-eval-gemma4-12b-{base,sft}` for crash-resilience on the long base run. Consider
`--limit 200` for the base arm first.

## The broader design: consultant as a variable (2×N factorial)

Teacher ∈ {base, SFT} × Consultant ∈ {Qwen-classifier (have it), self-consult (this handoff),
optionally unified, optionally a strong external judge like Claude}. The **core 2×2** is:

| | consultant = Qwen classifier | consultant = self (no external) |
|---|---|---|
| **base** | 49.62 (have it: `results/gemma4-12b-base`) | **run this** |
| **SFT** | 59.93 (have it: `results/gemma4-12b-sft`) | **run this** |

What each contrast answers:
- **SFT(self) vs SFT(Qwen)** — if close, the SFT **internalized** state-tracking and the external
  classifier is largely redundant (a strong, headline-worthy result). If it drops a lot, the gain
  leaned on the classifier.
- **base(self) vs base(Qwen)** — how much the *base* pipeline depends on the classifier crutch
  (expected: base self-consult ≪ base+Qwen).
- **SFT(self) vs base(self)** — the cleanest "is the SFT model itself a better Socratic
  state-tracker?" with no external help on either side. **This is the key comparison.**
- **Interaction** — does SFT's advantage shrink, hold, or *grow* when the crutch is removed? A gain
  that grows without the classifier is the strongest evidence the SFT learned the skill.

Note on what the metrics mean here: the scored `state_accuracy` is the **consultant's** prediction
(only indirectly shaped by the teacher, via the dialogue history the classifier reads), whereas
**ROUGE/BLEU directly measure teacher-turn quality** — and that's where the SFT's cleanest wins are
(ROUGE-1 28→48 on ZH). The consultant-variable runs that drop/weaken state prediction naturally
lean on the text metrics.

Stretch consultant levels (only if the core 2×2 is promising):
- **`--unified`** — fused single-call mode; another way the model self-tracks.
- **Strong external consultant** (Claude as classifier; configs exist:
  `configs/claude-*-as-consultant.env`) — upper bound on how much a *better* consultant lifts each
  teacher, and whether SFT still adds value on top of a strong classifier.
- **Oracle consultant (feed ground-truth state/action)** — small code change: a consultant that
  returns the dialogue's GT state instead of predicting. Makes `state_accuracy` trivially perfect,
  so compare on **ROUGE/BLEU** — this isolates *pure teacher-turn quality given correct state*, the
  cleanest "does SFT write better Socratic turns?" measure (removes classifier quality as a
  confound). Arguably more informative than fully-bare.
- **Fully bare (no consultant at all)** — needs code (strip the assessment+action lines, skip state
  scoring). SFT runs off-distribution; ROUGE/BLEU only. A robustness probe, not normal operation.

Keep each cell to ZH test first (the reference set); extend to EN/synthetic only for cells that
matter, since each base cell is expensive.

## Suggested order

1. Wire `NO_CONSULTANT=1` into the monitor (small, mirror the existing env-override block) + a test.
2. **SFT no-consultant, ZH test** (fast, ~1 h) — get the most important number first.
3. **base no-consultant, ZH test** — start `--limit 200` to get a quick read; full 681 if the box
   tolerates the ~30 h.
4. `--compare` the 2×2; write results into `docs/SFT_RESULTS_REPORT.md` + `EXPERIMENT_LOG.md` + #130.
5. Only then consider unified / Claude-consultant / parse-from-output.

## Definition of done

`results/gemma4-12b-{base,sft}-noconsult/metrics_summary.json` exist; the 2×2 table (Qwen vs
self-consult × base vs SFT) is filled and interpreted in `docs/SFT_RESULTS_REPORT.md`; #130 updated.
State whether the SFT's advantage survives removing the external classifier (the headline question).

## Where things live

- **Results so far:** `results/gemma4-12b-{base,sft}{,-en,-synth-zh,-synth-en}` + `-base-mtp`.
- **Models (HF, private):** `ulises-c/SocratesLM-12B-QLoRA` (adapter), `-12B` (merged BF16),
  `-12B-GGUF` (Q8_0). Served GGUF already staged in the weights dir.
- **Consultant ckpt:** `results/state-clf-qwen3.5-0.8b-lora-wandb/final`.
- **Key code:** `create_system` + `run_batch_evaluation` (`src/project/kele.py`); dual-role
  consultant (`src/project/socratic_teaching_system.py:socratic_teaching_consultant`); classifier
  variant (`src/project/socratic_teaching_bert_consultant.py`); monitor
  (`scripts/monitor_eval_gemma4_12b.sh`).
- **Box hazards:** unstable RTX 4000 Ada, stable at 85 W; power step-down inert (no passwordless
  sudo); one model at a time on 20 GB. See `docs/SFT_HANDOFF.md` + memories.
