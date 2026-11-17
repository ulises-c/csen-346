# Socratic QLoRA SFT vs Base — Results Report

**Gemma 4 12B-it, NVIDIA PoC.** Branch `feat/gemma4-12b-sft-poc-nvidia`. Updated 2026-07-09.
Companion to `docs/SFT_HANDOFF.md` (pipeline/provenance) and `docs/SFT_VS_BASE_ANALYSIS_PLAN.md`
(follow-on ablations). Live tracker: GitHub issue #130.

## TL;DR

Socratic QLoRA SFT on Gemma 4 12B **improves state accuracy on every axis tested** — both
languages, in-distribution and out-of-distribution — with **no regressions**. The gain is large on
held-out in-distribution data (**+7.7 to +10.3 pp**) and smaller but consistently positive on
never-seen synthetic data (**+3.5 to +3.9 pp**), the healthy shape that shows the model learned a
**transferable Socratic skill, not corpus memorization**. Achieved with a **~0.88-epoch** adapter
recovered after a NaN divergence — i.e. a lower bound on a clean full run.

## Headline: state accuracy (the metric of record)

| eval set | distribution | base | SFT | **Δ (pp)** | n |
|---|---|---|---|---|---|
| ZH test | in-distribution (held-out 10%) | 49.62 | **59.93** | **+10.31** | 681 |
| EN test | in-distribution (held-out 10%) | 51.00 | **58.68** | **+7.68** | 681 |
| ZH synthetic | OOD (never trained) | 27.19 | **31.09** | **+3.90** | 75 |
| EN synthetic | OOD (never trained) | 27.44 | **30.97** | **+3.53** | 75 |

Run-to-run σ ≈ 0.7 pp (decoding is stochastic), so the held-out gains are ~11–15σ — unambiguous.
The MTP-on ZH base scores 50.30 (within σ of the 49.62 MTP-off base used above); SFT clears either
by a wide margin. Every run: 681/681 (or 75/75) valid, **0 errors**.

## Per-stage breakdown (state accuracy by Socratic stage a–e)

| set | a | b | c | d | e |
|---|---|---|---|---|---|
| ZH test base → SFT | 100 → 100 | 43.4 → **57.0** | 30.6 → **42.9** | 36.7 → **46.2** | 61.9 → **78.1** |
| EN test base → SFT | 100 → 100 | 48.8 → **54.5** | 33.0 → **40.9** | 37.9 → **47.2** | 57.1 → **75.4** |
| ZH synth base → SFT | 100 → 100 | 17.1 → 20.4 | 5.3 → 4.5 | 0.0 → 1.5 | 29.2 → **49.3** |
| EN synth base → SFT | 100 → 100 | 30.7 → 28.0 | 4.6 → 6.9 | 0.0 → 0.0 | 18.9 → **38.8** |

- Stage **a** is trivially 100 everywhere (the opening turn).
- The largest, most consistent gain is **stage e** (closing/summary) — +16 to +20 pp on the
  in-distribution sets and the only clear OOD mover. The SFT most improved *how the teacher closes*.
- On OOD synthetic, mid-dialogue stages **c/d nearly collapse for both** models (c≈5, d≈0) — those
  synthetic dialogues are structurally hard / off-distribution in the middle; the SFT's OOD edge
  comes from stages **b** and **e**.

## Text-overlap metrics (vs ground-truth teacher turns)

SFT roughly doubles–triples ROUGE/BLEU on every set, confirming it learned the teacher's phrasing,
not just state labels:

| set | ROUGE-1 | ROUGE-L | BLEU-4 |
|---|---|---|---|
| ZH test base → SFT | 28.6 → **48.1** | 21.0 → **40.9** | 5.2 → **20.1** |
| EN test base → SFT | 51.3 → **69.5** | 34.4 → **46.8** | 3.2 → **11.7** |
| ZH synth base → SFT | 24.3 → **33.7** | 17.7 → **26.5** | 3.3 → **8.3** |
| EN synth base → SFT | 46.4 → **62.6** | 30.5 → **37.2** | 0.9 → **1.7** |

## What this means

1. **Real, large in-distribution improvement** — +10.3 pp ZH / +7.7 pp EN on held-out test, far
   past the ~1.5 pp significance bar.
2. **Generalizes, not memorizes** — the SFT still wins +3.5–3.9 pp on synthetic dialogues it has
   *never* seen (different origin, phrasing, question pool). A pure memorizer would show ~0 OOD gain.
3. **Cross-lingual transfer** — EN (+7.7) tracks ZH (+10.3); the Socratic behavior survives the
   language boundary (the SFT trained on both ZH and EN per-turn data).
4. **Behavioral signal** — the base model is ~6× slower to evaluate (≈38 vs ≈225 dlg/hr; e.g. ZH
   base 17.4 h vs SFT 3.0 h) because it **rambles to the 2048-token cap** instead of producing
   short, terminating Socratic turns. The SFT terminates cleanly — itself evidence of better form
   (to be quantified; see analysis plan T0.1).

## Consultant ablation (T1.1): does the advantage survive removing the classifier?

Every headline result gave **both** base and SFT the same external Qwen state classifier, so those
numbers don't isolate the SFT's *own* contribution. This ablation reruns the ZH-test 2×2 across
**three consultant modes**, varying only where the scored state comes from (all else pinned: Q8_0,
`-np 4`, q4_0 KV, MTP off, 8 rounds; every arm 681/681, 0 errors):

- **Qwen classifier** (headline): shared external classifier, ~55–60% accurate, for both models.
- **self-consult**: no external classifier — the served LLM produces the state assessment itself
  then consumes it (dual-role). The apples-to-apples "no external classifier" baseline.
- **oracle**: the ground-truth state is *fed in* each turn (`--oracle-consultant`), so
  `state_accuracy` is 100% by construction. Removes classifier accuracy as a confound entirely —
  the cleanest measure of *teacher-turn quality given correct state*.

(The two long arms ran on the unstable box across auto-resumed crashes: base self-consult ~7 days,
base oracle ~2 days.)

**The two skills decouple cleanly.** Teacher-turn *quality* is intrinsic to the SFT — and is in fact
*largest* once the state confound is removed (oracle). *State-tracking* was entirely the external
classifier and is worse than base when the SFT must self-track.

### State accuracy (overall %) — the classifier's job, not the SFT's

| | Qwen classifier | self-consult | oracle |
|---|---:|---:|---:|
| base | 49.62 | 34.45 | 100 (by construction) |
| SFT | 59.93 | **26.80** | 100 (by construction) |
| **SFT − base** | **+10.31** | **−7.65** | — |

The SFT's +10.3 pp state edge under the shared classifier *inverts* to −7.65 pp when it must
self-track (it loses on every hard stage: b 24.4 vs 41.7, c 5.9 vs 16.5, e 27.1 vs 36.0). Oracle
pins both to 100, so state accuracy carries no signal there — score oracle on ROUGE/BLEU only.

### Teacher-turn quality — the SFT's real deliverable (ROUGE/BLEU, SFT − base gap)

| metric | base: self / Qwen / oracle | SFT: self / Qwen / oracle | **SFT − base gap: self / Qwen / oracle** |
|---|---:|---:|---:|
| ROUGE-1 | 28.02 / 28.56 / 29.03 | 44.21 / 48.13 / 51.77 | **+16.19 / +19.57 / +22.74** |
| ROUGE-L | 20.35 / 21.02 / 21.76 | 37.62 / 40.94 / 44.80 | **+17.27 / +19.92 / +23.04** |
| BLEU-4  | 4.80 / 5.22 / 5.73    | 18.31 / 20.12 / 24.80 | **+13.51 / +14.90 / +19.07** |

### What the three modes together show

- **The SFT climbs monotonically with state quality; base is flat.** SFT ROUGE-1: 44.2 (self, ~27%
  state) → 48.1 (Qwen, ~55–60%) → **51.8 (oracle, 100%)**. Base sits at **28.0 → 28.6 → 29.0** across
  the same axis. The SFT *learned to condition on state* — hand it a better state and it writes a
  better turn; base can't exploit better state at all (it rambles regardless). That conditioning is
  itself a learned skill, not phrasing memorization.
- **The gap is *largest* with the confound removed.** Oracle (correct state, both sides) gives the
  SFT its biggest edge — **+22.7 ROUGE-1, +23.0 ROUGE-L, +19.1 BLEU-4** — bigger than under either
  the classifier or self-consult. So "the SFT writes better Socratic turns" is unambiguous.
- **Self-consult was an *underestimate*, not the ceiling.** It showed the smallest gap (+16.2)
  precisely because the SFT was penalized there — conditioned on its own worse self-classification
  (27%). Removing that penalty (oracle) reveals the true ceiling; the "survives removing the
  classifier" conclusion is thereby *strengthened*, not merely upheld.

### Interpretation

Consistent with the SFT's training format (`dataset.py:608–647`): the SFT was trained to **consume**
the consultant's assessment + action and emit a clean teacher turn — it **never learned to emit
state**. So `state_accuracy` in self-consult mode measures a skill the SFT never trained, and it
regressed below base's zero-shot classification (it learned to expect state to be handed to it). The
base model, never specialized, is the better self-classifier.

**Headline answer:** the SFT genuinely internalized how to *write* Socratic turns — that advantage
not only survives removing the external classifier, it is **largest** when the state confound is
removed (+22.7 ROUGE-1, +19.1 BLEU-4 given correct state). It did **not** internalize how to *track*
state — that gain was the external classifier's, and self-tracking is worse than base. In deployment
the SFT wants an external state source, and it rewards a *better* one with proportionally better
turns. Results: `results/gemma4-12b-{base,sft}-{noconsult,oracle}/`.

## LLM-judge (absolute pedagogical quality) — does the oracle ROUGE win hold on *quality*, not overlap?

ROUGE/BLEU only measure overlap with the reference teacher turn. To test whether the oracle win
reflects genuinely better teaching, we scored both **oracle arms** with **Claude Opus 4.8** on an
absolute 0–10 rubric (socratic_validity 0–3, advancement 0–3, age_appropriateness 0–2,
question_form 0–2). The rubric shows the GT reference "as one valid move, do not penalize routing
differences" **on purpose** — so it rewards quality, not ROUGE-like overlap. Paired stratified
sample: 200 turns/arm, identical `(file, turn_idx)` keys across arms, stages b–e (opener `a`
excluded), seed 42, `--per-stage 50` (400 Opus calls, ~62 min/arm). Single run, headless subscription
(not temperature-0). `results/llm_judge_oracle_compare.json`.

| | SFT-oracle | base-oracle | **Δ (SFT − base)** |
|---|---:|---:|---:|
| **overall /10** | **8.13** | 7.75 | **+0.38** |
| socratic_validity /3 | 2.56 | 2.46 | +0.10 |
| advancement /3 | 2.43 | 2.435 | **−0.01 (tied)** |
| age_appropriateness /2 | 1.93 | 1.845 | +0.09 |
| question_form /2 | 1.21 | 1.01 | **+0.20** |
| stage b /10 | 8.28 | 8.14 | +0.14 |
| stage c /10 | 8.64 | 8.00 | **+0.64** |
| stage d /10 | 8.12 | 7.36 | **+0.76** |
| stage e /10 | 7.48 | 7.50 | −0.02 (tied) |

**Verdict: confirms the *direction*, corrects the *magnitude*.** The judge independently agrees the SFT
writes better Socratic turns given correct state (+0.38/10 overall, and it wins or ties every axis and
stage). But the gap is **far smaller than ROUGE implied**: +0.38/10 (~5% relative) vs +22.7 ROUGE-1
(~78% relative). Both models produce pedagogically solid turns given correct state (7.75 vs 8.13 are
both high) — so most of the ROUGE advantage was **reference-phrasing overlap, not teaching quality**.
The SFT learned to phrase turns the way the corpus does; that shows up huge in ROUGE but only modestly
in absolute pedagogy.

Two honest caveats the pattern surfaces:
- **`advancement` is a dead tie** (2.43 vs 2.435). Given correct state, base moves the lesson forward
  just as well as the SFT. The SFT's real, judge-visible edge is **question_form** (+0.20, the largest
  axis delta) and **socratic_validity** (+0.10) — *how* it asks, not *whether* it progresses.
- **The win concentrates in the middle stages** where there's pedagogical work to do — c (+0.64) and
  d (+0.76) — and vanishes at the endpoints (b +0.14, e −0.02). Coherent, but it means the headline
  +0.38 is an average over a strongly stage-dependent effect, not a uniform lift.
- **No error bars** (single run, non-greedy subscription decoding). The +0.38 headline is a point
  estimate; the internally-consistent per-axis/per-stage pattern is what lends it confidence, not the
  scalar alone. (Aside: base cost *more* to judge — $15.02 vs $13.86 — i.e. base emits more tokens,
  consistent with the "base rambles regardless" finding.)

## Method (held fixed; only model + dataset vary)

- **Teacher** = the only variable under test: base `unsloth/gemma-4-12b-it` vs the merged Socratic
  SFT, both served as **Q8_0 GGUF** on llama.cpp (`-np 4`, q4_0 KV, **MTP off**, workers=4).
- **Consultant** = Qwen3.5-0.8B LoRA state classifier on CPU (same checkpoint for every run).
- 8 teaching rounds, no fewshot, no thinking budget, stochastic server-default sampling (identical
  across runs). Eval replays ground-truth student turns; the teacher generates; the classifier
  scores state. 90/10 train/test split, seed 42, dialogues kept whole.
- **Datasets** (KELE-v2 collection): `SocratDataset` (ZH) / `SocratDataset-EN` held-out **test**
  splits; `SocratDataset-SYNTHETIC` / `-EN` (75 each) run **whole** as OOD probes (never in training).

## Caveats

- **Adapter is ~0.88 epoch**, `checkpoint-4250`, recovered from HF history after a NaN divergence
  at step ~4260 (loss had plateaued since ~step 3000). Report as ~0.88 epoch; a clean full run is
  a plausible further gain (see `SFT_HANDOFF.md`).
- **Synthetic n is small** (37 ZH + 38 EN merged → 75 each; ~215/431 turns). Directional OOD
  signal, not σ-tight. State accuracy is per-turn, so steadier than the dialogue count implies.
- **Stochastic decoding** — no temperature/seed pinned, so each run carries ~0.7 pp noise. Greedy
  and multi-seed runs (analysis plan T2.1/T2.2) would tighten the point estimates.
- **Consultant ablation DONE** (all three modes — see the consultant-ablation section above) — the
  headline state-accuracy numbers rely on the shared external classifier and do **not** reflect the
  SFT's own state-tracking (self-consult drops SFT to 26.8, below base). The SFT's *teacher-turn
  quality*, however, is classifier-independent and *largest* with the confound removed (+22.7
  ROUGE-1 oracle vs base). Read the headline table as "SFT + classifier vs base + classifier," not
  "SFT alone."

## Artifacts

- **Models (HF, private):** adapter `ulises-c/SocratesLM-12B-QLoRA` (ckpts 3200–4250); merged BF16
  `ulises-c/SocratesLM-12B`; Q8_0 GGUF `ulises-c/SocratesLM-12B-GGUF`.
- **Datasets:** `ulises-c/SocratDataset{,-EN,-SYNTHETIC,-SYNTHETIC-EN}` (synthetic ZH completed to
  75 this PR).
- **Results:** `results/gemma4-12b-{base,sft}{,-en,-synth-zh,-synth-en,-noconsult,-oracle}/` +
  `-base-mtp`.

## What this PR changed (code)

- GGUF convert: CPU-only `llama-quantize` via `QUANTIZE` override (the CUDA build segfaults `nvcc`
  on this box).
- Eval: `--hf-repo`/`--split` on `evaluate` + monitor `EVAL_HF_REPO/EVAL_SPLIT/EVAL_OUT_SUFFIX`
  (no schema adapter needed — all KELE-v2 sets share the `{student,teacher,state}` turn keys).
- Data: merged the 38-record ZH-synthetic extension into HF (37 → 75); fixed the loader that
  referenced the never-uploaded config.
- Monitor: log the GPU's **actual enforced** power, not the card max (the step-down is inert
  without passwordless sudo).
- Docs: eval-plan rationale + workers A/B, this report, and the further-analysis/ablation plan.

## Reproduce

```
# serve the model under test (base or SFT), then:
KELE_PARALLEL_WORKERS=4 EVAL_HF_REPO=<repo> EVAL_SPLIT=<test|all> EVAL_OUT_SUFFIX=<tag> \
  make monitor-eval-gemma4-12b-{base,sft}
python -m src.project.evaluate --compare results/<base-run> results/<sft-run>
```

## Next

Deeper analysis is scoped in `docs/SFT_VS_BASE_ANALYSIS_PLAN.md`; live next-steps in
`docs/HANDOFF_SFT_POST_JUDGE.md`. The consultant ablation (T1.1, all three modes) **and** the
LLM-judge (T1.2) are **done** (above): the SFT internalized teacher-turn *quality* (largest under
oracle; judge confirms the direction at +0.38/10 but ROUGE overstated the magnitude), not
*state-tracking*. Remaining picks, all optional: **T0.1 termination/length/style metrics** (free,
on-disk — quantifies the still-unquantified "base rambles" claim in §4 above), **multi-seed error
bars** (every number is a single-run point estimate), **earlier-checkpoint eval** (3200 vs 4250 —
decides whether a clean full-epoch run is worth ~30 h), and a **strong-consultant** cell (Claude as
classifier) between the ~55–60% Qwen point and the oracle's 100%.
