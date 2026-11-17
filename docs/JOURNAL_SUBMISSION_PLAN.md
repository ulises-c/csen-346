# Journal submission plan

**Written:** 2026-11-17. **Status:** Phase 0 in progress; nothing after it has started.

**Goal:** get the paper to a state where every claim in it is backed by an artifact in this repo or
on Hugging Face, with confidence intervals, at no API cost.

Tick the boxes as items finish, and add a dated line under "Log" at the bottom when a phase closes
or a decision changes.

---

## 1. Constraints

- **Budget is $0.** No paid API calls. Two Claude subscriptions are available for small checks.
- **Main GPU is the R9700** (32 GB, ROCm 7.2.4, on `ollie-cachy`). A GTX 1070 is also on hand.
  An A6000 and smaller RTX cards are available occasionally.
- **The per-dialogue outputs behind the current paper are gone.** They were never committed, and
  the machines that held them are wiped or out of reach. Treat every number in the current draft as
  an archived figure that cannot be rescored, re-judged or given an interval.
- **What survives:**
  - `results/*/metrics_summary.json` and `results/*/judge_summary.json` in git (judge scores are
    per dialogue; state accuracy is not).
  - `results/wandb_export/csen346-eval.json`: cumulative metrics every 10 dialogues for the
    Gemma 4 12B base and fine-tune runs. See the README beside it.
  - Hugging Face: `ulises-c/socrates-state-classifier-qwen3.5-lora` (public, a retrain, not the
    paper's checkpoint), `ulises-c/SocratesLM-12B` and `ulises-c/SocratesLM-12B-QLoRA` (private),
    `ulises-c/SocratesLM-31B-stage2b-QLoRA` (public), `ulises-c/SocratTeachLLM`, and the datasets
    including `SocratDataset-SYNTHETIC` with its 75-dialogue extension.
  - On `ollie-cachy`: Gemma 4 31B Q5 GGUF, SocratTeachLLM weights, Qwen3.5-0.8B base.

## 2. What the paper will claim

Keep, and re-establish with new runs:

1. A small fine-tuned classifier is a better consultant than a general LLM for state tracking.
2. Text-overlap metrics and state accuracy rank systems differently.
3. In-distribution fine-tuning of the teacher buys phrasing overlap far more than pedagogy
   (SocratTeachLLM against its GLM-4-9B base; Gemma 4 12B fine-tune against its base, where
   oracle states gave about +23 ROUGE-1 but only +0.38 of 10 from the judge).
4. Schema fallback and small-sample effects, as methodological findings.

Change or drop:

- **Drop "overtakes frontier models" as a claim.** It was confounded and needs paid API runs to
  fix. Archived Claude numbers may appear as context, labelled as earlier runs with a different
  consultant and prompt stack.
- **Drop the cross-lingual "reproduces the headline" result.** It was a character-level ROUGE
  artifact.
- **Replace "contamination" with "the authors' split cannot be reproduced".** The seed is unknown.
- **Redo the clean-probe comparison** on the 75-dialogue synthetic set, same setup for every model.
- **Report state accuracy and judge score separately as primary results.** The unified score is
  secondary.

## 3. Phases

### Phase 0: housekeeping

- [x] Merge #129 (Gemma 12B fine-tune, oracle mode, Claude Code judge backend); close #101.
- [ ] Merge #183: language-aware ROUGE and `scripts/unified_robustness.py`.
- [ ] Merge #184: gfx1201 probe work, logs and upstream status.
- [ ] Merge the PR carrying this plan and the W&B export.
- [ ] Dependabot PRs: rebased onto the fixed lint job; merge the ones that pass, decide on the rest
      (see section 5).
- [ ] Delete `feat/gfx1201-rdna4-qlora-fla-training` after #184 merges.

### Phase 1: can the R9700 serve everything? (gates Phase 4)

Run on `ollie-cachy` in a separate git worktree. Each test is short.

- [ ] **T1. SocratTeachLLM in llama.cpp.** `scripts/convert_socratteachllm_to_gguf.sh`, then
      `make serve-socratteachllm-llamacpp`. The scripts record that an older llama.cpp could not
      load the result; the installed build is much newer. Pass = the server answers a chat request
      with sensible Chinese.
- [ ] **T2. Same for the GLM-4-9B base.** Download the base weights, then
      `scripts/convert_glm49b_chat_to_gguf.sh` and `scripts/serve_glm49b_chat_llamacpp.sh`.
- [ ] **T3. vLLM, only if T1 or T2 fails.** Source build of vLLM main with `--enforce-eager`, then
      `make test-vllm`. Upstream issue vllm#28649 is open, but the remaining complaints concern FP8.
- [ ] **T4. Gemma 4 31B timing and reproduction check.** `make serve-gemma4-31b`, then a
      50-dialogue run of the headline configuration with 4 workers. Record wall time. Compare with
      the archived 50-dialogue cell (51.58% state accuracy, ROUGE-1 38.76). The classifier is a
      different checkpoint, so expect a shift; a gap above about 5 points needs explaining before
      going on.
- [ ] **T5. Local judge throughput.** Serve the candidate judge (Phase 2) and time 20 dialogues.

**Gate.** If T1 or T3 passes and T4 is under about 8 hours for 681 dialogues, everything runs on
the R9700. If SocratTeachLLM cannot be served there, the fallback is `scripts/serve_teacher_local.sh`
(HF Transformers, slower) or borrowed A6000 time for the SocratTeachLLM and GLM runs only.

### Phase 2: preparation

- [ ] **Classifier of record.** `make _classifier-ckpt` pulls the Hugging Face checkpoint.
      Re-evaluate it on the 4,304-turn Chinese test set and replace the classifier table in the
      paper. Confirm which split it was trained on: its `eval_results.json` covers 8,526 turns,
      which suggests Chinese plus English.
- [ ] **BERT consultant.** Retrain with `scripts/train_stage_classifier.py` (minutes on any card);
      the weights are gone. Publish to Hugging Face this time.
- [ ] **Synthetic set.** Pull the 75-dialogue version from `ulises-c/SocratDataset-SYNTHETIC`
      (`n75_extension/`); the local file has 37.
- [ ] **Judge completeness check.** `scripts/llm_judge_eval.py` should exit non-zero, or mark the
      summary incomplete, when it judged fewer dialogues than the run contains. The archived Sonnet
      run was judged on 274 of 681 dialogues and nothing flagged it. Add a test.
- [ ] **Local judge.** Pick one pinned open-weight model already on `ollie-cachy` (Qwen 27B class),
      temperature 0, and confirm `llm_judge_eval.py --backend api --base-url …` works against a
      local OpenAI-compatible server. Record model file hash and quantization.
- [ ] **Output archiving.** After every run, upload `results/<run>/` including `dialogues/` to a
      Hugging Face dataset repo before starting the next run. `results/` is in `.gitignore`, so
      summaries reach git only when added with `git add -f`; that is how the Gemma 12B results were
      lost. Either un-ignore `results/*/*.json` or make the force-add part of the run script.
- [ ] **Seeds.** Fix the list up front: 42 for everything, plus two more for the headline cell.

### Phase 3: analyses that need no GPU

- [ ] **Block bootstrap on the Gemma 12B runs** from `results/wandb_export/`: state-accuracy
      intervals for base, fine-tune, no-consultant and oracle variants. Unpaired, blocks of 10
      dialogues.
- [ ] **Sample-size analysis, redone.** Rewrite `scripts/convergence_analysis.py` to resample with
      replacement and report standard errors. It can only be run on real outputs once Phase 4
      produces the first full run.
- [ ] **Weight sweep and judge bootstrap** (`scripts/unified_robustness.py`) rerun on the new runs.

### Phase 4: reruns

All on the Chinese test set, 10-shot prompt, seed 42, local judge on every dialogue, outputs
archived. Order matters: run 1 first, because the sample-size analysis on it decides whether the
rest can run at 400 dialogues.

| # | Consultant | Teacher | Set | Purpose |
|---|---|---|---|---|
| 1 | Qwen3.5 classifier | Gemma 4 31B | 681 | Headline |
| 2 | BERT | Gemma 4 31B | 681 | Consultant ablation |
| 3 | Gemma 4 31B (LLM as consultant) | Gemma 4 31B | 681 | Baseline: classifier against an LLM consultant, with outputs |
| 4 | Qwen3.5 classifier | SocratTeachLLM | 681 | Fine-tuned teacher |
| 5 | Qwen3.5 classifier | GLM-4-9B base | 681 | Its base model |
| 6 | Qwen3.5 classifier | Gemma 4 12B base, and fine-tune | 681 | Second fine-tune pair, replaces the lost June runs |
| 7 | Runs 1, 4, 5, 6 | | synthetic 75 | Clean probe; report difference of differences, no state accuracy |
| 8 | Run 1, seeds 2 and 3 | | 681 | Run-to-run variation |

The archived GPT-4o + SocratTeachLLM reproduction (25.94% state accuracy) stays in the paper as a
cited earlier figure without an interval. Run 3 is the baseline that has artifacts.

### Phase 5: ablations

Run at the dialogue count Phase 3 justifies, on the headline configuration unless stated.

- [ ] **Random states** fed to the teacher. Run first: it tests whether the teacher uses the state
      at all (archived judge scores were 8.19 against 8.32 across consultants).
- [ ] **Oracle consultant** (`--oracle-consultant`) for Gemma 4 31B and SocratTeachLLM. Tests the
      claim that state tracking explains the 12.79-point ROUGE-1 reproduction gap.
- [ ] **Offline against live classifier accuracy**, feeding ground-truth history, to explain the
      gap between 67.57% offline and 55.39% in the loop.
- [ ] **Exemplars:** 0 against 10, and stage-balanced against random.
- [ ] **Monotonicity guard** on and off.
- [ ] **Classifier learning curve** (train on 10%, 25%, 50%, 100%).

### Phase 6: is the judge trustworthy?

- [ ] **Second judge on a subset.** About 100 dialogues from runs 1 and 4, judged with
      `--backend claude-code` on a subscription. Report agreement with the local judge. Check the
      subscription terms first; this is a few hundred calls, not a full pass.
- [ ] **Judge without the reference answer**, same subset, to measure how much the reference
      anchors the score.
- [ ] **Human scores.** Two people, about 150 turns, blind to system. Report agreement between the
      humans and between humans and the judge.
- [ ] **Human relabel of about 200 turns** of state labels, for a label-noise ceiling on state
      accuracy.

### Phase 7: paper and repository

- [ ] Rewrite around section 2. Remove course-report language. Decide the venue and switch to its
      template; the current one is a non-anonymous ACL preprint style.
- [ ] One name throughout (MELE or KELE), and one state count (34 or 35).
- [ ] `LICENSE` and `CITATION.cff`; check the licences of the datasets and of SocratTeachLLM first.
- [ ] README: replace the `poetry run` commands, describe the metric change, link the archived
      outputs.
- [ ] Prune handoff notes and AMD diagnostics that the paper does not need (issue #110).
- [ ] A single script that regenerates every table and figure from the archived outputs.

## 4. Risks and open questions

- **The headline number will move.** Different classifier checkpoint, different hardware. Old and
  new numbers must not share a table.
- **R9700 serving may fail for the GLM-family models**, which would push runs 4 and 5 to slower
  serving or borrowed hardware.
- **A local judge is weaker than a frontier judge.** Phase 6 exists to measure that; if agreement
  with humans is poor, the judge score becomes a secondary result and state accuracy carries the
  paper.
- **Run 3 changes the baseline** from GPT-4o to an open model. The claim becomes "beats an LLM
  consultant of the same size class as the teacher", which is narrower than the current wording.
- **Open:** does the ex-partner still have the `results/` directory from the 5090 machine? If it
  turns up, the archived runs can be rescored and Phase 4 shrinks.
- **Open:** which venue, and whether it requires anonymisation.
- **Not needed for the paper:** the ROCm training fault. Inference on the R9700 is unaffected.
  Status is in `docs/HANDOFF_GFX1201_UPSTREAM_AMD.md`.

## 5. Dependabot

Eleven PRs were open on 2026-11-17, all failing Lint because CI ran an unpinned ruff. That is fixed
on main. The repository ruleset requires Lint and Test to pass and the branch to be up to date, so
each merge puts the others behind again; merge one, let Dependabot rebase, repeat.

Three need a decision, not just a rebase:

- **ruff 0.15.22 → 0.16.0:** the new formatter reformats about ten files. Merge it together with a
  formatting commit.
- **datasets 4 → 5** and **transformers 5.9 → 5.14:** run the classifier and evaluation smoke tests
  before merging, or hold them until Phase 4 is finished so the environment stays fixed while the
  paper's runs are produced.
- **bitsandbytes 0.49.2 → 0.50.0:** only matters for training; 0.50 is also what a ROCm 10 torch
  needs.

## Log

- 2026-11-17: plan written. #129 merged, #101 closed, #183 and #184 opened.
