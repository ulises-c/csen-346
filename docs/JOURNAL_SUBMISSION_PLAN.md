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
- [x] Merge #183: language-aware ROUGE and `scripts/unified_robustness.py`.
- [x] Merge #184: gfx1201 probe work, logs and upstream status.
- [x] Merge the PR carrying this plan and the W&B export.
- [ ] Dependabot PRs: rebased onto the fixed lint job; merge the ones that pass, decide on the rest
      (see section 5).
- [ ] Delete `feat/gfx1201-rdna4-qlora-fla-training` after #184 merges.

### Phase 1: can the R9700 serve everything? (gates Phase 4)

Run on `ollie-cachy` in a separate git worktree. Each test is short.

- [x] **T1. SocratTeachLLM in llama.cpp.** `scripts/convert_socratteachllm_to_gguf.sh`, then
      `make serve-socratteachllm-llamacpp`. The scripts record that an older llama.cpp could not
      load the result; the installed build is much newer. Pass = the server answers a chat request
      with sensible Chinese.
- [x] **T2. Same for the GLM-4-9B base.** Download the base weights, then
      `scripts/convert_glm49b_chat_to_gguf.sh` and `scripts/serve_glm49b_chat_llamacpp.sh`.
- [ ] **T3. vLLM, only if T1 or T2 fails.** *Not needed: T1 and T2 passed.* Source build of vLLM main with `--enforce-eager`, then
      `make test-vllm`. Upstream issue vllm#28649 is open, but the remaining complaints concern FP8.
- [x] **T4. Gemma 4 31B timing and reproduction check.** `make serve-gemma4-31b`, then a
      50-dialogue run of the headline configuration with 4 workers. Record wall time. Compare with
      the archived 50-dialogue cell (51.58% state accuracy, ROUGE-1 38.76). The classifier is a
      different checkpoint, so expect a shift; a gap above about 5 points needs explaining before
      going on.
- [x] **T5. Local judge throughput.** Serve the candidate judge (Phase 2) and time 20 dialogues.

**Gate.** If T1 or T3 passes and T4 is under about 8 hours for 681 dialogues, everything runs on
the R9700. If SocratTeachLLM cannot be served there, the fallback is `scripts/serve_teacher_local.sh`
(HF Transformers, slower) or borrowed A6000 time for the SocratTeachLLM and GLM runs only.

**Result (2026-11-17): the gate passes. Everything runs on the R9700.** Run on `ollie-cachy`,
llama.cpp build 11429 (ROCm backend), main at `5f2f645`, transformers 5.23.

| Test | Outcome |
|---|---|
| T1 SocratTeachLLM | Converts, loads and answers in Chinese. Q8_0, 10.0 GB file, 11.4 GB VRAM with 4 slots, about 50 tokens/s. |
| T2 GLM-4-9B base | Same, after downloading the base weights (18 GB). |
| T4 headline cell, 50 dialogues | 14 min with 4 workers. State accuracy 56.90%, ROUGE-1 38.93 (archived: 51.58%, 38.76). |
| T5 judge, Qwen3.8-27B Q5_K_M | 117 turns (20 dialogues) in 224 s with 4 workers; complete; mean score 9.05/10. |

Projected for 681 dialogues: about 3.2 hours for a Gemma 4 31B run and about 2.3 hours for a
judge pass.

What had to change, and what to watch:

- **GGUF conversion needed a tokenizer fix.** glm-4-9b-chat and SocratTeachLLM ship a tiktoken
  vocabulary with no `tokenizer.json`, so the converted file had no BPE merges and llama-server
  refused it ("cannot find tokenizer merges in model file"). This is the earlier "not loadable"
  failure, not an old-build problem. The convert scripts now take `tokenizer.json` from
  `zai-org/glm-4-9b-chat-hf`, whose vocabulary is identical (all 151,329 ids and the 14 added
  tokens match). Checked on 3,004 dataset strings (52,570 tokens): llama.cpp tokenizes every one
  the same as the model's own tiktoken vocabulary, and the served prompt starts with
  `[gMASK]<sop>` as the model's template requires.
- **Gemma 4 thinks by default on this llama.cpp build.** The first T4 attempt ran at 30
  dialogues/hour because the teacher wrote about 600 reasoning tokens per reply. It was stopped
  after 16 dialogues and rerun with `--reasoning off` (213 dialogues/hour). The archived headline
  run's wall time (681 dialogues in 32 minutes on a 5090) is only consistent with thinking off.
  `scripts/serve_gemma4_31b.sh` now pins it off. On the 16 dialogues both attempts share, ROUGE-1
  was 39.14 with thinking and 37.60 without, and state accuracy 54.26% and 53.76%; that sample is
  too small to call a difference. `run_config.json` does not record the reasoning setting, the
  git commit or the classifier revision; add them before Phase 4.
- **T4's state accuracy is 5.3 points above the archived cell**, just past the 5-point line set
  above. The classifier is the Hugging Face retrain, not the paper's checkpoint, and 297 turns
  give a 95% sampling error of at least ±5.6 points, so this is in line with a changed classifier. It
  confirms the headline number will move; the full run settles by how much.
- **The local judge scores high.** 9.05/10 on 20 dialogues leaves little room at the top. Compare
  with the archived Sonnet and Opus scores on the same configuration before fixing it as the
  judge of record; a stricter rubric or a second judge model may be needed (Phase 6).
- The judge script's `max_tokens=200` means a thinking judge model returns empty replies; the
  judge server must also run with `--reasoning off`.
- Port 8080 on `ollie-cachy` is held by `llama-swap`; the tests used 8085, 8086 and 8090.
- VRAM with Gemma 4 31B Q5 and four slots is 30.5 of 32 GB. Nothing else fits beside it; the
  classifier ran on the CPU.

Artifacts: `results/p1-r9700-qwen35clf-gemma31b-fewshot10-n50/` and
`results/p1-judge-throughput-n20/` (summaries in git; dialogues on `ollie-cachy` under
`~/github/csen-346-main/results/` and in `~/p1-logs/p1_results.tgz`).

### Phase 2: preparation

- [ ] **Classifier of record.** `make _classifier-ckpt` pulls the Hugging Face checkpoint.
      Re-evaluate it on the 4,304-turn Chinese test set and replace the classifier table in the
      paper. Confirm which split it was trained on: its `eval_results.json` covers 8,526 turns,
      which suggests Chinese plus English.
      **Corrected 2026-11-18: this checkpoint leaks.** It was trained on the seed-42 split of the
      English-plus-Chinese list, which is a different split from the Chinese-only one used for
      evaluation: 607 of the 681 evaluation dialogues are in its training set. It must be
      retrained on the Chinese-only train split before any state accuracy is reported. Details in
      `docs/handoffs/HANDOFF_JOURNAL_RERUNS.md`; check with
      `scripts/check_classifier_split_overlap.py`.
- [ ] **BERT consultant.** Retrain with `scripts/train_stage_classifier.py` (minutes on any card);
      the weights are gone. Publish to Hugging Face this time.
- [x] **Synthetic set.** (2026-11-18: the default config on Hugging Face already has all 75;
      nothing to pull.) Pull the 75-dialogue version from `ulises-c/SocratDataset-SYNTHETIC`
      (`n75_extension/`); the local file has 37.
- [x] **Judge completeness check.** `scripts/llm_judge_eval.py` should exit non-zero, or mark the
      summary incomplete, when it judged fewer dialogues than the run contains. The archived Sonnet
      run was judged on 274 of 681 dialogues and nothing flagged it. Add a test.
- [ ] **Local judge.** Pick one pinned open-weight model already on `ollie-cachy` (Qwen 27B class),
      temperature 0, and confirm `llm_judge_eval.py --backend api --base-url …` works against a
      local OpenAI-compatible server. Record model file hash and quantization.
- [ ] **Output archiving.** After every run, upload `results/<run>/` including `dialogues/` to a
      Hugging Face dataset repo before starting the next run. `results/` is in `.gitignore`, so
      summaries reach git only when added with `git add -f`; that is how the Gemma 12B results were
      lost. Either un-ignore `results/*/*.json` or make the force-add part of the run script.
      *Done in part 2026-11-17:* `.gitignore` now tracks `metrics_summary.json`,
      `judge_summary.json`, `run_config.json` and `semantic_r1.json` under `results/`. The Hugging
      Face dataset repo for dialogues is not created yet.
- [ ] **Seeds.** Fix the list up front: 42 for everything, plus two more for the headline cell.

### Phase 3: analyses that need no GPU

- [x] **Block bootstrap on the Gemma 12B runs** from `results/wandb_export/`: state-accuracy
      intervals for base, fine-tune, no-consultant and oracle variants. Unpaired, blocks of 10
      dialogues. *Done: `scripts/wandb_block_bootstrap.py`.* Intervals on the full 681-dialogue
      runs are about ±1.5 points. The synthetic-set runs have only 8 blocks, so their
      intervals are too narrow to rely on.
- [x] **Sample-size analysis, redone.** Rewrite `scripts/convergence_analysis.py` to resample with
      replacement and report standard errors. It can only be run on real outputs once Phase 4
      produces the first full run.
- [ ] **Weight sweep and judge bootstrap** (`scripts/unified_robustness.py`) rerun on the new runs.

### Phase 4: reruns

All on the Chinese test set, 10-shot prompt, seed 42, local judge on every dialogue, outputs
archived. Order matters: run 1 first, because the sample-size analysis on it decides whether the
rest can run at 400 dialogues.

| # | Consultant | Teacher | Set | Purpose |
|---|---|---|---|---|
| 1 | Qwen3.5 classifier | Gemma 4 31B | 681 | Headline. Run 2026-11-18 (54.50% state accuracy, ROUGE-1 36.98) **with a classifier that had seen the test dialogues; to be rerun** |
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

Eleven PRs were open on 2026-11-17, ten of them failing Lint because CI ran an unpinned ruff. That is fixed
on main. The repository ruleset requires Lint and Test to pass and the branch to be up to date, so
each merge puts the others behind again; merge one, let Dependabot rebase, repeat.

The repository's auto-merge workflow merges patch and minor bumps by itself once they are green.
On 2026-11-17, after the rebase, it merged **transformers 5.9 → 5.23** (#177) and
actions/checkout (#171). Phase 1 therefore runs on transformers 5.23; if anything in serving or
classifier loading breaks there, check that bump first.

Still needing a decision, not just a rebase:

- **ruff 0.15.22 → 0.17.5:** the newer formatter reformats about ten files. Merge it together with
  a formatting commit.
- **datasets 4 → 5:** a major bump. Run the classifier and evaluation smoke tests before merging,
  or hold it until Phase 4 is finished.
- **bitsandbytes 0.49.2 → 0.50.2:** only matters for training; 0.50 is also what a ROCm 10 torch
  needs.

**Auto-merge is paused** (2026-11-17) so the environment stays fixed while the paper's runs are
produced: the `Dependabot auto-merge` workflow is disabled and auto-merge was switched off on the
open PRs. Dependabot still opens PRs; they now wait for a manual merge. Re-enable with
`gh workflow enable dependabot-automerge.yml` once Phase 4 is finished.

## Log

- 2026-11-17: plan written. #129 merged, #101 closed, #183 and #184 opened.
- 2026-11-17: #183 and #184 merged. Dependabot auto-merge paused. transformers is at 5.23 on main.
- 2026-11-17: #185 merged; six Dependabot PRs merged, three held (#126, #174, #176).
- 2026-11-17: Phase 1 run on the R9700. T1, T2, T4 and T5 pass; T3 not needed. Judge completeness
  check, result-summary tracking and the W&B block bootstrap added.
- 2026-11-18: Phase 4 run 1 finished on the R9700 (3 h 18 min evaluation, 2 h 5 min judge pass).
  The local judge is lenient and flat across stages; the judge of record is undecided. Next steps
  are in `docs/handoffs/HANDOFF_JOURNAL_RERUNS.md`.
- 2026-11-18: found that the published classifier was trained on 607 of the 681 evaluation
  dialogues (split mismatch). Run 1 must be repeated with a retrained classifier.
