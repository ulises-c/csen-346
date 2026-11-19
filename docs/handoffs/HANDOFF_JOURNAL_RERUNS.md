# Handoff: journal reruns (Phase 4)

Updated 2026-11-18, afternoon. The plan of record is `docs/JOURNAL_SUBMISSION_PLAN.md`. This file
says where things stand, what to do next, and how to run it from any machine.

## Why the reruns exist

The per-dialogue outputs behind every number in the current paper are gone: they were never
committed, the RTX 4000 Ada machine is wiped, and the 5090 machine belongs to the project partner,
who was asked on 2026-11-17 and has not answered yet. Without them the archived runs cannot be
rescored with the fixed ROUGE, re-judged, or given confidence intervals. Phase 4 regenerates them
on the R9700.

**If the partner turns up with the old `results/` directory:** the archived runs can then be
rescored and bootstrapped directly, and the Sonnet and Opus teacher runs (which will not be rerun
under the no-cost rule) become usable again. The new runs are still worth keeping: they are
reproducible from a recorded commit, and the exact classifier checkpoint behind the paper's
headline is gone either way.

## Blocker found 2026-11-18: the classifier was trained on the evaluation dialogues

**Run 1's state accuracy cannot be reported.** The classifier used
(`ulises-c/socrates-state-classifier-qwen3.5-lora`) was trained on a different 90/10 split from
the one the evaluation uses.

- `load_dataset()` shuffles the list it loads with seed 42. Between 2026-05-21 and 2026-06-09 it
  loaded SocratDataset-EN plus SocratDataset (13,606 dialogues); now it loads SocratDataset alone
  (6,803). Same seed, different list, different split.
- The published classifier was trained in that window. Its card's 77,258 training turns and 8,526
  test turns are reproduced exactly by the concatenated split.
- Of run 1's 681 evaluation dialogues, **607 were in the classifier's training set in Chinese**,
  611 in English, 678 in at least one language. Three are clean.
- Measured effect in run 1: state accuracy 54.95% on the 607 seen dialogues against 50.88% on the
  74 whose Chinese original was unseen (456 turns, so about plus or minus 5 points; 71 of those
  74 were still seen in English). Small, and not separable from noise, but the number is not
  publishable either way.
- The paper's original classifier (lost) reported 4,304 test turns, which is the Chinese-only
  test split, so it appears to have been trained on the matching split. The archived 55.39% is
  probably clean; this cannot be confirmed without the checkpoint.
- Reproduce with `scripts/check_classifier_split_overlap.py results/<run>`.

An earlier note in the plan said the split was "confirmed" with no leakage. That was wrong: it
compared the seed, not the list being shuffled.

What run 1 is still good for: timing, the pipeline check, the judge validation (it scores teacher
replies, not states), and ROUGE/BLEU only as an approximation, since the states steer the teacher.

**Fix, in order:**

1. Done: `load_dataset()` now splits each repo separately, so loading a second language can no
   longer change which dialogues are held out. The Chinese-only split is unchanged (checked
   against run 1's 681 ids), and the two-repo train split now shares no id with the test set.
2. **Retrain the classifier** on the Chinese-only seed-42 train split (36,659 training turns plus
   1,929 held out; this is what `scripts/train_state_classifier_34way.py` does by default today).
   Command from the model card:
   `uv run python scripts/train_state_classifier_34way.py --model-id Qwen/Qwen3.5-0.8B-Base --lora --lora-r 8 --lora-alpha 16 --batch_size 8 --bf16-autocast`
   **Where it runs (settled 2026-11-18):**
   - **In progress on a borrowed RTX A6000** (`hw-0047`, reached with `ssh work-laptop` then
     `ssh hw-0047`). Started 17:14, 292 steps/min with the flash-linear-attention kernels,
     projected 1.3 hours. Everything is in `~/csen346-train/` there: the source at the commit in
     `SOURCE_COMMIT`, `status.log`, `train.log`, and the output in
     `results/state-clf-qwen3.5-0.8b-lora-zh/` (`final/` is the merged model, `test_eval.json`
     the offline accuracy). No credentials were put on that machine; the datasets and base model
     are public. The system python there has no `Python.h`, which Triton needs, so the venv uses
     a uv-managed python (`uv sync --extra cuda --no-dev --python-preference only-managed`).
     Copy the result back with
     `ssh work-laptop "ssh hw-0047 'tar czf - -C ~/csen346-train/results state-clf-qwen3.5-0.8b-lora-zh/final state-clf-qwen3.5-0.8b-lora-zh/test_eval.json'" > clf.tgz`.
   - **The R9700 works too, as a fallback.** `scripts/train_classifier_r9700.sh --probe` under
     the ROCm 10.1 torch (`~/rocm10-probe` on ollie-cachy, with the project's training libraries
     added): 40 steps clean, 111 steps/min, 20 GB VRAM, projected 3.4 hours, no GPU fault in the
     kernel log. Forty steps do not settle the old fault, which appeared around step 1230; a
     full run there would.
   - The M5 Max is out: with `--device mps` the run did not finish 40 steps in 10 minutes.
   - ollie-game-server (GTX 1070, 8 GB) does not answer ssh.
3. Verify the new checkpoint with `scripts/check_classifier_split_overlap.py` logic (its training
   ids against the test ids), record its offline accuracy on the 4,304 test turns, publish it as
   a new Hugging Face revision, and point `rerun_headline_r9700.sh` at it.
4. Repeat run 1.
5. Check the Gemma 12B fine-tunes for the same problem before using run 6: which split they were trained on
   has not been checked. The loader default was two-language until 2026-06-09.

The BERT consultant is not affected: it is retrained from the current loader (see below).

## State

- **Branch:** `feat/journal-phase1-side-items`, on top of main `5f2f645`. 257 tests pass; ruff,
  pyright, shellcheck and codespell are clean. No PR yet. Check `git branch -r` to see whether it
  has been pushed; on 2026-11-18 it existed only on the MacBook and, through a git bundle, in the
  ollie-cachy worktree `~/github/csen-346-main` (at `60a482c`, two commits behind the MacBook).
- **Constraints from the user:** no paid API calls. The R9700 (ollie-cachy) is in the user's
  bedroom and is loud: GPU jobs run in the daytime only and must end by evening.
- **Phase 1 passed:** the R9700 serves SocratTeachLLM, the GLM-4-9B base and Gemma 4 31B through
  llama.cpp, and Qwen3.8-27B as a judge.

## Done: run 1, the headline cell (2026-11-18)

Qwen3.5-0.8B LoRA classifier + Gemma 4 31B Q5, 10-shot, seed 42, 4 workers, thinking off. Started
by cron from m4-mini at 10:00, finished 15:23. Evaluation 3 h 18 min, judge 2 h 5 min.

| | New run | Archived (paper) |
|---|---|---|
| State accuracy | 54.50% | 55.39% |
| Per stage a / b / c / d / e | 100 / 50.59 / 35.41 / 41.07 / 70.45 | 100 / 46.39 / 35.42 / 46.36 / 78.45 |
| ROUGE-1 / ROUGE-2 / BLEU-4 | 36.98 / 16.63 / 9.06 | 37.65 / 16.74 / 9.61 |
| Turns scored | 4,134 | 3,974 |
| Judge, out of 10 | 9.11 (Qwen3.8-27B Q5_K_M) | 8.32 (Claude Sonnet 4.6) |

- Summaries are in git: `results/r1-qwen35clf-gemma31b-fewshot10-n681-seed42/`.
- All 681 dialogues: on ollie-cachy at `~/github/csen-346-main/results/r1-…/dialogues/` and
  `~/csen346-runs/r1-qwen35clf-gemma31b-fewshot10-n681-seed42.tgz`, and on the MacBook at
  `~/csen346-archive/`. **Not on Hugging Face yet.** From another machine, copy the tarball from
  ollie-cachy.
- The judge pass is complete (681 of 681 dialogues) and `run_config.json` records commit
  `60a482c`, `git_dirty: false`, the settings and the classifier hash.

Unexplained so far: the new run scores 160 more turns than the archived one, and the per-stage mix
moved (b up, d and e down). The classifier is a different checkpoint, which probably accounts for
the stage shift; the turn count needs a look.

## Decide first: which judge

The run used a local judge because of the no-cost rule: a Sonnet pass through the API is about $15
per run (about 3.2M input tokens), and eight or more runs need judging.

What the first pass shows about the local judge: it is lenient and flat. Mean 9.11, and its stage
averages span 1.0 point (8.35 to 9.36) where Sonnet's span 2.2 (7.14 to 9.38) on the archived run
of the same configuration. Those two passes are on different dialogues, so this is a hint, not a
measurement. Whether the local judge can separate systems is unknown until a second system is
judged.

Options:

1. **Keep the local judge, validate it.** Judge run 3 (the weakest cell) as well. If it scores
   clearly lower than run 1, the judge is usable. Then run the planned agreement check: Sonnet or
   Opus through the Claude subscription (`llm_judge_eval.py --backend claude-code`) on about 100
   dialogues of run 1, and report agreement.
2. **Sonnet through the subscription for everything.** `--backend claude-code` exists. A full pass
   is 4,134 sequential calls per run with no temperature control, and it counts against the
   subscription's usage limits; how many runs fit is untested. Check that this use is within the
   subscription terms.
3. **Stricter local judge.** Tighter rubric or a different model (Qwen3.6-27B is on ollie-cachy),
   tested on run 1's dialogues before use. Re-judging run 1 costs about 2 hours of GPU.
4. **Pay for Sonnet on the headline comparison only** (about $30 for runs 1 and 3). Outside the
   current no-cost rule; the user's call.

**Decided 2026-11-18 (user): validate the local judge, whichever model ends up as judge.**

Another candidate, suggested by the user: **Qwen3.8-Flash-Next** (125B mixture-of-experts, about
3B active per token). It is on ollie-cachy and already configured in llama-swap
(`~/.config/llama-swap/config.yaml`, model ids `qwen3.8-flash-next` and
`qwen3.8-flash-next-radiance`, port 8080); weights under `~/models/Qwen3.8-Flash-Next-Gyro/` and
`~/models/ista/IQ3_S/`. Unknown and to be measured before relying on it: throughput (most experts
sit in RAM, so probably one request at a time), whether replies are repeatable at temperature 0
with its draft model on, and how to pin it for the paper, since it runs on a non-standard engine
at a 3-bit-class quantization. A larger model is not automatically a stricter judge.

Validation, cheapest first:

1. **Controls on run 1's dialogues, no new evaluation needed: `scripts/judge_validation.py`**
   (built and run 2026-11-18). It samples 30 turns per stage (120 turns,
   seed 42), keeps the conversation history, and has the judge score seven replies per turn: the
   system's, the reference, and five bad controls, each aimed at one rubric axis (a reply swapped
   in from another dialogue, the previous reply repeated, a flat statement with no question,
   university-level jargon, the closing summary given early). It reports the paired drop against
   the system's reply with a dialogue-level bootstrap interval, flags controls the judge does not
   detect, and scores each system reply twice for repeatability. 887 judge calls, about 27
   minutes at Qwen3.8-27B's speed. Every judge gets the same items, and
   `--compare A.json B.json` gives item-level agreement between two judges.

   ```bash
   # on ollie-cachy, daytime, with a judge server up (reasoning off) on port 8086
   .venv/bin/python scripts/judge_validation.py results/r1-qwen35clf-gemma31b-fewshot10-n681-seed42 \
     --backend api --base-url http://127.0.0.1:8086/v1 --api-key local --model Qwen3.8-27B-Q5_K_M
   # Claude through the subscription, from any machine with the dialogues (smaller sample)
   uv run python scripts/judge_validation.py <run dir> --backend claude-code \
     --model claude-sonnet-4-6 --per-stage 10
   ```

   Run it for Qwen3.8-27B, Qwen3.6-27B and Qwen3.8-Flash-Next, then compare. The summary files
   are small; commit them. There is no wrapper yet that starts the judge server, runs this and
   stops the server; `scripts/rerun_headline_r9700.sh` stage 2 shows the server command.
   **Result for the two Qwen 27B judges** (120 turns, 767 items each, 26 minutes each on the
   R9700; files in `results/r1-…/judge_validation_*.json`):

   | | Qwen3.8-27B Q5_K_M | Qwen3.6-27B Q5_K_XL |
   |---|---|---|
   | System reply, mean | 9.17 | 8.34 |
   | Reference reply, mean | 9.34 | 9.57 |
   | Drop for swapped / repeat / statement / jargon / early closure | 6.5 / 5.1 / 8.8 / 9.2 / 7.6 | 7.1 / 4.2 / 8.3 / 8.3 / 7.8 |
   | Bad controls detected | 5 of 5 | 5 of 5 |
   | System replies given 10/10 | 73 of 120 | 67 of 120 |
   | System replies scored 6 or lower | 1 | 23 |
   | Same total on a repeat | 97% | 98% |

   Reading: neither judge is blind. Both put every kind of broken reply far below the real one,
   on the intended axes, and both are repeatable. That is a floor test and they pass it. The
   weakness is at the top: both give full marks to more than half of the real replies, although
   the rubric says a 10 should be rare, so they have little room to separate two competent
   systems. Qwen3.6-27B uses more of the scale and ranks the reference above the system reply
   (by 1.2 points; Qwen3.8 does not separate them). On real replies the two agree only
   moderately (rank correlation 0.60). The controls do not test fine distinctions; step 2 and
   step 3 below do. Qwen3.6-27B looks like the better candidate so far; a rubric that makes 10
   harder to earn is worth testing on these same items.

2. **A second system.** Judge run 3 once it exists and compare with run 1.
3. **Agreement with Claude** on about 100 dialogues through the subscription backend, and with
   human scores (Phase 6).

The Sonnet-judging-Sonnet objection no longer applies, because no Claude teacher run is being
rerun. Recommended: option 1, falling back to 3 if the judge does not separate runs 1 and 3.

Do not spend GPU time on further judge passes until this is settled. Evaluations can go ahead with
`SKIP_JUDGE=1`; judging can be added later because the dialogues are kept.

## Work to do, in order

Done on 2026-11-18 (all on the branch, PR #186):

- **Bootstrap intervals:** `scripts/state_accuracy_bootstrap.py` (resamples dialogues; `--paired`
  for two runs). Run 1: 54.50% [52.89, 56.13], stage-balanced 59.50% [57.81, 61.16]. These are
  from the run with the leaking classifier; the method stands, the numbers will be replaced.
- **Sample size:** `scripts/convergence_analysis.py` rewritten to resample with replacement.
  Standard error of state accuracy is 1.07 points at 400 dialogues and 0.82 at 681. Two
  independent runs can detect a difference of about 4.2 points at 400 and 3.3 at 681 (rule of
  thumb, 95% level, 80% power). Paired comparisons on the same dialogues should do better; one
  run cannot say by how much. `README.md` and `docs/CONVERGENCE_ANALYSIS.md` still describe the
  old method and its "400 is sufficient" conclusion and need rewriting.
- **Turn counts:** `docs/RUN1_TURN_COUNT_NOTE.md`. A dialogue stops when the consultant predicts
  `e34` (`src/project/kele.py`), so each system is scored on a different set of turns. Run 1
  dropped 170 of 4,304 turns in 126 dialogues; the archived run dropped about 330. Report turn
  counts beside every turn-level metric and compare by dialogue.
- **Synthetic set:** nothing to pull. The default config of `ulises-c/SocratDataset-SYNTHETIC`
  already has all 75 dialogues; use `--hf-repo ulises-c/SocratDataset-SYNTHETIC --split all`.

No GPU needed, still to do:

1. Merge PR #186.
2. Rewrite the sample-size sections of `README.md` and `docs/CONVERGENCE_ANALYSIS.md`.
3. Decide how to handle early-stopped dialogues in the paper (score all ground-truth turns, or
   report per-dialogue metrics).
4. Flash-Next as a judge: `llm_judge_eval.py` caps replies at 200 tokens and sends no reasoning
   setting; Flash-Next thinks by default behind llama-swap (port 8080, needs the llama-swap key).
   Add a token cap and reasoning-effort option, then run the validation against it.
5. Create the Hugging Face dataset repo for run outputs and upload run 1 (the user decides public
   or private).
6. Generalise `scripts/rerun_headline_r9700.sh` so the teacher and consultant are parameters. Keep
   its preflight, caps, resume and cleanup.
7. Retrain the BERT consultant (`scripts/train_stage_classifier.py`, minutes on any card) and
   publish it; the weights are gone.
8. Re-evaluate the Qwen3.5 classifier on the 4,304-turn Chinese test set for the paper's
   classifier table.
9. Pull the 75-dialogue synthetic set (`ulises-c/SocratDataset-SYNTHETIC`, `n75_extension/`).

GPU, daytime, one run per day unless shortened by item 3 (table in the plan, Phase 4):

10. Run 3: Gemma 4 31B as its own consultant. The baseline, and the judge test. Slower than run 1
    because the consultant is an LLM call per turn; time 20 dialogues first.
11. Run 2: BERT + Gemma.
12. Runs 4 and 5: classifier + SocratTeachLLM, classifier + GLM-4-9B base. GGUFs are in
    `~/Documents/models/weights/` on ollie-cachy.
13. Run 6: Gemma 4 12B base and fine-tune.
14. Run 7: synthetic set. Run 8: two more seeds of run 1.

Then Phases 5 to 7 in the plan (ablations, judge validation with humans, paper and release items).

## How to run a job

From any machine with ssh to m4-mini or ollie-cachy.

```bash
# start (wakes ollie-cachy if needed; refuses after 13:59)
ssh m4-mini '~/.local/bin/csen346-headline-launch'
# or, with the machine already on:
ssh ollie-cachy 'cd ~/github/csen-346-main && systemd-run --user --unit csen346-headline --collect \
  -p WorkingDirectory=$PWD -E HOME=$HOME /usr/bin/bash scripts/rerun_headline_r9700.sh'

ssh ollie-cachy 'cat ~/csen346-runs/STATUS'                    # progress, stage by stage
ssh ollie-cachy 'systemctl --user stop csen346-headline'       # stop the run and the GPU server
ssh ollie-cachy 'cd ~/github/csen-346-main && bash scripts/rerun_headline_r9700.sh --preflight'
```

Environment for the run script: `RUN_NAME`, `LIMIT`, `SKIP_JUDGE=1`, `EVAL_CAP_S`, `JUDGE_CAP_S`,
`JUDGE_MODEL`. Through the launcher, pass them as `RUN_ENV="K=V K=V"`. A stopped evaluation
resumes from the dialogues on disk; finished stages are skipped.

To schedule: a crontab line on m4-mini, for example
`0 10 19 11 * $HOME/.local/bin/csen346-headline-launch # csen346-headline-once`.

After a run: copy the three summary files into `results/<run>/` in the repo and commit; copy
`~/csen346-runs/<run>.tgz` off the machine; check `"complete": true` in `judge_summary.json`.

To get new code onto ollie-cachy: once the branch is on GitHub, `git fetch` in `~/github/csen-346`
there and check it out in the worktree `~/github/csen-346-main`. Leave the main checkout alone.

## Known problems

- **The launcher cannot remove its own crontab line.** macOS refuses crontab edits from inside
  cron ("Operation not permitted"). The 2026-11-18 entry was removed by hand. After a scheduled
  run, remove the line yourself: `crontab -l | grep -v csen346-headline-once | crontab -` over
  ssh. Fix: guard on the date inside the script, or schedule with launchd.
- **Wake-on-LAN is untested** in this launcher; ollie-cachy was already on when it fired. It
  reuses the user's `wake-cachy` helper.
- The launcher does not power ollie-cachy down afterwards.
- The judge log prints a dollar cost for the local model; it is a token count at API prices, not a
  charge.
- `scripts/convert_socratteachllm_to_gguf.sh` was fixed, but the SocratTeachLLM file on disk was
  made with the same method by hand; rerun the script once to confirm it.
- Port 8080 on ollie-cachy belongs to `llama-swap`. The run uses 8085 and 8086.

## Other open items

- ollie-cachy: two f16 intermediate GGUFs (about 38 GB) can be deleted; `~/p1-logs/` holds the
  Phase 1 logs; `~/rocm10-probe/` is the ROCm 10 probe venv.
- Dependabot auto-merge is paused until Phase 4 ends (`gh workflow enable
  dependabot-automerge.yml`). #126, #174 and #176 stay open.
- ollie-game-server (GTX 1070) has not answered ssh; not needed so far.
- The upstream ROCm reply (rocm-libraries#7992) is on hold at the user's request.
