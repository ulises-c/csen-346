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

**Fix:** retrain the classifier on the Chinese-only seed-42 train split (36,659 training turns
plus 1,929 held out, which is what `scripts/train_state_classifier_34way.py` does today), verify
with the overlap script, publish it as a new revision, then rerun run 1. Also make the split
explicit in code so this cannot recur: split by dialogue id, the same ids for every language.
Check the BERT consultant and the Gemma 12B fine-tunes for the same problem before using them.

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
   (built 2026-11-18, not yet run against a real judge). It samples 30 turns per stage (120 turns,
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
2. **A second system.** Judge run 3 once it exists and compare with run 1.
3. **Agreement with Claude** on about 100 dialogues through the subscription backend, and with
   human scores (Phase 6).

The Sonnet-judging-Sonnet objection no longer applies, because no Claude teacher run is being
rerun. Recommended: option 1, falling back to 3 if the judge does not separate runs 1 and 3.

Do not spend GPU time on further judge passes until this is settled. Evaluations can go ahead with
`SKIP_JUDGE=1`; judging can be added later because the dialogues are kept.

## Work to do, in order

No GPU needed:

1. Open the PR for the branch (after the user's go-ahead to push) and merge it.
2. Dialogue-level bootstrap interval for state accuracy on run 1. Per-dialogue outputs now exist;
   add it next to `scripts/unified_robustness.py`.
3. Rewrite `scripts/convergence_analysis.py` to resample with replacement and report standard
   errors; run it on run 1 to decide whether later runs can use 400 dialogues.
4. Explain the 160-turn difference from the archived run.
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
