# Handoff: journal reruns (Phase 4 start)

Written 2026-11-17, late evening. The plan of record is `docs/JOURNAL_SUBMISSION_PLAN.md`; read its
Phase 1 result block first. This file says what is scheduled, how to check it, and what to do next.

## State

- **Branch:** `feat/journal-phase1-side-items`, local on the MacBook (`~/github/csen-346`), **not
  pushed, no PR**. Ten commits on top of main `5f2f645`. 257 tests pass; ruff, pyright,
  shellcheck and codespell are clean. Pushing and opening the PR needs the user's go-ahead.
- **ollie-cachy:** the worktree `~/github/csen-346-main` is on the same branch (delivered with a git
  bundle, since the branch is not on GitHub). The main checkout `~/github/csen-346` is untouched.
  The GPU is idle.
- **Phase 1 passed:** the R9700 serves SocratTeachLLM, the GLM-4-9B base and Gemma 4 31B through
  llama.cpp, and Qwen3.8-27B as a judge. No borrowed GPU is needed.
- **Constraints from the user:** no paid API calls. The R9700 is in the user's bedroom and is loud:
  GPU jobs run in the daytime only and must be finished by evening. Never start a run that would
  go into the night.

## Scheduled: headline run, 2026-11-18 10:00 PST

A one-shot cron entry on **m4-mini** (`crontab -l` there, marker `csen346-headline-once`) runs
`~/.local/bin/csen346-headline-launch`, a copy of `scripts/launch_headline_from_m4.sh`. It:

1. sends Wake-on-LAN to ollie-cachy (`~/.local/bin/wake-cachy`) if ssh does not answer, and waits
   up to 6 minutes for ssh;
2. checks the worktree;
3. starts `scripts/rerun_headline_r9700.sh` on ollie-cachy as the systemd user unit
   `csen346-headline`;
4. removes its own crontab line, so it fires once.

It refuses to start after 13:59.

The run script does preflight checks (GPU idle, ports free, files present), then:

| Stage | What | Expected | Cap |
|---|---|---|---|
| 1 | Qwen3.5 classifier + Gemma 4 31B Q5, 10-shot, 681 dialogues, 4 workers, thinking off | about 3.2 h | 5 h |
| 2 | Judge pass with Qwen3.8-27B Q5_K_M, thinking off, 4 workers | about 2.3 h | 4 h |
| 3 | Tarball of the run directory to `~/csen346-runs/` | seconds | |

Expected finish: about 15:30. Output: `results/r1-qwen35clf-gemma31b-fewshot10-n681-seed42/` in the
worktree.

Tested on 2026-11-17: the run script end to end on 2 dialogues under systemd, and the launcher
from m4-mini's cron in `--dry-run` mode. **Not tested: the Wake-on-LAN step itself**, because that
needs the machine off. It reuses the user's existing `wake-cachy` helper.

### Check on it

```bash
ssh m4-mini 'cat ~/Library/Logs/csen346-headline-launch.log'          # did the launch happen
ssh ollie-cachy 'cat ~/csen346-runs/STATUS'                             # stage by stage
ssh ollie-cachy 'tail -1 ~/github/csen-346-main/results/r1-*/progress.log'
ssh ollie-cachy 'systemctl --user status csen346-headline --no-pager | head'
```

### Stop it

```bash
ssh ollie-cachy 'systemctl --user stop csen346-headline'   # stops the run and the GPU server
ssh m4-mini 'crontab -l | grep -v csen346-headline-once | crontab -'   # cancel before it fires
```

### If it failed or was stopped

The evaluation resumes from the dialogues on disk and finished stages are skipped, so start it
again the same way (daytime only):

```bash
ssh m4-mini '~/.local/bin/csen346-headline-launch'    # refuses after 13:59
```

The launch log and `STATUS` say which step failed. If Wake-on-LAN failed, the machine has to be
switched on by hand.

## When the run has finished

1. **Collect.** Copy `metrics_summary.json`, `judge_summary.json` and `run_config.json` from the
   run directory into the repo's `results/<run>/` (they are tracked now) and commit. Copy
   `~/csen346-runs/<run>.tgz` off the machine. The Hugging Face dataset repo for dialogues does
   not exist yet; creating it is the user's decision (public or private).
2. **Check completeness.** `judge_summary.json` must say `"complete": true` with 681 dialogues.
   `run_config.json` must say `"git_dirty": false`.
3. **Read the result against the archive**, in separate tables (never mix old and new numbers):
   archived headline 55.39% state accuracy on 681 dialogues; the 50-dialogue check gave 56.90%
   against an archived 51.58%.
4. **Judge sanity check.** The local judge gave 9.05/10 on 20 dialogues, which is near the ceiling.
   Compare its per-stage scores on this run with the archived Sonnet and Opus judge summaries for
   the same configuration (`results/t4-bert-gemma-fewshot10-n681/judge_summary.json`). If it
   cannot separate systems, try a stricter rubric or the other candidate (Qwen3.6-27B) before
   judging more runs. Decide this before spending GPU time on further judge passes.
5. **Statistics.** Rewrite `scripts/convergence_analysis.py` to resample with replacement (Phase 3)
   and run it and `scripts/unified_robustness.py` on the new outputs. Add a dialogue-level
   bootstrap interval for state accuracy; the per-dialogue outputs now exist.
6. **Update the plan's log** and tick Phase 4 run 1.

## Next runs (Phase 4), each a daytime job

`scripts/rerun_headline_r9700.sh` covers only the classifier + Gemma cell. The other cells need the
same wrapper with a different teacher or consultant; generalise the script and keep its preflight,
caps and cleanup.

- BERT consultant + Gemma: retrain BERT first (`scripts/train_stage_classifier.py`, minutes; the
  weights are gone) and publish it.
- Gemma 4 31B as its own consultant (the LLM-consultant baseline that replaces GPT-4o).
- Classifier + SocratTeachLLM and classifier + GLM-4-9B base. Both GGUFs are in
  `~/Documents/models/weights/` on ollie-cachy. A 9B model leaves room for more parallel slots.
- The 75-dialogue synthetic set for the headline and SocratTeachLLM cells
  (`ulises-c/SocratDataset-SYNTHETIC`, `n75_extension/`; the local copy has 37).
- Two more seeds for the headline cell.

## Open items

- Push the branch and open the PR (waiting for the user).
- Classifier: re-evaluate the Hugging Face checkpoint on the 4,304-turn Chinese test set for the
  paper's classifier table.
- `scripts/convert_socratteachllm_to_gguf.sh` was fixed, but the SocratTeachLLM file on disk was
  made with the same method by hand; rerun the script once to confirm it.
- ollie-cachy: two f16 intermediate GGUFs (about 38 GB) can be deleted; `~/p1-logs/` holds the
  Phase 1 logs and results tarball; `~/rocm10-probe/` is the ROCm 10 probe venv.
- Port 8080 on ollie-cachy belongs to `llama-swap`. The run uses 8085 and 8086.
- Dependabot auto-merge is paused until Phase 4 ends (`gh workflow enable
  dependabot-automerge.yml`). #126, #174 and #176 stay open.
- ollie-game-server (GTX 1070) has not answered ssh; not needed so far.
- The upstream ROCm reply (rocm-libraries#7992) is on hold at the user's request.
