# Why run 1 scored 4,134 turns and the archived run 3,974

Written 2026-11-18. Runs compared:

- new: `r1-qwen35clf-gemma31b-fewshot10-n681-seed42` (per-dialogue outputs available, 681 files);
- archived: `t4-bert-gemma-fewshot10-n681` (only `metrics_summary.json`, `judge_summary.json` and
  `run_config.json` survive; the dialogues are gone).

## Short answer

The evaluation loop stops a dialogue as soon as the system's state becomes `e34`, even when the
ground-truth dialogue has more student turns. The turns after that point are never generated and
never scored. The new run stopped 126 of 681 dialogues early; the archived run stopped about 217
to 219. The two runs used different state classifiers, and the classifier decides when `e34` is
predicted. So the hypothesis is confirmed for the mechanism and for the direction. That the
classifier checkpoint is the only cause cannot be proven, because the archived dialogues, the
archived classifier checkpoint and the archived code version are not available.

## What ends a dialogue (code)

- `src/project/kele.py`, `run_single_dialogue`, lines 193-215: the loop replays the ground-truth
  student turns one by one. After each turn, lines 214-215 break out of the loop if
  `system.current_state == "e34"`. Nothing else ends the loop, so the number of generated turns
  can be lower than the number of ground-truth turns but never higher.
- `max_teaching_rounds` (8 in both runs' `run_config.json`) does not end a dialogue. In
  `src/project/socratic_teaching_system.py`, lines 568-590, reaching the limit forces the state to
  `d33` unless the consultant predicted `e34`; the loop in `kele.py` keeps going.
- The stage guard in `socratic_teaching_system.py`, lines 537-553, stops the state from moving
  back to an earlier stage. It cannot produce `e34` by itself.
- The state stored at line 602 is the consultant's prediction after these two corrections. `e34`
  is the only state of stage e, so "the consultant predicts stage e" and "the dialogue ends" are
  the same event.

## What the new run shows (measured on its 681 dialogue files)

- Ground-truth turns: 4,304. Generated and scored turns: 4,134. Difference: 170.
- 126 dialogues have `num_turns_generated < num_turns_ground_truth`; none has more.
- Shortfall per cut dialogue: 1 turn in 88 dialogues, 2 turns in 34, 3 turns in 2, 4 turns in 2
  (88 + 68 + 6 + 8 = 170).
- In all 126, the predicted state of the last generated turn is `e34`, and the ground-truth state
  of that turn is not stage e (stage d in 105, stage c in 21). No dialogue has `e34` predicted
  before its last generated turn. So every cut is the `e34` break, triggered by a wrong
  prediction of the final state.
- Every ground-truth dialogue in the test set ends with exactly one `e34` turn. The 555 complete
  dialogues therefore account for the 555 stage-e turns scored, and each cut dialogue loses its
  stage-e turn.
- Ground-truth stage of the 170 dropped turns (from `references/KELE/SocratDataset.json`, matched
  by dialogue id): c 4, d 40, e 126. Full test set per stage: a 681, b 761, c 1,447, d 734,
  e 681.

## What the archived summaries show

- `metrics_summary.json`: 3,974 turns. `judge_summary.json`: 3,971 turns judged, per stage a 681,
  b 761, c 1,423, d 644, e 462. Three turns were scored by the metrics but not by the judge; which
  ones is not recorded.
- Against the full test set the archived judge counts lack c 24, d 90, e 219 turns (333 in total;
  330 against the metrics count). Stages a and b are complete in both runs, as expected: a cut can
  only happen after `e34` is predicted, and it removes turns from the end.
- Because each dialogue has one stage-e turn, at its end, the number of missing stage-e turns
  is the number of cut dialogues: 219 in the archived run, minus at most the 3 turns the judge
  skipped. The per-dialogue judged counts in the archived `judge_summary.json` give 220 dialogues
  shorter than their ground truth, which also includes judge skips. Both readings give 217 to 219
  cut dialogues, against 126 in the new run.
- Per dialogue, new generated turns minus archived judged turns: 0 in 507 dialogues, +1 in 119,
  +2 in 21, +3 in 9, and -1 in 25. So the new run is not uniformly longer: 25 dialogues were cut
  earlier in the new run than in the archived one. Of the dialogues short in either run, 111 are
  short in both, 109 only in the archived run and 15 only in the new run (archived side again
  subject to the 3 skipped turns).

## What cannot be determined

- The archived run's predicted states. Without them it cannot be checked directly that each
  archived cut was an early `e34`; this is inferred from the code and from the missing stage-e
  turns.
- Which classifier checkpoint the archived run used. Its `run_config.json` records no classifier
  path, hash or git commit, and `docs/handoffs/HANDOFF_JOURNAL_RERUNS.md` says the checkpoint is
  gone. The new run records `results/state-clf-qwen3.5-0.8b-lora-wandb/final` with a SHA-256.
- Which code version produced the archived run. The `e34` break has been in `kele.py` since commit
  `71f9984` (2026-04-13), before the archived run (2026-05-26), but the run does not record the
  commit it was made from.
- Whether other differences between the runs (teacher quantisation, server build, few-shot
  retrieval) also moved the count. They can only act through the predicted state, since nothing
  else ends a dialogue, but their share cannot be separated from the classifier's.
- The exact number of archived cut dialogues (217 to 219) and which three turns the judge skipped.

## What this means for comparing systems

Two systems are scored on different sets of turns. The dropped turns are not a random sample: they
are the last turns of dialogues in which the system predicted the final state too early.

- State accuracy. The turn with the wrong `e34` is scored (as wrong), the later turns are not. Had
  the dialogue continued, the code would have kept the state at `e34` (the stage guard and the
  round limit both leave `e34` in place), so the dropped stage-c and stage-d turns would have been
  wrong and the dropped final turn right. For the new run this gives, on all 4,304 ground-truth
  turns, 55.27% overall instead of 54.50%, and per stage c 35.31, d 38.83, e 75.92 instead of
  35.41, 41.07, 70.45 (stage-balanced 60.13 instead of 59.50). These are derived from the code and
  the counts above, not from a rerun. The effect is largest on stage e: its accuracy is computed
  only on dialogues that were not cut, and a system that cuts more dialogues has a smaller and
  different denominator.
- ROUGE, BLEU and judge scores. A cut dialogue contributes no summary turn and fewer stage-d
  turns. The archived run has 462 stage-e turns in its judge average and the new run 555, so the
  per-stage and overall means are taken over different turns. Part of any difference between the
  two runs in these metrics can come from which turns were scored; how much cannot be measured
  without the archived dialogues.
- Practical consequences: report the number of scored turns (overall and per stage) next to every
  turn-level metric; compare systems with dialogue-level resampling
  (`scripts/state_accuracy_bootstrap.py --paired`), which keeps each dialogue's turns together but
  does not align them; and, when the dialogues of both systems exist, add a check restricted to
  the turns both systems produced. Whether the evaluation should stop at `e34` at all is a
  separate decision; changing it would make new runs incomparable with every run made so far.
