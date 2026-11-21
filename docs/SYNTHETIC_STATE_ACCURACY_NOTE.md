# Why state accuracy is about 30% on the synthetic set

Written 2026-11-20. Runs compared (same classifier checkpoint, SHA-256 `696e6e04...`, same teacher
Gemma 4 31B, few-shot 10):

- test split: `r1b-qwen35clfzh-gemma31b-fewshot10-n681-seed42` (681 dialogues of SocratDataset);
- synthetic: `r7-synth75-qwen35clfzh-gemma31b-fewshot10` (75 dialogues of
  `ulises-c/SocratDataset-SYNTHETIC`, ids 100001 to 100075).

Everything below is counted from the saved `dialogues/*.json` files (fields `state` = predicted,
`ground_truth_state`, `num_turns_ground_truth`, `num_turns_generated`) and from the dataset files
`references/KELE/SocratDataset.json`, `references/synthetic/SocratDataset_SYNTHETIC.json` and
`data/synthetic_extension/socrat_synthetic_38.json`. No model was run. The ground-truth states in
the 75 saved synthetic dialogues equal the states in the two synthetic source files (75 of 75).

## Short answer

The main cause is (a): the synthetic set labels the sub-state inside each stage differently from
SocratDataset. The five-stage structure is the same, but the generator was given only the code
ranges per stage (`scripts/generate_synthetic_socrat.py`, prompt lines 65-71: "b2-b7", "c8-c29",
"d30-d33"), not the meaning of the 34 states, so its choice of code inside a stage does not follow
SocratDataset's usage. The classifier keeps predicting SocratDataset's frequent codes (`c12`,
`c17`, `d33`), which are rare in the synthetic labels. About two thirds of the drop comes from
this.

A smaller part is (b) at the stage level: the classifier is more often one stage behind on the
synthetic dialogues, mostly on the last two turns. It does not jump to `e34` early; it reaches
`e34` too late.

(c) is not a cause: only 6 of 75 synthetic dialogues were stopped early, against 140 of 681 on the
test split, and the round limit cannot be reached in dialogues of 5 or 6 turns.

## Numbers

### Size and stage structure are similar

| | test split (r1b) | synthetic (r7) |
|---|---|---|
| dialogues | 681 | 75 |
| ground-truth turns | 4,304 | 437 |
| scored turns | 4,120 | 431 |
| ground-truth turns per dialogue | 5 to 11 (5: 169, 6: 276, 7: 139, 8: 57, 9: 30, 10: 6, 11: 4) | 5: 13, 6: 62 |
| share of scored turns per stage a / b / c / d / e | 16.5 / 18.5 / 35.1 / 16.8 / 13.1 % | 17.4 / 20.4 / 30.6 / 15.5 / 16.0 % |
| most common stage sequence | `abccde` (244), `abcde` (168) | `abccde` (43), `abbcde` (10), `abcde` (8) |
| dialogues whose last ground-truth turn is `e34` | all | all |

Both sets use only valid codes. The scored test turns use all 34 states; the synthetic set uses
25 (`a1`, `b2`-`b6`, `c8`-`c21`, `d30`-`d33`, `e34`).

### The sub-state inside a stage is distributed differently

Share of each stage's ground-truth turns (training part of SocratDataset = the 6,122 dialogues not
in the test split, 38,588 turns):

| stage | SocratDataset train | SocratDataset test | synthetic |
|---|---|---|---|
| b | b6 35.2, b2 21.1, b5 20.5, b4 17.4, b7 4.0, b3 1.8 | b6 31.8, b2 21.3, b4 20.9, b5 19.6, b7 4.2, b3 2.2 | b2 34.1, b3 26.1, b4 19.3, b5 14.8, b6 5.7 |
| c | c16 25.3, c12 21.2, c22 17.4, c9 10.8, c17 7.9 | c16 23.6, c12 20.7, c22 18.0, c9 12.8, c17 8.8 | c8 21.2, c9 14.4, c10 13.6, c15 8.3, c11 6.8, c12 6.8, c14 6.8, c16 5.3 |
| d | d33 87.4, d32 6.5, d30 3.4, d31 2.8 | d33 87.7, d32 5.0, d30 3.7, d31 3.5 | d30 46.3, d31 31.3, d32 20.9, d33 1.5 |

- `d33` is 606 of the 692 scored stage-d test turns and 1 of the 67 synthetic ones. In
  SocratDataset 94.5% of all 6,803 dialogues have `d33` as the turn before the last; in the
  synthetic set 1 of 75 does.
- `c8` is the most frequent synthetic stage-c label (28 of 132 turns). In the test split it is 9
  of 1,447 stage-c turns, and the classifier never predicts it in either run.
- The synthetic stage-c labels lean to the low codes (`c8`-`c11` are 56% of stage c) and `c22`,
  18% of SocratDataset's stage c, does not occur.
- A check that needs no classifier: give every turn its correct stage and, inside the stage, the
  most frequent training label (`a1`, `b6`, `c16`, `d33`, `e34`). That scores 58.5% on the scored
  test turns and 36.4% on the scored synthetic turns.

### What the classifier predicts

Predicted labels on the synthetic run: `c12` 64 and `c17` 74 of 160 stage-c predictions (86%;
on the test split these two are 58% of 1,555), `d33` 50 of 57 stage-d predictions, no `b3`, no
`c8`, `c10`, `c11`, `c14`, `c15`. Largest off-diagonal cells on the synthetic run (ground truth →
predicted): `e34`→`d33` 28, `b3`→`b2` 12, `d30`→`c17` 12, `c8`→`c12` 11, `e34`→`c17` 11,
`d30`→`d33` 9.

Stage-level confusion (rows ground truth, columns predicted stage a b c d e; last column is the
number of exact state matches):

| | test split | exact | synthetic | exact |
|---|---|---|---|---|
| a | 681 0 0 0 0 | 681 | 75 0 0 0 0 | 75 |
| b | 0 720 41 0 0 | 359 | 0 79 9 0 0 | 22 |
| c | 0 114 1198 114 19 | 469 | 0 31 95 6 0 | 9 |
| d | 0 0 284 287 121 | 254 | 0 0 40 21 6 | 3 |
| e | 0 0 32 124 385 | 385 | 0 0 16 30 23 | 23 |

| | test split | synthetic |
|---|---|---|
| exact state accuracy | 52.1% (2,148 / 4,120) | 30.6% (132 / 431) |
| stage correct | 79.4% | 68.0% |
| exact, given the stage is correct: b / c / d | 49.9 / 39.1 / 88.5 % | 27.8 / 9.5 / 14.3 % |
| stage recall b / c / d / e | 94.6 / 82.9 / 41.5 / 71.2 % | 89.8 / 72.0 / 31.3 / 33.3 % |
| predicted stage is behind / ahead of ground truth | 13.5% / 7.2% | 27.1% / 4.9% |

### How much each part explains

The gap is 21.5 points (52.1 to 30.6).

- Keep the synthetic run's stage decisions and give the stage-correct turns the test split's
  rate of exact matches inside each stage: 44.8%. So the sub-state labels account for about
  14 points.
- Keep the synthetic run's exact-match rates inside each stage and give it the test split's stage
  recall: 37.5%. So the stage-level errors account for about 7 points.
- The stage mix does not matter: the synthetic stage mix with the test split's per-stage accuracy
  gives 54.1%.

### The stage-level part

- By turn position on the synthetic run: turn 5 (index 4) has ground truth c 3, d 59, e 13 and
  predictions c 41, d 24, e 10; turn 6 has ground truth e 56 and predictions c 12, d 25, e 19.
- Stage-e recall is 23 of 69. The misses are `d33` (28), `c17` (11), other (7). In these turns the
  student has already given the answer, for example dialogue 100001, last turn, student: "应该是化学
  变化！因为有新物质产生了。" (answer: 化学变化), predicted `d33`.
- SocratDataset dialogues almost always contain a `d33` turn in which the teacher asks for the
  final answer before `e34`. The synthetic dialogues go from a summary question labelled
  `d30`-`d32` (or straight from stage c, 8 of 75) to `e34`. The classifier appears to wait for the
  `d33` step. This is again a difference in how the dialogues are built and labelled, but it shows
  up as a stage error, so it is counted under (b) here. The saved files do not allow the two to be
  separated further.
- Dialogue length alone does not explain it: test dialogues with 5 or 6 ground-truth turns score
  48.0% and 56.2% exact (stage correct 72.3% and 83.9%); synthetic ones 33.8% and 30.1% (72.3%
  and 67.2%).

### The loop mechanics

- Early stop at `e34` (`src/project/kele.py`, `run_single_dialogue`): 6 of 75 synthetic dialogues
  (8%) were cut, each by one turn, against 140 of 681 (20.6%) on the test split. In all 6 the
  wrong `e34` came on a stage-d turn and the dropped turn was the final `e34` turn. Counting the 6
  dropped turns the way `docs/RUN1_TURN_COUNT_NOTE.md` does (state stays `e34`, so they would be
  right) gives 138 / 437 = 31.6% instead of 30.6%.
- Round limit: `max_teaching_rounds` is 8 and forces `d33` from the eighth teaching turn on. The
  synthetic dialogues have at most 6 turns, so it never applies.
- Turn alignment: turn 1 is `a1` and turn 2 is stage b in every dialogue of both runs, for ground
  truth and prediction, so there is no offset between predicted and ground-truth turns.
- Not measurable: the saved state is the classifier's output after the no-regression guard, so
  how often the guard replaced a prediction is unknown. The guard only ever keeps the state at a
  later stage than the raw prediction, so it cannot be the reason the saved stage is behind the
  ground truth (117 of 431 turns); the saved stage is ahead in 21 of 431 turns.

### Examples (synthetic run, ground truth → predicted)

- 100001: `a1`→`a1`, `b2`→`b2`, `c8`→`c12`, `c9`→`c17`, `e34`→`d33`. Stage right in 4 of 5 turns,
  state right in 2.
- 100010: `a1`→`a1`, `b3`→`b6`, `c8`→`c12`, `c13`→`c12`, `d32`→`c22`, `e34`→`d33`. Stage right in
  4 of 6, state right in 1.
- 100040: `a1`→`a1`, `b4`→`b4`, `c10`→`b4`, `c17`→`c17`, `d32`→`c17`, `e34`→`d33`.
- A turn labelled `c8` ("provide a counterexample" in the state table): dialogue 100010, teacher
  reference "你知道植物进行光合作用！那这些让气体进出的小孔，你觉得应该叫什么名字？". This asks
  for a name and contains no counterexample; the label does not follow the state's definition.
- A turn labelled `d31` ("show the correct concept and rule directly"): dialogue 100008, teacher
  reference "非常棒！那铁钉在潮湿空气中放久了，会发生哪种变化？". This asks for the final answer,
  which is what SocratDataset labels `d33`.

## The other teachers

All five synthetic runs use the same classifier and show the same pattern (from the dialogue
files; the first three columns equal `metrics_summary.json`):

| run | exact | c | d | stage correct | exact given stage, b / c / d | cut |
|---|---|---|---|---|---|---|
| r6-synth75 gemma12b | 28.7 | 5.3 | 0.0 | 68.3 | 25.9 / 7.1 / 0.0 | 2 |
| r6-synth75 gemma12bsft | 29.7 | 4.6 | 0.0 | 68.7 | 25.0 / 6.4 / 0.0 | 3 |
| r7-synth75 gemma31b | 30.6 | 6.8 | 4.5 | 68.0 | 27.8 / 9.5 / 14.3 | 6 |
| r7-synth75 glm49b | 24.3 | 6.1 | 0.0 | 64.0 | 17.9 / 7.6 / 0.0 | 1 |
| r7-synth75 socratteachllm | 31.5 | 7.6 | 1.5 | 77.5 | 23.4 / 8.3 / 3.3 | 5 |

On the test split the runs with this classifier have 76.9% to 80.6% stage-correct turns and
35.5% to 48.4% stage-d recall.

## What this means

- Exact 34-state accuracy on the synthetic set measures agreement with labels that were assigned
  without the state definitions. It should not be reported as classifier accuracy on new data.
- Stage-level accuracy is comparable between the two sets (the stage definitions were in the
  generator prompt): 79.4% on the test split against 68.0% on the synthetic set for this pair of
  runs. That 11-point drop is the part that may reflect the classifier off its training
  distribution, and part of it comes from the missing `d33` step in the synthetic dialogues.
- These are counts from one run per set. The intervals of the stage-level figures were not
  computed here.
