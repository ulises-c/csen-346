# Visualization & Stats Plan

Quick-reference TODO list of charts, tables, and statistics to produce for the paper and final report. Grouped by analysis area. All data sources are already on disk — no new runs needed for dataset-level items.

---

## 1. Dataset Analysis

> Source: `resources/KELE/SocratDataset.json` (6,803 dialogues, 42,892 turns)

- [ ] **Dialogue length distribution** — histogram of `dialogueRound` (turns per dialogue, range 5–12). Show mean (6.3) and std. Useful framing stat for the intro.
- [ ] **Stage turn distribution** — bar chart of total turns per stage (a: 6,803 · b: 7,480 · c: 14,445 · d: 7,361 · e: 6,803). Stage c is ~2× any other — motivates why c is the hardest to predict.
- [ ] **Question type breakdown** — pie or grouped bar: 选择题 (multiple choice) vs 判断题 (true/false). Include counts.
- [ ] **Grade / chapter heatmap** — rows = grade levels, cols = chapter numbers, cell = dialogue count. Shows dataset coverage / imbalance by curriculum position.
- [ ] **State label frequency** — horizontal bar chart of all 34 states sorted by count. Highlights rare states that the model has almost no training signal for.

---

## 2. Baseline Metrics

> Source: `results/baseline/metrics_summary.json` (681 dialogues, 4,294 turns)

- [ ] **Metric comparison table** — our run vs paper GPT-4o baseline vs paper SocratTeachLLM, for ROUGE-1/2/L and BLEU-4. Already in EXPERIMENT_LOG; format as a clean paper table with ± cells where available.
- [ ] **Radar / spider chart** — four axes (ROUGE-1, ROUGE-2, ROUGE-L, BLEU-4) with one polygon per system (paper SocratTeachLLM, paper GPT-4o, our baseline). Good summary figure for the paper.
- [ ] **Per-stage state accuracy bar chart** — bars for stages a/b/c/d/e with values 95.15 / 36.93 / 4.70 / 5.04 / 11.92. Add a dashed horizontal line at the overall mean (25.94%). The drop from b→c is the key story.
- [ ] **State accuracy heatmap** — rows = predicted stage (a–e), cols = ground truth stage (a–e), cell = accuracy or count. Visualizes which stages are confused with which. Stage c being misclassified as what?

---

## 3. Error Analysis

> Source: `results/baseline/dialogues/*.json` (681 files, one per dialogue)

- [ ] **State confusion matrix** — 5×5 or 34×34 version. At minimum show stage-level (5×5): where do wrong predictions land? Is c predicted as b? As a?
- [ ] **Per-dialogue ROUGE distribution** — box plots or violin plots of ROUGE-1/2/L per dialogue. Wide spread would suggest some dialogues are well-handled and others collapse entirely.
- [ ] **BLEU-4 by dialogue length** — scatter: x = `num_turns_generated`, y = BLEU-4. Any correlation? Longer dialogues harder?
- [ ] **State accuracy by question type** — grouped bar: 选择题 vs 判断题 × per-stage accuracy. Does the system handle one type better?
- [ ] **State accuracy by `dialogueRound`** — line chart bucketed by turn count (5, 6, 7, …, 12). Does accuracy degrade for longer dialogues?
- [ ] **Turn-level accuracy curve** — x = turn index within dialogue (1–N), y = fraction of turns where state prediction is correct. Shows if the model does better early (stage a) and worse mid-dialogue (stage c/d).

---

## 4. Latency & Cost

> Source: `results/baseline/dialogues/*.json` (`elapsed_seconds` per dialogue) + EXPERIMENT_LOG cost breakdown

- [ ] **Per-dialogue latency histogram** — `elapsed_seconds` distribution across 681 dialogues. Mark median, p95.
- [ ] **Cost breakdown table** — input tokens, output tokens, cost per dialogue, cost per turn, total. Already in EXPERIMENT_LOG; clean up for paper.
- [ ] **Latency vs dialogue length scatter** — `elapsed_seconds` vs `num_turns_generated`. Should be roughly linear; outliers flag API retry events.
- [ ] **Token usage by stage** — estimated input/output tokens per turn broken down by stage. Stage a should be cheapest (shorter context); stage c/d most expensive.

---

## 5. Experiment Comparison (fill in as runs complete)

> Source: `results/{experiment_name}/metrics_summary.json` — add a row per new run

- [ ] **Grouped bar chart: metrics by experiment** — x = experiment name (baseline, RAG, BERT-consultant, fused, …), clustered bars for ROUGE-1/2/L and BLEU-4. The main results figure.
- [ ] **State accuracy improvement table** — per-stage accuracy for each experiment variant, delta from baseline highlighted in green/red.
- [ ] **Cost vs accuracy scatter** — x = total API cost per run, y = overall state accuracy. Shows efficiency frontier across approaches.
- [ ] **ROUGE-1 vs BLEU-4 scatter** — one point per experiment. Are there approaches that help ROUGE but hurt BLEU (or vice versa)?

---

## 6. Approach-Specific Visuals (per improvement in IMPROVEMENT_PLAN.md)

- [ ] **RAG (#1)** — retrieval quality plot: top-k similarity scores for a sample of test questions. Shows whether retrieval is actually finding relevant exemplars.
- [ ] **BERT Classifier (#2)** — training loss / validation accuracy curves. Per-stage precision/recall/F1 table for the classifier alone.
- [ ] **Persistent Memory (#3)** — prompt token count before vs after memory compression, per turn. Shows the 30% token reduction claim.
- [ ] **Fused model (#4)** — training loss curve. Compare structured output accuracy (state + response jointly) vs two-pass baseline.
- [ ] **Data augmentation (#7, #9, #10)** — new vs original dataset size, stage distribution before/after augmentation (check if c-stage coverage improves).

---

## 7. Paper-Ready Figures (priority order)

1. **Figure 1** — KELE system architecture diagram (already conceptual; make a clean Mermaid or draw.io version).
2. **Figure 2** — Stage turn distribution bar chart (motivates the dataset's structure).
3. **Figure 3** — Per-stage state accuracy baseline (the key failure-mode figure).
4. **Figure 4** — Grouped metric comparison across experiments (main results).
5. **Table 1** — Baseline metrics vs paper, formatted.
6. **Table 2** — Per-stage accuracy across all experiment variants.
7. **Appendix** — State confusion matrix, per-dialogue ROUGE distributions.

---

## Implementation Notes

- Use `matplotlib` / `seaborn` for all charts. Save as PDF for Overleaf, PNG for slides.
- Parse dialogue JSON files with a single pass script (`src/project/viz.py` — to be created) that computes all per-dialogue stats in one sweep.
- Grade/chapter heatmap requires parsing Chinese grade strings (e.g., `"1小学一年级上册"`) — write a normalization helper.
- State confusion matrix needs the 34 state labels loaded from `resources/KELE/SocratDataset.json` — extract unique states during the single-pass script.
