# KELE Improvement Plan — 10 Novel Approaches

**CSEN 346 · Santa Clara University · 2026-04-14**

Ten domain-specific improvements over the KELE baseline (`SocratTeachLLM` teacher + GPT-4o consultant). All ten are composable — the final product does not need to use all ten, but each is independently defensible as a research contribution and each attacks a concrete failure mode we *measured* in the 2026-04-14 baseline run.

---

## Baseline weaknesses we are actually targeting

From `results/baseline/metrics_summary.json` (681 dialogues, 4,294 turns):

| Weakness | Evidence | Primary approach(es) targeting it |
|---|---|---|
| **Mid-stage state misclassification** | stage-c 4.7%, stage-d 5.04% vs stage-a 95.15% | #2, #4, #10 |
| **BLEU-4 gap** (19.60 vs paper 41.96) | surface-form drift; right meaning, wrong words | #6, #7, #9 |
| **Two-pass inference latency + cost** | ~$17.49/run, 4h34m wall time | #1, #2, #4 |
| **Stateless consultant** | re-reads history every turn; no student memory | #3, #4, #8 |
| **Loss ≠ pedagogical outcome** | trained on token overlap, not "did the student learn" | #5, #8, #9 |
| **Small dataset** (~6.8k dialogues) | under-trained representations of rare misconceptions | #7, #9, #10 |

---

## Group A — Ship-fast (days, not weeks)

### 1. Retrieval-Augmented Teacher (RAG)
**Core:** Index every dialogue in the SocratDataset train split by topic + concept using a Chinese sentence-encoder (e.g. `BAAI/bge-large-zh-v1.5`). At inference, for each incoming question retrieve the top-3 most similar past dialogues and prepend them as few-shot exemplars to the teacher prompt.

**Why it works here:** ROUGE/BLEU gap is driven by surface-form drift — the teacher produces fluent Chinese Socratic questions, but not in the *style* of the ground-truth teacher. Retrieval injects that style without any fine-tuning. This is also the single highest-leverage idea per hour of engineering work.

**Expected lift:** +3-6 points ROUGE-1, +4-8 points BLEU-4. Ceiling limited by retrieval quality on rare topics.

**Compose with:** Every other approach. Especially strong with #6 (semantic reward) and #8 (persistent student).

**Cost:** ~1 day to implement. Index fits in 1 GB. Zero training.

---

### 2. Hierarchical State Classifier (replace GPT-4o consultant)
**Core:** Replace the frontier-LLM consultant with a two-stage Chinese BERT classifier: first head predicts stage (a/b/c/d/e — 5-way), second head predicts within-stage state conditioned on stage (max 22-way for stage c). Trained on the ~42,000 (input, ground-truth-state) pairs from the SocratDataset train split.

**Why it works here:** The 30-way flat classification is WHY even GPT-4o gets stage-c right only 4.7% of the time — it's one of the hardest discrete classification tasks in the pipeline. A 125M-parameter BERT trained directly on the labels will crush a zero-shot 200B-parameter model on this task, because the task has *training data we've been ignoring.*

**MoE variant (the version Max mentioned):** Replace the second head with a Mixture-of-Experts where each expert specializes in one stage. Hard routing by stage prediction → within-stage specialist. Same parameter budget, better specialization.

**Expected lift:** +20-35 points on mid-stage accuracy. Consultant cost drops from $17/run to ~$0 (inference is free on the 5090). Latency drops ~4× per turn.

**Compose with:** Everything — this is pure plumbing replacement, downstream agents don't notice.

**Cost:** ~3 days. Training in ~1 hour on the 5090. Needs label cleanup.

---

### 3. Persistent Student Model Memory
**Core:** Maintain a growing structured representation of *this student's demonstrated misconceptions and competencies across the entire dialogue.* Every turn, a small adapter model updates this representation (JSON: `{"knows": [...], "confused_about": [...], "answered_correctly": int, ...}`). Both consultant and teacher receive the representation as context.

**Why it works here:** Socratic teaching requires *theory of mind.* The current system has none — every turn, both agents see the same raw history and have to re-derive the student's state from scratch. A separate persistent representation does this once and caches it. Also dramatically reduces prompt length (→ lower cost, higher throughput, cache hits).

**Expected lift:** +2-5 points state accuracy, substantial qualitative improvement in dialogue coherence. Secondary: ~30% token reduction per turn.

**Compose with:** #2 (classifier has a much better input feature), #4 (the fused model's internal state becomes explicit), #8 (RL reward can include "did the memory update reflect real student progress?").

**Cost:** ~4 days. Can bootstrap the update model with a prompted GPT-4o then distill.

---

## Group B — Substantive training wins (weeks)

### 4. Consultant-Teacher Fusion (single end-to-end model)
**Core:** Replace the two-agent architecture with a single model fine-tuned to produce a *structured output* in one forward pass: `{"state": "c16", "action": "...", "teacher_response": "..."}`. Train on the SocratDataset train split, where state and teacher response are ground truth.

**Why it works here:** The two-agent split is an *architectural artifact* of the paper's original need to use a closed model (GPT-4o) alongside a fine-tuned open model. With a single fine-tune we eliminate the round-trip, train the two tasks jointly (state picks that help generation will emerge naturally), and the model can learn implicit state representations that outperform the hand-designed 30-state schema.

**Expected lift:** +5-10 points ROUGE-1, latency halves, cost falls to teacher-only compute. Risk: single-model may overfit to surface form — mitigate with #6.

**Compose with:** #3 (memory feeds into the fused prompt), #6 (semantic reward during fine-tuning), #8 (multi-turn RL as the final training stage).

**Cost:** ~2 weeks. LoRA fine-tune of SocratTeachLLM on 5090. ~8h training time.

---

### 5. Process Reward Model + Best-of-N Sampling
**Core:** Train a small Chinese PRM (process reward model) on labeled (dialogue_history, teacher_response, student_next_turn) triples. Label = did the student's next response show progress toward the correct answer? At inference, sample N=8 teacher responses and pick the highest-scoring one.

**Why it works here:** PRMs have driven dramatic gains in math/code reasoning (OpenAI o1, DeepSeek-R1). They've never been applied to Socratic teaching, where "progress" is the defining signal. The infrastructure for multi-turn evaluation we already built makes labeling tractable: run rollouts, measure outcome, label each intermediate response.

**Expected lift:** +4-8 points BLEU-4 (best-of-N picks the most on-style response), +2-4 state accuracy, qualitative improvement. Cost: 8× inference at generation time.

**Compose with:** #4 (verifier on the fused model's output), #9 (student-simulator provides the rollout environment for PRM training).

**Cost:** ~2 weeks. PRM is small (~500M params). Training uses GPT-4o to label ~20k trajectories (~$50).

---

### 6. Semantic Reward Fine-Tuning (BLEU → BERTScore/embedding loss)
**Core:** Fine-tune the teacher with a reward that is *not* token-level BLEU/ROUGE. Use BERTScore against ground truth + a Chinese sentence-embedding cosine similarity. This optimizes for "meaning preserved" rather than "words matched."

**Why it works here:** Our baseline diagnostic is the smoking gun — ROUGE is close to paper, BLEU is far. BLEU-4 specifically requires 4-gram exact matches, which fails on valid paraphrases. The ground-truth teacher in SocratDataset has a specific phrasing style, but there are many equally-good Socratic questions. Training on semantic similarity lets the model generalize beyond the exact corpus style.

**Expected lift:** +5-10 points BLEU-4 *and* ROUGE, because better semantic training also produces better surface-form matching (counterintuitively). Risk: reward hacking — model learns to embed-match without real Socratic content. Mitigate with #5.

**Compose with:** #4 (fine-tuning objective), #9 (reward during RL).

**Cost:** ~1 week. Infrastructure is the main lift; reward computation is fast.

---

### 7. Counterfactual Dialogue Augmentation
**Core:** Use GPT-4o to generate 3 counterfactual variants of each SocratDataset training dialogue: (a) student holds one misconception different from the original, (b) student responds "one stage earlier" (more confused), (c) student is distracted or off-topic. Filter with a quality classifier. Fine-tune teacher on the 4× expanded dataset.

**Why it works here:** SocratDataset has ~6,800 dialogues, small for fine-tuning. The rare-misconception tail is especially underrepresented (why stage-c accuracy is poor — rare inductive-reasoning errors have few examples). Counterfactual generation systematically expands the tail.

**Expected lift:** +2-4 points across all metrics; most improvement on underrepresented stage transitions. Risk: synthetic data drift — mitigate with a held-out real test set (what we already have).

**Compose with:** #4, #9 (augments what gets trained on).

**Cost:** ~1 week + ~$200-400 in GPT-4o generation spend.

---

## Group C — High-risk, high-reward (month+)

### 8. Multi-Turn RL with Outcome-Based Reward
**Core:** Treat the dialogue as an MDP. Reward = `1{student_reaches_correct_answer} × 1/dialogue_length`. Train the teacher (after SFT) with PPO or REINFORCE, using a rollout environment where the consultant and a *student simulator* (see #9) interact with the teacher. Directly optimizes the *actual pedagogical goal* instead of token overlap.

**Why it works here:** This is the move from "imitate the teacher corpus" to "be an effective teacher." The paper never does this. It would be a legitimate research contribution. The reward is simple, unambiguous, and exactly what Socratic teaching is trying to achieve. Also: shorter successful dialogues are *better* — current training has no pressure for efficiency.

**Expected lift:** Hard to predict — could be dramatic (+10-15 across all metrics) or could collapse (reward hacking: teacher just tells answer immediately). Requires careful reward design with KL penalty to SFT baseline.

**Compose with:** #5 (PRM as dense intermediate reward), #9 (student simulator is the environment).

**Cost:** ~3-4 weeks. Needs rollout infrastructure + PPO training loop.

---

### 9. Student Simulator Self-Play
**Core:** Train a small "student" LLM on the student side of SocratDataset conversations, conditioned on a latent persona (misconception, knowledge level). Run teacher ↔ student-sim rollouts at scale, rank by outcome, fine-tune teacher on winning trajectories. Iterative self-play.

**Why it works here:** AlphaGo-style self-play requires an adversary; teaching has a *partner* instead. A student simulator is that partner. Lets us generate unbounded training data without collecting more human dialogues. Also the only way to scale #8's RL training — pure online RL against real students is not feasible.

**Expected lift:** Primarily unlocks #8. Standalone: +3-5 state accuracy via more diverse training trajectories.

**Compose with:** #7 (simulator personas = counterfactual generator), #8 (RL environment), #5 (PRM training data).

**Cost:** ~3 weeks. Student-sim is small (~1B params, LoRA on Qwen-1.5B). Needs persona-conditioning infrastructure.

---

### 10. Socratic Moves as Tool Calls (Architectural Rewrite)
**Core:** Reframe the 30 states as 30 *tools* the teacher can invoke: `ask_counterexample(topic)`, `check_understanding(concept)`, `provide_analogy(...)`, `summarize_so_far()`, etc. Teacher emits tool calls in structured JSON with grammar-constrained decoding. Each tool has a deterministic Chinese-language template that expands into the actual question text.

**Why it works here:** This is the most radical reframing. Turns the opaque 30-state schema into first-class structured actions. Advantages: (a) state classification becomes tool selection — a well-studied problem with grammar-constrained decoding; (b) tool outputs are interpretable and controllable; (c) we can add tools without retraining (teacher becomes extensible to new pedagogical moves); (d) BLEU improves because tool templates match ground-truth phrasing structure.

**Why it might fail:** Templates reduce expressiveness. Ground truth teachers are creative; templates are not. Mitigate with *parametrized* templates and a creativity-slot sampled from the teacher's latent space.

**Expected lift:** +10-15 BLEU-4 (templates match style), +15-25 mid-stage state accuracy (tool selection is easier than state ID), interpretability is a qualitative win.

**Compose with:** #2 (tool-call is just structured state+action emission — classifier becomes tool-router), #8 (reward rewards tool *effects* not words).

**Cost:** ~3-4 weeks. Biggest conceptual redesign. Benefits most from academic-style ablation in the paper.

---

## Recommended stack for the final product (our specific 5090 + API budget)

Given the timeline (final paper June 4) and our track record on the baseline run, the optimal stack is:

**MVP stack (5 approaches, ships in ~3 weeks):**
- **#1 RAG** — easy lift, improves BLEU/ROUGE immediately
- **#2 Hierarchical classifier (MoE variant)** — kills the mid-stage state problem *and* drops cost to zero
- **#3 Persistent student memory** — qualitative coherence + token reduction
- **#4 Fusion fine-tune** — halves latency, joint training signal
- **#6 Semantic reward fine-tuning** — directly addresses the BLEU gap

This stack attacks every measured weakness and is achievable pre-June-4.

**Stretch goals (if time permits):** #5 PRM + best-of-N → final quality gate. #7 counterfactual augmentation → underrepresented-stage boost. #10 tool-calling → paper-level novelty.

**Park for future work:** #8 RL, #9 student simulator — real research, but too expensive for a course timeline.

---

## Orthogonal lift: teacher base model swap

Max asked about "better teacher model." None of the 10 approaches requires SocratTeachLLM specifically — all work with any Chinese-capable base. Candidates to consider:

| Model | Params | Fit for 5090 | Expected baseline lift vs SocratTeachLLM |
|---|---|---|---|
| Qwen 3 32B | 32B | bf16 OOM; NF4 quant fits | +4-8 ROUGE-1 (stronger base), native Chinese |
| Gemma 3 27B | 27B | fits at Q5 | +3-6 ROUGE-1, weaker Chinese |
| GLM-4 9B Chat (non-fine-tuned) | 9B | fits easily | ~= baseline; isolates the fine-tune contribution |
| Qwen 2.5 14B Chat | 14B | fits at Q4 | +5-10 ROUGE-1; strong Chinese, faster than 32B |

**My recommendation for the primary "stronger teacher" experiment: Qwen 2.5 14B Chat.** Native Chinese, fits at Q4 alongside anything else we run, newer architecture than GLM4. A clean swap would be the lowest-effort baseline-beater.

---

## Sequencing

If I were deciding Monday morning:

1. **Week 1:** Implement #1 (RAG) + start #2 (classifier training). Also kick off `Qwen 2.5 14B` evaluation in parallel as a teacher-swap datapoint.
2. **Week 2-3:** #3 (memory) + #4 (fusion SFT on the new base). Ablation: fused vs two-agent.
3. **Week 4-5:** #6 (semantic reward training) on fused model. #7 (counterfactual augmentation) for training data.
4. **Week 6:** #5 (PRM + best-of-N) as the inference-time quality layer. Begin #10 if on track.
5. **Final weeks:** Ablations, paper writing, #10 experiments only if MVP is already exceeding baseline by wide margins.

Every improvement should be run against the same held-out test split we already have (seed=42, 10% = 681 dialogues) so the 2026-04-14 baseline is the unambiguous point of comparison.

---

## What this plan gives us

- **Five independent improvement vectors** (state classification, inference architecture, memory, training signal, data augmentation) — each defensible as a paper contribution.
- **Every approach grounded in a measured baseline weakness**, not speculation.
- **Clear compose-ability** — we can ship increments and stop when returns diminish.
- **Cost envelope:** the MVP stack fits entirely on the 5090 plus ~$200-400 of GPT-4o spend for data augmentation and evaluation reruns.
- **Ablation-friendly** — each approach can be toggled on/off for the paper's ablation table.
