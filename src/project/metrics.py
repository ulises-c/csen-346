"""
Metrics computation for KELE evaluation.

Reproduces the metrics from Table 1 of the paper:
- Text overlap: ROUGE-1, ROUGE-2, ROUGE-L, BLEU-4
- Single-turn (LLM-as-judge): PRR, NDAR, SPR, IAR
- Multi-turn (LLM-as-judge): Guidance, Logicality, Flexibility, Repetitiveness, Clarity
- State accuracy: how often our consultant picks the same state as the ground truth
"""

import json
from pathlib import Path

from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU


def compute_rouge(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L F1 scores."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        for key in scores:
            scores[key].append(result[key].fmeasure)

    return {key: sum(vals) / len(vals) * 100 for key, vals in scores.items()}


def compute_bleu(predictions: list[str], references: list[str]) -> float:
    """Compute BLEU-4 score."""
    bleu = BLEU(effective_order=True)
    # sacrebleu expects references as list of lists
    result = bleu.corpus_score(predictions, [references])
    return result.score


def compute_state_accuracy(dialogues_dir: Path) -> dict[str, float]:
    """Compute how often our consultant picks the correct state vs ground truth.

    Returns overall accuracy and per-stage accuracy (a, b, c, d, e).
    """
    correct = 0
    total = 0
    stage_correct: dict[str, int] = {}
    stage_total: dict[str, int] = {}

    for f in sorted(dialogues_dir.glob("*.json")):
        data = json.loads(f.read_text())
        if "error" in data:
            continue
        for turn in data.get("dialogue", []):
            gt_state = turn.get("ground_truth_state", "")
            pred_state = turn.get("state", "")
            if not gt_state:
                continue

            stage = gt_state[0]
            stage_total[stage] = stage_total.get(stage, 0) + 1
            total += 1

            if pred_state == gt_state:
                correct += 1
                stage_correct[stage] = stage_correct.get(stage, 0) + 1

    overall = (correct / total * 100) if total > 0 else 0.0
    per_stage = {}
    for stage in sorted(stage_total.keys()):
        per_stage[stage] = (
            stage_correct.get(stage, 0) / stage_total[stage] * 100
            if stage_total[stage] > 0
            else 0.0
        )

    return {"overall": overall, "per_stage": per_stage, "total_turns": total}


def extract_predictions_and_references(dialogues_dir: Path) -> tuple[list[str], list[str]]:
    """Extract teacher response predictions and ground truth references from dialogue outputs."""
    predictions = []
    references = []

    for f in sorted(dialogues_dir.glob("*.json")):
        data = json.loads(f.read_text())
        if "error" in data:
            continue
        for turn in data.get("dialogue", []):
            pred = turn.get("teacher_response", "")
            ref = turn.get("ground_truth_teacher", "")
            if pred and ref:
                predictions.append(pred)
                references.append(ref)

    return predictions, references


def compute_all_metrics(dialogues_dir: Path) -> dict:
    """Compute all automated metrics for a completed evaluation run."""
    predictions, references = extract_predictions_and_references(dialogues_dir)

    if not predictions:
        return {"error": "No dialogue data found"}

    rouge = compute_rouge(predictions, references)
    bleu = compute_bleu(predictions, references)
    state_acc = compute_state_accuracy(dialogues_dir)

    return {
        "n_turns": len(predictions),
        "rouge1": round(rouge["rouge1"], 2),
        "rouge2": round(rouge["rouge2"], 2),
        "rougeL": round(rouge["rougeL"], 2),
        "bleu4": round(bleu, 2),
        "state_accuracy": {
            "overall": round(state_acc["overall"], 2),
            "per_stage": {k: round(v, 2) for k, v in state_acc["per_stage"].items()},
            "total_turns": state_acc["total_turns"],
        },
    }


def format_metrics_table(metrics: dict) -> str:
    """Format metrics as a readable table for console output."""
    lines = [
        "=" * 50,
        "EVALUATION METRICS",
        "=" * 50,
        f"Total turns evaluated: {metrics.get('n_turns', 'N/A')}",
        "",
        "Text Overlap (vs ground-truth teacher responses):",
        f"  ROUGE-1:  {metrics.get('rouge1', 'N/A')}",
        f"  ROUGE-2:  {metrics.get('rouge2', 'N/A')}",
        f"  ROUGE-L:  {metrics.get('rougeL', 'N/A')}",
        f"  BLEU-4:   {metrics.get('bleu4', 'N/A')}",
        "",
        "State Classification Accuracy:",
        f"  Overall:  {metrics.get('state_accuracy', {}).get('overall', 'N/A')}%",
    ]

    per_stage = metrics.get("state_accuracy", {}).get("per_stage", {})
    for stage, acc in sorted(per_stage.items()):
        lines.append(f"  Stage {stage}: {acc}%")

    lines.append("=" * 50)
    return "\n".join(lines)
