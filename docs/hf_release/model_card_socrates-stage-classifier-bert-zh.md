---
# Suggested repository name, for the owner to confirm: ulises-c/socrates-stage-classifier-bert-zh
# Local checkpoint this card describes: results/stage_classifier_v1/final
# TODO(owner): add a `license:` field. It is left out on purpose. The base model is listed as mit on the Hub.
language: zh
library_name: transformers
pipeline_tag: text-classification
base_model: BAAI/bge-small-zh-v1.5
datasets:
- ulises-c/SocratDataset
metrics:
- accuracy
tags:
- text-classification
- education
- socratic-teaching
- stage-classification
- bert
- kele
---

# Socratic teaching-stage classifier, bge-small-zh, 5 stages (Chinese, clean retrain)

This model reads a tutoring dialogue up to the student's latest turn and predicts which of
five teaching stages the teacher should be in next. The stages are those of the SocRule
framework of the KELE paper:

| Label | Stage |
|---|---|
| `a` | Opening: the student poses the question |
| `b` | Concept probing |
| `c` | Inductive reasoning |
| `d` | Answer derivation |
| `e` | Summary: the dialogue ends |

It predicts the stage only, not the 34 finer states. It comes from the project "Reproducing
and Extending KELE" (https://github.com/ulises-c/csen-346). The project's evaluation pipeline
needs a 34-state prediction, so the pipeline's consultants are the two 34-state classifiers
and not this model. This model backs the stage-accuracy figure in the paper's classifier
table.

## Which checkpoint this is

The original weights of this classifier were not kept. This checkpoint was retrained on
2026-11-18 on the training part of the Chinese-only split described below.

A related checkpoint published earlier,
[ulises-c/socrates-state-classifier-qwen3.5-lora](https://huggingface.co/ulises-c/socrates-state-classifier-qwen3.5-lora),
was trained on a different split that put 607 of the 681 Chinese test dialogues into its
training set. That problem does not apply to this checkpoint: it is one of the clean retrains
made after the problem was found.

## Training data and split

- Data: [ulises-c/SocratDataset](https://huggingface.co/datasets/ulises-c/SocratDataset),
  6,803 Chinese dialogues.
- Split: 90/10 by dialogue, shuffled with seed 42 (`load_dataset()` in
  `src/project/dataset.py` of the code repository). The test part is 681 dialogues with 4,304
  turns. Only the training part was used for training.
- One example per turn: the dialogue so far as lines `学生: …` and `老师: …`, ending with the
  current student turn, cut to the last 4,000 characters. The label is the first letter of
  the state recorded for that turn.
- 5% of the training-part turns were held out for checks during training.

## Training

`scripts/train_stage_classifier.py --epochs 3 --batch_size 32`: base model
`BAAI/bge-small-zh-v1.5`, all weights trained, learning rate 2e-5, at most 512 tokens, seed
42. Training took 87 minutes on an Apple-silicon laptop GPU.

## Accuracy

From `test_eval.json`, written by `scripts/eval_stage_classifier_on_test.py`, on the 4,304
turns of the test part:

| Measure | Value |
|---|---|
| Stage accuracy, all turns | 86.64% (n = 4,304) |
| Stage a | 99.85% (n = 681) |
| Stage b | 90.28% (n = 761) |
| Stage c | 80.86% (n = 1,447) |
| Stage d | 75.07% (n = 734) |
| Stage e | 94.13% (n = 681) |

Confusion matrix from the same file (rows are the reference stage, columns the prediction):

| | a | b | c | d | e |
|---|---|---|---|---|---|
| **a** | 680 | 1 | 0 | 0 | 0 |
| **b** | 0 | 687 | 74 | 0 | 0 |
| **c** | 0 | 5 | 1,170 | 259 | 13 |
| **d** | 0 | 0 | 141 | 551 | 42 |
| **e** | 0 | 0 | 1 | 39 | 641 |

Most errors are between the neighbouring stages c and d. On the 5% of training-part turns
held out during training, accuracy was 87.30% (`training_metrics.json`).

## Files

`model.safetensors` sha256:
`de40c97f9e334eac72dfd770bc18d3e5ebab1ae1fa445e8860e3a09f48d56be3` (95,834,252 bytes).

## How to load it

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

repo = "REPO_ID"  # or a local folder
tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModelForSequenceClassification.from_pretrained(repo).eval()

history = "学生: 帮我解答一下这道题：所有的植物都是长在地上的。"
inputs = tokenizer(history[-4000:], truncation=True, max_length=512, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
print(model.config.id2label[int(logits.argmax(-1))])  # a
```

This snippet was run against the local checkpoint with transformers 5.23 on a CPU.
`scripts/eval_stage_classifier_on_test.py` in the code repository loads it the same way.

For later turns, add one line per earlier turn in order (`学生: …` for the student, `老师: …`
for the teacher) and end with the current student turn.

## Limits

- The figures are for turns whose history is the reference dialogue. The model has not been
  measured inside a running dialogue, where the history holds a teacher model's replies; the
  project's 34-state classifiers lose 13 to 16 points of state accuracy in that setting.
- It has only been tested on SocratDataset.
- Inputs longer than 512 tokens are truncated by the tokenizer from the right, so in a long
  dialogue the latest turn can be cut off. Training and evaluation had the same behaviour.
- Chinese only. Elementary-school science only. The stage labels come from the state labels
  in SocratDataset, which were produced by the KELE authors' pipeline, not by human
  annotators, so accuracy here is agreement with those labels.
- One training run, one seed. No confidence interval was computed for the accuracy figures.

## Contact

https://github.com/ulises-c/csen-346/issues
