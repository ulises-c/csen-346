---
# Suggested repository name, for the owner to confirm: ulises-c/socrates-state-classifier-bge-small-zh
# Local checkpoint this card describes: results/state-clf-bge-small-zh/final
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
- state-classification
- bert
- kele
---

# Socratic teaching-state classifier, bge-small-zh (Chinese, clean retrain)

This model reads a tutoring dialogue up to the student's latest turn and predicts which
teaching state the teacher should be in next. The states are the SocRule states of the KELE
paper: one opening state (`a1`), six concept-probing states (`b2` to `b7`), 22 inductive
reasoning states (`c8` to `c29`), four answer-derivation states (`d30` to `d33`) and one
closing state (`e34`). The first letter of a state is its stage.

It is a small (about 24 million parameters) alternative "consultant" in the project
"Reproducing and Extending KELE" (https://github.com/ulises-c/csen-346). The project's logs
call it the "BERT consultant". It is the weaker of the project's two state classifiers; the
stronger one is the Qwen3.5-0.8B LoRA classifier.

## Which checkpoint this is

Earlier checkpoints of this classifier were not kept. This one was retrained on 2026-11-18 on
the training part of the Chinese-only split described below.

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
  current student turn, cut to the last 4,000 characters. The label is the state recorded for
  that turn.
- 5% of the training-part turns were held out for checks during training.

## Training

`scripts/train_state_classifier_34way.py --device cpu --out-dir results/state-clf-bge-small-zh`,
other settings at their defaults: base model `BAAI/bge-small-zh-v1.5`, all weights trained,
5 epochs, batch size 32, learning rate 2e-5, at most 512 tokens, seed 42. Trained on a CPU.

## Accuracy

From `test_eval.json`, written by the training script, on the 4,304 turns of the test part:

| Measure | Value |
|---|---|
| State accuracy (exact state) | 61.59% (n = 4,304) |
| Stage accuracy implied by the predicted state, stage a | 100.00% (n = 681) |
| Stage b | 90.80% (n = 761) |
| Stage c | 74.84% (n = 1,447) |
| Stage d | 79.56% (n = 734) |
| Stage e | 95.59% (n = 681) |

Accuracy differs a great deal between states. Twenty-one of the 34 states were never
predicted correctly on the test part; each of them has at most 44 test turns. The frequent states carry the overall figure: `a1` 100%
(n = 681), `e34` 95.6% (n = 681), `d33` 78.3% (n = 644), `c12` 55.5% (n = 299), `c16` 54.8%
(n = 341).

## Files

`model.safetensors` sha256:
`07d79532334d51782fb7d088962a11bbb7c3befb18357a3f8a7b0b6b046d982f` (95,895,812 bytes). The
evaluation run that used this checkpoint records the same hash in its `run_config.json`
(`bert_consultant_sha256`).

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
print(model.config.id2label[int(logits.argmax(-1))])  # a1
```

This snippet was run against the local checkpoint with transformers 5.23 on a CPU. The
project's pipeline loads it the same way (`src/project/socratic_teaching_bert_consultant.py`,
option `--bert-consultant <folder>`).

For later turns, add one line per earlier turn in order (`学生: …` for the student, `老师: …`
for the teacher) and end with the current student turn.

## Limits

- The classifier is less accurate inside a running dialogue than on the test turns. On the
  test turns it sees the reference teacher replies as history. In the pipeline it sees the
  replies of the teacher model, and a wrong state changes what follows. With Gemma 4 31B as
  the teacher, state accuracy in the loop was 48.36% over the turns that were generated
  (3,805 turns, 681 dialogues), against 61.59% here. In that run 47.0% of the dialogues ended
  early because the classifier predicted the closing state before the reference dialogue
  ended.
- It has only been tested on SocratDataset.
- The config has 35 labels. Label 0 (`a0`) does not occur in the data and is never the
  correct answer.
- Inputs longer than 512 tokens are truncated by the tokenizer from the right, so in a long
  dialogue the latest turn can be cut off. Training and evaluation had the same behaviour.
- Chinese only. Elementary-school science only. The state labels in SocratDataset were
  produced by the KELE authors' pipeline, not by human annotators, so accuracy here is
  agreement with those labels.
- One training run, one seed. No confidence interval was computed for the accuracy figures.

## Contact

https://github.com/ulises-c/csen-346/issues
