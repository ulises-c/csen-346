---
# Suggested repository name, for the owner to confirm: ulises-c/socrates-state-classifier-qwen3.5-lora-zh
# Local checkpoint this card describes: results/state-clf-qwen3.5-0.8b-lora-zh/final
# TODO(owner): add a `license:` field. It is left out on purpose. The base model is listed as apache-2.0 on the Hub.
language: zh
library_name: transformers
pipeline_tag: text-classification
base_model: Qwen/Qwen3.5-0.8B-Base
datasets:
- ulises-c/SocratDataset
metrics:
- accuracy
tags:
- text-classification
- education
- socratic-teaching
- state-classification
- lora
- kele
---

# Socratic teaching-state classifier, Qwen3.5-0.8B with LoRA (Chinese, clean retrain)

This model reads a tutoring dialogue up to the student's latest turn and predicts which
teaching state the teacher should be in next. The states are the SocRule states of the KELE
paper: one opening state (`a1`), six concept-probing states (`b2` to `b7`), 22 inductive
reasoning states (`c8` to `c29`), four answer-derivation states (`d30` to `d33`) and one
closing state (`e34`). The first letter of a state is its stage.

It is the "consultant" in the project "Reproducing and Extending KELE"
(https://github.com/ulises-c/csen-346), where it replaces a language-model call: the
predicted state is handed to a teacher model that writes the reply.

## Which checkpoint this is

An earlier published checkpoint,
[ulises-c/socrates-state-classifier-qwen3.5-lora](https://huggingface.co/ulises-c/socrates-state-classifier-qwen3.5-lora),
was trained on a different split. That split was made over the Chinese and English datasets
together, and it put 607 of the 681 Chinese test dialogues into the training set. Its test
accuracy is therefore not a clean figure, and evaluation runs that used it should not be
reported.

This checkpoint is the clean retrain. It was trained on the training part of the
Chinese-only split and evaluated on the test part.

## Training data and split

- Data: [ulises-c/SocratDataset](https://huggingface.co/datasets/ulises-c/SocratDataset),
  6,803 Chinese dialogues.
- Split: 90/10 by dialogue, shuffled with seed 42 (`load_dataset()` in
  `src/project/dataset.py` of the code repository). The test part is 681 dialogues with 4,304
  turns. Only the training part was used for training.
- One example per turn: the dialogue so far as lines `学生: …` and `老师: …`, ending with the
  current student turn, cut to the last 4,000 characters. The label is the state recorded for
  that turn.
- The training part gives 38,588 turns; 5% of them (1,929) were held out for checks during
  training and 36,659 were trained on.

## Training

`scripts/train_state_classifier_34way.py --model-id Qwen/Qwen3.5-0.8B-Base --lora --lora-r 8 --lora-alpha 16 --batch_size 8 --bf16-autocast`,
other settings at their defaults: 5 epochs, learning rate 2e-5, at most 512 tokens, seed 42.
The LoRA adapters were merged into the base weights before saving, so this is a plain
sequence-classification checkpoint and needs no adapter library to load. Training took 1 hour
25 minutes on one RTX A6000.

## Accuracy

From `test_eval.json`, written by the training script, on the 4,304 turns of the test part:

| Measure | Value |
|---|---|
| State accuracy (exact state) | 68.45% (n = 4,304) |
| Stage accuracy implied by the predicted state, stage a | 100.00% (n = 681) |
| Stage b | 92.38% (n = 761) |
| Stage c | 85.00% (n = 1,447) |
| Stage d | 75.75% (n = 734) |
| Stage e | 96.92% (n = 681) |

The same file also holds `test_acc` = 68.38% from a second evaluation pass in the same
script. The two differ by three turns; the reason has not been looked into.

Accuracy differs a great deal between states. Fourteen of the 34 states were never predicted
correctly on the test part; each of them has at most 32 test turns. The frequent states carry
the overall figure: `a1` 100% (n = 681), `e34` 96.9% (n = 681), `d33` 74.5% (n = 644),
`c16` 65.1% (n = 341), `c12` 59.2% (n = 299).

## Files

`model.safetensors` sha256:
`696e6e044c5222e2a3e2f9746c76c57c308764cce941fff66c453014235c2506` (3,412,145,656 bytes,
32-bit floats). The evaluation runs that used this checkpoint record the same hash in their
`run_config.json` (`bert_consultant_sha256`).

## How to load it

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

repo = "REPO_ID"  # or a local folder
tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModelForSequenceClassification.from_pretrained(repo, attn_implementation="eager").eval()

history = "学生: 帮我解答一下这道题：所有的植物都是长在地上的。"
inputs = tokenizer(history[-4000:], truncation=True, max_length=512, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
print(model.config.id2label[int(logits.argmax(-1))])  # a1
```

This snippet was run against the local checkpoint with transformers 5.23 on a CPU. The
project's pipeline loads it the same way (`src/project/socratic_teaching_bert_consultant.py`,
option `--bert-consultant <folder>`), then casts the model to bfloat16.
`attn_implementation="eager"` is passed because the default attention path failed for this
architecture on an earlier transformers release.

For later turns, add one line per earlier turn in order (`学生: …` for the student, `老师: …`
for the teacher) and end with the current student turn.

## Limits

- The classifier is much less accurate inside a running dialogue than on the test turns. On
  the test turns it sees the reference teacher replies as history. In the pipeline it sees
  the replies of the teacher model, and a wrong state changes what follows. With Gemma 4 31B
  as the teacher, state accuracy in the loop was 52.14% over the turns that were generated
  (4,120 turns, 681 dialogues), against 68.45% here.
- It has only been tested on SocratDataset. On a 75-dialogue synthetic set whose sub-state
  labels follow a different convention, exact-state accuracy was about 30%; that figure
  reflects the labelling convention and should not be read as the model's accuracy.
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
