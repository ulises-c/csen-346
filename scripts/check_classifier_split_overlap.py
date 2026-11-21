"""Check whether an evaluation run's dialogues were in the state classifier's training data.

Why this exists: `load_dataset()` in src/project/kele.py shuffles whatever list it
loads with seed 42 and cuts it 90/10. From 2026-05-21 to 2026-06-09 its default
was SocratDataset-EN followed by SocratDataset (13,606 dialogues); since then it
is SocratDataset alone (6,803). The same seed on a list of a different length
gives a different split. The published classifier
(ulises-c/socrates-state-classifier-qwen3.5-lora) was trained in the first
period: its card reports 77,258 training turns and 8,526 test turns, which this
script reproduces from the concatenated split. Evaluation runs now use the
Chinese-only split, so most of their dialogues are in that classifier's
training set.

Usage:
  uv run python scripts/check_classifier_split_overlap.py results/<run>

Prints how many of the run's dialogues the classifier saw in training (in
Chinese, in English, in either), and the run's live state accuracy for seen and
unseen dialogues.
"""

import glob
import json
import random
import sys

from datasets import load_dataset as hf_load

en = [dict(r) for r in hf_load("ulises-c/SocratDataset-EN", split="train")]
zh = [dict(r) for r in hf_load("ulises-c/SocratDataset", split="train")]
print("en", len(en), "zh", len(zh), "keys", sorted(zh[0].keys()))
print(
    "ids aligned en/zh by position:", sum(a["id"] == b["id"] for a, b in zip(en, zh)), "of", len(zh)
)


def split(data, seed=42):
    rng = random.Random(seed)
    idx = list(range(len(data)))
    rng.shuffle(idx)
    cut = int(len(data) * 0.9)
    return sorted(idx[:cut]), sorted(idx[cut:])


# today's evaluation split: zh only
_, zh_test_idx = split(zh)
zh_test_ids = [zh[i]["id"] for i in zh_test_idx]
print("zh-only test dialogues:", len(zh_test_ids))

# classifier's split: EN first, then ZH (kele.py default from 2026-05-21 to 2026-06-09)
cat = en + zh
tr_idx, te_idx = split(cat)
print("concatenated: train", len(tr_idx), "test", len(te_idx))
n_en = len(en)
train_zh_ids = {cat[i]["id"] for i in tr_idx if i >= n_en}
train_en_ids = {cat[i]["id"] for i in tr_idx if i < n_en}
n_turns_train = sum(len(cat[i]["dialogue"]) for i in tr_idx)
n_turns_test = sum(len(cat[i]["dialogue"]) for i in te_idx)
print(
    "turns in concatenated train / test:",
    n_turns_train,
    n_turns_test,
    "(model card: 77,258 labeled train turns incl. 5% held out; 8,526 test)",
)

in_zh = [i for i in zh_test_ids if i in train_zh_ids]
in_en = [i for i in zh_test_ids if i in train_en_ids]
either = [i for i in zh_test_ids if i in train_zh_ids or i in train_en_ids]
clean = [i for i in zh_test_ids if i not in train_zh_ids and i not in train_en_ids]
print(
    f"of {len(zh_test_ids)} evaluation dialogues: Chinese original in classifier train {len(in_zh)}, "
    f"English translation in train {len(in_en)}, either {len(either)}, neither {len(clean)}"
)

# live state accuracy in today's run, by exposure
run = sys.argv[1]
groups = {
    "zh original in train": set(in_zh),
    "zh original NOT in train": set(zh_test_ids) - set(in_zh),
    "neither language in train": set(clean),
}
stats = {g: [0, 0, 0] for g in groups}
seen = set()
for f in glob.glob(f"{run}/dialogues/*.json"):
    d = json.load(open(f))
    seen.add(d["id"])
    ok = sum(t["state"] == t["ground_truth_state"] for t in d["dialogue"])
    for g, ids in groups.items():
        if d["id"] in ids:
            stats[g][0] += 1
            stats[g][1] += ok
            stats[g][2] += len(d["dialogue"])
print("run dialogue ids match the zh-only test split:", seen == set(zh_test_ids), len(seen))
for g, (n_dlg, ok, n) in stats.items():
    print(
        f"  {g:<28} dialogues {n_dlg:>4}  turns {n:>5}  state accuracy {100 * ok / max(n, 1):.2f}%"
    )
