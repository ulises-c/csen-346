"""Check the SFT training records against the evaluation test split.

The Gemma 4 12B fine-tune (ulises-c/SocratesLM-12B) was trained on 77,202
records from socrat-zh-sft and socrat-en-sft. That count is reproduced only by
a 90/10 cut over per-turn records, which put 3,858 turns from each language,
covering all 681 held-out test dialogues, into its training set. With the
loader fixed (split by dialogue), this script reports 77,172 records and no
test dialogue.

    uv run python scripts/check_sft_split_overlap.py
"""

from src.project import dataset as ds
from src.project.kele import load_dataset

test_ids = {d["id"] for d in load_dataset(split="test")}
train, evals = ds.load_split_pair(sources=["socrat-zh-sft", "socrat-en-sft"])
print(
    "test dialogues:", len(test_ids), "| train records:", len(train), "| eval records:", len(evals)
)
for src in ("socrat-zh-sft", "socrat-en-sft"):
    recs = [r for r in train if r["source"] == src]
    dlg = {int(str(r["id"]).rsplit("_", 1)[0]) for r in recs}
    in_test = [r for r in recs if int(str(r["id"]).rsplit("_", 1)[0]) in test_ids]
    print(
        f"{src}: {len(recs)} train turns from {len(dlg)} dialogues; "
        f"{len(in_test)} turns belong to {len(dlg & test_ids)} of the 681 test dialogues"
    )
# What the loader does when asked for the train split directly (dialogue-level split).
direct = ds.load_socrat_zh_sft(split="train")
dlg = {int(str(r["id"]).rsplit("_", 1)[0]) for r in direct}
print(
    "load_socrat_zh_sft(split='train'):",
    len(direct),
    "turns,",
    len(dlg & test_ids),
    "test dialogues touched",
)
