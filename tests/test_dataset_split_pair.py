"""load_split_pair must take each loader's own train/test split."""

from src.project import dataset


def test_split_pair_uses_each_loaders_own_split(monkeypatch):
    asked = []

    def fake_loader(split="train", seed=42):
        asked.append(split)
        # Per-turn records of 10 dialogues; dialogues 8 and 9 are held out.
        held_out = {8, 9}
        dialogues = [d for d in range(10) if (d in held_out) == (split == "test")]
        return [{"id": f"{d}_{t}", "messages": []} for d in dialogues for t in range(5)]

    monkeypatch.setitem(dataset._SOURCE_LOADERS, "fake-sft", fake_loader)

    train, test = dataset.load_split_pair(sources=["fake-sft"])

    assert sorted(asked) == ["test", "train"]  # never "all": that would be cut by turn
    train_dialogues = {r["id"].split("_")[0] for r in train}
    test_dialogues = {r["id"].split("_")[0] for r in test}
    assert train_dialogues.isdisjoint(test_dialogues)
    assert (len(train), len(test)) == (40, 10)
