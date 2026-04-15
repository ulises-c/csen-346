import json
import sys
import types

sys.modules.setdefault("openai", types.SimpleNamespace(Client=object))
from src.project import kele


class FakeSystem:
    def __init__(self, states, responses):
        self._states = list(states)
        self._responses = list(responses)
        self.current_state = "a1"
        self.reset_calls = 0

    def reset_session(self):
        self.reset_calls += 1
        self.current_state = "a1"

    def process_student_input(self, student_input):
        idx = min(self.reset_calls_input_index, len(self._responses) - 1)
        response = self._responses[self.reset_calls_input_index]
        self.current_state = self._states[self.reset_calls_input_index]
        self.reset_calls_input_index += 1
        return response

    @property
    def reset_calls_input_index(self):
        return getattr(self, "_index", 0)

    @reset_calls_input_index.setter
    def reset_calls_input_index(self, value):
        self._index = value


def make_dataset(path, ids):
    data = [
        {
            "id": item_id,
            "question": f"Q{item_id}",
            "answer": f"A{item_id}",
            "dialogue": [{"student": "s", "teacher": "t", "state": "a1"}],
        }
        for item_id in ids
    ]
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_load_dataset_train_test_split_is_deterministic(tmp_path):
    dataset_path = tmp_path / "dataset.json"
    make_dataset(dataset_path, range(1, 11))

    first = kele.load_dataset(dataset_path, split="test", seed=42)
    second = kele.load_dataset(dataset_path, split="test", seed=42)

    assert [item["id"] for item in first] == [item["id"] for item in second]


def test_load_dataset_train_and_test_do_not_overlap(tmp_path):
    dataset_path = tmp_path / "dataset.json"
    make_dataset(dataset_path, range(1, 21))

    train = kele.load_dataset(dataset_path, split="train", seed=7)
    test = kele.load_dataset(dataset_path, split="test", seed=7)

    train_ids = {item["id"] for item in train}
    test_ids = {item["id"] for item in test}

    assert len(train) == 18
    assert len(test) == 2
    assert train_ids.isdisjoint(test_ids)


def test_load_dataset_all_returns_full_dataset(tmp_path):
    dataset_path = tmp_path / "dataset.json"
    make_dataset(dataset_path, range(1, 6))

    data = kele.load_dataset(dataset_path, split="all")

    assert len(data) == 5


def test_run_single_dialogue_records_generated_and_ground_truth_fields():
    system = FakeSystem(states=["a1", "e34"], responses=["引导一", "总结"])
    item = {
        "id": 1,
        "question": "Q1",
        "answer": "A1",
        "dialogue": [
            {"student": "学生1", "teacher": "老师1", "state": "a1"},
            {"student": "学生2", "teacher": "老师2", "state": "e34"},
            {"student": "学生3", "teacher": "老师3", "state": "e34"},
        ],
    }

    result = kele.run_single_dialogue(system, item)

    assert result["id"] == 1
    assert result["num_turns_ground_truth"] == 3
    assert result["num_turns_generated"] == 2
    assert result["dialogue"][0]["teacher_response"] == "引导一"
    assert result["dialogue"][0]["ground_truth_teacher"] == "老师1"
    assert result["dialogue"][1]["state"] == "e34"


def test_run_single_dialogue_resets_system_before_replay():
    system = FakeSystem(states=["a1"], responses=["引导一"])
    item = {
        "id": 1,
        "question": "Q1",
        "answer": "A1",
        "dialogue": [{"student": "学生1", "teacher": "老师1", "state": "a1"}],
    }

    kele.run_single_dialogue(system, item)

    assert system.reset_calls == 1
