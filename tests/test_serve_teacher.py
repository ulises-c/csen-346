import contextlib
import sys

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from src.project import serve_teacher


class FakeTensor:
    def __init__(self, values):
        self.values = list(values)
        self.shape = (1, len(self.values))

    def to(self, device):
        return self

    def cpu(self):
        return self

    def __len__(self):
        return len(self.values)

    def __getitem__(self, item):
        if isinstance(item, int):
            return FakeTensor(self.values)
        if isinstance(item, slice):
            return FakeTensor(self.values[item])
        return FakeTensor(self.values)


class FakeTokenizer:
    eos_token_id = 99

    def __init__(self, fail_template=False):
        self.fail_template = fail_template
        self.last_prompt = None

    def apply_chat_template(self, messages, add_generation_prompt, return_tensors):
        if self.fail_template:
            raise RuntimeError("template failure")
        return FakeTensor([10, 11, 12])

    def __call__(self, prompt, return_tensors):
        self.last_prompt = prompt
        return type("TokenResult", (), {"input_ids": FakeTensor([21, 22])})()

    def decode(self, tokens, skip_special_tokens=True):
        return "老师回复"


class FakeModel:
    device = "cpu"

    def generate(self, input_ids, **kwargs):
        return FakeTensor([*input_ids.values, 31, 32])


class _FakeCuda:
    @staticmethod
    def empty_cache():
        pass


class FakeTorch:
    cuda = _FakeCuda

    @staticmethod
    @contextlib.contextmanager
    def inference_mode():
        yield

    @staticmethod
    def ones_like(t):
        return t


def make_client(tokenizer):
    app = serve_teacher.create_app()
    app.state.runtime = (tokenizer, FakeModel())
    return TestClient(app)


def test_list_models_returns_expected_shape():
    client = make_client(FakeTokenizer())

    response = client.get("/v1/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert payload["data"][0]["id"] == "SocratTeachLLM"


def test_healthz_reports_runtime_state():
    client = make_client(FakeTokenizer())

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_chat_completions_rejects_streaming(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    client = make_client(FakeTokenizer())

    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "你好"}], "stream": True},
    )

    assert response.status_code == 501
    assert response.json()["detail"] == "Streaming not supported"


def test_chat_completions_returns_openai_style_response(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    client = make_client(FakeTokenizer())

    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "你好"}], "max_tokens": 8},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"]["content"] == "老师回复"
    assert payload["usage"]["prompt_tokens"] == 3
    assert payload["usage"]["completion_tokens"] == 2


def test_chat_completions_uses_manual_prompt_fallback_when_template_fails(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    tokenizer = FakeTokenizer(fail_template=True)
    client = make_client(tokenizer)

    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "解释蒸发"}]},
    )

    assert response.status_code == 200
    assert "User: 解释蒸发" in tokenizer.last_prompt
    assert tokenizer.last_prompt.endswith("\nAssistant:")


def test_list_models_requires_api_key_when_configured(monkeypatch):
    monkeypatch.setenv("TEACHER_SERVER_API_KEY", "secret-token")
    app = serve_teacher.create_app()
    client = TestClient(app)

    unauthorized = client.get("/v1/models")
    authorized = client.get(
        "/v1/models",
        headers={"Authorization": "Bearer secret-token"},
    )

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200


def test_chat_completions_accepts_x_api_key_header(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setenv("TEACHER_SERVER_API_KEY", "secret-token")
    app = serve_teacher.create_app()
    app.state.runtime = (FakeTokenizer(), FakeModel())
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "你好"}]},
        headers={"x-api-key": "secret-token"},
    )

    assert response.status_code == 200
