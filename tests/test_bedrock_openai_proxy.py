"""bedrock_openai_proxy: request and response translation, and the HTTP round trip."""

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from scripts.bedrock_openai_proxy import from_converse, make_handler, to_converse


def test_system_messages_and_token_cap_are_translated():
    body = {
        "model": "us.some-model",
        "messages": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hello"},
        ],
        "max_tokens": 64,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }

    kwargs = to_converse(body, pass_temperature=False)

    assert kwargs == {
        "modelId": "us.some-model",
        "messages": [{"role": "user", "content": [{"text": "hello"}]}],
        "system": [{"text": "be brief"}],
        "inferenceConfig": {"maxTokens": 64},
    }
    assert to_converse(body, pass_temperature=True)["inferenceConfig"]["temperature"] == 0.2


def test_consecutive_same_role_messages_are_merged():
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}],
    }

    assert to_converse(body, False)["messages"] == [
        {"role": "user", "content": [{"text": "a\n\nb"}]}
    ]


def test_response_is_translated_with_usage():
    resp = {
        "output": {"message": {"content": [{"text": '{"state": "b4"}'}]}},
        "stopReason": "max_tokens",
        "usage": {"inputTokens": 900, "outputTokens": 30, "totalTokens": 930},
    }

    out = from_converse(resp, "m")

    assert out["choices"][0]["message"]["content"] == '{"state": "b4"}'
    assert out["choices"][0]["finish_reason"] == "length"
    assert out["usage"] == {"prompt_tokens": 900, "completion_tokens": 30, "total_tokens": 930}


class _FakeBedrock:
    def __init__(self, fail_with=None):
        self.calls = []
        self.fail_with = fail_with

    def converse(self, **kw):
        self.calls.append(kw)
        if self.fail_with:
            raise self.fail_with
        return {
            "output": {"message": {"content": [{"text": "ok"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 5, "outputTokens": 1, "totalTokens": 6},
        }


def _serve(client, log_path=None):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(client, False, log_path))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/v1/chat/completions"


def _post(url, body):
    req = urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def test_http_round_trip_and_call_log(tmp_path):
    client = _FakeBedrock()
    log = tmp_path / "calls.jsonl"
    server, url = _serve(client, log)
    try:
        out = _post(
            url, {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}
        )
    finally:
        server.shutdown()

    assert out["choices"][0]["message"]["content"] == "ok"
    assert client.calls[0]["modelId"] == "m"
    logged = json.loads(log.read_text().splitlines()[0])
    assert (logged["model"], logged["prompt_tokens"], logged["completion_tokens"]) == ("m", 5, 1)


def test_throttling_is_reported_as_429_so_the_pipeline_retries():
    class ThrottlingException(Exception):
        response = {"Error": {"Code": "ThrottlingException"}}

    server, url = _serve(_FakeBedrock(fail_with=ThrottlingException("slow down")))
    try:
        try:
            _post(url, {"model": "m", "messages": [{"role": "user", "content": "hi"}]})
            status = 200
        except urllib.error.HTTPError as e:
            status = e.code
    finally:
        server.shutdown()

    assert status == 429
