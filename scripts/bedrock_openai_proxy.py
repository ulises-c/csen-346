#!/usr/bin/env python3
"""Serve Amazon Bedrock models behind an OpenAI-compatible chat endpoint.

The pipeline talks to its consultant and teacher through the OpenAI chat-completions API. This
small local server lets a Bedrock model stand in as either, using the AWS credentials of the
machine it runs on, so no AWS credential has to be copied anywhere else:

    uv run --with boto3 python scripts/bedrock_openai_proxy.py --port 8090
    CONSULTANT_BASE_URL=http://localhost:8090/v1 CONSULTANT_MODEL_NAME=us.openai.gpt-5.4 ...

The request's ``model`` is passed to Bedrock's Converse API as the model or inference-profile
id. System messages become the Converse system prompt. ``max_tokens`` is forwarded;
``temperature`` is forwarded only with --pass-temperature, because several current models
reject it; ``response_format`` and other extras are ignored (the pipeline's prompts already ask
for JSON and its parser accepts loose JSON). Streaming is not supported.

Every call is appended to --log as one JSON line with the model, token counts and latency, so
the cost of a run can be added up afterwards. Calls are billed to the AWS account.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def to_converse(body: dict, pass_temperature: bool) -> dict:
    """Translate an OpenAI chat-completions request body into Converse keyword arguments."""
    system = []
    messages: list[dict] = []
    for m in body.get("messages", []):
        content = m.get("content") or ""
        if isinstance(content, list):  # OpenAI content parts: keep the text ones
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        if m.get("role") == "system":
            system.append({"text": content})
            continue
        role = "assistant" if m.get("role") == "assistant" else "user"
        if messages and messages[-1]["role"] == role:  # Converse needs alternating roles
            messages[-1]["content"][0]["text"] += "\n\n" + content
        else:
            messages.append({"role": role, "content": [{"text": content}]})
    config: dict = {}
    max_tokens = body.get("max_tokens") or body.get("max_completion_tokens")
    if max_tokens:
        config["maxTokens"] = int(max_tokens)
    if pass_temperature and body.get("temperature") is not None:
        config["temperature"] = float(body["temperature"])
    kwargs: dict = {"modelId": body["model"], "messages": messages}
    if system:
        kwargs["system"] = system
    if config:
        kwargs["inferenceConfig"] = config
    return kwargs


def from_converse(resp: dict, model: str) -> dict:
    """Translate a Converse response into an OpenAI chat-completions response body."""
    text = "".join(b.get("text", "") for b in resp["output"]["message"]["content"])
    usage = resp.get("usage", {})
    stop = {"end_turn": "stop", "max_tokens": "length", "stop_sequence": "stop"}.get(
        resp.get("stopReason", ""), "stop"
    )
    return {
        "id": f"chatcmpl-bedrock-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": stop}
        ],
        "usage": {
            "prompt_tokens": usage.get("inputTokens", 0),
            "completion_tokens": usage.get("outputTokens", 0),
            "total_tokens": usage.get("totalTokens", 0),
        },
    }


def make_handler(client, pass_temperature: bool, log_path: Path | None):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: dict, headers: dict | None = None) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format, *args):  # keep the console quiet; calls go to --log
            return

        def do_GET(self):
            if self.path.rstrip("/").endswith("/models") or self.path in ("/health", "/v1/health"):
                self._send(200, {"object": "list", "data": [], "status": "ok"})
            else:
                self._send(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if not self.path.rstrip("/").endswith("/chat/completions"):
                self._send(404, {"error": {"message": "only /v1/chat/completions is served"}})
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                if body.get("stream"):
                    self._send(400, {"error": {"message": "streaming is not supported"}})
                    return
                kwargs = to_converse(body, pass_temperature)
            except (ValueError, KeyError) as e:
                self._send(400, {"error": {"message": f"bad request: {e}"}})
                return
            started = time.time()
            try:
                resp = client.converse(**kwargs)
            except Exception as e:
                name = type(e).__name__
                code = getattr(e, "response", {}).get("Error", {}).get("Code", name)
                throttled = "Throttl" in code or "TooManyRequests" in code
                # 429 makes the OpenAI client raise RateLimitError, which the pipeline retries.
                self._send(
                    429 if throttled else 502,
                    {"error": {"message": f"{code}: {str(e)[:300]}", "type": code}},
                    {"Retry-After": "5"} if throttled else None,
                )
                return
            out = from_converse(resp, body["model"])
            if log_path is not None:
                line = {
                    "t": round(started, 3),
                    "model": body["model"],
                    "seconds": round(time.time() - started, 3),
                    **out["usage"],
                }
                with lock, log_path.open("a") as f:
                    f.write(json.dumps(line) + "\n")
            self._send(200, out)

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument(
        "--host", default="127.0.0.1", help="Address to listen on (default: this machine only)"
    )
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--region", default=None, help="AWS region; default: the profile's")
    ap.add_argument("--pass-temperature", action="store_true")
    ap.add_argument("--log", type=Path, default=None, help="Append one JSON line per call")
    args = ap.parse_args()

    import boto3  # type: ignore[import-not-found]  # optional: only this script needs it
    from botocore.config import Config  # type: ignore[import-not-found]

    client = boto3.client(
        "bedrock-runtime",
        region_name=args.region,
        config=Config(read_timeout=300, retries={"max_attempts": 4, "mode": "adaptive"}),
    )
    server = ThreadingHTTPServer(
        (args.host, args.port), make_handler(client, args.pass_temperature, args.log)
    )
    print(f"Bedrock proxy on http://{args.host}:{args.port}/v1", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
