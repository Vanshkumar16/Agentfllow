import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agentflow.local_model import OllamaModel


def test_local_model_rejects_non_loopback_endpoints():
    with pytest.raises(ValueError, match="remote endpoints are not allowed"):
        OllamaModel("gemma4:e2b", "https://example.com")


@pytest.mark.asyncio
async def test_ollama_adapter_uses_local_tags_and_reports_token_usage():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            payload = {"models": [{"name": "gemma4:e2b"}]}
            self._respond(payload)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            self._respond(
                {
                    "message": {"content": '{"action":{"tool":"finish","arguments":{}}}'},
                    "prompt_eval_count": 18,
                    "eval_count": 9,
                }
            )

        def _respond(self, payload):
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        model = OllamaModel(
            "gemma4:e2b", f"http://127.0.0.1:{server.server_port}", device="cpu"
        )
        await model.check_available()
        response = await model.complete("test prompt", max_output_tokens=32, timeout=2)
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()

    assert response.input_tokens == 18
    assert response.output_tokens == 9
    assert requests[0]["options"]["num_predict"] == 32
    assert requests[0]["options"]["num_gpu"] == 0


@pytest.mark.asyncio
async def test_ollama_uses_schema_format_when_server_version_supports_it():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            payload = {"version": "0.5.0"}
            self._respond(payload)

        def do_POST(self):
            requests.append(
                json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            )
            self._respond(
                {
                    "message": {"content": '{"action":{"tool":"finish","arguments":{}}}'},
                    "prompt_eval_count": 4,
                    "eval_count": 3,
                    "done": True,
                    "done_reason": "stop",
                    "total_duration": 1234,
                }
            )

        def _respond(self, payload):
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    schema = {
        "type": "object",
        "properties": {"action": {"type": "object"}},
        "required": ["action"],
    }
    try:
        model = OllamaModel(
            "gemma4:e2b", f"http://127.0.0.1:{server.server_port}"
        )
        response = await model.complete(
            "return JSON", 16, 2, response_schema=schema
        )
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()

    assert requests[0]["format"] == schema
    assert response.structured_output is True
    assert response.done is True
    assert response.done_reason == "stop"
    assert response.provider_metadata["total_duration"] == 1234


@pytest.mark.asyncio
async def test_ollama_preserves_safe_metadata_for_empty_or_non_string_content():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self._respond(
                {
                    "message": {"content": None},
                    "done": False,
                    "done_reason": "length",
                    "prompt_eval_count": 2,
                }
            )

        def _respond(self, payload):
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        model = OllamaModel(
            "gemma4:e2b", f"http://127.0.0.1:{server.server_port}"
        )
        response = await model.complete("test", 16, 2)
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()

    assert response.content == ""
    assert response.input_tokens == 2
    assert response.output_tokens is None
    assert response.done is False
    assert response.done_reason == "length"
    assert response.provider_metadata["content_type"] == "NoneType"
