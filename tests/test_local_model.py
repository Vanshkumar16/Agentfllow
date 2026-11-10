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
