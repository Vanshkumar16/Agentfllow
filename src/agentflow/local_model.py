"""Ollama-compatible local model client."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import math
import re
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import (
    HTTPRedirectHandler,
    ProxyHandler,
    Request,
    build_opener,
)


class LocalModelError(RuntimeError):
    """Raised when the configured local model cannot serve a request."""


GENERATION_TEMPERATURE = 0.1


@dataclass(frozen=True)
class ModelResponse:
    content: str
    input_tokens: int | None
    output_tokens: int | None


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        return None


class OllamaModel:
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        device: str = "auto",
    ) -> None:
        if not model.strip():
            raise ValueError("Model name must not be empty.")
        if device not in {"auto", "cpu"}:
            raise ValueError("device must be 'auto' or 'cpu'.")
        self.model = model.strip()
        self.base_url = base_url.rstrip("/")
        self._validate_local_url(self.base_url)
        self.device = device

    @staticmethod
    def _validate_local_url(base_url: str) -> None:
        try:
            parsed = urlsplit(base_url)
            _ = parsed.port
        except ValueError as exc:
            raise ValueError("The model endpoint must be a valid loopback URL.") from exc
        host = parsed.hostname
        is_loopback = host == "localhost"
        if host is not None and not is_loopback:
            try:
                is_loopback = ipaddress.ip_address(host).is_loopback
            except ValueError:
                is_loopback = False
        if (
            parsed.scheme not in {"http", "https"}
            or not is_loopback
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "The local model endpoint must use an HTTP(S) loopback URL "
                "such as http://localhost:11434; remote endpoints are not allowed."
            )

    @staticmethod
    def _open(request: Request, timeout: float):
        opener = build_opener(ProxyHandler({}), _NoRedirectHandler())
        return opener.open(request, timeout=timeout)

    async def list_models(self, timeout: float = 5.0) -> list[str]:
        return list((await self.list_model_details(timeout)).keys())

    async def list_model_details(
        self, timeout: float = 5.0
    ) -> dict[str, dict[str, object]]:
        return await asyncio.to_thread(self._list_model_details, timeout)

    def _list_model_details(self, timeout: float) -> dict[str, dict[str, object]]:
        request = Request(f"{self.base_url}/api/tags", method="GET")
        try:
            with self._open(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise LocalModelError(
                f"Could not query local model server at {self.base_url}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise LocalModelError("Local model server returned an invalid response.")
        models = payload.get("models")
        if not isinstance(models, list):
            raise LocalModelError("Local model server returned an invalid model list.")
        result: dict[str, dict[str, object]] = {}
        for item in models:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                continue
            details = item.get("details")
            result[item["name"]] = details if isinstance(details, dict) else {}
        return result

    async def check_available(self, timeout: float = 5.0) -> None:
        models = await self.list_models(timeout)
        if self.model not in models:
            raise LocalModelError(
                f"Model '{self.model}' is not installed on the local server. "
                f"Available models: {', '.join(models) or '(none)'}. "
                "Install the requested model with your local runtime, then retry."
            )

    async def complete(
        self, prompt: str, max_output_tokens: int, timeout: float
    ) -> ModelResponse:
        return await asyncio.to_thread(
            self._complete, prompt, max_output_tokens, timeout
        )

    def _complete(
        self, prompt: str, max_output_tokens: int, timeout: float
    ) -> ModelResponse:
        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be at least 1.")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be a positive finite number.")
        options: dict[str, int | float] = {
            "num_predict": max_output_tokens,
            "temperature": GENERATION_TEMPERATURE,
        }
        if self.device == "cpu":
            options["num_gpu"] = 0
        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": options,
            }
        ).encode("utf-8")
        request = Request(
            f"{self.base_url}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._open(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise LocalModelError(f"Local model request failed: {exc}") from exc

        if not isinstance(payload, dict):
            raise LocalModelError("Local model server returned an invalid response.")
        message = payload.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise LocalModelError("Local model returned an empty or malformed response.")
        prompt_tokens = payload.get("prompt_eval_count")
        output_tokens = payload.get("eval_count")
        return ModelResponse(
            content=content,
            input_tokens=prompt_tokens if type(prompt_tokens) is int else None,
            output_tokens=output_tokens if type(output_tokens) is int else None,
        )
