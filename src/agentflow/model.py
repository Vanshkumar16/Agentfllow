"""Model interfaces and the GitHub Copilot SDK adapter."""

from __future__ import annotations

from typing import Protocol


class TextModel(Protocol):
    async def complete(self, prompt: str) -> str:
        """Return a model response for a prompt."""


class CopilotModel:
    """Small adapter around the GitHub Copilot SDK."""

    def __init__(self, model: str) -> None:
        self.model = model
        self._client = None

    async def start(self) -> None:
        try:
            from copilot import CopilotClient
        except ImportError as exc:
            raise RuntimeError(
                "The GitHub Copilot SDK is required. Install requirements.txt."
            ) from exc

        self._client = CopilotClient()
        await self._client.start()

    async def complete(self, prompt: str) -> str:
        if self._client is None:
            raise RuntimeError("CopilotModel.start() must be called before complete().")

        session = await self._client.create_session({"model": self.model})
        response = await session.send_and_wait({"prompt": prompt})
        content = getattr(getattr(response, "data", None), "content", None)
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Copilot returned an empty response.")
        return content

    async def close(self) -> None:
        if self._client is not None:
            await self._client.stop()
            self._client = None
