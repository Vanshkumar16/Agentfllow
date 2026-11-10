"""Model-assisted completion assessment."""

from __future__ import annotations

import json

from .model import TextModel
from .planner import parse_json_object
from .schemas import Critique, StepResult, TestReport


class Critic:
    def __init__(self, model: TextModel) -> None:
        self._model = model

    async def review(
        self, goal: str, results: list[StepResult], tests: TestReport
    ) -> Critique:
        prompt = (
            "Assess whether the task is actually complete based only on the "
            "provided tool results and checks. Return JSON only: "
            '{"complete":true,"feedback":"brief reason"}. If evidence is '
            "insufficient or a tool failed, complete must be false.\n\n"
            f"Task: {goal}\n"
            f"Tool results: {json.dumps([result.__dict__ for result in results], default=str)}\n"
            f"Checks: {json.dumps(tests.checks)}"
        )
        data = parse_json_object(await self._model.complete(prompt))
        complete = data.get("complete")
        feedback = data.get("feedback")
        if not isinstance(complete, bool) or not isinstance(feedback, str):
            raise ValueError("Critic response requires boolean 'complete' and string 'feedback'.")
        return Critique(complete, feedback)
