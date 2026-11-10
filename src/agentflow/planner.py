"""Plan generation and response parsing."""

from __future__ import annotations

import json
from typing import Any

from .model import TextModel
from .schemas import PlanStep, TaskPlan
from .tools import ToolRegistry


def parse_json_object(response: str) -> dict[str, Any]:
    text = response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1]).strip()
            if text.startswith("json"):
                text = text[4:].lstrip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model response is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("Model response must be a JSON object.")
    return value


class Planner:
    def __init__(self, model: TextModel, tools: ToolRegistry) -> None:
        self._model = model
        self._tools = tools

    async def create_plan(self, goal: str, feedback: str = "") -> TaskPlan:
        prompt = (
            "Create a short executable plan for the user's task. "
            "Use only the listed tools. Return JSON only in the shape "
            '{"steps":[{"id":"step-1","description":"...","tool":"echo",'
            '"input":{"text":"..."}}]}. Each step needs a unique id. '
            "Do not claim work that a tool cannot perform.\n\n"
            f"Task: {goal}\n"
            f"Available tools: {json.dumps(self._tools.descriptions())}\n"
            f"Feedback from the previous attempt: {feedback or 'None'}"
        )
        data = parse_json_object(await self._model.complete(prompt))
        raw_steps = data.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise ValueError("Planner response must contain a non-empty 'steps' list.")

        steps: list[PlanStep] = []
        seen_ids: set[str] = set()
        for item in raw_steps:
            if not isinstance(item, dict):
                raise ValueError("Each plan step must be a JSON object.")
            step_id = item.get("id")
            description = item.get("description")
            tool = item.get("tool")
            arguments = item.get("input", {})
            if not all(isinstance(value, str) and value.strip() for value in (step_id, description, tool)):
                raise ValueError("Each step requires non-empty string id, description, and tool fields.")
            if step_id in seen_ids:
                raise ValueError(f"Duplicate plan step id: {step_id}")
            if not isinstance(arguments, dict):
                raise ValueError(f"Input for step '{step_id}' must be an object.")
            if tool not in self._tools.descriptions():
                raise ValueError(f"Planner selected unavailable tool '{tool}'.")
            seen_ids.add(step_id)
            steps.append(PlanStep(step_id, description, tool, arguments))
        return TaskPlan(steps)
