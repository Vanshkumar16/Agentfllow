"""Sequential execution of a validated plan."""

from __future__ import annotations

from .schemas import PlanStep, StepResult, TaskPlan
from .tools import ToolRegistry


class Executor:
    def __init__(self, tools: ToolRegistry) -> None:
        self._tools = tools

    def execute_step(self, step: PlanStep, attempts: int = 1) -> StepResult:
        try:
            output = self._tools.execute(step.tool, step.input)
        except Exception as exc:
            return StepResult(step.id, step.tool, error=f"{type(exc).__name__}: {exc}", attempts=attempts)
        return StepResult(step.id, step.tool, output=output, attempts=attempts)

    def execute(self, plan: TaskPlan) -> list[StepResult]:
        return [self.execute_step(step) for step in plan.steps]
