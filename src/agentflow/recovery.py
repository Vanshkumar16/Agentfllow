"""Bounded retry handling for failed tool steps."""

from __future__ import annotations

from .executor import Executor
from .schemas import PlanStep, StepResult


class Recovery:
    def __init__(self, executor: Executor, retries_per_step: int = 1) -> None:
        if retries_per_step < 0:
            raise ValueError("retries_per_step must not be negative.")
        self._executor = executor
        self._retries_per_step = retries_per_step

    def recover(
        self, steps: list[PlanStep], results: list[StepResult]
    ) -> list[StepResult]:
        steps_by_id = {step.id: step for step in steps}
        recovered: list[StepResult] = []
        for result in results:
            current = result
            step = steps_by_id.get(result.step_id)
            if step is None:
                recovered.append(result)
                continue
            for attempt in range(2, self._retries_per_step + 2):
                if current.succeeded:
                    break
                current = self._executor.execute_step(step, attempts=attempt)
            recovered.append(current)
        return recovered
