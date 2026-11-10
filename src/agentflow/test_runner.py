"""Deterministic checks on plan execution before model critique."""

from __future__ import annotations

from .schemas import StepResult, TestReport, TaskPlan


class TestRunner:
    def run(self, plan: TaskPlan, results: list[StepResult]) -> TestReport:
        checks: list[str] = []
        if not plan.steps:
            checks.append("Plan contains at least one step: failed")
        else:
            checks.append("Plan contains at least one step: passed")

        result_ids = [result.step_id for result in results]
        expected_ids = [step.id for step in plan.steps]
        if result_ids == expected_ids:
            checks.append("Every step produced a result in plan order: passed")
        else:
            checks.append("Every step produced a result in plan order: failed")

        failed_steps = [result.step_id for result in results if not result.succeeded]
        if failed_steps:
            checks.append(f"All tools succeeded: failed ({', '.join(failed_steps)})")
        else:
            checks.append("All tools succeeded: passed")
        return TestReport(
            passed=bool(plan.steps) and result_ids == expected_ids and not failed_steps,
            checks=checks,
        )
