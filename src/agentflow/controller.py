"""Compatibility controller for the original tool-only AgentFlow API."""

from __future__ import annotations

from .critic import Critic
from .executor import Executor
from .model import TextModel
from .planner import Planner
from .recovery import Recovery
from .schemas import IterationResult, RunResult, TaskRequest
from .test_runner import TestRunner
from .tools import ToolRegistry


class AgentController:
    def __init__(
        self,
        model: TextModel,
        tools: ToolRegistry | None = None,
        max_iterations: int = 3,
        retries_per_step: int = 1,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1.")
        self._model = model
        self._tools = tools or ToolRegistry()
        self._planner = Planner(model, self._tools)
        self._executor = Executor(self._tools)
        self._tests = TestRunner()
        self._critic = Critic(model)
        self._recovery = Recovery(self._executor, retries_per_step)
        self._max_iterations = max_iterations

    async def run(self, request: TaskRequest) -> RunResult:
        history: list[IterationResult] = []
        feedback = ""
        for _ in range(self._max_iterations):
            plan = await self._planner.create_plan(request.goal, feedback)
            results = self._executor.execute(plan)
            results = self._recovery.recover(plan.steps, results)
            tests = self._tests.run(plan, results)
            critique = await self._critic.review(request.goal, results, tests)
            history.append(IterationResult(plan, results, tests, critique))
            if critique.complete and tests.passed:
                return RunResult(request.goal, True, history, critique.feedback)
            feedback = critique.feedback
            if not tests.passed:
                feedback = f"{feedback}\nExecution checks: {'; '.join(tests.checks)}"
        summary = history[-1].critique.feedback if history else "No iteration was completed."
        return RunResult(request.goal, False, history, summary)

def main() -> None:
    from .cli import main as coding_agent_main

    coding_agent_main()


if __name__ == "__main__":
    main()