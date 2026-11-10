"""CLI and orchestration for the plan/execute/test/critique/recover loop."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import yaml

from .critic import Critic
from .executor import Executor
from .model import CopilotModel, TextModel
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


def _load_config(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except OSError as exc:
        raise RuntimeError(f"Could not read config file '{path}': {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"Config file '{path}' must contain a YAML object.")
    return config


async def _run_cli(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    model = CopilotModel(args.model or str(config.get("model", "gpt-4.1")))
    await model.start()
    try:
        controller = AgentController(
            model,
            max_iterations=(
                args.max_iterations
                if args.max_iterations is not None
                else int(config.get("max_iterations", 3))
            ),
            retries_per_step=(
                args.retries_per_step
                if args.retries_per_step is not None
                else int(config.get("retries_per_step", 1))
            ),
        )
        result = await controller.run(TaskRequest(args.goal))
        print(json.dumps(result.to_dict(), indent=2, default=str))
        return 0 if result.complete else 1
    finally:
        await model.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an agentflow task.")
    parser.add_argument("goal", help="Task for the agent to complete")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment.yaml"),
        help="YAML settings file",
    )
    parser.add_argument("--model", help="Override the model configured in YAML")
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--retries-per-step", type=int)
    args = parser.parse_args()
    try:
        exit_code = asyncio.run(_run_cli(args))
    except Exception as exc:
        parser.exit(2, f"agentflow: {type(exc).__name__}: {exc}\n")
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()