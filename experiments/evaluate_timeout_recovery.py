"""Deterministic fault-injection evaluation for AgentFlow timeout recovery.

This is a controlled software-path evaluation. It uses a scripted model and
mocked timeout outcomes; it does not measure live Ollama timeout reliability.
Run from the repository root:
    python experiments/evaluate_timeout_recovery.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from agentflow.budget import BudgetLimits
from agentflow.coding_agent import CodingAgent, CodingAgentConfig, RecoveryStrategy
from agentflow.local_model import ModelResponse
from agentflow.workspace import CommandResult, WorkspaceTools


class ScriptedModel:
    """Return preprogrammed responses or raise preprogrammed exceptions."""

    model = "scripted-timeout-model"
    device = "not_applicable"

    def __init__(self, responses: list[str | Exception]) -> None:
        self._responses = iter(responses)
        self.prompts: list[str] = []

    async def complete(
        self,
        prompt: str,
        max_output_tokens: int,
        timeout: float,
        response_schema: dict[str, Any] | None = None,
    ) -> ModelResponse:
        self.prompts.append(prompt)
        try:
            response = next(self._responses)
        except StopIteration as exc:
            raise AssertionError("ScriptedModel ran out of configured responses.") from exc
        if isinstance(response, Exception):
            raise response
        return ModelResponse(
            content=response,
            input_tokens=12,
            output_tokens=8,
            done=True,
            done_reason="stop",
        )


def action(tool: str, arguments: dict[str, Any] | None = None) -> str:
    return json.dumps({"action": {"tool": tool, "arguments": arguments or {}}})


def passing_tests(_timeout: float) -> CommandResult:
    return CommandResult(["pytest"], 0, "1 passed", "")


def timed_out_tests(_timeout: float) -> CommandResult:
    return CommandResult(
        ["pytest"], None, "", "injected test timeout", timed_out=True
    )


def timed_out_command(arguments: dict[str, Any], _timeout: float) -> CommandResult:
    command = arguments.get("command", ["python", "-c", "pass"])
    return CommandResult(
        list(command), None, "", "injected command timeout", timed_out=True
    )


async def run_case(case_id: str, root: Path) -> dict[str, Any]:
    """Run one deterministic timeout scenario in an isolated workspace."""
    root.mkdir(parents=True, exist_ok=True)
    limits = BudgetLimits()
    model_responses: list[str | Exception]
    workspace = WorkspaceTools(root, allow_tests=True)
    expected_decision = "REPLAN"
    expected_resolved = True
    expected_recovery_prompt = True

    if case_id == "T1_model_timeout_replan":
        model_responses = [TimeoutError("injected model timeout"), action("finish")]
        workspace.run_tests = passing_tests  # type: ignore[method-assign]

    elif case_id == "T2_test_timeout_replan":
        model_responses = [
            action("finish"),
            action("write_file", {"path": "recovered.py", "content": "value = 1\n"}),
            action("finish"),
        ]
        test_results = iter([timed_out_tests(1.0), passing_tests(1.0)])
        workspace.run_tests = lambda timeout: next(test_results)  # type: ignore[method-assign]

    elif case_id == "T3_command_timeout_replan":
        model_responses = [
            action("run_command", {"command": ["python", "-c", "pass"]}),
            action("finish"),
        ]
        workspace = WorkspaceTools(
            root,
            allow_tests=True,
            allow_shell=True,
            allowed_commands=("python",),
        )
        workspace.run_command = timed_out_command  # type: ignore[method-assign]
        workspace.run_tests = passing_tests  # type: ignore[method-assign]

    elif case_id == "T4_insufficient_tokens_stop":
        model_responses = [TimeoutError("injected model timeout")]
        limits = BudgetLimits(max_tokens=1200, max_output_tokens_per_call=1024)
        expected_decision = "STOP"
        expected_resolved = False
        expected_recovery_prompt = False
        workspace.run_tests = lambda _timeout: (_ for _ in ()).throw(  # type: ignore[method-assign]
            AssertionError("Tests must not run after the model times out.")
        )

    elif case_id == "T5_insufficient_time_stop":
        class DelayedTimeoutModel(ScriptedModel):
            async def complete(
                self,
                prompt: str,
                max_output_tokens: int,
                timeout: float,
                response_schema: dict[str, Any] | None = None,
            ) -> ModelResponse:
                self.prompts.append(prompt)
                await asyncio.sleep(0.02)
                raise TimeoutError("injected delayed model timeout")

        model = DelayedTimeoutModel([])
        limits = BudgetLimits(
            task_timeout_seconds=0.01,
            model_timeout_seconds=0.01,
            min_repair_seconds=0.001,
        )
        expected_decision = "STOP"
        expected_resolved = False
        expected_recovery_prompt = False
        workspace.run_tests = lambda _timeout: (_ for _ in ()).throw(  # type: ignore[method-assign]
            AssertionError("Tests must not run after the model times out.")
        )

    elif case_id == "T6_repair_limit_stop":
        model_responses = [TimeoutError("injected model timeout")]
        limits = BudgetLimits(max_repair_attempts=0)
        expected_decision = "STOP"
        expected_resolved = False
        expected_recovery_prompt = False
        workspace.run_tests = lambda _timeout: (_ for _ in ()).throw(  # type: ignore[method-assign]
            AssertionError("Tests must not run after the model times out.")
        )

    elif case_id == "T7_timeout_then_alternative_action":
        model_responses = [
            action("run_command", {"command": ["python", "-c", "pass"]}),
            action("write_file", {"path": "recovered.py", "content": "value = 7\n"}),
            action("finish"),
        ]
        workspace = WorkspaceTools(
            root,
            allow_tests=True,
            allow_shell=True,
            allowed_commands=("python",),
        )
        workspace.run_command = timed_out_command  # type: ignore[method-assign]
        workspace.run_tests = passing_tests  # type: ignore[method-assign]

    else:
        raise ValueError(f"Unknown scenario: {case_id}")

    if case_id != "T5_insufficient_time_stop":
        model = ScriptedModel(model_responses)

    started = time.monotonic()
    result = await CodingAgent(
        model,
        workspace,
        CodingAgentConfig(
            strategy=RecoveryStrategy.FAILURE_GUIDED,
            budgets=limits,
            max_actions_per_attempt=2,
        ),
    ).run(f"Execute controlled timeout scenario {case_id}.")

    elapsed = time.monotonic() - started
    first_attempt = result.attempts[0] if result.attempts else None
    observed_decision = (
        first_attempt.recovery_plan.get("decision")
        if first_attempt and first_attempt.recovery_plan
        else None
    )
    recovery_prompt_seen = any(
        "bounded timeout replan" in prompt.lower()
        for prompt in model.prompts
    )
    budget_events = result.budget.get("budget_events", [])
    token_overruns = sum(event.get("event") == "token_overrun" for event in budget_events)
    budget_rejections = sum(
        event.get("event") == "budget_rejected" for event in budget_events
    )
    expected_prompt_observed = (
        recovery_prompt_seen if expected_recovery_prompt else not recovery_prompt_seen
    )
    passed = (
        observed_decision == expected_decision
        and result.resolved is expected_resolved
        and expected_prompt_observed
        and token_overruns == 0
    )

    return {
        "record_type": "timeout_case",
        "scenario": case_id,
        "expected_decision": expected_decision,
        "observed_decision": observed_decision,
        "expected_resolved": expected_resolved,
        "resolved": result.resolved,
        "recovery_prompt_seen": recovery_prompt_seen,
        "model_calls": result.budget.get("model_calls"),
        "tool_calls": result.budget.get("tool_calls"),
        "test_runs": result.budget.get("test_runs"),
        "elapsed_seconds": round(elapsed, 6),
        "budget_rejections": budget_rejections,
        "token_overruns": token_overruns,
        "budget_stop_reason": result.budget.get("stop_reason"),
        "passed": passed,
        "result": result.to_dict(),
    }


async def evaluate(output_path: Path) -> int:
    scenarios = [
        "T1_model_timeout_replan",
        "T2_test_timeout_replan",
        "T3_command_timeout_replan",
        "T4_insufficient_tokens_stop",
        "T5_insufficient_time_stop",
        "T6_repair_limit_stop",
        "T7_timeout_then_alternative_action",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="agentflow-timeout-eval-") as temp:
        base = Path(temp)
        for index, scenario in enumerate(scenarios, start=1):
            row = await run_case(scenario, base / f"case-{index}")
            rows.append(row)
            print(
                f"{scenario}: {'PASS' if row['passed'] else 'FAIL'} "
                f"(expected={row['expected_decision']}, "
                f"observed={row['observed_decision']}, "
                f"resolved={row['resolved']})"
            )

    metadata = {
        "record_type": "run_metadata",
        "experiment": "AgentFlow deterministic timeout fault-injection",
        "evaluation_type": "scripted_model_and_mocked_timeout_outcomes",
        "live_ollama_used": False,
        "scenario_count": len(rows),
        "passed": sum(row["passed"] for row in rows),
        "failed": sum(not row["passed"] for row in rows),
        "note": (
            "This evaluates deterministic control-flow and budget decisions. "
            "It does not estimate real-world model timeout frequency or reliability."
        ),
    }
    with output_path.open("w", encoding="utf-8") as file:
        file.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    passed = metadata["passed"]
    failed = metadata["failed"]
    print(f"\nSummary: {passed}/{len(rows)} passed; {failed} failed.")
    print(f"Evidence written to: {output_path}")
    return 0 if failed == 0 else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run deterministic AgentFlow timeout fault-injection scenarios."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/timeout-recovery.jsonl"),
        help="JSONL evidence output path.",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(evaluate(args.output)))


if __name__ == "__main__":
    main()
