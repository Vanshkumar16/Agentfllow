"""Local coding-agent loop with selectable, budget-aware recovery policies."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Protocol

from .budget import BudgetLimits, BudgetTracker
from .failure import (
    FailureCategory,
    classify_failure,
    repeated_failure,
)
from .local_model import ModelResponse
from .workspace import WorkspaceError, WorkspaceTools


class CodingModel(Protocol):
    async def complete(
        self, prompt: str, max_output_tokens: int, timeout: float
    ) -> ModelResponse:
        """Return a single model response and provider-reported token usage."""


class RecoveryStrategy(str, Enum):
    SINGLE_SHOT = "single-shot"
    FIXED_RETRY = "fixed-retry"
    FAILURE_GUIDED = "failure-guided"


@dataclass(frozen=True)
class CodingAgentConfig:
    strategy: RecoveryStrategy = RecoveryStrategy.FAILURE_GUIDED
    budgets: BudgetLimits = field(default_factory=BudgetLimits)
    max_actions_per_attempt: int = 8
    max_prompt_chars: int = 24_000

    def __post_init__(self) -> None:
        if not isinstance(self.strategy, RecoveryStrategy):
            raise ValueError("strategy must be a RecoveryStrategy value.")
        if not isinstance(self.budgets, BudgetLimits):
            raise ValueError("budgets must be a BudgetLimits object.")
        if (
            not isinstance(self.max_actions_per_attempt, int)
            or isinstance(self.max_actions_per_attempt, bool)
            or self.max_actions_per_attempt < 1
        ):
            raise ValueError("max_actions_per_attempt must be at least 1.")
        if (
            not isinstance(self.max_prompt_chars, int)
            or isinstance(self.max_prompt_chars, bool)
            or self.max_prompt_chars < 4_000
        ):
            raise ValueError("max_prompt_chars must be at least 4000.")


@dataclass
class Attempt:
    number: int
    actions: list[dict[str, Any]] = field(default_factory=list)
    test: dict[str, Any] | None = None
    finished: bool = False
    passed: bool = False
    failure: str | None = None
    failure_category: str | None = None
    recovery_action: str | None = None
    recovery_decision: str | None = None


@dataclass(frozen=True)
class CodingRunResult:
    task: str
    workspace: str
    strategy: str
    resolved: bool
    attempts: list[Attempt]
    budget: dict[str, Any]
    budget_limits: dict[str, int | float]
    summary: str

    @property
    def repair_attempts(self) -> int:
        return max(0, len(self.attempts) - 1)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["repair_attempts"] = self.repair_attempts
        data["cost_usd"] = None
        data["cost_note"] = "The configured local model runtime does not report monetary cost."
        return data


class CodingAgent:
    def __init__(
        self,
        model: CodingModel,
        workspace: WorkspaceTools,
        config: CodingAgentConfig | None = None,
    ) -> None:
        self.model = model
        self.workspace = workspace
        self.config = config or CodingAgentConfig()

    async def run(self, task: str) -> CodingRunResult:
        if not task.strip():
            raise ValueError("Task must not be empty.")
        budget = BudgetTracker(self.config.budgets)
        if self.config.strategy is RecoveryStrategy.SINGLE_SHOT:
            return await self._run_single_shot(task, budget)
        attempts: list[Attempt] = []
        summary = "Task was not resolved."
        last_failure_details = ""
        previous_failure_fingerprint: str | None = None

        for repair_number in range(self.config.budgets.max_repair_attempts + 1):
            attempt = Attempt(number=repair_number)
            attempts.append(attempt)
            guidance = self._recovery_guidance(
                repair_number, last_failure_details, budget
            )
            attempt_failure_parts: list[str] = []

            for _ in range(self.config.max_actions_per_attempt):
                prompt = self._prompt(task, repair_number, guidance, attempt.actions, budget)
                max_tokens = budget.begin_model_call(prompt)
                if max_tokens is None:
                    attempt_failure_parts.append(budget.stop_reason or "Model call unavailable.")
                    break
                try:
                    response = await self.model.complete(
                        prompt,
                        max_output_tokens=max_tokens,
                        timeout=min(
                            self.config.budgets.model_timeout_seconds,
                            budget.remaining_seconds,
                        ),
                    )
                except (OSError, RuntimeError, TimeoutError) as exc:
                    budget.record_model_usage(None, None)
                    attempt_failure_parts.append(f"Model request failed: {exc}")
                    break

                budget.record_model_usage(response.input_tokens, response.output_tokens)
                if budget.stop_reason == "token budget exceeded by the completed model call":
                    attempt_failure_parts.append(budget.stop_reason)
                    break
                try:
                    action = _parse_action(response.content)
                except ValueError as exc:
                    attempt_failure_parts.append(str(exc))
                    break

                tool_name = action["tool"]
                if tool_name == "finish":
                    attempt.finished = True
                    break
                if not budget.begin_tool_call():
                    attempt_failure_parts.append(budget.stop_reason or "Tool call unavailable.")
                    break
                try:
                    output = self._execute_action(
                        tool_name,
                        action["arguments"],
                        min(
                            self.config.budgets.command_timeout_seconds,
                            budget.remaining_seconds,
                        ),
                    )
                    action_record = {"tool": tool_name, "arguments": action["arguments"], "output": output}
                except (WorkspaceError, OSError, ValueError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    action_record = {
                        "tool": tool_name,
                        "arguments": action["arguments"],
                        "error": error,
                    }
                    attempt_failure_parts.append(error)
                attempt.actions.append(action_record)

            action_assessment = classify_failure("\n".join(attempt_failure_parts))
            if action_assessment.category is FailureCategory.TIMEOUT:
                test_result = {
                    "passed": False,
                    "timed_out": True,
                    "error": (
                        "A model or action timed out; tests were not run to avoid "
                        "repeating timed-out work."
                    ),
                }
            else:
                test_result = self._run_tests(budget)
            attempt.test = test_result
            attempt.passed = (
                attempt.finished
                and test_result.get("passed") is True
                and not (budget.stop_reason and "budget" in budget.stop_reason)
            )
            if attempt.passed:
                summary = "Agent finished the task and the configured tests passed."
                break

            test_failure = _test_failure(test_result)
            failure_parts = attempt_failure_parts + ([test_failure] if test_failure else [])
            if not attempt.finished and not failure_parts:
                failure_parts.append("Agent used its action limit without finishing.")
            if budget.stop_reason:
                failure_parts.append(budget.stop_reason)
            last_failure_details = "\n".join(part for part in failure_parts if part)
            attempt.failure = last_failure_details or "Task was not confirmed complete."
            assessment = classify_failure(
                attempt.failure,
                timed_out=bool(test_result.get("timed_out")),
            )
            if (
                self.config.strategy is RecoveryStrategy.FAILURE_GUIDED
                and previous_failure_fingerprint == assessment.fingerprint
            ):
                assessment = repeated_failure(assessment)
            attempt.failure_category = assessment.category.value
            attempt.recovery_action = assessment.recovery_action
            last_failure_details = (
                f"Failure category: {assessment.category.value}\n"
                f"Recommended recovery: {assessment.recovery_action}\n"
                f"Failure evidence:\n{assessment.evidence}"
            )

            if repair_number >= self.config.budgets.max_repair_attempts:
                attempt.recovery_decision = "stop: repair-attempt limit reached"
                summary = attempt.failure
                break
            if assessment.category is FailureCategory.REPEATED_IDENTICAL:
                attempt.recovery_decision = "stop: repeated identical failure"
                summary = attempt.recovery_action
                break
            if (
                self.config.strategy is RecoveryStrategy.FAILURE_GUIDED
                and assessment.category is FailureCategory.TIMEOUT
            ):
                attempt.recovery_decision = "stop: timeout requires a bounded replan"
                summary = attempt.recovery_action
                break
            should_retry, decision = self._should_retry(
                attempt, test_result, attempt_failure_parts, budget
            )
            attempt.recovery_decision = decision
            if not should_retry:
                summary = decision
                break
            previous_failure_fingerprint = assessment.fingerprint
            summary = "Recovery scheduled within the remaining task budget."

        return CodingRunResult(
            task=task,
            workspace=str(self.workspace.root),
            strategy=self.config.strategy.value,
            resolved=any(attempt.passed for attempt in attempts),
            attempts=attempts,
            budget=budget.to_dict(),
            budget_limits=budget.limits_dict(),
            summary=summary,
        )

    async def _run_single_shot(
        self, task: str, budget: BudgetTracker
    ) -> CodingRunResult:
        attempt = Attempt(number=0)
        attempts = [attempt]
        failure: list[str] = []
        snapshot: dict[str, Any] = {"files": {}, "truncated": False}
        if budget.begin_tool_call():
            snapshot = self.workspace.snapshot(
                max_total_chars=max(1, self.config.max_prompt_chars // 2)
            )
            attempt.actions.append({"tool": "snapshot", "output": snapshot})
        else:
            failure.append(budget.stop_reason or "Workspace snapshot unavailable.")

        prompt = (
            "You are a local Python repair agent using a single-shot baseline. "
            "Return exactly one JSON object with 'complete' and 'patches' fields: "
            '{"complete":true,"patches":[{"path":"solution.py","content":"full file text"}]}. '
            "Provide every full file replacement needed in this one response. Do not "
            "use markdown fences, commands, or files outside the workspace. The "
            "'files' snapshot may be truncated; do not overwrite a file unless its "
            "complete current contents are present.\n\n"
            f"Task: {task}\n"
            f"Workspace snapshot: {json.dumps(snapshot, ensure_ascii=False)}\n"
            f"Budget: {json.dumps(budget.to_dict(include_events=False))}"
        )
        if len(prompt) > self.config.max_prompt_chars:
            failure.append("Single-shot prompt exceeds max_prompt_chars.")
        else:
            max_tokens = budget.begin_model_call(prompt)
            if max_tokens is None:
                failure.append(budget.stop_reason or "Model call unavailable.")
            else:
                try:
                    response = await self.model.complete(
                        prompt,
                        max_output_tokens=max_tokens,
                        timeout=min(
                            self.config.budgets.model_timeout_seconds,
                            budget.remaining_seconds,
                        ),
                    )
                except (OSError, RuntimeError, TimeoutError) as exc:
                    budget.record_model_usage(None, None)
                    failure.append(f"Model request failed: {exc}")
                else:
                    budget.record_model_usage(
                        response.input_tokens, response.output_tokens
                    )
                    if budget.stop_reason == "token budget exceeded by the completed model call":
                        failure.append(budget.stop_reason)
                    else:
                        try:
                            patch_response = _parse_patch_response(response.content)
                        except ValueError as exc:
                            failure.append(str(exc))
                        else:
                            attempt.finished = patch_response["complete"]
                            for patch in patch_response["patches"]:
                                if not budget.begin_tool_call():
                                    failure.append(
                                        budget.stop_reason or "File write unavailable."
                                    )
                                    break
                                try:
                                    output = self.workspace.write_file(patch)
                                except (WorkspaceError, OSError, ValueError) as exc:
                                    failure.append(f"{type(exc).__name__}: {exc}")
                                    attempt.actions.append(
                                        {
                                            "tool": "write_file",
                                            "arguments": {"path": patch["path"]},
                                            "error": failure[-1],
                                        }
                                    )
                                    break
                                attempt.actions.append(
                                    {
                                        "tool": "write_file",
                                        "arguments": {"path": patch["path"]},
                                        "output": output,
                                    }
                                )

        test_result = self._run_tests(budget)
        attempt.test = test_result
        attempt.passed = (
            attempt.finished
            and test_result.get("passed") is True
            and not (budget.stop_reason and "budget" in budget.stop_reason)
        )
        if not attempt.passed:
            test_failure = _test_failure(test_result)
            if test_failure:
                failure.append(test_failure)
            if budget.stop_reason:
                failure.append(budget.stop_reason)
            attempt.failure = "\n".join(failure) or "Task was not confirmed complete."
            assessment = classify_failure(
                attempt.failure,
                timed_out=bool(test_result.get("timed_out")),
            )
            attempt.failure_category = assessment.category.value
            attempt.recovery_action = assessment.recovery_action
            attempt.recovery_decision = "stop: single-shot strategy does not repair"
        summary = (
            "Single-shot patch applied and configured tests passed."
            if attempt.passed
            else attempt.failure or "Single-shot attempt did not resolve the task."
        )
        return CodingRunResult(
            task=task,
            workspace=str(self.workspace.root),
            strategy=self.config.strategy.value,
            resolved=attempt.passed,
            attempts=attempts,
            budget=budget.to_dict(),
            budget_limits=budget.limits_dict(),
            summary=summary,
        )

    def _execute_action(
        self, tool_name: str, arguments: dict[str, Any], timeout: float
    ) -> dict[str, Any]:
        if tool_name == "list_files":
            return self.workspace.list_files(arguments)
        if tool_name == "read_file":
            return self.workspace.read_file(arguments)
        if tool_name == "write_file":
            return self.workspace.write_file(arguments)
        if tool_name == "run_command":
            result = self.workspace.run_command(arguments, timeout)
            if not result.succeeded:
                raise WorkspaceError(
                    f"Command failed (returncode={result.returncode}, "
                    f"timed_out={result.timed_out}): {result.stderr or result.stdout}"
                )
            return result.to_dict()
        raise WorkspaceError(f"Unknown action '{tool_name}'.")

    def _run_tests(self, budget: BudgetTracker) -> dict[str, Any]:
        if not self.workspace.allow_tests:
            return {
                "passed": False,
                "error": "Test execution is disabled; pass --allow-tests to explicitly enable it.",
            }
        if not budget.begin_test_run():
            return {
                "passed": False,
                "error": budget.stop_reason or "Test run unavailable.",
            }
        try:
            result = self.workspace.run_tests(
                min(
                    self.config.budgets.command_timeout_seconds,
                    budget.remaining_seconds,
                )
            )
        except WorkspaceError as exc:
            return {"passed": False, "error": str(exc)}
        return {"passed": result.succeeded, **result.to_dict()}

    def _should_retry(
        self,
        attempt: Attempt,
        test_result: dict[str, Any],
        action_failures: list[str],
        budget: BudgetTracker,
    ) -> tuple[bool, str]:
        if self.config.strategy is RecoveryStrategy.SINGLE_SHOT:
            return False, "stop: single-shot strategy does not repair"
        if not self.workspace.allow_tests:
            return False, "stop: tests are disabled, so a repair cannot be verified"

        minimum_tokens = (
            self.config.budgets.min_repair_tokens
            if self.config.strategy is RecoveryStrategy.FAILURE_GUIDED
            else 1
        )
        minimum_seconds = (
            self.config.budgets.min_repair_seconds
            if self.config.strategy is RecoveryStrategy.FAILURE_GUIDED
            else 0
        )
        can_repair, budget_reason = budget.can_repair(
            minimum_tokens=minimum_tokens,
            minimum_seconds=minimum_seconds,
        )
        if not can_repair:
            return False, f"stop: {budget_reason}"

        if self.config.strategy is RecoveryStrategy.FIXED_RETRY:
            return True, "retry: fixed-retry policy and budget permit another attempt"

        actionable_failure = bool(action_failures) or (
            test_result.get("passed") is False
            and bool(test_result.get("stdout") or test_result.get("stderr") or test_result.get("error"))
        )
        if not actionable_failure:
            return False, "stop: no actionable failure signal for failure-guided recovery"
        return True, f"retry: actionable failure; {budget_reason}"

    def _recovery_guidance(
        self, repair_number: int, failure_details: str, budget: BudgetTracker
    ) -> str:
        if repair_number == 0:
            return "This is the initial attempt. Inspect the workspace before editing."
        if self.config.strategy is RecoveryStrategy.FAILURE_GUIDED:
            return (
                "This is a repair attempt. Follow the classified recovery action, "
                "use the exact failure evidence, and re-check relevant files before "
                "editing. Do not repeat an unchanged repair.\n"
                f"Prior failure assessment:\n{failure_details}\n"
                f"Remaining budget: {json.dumps(budget.to_dict(include_events=False))}"
            )
        return (
            "This is a fixed-retry attempt. Revisit the task and current workspace state, then "
            "make a different bounded attempt if appropriate. Detailed failure "
            "diagnostics are intentionally omitted for this baseline.\n"
            f"Remaining budget: {json.dumps(budget.to_dict(include_events=False))}"
        )

    def _prompt(
        self,
        task: str,
        repair_number: int,
        guidance: str,
        actions: list[dict[str, Any]],
        budget: BudgetTracker,
    ) -> str:
        available = ["list_files", "read_file", "write_file", "finish"]
        if self.workspace.allow_shell:
            available.insert(3, "run_command")
        context = (
            "You are a local Python software repair agent. Use only the provided "
            "workspace actions; never refer to files outside the workspace. Return "
            "exactly one JSON object per response in this format: "
            '{"action":{"tool":"read_file","arguments":{"path":"src/example.py"}}}. '
            'To finish, return {"action":{"tool":"finish","arguments":{}}}. '
            "Do not use markdown fences. Make small, task-focused edits. Do not "
            "claim completion unless you are ready for the configured tests.\n\n"
            f"Repair number: {repair_number}\n"
            f"Guidance: {guidance}\n"
            f"Available actions: {', '.join(available)}\n"
            f"Budget: {json.dumps(budget.to_dict(include_events=False))}"
        )
        if len(context) + len(task) + 80 > self.config.max_prompt_chars:
            raise ValueError("Task and recovery context exceed max_prompt_chars.")
        evidence = json.dumps(actions, ensure_ascii=False)
        evidence_limit = self.config.max_prompt_chars - len(context) - len(task) - 80
        if len(evidence) > evidence_limit:
            evidence = "[Earlier action evidence truncated]\n" + evidence[-evidence_limit:]
        return (
            f"{context}\n"
            f"Task: {task}\n"
            f"Actions and results from this attempt: {evidence}"
        )


def _parse_action(response: str) -> dict[str, Any]:
    text = response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1]).strip()
            if text.startswith("json"):
                text = text[4:].lstrip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model response is not valid JSON: {exc}") from exc
    action = payload.get("action") if isinstance(payload, dict) else None
    if not isinstance(action, dict):
        raise ValueError("Model response must contain an 'action' object.")
    tool = action.get("tool")
    arguments = action.get("arguments", {})
    if not isinstance(tool, str) or not tool.strip():
        raise ValueError("Action requires a non-empty tool name.")
    if not isinstance(arguments, dict):
        raise ValueError("Action arguments must be a JSON object.")
    return {"tool": tool, "arguments": arguments}


def _parse_patch_response(response: str) -> dict[str, Any]:
    text = response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1]).strip()
            if text.startswith("json"):
                text = text[4:].lstrip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Single-shot response is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("complete"), bool):
        raise ValueError("Single-shot response requires a boolean 'complete' field.")
    patches = payload.get("patches")
    if not isinstance(patches, list):
        raise ValueError("Single-shot response requires a 'patches' list.")
    validated: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    for patch in patches:
        if (
            not isinstance(patch, dict)
            or not isinstance(patch.get("path"), str)
            or not isinstance(patch.get("content"), str)
        ):
            raise ValueError("Each patch requires string 'path' and 'content' fields.")
        if patch["path"] in seen_paths:
            raise ValueError(f"Single-shot response contains duplicate patch path '{patch['path']}'.")
        seen_paths.add(patch["path"])
        validated.append({"path": patch["path"], "content": patch["content"]})
    return {"complete": payload["complete"], "patches": validated}


def _test_failure(test_result: dict[str, Any]) -> str:
    if test_result.get("passed") is True:
        return ""
    error = test_result.get("error")
    stdout = test_result.get("stdout", "")
    stderr = test_result.get("stderr", "")
    parts = [str(part) for part in (error, stdout, stderr) if part]
    return "\n".join(parts)[-8000:]
