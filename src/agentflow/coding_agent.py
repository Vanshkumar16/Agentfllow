"""Local coding-agent loop with selectable, budget-aware recovery policies."""

from __future__ import annotations

import json
import inspect
import math
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol

from .budget import BudgetLimits, BudgetTracker
from .failure import (
    FailureCategory,
    RecoveryDecision,
    classify_failure,
    repeated_failure,
)
from .local_model import ModelResponse
from .response_validation import (
    ResponseValidationError,
    action_response_schema,
    parse_action_response,
    parse_patch_response,
    patch_response_schema,
)
from .workspace import WorkspaceError, WorkspaceTools


class CodingModel(Protocol):
    async def complete(
        self,
        prompt: str,
        max_output_tokens: int,
        timeout: float,
        response_schema: dict[str, Any] | None = None,
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
    recovery_plan: dict[str, Any] | None = None
    patch_valid: bool | None = None
    response_valid: bool | None = None
    completion_reason: str | None = None


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
    run_id: str = ""
    experiment_id: str | None = None
    task_id: str | None = None
    patch_valid: bool | None = None
    final_test_passed: bool = False
    explicit_completion: bool = False
    budget_exhausted: bool = False
    completion_reason: str = ""
    per_call_logs: list[dict[str, Any]] = field(default_factory=list)

    @property
    def repair_attempts(self) -> int:
        return max(0, len(self.attempts) - 1)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["repair_attempts"] = self.repair_attempts
        data["strict_resolved"] = self.resolved
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

    async def run(
        self,
        task: str,
        *,
        experiment_id: str | None = None,
        task_id: str | None = None,
    ) -> CodingRunResult:
        if not task.strip():
            raise ValueError("Task must not be empty.")
        budget = BudgetTracker(self.config.budgets)
        run_id = uuid.uuid4().hex
        call_logs: list[dict[str, Any]] = []
        if self.config.strategy is RecoveryStrategy.SINGLE_SHOT:
            return await self._run_single_shot(
                task, budget, call_logs, run_id, experiment_id, task_id
            )
        attempts: list[Attempt] = []
        summary = "Task was not resolved."
        last_failure_details = ""
        recovery_context_action: str | None = None
        timeout_replan = False
        previous_failure_fingerprint: str | None = None
        patch_valid_state: bool | None = None

        for repair_number in range(self.config.budgets.max_repair_attempts + 1):
            attempt = Attempt(
                number=repair_number,
                recovery_action=recovery_context_action,
            )
            attempts.append(attempt)
            guidance = self._recovery_guidance(
                repair_number, last_failure_details, budget,
                timeout_replan=timeout_replan,
            )
            timeout_replan = False
            attempt_failure_parts: list[str] = []

            for _ in range(self.config.max_actions_per_attempt):
                try:
                    prompt = self._prompt(
                        task, repair_number, guidance, attempt.actions, budget
                    )
                except ValueError as exc:
                    attempt_failure_parts.append(str(exc))
                    break
                response, call_error, model_log = await self._invoke_model(
                    prompt,
                    action_response_schema(allow_shell=self.workspace.allow_shell),
                    budget,
                    call_logs,
                    run_id,
                    experiment_id,
                    task_id,
                    repair_number,
                    attempt.recovery_action,
                )
                if call_error is not None:
                    attempt_failure_parts.append(call_error)
                    break
                assert response is not None
                if budget.stop_reason == "token budget exceeded by the completed model call":
                    attempt_failure_parts.append(budget.stop_reason)
                    break
                termination_rejection = _response_termination_rejection(response)
                if termination_rejection is not None:
                    message, category = termination_rejection
                    model_log["validation_outcome"] = "invalid"
                    model_log["validation_phase"] = "termination"
                    model_log["failure_category"] = category.value
                    model_log["diagnostic"] = message
                    attempt.patch_valid = False
                    attempt.response_valid = False
                    patch_valid_state = False
                    attempt_failure_parts.append(message)
                    break
                try:
                    action = parse_action_response(
                        response.content,
                        allow_shell=self.workspace.allow_shell,
                    )
                except ResponseValidationError as exc:
                    message = _response_validation_message(exc, response)
                    model_log["validation_outcome"] = "invalid"
                    model_log["validation_phase"] = exc.phase
                    model_log["failure_category"] = _response_failure_category(exc, response).value
                    model_log["diagnostic"] = message[:1000]
                    attempt.patch_valid = False
                    attempt.response_valid = False
                    patch_valid_state = False
                    attempt_failure_parts.append(message)
                    break
                model_log["validation_outcome"] = "valid"
                model_log["validation_phase"] = "schema"
                attempt.response_valid = True

                tool_name = action["tool"]
                if tool_name == "finish":
                    attempt.finished = True
                    attempt.completion_reason = "agent emitted the explicit finish action"
                    break
                if not budget.begin_tool_call():
                    reason = budget.stop_reason or "Tool call unavailable."
                    attempt_failure_parts.append(reason)
                    call_logs.append(
                        _call_log(
                            "tool",
                            run_id,
                            experiment_id,
                            task_id,
                            self.config.strategy.value,
                            repair_number,
                            len(call_logs) + 1,
                            getattr(self.model, "model", None),
                            attempt.recovery_action,
                            "budget_rejected",
                            latency_seconds=None,
                            validation_outcome="not_applicable",
                            error=reason,
                        )
                    )
                    break
                tool_started = time.monotonic()
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
                    tool_status = "succeeded"
                except (WorkspaceError, OSError, ValueError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    action_record = {
                        "tool": tool_name,
                        "arguments": action["arguments"],
                        "error": error,
                    }
                    attempt_failure_parts.append(error)
                    tool_status = "failed"
                attempt.actions.append(action_record)
                if tool_name == "write_file":
                    attempt.patch_valid = tool_status == "succeeded"
                    patch_valid_state = attempt.patch_valid
                call_logs.append(
                    _call_log(
                        "tool",
                        run_id,
                        experiment_id,
                        task_id,
                        self.config.strategy.value,
                        repair_number,
                        len(call_logs) + 1,
                        getattr(self.model, "model", None),
                        attempt.recovery_action,
                        tool_status,
                        latency_seconds=time.monotonic() - tool_started,
                        validation_outcome="not_applicable",
                        tool=tool_name,
                    )
                )
                if tool_status == "failed":
                    if tool_name == "write_file":
                        attempt.patch_valid = False
                        patch_valid_state = False

            action_assessment = classify_failure("\n".join(attempt_failure_parts))
            if action_assessment.category is FailureCategory.TIMEOUT:
                timeout_reason = (
                    "A model or action timed out; tests were not run to avoid "
                    "repeating timed-out work."
                )
                test_result = {
                    "passed": False,
                    "timed_out": True,
                    "error": timeout_reason,
                }
                call_logs.append(
                    _call_log(
                        "test",
                        run_id,
                        experiment_id,
                        task_id,
                        self.config.strategy.value,
                        repair_number,
                        len(call_logs) + 1,
                        getattr(self.model, "model", None),
                        attempt.recovery_action,
                        "skipped_timeout",
                        latency_seconds=None,
                        validation_outcome="not_applicable",
                        passed=False,
                        timed_out=True,
                        error=timeout_reason,
                    )
                )
            else:
                test_result = self._run_tests(
                    budget,
                    call_logs,
                    run_id,
                    experiment_id,
                    task_id,
                    repair_number,
                    attempt.recovery_action,
                )
            attempt.test = test_result
            attempt.passed = (
                attempt.finished
                and test_result.get("passed") is True
                and not (budget.stop_reason and "budget" in budget.stop_reason)
            )
            if attempt.passed:
                summary = "Agent finished the task and the configured tests passed."
                attempt.completion_reason = "explicit finish action and final tests passed"
                break
            if test_result.get("passed") is True and not attempt.finished:
                attempt.completion_reason = (
                    "tests passed, but the model did not emit an explicit finish action"
                )
            elif test_result.get("passed") is True and budget.stop_reason:
                attempt.completion_reason = (
                    "tests passed and completion was signaled, but budget exhaustion "
                    "prevented strict resolution"
                )
            elif attempt.finished:
                attempt.completion_reason = (
                    "agent emitted an explicit finish action, but final tests did not pass"
                )

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
            recovery_context_action = assessment.recovery_action
            attempt.recovery_plan = {
                "suggested_decision": assessment.suggested_decision.value,
                "rationale": assessment.recovery_action,
                "failure_category": assessment.category.value,
                "fingerprint": assessment.fingerprint,
                "evidence": assessment.evidence,
                "attempt": repair_number,
            }
            last_failure_details = (
                f"Failure category: {assessment.category.value}\n"
                f"Recommended recovery: {assessment.recovery_action}\n"
                f"Failure evidence:\n{assessment.evidence}"
            )

            if repair_number >= self.config.budgets.max_repair_attempts:
                attempt.recovery_decision = "stop: repair-attempt limit reached"
                attempt.recovery_plan.update(
                    decision=RecoveryDecision.STOP.value,
                    rationale=attempt.recovery_decision,
                )
                summary = attempt.failure
                break
            if assessment.category is FailureCategory.REPEATED_IDENTICAL:
                attempt.recovery_decision = "stop: repeated identical failure"
                attempt.recovery_plan.update(
                    decision=RecoveryDecision.STOP.value,
                    rationale=attempt.recovery_decision,
                )
                summary = attempt.recovery_action
                break
            if (
                self.config.strategy is RecoveryStrategy.FAILURE_GUIDED
                and assessment.category is FailureCategory.TIMEOUT
            ):
                if not self.workspace.allow_tests:
                    replan = False
                    rationale = (
                        "stop: tests are disabled, so a timeout replan cannot be verified"
                    )
                else:
                    replan, budget_reason = budget.can_repair(
                        minimum_tokens=self.config.budgets.min_repair_tokens,
                        minimum_seconds=self.config.budgets.min_repair_seconds,
                    )
                    rationale = (
                        "replan: timeout recovery fits the remaining budget"
                        if replan
                        else f"stop: timeout replan unavailable; {budget_reason}"
                    )
                attempt.recovery_decision = rationale
                attempt.recovery_plan.update(
                    decision=(
                        RecoveryDecision.REPLAN.value
                        if replan
                        else RecoveryDecision.STOP.value
                    ),
                    rationale=rationale,
                )
                if replan:
                    timeout_replan = True
                    summary = "Timeout recovery scheduled within the remaining task budget."
                    continue
                summary = rationale
                break
            should_retry, decision = self._should_retry(
                attempt, test_result, attempt_failure_parts, budget
            )
            attempt.recovery_decision = decision
            if should_retry:
                actual_decision = (
                    RecoveryDecision.RETRY.value
                    if self.config.strategy is RecoveryStrategy.FIXED_RETRY
                    else assessment.suggested_decision.value
                )
            else:
                actual_decision = RecoveryDecision.STOP.value
            attempt.recovery_plan.update(
                decision=actual_decision,
                rationale=decision,
            )
            if not should_retry:
                summary = decision
                break
            previous_failure_fingerprint = assessment.fingerprint
            summary = "Recovery scheduled within the remaining task budget."

        final_attempt = attempts[-1] if attempts else None
        final_test_passed = bool(
            final_attempt
            and isinstance(final_attempt.test, dict)
            and final_attempt.test.get("passed") is True
        )
        explicit_completion = bool(final_attempt and final_attempt.finished)
        completion_reason = (
            final_attempt.completion_reason
            if final_attempt and final_attempt.completion_reason
            else "task did not reach strict completion"
        )
        return CodingRunResult(
            task=task,
            workspace=str(self.workspace.root),
            strategy=self.config.strategy.value,
            resolved=any(attempt.passed for attempt in attempts),
            attempts=attempts,
            budget=budget.to_dict(),
            budget_limits=budget.limits_dict(),
            summary=summary,
            run_id=run_id,
            experiment_id=experiment_id,
            task_id=task_id,
            patch_valid=patch_valid_state,
            final_test_passed=final_test_passed,
            explicit_completion=explicit_completion,
            budget_exhausted=budget.stop_reason is not None,
            completion_reason=completion_reason,
            per_call_logs=call_logs,
        )

    async def _run_single_shot(
        self,
        task: str,
        budget: BudgetTracker,
        call_logs: list[dict[str, Any]],
        run_id: str,
        experiment_id: str | None,
        task_id: str | None,
    ) -> CodingRunResult:
        attempt = Attempt(number=0)
        attempts = [attempt]
        failure: list[str] = []
        snapshot: dict[str, Any] = {"files": {}, "truncated": False}
        if budget.begin_tool_call():
            snapshot_started = time.monotonic()
            snapshot = self.workspace.snapshot(
                max_total_chars=max(1, self.config.max_prompt_chars // 2)
            )
            attempt.actions.append({"tool": "snapshot", "output": snapshot})
            call_logs.append(
                _call_log(
                    "tool",
                    run_id,
                    experiment_id,
                    task_id,
                    self.config.strategy.value,
                    0,
                    len(call_logs) + 1,
                    getattr(self.model, "model", None),
                    None,
                    "succeeded",
                    latency_seconds=time.monotonic() - snapshot_started,
                    validation_outcome="not_applicable",
                    tool="snapshot",
                )
            )
        else:
            reason = budget.stop_reason or "Workspace snapshot unavailable."
            failure.append(reason)
            call_logs.append(
                _call_log(
                    "tool",
                    run_id,
                    experiment_id,
                    task_id,
                    self.config.strategy.value,
                    0,
                    len(call_logs) + 1,
                    getattr(self.model, "model", None),
                    None,
                    "budget_rejected",
                    latency_seconds=None,
                    validation_outcome="not_applicable",
                    error=reason,
                    tool="snapshot",
                )
            )

        prompt = (
            "You are a local Python repair agent using a single-shot baseline. "
            "Return exactly one JSON object with 'complete' and 'patches' fields: "
            '{"complete":true,"patches":[{"path":"solution.py","content":"full file text"}]}. '
            "Use no other top-level or patch fields, use unique non-empty relative "
            "paths, and provide every full file replacement needed in this one response. Do not "
            "use markdown fences, commands, or files outside the workspace. The "
            "'files' snapshot may be truncated; do not overwrite a file unless its "
            "complete current contents are present.\n\n"
            f"Task: {task}\n"
            f"Workspace snapshot: {json.dumps(snapshot, ensure_ascii=False)}\n"
            f"Budget: {json.dumps(budget.to_dict(include_events=False))}"
        )
        if len(prompt) > self.config.max_prompt_chars:
            prompt_error = "Single-shot prompt exceeds max_prompt_chars."
            failure.append(prompt_error)
            call_logs.append(
                _call_log(
                    "model",
                    run_id,
                    experiment_id,
                    task_id,
                    self.config.strategy.value,
                    0,
                    len(call_logs) + 1,
                    getattr(self.model, "model", None),
                    None,
                    "rejected",
                    latency_seconds=None,
                    validation_outcome="not_called",
                    error=prompt_error,
                )
            )
        else:
            response, call_error, model_log = await self._invoke_model(
                prompt,
                patch_response_schema(),
                budget,
                call_logs,
                run_id,
                experiment_id,
                task_id,
                0,
                None,
            )
            if call_error is not None:
                failure.append(call_error)
            elif response is not None:
                if budget.stop_reason == "token budget exceeded by the completed model call":
                    failure.append(budget.stop_reason)
                elif (
                    termination_rejection := _response_termination_rejection(response)
                ) is not None:
                    message, category = termination_rejection
                    model_log["validation_outcome"] = "invalid"
                    model_log["validation_phase"] = "termination"
                    model_log["failure_category"] = category.value
                    model_log["diagnostic"] = message
                    attempt.patch_valid = False
                    attempt.response_valid = False
                    failure.append(message)
                else:
                    try:
                        patch_response = parse_patch_response(response.content)
                    except ResponseValidationError as exc:
                        message = _response_validation_message(exc, response)
                        model_log["validation_outcome"] = "invalid"
                        model_log["validation_phase"] = exc.phase
                        model_log["failure_category"] = _response_failure_category(exc, response).value
                        model_log["diagnostic"] = message[:1000]
                        attempt.patch_valid = False
                        attempt.response_valid = False
                        failure.append(message)
                    else:
                        model_log["validation_outcome"] = "valid"
                        model_log["validation_phase"] = "schema"
                        attempt.patch_valid = True
                        attempt.response_valid = True
                        attempt.finished = patch_response["complete"]
                        if attempt.finished:
                            attempt.completion_reason = (
                                "single-shot response explicitly set complete=true"
                            )
                        for patch in patch_response["patches"]:
                            if not budget.begin_tool_call():
                                reason = budget.stop_reason or "File write unavailable."
                                failure.append(reason)
                                call_logs.append(
                                    _call_log(
                                        "tool",
                                        run_id,
                                        experiment_id,
                                        task_id,
                                        self.config.strategy.value,
                                        0,
                                        len(call_logs) + 1,
                                        getattr(self.model, "model", None),
                                        None,
                                        "budget_rejected",
                                        latency_seconds=None,
                                        validation_outcome="not_applicable",
                                        error=reason,
                                        tool="write_file",
                                    )
                                )
                                break
                            write_started = time.monotonic()
                            try:
                                output = self.workspace.write_file(patch)
                            except (WorkspaceError, OSError, ValueError) as exc:
                                failure.append(f"{type(exc).__name__}: {exc}")
                                attempt.patch_valid = False
                                attempt.actions.append(
                                    {
                                        "tool": "write_file",
                                        "arguments": {"path": patch["path"]},
                                        "error": failure[-1],
                                    }
                                )
                                call_status = "failed"
                            else:
                                attempt.actions.append(
                                    {
                                        "tool": "write_file",
                                        "arguments": {"path": patch["path"]},
                                        "output": output,
                                    }
                                )
                                call_status = "succeeded"
                            call_logs.append(
                                _call_log(
                                    "tool",
                                    run_id,
                                    experiment_id,
                                    task_id,
                                    self.config.strategy.value,
                                    0,
                                    len(call_logs) + 1,
                                    getattr(self.model, "model", None),
                                    None,
                                    call_status,
                                    latency_seconds=time.monotonic() - write_started,
                                    validation_outcome="not_applicable",
                                    tool="write_file",
                                )
                            )

        if classify_failure("\n".join(failure)).category is FailureCategory.TIMEOUT:
            timeout_error = "Model request timed out; tests were not run."
            test_result = {"passed": False, "timed_out": True, "error": timeout_error}
            call_logs.append(
                _call_log(
                    "test",
                    run_id,
                    experiment_id,
                    task_id,
                    self.config.strategy.value,
                    0,
                    len(call_logs) + 1,
                    getattr(self.model, "model", None),
                    attempt.recovery_action,
                    "skipped_timeout",
                    latency_seconds=None,
                    validation_outcome="not_applicable",
                    passed=False,
                    timed_out=True,
                    error=timeout_error,
                )
            )
        else:
            test_result = self._run_tests(
                budget,
                call_logs,
                run_id,
                experiment_id,
                task_id,
                0,
                attempt.recovery_action,
            )
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
            attempt.recovery_plan = {
                "decision": RecoveryDecision.STOP.value,
                "rationale": attempt.recovery_decision,
                "failure_category": assessment.category.value,
                "fingerprint": assessment.fingerprint,
                "evidence": assessment.evidence,
                "attempt": 0,
            }
            if test_result.get("passed") is True and not attempt.finished:
                attempt.completion_reason = (
                    "tests passed, but the single-shot response did not explicitly set complete=true"
                )
            elif test_result.get("passed") is True and budget.stop_reason:
                attempt.completion_reason = (
                    "tests passed and complete=true was returned, but budget exhaustion "
                    "prevented strict resolution"
                )
            elif attempt.finished:
                attempt.completion_reason = (
                    "single-shot response explicitly completed, but final tests did not pass"
                )
        elif attempt.passed:
            attempt.completion_reason = "single-shot patch explicitly completed and tests passed"
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
            run_id=run_id,
            experiment_id=experiment_id,
            task_id=task_id,
            patch_valid=attempt.patch_valid,
            final_test_passed=test_result.get("passed") is True,
            explicit_completion=attempt.finished,
            budget_exhausted=budget.stop_reason is not None,
            completion_reason=attempt.completion_reason or "task did not reach strict completion",
            per_call_logs=call_logs,
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

    async def _invoke_model(
        self,
        prompt: str,
        response_schema: dict[str, Any],
        budget: BudgetTracker,
        call_logs: list[dict[str, Any]],
        run_id: str,
        experiment_id: str | None,
        task_id: str | None,
        attempt: int,
        recovery_action: str | None,
    ) -> tuple[ModelResponse | None, str | None, dict[str, Any]]:
        max_tokens = budget.begin_model_call(prompt)
        if max_tokens is None:
            rejection = budget.budget_events[-1] if budget.budget_events else {}
            entry = _call_log(
                "model",
                run_id,
                experiment_id,
                task_id,
                self.config.strategy.value,
                attempt,
                len(call_logs) + 1,
                getattr(self.model, "model", None),
                recovery_action,
                "budget_rejected",
                latency_seconds=None,
                validation_outcome="not_called",
                estimated_input_tokens=rejection.get("estimated_prompt_tokens"),
                error=budget.stop_reason or "Model call unavailable.",
            )
            call_logs.append(entry)
            return None, entry["error"], entry

        started = time.monotonic()
        try:
            complete = self.model.complete
            try:
                parameters = inspect.signature(complete).parameters.values()
            except (TypeError, ValueError):
                parameters = ()
            accepts_schema = any(
                parameter.name == "response_schema"
                or parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters
            )
            kwargs: dict[str, Any] = {
                "max_output_tokens": max_tokens,
                "timeout": min(
                    self.config.budgets.model_timeout_seconds,
                    budget.remaining_seconds,
                ),
            }
            if accepts_schema:
                kwargs["response_schema"] = response_schema
            raw_response = await complete(prompt, **kwargs)
            if isinstance(raw_response, ModelResponse):
                response = raw_response
            elif isinstance(raw_response, str):
                response = ModelResponse(raw_response, None, None)
            else:
                response = ModelResponse(
                    content="",
                    input_tokens=None,
                    output_tokens=None,
                    provider_metadata={
                        "content_type": type(
                            getattr(raw_response, "content", None)
                        ).__name__
                    },
                )
        except Exception as exc:
            accounting = budget.record_model_usage(None, None)
            latency = time.monotonic() - started
            error = f"Model request failed: {type(exc).__name__}: {exc}"
            entry = _call_log(
                "model",
                run_id,
                experiment_id,
                task_id,
                self.config.strategy.value,
                attempt,
                len(call_logs) + 1,
                getattr(self.model, "model", None),
                recovery_action,
                "failed",
                latency_seconds=latency,
                validation_outcome="not_available",
                response_termination="unknown",
                token_accounting=accounting,
                error=error[:1000],
            )
            call_logs.append(entry)
            return None, error, entry
        accounting = budget.record_model_usage(
            response.input_tokens, response.output_tokens
        )
        latency = time.monotonic() - started
        status = _termination_status(response)
        entry = _call_log(
            "model",
            run_id,
            experiment_id,
            task_id,
            self.config.strategy.value,
            attempt,
            len(call_logs) + 1,
            getattr(self.model, "model", None),
            recovery_action,
            "completed",
            latency_seconds=latency,
            validation_outcome="pending",
            response_termination=status,
            structured_output_requested=True,
            structured_output_used=response.structured_output,
            provider_metadata=_safe_provider_metadata(response.provider_metadata),
            token_accounting=accounting,
        )
        call_logs.append(entry)
        return response, None, entry

    def _run_tests(
        self,
        budget: BudgetTracker,
        call_logs: list[dict[str, Any]],
        run_id: str,
        experiment_id: str | None,
        task_id: str | None,
        attempt: int,
        recovery_action: str | None,
    ) -> dict[str, Any]:
        if not self.workspace.allow_tests:
            result = {
                "passed": False,
                "error": "Test execution is disabled; pass --allow-tests to explicitly enable it.",
            }
            call_logs.append(
                _call_log(
                    "test",
                    run_id,
                    experiment_id,
                    task_id,
                    self.config.strategy.value,
                    attempt,
                    len(call_logs) + 1,
                    getattr(self.model, "model", None),
                    recovery_action,
                    "disabled",
                    latency_seconds=None,
                    validation_outcome="not_applicable",
                    passed=False,
                )
            )
            return result
        if not budget.begin_test_run():
            reason = budget.stop_reason or "Test run unavailable."
            result = {
                "passed": False,
                "error": reason,
            }
            call_logs.append(
                _call_log(
                    "test",
                    run_id,
                    experiment_id,
                    task_id,
                    self.config.strategy.value,
                    attempt,
                    len(call_logs) + 1,
                    getattr(self.model, "model", None),
                    recovery_action,
                    "budget_rejected",
                    latency_seconds=None,
                    validation_outcome="not_applicable",
                    passed=False,
                    error=reason,
                )
            )
            return result
        started = time.monotonic()
        try:
            result = self.workspace.run_tests(
                min(
                    self.config.budgets.command_timeout_seconds,
                    budget.remaining_seconds,
                )
            )
        except WorkspaceError as exc:
            test_result = {"passed": False, "error": str(exc)}
            status = "failed"
        else:
            test_result = {"passed": result.succeeded, **result.to_dict()}
            status = "timed_out" if result.timed_out else "completed"
        call_logs.append(
            _call_log(
                "test",
                run_id,
                experiment_id,
                task_id,
                self.config.strategy.value,
                attempt,
                len(call_logs) + 1,
                getattr(self.model, "model", None),
                recovery_action,
                status,
                latency_seconds=time.monotonic() - started,
                validation_outcome="not_applicable",
                passed=test_result.get("passed") is True,
                returncode=test_result.get("returncode"),
                timed_out=test_result.get("timed_out", False),
            )
        )
        return test_result

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
        self,
        repair_number: int,
        failure_details: str,
        budget: BudgetTracker,
        *,
        timeout_replan: bool = False,
    ) -> str:
        if repair_number == 0:
            return "This is the initial attempt. Inspect the workspace before editing."
        if self.config.strategy is RecoveryStrategy.FAILURE_GUIDED:
            if timeout_replan:
                return (
                    "This is a bounded timeout replan. Do not repeat the timed-out "
                    "operation unchanged. Reinspect the current workspace and choose "
                    "a smaller alternative action that can finish and be verified "
                    "within the remaining budget.\n"
                    f"Prior failure assessment:\n{failure_details}\n"
                    f"Remaining budget: {json.dumps(budget.to_dict(include_events=False))}"
                )
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
            "claim completion unless you are ready for the configured tests. "
            "Use only these tool-specific arguments: list_files accepts optional "
            "path; read_file requires path; write_file requires path and content; "
            "finish accepts an empty object; run_command, when enabled, requires "
            "a command string array and accepts optional cwd. Do not add fields.\n\n"
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


def _call_log(
    call_type: str,
    run_id: str,
    experiment_id: str | None,
    task_id: str | None,
    strategy: str,
    attempt: int,
    sequence: int,
    model: Any,
    recovery_action: str | None,
    status: str,
    *,
    latency_seconds: float | None,
    validation_outcome: str,
    **details: Any,
) -> dict[str, Any]:
    return {
        "call_id": f"{run_id}:{sequence}",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "call_type": call_type,
        "run_id": run_id,
        "experiment_id": experiment_id,
        "task_id": task_id,
        "strategy": strategy,
        "attempt": attempt,
        "model": model if isinstance(model, str) else None,
        "recovery_action": recovery_action,
        "status": status,
        "latency_seconds": (
            round(latency_seconds, 6) if latency_seconds is not None else None
        ),
        "validation_outcome": validation_outcome,
        **details,
    }


def _safe_provider_metadata(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    safe: dict[str, Any] = {}
    for key, value in list(metadata.items())[:32]:
        if not isinstance(key, str):
            continue
        if isinstance(value, str):
            safe[key[:64]] = value[:256]
        elif isinstance(value, bool) or value is None:
            safe[key[:64]] = value
        elif isinstance(value, int):
            safe[key[:64]] = value
        elif isinstance(value, float) and math.isfinite(value):
            safe[key[:64]] = value
    return safe


def _termination_status(response: ModelResponse) -> str:
    reason = response.done_reason.casefold() if response.done_reason else ""
    if reason in {"length", "max_tokens", "max_output_tokens", "truncated"}:
        return "truncated"
    if response.done is False:
        return "incomplete"
    if response.done is True or reason in {"stop", "complete", "completed"}:
        return "complete"
    return "unknown"


def _response_termination_rejection(
    response: ModelResponse,
) -> tuple[str, FailureCategory] | None:
    status = _termination_status(response)
    if status == "complete":
        return None
    if status == "truncated":
        return (
            "Provider-confirmed truncated response: provider termination indicates an output length limit.",
            FailureCategory.TRUNCATED_RESPONSE,
        )
    if response.done is False:
        return (
            "Provider marked the response incomplete (done=false); the termination reason does not confirm length truncation.",
            FailureCategory.INCOMPLETE_RESPONSE,
        )
    return (
        "Provider termination status is unknown; response is treated as incomplete.",
        FailureCategory.INCOMPLETE_RESPONSE,
    )


def _response_validation_message(
    error: ResponseValidationError, response: ModelResponse
) -> str:
    status = _termination_status(response)
    if status == "truncated":
        return f"Provider-confirmed truncated response: {error}"
    if status != "complete":
        reason = (
            "Provider marked the response incomplete (done=false)"
            if response.done is False
            else "Provider termination status is unknown"
        )
        return f"{reason}; response may be incomplete: {error}"
    if error.phase == "schema":
        return f"Response schema validation failed: {error}"
    return f"Malformed response: {error}"


def _response_failure_category(
    error: ResponseValidationError, response: ModelResponse
) -> FailureCategory:
    status = _termination_status(response)
    if status == "truncated":
        return FailureCategory.TRUNCATED_RESPONSE
    if status != "complete":
        return FailureCategory.INCOMPLETE_RESPONSE
    if error.phase == "schema":
        return FailureCategory.SCHEMA_VALIDATION
    return FailureCategory.MALFORMED_RESPONSE


def _test_failure(test_result: dict[str, Any]) -> str:
    if test_result.get("passed") is True:
        return ""
    error = test_result.get("error")
    stdout = test_result.get("stdout", "")
    stderr = test_result.get("stderr", "")
    parts = [str(part) for part in (error, stdout, stderr) if part]
    return "\n".join(parts)[-8000:]
