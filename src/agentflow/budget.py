"""Per-task resource accounting and hard limits."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from math import ceil, isfinite
from typing import Any


_LOGGER = logging.getLogger(__name__)


def estimate_prompt_tokens(prompt: str) -> int:
    """Conservatively estimate tokens when the local runtime has no tokenizer API."""
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string.")
    if not prompt:
        return 0
    return max(1, ceil(len(prompt) / 2), ceil(len(prompt.encode("utf-8")) / 3))


@dataclass(frozen=True)
class BudgetLimits:
    max_model_calls: int = 16
    max_tool_calls: int = 24
    max_test_runs: int = 4
    max_tokens: int = 8192
    max_repair_attempts: int = 3
    task_timeout_seconds: float = 120.0
    model_timeout_seconds: float = 60.0
    command_timeout_seconds: float = 30.0
    max_output_tokens_per_call: int = 1024
    min_repair_tokens: int = 256
    min_repair_seconds: float = 10.0

    def __post_init__(self) -> None:
        integer_fields = (
            "max_model_calls",
            "max_tool_calls",
            "max_test_runs",
            "max_tokens",
            "max_output_tokens_per_call",
            "max_repair_attempts",
            "min_repair_tokens",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer.")
        for name in integer_fields[:5]:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1.")
        if not 0 <= self.max_repair_attempts <= 3:
            raise ValueError("max_repair_attempts must be between 0 and 3.")
        for name in (
            "task_timeout_seconds",
            "model_timeout_seconds",
            "command_timeout_seconds",
            "min_repair_seconds",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not isfinite(value):
                raise ValueError(f"{name} must be a finite number.")
        if self.task_timeout_seconds <= 0 or self.model_timeout_seconds <= 0 or self.command_timeout_seconds <= 0:
            raise ValueError("Timeouts must be positive.")
        if self.min_repair_tokens < 1 or self.min_repair_seconds < 0:
            raise ValueError("Repair thresholds must be non-negative and tokens positive.")


@dataclass(frozen=True)
class ModelCallReservation:
    estimated_input_tokens: int
    max_output_tokens: int

    @property
    def reserved_tokens(self) -> int:
        return self.estimated_input_tokens + self.max_output_tokens


class BudgetTracker:
    def __init__(self, limits: BudgetLimits) -> None:
        self.limits = limits
        self.started_at = time.monotonic()
        self.model_calls = 0
        self.tool_calls = 0
        self.test_runs = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self._input_usage_available = True
        self._output_usage_available = True
        self.estimated_input_tokens = 0
        self._charged_tokens = 0
        self._reservation: ModelCallReservation | None = None
        self.budget_events: list[dict[str, Any]] = []
        self.stop_reason: str | None = None

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.limits.task_timeout_seconds - self.elapsed_seconds)

    @property
    def tokens_used(self) -> int | None:
        if not self._input_usage_available or not self._output_usage_available:
            return None
        return self.input_tokens + self.output_tokens

    @property
    def remaining_tokens(self) -> int:
        reserved_tokens = (
            self._reservation.reserved_tokens if self._reservation is not None else 0
        )
        return max(0, self.limits.max_tokens - self._charged_tokens - reserved_tokens)

    @property
    def reserved_output_tokens(self) -> int:
        if self._reservation is None:
            return 0
        return self._reservation.max_output_tokens

    def _record_event(self, event: str, *, log_level: int | None = None, **details: Any) -> None:
        entry = {
            "event": event,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            **details,
        }
        self.budget_events.append(entry)
        if log_level is not None:
            _LOGGER.log(
                log_level,
                "Budget event: %s",
                event,
                extra={"budget_event": entry},
            )

    def _reject_budget(
        self,
        reason: str,
        *,
        budget_kind: str,
        estimated_prompt_tokens: int | None = None,
    ) -> None:
        if self.stop_reason is None:
            self.stop_reason = reason
        self._record_event(
            "budget_rejected",
            log_level=logging.WARNING,
            budget_kind=budget_kind,
            reason=reason,
            estimated_prompt_tokens=estimated_prompt_tokens,
            remaining_tokens=self.remaining_tokens,
            model_calls=self.model_calls,
        )

    def _time_available(self) -> bool:
        if self.remaining_seconds <= 0:
            self.stop_reason = "task time budget exhausted"
            return False
        return True

    def begin_model_call(self, prompt: str = "") -> int | None:
        if self._reservation is not None:
            raise RuntimeError("The previous model-call reservation has not been recorded.")
        estimated_prompt = estimate_prompt_tokens(prompt)
        if self.stop_reason is not None:
            self._reject_budget(
                self.stop_reason,
                budget_kind="model_call",
                estimated_prompt_tokens=estimated_prompt,
            )
            return None
        if self.model_calls >= self.limits.max_model_calls:
            self._reject_budget(
                "model-call budget exhausted",
                budget_kind="model_call",
                estimated_prompt_tokens=estimated_prompt,
            )
            return None
        if not self._time_available():
            self._reject_budget(
                self.stop_reason or "task time budget exhausted",
                budget_kind="task_time",
                estimated_prompt_tokens=estimated_prompt,
            )
            return None

        available_tokens = self.remaining_tokens - estimated_prompt
        if available_tokens < 1:
            self._reject_budget(
                "estimated prompt leaves no token budget for model output",
                budget_kind="tokens",
                estimated_prompt_tokens=estimated_prompt,
            )
            return None

        max_output_tokens = min(
            self.limits.max_output_tokens_per_call,
            available_tokens,
        )
        remaining_after_reservation = available_tokens - max_output_tokens
        self.model_calls += 1
        self.estimated_input_tokens += estimated_prompt
        self._reservation = ModelCallReservation(
            estimated_input_tokens=estimated_prompt,
            max_output_tokens=max_output_tokens,
        )
        self._record_event(
            "model_call_reserved",
            call=self.model_calls,
            estimated_input_tokens=estimated_prompt,
            reserved_output_tokens=max_output_tokens,
            remaining_tokens=remaining_after_reservation,
        )
        return max_output_tokens

    def record_model_usage(
        self, input_tokens: int | None, output_tokens: int | None
    ) -> None:
        reservation = self._reservation
        if reservation is None:
            raise RuntimeError("No model-call token reservation is pending.")
        if input_tokens is not None and (
            not isinstance(input_tokens, int)
            or isinstance(input_tokens, bool)
            or input_tokens < 0
        ):
            input_tokens = None
        if output_tokens is not None and (
            not isinstance(output_tokens, int)
            or isinstance(output_tokens, bool)
            or output_tokens < 0
        ):
            output_tokens = None

        if input_tokens is None:
            self._input_usage_available = False
        else:
            self.input_tokens += input_tokens
        if output_tokens is None:
            self._output_usage_available = False
        else:
            self.output_tokens += output_tokens

        charged_input = (
            input_tokens
            if input_tokens is not None
            else reservation.estimated_input_tokens
        )
        charged_output = (
            output_tokens
            if output_tokens is not None
            else reservation.max_output_tokens
        )
        charged_tokens = charged_input + charged_output
        self._charged_tokens += charged_tokens
        self._reservation = None

        actual_input_for_event = input_tokens
        actual_output_for_event = output_tokens
        actual_total = (
            input_tokens + output_tokens
            if input_tokens is not None and output_tokens is not None
            else None
        )
        reservation_overrun = (
            actual_total is not None
            and actual_total > reservation.reserved_tokens
        )
        prompt_estimate_overrun = (
            input_tokens is not None
            and input_tokens > reservation.estimated_input_tokens
        )
        if prompt_estimate_overrun:
            self._record_event(
                "token_overrun",
                log_level=logging.WARNING,
                reason="actual input tokens exceeded the prompt estimate",
                estimated_input_tokens=reservation.estimated_input_tokens,
                actual_input_tokens=actual_input_for_event,
                actual_output_tokens=actual_output_for_event,
            )
        if reservation_overrun:
            self._record_event(
                "token_overrun",
                log_level=logging.WARNING,
                reason="actual call usage exceeded its prompt and output reservation",
                reserved_tokens=reservation.reserved_tokens,
                actual_tokens=actual_total,
            )
        if self._charged_tokens > self.limits.max_tokens:
            self.stop_reason = "token budget exceeded by the completed model call"
            self._record_event(
                "token_overrun",
                log_level=logging.WARNING,
                reason=self.stop_reason,
                max_tokens=self.limits.max_tokens,
                charged_tokens=self._charged_tokens,
                actual_input_tokens=actual_input_for_event,
                actual_output_tokens=actual_output_for_event,
            )

    def begin_tool_call(self) -> bool:
        if self.tool_calls >= self.limits.max_tool_calls:
            self._reject_budget(
                "tool-call budget exhausted", budget_kind="tool_calls"
            )
            return False
        if not self._time_available():
            self._reject_budget(
                self.stop_reason or "task time budget exhausted",
                budget_kind="task_time",
            )
            return False
        self.tool_calls += 1
        return True

    def begin_test_run(self) -> bool:
        if self.test_runs >= self.limits.max_test_runs:
            self._reject_budget(
                "test-run budget exhausted", budget_kind="test_runs"
            )
            return False
        if not self._time_available():
            self._reject_budget(
                self.stop_reason or "task time budget exhausted",
                budget_kind="task_time",
            )
            return False
        self.test_runs += 1
        return True

    def can_repair(self, *, minimum_tokens: int, minimum_seconds: float) -> tuple[bool, str]:
        if self.model_calls >= self.limits.max_model_calls:
            return False, "model-call budget leaves no room for a repair"
        if self.tool_calls >= self.limits.max_tool_calls:
            return False, "tool-call budget leaves no room for a repair"
        if self.test_runs >= self.limits.max_test_runs:
            return False, "test-run budget leaves no room to validate a repair"
        if self.remaining_seconds < minimum_seconds:
            return False, "remaining time is below the repair threshold"
        if self.remaining_tokens < minimum_tokens:
            return False, "remaining token budget is below the repair threshold"
        if self.stop_reason is not None:
            return False, self.stop_reason
        return True, "repair fits the remaining budget"

    def to_dict(self, *, include_events: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "model_calls": self.model_calls,
            "tool_calls": self.tool_calls,
            "test_runs": self.test_runs,
            "input_tokens": self.input_tokens if self._input_usage_available else None,
            "output_tokens": self.output_tokens if self._output_usage_available else None,
            "total_tokens": self.tokens_used,
            "tokens_used": self.tokens_used,
            "estimated_input_tokens": self.estimated_input_tokens,
            "budget_tokens_charged": self._charged_tokens,
            "reserved_output_tokens": self.reserved_output_tokens,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "remaining_seconds": round(self.remaining_seconds, 3),
            "remaining_tokens": self.remaining_tokens,
            "stop_reason": self.stop_reason,
        }
        if include_events:
            data["budget_events"] = list(self.budget_events)
        return data

    def limits_dict(self) -> dict[str, int | float]:
        return asdict(self.limits)
