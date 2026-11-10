"""Per-task resource accounting and hard limits."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from math import isfinite


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


class BudgetTracker:
    def __init__(self, limits: BudgetLimits) -> None:
        self.limits = limits
        self.started_at = time.monotonic()
        self.model_calls = 0
        self.tool_calls = 0
        self.test_runs = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.token_usage_available = True
        self.stop_reason: str | None = None

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.limits.task_timeout_seconds - self.elapsed_seconds)

    @property
    def tokens_used(self) -> int | None:
        if not self.token_usage_available:
            return None
        return self.input_tokens + self.output_tokens

    @property
    def remaining_tokens(self) -> int | None:
        if self.tokens_used is None:
            return None
        return max(0, self.limits.max_tokens - self.tokens_used)

    def _time_available(self) -> bool:
        if self.remaining_seconds <= 0:
            self.stop_reason = "task time budget exhausted"
            return False
        return True

    def begin_model_call(self) -> int | None:
        if self.model_calls >= self.limits.max_model_calls:
            self.stop_reason = "model-call budget exhausted"
            return None
        if not self._time_available():
            return None
        remaining = self.remaining_tokens
        if remaining is not None and remaining <= 0:
            self.stop_reason = "token budget exhausted"
            return None
        self.model_calls += 1
        return min(
            self.limits.max_output_tokens_per_call,
            remaining if remaining is not None else self.limits.max_output_tokens_per_call,
        )

    def record_model_usage(
        self, input_tokens: int | None, output_tokens: int | None
    ) -> None:
        if input_tokens is None or output_tokens is None:
            self.token_usage_available = False
            return
        if input_tokens < 0 or output_tokens < 0:
            self.token_usage_available = False
            return
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        if self.input_tokens + self.output_tokens > self.limits.max_tokens:
            self.stop_reason = "token budget exceeded by the completed model call"

    def begin_tool_call(self) -> bool:
        if self.tool_calls >= self.limits.max_tool_calls:
            self.stop_reason = "tool-call budget exhausted"
            return False
        if not self._time_available():
            return False
        self.tool_calls += 1
        return True

    def begin_test_run(self) -> bool:
        if self.test_runs >= self.limits.max_test_runs:
            self.stop_reason = "test-run budget exhausted"
            return False
        if not self._time_available():
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
        remaining_tokens = self.remaining_tokens
        if remaining_tokens is not None and remaining_tokens < minimum_tokens:
            return False, "remaining token budget is below the repair threshold"
        if self.stop_reason is not None:
            return False, self.stop_reason
        return True, "repair fits the remaining budget"

    def to_dict(self) -> dict[str, int | float | str | None]:
        data = {
            "model_calls": self.model_calls,
            "tool_calls": self.tool_calls,
            "test_runs": self.test_runs,
            "input_tokens": self.input_tokens if self.token_usage_available else None,
            "output_tokens": self.output_tokens if self.token_usage_available else None,
            "tokens_used": self.tokens_used,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "remaining_seconds": round(self.remaining_seconds, 3),
            "stop_reason": self.stop_reason,
        }
        return data

    def limits_dict(self) -> dict[str, int | float]:
        return asdict(self.limits)
