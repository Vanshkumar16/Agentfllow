"""Typed data structures shared by the agent workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class TaskRequest:
    goal: str

    def __post_init__(self) -> None:
        if not self.goal.strip():
            raise ValueError("Task goal must not be empty.")


@dataclass(frozen=True)
class PlanStep:
    id: str
    description: str
    tool: str
    input: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskPlan:
    steps: list[PlanStep]


@dataclass(frozen=True)
class StepResult:
    step_id: str
    tool: str
    output: Any = None
    error: str | None = None
    attempts: int = 1

    @property
    def succeeded(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class TestReport:
    passed: bool
    checks: list[str]


@dataclass(frozen=True)
class Critique:
    complete: bool
    feedback: str


@dataclass(frozen=True)
class IterationResult:
    plan: TaskPlan
    steps: list[StepResult]
    tests: TestReport
    critique: Critique


@dataclass(frozen=True)
class RunResult:
    goal: str
    complete: bool
    iterations: list[IterationResult]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
