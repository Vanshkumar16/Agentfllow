"""Agent workflow orchestration."""

from .budget import BudgetLimits
from .coding_agent import CodingAgent, CodingAgentConfig, CodingRunResult, RecoveryStrategy
from .schemas import RunResult, TaskPlan, TaskRequest

__all__ = [
    "AgentController",
    "BudgetLimits",
    "CodingAgent",
    "CodingAgentConfig",
    "CodingRunResult",
    "RecoveryStrategy",
    "RunResult",
    "TaskPlan",
    "TaskRequest",
]


def __getattr__(name: str):
    if name == "AgentController":
        from .controller import AgentController

        return AgentController
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
