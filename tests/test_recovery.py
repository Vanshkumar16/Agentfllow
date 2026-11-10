from agentflow.executor import Executor
from agentflow.recovery import Recovery
from agentflow.schemas import PlanStep, StepResult
from agentflow.tools import ToolRegistry


def test_recovery_retries_failed_step_and_preserves_successful_step():
    registry = ToolRegistry()
    executor = Executor(registry)
    recovery = Recovery(executor, retries_per_step=2)
    steps = [
        PlanStep("bad", "Use an unknown tool", "not-allowed", {}),
        PlanStep("good", "Echo text", "echo", {"text": "hello"}),
    ]
    results = [
        StepResult("bad", "not-allowed", error="ToolError: not allowed"),
        StepResult("good", "echo", output={"text": "hello"}),
    ]

    recovered = recovery.recover(steps, results)

    assert recovered[0].attempts == 3
    assert recovered[0].succeeded is False
    assert recovered[1] == results[1]
