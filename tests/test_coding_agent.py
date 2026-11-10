import json
import sys

import pytest

from agentflow.budget import BudgetLimits
from agentflow.coding_agent import CodingAgent, CodingAgentConfig, RecoveryStrategy
from agentflow.failure import FailureCategory
from agentflow.local_model import ModelResponse
from agentflow.workspace import CommandResult, WorkspaceTools


class FakeModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts = []
        self.output_limits = []

    async def complete(self, prompt, max_output_tokens, timeout):
        self.prompts.append(prompt)
        self.output_limits.append(max_output_tokens)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        if isinstance(response, ModelResponse):
            return response
        return ModelResponse(response, input_tokens=12, output_tokens=8)


def _action(tool, arguments=None):
    return json.dumps(
        {"action": {"tool": tool, "arguments": arguments or {}}}
    )


@pytest.mark.asyncio
async def test_failure_guided_strategy_repairs_after_test_diagnostics(tmp_path):
    (tmp_path / "solution.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    (tmp_path / "test_solution.py").write_text(
        "from solution import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )
    model = FakeModel(
        [
            _action("finish"),
            _action("write_file", {"path": "solution.py", "content": "def add(a, b):\n    return a + b\n"}),
            _action("finish"),
        ]
    )
    workspace = WorkspaceTools(
        tmp_path, allow_tests=True, test_command=(sys.executable, "-m", "pytest", "-q")
    )

    result = await CodingAgent(
        model,
        workspace,
        CodingAgentConfig(
            strategy=RecoveryStrategy.FAILURE_GUIDED,
            budgets=BudgetLimits(max_model_calls=5, max_tool_calls=4, max_test_runs=2),
            max_actions_per_attempt=2,
        ),
    ).run("Fix add so it returns the sum.")

    assert result.resolved is True
    assert result.repair_attempts == 1
    assert result.budget["model_calls"] == 3
    assert result.budget["test_runs"] == 2
    assert "assert add(2, 3) == 5" in model.prompts[1]


@pytest.mark.asyncio
async def test_single_shot_never_retries(tmp_path):
    (tmp_path / "solution.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    (tmp_path / "test_solution.py").write_text(
        "from solution import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )
    model = FakeModel(
        [
            json.dumps(
                {
                    "complete": True,
                    "patches": [
                        {
                            "path": "solution.py",
                            "content": "def add(a, b):\n    return a - b\n",
                        }
                    ],
                }
            )
        ]
    )
    workspace = WorkspaceTools(
        tmp_path, allow_tests=True, test_command=(sys.executable, "-m", "pytest", "-q")
    )

    result = await CodingAgent(
        model,
        workspace,
        CodingAgentConfig(
            strategy=RecoveryStrategy.SINGLE_SHOT,
            budgets=BudgetLimits(max_test_runs=1),
        ),
    ).run("Fix the failing test.")

    assert result.resolved is False
    assert result.repair_attempts == 0
    assert len(model.prompts) == 1


@pytest.mark.asyncio
async def test_single_shot_applies_patch_with_one_model_generation(tmp_path):
    (tmp_path / "solution.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    (tmp_path / "test_solution.py").write_text(
        "from solution import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )
    model = FakeModel(
        [
            json.dumps(
                {
                    "complete": True,
                    "patches": [
                        {
                            "path": "solution.py",
                            "content": "def add(a, b):\n    return a + b\n",
                        }
                    ],
                }
            )
        ]
    )
    workspace = WorkspaceTools(
        tmp_path, allow_tests=True, test_command=(sys.executable, "-m", "pytest", "-q")
    )

    result = await CodingAgent(
        model,
        workspace,
        CodingAgentConfig(
            strategy=RecoveryStrategy.SINGLE_SHOT,
            budgets=BudgetLimits(max_model_calls=1, max_tool_calls=2, max_test_runs=1),
        ),
    ).run("Fix add so it returns the sum.")

    assert result.resolved is True
    assert result.budget["model_calls"] == 1
    assert result.budget["tool_calls"] == 2
    assert "return a - b" in model.prompts[0]


@pytest.mark.asyncio
async def test_fixed_retry_retries_without_failure_diagnostics(tmp_path):
    (tmp_path / "solution.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    (tmp_path / "test_solution.py").write_text(
        "from solution import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )
    model = FakeModel(
        [
            _action("finish"),
            _action("write_file", {"path": "solution.py", "content": "def add(a, b):\n    return a + b\n"}),
            _action("finish"),
        ]
    )
    workspace = WorkspaceTools(
        tmp_path, allow_tests=True, test_command=(sys.executable, "-m", "pytest", "-q")
    )

    result = await CodingAgent(
        model,
        workspace,
        CodingAgentConfig(
            strategy=RecoveryStrategy.FIXED_RETRY,
            budgets=BudgetLimits(max_model_calls=5, max_tool_calls=4, max_test_runs=2),
            max_actions_per_attempt=2,
        ),
    ).run("Fix add so it returns the sum.")

    assert result.resolved is True
    assert result.repair_attempts == 1
    assert "AssertionError" not in model.prompts[1]


@pytest.mark.asyncio
async def test_measured_token_overrun_prevents_action_execution(tmp_path, monkeypatch):
    monkeypatch.setattr("agentflow.budget.estimate_prompt_tokens", lambda prompt: 1)
    model = FakeModel(
        [
            ModelResponse(
                _action("write_file", {"path": "new.py", "content": "value = 1"}),
                input_tokens=8,
                output_tokens=4,
            )
        ]
    )
    workspace = WorkspaceTools(tmp_path, allow_tests=False)

    result = await CodingAgent(
        model,
        workspace,
        CodingAgentConfig(
            budgets=BudgetLimits(max_tokens=10, max_model_calls=2),
            max_actions_per_attempt=1,
        ),
    ).run("Create new.py.")

    assert result.resolved is False
    assert not (tmp_path / "new.py").exists()
    assert result.budget["tokens_used"] == 12
    assert "token budget exceeded" in result.budget["stop_reason"]
    assert any(
        event["event"] == "token_overrun"
        for event in result.budget["budget_events"]
    )


@pytest.mark.asyncio
async def test_prompt_budget_rejection_happens_before_model_call(tmp_path):
    model = FakeModel([])
    workspace = WorkspaceTools(tmp_path, allow_tests=False)

    result = await CodingAgent(
        model,
        workspace,
        CodingAgentConfig(budgets=BudgetLimits(max_tokens=10)),
    ).run("Fix the task.")

    assert model.prompts == []
    assert result.budget["model_calls"] == 0
    assert result.budget["stop_reason"] == (
        "estimated prompt leaves no token budget for model output"
    )
    assert result.budget["budget_events"][-1]["event"] == "budget_rejected"


@pytest.mark.asyncio
async def test_malformed_response_gets_controlled_regeneration_guidance(tmp_path):
    (tmp_path / "solution.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (tmp_path / "test_solution.py").write_text(
        "from solution import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )
    model = FakeModel(["not json", _action("finish")])
    workspace = WorkspaceTools(
        tmp_path, allow_tests=True, test_command=(sys.executable, "-m", "pytest", "-q")
    )

    result = await CodingAgent(
        model,
        workspace,
        CodingAgentConfig(
            strategy=RecoveryStrategy.FAILURE_GUIDED,
            budgets=BudgetLimits(max_model_calls=4, max_test_runs=2),
            max_actions_per_attempt=1,
        ),
    ).run("Confirm add returns the sum.")

    assert result.resolved is True
    assert result.attempts[0].failure_category == FailureCategory.MALFORMED_RESPONSE.value
    assert "controlled regeneration" in result.attempts[0].recovery_action
    assert "Failure category: empty_or_malformed_response" in model.prompts[1]
    assert "exactly the required JSON schema" in model.prompts[1]


@pytest.mark.asyncio
async def test_failure_guided_stops_when_the_same_failure_repeats(tmp_path):
    (tmp_path / "solution.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    (tmp_path / "test_solution.py").write_text(
        "from solution import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )
    model = FakeModel([_action("finish"), _action("finish"), _action("finish")])
    workspace = WorkspaceTools(
        tmp_path, allow_tests=True, test_command=(sys.executable, "-m", "pytest", "-q")
    )

    result = await CodingAgent(
        model,
        workspace,
        CodingAgentConfig(
            strategy=RecoveryStrategy.FAILURE_GUIDED,
            budgets=BudgetLimits(max_model_calls=6, max_test_runs=3),
            max_actions_per_attempt=1,
        ),
    ).run("Fix add so it returns the sum.")

    assert result.resolved is False
    assert len(result.attempts) == 2
    assert result.budget["model_calls"] == 2
    assert result.attempts[0].failure_category == FailureCategory.TEST_ASSERTION.value
    assert result.attempts[1].failure_category == FailureCategory.REPEATED_IDENTICAL.value
    assert result.attempts[1].recovery_decision == "stop: repeated identical failure"
    assert "Do not repeat the same repair" in result.attempts[1].recovery_action
    serialized = result.to_dict()["attempts"][1]
    assert serialized["failure_category"] == FailureCategory.REPEATED_IDENTICAL.value
    assert serialized["recovery_action"] == result.attempts[1].recovery_action


@pytest.mark.asyncio
async def test_failure_guided_stops_on_test_timeout(tmp_path, monkeypatch):
    model = FakeModel([_action("finish"), _action("finish")])
    workspace = WorkspaceTools(tmp_path, allow_tests=True)
    monkeypatch.setattr(
        workspace,
        "run_tests",
        lambda timeout: CommandResult(
            ["pytest"], None, "", "test command timed out", timed_out=True
        ),
    )

    result = await CodingAgent(
        model,
        workspace,
        CodingAgentConfig(
            strategy=RecoveryStrategy.FAILURE_GUIDED,
            budgets=BudgetLimits(max_model_calls=4, max_test_runs=2),
            max_actions_per_attempt=1,
        ),
    ).run("Inspect and fix the slow failing test.")

    assert result.resolved is False
    assert len(result.attempts) == 1
    assert result.attempts[0].failure_category == FailureCategory.TIMEOUT.value
    assert result.attempts[0].recovery_decision == (
        "stop: timeout requires a bounded replan"
    )


@pytest.mark.asyncio
async def test_failure_guided_does_not_run_tests_after_model_timeout(tmp_path, monkeypatch):
    model = FakeModel([TimeoutError("model request timed out")])
    workspace = WorkspaceTools(tmp_path, allow_tests=True)
    monkeypatch.setattr(
        workspace,
        "run_tests",
        lambda timeout: pytest.fail("Tests must not run after the model times out."),
    )

    result = await CodingAgent(
        model,
        workspace,
        CodingAgentConfig(strategy=RecoveryStrategy.FAILURE_GUIDED),
    ).run("Fix the task.")

    assert result.resolved is False
    assert result.attempts[0].failure_category == FailureCategory.TIMEOUT.value
    assert result.attempts[0].recovery_decision == (
        "stop: timeout requires a bounded replan"
    )
    assert "tests were not run" in result.attempts[0].test["error"]
