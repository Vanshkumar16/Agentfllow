import json
import sys

import pytest

from agentflow.budget import BudgetLimits
from agentflow.coding_agent import CodingAgent, CodingAgentConfig, RecoveryStrategy
from agentflow.local_model import ModelResponse
from agentflow.workspace import WorkspaceTools


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
async def test_measured_token_overrun_prevents_action_execution(tmp_path):
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
