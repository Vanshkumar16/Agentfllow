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
        return ModelResponse(response, input_tokens=12, output_tokens=8, done=True)


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
    assert result.attempts[0].recovery_plan["decision"] == "REVISE"
    assert result.patch_valid is True


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
    assert result.attempts[0].recovery_plan["decision"] == "RETRY"


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
    model = FakeModel(
        [
            ModelResponse("not json", input_tokens=12, output_tokens=8, done=True),
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
            budgets=BudgetLimits(max_model_calls=4, max_test_runs=2),
            max_actions_per_attempt=1,
        ),
    ).run("Confirm add returns the sum.")

    assert result.resolved is True
    assert result.attempts[0].failure_category == FailureCategory.MALFORMED_RESPONSE.value
    assert "controlled regeneration" in result.attempts[0].recovery_action
    assert "Failure category: empty_or_malformed_response" in model.prompts[1]
    assert "exactly the required JSON schema" in model.prompts[1]
    model_log = result.per_call_logs[0]
    assert model_log["response_termination"] == "complete"
    assert model_log["validation_outcome"] == "invalid"
    assert model_log["experiment_id"] is None
    assert model_log["task_id"] is None
    assert result.attempts[0].recovery_plan["decision"] == "RETRY"


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
    assert result.attempts[1].recovery_plan["decision"] == "STOP"
    assert "Do not repeat the same repair" in result.attempts[1].recovery_action
    serialized = result.to_dict()["attempts"][1]
    assert serialized["failure_category"] == FailureCategory.REPEATED_IDENTICAL.value
    assert serialized["recovery_action"] == result.attempts[1].recovery_action


@pytest.mark.asyncio
async def test_failure_guided_replans_after_test_timeout(tmp_path, monkeypatch):
    model = FakeModel(
        [
            _action("finish"),
            _action("write_file", {"path": "solution.py", "content": "value = 1"}),
            _action("finish"),
        ]
    )
    workspace = WorkspaceTools(tmp_path, allow_tests=True)
    test_results = iter(
        [
            CommandResult(
                ["pytest"], None, "", "test command timed out", timed_out=True
            ),
            CommandResult(["pytest"], 0, "1 passed", ""),
        ]
    )
    monkeypatch.setattr(
        workspace,
        "run_tests",
        lambda timeout: next(test_results),
    )

    result = await CodingAgent(
        model,
        workspace,
        CodingAgentConfig(
            strategy=RecoveryStrategy.FAILURE_GUIDED,
            budgets=BudgetLimits(max_model_calls=5, max_test_runs=2),
            max_actions_per_attempt=2,
        ),
    ).run("Inspect and fix the slow failing test.")

    assert result.resolved is True
    assert len(result.attempts) == 2
    assert result.attempts[0].failure_category == FailureCategory.TIMEOUT.value
    assert result.attempts[0].recovery_plan["decision"] == "REPLAN"
    assert "bounded timeout replan" in model.prompts[1]
    assert result.attempts[1].actions[0]["tool"] == "write_file"
    assert result.budget["model_calls"] == 3
    assert result.budget["test_runs"] == 2


@pytest.mark.asyncio
async def test_failure_guided_replans_after_model_timeout(tmp_path, monkeypatch):
    model = FakeModel(
        [TimeoutError("model request timed out"), _action("finish")]
    )
    workspace = WorkspaceTools(tmp_path, allow_tests=True)
    monkeypatch.setattr(
        workspace,
        "run_tests",
        lambda timeout: CommandResult(["pytest"], 0, "1 passed", ""),
    )

    result = await CodingAgent(
        model,
        workspace,
        CodingAgentConfig(
            strategy=RecoveryStrategy.FAILURE_GUIDED,
            budgets=BudgetLimits(max_model_calls=3, max_test_runs=2),
        ),
    ).run("Fix the task.")

    assert result.resolved is True
    assert len(result.attempts) == 2
    assert result.attempts[0].failure_category == FailureCategory.TIMEOUT.value
    assert result.attempts[0].recovery_plan["decision"] == "REPLAN"
    assert "bounded timeout replan" in model.prompts[1]
    assert "tests were not run" in result.attempts[0].test["error"]


@pytest.mark.asyncio
async def test_failure_guided_stops_timeout_replan_at_attempt_limit(tmp_path, monkeypatch):
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
        CodingAgentConfig(
            strategy=RecoveryStrategy.FAILURE_GUIDED,
            budgets=BudgetLimits(max_repair_attempts=0),
        ),
    ).run("Fix the task.")

    assert len(result.attempts) == 1
    assert result.budget["model_calls"] == 1
    assert result.attempts[0].recovery_plan["decision"] == "STOP"
    assert result.attempts[0].recovery_decision == "stop: repair-attempt limit reached"


@pytest.mark.asyncio
async def test_invalid_action_schema_is_rejected_before_tool_execution(tmp_path):
    model = FakeModel(
        [
            json.dumps(
                {
                    "action": {
                        "tool": "write_file",
                        "arguments": {
                            "path": "must-not-exist.py",
                            "content": "value = 1",
                            "unexpected": True,
                        },
                    }
                }
            )
        ]
    )
    result = await CodingAgent(
        model,
        WorkspaceTools(tmp_path),
        CodingAgentConfig(
            budgets=BudgetLimits(max_repair_attempts=0),
            max_actions_per_attempt=1,
        ),
    ).run("Do not apply malformed actions.")

    assert not (tmp_path / "must-not-exist.py").exists()
    assert result.patch_valid is False
    assert result.attempts[0].failure_category == FailureCategory.SCHEMA_VALIDATION.value
    assert result.per_call_logs[0]["validation_outcome"] == "invalid"
    assert result.attempts[0].recovery_plan["decision"] == "STOP"


@pytest.mark.asyncio
async def test_provider_truncation_is_distinguished_from_unknown_termination(tmp_path):
    truncated_model = FakeModel(
        [ModelResponse("not json", 12, 8, done=False, done_reason="length")]
    )
    truncated = await CodingAgent(
        truncated_model,
        WorkspaceTools(tmp_path),
        CodingAgentConfig(
            budgets=BudgetLimits(max_repair_attempts=0),
            max_actions_per_attempt=1,
        ),
    ).run("Handle a response that the provider says was cut off.")

    assert truncated.attempts[0].failure_category == FailureCategory.TRUNCATED_RESPONSE.value
    assert truncated.per_call_logs[0]["response_termination"] == "truncated"

    incomplete_model = FakeModel(
        [ModelResponse("not json", 12, 8, done=False)]
    )
    unknown = await CodingAgent(
        incomplete_model,
        WorkspaceTools(tmp_path),
        CodingAgentConfig(
            budgets=BudgetLimits(max_repair_attempts=0),
            max_actions_per_attempt=1,
        ),
    ).run("Handle invalid JSON with no termination metadata.")

    assert unknown.attempts[0].failure_category == FailureCategory.INCOMPLETE_RESPONSE.value
    assert unknown.per_call_logs[0]["response_termination"] == "incomplete"
    assert "truncated" not in unknown.attempts[0].failure.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("single_shot", [False, True])
async def test_valid_json_from_incomplete_provider_is_never_applied(tmp_path, single_shot):
    if single_shot:
        content = json.dumps(
            {
                "complete": True,
                "patches": [
                    {"path": "must-not-exist.py", "content": "value = 1"}
                ],
            }
        )
    else:
        content = _action(
            "write_file",
            {"path": "must-not-exist.py", "content": "value = 1"},
        )
    model = FakeModel(
        [
            ModelResponse(
                content,
                input_tokens=12,
                output_tokens=8,
                done=False,
                done_reason="stop",
            )
        ]
    )
    result = await CodingAgent(
        model,
        WorkspaceTools(tmp_path),
        CodingAgentConfig(
            strategy=(
                RecoveryStrategy.SINGLE_SHOT
                if single_shot
                else RecoveryStrategy.FAILURE_GUIDED
            ),
            budgets=BudgetLimits(max_repair_attempts=0),
            max_actions_per_attempt=1,
        ),
    ).run("Reject any payload from an incomplete provider response.")

    assert not (tmp_path / "must-not-exist.py").exists()
    assert result.resolved is False
    assert result.explicit_completion is False
    assert result.attempts[0].failure_category == FailureCategory.INCOMPLETE_RESPONSE.value
    model_log = next(call for call in result.per_call_logs if call["call_type"] == "model")
    assert model_log["validation_outcome"] == "invalid"
    assert model_log["validation_phase"] == "termination"
    assert model_log["failure_category"] == FailureCategory.INCOMPLETE_RESPONSE.value


@pytest.mark.asyncio
async def test_passing_tests_without_finish_are_not_strict_completion(tmp_path):
    (tmp_path / "solution.py").write_text(
        "def add(a, b):\n    return a + b\n", encoding="utf-8"
    )
    (tmp_path / "test_solution.py").write_text(
        "from solution import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )
    model = FakeModel([_action("read_file", {"path": "solution.py"})])
    workspace = WorkspaceTools(
        tmp_path, allow_tests=True, test_command=(sys.executable, "-m", "pytest", "-q")
    )

    result = await CodingAgent(
        model,
        workspace,
        CodingAgentConfig(
            budgets=BudgetLimits(
                max_repair_attempts=0,
                max_model_calls=1,
                max_test_runs=1,
            ),
            max_actions_per_attempt=1,
        ),
    ).run("Verify the existing implementation.", experiment_id="exp-1", task_id="dev-1")

    assert result.resolved is False
    assert result.to_dict()["strict_resolved"] is False
    assert result.final_test_passed is True
    assert result.explicit_completion is False
    assert "did not emit an explicit finish action" in result.completion_reason
    assert {row["call_type"] for row in result.per_call_logs} == {"model", "tool", "test"}
    assert all(row["experiment_id"] == "exp-1" for row in result.per_call_logs)
    assert all(row["task_id"] == "dev-1" for row in result.per_call_logs)
    assert result.per_call_logs[0]["token_accounting"]["usage_status"] == "measured"


@pytest.mark.asyncio
async def test_single_shot_validates_all_patches_before_writing_any(tmp_path):
    response = json.dumps(
        {
            "complete": True,
            "patches": [
                {"path": "first.py", "content": "value = 1"},
                {"path": "first.py", "content": "value = 2"},
            ],
        }
    )
    result = await CodingAgent(
        FakeModel([response]),
        WorkspaceTools(tmp_path),
        CodingAgentConfig(
            strategy=RecoveryStrategy.SINGLE_SHOT,
            budgets=BudgetLimits(max_repair_attempts=0),
        ),
    ).run("Reject duplicate patch paths.")

    assert not (tmp_path / "first.py").exists()
    assert result.patch_valid is False
    model_logs = [row for row in result.per_call_logs if row["call_type"] == "model"]
    assert model_logs[0]["validation_outcome"] == "invalid"
    assert result.explicit_completion is False
