import json
from itertools import islice

import pytest

from agentflow.budget import BudgetLimits
from agentflow.coding_agent import CodingAgent, CodingAgentConfig
from agentflow.failure import FailureCategory
from agentflow.local_model import ModelResponse
from agentflow.workspace import WorkspaceTools


class IncompleteWriteModel:
    model = "development-stub"

    def __init__(self, path):
        self.path = path

    async def complete(self, prompt, max_output_tokens, timeout):
        return ModelResponse(
            json.dumps(
                {
                    "action": {
                        "tool": "write_file",
                        "arguments": {
                            "path": self.path,
                            "content": "must not be applied",
                        },
                    }
                }
            ),
            input_tokens=1,
            output_tokens=1,
            done=False,
        )


def _pilot_tasks():
    with open("benchmarks/tasks.jsonl", encoding="utf-8") as tasks_file:
        rows = [json.loads(line) for line in islice(tasks_file, 5)]
    assert [row["id"] for row in rows] == [
        "py-fix-001",
        "py-fix-002",
        "py-fix-003",
        "py-fix-004",
        "py-fix-005",
    ]
    return rows


async def _check_pilot_fixture(task, workspace_root):
    for relative_path, content in task["files"].items():
        target = workspace_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    original = {
        relative_path: (workspace_root / relative_path).read_bytes()
        for relative_path in task["files"]
    }

    result = await CodingAgent(
        IncompleteWriteModel(next(iter(task["files"]))),
        WorkspaceTools(workspace_root),
        CodingAgentConfig(budgets=BudgetLimits(max_repair_attempts=0)),
    ).run(task["goal"], experiment_id="iteration-2-pilot-safety", task_id=task["id"])

    assert result.attempts[0].failure_category == FailureCategory.INCOMPLETE_RESPONSE.value
    assert result.attempts[0].actions == []
    assert all(
        (workspace_root / relative_path).read_bytes() == contents
        for relative_path, contents in original.items()
    )
    model_log = next(call for call in result.per_call_logs if call["call_type"] == "model")
    assert model_log["response_termination"] == "incomplete"
    assert model_log["validation_phase"] == "termination"


@pytest.mark.parametrize("task", _pilot_tasks(), ids=lambda task: task["id"])
@pytest.mark.asyncio
async def test_incomplete_valid_action_is_rejected_on_permitted_pilot_fixture(
    task, tmp_path
):
    await _check_pilot_fixture(task, tmp_path / task["id"])
