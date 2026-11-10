import json

import pytest

from agentflow.controller import AgentController
from agentflow.schemas import TaskRequest


class FakeModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts = []

    async def complete(self, prompt):
        self.prompts.append(prompt)
        return next(self.responses)


@pytest.mark.asyncio
async def test_controller_completes_after_successful_tool_execution():
    model = FakeModel(
        [
            json.dumps(
                {
                    "steps": [
                        {
                            "id": "calculate",
                            "description": "Multiply the numbers",
                            "tool": "calculator",
                            "input": {"expression": "17 * 23"},
                        }
                    ]
                }
            ),
            json.dumps({"complete": True, "feedback": "The result is 391."}),
        ]
    )

    result = await AgentController(model, retries_per_step=0).run(
        TaskRequest("Calculate 17 * 23.")
    )

    assert result.complete is True
    assert result.iterations[0].steps[0].output == {"result": 391}
    assert result.iterations[0].tests.passed is True


@pytest.mark.asyncio
async def test_controller_replans_when_critic_says_incomplete():
    model = FakeModel(
        [
            json.dumps(
                {
                    "steps": [
                        {
                            "id": "first",
                            "description": "Return an interim value",
                            "tool": "echo",
                            "input": {"text": "interim"},
                        }
                    ]
                }
            ),
            json.dumps({"complete": False, "feedback": "The requested result is missing."}),
            json.dumps(
                {
                    "steps": [
                        {
                            "id": "second",
                            "description": "Calculate the answer",
                            "tool": "calculator",
                            "input": {"expression": "6 * 7"},
                        }
                    ]
                }
            ),
            json.dumps({"complete": True, "feedback": "The answer is 42."}),
        ]
    )

    result = await AgentController(model, max_iterations=2, retries_per_step=0).run(
        TaskRequest("Calculate 6 * 7.")
    )

    assert result.complete is True
    assert len(result.iterations) == 2
    assert "requested result is missing" in model.prompts[2]
