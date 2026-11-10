import pytest

from agentflow.tools import ToolError, ToolRegistry


def test_calculator_supports_basic_arithmetic():
    assert ToolRegistry().execute("calculator", {"expression": "(2 + 3) * 4"}) == {
        "result": 20
    }


def test_calculator_rejects_code_execution():
    with pytest.raises(ToolError, match="Only numeric constants"):
        ToolRegistry().execute("calculator", {"expression": "__import__('os').getcwd()"})


def test_unknown_tool_is_rejected():
    with pytest.raises(ToolError, match="not allowed"):
        ToolRegistry().execute("run_shell", {"command": "echo unsafe"})
