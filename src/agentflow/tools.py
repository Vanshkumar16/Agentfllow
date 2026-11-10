"""A small, explicit allowlist of tools available to plans."""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable
from typing import Any


class ToolError(ValueError):
    """Raised when a tool name or tool input is invalid."""


def _calculate(arguments: dict[str, Any]) -> dict[str, float | int]:
    expression = arguments.get("expression")
    if not isinstance(expression, str) or not expression.strip():
        raise ToolError("'expression' must be a non-empty string.")

    binary_operators: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    unary_operators: dict[type[ast.unaryop], Callable[[Any], Any]] = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def evaluate(node: ast.AST) -> float | int:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in binary_operators:
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ToolError("Exponent magnitude must not exceed 100.")
            return binary_operators[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in unary_operators:
            return unary_operators[type(node.op)](evaluate(node.operand))
        raise ToolError("Only numeric constants and basic arithmetic are supported.")

    try:
        result = evaluate(ast.parse(expression, mode="eval"))
    except (SyntaxError, ZeroDivisionError, OverflowError) as exc:
        raise ToolError(f"Invalid arithmetic expression: {exc}") from exc
    return {"result": result}


def _echo(arguments: dict[str, Any]) -> dict[str, Any]:
    if "text" not in arguments:
        raise ToolError("'text' is required.")
    return {"text": arguments["text"]}


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, tuple[str, Callable[[dict[str, Any]], Any]]] = {
            "calculator": ("Evaluate a basic arithmetic expression.", _calculate),
            "echo": ("Return the provided text unchanged.", _echo),
        }

    def descriptions(self) -> dict[str, str]:
        return {name: description for name, (description, _) in self._tools.items()}

    def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        if not isinstance(arguments, dict):
            raise ToolError("Tool input must be a JSON object.")
        tool = self._tools.get(name)
        if tool is None:
            raise ToolError(f"Tool '{name}' is not allowed.")
        return tool[1](arguments)
