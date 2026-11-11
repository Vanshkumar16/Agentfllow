"""Strict JSON schemas and local validation for model response payloads."""

from __future__ import annotations

import json
import os
from typing import Any


class ResponseValidationError(ValueError):
    def __init__(self, message: str, *, phase: str) -> None:
        super().__init__(message)
        self.phase = phase


def action_response_schema(*, allow_shell: bool) -> dict[str, Any]:
    path = {"type": "string", "minLength": 1}
    tools = ["list_files", "read_file", "write_file", "finish"]
    if allow_shell:
        tools.insert(3, "run_command")
    return {
        "type": "object",
        "properties": {
            "action": {
                "type": "object",
                "properties": {
                    "tool": {"type": "string", "enum": tools},
                    "arguments": {"type": "object"},
                },
                "required": ["tool", "arguments"],
                "additionalProperties": False,
            }
        },
        "required": ["action"],
        "additionalProperties": False,
    }


def patch_response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "complete": {"type": "boolean"},
            "patches": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "minLength": 1},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["complete", "patches"],
        "additionalProperties": False,
    }


def parse_action_response(response: str, *, allow_shell: bool) -> dict[str, Any]:
    schema = action_response_schema(allow_shell=allow_shell)
    payload = _parse_json(response)
    _validate_schema(payload, schema)
    action = payload["action"]
    _validate_schema(
        action["arguments"],
        _action_argument_schema(action["tool"]),
        "$.action.arguments",
    )
    return {"tool": action["tool"], "arguments": action["arguments"]}


def parse_patch_response(response: str) -> dict[str, Any]:
    payload = _parse_json(response)
    _validate_schema(payload, patch_response_schema())
    validated: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    for patch in payload["patches"]:
        normalized_path = os.path.normcase(
            os.path.normpath(patch["path"].replace("/", os.sep))
        )
        if normalized_path in seen_paths:
            raise ResponseValidationError(
                f"Patch response contains duplicate normalized path '{patch['path']}'.",
                phase="schema",
            )
        seen_paths.add(normalized_path)
        validated.append({"path": patch["path"], "content": patch["content"]})
    return {"complete": payload["complete"], "patches": validated}


def _parse_json(response: str) -> Any:
    if not isinstance(response, str) or not response.strip():
        raise ResponseValidationError("Model response content is empty.", phase="syntax")
    try:
        return json.loads(
            response,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_json_constant,
        )
    except ResponseValidationError:
        raise
    except json.JSONDecodeError as exc:
        raise ResponseValidationError(
            f"Model response is not valid JSON: {exc}", phase="syntax"
        ) from exc


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ResponseValidationError(
                f"Model response contains duplicate JSON key '{key}'.",
                phase="syntax",
            )
        result[key] = value
    return result


def _reject_non_json_constant(value: str) -> None:
    raise ResponseValidationError(
        f"Model response contains non-JSON numeric constant '{value}'.",
        phase="syntax",
    )


def _validate_schema(value: Any, schema: dict[str, Any], path: str = "$") -> None:
    expected = schema.get("type")
    matches_type = {
        "object": lambda item: isinstance(item, dict),
        "array": lambda item: isinstance(item, list),
        "string": lambda item: isinstance(item, str),
        "boolean": lambda item: isinstance(item, bool),
    }
    if expected in matches_type and not matches_type[expected](value):
        raise ResponseValidationError(
            f"Response schema validation failed at {path}: expected {expected}.",
            phase="schema",
        )

    if "const" in schema and value != schema["const"]:
        raise ResponseValidationError(
            f"Response schema validation failed at {path}: value must be {schema['const']!r}.",
            phase="schema",
        )
    if "enum" in schema and value not in schema["enum"]:
        raise ResponseValidationError(
            f"Response schema validation failed at {path}: value is not an allowed option.",
            phase="schema",
        )
    if isinstance(value, str) and (
        len(value) < schema.get("minLength", 0)
        or (schema.get("minLength", 0) > 0 and not value.strip())
    ):
        raise ResponseValidationError(
            f"Response schema validation failed at {path}: string must not be empty.",
            phase="schema",
        )
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            raise ResponseValidationError(
                f"Response schema validation failed at {path}: array must not be empty.",
                phase="schema",
            )
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                _validate_schema(item, item_schema, f"{path}[{index}]")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                raise ResponseValidationError(
                    f"Response schema validation failed at {path}: missing required field '{key}'.",
                    phase="schema",
                )
        if schema.get("additionalProperties") is False:
            unknown = sorted(set(value) - set(properties))
            if unknown:
                raise ResponseValidationError(
                    f"Response schema validation failed at {path}: unexpected field '{unknown[0]}'.",
                    phase="schema",
                )
        for key, item in value.items():
            item_schema = properties.get(key)
            if isinstance(item_schema, dict):
                _validate_schema(item, item_schema, f"{path}.{key}")

    options = schema.get("oneOf")
    if isinstance(options, list):
        if isinstance(value, dict) and isinstance(value.get("tool"), str):
            matching = [
                option
                for option in options
                if isinstance(option.get("properties"), dict)
                and isinstance(option["properties"].get("tool"), dict)
                and option["properties"]["tool"].get("const") == value["tool"]
            ]
            if len(matching) == 1:
                _validate_schema(value, matching[0], path)
                return
            if not matching:
                raise ResponseValidationError(
                    f"Response schema validation failed at {path}.tool: unsupported action '{value['tool']}'.",
                    phase="schema",
                )
        successes = 0
        errors: list[ResponseValidationError] = []
        for option in options:
            try:
                _validate_schema(value, option, path)
            except ResponseValidationError as exc:
                errors.append(exc)
                continue
            successes += 1
        if successes != 1:
            if successes == 0 and errors:
                raise errors[0]
            raise ResponseValidationError(
                f"Response schema validation failed at {path}: expected exactly one matching action schema.",
                phase="schema",
            )


def _action_argument_schema(tool: str) -> dict[str, Any]:
    path = {"type": "string", "minLength": 1}
    schemas = {
        "list_files": {
            "type": "object",
            "properties": {"path": path},
            "additionalProperties": False,
        },
        "read_file": {
            "type": "object",
            "properties": {"path": path},
            "required": ["path"],
            "additionalProperties": False,
        },
        "write_file": {
            "type": "object",
            "properties": {"path": path, "content": {"type": "string"}},
            "required": ["path", "content"],
            "additionalProperties": False,
        },
        "finish": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        "run_command": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "minItems": 1,
                },
                "cwd": path,
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    }
    schema = schemas.get(tool)
    if schema is None:
        raise ResponseValidationError(
            f"Response schema validation failed at $.action.tool: unsupported action '{tool}'.",
            phase="schema",
        )
    return schema
