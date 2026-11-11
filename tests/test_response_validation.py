import pytest

from agentflow.response_validation import (
    ResponseValidationError,
    action_response_schema,
    parse_action_response,
    parse_patch_response,
    patch_response_schema,
)


def test_action_schema_is_strict_and_tool_specific():
    schema = action_response_schema(allow_shell=False)

    parsed = parse_action_response(
        '{"action":{"tool":"read_file","arguments":{"path":"solution.py"}}}',
        allow_shell=False,
    )
    assert parsed == {"tool": "read_file", "arguments": {"path": "solution.py"}}
    assert schema["properties"]["action"]["properties"]["tool"]["enum"]

    with pytest.raises(ResponseValidationError, match="unexpected field"):
        parse_action_response(
            '{"action":{"tool":"write_file","arguments":{"path":"x.py","content":"x","extra":true}}}',
            allow_shell=False,
        )
    with pytest.raises(ResponseValidationError, match="allowed option"):
        parse_action_response(
            '{"action":{"tool":"run_command","arguments":{"command":["python","-V"]}}}',
            allow_shell=False,
        )


def test_json_parser_rejects_duplicate_keys_and_nonstandard_constants():
    with pytest.raises(ResponseValidationError, match="duplicate JSON key"):
        parse_action_response(
            '{"action":{"tool":"finish","tool":"write_file","arguments":{}}}',
            allow_shell=False,
        )
    with pytest.raises(ResponseValidationError, match="non-JSON numeric constant"):
        parse_action_response(
            '{"action":{"tool":"finish","arguments":{"value":NaN}}}',
            allow_shell=False,
        )


def test_patch_schema_and_normalized_duplicate_paths_are_rejected():
    assert patch_response_schema()["additionalProperties"] is False
    with pytest.raises(ResponseValidationError, match="unexpected field"):
        parse_patch_response(
            '{"complete":true,"patches":[{"path":"a.py","content":"x","extra":1}]}'
        )
    with pytest.raises(ResponseValidationError, match="duplicate normalized path"):
        parse_patch_response(
            '{"complete":true,"patches":['
            '{"path":"src/../solution.py","content":"a"},'
            '{"path":"solution.py","content":"b"}]}'
        )
