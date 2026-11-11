import pytest

from agentflow.failure import FailureCategory, RecoveryDecision, classify_failure


@pytest.mark.parametrize(
    ("evidence", "category", "action_fragment"),
    [
        (
            "SyntaxError: invalid syntax",
            FailureCategory.SYNTAX_ERROR,
            "Revise the patch",
        ),
        (
            "ModuleNotFoundError: No module named 'missing_pkg'",
            FailureCategory.IMPORT_ERROR,
            "Inspect the failing import",
        ),
        (
            "E AssertionError: assert 3 == 4",
            FailureCategory.TEST_ASSERTION,
            "Re-evaluate the expected behavior",
        ),
        (
            "Traceback (most recent call last):\nValueError: bad value",
            FailureCategory.RUNTIME_EXCEPTION,
            "Inspect the traceback",
        ),
        (
            "Command timed out after 3 seconds",
            FailureCategory.TIMEOUT,
            "Stop the timed-out operation",
        ),
        (
            "Model response is not valid JSON",
            FailureCategory.MALFORMED_RESPONSE,
            "controlled regeneration",
        ),
    ],
)
def test_classifies_failure_categories_and_returns_recovery_action(
    evidence, category, action_fragment
):
    assessment = classify_failure(evidence)

    assert assessment.category is category
    assert action_fragment in assessment.recovery_action
    assert assessment.fingerprint
    assert isinstance(assessment.suggested_decision, RecoveryDecision)


def test_timeout_flag_classifies_empty_evidence_as_timeout():
    assert classify_failure("", timed_out=True).category is FailureCategory.TIMEOUT


def test_failure_fingerprint_ignores_test_run_durations():
    first = classify_failure("AssertionError: expected 1\n1 failed in 0.21s")
    second = classify_failure("AssertionError: expected 1\n1 failed in 0.48s")

    assert first.fingerprint == second.fingerprint


@pytest.mark.parametrize(
    ("evidence", "category", "decision"),
    [
        (
            "Provider-confirmed truncated response: invalid JSON",
            FailureCategory.TRUNCATED_RESPONSE,
            RecoveryDecision.RETRY,
        ),
        (
            "Provider termination status is unknown; response may be incomplete",
            FailureCategory.INCOMPLETE_RESPONSE,
            RecoveryDecision.RETRY,
        ),
        (
            "Response schema validation failed: missing field",
            FailureCategory.SCHEMA_VALIDATION,
            RecoveryDecision.RETRY,
        ),
    ],
)
def test_response_failures_have_specific_recovery_decisions(
    evidence, category, decision
):
    assessment = classify_failure(evidence)

    assert assessment.category is category
    assert assessment.suggested_decision is decision
