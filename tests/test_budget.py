import pytest

from agentflow.budget import BudgetLimits, BudgetTracker


def test_model_call_and_output_token_budgets_are_enforced():
    tracker = BudgetTracker(BudgetLimits(max_model_calls=1, max_tokens=10))

    assert tracker.begin_model_call() == 10
    tracker.record_model_usage(4, 7)
    assert tracker.tokens_used == 11
    assert tracker.begin_model_call() is None
    assert tracker.stop_reason == "model-call budget exhausted"


def test_token_usage_is_reported_as_unavailable_when_provider_omits_it():
    tracker = BudgetTracker(BudgetLimits())
    tracker.begin_model_call()
    tracker.record_model_usage(None, None)

    assert tracker.tokens_used is None
    assert tracker.to_dict()["input_tokens"] is None


def test_repair_attempts_cannot_exceed_three():
    with pytest.raises(ValueError, match="between 0 and 3"):
        BudgetLimits(max_repair_attempts=4)


def test_recovery_requires_remaining_tool_and_test_budgets():
    tracker = BudgetTracker(BudgetLimits(max_tool_calls=1, max_test_runs=1))
    assert tracker.begin_tool_call()
    assert tracker.begin_test_run()

    allowed, reason = tracker.can_repair(minimum_tokens=1, minimum_seconds=0)

    assert allowed is False
    assert "tool-call budget" in reason
