import pytest

from agentflow.budget import BudgetLimits, BudgetTracker


def test_model_call_and_output_token_budgets_are_enforced():
    tracker = BudgetTracker(BudgetLimits(max_model_calls=2, max_tokens=10))

    assert tracker.begin_model_call() == 10
    tracker.record_model_usage(4, 7)
    assert tracker.tokens_used == 11
    assert tracker.begin_model_call() is None
    assert tracker.stop_reason == "token budget exceeded by the completed model call"
    assert any(event["event"] == "token_overrun" for event in tracker.budget_events)


def test_prompt_estimate_and_output_are_reserved_before_generation():
    tracker = BudgetTracker(
        BudgetLimits(max_tokens=10, max_output_tokens_per_call=8)
    )

    assert tracker.begin_model_call("abcdefghijkl") == 4
    event = tracker.budget_events[-1]
    assert event["event"] == "model_call_reserved"
    assert event["estimated_input_tokens"] == 6
    assert event["reserved_output_tokens"] == 4
    assert tracker.remaining_tokens == 0


def test_call_is_rejected_if_estimated_prompt_leaves_no_output_budget():
    tracker = BudgetTracker(BudgetLimits(max_tokens=8))

    assert tracker.begin_model_call("abcdefghijklmnop") is None
    assert tracker.model_calls == 0
    assert tracker.stop_reason == "estimated prompt leaves no token budget for model output"
    assert tracker.budget_events[-1]["event"] == "budget_rejected"


def test_unknown_usage_is_charged_using_prompt_estimate_and_output_reservation():
    tracker = BudgetTracker(
        BudgetLimits(max_tokens=20, max_output_tokens_per_call=10)
    )

    assert tracker.begin_model_call("abcdefgh") == 10
    tracker.record_model_usage(None, None)

    assert tracker.tokens_used is None
    assert tracker.to_dict()["input_tokens"] is None
    assert tracker.to_dict()["budget_tokens_charged"] == 14
    assert tracker.remaining_tokens == 6
    accounting = next(
        event
        for event in tracker.budget_events
        if event["event"] == "model_call_accounted"
    )
    assert accounting["usage_status"] == "unavailable"
    assert accounting["charged_total_tokens"] == 14


def test_partial_measured_usage_is_separate_from_conservative_charge():
    tracker = BudgetTracker(
        BudgetLimits(max_tokens=30, max_output_tokens_per_call=10)
    )

    assert tracker.begin_model_call("abcdefgh") == 10
    accounting = tracker.record_model_usage(4, None)

    assert accounting["estimated_input_tokens"] == 4
    assert accounting["measured_input_tokens"] == 4
    assert accounting["measured_output_tokens"] is None
    assert accounting["charged_input_tokens"] == 4
    assert accounting["charged_output_tokens"] == 10
    assert accounting["usage_status"] == "partial"
    assert tracker.to_dict()["measured_input_tokens"] == 4
    assert tracker.to_dict()["measured_output_tokens"] is None
    assert tracker.to_dict()["budget_tokens_charged"] == 14


def test_actual_usage_overrun_is_logged_and_stops_future_calls(caplog):
    tracker = BudgetTracker(
        BudgetLimits(max_tokens=10, max_output_tokens_per_call=9)
    )

    assert tracker.begin_model_call("ab") == 9
    tracker.record_model_usage(9, 3)

    assert tracker.stop_reason == "token budget exceeded by the completed model call"
    assert tracker.to_dict()["input_tokens"] == 9
    assert tracker.to_dict()["output_tokens"] == 3
    assert tracker.to_dict()["total_tokens"] == 12
    assert tracker.to_dict()["tokens_used"] == 12
    assert tracker.begin_model_call("next prompt") is None
    assert tracker.budget_events[-1]["event"] == "budget_rejected"
    assert any(
        getattr(record, "budget_event", {}).get("event") == "token_overrun"
        for record in caplog.records
    )


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
    assert not tracker.begin_tool_call()
    assert tracker.begin_test_run()
    assert tracker.budget_events[-1]["event"] == "budget_rejected"

    allowed, reason = tracker.can_repair(minimum_tokens=1, minimum_seconds=0)

    assert allowed is False
    assert "tool-call budget" in reason
