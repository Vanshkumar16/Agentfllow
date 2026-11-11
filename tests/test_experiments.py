import json
from pathlib import Path

import pytest

from agentflow.budget import BudgetLimits
from experiments.evaluate import evaluate
from experiments.run_final_evaluation import (
    build_schedule,
    _validate_frozen_config,
)
from experiments.run_experiment import _optional_model_skips, _select_tasks, load_tasks


def test_controlled_benchmark_has_five_pilot_and_fifteen_held_out_tasks():
    tasks = load_tasks(Path("benchmarks/tasks.jsonl"))

    assert len(tasks) == 20
    assert len(_select_tasks(tasks, "pilot")) == 5
    assert len(_select_tasks(tasks, "held-out")) == 15


def test_final_evaluation_schedule_is_seeded_and_balanced():
    tasks = load_tasks(Path("benchmarks/tasks.jsonl"))
    held_out = _select_tasks(tasks, "held-out")

    first = build_schedule(held_out, 7)
    repeated = build_schedule(held_out, 7)
    other_seed = build_schedule(held_out, 8)

    assert [(task["id"], strategy) for task, strategy in first] == [
        (task["id"], strategy) for task, strategy in repeated
    ]
    assert [(task["id"], strategy) for task, strategy in first] != [
        (task["id"], strategy) for task, strategy in other_seed
    ]
    assert len(first) == 45
    assert {task["id"] for task, _ in first} == {task["id"] for task in held_out}
    assert all(
        sum(scheduled_task["id"] == task["id"] for scheduled_task, _ in first) == 3
        for task in held_out
    )


def test_final_evaluation_requires_fixed_model_and_budgets():
    _validate_frozen_config(
        {"name": "gemma4:e2b"},
        BudgetLimits(
            max_repair_attempts=3,
            task_timeout_seconds=120,
            max_tokens=8192,
        ),
    )

    with pytest.raises(ValueError, match="requires max_tokens=8192"):
        _validate_frozen_config(
            {"name": "gemma4:e2b"}, BudgetLimits(max_tokens=4096)
        )


def test_optional_e4b_requires_installed_model_and_measured_vram():
    candidates = [{"name": "gemma4:e4b-q4", "minimum_free_vram_gb": 4.5}]
    hardware = {
        "gpus": [{"name": "RTX 3050", "free_vram_mib": 6144, "total_vram_mib": 6144}]
    }
    installed = {
        "gemma4:e4b-q4": {
            "parameter_size": "4B",
            "quantization_level": "Q4_K_M",
            "family": "gemma4",
        }
    }

    runnable, skipped = _optional_model_skips(candidates, installed, hardware, "auto")
    assert len(runnable) == 1
    assert skipped == []

    runnable, skipped = _optional_model_skips(
        candidates,
        installed,
        {"gpus": [{"free_vram_mib": 1024}]},
        "auto",
    )
    assert runnable == []
    assert "below configured threshold" in skipped[0]["reason"]

    runnable, skipped = _optional_model_skips(
        candidates,
        {"gemma4:e4b-q4": {"parameter_size": "4B", "family": "gemma4"}},
        hardware,
        "auto",
    )
    assert runnable == []
    assert "quantized model" in skipped[0]["reason"]


def test_evaluator_reports_resolution_attempts_latency_and_tokens(tmp_path):
    results_file = tmp_path / "results.jsonl"
    rows = [
        {"record_type": "run_metadata", "split": "pilot"},
        {
            "record_type": "result",
            "task_id": "t1",
            "model": "local-model",
            "strategy": "failure-guided",
            "resolved": False,
            "patch_valid": True,
            "final_test_passed": True,
            "explicit_completion": False,
            "budget_exhausted": False,
            "repair_attempts": 1,
            "attempts": [
                {
                    "test": {"passed": False},
                    "failure_category": "test_assertion_failure",
                },
                {"test": {"passed": True}},
            ],
            "budget": {
                "model_calls": 3,
                "tool_calls": 1,
                "test_runs": 2,
                "elapsed_seconds": 4.0,
                "tokens_used": 100,
                "budget_tokens_charged": 100,
                "budget_events": [
                    {"event": "model_call_reserved"},
                    {"event": "budget_rejected", "budget_kind": "tool_calls"},
                ],
            },
            "budget_limits": {
                "max_model_calls": 10,
                "max_tool_calls": 10,
                "max_test_runs": 4,
                "task_timeout_seconds": 120,
                "max_tokens": 8192,
            },
            "per_call_logs": [
                {
                    "call_type": "model",
                    "validation_outcome": "valid",
                    "response_termination": "complete",
                    "token_accounting": {"usage_status": "measured"},
                },
                {"call_type": "tool"},
                {"call_type": "test"},
            ],
        },
    ]
    results_file.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    report = evaluate(results_file)

    assert report["metadata"][0]["split"] == "pilot"
    summary = report["runs"][0]
    assert summary["resolution_rate"] == 0.0
    assert summary["test_pass_rate"] == 1.0
    assert summary["patch_valid"] == 1
    assert summary["explicit_completion"] == 0
    assert summary["tests_passed_without_completion"] == 1
    assert summary["mean_repair_attempts"] == 1.0
    assert summary["mean_tokens_used"] == 100
    assert summary["failure_categories"] == {"test_assertion_failure": 1}
    assert summary["budget_rejection_events"] == 1
    assert summary["token_overrun_events"] == 0
    assert summary["budget_violation_cases"] == 0
    assert summary["model_calls_logged"] == 1
    assert summary["tool_calls_logged"] == 1
    assert summary["test_calls_logged"] == 1
    assert summary["measured_model_usage_calls"] == 1


def test_evaluator_reports_token_budget_violations(tmp_path):
    results_file = tmp_path / "violations.jsonl"
    rows = [
        {"record_type": "run_metadata", "split": "held-out"},
        {
            "record_type": "result",
            "task_id": "t2",
            "model": "local-model",
            "strategy": "failure-guided",
            "resolved": False,
            "attempts": [
                {
                    "test": {"passed": False},
                    "failure_category": "test_assertion_failure",
                }
            ],
            "budget": {
                "model_calls": 1,
                "tool_calls": 0,
                "test_runs": 1,
                "elapsed_seconds": 2,
                "tokens_used": 12,
                "budget_tokens_charged": 12,
                "budget_events": [{"event": "token_overrun"}],
            },
            "budget_limits": {
                "max_model_calls": 10,
                "max_tool_calls": 10,
                "max_test_runs": 4,
                "task_timeout_seconds": 120,
                "max_tokens": 10,
            },
        },
    ]
    results_file.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    summary = evaluate(results_file)["runs"][0]

    assert summary["token_overrun_events"] == 1
    assert summary["budget_violation_cases"] == 1
