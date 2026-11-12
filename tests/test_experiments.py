import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from agentflow.budget import BudgetLimits
from agentflow.coding_agent import RecoveryStrategy
from experiments.evaluate import evaluate
from experiments import run_experiment
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


def test_v2_benchmark_is_independent_and_has_frozen_split_sizes():
    original = load_tasks(Path("benchmarks/tasks.jsonl"))
    tasks = load_tasks(Path("benchmarks/tasks_v2.jsonl"))
    original_ids = {task["id"] for task in original}

    assert len(tasks) == 40
    assert len({task["id"] for task in tasks}) == 40
    assert original_ids.isdisjoint({task["id"] for task in tasks})
    assert sum(task["split"] == "pilot" for task in tasks) == 5
    assert sum(task["split"] == "held-out" for task in tasks) == 35
    assert len({task["category"] for task in tasks}) >= 30
    assert all(
        task["difficulty"] in {"introductory", "standard", "advanced"}
        and "solution.py" in task["files"]
        and "test_solution.py" in task["files"]
        for task in tasks
    )


@pytest.mark.asyncio
async def test_experiment_runner_records_reproducible_repetitions_and_hashes(
    tmp_path, monkeypatch
):
    task_path = tmp_path / "tasks.jsonl"
    task_bytes = b"".join(
        (
            json.dumps(
                {"id": f"t{i}", "goal": "complete task", "files": {"main.py": "x = 1\n"}}
            )
            + "\n"
        ).encode()
        for i in range(6)
    )
    task_path.write_bytes(task_bytes)
    config_path = tmp_path / "experiment.yaml"
    config_text = (
        "model:\n  name: test-model\n  device: cpu\n"
        "budgets:\n  max_repair_attempts: 1\n"
        "max_actions_per_attempt: 5\nmax_prompt_chars: 12000\n"
    )
    config_path.write_text(config_text, encoding="utf-8")
    config_bytes = config_path.read_bytes()

    class FakeModel:
        model = "test-model"
        device = "cpu"

        async def list_model_details(self):
            return {self.model: {}}

    monkeypatch.setattr(run_experiment, "OllamaModel", lambda *args: FakeModel())
    monkeypatch.setattr(run_experiment, "get_hardware_info", lambda: {"gpus": []})
    received_cases = []

    async def fake_run_case(
        task, model, strategy, limits, experiment_id, **kwargs
    ):
        received_cases.append((task["id"], strategy, kwargs))
        return {
            "record_type": "result",
            "task_id": task["id"],
            "strategy": strategy.value,
        }

    monkeypatch.setattr(run_experiment, "_run_case", fake_run_case)

    output_paths = [
        tmp_path / "first.jsonl",
        tmp_path / "second.jsonl",
        tmp_path / "other-seed.jsonl",
    ]
    for output_path in output_paths:
        await run_experiment.run(
            task_path,
            output_path,
            config_path,
            "pilot",
            repetitions=2,
            random_seed=82 if output_path.name == "other-seed.jsonl" else 81,
        )

    outputs = [
        [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for path in output_paths
    ]
    metadata = outputs[0][0]
    schedule = metadata["randomized_execution_order"]

    assert metadata["random_seed"] == 81
    assert metadata["repetitions"] == 2
    assert metadata["git_commit_hash"]
    assert isinstance(metadata["git_worktree_clean"], bool)
    assert metadata["dataset_sha256"] == hashlib.sha256(task_bytes).hexdigest()
    assert metadata["config_sha256"] == hashlib.sha256(config_bytes).hexdigest()
    assert metadata["max_actions_per_attempt"] == 5
    assert metadata["max_prompt_chars"] == 12000
    assert schedule == outputs[1][0]["randomized_execution_order"]
    assert schedule != outputs[2][0]["randomized_execution_order"]
    assert len(schedule) == 2 * 5 * len(run_experiment.STRATEGIES)
    for repetition_id in ("rep-001", "rep-002"):
        repetition_schedule = [
            row for row in schedule if row["repetition_id"] == repetition_id
        ]
        assert sorted(row["execution_order"] for row in repetition_schedule) == list(
            range(1, 16)
        )
        assert {
            (row["task_id"], row["strategy"]) for row in repetition_schedule
        } == {
            (f"t{i}", strategy.value)
            for i in range(5)
            for strategy in run_experiment.STRATEGIES
        }
        assert [
            (row["task_id"], row["strategy"]) for row in repetition_schedule
        ] != [
            (f"t{i}", strategy.value)
            for i in range(5)
            for strategy in run_experiment.STRATEGIES
        ]
    results = [row for row in outputs[0] if row["record_type"] == "result"]
    assert len(results) == 30
    assert all(row["random_seed"] == 81 for row in results)
    assert all(row["repetition_id"] in {"rep-001", "rep-002"} for row in results)
    assert all(1 <= row["execution_order"] <= 15 for row in results)
    scheduled_cases = {
        (row["repetition_id"], row["execution_order"]): (
            row["task_id"],
            row["strategy"],
        )
        for row in schedule
    }
    assert all(
        scheduled_cases[(row["repetition_id"], row["execution_order"])]
        == (row["task_id"], row["strategy"])
        for row in results
    )
    assert all(
        kwargs["max_actions_per_attempt"] == 5 for _, _, kwargs in received_cases
    )
    assert all(kwargs["max_prompt_chars"] == 12000 for _, _, kwargs in received_cases)


@pytest.mark.asyncio
async def test_experiment_runner_refuses_to_overwrite_results(tmp_path, monkeypatch):
    output_path = tmp_path / "existing.jsonl"
    original = "keep this result\n"
    output_path.write_text(original, encoding="utf-8")
    monkeypatch.setattr(
        run_experiment,
        "OllamaModel",
        lambda *args: pytest.fail("Model setup should not run for existing output."),
    )

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        await run_experiment.run(
            tmp_path / "missing-tasks.jsonl",
            output_path,
            tmp_path / "missing-config.yaml",
        )

    assert output_path.read_text(encoding="utf-8") == original


def test_run_case_passes_yaml_action_limits_to_coding_agent(tmp_path, monkeypatch):
    seen = {}

    class FakeAgent:
        def __init__(self, model, workspace, config):
            seen["config"] = config

        async def run(self, task, *, experiment_id, task_id):
            class Result:
                def to_dict(self):
                    return {}

            return Result()

    class FakeModel:
        model = "test-model"
        device = "cpu"

    monkeypatch.setattr(run_experiment, "CodingAgent", FakeAgent)

    asyncio.run(
        run_experiment._run_case(
            {"id": "t1", "goal": "test", "files": {"main.py": "x = 1\n"}},
            FakeModel(),
            RecoveryStrategy.FAILURE_GUIDED,
            BudgetLimits(),
            "experiment",
            max_actions_per_attempt=5,
            max_prompt_chars=12000,
        )
    )

    assert seen["config"].max_actions_per_attempt == 5
    assert seen["config"].max_prompt_chars == 12000


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
    assert summary["budget_rejections_by_kind"] == {"tool_calls": 1}
    assert summary["token_overrun_events"] == 0
    assert summary["budget_violation_cases"] == 0
    assert summary["token_usage"]["reported_usage_call_coverage"] == pytest.approx(
        1 / 3
    )
    assert summary["token_usage"]["missing_model_usage_calls"] == 2
    assert summary["token_usage"]["missing_result_token_usage_cases"] == 0
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
    assert summary["budget_violation_reasons"] == {
        "max_tokens": 1,
        "token_overrun": 1,
    }


def test_evaluator_reports_paired_repetition_statistics_and_usage_gaps(tmp_path):
    results_file = tmp_path / "paired.jsonl"
    strategies = ("single-shot", "fixed-retry", "failure-guided")
    resolved = {
        "single-shot": [True, False],
        "fixed-retry": [False, True],
        "failure-guided": [True, True],
    }
    test_passed = {
        "single-shot": [False, True],
        "fixed-retry": [True, False],
        "failure-guided": [True, True],
    }
    rows = [{"record_type": "run_metadata", "repetitions": 2}]
    for strategy in strategies:
        for repetition_index, repetition_id in enumerate(("rep-001", "rep-002")):
            passed = test_passed[strategy][repetition_index]
            row = {
                "record_type": "result",
                "model": "local-model",
                "strategy": strategy,
                "task_id": "task-1",
                "repetition_id": repetition_id,
                "resolved": resolved[strategy][repetition_index],
                "final_test_passed": passed,
                "repair_attempts": repetition_index,
                "budget": {
                    "elapsed_seconds": 2.0 + repetition_index,
                    "tokens_used": 20 + repetition_index,
                    "model_calls": 1,
                    "budget_events": (
                        [{"event": "budget_rejected", "budget_kind": "tokens"}]
                        if repetition_index == 1
                        else []
                    ),
                },
                "budget_limits": {
                    "max_tokens": 100,
                    "max_model_calls": 3,
                    "max_repair_attempts": 2,
                },
                "per_call_logs": [
                    {
                        "call_type": "model",
                        "token_accounting": {
                            "usage_status": (
                                "unavailable"
                                if strategy == "single-shot" and repetition_index == 1
                                else "measured"
                            )
                        },
                    }
                ],
            }
            rows.append(row)
    results_file.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    report = evaluate(results_file)

    assert len(report["paired_comparisons"]) == 3
    single_summary = next(
        summary
        for summary in report["runs"]
        if summary["strategy"] == "single-shot"
    )
    assert single_summary["strict_resolution_rate"] == 0.5
    assert single_summary["final_test_pass_rate"] == 0.5
    assert single_summary["metrics_across_repetitions"]["strict_resolution"][
        "standard_deviation"
    ] > 0
    assert single_summary["metrics_across_repetitions"]["strict_resolution"][
        "confidence_interval_95"
    ] is not None
    assert single_summary["mean_repair_attempts"] == 0.5
    assert single_summary["mean_latency_seconds"] == 2.5
    assert single_summary["token_usage"]["reported_usage_call_coverage"] == 0.5
    assert single_summary["token_usage"]["missing_model_usage_calls"] == 1
    assert single_summary["budget_rejections_by_kind"] == {"tokens": 1}

    paired = next(
        comparison
        for comparison in report["paired_comparisons"]
        if comparison["strategy_a"] == "single-shot"
        and comparison["strategy_b"] == "fixed-retry"
    )
    assert paired["matched_task_count"] == 1
    assert paired["matched_repetition_count"] == 2
    assert paired["per_task"][0]["task_id"] == "task-1"
    assert paired["per_task"][0]["mean_difference_a_minus_b"][
        "strict_resolution"
    ]["mean"] == 0
    assert paired["per_task"][0]["mean_difference_a_minus_b"][
        "strict_resolution"
    ]["confidence_interval_95"] is not None
    assert (
        paired["per_task"][0]["mean_difference_a_minus_b"]["final_test_passed"][
            "mean"
        ]
        == 0
    )
