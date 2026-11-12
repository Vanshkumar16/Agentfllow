import asyncio
import hashlib
import json
import random
from pathlib import Path

import pytest

from agentflow.budget import BudgetLimits
from agentflow.coding_agent import RecoveryStrategy
from experiments.evaluate import evaluate
from experiments import run_experiment
from experiments import prepare_research_artifacts
from experiments.prepare_research_artifacts import _validate_pilot_evidence
from experiments.prepare_heldout_research_artifacts import (
    ARTIFACT_FILES,
    prepare as prepare_heldout_artifacts,
)
from experiments.run_final_evaluation import (
    build_schedule,
    _validate_frozen_config,
)
from experiments.run_experiment import _optional_model_skips, _select_tasks, load_tasks
from experiments.validate_heldout_results import (
    EXPECTED_CONFIG_SHA256,
    EXPECTED_DATASET_SHA256,
    EXPECTED_MODEL,
    EXPECTED_REPETITIONS,
    EXPECTED_RESULT_COUNT,
    EXPECTED_SEED,
    EXPECTED_STRATEGIES,
    EXPECTED_TASK_COUNT,
    validate_results,
)


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


@pytest.fixture
def pilot_evidence(tmp_path):
    dataset_path = tmp_path / "tasks_v2.jsonl"
    dataset_path.write_bytes(Path("benchmarks/tasks_v2.jsonl").read_bytes())
    raw_rows = [
        json.loads(line)
        for line in Path(
            "benchmarks/research-v2/raw/v2-pilot-frozen.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    metadata = next(row for row in raw_rows if row["record_type"] == "run_metadata")
    results = [row for row in raw_rows if row["record_type"] == "result"]
    return dataset_path, raw_rows, metadata, results


def test_pilot_evidence_validator_accepts_complete_balanced_pilot(pilot_evidence):
    dataset_path, raw_rows, metadata, _ = pilot_evidence

    validated = _validate_pilot_evidence(dataset_path, raw_rows, metadata)

    assert len(validated) == 15


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda rows: rows.pop(), "exactly 15 result records"),
        (
            lambda rows: rows.__setitem__(14, dict(rows[0])),
            "Duplicate pilot observations",
        ),
        (
            lambda rows: rows[0].__setitem__("task_id", "unexpected-task"),
            "Task ID mismatch",
        ),
        (
            lambda rows: [row for row in rows if row["task_id"] != "v2-001"],
            "exactly 15 result records",
        ),
        (
            lambda rows: [
                row for row in rows if row["strategy"] != "single-shot"
            ],
            "exactly 15 result records",
        ),
        (
            lambda rows: rows[0].__setitem__("strategy", "unexpected-strategy"),
            "Expected strategies",
        ),
        (
            lambda rows: rows[0].__setitem__("repetition_id", "rep-002"),
            "repetition",
        ),
        (
            lambda rows: rows[0].pop("repetition_id"),
            "repetition",
        ),
    ],
)
def test_pilot_evidence_validator_rejects_invalid_result_sets(
    pilot_evidence, mutate, message
):
    dataset_path, raw_rows, metadata, results = pilot_evidence
    results = [dict(row) for row in results]
    changed = mutate(results)
    if isinstance(changed, list):
        results = changed
    candidate_rows = [metadata, *results]

    with pytest.raises(ValueError, match=message):
        _validate_pilot_evidence(dataset_path, candidate_rows, metadata)


def test_pilot_evidence_validator_rejects_wrong_metadata_task_count(pilot_evidence):
    dataset_path, raw_rows, metadata, _ = pilot_evidence
    wrong_metadata = {**metadata, "task_count": 4}
    candidate_rows = [
        wrong_metadata if row is metadata else row for row in raw_rows
    ]

    with pytest.raises(ValueError, match="task_count"):
        _validate_pilot_evidence(dataset_path, candidate_rows, wrong_metadata)


def test_pilot_evidence_validator_rejects_unexpected_record_type(pilot_evidence):
    dataset_path, raw_rows, metadata, _ = pilot_evidence
    candidate_rows = [*raw_rows, {"record_type": "unknown"}]

    with pytest.raises(ValueError, match="Unexpected JSONL record types"):
        _validate_pilot_evidence(dataset_path, candidate_rows, metadata)


def test_pilot_evidence_validator_rejects_duplicate_pilot_dataset_ids(
    pilot_evidence,
):
    dataset_path, raw_rows, metadata, _ = pilot_evidence
    tasks = [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
    ]
    tasks[1]["id"] = tasks[0]["id"]
    dataset_path.write_text(
        "\n".join(json.dumps(task) for task in tasks) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="unique pilot tasks"):
        _validate_pilot_evidence(dataset_path, raw_rows, metadata)


def test_invalid_pilot_evidence_never_reaches_artifact_generation(
    tmp_path, monkeypatch, pilot_evidence
):
    _, raw_rows, _, _ = pilot_evidence
    metadata = next(row for row in raw_rows if row["record_type"] == "run_metadata")
    results = [row for row in raw_rows if row["record_type"] == "result"][:-1]
    results_path = tmp_path / "invalid-pilot.jsonl"
    results_path.write_text(
        "\n".join(
            json.dumps(row) for row in [metadata, *results]
        ) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "artifacts"
    monkeypatch.setattr(
        prepare_research_artifacts,
        "_write_csv",
        lambda *args, **kwargs: pytest.fail("Artifact generation was reached."),
    )

    with pytest.raises(ValueError, match="exactly 15 result records"):
        prepare_research_artifacts.prepare(results_path, output_dir)

    assert not output_dir.exists()


@pytest.fixture
def heldout_evidence(tmp_path):
    dataset_bytes = Path("benchmarks/tasks_v2.jsonl").read_bytes()
    tasks = [
        json.loads(line)
        for line in dataset_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]
    task_ids = [task["id"] for task in tasks if task["split"] == "held-out"]
    generator = random.Random(EXPECTED_SEED)
    schedule = []
    for repetition in range(1, EXPECTED_REPETITIONS + 1):
        cases = [
            (task_id, strategy)
            for task_id in task_ids
            for strategy in EXPECTED_STRATEGIES
        ]
        generator.shuffle(cases)
        schedule.extend(
            {
                "repetition_id": f"rep-{repetition:03d}",
                "execution_order": order,
                "task_id": task_id,
                "strategy": strategy,
            }
            for order, (task_id, strategy) in enumerate(cases, start=1)
        )
    experiment_id = "heldout-test-experiment"
    metadata = {
        "record_type": "run_metadata",
        "experiment_id": experiment_id,
        "split": "held-out",
        "task_count": EXPECTED_TASK_COUNT,
        "repetitions": EXPECTED_REPETITIONS,
        "random_seed": EXPECTED_SEED,
        "random_seed_scope": "task/strategy execution order; not the model runtime",
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "primary_model": EXPECTED_MODEL,
        "temperature": 0.1,
        "ollama_server_version": "0.40.13",
        "ollama_cli_version": "0.40.13",
        "budgets": {
            "max_model_calls": 16,
            "max_tool_calls": 24,
            "max_test_runs": 4,
            "max_tokens": 8192,
            "max_repair_attempts": 3,
            "task_timeout_seconds": 120,
            "model_timeout_seconds": 60,
            "command_timeout_seconds": 30,
            "max_output_tokens_per_call": 1024,
            "min_repair_tokens": 256,
            "min_repair_seconds": 10,
        },
        "max_actions_per_attempt": 8,
        "max_prompt_chars": 24000,
        "device_requested": "auto",
        "hardware": {
            "platform": "test",
            "python": "3.12.0",
            "nvidia_smi_available": False,
            "gpus": [],
        },
        "git_commit_hash": "a" * 40,
        "git_worktree_clean": True,
        "model_inventory": {EXPECTED_MODEL: {"digest": "b" * 64, "details": {}}},
        "strategies": list(EXPECTED_STRATEGIES),
        "randomized_execution_order": schedule,
    }
    results = []
    for index, item in enumerate(schedule):
        passed = index % 2 == 0
        results.append(
            {
                "record_type": "result",
                "experiment_id": experiment_id,
                "model": EXPECTED_MODEL,
                "task_id": item["task_id"],
                "strategy": item["strategy"],
                "repetition_id": item["repetition_id"],
                "execution_order": item["execution_order"],
                "random_seed": EXPECTED_SEED,
                "resolved": passed,
                "final_test_passed": passed,
                "repair_attempts": index % 3,
                "attempts": [
                    {"failure_category": "test_assertion_failure"}
                    if not passed
                    else {}
                ],
                "budget": {
                    "elapsed_seconds": float(index % 7 + 1),
                    "tokens_used": index % 100 + 50,
                    "budget_tokens_charged": index % 100 + 50,
                    "model_calls": 1,
                    "tool_calls": 1,
                    "test_runs": 1,
                    "budget_events": [],
                },
                "budget_limits": {
                    "max_tokens": 8192,
                    "max_model_calls": 16,
                    "max_tool_calls": 24,
                    "max_test_runs": 4,
                    "max_repair_attempts": 3,
                    "task_timeout_seconds": 120,
                },
                "per_call_logs": [
                    {
                        "call_type": "model",
                        "status": "completed",
                        "token_accounting": {"usage_status": "measured"},
                    }
                ],
            }
        )
    runtime_observations = [
        {
            "record_type": "runtime_observation",
            "phase": phase,
            "observed_at": "2026-01-01T00:00:00+00:00",
            "ollama_processes": "test",
        }
        for phase in (
            "before_execution",
            "after_first_case",
            "after_execution",
        )
    ]
    completion = {
        "record_type": "evaluation_complete",
        "experiment_id": experiment_id,
        "status": "complete",
        "result_count": EXPECTED_RESULT_COUNT,
        "expected_result_count": EXPECTED_RESULT_COUNT,
        "scheduled_case_count_per_model": EXPECTED_RESULT_COUNT,
        "completed_models": [EXPECTED_MODEL],
        "repetitions": EXPECTED_REPETITIONS,
        "task_count": EXPECTED_TASK_COUNT,
        "strategies": list(EXPECTED_STRATEGIES),
        "random_seed": EXPECTED_SEED,
    }
    records = [metadata, *runtime_observations, *results, completion]
    path = tmp_path / "v2-held-out.jsonl"
    path.write_text(
        "\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8"
    )
    return path, records


def test_heldout_validator_accepts_complete_seeded_evidence(heldout_evidence):
    results_path, _ = heldout_evidence

    validated = validate_results(results_path)

    assert validated["result_count"] == 315
    assert len(validated["results"]) == EXPECTED_RESULT_COUNT
    assert len(validated["runtime_observations"]) == 3


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda rows: rows.pop(-2), "exactly 315 result records"),
        (
            lambda rows: rows[0]["randomized_execution_order"].pop(),
            "seeded schedule",
        ),
        (
            lambda rows: rows[0].__setitem__("config_sha256", "0" * 64),
            "config hash",
        ),
        (
            lambda rows: rows[-1].__setitem__("status", "running"),
            "status",
        ),
    ],
)
def test_heldout_validator_rejects_incomplete_or_mutated_evidence(
    tmp_path, heldout_evidence, mutation, message
):
    _, original_rows = heldout_evidence
    rows = json.loads(json.dumps(original_rows))
    mutation(rows)
    invalid_path = tmp_path / "invalid-heldout.jsonl"
    invalid_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match=message):
        validate_results(invalid_path)


def test_heldout_validator_rejects_duplicate_result_schedule_key(
    tmp_path, heldout_evidence
):
    _, original_rows = heldout_evidence
    rows = json.loads(json.dumps(original_rows))
    result_indices = [
        index for index, row in enumerate(rows) if row["record_type"] == "result"
    ]
    rows[result_indices[-1]] = dict(rows[result_indices[0]])
    invalid_path = tmp_path / "duplicate-heldout.jsonl"
    invalid_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Duplicate result schedule key"):
        validate_results(invalid_path)


def test_heldout_artifacts_require_valid_evidence_before_writing(
    tmp_path, heldout_evidence
):
    _, rows = heldout_evidence
    invalid_path = tmp_path / "invalid-heldout.jsonl"
    invalid_path.write_text(
        "\n".join(json.dumps(row) for row in [*rows[:-2], rows[-1]]) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "artifacts"

    with pytest.raises(ValueError, match="exactly 315 result records"):
        prepare_heldout_artifacts(invalid_path, output_dir)

    assert not output_dir.exists()


def test_heldout_artifact_generator_preserves_raw_log_and_prevents_overwrite(
    tmp_path, heldout_evidence
):
    raw_path, _ = heldout_evidence
    output_dir = tmp_path / "heldout-artifacts"

    prepare_heldout_artifacts(raw_path, output_dir)

    assert {
        path for path in ARTIFACT_FILES if (output_dir / path).is_file()
    } == set(ARTIFACT_FILES)
    assert (
        output_dir.joinpath("raw", "v2-held-out.jsonl").read_bytes()
        == raw_path.read_bytes()
    )
    summary = (output_dir / "summary.csv").read_text(encoding="utf-8")
    assert "strict_resolution_task_clustered_ci95_lower" in summary.splitlines()[0]
    assert "missing_result_token_usage_cases" in summary.splitlines()[0]
    readme = (output_dir / "README.md").read_text(encoding="utf-8")
    assert "no strategy" in readme
    with pytest.raises(FileExistsError, match="already exist"):
        prepare_heldout_artifacts(raw_path, output_dir)


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

    monkeypatch.setattr(
        run_experiment,
        "OllamaModel",
        lambda *args: FakeModel(),
    )
    monkeypatch.setattr(
        run_experiment,
        "_query_model_inventory",
        lambda *args: {"test-model": {"details": {}, "digest": "test-digest"}},
    )
    monkeypatch.setattr(run_experiment, "get_hardware_info", lambda: {"gpus": []})
    monkeypatch.setattr(run_experiment, "_ollama_server_version", lambda *args: "test")
    monkeypatch.setattr(run_experiment, "_ollama_cli_version", lambda: "test")
    monkeypatch.setattr(run_experiment, "_ollama_process_snapshot", lambda: "test process")
    received_cases = []

    async def fake_run_case(
        task, model, strategy, limits, experiment_id, **kwargs
    ):
        received_cases.append((task["id"], strategy, kwargs))
        return {
            "record_type": "result",
            "experiment_id": experiment_id,
            "task_id": task["id"],
            "strategy": strategy.value,
            "model": model.model,
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
    assert metadata["model_inventory"]["test-model"]["digest"] == "test-digest"
    assert metadata["ollama_server_version"] == "test"
    assert metadata["temperature"] == 0.1
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
    completion = next(
        row for row in outputs[0] if row["record_type"] == "evaluation_complete"
    )
    observations = [
        row for row in outputs[0] if row["record_type"] == "runtime_observation"
    ]
    assert len(results) == 30
    assert completion["status"] == "complete"
    assert completion["result_count"] == completion["expected_result_count"] == 30
    assert {row["phase"] for row in observations} == {
        "before_execution",
        "after_first_case",
        "after_execution",
    }
    assert not output_paths[0].with_name("first.jsonl.partial").exists()
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
async def test_experiment_runner_keeps_interrupted_output_partial(tmp_path, monkeypatch):
    task_path = tmp_path / "tasks.jsonl"
    task_path.write_text(
        "\n".join(
            json.dumps(
                {"id": f"t{i}", "goal": "complete task", "files": {"main.py": "x=1\n"}}
            )
            for i in range(6)
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        "model:\n  name: test-model\n  device: cpu\nbudgets:\n  max_repair_attempts: 1\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "interrupted.jsonl"

    class FakeModel:
        model = "test-model"
        device = "cpu"

        async def list_model_inventory(self):
            return {self.model: {"details": {}, "digest": "test-digest"}}

    async def fail_case(*args, **kwargs):
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(run_experiment, "OllamaModel", lambda *args: FakeModel())
    monkeypatch.setattr(
        run_experiment,
        "_query_model_inventory",
        lambda *args: {"test-model": {"details": {}, "digest": "test-digest"}},
    )
    monkeypatch.setattr(run_experiment, "get_hardware_info", lambda: {"gpus": []})
    monkeypatch.setattr(run_experiment, "_ollama_server_version", lambda *args: None)
    monkeypatch.setattr(run_experiment, "_ollama_cli_version", lambda: None)
    monkeypatch.setattr(run_experiment, "_ollama_process_snapshot", lambda: None)
    monkeypatch.setattr(run_experiment, "_run_case", fail_case)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        await run_experiment.run(
            task_path, output_path, config_path, split="pilot", random_seed=2
        )

    partial_path = output_path.with_name(f"{output_path.name}.partial")
    assert not output_path.exists()
    assert partial_path.exists()
    partial_records = [
        json.loads(line)
        for line in partial_path.read_text(encoding="utf-8").splitlines()
    ]
    assert not any(
        row.get("record_type") == "evaluation_complete" for row in partial_records
    )


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


@pytest.mark.asyncio
async def test_experiment_runner_refuses_to_reuse_partial_results(tmp_path, monkeypatch):
    output_path = tmp_path / "incomplete.jsonl"
    partial_path = output_path.with_name(f"{output_path.name}.partial")
    original = "keep incomplete evidence for inspection\n"
    partial_path.write_text(original, encoding="utf-8")
    monkeypatch.setattr(
        run_experiment,
        "OllamaModel",
        lambda *args: pytest.fail("Model setup should not run for partial output."),
    )

    with pytest.raises(FileExistsError, match="incomplete experiment file"):
        await run_experiment.run(
            tmp_path / "missing-tasks.jsonl",
            output_path,
            tmp_path / "missing-config.yaml",
        )

    assert partial_path.read_text(encoding="utf-8") == original


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


def test_evaluator_excludes_budget_rejected_model_calls_from_usage_coverage(tmp_path):
    results_file = tmp_path / "rejected-model-calls.jsonl"
    row = {
        "record_type": "result",
        "task_id": "t1",
        "model": "local-model",
        "strategy": "failure-guided",
        "resolved": False,
        "budget": {
            "model_calls": 1,
            "budget_events": [
                {
                    "event": "budget_rejected",
                    "budget_kind": "tokens",
                    "reason": "estimated prompt leaves no token budget for model output",
                }
            ],
        },
        "per_call_logs": [
            {
                "call_type": "model",
                "status": "completed",
                "token_accounting": {"usage_status": "measured"},
            },
            {
                "call_type": "model",
                "status": "budget_rejected",
                "validation_outcome": "not_called",
            },
        ],
    }
    results_file.write_text(json.dumps(row) + "\n", encoding="utf-8")

    summary = evaluate(results_file)["runs"][0]
    token_usage = summary["token_usage"]

    assert token_usage["model_calls_logged"] == 2
    assert token_usage["provider_call_attempts"] == 1
    assert token_usage["budget_rejected_model_call_attempts"] == 1
    assert token_usage["missing_model_usage_calls"] == 0
    assert token_usage["reported_usage_call_coverage"] == 1.0
    assert token_usage["fully_measured_usage_call_coverage"] == 1.0
    assert summary["budget_rejection_events"] == 1


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


def test_evaluator_bootstraps_task_clusters_and_pairs_repetitions(tmp_path):
    results_file = tmp_path / "clustered.jsonl"
    rows = [{"record_type": "run_metadata", "repetitions": 2}]
    observations = [
        ("single-shot", "task-a", "rep-002", False, 2),
        ("fixed-retry", "task-a", "rep-001", False, 3),
        ("single-shot", "task-b", "rep-001", False, None),
        ("failure-guided", "task-a", "rep-001", True, 4),
        ("fixed-retry", "task-a", "rep-002", True, 5),
        ("single-shot", "task-a", "rep-001", True, 1),
        ("fixed-retry", "task-b", "rep-001", True, 6),
        ("failure-guided", "task-b", "rep-001", True, 7),
    ]
    for strategy, task_id, repetition_id, resolved, tokens in observations:
        budget = {"elapsed_seconds": 1.0}
        if tokens is not None:
            budget["tokens_used"] = tokens
        rows.append(
            {
                "record_type": "result",
                "model": "local-model",
                "strategy": strategy,
                "task_id": task_id,
                "repetition_id": repetition_id,
                "resolved": resolved,
                "final_test_passed": resolved,
                "repair_attempts": 0,
                "budget": budget,
            }
        )
    results_file.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    report = evaluate(results_file, bootstrap_seed=19)
    repeat_report = evaluate(results_file, bootstrap_seed=19)
    single = next(
        item
        for item in report["statistical_analysis"]["strategy_estimates"]
        if item["strategy"] == "single-shot"
    )
    paired = next(
        item
        for item in report["paired_comparisons"]
        if item["strategy_a"] == "single-shot"
        and item["strategy_b"] == "fixed-retry"
    )

    assert single["metrics"]["strict_resolution"]["n_tasks"] == 2
    assert single["metrics"]["strict_resolution"]["mean"] == pytest.approx(0.25)
    assert single["metrics"]["tokens_used"]["missing_tasks"] == 1
    assert report["statistical_analysis"]["bootstrap_samples"] >= 10_000
    assert report["statistical_analysis"] == repeat_report["statistical_analysis"]
    assert paired["matched_task_count"] == 2
    assert paired["matched_repetition_count"] == 3
    assert (
        paired["mean_difference_across_paired_cases"]["strict_resolution"]["mean"]
        == pytest.approx(-0.5)
    )
    assert (
        paired["mean_difference_across_paired_cases"]["strict_resolution"][
            "tasks_favoring_strategy_a"
        ]
        == 0
    )
    assert (
        paired["mean_difference_across_paired_cases"]["strict_resolution"][
            "tasks_favoring_strategy_b"
        ]
        == 1
    )
    assert (
        paired["mean_difference_across_paired_cases"]["strict_resolution"][
            "tasks_tied"
        ]
        == 1
    )


def test_evaluator_rejects_duplicate_task_repetition_observations(tmp_path):
    results_file = tmp_path / "duplicate.jsonl"
    row = {
        "record_type": "result",
        "model": "local-model",
        "strategy": "single-shot",
        "task_id": "task-a",
        "repetition_id": "rep-001",
        "resolved": True,
    }
    results_file.write_text(
        f"{json.dumps(row)}\n{json.dumps(row)}\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Duplicate result observation"):
        evaluate(results_file)


def test_evaluator_requires_ten_thousand_bootstrap_resamples(tmp_path):
    results_file = tmp_path / "empty.jsonl"
    results_file.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="at least 10000"):
        evaluate(results_file, bootstrap_samples=9999)
