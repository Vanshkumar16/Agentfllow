"""Strictly validate the frozen v2 held-out experiment JSONL evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any


EXPECTED_DATASET_SHA256 = (
    "dece185db26d52aadce87df77a9b3726dc529c4f0b16a543f455c112160b1a39"
)
EXPECTED_CONFIG_SHA256 = (
    "2008d3eb447ee98b9212078c6cadd4f5b8e1064f13fda853cc73dc55802620c9"
)
EXPECTED_MODEL = "gemma4:e2b"
EXPECTED_REPETITIONS = 3
EXPECTED_SEED = 20261113
EXPECTED_TASK_COUNT = 35
EXPECTED_STRATEGIES = ("single-shot", "fixed-retry", "failure-guided")
EXPECTED_RESULT_COUNT = (
    EXPECTED_TASK_COUNT * EXPECTED_REPETITIONS * len(EXPECTED_STRATEGIES)
)
ALLOWED_RECORD_TYPES = {
    "run_metadata",
    "runtime_observation",
    "result",
    "evaluation_complete",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_number} must be a JSON object.")
            rows.append(row)
    return rows


def _expected_schedule(task_ids: list[str]) -> list[dict[str, Any]]:
    generator = random.Random(EXPECTED_SEED)
    schedule: list[dict[str, Any]] = []
    for repetition in range(1, EXPECTED_REPETITIONS + 1):
        cases = [
            (task_id, strategy)
            for task_id in task_ids
            for strategy in EXPECTED_STRATEGIES
        ]
        generator.shuffle(cases)
        repetition_id = f"rep-{repetition:03d}"
        schedule.extend(
            {
                "repetition_id": repetition_id,
                "execution_order": execution_order,
                "task_id": task_id,
                "strategy": strategy,
            }
            for execution_order, (task_id, strategy) in enumerate(cases, start=1)
        )
    return schedule


def validate_results(
    results_path: Path, *, repository_root: Path | None = None
) -> dict[str, Any]:
    root = repository_root or Path(__file__).resolve().parents[1]
    dataset_path = root / "benchmarks" / "tasks_v2.jsonl"
    config_path = root / "configs" / "experiment_v2.yaml"
    if _sha256(dataset_path) != EXPECTED_DATASET_SHA256:
        raise ValueError("The checked-out v2 dataset does not match the frozen SHA-256.")
    if _sha256(config_path) != EXPECTED_CONFIG_SHA256:
        raise ValueError("The checked-out v2 config does not match the frozen SHA-256.")

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to validate the frozen config.") from exc
    config = yaml.safe_load(config_path.read_bytes())
    if not isinstance(config, dict):
        raise ValueError("The frozen config must contain a YAML object.")
    model_config = config.get("model")
    if not isinstance(model_config, dict) or model_config.get("name") != EXPECTED_MODEL:
        raise ValueError("The frozen config does not specify the expected model.")

    task_rows = [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    task_ids = [
        row["id"] for row in task_rows if isinstance(row, dict) and row.get("split") == "held-out"
    ]
    if len(task_ids) != EXPECTED_TASK_COUNT or len(set(task_ids)) != EXPECTED_TASK_COUNT:
        raise ValueError("The frozen dataset must contain exactly 35 unique held-out tasks.")

    rows = _read_jsonl(results_path)
    if not rows:
        raise ValueError("The held-out results file is empty.")
    record_types = [row.get("record_type") for row in rows]
    unexpected_types = set(record_types) - ALLOWED_RECORD_TYPES
    if unexpected_types:
        raise ValueError(
            f"Unexpected JSONL record types: {sorted(unexpected_types, key=str)}."
        )
    metadata_rows = [row for row in rows if row.get("record_type") == "run_metadata"]
    completion_rows = [
        row for row in rows if row.get("record_type") == "evaluation_complete"
    ]
    result_rows = [row for row in rows if row.get("record_type") == "result"]
    observations = [
        row for row in rows if row.get("record_type") == "runtime_observation"
    ]
    if len(metadata_rows) != 1 or rows[0] is not metadata_rows[0]:
        raise ValueError("The file must start with exactly one run_metadata record.")
    if len(completion_rows) != 1 or rows[-1] is not completion_rows[0]:
        raise ValueError("The file must end with exactly one evaluation_complete record.")
    if len(result_rows) != EXPECTED_RESULT_COUNT:
        raise ValueError(
            f"Expected exactly {EXPECTED_RESULT_COUNT} result records, "
            f"found {len(result_rows)}."
        )

    metadata = metadata_rows[0]
    if metadata.get("split") != "held-out":
        raise ValueError("Run metadata split must be 'held-out'.")
    if metadata.get("task_count") != EXPECTED_TASK_COUNT:
        raise ValueError("Run metadata task_count must equal 35.")
    if metadata.get("repetitions") != EXPECTED_REPETITIONS:
        raise ValueError("Run metadata repetitions must equal 3.")
    if metadata.get("random_seed") != EXPECTED_SEED:
        raise ValueError("Run metadata random_seed must equal 20261113.")
    if metadata.get("dataset_sha256") != EXPECTED_DATASET_SHA256:
        raise ValueError("Run metadata dataset hash does not match the frozen dataset.")
    if metadata.get("config_sha256") != EXPECTED_CONFIG_SHA256:
        raise ValueError("Run metadata config hash does not match the frozen config.")
    if metadata.get("primary_model") != EXPECTED_MODEL:
        raise ValueError(f"Run metadata primary_model must be {EXPECTED_MODEL}.")
    if metadata.get("budgets") != config.get("budgets"):
        raise ValueError("Run metadata budgets do not match the frozen configuration.")
    if metadata.get("max_actions_per_attempt") != config.get(
        "max_actions_per_attempt", 8
    ):
        raise ValueError("Run metadata max_actions_per_attempt differs from config.")
    if metadata.get("max_prompt_chars") != config.get("max_prompt_chars", 24_000):
        raise ValueError("Run metadata max_prompt_chars differs from config.")
    if metadata.get("device_requested") != model_config.get("device", "auto"):
        raise ValueError("Run metadata requested device differs from config.")
    if metadata.get("temperature") != 0.1:
        raise ValueError("Run metadata temperature must be the frozen value 0.1.")
    if not isinstance(metadata.get("ollama_server_version"), str) or not metadata[
        "ollama_server_version"
    ]:
        raise ValueError("Run metadata must record the Ollama server version.")
    if not isinstance(metadata.get("ollama_cli_version"), str) or not metadata[
        "ollama_cli_version"
    ]:
        raise ValueError("Run metadata must record the Ollama CLI version.")
    if metadata.get("git_worktree_clean") is not True:
        raise ValueError("Held-out evidence must record a clean Git worktree.")
    commit_hash = metadata.get("git_commit_hash")
    if not isinstance(commit_hash, str) or re.fullmatch(r"[0-9a-f]{40}", commit_hash) is None:
        raise ValueError("Run metadata must contain a full 40-character Git commit hash.")
    inventory = metadata.get("model_inventory")
    model_record = inventory.get(EXPECTED_MODEL) if isinstance(inventory, dict) else None
    digest = model_record.get("digest") if isinstance(model_record, dict) else None
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("Run metadata must record the installed model's SHA-256 digest.")
    if not isinstance(metadata.get("experiment_id"), str) or not metadata["experiment_id"]:
        raise ValueError("Run metadata must contain an experiment_id.")
    hardware = metadata.get("hardware")
    if (
        not isinstance(hardware, dict)
        or not isinstance(hardware.get("platform"), str)
        or not isinstance(hardware.get("python"), str)
        or not isinstance(hardware.get("nvidia_smi_available"), bool)
        or not isinstance(hardware.get("gpus"), list)
    ):
        raise ValueError("Run metadata must contain complete platform and GPU metadata.")

    expected_schedule = _expected_schedule(task_ids)
    if metadata.get("strategies") != list(EXPECTED_STRATEGIES):
        raise ValueError("Run metadata strategies do not match the frozen protocol.")
    if metadata.get("randomized_execution_order") != expected_schedule:
        raise ValueError("Run metadata execution schedule is not the frozen seeded schedule.")
    expected_by_key = {
        (
            item["repetition_id"],
            item["execution_order"],
        ): (item["task_id"], item["strategy"])
        for item in expected_schedule
    }
    observed_keys: set[tuple[str, int]] = set()
    for row in result_rows:
        repetition_id = row.get("repetition_id")
        execution_order = row.get("execution_order")
        if (
            not isinstance(repetition_id, str)
            or not isinstance(execution_order, int)
            or isinstance(execution_order, bool)
        ):
            raise ValueError("Each result must record repetition_id and execution_order.")
        key = (repetition_id, execution_order)
        if key in observed_keys:
            raise ValueError(f"Duplicate result schedule key: {key}.")
        observed_keys.add(key)
        if key not in expected_by_key:
            raise ValueError(f"Unexpected result schedule key: {key}.")
        if (row.get("task_id"), row.get("strategy")) != expected_by_key[key]:
            raise ValueError(f"Result identity does not match scheduled case {key}.")
        if row.get("model") != EXPECTED_MODEL:
            raise ValueError(f"Result {key} has an unexpected model.")
        if row.get("experiment_id") != metadata["experiment_id"]:
            raise ValueError(f"Result {key} has a mismatched experiment_id.")
        if row.get("random_seed") != EXPECTED_SEED:
            raise ValueError(f"Result {key} has an unexpected random_seed.")
        if not isinstance(row.get("resolved"), bool):
            raise ValueError(f"Result {key} must report boolean resolved.")
        if not isinstance(row.get("final_test_passed"), bool):
            raise ValueError(f"Result {key} must report boolean final_test_passed.")
    if observed_keys != set(expected_by_key):
        raise ValueError(
            f"Result schedule is incomplete: {len(observed_keys)}/"
            f"{len(expected_by_key)} scheduled cases."
        )

    completion = completion_rows[0]
    expected_completion = {
        "status": "complete",
        "result_count": EXPECTED_RESULT_COUNT,
        "expected_result_count": EXPECTED_RESULT_COUNT,
        "scheduled_case_count_per_model": EXPECTED_RESULT_COUNT,
        "completed_models": [EXPECTED_MODEL],
        "repetitions": EXPECTED_REPETITIONS,
        "task_count": EXPECTED_TASK_COUNT,
        "strategies": list(EXPECTED_STRATEGIES),
        "random_seed": EXPECTED_SEED,
        "experiment_id": metadata["experiment_id"],
    }
    for key, expected_value in expected_completion.items():
        if completion.get(key) != expected_value:
            raise ValueError(
                f"Completion record field '{key}' must equal {expected_value!r}."
            )
    expected_phases = {"before_execution", "after_first_case", "after_execution"}
    phases = [row.get("phase") for row in observations]
    if len(phases) != len(expected_phases) or set(phases) != expected_phases:
        raise ValueError("Expected one runtime observation for each run phase.")

    return {
        "metadata": metadata,
        "completion": completion,
        "results": result_rows,
        "runtime_observations": observations,
        "result_count": len(result_rows),
        "raw_log_sha256": _sha256(results_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results", nargs="?", type=Path, default=Path("results/v2-held-out.jsonl")
    )
    args = parser.parse_args()
    try:
        result = validate_results(args.results)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"held-out validation: {type(exc).__name__}: {exc}\n")
    print(
        f"Held-out evidence validation passed: {result['result_count']}/"
        f"{EXPECTED_RESULT_COUNT} result records; SHA-256 "
        f"{result['raw_log_sha256']}."
    )


if __name__ == "__main__":
    main()
