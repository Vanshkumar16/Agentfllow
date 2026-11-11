"""Run a frozen, randomized final evaluation on the held-out benchmark tasks."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import random
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import ProxyHandler, build_opener

from agentflow.budget import BudgetLimits
from agentflow.coding_agent import RecoveryStrategy
from agentflow.local_model import (
    GENERATION_TEMPERATURE,
    LocalModelError,
    OllamaModel,
)

if __package__:
    from .run_experiment import (
        STRATEGIES,
        _run_case,
        _select_tasks,
        get_hardware_info,
        load_tasks,
    )
else:
    from run_experiment import (
        STRATEGIES,
        _run_case,
        _select_tasks,
        get_hardware_info,
        load_tasks,
    )

EXPECTED_MODEL = "gemma4:e2b"
EXPECTED_SPLIT = "held-out"
EXPECTED_REPAIR_ATTEMPTS = 3
EXPECTED_TIMEOUT_SECONDS = 120
EXPECTED_TOKEN_BUDGET = 8192
DEFAULT_RANDOM_SEED = 20261111
FROZEN_AGENT_REVISION = "7e1ab1ea56f93136c1c06dddf22f6b598d263288"


def _git(command: list[str]) -> str:
    completed = subprocess.run(
        ["git", *command],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=10,
        shell=False,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Git command failed: {' '.join(command)}: {completed.stderr.strip()}")
    return completed.stdout.strip()


def _assert_frozen_agent(revision: str) -> str:
    if _git(["merge-base", "--is-ancestor", FROZEN_AGENT_REVISION, revision]) != "":
        raise RuntimeError("The frozen agent revision is not an ancestor of HEAD.")
    comparison = subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            FROZEN_AGENT_REVISION,
            "--",
            "src/agentflow",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=10,
        shell=False,
        check=False,
    )
    if comparison.returncode != 0:
        raise RuntimeError(
            "Agent sources differ from the frozen implementation revision "
            f"{FROZEN_AGENT_REVISION}."
        )
    return FROZEN_AGENT_REVISION


def _ollama_server_version(base_url: str) -> str | None:
    try:
        opener = build_opener(ProxyHandler({}))
        with opener.open(f"{base_url.rstrip('/')}/api/version", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, TimeoutError, json.JSONDecodeError):
        return None
    version = payload.get("version") if isinstance(payload, dict) else None
    return version if isinstance(version, str) else None


def _ollama_cli_version() -> str | None:
    try:
        completed = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            shell=False,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or completed.stderr.strip() or None


def _gpu_driver_info() -> list[dict[str, str]]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3,
            shell=False,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if completed.returncode != 0:
        return []
    result = []
    for line in completed.stdout.splitlines():
        name, separator, driver = line.partition(",")
        if separator:
            result.append({"name": name.strip(), "driver_version": driver.strip()})
    return result


def _ollama_process_list() -> str | None:
    try:
        completed = subprocess.run(
            ["ollama", "ps"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            shell=False,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def build_schedule(
    tasks: list[dict[str, Any]], random_seed: int
) -> list[tuple[dict[str, Any], RecoveryStrategy]]:
    cases = [(task, strategy) for task in tasks for strategy in STRATEGIES]
    random.Random(random_seed).shuffle(cases)
    return cases


def _validate_frozen_config(
    model_config: dict[str, Any], limits: BudgetLimits
) -> None:
    if str(model_config.get("name", "")) != EXPECTED_MODEL:
        raise ValueError(f"Final evaluation requires model '{EXPECTED_MODEL}'.")
    if limits.max_repair_attempts != EXPECTED_REPAIR_ATTEMPTS:
        raise ValueError("Final evaluation requires max_repair_attempts=3.")
    if limits.task_timeout_seconds != EXPECTED_TIMEOUT_SECONDS:
        raise ValueError("Final evaluation requires task_timeout_seconds=120.")
    if limits.max_tokens != EXPECTED_TOKEN_BUDGET:
        raise ValueError("Final evaluation requires max_tokens=8192.")


async def run(
    tasks_path: Path,
    config_path: Path,
    output_path: Path,
    random_seed: int,
) -> None:
    import yaml

    revision = _git(["rev-parse", "HEAD"])
    frozen_agent_revision = _assert_frozen_agent(revision)
    dirty_state = _git(["status", "--porcelain", "--untracked-files=normal"])
    if dirty_state:
        raise RuntimeError(
            "Final evaluation requires a clean frozen Git revision. "
            "Commit or remove pending changes before running."
        )

    config_bytes = config_path.read_bytes()
    dataset_bytes = tasks_path.read_bytes()
    try:
        config = yaml.safe_load(config_bytes)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in config file '{config_path}': {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"Config file '{config_path}' must contain a YAML object.")

    model_config = config.get("model", {})
    budget_config = config.get("budgets", {})
    if not isinstance(model_config, dict) or not isinstance(budget_config, dict):
        raise ValueError("Config 'model' and 'budgets' values must be YAML objects.")
    limits = BudgetLimits(**budget_config)
    _validate_frozen_config(model_config, limits)
    model = OllamaModel(
        EXPECTED_MODEL,
        str(model_config.get("base_url", "http://localhost:11434")),
        str(model_config.get("device", "auto")),
    )
    installed_models = await model.list_model_details()
    if model.model not in installed_models:
        raise LocalModelError(f"Required model '{model.model}' is not installed.")

    tasks = _select_tasks(load_tasks(tasks_path), EXPECTED_SPLIT)
    if len(tasks) != 15:
        raise ValueError(f"Expected 15 held-out tasks, found {len(tasks)}.")
    cases = build_schedule(tasks, random_seed)
    order = [
        {"order": index, "task_id": task["id"], "strategy": strategy.value}
        for index, (task, strategy) in enumerate(cases, start=1)
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite evaluation results: {output_path}"
        )

    hardware = get_hardware_info()
    metadata = {
        "record_type": "run_metadata",
        "evaluation": "frozen-final",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": EXPECTED_SPLIT,
        "task_count": len(tasks),
        "case_count": len(cases),
        "pilot_tasks_previously_used": [
            task["id"] for task in _select_tasks(load_tasks(tasks_path), "pilot")
        ],
        "pilot_tasks_excluded_from_final_evaluation": True,
        "random_seed": random_seed,
        "randomized_execution_order": order,
        "agent_git_revision": frozen_agent_revision,
        "evaluation_harness_revision": revision,
        "git_worktree_clean_before_run": True,
        "dataset_path": str(tasks_path),
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "fresh_workspace_policy": (
            "Each case receives a new temporary workspace populated from the "
            "same immutable benchmark task fixture; no modified workspace is reused."
        ),
        "test_command": [sys.executable, "-m", "pytest", "-q"],
        "model": model.model,
        "model_runtime_details": installed_models[model.model],
        "ollama_cli_version": _ollama_cli_version(),
        "ollama_server_version": _ollama_server_version(model.base_url),
        "generation_settings": {
            "temperature": GENERATION_TEMPERATURE,
            "num_predict": (
                "per-call output reservation, capped by "
                f"{limits.max_output_tokens_per_call}"
            ),
            "stream": False,
            "device_requested": model.device,
            "device_selected": None,
            "device_selection_note": (
                "Ollama does not provide effective device placement in the AgentFlow response."
            ),
            "random_seed_applied_to_model": None,
        },
        "token_estimation": (
            "max(ceil(prompt_characters / 2), ceil(prompt_utf8_bytes / 3))"
        ),
        "budgets": asdict(limits),
        "hardware": hardware,
        "gpu_driver": _gpu_driver_info(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "strategies": [strategy.value for strategy in STRATEGIES],
    }

    with output_path.open("x", encoding="utf-8") as output_file:
        output_file.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        output_file.flush()
        for index, (task, strategy) in enumerate(cases, start=1):
            row = await _run_case(task, model, strategy, limits)
            row["evaluation_order"] = index
            row["random_seed"] = random_seed
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            output_file.flush()
            print(
                f"[{index}/{len(cases)}] {task['id']} / {strategy.value}: "
                f"resolved={row['resolved']}",
                flush=True,
            )
        output_file.write(
            json.dumps(
                {
                    "record_type": "evaluation_complete",
                    "completed_case_count": len(cases),
                    "expected_case_count": len(cases),
                }
            )
            + "\n"
        )
        output_file.flush()
        output_file.write(
            json.dumps(
                {
                    "record_type": "runtime_observation",
                    "phase": "after-evaluation",
                    "ollama_process_list": _ollama_process_list(),
                    "hardware": get_hardware_info(),
                    "gpu_driver": _gpu_driver_info(),
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen, randomized held-out AgentFlow evaluation."
    )
    parser.add_argument("--tasks", type=Path, default=Path("benchmarks/tasks.jsonl"))
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiment.yaml")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/final-held-out.jsonl")
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    args = parser.parse_args()
    try:
        asyncio.run(run(args.tasks, args.config, args.output, args.seed))
    except (FileExistsError, LocalModelError, OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"final evaluation: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
