"""Run controlled Python repair tasks across recovery strategies."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import platform
import re
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from agentflow.budget import BudgetLimits
from agentflow.coding_agent import CodingAgent, CodingAgentConfig, RecoveryStrategy
from agentflow.local_model import LocalModelError, OllamaModel
from agentflow.workspace import WorkspaceTools

STRATEGIES = (
    RecoveryStrategy.SINGLE_SHOT,
    RecoveryStrategy.FIXED_RETRY,
    RecoveryStrategy.FAILURE_GUIDED,
)


def load_tasks(path: Path) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                task = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(task, dict):
                raise ValueError(f"Benchmark line {line_number} must be a JSON object.")
            task_id, goal, files = task.get("id"), task.get("goal"), task.get("files")
            if not isinstance(task_id, str) or not task_id or task_id in seen_ids:
                raise ValueError(f"Benchmark line {line_number} has a missing or duplicate id.")
            if not isinstance(goal, str) or not goal.strip():
                raise ValueError(f"Benchmark task '{task_id}' has no non-empty goal.")
            if not isinstance(files, dict) or not files:
                raise ValueError(f"Benchmark task '{task_id}' must include a files object.")
            if any(not isinstance(name, str) or not isinstance(text, str) for name, text in files.items()):
                raise ValueError(f"Benchmark task '{task_id}' files must map paths to text.")
            seen_ids.add(task_id)
            tasks.append(task)
    if len(tasks) < 6:
        raise ValueError("Benchmark needs at least 6 tasks to provide a 5-task pilot and held-out split.")
    return tasks


def get_hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor() or None,
        "python": platform.python_version(),
        "nvidia_smi_available": False,
        "gpus": [],
    }
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
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
        return info
    if completed.returncode != 0:
        return info
    info["nvidia_smi_available"] = True
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            total_mib, free_mib = int(parts[1]), int(parts[2])
        except ValueError:
            continue
        info["gpus"].append(
            {
                "name": parts[0],
                "total_vram_mib": total_mib,
                "free_vram_mib": free_mib,
            }
        )
    return info


def _write_task_files(root: Path, files: dict[str, str]) -> None:
    resolved_root = root.resolve()
    for relative, content in files.items():
        target = (resolved_root / relative).resolve()
        try:
            target.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError(f"Benchmark file path escapes task workspace: {relative}") from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


async def _run_case(
    task: dict[str, Any],
    model: OllamaModel,
    strategy: RecoveryStrategy,
    limits: BudgetLimits,
) -> dict[str, Any]:
    with TemporaryDirectory(prefix=f"agentflow-{task['id']}-") as temp_dir:
        workspace_path = Path(temp_dir)
        _write_task_files(workspace_path, task["files"])
        workspace = WorkspaceTools(
            workspace_path,
            allow_tests=True,
            test_command=(sys.executable, "-m", "pytest", "-q"),
        )
        agent = CodingAgent(
            model,
            workspace,
            CodingAgentConfig(strategy=strategy, budgets=limits),
        )
        result = await agent.run(task["goal"])
        return {
            "record_type": "result",
            "task_id": task["id"],
            "model": model.model,
            "device_requested": model.device,
            "device_selected": None,
            "device_selection_note": "The local model server does not report the effective device.",
            **result.to_dict(),
        }


def _select_tasks(tasks: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    if split == "pilot":
        return tasks[:5]
    if split == "held-out":
        return tasks[5:]
    if split == "all":
        return tasks
    raise ValueError(f"Unknown benchmark split '{split}'.")


def _optional_model_skips(
    candidates: list[dict[str, Any]],
    installed_models: dict[str, dict[str, object]],
    hardware: dict[str, Any],
    device: str,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    runnable: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    gpus = hardware.get("gpus", [])
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("Each optional E4B model must be a YAML object.")
        name = candidate.get("name")
        minimum_free = candidate.get("minimum_free_vram_gb")
        if not isinstance(name, str) or not name.strip():
            skipped.append({"model": str(name), "reason": "candidate has no model name"})
            continue
        if device == "cpu" or len(gpus) != 1:
            skipped.append(
                {
                    "model": name,
                    "reason": "optional E4B evaluation requires auto device mode and exactly one visible NVIDIA GPU",
                }
            )
            continue
        details = installed_models.get(name)
        if details is None:
            skipped.append({"model": name, "reason": "model is not already installed; automatic downloads are disabled"})
            continue
        parameter_size = details.get("parameter_size")
        if not isinstance(parameter_size, str) or re.fullmatch(
            r"4(?:\.\d+)?\s*B", parameter_size.strip(), flags=re.IGNORECASE
        ) is None:
            skipped.append({"model": name, "reason": "runtime metadata does not verify a 4B parameter size"})
            continue
        quantization = details.get("quantization_level")
        if not isinstance(quantization, str) or re.match(
            r"^Q\d", quantization.strip(), flags=re.IGNORECASE
        ) is None:
            skipped.append({"model": name, "reason": "runtime metadata does not verify a quantized model"})
            continue
        family = details.get("family")
        if "gemma4" not in name.lower() and (
            not isinstance(family, str) or "gemma4" not in family.lower()
        ):
            skipped.append({"model": name, "reason": "runtime metadata does not identify a Gemma 4 model"})
            continue
        try:
            if isinstance(minimum_free, bool):
                raise ValueError("minimum_free_vram_gb cannot be a boolean")
            minimum_gb = float(minimum_free)
            required_mib = int(minimum_gb * 1024)
        except (TypeError, ValueError, OverflowError):
            skipped.append({"model": name, "reason": "minimum_free_vram_gb is missing or invalid"})
            continue
        if not math.isfinite(minimum_gb) or minimum_gb <= 0:
            skipped.append({"model": name, "reason": "minimum_free_vram_gb must be positive"})
            continue
        free_mib = gpus[0]["free_vram_mib"]
        if free_mib < required_mib:
            skipped.append(
                {
                    "model": name,
                    "reason": f"free VRAM {free_mib} MiB is below configured threshold {required_mib} MiB",
                }
            )
            continue
        runnable.append(candidate)
    return runnable, skipped


async def run(
    tasks_path: Path,
    output_path: Path,
    config_path: Path,
    split: str = "pilot",
) -> None:
    import yaml

    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(f"Could not read config file '{config_path}': {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in config file '{config_path}': {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"Config file '{config_path}' must contain a YAML object.")

    model_config = config.get("model", {})
    budget_config = config.get("budgets", {})
    if not isinstance(model_config, dict) or not isinstance(budget_config, dict):
        raise ValueError("Config 'model' and 'budgets' values must be YAML objects.")
    limits = BudgetLimits(**budget_config)
    device = str(model_config.get("device", "auto"))
    base_url = str(model_config.get("base_url", "http://localhost:11434"))
    primary = OllamaModel(str(model_config.get("name", "gemma4:e2b")), base_url, device)
    primary_error: str | None = None
    try:
        installed_models = await primary.list_model_details()
    except LocalModelError as exc:
        installed_models = {}
        primary_error = str(exc)
    if primary_error is None and primary.model not in installed_models:
        primary_error = (
            f"Primary model '{primary.model}' is not installed at {base_url}. "
            f"Available models: {', '.join(installed_models) or '(none)'}. "
            "No model was downloaded."
        )
    hardware = get_hardware_info()
    candidates = config.get("optional_e4b_models", [])
    if not isinstance(candidates, list):
        raise ValueError("'optional_e4b_models' must be a YAML list.")
    runnable_optional, skipped_optional = _optional_model_skips(
        candidates, installed_models, hardware, device
    )
    if primary_error is not None:
        skipped_optional.extend(
            {
                "model": str(candidate.get("name", "(unnamed)")),
                "reason": "primary E2B baseline is unavailable; optional-model comparison was not run",
            }
            for candidate in candidates
            if isinstance(candidate, dict)
        )
        runnable_optional = []
    model_runs = ([(primary, None)] if primary_error is None else []) + [
        (OllamaModel(str(candidate["name"]), base_url, device), candidate)
        for candidate in runnable_optional
    ]
    if primary_error is not None:
        skipped_optional.insert(
            0, {"model": primary.model, "reason": primary_error}
        )

    selected_tasks = _select_tasks(load_tasks(tasks_path), split)
    if not selected_tasks:
        raise ValueError(f"No tasks selected for split '{split}'.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        metadata = {
            "record_type": "run_metadata",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset": str(tasks_path),
            "split": split,
            "task_count": len(selected_tasks),
            "strategies": [strategy.value for strategy in STRATEGIES],
            "hardware": hardware,
            "primary_model": primary.model,
            "primary_model_status": "available" if primary_error is None else "unavailable",
            "device_requested": device,
            "device_selected": None,
            "device_selection_note": "The local model server does not report the effective device.",
            "budgets": asdict(limits),
            "model_skips": skipped_optional,
        }
        output_file.write(json.dumps(metadata) + "\n")
        for item in skipped_optional:
            output_file.write(json.dumps({"record_type": "model_skipped", **item}) + "\n")
        output_file.flush()
        for model, optional_candidate in model_runs:
            if optional_candidate is not None:
                current_hardware = get_hardware_info()
                still_fits, current_skips = _optional_model_skips(
                    [optional_candidate],
                    installed_models,
                    current_hardware,
                    device,
                )
                if not still_fits:
                    skipped_row = {
                        "record_type": "model_skipped",
                        "model": model.model,
                        "reason": current_skips[0]["reason"],
                    }
                    output_file.write(json.dumps(skipped_row) + "\n")
                    output_file.flush()
                    continue
                output_file.write(
                    json.dumps(
                        {
                            "record_type": "run_metadata",
                            "phase": "optional-model-preflight",
                            "model": model.model,
                            "hardware": current_hardware,
                        }
                    )
                    + "\n"
                )
            for task in selected_tasks:
                for strategy in STRATEGIES:
                    row = await _run_case(task, model, strategy, limits)
                    output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                    output_file.flush()
    if primary_error is not None:
        raise LocalModelError(primary_error)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare AgentFlow recovery strategies.")
    parser.add_argument("--tasks", type=Path, default=Path("benchmarks/tasks.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/latest.jsonl"))
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--split", choices=("pilot", "held-out", "all"), default="pilot")
    args = parser.parse_args()
    try:
        asyncio.run(run(args.tasks, args.output, args.config, args.split))
    except (LocalModelError, OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"experiment: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
