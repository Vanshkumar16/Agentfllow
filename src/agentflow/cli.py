"""Command-line entry point for local code repair."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import yaml

from .budget import BudgetLimits
from .coding_agent import CodingAgent, CodingAgentConfig, RecoveryStrategy
from .local_model import LocalModelError, OllamaModel
from .workspace import WorkspaceError, WorkspaceTools


def _read_config(path: Path) -> dict[str, Any]:
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(f"Could not read config file '{path}': {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in config file '{path}': {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"Config file '{path}' must contain a YAML object.")
    return config


def _apply_budget_overrides(
    config: dict[str, Any], args: argparse.Namespace
) -> BudgetLimits:
    overrides = {
        "max_model_calls": args.max_model_calls,
        "max_tool_calls": args.max_tool_calls,
        "max_test_runs": args.max_test_runs,
        "max_tokens": args.max_tokens,
        "max_repair_attempts": args.max_repair_attempts,
        "task_timeout_seconds": args.timeout,
        "command_timeout_seconds": args.command_timeout,
    }
    configured_budgets = config.get("budgets", {})
    if not isinstance(configured_budgets, dict):
        raise ValueError("Config 'budgets' must be a YAML object.")
    values = dict(configured_budgets)
    values.update({key: value for key, value in overrides.items() if value is not None})
    return BudgetLimits(**values)


async def _run(args: argparse.Namespace) -> int:
    config = _read_config(args.config)
    model_config = config.get("model", {})
    workspace_config = config.get("workspace", {})
    if not isinstance(model_config, dict) or not isinstance(workspace_config, dict):
        raise ValueError("Config 'model' and 'workspace' values must be YAML objects.")
    model_name = args.model or str(model_config.get("name", "gemma4:e2b"))
    base_url = args.base_url or str(model_config.get("base_url", "http://localhost:11434"))
    device = args.device or str(model_config.get("device", "auto"))
    model = OllamaModel(model_name, base_url, device)
    await model.check_available()

    workspace_path = args.workspace or Path(str(workspace_config.get("path", ".")))
    raw_test_command = workspace_config.get("test_command")
    if raw_test_command is not None and (
        not isinstance(raw_test_command, list)
        or any(not isinstance(part, str) or not part for part in raw_test_command)
    ):
        raise ValueError("workspace.test_command must be a non-empty list of strings.")
    test_command = tuple(raw_test_command) if raw_test_command else None
    configured_commands = workspace_config.get("allowed_commands", [])
    if not isinstance(configured_commands, list) or any(
        not isinstance(command, str) for command in configured_commands
    ):
        raise ValueError("workspace.allowed_commands must be a list of executable names.")
    allowed_commands = tuple(configured_commands) + tuple(args.allowed_command or [])
    workspace = WorkspaceTools(
        workspace_path,
        allow_tests=args.allow_tests,
        allow_shell=args.allow_shell,
        allowed_commands=allowed_commands,
        test_command=test_command,
    )
    strategy = RecoveryStrategy(args.strategy)
    limits = _apply_budget_overrides(config, args)
    agent_config = CodingAgentConfig(
        strategy=strategy,
        budgets=limits,
        max_actions_per_attempt=int(
            config.get("max_actions_per_attempt", 8)
        ),
        max_prompt_chars=int(config.get("max_prompt_chars", 24_000)),
    )
    result = await CodingAgent(model, workspace, agent_config).run(args.task)
    output = result.to_dict()
    output["model"] = model.model
    output["device_requested"] = model.device
    output["device_selected"] = None
    output["device_selection_note"] = "The local model server does not report the effective device."
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0 if result.resolved else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repair a local Python workspace with budget-aware recovery."
    )
    parser.add_argument("task", help="Coding task for the local agent")
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--workspace", type=Path, help="Workspace root (defaults to config or current directory)")
    parser.add_argument("--model", help="Override the local model name")
    parser.add_argument("--base-url", help="Override the local Ollama-compatible URL")
    parser.add_argument("--device", choices=("auto", "cpu"), help="Use runtime GPU placement or force CPU")
    parser.add_argument(
        "--strategy",
        choices=[strategy.value for strategy in RecoveryStrategy],
        default=RecoveryStrategy.FAILURE_GUIDED.value,
    )
    parser.add_argument(
        "--allow-tests",
        action="store_true",
        help="Opt in to running the configured test command in the workspace",
    )
    parser.add_argument(
        "--allow-shell",
        action="store_true",
        help="Opt in to model-requested argv commands; requires an executable allowlist",
    )
    parser.add_argument(
        "--allowed-command",
        action="append",
        help="Executable allowed for run_command when --allow-shell is set (repeatable)",
    )
    parser.add_argument("--max-model-calls", type=int)
    parser.add_argument("--max-tool-calls", type=int)
    parser.add_argument("--max-test-runs", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--max-repair-attempts", type=int)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--command-timeout", type=float)
    args = parser.parse_args()
    try:
        exit_code = asyncio.run(_run(args))
    except (LocalModelError, WorkspaceError, OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"agentflow: {type(exc).__name__}: {exc}\n")
    raise SystemExit(exit_code)
