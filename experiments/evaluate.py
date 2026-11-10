"""Aggregate resolution, tests, recovery, latency, and token metrics."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def evaluate(path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    completions: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as result_file:
        for line_number, line in enumerate(result_file, start=1):
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Result line {line_number} must be a JSON object.")
                kind = row.get("record_type")
                if kind == "run_metadata":
                    metadata.append(row)
                elif kind == "runtime_observation":
                    observations.append(row)
                elif kind == "evaluation_complete":
                    completions.append(row)
                elif kind == "model_skipped":
                    skipped.append(row)
                elif kind == "result":
                    rows.append(row)
                else:
                    raise ValueError(
                        f"Result line {line_number} has unknown record_type '{kind}'."
                    )

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("model", "unknown")), str(row.get("strategy", "unknown")))
        groups.setdefault(key, []).append(row)

    summaries = []
    for (model, strategy), group in sorted(groups.items()):
        resolved = sum(row.get("resolved") is True for row in group)
        test_passed = sum(
            bool(
                row.get("attempts")
                and isinstance(row["attempts"][-1].get("test"), dict)
                and row["attempts"][-1]["test"].get("passed") is True
            )
            for row in group
        )
        attempts = [
            len(row.get("attempts", []))
            for row in group
        ]
        repairs = [int(row.get("repair_attempts", max(0, count - 1))) for row, count in zip(group, attempts)]
        latencies = [
            row["budget"]["elapsed_seconds"]
            for row in group
            if isinstance(row.get("budget"), dict)
            and isinstance(row["budget"].get("elapsed_seconds"), (int, float))
        ]
        tokens = [
            row["budget"]["tokens_used"]
            for row in group
            if isinstance(row.get("budget"), dict)
            and isinstance(row["budget"].get("tokens_used"), int)
        ]
        failure_categories = Counter(
            attempt["failure_category"]
            for row in group
            for attempt in row.get("attempts", [])
            if isinstance(attempt, dict)
            and isinstance(attempt.get("failure_category"), str)
        )
        budget_rejection_events = sum(
            event.get("event") == "budget_rejected"
            for row in group
            for event in _budget_events(row)
        )
        token_overrun_events = sum(
            event.get("event") == "token_overrun"
            for row in group
            for event in _budget_events(row)
        )
        budget_violation_cases = sum(
            _has_budget_violation(row) for row in group
        )
        summaries.append(
            {
                "model": model,
                "strategy": strategy,
                "tasks": len(group),
                "resolved": resolved,
                "resolution_rate": resolved / len(group),
                "test_passed": test_passed,
                "test_pass_rate": test_passed / len(group),
                "mean_attempts": sum(attempts) / len(attempts),
                "mean_repair_attempts": sum(repairs) / len(repairs),
                "mean_latency_seconds": sum(latencies) / len(latencies) if latencies else None,
                "mean_tokens_used": sum(tokens) / len(tokens) if tokens else None,
                "token_measured_tasks": len(tokens),
                "mean_model_calls": _mean_budget_metric(group, "model_calls"),
                "mean_tool_calls": _mean_budget_metric(group, "tool_calls"),
                "mean_test_runs": _mean_budget_metric(group, "test_runs"),
                "failure_categories": dict(sorted(failure_categories.items())),
                "budget_rejection_events": budget_rejection_events,
                "token_overrun_events": token_overrun_events,
                "budget_violation_cases": budget_violation_cases,
                "cost_usd": None,
                "cost_note": "Local inference does not report monetary cost.",
            }
        )
    return {
        "runs": summaries,
        "metadata": metadata,
        "runtime_observations": observations,
        "completions": completions,
        "skipped_models": skipped,
    }


def _mean_budget_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [
        row["budget"][key]
        for row in rows
        if isinstance(row.get("budget"), dict)
        and isinstance(row["budget"].get(key), (int, float))
    ]
    return sum(values) / len(values) if values else None


def _budget_events(row: dict[str, Any]) -> list[dict[str, Any]]:
    budget = row.get("budget")
    events = budget.get("budget_events") if isinstance(budget, dict) else None
    if not isinstance(events, list):
        return []
    return [event for event in events if isinstance(event, dict)]


def _has_budget_violation(row: dict[str, Any]) -> bool:
    budget = row.get("budget")
    limits = row.get("budget_limits")
    if not isinstance(budget, dict) or not isinstance(limits, dict):
        return False
    if any(event.get("event") == "token_overrun" for event in _budget_events(row)):
        return True
    metrics = (
        ("budget_tokens_charged", "max_tokens"),
        ("model_calls", "max_model_calls"),
        ("tool_calls", "max_tool_calls"),
        ("test_runs", "max_test_runs"),
        ("elapsed_seconds", "task_timeout_seconds"),
    )
    if any(
        isinstance(budget.get(metric), (int, float))
        and isinstance(limits.get(limit), (int, float))
        and budget[metric] > limits[limit]
        for metric, limit in metrics
    ):
        return True
    repair_attempts = row.get("repair_attempts")
    max_repairs = limits.get("max_repair_attempts")
    return (
        isinstance(repair_attempts, (int, float))
        and isinstance(max_repairs, (int, float))
        and repair_attempts > max_repairs
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", type=Path, default=Path("results/latest.jsonl"))
    args = parser.parse_args()
    print(json.dumps(evaluate(args.results), indent=2))


if __name__ == "__main__":
    main()
