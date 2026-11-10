"""Aggregate resolution, tests, recovery, latency, and token metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def evaluate(path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
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
                "cost_usd": None,
                "cost_note": "Local inference does not report monetary cost.",
            }
        )
    return {"runs": summaries, "metadata": metadata, "skipped_models": skipped}


def _mean_budget_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [
        row["budget"][key]
        for row in rows
        if isinstance(row.get("budget"), dict)
        and isinstance(row["budget"].get(key), (int, float))
    ]
    return sum(values) / len(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", type=Path, default=Path("results/latest.jsonl"))
    args = parser.parse_args()
    print(json.dumps(evaluate(args.results), indent=2))


if __name__ == "__main__":
    main()
