"""Summarize task completion and execution reliability from JSONL results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def evaluate(path: Path) -> dict[str, Any]:
    rows = []
    with path.open(encoding="utf-8") as result_file:
        for line_number, line in enumerate(result_file, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
    completed = sum(row.get("complete") is True for row in rows)
    total_iterations = sum(len(row.get("iterations", [])) for row in rows)
    total_steps = sum(
        len(iteration.get("steps", []))
        for row in rows
        for iteration in row.get("iterations", [])
    )
    failed_steps = sum(
        step.get("error") is not None
        for row in rows
        for iteration in row.get("iterations", [])
        for step in iteration.get("steps", [])
    )
    return {
        "tasks": len(rows),
        "completed": completed,
        "completion_rate": completed / len(rows) if rows else 0.0,
        "iterations": total_iterations,
        "steps": total_steps,
        "failed_steps": failed_steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", type=Path, default=Path("results/latest.jsonl"))
    args = parser.parse_args()
    print(json.dumps(evaluate(args.results), indent=2))


if __name__ == "__main__":
    main()
