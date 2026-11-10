"""Print task-level outcomes and iteration counts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", type=Path, default=Path("results/latest.jsonl"))
    args = parser.parse_args()
    with args.results.open(encoding="utf-8") as result_file:
        for line_number, line in enumerate(result_file, start=1):
            if not line.strip():
                continue
            try:
                result = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            status = "complete" if result.get("complete") is True else "incomplete"
            print(
                f"{result.get('task_id', line_number)}: {status}; "
                f"{len(result.get('iterations', []))} iteration(s); "
                f"{result.get('summary', '')}"
            )


if __name__ == "__main__":
    main()
