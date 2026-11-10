"""Print an easy-to-scan comparison table from JSONL experiment results."""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__:
    from .evaluate import evaluate
else:
    from evaluate import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", type=Path, default=Path("results/latest.jsonl"))
    args = parser.parse_args()
    report = evaluate(args.results)
    if not report["runs"]:
        print("No completed task records are present in this results file.")
    for row in report["runs"]:
        print(
            f"{row['model']} | {row['strategy']} | "
            f"resolved={row['resolved']}/{row['tasks']} "
            f"({row['resolution_rate']:.1%}) | tests={row['test_pass_rate']:.1%} | "
            f"repairs={row['mean_repair_attempts']:.2f} | "
            f"latency={row['mean_latency_seconds']}s | "
            f"tokens={row['mean_tokens_used']}"
        )
    for row in report["skipped_models"]:
        print(f"Skipped model {row['model']}: {row['reason']}")


if __name__ == "__main__":
    main()
