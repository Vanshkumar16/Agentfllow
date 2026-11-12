"""Build reproducibility tables, figures, and environment metadata from a run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

if __package__:
    from .evaluate import evaluate
else:
    from evaluate import evaluate


STRATEGIES = ("single-shot", "fixed-retry", "failure-guided")
METRICS = (
    "strict_resolution",
    "final_test_passed",
    "repair_attempts",
    "latency_seconds",
    "tokens_used",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(row)
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _ollama_version() -> str | None:
    try:
        completed = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = (completed.stdout or completed.stderr).strip()
    return output if completed.returncode == 0 else None


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _svg_bar_chart(
    path: Path,
    title: str,
    subtitle: str,
    values: list[tuple[str, float]],
    maximum: float,
    suffix: str,
) -> None:
    width, height = 900, 300
    left, top, chart_width, bar_height, gap = 220, 92, 580, 30, 34
    colors = ("#4c78a8", "#f58518", "#54a24b")
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="24" y="34" font-family="Arial,sans-serif" font-size="22" font-weight="bold">{title}</text>',
        f'<text x="24" y="60" font-family="Arial,sans-serif" font-size="13" fill="#444">{subtitle}</text>',
    ]
    for index, (label, value) in enumerate(values):
        y = top + index * (bar_height + gap)
        bar_width = chart_width * max(0, min(value, maximum)) / maximum if maximum else 0
        color = colors[index % len(colors)]
        elements.extend(
            (
                f'<text x="{left - 12}" y="{y + 21}" text-anchor="end" font-family="Arial,sans-serif" font-size="15">{label}</text>',
                f'<rect x="{left}" y="{y}" width="{chart_width}" height="{bar_height}" rx="4" fill="#eeeeee"/>',
                f'<rect x="{left}" y="{y}" width="{bar_width:.2f}" height="{bar_height}" rx="4" fill="{color}"/>',
                f'<text x="{left + chart_width + 12}" y="{y + 21}" font-family="Arial,sans-serif" font-size="15">{value:g}{suffix}</text>',
            )
        )
    elements.append(
        f'<text x="{left}" y="{height - 20}" font-family="Arial,sans-serif" font-size="12" fill="#555">0</text>'
        f'<text x="{left + chart_width}" y="{height - 20}" text-anchor="end" font-family="Arial,sans-serif" font-size="12" fill="#555">{maximum:g}{suffix}</text>'
    )
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def prepare(raw_path: Path, output_dir: Path, *, force: bool = False) -> None:
    raw_path = raw_path.resolve()
    root = Path(__file__).resolve().parents[1]
    dataset_path = root / "benchmarks" / "tasks_v2.jsonl"
    config_path = root / "configs" / "experiment_v2.yaml"
    destination = output_dir.resolve()
    files = (
        "raw/v2-pilot-frozen.jsonl",
        "environment.json",
        "summary.csv",
        "task_metrics.csv",
        "paired_task_comparisons.csv",
        "failure_categories.csv",
        "pilot-success-rates.svg",
        "pilot-latency.svg",
        "pilot-token-usage.svg",
        "README.md",
    )
    existing = [destination / name for name in files if (destination / name).exists()]
    if existing and not force:
        raise FileExistsError(
            "Research artifacts already exist; use --force to regenerate derived files: "
            + ", ".join(str(path) for path in existing)
        )

    raw_rows = _read_jsonl(raw_path)
    metadata_rows = [row for row in raw_rows if row.get("record_type") == "run_metadata"]
    if len(metadata_rows) != 1:
        raise ValueError("Expected exactly one run_metadata record in the raw log.")
    metadata = metadata_rows[0]
    if metadata.get("split") != "pilot":
        raise ValueError("This artifact builder currently accepts pilot logs only.")
    if metadata.get("dataset_sha256") != _sha256(dataset_path):
        raise ValueError("Raw log dataset hash does not match benchmarks/tasks_v2.jsonl.")
    if metadata.get("config_sha256") != _sha256(config_path):
        raise ValueError("Raw log config hash does not match configs/experiment_v2.yaml.")

    report = evaluate(raw_path)
    results = [row for row in raw_rows if row.get("record_type") == "result"]
    strategies = {str(row.get("strategy")) for row in results}
    if strategies != set(STRATEGIES):
        raise ValueError(f"Pilot log must include all strategies, found {sorted(strategies)}.")

    destination.mkdir(parents=True, exist_ok=True)
    (destination / "raw").mkdir(exist_ok=True)
    raw_copy = destination / "raw" / "v2-pilot-frozen.jsonl"
    if raw_copy.exists() and _sha256(raw_copy) != _sha256(raw_path):
        raise ValueError(f"Refusing to overwrite a different raw log: {raw_copy}")
    if not raw_copy.exists() and raw_path != raw_copy.resolve():
        raw_copy.write_bytes(raw_path.read_bytes())

    environment = {
        "artifact_prepared_at": datetime.now(timezone.utc).isoformat(),
        "artifact_preparation_versions": {
            "python": platform.python_version(),
            "pyyaml": _package_version("PyYAML"),
            "pytest": _package_version("pytest"),
            "ollama_cli": _ollama_version(),
            "note": (
                "Package and Ollama versions were queried during artifact preparation; "
                "the pilot log did not record those runtime versions."
            ),
        },
        "pilot_runtime": {
            "python": metadata.get("hardware", {}).get("python"),
            "platform": metadata.get("hardware", {}).get("platform"),
            "processor": metadata.get("hardware", {}).get("processor"),
            "gpus": metadata.get("hardware", {}).get("gpus", []),
            "model": metadata.get("primary_model"),
            "model_status": metadata.get("primary_model_status"),
            "device_requested": metadata.get("device_requested"),
            "device_selected": metadata.get("device_selected"),
            "device_selection_note": metadata.get("device_selection_note"),
            "budgets": metadata.get("budgets"),
        },
        "reproducibility": {
            "git_commit_hash": metadata.get("git_commit_hash"),
            "git_worktree_clean": metadata.get("git_worktree_clean"),
            "dataset_sha256": metadata.get("dataset_sha256"),
            "config_sha256": metadata.get("config_sha256"),
            "raw_log_sha256": _sha256(raw_path),
            "experiment_id": metadata.get("experiment_id"),
            "random_seed": metadata.get("random_seed"),
            "random_seed_scope": metadata.get("random_seed_scope"),
            "split": metadata.get("split"),
            "task_count": metadata.get("task_count"),
            "repetitions": metadata.get("repetitions"),
        },
    }
    (destination / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    summary_rows = []
    for row in report["runs"]:
        summary_rows.append(
            {
                "model": row["model"],
                "strategy": row["strategy"],
                "tasks": row["tasks"],
                "repetitions": row["repetitions"],
                "resolved": row["resolved"],
                "strict_resolution_rate": row["strict_resolution_rate"],
                "strict_resolution_ci95": json.dumps(
                    row["strict_resolution_confidence_interval_95"]
                ),
                "final_test_passed": row["test_passed"],
                "final_test_pass_rate": row["final_test_pass_rate"],
                "final_test_pass_ci95": json.dumps(
                    row["final_test_pass_confidence_interval_95"]
                ),
                "mean_repair_attempts": row["mean_repair_attempts"],
                "mean_latency_seconds": row["mean_latency_seconds"],
                "mean_tokens_used": row["mean_tokens_used"],
                "model_calls_logged": row["model_calls_logged"],
                "token_usage_coverage": row["token_usage"]["reported_usage_call_coverage"],
                "missing_model_usage_calls": row["token_usage"]["missing_model_usage_calls"],
                "budget_rejection_events": row["budget_rejection_events"],
                "budget_violation_cases": row["budget_violation_cases"],
                "failure_categories": json.dumps(row["failure_categories"], sort_keys=True),
            }
        )
    _write_csv(destination / "summary.csv", list(summary_rows[0]), summary_rows)

    task_rows = [
        {"model": run["model"], "strategy": run["strategy"], **task}
        for run in report["runs"]
        for task in run["per_task_metrics"]
    ]
    _write_csv(destination / "task_metrics.csv", list(task_rows[0]), task_rows)

    paired_rows = []
    for pair in report["paired_comparisons"]:
        for task in pair["per_task"]:
            row: dict[str, Any] = {
                "model": pair["model"],
                "strategy_a": pair["strategy_a"],
                "strategy_b": pair["strategy_b"],
                "task_id": task["task_id"],
                "matched_repetitions": task["matched_repetitions"],
            }
            for metric in METRICS:
                details = task["mean_difference_a_minus_b"][metric]
                row[f"{metric}_mean_a_minus_b"] = details["mean"]
            paired_rows.append(row)
    if paired_rows:
        _write_csv(destination / "paired_task_comparisons.csv", list(paired_rows[0]), paired_rows)

    failure_counts: dict[str, int] = {}
    for result in results:
        for attempt in result.get("attempts", []):
            category = attempt.get("failure_category") if isinstance(attempt, dict) else None
            if isinstance(category, str):
                failure_counts[category] = failure_counts.get(category, 0) + 1
    failures = [
        {"failure_category": category, "count": count}
        for category, count in sorted(failure_counts.items())
    ]
    _write_csv(
        destination / "failure_categories.csv",
        ["failure_category", "count"],
        failures,
    )

    summary_by_strategy = {row["strategy"]: row for row in summary_rows}
    success_values = [
        (strategy, summary_by_strategy[strategy]["strict_resolution_rate"] * 100)
        for strategy in STRATEGIES
    ]
    result_table = "\n".join(
        "| {strategy} | {resolved}/{tasks} ({resolution:.0%}) | "
        "{passed}/{tasks} ({test_rate:.0%}) | {latency:.2f} | {tokens:.0f} | "
        "{coverage:.1%} |".format(
            strategy=row["strategy"],
            resolved=row["resolved"],
            tasks=row["tasks"],
            resolution=row["strict_resolution_rate"],
            passed=row["final_test_passed"],
            test_rate=row["final_test_pass_rate"],
            latency=row["mean_latency_seconds"],
            tokens=row["mean_tokens_used"],
            coverage=row["token_usage_coverage"],
        )
        for row in summary_rows
    )
    _svg_bar_chart(
        destination / "pilot-success-rates.svg",
        "V2 pilot: success rates",
        "Strict resolution; five pilot tasks, one repetition. Final test pass is reported separately in summary.csv.",
        success_values,
        100,
        "%",
    )
    latency_values = [
        (strategy, summary_by_strategy[strategy]["mean_latency_seconds"])
        for strategy in STRATEGIES
    ]
    _svg_bar_chart(
        destination / "pilot-latency.svg",
        "V2 pilot: mean elapsed time",
        "Seconds per task; descriptive pilot data only (n=5, one repetition).",
        latency_values,
        max(1, max(value for _, value in latency_values) * 1.15),
        " s",
    )
    token_values = [
        (strategy, summary_by_strategy[strategy]["mean_tokens_used"])
        for strategy in STRATEGIES
    ]
    _svg_bar_chart(
        destination / "pilot-token-usage.svg",
        "V2 pilot: mean charged tokens",
        "Tokens per task; provider usage coverage varies by strategy.",
        token_values,
        max(1, max(value for _, value in token_values) * 1.15),
        "",
    )

    relative_raw = raw_copy.relative_to(root).as_posix()
    readme = f"""# AgentFlow independent benchmark v2 research artifacts

## Status

This package contains the frozen v2 dataset and protocol references and the
**five-task pilot only**. It is not the 35-task held-out evaluation and is not
evidence of statistical superiority. The planned held-out evaluation remains
315 cases (35 tasks x 3 strategies x 3 repetitions) and must be run separately.

## Frozen inputs and analysis

- Dataset: [`../tasks_v2.jsonl`](../tasks_v2.jsonl), SHA-256
  `{metadata['dataset_sha256']}`.
- Configuration: [`../../configs/experiment_v2.yaml`](../../configs/experiment_v2.yaml),
  SHA-256 `{metadata['config_sha256']}`.
- Frozen code revision: `{metadata['git_commit_hash']}`; clean worktree recorded:
  `{metadata['git_worktree_clean']}`.
- Raw pilot log: [`{relative_raw.split('/')[-1]}`](raw/{relative_raw.split('/')[-1]}),
  SHA-256 `{environment['reproducibility']['raw_log_sha256']}`.
- Analysis: [`../../experiments/evaluate.py`](../../experiments/evaluate.py) and
  [`../../experiments/analyze_results.py`](../../experiments/analyze_results.py).
- Artifact generator: [`../../experiments/prepare_research_artifacts.py`](../../experiments/prepare_research_artifacts.py).

## Reproduce

From the repository root, with the frozen revision checked out and the configured
Gemma E2B model already available in the local Ollama service:

```powershell
$env:PYTHONPATH = 'src'
python experiments/run_experiment.py --tasks benchmarks/tasks_v2.jsonl `
  --config configs/experiment_v2.yaml --split pilot --repetitions 1 `
  --seed 20261112 --output results/v2-pilot-reproduced.jsonl
python experiments/evaluate.py results/v2-pilot-reproduced.jsonl
python experiments/analyze_results.py results/v2-pilot-reproduced.jsonl
python experiments/prepare_research_artifacts.py `
  --results results/v2-pilot-reproduced.jsonl `
  --output-dir benchmarks/research-v2-reproduced
```

The original raw log is preserved and is never overwritten. The randomized seed
controls case ordering, not model sampling. Device selection was requested as
`auto`; the local model server did not report the effective device.

## Files

- `environment.json`: pilot hardware/configuration and reproducibility hashes;
  tool versions queried during artifact preparation are explicitly distinguished
  from versions embedded in the run log.
- `summary.csv`: strategy-level rates, effort, usage coverage, failures, and
  budget outcomes.
- `task_metrics.csv`: per-task/per-strategy observations.
- `paired_task_comparisons.csv`: per-task paired differences (`strategy_a -
  strategy_b`) for the metrics supported by the evaluator.
- `failure_categories.csv`: observed attempt failure categories.
- `pilot-success-rates.svg`, `pilot-latency.svg`, `pilot-token-usage.svg`:
  descriptive figures; no inferential claim is made from the pilot.
- `raw/`: immutable JSONL run record.

## Pilot observations and limitations

There are 15 result records: five tasks under each strategy, one repetition,
with no missing cases in the saved pilot log. Strict resolution and final test
pass are distinct outcomes. Token totals are provider-reported and usage-call
coverage is not 100% for every strategy. The effective inference device and
Ollama version at execution time were not recorded in the raw log. Five tasks
and one repetition are only a protocol/logging pilot; confidence intervals or
paired differences here are not a basis for a superiority claim.

| Strategy | Strict resolution | Final tests passed | Mean latency (s) | Mean tokens | Usage coverage |
|---|---:|---:|---:|---:|---:|
{result_table}

The pilot outcome is descriptive only. The reported confidence intervals in
`summary.csv` should not be interpreted as evidence that any strategy is
superior.

The first exploratory pilot log (`results/v2-pilot.jsonl`) predates the final
v2 dataset hash and is excluded from these tables. Only the frozen log above is
included in this package.
"""
    (destination / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/v2-pilot-frozen.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/research-v2"))
    parser.add_argument("--force", action="store_true", help="replace generated artifacts in output-dir")
    args = parser.parse_args()
    prepare(args.results, args.output_dir, force=args.force)


if __name__ == "__main__":
    main()
