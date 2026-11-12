"""Generate reproducibility artifacts from validated v2 held-out results."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

if __package__:
    from .evaluate import evaluate
    from .prepare_research_artifacts import _svg_bar_chart, _write_csv
    from .validate_heldout_results import validate_results
else:
    from evaluate import evaluate
    from prepare_research_artifacts import _svg_bar_chart, _write_csv
    from validate_heldout_results import validate_results


ARTIFACT_FILES = (
    "raw/v2-held-out.jsonl",
    "environment.json",
    "summary.csv",
    "task_metrics.csv",
    "paired_task_comparisons.csv",
    "failure_categories.csv",
    "budget_analysis.csv",
    "statistical_analysis.json",
    "heldout-success-rates.svg",
    "heldout-latency.svg",
    "heldout-token-usage.svg",
    "heldout-repair-attempts.svg",
    "README.md",
)


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _flatten_per_task_metrics(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for run in report["runs"]:
        for task in run["per_task_metrics"]:
            row: dict[str, Any] = {
                "model": run["model"],
                "strategy": run["strategy"],
                "task_id": task["task_id"],
                "repetitions": task["repetitions"],
            }
            for metric in (
                "strict_resolution",
                "final_test_passed",
                "repair_attempts",
                "latency_seconds",
                "tokens_used",
            ):
                summary = task[metric]
                row[f"{metric}_mean"] = summary.get(
                    "mean", summary.get("rate")
                )
                row[f"{metric}_standard_deviation"] = summary.get(
                    "standard_deviation"
                )
                row[f"{metric}_ci95"] = json.dumps(
                    summary["confidence_interval_95"], sort_keys=True
                )
            rows.append(row)
    return rows


def _flatten_paired_comparisons(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for comparison in report["paired_comparisons"]:
        clustered = comparison["mean_difference_across_paired_cases"]
        for task in comparison["per_task"]:
            for metric, summary in task["mean_difference_a_minus_b"].items():
                interval = summary["confidence_interval_95"]
                overall = clustered[metric]
                overall_interval = overall["confidence_interval_95"]
                rows.append(
                    {
                        "comparison_level": "task",
                        "model": comparison["model"],
                        "strategy_a": comparison["strategy_a"],
                        "strategy_b": comparison["strategy_b"],
                        "task_id": task["task_id"],
                        "matched_task_count": comparison["matched_task_count"],
                        "matched_repetitions": task["matched_repetitions"],
                        "metric": metric,
                        "mean_difference_a_minus_b": summary["mean"],
                        "repetition_standard_deviation": summary[
                            "standard_deviation"
                        ],
                        "repetition_ci95_lower": (
                            interval["lower"] if interval is not None else None
                        ),
                        "repetition_ci95_upper": (
                            interval["upper"] if interval is not None else None
                        ),
                        "task_clustered_mean_difference": overall["mean"],
                        "task_clustered_ci95_lower": (
                            overall_interval["lower"]
                            if overall_interval is not None
                            else None
                        ),
                        "task_clustered_ci95_upper": (
                            overall_interval["upper"]
                            if overall_interval is not None
                            else None
                        ),
                        "matched_metric_task_count": overall[
                            "matched_metric_task_count"
                        ],
                        "missing_metric_task_count": overall[
                            "missing_metric_task_count"
                        ],
                        "tasks_favoring_strategy_a": overall[
                            "tasks_favoring_strategy_a"
                        ],
                        "tasks_favoring_strategy_b": overall[
                            "tasks_favoring_strategy_b"
                        ],
                        "tasks_tied": overall["tasks_tied"],
                    }
                )
        for metric, overall in clustered.items():
            interval = overall["confidence_interval_95"]
            rows.append(
                {
                    "comparison_level": "aggregate",
                    "model": comparison["model"],
                    "strategy_a": comparison["strategy_a"],
                    "strategy_b": comparison["strategy_b"],
                    "task_id": "ALL_TASKS",
                    "matched_task_count": comparison["matched_task_count"],
                    "matched_repetitions": comparison["matched_repetition_count"],
                    "metric": metric,
                    "mean_difference_a_minus_b": overall["mean"],
                    "repetition_standard_deviation": None,
                    "repetition_ci95_lower": None,
                    "repetition_ci95_upper": None,
                    "task_clustered_mean_difference": overall["mean"],
                    "task_clustered_ci95_lower": (
                        interval["lower"] if interval is not None else None
                    ),
                    "task_clustered_ci95_upper": (
                        interval["upper"] if interval is not None else None
                    ),
                    "matched_metric_task_count": overall[
                        "matched_metric_task_count"
                    ],
                    "missing_metric_task_count": overall[
                        "missing_metric_task_count"
                    ],
                    "tasks_favoring_strategy_a": overall[
                        "tasks_favoring_strategy_a"
                    ],
                    "tasks_favoring_strategy_b": overall[
                        "tasks_favoring_strategy_b"
                    ],
                    "tasks_tied": overall["tasks_tied"],
                }
            )
    return rows


def _flatten_failure_categories(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "model": run["model"],
            "strategy": run["strategy"],
            "failure_category": category,
            "count": count,
        }
        for run in report["runs"]
        for category, count in sorted(run["failure_categories"].items())
    ]


def _budget_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "model": run["model"],
            "strategy": run["strategy"],
            "result_cases": run["tasks"],
            "budget_rejection_events": run["budget_rejection_events"],
            "budget_rejections_by_kind": json.dumps(
                run["budget_rejections_by_kind"], sort_keys=True
            ),
            "budget_violation_cases": run["budget_violation_cases"],
            "budget_violation_rate": run["budget_violation_rate"],
            "budget_violation_reasons": json.dumps(
                run["budget_violation_reasons"], sort_keys=True
            ),
            "provider_call_attempts": run["token_usage"]["provider_call_attempts"],
            "budget_rejected_model_call_attempts": run["token_usage"][
                "budget_rejected_model_call_attempts"
            ],
            "missing_model_usage_calls": run["token_usage"][
                "missing_model_usage_calls"
            ],
            "reported_usage_call_coverage": run["token_usage"][
                "reported_usage_call_coverage"
            ],
            "missing_result_token_usage_cases": run["token_usage"][
                "missing_result_token_usage_cases"
            ],
        }
        for run in report["runs"]
    ]


def _summary_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    clustered = report["statistical_analysis"]["strategy_estimates"]
    clustered_by_key = {
        (row["model"], row["strategy"]): row["metrics"] for row in clustered
    }
    for run in report["runs"]:
        repetitions = run["metrics_across_repetitions"]
        cluster_metrics = clustered_by_key[(run["model"], run["strategy"])]
        row = {
            "model": run["model"],
            "strategy": run["strategy"],
            "result_cases": run["tasks"],
            "repetitions": run["repetitions"],
            "strict_resolution_rate": run["strict_resolution_rate"],
            "final_test_pass_rate": run["final_test_pass_rate"],
            "mean_repair_attempts": run["mean_repair_attempts"],
            "mean_latency_seconds": run["mean_latency_seconds"],
            "mean_tokens_used": run["mean_tokens_used"],
            "token_measured_tasks": run["token_measured_tasks"],
            "missing_result_token_usage_cases": run["token_usage"][
                "missing_result_token_usage_cases"
            ],
            "reported_usage_call_coverage": run["token_usage"][
                "reported_usage_call_coverage"
            ],
            "missing_model_usage_calls": run["token_usage"][
                "missing_model_usage_calls"
            ],
            "budget_rejected_model_call_attempts": run["token_usage"][
                "budget_rejected_model_call_attempts"
            ],
            "budget_violation_cases": run["budget_violation_cases"],
        }
        for metric in (
            "strict_resolution",
            "final_test_passed",
            "repair_attempts",
            "latency_seconds",
            "tokens_used",
        ):
            repetition_summary = repetitions[metric]
            row[f"{metric}_repetition_mean"] = repetition_summary["mean"]
            row[f"{metric}_repetition_standard_deviation"] = repetition_summary[
                "standard_deviation"
            ]
            cluster_summary = cluster_metrics[metric]
            row[f"{metric}_task_clustered_mean"] = cluster_summary["mean"]
            row[f"{metric}_task_clustered_standard_deviation"] = cluster_summary[
                "standard_deviation"
            ]
            interval = cluster_summary["confidence_interval_95"]
            row[f"{metric}_task_clustered_ci95_lower"] = (
                interval["lower"] if interval is not None else None
            )
            row[f"{metric}_task_clustered_ci95_upper"] = (
                interval["upper"] if interval is not None else None
            )
            row[f"{metric}_missing_tasks"] = cluster_summary["missing_tasks"]
        rows.append(row)
    return rows


def _write_readme(
    destination: Path,
    validated: dict[str, Any],
    report: dict[str, Any],
    raw_sha256: str,
) -> None:
    metadata = validated["metadata"]
    lines = [
        "# AgentFlow independent benchmark v2: held-out evaluation",
        "",
        "This package contains the validated held-out evaluation evidence, derived",
        "tables, statistical analysis, runtime environment metadata, and figures.",
        "The 35-task held-out set was evaluated with three strategies and three",
        "repetitions (315 scheduled cases). The five-task pilot was separate.",
        "",
        "## Frozen protocol and reproducibility",
        "",
        "- Dataset: [`../../tasks_v2.jsonl`](../../tasks_v2.jsonl), SHA-256",
        f"  `{metadata['dataset_sha256']}`.",
        "- Configuration: [`../../../configs/experiment_v2.yaml`](../../../configs/experiment_v2.yaml), SHA-256",
        f"  `{metadata['config_sha256']}`.",
        f"- Commit: `{metadata['git_commit_hash']}` (clean worktree recorded: `{metadata['git_worktree_clean']}`).",
        f"- Model: `{metadata['primary_model']}`; Ollama model digest is in `environment.json`.",
        f"- Random seed: `{metadata['random_seed']}`; it randomizes task/strategy order, not inference sampling.",
        f"- Temperature: `{metadata['temperature']}`.",
        f"- Raw evidence: [`raw/v2-held-out.jsonl`](raw/v2-held-out.jsonl), SHA-256 `{raw_sha256}`.",
        "- The original frozen pilot evidence was not changed.",
        "",
        "Reproduce from the repository root with the frozen code, config, dataset,",
        "and locally installed model available:",
        "",
        "```powershell",
        "$env:PYTHONPATH = 'src'",
        "python experiments/run_experiment.py --tasks benchmarks/tasks_v2.jsonl `",
        "  --config configs/experiment_v2.yaml --split held-out --repetitions 3 `",
        "  --seed 20261113 --output results/v2-held-out.jsonl",
        "# An interrupted run leaves results/v2-held-out.jsonl.partial; inspect it before retrying.",
        "python experiments/validate_heldout_results.py results/v2-held-out.jsonl",
        "python experiments/evaluate.py results/v2-held-out.jsonl",
        "python experiments/analyze_results.py results/v2-held-out.jsonl",
        "python experiments/prepare_heldout_research_artifacts.py `",
        "  --results results/v2-held-out.jsonl `",
        "  --output-dir benchmarks/research-v2/held-out",
        "```",
        "",
        "## Results",
        "",
        "Strict resolution and final test-pass rate are separate outcomes. Values",
        "and task-clustered 95% confidence intervals are descriptive; no strategy",
        "is claimed superior without evidence supporting that conclusion.",
        "",
        "| Strategy | Strict resolution | Final tests passed | Mean repairs | Mean latency (s) | Mean tokens | Usage coverage |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for run in report["runs"]:
        usage = run["token_usage"]["reported_usage_call_coverage"]
        usage_text = f"{usage:.1%}" if usage is not None else "not measured"
        lines.append(
            f"| {run['strategy']} | {run['strict_resolution_rate']:.1%} | "
            f"{run['final_test_pass_rate']:.1%} | {run['mean_repair_attempts']} | "
            f"{run['mean_latency_seconds']} | {run['mean_tokens_used']} | {usage_text} |"
        )
    lines.extend(
        [
            "",
            "## Statistical method and limitations",
            "",
            f"- Strategy estimates use a seeded task-clustered percentile bootstrap with {report['statistical_analysis']['bootstrap_samples']:,} resamples; repetitions remain within each sampled task.",
            "- Paired comparisons first match strategy observations by task and repetition, average differences within task, and then bootstrap tasks. The evaluator reports matched task/repetition counts and missing metric coverage.",
            "- Repetition-level means and standard deviations are reported separately from task-clustered estimates.",
            "- Provider usage coverage excludes budget-rejected calls, which are reported separately. Missing provider usage and missing result-level token totals are reported explicitly.",
            "- Budget rejection events and budget violations are distinct and shown in `budget_analysis.csv`.",
            "- Model execution is local. `environment.json` records the runtime versions, model digest, hardware/GPU driver, and available Ollama process-list observations; unreported effective device details are not inferred.",
            "- Inference is not deterministic even with a fixed execution-order seed. The seed does not control model sampling.",
            "",
            "## Files",
            "",
            "- `raw/v2-held-out.jsonl`: immutable raw run log.",
            "- `environment.json`: hashes, software, hardware, runtime, and completion metadata.",
            "- `summary.csv`: repetition-level and task-clustered strategy metrics, CIs, and token-usage coverage.",
            "- `task_metrics.csv`: per-task/per-strategy metrics and repetition dispersion.",
            "- `paired_task_comparisons.csv`: task-level paired differences for all strategy pairs.",
            "- `statistical_analysis.json`: bootstrap method, estimates, paired comparisons, and uncertainty.",
            "- `budget_analysis.csv`: provider-usage gaps, budget rejections, and violations.",
            "- `failure_categories.csv`: observed failures by strategy.",
            "- `heldout-*.svg`: descriptive success, latency, token, and repair charts.",
            "",
        ]
    )
    (destination / "README.md").write_text("\n".join(lines), encoding="utf-8")


def prepare(
    raw_path: Path, output_dir: Path, *, force: bool = False
) -> None:
    validated = validate_results(raw_path)
    report = evaluate(raw_path)
    destination = output_dir.resolve()
    existing = [
        destination / name
        for name in ARTIFACT_FILES
        if (destination / name).exists()
    ]
    raw_destination = destination / "raw" / "v2-held-out.jsonl"
    raw_sha256 = validated["raw_log_sha256"]
    if raw_destination.exists():
        existing_raw_sha256 = hashlib.sha256(raw_destination.read_bytes()).hexdigest()
        if existing_raw_sha256 != raw_sha256:
            raise ValueError(f"Refusing to overwrite different raw evidence: {raw_destination}")
        existing = [path for path in existing if path != raw_destination]
    if existing and not force:
        raise FileExistsError(
            "Held-out research artifacts already exist; use --force to regenerate "
            "derived files: " + ", ".join(str(path) for path in existing)
        )

    destination.mkdir(parents=True, exist_ok=True)
    (destination / "raw").mkdir(exist_ok=True)
    if not raw_destination.exists() and raw_path.resolve() != raw_destination.resolve():
        raw_destination.write_bytes(raw_path.read_bytes())

    metadata = validated["metadata"]
    environment = {
        "artifact_prepared_at": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": metadata.get("hardware", {}).get("python"),
            "platform": metadata.get("hardware", {}).get("platform"),
            "processor": metadata.get("hardware", {}).get("processor"),
            "gpus": metadata.get("hardware", {}).get("gpus", []),
            "model": metadata.get("primary_model"),
            "model_inventory": metadata.get("model_inventory"),
            "temperature": metadata.get("temperature"),
            "device_requested": metadata.get("device_requested"),
            "device_selected": metadata.get("device_selected"),
            "device_selection_note": metadata.get("device_selection_note"),
            "ollama_server_version": metadata.get("ollama_server_version"),
            "ollama_cli_version": metadata.get("ollama_cli_version"),
            "runtime_observations": validated["runtime_observations"],
            "budgets": metadata.get("budgets"),
        },
        "reproducibility": {
            "git_commit_hash": metadata.get("git_commit_hash"),
            "git_worktree_clean": metadata.get("git_worktree_clean"),
            "dataset_sha256": metadata.get("dataset_sha256"),
            "config_sha256": metadata.get("config_sha256"),
            "raw_log_sha256": raw_sha256,
            "experiment_id": metadata.get("experiment_id"),
            "random_seed": metadata.get("random_seed"),
            "random_seed_scope": metadata.get("random_seed_scope"),
            "split": metadata.get("split"),
            "task_count": metadata.get("task_count"),
            "repetitions": metadata.get("repetitions"),
            "result_count": validated["result_count"],
            "completion": validated["completion"],
        },
        "artifact_preparation_versions": {
            "python": platform.python_version(),
            "pyyaml": _package_version("PyYAML"),
            "pytest": _package_version("pytest"),
        },
    }
    (destination / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    summary_rows = _summary_rows(report)
    summary_fields = list(summary_rows[0]) if summary_rows else []
    _write_csv(destination / "summary.csv", summary_fields, summary_rows)
    task_rows = _flatten_per_task_metrics(report)
    _write_csv(
        destination / "task_metrics.csv",
        list(task_rows[0]) if task_rows else [],
        task_rows,
    )
    paired_rows = _flatten_paired_comparisons(report)
    _write_csv(
        destination / "paired_task_comparisons.csv",
        list(paired_rows[0]) if paired_rows else [],
        paired_rows,
    )
    failure_rows = _flatten_failure_categories(report)
    _write_csv(
        destination / "failure_categories.csv",
        ["model", "strategy", "failure_category", "count"],
        failure_rows,
    )
    budget_rows = _budget_rows(report)
    _write_csv(
        destination / "budget_analysis.csv",
        list(budget_rows[0]) if budget_rows else [],
        budget_rows,
    )
    analysis = {
        "statistical_analysis": report["statistical_analysis"],
        "paired_comparisons": report["paired_comparisons"],
    }
    (destination / "statistical_analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    strategies = ("single-shot", "fixed-retry", "failure-guided")
    values = {row["strategy"]: row for row in summary_rows}
    charts = (
        (
            "heldout-success-rates.svg",
            "Held-out success rates",
            "Strict resolution; final test pass is reported separately in summary.csv",
            [
                (strategy, _number(values[strategy]["strict_resolution_rate"]) or 0)
                for strategy in strategies
            ],
            1.0,
            "",
        ),
        (
            "heldout-latency.svg",
            "Held-out mean latency",
            "Mean elapsed seconds across result cases",
            [
                (strategy, _number(values[strategy]["mean_latency_seconds"]) or 0)
                for strategy in strategies
            ],
            max(
                (
                    _number(values[strategy]["mean_latency_seconds"]) or 0
                    for strategy in strategies
                ),
                default=0,
            ),
            "s",
        ),
        (
            "heldout-token-usage.svg",
            "Held-out mean token usage",
            "Provider-reported usage; missing observations are listed in summary.csv",
            [
                (strategy, _number(values[strategy]["mean_tokens_used"]) or 0)
                for strategy in strategies
            ],
            max(
                (
                    _number(values[strategy]["mean_tokens_used"]) or 0
                    for strategy in strategies
                ),
                default=0,
            ),
            "",
        ),
        (
            "heldout-repair-attempts.svg",
            "Held-out repair attempts",
            "Mean repair attempts per result case",
            [
                (strategy, _number(values[strategy]["mean_repair_attempts"]) or 0)
                for strategy in strategies
            ],
            max(
                (
                    _number(values[strategy]["mean_repair_attempts"]) or 0
                    for strategy in strategies
                ),
                default=0,
            ),
            "",
        ),
    )
    for filename, title, subtitle, chart_values, maximum, suffix in charts:
        _svg_bar_chart(
            destination / filename,
            title,
            subtitle,
            chart_values,
            maximum,
            suffix,
        )
    _write_readme(destination, validated, report, raw_sha256)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=Path("results/v2-held-out.jsonl")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("benchmarks/research-v2/held-out")
    )
    parser.add_argument(
        "--force", action="store_true", help="regenerate derived artifacts only"
    )
    args = parser.parse_args()
    try:
        prepare(args.results, args.output_dir, force=args.force)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"held-out artifacts: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
