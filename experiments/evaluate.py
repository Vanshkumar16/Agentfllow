"""Aggregate resolution, tests, recovery, latency, and token metrics."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence, TypeGuard


STRATEGIES = ("single-shot", "fixed-retry", "failure-guided")
PAIRED_METRICS = (
    "strict_resolution",
    "final_test_passed",
    "repair_attempts",
    "latency_seconds",
    "tokens_used",
)


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
        test_passed = sum(_final_test_passed(row) for row in group)
        patch_valid = sum(row.get("patch_valid") is True for row in group)
        explicit_completion = sum(
            row.get("explicit_completion") is True for row in group
        )
        budget_exhausted = sum(
            row.get("budget_exhausted") is True for row in group
        )
        tests_passed_without_completion = sum(
            row.get("final_test_passed") is True
            and row.get("explicit_completion") is False
            for row in group
        )
        attempts = [_attempt_count(row) for row in group]
        repairs = [_repair_attempts(row) for row in group]
        latencies = [_latency(row) for row in group]
        tokens = [_tokens_used(row) for row in group]
        valid_latencies = [value for value in latencies if value is not None]
        valid_tokens = [value for value in tokens if value is not None]
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
        budget_rejections_by_kind = Counter(
            str(event.get("budget_kind", "unknown"))
            for row in group
            for event in _budget_events(row)
            if event.get("event") == "budget_rejected"
        )
        token_overrun_events = sum(
            event.get("event") == "token_overrun"
            for row in group
            for event in _budget_events(row)
        )
        budget_violation_reasons = Counter(
            reason
            for row in group
            for reason in _budget_violation_reasons(row)
        )
        budget_violation_cases = sum(bool(_budget_violation_reasons(row)) for row in group)
        call_logs = [
            call
            for row in group
            for call in _per_call_logs(row)
        ]
        call_counts = Counter(call.get("call_type") for call in call_logs)
        model_call_logs = [
            call for call in call_logs if call.get("call_type") == "model"
        ]
        usage_statuses = Counter(
            accounting.get("usage_status")
            for call in model_call_logs
            if isinstance((accounting := call.get("token_accounting")), dict)
        )
        missing_usage_calls = sum(
            not isinstance(call.get("token_accounting"), dict)
            or call["token_accounting"].get("usage_status") not in {
                "measured",
                "partial",
                "unavailable",
            }
            for call in model_call_logs
        )
        model_calls_from_budgets = sum(
            _int_metric(row, "model_calls") or 0 for row in group
        )
        unlogged_model_calls = max(0, model_calls_from_budgets - len(model_call_logs))
        missing_model_usage_calls = (
            usage_statuses.get("unavailable", 0)
            + missing_usage_calls
            + unlogged_model_calls
        )
        usage_call_denominator = max(model_calls_from_budgets, len(model_call_logs))
        repetition_summaries = _summarize_repetitions(group)
        task_summaries = _summarize_tasks(group)
        token_usage = {
            "model_calls_logged": len(model_call_logs),
            "model_calls_from_budgets": model_calls_from_budgets,
            "unlogged_model_calls": unlogged_model_calls,
            "measured_model_usage_calls": usage_statuses.get("measured", 0),
            "partial_model_usage_calls": usage_statuses.get("partial", 0),
            "unavailable_model_usage_calls": usage_statuses.get("unavailable", 0),
            "missing_model_usage_calls": missing_model_usage_calls,
            "reported_usage_call_coverage": (
                (usage_statuses.get("measured", 0) + usage_statuses.get("partial", 0))
                / usage_call_denominator
                if usage_call_denominator
                else None
            ),
            "fully_measured_usage_call_coverage": (
                usage_statuses.get("measured", 0) / usage_call_denominator
                if usage_call_denominator
                else None
            ),
            "result_token_usage_cases": len(valid_tokens),
            "missing_result_token_usage_cases": len(group) - len(valid_tokens),
            "missing_result_token_usage_task_ids": [
                _task_id(row)
                for row, token_count in zip(group, tokens)
                if token_count is None
            ],
        }
        summaries.append(
            {
                "model": model,
                "strategy": strategy,
                "tasks": len(group),
                "repetitions": len(
                    {_repetition_id(row) for row in group}
                ),
                "resolved": resolved,
                "resolution_rate": resolved / len(group),
                "strict_resolution_rate": resolved / len(group),
                "strict_resolution_confidence_interval_95": _proportion_confidence_interval(
                    resolved, len(group)
                )["confidence_interval_95"],
                "test_passed": test_passed,
                "test_pass_rate": test_passed / len(group),
                "final_test_pass_rate": test_passed / len(group),
                "final_test_pass_confidence_interval_95": _proportion_confidence_interval(
                    test_passed, len(group)
                )["confidence_interval_95"],
                "patch_valid": patch_valid,
                "explicit_completion": explicit_completion,
                "budget_exhausted": budget_exhausted,
                "tests_passed_without_completion": tests_passed_without_completion,
                "mean_attempts": _mean([value for value in attempts if value is not None]),
                "mean_repair_attempts": _mean([value for value in repairs if value is not None]),
                "mean_latency_seconds": _mean(valid_latencies),
                "mean_tokens_used": _mean(valid_tokens),
                "token_measured_tasks": len(valid_tokens),
                "mean_model_calls": _mean_budget_metric(group, "model_calls"),
                "mean_tool_calls": _mean_budget_metric(group, "tool_calls"),
                "mean_test_runs": _mean_budget_metric(group, "test_runs"),
                "failure_categories": dict(sorted(failure_categories.items())),
                "budget_rejection_events": budget_rejection_events,
                "budget_rejections_by_kind": dict(sorted(budget_rejections_by_kind.items())),
                "token_overrun_events": token_overrun_events,
                "budget_violation_cases": budget_violation_cases,
                "budget_violation_rate": budget_violation_cases / len(group),
                "budget_violation_reasons": dict(sorted(budget_violation_reasons.items())),
                "token_usage": token_usage,
                "metrics_across_repetitions": repetition_summaries,
                "per_task_metrics": task_summaries,
                "model_calls_logged": call_counts.get("model", 0),
                "tool_calls_logged": call_counts.get("tool", 0),
                "test_calls_logged": call_counts.get("test", 0),
                "invalid_model_responses": sum(
                    call.get("validation_outcome") == "invalid"
                    for call in model_call_logs
                ),
                "unknown_termination_calls": sum(
                    call.get("response_termination") == "unknown"
                    for call in model_call_logs
                ),
                "measured_model_usage_calls": usage_statuses.get("measured", 0),
                "partial_model_usage_calls": usage_statuses.get("partial", 0),
                "unavailable_model_usage_calls": usage_statuses.get("unavailable", 0),
                "cost_usd": None,
                "cost_note": "Local inference does not report monetary cost.",
            }
        )
    return {
        "runs": summaries,
        "paired_comparisons": _paired_comparisons(rows),
        "metadata": metadata,
        "runtime_observations": observations,
        "completions": completions,
        "skipped_models": skipped,
    }


def _mean(values: Sequence[float | int]) -> float | None:
    return sum(values) / len(values) if values else None


def _is_number(value: Any) -> TypeGuard[int | float]:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _int_metric(row: dict[str, Any], key: str) -> int | None:
    budget = row.get("budget")
    if not isinstance(budget, dict):
        return None
    value = budget.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _attempt_count(row: dict[str, Any]) -> int | None:
    attempts = row.get("attempts")
    return len(attempts) if isinstance(attempts, list) else None


def _repair_attempts(row: dict[str, Any]) -> int | None:
    value = row.get("repair_attempts")
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    count = _attempt_count(row)
    return max(0, count - 1) if count is not None else None


def _latency(row: dict[str, Any]) -> float | None:
    budget = row.get("budget")
    if not isinstance(budget, dict):
        return None
    value = budget.get("elapsed_seconds")
    return float(value) if _is_number(value) and value >= 0 else None


def _tokens_used(row: dict[str, Any]) -> int | None:
    budget = row.get("budget")
    if not isinstance(budget, dict):
        return None
    value = budget.get("tokens_used")
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _task_id(row: dict[str, Any]) -> str:
    return str(row.get("task_id", "unknown"))


def _repetition_id(row: dict[str, Any]) -> str:
    repetition = row.get("repetition_id")
    return str(repetition) if repetition is not None else "single-run"


def _final_test_passed(row: dict[str, Any]) -> bool:
    final_test_passed = row.get("final_test_passed")
    if isinstance(final_test_passed, bool):
        return final_test_passed
    attempts = row.get("attempts")
    if not isinstance(attempts, list) or not attempts or not isinstance(attempts[-1], dict):
        return False
    test = attempts[-1].get("test")
    return isinstance(test, dict) and test.get("passed") is True


def _mean_budget_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [
        row["budget"][key]
        for row in rows
        if isinstance(row.get("budget"), dict)
        and _is_number(row["budget"].get(key))
    ]
    return sum(values) / len(values) if values else None


def _mean_confidence_interval(values: list[float]) -> dict[str, Any]:
    count = len(values)
    mean = _mean(values)
    standard_deviation = statistics.stdev(values) if count > 1 else None
    if mean is None or count < 2 or standard_deviation is None:
        lower = upper = None
    else:
        margin = _t_critical_95(count - 1) * standard_deviation / math.sqrt(count)
        lower, upper = mean - margin, mean + margin
    return {
        "n": count,
        "mean": mean,
        "standard_deviation": standard_deviation,
        "confidence_interval_95": (
            {"lower": lower, "upper": upper} if lower is not None else None
        ),
    }


def _proportion_confidence_interval(successes: int, total: int) -> dict[str, Any]:
    if total == 0:
        return {"n": 0, "successes": 0, "rate": None, "confidence_interval_95": None}
    rate = successes / total
    z = 1.959963984540054
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    half_width = (
        z
        * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total))
        / denominator
    )
    return {
        "n": total,
        "successes": successes,
        "rate": rate,
        "confidence_interval_95": {
            "lower": max(0.0, center - half_width),
            "upper": min(1.0, center + half_width),
        },
    }


def _t_critical_95(degrees_of_freedom: int) -> float:
    critical_values = (
        0.0,
        12.706,
        4.303,
        3.182,
        2.776,
        2.571,
        2.447,
        2.365,
        2.306,
        2.262,
        2.228,
        2.201,
        2.179,
        2.160,
        2.145,
        2.131,
        2.120,
        2.110,
        2.101,
        2.093,
        2.086,
        2.080,
        2.074,
        2.069,
        2.064,
        2.060,
        2.056,
        2.052,
        2.048,
        2.045,
        2.042,
    )
    if degrees_of_freedom < len(critical_values):
        return critical_values[degrees_of_freedom]
    return 1.96


def _metric_values(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {metric: [] for metric in PAIRED_METRICS}
    for row in rows:
        values["strict_resolution"].append(float(row.get("resolved") is True))
        values["final_test_passed"].append(float(_final_test_passed(row)))
        repairs = _repair_attempts(row)
        latency = _latency(row)
        tokens = _tokens_used(row)
        if repairs is not None:
            values["repair_attempts"].append(float(repairs))
        if latency is not None:
            values["latency_seconds"].append(latency)
        if tokens is not None:
            values["tokens_used"].append(float(tokens))
    return values


def _summarize_repetitions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    repetitions: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        repetitions.setdefault(_repetition_id(row), []).append(row)
    per_repetition: dict[str, dict[str, float | None]] = {}
    for repetition_id, group in sorted(repetitions.items()):
        measurements = _metric_values(group)
        per_repetition[repetition_id] = {}
        for metric, values in measurements.items():
            per_repetition[repetition_id][metric] = _mean(values)
    summary: dict[str, Any] = {
        "count": len(per_repetition),
        "per_repetition": per_repetition,
    }
    for metric in PAIRED_METRICS:
        samples: list[float] = []
        for value in per_repetition.values():
            sample = value[metric]
            if sample is not None:
                samples.append(sample)
        if metric in {"strict_resolution", "final_test_passed"}:
            summary[metric] = _mean_confidence_interval(samples)
            summary[metric]["confidence_interval_method"] = (
                "Student-t interval over repetition-level rates"
            )
        else:
            summary[metric] = _mean_confidence_interval(samples)
    return summary


def _summarize_tasks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(_task_id(row), []).append(row)
    summaries = []
    for task_id, group in sorted(by_task.items()):
        values = _metric_values(group)
        summaries.append(
            {
                "task_id": task_id,
                "repetitions": len(
                    {_repetition_id(row) for row in group}
                ),
                "strict_resolution": _proportion_confidence_interval(
                    sum(row.get("resolved") is True for row in group), len(group)
                ),
                "final_test_passed": _proportion_confidence_interval(
                    sum(_final_test_passed(row) for row in group), len(group)
                ),
                "repair_attempts": _mean_confidence_interval(values["repair_attempts"]),
                "latency_seconds": _mean_confidence_interval(values["latency_seconds"]),
                "tokens_used": _mean_confidence_interval(values["tokens_used"]),
            }
        )
    return summaries


def _paired_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    models: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        model = str(row.get("model", "unknown"))
        strategy = str(row.get("strategy", "unknown"))
        models.setdefault(model, {}).setdefault(strategy, []).append(row)

    comparisons: list[dict[str, Any]] = []
    for model, strategies_by_name in sorted(models.items()):
        strategy_names = sorted(strategies_by_name)
        strategy_pairs = list(combinations(strategy_names, 2))
        # Keep the canonical strategy ordering where the standard three are present.
        if all(strategy in strategy_names for strategy in STRATEGIES):
            strategy_pairs = list(combinations(STRATEGIES, 2))
        for strategy_a, strategy_b in strategy_pairs:
            by_key: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = {}
            for strategy, strategy_rows in (
                (strategy_a, strategies_by_name[strategy_a]),
                (strategy_b, strategies_by_name[strategy_b]),
            ):
                for row in strategy_rows:
                    key = (_task_id(row), _repetition_id(row))
                    by_key.setdefault(key, {}).setdefault(strategy, []).append(row)
            task_pairs: dict[str, dict[str, list[tuple[dict[str, Any], dict[str, Any]]]]] = {}
            for (task_id, _), strategy_rows in by_key.items():
                rows_a = strategy_rows.get(strategy_a, [])
                rows_b = strategy_rows.get(strategy_b, [])
                for row_a, row_b in zip(rows_a, rows_b):
                    task_pairs.setdefault(task_id, {}).setdefault("pairs", []).append(
                        (row_a, row_b)
                    )

            per_task = []
            aggregate_differences: dict[str, list[float]] = {
                metric: [] for metric in PAIRED_METRICS
            }
            for task_id, pair_group in sorted(task_pairs.items()):
                pairs = pair_group["pairs"]
                differences: dict[str, list[float]] = {
                    metric: [] for metric in PAIRED_METRICS
                }
                for row_a, row_b in pairs:
                    metric_values_a = _metric_values([row_a])
                    metric_values_b = _metric_values([row_b])
                    for metric in PAIRED_METRICS:
                        if metric_values_a[metric] and metric_values_b[metric]:
                            difference = (
                                metric_values_a[metric][0] - metric_values_b[metric][0]
                            )
                            differences[metric].append(difference)
                            aggregate_differences[metric].append(difference)
                per_task.append(
                    {
                        "task_id": task_id,
                        "matched_repetitions": len(pairs),
                        "mean_difference_a_minus_b": {
                            metric: _mean_confidence_interval(samples)
                            for metric, samples in differences.items()
                        },
                    }
                )
            comparisons.append(
                {
                    "model": model,
                    "strategy_a": strategy_a,
                    "strategy_b": strategy_b,
                    "difference_direction": "strategy_a minus strategy_b",
                    "matched_task_count": len(per_task),
                    "matched_repetition_count": sum(
                        item["matched_repetitions"] for item in per_task
                    ),
                    "mean_difference_across_paired_cases": {
                        metric: _mean_confidence_interval(samples)
                        for metric, samples in aggregate_differences.items()
                    },
                    "per_task": per_task,
                }
            )
    return comparisons


def _budget_events(row: dict[str, Any]) -> list[dict[str, Any]]:
    budget = row.get("budget")
    events = budget.get("budget_events") if isinstance(budget, dict) else None
    if not isinstance(events, list):
        return []
    return [event for event in events if isinstance(event, dict)]


def _per_call_logs(row: dict[str, Any]) -> list[dict[str, Any]]:
    calls = row.get("per_call_logs")
    if not isinstance(calls, list):
        return []
    return [call for call in calls if isinstance(call, dict)]


def _budget_violation_reasons(row: dict[str, Any]) -> list[str]:
    budget = row.get("budget")
    limits = row.get("budget_limits")
    reasons = []
    if any(event.get("event") == "token_overrun" for event in _budget_events(row)):
        reasons.append("token_overrun")
    if not isinstance(budget, dict) or not isinstance(limits, dict):
        return reasons
    metrics = (
        ("budget_tokens_charged", "max_tokens"),
        ("model_calls", "max_model_calls"),
        ("tool_calls", "max_tool_calls"),
        ("test_runs", "max_test_runs"),
        ("elapsed_seconds", "task_timeout_seconds"),
    )
    reasons.extend(
        limit
        for metric, limit in metrics
        if _is_number(budget.get(metric))
        and _is_number(limits.get(limit))
        and budget[metric] > limits[limit]
    )
    repair_attempts = row.get("repair_attempts")
    max_repairs = limits.get("max_repair_attempts")
    if _is_number(repair_attempts) and _is_number(max_repairs) and repair_attempts > max_repairs:
        reasons.append("max_repair_attempts")
    return reasons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", type=Path, default=Path("results/latest.jsonl"))
    args = parser.parse_args()
    print(json.dumps(evaluate(args.results), indent=2))


if __name__ == "__main__":
    main()
