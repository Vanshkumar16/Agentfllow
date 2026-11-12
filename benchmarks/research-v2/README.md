# AgentFlow independent benchmark v2 research artifacts

## Status

This package contains the frozen v2 dataset and protocol references and the
**five-task pilot only**. It is not the 35-task held-out evaluation and is not
evidence of statistical superiority. The planned held-out evaluation remains
315 cases (35 tasks x 3 strategies x 3 repetitions) and must be run separately.

## Frozen inputs and analysis

- Dataset: [`../tasks_v2.jsonl`](../tasks_v2.jsonl), SHA-256
  `dece185db26d52aadce87df77a9b3726dc529c4f0b16a543f455c112160b1a39`.
- Configuration: [`../../configs/experiment_v2.yaml`](../../configs/experiment_v2.yaml),
  SHA-256 `2008d3eb447ee98b9212078c6cadd4f5b8e1064f13fda853cc73dc55802620c9`.
- Frozen code revision: `bd01c0047d2736bd158311d92cdeb4ddb736624c`; clean worktree recorded:
  `True`.
- Raw pilot log: [`v2-pilot-frozen.jsonl`](raw/v2-pilot-frozen.jsonl),
  SHA-256 `095b0dbf1bfb5aa63f7d3ff64aea35265d7b7044bc4a2ec33dc16511df116c7b`.
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
  budget outcomes. Provider usage coverage excludes model-call records marked
  `budget_rejected`; those attempts are counted separately from calls that
  reached the provider but lack usage telemetry.
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

The six token-budget-rejected model-call attempts were rejected before inference
and are excluded from the provider-usage coverage denominator. They are reported
in `summary.csv` as `budget_rejected_model_call_attempts`, separate from
`missing_model_usage_calls`.

| Strategy | Strict resolution | Final tests passed | Mean latency (s) | Mean tokens | Provider usage coverage | Budget-rejected attempts |
|---|---:|---:|---:|---:|---:|---:|
| failure-guided | 1/5 (20%) | 1/5 (20%) | 35.28 | 6236 | 100.0% | 2 |
| fixed-retry | 1/5 (20%) | 1/5 (20%) | 42.77 | 6770 | 100.0% | 4 |
| single-shot | 0/5 (0%) | 0/5 (0%) | 11.90 | 1570 | 100.0% | 0 |

The pilot outcome is descriptive only. The reported confidence intervals in
`summary.csv` should not be interpreted as evidence that any strategy is
superior.

The first exploratory pilot log (`results/v2-pilot.jsonl`) predates the final
v2 dataset hash and is excluded from these tables. Only the frozen log above is
included in this package.
