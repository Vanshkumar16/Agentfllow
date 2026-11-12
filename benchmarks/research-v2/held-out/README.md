# AgentFlow independent benchmark v2: held-out evaluation

This package contains the validated held-out evaluation evidence, derived
tables, statistical analysis, runtime environment metadata, and figures.
The 35-task held-out set was evaluated with three strategies and three
repetitions (315 scheduled cases). The five-task pilot was separate.

## Frozen protocol and reproducibility

- Dataset: [`../../tasks_v2.jsonl`](../../tasks_v2.jsonl), SHA-256
  `dece185db26d52aadce87df77a9b3726dc529c4f0b16a543f455c112160b1a39`.
- Configuration: [`../../../configs/experiment_v2.yaml`](../../../configs/experiment_v2.yaml), SHA-256
  `2008d3eb447ee98b9212078c6cadd4f5b8e1064f13fda853cc73dc55802620c9`.
- Commit: `8a649d4b6cb9b497556db747a1962fb7a91f39ca` (clean worktree recorded: `True`).
- Model: `gemma4:e2b`; Ollama model digest is in `environment.json`.
- Random seed: `20261113`; it randomizes task/strategy order, not inference sampling.
- Temperature: `0.1`.
- Raw evidence: [`raw/v2-held-out.jsonl`](raw/v2-held-out.jsonl), SHA-256 `d9aa01cf08172116a2cdf385a64051465ac9068485dfbfa493df4fdad3945ce0`.
- The original frozen pilot evidence was not changed.

Reproduce from the repository root with the frozen code, config, dataset,
and locally installed model available:

```powershell
$env:PYTHONPATH = 'src'
python experiments/run_experiment.py --tasks benchmarks/tasks_v2.jsonl `
  --config configs/experiment_v2.yaml --split held-out --repetitions 3 `
  --seed 20261113 --output results/v2-held-out.jsonl
# An interrupted run leaves results/v2-held-out.jsonl.partial; inspect it before retrying.
python experiments/validate_heldout_results.py results/v2-held-out.jsonl
python experiments/evaluate.py results/v2-held-out.jsonl
python experiments/analyze_results.py results/v2-held-out.jsonl
python experiments/prepare_heldout_research_artifacts.py `
  --results results/v2-held-out.jsonl `
  --output-dir benchmarks/research-v2/held-out
```

## Results

Strict resolution and final test-pass rate are separate outcomes. Values
and task-clustered 95% confidence intervals are descriptive; no strategy
is claimed superior without evidence supporting that conclusion.

| Strategy | Strict resolution (95% task CI) | Final tests passed (95% task CI) | Mean repairs | Mean latency (s) | Mean tokens | Usage coverage | Rejected calls | Violations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| failure-guided | 21.9% [9.5%, 35.2%] | 22.9% [10.5%, 36.2%] | 0.89 | 35.02 | 6259 | 100.0% | 43 | 0 |
| fixed-retry | 21.9% [11.4%, 33.3%] | 27.6% [15.2%, 41.0%] | 1.54 | 41.65 | 6736 | 100.0% | 82 | 0 |
| single-shot | 5.7% [0.0%, 13.3%] | 5.7% [0.0%, 13.3%] | 0.00 | 12.00 | 1582 | 100.0% | 0 | 0 |

## Paired strategy differences

Each estimate is strategy A minus strategy B in percentage points; negative
values favor strategy B for these success-rate metrics.

| Pair | Matched tasks | Strict resolution difference (95% task CI) | Final test-pass difference (95% task CI) |
|---|---:|---:|---:|
| single-shot vs fixed-retry | 35 | -16.2 pp [-28.6, -4.8] | -21.9 pp [-36.2, -8.6] |
| single-shot vs failure-guided | 35 | -16.2 pp [-29.5, -3.8] | -17.1 pp [-31.4, -3.8] |
| fixed-retry vs failure-guided | 35 | +0.0 pp [-5.7, +5.7] | +4.8 pp [+0.0, +10.5] |

## Statistical method and limitations

- Strategy estimates use a seeded task-clustered percentile bootstrap with 10,000 resamples; repetitions remain within each sampled task.
- Paired comparisons first match strategy observations by task and repetition, average differences within task, and then bootstrap tasks. The evaluator reports matched task/repetition counts and missing metric coverage.
- Pairwise intervals are exploratory and are not multiplicity-adjusted; they are not by themselves a confirmatory superiority test.
- Repetition-level means and standard deviations are reported separately from task-clustered estimates.
- Provider usage coverage excludes budget-rejected calls, which are reported separately. Missing provider usage and missing result-level token totals are reported explicitly.
- Budget rejection events and budget violations are distinct and shown in `budget_analysis.csv`.
- Model execution is local. `environment.json` records the runtime versions, model digest, hardware/GPU driver, and available Ollama process-list observations; unreported effective device details are not inferred.
- Inference is not deterministic even with a fixed execution-order seed. The seed does not control model sampling.

## Files

- `raw/v2-held-out.jsonl`: immutable raw run log.
- `environment.json`: hashes, software, hardware, runtime, and completion metadata.
- `summary.csv`: repetition-level and task-clustered strategy metrics, CIs, and token-usage coverage.
- `task_metrics.csv`: per-task/per-strategy metrics and repetition dispersion.
- `paired_task_comparisons.csv`: task-level paired differences for all strategy pairs.
- `statistical_analysis.json`: bootstrap method, estimates, paired comparisons, and uncertainty.
- `budget_analysis.csv`: provider-usage gaps, budget rejections, and violations.
- `failure_categories.csv`: observed failures by strategy.
- `heldout-*.svg`: descriptive success, latency, token, and repair charts.
