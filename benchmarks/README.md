# Controlled Python repair benchmark

`tasks.jsonl` contains 20 self-contained Python repair tasks. Each line provides
an `id`, a task `goal`, and an initial `files` mapping copied into a fresh
temporary workspace for each model/strategy run. The first five tasks are the
pilot; the remaining 15 are held out.

```powershell
python experiments/run_experiment.py --split pilot --repetitions 3 --seed 20261112 `
  --output results/pilot-repetitions.jsonl
python experiments/evaluate.py results/pilot-repetitions.jsonl
python experiments/analyze_results.py results/pilot-repetitions.jsonl
python experiments/run_experiment.py --split held-out --output results/held-out.jsonl
```

The harness compares single-shot, fixed-retry, and failure-guided strategies
with identical task inputs and per-task budgets. It does not install model
weights and records model, device, hardware, and budget metadata. `--seed`
reproducibly shuffles task/strategy order, while `--repetitions` repeats every
case with a fresh workspace and budget. Results include repetition IDs,
execution order, the Git commit, and SHA-256 hashes of the dataset and YAML
configuration. Existing output files are never overwritten; select a new
`--output` path for every run. SWE-bench Verified is an optional future
evaluation, not part of the default run.

The evaluator reports strict resolution and final test-pass rates separately,
with per-task/repetition metrics, 95% confidence intervals, and paired
comparisons for matched task/repetition cases. Its token-usage coverage and
missing-usage counts distinguish unavailable measurements from result rows
without token totals. Budget rejections are guards; budget violations are
reported separately.

## Independent benchmark v2

[`tasks_v2.jsonl`](tasks_v2.jsonl) is a new 40-task set. IDs use the `v2-`
prefix and do not overlap the original 20 `py-fix-*` tasks. The first five
fixtures are the pilot; the remaining 35 are held out and must not be used to
tune the agent or protocol. Every task has its own starter module and
independent pytest contract, with no external service or package dependency.

The task types include configuration precedence, delimited records,
pagination, cache expiry, URL normalization, retry and rate-limit policy,
bounded queues and caches, dependency graphs, ingestion and serialization,
input validation, path safety, interval scheduling, transactional state,
environment parsing, streaming, Unicode, filesystem discovery, and response
envelopes. Difficulty labels are consistent across the set: `introductory`
tasks isolate one boundary/precedence rule, `standard` tasks combine a few
related rules, and `advanced` tasks require coordinated state or graph
behavior. All tasks use the same local Python/pytest environment and provide
multiple explicit assertions including boundary or invalid-input behavior.

Run the pilot and then, after freezing the protocol, the held-out evaluation
with the dedicated configuration and separate raw logs:

```powershell
python experiments/run_experiment.py --tasks benchmarks/tasks_v2.jsonl `
  --config configs/experiment_v2.yaml --split pilot --repetitions 1 --seed 20261112 `
  --output results/v2-pilot.jsonl
python experiments/evaluate.py results/v2-pilot.jsonl
python experiments/run_experiment.py --tasks benchmarks/tasks_v2.jsonl `
  --config configs/experiment_v2.yaml --split held-out --repetitions 3 --seed 20261113 `
  --output results/v2-held-out.jsonl
python experiments/evaluate.py results/v2-held-out.jsonl
```

The held-out design is 35 tasks x 3 strategies x 3 repetitions = 315 cases.
Do not begin it until pilot execution reliability and logging are checked and
the dataset/configuration hashes and agent/protocol revision are frozen. Keep
the JSONL files unchanged; report missing observations and do not claim
strategy superiority without adequate paired evidence.

The reproducibility package and pilot-only tables/figures are in
[`research-v2/`](research-v2/README.md). Generate those artifacts from a raw run
with `python experiments/prepare_research_artifacts.py --results <jsonl>`.
The generator verifies dataset/configuration hashes and refuses to replace
existing outputs unless `--force` is provided. Pilot artifacts are not a
substitute for the held-out results.

## Frozen final evaluation

After freezing the agent implementation in a clean Git commit, run the
randomized final evaluation on the 15 held-out tasks:

```powershell
python experiments/run_final_evaluation.py --seed 20261111
python experiments/evaluate.py results/final-held-out.jsonl
python experiments/analyze_results.py results/final-held-out.jsonl
```

The runner refuses a dirty worktree or configuration that differs from the
required Gemma E2B model, 120-second task timeout, three repairs, or 8,192-token
budget. It records the frozen Git revision, config and dataset hashes, Ollama
versions, generation settings, Python/OS/GPU/driver details, the seeded
randomized task/strategy order, and a post-run `ollama ps` observation. Each
task/strategy case starts in a new temporary workspace populated from the
benchmark fixture. The first five pilot tasks are excluded from this final
evaluation; they have been used in prior development experiments and are not
described as untouched holdout tasks.

Aggregate reports include classified failure counts, budget rejection events,
token overruns, and cases that exceeded a configured budget. A budget rejection
is a guard stopping an inadmissible call; it is reported separately from a
budget violation.

The archived 45-case JSONL, frozen source revision, config, and task dataset are
historical evidence and must remain unchanged. Do not execute or tune against
the 15 frozen tasks with a later implementation. SWE-bench Verified is the
selected external benchmark for independent evaluation, but it is not present
in this workspace. The dataset was not downloaded and the independent run is
deferred until an authorized copy and a separate protocol are available.
