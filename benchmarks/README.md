# Controlled Python repair benchmark

`tasks.jsonl` contains 20 self-contained Python repair tasks. Each line provides
an `id`, a task `goal`, and an initial `files` mapping copied into a fresh
temporary workspace for each model/strategy run. The first five tasks are the
pilot; the remaining 15 are held out.

```powershell
python experiments/run_experiment.py --split pilot
python experiments/evaluate.py results/latest.jsonl
python experiments/analyze_results.py results/latest.jsonl
python experiments/run_experiment.py --split held-out --output results/held-out.jsonl
```

The harness compares single-shot, fixed-retry, and failure-guided strategies
with identical task inputs and per-task budgets. It does not install model
weights and records model, device, hardware, and budget metadata. SWE-bench
Verified is an optional future evaluation, not part of the default run.

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
