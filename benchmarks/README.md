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
