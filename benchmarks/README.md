# Benchmarks

Each line in `tasks.jsonl` is a JSON object with an optional `id` and a required
`goal`. Run the benchmark with:

```powershell
python experiments/run_experiment.py
python experiments/evaluate.py results/latest.jsonl
python experiments/analyze_results.py results/latest.jsonl
```

The default benchmark uses the same tool allowlist as the interactive agent.
