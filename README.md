# Agentflow

Agentflow is a small Python orchestration example that runs a task through
**plan → execute → test → critique → recover/replan**. It uses the GitHub
Copilot SDK for planning and critique; tool execution is restricted to an
explicit local allowlist (`calculator` and `echo`).

## Requirements

- Python 3.10+
- GitHub Copilot CLI installed and authenticated

## Install and run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
copilot --version
agentflow "Calculate 17 * 23 and report the result."
```

The default settings are in `configs/experiment.yaml`. Override them per run:

```powershell
agentflow "Calculate 17 * 23." --model gpt-4.1 --max-iterations 3 --retries-per-step 1
```

Results are emitted as JSON to stdout. Exit code `0` means the critic confirmed
completion; exit code `1` means the iteration limit was reached without
confirmation. Configuration, model, and execution errors are reported with
exit code `2`.

## Workflow

1. **Planner** requests a JSON plan constrained to registered tools.
2. **Executor** runs steps sequentially and records outputs or errors.
3. **Test runner** checks that each planned step produced a result and that no
   tool failed.
4. **Recovery** retries failed tool steps up to the configured bound.
5. **Critic** assesses completion from the goal, tool evidence, and checks.
   If incomplete, the planner receives its feedback and creates a revised plan.

The orchestration is covered by fake-model tests and does not require a live
Copilot connection to run the test suite:

```powershell
pytest
```

## Experiments

Run the JSONL benchmark and summarize its results:

```powershell
python experiments/run_experiment.py
python experiments/evaluate.py results/latest.jsonl
python experiments/analyze_results.py results/latest.jsonl
```
