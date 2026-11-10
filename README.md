# AgentFlow: Budget-Aware Recovery for Local Software Engineering Agents

AgentFlow is a research prototype for local Python code repair. It uses a local
Ollama-compatible chat endpoint, confines file operations to a selected
workspace, runs tests only after explicit opt-in, and compares three recovery
strategies:

- **Single-shot:** one attempt, no repairs.
- **Fixed retry:** retry an unsuccessful task while budgets permit.
- **Failure-guided:** retry only when the failure evidence is actionable and
  the remaining time/token/tool/test budgets meet configured thresholds.

The pilot benchmark is a controlled set of 20 small Python bug-fixing tasks:
five pilot tasks and 15 held-out tasks. SWE-bench Verified is not required.

## Requirements and local model

- Python 3.10+
- A local Ollama-compatible server and a locally installed, supported model.
- The default model identifier is `gemma4:e2b`; change it in
  `configs/experiment.yaml` to the exact tag supported by your local runtime.
  AgentFlow does not download model weights.

Install the runtime and model separately, then install the project:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

By default, the local runtime chooses its available compute device. Force its
CPU path with `--device cpu`; use `auto` to let the runtime choose GPU or CPU.
If GPU placement fails, retry explicitly with `--device cpu`.

## Run on a workspace

```powershell
agentflow "Fix the failing Python tests in this project." --workspace . --allow-tests
```

Test execution is disabled unless `--allow-tests` is passed. Arbitrary command
execution is separately disabled by default; enabling it also requires an
executable allowlist:

```powershell
agentflow "Run the project's formatter." --workspace . --allow-tests `
  --allow-shell --allowed-command ruff
```

Command arguments are passed without a shell, commands run with the workspace
as their working directory, and each command is time-limited. Only add
executables you trust: an allowlisted program can still modify data. The agent
does not provide an operating-system sandbox.

The default strategy is failure-guided. Select a baseline and override task
budgets with flags:

```powershell
agentflow "Fix the failing Python tests." --workspace . --allow-tests `
  --strategy fixed-retry --max-repair-attempts 2 --max-tokens 6000 `
  --max-model-calls 12 --timeout 120 --command-timeout 30
```

All settings, including the default limit of at most three repair attempts and
120 seconds per task, are in `configs/experiment.yaml`. The CLI emits structured
JSON. Exit codes: `0` resolved with tests passing; `1` unresolved or budget
exhausted; `2` configuration, model, or execution setup error.

## Experiment

The default run evaluates the five-task pilot across all three strategies. Run
the 15 held-out tasks only after reviewing the pilot:

```powershell
python experiments/run_experiment.py --split pilot
python experiments/evaluate.py results/latest.jsonl
python experiments/analyze_results.py results/latest.jsonl
python experiments/run_experiment.py --split held-out --output results/held-out.jsonl
```

Each task/strategy run starts from a fresh workspace copy and fresh budgets.
Reports include resolution rate, final test pass rate, attempts, latency, model
calls, tool calls, test runs, token counts when the local provider reports them,
and measured hardware details. Monetary cost is reported as unavailable for
local inference. If the provider does not supply token usage, token metrics are
`null`; they are not estimated or fabricated.

Optional E4B candidates can be configured under `optional_e4b_models`. A
candidate is evaluated only if its exact local model tag is already installed,
an NVIDIA GPU is visible, the configured free-VRAM threshold is met, and the
runtime is not forced to CPU. The memory check is a preflight guard, not a
guarantee that model loading will succeed. AgentFlow never downloads the
candidate automatically.

## Research framing and limitations

The workflow is informed by [ReAct](https://arxiv.org/abs/2210.03629)
(reasoning and acting) and [Reflexion](https://arxiv.org/abs/2303.11366)
(failure-informed retries); SWE-bench Verified is an optional future benchmark.
This prototype's small controlled task set supports pilot observations only;
it does not establish generalization or a performance improvement. Record the
generated result files and actual runtime/model/hardware before drawing
conclusions.

## Tests

The unit suite uses fake model responses and temporary workspaces; it does not
need a live model or GPU:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m compileall -q src experiments
```

The original calculator/echo-based AgentFlow API remains available for
compatibility; install the `copilot` extra with `pip install -e ".[copilot]"`
to use its Copilot SDK adapter. The `agentflow` executable uses the local
coding-agent workflow.
