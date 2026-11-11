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

## Evaluation status

The frozen final evaluation has completed all 45 cases: each of the 15 held-out
tasks was run under single-shot, fixed-retry, and failure-guided recovery. The
five pilot tasks used during development were excluded. Results are small-sample
observations and do not establish strategy superiority.

| Strategy | Resolved | Final tests passed | Mean repairs | Mean latency | Mean tokens used |
|---|---:|---:|---:|---:|---:|
| Single-shot | 10/15 (66.7%) | 10/15 (66.7%) | 0.00 | 10.44 s | 1,012.38 (13 measured) |
| Fixed retry | 8/15 (53.3%) | 10/15 (66.7%) | 1.07 | 37.01 s | 4,674.55 (11 measured) |
| Failure-guided | 8/15 (53.3%) | 8/15 (53.3%) | 0.67 | 29.98 s | 4,141.33 (12 measured) |

There were nine budget rejections, zero token overruns, and zero cases exceeding
the configured budgets. Rejections are protected refusals, not violations.
Fixed-retry had two cases (`py-fix-019`, `py-fix-020`) whose final tests passed
but whose agent outcome was marked unresolved because the model did not signal
completion; resolution and final test pass are reported separately. Nine
task/strategy records have missing provider token totals, 36 have positive
measured totals, and none report zero usage. Token averages include only cases
with valid provider-reported usage.

The run used Gemma `gemma4:e2b` with Ollama 0.40.13, temperature 0.1, 120
seconds per task, up to three repairs, and an 8,192-token budget. It ran on
Windows 11 / Python 3.14.6 with an NVIDIA GeForce RTX 3050 Laptop GPU. The
frozen agent revision is `7e1ab1ea56f93136c1c06dddf22f6b598d263288`; the
evaluation harness revision is `8fa80e36c93ba62dc033aa418754d2eaa873bda7`.
See the [audited final evaluation report](benchmarks/final-evaluation-report.md),
[benchmark protocol](benchmarks/README.md), and the [archived evidence bundle](benchmarks/evidence/final-held-out/).
The archive preserves the exact JSONL log, config, dataset, their hashes, and
the source revisions.

## Failure-guided recovery

Failure-guided attempts classify failure evidence and attach a recovery action
to each attempt:

| Failure category | Recovery action |
|---|---|
| Syntax error | Revise the patch to fix syntax. |
| Import error | Inspect dependencies, module paths, and imports. |
| Test assertion failure | Re-evaluate the expected behavior and task contract. |
| Runtime exception | Inspect the traceback and fix its underlying cause. |
| Timeout | Stop the timed-out work; do not automatically repeat it. |
| Empty or malformed response | Retry with controlled regeneration using the required JSON schema. |
| Repeated identical failure | Stop instead of repeating an unchanged repair. |

The attempt records include `failure_category`, `recovery_action`, and
`recovery_decision`. A repeated failure is detected by comparing normalized
failure fingerprints; fixed-retry remains an unconditional baseline, while
failure-guided recovery stops when an identical failure recurs.

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

After running the pilot, refresh the summary with the evaluation and analysis
commands above, then update the measured results in this README if you rerun
with a different model, configuration, or hardware.

For a reproducible final evaluation, freeze the agent in a clean Git commit
and run the seeded held-out protocol described in
[benchmarks/README.md](benchmarks/README.md). It excludes the five tasks
previously used for pilot development and records the frozen revision,
randomized order, environment, settings, failure categories, and budget
violations.

Before each generation, AgentFlow estimates prompt tokens and reserves an
output allowance, passing that allowance to Ollama as `num_predict`. Calls
that cannot fit an estimated prompt and at least one output token are rejected.
Ollama does not expose a tokenizer endpoint through this adapter, so prompt
counting estimates `max(ceil(characters / 2), ceil(UTF-8 bytes / 3))` rather
than using the model's exact tokenizer. Actual provider-reported input, output,
and total counts are recorded after each call; any estimate shortfall or
total-token overrun is logged in the run's `budget_events`, and further
recovery calls stop after a budget overrun. The reservation reduces overruns
but cannot guarantee an exact pre-generation input-token count without the
model tokenizer.

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
