# Goal: Budget-Aware Recovery for Local Software Engineering Agents

## User Request

“AgentFlow: Budget-Aware Recovery for Local Software Engineering Agents — this is my target to achieve and build something like this.”

The user selected a runnable research prototype with tests and a benchmark; budgets measured in model calls and tool/test executions, with token/cost reporting when exposed; workspace edits and tests with shell execution requiring opt-in; comparison against fixed-retry and no-recovery baselines; and the following constraints:

- Research references: ReAct, Reflexion, and SWE-bench Verified.
- Target: local NVIDIA RTX 3050 with 6 GB VRAM; CPU fallback if necessary.
- Initial model: Gemma 4 E2B; evaluate a quantized E4B variant only if it fits.
- Maximum 3 repair attempts per task; configurable inference/token budget and command timeout.
- Start with a small controlled Python bug-fixing benchmark; use SWE-bench Verified only if environment and time permit.
- Compare single-shot generation, fixed retries, and failure-guided AgentFlow recovery.
- No fine-tuning or multi-GPU infrastructure; do not make unsupported performance claims.
- Record actual experimental results before drawing conclusions.
- 120 seconds per task by default, adjustable for larger repositories.
- Pilot on 5 tasks; initially target 15–30 held-out tasks.
- Report resolution rate, test pass rate, attempts, latency, and token consumption.
- Interpret small-scale experiments as preliminary, not generalizable findings.

## Refined Goal

Extend the existing Python AgentFlow scaffold into a runnable, local-first software-engineering agent that can inspect a configured workspace, plan and make bounded code repairs, run tests, and recover from failures while enforcing explicit model-call, token, tool/test, repair-attempt, and time budgets. Provide configurable single-shot, fixed-retry, and failure-guided recovery strategies, a safe local-model integration targeting Gemma 4 E2B with CPU fallback, and an evaluation harness for controlled Python bug-fixing tasks. Report measured outcomes and environment details without claiming improvements unsupported by the actual runs; GPU-specific experiments and SWE-bench Verified remain optional when unavailable or impractical.

## Acceptance Criteria

- [ ] The existing AgentFlow package exposes a local model interface configurable for the requested Gemma 4 E2B baseline, supports CPU fallback, and reports unavailable/incompatible model or hardware conditions explicitly; a quantized E4B variant is attempted only when configured and the runtime can establish that it fits the available device.
- [ ] The agent can operate on a configured local Python workspace, inspect files, propose/apply bounded repairs, and run tests. Shell/command execution is disabled unless explicitly opted in, confined to the configured workspace, and subject to a configurable timeout; unsafe or out-of-scope actions fail visibly.
- [ ] Single-shot, fixed-retry, and budget-aware failure-guided recovery strategies are selectable and use the same task inputs and evaluation contract.
- [ ] Per-task budgets are configurable and enforced for model calls/tokens when measurable, tool/test executions, no more than 3 repair attempts by default, and 120 seconds by default. Budget exhaustion stops further work and is represented in structured results rather than treated as success.
- [ ] Structured per-task outputs and aggregate reports include resolution rate, test pass rate, attempts, latency, model calls, tool/test counts, and token consumption when available; unavailable metrics are identified as unavailable, not fabricated.
- [ ] A controlled Python bug-fixing benchmark contains at least 15 tasks, with a 5-task pilot path and held-out evaluation support. SWE-bench Verified is optional and is not required for completion.
- [ ] Automated tests cover the recovery strategies, budget boundary enforcement, command timeout/permission behavior, result accounting, and benchmark/report generation; project tests pass without requiring a live model or GPU.
- [ ] Documentation explains setup, local-model and CPU/GPU configuration, safety boundaries, evaluation procedure, baselines, limitations, and how to record results. Any experimental claims are tied to recorded runs and clearly described as preliminary.

## Scope Boundaries

**In scope:**
- Evolving the existing Python AgentFlow scaffold into a local software-engineering workflow.
- Local model runtime integration and explicit CPU fallback; GPU evaluation when suitable hardware/model artifacts are available.
- Workspace-confined file and test operations, with opt-in, bounded command execution.
- Configurable budget-aware recovery and comparable baseline strategies.
- A controlled Python task benchmark, pilot/held-out runners, structured result accounting, and documentation.

**Out of scope:**
- Model fine-tuning, multi-GPU execution, automatic deployment, or an always-on autonomous service.
- Running shell commands without explicit opt-in, destructive operations, or modifying files outside the configured workspace.
- Requiring SWE-bench Verified or claiming broad generalization from a small-scale benchmark.
- Claiming E4B, GPU, or quality results unless the corresponding model/device/run is actually available and recorded.

## Applicable Project Conventions

**Quality gate command:**
- `.\.venv\Scripts\python.exe -m pytest`
- `.\.venv\Scripts\python.exe -m compileall -q src experiments`

**Commit convention:**
- Conventional commits with goal workflow role markers: `type(scope): [B/I] description`.
- Include the applicable `Assisted-by:` trailer and the workspace-required `Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>` trailer.

**Guidelines:**
- No `AGENTS.md`, `CONSTITUTION.md`, `.agents/guidelines/`, or `.github/guidelines/` files were found.

**Rules:**
- Python package source is under `src/agentflow`; tests are under `tests`; pytest is configured in `pyproject.toml`.
- Workspace was initialized as Git for this goal workflow. Initial baseline commit: `267ed4095cd51d2540b4a1933e7651472850a763`.
