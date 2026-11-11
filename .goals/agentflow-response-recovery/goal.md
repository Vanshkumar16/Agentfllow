# Goal: Reliable response and recovery pipeline

## User Request

Improve the existing AgentFlow implementation for structured Ollama response
handling, failure-specific recovery, strict budget accounting, completion
verification, and reproducible per-call experiment logging. Preserve the
original 45-case evaluation unchanged, do not tune against its frozen tasks,
and defer an independent benchmark if a suitable external benchmark is
unavailable. Document original results, limitations, and independent
evaluation as future work.

## Refined Goal

Improve AgentFlow's existing local Ollama coding-agent workflow without
unnecessarily rewriting working components. Implement and verify schema-driven
response validation, explicit failure-specific recovery, centralized
budget and measured-usage accounting, distinct patch/test/protocol completion
outcomes, and per-call experiment telemetry. Preserve the original frozen
implementation revision and all 45-case run evidence exactly; use only
non-held-out development cases for development checks. Do not run or tune on
the frozen 15-task evaluation set. An independent benchmark may be run only if
a suitable external dataset is available; otherwise defer it and document why.

## Acceptance Criteria

- [ ] Ollama model responses use supported structured-output/schema features
  when available and compatible; every response is checked for non-empty
  content, JSON validity, and schema validity before any action or patch is
  applied.
- [ ] Incomplete or truncated responses are distinguished only when provider
  termination metadata supports that conclusion; otherwise they remain
  uncertain/incomplete. Preserve safe diagnostic response/provider metadata.
- [ ] Invalid responses can be regenerated within existing call, repair, time,
  and token limits; failure classification and recovery decisions carry
  actionable evidence, attempt context, and explicit RETRY/REVISE/REPLAN/STOP
  rationale, including normalized repeated-failure detection.
- [ ] Central budget enforcement preserves the existing experimental ceilings
  (8,192 total tokens, 16 model calls, 24 tool calls, 4 test runs, 3 repairs,
  120-second task timeout, 60-second model timeout, 30-second command timeout,
  1,024 output-token per-call cap). It records measured usage separately from
  estimates and reserved/charged tokens, handles unavailable provider usage
  conservatively, and records rejection and overrun events distinctly.
- [ ] Results distinguish patch validity, final test pass, explicit agent
  completion, budget exhaustion, and strict resolved status; a passing patch
  without an explicit completion signal has a recorded reason.
- [ ] Per-call model/tool/test telemetry is machine-readable and correlated
  with experiment, task, strategy, attempt, model, recovery action, latency,
  usage and validation outcomes. Existing aggregate records remain usable or
  are migrated/documented without changing archived records.
- [ ] Single-shot, fixed-retry, and improved failure-guided strategies remain
  separate; improved recovery behavior does not alter baseline policies.
- [ ] Add tests for the specified response, recovery, accounting, completion,
  and logging cases without weakening existing assertions. The full unit test
  suite, compilation check, and dependency check pass.
- [ ] Development evaluation uses non-held-out cases only and reports actual
  outcomes without hardcoded success or fabricated measurements.
- [ ] The original 45-case JSONL, frozen agent source revision, config, and
  dataset archive remain byte-identical. No execution, tuning, or replacement
  of the original frozen evaluation tasks occurs during this goal.
- [ ] README and research documentation report the original measured results,
  limitations, and deferred independent evaluation. If no suitable external
  benchmark is available, state that it was not run and why; do not create a
  hurried synthetic benchmark.

## Scope Boundaries

**In scope:**
- Evolve the current local Ollama response adapter, coding agent, failure
  classifier, budget tracker, experiment runner/evaluator, tests, and docs as
  needed to satisfy the acceptance criteria.
- Use only development fixtures outside the frozen 15-task evaluation split.
- Preserve the original evaluation as historical baseline evidence and archive.
- Identify available external benchmark options and record availability or
  blockers without silently substituting a new benchmark.

**Out of scope:**
- Re-running, tuning against, editing, regenerating, or replacing any part of
  the original 45-case final evaluation.
- Claiming improved performance without independent measured evidence.
- Changing the configured maximum repair-attempt count to increase success.
- Downloading benchmark datasets or model weights, contacting external
  services, or adding dependencies without necessity and user approval.
- Creating a synthetic independent evaluation set when no suitable external
  dataset is available.
- Rewriting unrelated planner/executor/controller compatibility surfaces.

## Applicable Project Conventions

**Quality gate command:**
- `.\.venv\Scripts\python.exe -m pytest -q`
- `.\.venv\Scripts\python.exe -m compileall -q src experiments tests`
- `.\.venv\Scripts\python.exe -m pip check`

**Commit convention:**
- Conventional commits are used in repository history. Goal iteration commits
  use `type(scope): [B/I] imperative description`, title at most 72 characters.
- Include `Assisted-by: OpenAI:GPT-5.6 Luna` for Builder commits and
  `Assisted-by: OpenAI:GPT-5.6 Sol` for Inspector commits.
- Include the repository-required trailer:
  `Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>`.

**Guidelines:**
- No `AGENTS.md`, `CONSTITUTION.md`, `.agents/guidelines/`, or
  `.github/guidelines/` files were found.
- Project metadata and pytest configuration are in `pyproject.toml`.

**Rules:**
- Python package supports Python 3.10+ and uses PyYAML as its runtime
  dependency; pytest and pytest-asyncio are test extras.
- Existing frozen evaluation evidence and its source/config/dataset hashes are
  preserved under `benchmarks/evidence/final-held-out/`.
- The existing per-task ceilings are configured in `configs/experiment.yaml`.
