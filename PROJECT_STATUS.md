# Project readiness and evaluation results

Checked: 2026-11-11

## Summary

The configured `gemma4:e2b` model is installed and responds to local
generation. The pre-generation budget controller and structured failure-guided
recovery are frozen. The randomized final evaluation completed all 45
task/strategy cases on the 15 held-out tasks. All 46 repository tests pass.
These results are small-sample observations, not evidence of strategy
superiority or broad generalization.

## What works

- The unit suite passes: 46 tests.
- Python modules compile, and the installed dependency check passes.
- The experiment runner, three strategy paths, and controlled 20-task
  benchmark are available.
- The budget controller now estimates prompt tokens and reserves bounded
  output tokens before each generation, records actual provider usage, and
  stores structured rejection/overrun events.
- Failure-guided attempts classify syntax, import, assertion, runtime,
  timeout, malformed-response, repeated-identical, and unknown failures;
  attempts store the category and recovery action.
- Failure-guided timeouts schedule a bounded replan when remaining time, token,
  call, and test budgets permit it. The next attempt is directed to choose a
  smaller alternative; otherwise recovery stops. The agent skips the configured
  test run within an attempt when its model/action has timed out.
- Repeated identical failure fingerprints stop failure-guided recovery before
  another retry. The fixed-retry strategy remains an unconditional baseline.
- The local Ollama-compatible endpoint is reachable.
- `ollama show gemma4:e2b` confirms the installed tag uses the `gemma4`
  architecture and Q4_K_M quantization. Ollama reports 4.6B parameters for
  this package.
- The recorded hardware preflight sees an NVIDIA GeForce RTX 3050 6GB Laptop
  GPU. The endpoint does not report whether inference actually uses the GPU.

## Frozen final held-out evaluation

See [benchmarks/final-evaluation-report.md](benchmarks/final-evaluation-report.md)
for the full methodology, metadata, interpretation, and limitations.

| Strategy | Resolved | Final tests passed | Mean repair attempts | Mean latency | Mean tokens |
|---|---:|---:|---:|---:|---:|
| Single-shot | 10/15 (66.7%) | 10/15 (66.7%) | 0.00 | 10.44 s | 1,012.38 (13 measured) |
| Fixed retry | 8/15 (53.3%) | 10/15 (66.7%) | 1.07 | 37.01 s | 4,674.55 (11 measured) |
| Failure-guided | 8/15 (53.3%) | 8/15 (53.3%) | 0.67 | 29.98 s | 4,141.33 (12 measured) |

There were nine budget rejections, zero token overruns, and zero budget
violations. Fixed-retry had two cases with final tests passing but an
`unresolved` agent outcome (`py-fix-019`, `py-fix-020`); the metrics preserve
these separate outcomes because the model did not issue the required finish
action. Nine task/strategy cases have missing provider token totals, 36 have
positive measured totals, and zero report zero usage. The five pilot tasks were
used during development and excluded from the final evaluation.

The per-case log, exact config and dataset snapshots, hashes, and source
revisions are preserved in
[benchmarks/evidence/final-held-out/](benchmarks/evidence/final-held-out/).
See the [audited final evaluation report](benchmarks/final-evaluation-report.md)
for paired task results, token-accounting caveats, and the response-failure
taxonomy.

## Pilot result after budget-controller update (5 tasks)

This pilot predates the failure-classification change and is historical; it
must not be combined with the final held-out results.

The run used `gemma4:e2b`, `device: auto`, max 3 repairs, max 8,192 measured
tokens, and a 120-second per-task limit. Evaluation results:

| Strategy | Resolved / tests passed | Mean repair attempts | Mean latency | Mean tokens |
|---|---:|---:|---:|---:|
| Single-shot | 4/5 (80%) | 0.00 | 11.56 s | 627.75 across 4 measured tasks |
| Fixed retry | 5/5 (100%) | 0.00 | 21.41 s | 2,815.4 |
| Failure-guided | 5/5 (100%) | 0.00 | 20.29 s | 2,763.8 |

`py-fix-003` failed in single-shot because the model returned an empty or
malformed response. Single-shot token usage is unavailable for that response.
Across the 15 task/strategy records, there were no token-budget rejection or
overrun events. The highest accounted per-task token budget use was 3,648 of
8,192 tokens.

The controller now estimates prompt length as
`max(ceil(characters / 2), ceil(UTF-8 bytes / 3))` and reserves an output
allowance before generation. The estimate is not the model's exact
tokenization: Ollama's adapter provides actual counts only after generation.
An underestimate can still cause a logged overrun, after which further
recovery calls stop.
The refreshed pilot did not encounter an estimate overrun, but this five-task
sample does not guarantee that future prompts will be estimated exactly.

The hardware preflight detected an NVIDIA GeForce RTX 3050 6GB Laptop GPU.
After the run, `ollama ps` showed `gemma4:e2b` at 100% GPU. AgentFlow's own
metadata leaves `device_selected` unknown because the Ollama API does not
report placement.

The result records are in `results/latest.jsonl` (local, ignored by Git).
Refresh the aggregate report with:

```powershell
python experiments/evaluate.py results/latest.jsonl
python experiments/analyze_results.py results/latest.jsonl
```

## Remaining work and limitations

- Both the pilot and final benchmark are small controlled task sets and are
  not statistically conclusive; SWE-bench Verified has not been run.
- One single-shot task received an empty/malformed model response.
- Exact prompt tokenization is not available from the configured Ollama
  adapter; the controller uses a conservative estimate and reports overruns.
- AgentFlow's result JSON cannot independently identify actual device
  placement; the GPU observation came from `ollama ps` after the pilot.
- The unit suite validates code with fake models; it does not establish model
  quality beyond the recorded pilot.

Review the pilot before running the held-out split. Do not claim strategy
superiority or broad generalization from this small sample.
