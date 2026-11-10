# Project readiness and blockers

Checked: 2026-11-11

## Summary

The configured `gemma4:e2b` model is installed and responds to local
generation. The updated pre-generation budget controller passed its automated
tests and was exercised in the 5-task pilot across all three strategies. The
failure-guided path now classifies errors and suppresses repeated identical
retries. All 43 repository unit tests pass. Results are preliminary: 14/15
strategy cases resolved, with the single-shot baseline accounting for the one
unresolved case.

## What works

- The unit suite passes: 43 tests.
- Python modules compile, and the installed dependency check passes.
- The experiment runner, three strategy paths, and controlled 20-task
  benchmark are available.
- The budget controller now estimates prompt tokens and reserves bounded
  output tokens before each generation, records actual provider usage, and
  stores structured rejection/overrun events.
- Failure-guided attempts classify syntax, import, assertion, runtime,
  timeout, malformed-response, repeated-identical, and unknown failures;
  attempts store the category and recovery action.
- Timeouts stop automatic failure-guided retries; the agent also skips the
  configured test run when the model/action itself has timed out.
- Repeated identical failure fingerprints stop failure-guided recovery before
  another retry. The fixed-retry strategy remains an unconditional baseline.
- The local Ollama-compatible endpoint is reachable.
- `ollama show gemma4:e2b` confirms the installed tag uses the `gemma4`
  architecture and Q4_K_M quantization. Ollama reports 4.6B parameters for
  this package.
- The recorded hardware preflight sees an NVIDIA GeForce RTX 3050 6GB Laptop
  GPU. The endpoint does not report whether inference actually uses the GPU.

## Pilot result after budget-controller update (5 tasks)

This pilot predates the failure-classification change below. The classification
and repeated-failure behavior is covered by the automated tests, but has not
been rerun against the live model.

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

- The pilot is only five controlled tasks and is not statistically conclusive.
- The 15 held-out tasks and SWE-bench Verified have not been run.
- One single-shot task received an empty/malformed model response.
- Exact prompt tokenization is not available from the configured Ollama
  adapter; the controller uses a conservative estimate and reports overruns.
- AgentFlow's result JSON cannot independently identify actual device
  placement; the GPU observation came from `ollama ps` after the pilot.
- The unit suite validates code with fake models; it does not establish model
  quality beyond the recorded pilot.

Review the pilot before running the held-out split. Do not claim strategy
superiority or broad generalization from this small sample.
