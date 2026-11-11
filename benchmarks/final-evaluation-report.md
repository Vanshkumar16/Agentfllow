# Frozen held-out evaluation report

## Result

The frozen evaluation completed all 45 task/strategy cases: 15 held-out tasks
run once each with single-shot, fixed-retry, and failure-guided recovery.
Results are exploratory measurements on this small benchmark, not evidence that
one strategy generalizes or is superior.

| Strategy | Resolved | Final tests passed | Mean repairs | Mean latency | Mean tokens used |
|---|---:|---:|---:|---:|---:|
| Single-shot | 10/15 (66.7%) | 10/15 (66.7%) | 0.00 | 10.44 s | 1,012.38 (13/15 measured) |
| Fixed retry | 8/15 (53.3%) | 10/15 (66.7%) | 1.07 | 37.01 s | 4,674.55 (11/15 measured) |
| Failure-guided | 8/15 (53.3%) | 8/15 (53.3%) | 0.67 | 29.98 s | 4,141.33 (12/15 measured) |

Resolution is the agent's recorded `resolved` outcome; final test pass is
reported independently from the last recorded test result. For fixed-retry,
`py-fix-019` and `py-fix-020` had passing final tests but were recorded as
unresolved. The agent defines an attempt as passed only when the model emits
the explicit `finish` action, the configured tests pass, and the budget has not
stopped the run. Both fixed-retry cases had passing tests on their final
attempt, but no `finish` action: generation returned empty content after the
patch had been applied. Thus `resolved=false` is consistent with the agent's
strict completion protocol, but is a false negative if benchmark resolution
means "the patch currently passes the tests." The test-pass metric captures
that distinction. For `py-fix-019`, fixed-retry's repair corrected the
left-rotation behavior; for `py-fix-020`, its repair corrected the `1` prime
case. The agent was not changed after the evaluation.

## Paired task-level outcomes

Each cell is `resolved/final-test-passed (repair attempts)`. The test result is
the last recorded test run for that task/strategy. Rows are paired by task ID;
`1/0` means the agent reported completion but the final test failed, and
`0/1` means tests passed without an explicit agent completion signal.

| Task | Single-shot | Fixed retry | Failure-guided |
|---|---:|---:|---:|
| `py-fix-006` | 1/1 (0) | 1/1 (0) | 1/1 (0) |
| `py-fix-007` | 1/1 (0) | 1/1 (0) | 1/1 (0) |
| `py-fix-008` | 1/1 (0) | 1/1 (0) | 1/1 (0) |
| `py-fix-009` | 0/0 (0) | 0/0 (2) | 0/0 (1) |
| `py-fix-010` | 1/1 (0) | 0/0 (3) | 0/0 (2) |
| `py-fix-011` | 1/1 (0) | 1/1 (0) | 1/1 (0) |
| `py-fix-012` | 1/1 (0) | 1/1 (0) | 0/0 (1) |
| `py-fix-013` | 1/1 (0) | 1/1 (0) | 1/1 (0) |
| `py-fix-014` | 0/0 (0) | 0/0 (3) | 0/0 (1) |
| `py-fix-015` | 1/1 (0) | 0/0 (3) | 0/0 (2) |
| `py-fix-016` | 1/1 (0) | 1/1 (0) | 1/1 (0) |
| `py-fix-017` | 0/0 (0) | 0/0 (2) | 0/0 (2) |
| `py-fix-018` | 1/1 (0) | 1/1 (0) | 1/1 (0) |
| `py-fix-019` | 0/0 (0) | 0/1 (1) | 0/0 (1) |
| `py-fix-020` | 0/0 (0) | 0/1 (2) | 1/1 (0) |

This paired view shows that all three strategies agreed on 11 tasks under the
strict `resolved` metric. Single-shot alone resolved `py-fix-010` and
`py-fix-015`; fixed-retry alone resolved none; failure-guided alone resolved
`py-fix-020`. Fixed-retry's tests passed without a completion signal on
`py-fix-019` and `py-fix-020`.

| Strategy | Budget rejections | Token overruns | Budget-violation cases |
|---|---:|---:|---:|
| Single-shot | 0 | 0 | 0 |
| Fixed retry | 5 | 0 | 0 |
| Failure-guided | 4 | 0 | 0 |

Rejections are controller guard events that declined a call which could not fit
the estimated prompt and reserved output allowance. They are distinct from
budget violations.

## Token accounting audit

| Strategy | Positive measured totals | Missing totals | Exactly zero totals |
|---|---:|---:|---:|
| Single-shot | 13/15 | 2/15 | 0 |
| Fixed retry | 11/15 | 4/15 | 0 |
| Failure-guided | 12/15 | 3/15 | 0 |

The nine missing totals have null input, output, and total usage; they are not
zero-token calls. In eight of those nine task/strategy records, at least one
attempt explicitly records that the Ollama client rejected empty or non-string
response content. The remaining missing record is failure-guided
`py-fix-009`; its aggregate JSONL does not identify which call lacked provider
usage. The log does not preserve per-call usage records, so it is not possible
to attribute each null aggregate to a specific call or distinguish a valid
response with missing usage fields from every other missing-usage path.

When usage is unavailable, the budget controller charges the prompt estimate
and reserved output allowance to enforce the cap; that `budget_tokens_charged`
is a safety-accounting value, not measured token consumption. The token means
above use only positive provider-reported totals. There are no observed
zero-usage totals.

## Malformed-response audit

The following are attempt counts based on the recorded failure text, not
unique tasks. A repeated parse failure is counted at each occurrence.

| Strategy | Empty/non-string content rejected by client | JSON parser failures | JSON schema/patch validation failures | Truncation attribution |
|---|---:|---:|---:|
| Single-shot | 2 | 3 | 0 | Not recorded |
| Fixed retry | 6 | 10 | 0 | Not recorded |
| Failure-guided | 2 | 9 | 0 | Not recorded |

The Ollama adapter maps blank and non-string `message.content` to the same
exception, so those two cases cannot be separated from the saved evidence.
Parser failures include unterminated strings and missing delimiters. They may
be consistent with truncated generation, but the raw model output and Ollama
`done_reason` were not saved; none can be confidently labeled as truncated.
The ten empty/non-string-content events are failed completion calls. The 22
JSON parser failures occurred after response content reached the JSON parser;
the aggregate evidence does not retain per-call metadata to correlate their
individual provider usage.
There were no recorded valid-JSON schema or patch-validation errors. The
failure classifier's broad `empty_or_malformed_response` category combines
some parser failures and other errors, so the counts above were re-audited
from the specific stored failure text rather than copied from that category.

## Protocol and environment

- Model: `gemma4:e2b`, Gemma 4.6B, GGUF Q4_K_M; Ollama CLI and server 0.40.13.
- Platform: Windows 11 (10.0.26200), Python 3.14.6; NVIDIA GeForce RTX 3050
  Laptop GPU, 6 GB VRAM, driver 616.92. Device was requested as `auto`;
  Ollama showed 100% GPU in the post-run process observation, while effective
  device selection was not available per case.
- Generation: temperature 0.1, non-streaming; `num_predict` bounded by the
  output reservation and capped at 1,024.
- Budgets: 8,192 tokens, 120 seconds per task, at most three repair attempts;
  configured model-call, tool-call, test-run, and command timeouts were also
  applied.
- Test command: `python -m pytest -q`.
- Each case got a new temporary workspace populated from the same benchmark
  fixture. Execution order was shuffled with seed `20261111`.
- The five pilot tasks (`py-fix-001` through `py-fix-005`) were previously used
  during development and were excluded. These are the remaining 15 tasks, not
  untouched tasks.
- Frozen agent revision:
  `7e1ab1ea56f93136c1c06dddf22f6b598d263288`.
- Evaluation harness revision:
  `8fa80e36c93ba62dc033aa418754d2eaa873bda7`.
- Dataset SHA-256:
  `b67ac971ead28a70a27a43f396c54cb5344abcfaf213bc5b77fdc9e4c0b7fff9`.
- Config SHA-256:
  `2008d3eb447ee98b9212078c6cadd4f5b8e1064f13fda853cc73dc55802620c9`.
- Complete evidence bundle: [benchmarks/evidence/final-held-out/](evidence/final-held-out/).
  It contains byte-for-byte copies of the JSONL run log, config, and benchmark
  dataset, with hashes and both source revisions recorded in its README.

The complete randomized order, per-case records, hashes, environment metadata,
budget events, and completion record are in the archived JSONL. It contains 45
results, one metadata record, one completion record, and one post-run runtime
observation.

## Interpretation and limitations

Single-shot had the highest strict resolution rate and lowest mean latency
and measured token use in this run. Fixed-retry's final-test pass rate matched
single-shot despite two passing patches lacking an explicit completion
signal. Failure-guided recovery did not outperform these baselines on this
sample. The frequency of blank-content and JSON parser failures is a notable
observed response-format issue; the retained data cannot establish its cause
or determine whether parser failures were truncated generations.

The sample is small, task-specific, and uses one model and runtime. Token
means are based on incomplete provider-reported usage, and prompt token counts
are estimated rather than obtained from Gemma's exact tokenizer. Results do
not establish statistical significance or broad task performance. Do not tune
the agent against these evaluation tasks and present a rerun as an untouched
holdout.

## Subsequent response-recovery work and independent evaluation

The response-validation and recovery implementation developed after this
freeze is not represented by the measurements above. The archived results,
their source/config/dataset copies, and frozen source revision remain
historical baseline evidence; the frozen 15 tasks were not executed or used
for development checks during that subsequent work.

SWE-bench Verified is the selected external benchmark for independent
evaluation. No authorized dataset copy is available in this workspace, and
downloading a dataset was outside the approved scope, so the independent
evaluation was deferred. No replacement or synthetic benchmark was created.
The response-recovery changes therefore have unit/development validation only;
they have no claimed performance gain or independent task-resolution result.
