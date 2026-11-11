# Final held-out evidence archive

This directory preserves the final evaluation evidence as run. The JSONL
contains 45 task/strategy results, one run-metadata record, one completion
record, and one post-run runtime observation.

## Files and hashes

| File | SHA-256 |
|---|---|
| `final-held-out.jsonl` | `79a055b3be67fc85294a2334207c50e07a0211975d2ee78f4a09a8b2fbe619ec` |
| `experiment.yaml` | `2008d3eb447ee98b9212078c6cadd4f5b8e1064f13fda853cc73dc55802620c9` |
| `tasks.jsonl` | `b67ac971ead28a70a27a43f396c54cb5344abcfaf213bc5b77fdc9e4c0b7fff9` |

These are byte-for-byte copies of the files used for the run. The archived
dataset contains all 20 benchmark tasks. The five pilot tasks
(`py-fix-001` through `py-fix-005`) were excluded; the other 15 tasks comprise
the evaluated held-out split.

## Source revisions

- Frozen agent: `7e1ab1ea56f93136c1c06dddf22f6b598d263288`
- Evaluation harness: `8fa80e36c93ba62dc033aa418754d2eaa873bda7`

The agent source tree was unchanged from its frozen revision during the run.
Both commits are in the repository's Git history. The held-out run used seed
`20261111`, Gemma `gemma4:e2b`, Ollama 0.40.13, temperature 0.1, a 120-second
task timeout, at most three repairs, and an 8,192-token budget.

See [the audited evaluation report](../../final-evaluation-report.md) for
outcomes, token-usage caveats, failure classification, and paired task results.
