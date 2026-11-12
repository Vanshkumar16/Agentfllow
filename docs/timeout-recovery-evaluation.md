# AgentFlow: Timeout Recovery Evaluation

## 1. Objective

Evaluate whether AgentFlow correctly handles model, test-runner, and command timeouts through bounded failure-guided replanning.

The evaluation focuses on two requirements:

1. Replan when a timeout occurs and sufficient recovery budget remains.
2. Stop when recovery is unavailable because of token, time, or repair-attempt limits.

## 2. Evaluation Methodology

The evaluation uses deterministic fault injection.

- A scripted model supplies predefined responses.
- Timeout conditions are injected into model, test-runner, and command execution paths.
- Each scenario runs in an isolated temporary workspace.
- Expected recovery decisions are compared with observed decisions.
- Budget events and final task resolution are recorded.

This evaluation does not use live Ollama inference.

## 3. Experimental Scenarios

| ID | Scenario | Expected decision |
|---|---|---|
| T1 | Model timeout with sufficient budget | REPLAN |
| T2 | Test-runner timeout with sufficient budget | REPLAN |
| T3 | Command timeout with sufficient budget | REPLAN |
| T4 | Insufficient remaining token budget | STOP |
| T5 | Insufficient remaining time budget | STOP |
| T6 | Repair-attempt limit reached | STOP |
| T7 | Timeout followed by an alternative action | REPLAN and resolve |

## 4. Results

**Observed result: 7/7 scenarios passed.**

| Metric | Result |
|---|---:|
| Total scenarios | 7 |
| Passed scenarios | 7 |
| Failed scenarios | 0 |
| Decision mismatches | 0 |
| Token overrun events | 0 |

All seven scenarios produced their expected recovery decisions and resolution outcomes.

## 5. Implementation Details

Timeout recovery is integrated into the failure-guided orchestration path.

The implementation:

- Detects model request timeouts.
- Detects test-runner timeouts.
- Detects command execution timeouts.
- Records timeout events in tool-call logs.
- Stops the current action loop after a command timeout.
- Checks remaining token and time budgets before replanning.
- Stops when the repair-attempt limit is reached.
- Provides recovery guidance that discourages repeating the timed-out operation unchanged.

Command timeout classification uses structured timeout information on `WorkspaceError`, rather than relying on exception-message parsing.

## 6. Budget Enforcement

Replanning is conditional on the remaining task budget.

The agent checks:

- Minimum remaining tokens required for recovery.
- Minimum remaining execution time.
- Maximum permitted repair attempts.

When these conditions are not satisfied, the recovery decision is STOP rather than an additional model call.

## 7. Limitations

This evaluation has important limitations:

1. The model is scripted rather than a live language model.
2. Timeout outcomes are injected or mocked.
3. Seven scenarios provide limited coverage of possible execution conditions.
4. The evaluation does not measure real-world timeout frequency.
5. It does not establish that timeout replanning improves software-engineering task resolution rates.
6. It does not establish generalization to other models, hardware, or workloads.

Therefore, the 7/7 result demonstrates that the tested control-flow and budget decisions behaved as expected under the defined scenarios. It is not evidence of general timeout-recovery reliability.

## 8. Reproducibility

Run from the repository root:

```bash
python experiments/evaluate_timeout_recovery.py
```

The evaluation writes its JSONL evidence to:

```text
results/timeout-recovery.jsonl
```

The evaluation implementation is:

```text
experiments/evaluate_timeout_recovery.py
```

The production implementation is located in:

```text
src/agentflow/coding_agent.py
src/agentflow/workspace.py
```

## 9. Conclusion

The controlled evaluation verified all seven predefined timeout-recovery scenarios.

AgentFlow correctly selected REPLAN when recovery was permitted and STOP when the configured budget or repair constraints prevented recovery.

Further empirical evaluation with live model inference and independently selected software-engineering tasks is required before making broader performance or reliability claims.
