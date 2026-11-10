import json
from pathlib import Path

from experiments.evaluate import evaluate
from experiments.run_experiment import _optional_model_skips, _select_tasks, load_tasks


def test_controlled_benchmark_has_five_pilot_and_fifteen_held_out_tasks():
    tasks = load_tasks(Path("benchmarks/tasks.jsonl"))

    assert len(tasks) == 20
    assert len(_select_tasks(tasks, "pilot")) == 5
    assert len(_select_tasks(tasks, "held-out")) == 15


def test_optional_e4b_requires_installed_model_and_measured_vram():
    candidates = [{"name": "gemma4:e4b-q4", "minimum_free_vram_gb": 4.5}]
    hardware = {
        "gpus": [{"name": "RTX 3050", "free_vram_mib": 6144, "total_vram_mib": 6144}]
    }
    installed = {
        "gemma4:e4b-q4": {
            "parameter_size": "4B",
            "quantization_level": "Q4_K_M",
            "family": "gemma4",
        }
    }

    runnable, skipped = _optional_model_skips(candidates, installed, hardware, "auto")
    assert len(runnable) == 1
    assert skipped == []

    runnable, skipped = _optional_model_skips(
        candidates,
        installed,
        {"gpus": [{"free_vram_mib": 1024}]},
        "auto",
    )
    assert runnable == []
    assert "below configured threshold" in skipped[0]["reason"]

    runnable, skipped = _optional_model_skips(
        candidates,
        {"gemma4:e4b-q4": {"parameter_size": "4B", "family": "gemma4"}},
        hardware,
        "auto",
    )
    assert runnable == []
    assert "quantized model" in skipped[0]["reason"]


def test_evaluator_reports_resolution_attempts_latency_and_tokens(tmp_path):
    results_file = tmp_path / "results.jsonl"
    rows = [
        {"record_type": "run_metadata", "split": "pilot"},
        {
            "record_type": "result",
            "task_id": "t1",
            "model": "local-model",
            "strategy": "failure-guided",
            "resolved": True,
            "repair_attempts": 1,
            "attempts": [{"test": {"passed": False}}, {"test": {"passed": True}}],
            "budget": {
                "model_calls": 3,
                "tool_calls": 1,
                "test_runs": 2,
                "elapsed_seconds": 4.0,
                "tokens_used": 100,
            },
        },
    ]
    results_file.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    report = evaluate(results_file)

    assert report["metadata"][0]["split"] == "pilot"
    summary = report["runs"][0]
    assert summary["resolution_rate"] == 1.0
    assert summary["test_pass_rate"] == 1.0
    assert summary["mean_repair_attempts"] == 1.0
    assert summary["mean_tokens_used"] == 100
