"""Run every task in the JSONL benchmark and save structured results."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from agentflow.controller import AgentController
from agentflow.model import CopilotModel
from agentflow.schemas import TaskRequest


async def run(tasks_path: Path, output_path: Path, model_name: str) -> None:
    model = CopilotModel(model_name)
    await model.start()
    try:
        controller = AgentController(model)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tasks_path.open(encoding="utf-8") as tasks_file, output_path.open(
            "w", encoding="utf-8"
        ) as output_file:
            for line_number, line in enumerate(tasks_file, start=1):
                if not line.strip():
                    continue
                task = json.loads(line)
                goal = task.get("goal")
                if not isinstance(goal, str) or not goal.strip():
                    raise ValueError(f"Benchmark line {line_number} has no non-empty goal.")
                result = await controller.run(TaskRequest(goal))
                output_file.write(json.dumps({"task_id": task.get("id"), **result.to_dict()}, default=str))
                output_file.write("\n")
    finally:
        await model.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, default=Path("benchmarks/tasks.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/latest.jsonl"))
    parser.add_argument("--model", default="gpt-4.1")
    args = parser.parse_args()
    asyncio.run(run(args.tasks, args.output, args.model))


if __name__ == "__main__":
    main()
