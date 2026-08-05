# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concise examples for the Evaluator plugin SDK surfaces.

These functions are intentionally not called at import time. Copy the one that
matches the feature being used and supply a configured NeMo Platform client.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def capital_france_metric() -> Any:
    """Return an output-only metric suitable for stored agent-evaluation tasks."""
    from nemo_evaluator_sdk import StringCheckMetric

    return StringCheckMetric(
        operation="equals",
        left_template="{{sample.output_text | trim}}",
        right_template="Paris",
    )


def evaluate_standalone() -> Any:
    """Evaluate one deterministic metric in process."""
    from nemo_evaluator_sdk import Evaluator, ExactMatchMetric

    return Evaluator().run_sync(
        metrics=ExactMatchMetric(
            reference="{{item.expected}}",
            candidate="{{item.output}}",
        ),
        dataset=[
            {"expected": "Paris", "output": "Paris"},
            {"expected": "Paris", "output": "London"},
        ],
    )


def submit_and_collect(client: Any, output_dir: Path) -> tuple[Any, Path]:
    """Submit one metric, wait for completion, and retrieve its artifacts."""
    from nemo_evaluator_sdk import ExactMatchMetric

    job = client.evaluator.submit(
        metric=ExactMatchMetric(
            reference="{{item.expected}}",
            candidate="{{item.output}}",
        ),
        dataset=[{"expected": "Paris", "output": "Paris"}],
    )
    job.wait_until_done()
    return job.get_result(), job.download_artifacts(output_dir)


def store_resources(client: Any) -> None:
    """Store one metric, task, and taskset."""
    from nemo_evaluator.api.schemas import (
        MetricRef,
        TaskInput,
        TaskInputs,
        TaskRef,
        TasksetInput,
    )

    client.evaluator.metrics.create(
        "answer-exact",
        metric=capital_france_metric(),
    )
    client.evaluator.tasks.create(
        "capital-france",
        task=TaskInput(
            intent="Name the capital of France.",
            inputs=TaskInputs(instruction="What is the capital of France?"),
            metrics=[MetricRef("default/answer-exact")],
        ),
    )
    client.evaluator.tasksets.create(
        "geography",
        taskset=TasksetInput(tasks=[TaskRef("default/capital-france")]),
    )


def build_agent_eval_spec(metric_bundle: Any) -> Any:
    """Build a durable task evaluation with a runner target."""
    from nemo_evaluator.api.schemas import TaskInputs
    from nemo_evaluator.jobs.agent_spec import (
        AgentEvalInputSpec,
        AgentEvalTaskInput,
        CodexRunnerTarget,
    )

    return AgentEvalInputSpec(
        tasks=[
            AgentEvalTaskInput(
                id="capital-france",
                intent="Name the capital of France.",
                inputs=TaskInputs(instruction="What is the capital of France?"),
                metrics=[metric_bundle],
            )
        ],
        target=CodexRunnerTarget(),
        max_concurrent_tasks=2,
        labels={"benchmark": "geography-smoke"},
    )
