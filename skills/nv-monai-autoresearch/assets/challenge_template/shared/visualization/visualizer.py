# miagent-template-source: assets/challenge_template/shared/visualization/visualizer.py
# miagent-template-contract: active; keep the miagent-template-source loaded for creation/material updates; reload after resume or compaction
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stable shared-visualization API to implement for this challenge."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

Artifact = str | Path
Artifacts = Artifact | Sequence[Artifact] | Mapping[str, Artifact]

# Remove this marker after implementing and smoke-testing both entry points.
TEMPLATE_STUB = True


def visualize_case(
    inputs: Artifacts,
    prediction: Artifacts,
    output_dir: Artifact,
    *,
    ground_truth: Artifacts | None = None,
    case_id: str,
    intermediates: Mapping[str, Artifact] | None = None,
    metrics: Mapping[str, float] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Render one task-specific case for quick debugging."""
    raise NotImplementedError("Implement the challenge-specific case renderer")


def visualize_run(manifest: Artifact, output_html: Artifact) -> Path:
    """Render an indexed run viewer from a normalized JSON manifest."""
    raise NotImplementedError("Implement the challenge-specific run renderer")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render challenge results")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    visualize_run(args.manifest, args.output)


if __name__ == "__main__":
    main()
