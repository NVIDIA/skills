#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check whether an Isaac Lab training run produced its expected final artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def verify(
    run_dir: Path, console: Path, expected_step: int, iterations: int,
    video_length: int, min_video_clips: int = 1,
) -> dict:
    """Return artifact evidence and problems, including unreadable TensorBoard data."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    problems: list[str] = []
    tag = "Metrics/terminal_success_rate"
    scalars = []
    try:
        event_reader = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
        event_reader.Reload()
        scalars = event_reader.Scalars(tag) if tag in event_reader.Tags()["scalars"] else []
    except Exception as error:
        problems.append(f"cannot read TensorBoard scalars from {run_dir}: {error}")
    if len(scalars) != iterations:
        problems.append(f"{tag}: found {len(scalars)} samples, expected {iterations}")
    last_step = scalars[-1].step if scalars else None
    if last_step != expected_step:
        problems.append(f"last scalar step {last_step}, expected {expected_step}")

    checkpoint = run_dir / f"model_{expected_step}.pt"
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        problems.append(f"missing or empty final checkpoint: {checkpoint}")

    if not console.is_file() or "Training time:" not in console.read_text(errors="replace"):
        problems.append(f"final training-time line missing from {console}")

    videos = sorted((run_dir / "videos/train").glob("*.mp4")) if min_video_clips else []
    if len(videos) < min_video_clips:
        problems.append(f"found {len(videos)} training MP4s, expected at least {min_video_clips}")
    if videos:
        import imageio.v2 as imageio
    for video in videos:
        try:
            with imageio.get_reader(str(video), format="ffmpeg") as reader:
                if reader.get_data(video_length - 1).size == 0:
                    problems.append(f"empty final frame in {video}")
        except (IndexError, OSError, RuntimeError, ValueError) as error:
            problems.append(f"cannot decode frame {video_length - 1} in {video}: {error}")

    exports = sorted(
        path for path in run_dir.rglob("*")
        if path.is_file() and (path.suffix.lower() in {".onnx", ".jit", ".ts"}
                               or path.name.endswith(".onnx.data")
                               or (path.suffix.lower() == ".pt" and "exported" in path.relative_to(run_dir).parts))
    )

    return {
        "artifact_complete": not problems,
        "exit_code": "unavailable; this check cannot determine the trainer process exit code",
        "run_dir": str(run_dir),
        "last_step": last_step,
        "scalar_count": len(scalars),
        "checkpoint": str(checkpoint),
        "videos": [str(video) for video in videos],
        "exports": [str(path) for path in exports],
        "problems": problems,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--console", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--video-length", type=int, default=200)
    parser.add_argument("--min-video-clips", type=int, default=1)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    if args.iterations <= 0 or args.video_length <= 0:
        parser.error("--iterations and --video-length must be positive")
    if args.min_video_clips < 0:
        parser.error("--min-video-clips must be nonnegative")
    clips = 0 if args.no_video else args.min_video_clips
    result = verify(args.run_dir, args.console, args.expected_step, args.iterations, args.video_length, clips)
    print(json.dumps(result, indent=2))
    return 0 if result["artifact_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
