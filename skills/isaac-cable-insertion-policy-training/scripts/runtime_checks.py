#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Probe standalone training installation, video dependencies, and TensorBoard.

Usage: From the isaaclab_training root, run `.venv/bin/python /path/to/runtime_checks.py
preflight --video`, or use the `progress` subcommand for a running attempt.
The `installation --root PATH` probe checks the package venv and locked Isaac Lab wheel.
The `dependencies --root PATH` probe audits installed requirements and project overrides.
Arguments: preflight accepts --video; progress requires --run-dir and
--started-epoch, with an optional --start-iteration (default 0).
Output: JSON dependency results or latest TensorBoard evidence on stdout.
Exit codes: 0 on success, 2 on an argument error or failed check (JSON error).
These bounded, CPU-only checks do not import Isaac Sim or load checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import re
import sys
import tempfile
import tomllib
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit


SUPPORTED_LEAPP_VERSION = "0.6.1"


def installation(root: Path) -> dict:
    """Check installed provenance without importing Isaac Sim or needing a Lab checkout."""
    root = root.resolve()
    venv = root / ".venv"
    if Path(sys.prefix).resolve() != venv.resolve():
        raise RuntimeError("run installation checks with training_root/.venv/bin/python")
    project = tomllib.loads((root / "pyproject.toml").read_text())
    if project.get("project", {}).get("name") != "isaaclab-training":
        raise ValueError("training_root must contain the standalone isaaclab-training pyproject.toml")
    lock_path = root / "uv.lock"
    lock = tomllib.loads(lock_path.read_text())
    pin = project.get("tool", {}).get("uv", {}).get("sources", {}).get("isaaclab", {})
    rev = pin.get("rev", "")
    if (pin.get("git") != "https://github.com/isaac-sim/IsaacLab.git"
            or pin.get("subdirectory") != "tools/wheel_builder"
            or not re.fullmatch(r"[0-9a-f]{40}", rev)):
        raise ValueError("standalone project must pin the official Isaac Lab wheel-builder Git commit")

    def locked(name: str) -> dict:
        matches = [item for item in lock.get("package", []) if item["name"] == name]
        if len(matches) != 1:
            raise ValueError(f"uv.lock must contain exactly one {name} package")
        return matches[0]

    lab_lock = locked("isaaclab")
    git_source = urlsplit(lab_lock.get("source", {}).get("git", ""))
    query = parse_qs(git_source.query)
    if (git_source.fragment != rev or query.get("rev") != [rev]
            or query.get("subdirectory") != [pin["subdirectory"]]
            or git_source._replace(query="", fragment="").geturl() != pin["git"]):
        raise ValueError("Isaac Lab pin and uv.lock disagree; restore the release's matching files")

    lab = importlib.metadata.distribution("isaaclab")
    direct = json.loads(lab.read_text("direct_url.json") or "{}")
    if (direct.get("url") != pin["git"] or direct.get("subdirectory") != pin["subdirectory"]
            or direct.get("vcs_info", {}).get("vcs") != "git"
            or direct.get("vcs_info", {}).get("commit_id") != rev
            or direct.get("dir_info", {}).get("editable")
            or lab.version != lab_lock["version"]):
        raise RuntimeError("installed Isaac Lab does not match the locked wheel-builder dependency; "
                           "sync the standalone project, not a source IsaacLab environment")
    lab_spec = importlib.util.find_spec("isaaclab")
    if not lab_spec or not lab_spec.origin or not Path(lab_spec.origin).resolve().is_relative_to(venv.resolve()):
        raise RuntimeError("Isaac Lab import resolves outside the standalone .venv")

    training = importlib.metadata.distribution("isaaclab-training")
    training_lock = locked("isaaclab-training")
    if (training_lock.get("source") != {"editable": "."}
            or training.version != training_lock["version"]):
        raise RuntimeError("IsaacLabTraining installation does not match the editable project in uv.lock")
    editable = json.loads(training.read_text("direct_url.json") or "{}")
    training_url = urlsplit(editable.get("url", ""))
    if (not editable.get("dir_info", {}).get("editable") or training_url.scheme != "file"
            or Path(unquote(training_url.path)).resolve() != root):
        raise RuntimeError("IsaacLabTraining must be editable from training_root")
    training_spec = importlib.util.find_spec("isaaclab_training")
    source = root / "src/isaaclab_training/__init__.py"
    if not training_spec or not training_spec.origin or Path(training_spec.origin).resolve() != source.resolve():
        raise RuntimeError("IsaacLabTraining import does not resolve to training_root/src")
    sim_version = importlib.metadata.version("isaacsim")
    if sim_version != locked("isaacsim")["version"]:
        raise RuntimeError("installed Isaac Sim does not match uv.lock; sync with --group sim")
    return {"training_root": str(root), "isaaclab_training_source": str(source),
            "isaaclab_version": lab.version, "isaaclab_commit": rev,
            "isaaclab_source": pin["git"], "isaac_sim_version": sim_version,
            "lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest()}


def preflight(video: bool) -> dict:
    importlib.import_module("tensorboard.backend.event_processing.event_accumulator")

    result = {"tensorboard": True, "video": video}
    if video:
        import imageio.v2 as imageio
        import numpy as np
        from moviepy.editor import ImageSequenceClip

        # Exercise the same encoder used by IsaacLab's recorder and the decoder
        # used by verify_run. Importing the packages alone does not check FFmpeg.
        with tempfile.TemporaryDirectory(prefix="isaac-video-check-") as directory:
            path = str(Path(directory) / "probe.mp4")
            clip = ImageSequenceClip([np.zeros((16, 16, 3), dtype=np.uint8)] * 2, fps=2)
            try:
                clip.write_videofile(path, codec="libx264", audio=False, logger=None)
            finally:
                clip.close()
            with imageio.get_reader(path, format="ffmpeg") as reader:
                if not reader.get_data(1).size:
                    raise RuntimeError("video encoder produced an empty frame")
        result["encoder"] = "libx264"
    return result


def dependencies(root: Path) -> dict:
    """Audit installed base requirements; permit only explicit, satisfied project overrides."""
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    root = root.resolve()
    if Path(sys.prefix).resolve() != (root / ".venv").resolve():
        raise RuntimeError("run dependency checks with training_root/.venv/bin/python")
    project = tomllib.loads((root / "pyproject.toml").read_text())
    lock = tomllib.loads((root / "uv.lock").read_text())
    overrides = {}
    for value in project.get("tool", {}).get("uv", {}).get("override-dependencies", []):
        requirement = Requirement(value)
        if not requirement.marker or requirement.marker.evaluate({"extra": ""}):
            overrides[canonicalize_name(requirement.name)] = requirement
    installed = {canonicalize_name(dist.metadata["Name"]): dist
                 for dist in importlib.metadata.distributions() if dist.metadata.get("Name")}
    problems = []
    accepted = []
    providers = []
    if "onnxruntime" in installed and "onnxruntime-gpu" in installed:
        problems.append("onnxruntime and onnxruntime-gpu both provide the onnxruntime import; "
                        "resolve the release's provider choice before training")
    # Optional groups may be absent, but any installed locked package must retain its pin.
    locked_versions = {}
    for package in lock.get("package", []):
        name = canonicalize_name(package["name"])
        locked_versions.setdefault(name, set()).add(package["version"])
    for name, versions in locked_versions.items():
        if name in installed and installed[name].version not in versions:
            problems.append(f"{name}=={installed[name].version} differs from uv.lock {sorted(versions)}")
    for name, dist in sorted(installed.items()):
        for value in dist.requires or []:
            requirement = Requirement(value)
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            target = canonicalize_name(requirement.name)
            found = installed.get(target)
            if found is None:
                # The supported LEAPP release uses the same onnxruntime import
                # from the GPU distribution. Keep this substitution version-bound;
                # never install both providers or waive another missing package.
                gpu = installed.get("onnxruntime-gpu")
                if (name == "leapp" and dist.version == SUPPORTED_LEAPP_VERSION
                        and locked_versions.get("leapp") == {SUPPORTED_LEAPP_VERSION}
                        and target == "onnxruntime" and not requirement.url
                        and not requirement.extras and gpu is not None
                        and gpu.version == "1.26.0"
                        and requirement.specifier.contains(gpu.version, prereleases=True)):
                    providers.append({"package": name, "version": dist.version,
                                      "requirement": str(requirement),
                                      "provider": "onnxruntime-gpu", "installed": gpu.version,
                                      "evidence": "https://nvidia-isaac.github.io/leapp/leapp_runtime.html"})
                    continue
                problems.append(f"{name} requires {requirement}; {target} is missing")
                continue
            if requirement.url:
                # Direct-source provenance is checked separately by installation().
                continue
            if requirement.specifier.contains(found.version, prereleases=True):
                continue
            override = overrides.get(target)
            if (override and target in locked_versions and not override.url and override.specifier
                    and override.specifier.contains(found.version, prereleases=True)):
                accepted.append({"package": name, "requirement": str(requirement),
                                 "installed": found.version, "override": str(override)})
            else:
                problems.append(f"{name} requires {requirement}; installed {target}=={found.version}")
    if problems:
        raise RuntimeError("installed dependency audit failed: " + "; ".join(problems))
    result = {"checked_distributions": len(installed), "project_override_exceptions": accepted}
    if providers:
        result["provider_substitutions"] = providers
    return result


def tensorboard_progress(run_dir: Path, started_epoch: float, start_iteration: int) -> dict:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    # Keep only the most recent scalar per tag. Reload tolerates an incomplete
    # trailing event; the caller also bounds this process's running time.
    reader = EventAccumulator(str(run_dir), size_guidance={"scalars": 1})
    reader.Reload()
    evidence = []
    metrics = {}
    for tag in reader.Tags().get("scalars", []):
        # Exclude time-indexed reward tags and unrelated summaries.
        if not (tag.startswith(("Metrics/", "Loss/")) or tag == "Perf/total_fps"):
            continue
        for event in reader.Scalars(tag):
            if event.step >= start_iteration and event.wall_time >= started_epoch:
                evidence.append((event.step, event.wall_time))
                if tag.startswith("Metrics/") and math.isfinite(event.value):
                    metrics[tag] = event.value
    if not evidence:
        return {}
    step, epoch = max(evidence)
    return {"step": step, "epoch": epoch, "source": "tensorboard", "metrics": metrics}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("preflight")
    check.add_argument("--video", action="store_true")
    installed = commands.add_parser("installation")
    installed.add_argument("--root", type=Path, required=True)
    dependency_check = commands.add_parser("dependencies")
    dependency_check.add_argument("--root", type=Path, required=True)
    progress = commands.add_parser("progress")
    progress.add_argument("--run-dir", type=Path, required=True)
    progress.add_argument("--started-epoch", type=float, required=True)
    progress.add_argument("--start-iteration", type=int, default=0)
    args = parser.parse_args()
    try:
        if args.command == "installation":
            result = installation(args.root)
        elif args.command == "dependencies":
            result = dependencies(args.root)
        elif args.command == "preflight":
            result = preflight(args.video)
        else:
            result = tensorboard_progress(args.run_dir, args.started_epoch, args.start_iteration)
        print(json.dumps(result, allow_nan=False))
        return 0
    except Exception as error:
        print(json.dumps({"error": f"{type(error).__name__}: {error}"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
