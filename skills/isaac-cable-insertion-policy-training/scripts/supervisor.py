#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable, restartable DisplayPort training campaign supervisor.

Run this command from the standalone isaaclab_training package directory. The campaign
plan and all mutable state live in a unique skill-local logs directory.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import time
import traceback
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from reporting import final_iteration, write_report as write_campaign_report
from launch_environment import build_environment, normalize_environment
from storage import (SCHEMA_VERSION, append_record, atomic_json,
                     cleanup_empty_logs, completed_result, evaluation_provenance, fingerprint, local_path, read_session,
                     training_fingerprint,
                     run_folder, save_state, slug, timestamp)


SKILL = Path(__file__).resolve().parent.parent
LOGS = SKILL / "logs"
DEFAULT_TASK = "IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace-ROS-Inference"
EXPERIMENT = "displayport_insertion_rizon4s"
RUN_ID = re.compile(r"^[a-z][a-z0-9_-]{0,47}$")
ITERATION = re.compile(r"Learning iteration\s+(\d+)/(\d+)")
METRIC = re.compile(r"^\s*([A-Za-z_]+/[A-Za-z_]+):\s*(-?[\d.eE+]+)", re.MULTILINE)
WATCH_METRICS = {
    "Metrics/terminal_success_rate", "Metrics/success_rate",
    "Metrics/plug_socket_pos_error_m", "Metrics/plug_socket_keypoint_dist_m",
    "Episode_Termination/time_out", "Episode_Termination/plug_dropped",
    "Episode_Termination/plug_orientation_exceeded",
    "Episode_Reward/plug_socket_keypoint_tracking",
    "Episode_Reward/plug_socket_keypoint_tracking_exp",
}
GUIDE = "https://nvidia-isaac-ros.github.io/manipulation/training/tasks/displayport_insertion.html#step-2-full-scale-training"
EULA_URL = "https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html"
EULA_SCOPE = "isaac-sim-6.1"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def eula_record_path() -> Path:
    """Keep acceptance local to the OS user and outside Git-tracked skill files."""
    return LOGS / f".eula-acceptance-uid{os.getuid()}.json"


def eula_record() -> dict | None:
    path = eula_record_path()
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise RuntimeError(f"EULA acceptance record has unsafe ownership or permissions: {path}")
    try:
        value = read_json(path)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"EULA acceptance record is unreadable: {path}") from error
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        return None
    if (value.get("accepted") is True and value.get("uid") == os.getuid()
            and value.get("agreement_url") == EULA_URL and value.get("scope") == EULA_SCOPE
            and isinstance(value.get("accepted_at_utc"), str)
            and value.get("source") in {"interactive_prompt", "explicit_user_reply"}):
        return value
    return None


def record_eula_acceptance(source: str) -> dict:
    if source not in {"interactive_prompt", "explicit_user_reply"}:
        raise ValueError("EULA acceptance source must identify an explicit user confirmation")
    LOGS.mkdir(exist_ok=True)
    value = {
        "schema_version": 1, "accepted": True, "accepted_at_utc": utc(),
        "agreement_url": EULA_URL, "scope": EULA_SCOPE, "uid": os.getuid(), "source": source,
    }
    path = eula_record_path()
    temp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
    return value


def require_eula_record() -> dict:
    record = eula_record()
    if record is None:
        raise RuntimeError(
            f"Omniverse EULA acceptance is required before training. Read {EULA_URL}; "
            "after the user explicitly accepts, run eula-accept --user-confirmed. "
            "A plan's eula_accepted field alone is not proof of acceptance."
        )
    return record


def safe_remote(value: str) -> str:
    """Keep a useful Git remote identity without logging URL credentials."""
    if "://" not in value:
        authority, separator, path = value.partition(":")
        if not separator or "/" in authority:
            return value
        host = authority.rsplit("@", 1)[-1]
        value = urlunsplit(("ssh", host, "/" + path, "", ""))
    parsed = urlsplit(value)
    host = parsed.hostname or ""
    if parsed.port:
        host += f":{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path, "", ""))


def emit(session: Path, state: dict, kind: str, **details: Any) -> None:
    record = {"utc": utc(), "kind": kind, **details}
    append_record(session / "events.jsonl", record)
    state["updated_utc"] = record["utc"]
    save_state(session, state)
    write_report(session, state)


def positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def optional_seconds(value: Any, name: str) -> int | None:
    return None if value is None else positive_int(value, name)


def json_value(value: Any) -> bool:
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, list):
        return all(json_value(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and json_value(item) for key, item in value.items())
    return False


def normalize_run(raw: dict, *, used: set[str] | None = None, task: str = DEFAULT_TASK) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("each run must be a JSON object")
    allowed = {
        "id", "hypothesis", "num_envs", "seed", "max_iterations", "num_steps_per_env",
        "video", "video_length", "video_interval", "video_schedule", "overrides", "purpose", "parent_checkpoint",
        "retry_of", "parent_total_updates", "parent_training_seconds",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown run keys: {sorted(unknown)}")
    run_id = raw.get("id")
    if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id):
        raise ValueError("run id must match ^[a-z][a-z0-9_-]{0,47}$")
    if used is not None and run_id in used:
        raise ValueError(f"duplicate run id: {run_id}")
    task_space = "TaskSpace" in task
    rollout = positive_int(raw.get("num_steps_per_env", 128 if task_space else 512), "num_steps_per_env")
    iterations = positive_int(raw.get("max_iterations", 1000 if task_space else 1500), "max_iterations")
    purpose = raw.get("purpose", "training")
    if purpose not in {"training", "smoke"}:
        raise ValueError("purpose must be training or smoke")
    video = raw.get("video", True)
    if not isinstance(video, bool):
        raise ValueError("video must be a boolean")
    schedule = raw.get("video_schedule", "interval")
    if schedule not in {"interval", "start_end"}:
        raise ValueError("video_schedule must be interval or start_end")
    video_length = positive_int(raw.get("video_length", 200), "video_length")
    total_steps = iterations * rollout
    if schedule == "start_end":
        if not video or "video_interval" in raw or total_steps < video_length * 2:
            raise ValueError("start_end video needs two complete clips and no explicit video_interval")
        video_interval = total_steps - video_length
    else:
        video_interval = positive_int(raw.get("video_interval", rollout * 150), "video_interval")
    overrides = raw.get("overrides", {})
    if not isinstance(overrides, dict) or any(
        not isinstance(key, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", key)
        or not json_value(value)
        for key, value in overrides.items()
    ):
        raise ValueError("overrides must map config keys to JSON values")
    overrides = dict(overrides)
    if {"agent.experiment_name", "agent.run_name"}.intersection(overrides):
        raise ValueError("the supervisor owns agent.experiment_name and agent.run_name; omit these overrides")
    if "agent.num_steps_per_env" in overrides or "agent.max_iterations" in overrides:
        raise ValueError("set rollout and iteration values through named run fields")
    divisor_key = "env.events.reset_plug_curriculum.params.num_steps_per_env"
    if divisor_key in overrides and int(overrides[divisor_key]) != rollout:
        raise ValueError("curriculum num_steps_per_env must match agent.num_steps_per_env")
    overrides[divisor_key] = rollout
    # Record and launch the same anneal checked here, including the release default.
    anneal = overrides.setdefault("env.events.reset_plug_curriculum.params.anneal_end_iter", 333 if task_space else 500)
    if purpose == "training" and iterations < int(anneal):
        raise ValueError("training stops before curriculum anneal_end_iter; set a shorter anneal or purpose=smoke")
    parent = raw.get("parent_checkpoint")
    if parent is not None and (not isinstance(parent, str) or not Path(parent).is_absolute()):
        raise ValueError("parent_checkpoint must be an absolute path")
    for field in ("parent_total_updates", "parent_training_seconds"):
        value = raw.get(field)
        if value is not None and (not parent or isinstance(value, bool)
                                  or not isinstance(value, (int, float)) or value < 0):
            raise ValueError(f"{field} requires a parent checkpoint and a nonnegative number")
    if raw.get("parent_total_updates") is not None and not isinstance(raw["parent_total_updates"], int):
        raise ValueError("parent_total_updates must be an integer")
    if raw.get("retry_of") is not None and not isinstance(raw["retry_of"], str):
        raise ValueError("retry_of must be a run id")
    num_envs = positive_int(raw.get("num_envs", 256), "num_envs")
    batches = positive_int(overrides.get("agent.algorithm.num_mini_batches", 16), "num_mini_batches")
    if num_envs < batches or num_envs % batches:
        raise ValueError("recurrent PPO num_envs must be a positive multiple of num_mini_batches (default 16)")
    return {
        "id": run_id,
        "hypothesis": str(raw.get("hypothesis", "")),
        "purpose": purpose,
        "num_envs": num_envs,
        "seed": int(raw.get("seed", 42)),
        "max_iterations": iterations,
        "num_steps_per_env": rollout,
        "video": video,
        "video_length": video_length,
        "video_interval": video_interval,
        "video_schedule": schedule,
        "overrides": overrides,
        "parent_checkpoint": parent,
        "retry_of": raw.get("retry_of"),
        "parent_total_updates": raw.get("parent_total_updates"),
        "parent_training_seconds": raw.get("parent_training_seconds"),
    }


def normalize_plan(raw: dict) -> dict:
    """Validate a session plan and fill in training, evaluation, and budget defaults."""
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("plan schema_version must be 1")
    if "isaaclab_root" in raw:
        raise ValueError("source-IsaacLab plans are no longer supported; use training_root pointing to "
                         "isaac_ros_manipulation/isaaclab_training and create a new session")
    allowed = {
        "schema_version", "training_root", "task", "output_root", "max_completed_runs",
        "training_budget_seconds", "evaluation_budget_seconds", "decision_mode", "runs",
        "evaluation", "timeouts", "eula_accepted", "name", "objective", "environment",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown plan keys: {sorted(unknown)}")
    root = Path(raw.get("training_root", ""))
    if not root.is_absolute() or not root.is_dir():
        raise ValueError("training_root must be an existing absolute directory")
    default_output = root / "logs/rsl_rl" / EXPERIMENT
    output = Path(raw.get("output_root", default_output))
    if not output.is_absolute():
        raise ValueError("output_root must be absolute")
    if output.resolve() != default_output.resolve():
        raise ValueError(f"custom output_root is unsupported; use {default_output.resolve()}")
    mode = raw.get("decision_mode", "manual")
    if mode not in {"manual", "preplanned"}:
        raise ValueError("decision_mode must be manual or preplanned")
    max_runs = positive_int(raw.get("max_completed_runs", 3), "max_completed_runs")
    runs = raw.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("runs must contain at least one run")
    used: set[str] = set()
    normalized_runs = []
    for item in runs:
        run = normalize_run(item, used=used, task=str(raw.get("task", DEFAULT_TASK)))
        used.add(run["id"])
        normalized_runs.append(run)
    if len(normalized_runs) > max_runs:
        raise ValueError("initial runs exceed max_completed_runs")
    evaluation = raw.get("evaluation", {"enabled": False})
    if not isinstance(evaluation, dict):
        raise ValueError("evaluation must be an object")
    enabled = evaluation.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("evaluation.enabled must be a boolean")
    seeds = evaluation.get("seeds", [])
    if not isinstance(seeds, list) or any(not isinstance(seed, int) for seed in seeds):
        raise ValueError("evaluation.seeds must be a list of integers")
    if enabled and not seeds:
        raise ValueError("evaluation.seeds is required when evaluation is enabled")
    if len(set(seeds)) != len(seeds):
        raise ValueError("evaluation.seeds must be unique")
    group = evaluation.get("group", "tuning")
    if group not in {"tuning", "screening", "independent"}:
        raise ValueError("evaluation.group must be tuning, screening, or independent")
    checkpoint_steps = evaluation.get("checkpoint_steps", ["final"])
    if not isinstance(checkpoint_steps, list) or not checkpoint_steps or any(
        step != "final" and (not isinstance(step, int) or step < 0) for step in checkpoint_steps
    ):
        raise ValueError("evaluation.checkpoint_steps must contain update numbers or 'final'")
    timeouts = raw.get("timeouts", {})
    if not isinstance(timeouts, dict):
        raise ValueError("timeouts must be an object")
    accepted = raw.get("eula_accepted", False)
    if not isinstance(accepted, bool):
        raise ValueError("eula_accepted must be a boolean")
    return {
        "schema_version": 1,
        "name": slug(str(raw.get("name", "isaac-cable-insertion-policy-training"))),
        "objective": str(raw.get("objective", "Compare fixed-distribution policy success within the agreed budget.")),
        "training_root": str(root.resolve()),
        "environment": normalize_environment(raw.get("environment")),
        "task": str(raw.get("task", DEFAULT_TASK)),
        "output_root": str(output.resolve()),
        "max_completed_runs": max_runs,
        "training_budget_seconds": optional_seconds(raw.get("training_budget_seconds"), "training_budget_seconds"),
        "evaluation_budget_seconds": optional_seconds(raw.get("evaluation_budget_seconds"), "evaluation_budget_seconds"),
        "decision_mode": mode,
        "runs": normalized_runs,
        "evaluation": {
            "enabled": enabled, "seeds": seeds, "group": group,
            "num_envs": positive_int(evaluation.get("num_envs", 256), "evaluation.num_envs"),
            "max_steps": positive_int(evaluation.get("max_steps", 260), "evaluation.max_steps"),
            "checkpoint_steps": checkpoint_steps,
        },
        "timeouts": {
            "startup_seconds": positive_int(timeouts.get("startup_seconds", 1800), "startup_seconds"),
            "stall_seconds": positive_int(timeouts.get("stall_seconds", 1800), "stall_seconds"),
            "grace_seconds": positive_int(timeouts.get("grace_seconds", 90), "grace_seconds"),
        },
        "eula_accepted": accepted,
    }


def session_state(session: Path) -> tuple[dict, dict]:
    return read_session(session)


def require_standalone_session(plan: dict) -> None:
    if "training_root" not in plan:
        raise RuntimeError("source-IsaacLab sessions are read-only; start a new standalone session "
                           "with training_root; historical checkpoints may be supplied explicitly")


def create_session(plan_path: Path) -> Path:
    plan = normalize_plan(read_json(plan_path))
    if Path.cwd().resolve() != Path(plan["training_root"]):
        raise RuntimeError(f"run this command from {plan['training_root']}")
    confirmation = require_eula_record()
    plan["eula_accepted"] = True
    plan["eula_confirmation"] = confirmation
    LOGS.mkdir(exist_ok=True)
    created = utc()
    base = timestamp(created) + "__" + plan["name"]
    name = base
    session = LOGS / name
    suffix = 1
    while True:
        try:
            session.mkdir()
            break
        except FileExistsError:
            suffix += 1
            name = f"{base}__{suffix:02d}"
            session = LOGS / name
    hashes = {}
    for script in ("supervisor.py", "reporting.py", "storage.py", "checkpoint.py", "evaluate.py", "verify_run.py", "runtime_checks.py", "launch_environment.py"):
        source = SKILL / "scripts" / script
        hashes[script] = fingerprint(source)["sha256"]
    state = {
        "session_id": name, "status": "created", "created_utc": created, "updated_utc": created,
        "queue": plan["runs"], "attempts": [], "completed_runs": 0,
        "current": None, "training_seconds": 0.0, "evaluation_seconds": 0.0,
        "launcher": None, "supervisor_pid": None,
        "code_sha256": hashes,
    }
    atomic_json(session / "session.json", {"schema_version": SCHEMA_VERSION,
                                           "plan": plan, "state": state})
    (session / "events.jsonl").touch()
    emit(session, state, "session_created", message=f"Plan copied from {plan_path}")
    emit(session, state, "eula_acceptance_reused", message=confirmation["accepted_at_utc"])
    return session


def ensure_session_code(state: dict) -> None:
    for name, expected in state["code_sha256"].items():
        if fingerprint(SKILL / "scripts" / name)["sha256"] != expected:
            raise RuntimeError("session code changed; restore the original skill version or start a new session "
                               "rather than silently changing an experiment")


def process_birth(pid: int) -> str | None:
    try:
        data = Path(f"/proc/{pid}/stat").read_text()
        fields = data.rsplit(") ", 1)[1].split()
        return fields[19] if fields[0] not in {"Z", "X"} else None
    except (FileNotFoundError, PermissionError, IndexError, ValueError):
        return None


def process_alive(pid: int | None, birth: str | None) -> bool:
    return bool(pid and birth and process_birth(pid) == birth)


def tail(path: Path, length: int = 262144) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - length))
        return stream.read().decode(errors="replace")


def progress(console: Path) -> tuple[int | None, dict[str, float]]:
    text = tail(console)
    iterations = list(ITERATION.finditer(text))
    step = int(iterations[-1].group(1)) if iterations else None
    metrics: dict[str, float] = {}
    for key, number in METRIC.findall(text):
        if key in WATCH_METRICS:
            metrics[key] = float(number)
    return step, metrics


def reported_eta(console: Path) -> int | None:
    values = re.findall(r"\bETA:\s*(\d+):(\d+):(\d+)", tail(console))
    if not values:
        return None
    hours, minutes, seconds = map(int, values[-1])
    return hours * 3600 + minutes * 60 + seconds


def run_command(run: dict, plan: dict, session: Path) -> tuple[list[str], dict[str, str]]:
    command = [
        selected_uv(plan), "run", "--no-sync", "isaaclab", "train", "--rl_library", "rsl_rl",
        "--task", plan["task"], "--num_envs", str(run["num_envs"]),
        "--seed", str(run["seed"]), "--max_iterations", str(run["max_iterations"]),
        "--run_name", run["run_name"],
        "--experiment_name", EXPERIMENT,
    ]
    if run["video"]:
        command.extend(["--viz", "kit", "--video", "--video_length", str(run["video_length"]),
                        "--video_interval", str(run["video_interval"])])
    command.append(f"agent.num_steps_per_env={run['num_steps_per_env']}")
    for key, value in sorted(run["overrides"].items()):
        scalar = json.dumps(value, separators=(",", ":")) if isinstance(value, (list, dict, bool)) or value is None else str(value)
        command.append(f"{key}={scalar}")
    if run["parent_checkpoint"]:
        command.extend(["--checkpoint", run["parent_checkpoint"]])
    return command, training_environment(plan)


def parent_iteration(run: dict, plan: dict) -> int:
    """Read the RSL-RL resume index without executing checkpoint pickle objects."""
    checkpoint = run["parent_checkpoint"]
    if checkpoint is None:
        return 0
    path = Path(checkpoint)
    if not path.is_file():
        raise RuntimeError(f"parent checkpoint is missing: {path}")
    command = [
        str(Path(plan["training_root"]) / ".venv/bin/python"),
        str(Path(__file__).resolve().parent / "checkpoint.py"),
        str(path),
    ]
    result = subprocess.run(command, cwd=plan["training_root"], capture_output=True,
                            text=True, check=False, timeout=120, env=training_environment(plan))
    if result.returncode != 0:
        raise RuntimeError(f"cannot safely read parent checkpoint iteration: {path}: {result.stderr}")
    try:
        step = int(result.stdout)
    except ValueError as error:
        raise RuntimeError(f"parent checkpoint has no integer iteration: {path}") from error
    if step < 0:
        raise RuntimeError(f"parent checkpoint iteration is negative: {path}")
    return step


def selected_uv(plan: dict) -> str:
    configuration = plan.get("environment")
    if configuration is None:
        configuration = normalize_environment(None)
    return configuration["uv_executable"]


def training_environment(plan: dict) -> dict[str, str]:
    configuration = plan.get("environment")
    if configuration is None:
        configuration = normalize_environment(None)
    return build_environment(Path(plan["training_root"]), configuration,
                             eula_accepted=plan.get("eula_accepted", False))


def runtime_check(plan: dict, arguments: list[str], timeout: int = 15) -> dict:
    """Keep optional package imports in the training venv, outside the supervisor."""
    command = [str(Path(plan["training_root"]) / ".venv/bin/python"),
               str(SKILL / "scripts/runtime_checks.py"), *arguments]
    result = subprocess.run(command, cwd=plan["training_root"], env=training_environment(plan),
                            capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError((result.stdout or result.stderr).strip()[-2000:])
    value = json.loads(result.stdout)
    if not isinstance(value, dict):
        raise ValueError("runtime check did not return a JSON object")
    return value


def check_preflight(plan: dict) -> dict:
    """Verify the standalone package, locked Isaac Lab wheel, and training runtime."""
    root = Path(plan["training_root"])
    if Path.cwd().resolve() != root:
        raise RuntimeError(f"run from the training package root: {root}")
    if not (root / ".venv/bin/python").is_file():
        raise RuntimeError("standalone training .venv is missing; run setup from isaaclab_training")
    if not shutil.which(selected_uv(plan)) or not shutil.which("nvidia-smi", path=training_environment(plan)["PATH"]):
        raise RuntimeError("uv and nvidia-smi are required")
    try:
        checks = runtime_check(plan, ["preflight"] +
                               (["--video"] if any(run["video"] for run in plan["runs"]) else []), timeout=60)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        raise RuntimeError(
            "Training dependency preflight failed before trainer launch: "
            f"{error}. Verify TensorBoard and, for video, recorder dependencies "
            "(moviepy<2, imageio and FFmpeg). Follow references/setup.md to repair "
            "the standalone training environment."
        ) from error
    installation = runtime_check(plan, ["installation", "--root", str(root)], timeout=60)
    dependency_audit = runtime_check(plan, ["dependencies", "--root", str(root)], timeout=60)
    def output(command: list[str]) -> str:
        result = subprocess.run(command, cwd=root, env=training_environment(plan),
                                capture_output=True, text=True, check=True, timeout=60)
        return result.stdout.strip()
    repo = Path(output(["git", "rev-parse", "--show-toplevel"]))
    if root != repo / "isaaclab_training":
        raise RuntimeError("training_root must be the manipulation repository's isaaclab_training directory")
    git_head = output(["git", "rev-parse", "HEAD"])
    git_branch = output(["git", "branch", "--show-current"])
    if git_branch:
        git_tag = ""
    else:
        try:
            git_tag = output(["git", "describe", "--tags", "--exact-match"])
        except subprocess.CalledProcessError:
            git_tag = ""
    isaac_sim_version = installation["isaac_sim_version"]
    if not isaac_sim_version.startswith("6.1."):
        raise RuntimeError(f"Isaac Sim {isaac_sim_version} does not match the accepted {EULA_SCOPE} scope")
    task = plan["task"]
    output([
        selected_uv(plan), "run", "--no-sync", "python", "-c",
        "import gymnasium as gym, isaaclab_training.tasks; "
        f"gym.spec({task!r}); print('registered')",
    ])
    gpu = output(["nvidia-smi", "--query-gpu=name,memory.free", "--format=csv,noheader"])
    try:
        remote = safe_remote(output(["git", "remote", "get-url", "origin"]))
    except subprocess.CalledProcessError:
        remote = ""
    canonical_remote = remote.removesuffix(".git")
    canonical_remote = canonical_remote.replace("ssh://github.com/", "https://github.com/")
    if remote and canonical_remote.lower() != (
        "https://github.com/nvidia-isaac-ros/isaac_ros_manipulation"
    ):
        raise RuntimeError("training repository origin is not NVIDIA-ISAAC-ROS/isaac_ros_manipulation")
    return {
        "runtime_checks": checks,
        "launch_environment": training_environment(plan),
        "dependency_audit": dependency_audit,
        **installation,
        "training_repository": str(repo),
        "training_head": git_head,
        "training_ref": git_branch or git_tag or git_head,
        "training_dirty": bool(output(["git", "status", "--porcelain"])),
        "training_remote": remote,
        "uv_version": output([selected_uv(plan), "--version"]),
        "python_version": output([selected_uv(plan), "run", "--no-sync", "python", "--version"]),
        "disk_free_bytes": shutil.disk_usage(root).free,
        "gpu": gpu,
    }


def refresh_progress(session: Path, plan: dict, state: dict, attempt: dict, *, force: bool = False) -> None:
    """Merge advancing evidence from this attempt; never treat file growth as an update."""
    console = local_path(session, attempt["console"])
    now = time.time()
    start = attempt.get("start_iteration", 0)
    previous = attempt.get("last_iteration")
    step, metrics = progress(console)
    evidence = []
    if step is not None and step >= start:
        evidence.append({"step": step, "epoch": console.stat().st_mtime,
                         "source": "console", "metrics": metrics})

    matches = sorted(Path(plan["output_root"]).glob(f"*_{attempt['run_name']}"))
    old_error = attempt.get("progress_probe_error")
    if len(matches) > 1:
        attempt["progress_probe_error"] = "Multiple output directories match this attempt; artifacts ignored."
    elif len(matches) == 1:
        run_dir = matches[0]
        attempt["run_dir"] = str(run_dir)
        if force or (now - attempt.get("progress_probe_epoch", 0) >= 30
                     and (step is None or (previous is not None and step <= previous))):
            attempt["progress_probe_epoch"] = now
            try:
                value = runtime_check(plan, ["progress", "--run-dir", str(run_dir),
                                            "--started-epoch", str(attempt["started_epoch"]),
                                            "--start-iteration", str(start)])
                if value:
                    evidence.append(value)
                attempt.pop("progress_probe_error", None)
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                # An incomplete/unreadable event file cannot disable the other sources.
                attempt["progress_probe_error"] = str(error)[-2000:]
        snapshots = {}
        for checkpoint in run_dir.glob("model_*.pt"):
            match = re.fullmatch(r"model_(\d+)\.pt", checkpoint.name)
            if not match:
                continue
            try:
                info = checkpoint.stat()
            except FileNotFoundError:
                continue
            label = int(match[1])
            if label < start or not info.st_size or info.st_mtime < attempt["started_epoch"]:
                continue
            snapshot = [info.st_size, info.st_mtime_ns]
            snapshots[checkpoint.name] = snapshot
            if attempt.get("checkpoint_snapshots", {}).get(checkpoint.name) == snapshot:
                evidence.append({"step": label, "epoch": info.st_mtime,
                                 "source": "checkpoint", "checkpoint": str(checkpoint), "metrics": {}})
        attempt["checkpoint_snapshots"] = snapshots
    if evidence:
        latest = max(evidence, key=lambda item: (item["step"], item["epoch"]))
        if previous is None or latest["step"] > previous:
            attempt["last_iteration"] = latest["step"]
            attempt["last_progress_epoch"] = min(now, latest["epoch"])
            attempt["progress_evidence"] = latest
            attempt.setdefault("metrics", {}).update(latest.get("metrics", {}))
            emit(session, state, "iteration", run_id=attempt["run_id"],
                 iteration=latest["step"], source=latest["source"])
    if old_error != attempt.get("progress_probe_error"):
        emit(session, state, "progress_probe", run_id=attempt["run_id"],
             error=attempt.get("progress_probe_error"))


def monitor_stop_reason(plan: dict, state: dict, attempt: dict) -> tuple[str, str] | None:
    now = time.time()
    elapsed = max(0, now - attempt["started_epoch"])
    budget = plan["training_budget_seconds"]
    if budget is not None and state["training_seconds"] + elapsed >= budget:
        return "budget", "training budget reached"
    if attempt.get("last_iteration") is None:
        if elapsed >= plan["timeouts"]["startup_seconds"]:
            return "startup_observation_timeout", "no training progress observed before the startup deadline"
    elif now - attempt["last_progress_epoch"] >= plan["timeouts"]["stall_seconds"]:
        return "progress_timeout", "no advancing training progress observed before the stall deadline"
    return None


def terminate_group(pid: int, birth: str | None, grace: int) -> None:
    if not process_alive(pid, birth):
        return
    os.killpg(pid, signal.SIGTERM)
    deadline = time.monotonic() + grace
    while process_alive(pid, birth) and time.monotonic() < deadline:
        time.sleep(0.5)
    if process_alive(pid, birth):
        os.killpg(pid, signal.SIGKILL)


def monitor(session: Path, plan: dict, state: dict, attempt: dict, process: subprocess.Popen | None) -> None:
    timeouts = plan["timeouts"]
    while True:
        code = process.poll() if process is not None else None
        alive = process_alive(attempt.get("pid"), attempt.get("process_birth"))
        if code is not None or not alive:
            break
        refresh_progress(session, plan, state, attempt)
        reason = monitor_stop_reason(plan, state, attempt)
        if reason and reason[0] != "budget":
            # Recheck all sources immediately before an observation-based stop.
            refresh_progress(session, plan, state, attempt, force=True)
            reason = monitor_stop_reason(plan, state, attempt)
        # The trainer may have exited normally during the bounded artifact probe.
        if (process is not None and process.poll() is not None) or not process_alive(attempt.get("pid"), attempt.get("process_birth")):
            break
        if reason:
            attempt["stop_origin"] = "supervisor"
            attempt["stop_kind"] = reason[0]
            attempt["stop_reason"] = f"Supervisor stopped trainer: {reason[1]}"
            emit(session, state, "trainer_stop_requested", run_id=attempt["run_id"],
                 message=attempt["stop_reason"], last_progress=attempt.get("progress_evidence"),
                 probe_error=attempt.get("progress_probe_error"))
            terminate_group(attempt["pid"], attempt["process_birth"], timeouts["grace_seconds"])
            break
        time.sleep(10)
    if process is not None:
        attempt["exit_code"] = process.wait()
    else:
        attempt["exit_code"] = None
    refresh_progress(session, plan, state, attempt, force=True)
    attempt.setdefault("stop_origin", "trainer_exit" if attempt["exit_code"] is not None else "unknown")
    attempt["ended_utc"] = utc()
    attempt["elapsed_seconds"] = max(0.0, time.time() - attempt["started_epoch"])
    state["training_seconds"] += attempt["elapsed_seconds"]
    attempt["additional_updates_completed"] = (
        max(0, attempt["last_iteration"] - attempt.get("start_iteration", 0) + 1)
        if attempt.get("last_iteration") is not None else 0
    )
    attempt["status"] = "verifying"
    emit(session, state, "trainer_exited", run_id=attempt["run_id"], exit_code=attempt["exit_code"])


def verify_attempt(session: Path, plan: dict, state: dict, attempt: dict) -> None:
    ensure_session_code(state)
    run = attempt["spec"]
    output = Path(plan["output_root"])
    matches = sorted(output.glob(f"*_{attempt['run_name']}"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one IsaacLab output directory for {attempt['run_name']}, found {len(matches)}")
    run_dir = matches[0]
    attempt["run_dir"] = str(run_dir)
    command = [
        selected_uv(plan), "run", "--no-sync", "python", str(SKILL / "scripts/verify_run.py"),
        "--run-dir", str(run_dir), "--console", str(local_path(session, attempt["console"])),
        "--expected-step", str(final_iteration(attempt)),
        "--iterations", str(run["max_iterations"]),
        "--video-length", str(run["video_length"]),
        "--min-video-clips", "2" if run["video_schedule"] == "start_end" else "1",
    ]
    if not run["video"]:
        command.append("--no-video")
    result = subprocess.run(command, cwd=plan["training_root"], capture_output=True,
                            text=True, check=False, timeout=300, env=training_environment(plan))
    attempt["verification_process"] = {"argv": command, "exit_code": result.returncode,
                                        "stdout": result.stdout, "stderr": result.stderr}
    if result.returncode != 0:
        raise RuntimeError(f"artifact verification failed ({result.returncode}): {result.stdout} {result.stderr}")
    attempt["verification"] = json.loads(result.stdout)
    capture_configs(session, plan, attempt, required=True)
    attempt["artifacts"] = [fingerprint(Path(path)) for path in
                            [attempt["verification"]["checkpoint"],
                             *attempt["verification"].get("videos", []),
                             *attempt["verification"].get("exports", [])] if Path(path).is_file()]
    attempt["additional_updates_completed"] = run["max_iterations"]
    parent_updates = run.get("parent_total_updates")
    attempt["total_lineage_updates"] = (run["max_iterations"] if not run["parent_checkpoint"] else
                                        parent_updates + run["max_iterations"] if parent_updates is not None else None)
    attempt["metrics"] = progress(local_path(session, attempt["console"]))[1]
    attempt["status"] = "artifact_verified" if attempt["exit_code"] is None else "verified"
    emit(session, state, "artifacts_verified", run_id=attempt["run_id"],
         exit_code=attempt["exit_code"], checkpoint=attempt["verification"]["checkpoint"])


def capture_configs(session: Path, plan: dict, attempt: dict, *, required: bool = False) -> None:
    """Save the trainer's resolved YAML files in this attempt's params directory."""
    matches = sorted(Path(plan["output_root"]).glob(f"*_{attempt['run_name']}"))
    if len(matches) != 1:
        if required:
            raise RuntimeError("resolved config output directory is unavailable or ambiguous")
        return
    attempt["run_dir"] = str(matches[0])
    files = []
    configs = []
    for name in ("agent.yaml", "env.yaml"):
        path = matches[0] / "params" / name
        if path.is_file():
            files.append(path)
        elif required:
            raise RuntimeError(f"resolved config missing: {path}")
    if files:
        destination = run_folder(session, attempt) / "params"
        destination.mkdir(parents=True, exist_ok=True)
        for path in files:
            saved = destination / path.name
            temporary = saved.with_name(f".{saved.name}.{os.getpid()}.tmp")
            temporary.write_bytes(path.read_bytes())
            temporary.replace(saved)
            configs.append({**fingerprint(saved), "path": str(saved.relative_to(session)),
                            "source_path": str(path)})
    attempt["resolved_configs"] = configs


def evaluation_process(session: Path, plan: dict, state: dict, attempt: dict,
                       command: list[str], console: Path, identity: str) -> int | None:
    """Persist child identity before waiting; recover a live evaluator on resume."""
    job = attempt.get("evaluation_job")
    process = None
    if job and job["identity"] != identity:
        raise RuntimeError("unfinished evaluation must be recovered before changing its configuration")
    if not job:
        with console.open("a") as stream:
            stream.write(f"\n=== {utc()} {identity} ===\n{shlex.join(command)}\n")
            stream.flush()
            process = subprocess.Popen(command, cwd=plan["training_root"], stdout=stream,
                                       stderr=subprocess.STDOUT, env=training_environment(plan),
                                       stdin=subprocess.DEVNULL, start_new_session=True)
        job = {"identity": identity, "pid": process.pid, "birth": process_birth(process.pid),
               "started_epoch": time.time(), "argv": command}
        attempt["evaluation_job"] = job
        emit(session, state, "evaluation_started", run_id=attempt["run_id"], **job)
    while (process.poll() is None if process else process_alive(job["pid"], job["birth"])):
        elapsed = time.time() - job["started_epoch"]
        budget = plan["evaluation_budget_seconds"]
        if elapsed >= 1800 or (budget is not None and state["evaluation_seconds"] + elapsed >= budget):
            terminate_group(job["pid"], job["birth"], plan["timeouts"]["grace_seconds"])
            job["stop_reason"] = "evaluation timeout or budget reached"
            break
        time.sleep(2)
    code = process.wait() if process else job.get("exit_code")
    if not job.get("accounted"):
        elapsed = max(0, time.time() - job["started_epoch"])
        state["evaluation_seconds"] += elapsed
        job.update(exit_code=code, seconds=elapsed, accounted=True)
        emit(session, state, "evaluation_exited", run_id=attempt["run_id"], **job)
    return code


def evaluate_attempt(session: Path, plan: dict, state: dict, attempt: dict) -> None:
    ensure_session_code(state)
    config = plan["evaluation"]
    if not config["enabled"] or attempt["spec"]["purpose"] == "smoke":
        return
    provenance = evaluation_provenance(session, plan, attempt)
    record = run_folder(session, attempt)
    record.mkdir(parents=True, exist_ok=True)
    journal = record / "evaluations.jsonl"
    console = record / "evaluation.log"
    group = config.get("group", "tuning")
    attempt.setdefault("evaluations", {})
    for requested_step in config["checkpoint_steps"]:
        step = final_iteration(attempt) if requested_step == "final" else requested_step
        checkpoint = Path(attempt["run_dir"]) / f"model_{step}.pt"
        if not checkpoint.is_file():
            raise RuntimeError(f"evaluation checkpoint missing: {checkpoint}")
        condition = {"group": group, "task": plan["task"], "spec": attempt["spec"],
                     "config": {key: value for key, value in config.items()
                                if key not in {"seeds", "checkpoint_steps", "enabled"}},
                     "checkpoint_sha256": fingerprint(checkpoint)["sha256"],
                     "provenance": provenance}
        evaluation_id = hashlib.sha256(json.dumps(condition, sort_keys=True).encode()).hexdigest()
        label = str(step) if group == "tuning" else f"{group}:{step}"
        existing = attempt["evaluations"].get(label)
        if (existing and existing.get("evaluation_id") == evaluation_id
                and existing.get("seeds") == config["seeds"]):
            continue
        results = []
        for seed in config["seeds"]:
            identity = f"{evaluation_id}:seed{seed}"
            payload = completed_result(journal, evaluation_id, seed)
            # Even with a committed result, reconcile the child before using the GPU again.
            if not payload or attempt.get("evaluation_job", {}).get("identity") == identity:
                budget = plan["evaluation_budget_seconds"]
                if not attempt.get("evaluation_job") and budget is not None and state["evaluation_seconds"] >= budget:
                    raise RuntimeError("evaluation budget reached")
                command = [
                    selected_uv(plan), "run", "--no-sync", "python", str(SKILL / "scripts/evaluate.py"),
                    "--task", plan["task"], "--checkpoint", str(checkpoint),
                    "--journal", str(journal), "--evaluation-id", evaluation_id,
                    "--seed", str(seed), "--num_envs", str(config["num_envs"]),
                    "--max_steps", str(config["max_steps"]),
                    "--session", str(session), "--run-id", attempt["run_id"],
                ]
                code = evaluation_process(session, plan, state, attempt, command, console, identity)
                payload = completed_result(journal, evaluation_id, seed)
                job = attempt.pop("evaluation_job", {})
                append_record(journal, {"kind": "process", "evaluation_id": evaluation_id,
                                       "seed": seed, "exit_code": code, **job})
                save_state(session, state)
                if code not in (0, None) or not payload or job.get("stop_reason"):
                    raise RuntimeError(f"evaluation checkpoint {step}, seed {seed} failed; inspect {console}")
            results.append({"seed": seed, "successes": payload["terminal_success_count"],
                            "episodes": payload["num_envs"]})
        successes = sum(item["successes"] for item in results)
        episodes = sum(item["episodes"] for item in results)
        attempt["evaluations"][label] = {
            "evaluation_id": evaluation_id, "group": group, "step": step,
            "checkpoint": str(checkpoint), "successes": successes, "episodes": episodes,
            "rate": successes / episodes, "seeds": list(config["seeds"]),
            "config": json.loads(json.dumps(config)),
            "records": str(journal.relative_to(session)),
            "condition": "first episode; 0% near goal; 20–60 mm approach; no observation corruption; 3 mm geometric success",
        }
        emit(session, state, "evaluation_complete", run_id=attempt["run_id"],
             step=step, group=group, result=attempt["evaluations"][label])


def diagnose(console: Path) -> str:
    text = tail(console).lower()
    if "collisionstacksize" in text and "overflow" in text:
        return "PhysX collision stack overflow: inspect GPU memory, then consider sim.physics.gpu_collision_stack_size=2147483648."
    if "cuda out of memory" in text or "outofmemoryerror" in text:
        return "CUDA out of memory: reduce --num_envs (for example 256 to 128); record any physics trade-off."
    if "video" in text and ("source unavailable" in text or "no camera" in text):
        return "Video source unavailable: omit --viz none and verify Kit/video extras."
    return "Inspect the console and configuration; retry only after a bounded diagnosis."


def execute_next(session: Path, plan: dict, state: dict) -> None:
    ensure_session_code(state)
    spec = state["queue"].pop(0)
    started = utc()
    number = len(state["attempts"]) + 1
    run_name = f"{timestamp(started)}__r{number:02d}__{slug(spec['id'])}"
    attempt = {
        "run_id": spec["id"], "run_name": run_name, "spec": spec, "status": "launching",
        "run_number": number, "record_dir": f"runs/{run_name}",
        "started_utc": started, "started_epoch": time.time(), "last_iteration": None,
        "last_progress_epoch": time.time(), "exit_code": None,
    }
    state["current"] = attempt
    state["attempts"].append(attempt)
    state["status"] = "running"
    record = run_folder(session, attempt)
    record.mkdir(parents=True, exist_ok=False)
    command, environment = run_command({**spec, "run_name": run_name}, plan, session)
    attempt["command"] = command
    attempt["environment"] = environment
    attempt["console"] = str((record / "train.log").relative_to(session))
    attempt["cwd"] = plan["training_root"]
    emit(session, state, "run_launching", run_id=spec["id"],
         message=spec["hypothesis"], command=shlex.join(command))
    try:
        attempt["training_sha256"] = training_fingerprint(Path(plan["training_root"]))
        attempt["start_iteration"] = parent_iteration(spec, plan)
        attempt["final_iteration"] = final_iteration(attempt)
        if spec["parent_checkpoint"]:
            emit(session, state, "parent_checkpoint_loaded", run_id=spec["id"],
                 message=f"resume labels {attempt['start_iteration']}–{attempt['final_iteration']}")
        if plan["evaluation"]["enabled"]:
            for requested in plan["evaluation"]["checkpoint_steps"]:
                if requested != "final" and not attempt["start_iteration"] <= requested <= attempt["final_iteration"]:
                    raise ValueError(f"evaluation checkpoint step {requested} is outside this run's "
                                     f"{attempt['start_iteration']}–{attempt['final_iteration']} labels")
        with local_path(session, attempt["console"]).open("wb") as stream:
            process = subprocess.Popen(command, cwd=plan["training_root"], env=environment,
                                       stdin=subprocess.DEVNULL, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        attempt["pid"] = process.pid
        attempt["process_birth"] = process_birth(process.pid)
        attempt["status"] = "running"
        emit(session, state, "trainer_started", run_id=spec["id"], pid=process.pid)
        monitor(session, plan, state, attempt, process)
        if attempt.get("stop_reason") or attempt["exit_code"] not in (0, None):
            raise RuntimeError(attempt.get("stop_reason") or f"trainer exit {attempt['exit_code']}")
        verify_attempt(session, plan, state, attempt)
        evaluate_attempt(session, plan, state, attempt)
        attempt["status"] = "complete"
        state["completed_runs"] += 1
        emit(session, state, "run_complete", run_id=spec["id"],
             message=f"checkpoint {attempt['verification']['checkpoint']}")
    except Exception as error:
        attempt["status"] = "evaluation_failed" if attempt.get("verification") else "failed"
        attempt["error"] = str(error)
        try:
            capture_configs(session, plan, attempt)
        except Exception as capture_error:
            attempt["config_capture_error"] = str(capture_error)
        attempt["diagnosis"] = (attempt.get("stop_reason") or
                                diagnose(local_path(session, attempt["console"])))
        state["status"] = "blocked"
        emit(session, state, "run_failed", run_id=spec["id"], message=str(error),
             diagnosis=attempt["diagnosis"])


def reconcile_current(session: Path, plan: dict, state: dict) -> None:
    """Recover a saved attempt; leave completed attempts for the campaign decision gate."""
    attempt = state.get("current")
    if not attempt or attempt["status"] == "complete":
        return
    if attempt["status"] == "failed":
        state["status"] = "blocked"
        return
    if attempt.get("pid") and process_alive(attempt["pid"], attempt.get("process_birth")):
        emit(session, state, "trainer_recovered", run_id=attempt["run_id"],
             message="Monitoring an existing trainer after supervisor restart")
        monitor(session, plan, state, attempt, None)
    elif attempt["status"] in {"launching", "running"}:
        attempt["exit_code"] = None
        attempt["ended_utc"] = utc()
        attempt["elapsed_seconds"] = max(0, time.time() - attempt["started_epoch"])
        state["training_seconds"] += attempt["elapsed_seconds"]
        attempt["elapsed_estimated"] = True
        attempt["status"] = "verifying"
        emit(session, state, "trainer_disappeared", run_id=attempt["run_id"],
             message="Exit code unavailable; checking final artifacts")
    try:
        if attempt.get("stop_reason") or attempt.get("exit_code") not in (0, None):
            raise RuntimeError(attempt.get("stop_reason") or f"trainer exit {attempt['exit_code']}")
        if attempt["status"] != "artifact_verified" and attempt["status"] != "verified":
            verify_attempt(session, plan, state, attempt)
        evaluate_attempt(session, plan, state, attempt)
        attempt["status"] = "complete"
        state["completed_runs"] += 1
        state["status"] = "running"
        emit(session, state, "run_recovered", run_id=attempt["run_id"],
             message="Recovered from saved artifacts")
    except Exception as error:
        attempt["status"] = "evaluation_failed" if attempt.get("verification") else "failed"
        attempt["error"] = str(error)
        state["status"] = "blocked"
        emit(session, state, "recovery_failed", run_id=attempt["run_id"], message=str(error))


def campaign(session: Path) -> int:
    """Run or recover a campaign while honoring manual decisions and resource limits."""
    if not (session / "session.json").is_file():
        raise RuntimeError("legacy sessions are read-only; start a new session with the desired parent checkpoint")
    require_standalone_session(session_state(session)[0])
    lock = (session / ".lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("another supervisor is already running for this session") from error
    try:
        plan, state = session_state(session)
        if state["status"] in {"complete", "closed"}:
            return 0
        state["supervisor_pid"] = os.getpid()
        state["supervisor_birth"] = process_birth(os.getpid())
        emit(session, state, "supervisor_started", pid=os.getpid())
        try:
            require_eula_record()
            ensure_session_code(state)
            state["failure_phase"] = "preflight"
            pending_runs = list(state["queue"])
            if state.get("current"):
                pending_runs.append(state["current"]["spec"])
            state["preflight"] = check_preflight({**plan, "runs": pending_runs})
            state.pop("failure_phase", None)
            state.pop("error", None)
            emit(session, state, "preflight_complete", message=state["preflight"]["training_ref"])
            reconcile_current(session, plan, state)
            if state.get("current") and state["current"]["status"] == "complete":
                state["current"] = None
                if plan["decision_mode"] == "manual" and state["completed_runs"] < plan["max_completed_runs"]:
                    state["status"] = "decision_pending"
                    emit(session, state, "decision_pending", message="Inspect results and queue the next run")
            while state["status"] not in {"blocked", "decision_pending"} and state["queue"] and state["completed_runs"] < plan["max_completed_runs"]:
                if plan["training_budget_seconds"] is not None and state["training_seconds"] >= plan["training_budget_seconds"]:
                    state["status"] = "blocked"
                    emit(session, state, "budget_exhausted", message="Training budget exhausted before next run")
                    break
                execute_next(session, plan, state)
                if state["status"] == "blocked":
                    break
                state["current"] = None
                if plan["decision_mode"] == "manual" and state["completed_runs"] < plan["max_completed_runs"]:
                    state["status"] = "decision_pending"
                    emit(session, state, "decision_pending", message="Inspect results and queue the next run")
                    break
            if state["status"] not in {"blocked", "decision_pending"}:
                state["status"] = "complete"
                emit(session, state, "session_complete", message="All planned runs finished")
        except Exception as error:
            state["status"] = "blocked"
            state["error"] = str(error)
            emit(session, state, "supervisor_error", message=str(error), traceback=traceback.format_exc())
            raise
        finally:
            state["supervisor_pid"] = None
            state["supervisor_birth"] = None
            save_state(session, state)
            write_report(session, state)
            if state["status"] in {"complete", "closed"}:
                cleanup_empty_logs(session)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    return 2 if state["status"] == "blocked" else 0


def write_report(session: Path, state: dict) -> None:
    write_campaign_report(session, state, guide=GUIDE, eula_scope=EULA_SCOPE)


def launcher(session: Path, choice: str) -> str:
    plan, state = session_state(session)
    require_standalone_session(plan)
    if state["status"] in {"complete", "closed"}:
        raise RuntimeError("session is already complete")
    require_eula_record()
    if process_alive(state.get("supervisor_pid"), state.get("supervisor_birth")):
        raise RuntimeError("supervisor already running")
    root = Path(plan["training_root"])
    if Path.cwd().resolve() != root:
        raise RuntimeError(f"run this command from {root}")
    if choice == "auto":
        choice = "tmux" if shutil.which("tmux") else "detached"
    environment = training_environment(plan)
    command = [str(root / ".venv/bin/python"), str(Path(__file__).resolve()), "run", "--session", str(session)]
    # tmux/systemd may have an older server environment: reset it before Python starts.
    clean_command = ["/usr/bin/env", "-i", *[f"{key}={value}" for key, value in sorted(environment.items())], *command]
    if choice == "tmux":
        if not shutil.which("tmux"):
            raise RuntimeError("tmux is unavailable; use --launcher detached or foreground")
        tmux_name = "dp-" + session.name
        shell = "cd " + shlex.quote(str(root)) + " && " + shlex.join(clean_command)
        shell += " >> " + shlex.quote(str(session / "supervisor.log")) + " 2>&1"
        state["launcher"] = {"type": "tmux", "session": tmux_name}
    elif choice == "systemd":
        if not shutil.which("systemd-run"):
            raise RuntimeError("systemd-run is unavailable")
        unit = "displayport-" + session.name.lower()
        state["launcher"] = {"type": "systemd", "unit": unit}
    elif choice == "detached":
        state["launcher"] = {"type": "detached"}
    elif choice == "foreground":
        state["launcher"] = {"type": "foreground"}
    else:
        raise ValueError(f"unknown launcher {choice}")
    emit(session, state, "supervisor_launched", message=choice)
    try:
        if choice == "tmux":
            subprocess.run(["tmux", "new-session", "-d", "-s", tmux_name, shell], check=True)
        elif choice == "systemd":
            subprocess.run(["systemd-run", "--user", "--collect", "--unit", unit,
                            f"--property=WorkingDirectory={root}", *clean_command], check=True)
        elif choice == "detached":
            with (session / "supervisor.log").open("ab") as stream:
                subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=subprocess.STDOUT, env=environment,
                                 start_new_session=True)
        else:
            if campaign(session):
                raise RuntimeError(f"campaign blocked; inspect {session / 'report.md'}")
    except Exception as error:
        _, latest = session_state(session)
        latest["status"] = "blocked"
        emit(session, latest, "launch_failed", message=str(error))
        raise
    return choice


def queue_run(session: Path, spec_path: Path) -> None:
    with idle_session(session) as (plan, state):
        queue_idle_run(session, plan, state, spec_path)


def queue_idle_run(session: Path, plan: dict, state: dict, spec_path: Path) -> None:
    if state.get("pending_evaluation") or any(item.get("evaluation_job") for item in state["attempts"]):
        raise RuntimeError("recover the unfinished evaluation before queueing another run")
    if state["status"] not in {"decision_pending", "blocked"}:
        raise RuntimeError("queue a new run only at a decision point or after a failure")
    if state["completed_runs"] >= plan["max_completed_runs"]:
        raise RuntimeError("completed run limit reached")
    used = {item["run_id"] for item in state["attempts"]}
    used.update(item["id"] for item in state["queue"])
    spec = normalize_run(read_json(spec_path), used=used, task=plan["task"])
    failed = state.get("current")
    if failed and failed.get("status") in {"failed", "evaluation_failed"}:
        spec["retry_of"] = spec.get("retry_of") or failed["run_id"]
    if spec.get("retry_of") and spec["retry_of"] not in {item["run_id"] for item in state["attempts"]}:
        raise ValueError("retry_of must refer to an existing attempt in this session")
    state["queue"].insert(0, spec)
    state["current"] = None
    state["status"] = "created"
    emit(session, state, "run_queued", run_id=spec["id"], message=spec["hypothesis"])


@contextmanager
def idle_session(session: Path, *, recover_evaluator: bool = False):
    if not (session / "session.json").is_file():
        raise RuntimeError("legacy sessions are read-only; start a new session")
    with (session / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("supervisor is active; retry at a decision point") from error
        try:
            plan, state = session_state(session)
            require_standalone_session(plan)
            if Path.cwd().resolve() != Path(plan["training_root"]):
                raise RuntimeError(f"run from the training package root: {plan['training_root']}")
            if any(process_alive(item.get("pid"), item.get("process_birth")) for item in state["attempts"]):
                raise RuntimeError("trainer is still active; resume monitoring before changing the session")
            if not recover_evaluator and any(
                process_alive(item.get("evaluation_job", {}).get("pid"), item.get("evaluation_job", {}).get("birth"))
                for item in state["attempts"]
            ):
                raise RuntimeError("evaluator is still active; recover it before changing the session")
            yield plan, state
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def evaluate_saved(session: Path, run_id: str, config_path: Path) -> None:
    """Checkpoint screening/validation shares the originating run's log and journal."""
    with idle_session(session, recover_evaluator=True) as (plan, state):
        require_eula_record()
        ensure_session_code(state)
        attempt = next(item for item in state["attempts"] if item["run_id"] == run_id)
        if not attempt.get("verification"):
            raise RuntimeError("evaluate requires a verified run")
        config = read_json(config_path)
        raw = {key: value for key, value in plan.items() if key != "eula_confirmation"}
        raw["runs"] = [{key: value for key, value in run.items()
                        if key != "video_interval" or run["video_schedule"] != "start_end"}
                       for run in plan["runs"]]
        raw["evaluation"] = config
        evaluation = normalize_plan(raw)["evaluation"]
        if not evaluation["enabled"]:
            raise ValueError("evaluation must be enabled")
        if evaluation["group"] == "independent":
            used = {seed for item in state["attempts"] for result in item.get("evaluations", {}).values()
                    if result.get("group", "tuning") != "independent" for seed in result["seeds"]}
            if used.intersection(evaluation["seeds"]):
                raise ValueError("independent validation seeds overlap screening/tuning seeds")
        request = {"run_id": run_id, "evaluation": evaluation}
        if state.get("pending_evaluation") not in (None, request):
            raise RuntimeError("recover the pending evaluation with its original config first")
        for requested in evaluation["checkpoint_steps"]:
            step = final_iteration(attempt) if requested == "final" else requested
            checkpoint = Path(attempt["run_dir"]) / f"model_{step}.pt"
            if not checkpoint.is_file():
                raise RuntimeError(f"evaluation checkpoint missing: {checkpoint}")
        evaluation_provenance(session, plan, attempt)
        state["pending_evaluation"] = request
        emit(session, state, "checkpoint_evaluation_requested", **request)
        try:
            evaluate_attempt(session, {**plan, "evaluation": evaluation}, state, attempt)
        except Exception as error:
            state["evaluation_error"] = str(error)
            emit(session, state, "checkpoint_evaluation_failed", run_id=run_id, message=str(error))
            raise
        state.pop("pending_evaluation", None)
        state.pop("evaluation_error", None)
        emit(session, state, "checkpoint_evaluation_finished", run_id=run_id)


def cancel_evaluation(session: Path) -> None:
    """Release a failed request without discarding evidence or interrupting a child."""
    with idle_session(session) as (plan, state):
        request = state.get("pending_evaluation")
        if request is None:
            raise RuntimeError("no pending evaluation to cancel")
        for attempt in state["attempts"]:
            job = attempt.get("evaluation_job")
            if not job:
                continue
            # idle_session already rejected live children. Reconcile elapsed cost
            # and invalidate any result committed before an interrupted exit.
            record = run_folder(session, attempt)
            evaluation_process(session, plan, state, attempt, job["argv"],
                               record / "evaluation.log", job["identity"])
            job = attempt.pop("evaluation_job")
            identity, _, seed = job["identity"].rpartition(":seed")
            append_record(record / "evaluations.jsonl", {
                **job, "kind": "process", "evaluation_id": identity, "seed": int(seed),
                "stop_reason": "evaluation cancelled",
            })
        state.pop("pending_evaluation")
        error = state.pop("evaluation_error", None)
        emit(session, state, "checkpoint_evaluation_cancelled", request=request, error=error)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("eula-status", help="show the current user's remembered Omniverse EULA acceptance")
    accept = commands.add_parser("eula-accept", help="record an explicit user acceptance of the Omniverse EULA")
    accept.add_argument("--user-confirmed", action="store_true",
                        help="the agent received an explicit Yes from this OS user after showing the license")
    commands.add_parser("eula-revoke", help="forget the current user's remembered acceptance")
    preflight = commands.add_parser("preflight", help="verify a prepared checkout without creating a session or training")
    preflight.add_argument("--plan", type=Path, required=True)
    start = commands.add_parser("start", help="create a campaign and launch its supervisor")
    start.add_argument("--plan", type=Path, required=True)
    start.add_argument("--launcher", choices=("auto", "tmux", "detached", "systemd", "foreground"), default="auto")
    for name in ("resume", "run", "status", "report", "queue", "finish", "evaluate", "cancel-evaluation", "note"):
        sub = commands.add_parser(name)
        sub.add_argument("--session", type=Path, required=True)
        if name == "resume":
            sub.add_argument("--launcher", choices=("auto", "tmux", "detached", "systemd", "foreground"), default="auto")
        if name == "queue":
            sub.add_argument("--spec", type=Path, required=True)
        if name == "finish":
            sub.add_argument("--reason", required=True)
        if name == "evaluate":
            sub.add_argument("--run-id", required=True)
            sub.add_argument("--config", type=Path, required=True,
                             help="Evaluation object with group, seeds, environments and checkpoint_steps")
        if name == "note":
            sub.add_argument("--message", required=True)
            sub.add_argument("--finding", action="store_true")
            sub.add_argument("--next-action")
    args = parser.parse_args()
    try:
        if args.command == "eula-status":
            record = eula_record()
            print(json.dumps({"accepted": record is not None, "agreement_url": EULA_URL,
                              "scope": EULA_SCOPE, "record": str(eula_record_path()),
                              "accepted_at_utc": record["accepted_at_utc"] if record else None}, indent=2))
        elif args.command == "eula-accept":
            if args.user_confirmed:
                source = "explicit_user_reply"
            elif sys.stdin.isatty():
                print(f"Read the NVIDIA Omniverse License Agreement: {EULA_URL}")
                if input("Do you accept it for Isaac Sim 6.1? Type YES to accept: ").strip() != "YES":
                    raise RuntimeError("EULA acceptance was not given")
                source = "interactive_prompt"
            else:
                raise RuntimeError("Ask the user first, then pass --user-confirmed after an explicit Yes")
            record = record_eula_acceptance(source)
            print(json.dumps({"accepted": True, "accepted_at_utc": record["accepted_at_utc"],
                              "record": str(eula_record_path())}))
        elif args.command == "eula-revoke":
            eula_record_path().unlink(missing_ok=True)
            print(json.dumps({"accepted": False, "record": str(eula_record_path())}))
        elif args.command == "preflight":
            plan = normalize_plan(read_json(args.plan))
            require_eula_record()
            plan["eula_accepted"] = True
            print(json.dumps(check_preflight(plan), indent=2))
        elif args.command == "start":
            session = create_session(args.plan)
            selected = launcher(session, args.launcher)
            print(json.dumps({"session": str(session), "launcher": selected}))
        else:
            session = args.session.resolve()
            if not session.is_dir() or not any((session / name).is_file() for name in ("session.json", "state.json")):
                raise RuntimeError(f"invalid session: {session}")
            legacy = not (session / "session.json").is_file()
            if legacy and args.command != "status":
                raise RuntimeError("legacy sessions support status only; start a new session to continue training")
            if args.command != "status":
                require_standalone_session(session_state(session)[0])
            if args.command == "run":
                return campaign(session)
            elif args.command == "resume":
                selected = launcher(session, args.launcher)
                print(json.dumps({"session": str(session), "launcher": selected}))
            elif args.command == "status":
                _, state = session_state(session)
                current = state.get("current")
                active = max(0, time.time() - current["started_epoch"]) if current and current.get("status") in {"launching", "running"} else 0
                step = current.get("last_iteration") if current else None
                total = current["spec"]["max_iterations"] if current else None
                eta = reported_eta(local_path(session, current["console"])) if current and current.get("status") == "running" else None
                progressed = step - current.get("start_iteration", 0) + 1 if step is not None else 0
                if eta is None and progressed >= 5:
                    eta = (active / progressed) * max(0, total - progressed)
                print(json.dumps({"session": str(session), "status": state["status"],
                                  "completed_runs": state["completed_runs"],
                                  "training_hours": round((state["training_seconds"] + active) / 3600, 3),
                                  "evaluation_hours": round(state["evaluation_seconds"] / 3600, 3),
                                  "current_run": current["run_id"] if current else None,
                                  "current_iteration": step,
                                  "progress_evidence": current.get("progress_evidence") if current else None,
                                  "progress_probe_error": current.get("progress_probe_error") if current else None,
                                  "stop_origin": current.get("stop_origin") if current else None,
                                  "stop_reason": current.get("stop_reason") if current else None,
                                  "failure_phase": state.get("failure_phase"),
                                  "estimated_seconds_remaining": round(eta) if eta is not None else None,
                                  "supervisor_alive": process_alive(state.get("supervisor_pid"), state.get("supervisor_birth")),
                                  "trainer_alive": process_alive(current.get("pid"), current.get("process_birth")) if current else False,
                                  "report": str(session / ("final_report.md" if legacy else "report.md"))}, indent=2))
            elif args.command == "queue":
                queue_run(session, args.spec)
            elif args.command == "evaluate":
                evaluate_saved(session, args.run_id, args.config)
            elif args.command == "cancel-evaluation":
                cancel_evaluation(session)
            elif args.command == "note":
                with idle_session(session) as (_, state):
                    if args.finding:
                        state.setdefault("findings", []).append(args.message)
                    if args.next_action:
                        state["next_action"] = args.next_action
                    emit(session, state, "agent_note", message=args.message)
            elif args.command == "finish":
                with idle_session(session) as (_, state):
                    if state["status"] not in {"decision_pending", "blocked"}:
                        raise RuntimeError("finish is allowed only at a decision point or after a failure")
                    state["status"] = "closed"
                    emit(session, state, "session_closed", message=args.reason)
                    cleanup_empty_logs(session)
            elif args.command == "report":
                _, state = session_state(session)
                write_report(session, state)
                print(session / "report.md")
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
