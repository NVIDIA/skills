#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packaged Docker consumer for PAS action requests.

Implements the platform four-verb contract without asking an agent to assemble
mounts, credentials, job-record ordering, or lifecycle state by hand.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import stat
import subprocess
import sys
from typing import Any

import run_deft_action as action
from deft_action_contract import load_state


REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
JOB_RECORD = REPO_ROOT / "scripts" / "tao_job_record.py"


def _job_env(request: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    env["TAO_STATE_DIR"] = request["job_state_dir"]
    return env


def _run(
    command: list[str], *, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def _require_forwarded_environment(request: dict[str, Any]) -> None:
    missing = [name for name in request["forward_env"] if not os.environ.get(name)]
    if missing:
        raise ValueError(
            "missing approved credential environment at point of use: "
            + ", ".join(missing)
        )


def _job_record_path(request: dict[str, Any], job_id: str) -> pathlib.Path:
    return pathlib.Path(request["job_state_dir"]) / "jobs" / f"{job_id}.json"


def _open_job(request: dict[str, Any]) -> tuple[str, pathlib.Path]:
    bundle = request["spec_bundle"]
    command = [
        sys.executable,
        str(JOB_RECORD),
        "open",
        "--platform",
        "docker",
        "--image",
        request["record_image"],
        "--network-arch",
        bundle["network_arch"],
        "--action",
        bundle["action"],
        "--storage-tier",
        "A",
        "--results-dir",
        request["stage_dir"],
    ]
    for excluded in bundle["upload_excludes"]:
        command.extend(["--upload-exclude", excluded])
    completed = _run(command, env=_job_env(request))
    if completed.returncode != 0:
        raise ValueError(
            "job-record open failed: "
            + (completed.stderr or completed.stdout).strip()
        )
    job_id = completed.stdout.strip()
    if not job_id:
        raise ValueError("job-record open did not return an id")
    return job_id, _job_record_path(request, job_id)


def _mark(
    request: dict[str, Any],
    job_id: str,
    state: str,
    *,
    backend_ref: str | None = None,
    message: str = "",
) -> None:
    command = [
        sys.executable,
        str(JOB_RECORD),
        "mark",
        job_id,
        "--state",
        state,
        "--source",
        "agent",
    ]
    if backend_ref:
        command.extend(["--backend-ref", backend_ref])
    if message:
        command.extend(["--message", message])
    completed = _run(command, env=_job_env(request))
    if completed.returncode != 0:
        raise ValueError(
            "job-record transition failed: "
            + (completed.stderr or completed.stdout).strip()
        )


def _load(args: argparse.Namespace) -> tuple[pathlib.Path, dict[str, Any]]:
    request_path, request = action._load_request(args.request)  # noqa: SLF001
    if request["platform"] != "docker":
        raise ValueError("Docker consumer requires a platform=docker request")
    return request_path, request


def _docker_command(request: dict[str, Any], job_id: str) -> list[str]:
    state = load_state(pathlib.Path(request["results_dir"]))
    config = state["config"]
    gpu_ids = config.get("gpu_ids")
    num_gpus = request["spec_bundle"]["compute_shape"]["gpus"]
    if (
        not isinstance(gpu_ids, list)
        or len(gpu_ids) != num_gpus
        or len(set(gpu_ids)) != len(gpu_ids)
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in gpu_ids
        )
    ):
        raise ValueError("immutable state contains an invalid Docker GPU allocation")
    uid = os.getuid()
    gid = os.getgid()
    if uid == 0:
        raise ValueError(
            "refusing a writable PAS Docker launch as UID 0; run as the approved "
            "non-root workspace owner"
        )
    command = [
        "docker",
        "run",
        "--detach",
        "--name",
        job_id,
        "--label",
        f"tao-job={job_id}",
        "--runtime=nvidia",
        "--gpus",
        '"device=' + ",".join(str(value) for value in gpu_ids) + '"',
        "--shm-size=8g",
        "--user",
        f"{uid}:{gid}",
    ]
    for supplemental_gid in sorted(set(os.getgroups()) - {gid}):
        command.extend(["--group-add", str(supplemental_gid)])
    for mount in request["mounts"]:
        rendered = f"{mount['source']}:{mount['target']}"
        if mount["read_only"]:
            rendered += ":ro"
        command.extend(["--volume", rendered])
    for name, value in request["environment"].items():
        command.extend(["--env", f"{name}={value}"])
    # PyTorch calls getpass.getuser() while initializing Inductor. The numeric
    # host UID intentionally has no image passwd entry, so Docker supplies a
    # stable, non-secret identity as platform-owned runtime configuration.
    command.extend(["--env", "USER=tao", "--env", "LOGNAME=tao"])
    for name in request["forward_env"]:
        # Docker reads the value from this process environment. The secret is
        # never placed in argv, request JSON, output, or the job-record.
        command.extend(["--env", name])
    bundle = request["spec_bundle"]
    command.extend(
        [request["workload_image"], bundle["command"], *bundle["args"]]
    )
    return command


def _write_submit_error_log(request: dict[str, Any], detail: str) -> pathlib.Path:
    log_path = action.safe_absolute_path(pathlib.Path(request["log_path"]), "action log")
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(log_path, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise ValueError("action log path is not a regular file")
        handle.write(detail.strip() or "Docker submit failed without diagnostics")
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return log_path


def submit(args: argparse.Namespace) -> dict[str, Any]:
    request_path, request = _load(args)
    _require_forwarded_environment(request)
    image_check = _run(["docker", "image", "inspect", request["workload_image"]])
    if image_check.returncode != 0:
        raise ValueError(
            "approved workload image is absent; acquire it through the approved "
            "platform step before submit"
        )
    reconciliation = action.reconcile_request(argparse.Namespace(request=request_path))
    if reconciliation["state"] == "NO_JOB_RECORD":
        job_id, job_path = _open_job(request)
    elif reconciliation["state"] == "JOB_OPENED_UNBOUND":
        job_id = reconciliation["job_id"]
        job_path = pathlib.Path(reconciliation["job_record"])
    else:
        raise ValueError(
            "request already has a bound job; use status/logs/cancel instead of submit"
        )
    action.bind_job(
        argparse.Namespace(request=request_path, job_record=job_path)
    )
    completed = _run(_docker_command(request, job_id))
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        _write_submit_error_log(request, detail)
        _mark(
            request,
            job_id,
            "ERROR",
            backend_ref=f"submit-error:docker:{job_id}",
            message="Docker submit failed before a container handle was created",
        )
        action.finalize(
            argparse.Namespace(
                request=request_path,
                job_record=job_path,
                native_exit_code=completed.returncode,
            )
        )
        raise ValueError(
            "Docker submit failed: " + detail
        )
    container_id = completed.stdout.strip()
    if not container_id:
        raise ValueError("Docker submit returned no container id")
    _mark(
        request,
        job_id,
        "RUNNING",
        backend_ref=f"docker:{container_id}",
        message="Docker container launched",
    )
    return {"job_id": job_id, "backend_ref": f"docker:{container_id}", "state": "RUNNING"}


def _bound(
    request_path: pathlib.Path, request: dict[str, Any]
) -> tuple[str, pathlib.Path, str]:
    reconciliation = action.reconcile_request(argparse.Namespace(request=request_path))
    if reconciliation["state"] not in {"BOUND", "BOUND_BACKEND_RECONCILIATION_REQUIRED"}:
        raise ValueError("request has no submitted Docker job")
    return (
        reconciliation["job_id"],
        pathlib.Path(reconciliation["job_record"]),
        reconciliation["state"],
    )


def status(args: argparse.Namespace) -> dict[str, Any]:
    request_path, request = _load(args)
    job_id, _, binding_state = _bound(request_path, request)
    completed = _run(["docker", "inspect", "--format", "{{json .State}}", job_id])
    if completed.returncode != 0:
        raise ValueError("Docker backend object is missing for the bound job")
    native = json.loads(completed.stdout)
    native_status = str(native.get("Status", "")).lower()
    exit_code = native.get("ExitCode")
    if native_status in {"created", "restarting"}:
        mapped = "PENDING"
    elif native_status in {"running", "paused"}:
        mapped = "RUNNING"
    elif native_status == "exited":
        mapped = "COMPLETE" if exit_code == 0 else "ERROR"
    elif native_status in {"dead", "removing"}:
        mapped = "ERROR"
    else:
        mapped = "UNKNOWN"
    if binding_state == "BOUND_BACKEND_RECONCILIATION_REQUIRED":
        _mark(
            request,
            job_id,
            "RUNNING",
            backend_ref=f"docker:{job_id}",
            message="reconciled deterministic Docker container",
        )
    if mapped in {"COMPLETE", "ERROR"}:
        _mark(request, job_id, mapped, message=f"Docker state={native_status}")
    return {"job_id": job_id, "state": mapped, "native_exit_code": exit_code}


def logs(args: argparse.Namespace) -> dict[str, Any]:
    request_path, request = _load(args)
    job_id, _, binding_state = _bound(request_path, request)
    if binding_state == "BOUND_BACKEND_RECONCILIATION_REQUIRED":
        _mark(
            request,
            job_id,
            "RUNNING",
            backend_ref=f"docker:{job_id}",
            message="reconciled deterministic Docker container",
        )
    log_path = action.safe_absolute_path(pathlib.Path(request["log_path"]), "action log")
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(log_path, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise ValueError("action log path is not a regular file")
        completed = subprocess.run(
            ["docker", "logs", job_id],
            check=False,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if completed.returncode != 0:
        raise ValueError("Docker log capture failed")
    return {"job_id": job_id, "log_path": str(log_path)}


def cancel(args: argparse.Namespace) -> dict[str, Any]:
    request_path, request = _load(args)
    job_id, _, binding_state = _bound(request_path, request)
    if binding_state == "BOUND_BACKEND_RECONCILIATION_REQUIRED":
        _mark(
            request,
            job_id,
            "RUNNING",
            backend_ref=f"docker:{job_id}",
            message="reconciled deterministic Docker container",
        )
    completed = _run(["docker", "rm", "--force", job_id])
    if completed.returncode != 0:
        raise ValueError(
            "Docker cancel failed: " + (completed.stderr or completed.stdout).strip()
        )
    _mark(request, job_id, "CANCELED", message="Docker container canceled")
    return {"job_id": job_id, "state": "CANCELED"}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="verb", required=True)
    for verb in ("submit", "status", "logs", "cancel"):
        child = sub.add_parser(verb)
        child.add_argument("--request", required=True, type=pathlib.Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = globals()[args.verb](args)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"run_deft_docker_action: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
