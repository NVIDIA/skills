# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Legacy schema-v1 Docker adapter for existing PAS DEFT runs.

New runs use ``run_deft_action.py`` and a selected platform consumer. This
compatibility wrapper reconstructs every value from ``deft_state.json`` on each call;
it never relies on a previous shell export.  It records the Docker exit code
and log path atomically and never places credential values in argv or status.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import getpass
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import time
from typing import Any

from command_contract import (
    command_sha256,
    derived_spec_evidence,
    expected_container_command,
    expected_hf_forwarding,
    expected_fresh_outputs,
    expected_image_kind,
    expected_stage_directory,
    validate_content_bound_outputs,
)


RUN_SPEC_NAMES = (
    "deft_config.yaml",
    "tao_spec.yaml",
    "text_embed_spec.yaml",
    "image_embed_spec.yaml",
    "mining_spec.yaml",
    "approval.json",
)
PINNED_IMAGES = {
    "pyt": "nvcr.io/nvidia/tao/tao-toolkit:7.2.0-pyt",  # versions-key: images.tao_toolkit.deft_pas_pyt
    "ds": "nvcr.io/nvidia/tao/tao-toolkit:7.2.0-data-services",  # versions-key: images.tao_toolkit.deft_pas_data_services
}


def _atomic_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _under(path: pathlib.Path, root: pathlib.Path, name: str) -> pathlib.Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{name} must be under {root}: {resolved}") from exc
    return resolved


def _fresh_output_path(
    path: pathlib.Path, root: pathlib.Path, name: str
) -> pathlib.Path:
    """Resolve an output lexically so unlink never follows a final symlink."""
    raw = path.expanduser()
    if not raw.is_absolute():
        raw = pathlib.Path.cwd() / raw
    absolute = pathlib.Path(os.path.abspath(raw))
    try:
        absolute.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{name} must be under {root}: {absolute}") from exc
    parent = absolute.parent
    try:
        parent.resolve().relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{name} parent must remain under {root}: {parent}") from exc
    if parent.resolve() != parent:
        raise ValueError(f"{name} parent must not traverse a symlink: {parent}")
    if absolute.is_symlink():
        raise ValueError(f"{name} must not be a symlink: {absolute}")
    return absolute


def _load_state(results_dir: pathlib.Path) -> dict[str, Any]:
    path = results_dir / "deft_state.json"
    if not path.is_file():
        raise ValueError(f"state file not found: {path}")
    payload = json.loads(path.read_text())
    if (
        not isinstance(payload, dict)
        or payload.get("workflow") != "tao-run-deft-pas"
        or payload.get("schema_version") != "3"
    ):
        raise ValueError(f"invalid PAS DEFT state: {path}")
    if pathlib.Path(str(payload.get("results_dir", ""))).resolve() != results_dir:
        raise ValueError("state.results_dir does not match --results-dir")
    return payload


def _workspace_child(path: pathlib.Path, workspace: pathlib.Path, name: str) -> pathlib.Path:
    resolved = path.expanduser().resolve()
    try:
        relative = resolved.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(f"{name} must be under workspace {workspace}: {resolved}") from exc
    if relative == pathlib.Path("."):
        raise ValueError(f"{name} must be a child of workspace, not workspace itself")
    return resolved


def _validated_runtime_paths(
    results_dir: pathlib.Path, config: dict[str, Any]
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    workspace = pathlib.Path(str(config.get("workspace", ""))).expanduser().resolve()
    if not workspace.is_dir() or workspace == pathlib.Path(workspace.anchor):
        raise ValueError(f"state workspace must be an existing non-root directory: {workspace}")
    _workspace_child(results_dir, workspace, "state results_dir")
    dataset_root = _workspace_child(
        pathlib.Path(str(config.get("dataset_root", ""))), workspace, "state dataset_root"
    )
    if not dataset_root.is_dir():
        raise ValueError(f"state dataset_root is not an existing directory: {dataset_root}")
    if results_dir in dataset_root.parents or dataset_root in results_dir.parents:
        raise ValueError("state results_dir and dataset_root must not contain one another")
    if dataset_root.parent == workspace:
        raise ValueError("state dataset_root must be nested below a workspace data directory")
    if dataset_root.parent == pathlib.Path(dataset_root.anchor):
        raise ValueError("refusing to mount a filesystem root as the dataset parent")

    config_dir = pathlib.Path(str(config.get("config_dir", ""))).expanduser().resolve()
    expected_config_dir = results_dir / "config"
    if config_dir != expected_config_dir or not config_dir.is_dir():
        raise ValueError(
            f"state config_dir must be the existing run config directory {expected_config_dir}"
        )
    expected_hashes = config.get("spec_sha256")
    if not isinstance(expected_hashes, dict) or set(expected_hashes) != set(RUN_SPEC_NAMES):
        raise ValueError("state config.spec_sha256 must bind all immutable run config files")
    for name in RUN_SPEC_NAMES:
        path = config_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"run spec is missing or empty: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected_hashes.get(name) != actual:
            raise ValueError(f"run spec changed after approval: {path}")
    if pathlib.Path(str(config.get("deft_config", ""))).resolve() != config_dir / "deft_config.yaml":
        raise ValueError("state deft_config path does not match the immutable run config")
    if pathlib.Path(str(config.get("tao_spec", ""))).resolve() != config_dir / "tao_spec.yaml":
        raise ValueError("state tao_spec path does not match the immutable run config")
    if config.get("platform") != "docker":
        raise ValueError("state platform must be docker")
    return workspace, dataset_root, config_dir


def _reject_sensitive_argv(command: list[str]) -> None:
    secret_values = [
        value for name in ("NGC_KEY", "HF_TOKEN")
        if (value := os.environ.get(name))
    ]
    sensitive_markers = ("password=", "token=", "api_key=", "apikey=")
    sensitive_flags = {
        "--password",
        "--password-stdin",
        "--token",
        "--api-key",
        "--api_key",
        "--apikey",
        "-p",
    }
    for item in command:
        lowered = item.lower()
        if lowered in sensitive_flags or any(marker in lowered for marker in sensitive_markers):
            raise ValueError(
                "container command must not carry credentials in argv; use "
                "the wrapper's explicit environment forwarding option"
            )
        if any(secret in item for secret in secret_values):
            raise ValueError("container command argv contains a credential value")


def _docker_gpu_args(config: dict[str, Any]) -> list[str]:
    """Return the exact immutable Docker device request from approved state."""
    gpu_ids = config.get("gpu_ids")
    num_gpus = config.get("num_gpus")
    valid_ids = (
        isinstance(gpu_ids, list)
        and bool(gpu_ids)
        and all(
            isinstance(gpu_id, int)
            and not isinstance(gpu_id, bool)
            and gpu_id >= 0
            for gpu_id in gpu_ids
        )
        and len(set(gpu_ids)) == len(gpu_ids)
    )
    if not valid_ids or num_gpus != len(gpu_ids):
        raise ValueError(
            "state.config.gpu_ids must be a non-empty unique integer list "
            "matching state.config.num_gpus"
        )
    device_ids = ",".join(str(gpu_id) for gpu_id in gpu_ids)
    # Docker parses --gpus as CSV. Literal double quotes keep a multi-device
    # selector together when argv is passed directly without a shell.
    return ["--gpus", f'"device={device_ids}"']


def _container_is_running(container_name: str) -> bool:
    """Read Docker state without mutating a possibly orphaned container."""
    inspected = subprocess.run(
        [
            "docker",
            "container",
            "inspect",
            "--format",
            "{{.State.Running}}",
            container_name,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if inspected.returncode == 0:
        value = inspected.stdout.strip().lower()
        if value not in {"true", "false"}:
            raise ValueError(
                f"Docker returned an unrecognized state for {container_name}: {value!r}"
            )
        return value == "true"
    combined = (inspected.stdout + inspected.stderr).lower()
    if "no such object" in combined or "no such container" in combined:
        return False
    raise ValueError(
        f"cannot inspect prior container {container_name}; Docker said: "
        f"{(inspected.stderr or inspected.stdout).strip() or 'unknown error'}"
    )


def _container_exit_code(container_name: str) -> int | None:
    """Return a retained container's exit code, or None after Docker --rm."""
    inspected = subprocess.run(
        [
            "docker",
            "container",
            "inspect",
            "--format",
            "{{.State.ExitCode}}",
            container_name,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if inspected.returncode == 0:
        try:
            return int(inspected.stdout.strip())
        except ValueError as exc:
            raise ValueError(
                f"Docker returned an invalid exit code for {container_name}: "
                f"{inspected.stdout.strip()!r}"
            ) from exc
    combined = (inspected.stdout + inspected.stderr).lower()
    if "no such object" in combined or "no such container" in combined:
        return None
    raise ValueError(
        f"cannot inspect prior container {container_name}; Docker said: "
        f"{(inspected.stderr or inspected.stdout).strip() or 'unknown error'}"
    )


def _last_execution_status(log_path: pathlib.Path) -> str | None:
    """Stream the complete log and return its final TAO status marker.

    Reconciliation is an exceptional recovery path, so a linear scan is a
    better tradeoff than a fixed tail: it uses bounded memory and cannot lose
    durable success evidence merely because later diagnostics made the log
    larger than an arbitrary window.
    """
    final_status = None
    pattern = re.compile(r"^Execution status:\s*(PASS|FAIL)\s*$", re.IGNORECASE)
    with log_path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = pattern.match(line)
            if match:
                final_status = match.group(1).upper()
    return final_status


def _validate_interrupted_identity(
    existing: dict[str, Any],
    *,
    name: str,
    image_kind: str,
    image: str,
    command: list[str],
    passed_hf_token: bool,
    container_name: str,
    cidfile_path: pathlib.Path,
    log_path: pathlib.Path,
    fresh_outputs: list[str],
) -> None:
    expected = {
        "schema_version": "1",
        "workflow": "tao-run-deft-pas",
        "kind": "container",
        "name": name,
        "image_kind": image_kind,
        "image": image,
        "command": command,
        "command_sha256": command_sha256(command),
        "passed_hf_token": passed_hf_token,
        "container_name": container_name,
        "cidfile": str(cidfile_path),
        "log_path": str(log_path),
        "fresh_outputs": fresh_outputs,
    }
    mismatches = [key for key, value in expected.items() if existing.get(key) != value]
    if mismatches:
        raise ValueError(
            "existing running status does not match the approved command identity: "
            + ", ".join(mismatches)
        )


def _reconcile_interrupted_success(
    existing: dict[str, Any],
    *,
    status_path: pathlib.Path,
    log_path: pathlib.Path,
    fresh_outputs: list[str],
    container_name: str,
) -> bool:
    """Close a status left running only when durable success evidence agrees."""
    started_ns = existing.get("started_ns")
    if (
        not isinstance(started_ns, int)
        or isinstance(started_ns, bool)
        or started_ns < 1
        or not log_path.is_file()
        or log_path.stat().st_size == 0
        or log_path.is_symlink()
        or log_path.resolve() != log_path
    ):
        return False
    for raw in fresh_outputs:
        output = pathlib.Path(raw)
        if (
            output.is_symlink()
            or not output.is_file()
            or output.stat().st_size == 0
            or output.stat().st_mtime_ns < started_ns
        ):
            return False

    docker_exit_code = _container_exit_code(container_name)
    log_status = _last_execution_status(log_path)
    if docker_exit_code is not None:
        if docker_exit_code != 0 or log_status == "FAIL":
            return False
        source = "docker"
    else:
        if log_status != "PASS":
            return False
        docker_exit_code = 0
        source = "container_log"

    finished_at = dt.datetime.fromtimestamp(
        log_path.stat().st_mtime, tz=dt.timezone.utc
    ).isoformat(timespec="seconds")
    _atomic_json(
        status_path,
        {
            **existing,
            "finished_at": finished_at,
            "status": "ok",
            "exit_code": 0,
            "docker_exit_code": docker_exit_code,
            "artifact_error": None,
            "reconciled_after_wrapper_exit": True,
            "reconciliation_source": source,
        },
    )
    return True


def _launch_label(name: str, stage_dir: pathlib.Path, results_dir: pathlib.Path) -> str:
    if name == "pool_embed":
        return "baseline"
    relative = stage_dir.relative_to(results_dir)
    if relative.parts and relative.parts[0] == "zs":
        return "baseline"
    if relative.parts:
        match = re.fullmatch(r"iter_([1-9][0-9]*)", relative.parts[0])
        if match:
            return f"iter{match.group(1)}"
    raise ValueError(
        f"cannot derive baseline/iterN command scope from --stage-dir {stage_dir}"
    )


def _load_existing_status(path: pathlib.Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"existing command status is invalid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"existing command status root must be an object: {path}")
    return payload


def _validate_derived_spec_input(
    name: str, label: str, results_dir: pathlib.Path
) -> dict[str, str]:
    evidence = derived_spec_evidence(name, label, results_dir)
    if evidence is None:
        return {}
    spec_path, status_path, producer = evidence
    payload = _load_existing_status(status_path)
    if (
        payload is None
        or payload.get("schema_version") != "1"
        or payload.get("workflow") != "tao-run-deft-pas"
        or payload.get("kind") != "host"
        or payload.get("name") != producer
        or payload.get("status") != "ok"
        or payload.get("exit_code") != 0
        or str(spec_path) not in payload.get("fresh_outputs", [])
    ):
        raise ValueError(
            f"{name} requires successful content-bound {producer} evidence: {status_path}"
        )
    return validate_content_bound_outputs(payload, [spec_path], str(status_path))


def run(args: argparse.Namespace) -> tuple[pathlib.Path, pathlib.Path, int]:
    results_dir = args.results_dir.expanduser().resolve()
    state = _load_state(results_dir)
    config = state.get("config")
    if not isinstance(config, dict):
        raise ValueError("state.config must be an object")
    image_key = "pyt_image" if args.image == "pyt" else "ds_image"
    image = str(config.get(image_key, "")).strip()
    if image != PINNED_IMAGES[args.image]:
        raise ValueError(
            f"state.config.{image_key} must be the pinned {args.image} image"
        )
    workspace, dataset_root, config_dir = _validated_runtime_paths(
        results_dir, config
    )
    patches_dir = pathlib.Path(__file__).resolve().parent.parent / "patches"
    if not patches_dir.is_dir():
        raise ValueError(f"container compatibility patches are missing: {patches_dir}")

    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", args.name):
        raise ValueError("--name must contain only lowercase letters, digits, ._- characters")
    _reject_sensitive_argv(args.command)
    stage_dir = _under(args.stage_dir, results_dir, "--stage-dir")
    label = _launch_label(args.name, stage_dir, results_dir)
    if label.startswith("iter"):
        number = int(label[4:])
        maximum = state.get("max_iterations")
        if (
            not isinstance(maximum, int)
            or isinstance(maximum, bool)
            or not 1 <= number <= maximum
        ):
            raise ValueError(
                f"container iteration {number} is outside approved range 1..{maximum}"
            )
    expected_stage = expected_stage_directory(args.name, label, results_dir)
    if stage_dir != expected_stage:
        raise ValueError(
            f"--stage-dir for {args.name} must be {expected_stage}, got {stage_dir}"
        )
    expected_command = expected_container_command(args.name, label, config)
    if args.command != expected_command:
        raise ValueError(
            f"container argv for {args.name} does not match the immutable workflow command"
        )
    expected_kind = expected_image_kind(args.name)
    if args.image != expected_kind:
        raise ValueError(
            f"--image for {args.name} must be {expected_kind}, got {args.image}"
        )
    required_hf = expected_hf_forwarding(args.name, config)
    if bool(args.pass_hf_token) != required_hf:
        raise ValueError(
            f"--pass-hf-token for {args.name} must be {required_hf} per immutable approval"
        )
    input_sha256 = _validate_derived_spec_input(
        args.name, label, results_dir
    )
    stage_dir.mkdir(parents=True, exist_ok=True)
    log_path = stage_dir / f"{args.name}.log"
    status_path = stage_dir / f"{args.name}.status.json"
    cidfile_path = stage_dir / f"{args.name}.cid"
    lock_path = stage_dir / f"{args.name}.launch.lock"
    identity = hashlib.sha256(
        f"{results_dir}\0{stage_dir.relative_to(results_dir)}\0{args.name}".encode()
    ).hexdigest()[:20]
    container_name = f"tao-deft-{identity}"

    fresh_outputs: list[str] = []
    for raw in args.fresh_output:
        output = _fresh_output_path(raw, results_dir, "--fresh-output")
        if output.exists() and not output.is_file():
            raise ValueError(f"--fresh-output must name a file, not a directory: {output}")
        fresh_outputs.append(str(output))
    expected_outputs = [
        str(path) for path in expected_fresh_outputs(args.name, label, results_dir)
    ]
    if fresh_outputs != expected_outputs:
        raise ValueError(
            f"--fresh-output for {args.name} must be {expected_outputs}, "
            f"got {fresh_outputs}"
        )

    cache_dir = workspace / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_mount = dataset_root.parent
    command = [
        "docker",
        "run",
        *_docker_gpu_args(config),
        "--rm",
        "--name",
        container_name,
        "--cidfile",
        str(cidfile_path),
        "--shm-size=8g",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--env",
        f"USER={getpass.getuser()}",
        "--env",
        f"LOGNAME={getpass.getuser()}",
        "--env",
        "HOME=/tmp",
        "--env",
        "PYTHONPATH=/patches",
        "--env",
        "HF_HOME=/cache/huggingface",
        "--env",
        "XDG_CACHE_HOME=/cache",
        "--volume",
        f"{results_dir}:/results",
        "--volume",
        f"{data_mount}:/data:ro",
        "--volume",
        f"{data_mount}:{data_mount}:ro",
        "--volume",
        f"{config_dir}:/specs:ro",
        "--volume",
        f"{patches_dir}:/patches:ro",
        "--volume",
        f"{cache_dir}:/cache",
    ]
    # ``--env HF_TOKEN`` copies the already-exported value without exposing it
    # in the process list, status JSON, or transcript.  Public models need none.
    if args.pass_hf_token:
        if not os.environ.get("HF_TOKEN"):
            raise ValueError("--pass-hf-token requires HF_TOKEN in the process environment")
        command.extend(["--env", "HF_TOKEN"])
    command.extend([image, *args.command])

    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(
                f"another wrapper process owns this stage launch: {lock_path}"
            ) from exc

        existing = _load_existing_status(status_path)
        prior_attempt = 0
        lineage_started_ns = None
        if existing is not None:
            raw_attempt = existing.get("attempt", 1)
            if (
                not isinstance(raw_attempt, int)
                or isinstance(raw_attempt, bool)
                or raw_attempt < 1
            ):
                raise ValueError(f"existing command status has invalid attempt: {status_path}")
            prior_attempt = raw_attempt
            lineage_started_ns = existing.get(
                "lineage_started_ns", existing.get("started_ns")
            )
            if (
                not isinstance(lineage_started_ns, int)
                or isinstance(lineage_started_ns, bool)
                or lineage_started_ns < 1
            ):
                raise ValueError(
                    f"existing command status has invalid attempt lineage: {status_path}"
                )
        if existing is not None and existing.get("status") == "running":
            prior_name = existing.get("container_name")
            if prior_name != container_name:
                raise ValueError(
                    "existing running status lacks the deterministic container identity; "
                    f"inspect and recover manually before replacing {status_path}"
                )
            _validate_interrupted_identity(
                existing,
                name=args.name,
                image_kind=args.image,
                image=image,
                command=list(args.command),
                passed_hf_token=bool(args.pass_hf_token),
                container_name=container_name,
                cidfile_path=cidfile_path,
                log_path=log_path,
                fresh_outputs=fresh_outputs,
            )
            if _container_is_running(container_name):
                raise ValueError(
                    f"prior container {container_name} is still running; wait for it to "
                    "exit, then rerun the same command (the wrapper never auto-kills it)"
                )
            if _reconcile_interrupted_success(
                existing,
                status_path=status_path,
                log_path=log_path,
                fresh_outputs=fresh_outputs,
                container_name=container_name,
            ):
                return status_path, log_path, 0
        if prior_attempt >= 2:
            raise ValueError(
                f"attempt budget exhausted for {args.name} (attempt={prior_attempt}); "
                "commit a terminal stage error and hard-stop instead of retrying"
            )

        try:
            cidfile_path.unlink()
        except FileNotFoundError:
            pass
        started_ns = time.time_ns()
        if lineage_started_ns is None:
            lineage_started_ns = started_ns
        started_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        # Invalidate any previous successful attempt *before* deleting outputs
        # or launching Docker. A deterministic container name plus the launch
        # lock prevents two attempts from concurrently writing those outputs.
        running_payload = {
            "schema_version": "1",
            "workflow": "tao-run-deft-pas",
            "kind": "container",
            "name": args.name,
            "attempt": prior_attempt + 1,
            "image_kind": args.image,
            "image": image,
            "command": list(args.command),
            "command_sha256": command_sha256(list(args.command)),
            "passed_hf_token": bool(args.pass_hf_token),
            "container_name": container_name,
            "cidfile": str(cidfile_path),
            "started_at": started_at,
            "started_ns": started_ns,
            "lineage_started_ns": lineage_started_ns,
            "finished_at": None,
            "status": "running",
            "exit_code": None,
            "log_path": str(log_path),
            "fresh_outputs": fresh_outputs,
            "input_sha256": input_sha256,
        }
        _atomic_json(status_path, running_payload)
        for raw in fresh_outputs:
            try:
                pathlib.Path(raw).unlink()
            except FileNotFoundError:
                pass
        with log_path.open("wb") as log:
            completed = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, check=False
            )
        returncode = completed.returncode
        artifact_error = None
        if returncode == 0:
            for raw in fresh_outputs:
                output = pathlib.Path(raw)
                if (
                    output.is_symlink()
                    or not output.is_file()
                    or output.stat().st_size == 0
                ):
                    artifact_error = f"fresh output is missing, empty, or a symlink: {output}"
                    break
                if output.stat().st_mtime_ns < started_ns:
                    artifact_error = f"fresh output predates this launch: {output}"
                    break
        if artifact_error is not None:
            returncode = 3
            with log_path.open("ab") as log:
                log.write(
                    ("\nrun_deft_container: " + artifact_error + "\n").encode("utf-8")
                )
        finished_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        payload = {
            **running_payload,
            "finished_at": finished_at,
            "status": "ok" if returncode == 0 else "error",
            "exit_code": returncode,
            "docker_exit_code": completed.returncode,
            "artifact_error": artifact_error,
        }
        _atomic_json(status_path, payload)
        return status_path, log_path, returncode


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=pathlib.Path)
    parser.add_argument("--image", required=True, choices=("pyt", "ds"))
    parser.add_argument("--stage-dir", required=True, type=pathlib.Path)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--pass-hf-token",
        action="store_true",
        help="Forward the existing HF_TOKEN by environment name; off by default.",
    )
    parser.add_argument(
        "--fresh-output",
        action="append",
        default=[],
        type=pathlib.Path,
        help="Delete this exact results-scoped file before launch; repeatable.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        print("run_deft_container: command after -- is required", file=sys.stderr)
        return 2
    try:
        status, log, returncode = run(args)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"run_deft_container: {exc}", file=sys.stderr)
        return 2
    outcome = "ok" if returncode == 0 else "error"
    print(
        f"container={outcome} exit_code={returncode} status={status} log={log}",
        file=sys.stderr,
    )
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
