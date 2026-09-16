#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Durably gate a Docker Cosmos evaluation on a completed training job.

The process is restart-safe: it owns an advisory lock, persists its state with
atomic renames, polls Docker as the source of truth, and only opens an
evaluation job-record after the requested checkpoint has been validated.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
from pathlib import Path
import pwd
import struct
import subprocess
import sys
import time
from typing import Any


LOG = logging.getLogger("cosmos-train-eval-gate")


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    LOG.debug("running: %s", " ".join(command))
    return subprocess.run(command, check=check, text=True, capture_output=True)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.exists():
        return {} if default is None else default.copy()
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def docker_state(container: str) -> tuple[str, int]:
    result = run(
        ["docker", "inspect", "--format", "{{.State.Status}} {{.State.ExitCode}}", container],
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"cannot inspect Docker container {container}: {result.stderr.strip()}")
    status, exit_code = result.stdout.strip().split()
    return status, int(exit_code)


def docker_id(container: str) -> str | None:
    result = run(["docker", "inspect", "--format", "{{.Id}}", container], check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def record(job_helper: Path, job_id: str) -> dict[str, Any]:
    result = run([sys.executable, str(job_helper), "show", job_id])
    return json.loads(result.stdout)


def mark(
    job_helper: Path,
    job_id: str,
    state: str,
    message: str,
    *,
    backend_ref: str | None = None,
    err_class: str | None = None,
) -> None:
    current = record(job_helper, job_id)
    if current.get("terminal_state"):
        LOG.info("job %s is already terminal: %s", job_id, current["terminal_state"])
        return
    command = [
        sys.executable,
        str(job_helper),
        "mark",
        job_id,
        "--state",
        state,
        "--message",
        message,
        "--source",
        "poller",
    ]
    if backend_ref:
        command.extend(["--backend-ref", backend_ref])
    if err_class:
        command.extend(["--err-class", err_class])
    run(command)


def validate_safetensors(path: Path) -> None:
    size = path.stat().st_size
    if size < 10:
        raise ValueError(f"safetensors file is truncated: {path} ({size} bytes)")
    with path.open("rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        if header_size <= 2 or header_size > size - 8:
            raise ValueError(f"invalid safetensors header length in {path}: {header_size}")
        header = json.loads(stream.read(header_size))
    if not isinstance(header, dict) or not any(key != "__metadata__" for key in header):
        raise ValueError(f"safetensors contains no tensor index: {path}")


def validate_dense_checkpoint(checkpoint: Path) -> None:
    config_path = checkpoint / "config.json"
    config = load_json(config_path)
    if not config.get("model_type"):
        raise ValueError(f"dense checkpoint config has no model_type: {config_path}")

    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.exists():
        index = load_json(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"dense checkpoint weight_map is empty: {index_path}")
        weight_files = sorted({checkpoint / str(name) for name in weight_map.values()})
    else:
        weight_files = sorted(checkpoint.glob("*.safetensors"))
    if not weight_files:
        raise ValueError(f"dense checkpoint has no safetensors weights: {checkpoint}")
    missing = [path for path in weight_files if not path.is_file()]
    if missing:
        raise ValueError(f"dense checkpoint index references missing weights: {missing}")
    for weights_path in weight_files:
        validate_safetensors(weights_path)


def find_and_validate_checkpoint(
    train_results: Path,
    epoch: int,
    checkpoint_kind: str,
    expected_config_path: Path | None = None,
) -> Path:
    matches = sorted(train_results.glob(f"**/safetensors/epoch_{epoch}"))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one epoch_{epoch} safetensors directory below "
            f"{train_results}, found {len(matches)}: {matches}"
        )
    checkpoint = matches[0]
    if checkpoint_kind == "dense":
        validate_dense_checkpoint(checkpoint)
    else:
        if expected_config_path is None:
            raise ValueError("adapter validation requires --expected-adapter-config")
        config_path = checkpoint / "adapter_config.json"
        weights_path = checkpoint / "adapter_model.safetensors"
        actual = load_json(config_path)
        expected = load_json(expected_config_path)
        mismatches = {
            key: {"expected": value, "actual": actual.get(key)}
            for key, value in expected.items()
            if actual.get(key) != value
        }
        if mismatches:
            raise ValueError(f"adapter config mismatch: {json.dumps(mismatches, sort_keys=True)}")
        validate_safetensors(weights_path)
    LOG.info(
        "validated epoch-%d %s checkpoint at %s", epoch, checkpoint_kind, checkpoint
    )
    return checkpoint


def require_structured_success(status_path: Path) -> None:
    if not status_path.is_file():
        raise ValueError(f"training structured status file is missing: {status_path}")
    terminal: str | None = None
    with status_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid structured status JSON at {status_path}:{line_number}: {exc}"
                ) from exc
            status = event.get("status") if isinstance(event, dict) else None
            if status in {"SUCCESS", "FAILURE"}:
                terminal = status
    if terminal != "SUCCESS":
        raise ValueError(
            f"training structured terminal status is {terminal or 'missing'}, expected SUCCESS"
        )


def open_eval_record(args: argparse.Namespace) -> tuple[str, Path]:
    command = [
        sys.executable,
        str(args.job_helper),
        "open",
        "--platform",
        "docker",
        "--image",
        args.image,
        "--network-arch",
        "cosmos-reason",
        "--action",
        "evaluate",
        "--storage-tier",
        "A",
        "--results-root",
        str(args.eval_results_root),
        "--parent-job",
        args.train_job,
    ]
    result = run(command)
    job_id = result.stdout.strip().splitlines()[-1]
    job_record = record(args.job_helper, job_id)
    return job_id, Path(job_record["results_dir"])


def launch_eval(
    args: argparse.Namespace, job_id: str, results_dir: Path, checkpoint: Path
) -> str:
    for directory in (
        results_dir,
        results_dir / ".tao-runtime" / "home",
        results_dir / ".cache",
        results_dir / "tmp",
        args.eval_cache,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    lint = run(
        [sys.executable, str(args.redactor), "lint", str(args.eval_spec)], check=False
    )
    if lint.returncode:
        raise RuntimeError(f"evaluation spec failed secret lint: {lint.stdout}{lint.stderr}")
    command = [
        "docker",
        "run",
        "--detach",
        "--name",
        job_id,
        "--gpus",
        f"device={args.gpu}",
        "--network",
        "host",
        "--ipc",
        "host",
        "--shm-size",
        args.shm_size,
        "--user",
        args.user,
    ]
    for group_id in args.group_add:
        command.extend(["--group-add", group_id])
    command.extend([
        "--env",
        "HOME=/results/.tao-runtime/home",
        "--env",
        "COSMOS_CACHE=/cache",
        "--env",
        f"USER={args.runtime_user}",
        "--env",
        f"LOGNAME={args.runtime_user}",
        "--env",
        "XDG_CACHE_HOME=/results/.cache",
        "--env",
        "TMPDIR=/results/tmp",
        "--env",
        f"TAO_API_JOB_ID={job_id}",
        "--env",
        "COSMOS_VIDEO_CACHE_ITEMS=16",
        "--env",
        "PYTHONUNBUFFERED=1",
        "--mount",
        f"type=bind,src={args.eval_cache},dst=/cache",
        "--mount",
        f"type=bind,src={args.dataset_root},dst={args.dataset_mount_destination},readonly",
        "--mount",
        f"type=bind,src={args.eval_spec.parent},dst=/specs,readonly",
        "--mount",
        f"type=bind,src={checkpoint},dst=/adapter,readonly",
        "--mount",
        f"type=bind,src={args.base_model},dst=/base-model,readonly",
        "--mount",
        f"type=bind,src={results_dir},dst=/results",
        args.image,
        "cosmos-rl-evaluate",
        "--config",
        f"/specs/{args.eval_spec.name}",
    ])
    result = run(command)
    container_id = result.stdout.strip()
    mark(
        args.job_helper,
        job_id,
        "RUNNING",
        "Docker evaluation container launched by durable train/eval gate",
        backend_ref=container_id,
    )
    return container_id


def find_results(results_dir: Path) -> dict[str, Any]:
    matches = sorted(results_dir.glob("evaluation/**/results.json"))
    if len(matches) != 1:
        raise ValueError(f"expected one evaluation results.json, found {len(matches)}: {matches}")
    with matches[0].open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if isinstance(payload, dict):
        result = payload.copy()
    elif isinstance(payload, list):
        if not payload or not all(isinstance(item, dict) for item in payload):
            raise ValueError(f"evaluation result list is empty or malformed: {matches[0]}")
        if not all("response" in item and "gt" in item for item in payload):
            raise ValueError(f"evaluation result rows lack response/gt fields: {matches[0]}")
        correct = sum(
            str(item["response"]).strip() == str(item["gt"]).strip() for item in payload
        )
        result = {
            "accuracy": correct / len(payload),
            "correct_samples": correct,
            "total_samples": len(payload),
        }
    else:
        raise ValueError(f"unsupported evaluation results shape in {matches[0]}")
    result["results_file"] = str(matches[0])
    return result


def save_summary(
    args: argparse.Namespace,
    state: dict[str, Any],
    workflow_state: str,
    error: str | None = None,
) -> None:
    summary: dict[str, Any] = {
        "workflow_state": workflow_state,
        "train_job": args.train_job,
        "train_container": args.train_container,
        "checkpoint": state.get("checkpoint"),
        "evaluation_job": state.get("evaluation_job"),
        "evaluation_container": state.get("evaluation_container"),
        "evaluation_results_dir": state.get("evaluation_results_dir"),
        "evaluation_results": state.get("evaluation_results"),
    }
    if error:
        summary["error"] = error
    atomic_json(args.summary_file, summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-job", required=True)
    parser.add_argument("--train-container", required=True)
    parser.add_argument("--train-results", required=True, type=Path)
    parser.add_argument("--checkpoint-epoch", required=True, type=int)
    parser.add_argument(
        "--checkpoint-kind", choices=("adapter", "dense"), default="adapter"
    )
    parser.add_argument("--expected-adapter-config", type=Path)
    parser.add_argument("--train-status-file", type=Path)
    parser.add_argument("--eval-spec", required=True, type=Path)
    parser.add_argument("--eval-results-root", required=True, type=Path)
    parser.add_argument("--eval-cache", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument(
        "--dataset-mount-destination",
        type=Path,
        help="container destination for --dataset-root (default: same absolute path)",
    )
    parser.add_argument("--base-model", required=True, type=Path)
    parser.add_argument("--image", required=True)
    parser.add_argument("--job-helper", required=True, type=Path)
    parser.add_argument("--redactor", required=True, type=Path)
    parser.add_argument("--state-file", required=True, type=Path)
    parser.add_argument("--summary-file", required=True, type=Path)
    parser.add_argument("--log-file", required=True, type=Path)
    parser.add_argument("--pid-file", required=True, type=Path)
    parser.add_argument("--lock-file", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--shm-size", default="16g")
    parser.add_argument(
        "--user",
        default=f"{os.getuid()}:{os.getgid()}",
        help="container UID:GID (default: invoking process UID:GID)",
    )
    parser.add_argument(
        "--runtime-user",
        default=pwd.getpwuid(os.getuid()).pw_name,
        help="USER/LOGNAME inside the container (default: invoking account)",
    )
    parser.add_argument(
        "--group-add",
        action="append",
        default=[],
        metavar="GID",
        help="supplementary container group; repeat for video/render device groups",
    )
    args = parser.parse_args()
    if args.dataset_mount_destination is None:
        args.dataset_mount_destination = args.dataset_root
    return args


def main() -> int:
    args = parse_args()
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log_file), logging.StreamHandler()],
    )
    args.lock_file.parent.mkdir(parents=True, exist_ok=True)
    lock_stream = args.lock_file.open("a+")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        LOG.info("another gate process already owns %s", args.lock_file)
        return 0
    atomic_json(args.pid_file, {"pid": os.getpid()})
    state = load_json(args.state_file)
    try:
        while "checkpoint" not in state:
            status, exit_code = docker_state(args.train_container)
            LOG.info("training backend state=%s exit_code=%d", status, exit_code)
            if status in {"created", "running", "restarting", "paused"}:
                time.sleep(args.poll_seconds)
                continue
            if status != "exited" or exit_code != 0:
                message = f"training Docker container ended state={status} exit_code={exit_code}"
                mark(
                    args.job_helper,
                    args.train_job,
                    "ERROR",
                    message,
                    err_class="ERR_PROGRAM" if status == "exited" else "ERR_INFRA",
                )
                save_summary(args, state, "ERROR", message)
                return 1
            if args.train_status_file is not None:
                require_structured_success(args.train_status_file)
            checkpoint = find_and_validate_checkpoint(
                args.train_results,
                args.checkpoint_epoch,
                args.checkpoint_kind,
                args.expected_adapter_config,
            )
            mark(
                args.job_helper,
                args.train_job,
                "COMPLETE",
                "training Docker container exited successfully with validated structured status and checkpoint",
            )
            state["checkpoint"] = str(checkpoint)
            atomic_json(args.state_file, state)

        if "evaluation_job" not in state:
            job_id, results_dir = open_eval_record(args)
            state.update(
                evaluation_job=job_id,
                evaluation_results_dir=str(results_dir),
            )
            atomic_json(args.state_file, state)
            container_id = launch_eval(args, job_id, results_dir, Path(state["checkpoint"]))
            state["evaluation_container"] = container_id
            atomic_json(args.state_file, state)

        eval_job = state["evaluation_job"]
        eval_results_dir = Path(state["evaluation_results_dir"])
        eval_container = state.get("evaluation_container")
        if not eval_container:
            eval_record = record(args.job_helper, eval_job)
            eval_container = eval_record.get("backend_ref")
            if not eval_container:
                eval_container = docker_id(eval_job)
            if not eval_container:
                eval_container = launch_eval(
                    args, eval_job, eval_results_dir, Path(state["checkpoint"])
                )
            else:
                mark(
                    args.job_helper,
                    eval_job,
                    "RUNNING",
                    "durable train/eval gate recovered existing Docker evaluation container",
                    backend_ref=eval_container,
                )
            state["evaluation_container"] = eval_container
            atomic_json(args.state_file, state)

        while True:
            status, exit_code = docker_state(eval_container)
            LOG.info("evaluation backend state=%s exit_code=%d", status, exit_code)
            if status in {"created", "running", "restarting", "paused"}:
                time.sleep(args.poll_seconds)
                continue
            if status == "exited" and exit_code == 0:
                results = find_results(eval_results_dir)
                state["evaluation_results"] = results
                atomic_json(args.state_file, state)
                mark(
                    args.job_helper,
                    eval_job,
                    "COMPLETE",
                    "evaluation Docker container exited successfully and results.json was validated",
                )
                save_summary(args, state, "COMPLETE")
                LOG.info("workflow completed successfully")
                return 0
            message = f"evaluation Docker container ended state={status} exit_code={exit_code}"
            mark(
                args.job_helper,
                eval_job,
                "ERROR",
                message,
                err_class="ERR_PROGRAM" if status == "exited" else "ERR_INFRA",
            )
            save_summary(args, state, "ERROR", message)
            return 1
    except Exception as exc:
        LOG.exception("workflow gate failed")
        if "checkpoint" not in state:
            mark(
                args.job_helper,
                args.train_job,
                "ERROR",
                f"training gate validation failed: {exc}",
                err_class="ERR_PROGRAM",
            )
        elif state.get("evaluation_job"):
            mark(
                args.job_helper,
                state["evaluation_job"],
                "ERROR",
                f"evaluation gate validation failed: {exc}",
                err_class="ERR_PROGRAM",
            )
        save_summary(args, state, "ERROR", str(exc))
        return 1
    finally:
        args.pid_file.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
