#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare and finalize one platform-dispatched PAS DEFT TAO action.

``prepare`` validates the exact immutable action, writes a schema-valid
platform-neutral spec bundle plus a concrete staging/mount contract, and
invalidates stale outputs. The selected TAO platform skill then owns the
native submit/status/logs/cancel lifecycle and job-record. ``finalize`` binds
that terminal job-record and native exit status to fresh workflow outputs.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
import pathlib
import stat
import subprocess
import sys
import time
import urllib.parse
from typing import Any

import jsonschema

from command_contract import command_sha256
from deft_action_contract import (
    WORKFLOW,
    ActionContext,
    atomic_json,
    load_existing_status,
    safe_absolute_path,
    validate_action,
)


REQUEST_SCHEMA_VERSION = "1"
STATUS_SCHEMA_VERSION = "2"
REMOTE_PLATFORMS = frozenset({"slurm", "kubernetes", "brev"})
JOB_IDENTITY_FIELDS = (
    "schema_version",
    "id",
    "platform",
    "image",
    "network_arch",
    "action",
    "results_dir",
    "storage_tier",
    "upload_excludes",
    "submitted_at",
)

# Only actions that dereference image paths need the dataset tree.  Text-only
# embedding and k-NN actions operate entirely on run-owned parquet artifacts.
# Keeping this decision in the producer makes every remote consumer stage the
# same minimal, truthful input set instead of relying on platform-specific
# exclusions.
DATASET_ACTIONS = frozenset(
    {
        "evaluate",
        "train",
        "viz_weak_embed",
        "viz_mined_embed",
        "viz_previous_embed",
    }
)


def _artifact_schema(name: str) -> dict[str, Any]:
    path = (
        pathlib.Path(__file__).resolve().parents[3]
        / "core"
        / "tao-artifacts"
        / "references"
        / name
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _reference_schema(name: str) -> dict[str, Any]:
    path = pathlib.Path(__file__).resolve().parent.parent / "references" / name
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_json(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _aware_timestamp(value: Any, name: str) -> dt.datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty timezone-aware timestamp")
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} is not a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone offset")
    return parsed.astimezone(dt.timezone.utc)


def _relative_output(path: pathlib.Path, root: pathlib.Path) -> str:
    return path.relative_to(root).as_posix()


def _attempt_path(
    stage_dir: pathlib.Path, name: str, attempt: int, suffix: str
) -> pathlib.Path:
    if attempt not in {1, 2}:
        raise ValueError(f"action attempt must be 1 or 2, got {attempt!r}")
    infix = "" if attempt == 1 else ".attempt-2"
    return stage_dir / f"{name}{infix}.{suffix}"


def _request_path_for(context: ActionContext, attempt: int) -> pathlib.Path:
    return _attempt_path(context.stage_dir, context.name, attempt, "action.json")


def _request_path_from_payload(request: dict[str, Any]) -> pathlib.Path:
    return _attempt_path(
        pathlib.Path(request["stage_dir"]),
        request["name"],
        request["attempt"],
        "action.json",
    )


def _runtime_dir_for(context: ActionContext, attempt: int) -> pathlib.Path:
    return (
        context.stage_dir
        / ".tao-runtime"
        / f"{context.name}.attempt-{attempt}"
    )


def _job_state_dir() -> pathlib.Path:
    raw = os.environ.get("TAO_STATE_DIR")
    if raw is not None and not raw.strip():
        raise ValueError("TAO_STATE_DIR is set but empty")
    path = pathlib.Path(raw) if raw is not None else pathlib.Path.home() / ".tao"
    return safe_absolute_path(path, "TAO job state directory")


@contextlib.contextmanager
def _exclusive_lock(path: pathlib.Path):
    path = safe_absolute_path(path, "action launch lock")
    try:
        fd = os.open(
            path,
            os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except OSError as exc:
        raise ValueError(f"cannot open safe action launch lock {path}: {exc}") from exc
    with os.fdopen(fd, "a+", encoding="utf-8") as lock:
        if not stat.S_ISREG(os.fstat(lock.fileno()).st_mode):
            raise ValueError(f"action launch lock is not a regular file: {path}")
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"another process owns this action launch: {path}") from exc
        yield


def _request_lock(request: dict[str, Any]):
    return _exclusive_lock(
        pathlib.Path(request["stage_dir"]) / f"{request['name']}.launch.lock"
    )


def _action_id(context: ActionContext, attempt: int, started_ns: int) -> str:
    """Bind the job-record action to one prepared run/stage attempt."""
    identity = "\0".join(
        (
            str(context.results_dir),
            context.stage_dir.relative_to(context.results_dir).as_posix(),
            context.name,
            str(attempt),
            str(started_ns),
        )
    )
    suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"deft-pas-{context.name}-{suffix}"


def _bundle(
    context: ActionContext,
    execution_image: str,
    attempt: int,
    started_ns: int,
) -> dict[str, Any]:
    network = "clip" if context.image_kind == "pyt" else "data-services"
    declared_inputs = [
        {
            "spec_key": "workflow_results",
            "type": "folder",
            "uri": str(context.results_dir),
        },
        {
            "spec_key": "run_config",
            "type": "folder",
            "uri": str(context.config_dir),
        },
        {
            "spec_key": "compatibility_patches",
            "type": "folder",
            "uri": str(context.patches_dir),
        },
        {
            "spec_key": "model_cache",
            "type": "folder",
            "uri": str(context.cache_dir),
        },
    ]
    if context.name in DATASET_ACTIONS:
        declared_inputs.insert(
            1,
            {
                "spec_key": "dataset_parent",
                "type": "folder",
                "uri": str(context.dataset_root.parent),
            },
        )

    bundle = {
        "network_arch": network,
        "action": _action_id(context, attempt, started_ns),
        "image": execution_image,
        "mode": "args",
        "command": context.command[0],
        "args": context.command[1:],
        "declared_inputs": declared_inputs,
        "declared_outputs": [
            {
                # The job-record binds results_dir to this action's stage, so
                # output keys are relative to that exact job root rather than
                # redundantly repeating the workflow-level path.
                "spec_key": _relative_output(output, context.stage_dir),
                "type": "file",
            }
            for output in context.fresh_outputs
        ],
        "upload_excludes": [".tao-runtime/", "*.launch.lock"],
        "compute_shape": {
            "gpus": int(context.config["num_gpus"]),
            "nodes": 1,
        },
    }
    jsonschema.validate(bundle, _artifact_schema("spec_bundle.schema.json"))
    return bundle


def _mounts(context: ActionContext) -> list[dict[str, Any]]:
    """Return aliases required by existing specs and embedded parquet paths."""
    data_parent = context.dataset_root.parent
    mounts = [
        {"source": str(context.results_dir), "target": "/results", "read_only": False},
        {
            "source": str(context.results_dir),
            "target": str(context.results_dir),
            "read_only": False,
        },
        {"source": str(context.config_dir), "target": "/specs", "read_only": True},
        {"source": str(context.patches_dir), "target": "/patches", "read_only": True},
        {"source": str(context.cache_dir), "target": "/cache", "read_only": False},
    ]
    if context.name in DATASET_ACTIONS:
        mounts[2:2] = [
            {"source": str(data_parent), "target": "/data", "read_only": True},
            {
                "source": str(data_parent),
                "target": str(data_parent),
                "read_only": True,
            },
        ]
    return mounts


def _request(
    context: ActionContext,
    attempt: int,
    started_ns: int,
    *,
    started_at: str | None = None,
    job_state_dir: pathlib.Path | None = None,
) -> dict[str, Any]:
    if context.platform == "virtualenv":
        if context.virtualenv is None:
            raise ValueError("virtualenv platform has no validated action profile")
        virtualenv = str(context.virtualenv)
        record_image = str(context.virtualenv / "bin" / "python")
        entrypoint_sha256 = _sha256_file(
            context.virtualenv / "bin" / context.command[0]
        )
    else:
        virtualenv = None
        record_image = context.image
        entrypoint_sha256 = None
    bundle = _bundle(context, record_image, attempt, started_ns)
    log_path = _attempt_path(context.stage_dir, context.name, attempt, "log")
    payload = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "workflow": WORKFLOW,
        "platform": context.platform,
        "name": context.name,
        "attempt": attempt,
        "label": context.label,
        "image_kind": context.image_kind,
        "record_image": record_image,
        "workload_image": context.image,
        "passed_hf_token": context.pass_hf_token,
        "forward_env": ["HF_TOKEN"] if context.pass_hf_token else [],
        "spec_bundle": bundle,
        "mounts": _mounts(context),
        "environment": {
            "HOME": "/tmp",  # nosec B108 - container HOME inside ephemeral job sandbox
            "PYTHONPATH": "/patches",
            "HF_HOME": "/cache/huggingface",
            "XDG_CACHE_HOME": "/cache",
        },
        "virtualenv": virtualenv,
        "virtualenv_entrypoint_sha256": entrypoint_sha256,
        "virtualenv_shim": str(pathlib.Path(__file__).resolve().parent / "run_deft_cli.py"),
        "results_dir": str(context.results_dir),
        "stage_dir": str(context.stage_dir),
        "platform_runtime_dir": str(_runtime_dir_for(context, attempt)),
        "status_path": str(context.status_path),
        "log_path": str(log_path),
        "fresh_outputs": [str(path) for path in context.fresh_outputs],
        "staging_absent_paths": [
            *[str(path) for path in context.fresh_outputs],
            str(log_path),
        ],
        "freshness_contract": (
            "local-mtime-after-prepare"
            if context.platform == "virtualenv"
            or (
                context.platform == "docker"
                and context.config.get("docker_remote") is not True
            )
            else "remote-mirror-with-delete-before-submit"
        ),
        "staging_receipt_path": str(
            _attempt_path(context.stage_dir, context.name, attempt, "staged.json")
        ),
        "job_binding_path": str(
            _attempt_path(context.stage_dir, context.name, attempt, "job-binding.json")
        ),
        "job_state_dir": str(job_state_dir or _job_state_dir()),
        "started_at": started_at
        or dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "started_ns": started_ns,
    }
    payload["request_sha256"] = _sha256_json(payload)
    return payload


def prepare(args: argparse.Namespace) -> tuple[pathlib.Path, dict[str, Any]]:
    context = validate_action(
        results_dir=args.results_dir,
        image_kind=args.image,
        stage_dir=args.stage_dir,
        name=args.name,
        pass_hf_token=args.pass_hf_token,
        fresh_outputs=args.fresh_output,
        command=args.command,
    )
    with _exclusive_lock(context.lock_path):
        existing = load_existing_status(context.status_path)
        attempt_one_path = _request_path_for(context, 1)
        attempt_two_path = _request_path_for(context, 2)
        archived_status_path = (
            context.stage_dir / f"{context.name}.attempt-1.status.json"
        )
        safe_absolute_path(archived_status_path, "archived attempt-1 status")

        if existing is None:
            if attempt_two_path.exists():
                if not archived_status_path.is_file():
                    raise ValueError(
                        "attempt-2 request exists without archived terminal attempt-1 status"
                    )
                _, request = _load_request(attempt_two_path)
                return attempt_two_path, request
            if attempt_one_path.exists():
                if archived_status_path.exists():
                    raise ValueError(
                        "archived attempt-1 status exists without its attempt-2 request"
                    )
                # Crash-safe initial prepare: return the same immutable attempt.
                _, request = _load_request(attempt_one_path)
                return attempt_one_path, request
            prior_attempt = 0
        else:
            raw_attempt = existing.get("attempt")
            if (
                not isinstance(raw_attempt, int)
                or isinstance(raw_attempt, bool)
                or raw_attempt not in {1, 2}
            ):
                raise ValueError(
                    f"existing command status has invalid attempt: {context.status_path}"
                )
            prior_attempt = raw_attempt
            if existing.get("status") == "ok":
                raise ValueError(
                    f"action already completed successfully: {context.status_path}"
                )
            prior_request_path = _request_path_for(context, prior_attempt)
            if not prior_request_path.exists():
                raise ValueError(
                    "cannot retry without the prior immutable action request"
                )
            prior_request_path, prior_request = _load_request(prior_request_path)
            _validate_retry_lineage(
                context,
                prior_request_path,
                prior_request,
                existing,
            )
            if prior_attempt >= 2:
                raise ValueError(
                    f"attempt budget exhausted for {context.name} (attempt={prior_attempt})"
                )
            if archived_status_path.exists():
                archived = load_existing_status(archived_status_path)
                if archived != existing:
                    raise ValueError(
                        "archived attempt-1 status conflicts with active terminal status"
                    )
            else:
                atomic_json(archived_status_path, existing)
            if attempt_two_path.exists():
                # Recovery after the attempt-2 request became durable but before
                # the fixed active-status path was cleared.
                _, request = _load_request(attempt_two_path)
                context.status_path.unlink()
                return attempt_two_path, request

        for output in context.fresh_outputs:
            try:
                output.unlink()
            except FileNotFoundError:
                pass
        attempt = prior_attempt + 1
        log_path = _attempt_path(context.stage_dir, context.name, attempt, "log")
        try:
            log_path.unlink()
        except FileNotFoundError:
            pass
        for evidence in (
            _attempt_path(context.stage_dir, context.name, attempt, "staged.json"),
            _attempt_path(context.stage_dir, context.name, attempt, "job-binding.json"),
        ):
            safe_absolute_path(evidence, "prior attempt evidence")
            try:
                evidence.unlink()
            except FileNotFoundError:
                pass
        started_ns = time.time_ns()
        payload = _request(context, attempt, started_ns)
        request_path = _request_path_for(context, attempt)
        atomic_json(request_path, payload)
        if existing is not None:
            # The prior terminal status is durable at its attempt-specific
            # archive before the fixed active path is cleared.  A repeated
            # prepare now returns the already-written attempt-2 request.
            context.status_path.unlink()
    return request_path, payload


def _load_request(path: pathlib.Path) -> tuple[pathlib.Path, dict[str, Any]]:
    resolved = safe_absolute_path(path, "action request", require_exists=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"action request is missing or unsafe: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid action request: {resolved}")
    expected_hash = payload.pop("request_sha256", None)
    actual_hash = _sha256_json(payload)
    payload["request_sha256"] = expected_hash
    if not isinstance(expected_hash, str) or expected_hash != actual_hash:
        raise ValueError(f"action request digest mismatch: {resolved}")
    try:
        jsonschema.validate(payload, _reference_schema("action-request.schema.json"))
        jsonschema.validate(
            payload["spec_bundle"], _artifact_schema("spec_bundle.schema.json")
        )
    except jsonschema.ValidationError as exc:
        raise ValueError(f"action request schema violation: {exc.message}") from exc
    _aware_timestamp(payload["started_at"], "action request started_at")
    expected = safe_absolute_path(
        _request_path_from_payload(payload), "action request path"
    )
    if resolved != expected:
        raise ValueError(f"action request must remain at {expected}, got {resolved}")
    bundle = payload["spec_bundle"]
    context = validate_action(
        results_dir=pathlib.Path(payload["results_dir"]),
        image_kind=payload["image_kind"],
        stage_dir=pathlib.Path(payload["stage_dir"]),
        name=payload["name"],
        pass_hf_token=payload["passed_hf_token"],
        fresh_outputs=[pathlib.Path(item) for item in payload["fresh_outputs"]],
        command=[bundle["command"], *bundle["args"]],
        mutate=False,
        require_forwarded_credentials=False,
        verify_virtualenv_runtime=False,
    )
    expected_payload = _request(
        context,
        payload["attempt"],
        payload["started_ns"],
        started_at=payload["started_at"],
        job_state_dir=safe_absolute_path(
            pathlib.Path(payload["job_state_dir"]), "action request job_state_dir"
        ),
    )
    if payload != expected_payload:
        raise ValueError(
            "action request fields do not match immutable workflow state and paths"
        )
    return resolved, payload


def _load_job_record(path: pathlib.Path) -> dict[str, Any]:
    resolved = safe_absolute_path(path, "job-record", require_exists=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"job-record is missing or unsafe: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    jsonschema.validate(payload, _artifact_schema("job_record.schema.json"))
    return payload


def _job_record_path(request: dict[str, Any], job: dict[str, Any]) -> pathlib.Path:
    return safe_absolute_path(
        pathlib.Path(request["job_state_dir"]) / "jobs" / f"{job['id']}.json",
        "bound job-record path",
    )


def _job_identity_sha256(job: dict[str, Any]) -> str:
    return _sha256_json({field: job.get(field) for field in JOB_IDENTITY_FIELDS})


def _ownership_mismatches(
    request: dict[str, Any], job: dict[str, Any]
) -> list[str]:
    bundle = request["spec_bundle"]
    mismatches = []
    for field, expected in (
        ("platform", request["platform"]),
        ("image", request["record_image"]),
        ("network_arch", bundle["network_arch"]),
        ("action", bundle["action"]),
        ("upload_excludes", bundle["upload_excludes"]),
    ):
        if job.get(field) != expected:
            mismatches.append(f"{field}={job.get(field)!r}, expected {expected!r}")
    return mismatches


def _canonical_remote_scope(value: Any) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError("remote backend scope must be a non-empty canonical string")
    if any(ord(character) < 32 for character in value) or "\\" in value:
        raise ValueError(
            "remote backend scope must not contain control characters or backslashes"
        )
    if value.startswith("/"):
        path = pathlib.PurePosixPath(value)
        if (
            value == "/"
            or path.as_posix() != value
            or ".." in path.parts
            or "." in path.parts
        ):
            raise ValueError("remote backend scope path must be normalized and non-root")
        return value
    parsed = urllib.parse.urlsplit(value)
    remote_path = pathlib.PurePosixPath(parsed.path)
    decoded_path = urllib.parse.unquote(parsed.path)
    decoded_parts = pathlib.PurePosixPath(decoded_path).parts
    if (
        not parsed.scheme
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or not parsed.path
        or remote_path.as_posix() != parsed.path
        or ".." in remote_path.parts
        or "." in remote_path.parts
        or ".." in decoded_parts
        or "." in decoded_parts
        or "\\" in decoded_path
    ):
        raise ValueError(
            "remote backend scope must be an absolute compute path or a "
            "credential-free normalized persistence URI"
        )
    return value


def _validate_job_ownership(
    request_path: pathlib.Path,
    request: dict[str, Any],
    job_path: pathlib.Path,
    job: dict[str, Any],
) -> None:
    expected_path = _job_record_path(request, job)
    if job_path != expected_path:
        raise ValueError(
            f"job-record must be the request-bound state record {expected_path}, got {job_path}"
        )
    mismatches = _ownership_mismatches(request, job)
    if mismatches:
        raise ValueError("job-record does not own this action: " + "; ".join(mismatches))
    request_started = _aware_timestamp(request["started_at"], "action request started_at")
    job_submitted = _aware_timestamp(job["submitted_at"], "job-record submitted_at")
    if job_submitted < request_started:
        raise ValueError("job-record predates the prepared action request")
    if request_path != _request_path_from_payload(request):
        raise ValueError("action request path is inconsistent during job binding")


def attest_staged(args: argparse.Namespace) -> pathlib.Path:
    """Record the platform consumer's post-sync absence check for remote jobs."""
    request_path, request = _load_request(args.request)
    if request["freshness_contract"] != "remote-mirror-with-delete-before-submit":
        raise ValueError("staging attestation is valid only for remote platforms")
    checked = [str(pathlib.Path(raw)) for raw in args.absent_path]
    if checked != request["staging_absent_paths"]:
        raise ValueError(
            "--absent-path must repeat every request staging_absent_paths entry in order"
        )
    backend_scope = _canonical_remote_scope(args.backend_scope)
    payload = {
        "schema_version": "1",
        "workflow": WORKFLOW,
        "platform": request["platform"],
        "request_path": str(request_path),
        "request_sha256": request["request_sha256"],
        "backend_scope": backend_scope,
        "checked_paths_absent": checked,
        "checked_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    payload["receipt_sha256"] = _sha256_json(payload)
    jsonschema.validate(payload, _reference_schema("staging-receipt.schema.json"))
    path = safe_absolute_path(
        pathlib.Path(request["staging_receipt_path"]), "staging receipt"
    )
    atomic_json(path, payload)
    return path


def _validate_staging_receipt(
    request: dict[str, Any], job: dict[str, Any]
) -> tuple[str | None, str]:
    if request["freshness_contract"] != "remote-mirror-with-delete-before-submit":
        expected = str(safe_absolute_path(pathlib.Path(request["stage_dir"]), "action stage"))
        if job.get("results_dir") != expected:
            raise ValueError(
                f"local job-record results_dir must equal the action stage {expected}"
            )
        return None, expected
    path = safe_absolute_path(
        pathlib.Path(str(request["staging_receipt_path"])),
        "staging receipt",
        require_exists=True,
    )
    if not path.is_file() or path.is_symlink() or path.resolve() != path:
        raise ValueError(f"remote action lacks a safe staging receipt: {path}")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    jsonschema.validate(receipt, _reference_schema("staging-receipt.schema.json"))
    body = dict(receipt)
    digest = body.pop("receipt_sha256", None)
    actual = _sha256_json(body)
    if digest != actual:
        raise ValueError(f"staging receipt digest mismatch: {path}")
    expected = {
        "platform": request["platform"],
        "request_path": str(_request_path_from_payload(request)),
        "request_sha256": request["request_sha256"],
        "checked_paths_absent": request["staging_absent_paths"],
    }
    for field, value in expected.items():
        if receipt.get(field) != value:
            raise ValueError(f"staging receipt {field} does not match the action request")
    checked_at = _aware_timestamp(receipt["checked_at"], "staging receipt checked_at")
    submitted_at = _aware_timestamp(job["submitted_at"], "job-record submitted_at")
    if submitted_at < checked_at:
        raise ValueError("job-record predates the remote output-absence attestation")
    backend_scope = _canonical_remote_scope(receipt.get("backend_scope"))
    if job.get("results_dir") != backend_scope:
        raise ValueError(
            "remote job-record results_dir must equal the attested backend scope"
        )
    return digest, backend_scope


def _binding_payload(
    *,
    request_path: pathlib.Path,
    request: dict[str, Any],
    job_path: pathlib.Path,
    job: dict[str, Any],
    staging_receipt_sha256: str | None,
    results_scope: str,
    bound_at: str,
) -> dict[str, Any]:
    _aware_timestamp(bound_at, "job binding bound_at")
    payload = {
        "schema_version": "1",
        "workflow": WORKFLOW,
        "platform": request["platform"],
        "request_path": str(request_path),
        "request_sha256": request["request_sha256"],
        "job_record_path": str(job_path),
        "job_id": job["id"],
        "job_identity_sha256": _job_identity_sha256(job),
        "results_scope": results_scope,
        "staging_receipt_sha256": staging_receipt_sha256,
        "bound_at": bound_at,
    }
    payload["binding_sha256"] = _sha256_json(payload)
    jsonschema.validate(payload, _reference_schema("job-binding.schema.json"))
    return payload


def _load_job_binding(
    request_path: pathlib.Path,
    request: dict[str, Any],
    job_path: pathlib.Path,
    job: dict[str, Any],
) -> dict[str, Any]:
    path = safe_absolute_path(
        pathlib.Path(request["job_binding_path"]),
        "job binding",
        require_exists=True,
    )
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"job binding is missing or unsafe: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    jsonschema.validate(payload, _reference_schema("job-binding.schema.json"))
    body = dict(payload)
    digest = body.pop("binding_sha256", None)
    if digest != _sha256_json(body):
        raise ValueError(f"job binding digest mismatch: {path}")
    staging_digest, results_scope = _validate_staging_receipt(request, job)
    expected = _binding_payload(
        request_path=request_path,
        request=request,
        job_path=job_path,
        job=job,
        staging_receipt_sha256=staging_digest,
        results_scope=results_scope,
        bound_at=payload["bound_at"],
    )
    if payload != expected:
        raise ValueError("job binding does not match the immutable request and job")
    return payload


def _validate_retry_lineage(
    context: ActionContext,
    request_path: pathlib.Path,
    request: dict[str, Any],
    status: dict[str, Any],
) -> None:
    """Require one fully finalized terminal attempt before minting a retry."""
    if status.get("schema_version") == "1":
        if request.get("platform") != "docker" or request.get("attempt") != 1:
            raise ValueError(
                "legacy command status can establish only Docker attempt-1 lineage"
            )
        legacy_expected = {
            "workflow": WORKFLOW,
            "kind": "container",
            "name": context.name,
            "attempt": 1,
            "image_kind": request["image_kind"],
            "image": request["workload_image"],
            "command": [
                request["spec_bundle"]["command"],
                *request["spec_bundle"]["args"],
            ],
            "command_sha256": command_sha256(
                [
                    request["spec_bundle"]["command"],
                    *request["spec_bundle"]["args"],
                ]
            ),
            "passed_hf_token": request["passed_hf_token"],
            "started_at": request["started_at"],
            "started_ns": request["started_ns"],
            "log_path": request["log_path"],
            "fresh_outputs": request["fresh_outputs"],
            "status": "error",
        }
        for field, value in legacy_expected.items():
            if status.get(field) != value:
                raise ValueError(
                    f"existing legacy command status {field} does not match "
                    "prior action lineage"
                )
        if (
            not isinstance(status.get("exit_code"), int)
            or isinstance(status["exit_code"], bool)
            or status["exit_code"] == 0
        ):
            raise ValueError(
                "existing legacy error status must record a nonzero exit_code"
            )
        finished_at = _aware_timestamp(
            status.get("finished_at"), "existing legacy status finished_at"
        )
        if finished_at < _aware_timestamp(
            request["started_at"], "prior action request started_at"
        ):
            raise ValueError(
                "existing legacy status finished_at predates the prior action"
            )
        log_path = safe_absolute_path(
            pathlib.Path(request["log_path"]),
            "prior legacy action log",
            require_exists=True,
        )
        if (
            not log_path.is_file()
            or log_path.is_symlink()
            or log_path.stat().st_size == 0
        ):
            raise ValueError(
                "cannot retry without the immutable prior legacy action log"
            )
        return
    try:
        jsonschema.validate(
            status, _reference_schema("platform-action-status.schema.json")
        )
    except jsonschema.ValidationError as exc:
        raise ValueError(
            f"existing command status schema violation: {exc.message}"
        ) from exc
    expected = {
        "schema_version": STATUS_SCHEMA_VERSION,
        "workflow": WORKFLOW,
        "kind": "platform_action",
        "name": context.name,
        "attempt": request["attempt"],
        "platform": request["platform"],
        "request_path": str(request_path),
        "request_sha256": request["request_sha256"],
        "status": "error",
        "image_kind": request["image_kind"],
        "image": request["workload_image"],
        "command": [
            request["spec_bundle"]["command"],
            *request["spec_bundle"]["args"],
        ],
        "command_sha256": command_sha256(
            [
                request["spec_bundle"]["command"],
                *request["spec_bundle"]["args"],
            ]
        ),
        "passed_hf_token": request["passed_hf_token"],
        "started_at": request["started_at"],
        "started_ns": request["started_ns"],
        "log_path": request["log_path"],
        "fresh_outputs": request["fresh_outputs"],
        "freshness_contract": request["freshness_contract"],
    }
    for field, value in expected.items():
        if status.get(field) != value:
            raise ValueError(
                f"existing command status {field} does not match prior action lineage"
            )
    if (
        not isinstance(status.get("exit_code"), int)
        or isinstance(status["exit_code"], bool)
        or status["exit_code"] == 0
    ):
        raise ValueError("existing error status must record a nonzero exit_code")
    finished_at = _aware_timestamp(
        status.get("finished_at"), "existing status finished_at"
    )
    if finished_at < _aware_timestamp(
        request["started_at"], "prior action request started_at"
    ):
        raise ValueError("existing status finished_at predates the prior action")
    log_path = safe_absolute_path(
        pathlib.Path(request["log_path"]), "prior action log", require_exists=True
    )
    if not log_path.is_file() or log_path.is_symlink() or log_path.stat().st_size == 0:
        raise ValueError("cannot retry without the immutable prior action log")
    matches = _matching_job_records(request_path, request)
    if len(matches) != 1:
        raise ValueError(
            "existing error status must have exactly one request-owned job-record"
        )
    job_path, job = matches[0]
    binding = _load_job_binding(request_path, request, job_path, job)
    if job.get("terminal_state") not in {"COMPLETE", "ERROR", "CANCELED"}:
        raise ValueError("cannot retry while the prior native job is nonterminal")
    _validate_terminal_job(job)
    if not isinstance(job.get("backend_ref"), str) or not job["backend_ref"].strip():
        raise ValueError("cannot retry without the prior native backend reference")
    for field, expected_value in (
        ("job_id", job["id"]),
        ("backend_ref", job["backend_ref"]),
        ("storage_tier", job["storage_tier"]),
        ("backend_state", job["terminal_state"]),
        ("job_binding_sha256", binding["binding_sha256"]),
        ("results_scope", binding["results_scope"]),
        ("staging_receipt_sha256", binding["staging_receipt_sha256"]),
    ):
        if status.get(field) != expected_value:
            raise ValueError(
                f"existing command status {field} does not match prior job evidence"
            )
    expected_exit = (
        status["backend_exit_code"] if status["backend_exit_code"] else 3
    )
    if status["exit_code"] != expected_exit or status["exit_code"] == 0:
        raise ValueError(
            "existing command status exit_code is inconsistent with terminal evidence"
        )


def bind_job(args: argparse.Namespace) -> pathlib.Path:
    """Bind one freshly opened job-record to a request before native submit."""
    request_path, request = _load_request(args.request)
    with _request_lock(request):
        job_path = safe_absolute_path(args.job_record, "job-record", require_exists=True)
        job = _load_job_record(job_path)
        _validate_job_ownership(request_path, request, job_path, job)
        if (
            job.get("backend_ref") is not None
            or job.get("terminal_state") is not None
            or len(job.get("transitions", [])) != 1
            or job["transitions"][0].get("state") != "PENDING"
        ):
            raise ValueError("job binding must occur after open and before native submit")
        staging_digest, results_scope = _validate_staging_receipt(request, job)
        path = safe_absolute_path(pathlib.Path(request["job_binding_path"]), "job binding")
        if path.exists():
            # Binding is immutable and idempotent for the same record.  A
            # second concurrently opened record fails validation here and can
            # never overwrite the first winner's binding.
            _load_job_binding(request_path, request, job_path, job)
            return path
        payload = _binding_payload(
            request_path=request_path,
            request=request,
            job_path=job_path,
            job=job,
            staging_receipt_sha256=staging_digest,
            results_scope=results_scope,
            bound_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        )
        atomic_json(path, payload)
    return path


def _matching_job_records(
    request_path: pathlib.Path, request: dict[str, Any]
) -> list[tuple[pathlib.Path, dict[str, Any]]]:
    jobs = safe_absolute_path(
        pathlib.Path(request["job_state_dir"]) / "jobs", "job-record directory"
    )
    if not jobs.exists():
        return []
    if not jobs.is_dir():
        raise ValueError(f"job-record directory is not a directory: {jobs}")
    matches: list[tuple[pathlib.Path, dict[str, Any]]] = []
    action_id = request["spec_bundle"]["action"]
    for path in sorted(jobs.glob("*.json")):
        try:
            candidate = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # The job directory is shared across actions and releases. A file
            # that cannot identify itself as this action is unrelated state.
            continue
        if not isinstance(candidate, dict) or candidate.get("action") != action_id:
            continue
        try:
            job = _load_job_record(path)
        except (
            OSError,
            ValueError,
            json.JSONDecodeError,
            jsonschema.ValidationError,
        ) as exc:
            raise ValueError(
                f"cannot reconcile malformed matching job-record {path}: {exc}"
            ) from exc
        _validate_job_ownership(request_path, request, path, job)
        matches.append((path, job))
    if len(matches) > 1:
        raise ValueError("multiple job-records claim the same prepared action")
    return matches


def reconcile_request(args: argparse.Namespace) -> dict[str, Any]:
    request_path, request = _load_request(args.request)
    matches = _matching_job_records(request_path, request)
    binding_path = pathlib.Path(request["job_binding_path"])
    if not matches:
        if binding_path.exists():
            raise ValueError("job binding exists without its request-owned job-record")
        return {
            "state": "NO_JOB_RECORD",
            "request": str(request_path),
            "action": request["spec_bundle"]["action"],
        }
    job_path, job = matches[0]
    if binding_path.exists():
        _load_job_binding(request_path, request, job_path, job)
        state = (
            "BOUND"
            if job.get("backend_ref")
            else "BOUND_BACKEND_RECONCILIATION_REQUIRED"
        )
    else:
        state = "JOB_OPENED_UNBOUND"
    return {
        "state": state,
        "request": str(request_path),
        "action": request["spec_bundle"]["action"],
        "job_id": job["id"],
        "job_record": str(job_path),
        "backend_ref_present": bool(job.get("backend_ref")),
        "terminal_state": job.get("terminal_state"),
    }


def _validate_terminal_job(job: dict[str, Any]) -> None:
    terminal = job.get("terminal_state")
    transitions = job.get("transitions")
    if not isinstance(transitions, list) or len(transitions) < 3:
        raise ValueError(
            "terminal job-record must preserve PENDING, RUNNING, and terminal transitions"
        )
    states = [item.get("state") if isinstance(item, dict) else None for item in transitions]
    if states[0] != "PENDING" or states[-1] != terminal or "RUNNING" not in states[1:-1]:
        raise ValueError(
            "terminal job-record transition lineage must be PENDING -> RUNNING -> terminal"
        )
    terminal_states = {"COMPLETE", "ERROR", "CANCELED"}
    if any(state in terminal_states for state in states[:-1]):
        raise ValueError("job-record has a transition after an earlier terminal state")
    timestamps = [
        _aware_timestamp(item.get("ts"), "job-record transition timestamp")
        for item in transitions
    ]
    if timestamps != sorted(timestamps):
        raise ValueError("job-record transitions are not timestamp ordered")


def finalize(args: argparse.Namespace) -> tuple[pathlib.Path, int]:
    request_path, request = _load_request(args.request)
    job_path = safe_absolute_path(args.job_record, "job-record", require_exists=True)
    job = _load_job_record(job_path)
    _validate_job_ownership(request_path, request, job_path, job)
    binding = _load_job_binding(request_path, request, job_path, job)
    bundle = request["spec_bundle"]
    if not isinstance(job.get("backend_ref"), str) or not job["backend_ref"].strip():
        raise ValueError("job-record lacks the native backend reference")
    staging_receipt_sha256 = binding["staging_receipt_sha256"]
    terminal = job.get("terminal_state")
    if terminal not in {"COMPLETE", "ERROR", "CANCELED"}:
        raise ValueError("job-record is not terminal; poll the selected platform first")
    _validate_terminal_job(job)
    log_path = pathlib.Path(str(request["log_path"]))
    if (
        not log_path.is_absolute()
        or not log_path.is_file()
        or log_path.stat().st_size == 0
        or log_path.is_symlink()
        or log_path.resolve() != log_path
    ):
        raise ValueError(
            f"platform logs must be captured at the immutable action log path: {log_path}"
        )
    stage_dir = pathlib.Path(str(request["stage_dir"])).resolve()
    log_path.relative_to(stage_dir)
    secret_linter = pathlib.Path(__file__).resolve().parents[4] / "scripts" / "redact_secrets.py"
    linted = subprocess.run(
        [sys.executable, str(secret_linter), "lint", str(log_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if linted.returncode != 0:
        raise ValueError(
            "platform log failed credential lint; redact the captured log before finalization"
        )
    started_ns = request.get("started_ns")
    if not isinstance(started_ns, int) or isinstance(started_ns, bool) or started_ns < 1:
        raise ValueError("action request started_ns is invalid")

    artifact_error = None
    if terminal == "COMPLETE":
        if args.native_exit_code != 0:
            raise ValueError("a COMPLETE backend must report --native-exit-code 0")
        for raw in request["fresh_outputs"]:
            output = pathlib.Path(raw)
            if (
                output.is_symlink()
                or not output.is_file()
                or output.stat().st_size == 0
            ):
                artifact_error = f"fresh output is missing, empty, or a symlink: {output}"
                break
            if (
                request["freshness_contract"] == "local-mtime-after-prepare"
                and output.stat().st_mtime_ns < started_ns
            ):
                artifact_error = f"fresh output predates this action: {output}"
                break
    backend_exit_code = args.native_exit_code
    success = terminal == "COMPLETE" and backend_exit_code == 0 and artifact_error is None
    payload = {
        "schema_version": STATUS_SCHEMA_VERSION,
        "workflow": WORKFLOW,
        "kind": "platform_action",
        "name": request["name"],
        "attempt": request["attempt"],
        "platform": request["platform"],
        "job_id": job["id"],
        "backend_ref": job["backend_ref"],
        "storage_tier": job["storage_tier"],
        "backend_state": terminal,
        "backend_exit_code": backend_exit_code,
        "image_kind": request["image_kind"],
        "image": request["workload_image"],
        "command": [bundle["command"], *bundle["args"]],
        "command_sha256": command_sha256([bundle["command"], *bundle["args"]]),
        "passed_hf_token": request["passed_hf_token"],
        "request_path": str(request_path),
        "request_sha256": request["request_sha256"],
        "job_binding_sha256": binding["binding_sha256"],
        "results_scope": binding["results_scope"],
        "staging_receipt_sha256": staging_receipt_sha256,
        "started_at": request["started_at"],
        "started_ns": started_ns,
        "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "status": "ok" if success else "error",
        "exit_code": 0 if success else (backend_exit_code if backend_exit_code else 3),
        "log_path": str(log_path),
        "fresh_outputs": request["fresh_outputs"],
        "freshness_contract": request["freshness_contract"],
        "artifact_error": artifact_error,
    }
    jsonschema.validate(
        payload, _reference_schema("platform-action-status.schema.json")
    )
    status_path = pathlib.Path(str(request["status_path"]))
    atomic_json(status_path, payload)
    return status_path, 0 if success else 3


def _common_action_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--results-dir", required=True, type=pathlib.Path)
    parser.add_argument("--image", required=True, choices=("pyt", "ds"))
    parser.add_argument("--stage-dir", required=True, type=pathlib.Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--pass-hf-token", action="store_true")
    parser.add_argument(
        "--fresh-output", action="append", default=[], type=pathlib.Path
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="verb", required=True)
    prepare_parser = sub.add_parser("prepare")
    _common_action_args(prepare_parser)
    staged_parser = sub.add_parser("attest-staged")
    staged_parser.add_argument("--request", required=True, type=pathlib.Path)
    staged_parser.add_argument("--backend-scope", required=True)
    staged_parser.add_argument(
        "--absent-path", action="append", default=[], required=True
    )
    reconcile_parser = sub.add_parser("reconcile")
    reconcile_parser.add_argument("--request", required=True, type=pathlib.Path)
    bind_parser = sub.add_parser("bind-job")
    bind_parser.add_argument("--request", required=True, type=pathlib.Path)
    bind_parser.add_argument("--job-record", required=True, type=pathlib.Path)
    finalize_parser = sub.add_parser("finalize")
    finalize_parser.add_argument("--request", required=True, type=pathlib.Path)
    finalize_parser.add_argument("--job-record", required=True, type=pathlib.Path)
    finalize_parser.add_argument("--native-exit-code", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.verb == "prepare":
        if args.command and args.command[0] == "--":
            args.command = args.command[1:]
        if not args.command:
            print("run_deft_action: command after -- is required", file=sys.stderr)
            return 2
    try:
        if args.verb == "prepare":
            path, payload = prepare(args)
            reconciliation = reconcile_request(
                argparse.Namespace(request=path)
            )
            print(
                json.dumps(
                    {
                        "request": str(path),
                        "platform": payload["platform"],
                        "action": payload["spec_bundle"]["action"],
                        "attempt": payload["attempt"],
                        "reconciliation": reconciliation,
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.verb == "attest-staged":
            path = attest_staged(args)
            print(json.dumps({"staging_receipt": str(path)}, sort_keys=True))
            return 0
        if args.verb == "reconcile":
            print(json.dumps(reconcile_request(args), sort_keys=True))
            return 0
        if args.verb == "bind-job":
            path = bind_job(args)
            print(json.dumps({"job_binding": str(path)}, sort_keys=True))
            return 0
        path, returncode = finalize(args)
        print(json.dumps({"status": str(path), "exit_code": returncode}, sort_keys=True))
        return returncode
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
        jsonschema.ValidationError,
    ) as exc:
        print(f"run_deft_action: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
