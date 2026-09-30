# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Platform-neutral validation for one TAO action in the PAS DEFT loop."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
from dataclasses import dataclass
from typing import Any

from command_contract import (
    expected_container_command,
    expected_fresh_outputs,
    expected_hf_forwarding,
    expected_image_kind,
    expected_stage_directory,
)
from virtualenv_runtime import validate_tao_virtualenv


WORKFLOW = "tao-run-deft-pas"
SUPPORTED_PLATFORMS = ("docker", "slurm", "kubernetes", "brev", "virtualenv")
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


def safe_absolute_path(
    path: pathlib.Path, name: str, *, require_exists: bool = False
) -> pathlib.Path:
    """Return one lexical absolute path after rejecting every symlink hop.

    Resolving first and validating later loses whether the caller supplied a
    symlink.  Keep the lexical path, normalize only ``.``/``..``, and compare it
    with ``resolve(strict=False)`` so existing symlinks in any parent are
    rejected even when the final path has not been created yet.
    """
    expanded = path.expanduser()
    if not expanded.is_absolute():
        raise ValueError(f"{name} must be an absolute path: {path}")
    lexical = pathlib.Path(os.path.abspath(expanded))
    if lexical == pathlib.Path(lexical.anchor):
        raise ValueError(f"{name} must not be a filesystem root: {lexical}")
    if lexical.resolve(strict=False) != lexical:
        raise ValueError(f"{name} must not contain or traverse a symlink: {lexical}")
    if require_exists and not lexical.exists():
        raise ValueError(f"{name} does not exist: {lexical}")
    return lexical


@dataclass(frozen=True)
class ActionContext:
    """Validated immutable inputs shared by every platform consumer."""

    state: dict[str, Any]
    config: dict[str, Any]
    platform: str
    image_kind: str
    image: str
    virtualenv: pathlib.Path | None
    name: str
    label: str
    command: list[str]
    pass_hf_token: bool
    results_dir: pathlib.Path
    workspace: pathlib.Path
    dataset_root: pathlib.Path
    config_dir: pathlib.Path
    patches_dir: pathlib.Path
    cache_dir: pathlib.Path
    stage_dir: pathlib.Path
    status_path: pathlib.Path
    log_path: pathlib.Path
    request_path: pathlib.Path
    lock_path: pathlib.Path
    fresh_outputs: list[pathlib.Path]


def atomic_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically without leaving a partial evidence file."""
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _under(path: pathlib.Path, root: pathlib.Path, name: str) -> pathlib.Path:
    resolved = safe_absolute_path(path, name)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{name} must be under {root}: {resolved}") from exc
    return resolved


def fresh_output_path(
    path: pathlib.Path, root: pathlib.Path, name: str
) -> pathlib.Path:
    """Resolve an output lexically so deletion never follows a final symlink."""
    absolute = safe_absolute_path(path, name)
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
    return absolute


def load_state(results_dir: pathlib.Path) -> dict[str, Any]:
    path = results_dir / "deft_state.json"
    if (
        not path.is_file()
        or path.is_symlink()
        or path.resolve(strict=False) != path
    ):
        raise ValueError(f"state file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("workflow") != WORKFLOW
        or payload.get("schema_version") != "3"
    ):
        raise ValueError(f"invalid PAS DEFT state: {path}")
    if pathlib.Path(str(payload.get("results_dir", ""))).resolve() != results_dir:
        raise ValueError("state.results_dir does not match --results-dir")
    return payload


def _workspace_child(
    path: pathlib.Path, workspace: pathlib.Path, name: str
) -> pathlib.Path:
    resolved = safe_absolute_path(path, name)
    try:
        relative = resolved.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(f"{name} must be under workspace {workspace}: {resolved}") from exc
    if relative == pathlib.Path("."):
        raise ValueError(f"{name} must be a child of workspace, not workspace itself")
    return resolved


def validate_runtime_paths(
    results_dir: pathlib.Path, config: dict[str, Any]
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    """Validate local workflow paths without imposing an execution backend."""
    workspace = safe_absolute_path(
        pathlib.Path(str(config.get("workspace", ""))), "state workspace"
    )
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
        raise ValueError("refusing to stage a filesystem root as the dataset parent")

    config_dir = safe_absolute_path(
        pathlib.Path(str(config.get("config_dir", ""))), "state config_dir"
    )
    expected_config_dir = results_dir / "config"
    if config_dir != expected_config_dir or not config_dir.is_dir():
        raise ValueError(
            f"state config_dir must be the existing run config directory {expected_config_dir}"
        )
    expected_hashes = config.get("spec_sha256")
    if not isinstance(expected_hashes, dict) or set(expected_hashes) != set(RUN_SPEC_NAMES):
        raise ValueError("state.config.spec_sha256 must bind all immutable run config files")
    for spec_name in RUN_SPEC_NAMES:
        path = config_dir / spec_name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"run spec is missing or empty: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected_hashes.get(spec_name) != actual:
            raise ValueError(f"run spec changed after approval: {path}")
    if pathlib.Path(str(config.get("deft_config", ""))).resolve() != config_dir / "deft_config.yaml":
        raise ValueError("state deft_config path does not match the immutable run config")
    if pathlib.Path(str(config.get("tao_spec", ""))).resolve() != config_dir / "tao_spec.yaml":
        raise ValueError("state tao_spec path does not match the immutable run config")
    platform = config.get("platform")
    if platform not in SUPPORTED_PLATFORMS:
        raise ValueError(
            "state platform must be one of " + ", ".join(SUPPORTED_PLATFORMS)
        )
    docker_remote = config.get("docker_remote", False)
    if not isinstance(docker_remote, bool):
        raise ValueError("state.config.docker_remote must be boolean")
    if docker_remote and platform != "docker":
        raise ValueError(
            "state.config.docker_remote may be true only for platform=docker"
        )
    return workspace, dataset_root, config_dir


def reject_sensitive_argv(command: list[str]) -> None:
    secret_values = [
        value
        for name in ("NGC_KEY", "HF_TOKEN", "BREV_API_TOKEN")
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
                "action command must not carry credentials in argv; forward approved "
                "environment variable names through the selected platform"
            )
        if any(secret in item for secret in secret_values):
            raise ValueError("action command argv contains a credential value")


def launch_label(name: str, stage_dir: pathlib.Path, results_dir: pathlib.Path) -> str:
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


def load_existing_status(path: pathlib.Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    if not path.is_file() or path.is_symlink() or path.resolve(strict=False) != path:
        raise ValueError(f"existing command status is missing or unsafe: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"existing command status is invalid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"existing command status root must be an object: {path}")
    return payload


def validate_action(
    *,
    results_dir: pathlib.Path,
    image_kind: str,
    stage_dir: pathlib.Path,
    name: str,
    pass_hf_token: bool,
    fresh_outputs: list[pathlib.Path],
    command: list[str],
    mutate: bool = True,
    require_forwarded_credentials: bool = True,
    verify_virtualenv_runtime: bool = True,
) -> ActionContext:
    """Validate a requested PAS TAO action against immutable workflow state."""
    results_dir = safe_absolute_path(results_dir, "--results-dir", require_exists=True)
    state = load_state(results_dir)
    config = state.get("config")
    if not isinstance(config, dict):
        raise ValueError("state.config must be an object")
    if image_kind not in PINNED_IMAGES:
        raise ValueError(f"unsupported image kind: {image_kind!r}")
    image_key = "pyt_image" if image_kind == "pyt" else "ds_image"
    image = str(config.get(image_key, "")).strip()
    if image != PINNED_IMAGES[image_kind]:
        raise ValueError(f"state.config.{image_key} must be the pinned {image_kind} image")
    workspace, dataset_root, config_dir = validate_runtime_paths(results_dir, config)
    platform = str(config["platform"])
    patches_dir = pathlib.Path(__file__).resolve().parent.parent / "patches"
    if not patches_dir.is_dir():
        raise ValueError(f"container compatibility patches are missing: {patches_dir}")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", name):
        raise ValueError("--name must contain only lowercase letters, digits, ._- characters")
    reject_sensitive_argv(command)
    normalized_stage = _under(stage_dir, results_dir, "--stage-dir")
    label = launch_label(name, normalized_stage, results_dir)
    if label.startswith("iter"):
        number = int(label[4:])
        maximum = state.get("max_iterations")
        if (
            not isinstance(maximum, int)
            or isinstance(maximum, bool)
            or not 1 <= number <= maximum
        ):
            raise ValueError(
                f"action iteration {number} is outside approved range 1..{maximum}"
            )
    expected_stage = expected_stage_directory(name, label, results_dir)
    if normalized_stage != expected_stage:
        raise ValueError(
            f"--stage-dir for {name} must be {expected_stage}, got {normalized_stage}"
        )
    expected_command = expected_container_command(name, label, config)
    if command != expected_command:
        raise ValueError(
            f"action argv for {name} does not match the immutable workflow command"
        )
    expected_kind = expected_image_kind(name)
    if image_kind != expected_kind:
        raise ValueError(f"--image for {name} must be {expected_kind}, got {image_kind}")
    virtualenv: pathlib.Path | None = None
    legacy_virtualenv = config.get("virtualenv")
    virtualenvs = config.get("virtualenvs")
    if platform == "virtualenv":
        if isinstance(virtualenvs, dict) and set(virtualenvs) == set(PINNED_IMAGES):
            selected = virtualenvs.get(image_kind)
        elif isinstance(legacy_virtualenv, str) and legacy_virtualenv.strip():
            # Compatibility is truthful only when the same environment passed
            # both profile contracts during initialization.
            selected = legacy_virtualenv
        else:
            raise ValueError(
                "virtualenv platform requires state.config.virtualenvs.pyt and .ds"
            )
        if not isinstance(selected, str) or not selected.strip():
            raise ValueError(f"state.config.virtualenvs.{image_kind} must be an absolute path")
        if pathlib.Path(selected).expanduser().resolve() == (workspace / ".venv").resolve():
            raise ValueError(
                "TAO execution profile must be separate from the workspace control .venv"
            )
        required_gpus = config.get("num_gpus")
        if not isinstance(required_gpus, int) or isinstance(required_gpus, bool):
            raise ValueError("state.config.num_gpus must be an integer")
        selected_path = safe_absolute_path(
            pathlib.Path(selected), f"state.config.virtualenvs.{image_kind}"
        )
        virtualenv = (
            validate_tao_virtualenv(
                selected_path,
                profile=image_kind,
                probe_imports=True,
                required_cli=command[0],
                minimum_gpus=required_gpus,
            )
            if verify_virtualenv_runtime
            else selected_path
        )
    elif legacy_virtualenv is not None or virtualenvs is not None:
        raise ValueError(
            "state virtualenv configuration is valid only for platform=virtualenv"
        )
    required_hf = expected_hf_forwarding(name, config)
    if bool(pass_hf_token) != required_hf:
        raise ValueError(
            f"--pass-hf-token for {name} must be {required_hf} per immutable approval"
        )
    if (
        require_forwarded_credentials
        and pass_hf_token
        and not os.environ.get("HF_TOKEN")
    ):
        raise ValueError("--pass-hf-token requires HF_TOKEN in the process environment")

    normalized_outputs: list[pathlib.Path] = []
    for raw in fresh_outputs:
        output = fresh_output_path(raw, results_dir, "--fresh-output")
        if output.exists() and not output.is_file():
            raise ValueError(f"--fresh-output must name a file, not a directory: {output}")
        normalized_outputs.append(output)
    expected_outputs = expected_fresh_outputs(name, label, results_dir)
    if normalized_outputs != expected_outputs:
        raise ValueError(
            f"--fresh-output for {name} must be {[str(p) for p in expected_outputs]}, "
            f"got {[str(p) for p in normalized_outputs]}"
        )

    cache_dir = safe_absolute_path(workspace / "cache", "workspace cache")
    # Validate every mutation target before creating any of them.  In
    # particular, an existing ``results/config`` or ``workspace/cache``
    # symlink must never redirect action writes outside the approved tree.
    safe_absolute_path(normalized_stage, "--stage-dir")
    if mutate:
        normalized_stage.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
    return ActionContext(
        state=state,
        config=config,
        platform=platform,
        image_kind=image_kind,
        image=image,
        virtualenv=virtualenv,
        name=name,
        label=label,
        command=list(command),
        pass_hf_token=bool(pass_hf_token),
        results_dir=results_dir,
        workspace=workspace,
        dataset_root=dataset_root,
        config_dir=config_dir,
        patches_dir=patches_dir,
        cache_dir=cache_dir,
        stage_dir=normalized_stage,
        status_path=normalized_stage / f"{name}.status.json",
        log_path=normalized_stage / f"{name}.log",
        request_path=normalized_stage / f"{name}.action.json",
        lock_path=normalized_stage / f"{name}.launch.lock",
        fresh_outputs=normalized_outputs,
    )


def backend_exit_code(payload: dict[str, Any]) -> Any:
    """Read native success from current evidence, preserving v1 Docker records."""
    if payload.get("schema_version") == "2":
        return payload.get("backend_exit_code")
    return payload.get("docker_exit_code")


def action_kind_is_valid(payload: dict[str, Any]) -> bool:
    """Accept current platform evidence and legacy Docker evidence."""
    version = payload.get("schema_version")
    return (version == "2" and payload.get("kind") == "platform_action") or (
        version == "1" and payload.get("kind") == "container"
    )


def platform_evidence_error(
    payload: dict[str, Any], expected_platform: str
) -> str | None:
    """Return why action evidence is not native-success proof, if anything."""
    if expected_platform not in SUPPORTED_PLATFORMS:
        return "expected platform is not a supported TAO platform"
    if not action_kind_is_valid(payload):
        return "kind/schema_version does not identify supported action evidence"
    if backend_exit_code(payload) != 0:
        return "native backend exit code is not zero"
    if payload.get("artifact_error") is not None:
        return "artifact freshness validation failed"
    if payload.get("schema_version") == "2":
        platform = payload.get("platform")
        if platform not in SUPPORTED_PLATFORMS:
            return "platform is not a supported TAO platform"
        if platform != expected_platform:
            return "platform does not match the initialized workflow platform"
        if payload.get("backend_state") != "COMPLETE":
            return "backend_state is not COMPLETE"
        if not isinstance(payload.get("job_id"), str) or not payload["job_id"].strip():
            return "job_id is missing"
        if not isinstance(payload.get("backend_ref"), str) or not payload["backend_ref"].strip():
            return "backend_ref is missing"
        for field in ("request_sha256", "job_binding_sha256"):
            digest = payload.get(field)
            if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                return f"{field} is missing or invalid"
        results_scope = payload.get("results_scope")
        if not isinstance(results_scope, str) or not results_scope.strip():
            return "results_scope is missing"
        if payload.get("freshness_contract") == "remote-mirror-with-delete-before-submit":
            if not remote_freshness_attested(payload):
                return "remote output-absence staging receipt is missing"
        elif (
            payload.get("freshness_contract") != "local-mtime-after-prepare"
            or payload.get("staging_receipt_sha256") is not None
            or platform not in {"docker", "virtualenv"}
        ):
            return "local freshness evidence is inconsistent"
    elif expected_platform != "docker":
        return "legacy container evidence is valid only for Docker workflows"
    return None


def remote_freshness_attested(payload: dict[str, Any]) -> bool:
    """True only for current remote evidence with a digest-bound receipt."""
    digest = payload.get("staging_receipt_sha256")
    return (
        payload.get("schema_version") == "2"
        and payload.get("platform") in {"docker", "slurm", "kubernetes", "brev"}
        and payload.get("freshness_contract")
        == "remote-mirror-with-delete-before-submit"
        and isinstance(digest, str)
        and re.fullmatch(r"[0-9a-f]{64}", digest) is not None
    )
