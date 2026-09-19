#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render one producer action request as a single-pod Kubernetes Job.

The data mover first mirrors each distinct request ``mounts[].source`` into a
job-owned directory on one PVC.  Its receipt records the relative PVC path for
each source.  This renderer preserves every producer-declared target and access
mode, including multiple target aliases for the same source, by mounting that
PVC subPath once per target.

Credentials are deliberately not accepted as values.  Each ``forward_env``
name is projected from the same key of an optional, pre-created Secret; keys
that the producer did not approve are never imported.  Registry authentication
is an independent optional image-pull Secret.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import posixpath
import re
import shlex
import stat
import sys
import tempfile
from typing import Any


REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
DEFAULT_TEMPLATE = REPO_ROOT / "templates/k8s/action-job.yaml.tmpl"
MARKER_RE = re.compile(r"@@[A-Z][A-Z0-9_]*@@")
ENV_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
DNS_LABEL_RE = re.compile(r"[a-z0-9](?:[-a-z0-9]*[a-z0-9])?")
DNS_SUBDOMAIN_RE = re.compile(
    r"[a-z0-9](?:[-a-z0-9.]*[a-z0-9])?"
)
QUANTITY_RE = re.compile(r"[1-9][0-9]*(?:Ki|Mi|Gi|Ti|Pi|Ei)")
MAX_JOB_NAME = 52
REQUEST_SCHEMA_VERSION = "1"
CONFIG_FORMATS = {"json": "json", "toml": "toml", "yaml": "yaml"}
SHELL_META = ("\n", ";", "&&", "||", "$(", "`", "|", ">", "<")


class RenderError(ValueError):
    """The action request or staging receipt cannot be rendered safely."""


def _load_object(path: pathlib.Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RenderError(f"cannot read {label} JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RenderError(f"{label} JSON root must be an object")
    return payload


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise RenderError(f"{label} must be a non-empty string without NUL")
    return value


def _string_value(value: Any, label: str) -> str:
    if not isinstance(value, str) or "\x00" in value:
        raise RenderError(f"{label} must be a string without NUL")
    return value


def _dns_name(value: str, label: str, *, label_only: bool = False) -> str:
    pattern = DNS_LABEL_RE if label_only else DNS_SUBDOMAIN_RE
    maximum = 63 if label_only else 253
    if len(value) > maximum or pattern.fullmatch(value) is None:
        kind = "DNS label" if label_only else "DNS subdomain"
        raise RenderError(f"{label} must be a valid Kubernetes {kind}")
    return value


def kubernetes_job_name(job_id: str) -> str:
    """Map an immutable job-record id to a stable Kubernetes-safe name.

    Job-record components may contain underscores, uppercase letters, dots, or
    enough text to exceed Kubernetes' practical Job-name length. Preserve an
    already-safe short id; otherwise append a digest so normalization and
    truncation cannot merge two record ids into one backend object.
    """
    original = _nonempty_string(job_id, "job id")
    if len(original) <= MAX_JOB_NAME and DNS_LABEL_RE.fullmatch(original):
        return original
    slug = re.sub(r"[^a-z0-9]+", "-", original.lower()).strip("-")
    if not slug:
        slug = "tao-job"
    digest = hashlib.sha256(original.encode("utf-8")).hexdigest()[:10]
    prefix = slug[: MAX_JOB_NAME - len(digest) - 1].rstrip("-") or "tao-job"
    return f"{prefix}-{digest}"


def _absolute_clean_path(value: Any, label: str) -> str:
    path = _nonempty_string(value, label)
    if (
        not path.startswith("/")
        or path == "/"
        or "\\" in path
        or any(ord(ch) < 32 for ch in path)
    ):
        raise RenderError(f"{label} must be an absolute non-root POSIX path")
    normalized = posixpath.normpath(path)
    if normalized != path or "//" in path:
        raise RenderError(f"{label} must be normalized and contain no traversal")
    return path


def _safe_sub_path(value: Any, label: str) -> str:
    path = _nonempty_string(value, label)
    if path.startswith("/") or "\\" in path or any(ord(ch) < 32 for ch in path):
        raise RenderError(f"{label} must be a safe relative POSIX path")
    normalized = posixpath.normpath(path)
    if normalized != path or path in {".", ".."} or path.startswith("../"):
        raise RenderError(f"{label} must be normalized and contain no traversal")
    return path


def _toml_key(value: Any) -> str:
    key = _nonempty_string(value, "TOML key")
    if re.fullmatch(r"[A-Za-z0-9_-]+", key):
        return key
    return json.dumps(key)


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RenderError("TOML config values must be finite")
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{ " + ", ".join(
            f"{_toml_key(key)} = {_toml_value(item)}"
            for key, item in sorted(value.items())
        ) + " }"
    raise RenderError(f"no TOML representation for {type(value).__name__}")


def _toml_document(spec: dict[str, Any], prefix: tuple[str, ...] = ()) -> str:
    scalars: list[str] = []
    tables: list[tuple[str, dict[str, Any]]] = []
    for raw_key in sorted(spec):
        key = _toml_key(raw_key)
        value = spec[raw_key]
        if isinstance(value, dict):
            tables.append((key, value))
        else:
            scalars.append(f"{key} = {_toml_value(value)}")
    sections: list[str] = []
    if scalars:
        sections.append("\n".join(scalars))
    for key, value in tables:
        name = ".".join((*prefix, key))
        body = _toml_document(value, (*prefix, key)).rstrip("\n")
        sections.append(f"[{name}]" + (f"\n{body}" if body else ""))
    return "\n\n".join(sections) + "\n"


def _config_content(bundle: dict[str, Any]) -> tuple[str, str, str]:
    spec = bundle.get("spec")
    if not isinstance(spec, dict):
        raise RenderError("spec_bundle.spec must be a nested object for mode=config")
    raw_format = bundle.get("config_format")
    if raw_format not in CONFIG_FORMATS:
        raise RenderError(
            "spec_bundle.config_format must be one of json, toml, or yaml"
        )
    extension = CONFIG_FORMATS[raw_format]
    try:
        if raw_format == "toml":
            content = _toml_document(spec)
        else:
            # JSON is valid YAML. Keeping both formats on the stdlib serializer
            # avoids making Kubernetes submission depend on a host YAML package.
            content = json.dumps(
                spec, allow_nan=False, indent=2, sort_keys=True
            ) + "\n"
    except (TypeError, ValueError) as exc:
        raise RenderError(f"spec_bundle.spec is not {raw_format}-serializable: {exc}") from exc
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return content, extension, digest


def _read_regular_file(path: pathlib.Path, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RenderError(f"cannot open {label} {path}: {exc}") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise RenderError(f"{label} must be a regular non-symlink file: {path}")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            return handle.read()
    except OSError as exc:
        raise RenderError(f"cannot read {label} {path}: {exc}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def materialize_config(
    request: dict[str, Any], output_dir: pathlib.Path
) -> pathlib.Path:
    """Write one content-addressed config file for later PVC staging."""
    if request.get("schema_version") != REQUEST_SCHEMA_VERSION:
        raise RenderError(
            f"request.schema_version must be {REQUEST_SCHEMA_VERSION!r}"
        )
    if request.get("platform") != "kubernetes":
        raise RenderError("request.platform must be 'kubernetes'")
    bundle = request.get("spec_bundle")
    if not isinstance(bundle, dict):
        raise RenderError("request.spec_bundle must be an object")
    if bundle.get("mode") != "config":
        raise RenderError("materialize-config requires spec_bundle.mode=config")
    if "args" in bundle:
        raise RenderError("spec_bundle.mode=config must not carry args")
    content, extension, digest = _config_content(bundle)

    raw_output_dir = output_dir.expanduser()
    if raw_output_dir.is_symlink():
        raise RenderError("config output directory must not be a symlink")
    try:
        raw_output_dir.mkdir(parents=True, exist_ok=True)
        root = raw_output_dir.resolve(strict=True)
    except OSError as exc:
        raise RenderError(f"cannot prepare config output directory: {exc}") from exc
    if root == pathlib.Path(root.anchor) or not root.is_dir():
        raise RenderError("config output directory must be a non-root directory")

    destination = root / f"tao-action-config-{digest}.{extension}"
    encoded = content.encode("utf-8")
    if destination.exists() or destination.is_symlink():
        if _read_regular_file(destination, "materialized config") != encoded:
            raise RenderError(f"materialized config content mismatch: {destination}")
        return destination

    temporary: pathlib.Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=destination.name + ".",
            suffix=".tmp",
            dir=root,
            delete=False,
        ) as handle:
            temporary = pathlib.Path(handle.name)
            os.fchmod(handle.fileno(), 0o600)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # link() publishes the complete file without ever replacing an
            # existing path. Concurrent materializers either create the same
            # content-addressed file or verify the winner below.
            os.link(temporary, destination)
        except FileExistsError:
            if _read_regular_file(destination, "materialized config") != encoded:
                raise RenderError(
                    f"materialized config content mismatch: {destination}"
                )
    except OSError as exc:
        raise RenderError(f"cannot materialize config in {root}: {exc}") from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    return destination


def _is_shell_script(command: str) -> bool:
    return any(marker in command for marker in SHELL_META)


def _command_bundle(
    request: dict[str, Any], config_path: str | None
) -> tuple[list[str], list[str], str, int]:
    bundle = request.get("spec_bundle")
    if not isinstance(bundle, dict):
        raise RenderError("request.spec_bundle must be an object")
    mode = bundle.get("mode")
    command_text = _nonempty_string(bundle.get("command"), "spec_bundle.command")
    if mode == "args":
        if "spec" in bundle or "config_format" in bundle:
            raise RenderError(
                "spec_bundle.mode=args must not carry spec or config_format"
            )
        if config_path is not None:
            raise RenderError("--config-source must be omitted for spec_bundle.mode=args")
        raw_args = bundle.get("args")
        if not isinstance(raw_args, list):
            raise RenderError("spec_bundle.args must be an array")
        args = [
            _string_value(value, f"spec_bundle.args[{index}]")
            for index, value in enumerate(raw_args)
        ]
    elif mode == "config":
        if "args" in bundle:
            raise RenderError("spec_bundle.mode=config must not carry args")
        _config_content(bundle)
        if config_path is None:
            raise RenderError(
                "spec_bundle.mode=config requires --config-source; run "
                "materialize-config, stage the returned file, then render"
            )
        if "{config_path}" not in command_text:
            raise RenderError(
                "spec_bundle.mode=config requires {config_path} in command"
            )
        command_text = command_text.replace("{config_path}", config_path)
        args = []
    else:
        raise RenderError("spec_bundle.mode must be 'args' or 'config'")

    if mode == "config" and _is_shell_script(command_text):
        command = ["/bin/sh", "-c"]
        args = [command_text, "tao-action", *args]
    else:
        try:
            command = shlex.split(command_text)
        except ValueError as exc:
            raise RenderError(f"spec_bundle.command cannot be parsed: {exc}") from exc
        if not command:
            raise RenderError("spec_bundle.command must contain an executable")
    image = _nonempty_string(bundle.get("image"), "spec_bundle.image")
    workload_image = request.get("workload_image")
    if workload_image is not None and workload_image != image:
        raise RenderError("request.workload_image must equal spec_bundle.image")
    shape = bundle.get("compute_shape")
    if not isinstance(shape, dict):
        raise RenderError("spec_bundle.compute_shape must be an object")
    gpus = shape.get("gpus")
    nodes = shape.get("nodes")
    if not isinstance(gpus, int) or isinstance(gpus, bool) or gpus < 0:
        raise RenderError("spec_bundle.compute_shape.gpus must be a non-negative integer")
    if nodes != 1:
        raise RenderError("single-pod action rendering requires compute_shape.nodes=1")
    return command, args, image, gpus


def _staged_sources(payload: dict[str, Any]) -> dict[str, str]:
    if payload.get("schema_version") != "1":
        raise RenderError("staging map schema_version must be '1'")
    rows = payload.get("sources")
    if not isinstance(rows, list) or not rows:
        raise RenderError("staging map sources must be a non-empty array")
    result: dict[str, str] = {}
    sub_paths: dict[str, str] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RenderError(f"staging map sources[{index}] must be an object")
        source = _absolute_clean_path(row.get("source"), f"staging sources[{index}].source")
        sub_path = _safe_sub_path(row.get("sub_path"), f"staging sources[{index}].sub_path")
        if source in result:
            raise RenderError(f"staging map repeats source: {source}")
        if sub_path in sub_paths:
            raise RenderError(
                "staging map assigns one PVC subPath to distinct sources: "
                f"{sub_paths[sub_path]} and {source}"
            )
        result[source] = sub_path
        sub_paths[sub_path] = source
    return result


def _mounts(
    request: dict[str, Any],
    staged: dict[str, str],
    config_mount: tuple[str, str] | None = None,
) -> tuple[list[dict[str, Any]], list[tuple[pathlib.PurePosixPath, bool]]]:
    rows = request.get("mounts")
    if not isinstance(rows, list) or not rows:
        raise RenderError("request.mounts must be a non-empty array")
    rendered: list[dict[str, Any]] = [
        {"name": "dshm", "mountPath": "/dev/shm", "readOnly": False}
    ]
    used_sources: set[str] = set()
    targets: set[str] = {"/dev/shm"}
    source_modes: list[tuple[pathlib.PurePosixPath, bool]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RenderError(f"request.mounts[{index}] must be an object")
        source = _absolute_clean_path(row.get("source"), f"mounts[{index}].source")
        target = _absolute_clean_path(row.get("target"), f"mounts[{index}].target")
        read_only = row.get("read_only")
        if not isinstance(read_only, bool):
            raise RenderError(f"mounts[{index}].read_only must be boolean")
        if source not in staged:
            raise RenderError(f"mount source has no staged PVC subPath: {source}")
        if target in targets:
            raise RenderError(f"request repeats Kubernetes mount target: {target}")
        targets.add(target)
        used_sources.add(source)
        source_modes.append((pathlib.PurePosixPath(source), read_only))
        rendered.append(
            {
                "name": "workspace",
                "mountPath": target,
                "subPath": staged[source],
                "readOnly": read_only,
            }
        )
    if config_mount is not None:
        source, target = config_mount
        if source in used_sources:
            raise RenderError(
                "materialized config must have its own staged file source"
            )
        if source not in staged:
            raise RenderError(
                f"materialized config has no staged PVC subPath: {source}"
            )
        if target in targets:
            raise RenderError(
                f"materialized config repeats Kubernetes mount target: {target}"
            )
        used_sources.add(source)
        targets.add(target)
        rendered.append(
            {
                "name": "workspace",
                "mountPath": target,
                "subPath": staged[source],
                "readOnly": True,
            }
        )
    extras = sorted(set(staged) - used_sources)
    if extras:
        raise RenderError("staging map contains undeclared mount sources: " + ", ".join(extras))
    return rendered, source_modes


def _config_mount(
    request: dict[str, Any],
    staged: dict[str, str],
    config_source: pathlib.Path | None,
) -> tuple[str | None, tuple[str, str] | None]:
    bundle = request.get("spec_bundle")
    if not isinstance(bundle, dict):
        raise RenderError("request.spec_bundle must be an object")
    mode = bundle.get("mode")
    if mode != "config":
        if config_source is not None:
            raise RenderError("--config-source is valid only for spec_bundle.mode=config")
        return None, None
    if config_source is None:
        return None, None

    lexical = config_source.expanduser()
    source = _absolute_clean_path(str(lexical), "config source")
    path = pathlib.Path(source)
    content, extension, digest = _config_content(bundle)
    if _read_regular_file(path, "config source") != content.encode("utf-8"):
        raise RenderError(
            "config source content does not match spec_bundle.spec; rerun "
            "materialize-config and restage the returned file"
        )
    if source not in staged:
        raise RenderError(f"materialized config has no staged PVC subPath: {source}")
    target = f"/tao-action-config/spec-{digest}.{extension}"
    return target, (source, target)


def _require_writable_outputs(
    request: dict[str, Any], source_modes: list[tuple[pathlib.PurePosixPath, bool]]
) -> None:
    outputs = request.get("fresh_outputs", [])
    if not isinstance(outputs, list):
        raise RenderError("request.fresh_outputs must be an array")
    for index, raw in enumerate(outputs):
        output = pathlib.PurePosixPath(
            _absolute_clean_path(raw, f"fresh_outputs[{index}]")
        )
        writable = False
        for source, read_only in source_modes:
            try:
                output.relative_to(source)
            except ValueError:
                continue
            if not read_only:
                writable = True
                break
        if not writable:
            raise RenderError(
                f"fresh output is not covered by a writable declared mount: {output}"
            )


def _environment(
    request: dict[str, Any], credential_secret: str | None
) -> list[dict[str, Any]]:
    raw_environment = request.get("environment", {})
    if not isinstance(raw_environment, dict):
        raise RenderError("request.environment must be an object")
    environment: list[dict[str, Any]] = []
    for name in sorted(raw_environment):
        if ENV_NAME_RE.fullmatch(name) is None:
            raise RenderError(f"invalid environment variable name: {name!r}")
        value = _string_value(raw_environment[name], f"environment.{name}")
        environment.append({"name": name, "value": value})

    raw_forward = request.get("forward_env", [])
    if not isinstance(raw_forward, list):
        raise RenderError("request.forward_env must be an array")
    forward: list[str] = []
    for index, raw in enumerate(raw_forward):
        name = _nonempty_string(raw, f"forward_env[{index}]")
        if ENV_NAME_RE.fullmatch(name) is None:
            raise RenderError(f"invalid forwarded environment variable name: {name!r}")
        if name in forward:
            raise RenderError(f"request.forward_env repeats {name}")
        if name in raw_environment:
            raise RenderError(f"credential {name} must not be present in request.environment")
        forward.append(name)
    if forward and credential_secret is None:
        raise RenderError(
            "request.forward_env is non-empty but no --credential-secret was supplied"
        )
    if not forward and credential_secret is not None:
        raise RenderError(
            "--credential-secret must be omitted when request.forward_env is empty"
        )
    if credential_secret is not None:
        environment.extend(
            {
                "name": name,
                "valueFrom": {
                    "secretKeyRef": {"name": credential_secret, "key": name}
                },
            }
            for name in forward
        )
    return environment


def render_action_job(
    request: dict[str, Any],
    staging_map: dict[str, Any],
    *,
    job_id: str,
    namespace: str,
    pvc_claim: str,
    credential_secret: str | None = None,
    image_pull_secret: str | None = None,
    config_source: pathlib.Path | None = None,
    ttl_seconds: int = 3600,
    shm_size: str = "16Gi",
    template_path: pathlib.Path = DEFAULT_TEMPLATE,
) -> str:
    """Validate inputs and return a complete Kubernetes Job manifest."""
    if request.get("schema_version") != REQUEST_SCHEMA_VERSION:
        raise RenderError(
            f"request.schema_version must be {REQUEST_SCHEMA_VERSION!r}"
        )
    if request.get("platform") != "kubernetes":
        raise RenderError("request.platform must be 'kubernetes'")
    job_id = _nonempty_string(job_id, "job id")
    job_name = kubernetes_job_name(job_id)
    namespace = _dns_name(namespace, "namespace", label_only=True)
    pvc_claim = _dns_name(pvc_claim, "PVC claim")
    if credential_secret is not None:
        credential_secret = _dns_name(credential_secret, "credential secret")
    if image_pull_secret is not None:
        image_pull_secret = _dns_name(image_pull_secret, "image pull secret")
    if (
        not isinstance(ttl_seconds, int)
        or isinstance(ttl_seconds, bool)
        or not 0 <= ttl_seconds <= 604800
    ):
        raise RenderError("ttl_seconds must be an integer from 0 through 604800")
    if QUANTITY_RE.fullmatch(shm_size) is None:
        raise RenderError("shm_size must be a positive binary Kubernetes quantity such as 16Gi")

    staged = _staged_sources(staging_map)
    config_path, config_mount = _config_mount(request, staged, config_source)
    command, args, image, gpus = _command_bundle(request, config_path)
    volume_mounts, source_modes = _mounts(request, staged, config_mount)
    _require_writable_outputs(request, source_modes)
    environment = _environment(request, credential_secret)
    image_pull_secrets = (
        [{"name": image_pull_secret}] if image_pull_secret is not None else []
    )

    values: dict[str, Any] = {
        "JOB_NAME_JSON": job_name,
        "JOB_ID_JSON": job_id,
        "NAMESPACE_JSON": namespace,
        "TTL_SECONDS_JSON": ttl_seconds,
        "IMAGE_PULL_SECRETS_JSON": image_pull_secrets,
        "IMAGE_JSON": image,
        "COMMAND_JSON": command,
        "ARGS_JSON": args,
        "NUM_GPUS_JSON": str(gpus),
        "ENV_JSON": environment,
        "VOLUME_MOUNTS_JSON": volume_mounts,
        "SHM_SIZE_JSON": shm_size,
        "PVC_CLAIM_JSON": pvc_claim,
    }
    try:
        rendered = template_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RenderError(f"cannot read Kubernetes action template: {exc}") from exc
    for marker, value in values.items():
        rendered = rendered.replace(f"@@{marker}@@", json.dumps(value, separators=(",", ":")))
    remaining = MARKER_RE.findall(rendered)
    if remaining:
        raise RenderError(f"unresolved template markers: {sorted(set(remaining))}")
    return rendered


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    name = commands.add_parser(
        "name", help="print the deterministic Kubernetes name for a job-record id"
    )
    name.add_argument("--job-id", required=True)

    materialize = commands.add_parser(
        "materialize-config",
        help="write a content-addressed mode=config file for PVC staging",
    )
    materialize.add_argument("--request", required=True, type=pathlib.Path)
    materialize.add_argument("--output-dir", required=True, type=pathlib.Path)

    render = commands.add_parser("render", help="render a complete Job manifest")
    render.add_argument("--request", required=True, type=pathlib.Path)
    render.add_argument("--staging-map", required=True, type=pathlib.Path)
    render.add_argument("--job-id", required=True)
    render.add_argument(
        "--namespace", default=os.environ.get("TAO_K8S_NAMESPACE", "default")
    )
    render.add_argument("--pvc-claim", required=True)
    render.add_argument("--credential-secret")
    render.add_argument("--image-pull-secret")
    render.add_argument("--config-source", type=pathlib.Path)
    render.add_argument("--ttl-seconds", type=int, default=3600)
    render.add_argument("--shm-size", default="16Gi")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.operation == "name":
            print(kubernetes_job_name(args.job_id))
            return 0
        request = _load_object(args.request, "action request")
        if args.operation == "materialize-config":
            print(materialize_config(request, args.output_dir))
            return 0
        staging_map = _load_object(args.staging_map, "staging map")
        manifest = render_action_job(
            request,
            staging_map,
            job_id=args.job_id,
            namespace=args.namespace,
            pvc_claim=args.pvc_claim,
            credential_secret=args.credential_secret,
            image_pull_secret=args.image_pull_secret,
            config_source=args.config_source,
            ttl_seconds=args.ttl_seconds,
            shm_size=args.shm_size,
        )
    except RenderError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    sys.stdout.write(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
