#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute one prepared PAS TAO CLI action inside the selected virtualenv."""

from __future__ import annotations

import hashlib
import os
import pathlib
import re
import stat
import sys
import tempfile

import yaml

from run_deft_action import _load_request, _request_path_from_payload


REQUEST_SCHEMA_VERSION = "1"
WORKFLOW = "tao-run-deft-pas"
_BASE_MOUNT_TARGETS = frozenset({"/results", "/specs", "/patches", "/cache"})
_BENIGN_RUNTIME_ENV = frozenset(
    {
        "CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NCCL_DEBUG",
        "NCCL_SOCKET_IFNAME",
        "NCCL_IB_DISABLE",
        "NCCL_P2P_DISABLE",
        "NCCL_SHM_DISABLE",
        "NCCL_ASYNC_ERROR_HANDLING",
    }
)
_CERTIFICATE_ENV = frozenset(
    {"SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"}
)
_SYSTEM_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


def _absolute_normalized(value: str) -> bool:
    return (
        value.startswith("/")
        and value != "/"
        and os.path.normpath(value) == value
    )


def _request_contract(
    request_path: pathlib.Path, request: object
) -> tuple[dict, dict[str, str]]:
    """Validate the shim's required request subset without workload jsonschema."""
    if not isinstance(request, dict):
        raise ValueError("action request root must be an object")
    if request.get("schema_version") != REQUEST_SCHEMA_VERSION:
        raise ValueError("unsupported action request schema_version")
    if request.get("workflow") != WORKFLOW:
        raise ValueError("action request workflow mismatch")
    if request.get("platform") != "virtualenv":
        raise ValueError("action request platform must be virtualenv")
    name = request.get("name")
    stage_dir = request.get("stage_dir")
    if not isinstance(name, str) or not re.fullmatch(
        r"[a-z0-9][a-z0-9_.-]*", name
    ):
        raise ValueError("action request name is invalid")
    if not isinstance(stage_dir, str) or not _absolute_normalized(stage_dir):
        raise ValueError("action request stage_dir must be an absolute normalized path")
    stage = pathlib.Path(stage_dir)
    if stage.resolve() != stage:
        raise ValueError("action request stage_dir must not traverse symlinks")
    attempt = request.get("attempt")
    if (
        not isinstance(attempt, int)
        or isinstance(attempt, bool)
        or attempt not in {1, 2}
    ):
        raise ValueError("action request attempt must be 1 or 2")
    expected_request = _request_path_from_payload(request)
    if request_path != expected_request:
        raise ValueError(f"action request must remain at {expected_request}")

    virtualenv = request.get("virtualenv")
    record_image = request.get("record_image")
    if not isinstance(virtualenv, str) or not _absolute_normalized(virtualenv):
        raise ValueError("action request virtualenv must be an absolute normalized path")
    if record_image != str(pathlib.Path(virtualenv) / "bin" / "python"):
        raise ValueError("action request record_image does not bind its virtualenv")

    bundle = request.get("spec_bundle")
    if not isinstance(bundle, dict):
        raise ValueError("action request spec_bundle must be an object")
    bundle_command = bundle.get("command")
    bundle_args = bundle.get("args")
    if (
        bundle_command not in {"clip", "embedding", "tmm"}
        or not isinstance(bundle_args, list)
        or any(not isinstance(item, str) for item in bundle_args)
    ):
        raise ValueError("action request bundle command/args are invalid")
    expected_kind = "pyt" if bundle_command == "clip" else "ds"
    if request.get("image_kind") != expected_kind:
        raise ValueError("action request image_kind does not own the TAO CLI")

    mounts = request.get("mounts")
    if not isinstance(mounts, list) or not mounts:
        raise ValueError("action request has no mount contract")
    aliases: dict[str, str] = {}
    for index, item in enumerate(mounts):
        if not isinstance(item, dict) or set(item) != {"source", "target", "read_only"}:
            raise ValueError(f"action request mount {index} is malformed")
        source = item["source"]
        target = item["target"]
        if (
            not isinstance(source, str)
            or not _absolute_normalized(source)
            or pathlib.Path(source).resolve() != pathlib.Path(source)
        ):
            raise ValueError(f"action request mount {index} source is unsafe")
        if not isinstance(target, str) or not _absolute_normalized(target):
            raise ValueError(f"action request mount {index} target is unsafe")
        if not isinstance(item["read_only"], bool):
            raise ValueError(f"action request mount {index} read_only must be boolean")
        if target in aliases:
            raise ValueError(f"action request mount target is duplicated: {target}")
        aliases[target] = source
    missing = sorted(_BASE_MOUNT_TARGETS - aliases.keys())
    if missing:
        raise ValueError("action request lacks required mount targets: " + ", ".join(missing))
    results_dir = request.get("results_dir")
    if (
        not isinstance(results_dir, str)
        or not _absolute_normalized(results_dir)
        or pathlib.Path(results_dir).resolve() != pathlib.Path(results_dir)
    ):
        raise ValueError("action request results_dir must be an absolute safe path")
    if aliases["/results"] != results_dir:
        raise ValueError("action request /results mount does not bind results_dir")
    if pathlib.Path(results_dir) not in stage.parents:
        raise ValueError("action request stage_dir must remain below results_dir")
    return bundle, aliases


def _translate(value: str, aliases: dict[str, str]) -> str:
    """Translate a compute-frame path or key=value token to a local path."""
    prefix = ""
    candidate = value
    if "=" in value:
        key, candidate = value.split("=", 1)
        prefix = key + "="
    for target in sorted(aliases, key=len, reverse=True):
        if candidate == target or candidate.startswith(target + "/"):
            candidate = aliases[target] + candidate[len(target):]
            break
    return prefix + candidate


def _translate_tree(value, aliases: dict[str, str]):
    if isinstance(value, str):
        return _translate(value, aliases)
    if isinstance(value, list):
        return [_translate_tree(item, aliases) for item in value]
    if isinstance(value, dict):
        return {key: _translate_tree(item, aliases) for key, item in value.items()}
    return value


def _certificate_path(value: str, name: str) -> str:
    """Accept only an existing normalized certificate file/directory path."""
    if not _absolute_normalized(value):
        raise ValueError(f"{name} must be an absolute normalized path")
    path = pathlib.Path(value)
    if path.resolve() != path or path.is_symlink() or not path.exists():
        raise ValueError(f"{name} must be an existing non-symlink path")
    if name == "SSL_CERT_DIR" and not path.is_dir():
        raise ValueError("SSL_CERT_DIR must identify a directory")
    if name != "SSL_CERT_DIR" and not path.is_file():
        raise ValueError(f"{name} must identify a regular file")
    return value


def _open_verified_entrypoint(
    executable: pathlib.Path, expected_sha256: object, python: pathlib.Path
) -> tuple[int, str]:
    """Pin the prepared console script and verify its content before exec."""
    if not isinstance(expected_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_sha256
    ):
        raise ValueError("action request lacks the approved virtualenv entrypoint digest")
    try:
        fd = os.open(
            executable,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise ValueError(f"cannot open approved TAO CLI without symlink traversal: {exc}") from exc
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o111 == 0:
            raise ValueError("approved TAO CLI is not an executable regular file")
        digest = hashlib.sha256()
        first_line = bytearray()
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            if b"\n" not in first_line:
                first_line.extend(chunk[: 4096 - len(first_line)])
        if digest.hexdigest() != expected_sha256:
            raise ValueError("approved TAO CLI changed after action preparation")
        shebang = bytes(first_line).splitlines()[0] if first_line else b""
        if shebang != f"#!{python}".encode("utf-8"):
            raise ValueError("approved TAO CLI shebang no longer binds the selected profile")
        pinned = f"/proc/self/fd/{fd}"
        if not pathlib.Path(pinned).exists():
            raise ValueError("pinned virtualenv execution requires Linux procfs")
        os.set_inheritable(fd, True)
        return fd, pinned
    except Exception:
        os.close(fd)
        raise


def _execution_environment(
    request: dict,
    aliases: dict[str, str],
    approved_virtualenv: str,
) -> dict[str, str]:
    """Build the exact child environment without inheriting ambient secrets."""
    configured = request.get("environment")
    if not isinstance(configured, dict) or set(configured) != {
        "HOME",
        "PYTHONPATH",
        "HF_HOME",
        "XDG_CACHE_HOME",
    }:
        raise ValueError("action request environment contract is malformed")
    environment = {
        name: _translate(value, aliases)
        for name, value in configured.items()
        if isinstance(value, str)
    }
    if set(environment) != set(configured):
        raise ValueError("action request environment values must be strings")
    expected_patch = pathlib.Path(__file__).resolve().parent.parent / "patches"
    if environment["PYTHONPATH"] != str(expected_patch):
        raise ValueError("action request PYTHONPATH must bind the packaged patches")
    environment.update(
        {
            "VIRTUAL_ENV": approved_virtualenv,
            "PATH": f"{approved_virtualenv}/bin:{_SYSTEM_PATH}",
            "PYTHONUNBUFFERED": "1",
        }
    )
    for name in _BENIGN_RUNTIME_ENV:
        value = os.environ.get(name)
        if value is not None:
            environment[name] = value
    for name in _CERTIFICATE_ENV:
        value = os.environ.get(name)
        if value is not None:
            environment[name] = _certificate_path(value, name)
    forward = request.get("forward_env")
    if not isinstance(forward, list) or any(name != "HF_TOKEN" for name in forward):
        raise ValueError("action request forward_env is not allowlisted")
    for name in forward:
        value = os.environ.get(name)
        if not value:
            raise ValueError(f"approved forwarded variable {name} is unset")
        environment[name] = value
    return environment


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if len(raw) < 3 or raw[0] != "--request" or "--" not in raw[2:]:
        print(
            "run_deft_cli: expected --request ACTION_JSON -- COMMAND ...",
            file=sys.stderr,
        )
        return 2
    separator = raw.index("--", 2)
    if separator != 2:
        print("run_deft_cli: unexpected arguments before --", file=sys.stderr)
        return 2
    supplied_request = pathlib.Path(raw[1]).expanduser()
    request_path = pathlib.Path(os.path.abspath(supplied_request))
    if not request_path.is_file() or request_path.is_symlink():
        print("run_deft_cli: action request is missing or unsafe", file=sys.stderr)
        return 2
    command = raw[separator + 1:]
    if not command:
        print("run_deft_cli: command after -- is required", file=sys.stderr)
        return 2
    if "/" in command[0] or command[0] not in {"clip", "embedding", "tmm"}:
        print("run_deft_cli: executable must be clip, embedding, or tmm", file=sys.stderr)
        return 2
    venv = os.environ.get("VIRTUAL_ENV")
    if not venv:
        print("run_deft_cli: VIRTUAL_ENV is not set by the platform runner", file=sys.stderr)
        return 2
    try:
        request_path, request = _load_request(request_path)
        bundle, aliases = _request_contract(request_path, request)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"run_deft_cli: {exc}", file=sys.stderr)
        return 2
    expected = [bundle["command"], *bundle["args"]]
    if command != expected:
        print(
            "run_deft_cli: command does not match the prepared virtualenv action",
            file=sys.stderr,
        )
        return 2
    approved_virtualenv = request.get("virtualenv")
    if (
        not isinstance(approved_virtualenv, str)
        or not approved_virtualenv.strip()
        or pathlib.Path(venv).expanduser().resolve()
        != pathlib.Path(approved_virtualenv).expanduser().resolve()
    ):
        print(
            "run_deft_cli: VIRTUAL_ENV does not match the action's approved profile",
            file=sys.stderr,
        )
        return 2
    translated = [_translate(item, aliases) for item in command]
    if "-e" in translated:
        index = translated.index("-e") + 1
        if index >= len(translated):
            print("run_deft_cli: -e is missing its config path", file=sys.stderr)
            return 2
        original_source = command[index]
        source = pathlib.Path(translated[index])
        approved_specs = pathlib.Path(aliases["/specs"])
        if (
            not original_source.startswith("/specs/")
            or not source.is_absolute()
            or source.resolve() != source
            or approved_specs not in source.parents
            or source.is_symlink()
        ):
            print(
                "run_deft_cli: action config must be a regular file below the "
                "approved /specs mount",
                file=sys.stderr,
            )
            return 2
        try:
            spec = yaml.safe_load(source.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            print(f"run_deft_cli: cannot translate action config {source}: {exc}", file=sys.stderr)
            return 2
        destination = request_path.parent / f"{request['name']}.virtualenv.yaml"
        if destination.parent != request_path.parent:
            print(
                "run_deft_cli: translated config destination escaped the approved stage",
                file=sys.stderr,
            )
            return 2
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=request_path.parent
        )
        temporary = pathlib.Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                yaml.safe_dump(
                    _translate_tree(spec, aliases),
                    handle,
                    sort_keys=False,
                    allow_unicode=True,
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        translated[index] = str(destination)
    approved_python = pathlib.Path(approved_virtualenv).resolve() / "bin" / "python"
    executable = pathlib.Path(approved_virtualenv).resolve() / "bin" / translated[0]
    try:
        executable_fd, pinned_executable = _open_verified_entrypoint(
            executable,
            request.get("virtualenv_entrypoint_sha256"),
            approved_python,
        )
    except ValueError as exc:
        print(f"run_deft_cli: {exc}", file=sys.stderr)
        return 127
    try:
        environment = _execution_environment(request, aliases, approved_virtualenv)
    except ValueError as exc:
        print(f"run_deft_cli: {exc}", file=sys.stderr)
        return 2
    try:
        os.execve(pinned_executable, [str(executable), *translated[1:]], environment)
    except OSError as exc:
        os.close(executable_fd)
        print(f"run_deft_cli: cannot exec pinned TAO CLI: {exc}", file=sys.stderr)
        return 127
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
