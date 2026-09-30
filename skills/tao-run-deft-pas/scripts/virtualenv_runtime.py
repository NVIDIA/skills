# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Immutable runtime contract for PAS's two virtualenv execution profiles."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import subprocess
from typing import Any

import jsonschema


PROFILE_NAMES = ("pyt", "ds")
REFERENCE_DIR = pathlib.Path(__file__).resolve().parent.parent / "references"
MANIFEST_PATH = REFERENCE_DIR / "virtualenv-runtime-manifest.json"
MANIFEST_SCHEMA_PATH = REFERENCE_DIR / "virtualenv-runtime-manifest.schema.json"


_ENV_PROBE = r"""
import importlib
import importlib.metadata as metadata
import json
import pathlib
import platform
import re
import sys

contract = json.loads(sys.argv[1])
errors = []
facts = {
    "implementation": sys.implementation.name,
    "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
    "machine": platform.machine(),
    "glibc": platform.libc_ver()[1],
    "prefix": str(pathlib.Path(sys.prefix).resolve()),
    "base_prefix": str(pathlib.Path(sys.base_prefix).resolve()),
    "distributions": {},
    "entrypoints": {},
    "imports": [],
}

def normalized(name):
    return re.sub(r"[-_.]+", "-", name).lower()

for name, expected_version in contract["distributions"].items():
    try:
        actual_version = metadata.version(name)
    except metadata.PackageNotFoundError:
        errors.append(f"missing distribution {name}=={expected_version}")
        continue
    facts["distributions"][name] = actual_version
    if actual_version != expected_version:
        errors.append(
            f"distribution {name} must be {expected_version}, got {actual_version}"
        )

console_scripts = list(metadata.entry_points(group="console_scripts"))
for name, expected in contract["entrypoints"].items():
    matches = [item for item in console_scripts if item.name == name]
    if len(matches) != 1:
        errors.append(f"console script {name} must have exactly one metadata owner")
        continue
    entrypoint = matches[0]
    owner = entrypoint.dist.metadata.get("Name", "") if entrypoint.dist else ""
    facts["entrypoints"][name] = {"distribution": owner, "target": entrypoint.value}
    if normalized(owner) != normalized(expected["distribution"]):
        errors.append(
            f"console script {name} must be owned by {expected['distribution']}, got {owner}"
        )
    if entrypoint.value != expected["target"]:
        errors.append(
            f"console script {name} must target {expected['target']}, got {entrypoint.value}"
        )

if contract["probe_imports"]:
    for module in contract["imports"]:
        try:
            importlib.import_module(module)
        except Exception as exc:
            errors.append(f"import {module} failed: {type(exc).__name__}: {exc}")
        else:
            facts["imports"].append(module)
    try:
        torch = importlib.import_module("torch")
    except Exception:
        pass
    else:
        facts["torch_version"] = str(torch.__version__)
        facts["torch_cuda_build"] = str(torch.version.cuda)

print("PAS_VENV_CONTRACT=" + json.dumps({"errors": errors, "facts": facts}, sort_keys=True))
"""


def load_runtime_manifest() -> dict[str, Any]:
    """Load and schema-check the checked-in acquisition/verification contract."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    schema = json.loads(MANIFEST_SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(manifest, schema)
    return manifest


def profile_contract(profile: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if profile not in PROFILE_NAMES:
        raise ValueError(f"virtualenv profile must be one of {', '.join(PROFILE_NAMES)}")
    manifest = load_runtime_manifest()
    return manifest, manifest["profiles"][profile]


def _version_tuple(value: str) -> tuple[int, ...]:
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", value):
        return ()
    return tuple(int(item) for item in value.split("."))


def _completed_error(completed: subprocess.CompletedProcess[str], label: str) -> str:
    detail = (completed.stderr or completed.stdout).strip()
    if len(detail) > 1000:
        detail = detail[:1000] + "..."
    return f"{label} failed (exit {completed.returncode})" + (f": {detail}" if detail else "")


def validate_tao_virtualenv(
    path: pathlib.Path,
    *,
    profile: str,
    probe_imports: bool,
    required_cli: str | None = None,
    minimum_gpus: int | None = None,
) -> pathlib.Path:
    """Verify one profile by ABI, package metadata, imports, pip, and CUDA.

    Merely placing executable files in ``bin`` is deliberately insufficient:
    each CLI must be the console script owned by the pinned TAO distribution.
    ``minimum_gpus`` opts into the real tensor-allocation probe and is intended
    for the approved preflight/action path, not read-only discovery.
    """
    manifest, configured = profile_contract(profile)
    resolved = path.expanduser().resolve()
    python = resolved / "bin" / "python"
    if (
        not (resolved / "pyvenv.cfg").is_file()
        or not python.is_file()
        or not os.access(python, os.X_OK)
    ):
        raise ValueError(f"approved {profile} virtualenv is missing or invalid: {resolved}")
    try:
        with python.open("rb") as handle:
            python_magic = handle.read(4)
    except OSError as exc:
        raise ValueError(f"cannot inspect {profile} virtualenv interpreter: {exc}") from exc
    if python_magic != b"\x7fELF":
        raise ValueError(
            f"approved {profile} virtualenv interpreter is not a Linux executable"
        )

    entrypoints = configured["entrypoints"]
    if required_cli is not None:
        if required_cli not in entrypoints:
            raise ValueError(
                f"TAO CLI {required_cli!r} does not belong to virtualenv profile {profile}"
            )
        checked_entrypoints = {required_cli: entrypoints[required_cli]}
    else:
        checked_entrypoints = dict(entrypoints)
    for name in checked_entrypoints:
        executable = resolved / "bin" / name
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise ValueError(
                f"approved {profile} virtualenv lacks TAO CLI entrypoint: {name}"
            )
        try:
            first_line = executable.open("rb").readline(4096).decode("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise ValueError(
                f"cannot inspect {profile} console script {executable}: {exc}"
            ) from exc
        if first_line.rstrip("\r\n") != f"#!{python}":
            raise ValueError(
                f"{profile} console script {name} is not bound to {python}"
            )

    probe_contract = {
        "distributions": configured["distributions"],
        "entrypoints": checked_entrypoints,
        "imports": configured["imports"],
        "probe_imports": probe_imports,
    }
    completed = subprocess.run(
        [str(python), "-I", "-c", _ENV_PROBE, json.dumps(probe_contract)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(_completed_error(completed, f"{profile} metadata probe"))
    try:
        evidence = next(
            (
                line.removeprefix("PAS_VENV_CONTRACT=")
                for line in reversed(completed.stdout.splitlines())
                if line.startswith("PAS_VENV_CONTRACT=")
            ),
            None,
        )
        payload = json.loads(evidence) if evidence is not None else None
        errors = payload["errors"]
        facts = payload["facts"]
        if not isinstance(errors, list) or any(
            not isinstance(item, str) for item in errors
        ) or not isinstance(facts, dict):
            raise TypeError("invalid evidence types")
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"{profile} metadata probe returned invalid JSON") from exc
    python_contract = manifest["python"]
    if facts.get("implementation") != python_contract["implementation"]:
        errors.append(
            f"Python implementation must be {python_contract['implementation']}, "
            f"got {facts.get('implementation')}"
        )
    if facts.get("python_major_minor") != python_contract["major_minor"]:
        errors.append(
            f"Python ABI must be {python_contract['major_minor']}, "
            f"got {facts.get('python_major_minor')}"
        )
    if facts.get("machine") != python_contract["machine"]:
        errors.append(
            f"machine must be {python_contract['machine']}, got {facts.get('machine')}"
        )
    if facts.get("prefix") != str(resolved) or facts.get("base_prefix") == str(resolved):
        errors.append(
            "interpreter sys.prefix does not identify the approved isolated virtualenv"
        )
    if _version_tuple(str(facts.get("glibc", ""))) < _version_tuple(
        python_contract["minimum_glibc"]
    ):
        errors.append(
            f"glibc must be >= {python_contract['minimum_glibc']}, got {facts.get('glibc')}"
        )
    if probe_imports:
        cuda_contract = manifest["cuda"]
        if facts.get("torch_version") != cuda_contract["torch_version"]:
            errors.append(
                f"torch runtime must be {cuda_contract['torch_version']}, "
                f"got {facts.get('torch_version')}"
            )
        if facts.get("torch_cuda_build") != cuda_contract["torch_cuda_build"]:
            errors.append(
                f"torch CUDA build must be {cuda_contract['torch_cuda_build']}, "
                f"got {facts.get('torch_cuda_build')}"
            )
    if errors:
        raise ValueError(
            f"approved {profile} virtualenv violates its runtime contract: "
            + "; ".join(errors)
        )

    if probe_imports:
        checked = subprocess.run(
            [str(python), "-m", "pip", "check"],
            capture_output=True,
            text=True,
            check=False,
        )
        if checked.returncode != 0:
            raise ValueError(_completed_error(checked, f"{profile} pip check"))

    if minimum_gpus is not None:
        if minimum_gpus < 1:
            raise ValueError("minimum_gpus must be at least 1")
        cuda_probe = pathlib.Path(__file__).resolve().parent / "check_pas_cuda_runtime.py"
        command = [str(python), str(cuda_probe), "--min-gpus", str(minimum_gpus)]
        for name in checked_entrypoints:
            command.extend(["--require-cli", name])
        environment = dict(os.environ)
        environment["PATH"] = str(resolved / "bin") + os.pathsep + environment.get("PATH", "")
        probed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=environment,
            check=False,
        )
        if probed.returncode != 0 or "PAS_CUDA_PROBE=PASS" not in probed.stdout:
            raise ValueError(_completed_error(probed, f"{profile} CUDA runtime probe"))
    return resolved


def resolve_virtualenv_profiles(
    *,
    platform: str,
    legacy: pathlib.Path | None,
    pyt: pathlib.Path | None,
    ds: pathlib.Path | None,
    probe_imports: bool,
) -> dict[str, pathlib.Path] | None:
    """Resolve explicit dual profiles, with truthful same-env compatibility."""
    supplied = [value is not None for value in (legacy, pyt, ds)]
    if platform != "virtualenv":
        if any(supplied):
            raise ValueError("virtualenv arguments are valid only when --platform virtualenv")
        return None
    if legacy is not None and (pyt is not None or ds is not None):
        raise ValueError("--virtualenv cannot be combined with profile-specific virtualenvs")
    if legacy is not None:
        pyt = ds = legacy
    if pyt is None or ds is None:
        raise ValueError(
            "--pyt-virtualenv and --ds-virtualenv are both required for platform=virtualenv"
        )
    return {
        profile: validate_tao_virtualenv(
            path,
            profile=profile,
            probe_imports=probe_imports,
        )
        for profile, path in (("pyt", pyt), ("ds", ds))
    }


def lock_status(profile: str) -> dict[str, Any]:
    """Return deterministic lock readiness without resolving or downloading."""
    _, configured = profile_contract(profile)
    lock = REFERENCE_DIR / configured["lock_file"]
    actual_sha256 = hashlib.sha256(lock.read_bytes()).hexdigest() if lock.is_file() else None
    complete = (
        configured["lock_status"] == "complete"
        and isinstance(configured["lock_sha256"], str)
        and actual_sha256 == configured["lock_sha256"]
    )
    return {
        "profile": profile,
        "lock_file": str(lock),
        "declared_status": configured["lock_status"],
        "declared_sha256": configured["lock_sha256"],
        "actual_sha256": actual_sha256,
        # Verification consumes the manifest's exact runtime facts and does
        # not install anything.  Keep it available for supplied environments
        # even while reproducible acquisition remains fail-closed.
        "prebuilt_verification_available": True,
        "ready_to_install": complete,
        "blocker": None if complete else (
            "generate the complete transitive CPython 3.12/Linux x86_64/CUDA 13.0 "
            "hash lock with an approved resolver, review every artifact, then set "
            "lock_status=complete and record the lock SHA-256 in the runtime manifest"
        ),
    }
