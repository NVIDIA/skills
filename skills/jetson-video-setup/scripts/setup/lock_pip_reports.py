#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create the isolated venv, then materialize and apply a hash-bound pip lock.

Run directly as ``python3 -I lock_pip_reports.py <action> ...``. ``create-venv`` builds one new
isolated environment, never deleting or reusing a target; ``materialize-apply`` writes the
reviewed lock plus its manifest and applies that exact digest. Every artifact is created
exclusively and is never rewritten. This helper owns the lock, the manifest, the apply report,
and the PyNvVideoCodec wheel evidence -- "the wheel whose SHA-256 matched the lock".
Authenticating the extension the interpreter actually loads against that wheel's ``RECORD``
belongs to ``verify_pynvc_sample.py``, which also owns validated-venv registry publication
through its own ``--register-current``.
"""

# pylint: disable=missing-function-docstring,too-many-arguments,too-many-positional-arguments

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.metadata
import json
import os
import re
import shutil
import sys
import tempfile
import time
import venv
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

_SETUP_DIR = Path(__file__).resolve().parent
if str(_SETUP_DIR) not in sys.path:
    sys.path.insert(0, str(_SETUP_DIR))

from setup_contract import (  # noqa: E402  # pylint: disable=wrong-import-position
    file_identity, read_json, require_isolated,
    run_command, sha256_bytes, system_executable, utc_now, write_new_bytes, write_new_json)

SCHEMA_VERSION = "1.0"
DEFAULT_ALLOWED_HOSTS = ("files.pythonhosted.org", "download.pytorch.org",
                         "download-r2.pytorch.org", "pypi.nvidia.com")
# The one permitted source artifact: SHA-256-pinned bytes, so inspecting its declared build and
# runtime dependencies cannot add assurance. pip enforces them via --check-build-dependencies.
PYCUDA_SOURCE_POLICY = {
    "name": "pycuda", "version": "2026.1",
    "url": ("https://files.pythonhosted.org/packages/75/93/"
            "36ffd54430924dcb03a88c97519d56589e573d0dcc999c77a4c4d8a90306/pycuda-2026.1.tar.gz"),
    "sha256": "759516160628ba06f32ce7e563e3f5b9214691dc9528a03ea99ea1073f4e14ba"}
INSTALL_CONTRACT = {
    "no_dependency_resolution": True, "hash_checking": True, "venv_reuse": False,
    "direct_https_artifacts_only": True, "source_preflight_before_mutation": True,
    "wheel_before_source": True, "build_isolation": False, "build_dependencies_checked": True,
    "source_archive_hash_verified": True, "proxy_forwarded_to_children": True,
    "build_environment": "explicit PATH/CPATH/LIBRARY_PATH scoped to the build subprocess"}
# Internal markers: creation proves the applying venv; applied is written once, barring a redo.
CLEAN_VENV_MARKER_NAME, CLEAN_VENV_MARKER_KIND = ".nvcodec-clean-venv.json", (
    "nvcodec-clean-venv-marker")
APPLIED_MARKER_NAME, APPLIED_MARKER_KIND = ".nvcodec-clean-venv-applied.json", (
    "nvcodec-clean-venv-applied-marker")
# run_command's allowlist excludes PIP_*, so this is the complete pip config for every child.
_PIP_ENV = {"PIP_CONFIG_FILE": os.devnull, "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INPUT": "1"}
_SYSTEM_BUILD_TOOLS = frozenset({"gcc", "g++", "make"})
_SOURCE_ARCHIVE_LIMIT, _CHILD_TIMEOUT = 64 * 1024 * 1024, 3600.0
_STREAM_TAIL_BYTES = 8000
_SHA256 = re.compile(r"[0-9a-f]{64}")
_NAME = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?")
# packaging helper -> submodule, resolved from packaging or pip's vendored copy.
_PACKAGING_HELPERS = {"Requirement": "requirements", "Version": "version",
                      "default_environment": "markers"}
_EXISTS = "The venv target already exists; it was not deleted, reused, or modified."
_APPEARED = "The venv target appeared during reservation; it was not reused or modified."
_RECOVERY = "The partial target was preserved. Choose a different new target to retry."


class ApplyBlockedError(ValueError):
    """A complete, safely stopped apply outcome caused by external policy drift."""


class _StageFailed(Exception):
    """A pip stage exited non-zero, so the environment may already be mutated."""

    def __init__(self, stage: str) -> None:
        super().__init__(stage)
        self.stage = stage


class _VerificationFailed(Exception):
    """Post-mutation evidence did not match the reviewed lock."""

    def __init__(self, stage: str, cause: BaseException) -> None:
        super().__init__(str(cause))
        self.stage = stage
        self.cause = cause


class _PipCheckFailed(Exception):
    """`pip check` reported an inconsistent environment after application."""


class _EnvironmentMismatch(Exception):
    """Post-install check found a difference: a completed negative, so exit 2, not 3."""


class _Packaging:
    """Packaging helpers resolved on first use, from packaging or pip's vendored copy."""

    def __getattr__(self, name: str) -> Any:
        module = _PACKAGING_HELPERS.get(name)
        if module is None:
            raise AttributeError(name)
        try:
            source = importlib.import_module(f"packaging.{module}")
        except ImportError:
            source = importlib.import_module(f"pip._vendor.packaging.{module}")
        value = getattr(source, name)
        setattr(self, name, value)
        return value


_pkg = _Packaging()


def _absolute(path: Any) -> Path:
    """Normalize dot segments without following a target symlink."""
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _digest(path: Any, label: str) -> str:
    return file_identity(path, label=label)["sha256"]


def _emit(path: Path) -> None:
    # Replay the persisted artifact verbatim, so stdout and disk can never disagree.
    sys.stdout.write(path.read_text(encoding="utf-8"))


def _render(value: Any) -> str:
    """Render one JSON document for stdout only; every on-disk artifact uses write_new_json."""
    return json.dumps(value, allow_nan=False, sort_keys=True, indent=2)


def _prepared_parent(path: Path) -> Path:
    """Create every missing parent privately, rejecting a symlinked path component."""
    requested = _absolute(path)
    current = Path(requested.anchor)
    for part in requested.parent.parts[1:]:
        current /= part
        if current.is_symlink() or (current.exists() and not current.is_dir()):
            raise ValueError(f"path parent component is a symlink or not a directory: {current}")
        current.mkdir(mode=0o700, exist_ok=True)
    return requested.parent.resolve(strict=True) / requested.name


def _fresh_output_path(path: Any, *, label: str) -> Path:
    """Refuse an output path that already exists, before anything is created."""
    requested = _absolute(path)
    if not requested.parent.is_dir():
        raise FileNotFoundError(f"{label} parent does not exist: {requested.parent}")
    resolved = requested.parent.resolve(strict=True) / requested.name
    if os.path.lexists(resolved):
        raise FileExistsError(f"{label} must be a fresh path: {resolved}")
    return resolved


def _assert_distinct_paths(paths: list[Path]) -> None:
    resolved = [_absolute(path) for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("resolver, lock, manifest, and report paths must be distinct")


def _canonical_name(name: str) -> str:
    """PEP 503 normalization, independent of whether packaging itself is importable."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _validated_name(name: str) -> str:
    if not _NAME.fullmatch(name):
        raise ValueError(f"invalid package name in pip report: {name!r}")
    return _canonical_name(name)


def _load_report(path: Path) -> dict[str, Any]:
    data = read_json(Path(path))
    if not isinstance(data, dict) or data.get("version") != "1":
        raise ValueError(f"unsupported pip report version in {path}")
    if (not isinstance(data.get("pip_version"), str) or not data["pip_version"]
            or not isinstance(data.get("environment"), dict) or not data["environment"]):
        raise ValueError(f"pip report lacks a pip_version and target environment: {path}")
    return data


def _resolution_contract(reports: list[Path]) -> dict[str, Any]:
    contracts = [{"pip_version": data["pip_version"], "environment": data["environment"]}
                 for data in (_load_report(path) for path in reports)]
    if any(contract != contracts[0] for contract in contracts[1:]):
        raise ValueError("resolver reports do not describe the same pip/environment contract")
    return contracts[0]


def _archive_sha256(download_info: Any) -> str:
    archive = download_info.get("archive_info") if isinstance(download_info, dict) else None
    if not isinstance(archive, dict):
        raise ValueError("pip report entry has no archive_info")
    hashes = archive.get("hashes")
    digest = hashes.get("sha256") if isinstance(hashes, dict) else None
    if not digest:
        value = archive.get("hash")
        digest = value.split("=", 1)[1] if isinstance(value, str) and "=" in value else None
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest.lower()):
        raise ValueError("pip report entry has no exact SHA-256 archive hash")
    return digest.lower()


def _locked_url(url: str, digest: str, allowed_hosts: set[str]) -> str:
    """Bind an artifact to default-port HTTPS on an allowlisted, credential-free origin."""
    parsed = urlsplit(url)
    if parsed.scheme != "https" or (parsed.hostname or "").lower() not in allowed_hosts:
        raise ValueError(f"unapproved artifact origin: {url!r}")
    if parsed.username or parsed.password:
        raise ValueError("artifact URL must not contain credentials")
    if parsed.port not in {None, 443}:
        raise ValueError("artifact URL must use the default HTTPS port")
    if parsed.query:
        raise ValueError("artifact URL must not contain an unstable query string")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", f"sha256={digest}"))


class _LockedRedirectHandler(HTTPRedirectHandler):
    """Validate every resolved redirect before urllib opens the next URL."""

    def __init__(self, digest: str, allowed_hosts: set[str]) -> None:
        super().__init__()
        self._digest = digest
        self._allowed_hosts = set(allowed_hosts)

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        try:
            _locked_url(newurl, self._digest, self._allowed_hosts)
        except ValueError:
            fp.close()
            raise
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _artifact_type(url: str) -> str:
    path = unquote(urlsplit(url).path).lower()
    if path.endswith(".whl"):
        return "wheel"
    if path.endswith((".tar.gz", ".tar.bz2", ".tar.xz", ".zip")):
        return "source"
    raise ValueError(f"unsupported Python artifact type: {url!r}")


def _artifact_filename(url: str) -> str:
    filename = PurePosixPath(unquote(urlsplit(url).path)).name
    if not filename:
        raise ValueError(f"artifact URL has no filename: {url!r}")
    return filename


def _locked_requirement(name: str, url: str, digest: str, allowed_hosts: set[str]) -> str:
    return f"{name} @ {_locked_url(url, digest, allowed_hosts)} --hash=sha256:{digest}"


def lock_text(artifacts: list[dict[str, Any]]) -> str:
    return "".join(f"{item['locked_requirement']}\n" for item in artifacts)


def _artifact_identity(artifacts: list[dict[str, Any]]) -> set[tuple[str, str, str]]:
    return {(item["normalized_name"], item["version"], item["sha256"]) for item in artifacts}


def _report_entry(entry: Any, report: Path, allowed_hosts: set[str]) -> dict[str, Any]:
    metadata = entry.get("metadata") if isinstance(entry, dict) else None
    download = entry.get("download_info") if isinstance(entry, dict) else None
    if not isinstance(metadata, dict) or not isinstance(download, dict):
        raise ValueError(f"pip report entry lacks metadata/download_info: {report}")
    name, version, url = metadata.get("name"), metadata.get("version"), download.get("url")
    if not all(isinstance(value, str) and value for value in (name, version, url)):
        raise ValueError(f"pip report entry lacks name/version/url: {report}")
    _pkg.Version(version)
    if not isinstance(entry.get("requested"), bool):
        raise ValueError(f"pip report entry has no requested flag: {report}")
    if entry.get("is_yanked") is True:
        raise ValueError(f"pip report selected a yanked artifact: {name}=={version}")
    digest = _archive_sha256(download)
    return {"name": name, "normalized_name": _validated_name(name), "version": version,
            "url": url, "filename": _artifact_filename(url), "sha256": digest,
            "artifact_type": _artifact_type(url), "requested": entry["requested"],
            "locked_requirement": _locked_requirement(name, url, digest, allowed_hosts)}


def _report_records(reports: list[Path], allowed_hosts: set[str]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for report in reports:
        installs = _load_report(report).get("install")
        if not isinstance(installs, list) or not installs:
            raise ValueError(f"pip report has no resolved install set: {report}")
        for entry in installs:
            record = _report_entry(entry, report, allowed_hosts)
            previous = records.setdefault(record["normalized_name"], record)
            # Merge only the same wheel bytes/metadata. When approved mirrors expose the
            # same artifact, retain the lexically first URL for order-independent output.
            if previous is not record and any(previous[key] != record[key] for key in (
                    "version", "sha256", "artifact_type", "filename")):
                raise ValueError(f"conflicting resolved artifacts for {record['name']}")
            previous["requested"] = previous["requested"] or record["requested"]
            canonical_url = min(previous["url"], record["url"])
            previous["url"] = canonical_url
            previous["locked_requirement"] = _locked_requirement(
                previous["name"], canonical_url, previous["sha256"], allowed_hosts
            )
    return records


def collect_artifacts(reports: list[Path], requirements: list[str],
                      allowed_hosts: set[str]) -> list[dict[str, Any]]:
    """Return one exact artifact per normalized package and enforce the reviewed requirements."""
    if not requirements:
        raise ValueError("at least one policy requirement is required")
    records = _report_records(reports, allowed_hosts)
    required: set[str] = set()
    for raw in requirements:
        requirement = _pkg.Requirement(raw)
        required.add(_canonical_name(requirement.name))
        record = records.get(_canonical_name(requirement.name))
        if not record:
            raise ValueError(f"required package is absent from reports: {requirement.name}")
        if requirement.url:
            raise ValueError("policy requirements must use version specifiers, not direct URLs")
        if not record["requested"]:
            raise ValueError(f"policy root was not requested in a report: {requirement.name}")
        if requirement.specifier and not requirement.specifier.contains(record["version"],
                                                                        prereleases=True):
            raise ValueError(f"resolved {record['name']}=={record['version']} violates {raw}")
        if (_pkg.Version(record["version"]).is_prerelease
                and str(requirement.specifier) != f"=={record['version']}"):
            raise ValueError(f"prerelease {record['name']} requires an exact policy pin")
    unexpected = sorted(name for name, item in records.items()
                        if item["requested"] and name not in required)
    if unexpected:
        raise ValueError(f"reports contain unexpected requested roots: {unexpected}")
    loose = sorted(name for name, item in records.items()
                   if not item["requested"] and _pkg.Version(item["version"]).is_prerelease)
    if loose:
        raise ValueError(f"transitive prerelease artifacts are forbidden: {loose}")
    return [records[name] for name in sorted(records)]


def _source_names(source_requirements: list[str]) -> set[str]:
    names: set[str] = set()
    for raw in source_requirements:
        requirement = _pkg.Requirement(raw)
        if requirement.url:
            raise ValueError("source policy requirements must not use direct URLs")
        names.add(_canonical_name(requirement.name))
    return names


def _validate_source_policy(artifacts: list[dict[str, Any]], source_requirements: list[str]):
    expected = _source_names(source_requirements)
    observed = {item["normalized_name"] for item in artifacts if item["artifact_type"] == "source"}
    if observed != expected:
        raise ValueError("source artifact set does not match the explicit source policy: "
                         f"expected {sorted(expected)}, observed {sorted(observed)}")


def _policy_source_artifacts(source_requirements: list[str],
                             allowed_hosts: set[str]) -> list[dict[str, Any]]:
    if not source_requirements:
        return []
    if source_requirements != ["pycuda==2026.1"]:
        raise ValueError("the only permitted source policy is exact pycuda==2026.1")
    name, url, digest = (PYCUDA_SOURCE_POLICY[key] for key in ("name", "url", "sha256"))
    return [{"name": name, "normalized_name": name, "version": PYCUDA_SOURCE_POLICY["version"],
             "url": url, "filename": _artifact_filename(url), "sha256": digest,
             "artifact_type": "source", "requested": True,
             "locked_requirement": _locked_requirement(name, url, digest, allowed_hosts)}]


def _venv_outcome(result: dict[str, Any], status: str, exit_code: int,
                  **fields: Any) -> tuple[dict[str, Any], int]:
    result.update({"status": status, **fields})
    return result, exit_code


def create_clean_venv(target: Path) -> tuple[dict[str, Any], int]:
    """Create a reserved, absent target; an existing one is reported, never deleted or reused."""
    target = _absolute(target)
    base = str(Path(getattr(sys, "_base_executable", None) or sys.executable).resolve())
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "kind": "nvcodec-clean-venv-creation",
        "target": str(target), "creator_executable": sys.executable, "base_executable": base,
        "reuse_allowed": False, "delete_existing": False, "system_site_packages": False}
    exists = {"created": False, "mutated": False, "target_preexisted": True}
    if os.path.lexists(target):
        return _venv_outcome(result, "target_exists", 2, **exists, reason=_EXISTS)
    try:
        target = _prepared_parent(target)
        target.mkdir(mode=0o700)
    except FileExistsError:
        return _venv_outcome(result, "target_exists", 2, **exists, reason=_APPEARED)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return _venv_outcome(result, "creation_failed", 3, created=False, mutated=False,
                             target_preexisted=False, reason=str(exc))
    try:
        reserved = target.stat()
        # venv and Debian's patched ensurepip write to stdout; this CLI's stdout is JSON only.
        with contextlib.redirect_stdout(sys.stderr):
            venv.EnvBuilder(system_site_packages=False, clear=False, symlinks=os.name != "nt",
                            upgrade=False, with_pip=True, prompt="nvcodec",
                            upgrade_deps=False).create(target)
        current = target.stat()
        if target.is_symlink() or (current.st_dev, current.st_ino) != (reserved.st_dev,
                                                                       reserved.st_ino):
            raise RuntimeError("reserved venv target identity changed during creation")
        interpreter = target / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        configuration = target / "pyvenv.cfg"
        if not interpreter.is_file() or not configuration.is_file():
            raise RuntimeError("venv creation did not produce its interpreter and pyvenv.cfg")
        if configuration.is_symlink() or not configuration.resolve(
                strict=True).is_relative_to(target):
            raise RuntimeError("pyvenv.cfg must be a regular file inside the reserved target")
        # Mirrors this record with the resolved target path; only the apply stage reads it.
        marker = write_new_json(target / CLEAN_VENV_MARKER_NAME, {
            **result, "kind": CLEAN_VENV_MARKER_KIND, "created_at": utc_now(),
            "target": str(target), "target_python": str(interpreter),
            "target_preexisted": False})
        return _venv_outcome(
            result, "created", 0, created=True, mutated=True, target_preexisted=False,
            target_python=str(interpreter), configuration=str(configuration), marker=str(marker),
            marker_sha256=_digest(marker, "clean venv marker"), completed_at=utc_now())
    # SystemExit is deliberate: Debian/Ubuntu patch venv to sys.exit() with no python3-venv.
    except (Exception, SystemExit) as exc:  # pylint: disable=broad-exception-caught
        return _venv_outcome(result, "creation_failed_partial", 3, created=False, mutated=True,
                             target_preexisted=False, target_exists=target.exists(),
                             reason=f"{type(exc).__name__}: {exc}", recovery=_RECOVERY)


def _assert_clean_venv() -> Path:
    """Prove the applying interpreter is the fresh, never-applied venv holding only pip tooling."""
    if sys.prefix == getattr(sys, "base_prefix", sys.prefix):
        raise ValueError("locked application requires the newly created virtual environment")
    marker_path = Path(sys.prefix) / CLEAN_VENV_MARKER_NAME
    if marker_path.is_symlink() or not marker_path.is_file():
        raise ValueError("new virtual environment has no clean-creation marker")
    if os.path.lexists(Path(sys.prefix) / APPLIED_MARKER_NAME):
        raise ValueError("this virtual environment already consumed a locked apply")
    marker = read_json(marker_path)
    if (not isinstance(marker, dict) or marker.get("schema_version") != SCHEMA_VERSION
            or marker.get("kind") != CLEAN_VENV_MARKER_KIND
            or marker.get("target_preexisted") is not False):
        raise ValueError("clean virtual-environment marker is invalid")
    unexpected = sorted({_canonical_name(item.metadata.get("Name", ""))
                         for item in importlib.metadata.distributions()} - {"pip", "setuptools"})
    if unexpected:
        raise ValueError(f"new virtual environment is not clean: {unexpected}")
    return marker_path


def _seal_applied_venv(manifest_digest: str, wheel_evidence: dict[str, Any] | None) -> str:
    """Create the one-shot applied marker, so a second apply into this venv is refused."""
    return _digest(write_new_json(Path(sys.prefix) / APPLIED_MARKER_NAME, {
        "schema_version": SCHEMA_VERSION, "kind": APPLIED_MARKER_KIND,
        "apply_status": "applied_verified", "apply_completed_at": utc_now(),
        "manifest_sha256": manifest_digest, "pynvc_wheel": wheel_evidence}), "applied marker")


def materialize(reports: list[Path], requirements: list[str], source_requirements: list[str],
                lock_path: Path, manifest_path: Path) -> dict[str, Any]:
    allowed_hosts = set(DEFAULT_ALLOWED_HOSTS)
    lock_path = _fresh_output_path(lock_path, label="pip lock")
    manifest_path = _fresh_output_path(manifest_path, label="pip lock manifest")
    identities = [file_identity(path, label="pip resolver report") for path in reports]
    resolved = [Path(item["path"]) for item in identities]
    if len(set(resolved)) != len(resolved):
        raise ValueError("resolver report paths must be unique")
    _assert_distinct_paths([*resolved, lock_path, manifest_path])
    source_names = _source_names(source_requirements)
    artifacts = collect_artifacts(
        resolved, [item for item in requirements
                   if _canonical_name(_pkg.Requirement(item).name) not in source_names],
        allowed_hosts)
    artifacts.extend(_policy_source_artifacts(source_requirements, allowed_hosts))
    artifacts.sort(key=lambda item: item["normalized_name"])
    _validate_source_policy(artifacts, source_requirements)
    contract = _resolution_contract(resolved)
    if contract["pip_version"] != importlib.metadata.version("pip"):
        raise ValueError("resolver reports were created by a different pip version")
    if contract["environment"] != _pkg.default_environment():
        raise ValueError("resolver reports describe a different Python/platform environment")
    lock_path = write_new_bytes(lock_path, lock_text(artifacts).encode("utf-8"))
    manifest = {
        "schema_version": SCHEMA_VERSION, "kind": "nvcodec-pip-lock-manifest",
        "generated_at": utc_now(), "interpreter": sys.executable,
        "requirements": requirements, "source_requirements": source_requirements,
        "allowed_hosts": sorted(allowed_hosts), "resolution_contract": contract,
        "resolution_reports": [{"path": item["path"], "sha256": item["sha256"]}
                               for item in identities],
        "lock": {"path": str(lock_path), "sha256": _digest(lock_path, "pip lock")},
        "artifacts": artifacts, "install_contract": INSTALL_CONTRACT}
    write_new_json(manifest_path, manifest)
    return manifest


def _authenticated_manifest_artifacts(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Re-enforce the origin/hash policy on every artifact this apply will install."""
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("lock manifest has no artifacts")
    allowed_hosts = set(manifest["allowed_hosts"])
    for item in artifacts:
        if not isinstance(item, dict):
            raise ValueError("lock manifest artifact record is invalid")
        name, url = str(item.get("name", "")), str(item.get("url", ""))
        digest = str(item.get("sha256", ""))
        if _validated_name(name) != item.get("normalized_name"):
            raise ValueError(f"lock manifest artifact name is invalid: {name!r}")
        if not _SHA256.fullmatch(digest):
            raise ValueError(f"lock manifest artifact has no exact SHA-256: {name}")
        if (item.get("locked_requirement") != _locked_requirement(name, url, digest, allowed_hosts)
                or item.get("filename") != _artifact_filename(url)
                or item.get("artifact_type") != _artifact_type(url)):
            raise ValueError(f"lock manifest artifact is not bound to its reviewed URL: {name}")
    _validate_source_policy(artifacts, manifest["source_requirements"])
    return artifacts


def _bound_file(record: Any, label: str) -> Path:
    """Return a recorded path only while the file on disk still hashes to its recorded digest."""
    path = record.get("path") if isinstance(record, dict) else None
    if not isinstance(path, str) or not path:
        raise ValueError(f"lock manifest has no {label} path")
    if _digest(Path(path), label) != record.get("sha256"):
        raise ValueError(f"{label} drift: {path}")
    return Path(path)


def verify_manifest(manifest_path: Path) -> dict[str, Any]:
    """Authenticate the manifest, its lock, and the environment that is applying it."""
    manifest = read_json(Path(manifest_path))
    if not isinstance(manifest, dict) or manifest.get("kind") != "nvcodec-pip-lock-manifest":
        raise ValueError("not an nvcodec-pip-lock-manifest")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported pip lock manifest schema")
    if manifest.get("interpreter") != sys.executable:
        raise ValueError("lock manifest was created by a different Python interpreter")
    if sorted(manifest.get("allowed_hosts") or ()) != sorted(DEFAULT_ALLOWED_HOSTS):
        raise ValueError("lock manifest allowed_hosts differs from the reviewed setup policy")
    for key in ("requirements", "source_requirements"):
        values = manifest.get(key)
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise ValueError(f"lock manifest {key} are invalid")
    contract = manifest.get("resolution_contract")
    if not isinstance(contract, dict):
        raise ValueError("lock manifest has no resolver environment contract")
    if contract.get("pip_version") != importlib.metadata.version("pip"):
        raise ValueError("the applying pip version differs from the resolver pip version")
    if contract.get("environment") != _pkg.default_environment():
        raise ValueError("the applying Python/platform differs from the resolver environment")
    records = manifest.get("resolution_reports")
    if not isinstance(records, list) or not records:
        raise ValueError("lock manifest has no resolution reports")
    for record in records:
        _bound_file(record, "resolver report")
    lock_path = _bound_file(manifest.get("lock"), "pip lock")
    if lock_path.read_text(encoding="utf-8") != lock_text(
            _authenticated_manifest_artifacts(manifest)):
        raise ValueError("pip lock content does not match the reviewed artifact set")
    return manifest


def _download_authenticated_source(artifact: dict[str, Any], destination: Path,
                                   allowed_hosts: set[str]) -> dict[str, Any]:
    """Fetch one locked sdist, validating every redirect target and then the exact hash."""
    parsed = urlsplit(_locked_url(artifact["url"], artifact["sha256"], allowed_hosts))
    filename = PurePosixPath(unquote(parsed.path)).name
    if not filename or _artifact_type(parsed.path) != "source":
        raise ValueError("source artifact URL has no supported archive filename")
    destination.mkdir(mode=0o700, exist_ok=True)
    request = Request(urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", "")),
                      headers={"User-Agent": "jetson-videosdk-skill/1.0"})
    chunks: list[bytes] = []
    observed = 0
    opener = build_opener(_LockedRedirectHandler(artifact["sha256"], allowed_hosts))
    with opener.open(request, timeout=120) as response:  # nosec B310
        # Defense in depth for any response path that bypasses redirect handling.
        _locked_url(response.geturl(), artifact["sha256"], allowed_hosts)
        for block in iter(lambda: response.read(1024 * 1024), b""):
            observed += len(block)
            if observed > _SOURCE_ARCHIVE_LIMIT:
                raise ValueError("source archive exceeds the reviewed size limit")
            chunks.append(block)
    payload = b"".join(chunks)
    if sha256_bytes(payload) != artifact["sha256"]:
        raise ValueError("downloaded source does not match its reviewed SHA-256")
    return {"path": str(write_new_bytes(destination / filename, payload)),
            "sha256": artifact["sha256"], "size_bytes": observed, "source_url": artifact["url"]}


def _parse_build_environment(values: list[str]) -> dict[str, str]:
    allowed = {"PATH", "CPATH", "LIBRARY_PATH"}
    result: dict[str, str] = {}
    for value in values:
        name, separator, content = value.partition("=")
        if not separator or name not in allowed or not content or name in result:
            raise ValueError("build environment must contain one non-empty PATH, CPATH, and"
                             " LIBRARY_PATH assignment")
        result[name] = content
    if set(result) != allowed:
        raise ValueError("PyCUDA source builds require PATH, CPATH, and LIBRARY_PATH")
    for name, content in result.items():
        if any(not value or not Path(value).is_absolute() for value in content.split(":")):
            raise ValueError(f"PyCUDA {name} entries must be non-empty absolute paths")
    return result


def _build_tool_identities(environment: dict[str, str]) -> dict[str, Any]:
    """Identify the tools this build will use; system compilers must be the canonical ones."""
    identities: dict[str, Any] = {}
    for name in ("nvcc", "gcc", "g++", "make"):
        located = shutil.which(name, path=environment.get("PATH"))
        if not located:
            raise ValueError(f"authenticated PyCUDA build tool is unavailable: {name}")
        invocation, identity_path = Path(located), Path(located)
        if name in _SYSTEM_BUILD_TOOLS:
            identity_path = Path(system_executable(name))
            if invocation.parent != Path("/usr/bin") or invocation.resolve(strict=True) != (
                    identity_path):
                raise ValueError(
                    f"PyCUDA build tool {name} is not the canonical system executable")
        identities[name] = {"invocation": str(invocation), **file_identity(
            identity_path, label=f"PyCUDA build tool {name}")}
    return identities


def _pip_argv(*arguments: str) -> list[str]:
    """The only interpreter/pip prefix any child command in this module may use."""
    return [sys.executable, "-I", "-m", "pip", "--isolated", *arguments]


def _pip_install_argv(report_path: Path, requirements_path: Path, *, no_index: bool,
                      build: bool = False) -> list[str]:
    """The only permitted install shape: hash-checked, dependency-free, forced reinstall."""
    return _pip_argv("install", "--disable-pip-version-check", "--no-input", "--no-cache-dir",
                     *(["--no-index"] if no_index else []),
                     *(["--no-build-isolation", "--check-build-dependencies"] if build else []),
                     "--force-reinstall", "--no-deps", "--require-hashes",
                     "--report", str(report_path), "-r", str(requirements_path))


def _run_child(argv: list[str], *, environment: dict[str, str], cwd: Path) -> dict[str, Any]:
    started, monotonic = utc_now(), time.monotonic()
    done = run_command(argv, cwd=cwd, timeout=_CHILD_TIMEOUT, extra_env=environment)
    record = {"argv": argv, "cwd": str(cwd), "started_at": started, "ended_at": utc_now(),
              "duration_seconds": time.monotonic() - monotonic, "returncode": done.returncode}
    # Each stream is bounded to its last _STREAM_TAIL_BYTES; nothing is dropped silently.
    for name in ("stdout", "stderr"):
        data = getattr(done, name) or b""
        record[f"{name}_tail"] = data[-_STREAM_TAIL_BYTES:].decode("utf-8", "replace")
        record[f"{name}_bytes"] = len(data)
        record[f"{name}_truncated"] = len(data) > _STREAM_TAIL_BYTES
    return record


def _validate_local_source_report(report_path: Path, artifact: dict[str, Any],
                                  source: dict[str, Any], contract: dict[str, Any]) -> None:
    """Confirm the source stage installed exactly the archive this process authenticated."""
    data = _load_report(report_path)
    if {"pip_version": data["pip_version"], "environment": data["environment"]} != contract:
        raise ValueError("source apply report environment differs from the resolver")
    installs = data.get("install")
    if not isinstance(installs, list) or len(installs) != 1:
        raise ValueError("source apply report must contain exactly one artifact")
    metadata = installs[0].get("metadata") or {}
    download = installs[0].get("download_info") or {}
    if _canonical_name(str(metadata.get("name", ""))) != artifact["normalized_name"]:
        raise ValueError("source apply report has the wrong project")
    if _pkg.Version(str(metadata.get("version", ""))) != _pkg.Version(artifact["version"]):
        raise ValueError("source apply report has the wrong version")
    if _archive_sha256(download) != source["sha256"]:
        raise ValueError("source apply report has the wrong archive hash")
    url = urlsplit(str(download.get("url", "")))
    if url.scheme != "file" or Path(unquote(url.path)).resolve() != Path(source["path"]):
        raise ValueError("source apply report did not install the authenticated archive")


def _pynvc_wheel_evidence(wheels: list[dict[str, Any]],
                          wheel_report: Path | None) -> dict[str, Any] | None:
    """Name the wheel whose SHA-256 matched the lock, installed under --require-hashes.

    Proven upstream, unrepeated: _wheel_stage equates this apply report's artifact identity with
    the lock. verify_pynvc_sample.py owns RECORD authentication of the loaded extension.
    """
    locked = [item for item in wheels if item["normalized_name"] == "pynvvideocodec"]
    if not locked:
        return None
    if len(locked) != 1:
        raise ValueError("reviewed lock must select exactly one PyNvVideoCodec wheel")
    if wheel_report is None or not Path(wheel_report).is_file():
        raise ValueError("PyNvVideoCodec evidence requires the persisted wheel stage report")
    artifact = locked[0]
    return {"schema_version": SCHEMA_VERSION, "kind": "nvcodec-pynvc-wheel-evidence",
            "name": artifact["name"], "normalized_name": "pynvvideocodec",
            "version": artifact["version"], "filename": artifact["filename"],
            "sha256": artifact["sha256"], "url": artifact["url"], "hash_checked_install": True,
            "apply_report": str(wheel_report),
            "installed_version": importlib.metadata.version("PyNvVideoCodec"),
            "record_authority": "verify_pynvc_sample.py"}


def _preflight_apply_outputs(install_report: Path) -> dict[str, Path]:
    """Reserve every evidence path this apply may write before anything is created."""
    report = _fresh_output_path(install_report, label="pip install report")
    stem, suffix = (report.stem, report.suffix) if report.suffix else (report.name, ".json")
    return {"install_report": report,
            **{stage: _fresh_output_path(report.with_name(f"{stem}.{stage}{suffix}"),
                                         label=f"{stage} stage report")
               for stage in ("wheels", "sources")}}


class _Apply:  # pylint: disable=too-many-instance-attributes
    """One locked application: authenticate every artifact, then mutate."""

    def __init__(self, manifest_path: Path, install_report: Path, expected_manifest_sha256: str,
                 build_environment_values: list[str], state: dict[str, bool]) -> None:
        self.manifest_path = _absolute(manifest_path)
        self.state = state
        self.digest = _digest(self.manifest_path, "lock manifest")
        if expected_manifest_sha256.lower() != self.digest:
            raise ValueError("lock manifest does not match the expected SHA-256")
        self.manifest = verify_manifest(self.manifest_path)
        artifacts = self.manifest["artifacts"]
        self.sources = [item for item in artifacts if item["artifact_type"] == "source"]
        self.wheels = [item for item in artifacts if item["artifact_type"] == "wheel"]
        if len(self.sources) > 1:
            raise ValueError("the reviewed policy permits at most one source artifact")
        if self.sources and not self.wheels:
            raise ValueError("source builds require a reviewed wheel bootstrap stage")
        if not self.sources and build_environment_values:
            raise ValueError("build environment was supplied without a source artifact")
        self.build_environment = (_parse_build_environment(build_environment_values)
                                  if self.sources else {})
        self.paths = _preflight_apply_outputs(install_report)
        self.report = self.paths["install_report"]
        _assert_distinct_paths([self.manifest_path, Path(self.manifest["lock"]["path"]),
                                *(Path(item["path"])
                                  for item in self.manifest["resolution_reports"]),
                                self.report, self.paths["wheels"], self.paths["sources"]])
        self.marker_path = _assert_clean_venv()
        self.summary: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION, "kind": "nvcodec-pip-lock-apply",
            "status": "running", "manifest": str(self.manifest_path),
            "manifest_sha256": self.digest, "clean_venv_marker": str(self.marker_path),
            "clean_venv_marker_status": "unconsumed",
            "clean_venv_marker_sha256": _digest(self.marker_path, "clean venv marker"),
            "started_at": utc_now(), "stages": []}

    def _stop(self, status: str, exit_code: int, **fields: Any) -> int:
        self.summary.update({"status": status, "completed_at": utc_now(), **fields})
        write_new_json(self.report, self.summary)
        return exit_code

    def _stage(self, stage: str, command: list[str], root: Path, environment: dict[str, str],
               persistent: Path, report_path: Path, **extra: Any) -> dict[str, Any]:
        record = {"stage": stage,
                  **_run_child(command, environment=environment, cwd=root), **extra}
        self.summary["stages"].append(record)
        if report_path.is_file():
            saved = write_new_bytes(persistent, report_path.read_bytes())
            record["pip_report"] = str(saved)
            record["pip_report_sha256"] = _digest(saved, "persisted pip report")
        if record["returncode"] != 0:
            raise _StageFailed(stage)
        return record

    def _wheel_stage(self, root: Path) -> list[dict[str, Any]]:
        """Install every locked wheel with hashes and no dependency resolution."""
        requirements = write_new_bytes(root / "wheels.requirements.txt",
                                       lock_text(self.wheels).encode("utf-8"))
        report_path = root / "wheels.pip-report.json"
        self._stage("wheels", _pip_install_argv(report_path, requirements, no_index=False), root,
                    _PIP_ENV, self.paths["wheels"], report_path)
        try:
            observed = collect_artifacts(
                [report_path], [f"{item['name']}=={item['version']}" for item in self.wheels],
                set(self.manifest["allowed_hosts"]))
            if _resolution_contract([report_path]) != self.manifest["resolution_contract"]:
                raise ValueError("apply report environment differs from resolver environment")
            if _artifact_identity(observed) != _artifact_identity(self.wheels):
                raise ValueError("wheels apply report differs from the reviewed lock")
        except Exception as exc:  # pylint: disable=broad-exception-caught
            raise _VerificationFailed("wheels", exc) from exc
        return observed

    def _source_stage(self, artifact: dict[str, Any], source: dict[str, Any], root: Path) -> None:
        """Build and install the hash-pinned sdist with the scoped CUDA build environment."""
        report_path = root / "source.pip-report.json"
        requirements = write_new_bytes(root / "source.requirements.txt", (
            f"{artifact['name']} @ {Path(source['path']).as_uri()}#sha256={source['sha256']} "
            f"--hash=sha256:{source['sha256']}\n").encode("utf-8"))
        environment = {**_PIP_ENV, **self.build_environment}
        self._stage("sources", _pip_install_argv(report_path, requirements, no_index=True,
                                                 build=True), root, environment,
                    self.paths["sources"], report_path, source=source,
                    build_environment=self.build_environment,
                    build_tool_identities=_build_tool_identities(environment))
        try:
            _validate_local_source_report(report_path, artifact, source,
                                          self.manifest["resolution_contract"])
        except Exception as exc:  # pylint: disable=broad-exception-caught
            raise _VerificationFailed("sources", exc) from exc

    def _run_stages(self, root: Path) -> list[dict[str, Any]]:
        """Authenticate every artifact before mutating, then install wheels before sources."""
        allowed_hosts = set(self.manifest["allowed_hosts"])
        try:
            sources = [_download_authenticated_source(item, root / "sources", allowed_hosts)
                       for item in self.sources]
        except (OSError, ValueError) as exc:
            raise ApplyBlockedError(f"authenticated artifact preflight blocked: {exc}") from exc
        self.summary["source_preflight"] = {
            "status": "authenticated_before_mutation" if sources else "not_applicable",
            "sources": sources}
        # The one manifest re-check, across the network preflight window, before any mutation.
        if _digest(self.manifest_path, "lock manifest") != self.digest:
            raise ValueError("lock manifest changed during authenticated source preflight")
        self.state["mutated"] = True
        applied = self._wheel_stage(root) if self.wheels else []
        for artifact, source in zip(self.sources, sources):
            self._source_stage(artifact, source, root)
            applied.append(artifact)
        return applied

    def _verify_environment(self, applied: list[dict[str, Any]]) -> dict[str, str]:
        """Confirm the installed versions match the lock and that `pip check` is clean."""
        if _artifact_identity(applied) != _artifact_identity(self.manifest["artifacts"]):
            raise _EnvironmentMismatch("combined apply evidence differs from the reviewed lock")
        installed: dict[str, str] = {}
        for item in self.manifest["artifacts"]:
            observed = importlib.metadata.version(item["name"])
            if observed != item["version"]:
                raise _EnvironmentMismatch(f"installed {item['name']} version {observed} differs"
                                           f" from the locked {item['version']}")
            installed[item["normalized_name"]] = observed
        checked = _run_child(_pip_argv("check"), environment=_PIP_ENV, cwd=self.report.parent)
        self.summary["pip_check"] = checked
        if checked["returncode"] != 0:
            raise _PipCheckFailed()
        return installed

    def run(self) -> int:
        try:
            with tempfile.TemporaryDirectory(prefix="nvcodec-pip-apply-",
                                             dir=self.report.parent) as temporary:
                root = Path(temporary)
                root.chmod(0o700)
                self.summary["private_staging_root"] = {"path": str(root), "mode": "0700"}
                applied = self._run_stages(root)
            installed = self._verify_environment(applied)
            evidence = _pynvc_wheel_evidence(self.wheels,
                                             self.paths["wheels"] if self.wheels else None)
        except _StageFailed as exc:
            return self._stop("apply_failed_mutation_possible", 2, failed_stage=exc.stage)
        except _PipCheckFailed:
            return self._stop("mutated_verification_failed", 2,
                              verification_error="pip check reported an inconsistent environment")
        except _EnvironmentMismatch as exc:
            return self._stop("mutated_verification_failed", 2, verification_error=str(exc))
        except _VerificationFailed as exc:
            self._stop("mutated_verification_failed", 3, failed_stage=exc.stage,
                       verification_error=str(exc.cause))
            raise exc.cause
        except Exception as exc:  # pylint: disable=broad-exception-caught
            if self.state["mutated"]:
                self._stop("mutated_verification_failed", 3, verification_error=str(exc))
            raise
        return self._stop("applied_verified", 0, installed_versions=installed,
                          pynvc_wheel=evidence,
                          applied_marker_sha256=_seal_applied_venv(self.digest, evidence))


def apply_lock(manifest_path: Path, install_report: Path, expected_manifest_sha256: str,
               build_environment_values: list[str]) -> int:
    """Apply a lock and emit exactly one persisted JSON result."""
    install_report = _fresh_output_path(install_report, label="pip install report")
    state = {"mutated": False}
    try:
        result = _Apply(manifest_path, install_report, expected_manifest_sha256,
                        build_environment_values, state).run()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        if not install_report.is_file():
            write_new_json(install_report, {
                "schema_version": SCHEMA_VERSION, "kind": "nvcodec-pip-lock-apply",
                "status": ("mutated_verification_failed" if state["mutated"]
                           else "validation_failed_no_mutation"),
                "manifest": str(_absolute(manifest_path)), "error": str(exc),
                "clean_venv_marker": None, "clean_venv_marker_status": None,
                "clean_venv_marker_sha256": None,
                "completed_at": utc_now()})
        _emit(install_report)
        return 2 if isinstance(exc, ApplyBlockedError) else 3
    _emit(install_report)
    return result


def materialize_apply(reports: list[Path], requirements: list[str],
                      source_requirements: list[str], lock_path: Path, manifest_path: Path,
                      install_report: Path, build_environment_values: list[str]) -> int:
    _preflight_apply_outputs(install_report)
    materialize(reports, requirements, source_requirements, lock_path, manifest_path)
    return apply_lock(manifest_path, install_report, _digest(manifest_path, "lock manifest"),
                      build_environment_values)


def _action_error(kind: str, output: Path | None, exc: BaseException,
                  **context: Any) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "kind": kind, "status": "error", **context,
            "output": str(output) if output is not None else None, "error": str(exc)}


def _action_main(output: Path | None, produce: Any, kind: str, **context: Any) -> int:
    """Run one action under isolation, then persist and echo exactly one JSON document."""
    try:
        require_isolated()
        if output is not None:
            _fresh_output_path(output, label="action report")
        result, exit_code = produce()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        result, exit_code = _action_error(kind, output, exc, **context), 3
    try:
        if output is not None:
            write_new_json(output, result)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        result, exit_code = _action_error(kind, output, exc, **context), 3
    print(_render(result))
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    venv_parser = actions.add_parser(
        "create-venv", help="create one new isolated venv without deleting or reusing a target")
    venv_parser.add_argument("--target", type=Path, required=True)
    venv_parser.add_argument("--output", type=Path)
    materialize_parser = actions.add_parser("materialize")
    pipeline_parser = actions.add_parser(
        "materialize-apply", help="materialize the reviewed lock and apply that exact digest")
    for lock_parser in (materialize_parser, pipeline_parser):
        lock_parser.add_argument("--report", action="append", type=Path, required=True)
        lock_parser.add_argument("--require", action="append", default=[])
        lock_parser.add_argument("--source-require", action="append", default=[])
        lock_parser.add_argument("--allowed-host", action="append")
        lock_parser.add_argument("--lock", type=Path, required=True)
        lock_parser.add_argument("--manifest", type=Path, required=True)
    pipeline_parser.add_argument("--install-report", type=Path, required=True)
    pipeline_parser.add_argument("--build-env", action="append", default=[])
    apply_parser = actions.add_parser("apply")
    apply_parser.add_argument("--manifest", type=Path, required=True)
    apply_parser.add_argument("--install-report", type=Path, required=True)
    apply_parser.add_argument("--expected-manifest-sha256", required=True)
    apply_parser.add_argument("--build-env", action="append", default=[])
    args = parser.parse_args()
    output = _absolute(args.output) if getattr(args, "output", None) is not None else None
    if args.action == "create-venv":
        target = _absolute(args.target)
        if output is not None and (output == target or target in output.parents):
            parser.error("--output must be outside the requested venv target")
        return _action_main(output, lambda: create_clean_venv(target),
                            "nvcodec-clean-venv-error", target=str(target))
    try:
        require_isolated()
        if args.action == "apply":
            return apply_lock(args.manifest, args.install_report,
                              args.expected_manifest_sha256, args.build_env)
        if args.allowed_host is not None and set(args.allowed_host) != set(DEFAULT_ALLOWED_HOSTS):
            raise ValueError("--allowed-host must name exactly the fixed approved host set")
        if args.action == "materialize":
            print(_render(materialize(args.report, args.require, args.source_require,
                                      args.lock, args.manifest)))
            return 0
        return materialize_apply(args.report, args.require, args.source_require, args.lock,
                                 args.manifest, args.install_report, args.build_env)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(_render({"schema_version": SCHEMA_VERSION, "kind": "nvcodec-pip-lock-error",
                       "status": "error", "error": str(exc)}))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
