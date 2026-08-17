#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private capability evidence and live PyNvVideoCodec authentication helpers.

Authentication is computed locally from the standard library only: the imported
package is checked for an exact version and every file the process actually
executed is proven to be owned by the installed wheel ``RECORD``. No artifact,
marker, or attestation produced by another skill is consumed.
"""

# pylint: disable=missing-function-docstring,protected-access
# pylint: disable=too-many-locals,unidiomatic-typecheck

from __future__ import annotations

import base64
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
import re
import stat
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

EXPECTED_PYNVC_VERSION = "2.1.0"
ENVIRONMENT_KIND = "nvcodec-environment"
ENVIRONMENT_SCHEMA_VERSION = "1.2"
ENVIRONMENT_SUPPORTED_MAJOR = 1
LOCAL_BINDING_KIND = "nvcodec-pynvc-runtime-binding"
LOCAL_BINDING_SCHEMA_VERSION = "1.0"
KNOWN_STALE_RECORD = "known_upstream_stale_record"
# The single evidenced RECORD exception: public PyNvVideoCodec 2.1.0 ships a stale
# RECORD row for its own native extension. Nothing else may miss its RECORD hash.
_NATIVE_EXTENSION_MEMBER = re.compile(
    r"PyNvVideoCodec/_?PyNvVideoCodec(?:_(?:121|130))?\.[^/]*\.so"
)
_IMPORT_VARIABLES = (
    "PYTHONPATH",
    "PYTHONHOME",
    "PYTHONSTARTUP",
    "PYTHONEXECUTABLE",
    "PYTHONUSERBASE",
    "PYTHONWARNINGS",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONOPTIMIZE",
    "LD_PRELOAD",
    "LD_AUDIT",
    "LD_LIBRARY_PATH",
    "LIBRARY_PATH",
    "DYLD_INSERT_LIBRARIES",
    "DYLD_LIBRARY_PATH",
    "DYLD_FRAMEWORK_PATH",
)
_IMPORT_SEAL = object()


class CapabilityAuthorityUnavailable(ValueError):
    """The selected live environment has no usable PyNvVideoCodec authority."""


def _setup_skill_path() -> Path:
    """Return the setup marker in the lexical installed-skill catalog."""
    return Path(__file__).absolute().parents[2] / "jetson-video-setup" / "SKILL.md"


def setup_dependency(*, surface: str, needed_for: str) -> dict[str, Any]:
    """Describe whether the canonical setup sibling can handle one remediation."""
    if surface not in {"native", "pynvc"}:
        raise ValueError("setup dependency surface must be native or pynvc")
    candidate = _setup_skill_path()
    try:
        details = candidate.lstat()
        resolved_root = candidate.parent.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
    except OSError:
        installed = False
    else:
        installed = bool(
            stat.S_ISREG(details.st_mode)
            and not stat.S_ISLNK(details.st_mode)
            and resolved_root.is_dir()
            and resolved == resolved_root / "SKILL.md"
        )
    action = (
        f"use jetson-video-setup to configure or repair the {surface} surface, "
        "then retry this capability query"
        if installed
        else f"install jetson-video-setup as a sibling skill, use it to configure "
        f"or repair the {surface} surface, then retry this capability query"
    )
    return {
        "skill": "jetson-video-setup",
        "installed": installed,
        "needed_for": needed_for,
        "next_action": action,
    }


def add_setup_remediation(
    result: dict[str, Any], *, surface: str, reason: str
) -> None:
    """Attach one additive, non-mutating, customer-actionable setup handoff."""
    result["remediation"] = {
        "route": "jetson-video-setup",
        "surface": surface,
        "mutation_performed": False,
        "reason": reason,
        "dependency": setup_dependency(
            surface=surface,
            needed_for=f"{surface} capability-query prerequisites",
        ),
    }


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-RFC JSON numeric constant is forbidden: {value}")


def strict_json_loads(value: str | bytes) -> Any:
    return json.loads(value, parse_constant=_reject_json_constant)


def _require_finite_json(value: Any, *, location: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite numeric value is forbidden at {location}")
    if isinstance(value, dict):
        for key, item in value.items():
            _require_finite_json(item, location=f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_finite_json(item, location=f"{location}[{index}]")


def strict_json_dumps(value: Any, **kwargs: Any) -> str:
    _require_finite_json(value)
    return json.dumps(value, allow_nan=False, **kwargs)


def exclusive_write_bytes(path: Path, content: bytes, *, mode: int = 0o600) -> Path:
    """Create one fresh regular file without following any symlink component."""
    requested = path.expanduser().absolute()
    if not requested.parent.is_dir():
        raise FileNotFoundError(
            f"evidence parent directory does not exist: {requested.parent}"
        )
    current = Path(requested.anchor)
    for part in requested.parent.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"evidence parent contains a symlink component: {current}")
    resolved = requested.parent.resolve(strict=True) / requested.name
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(resolved, flags, mode)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        resolved.unlink(missing_ok=True)
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return resolved


def exclusive_write_text(path: Path, content: str) -> None:
    exclusive_write_bytes(path, content.encode("utf-8"))


def _canonical_json_sha256(value: Any) -> str:
    rendered = strict_json_dumps(value, sort_keys=True, separators=(",", ":"))
    return _sha256_bytes(rendered.encode("utf-8"))


def _record_entry(distribution: Any, path: Path) -> tuple[Path, str, str, str]:
    resolved = path.expanduser().resolve()
    matching = [
        entry
        for entry in distribution.files or []
        if Path(distribution.locate_file(entry)).resolve() == resolved
    ]
    if len(matching) != 1:
        raise ValueError(
            f"expected one distribution file entry for {resolved}, found {len(matching)}"
        )
    record_text = distribution.read_text("RECORD")
    if not record_text:
        raise ValueError("installed PyNvVideoCodec distribution has no readable RECORD")
    entry_name = str(matching[0]).replace("\\", "/")
    rows = [
        row
        for row in csv.reader(io.StringIO(record_text))
        if row and row[0].replace("\\", "/") == entry_name
    ]
    if len(rows) != 1:
        raise ValueError(f"expected one wheel RECORD row for {entry_name}, found {len(rows)}")
    record_path, encoded_hash, recorded_size = (rows[0] + ["", ""])[:3]
    return resolved, record_path, encoded_hash, recorded_size


def _wheel_owned_identity(distribution: Any, path: Path, *, label: str) -> dict[str, Any]:
    """Prove one executed file is a wheel-RECORD member and hash it locally.

    Ownership is the assertion: the loaded path must resolve to exactly one
    installed distribution entry and exactly one ``RECORD`` row. The recorded
    digest is then re-derived here with :mod:`hashlib` and the comparison is
    published as evidence rather than silently trusted, so a known upstream
    ``RECORD`` inconsistency stays visible instead of requiring an external
    attestation to excuse it.
    """
    resolved, record_path, encoded_hash, recorded_size = _record_entry(distribution, path)
    entry_name = record_path.replace("\\", "/")
    observed_sha256 = _sha256_file(resolved)
    observed_size = resolved.stat().st_size
    record: dict[str, Any] = {
        "record_path": record_path,
        "hash": encoded_hash or None,
        "size_bytes": int(recorded_size) if str(recorded_size).isdigit() else None,
    }
    if not encoded_hash or "=" not in encoded_hash:
        record["consistency"] = "record_hash_absent"
    else:
        algorithm, expected = encoded_hash.split("=", 1)
        if algorithm.lower() != "sha256":
            raise ValueError(
                f"{label} wheel RECORD hash algorithm must be sha256, got {algorithm}"
            )
        actual = (
            base64.urlsafe_b64encode(bytes.fromhex(observed_sha256))
            .rstrip(b"=")
            .decode("ascii")
        )
        size_agrees = record["size_bytes"] in (None, observed_size)
        record["consistency"] = (
            "matches_record" if actual == expected and size_agrees else "upstream_mismatch"
        )
    return {
        "path": str(resolved),
        "sha256": observed_sha256,
        "size_bytes": observed_size,
        "wheel_member": entry_name,
        "record": record,
    }


def _require_record_consistency(
    identity: dict[str, Any], *, label: str, version: str, allow_native_stale: bool = False
) -> None:
    """Fail on a RECORD mismatch, permitting only the evidenced native-extension row.

    The exception is scoped by exact package version and by exact RECORD member:
    the public PyNvVideoCodec 2.1.0 native extension. Ownership of the executed
    file is still proven independently, so the exception narrows to the recorded
    digest alone and is published in the report rather than hidden.
    """
    record = identity["record"]
    if record["consistency"] == "matches_record":
        return
    member = identity["wheel_member"]
    if (
        allow_native_stale
        and record["consistency"] == "upstream_mismatch"
        and version == EXPECTED_PYNVC_VERSION
        and _NATIVE_EXTENSION_MEMBER.fullmatch(member) is not None
    ):
        record["consistency"] = KNOWN_STALE_RECORD
        record["exception"] = (
            f"known upstream PyNvVideoCodec {EXPECTED_PYNVC_VERSION} stale RECORD row for"
            " its native extension; wheel ownership of the executed file is still proven"
        )
        return
    raise ValueError(
        f"PyNvVideoCodec {label} does not match its wheel RECORD"
        f" ({record['consistency']}): {member}"
    )


def scalar_attributes(value: Any) -> dict[str, Any]:
    """Preserve every finite scalar field of a capability result, mapping or object."""
    if isinstance(value, Mapping):
        if any(isinstance(item, float) and not math.isfinite(item) for item in value.values()):
            raise ValueError("capability API returned a non-finite numeric field")
        return {
            str(key): item
            for key, item in value.items()
            if (isinstance(item, (str, int, bool)) or item is None)
            or (isinstance(item, float) and math.isfinite(item))
        }
    result: dict[str, Any] = {}
    for name in dir(value):
        if name.startswith("_"):
            continue
        try:
            item = getattr(value, name)
        except Exception:  # pylint: disable=broad-exception-caught
            continue
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("capability API returned a non-finite numeric field")
        if (isinstance(item, (str, int, bool)) or item is None) or (
            isinstance(item, float) and math.isfinite(item)
        ):
            result[name] = item
    return result


def linked_nvenc_api(module: Any) -> dict[str, Any]:
    """Read the NVENC API level the loaded extension is linked against."""
    suffix = getattr(module, "module_suffix", None)
    extension_file = getattr(getattr(module, "_PyNvVideoCodec", None), "__file__", None)
    combined = " ".join(str(item) for item in (suffix, extension_file) if item)
    value = "13.0" if "_130" in combined else ("12.1" if "_121" in combined else None)
    return {
        "status": "observed" if value else "unknown",
        "value": value,
        "module_suffix": str(suffix) if suffix else None,
        "extension_file": str(extension_file) if extension_file else None,
        "provenance": "loaded PyNvVideoCodec extension filename/module_suffix",
    }


def _active_import_environment() -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted((name, value) for name in _IMPORT_VARIABLES if (value := os.environ.get(name)))
    )


def _process_context() -> tuple[int, str, str, int]:
    return (
        os.getpid(),
        str(Path(sys.executable).resolve()),
        str(Path(sys.prefix).resolve()),
        id(os.environ),
    )


class _ImportValidation:
    """Opaque process-local proof that import inputs were validated."""

    __slots__ = ("_seal", "_context", "_module_id", "_environment")

    def __init__(self) -> None:
        self._seal = _IMPORT_SEAL
        self._context = _process_context()
        self._module_id: int | None = None
        self._environment: tuple[tuple[str, str], ...] | None = None


def _assert_isolated() -> None:
    if not sys.flags.ignore_environment or not sys.flags.no_user_site:
        raise ValueError(
            "Python authentication must run isolated with environment/user-site disabled"
            " (python -I)"
        )


def assert_safe_python_import_environment() -> _ImportValidation:
    ambient = _active_import_environment()
    if ambient:
        names = ", ".join(name for name, _value in ambient)
        raise ValueError(
            "ambient Python import injection variables must be absent during authentication: "
            + names
        )
    _assert_isolated()
    return _ImportValidation()


def _assert_preimport(validation: object, module: Any) -> _ImportValidation:
    if type(validation) is not _ImportValidation or validation._seal is not _IMPORT_SEAL:
        raise ValueError("PyNvVideoCodec pre-import environment validation is invalid")
    _assert_isolated()
    if validation._context != _process_context():
        raise ValueError(
            "PyNvVideoCodec pre-import environment validation belongs to a different process"
            " or interpreter"
        )
    current = _active_import_environment()
    if validation._environment is None:
        validation._module_id = id(module)
        validation._environment = current
    elif validation._module_id != id(module):
        raise ValueError(
            "PyNvVideoCodec pre-import environment validation is bound to another module"
        )
    elif validation._environment != current:
        raise ValueError(
            "Python import injection environment changed after PyNvVideoCodec authentication"
        )
    return validation


def live_pynvc_identity(
    module: Any,
    *,
    expected_version: str = EXPECTED_PYNVC_VERSION,
    preimport_validation: object | None = None,
) -> dict[str, Any]:
    """Authenticate the imported PyNvVideoCodec locally, with no external attestation.

    Three local facts are established with the standard library alone: the
    import happened under an isolated interpreter, the distribution and module
    both report the exact expected version, and each file the process actually
    executed is a wheel-``RECORD`` member inside the running ``sys.prefix``.

    The executed extension is taken from the imported ``_PyNvVideoCodec.__file__``
    rather than located through metadata, so ``loaded_path`` proves which binary
    the interpreter really ran instead of merely which one exists on disk.
    """
    if preimport_validation is None:
        assert_safe_python_import_environment()
    else:
        _assert_preimport(preimport_validation, module)
    distribution = importlib.metadata.distribution("PyNvVideoCodec")
    distribution_version = str(distribution.version)
    module_version = str(getattr(module, "__version__", "") or "")
    if distribution_version != expected_version or module_version != expected_version:
        raise ValueError(
            "PyNvVideoCodec distribution and imported module must both be exact public "
            f"{expected_version}"
        )
    module_path_value = getattr(module, "__file__", None)
    extension_path_value = getattr(getattr(module, "_PyNvVideoCodec", None), "__file__", None)
    if not module_path_value or not extension_path_value:
        raise ValueError("imported PyNvVideoCodec package/extension paths are unavailable")
    module_path = Path(module_path_value).resolve()
    extension_path = Path(extension_path_value).resolve()
    prefix = Path(sys.prefix).resolve()
    for label, owned_path in (("module", module_path), ("extension", extension_path)):
        if not owned_path.is_relative_to(prefix):
            raise ValueError(
                f"PyNvVideoCodec {label} path {owned_path} is outside current sys.prefix {prefix}"
            )
    linked_api = linked_nvenc_api(module)
    if linked_api["value"] is None:
        raise ValueError("loaded PyNvVideoCodec extension API suffix is not recognized")
    module_identity = _wheel_owned_identity(distribution, module_path, label="module")
    _require_record_consistency(
        module_identity, label="module", version=distribution_version
    )
    module_identity.update({"loaded_path": str(module_path_value), "version": module_version})
    extension_identity = _wheel_owned_identity(distribution, extension_path, label="extension")
    _require_record_consistency(
        extension_identity,
        label="extension",
        version=distribution_version,
        allow_native_stale=True,
    )
    extension_identity.update(
        {
            "loaded_path": str(extension_path_value),
            "linked_nvenc_api": linked_api["value"],
            "module_suffix": linked_api["module_suffix"],
        }
    )
    interpreter = Path(sys.executable).resolve(strict=True)
    return {
        "status": "verified",
        "authentication": "local_wheel_record_ownership",
        # Lexical: exactly what was invoked. Resolved identity: what actually ran.
        "interpreter": sys.executable,
        "interpreter_identity": {
            "path": str(interpreter),
            "sha256": _sha256_file(interpreter),
            "size_bytes": interpreter.stat().st_size,
        },
        "sys_prefix": str(prefix),
        "version": distribution_version,
        "module": module_identity,
        "extension": extension_identity,
    }


def require_environment_schema(data: Any) -> str:
    """Reject an unknown major first, then pin the exact supported environment."""
    if not isinstance(data, dict):
        raise ValueError("environment manifest must be a JSON object")
    if data.get("kind") != ENVIRONMENT_KIND:
        raise ValueError(f"environment manifest kind must be {ENVIRONMENT_KIND}")
    version = data.get("schema_version")
    if not isinstance(version, str) or re.fullmatch(r"\d+\.\d+", version) is None:
        raise ValueError("environment schema_version must be MAJOR.MINOR")
    if int(version.split(".", 1)[0]) != ENVIRONMENT_SUPPORTED_MAJOR:
        raise ValueError(
            f"unknown {ENVIRONMENT_KIND} major in {version}; this skill implements major"
            f" {ENVIRONMENT_SUPPORTED_MAJOR}"
        )
    if version != ENVIRONMENT_SCHEMA_VERSION:
        raise ValueError(
            f"{ENVIRONMENT_KIND} {version} is not the pinned {ENVIRONMENT_SCHEMA_VERSION}"
        )
    if data.get("mode") != "live":
        raise ValueError("capability queries require a live environment manifest")
    return version


def _require_same_file(value: Any, expected: str, *, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"environment pynvc.identity.{label} is missing or malformed")
    if str(Path(value).expanduser().resolve()) != expected:
        raise ValueError(
            f"environment pynvc.identity.{label} is not the file this process actually loaded"
        )


def declared_pynvc_identity(data: dict[str, Any]) -> dict[str, Any]:
    """Return the declared ``pynvc.identity`` block of a 1.2 manifest.

    Schema 1.2 publishes the Python surface as the top-level ``pynvc`` member,
    whose ``identity`` block is what this skill authenticated at probe time. An
    absent or unverified authority is a classification, not a hard failure, so it
    raises :class:`CapabilityAuthorityUnavailable` rather than ``ValueError``.
    """
    surface = data.get("pynvc")
    if not isinstance(surface, dict) or surface.get("imported") is not True:
        raise CapabilityAuthorityUnavailable(
            "environment manifest reports no imported pynvc surface"
        )
    identity = surface.get("identity")
    if not isinstance(identity, dict) or identity.get("status") != "verified":
        raise CapabilityAuthorityUnavailable(
            "environment pynvc.identity is not a verified authority"
        )
    return identity


def declared_distribution_version(declared: dict[str, Any]) -> Any:
    """Read the manifest's PyNvVideoCodec distribution version by either spelling.

    The established 1.2 producer carries it as ``identity.distribution``; this
    skill additionally publishes the flattened ``identity.version``. Both mean the
    installed distribution version, so either satisfies the cross-check and
    neither is required on its own. ``distribution`` is accepted as the bare
    version or as a mapping carrying one, which keeps this a required-subset read.
    """
    if "version" in declared:
        return declared.get("version")
    distribution = declared.get("distribution")
    if isinstance(distribution, Mapping):
        return distribution.get("version")
    return distribution


def require_pynvc_identity(data: dict[str, Any], identity: dict[str, Any]) -> dict[str, Any]:
    """Validate only the required ``pynvc.identity`` subset against the live process.

    Additive optional keys anywhere in the block are ignored by construction:
    every check names one required field, and nested blocks are compared key by
    required key rather than by set equality. ``extension.loaded_path`` and
    ``module.{version,path}`` are compared against the imported objects, so a
    manifest that describes a different binary than the one executed fails closed.

    ``identity.version`` and ``identity.interpreter_identity`` are additive
    refinements, not requirements: the established producer closes ``identity`` to
    ``{status, interpreter, sys_prefix, dist_info_path, distribution, module,
    extension}``. The distribution version is read through whichever spelling is
    present, and a resolved interpreter identity is held to the exact running
    interpreter only when the manifest actually publishes one. The lexical
    ``identity.interpreter`` is required either way, so interpreter agreement is
    still asserted for a manifest that carries no resolved identity.
    """
    declared = declared_pynvc_identity(data)
    for field in ("interpreter", "sys_prefix"):
        if declared.get(field) != identity[field]:
            raise ValueError(
                f"environment pynvc.identity.{field} does not match the live PyNvVideoCodec import"
            )
    if declared_distribution_version(declared) != identity["version"]:
        raise ValueError(
            "environment pynvc.identity distribution version does not match the live"
            " PyNvVideoCodec import"
        )
    for field, additive, required in (
        (
            "interpreter_identity",
            True,
            {
                "path": identity["interpreter_identity"]["path"],
                "sha256": identity["interpreter_identity"]["sha256"],
            },
        ),
        (
            "extension",
            False,
            {
                "path": identity["extension"]["path"],
                "sha256": identity["extension"]["sha256"],
            },
        ),
        ("module", False, {"version": identity["module"]["version"]}),
    ):
        observed = declared.get(field)
        if observed is None and additive:
            continue
        if not isinstance(observed, dict):
            raise ValueError(f"environment pynvc.identity.{field} is missing or malformed")
        for key, value in required.items():
            if observed.get(key) != value:
                raise ValueError(
                    f"environment pynvc.identity.{field}.{key} differs from the live process"
                )
    _require_same_file(
        declared["extension"].get("loaded_path"),
        identity["extension"]["path"],
        label="extension.loaded_path",
    )
    _require_same_file(
        declared["module"].get("path"), identity["module"]["path"], label="module.path"
    )
    return declared


def require_declared_pynvc_authority(path: Path, *, gpu: int) -> None:
    """Classify an absent Py surface before importing its runtime authority."""
    requested = path.expanduser().absolute()
    if requested.is_symlink():
        raise ValueError("environment manifest must not be a symlink")
    data = strict_json_loads(requested.read_bytes())
    require_environment_schema(data)
    if data.get("selected_gpu") != gpu:
        raise ValueError("environment selected_gpu does not match --gpu")
    declared_pynvc_identity(data)


def authenticated_environment(
    path: Path,
    *,
    gpu: int,
    module: Any,
    preimport_validation: object | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind capability queries to the exact live 1.2 manifest and the imported wheel."""
    requested = path.expanduser().absolute()
    if requested.is_symlink():
        raise ValueError("environment manifest must not be a symlink")
    data = strict_json_loads(requested.read_bytes())
    require_environment_schema(data)
    if data.get("selected_gpu") != gpu:
        raise ValueError("environment selected_gpu does not match --gpu")
    identity = live_pynvc_identity(module, preimport_validation=preimport_validation)
    require_pynvc_identity(data, identity)
    # environment_binding re-reads and compares the snapshot so replacement races fail closed.
    binding = environment_binding(requested, data, identity)
    binding["selected_gpu"] = gpu
    return data, binding


def local_pynvc_binding(
    module: Any,
    *,
    gpu: int,
    preimport_validation: object | None = None,
) -> dict[str, Any]:
    """Authenticate the exact invoking interpreter and wheel without setup evidence."""
    identity = live_pynvc_identity(
        module, preimport_validation=preimport_validation
    )
    return {
        "schema_version": LOCAL_BINDING_SCHEMA_VERSION,
        "kind": LOCAL_BINDING_KIND,
        "mode": "live",
        "selected_gpu": gpu,
        "interpreter": sys.executable,
        "interpreter_identity": identity["interpreter_identity"],
        "pynvc_identity": identity,
    }


def require_local_pynvc_binding(binding: Any, *, gpu: int) -> None:
    """Require the closed process-local binding produced above."""
    if not isinstance(binding, dict):
        raise ValueError("local PyNvVideoCodec runtime binding is mandatory")
    expected = {
        "schema_version": LOCAL_BINDING_SCHEMA_VERSION,
        "kind": LOCAL_BINDING_KIND,
        "mode": "live",
        "selected_gpu": gpu,
        "interpreter": sys.executable,
    }
    if set(binding) != {*expected, "interpreter_identity", "pynvc_identity"}:
        raise ValueError("local PyNvVideoCodec runtime binding shape is invalid")
    for field, value in expected.items():
        if binding.get(field) != value:
            raise ValueError(
                f"local PyNvVideoCodec runtime binding {field} is not authenticated"
            )
    identity = binding.get("pynvc_identity")
    interpreter_identity = binding.get("interpreter_identity")
    if (
        not isinstance(identity, dict)
        or identity.get("status") != "verified"
        or identity.get("authentication") != "local_wheel_record_ownership"
        or not isinstance(interpreter_identity, dict)
        or interpreter_identity != identity.get("interpreter_identity")
    ):
        raise ValueError("local PyNvVideoCodec wheel/interpreter identity is invalid")


def reauthenticate_local_pynvc_binding(
    binding: dict[str, Any],
    *,
    gpu: int,
    module: Any,
    preimport_validation: object | None = None,
) -> None:
    """Recompute and compare one local binding at a query boundary."""
    require_local_pynvc_binding(binding, gpu=gpu)
    observed = local_pynvc_binding(
        module, gpu=gpu, preimport_validation=preimport_validation
    )
    if observed != binding:
        raise ValueError(
            "local PyNvVideoCodec wheel or interpreter identity changed during queries"
        )


def require_authenticated_binding(environment: Any, *, gpu: int) -> None:
    """Require the required binding subset produced by ``authenticated_environment``."""
    if not isinstance(environment, dict):
        raise ValueError("authenticated environment binding is mandatory")
    expected = {
        "schema_version": ENVIRONMENT_SCHEMA_VERSION,
        "kind": ENVIRONMENT_KIND,
        "mode": "live",
        "selected_gpu": gpu,
        "interpreter": sys.executable,
    }
    for field, value in expected.items():
        if environment.get(field) != value:
            raise ValueError(f"capability environment binding {field} is not authenticated")
    if not isinstance(environment.get("path"), str) or not environment["path"]:
        raise ValueError("capability environment binding has no manifest path")
    if type(environment.get("size_bytes")) is not int:
        raise ValueError("capability environment binding has no manifest size")
    for field in ("sha256", "canonical_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", str(environment.get(field))) is None:
            raise ValueError(f"capability environment binding {field} is not a SHA-256 digest")
    identity = environment.get("interpreter_identity")
    if not isinstance(identity, dict) or not isinstance(identity.get("path"), str):
        raise ValueError("capability environment binding has no resolved interpreter identity")


def _regular_file_snapshot(
    path: Path, *, label: str
) -> tuple[bytes, dict[str, Any]]:
    requested = Path(os.path.abspath(os.fspath(path.expanduser())))
    if not requested.parent.is_dir():
        raise FileNotFoundError(f"{label} parent directory does not exist: {requested.parent}")
    current = Path(requested.anchor)
    for part in requested.parent.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{label} parent contains a symlink component: {current}")
    if requested.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {requested}")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(requested, flags)
    try:
        opened_stat = os.fstat(descriptor)
        if not stat.S_ISREG(opened_stat.st_mode):
            raise ValueError(f"{label} must be a regular file: {requested}")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            raw = handle.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    terminal_stat = os.stat(requested, follow_symlinks=False)
    opened_key = (opened_stat.st_dev, opened_stat.st_ino, opened_stat.st_size)
    terminal_key = (terminal_stat.st_dev, terminal_stat.st_ino, terminal_stat.st_size)
    if opened_key != terminal_key or not stat.S_ISREG(terminal_stat.st_mode):
        raise ValueError(f"{label} identity changed while it was being read")
    resolved = requested.parent.resolve(strict=True) / requested.name
    return raw, {
        "path": str(resolved),
        "device": opened_stat.st_dev,
        "inode": opened_stat.st_ino,
        "size_bytes": len(raw),
        "sha256": _sha256_bytes(raw),
    }


def environment_binding(
    path: Path, data: dict[str, Any], identity: dict[str, Any]
) -> dict[str, Any]:
    """Bind a report to the exact environment artifact bytes it was produced from.

    The recorded ``{path, size_bytes, sha256, canonical_sha256}`` is what lets a
    downstream consumer reject a report produced against a different environment
    and detect an environment that changed after production.
    """
    raw, initial = _regular_file_snapshot(path, label="environment manifest")
    parsed = strict_json_loads(raw)
    if parsed != data:
        raise ValueError("environment manifest changed after it was loaded")
    terminal_raw, terminal = _regular_file_snapshot(path, label="environment manifest")
    if terminal_raw != raw or terminal != initial:
        raise ValueError("environment manifest inode/hash changed during authentication")
    return {
        "path": initial["path"],
        "size_bytes": initial["size_bytes"],
        "sha256": _sha256_bytes(raw),
        "canonical_sha256": _canonical_json_sha256(parsed),
        "schema_version": data.get("schema_version"),
        "kind": data.get("kind"),
        "mode": data.get("mode"),
        # Lexical: exactly what was invoked. Resolved identity: what actually ran.
        "interpreter": sys.executable,
        "interpreter_identity": identity["interpreter_identity"],
        "pynvc_identity": identity,
    }


def enum_member(module: Any, containers: tuple[str, ...], names: tuple[str, ...]) -> Any:
    for container_name in containers:
        container = getattr(module, container_name, None)
        if container is None:
            continue
        for name in names:
            if hasattr(container, name):
                return getattr(container, name)
    for name in names:
        if hasattr(module, name):
            return getattr(module, name)
    return None
