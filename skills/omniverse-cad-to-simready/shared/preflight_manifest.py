#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
from typing import Any
from urllib.request import Request, urlopen


PREFLIGHT_MANIFEST_ENV = "PHYSICAL_AI_PREFLIGHT_MANIFEST"
PREFLIGHT_REQUIRED_ENV = "PHYSICAL_AI_REQUIRE_PREFLIGHT"
DEFAULT_MANIFEST_NAME = "cad-to-simready-preflight.json"
SERVICE_HEALTH_CHECK_TIMEOUT_SECONDS = 5.0
_ENTRY_PATH_KEYS = ("root", "path", "venv")


def cad_to_simready_home() -> Path:
    return Path(os.environ.get("OMNIVERSE_CAD_TO_SIMREADY_HOME", "~/.omniverse-cad-to-simready")).expanduser()


def default_manifest_path() -> Path:
    state_root = os.environ.get("OMNIVERSE_CAD_TO_SIMREADY_STATE")
    if state_root:
        return Path(state_root).expanduser() / DEFAULT_MANIFEST_NAME
    return cad_to_simready_home() / "state" / DEFAULT_MANIFEST_NAME


def configured_manifest_path() -> Path:
    explicit = os.environ.get(PREFLIGHT_MANIFEST_ENV)
    return Path(explicit).expanduser() if explicit else default_manifest_path()


def load_preflight_manifest(path: Path | None = None) -> tuple[dict[str, Any] | None, Path, str | None]:
    manifest_path = path.expanduser() if path is not None else configured_manifest_path()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, manifest_path, f"preflight manifest was not found: {manifest_path}"
    except (OSError, json.JSONDecodeError) as exc:
        return None, manifest_path, f"preflight manifest could not be read: {exc}"
    if not isinstance(payload, dict):
        return None, manifest_path, f"preflight manifest is not a JSON object: {manifest_path}"
    return payload, manifest_path, None


def preflight_required() -> bool:
    return os.environ.get(PREFLIGHT_REQUIRED_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def preflight_block_message(target: str, path: Path | None = None) -> str:
    manifest_path = path or configured_manifest_path()
    return (
        f"cad-to-simready preflight has not prepared `{target}`. "
        "Run `preflight/scripts/preflight.py` for the required targets and source the generated env file, "
        f"or set {PREFLIGHT_MANIFEST_ENV} to a ready manifest. Expected manifest: {manifest_path}"
    )


def _is_remote_or_invocation_endpoint(base_url: str) -> bool:
    lowered = base_url.lower()
    return lowered.startswith("https://") or "nvcf" in lowered or "invocation" in lowered


def _service_health_urls(base_url: str) -> list[str]:
    clean = base_url.rstrip("/")
    return [f"{clean}/health", f"{clean}/v2/health/ready"]


def _service_url_is_healthy(base_url: str, timeout: float = SERVICE_HEALTH_CHECK_TIMEOUT_SECONDS) -> bool:
    """Lightweight liveness probe for a manifest-recorded service base URL.

    Mirrors the health-check convention the preflight bootstrap itself uses
    (`GET {base_url}/health` or `/v2/health/ready`, HTTP 2xx = healthy) so a
    consumer does not keep trusting a service that was healthy when preflight
    ran but has since stopped. Remote/invocation endpoints are not probed
    here, consistent with how preflight itself treats them as pre-authenticated
    and not reachable with a generic unauthenticated GET.
    """
    if _is_remote_or_invocation_endpoint(base_url):
        return True
    for url in _service_health_urls(base_url):
        try:
            with urlopen(Request(url, method="GET"), timeout=timeout) as response:
                if 200 <= response.status < 300:
                    return True
        except Exception:
            continue
    return False


def _path_is_present(value: Any) -> bool:
    if not value:
        return False
    return Path(str(value)).expanduser().exists()


def _executable_is_present(value: Any) -> bool:
    if not value:
        return False
    text = str(value)
    if _path_is_present(text):
        return True
    has_separator = os.sep in text or (os.altsep is not None and os.altsep in text)
    return not has_separator and shutil.which(text) is not None


def _entry_freshness_error(entry: dict[str, Any]) -> str | None:
    """Return a reason if a recorded `ready`/`present` entry no longer reflects reality.

    A preflight manifest is a point-in-time snapshot. If the checkout, venv,
    executable, or service it recorded as `ready` has since disappeared or
    stopped responding, callers must not keep trusting the cached status.
    Returns None when the entry still checks out.
    """
    base_url = entry.get("base_url")
    if base_url:
        base_url = str(base_url)
        if not _service_url_is_healthy(base_url):
            return f"recorded service at {base_url} is no longer reachable"
        return None
    for key in _ENTRY_PATH_KEYS:
        value = entry.get(key)
        if value and not _path_is_present(value):
            return f"recorded {key} no longer exists: {value}"
    executable = entry.get("executable")
    if executable and not _executable_is_present(executable):
        return f"recorded executable no longer exists: {executable}"
    return None


def runtime_entry(manifest: dict[str, Any] | None, name: str) -> dict[str, Any] | None:
    if not manifest:
        return None
    runtimes = manifest.get("runtimes")
    if not isinstance(runtimes, dict):
        return None
    entry = runtimes.get(name)
    return entry if isinstance(entry, dict) else None


def upstream_entry(manifest: dict[str, Any] | None, name: str) -> dict[str, Any] | None:
    if not manifest:
        return None
    upstreams = manifest.get("upstreams")
    if not isinstance(upstreams, dict):
        return None
    entry = upstreams.get(name)
    return entry if isinstance(entry, dict) else None


def service_entry(manifest: dict[str, Any] | None, name: str) -> dict[str, Any] | None:
    if not manifest:
        return None
    services = manifest.get("services")
    if not isinstance(services, dict):
        return None
    entry = services.get(name)
    return entry if isinstance(entry, dict) else None


def manifest_env_value(manifest: dict[str, Any] | None, name: str) -> str | None:
    if not manifest:
        return None
    env = manifest.get("env")
    if not isinstance(env, dict):
        return None
    value = env.get(name)
    return str(value) if value else None


def manifest_path_value(entry: dict[str, Any] | None, key: str) -> Path | None:
    if not entry:
        return None
    value = entry.get(key)
    if not value:
        return None
    path = Path(str(value)).expanduser().resolve()
    return path if path.exists() else None


def ready_path_from_runtime(manifest: dict[str, Any] | None, runtime_name: str, key: str = "root") -> Path | None:
    entry = runtime_entry(manifest, runtime_name)
    if not entry or entry.get("status") != "ready":
        return None
    return manifest_path_value(entry, key)


def ready_path_from_upstream(manifest: dict[str, Any] | None, upstream_name: str) -> Path | None:
    entry = upstream_entry(manifest, upstream_name)
    if not entry or entry.get("status") not in {"ready", "present"}:
        return None
    return manifest_path_value(entry, "path")


def ready_executable_from_runtime(manifest: dict[str, Any] | None, runtime_name: str) -> str | None:
    entry = runtime_entry(manifest, runtime_name)
    if not entry or entry.get("status") != "ready":
        return None
    executable = entry.get("executable")
    if not executable or not _executable_is_present(executable):
        return None
    return str(executable)


def ready_service_url(manifest: dict[str, Any] | None, service_name: str) -> str | None:
    entry = service_entry(manifest, service_name)
    if entry and entry.get("status") == "ready" and entry.get("base_url"):
        base_url = str(entry["base_url"]).rstrip("/")
        if _service_url_is_healthy(base_url):
            return base_url
    env_name_by_service = {
        "material": "CONTENT_AGENTS_MATERIAL_AGENT_BASE_URL",
        "physics": "CONTENT_AGENTS_PHYSICS_AGENT_BASE_URL",
        "texture": "CONTENT_AGENTS_TEXTURE_AGENT_BASE_URL",
        "ovrtx": "RENDER_ENDPOINT",
    }
    env_name = env_name_by_service.get(service_name)
    value = manifest_env_value(manifest, env_name) if env_name else None
    if not value:
        return None
    # The manifest `env` block mirrors the same base URLs preflight recorded under
    # `services`, so it needs the same liveness probe; otherwise a stopped service
    # is still handed back through this fallback.
    base_url = value.rstrip("/")
    return base_url if _service_url_is_healthy(base_url) else None


def preflight_status_check(target: str, component: str) -> dict[str, Any]:
    manifest, path, error = load_preflight_manifest()
    if error:
        return {
            "name": f"preflight_{target}_ready",
            "passed": False,
            "severity": "error",
            "message": preflight_block_message(target, path),
        }
    entry = runtime_entry(manifest, component) or service_entry(manifest, component) or upstream_entry(manifest, component)
    passed = bool(entry and entry.get("status") in {"ready", "present"})
    staleness_reason: str | None = None
    if passed:
        staleness_reason = _entry_freshness_error(entry)
        if staleness_reason:
            passed = False
    detail = f"Preflight manifest: {path}"
    if entry:
        detail = f"{detail}; {component} status: {entry.get('status')}"
        if staleness_reason:
            detail = f"{detail}; {staleness_reason}"
    return {
        "name": f"preflight_{target}_ready",
        "passed": passed,
        "severity": "error",
        "message": detail if passed else preflight_block_message(target, path),
    }
