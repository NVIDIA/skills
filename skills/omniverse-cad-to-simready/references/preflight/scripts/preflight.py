#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare local cad-to-simready dependencies and write a preflight manifest.

Usage:
    python3 scripts/preflight.py [--check-only]
    python3 scripts/preflight.py --report preflight.json --env-file preflight.env

Arguments:
    --check-only              Verify local readiness without cloning, installing, or deploying.
    --skip-content-agents     Do not verify or deploy Content Agents services.
    --skip-deploy             Verify Content Agents endpoints but do not start services.
    --report PATH             Write the preflight manifest JSON.
    --env-file PATH           Write shell exports for downstream references.
    --powershell-env-file PATH
                              Write PowerShell env assignments for downstream references.
    --markdown-report PATH    Write a human-readable readiness report.

Exit codes:
    0 - dependencies and services are ready or explicitly skipped
    1 - one or more dependencies or services are blocked
    2 - unexpected error (crash or malformed input)
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass, field
from datetime import datetime, timezone
import ipaddress
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, urlunparse
from urllib.request import Request, urlopen

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared"))

from script_utils import tail_text
from upstream_versions import (
    UPSTREAM_VERSION_MANIFEST_PATH,
    expected_versions,
    git_upstreams,
    package_requirement,
    package_version,
    runtime_package_names,
    runtime_requirements,
    version_mismatches,
    version_probe_code,
)


SKILL = "cad-to-simready-preflight"
SCHEMA_VERSION = "1.0"
DEFAULT_TARGETS = ("conversion", "validation", "content-agents")
SECRET_NAME_PARTS = ("KEY", "TOKEN", "SECRET", "PASSWORD")
DEFAULT_SERVICE_URLS = {
    "ovrtx": "http://localhost:8001",
    "material": "http://localhost:8100",
    "physics": "http://localhost:8200",
    "texture": "http://localhost:8300",
}
SMOKE_CAMERA_TRANSLATE_X = 3
SMOKE_CAMERA_TRANSLATE_Y = 3
SMOKE_CAMERA_TRANSLATE_Z = 3
SMOKE_CAMERA_ROTATE_X = -30
SMOKE_CAMERA_ROTATE_Y = 45
SMOKE_CAMERA_ROTATE_Z = 0
SMOKE_LIGHT_ROTATE_X = -45
SMOKE_LIGHT_ROTATE_Y = 30
SMOKE_LIGHT_ROTATE_Z = 0
SMOKE_CAMERA_TRANSLATE = f"{SMOKE_CAMERA_TRANSLATE_X}, {SMOKE_CAMERA_TRANSLATE_Y}, {SMOKE_CAMERA_TRANSLATE_Z}"
SMOKE_CAMERA_ROTATE = f"{SMOKE_CAMERA_ROTATE_X}, {SMOKE_CAMERA_ROTATE_Y}, {SMOKE_CAMERA_ROTATE_Z}"
SMOKE_LIGHT_ROTATE = f"{SMOKE_LIGHT_ROTATE_X}, {SMOKE_LIGHT_ROTATE_Y}, {SMOKE_LIGHT_ROTATE_Z}"
OVRTX_RENDER_SMOKE_USDA_TEMPLATE = """#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Y"
)

def Xform "World"
{
    def Cube "Cube"
    {
        double size = 1
    }

    def Camera "Camera"
    {
        float focalLength = 35
        float horizontalAperture = 36
        float verticalAperture = 36
        float2 clippingRange = (0.1, 1000)
        double3 xformOp:translate = (__SMOKE_CAMERA_TRANSLATE__)
        float3 xformOp:rotateXYZ = (__SMOKE_CAMERA_ROTATE__)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]
    }

    def DistantLight "KeyLight"
    {
        float intensity = 5000
        float3 xformOp:rotateXYZ = (__SMOKE_LIGHT_ROTATE__)
        uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
    }
}
"""
OVRTX_RENDER_SMOKE_USDA = (
    OVRTX_RENDER_SMOKE_USDA_TEMPLATE.replace("__SMOKE_CAMERA_TRANSLATE__", SMOKE_CAMERA_TRANSLATE)
    .replace("__SMOKE_CAMERA_ROTATE__", SMOKE_CAMERA_ROTATE)
    .replace("__SMOKE_LIGHT_ROTATE__", SMOKE_LIGHT_ROTATE)
)
CONTENT_AGENTS_SECRET_ENV_NAMES = (
    "NVIDIA_API_KEY",
    "NGC_API_KEY",
    "NVCF_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "WU_NIM_API_KEY",
    "MA_NVIDIA_API_KEY",
    "MA_IMAGE_GEN_API_KEY",
    "MA_CLUSTER_EMBEDDING_API_KEY",
    "MA_NIM_API_KEY",
    "PA_NVIDIA_API_KEY",
    "PA_NIM_API_KEY",
    "TA_IMAGE_GEN_API_KEY",
)
CONTENT_AGENTS_PROVIDER_SECRET_ENV_NAMES = (
    "NVIDIA_API_KEY",
    "NGC_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "WU_NIM_API_KEY",
    "MA_NVIDIA_API_KEY",
    "MA_NIM_API_KEY",
    "PA_NVIDIA_API_KEY",
    "PA_NIM_API_KEY",
    "TA_IMAGE_GEN_API_KEY",
)
CONTENT_AGENTS_BACKEND_API_KEY_ENV_NAMES = {
    "nim": ("NVIDIA_API_KEY", "WU_NIM_API_KEY", "MA_NIM_API_KEY", "PA_NIM_API_KEY"),
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "gemini": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
}
CONTENT_AGENTS_DEPENDENCY_API_KEY_ENV_NAMES = {
    "image_gen": ("MA_IMAGE_GEN_API_KEY", "TA_IMAGE_GEN_API_KEY"),
}
CONTENT_AGENTS_MODEL_ENV_NAMES = (
    "CONTENT_AGENTS_VLM_BACKEND",
    "CONTENT_AGENTS_VLM_MODEL",
    "CONTENT_AGENTS_VLM_ENDPOINT",
    "CONTENT_AGENTS_VLM_API_KEY_ENV",
    "MA_VLM_API_KEY_ENV",
    "PA_VLM_API_KEY_ENV",
    "CONTENT_AGENTS_LLM_BACKEND",
    "CONTENT_AGENTS_LLM_MODEL",
    "CONTENT_AGENTS_LLM_ENDPOINT",
    "CONTENT_AGENTS_LLM_API_KEY_ENV",
    "MA_LLM_API_KEY_ENV",
    "TA_LLM_API_KEY_ENV",
    "CONTENT_AGENTS_IMAGE_GEN_BACKEND",
    "CONTENT_AGENTS_IMAGE_GEN_MODEL",
    "CONTENT_AGENTS_IMAGE_GEN_ENDPOINT",
    "CONTENT_AGENTS_IMAGE_GEN_API_KEY_ENV",
    "MA_IMAGE_GEN_API_KEY_ENV",
    "TA_IMAGE_GEN_API_KEY_ENV",
)
CONTENT_AGENTS_API_KEY_ENV_NAME_ENV_NAMES = (
    "CONTENT_AGENTS_VLM_API_KEY_ENV",
    "MA_VLM_API_KEY_ENV",
    "PA_VLM_API_KEY_ENV",
    "CONTENT_AGENTS_LLM_API_KEY_ENV",
    "MA_LLM_API_KEY_ENV",
    "TA_LLM_API_KEY_ENV",
    "CONTENT_AGENTS_IMAGE_GEN_API_KEY_ENV",
    "MA_IMAGE_GEN_API_KEY_ENV",
    "TA_IMAGE_GEN_API_KEY_ENV",
)
CREDENTIAL_ENV_NAME_PATTERN = re.compile(r"^[A-Z_][A-Z0-9_]*$")
SIMREADY_RUNTIME_REQUIREMENTS = runtime_requirements("simready_validate")
SIMREADY_NO_DEPS_REQUIREMENTS = runtime_requirements("simready_validate", no_deps=True)
SIMREADY_RUNTIME_PACKAGE_NAMES = (
    *runtime_package_names("simready_validate"),
    *runtime_package_names("simready_validate", no_deps=True),
)
DEFAULT_CONVERSION_TOOLS = {
    "repo-python",
    "usd-convert-cad",
    "usd-convert-gsplat",
}
UV_INSTALL_HINT = "Install uv with: curl -LsSf https://astral.sh/uv/install.sh | sh"
OPENUSD_IMPORT_CHECK = "from pxr import Usd, UsdGeom, UsdPhysics; print(Usd.GetVersion())"
USD_CONVERT_CAD_PYTHON_ENV = "USD_CONVERT_CAD_PYTHON"
USD_CONVERT_CAD_PIP_PACKAGE = package_requirement("usd-convert-cad")
USD_CONVERT_CAD_VERSION = package_version("usd-convert-cad")
USD_CONVERT_CAD_IMPORT_CHECK = (
    "import usd_convert_cad as m; "
    "print(getattr(m, '__version__', None) or "
    "(m.get_version() if hasattr(m, 'get_version') else 'unknown'))"
)
USD_CONVERT_CAD_INSTALL_HINT = (
    "usd-convert-cad 0.2 is a self-contained wheel. Install it into an isolated Python 3.12 "
    f"environment and expose that interpreter via {USD_CONVERT_CAD_PYTHON_ENV}: "
    'python3.12 -m venv "$HOME/.omniverse-cad-to-simready/venvs/usd-convert-cad" '
    '&& "$HOME/.omniverse-cad-to-simready/venvs/usd-convert-cad/bin/python" -m pip install '
    f"{USD_CONVERT_CAD_PIP_PACKAGE} --extra-index-url https://pypi.nvidia.com "
    f'&& export {USD_CONVERT_CAD_PYTHON_ENV}="$HOME/.omniverse-cad-to-simready/venvs/usd-convert-cad/bin/python"'
)
ASSET_VALIDATOR_IMPORT_CHECK = "import omni.asset_validator"
USD_SUFFIXES = {".usd", ".usda", ".usdc", ".usdz"}
REPO_PYTHON_SOURCE_FORMATS = {"urdf", "mjcf", "mujoco"}
# Source suffixes usd-convert-cad advertises in its Supported Formats table
# (`.agents/skills/usd-convert-cad/SKILL.md`). `.ply` intentionally routes to
# usd-convert-gsplat, which is checked ahead of CAD in `_normalized_source_format`.
CAD_SOURCE_SUFFIXES = {
    ".3dm",
    ".3ds",
    ".3dxml",
    ".3mf",
    ".asm",
    ".catpart",
    ".catproduct",
    ".cgr",
    ".dae",
    ".dgn",
    ".dwg",
    ".dxf",
    ".fbx",
    ".glb",
    ".gltf",
    ".iam",
    ".ifc",
    ".ifczip",
    ".iges",
    ".igs",
    ".ipt",
    ".jt",
    ".obj",
    ".par",
    ".prt",
    ".psm",
    ".pwd",
    ".rfa",
    ".rvt",
    ".sab",
    ".sat",
    ".sldasm",
    ".sldprt",
    ".step",
    ".stl",
    ".stp",
    ".x_b",
    ".x_t",
    ".xmt",
    ".xmt_txt",
}
GSPLAT_SOURCE_SUFFIXES = {".ply", ".splat", ".ksplat"}
DOCKER_HOST_ALIAS = "host.docker.internal"
# Hosts that name the local machine from outside a container and therefore have
# to be rewritten to the Docker host alias before an agent container can reach
# them.
LOCAL_RENDER_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
# Hosts that mean "the renderer runs on this machine", including the Docker host
# alias, which is already container-reachable and so needs no rewrite.
LOCAL_RENDER_ENDPOINT_HOSTS = LOCAL_RENDER_HOSTS | {DOCKER_HOST_ALIAS}
# Hosts upstream Content Agents readiness accepts as a local renderer. Mirrors
# `_LOCAL_RENDER_HOSTS` in the upstream collection's
# `apps/{material,physics}_agent_service/service/config.py`, read at
# `nvidia-omniverse/content-agents` `v0.5.2` (`36dbf3f`). Upstream matches the
# hostname exactly, so a host-local renderer named any other way -- notably the
# Docker host alias -- fails the upstream render gate.
#
# The two services differ: physics also accepts `physics-ovrtx-rendering-api`.
# Take the union, because this set is only used to decide whether upstream would
# already accept an endpoint. Over-listing means the skill skips a compensation
# it did not need; under-listing means it writes a provider credential that was
# never required. Texture has no render gate at all, so it is not mirrored here.
UPSTREAM_LOCAL_RENDER_HOSTS = {
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "::1",
    "ovrtx-rendering-api",
    "physics-ovrtx-rendering-api",
}
CONTAINER_RENDER_ENV_KEYS = ("RENDER_ENDPOINT", "OVRTX_RENDER_ENDPOINT", "CONTENT_AGENTS_RENDER_BASE_URL")
# usd-convert-cad 0.2 conversion runs from the self-contained pip wheel (see
# `_check_usd_convert_cad`). This checkout is cloned only as the capability and
# agent-guidance source: `usd-convert-cad`'s SKILL.md holds the authoritative
# Supported Formats table and conversion workflow guidance. It is not used as a
# conversion runtime.
UPSTREAMS = git_upstreams()


@dataclass
class Step:
    name: str
    status: str
    message: str
    command: list[str] = field(default_factory=list)
    returncode: int | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "status": self.status,
            "message": self.message,
        }
        if self.command:
            payload["command"] = self.command
        if self.returncode is not None:
            payload["returncode"] = self.returncode
        if self.stdout_tail:
            payload["stdout_tail"] = self.stdout_tail
        if self.stderr_tail:
            payload["stderr_tail"] = self.stderr_tail
        return payload


def _checkout_name_from_repo_url(repo_url: str) -> str:
    name = urlparse(repo_url).path.rstrip("/").rsplit("/", 1)[-1]
    return name[:-4] if name.endswith(".git") else name


def _default_home() -> Path:
    return Path(os.environ.get("OMNIVERSE_CAD_TO_SIMREADY_HOME", "~/.omniverse-cad-to-simready")).expanduser()


def _default_state_root(home: Path) -> Path:
    return Path(os.environ.get("OMNIVERSE_CAD_TO_SIMREADY_STATE", home / "state")).expanduser()


def _default_upstream_root(home: Path) -> Path:
    return Path(os.environ.get("OMNIVERSE_CAD_TO_SIMREADY_UPSTREAM_ROOT", home / "upstreams")).expanduser()


def _default_venv_root(home: Path) -> Path:
    return Path(os.environ.get("OMNIVERSE_CAD_TO_SIMREADY_VENV_ROOT", home / "venvs")).expanduser()


def _is_secret_name(name: str) -> bool:
    upper = name.upper()
    return any(part in upper for part in SECRET_NAME_PARTS)


def _redaction_values(env: dict[str, str]) -> list[str]:
    return [value for name, value in env.items() if value and _is_secret_name(name) and len(value) >= 4]


def _redact(text: str, secrets: list[str]) -> str:
    redacted = text
    for secret in secrets:
        redacted = redacted.replace(secret, "<redacted>")
    return redacted


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int = 900,
) -> Step:
    command_env = env or os.environ.copy()
    secrets = _redaction_values(command_env)
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            env=command_env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return Step(
            name=Path(command[0]).name,
            status="blocked",
            message=f"command could not complete: {exc}",
            command=command,
        )
    status = "ready" if completed.returncode == 0 else "blocked"
    return Step(
        name=Path(command[0]).name,
        status=status,
        message="command completed" if completed.returncode == 0 else f"command failed with exit {completed.returncode}",
        command=command,
        returncode=completed.returncode,
        stdout_tail=tail_text(_redact(completed.stdout or "", secrets)),
        stderr_tail=tail_text(_redact(completed.stderr or "", secrets)),
    )


def _package_versions_step(
    python: Path,
    package_names: tuple[str, ...],
    *,
    name: str,
    import_check: str = "",
) -> tuple[Step, dict[str, str | None]]:
    code = version_probe_code(package_names)
    if import_check:
        code = f"{import_check}; {code}"
    step = _run([str(python), "-c", code], timeout=120)
    step.name = name
    if step.status == "blocked":
        step.message = f"could not verify pinned Python runtime: {step.stderr_tail or step.message}"
        return step, {}
    try:
        installed = json.loads(step.stdout_tail.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        step.status = "blocked"
        step.message = f"Python version probe did not return JSON: {exc}"
        return step, {}
    mismatches = version_mismatches(expected_versions(package_names), installed)
    if mismatches:
        step.status = "blocked"
        step.message = "; ".join(mismatches)
    else:
        step.message = "pinned Python package versions match the upstream version manifest"
    return step, installed


def _container_reachable_render_endpoint(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in LOCAL_RENDER_HOSTS:
        return value
    host = DOCKER_HOST_ALIAS
    if parsed.port:
        host = f"{host}:{parsed.port}"
    return urlunparse(parsed._replace(netloc=host))


def _endpoint_host(value: str | None) -> str:
    if not value:
        return ""
    return (urlparse(value).hostname or "").strip("[]").lower()


def _is_local_render_endpoint(value: str | None) -> bool:
    """Report whether a render endpoint is served by this machine.

    ``host.docker.internal`` is the Docker host alias the upstream Content
    Agents collection uses to reach a host-local OVRTX renderer, so it is local
    even though it is not a loopback name. Keep this predicate in agreement with
    ``references/ovrtx-render-service/scripts/run.py`` and
    ``references/content-agents/scripts/content_agent_client.py``.
    """
    host = _endpoint_host(value)
    if not host:
        return False
    if host in LOCAL_RENDER_ENDPOINT_HOSTS or host.endswith(".localhost"):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_loopback or address.is_unspecified


def _upstream_render_gate_rejects_local_endpoint(value: str | None) -> bool:
    """Report whether upstream demands a render key for a host-local renderer.

    Upstream readiness matches an exact hostname allowlist, so a renderer that
    is on this machine but reached through another name -- notably the
    ``host.docker.internal`` Docker host alias -- fails the upstream render gate
    and drives ``api_keys_configured: false`` even while rendering works over
    HTTP. That is the known upstream gap behind NVBug 6534667, not a missing
    workflow credential.
    """
    if not _is_local_render_endpoint(value):
        return False
    return _endpoint_host(value) not in UPSTREAM_LOCAL_RENDER_HOSTS


def _content_agents_deploy_env(
    model_options: dict[str, dict[str, str]] | None = None,
    *,
    render_endpoint: str | None = None,
) -> dict[str, str]:
    """Translate host-local renderer URLs for agent containers during deploy."""
    env = os.environ.copy()
    env.setdefault("BUILDKIT_PROGRESS", "plain")
    env.setdefault("COMPOSE_BAKE", "false")
    for key in CONTAINER_RENDER_ENV_KEYS:
        if env.get(key):
            env[key] = _container_reachable_render_endpoint(env[key])
    if render_endpoint:
        env["RENDER_ENDPOINT"] = _container_reachable_render_endpoint(render_endpoint)
    if not env.get("RENDER_ENDPOINT"):
        for key in ("OVRTX_RENDER_ENDPOINT", "CONTENT_AGENTS_RENDER_BASE_URL"):
            if env.get(key):
                env["RENDER_ENDPOINT"] = _container_reachable_render_endpoint(env[key])
                break
    if model_options:
        vlm = model_options.get("vlm", {})
        llm = model_options.get("llm", {})
        if vlm.get("backend"):
            env["CONTENT_AGENTS_VLM_BACKEND"] = vlm["backend"]
            env["MA_VLM_BACKEND"] = vlm["backend"]
            env["PA_VLM_BACKEND"] = vlm["backend"]
        vlm_api_key_env = _content_agents_api_key_env(vlm)
        if vlm_api_key_env:
            env["MA_VLM_API_KEY_ENV"] = vlm_api_key_env
            env["PA_VLM_API_KEY_ENV"] = vlm_api_key_env
        if vlm.get("model"):
            env["CONTENT_AGENTS_VLM_MODEL"] = vlm["model"]
            env["MA_VLM_MODEL"] = vlm["model"]
            env["PA_VLM_MODEL"] = vlm["model"]
        if vlm.get("endpoint"):
            env["CONTENT_AGENTS_VLM_ENDPOINT"] = vlm["endpoint"]
            if vlm.get("backend", "").lower() == "openai":
                env["MA_VLM_BASE_URL"] = vlm["endpoint"]
                env["PA_VLM_BASE_URL"] = vlm["endpoint"]
                env.pop("MA_VLM_NIM_BASE_URL", None)
                env.pop("PA_VLM_NIM_BASE_URL", None)
            elif vlm.get("backend", "").lower() == "nim":
                env["MA_VLM_NIM_BASE_URL"] = vlm["endpoint"]
                env["PA_VLM_NIM_BASE_URL"] = vlm["endpoint"]
        if _content_agents_needs_default_temperature(vlm):
            env["MA_VLM_TEMPERATURE"] = "1.0"
            env["PA_VLM_TEMPERATURE"] = "1.0"
        if llm.get("backend"):
            env["CONTENT_AGENTS_LLM_BACKEND"] = llm["backend"]
            env["MA_LLM_BACKEND"] = llm["backend"]
            env["TA_LLM_BACKEND"] = llm["backend"]
        llm_api_key_env = _content_agents_api_key_env(llm)
        if llm_api_key_env:
            # MA_ and TA_ deliberately differ here. The Material Agent service
            # re-normalizes its pointer through format_env_reference() before
            # writing its durable config (upstream
            # material_agent_service/service/routers/pipeline_router.py), so a
            # bare name is safe on that side and is left exactly as-is.
            #
            # The Texture Agent service has no such normalization: it copies
            # TA_LLM_API_KEY_ENV verbatim into auto_prompt.llm.api_key_env of
            # its generated pipeline config (upstream
            # texture_agent_service/service/routers/pipeline_router.py, and
            # texture_agent/config/unified_config.py for the local pipeline).
            # ensure_no_inline_secrets() then rejects the bare name as an
            # inline secret, surfacing as
            # HTTPException(400, "Pipeline configuration is invalid") -- the
            # second half of GH #260 / NVBug 6539618, reported at
            # auto_prompt.llm.api_key_env. This is the same Material-normalizes
            # / Texture-does-not asymmetry that affects image_gen below.
            env["MA_LLM_API_KEY_ENV"] = llm_api_key_env
            env["TA_LLM_API_KEY_ENV"] = _format_env_reference(llm_api_key_env) or llm_api_key_env
        if llm.get("model"):
            env["CONTENT_AGENTS_LLM_MODEL"] = llm["model"]
            env["MA_LLM_MODEL"] = llm["model"]
            env["TA_LLM_MODEL"] = llm["model"]
        if llm.get("endpoint"):
            env["CONTENT_AGENTS_LLM_ENDPOINT"] = llm["endpoint"]
            if llm.get("backend", "").lower() == "openai":
                env["MA_LLM_BASE_URL"] = llm["endpoint"]
                env["TA_LLM_BASE_URL"] = llm["endpoint"]
                env.pop("MA_LLM_NIM_BASE_URL", None)
            elif llm.get("backend", "").lower() == "nim":
                env["MA_LLM_NIM_BASE_URL"] = llm["endpoint"]
                env["TA_LLM_BASE_URL"] = llm["endpoint"]
        if _content_agents_needs_default_temperature(llm):
            env["MA_LLM_TEMPERATURE"] = "1.0"
            env["TA_LLM_TEMPERATURE"] = "1.0"
        image_gen = model_options.get("image_gen", {})
        if image_gen.get("backend"):
            env["CONTENT_AGENTS_IMAGE_GEN_BACKEND"] = image_gen["backend"]
            env["MA_IMAGE_GEN_BACKEND"] = image_gen["backend"]
            env["TA_IMAGE_GEN_BACKEND"] = image_gen["backend"]
        # image_gen must use the durable ``${NAME}`` reference form. Unlike the
        # VLM/LLM pointers -- which the Material Agent service re-normalizes via
        # format_env_reference() before it writes its durable config -- the
        # Texture Agent service copies TA_IMAGE_GEN_API_KEY_ENV verbatim into
        # the generated pipeline config (upstream
        # texture_agent_service/service/routers/pipeline_router.py). That config
        # is then screened by ensure_no_inline_secrets(), which rejects a bare
        # name in an ``*_api_key_env`` field and surfaces as
        # HTTPException(400, "Pipeline configuration is invalid") -- the failure
        # reported in GH #260 / NVBug 6539618.
        #
        # This value is delivered to the container through the *process*
        # environment of the deploy/compose subprocess, where Compose treats it
        # as an interpolation source and passes it through literally. It must
        # not be written into the ``--env-file`` instead: Compose interpolates
        # unquoted ``${...}`` inside env-file values, which would expand the
        # reference to the raw secret. See _apply_collection_dependency.
        image_gen_api_key_env = _format_env_reference(
            _content_agents_api_key_env(image_gen, name="image_gen")
        )
        if image_gen_api_key_env:
            env["MA_IMAGE_GEN_API_KEY_ENV"] = image_gen_api_key_env
            env["TA_IMAGE_GEN_API_KEY_ENV"] = image_gen_api_key_env
        if image_gen.get("model"):
            env["CONTENT_AGENTS_IMAGE_GEN_MODEL"] = image_gen["model"]
            env["MA_IMAGE_GEN_MODEL"] = image_gen["model"]
            env["TA_IMAGE_GEN_MODEL"] = image_gen["model"]
        if image_gen.get("endpoint"):
            env["CONTENT_AGENTS_IMAGE_GEN_ENDPOINT"] = image_gen["endpoint"]
            env["MA_IMAGE_GEN_BASE_URL"] = image_gen["endpoint"]
            env["TA_IMAGE_GEN_BASE_URL"] = image_gen["endpoint"]
    return env


def _content_agents_provider_credentials_available(
    model_options: dict[str, dict[str, str]] | None = None,
) -> bool:
    names = set(CONTENT_AGENTS_PROVIDER_SECRET_ENV_NAMES)
    if model_options:
        for group in model_options.values():
            api_key_env = _parse_env_reference(group.get("api_key_env", ""))
            if api_key_env:
                names.add(api_key_env)
    return any(os.environ.get(name) for name in names)


def _env_or_arg(args: argparse.Namespace, attr: str, *env_names: str) -> str:
    value = getattr(args, attr, None)
    if value:
        return str(value).strip()
    for name in env_names:
        value = os.environ.get(name)
        if value:
            return value.strip()
    return ""


def _infer_content_agents_backend(model_name: str, endpoint_value: str) -> str:
    normalized = model_name.lower()
    if endpoint_value:
        return "openai"
    if normalized.startswith("claude-") or normalized.startswith("claude."):
        return "anthropic"
    if normalized.startswith("gemini-") or normalized.startswith("models/gemini"):
        return "gemini"
    if normalized.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    return ""


def _default_content_agents_api_key_env(backend: str, *, preferred_names: tuple[str, ...] = ()) -> str:
    for name in preferred_names:
        if os.environ.get(name):
            return name
    names = CONTENT_AGENTS_BACKEND_API_KEY_ENV_NAMES.get(backend.lower(), ())
    for name in names:
        if os.environ.get(name):
            return name
    if preferred_names:
        return preferred_names[0]
    return names[0] if names else ""


# Mirrors upstream world_understanding/utils/credentials.py (_ENV_NAME_RE /
# _DURABLE_ENV_REFERENCE_RE). Upstream's durable boundaries -- the generated
# pipeline config written by each agent service -- accept only the unambiguous
# ``${NAME}`` reference form for ``*_api_key_env`` fields. Runtime consumers
# call parse_env_reference(allow_legacy_bare=True) and so accept both forms,
# which makes ``${NAME}`` strictly the more compatible thing to emit.
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DURABLE_ENV_REFERENCE_RE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


def _parse_env_reference(value: str) -> str:
    """Return the bare variable name carried by ``NAME`` or ``${NAME}``.

    Mirrors upstream ``parse_env_reference(..., allow_legacy_bare=True)``.
    Returns "" for anything that is not a usable reference, so a
    credential-bearing value is never echoed back to the caller.
    """
    stripped = (value or "").strip()
    match = _DURABLE_ENV_REFERENCE_RE.match(stripped)
    if match:
        return match.group(1)
    if _ENV_NAME_RE.match(stripped):
        return stripped
    return ""


def _format_env_reference(value: str) -> str:
    """Return the canonical durable ``${NAME}`` form, or "" when unusable.

    Mirrors upstream ``format_env_reference``. This only ever renders a
    variable *name*; it never reads the variable's value.
    """
    name = _parse_env_reference(value)
    return f"${{{name}}}" if name else ""


def _read_credential_env(name: str) -> str:
    """Read a caller-named credential env var, bounded to a well-formed name.

    api_key_env / api_key_env_name style values come from CLI flags or an env
    file rather than a fixed allowlist, so this guards against forwarding an
    unintended environment variable (empty string, stray whitespace, or a
    malformed name) into a generated collection config or .env file.
    """
    if not CREDENTIAL_ENV_NAME_PATTERN.match(name):
        return ""
    return os.environ.get(name, "").strip()

def _content_agents_api_key_env(options: dict[str, str], *, name: str = "") -> str:
    explicit = options.get("api_key_env", "").strip()
    if explicit:
        return explicit
    backend = options.get("backend", "").lower()
    if not backend or backend == "nim":
        return ""
    preferred_names = CONTENT_AGENTS_DEPENDENCY_API_KEY_ENV_NAMES.get(name, ())
    return _default_content_agents_api_key_env(backend, preferred_names=preferred_names)


def _content_agents_needs_default_temperature(options: dict[str, str]) -> bool:
    return (
        options.get("backend", "").lower() == "openai"
        and options.get("model", "").lower().startswith("gpt-")
    )


def _content_agents_model_options(args: argparse.Namespace) -> dict[str, dict[str, str]]:
    vlm_model = _env_or_arg(args, "content_agents_vlm_model", "CONTENT_AGENTS_VLM_MODEL", "MA_VLM_MODEL", "PA_VLM_MODEL")
    vlm_endpoint = _env_or_arg(args, "content_agents_vlm_endpoint", "CONTENT_AGENTS_VLM_ENDPOINT", "MA_VLM_NIM_BASE_URL", "PA_VLM_NIM_BASE_URL")
    vlm_backend = _env_or_arg(args, "content_agents_vlm_backend", "CONTENT_AGENTS_VLM_BACKEND", "MA_VLM_BACKEND", "PA_VLM_BACKEND")
    if not vlm_backend and (vlm_model or vlm_endpoint):
        vlm_backend = _infer_content_agents_backend(vlm_model, vlm_endpoint)
    vlm_api_key_env = _env_or_arg(args, "content_agents_vlm_api_key_env", "CONTENT_AGENTS_VLM_API_KEY_ENV", "MA_VLM_API_KEY_ENV", "PA_VLM_API_KEY_ENV")

    llm_model = _env_or_arg(args, "content_agents_llm_model", "CONTENT_AGENTS_LLM_MODEL", "MA_LLM_MODEL", "TA_LLM_MODEL")
    llm_endpoint = _env_or_arg(args, "content_agents_llm_endpoint", "CONTENT_AGENTS_LLM_ENDPOINT", "MA_LLM_NIM_BASE_URL", "TA_LLM_BASE_URL")
    llm_backend = _env_or_arg(args, "content_agents_llm_backend", "CONTENT_AGENTS_LLM_BACKEND", "MA_LLM_BACKEND", "TA_LLM_BACKEND")
    if not llm_backend and (llm_model or llm_endpoint):
        llm_backend = _infer_content_agents_backend(llm_model, llm_endpoint)
    llm_api_key_env = _env_or_arg(args, "content_agents_llm_api_key_env", "CONTENT_AGENTS_LLM_API_KEY_ENV", "MA_LLM_API_KEY_ENV", "TA_LLM_API_KEY_ENV")

    image_gen_model = _env_or_arg(args, "content_agents_image_gen_model", "CONTENT_AGENTS_IMAGE_GEN_MODEL", "MA_IMAGE_GEN_MODEL", "TA_IMAGE_GEN_MODEL")
    image_gen_endpoint = _env_or_arg(args, "content_agents_image_gen_endpoint", "CONTENT_AGENTS_IMAGE_GEN_ENDPOINT", "MA_IMAGE_GEN_BASE_URL", "TA_IMAGE_GEN_BASE_URL")
    image_gen_backend = _env_or_arg(args, "content_agents_image_gen_backend", "CONTENT_AGENTS_IMAGE_GEN_BACKEND", "MA_IMAGE_GEN_BACKEND", "TA_IMAGE_GEN_BACKEND")
    if not image_gen_backend and (image_gen_model or image_gen_endpoint):
        image_gen_backend = _infer_content_agents_backend(image_gen_model, image_gen_endpoint)
    image_gen_api_key_env = _env_or_arg(args, "content_agents_image_gen_api_key_env", "CONTENT_AGENTS_IMAGE_GEN_API_KEY_ENV", "MA_IMAGE_GEN_API_KEY_ENV", "TA_IMAGE_GEN_API_KEY_ENV")

    return {
        "vlm": {
            "backend": vlm_backend,
            "model": vlm_model,
            "endpoint": vlm_endpoint,
            "api_key_env": vlm_api_key_env,
        },
        "llm": {
            "backend": llm_backend,
            "model": llm_model,
            "endpoint": llm_endpoint,
            "api_key_env": llm_api_key_env,
        },
        "image_gen": {
            "backend": image_gen_backend,
            "model": image_gen_model,
            "endpoint": image_gen_endpoint,
            "api_key_env": image_gen_api_key_env,
        },
    }


def _has_content_agents_model_options(model_options: dict[str, dict[str, str]] | None) -> bool:
    if not model_options:
        return False
    return any(value for group in model_options.values() for value in group.values())


def _apply_collection_dependency(
    config: dict[str, Any],
    name: str,
    options: dict[str, str],
    *,
    default_backend: str,
) -> None:
    if not any(options.values()):
        return
    dependencies = config.setdefault("dependencies", {})
    if not isinstance(dependencies, dict):
        dependencies = {}
        config["dependencies"] = dependencies
    dependency_config = dependencies.setdefault(name, {})
    if not isinstance(dependency_config, dict):
        dependency_config = {}
        dependencies[name] = dependency_config
    dependency_config["enabled"] = True
    dependency_config.setdefault("provider", "external")
    dependency_config.setdefault("endpoint", "")
    selected_backend = options.get("backend") or default_backend
    dependency_config["backend"] = selected_backend
    api_key_env = _content_agents_api_key_env({**options, "backend": selected_backend}, name=name)
    if api_key_env:
        # The collection config carries the *bare* variable name on purpose.
        # Upstream deploy.py copies this value verbatim and unquoted into
        # .collection.generated.env, and Docker Compose interpolates unquoted
        # ``${...}`` inside an --env-file value -- so a durable ``${NAME}``
        # reference here would expand to the raw secret and hand the container
        # an inline credential. The durable reference is instead exported
        # through the deploy process environment (see
        # _content_agents_deploy_env), which Compose passes through literally
        # and which takes precedence over the --env-file value.
        dependency_config.setdefault("api_key_env", _parse_env_reference(api_key_env) or api_key_env)
    if options.get("model"):
        dependency_config["model"] = options["model"]
    if options.get("endpoint"):
        if selected_backend == "openai":
            dependency_config["endpoint"] = ""
            dependency_config["base_url"] = options["endpoint"]
            # image_gen deliberately does not forward a resolved literal key.
            # Upstream's own collection.yaml ships ``api_key: not-used`` for
            # image_gen, so this setdefault was already inert against real
            # upstream; and the Texture Agent service copies a literal
            # image_gen api_key straight into its durable pipeline config,
            # where ensure_no_inline_secrets() rejects it as an inline secret.
            # The ``${NAME}`` pointer wired in _content_agents_deploy_env is
            # what carries the image_gen credential instead.
            if api_key_env and name != "image_gen":
                # api_key_env may arrive as a bare name or as ``${NAME}`` when
                # an operator supplies the durable form; resolve to the bare
                # variable name, then read it through the credential-name guard
                # so a malformed name never reaches os.environ.
                api_key_value = _read_credential_env(_parse_env_reference(api_key_env))
                if api_key_value:
                    dependency_config.setdefault("api_key", api_key_value)
        else:
            dependency_config["endpoint"] = options["endpoint"]
    if _content_agents_needs_default_temperature(options):
        dependency_config["temperature"] = 1.0


def _content_agents_openai_endpoint_options(
    model_options: dict[str, dict[str, str]] | None,
) -> list[str]:
    if not model_options:
        return []
    return [
        name
        for name, options in model_options.items()
        if options.get("backend", "").lower() == "openai" and bool(options.get("endpoint"))
    ]


def _collection_deploy_supports_openai_base_url(deploy_py: Path) -> bool:
    try:
        text = deploy_py.read_text(encoding="utf-8")
    except OSError:
        return False
    if "OPENAI_BASE_URL" in text:
        return True
    return (
        "set_model_endpoint_env" in text
        and 'vlm_backend == "openai"' in text
        and 'llm_backend == "openai"' in text
        and "_BASE_URL" in text
        and "backend=openai uses base_url" in text
    )


def _prepare_content_agents_collection_config(
    world_root: Path,
    *,
    include_texture: bool,
    model_options: dict[str, dict[str, str]] | None,
    enabled_agents: set[str] | None = None,
    render_endpoint: str | None = None,
    config_suffix: str = "",
) -> tuple[Path | None, Step | None]:
    """Write a non-secret upstream collection config when local overrides exist."""
    deploy_py = world_root / "deploy" / "collection" / "deploy.py"
    if not deploy_py.is_file():
        return None, None
    openai_endpoint_roles = _content_agents_openai_endpoint_options(model_options)
    if openai_endpoint_roles and not _collection_deploy_supports_openai_base_url(deploy_py):
        return None, Step(
            name="content_agents_collection_config",
            status="blocked",
            message=(
                "upstream Content Agents collection deploy does not support "
                "OpenAI backend base_url deployment for "
                + ", ".join(sorted(openai_endpoint_roles))
            ),
        )

    needs_generated_config = (
        include_texture
        or enabled_agents is not None
        or bool(render_endpoint)
        or _has_content_agents_model_options(model_options)
    )
    if not needs_generated_config:
        return None, None

    base_config = world_root / "deploy" / "collection" / "collection.yaml"
    if not base_config.is_file():
        return None, Step(
            name="content_agents_collection_config",
            status="blocked",
            message="upstream Content Agents collection.yaml was not found",
        )

    try:
        config = yaml.safe_load(base_config.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        return None, Step(
            name="content_agents_collection_config",
            status="blocked",
            message=f"upstream Content Agents collection.yaml could not be read: {exc}",
        )
    if not isinstance(config, dict):
        return None, Step(
            name="content_agents_collection_config",
            status="blocked",
            message="upstream Content Agents collection.yaml must contain a YAML mapping",
        )

    agents = config.setdefault("agents", {})
    if not isinstance(agents, dict):
        agents = {}
        config["agents"] = agents
    if enabled_agents is not None:
        for name in ("material", "physics", "texture"):
            entry = agents.setdefault(name, {})
            if not isinstance(entry, dict):
                entry = {}
                agents[name] = entry
            entry["enabled"] = name in enabled_agents
    elif include_texture:
        texture = agents.setdefault("texture", {})
        if not isinstance(texture, dict):
            texture = {}
            agents["texture"] = texture
        texture["enabled"] = True

    if render_endpoint:
        dependencies = config.setdefault("dependencies", {})
        if not isinstance(dependencies, dict):
            dependencies = {}
            config["dependencies"] = dependencies
        render = dependencies.setdefault("render", {})
        if not isinstance(render, dict):
            render = {}
            dependencies["render"] = render
        render["enabled"] = True
        render["provider"] = "external"
        render["endpoint"] = _container_reachable_render_endpoint(render_endpoint)

    options = model_options or {}
    _apply_collection_dependency(config, "vlm", options.get("vlm", {}), default_backend="nim")
    _apply_collection_dependency(config, "llm", options.get("llm", {}), default_backend="nim")
    _apply_collection_dependency(config, "image_gen", options.get("image_gen", {}), default_backend="openai")

    suffix = f".{config_suffix}" if config_suffix else ""
    generated_config = world_root / "deploy" / "collection" / f".collection.cad-to-simready{suffix}.yaml"
    try:
        generated_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    except OSError as exc:
        return None, Step(
            name="content_agents_collection_config",
            status="blocked",
            message=f"upstream Content Agents collection config could not be written: {exc}",
        )
    try:
        generated_config.chmod(0o600)
    except OSError:
        pass

    configured: list[str] = []
    if enabled_agents is not None:
        configured.append("agents=" + ",".join(sorted(enabled_agents)))
    if include_texture:
        configured.append("texture")
    if render_endpoint:
        configured.append("render")
    if _has_content_agents_model_options({"vlm": options.get("vlm", {})}):
        configured.append("vlm")
    if _has_content_agents_model_options({"llm": options.get("llm", {})}):
        configured.append("llm")
    if _has_content_agents_model_options({"image_gen": options.get("image_gen", {})}):
        configured.append("image_gen")
    return generated_config, Step(
        name="content_agents_collection_config",
        status="ready",
        message=f"wrote upstream collection override for {', '.join(configured)}",
        command=[],
    )


def _path_entries(*entries: Path | None) -> list[str]:
    seen: set[str] = set()
    values: list[str] = []
    for entry in entries:
        if entry is None:
            continue
        value = str(entry.expanduser())
        if value in seen or not Path(value).is_dir():
            continue
        seen.add(value)
        values.append(value)
    return values


def _path_env_with_entries(*entries: Path | None) -> str | None:
    extra = _path_entries(*entries)
    if not extra:
        return None
    return os.pathsep.join([*extra, os.environ.get("PATH", "")])


def _user_tool_dirs() -> list[str]:
    home = Path.home()
    uv_python_dirs = sorted((home / ".local" / "share" / "uv" / "python").glob("*/bin"))
    return _path_entries(home / ".local" / "bin", home / "bin", *uv_python_dirs)


def _which(name: str, *, extra_dirs: list[str] | None = None) -> str | None:
    search_dirs = [*(extra_dirs or []), *_user_tool_dirs()]
    search_path = os.environ.get("PATH", "")
    if search_dirs:
        search_path = os.pathsep.join([*search_dirs, search_path])
    return shutil.which(name, path=search_path)


def _env_with_extra_path(*entries: Path | None) -> dict[str, str]:
    env = os.environ.copy()
    path_value = _path_env_with_entries(*entries, *[Path(value) for value in _user_tool_dirs()])
    if path_value:
        env["PATH"] = path_value
    return env


def _csv_values(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _selected_targets(raw: str, *, skip_content_agents: bool) -> tuple[str, ...]:
    values = _csv_values(raw)
    if values:
        selected = tuple(target for target in values if target in DEFAULT_TARGETS)
    else:
        selected = tuple(DEFAULT_TARGETS)
    if skip_content_agents:
        selected = tuple(target for target in selected if target != "content-agents")
    return selected


def _normalized_source_format(source_asset: Path | None, source_format: str) -> str:
    if source_format:
        return source_format.strip().lower().lstrip(".")
    if source_asset is None:
        return ""
    suffix = source_asset.suffix.lower()
    if suffix in USD_SUFFIXES:
        return "usd"
    if suffix == ".urdf":
        return "urdf"
    if suffix in {".mjcf", ".xml"}:
        return "mujoco"
    if suffix in GSPLAT_SOURCE_SUFFIXES:
        return "gsplat"
    if suffix in CAD_SOURCE_SUFFIXES:
        return "cad"
    return suffix.lstrip(".")


def _inferred_conversion_tools(source_format: str) -> set[str] | None:
    if not source_format:
        return None
    if source_format in {"usd", "openusd"}:
        return set()
    if source_format in REPO_PYTHON_SOURCE_FORMATS:
        return {"repo-python"}
    if source_format in {"cad", "mesh", "scene", "obj", "stl", "dae", "gltf", "glb", "fbx", "step", "stp"}:
        return {"usd-convert-cad"}
    if source_format in {"gsplat", "splat", "ksplat"}:
        return {"usd-convert-gsplat"}
    return None


def _selected_conversion_tools(raw: str, source_asset: Path | None, source_format: str) -> tuple[set[str], dict[str, Any]]:
    explicit_tools = {tool for tool in _csv_values(raw) if tool in DEFAULT_CONVERSION_TOOLS}
    normalized_source_format = _normalized_source_format(source_asset, source_format)
    inferred_tools = _inferred_conversion_tools(normalized_source_format)
    if explicit_tools:
        tools = explicit_tools
        reason = "explicit"
    elif inferred_tools is not None:
        tools = inferred_tools
        reason = "source-format"
    else:
        tools = set(DEFAULT_CONVERSION_TOOLS)
        reason = "default"
    return tools, {
        "source_asset": str(source_asset) if source_asset else "",
        "source_format": normalized_source_format,
        "conversion_tools_reason": reason,
    }


def _project_venv_dir(project_root: Path | None) -> Path | None:
    if project_root is None:
        return None
    venv = project_root / ".venv"
    return venv if venv.exists() else None


def _project_venv_bin(project_root: Path | None) -> Path | None:
    venv = _project_venv_dir(project_root)
    if venv is None:
        return None
    return _venv_bin_dir(venv)


def _project_venv_python(project_root: Path | None) -> Path | None:
    venv_bin = _project_venv_bin(project_root)
    if venv_bin is None:
        return None
    python = venv_bin / ("python.exe" if os.name == "nt" else "python")
    return python if python.exists() else None


def _find_executable(name: str, *, project_root: Path | None = None) -> str | None:
    extra_dirs = _path_entries(_project_venv_bin(project_root))
    return _which(name, extra_dirs=extra_dirs)


def _find_project_root(explicit: Path | None) -> Path | None:
    candidates: list[Path] = []
    if explicit:
        candidates.append(explicit.expanduser().resolve())
    candidates.append(Path.cwd().resolve())
    candidates.extend(Path(__file__).resolve().parents)
    for candidate in candidates:
        current = candidate
        while True:
            if (current / "pyproject.toml").is_file():
                return current
            if current.parent == current:
                break
            current = current.parent
    return None


def _upstream_path(name: str, upstream_root: Path) -> Path:
    spec = UPSTREAMS[name]
    override = os.environ.get(str(spec["env"]))
    if override:
        return Path(override).expanduser().resolve()
    checkout = str(spec["checkout"]) or _checkout_name_from_repo_url(str(spec["url"]))
    return (upstream_root / checkout).expanduser().resolve()


def _git_commit(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    completed = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30, check=False)
    return completed.stdout.strip() if completed.returncode == 0 else None


def _git_dirty(path: Path) -> bool:
    completed = subprocess.run(["git", "-C", str(path), "status", "--porcelain"], capture_output=True, text=True, timeout=30, check=False)
    return bool(completed.stdout.strip()) if completed.returncode == 0 else False


def _ensure_upstream(name: str, upstream_root: Path, *, check_only: bool, no_update: bool) -> tuple[dict[str, Any], list[Step]]:
    spec = UPSTREAMS[name]
    path = _upstream_path(name, upstream_root)
    expected_ref = str(spec["ref"])
    expected_commit = str(spec["commit"])
    explicitly_overridden = bool(os.environ.get(str(spec["env"])))
    steps: list[Step] = []
    if not path.exists():
        if check_only:
            return (
                {
                    "status": "blocked",
                    "path": str(path),
                    "url": spec["url"],
                    "ref": expected_ref,
                    "expected_commit": expected_commit,
                    "version_manifest": str(UPSTREAM_VERSION_MANIFEST_PATH),
                    "message": f"checkout is missing: {path}; requires {expected_ref} ({expected_commit})",
                },
                steps,
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        clone_env = os.environ.copy()
        clone_env["GIT_LFS_SKIP_SMUDGE"] = "1"
        command = [
            "git",
            "clone",
            "--filter=blob:none",
            "--depth",
            "1",
            "--single-branch",
            "--branch",
            expected_ref,
            str(spec["url"]),
            str(path),
        ]
        steps.append(_run(command, env=clone_env, timeout=1800))
    elif (path / ".git").exists() and not check_only and not no_update:
        if explicitly_overridden:
            steps.append(
                Step(
                    name=f"{name}_update",
                    status="skipped",
                    message="explicit upstream override is never mutated; verifying its pinned commit",
                )
            )
        elif _git_dirty(path):
            steps.append(Step(name=f"{name}_update", status="skipped", message="checkout has local changes; skipped automatic update"))
        else:
            steps.append(_run(["git", "-C", str(path), "fetch", "--depth", "1", "origin", expected_ref], timeout=600))
            if steps[-1].status != "blocked":
                steps.append(_run(["git", "-C", str(path), "checkout", "--detach", expected_commit], timeout=120))

    present = path.exists()
    actual_commit = _git_commit(path)
    matches_pin = actual_commit == expected_commit
    status = "ready" if present and matches_pin else "blocked"
    if any(step.status == "blocked" for step in steps) or not (path / ".git").exists():
        status = "blocked"
    if not present:
        message = f"checkout is missing: {path}; requires {expected_ref} ({expected_commit})"
    elif actual_commit is None:
        message = f"checkout at {path} is not a Git worktree; requires {expected_ref} ({expected_commit})"
    elif not matches_pin:
        message = f"requires {expected_ref} ({expected_commit}), found {actual_commit} at {path}"
    else:
        message = f"checkout matches pinned {expected_ref} ({expected_commit})"
    return (
        {
            "status": status,
            "path": str(path),
            "url": spec["url"],
            "ref": expected_ref,
            "expected_commit": expected_commit,
            "commit": actual_commit,
            "version_manifest": str(UPSTREAM_VERSION_MANIFEST_PATH),
            "message": message,
        },
        steps,
    )


def _runtime_entry(status: str, message: str, **extra: Any) -> dict[str, Any]:
    payload = {"status": status, "message": message}
    payload.update({key: value for key, value in extra.items() if value is not None and value != ""})
    return payload


def _manifest_blockers(
    upstreams: dict[str, dict[str, Any]],
    runtimes: dict[str, dict[str, Any]],
    services: dict[str, dict[str, Any]],
) -> list[str]:
    blockers: list[str] = []
    for category, entries in (
        ("upstream", upstreams),
        ("runtime", runtimes),
        ("service", services),
    ):
        for name, entry in sorted(entries.items()):
            if entry.get("status") == "blocked":
                blockers.append(f"{category} {name}: {entry.get('message')}")
    return blockers


def _nearest_existing_parent(path: Path) -> Path:
    current = path
    while not current.exists() and current.parent != current:
        current = current.parent
    return current


def _check_request_inputs(source_asset: Path | None, output_root: Path | None, *, check_only: bool) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    blockers: list[str] = []

    if source_asset is None:
        checks.append({"name": "source_asset", "status": "skipped", "message": "no source asset was provided"})
    elif not source_asset.exists():
        message = f"source asset does not exist: {source_asset}"
        checks.append({"name": "source_asset", "status": "blocked", "message": message})
        blockers.append(message)
    elif not os.access(source_asset, os.R_OK):
        message = f"source asset is not readable: {source_asset}"
        checks.append({"name": "source_asset", "status": "blocked", "message": message})
        blockers.append(message)
    else:
        checks.append({"name": "source_asset", "status": "ready", "message": "source asset exists and is readable"})

    if output_root is None:
        checks.append({"name": "output_root", "status": "skipped", "message": "no output root was provided"})
    else:
        try:
            if output_root.exists():
                if not output_root.is_dir():
                    message = f"output root exists but is not a directory: {output_root}"
                    checks.append({"name": "output_root", "status": "blocked", "message": message})
                    blockers.append(message)
                elif not os.access(output_root, os.W_OK):
                    message = f"output root is not writable: {output_root}"
                    checks.append({"name": "output_root", "status": "blocked", "message": message})
                    blockers.append(message)
                else:
                    checks.append({"name": "output_root", "status": "ready", "message": "output root exists and is writable"})
            elif check_only:
                parent = _nearest_existing_parent(output_root.parent)
                if parent.exists() and os.access(parent, os.W_OK):
                    checks.append({"name": "output_root", "status": "ready", "message": f"output root can be created under {parent}"})
                else:
                    message = f"output root parent is not writable or does not exist: {output_root.parent}"
                    checks.append({"name": "output_root", "status": "blocked", "message": message})
                    blockers.append(message)
            else:
                output_root.mkdir(parents=True, exist_ok=True)
                checks.append({"name": "output_root", "status": "ready", "message": "output root was created"})
        except OSError as exc:
            message = f"output root could not be prepared: {exc}"
            checks.append({"name": "output_root", "status": "blocked", "message": message})
            blockers.append(message)

    status = "blocked" if blockers else ("skipped" if source_asset is None and output_root is None else "ready")
    message = "request inputs are ready" if status == "ready" else "request input checks were skipped" if status == "skipped" else "; ".join(blockers)
    return _runtime_entry(
        status,
        message,
        source_asset=str(source_asset) if source_asset else "",
        output_root=str(output_root) if output_root else "",
        checks=checks,
    )


def _check_repo_python(project_root: Path | None, *, check_only: bool, skip_uv_sync: bool) -> tuple[dict[str, Any], list[Step]]:
    uv = _which("uv")
    steps: list[Step] = []
    if project_root is None:
        return _runtime_entry("skipped", "no pyproject.toml found near cwd or preflight script", executable=sys.executable), steps
    if uv is None:
        return _runtime_entry(
            "blocked",
            "`uv` was not found on PATH",
            project_root=str(project_root),
            executable=sys.executable,
            install_hint=UV_INSTALL_HINT,
        ), steps
    if not check_only and not skip_uv_sync:
        steps.append(_run([uv, "sync", "--dev", "--python", "3.12"], cwd=project_root, timeout=1800))
    status = "blocked" if any(step.status == "blocked" for step in steps) else "ready"
    message = "repo Python environment is synchronized" if status == "ready" else "repo Python environment sync failed"
    if check_only or skip_uv_sync:
        message = "repo Python environment sync was not run"
    venv = _project_venv_dir(project_root)
    repo_python = _project_venv_python(project_root)
    return _runtime_entry(
        status,
        message,
        project_root=str(project_root),
        executable=str(repo_python or sys.executable),
        uv=uv,
        venv=str(venv) if venv else "",
    ), steps


def _check_openusd_python(project_root: Path | None) -> tuple[dict[str, Any], list[Step]]:
    python = str(_project_venv_python(project_root) or Path(sys.executable))
    step = _run([python, "-c", OPENUSD_IMPORT_CHECK], env=_env_with_extra_path(_project_venv_bin(project_root)), timeout=60)
    status = "ready" if step.status == "ready" else "blocked"
    message = "OpenUSD Python APIs are importable" if status == "ready" else "OpenUSD Python APIs are not importable"
    return _runtime_entry(status, message, executable=python), [step]


def _check_asset_validator(project_root: Path | None) -> tuple[dict[str, Any], list[Step]]:
    executable = _find_executable("omni_asset_validate", project_root=project_root)
    if executable:
        return _runtime_entry("ready", "omni_asset_validate CLI is on PATH", executable=executable), []
    python = str(_project_venv_python(project_root) or Path(sys.executable))
    step = _run([python, "-c", ASSET_VALIDATOR_IMPORT_CHECK], env=_env_with_extra_path(_project_venv_bin(project_root)), timeout=60)
    status = "ready" if step.status == "ready" else "blocked"
    message = "omni.asset_validator Python module is importable" if status == "ready" else "omni_asset_validate CLI and omni.asset_validator module are unavailable"
    return _runtime_entry(status, message, executable=python), [step]


def _check_git_lfs(*, install_lfs: bool, check_only: bool) -> tuple[dict[str, Any], list[Step]]:
    git_lfs = _which("git-lfs") or _which("git")
    if _which("git-lfs") is None:
        return _runtime_entry("blocked", "`git-lfs` was not found on PATH"), []
    steps: list[Step] = []
    if install_lfs and not check_only:
        steps.append(_run(["git", "lfs", "install"], timeout=120))
        steps.append(_run(["git", "lfs", "pull"], timeout=1800))
    status = "blocked" if any(step.status == "blocked" for step in steps) else "ready"
    return _runtime_entry(status, "Git LFS is available", executable=git_lfs), steps


def _resolve_usd_convert_cad_python(project_root: Path | None) -> Path:
    """Resolve the interpreter expected to hold the usd-convert-cad wheel.

    The wheel bundles its own ``pxr`` runtime, so it belongs in a dedicated
    interpreter. Order: ``USD_CONVERT_CAD_PYTHON`` env -> the project ``.venv``
    -> the current interpreter.
    """
    env_python = os.environ.get(USD_CONVERT_CAD_PYTHON_ENV)
    if env_python:
        # Do not call Path.resolve(): venv/bin/python is commonly a symlink to
        # the base interpreter, and dereferencing it loses the venv package
        # context when that target is executed directly.
        return Path(os.path.abspath(os.fspath(Path(env_python).expanduser())))
    return _project_venv_python(project_root) or Path(sys.executable)


def _usd_convert_cad_skill_md(skill_source: dict[str, Any] | None) -> str | None:
    """Locate the SKILL.md inside a cloned usd-convert-cad checkout, if present."""
    if not skill_source:
        return None
    root = skill_source.get("path")
    if not root:
        return None
    checkout = Path(str(root))
    candidates = [
        checkout / "skills" / "omniverse-cad-to-usd" / "SKILL.md",
        checkout / ".agents" / "skills" / "usd-convert-cad" / "SKILL.md",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    if checkout.exists():
        matches = sorted(checkout.glob("**/SKILL.md"))
        if matches:
            return str(matches[0])
    return None


def _check_usd_convert_cad(
    *,
    project_root: Path | None,
    check_only: bool,
    skill_source: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[Step]]:
    steps: list[Step] = []
    python = _resolve_usd_convert_cad_python(project_root)
    skill_md = _usd_convert_cad_skill_md(skill_source)
    if not python.exists():
        runtime = _runtime_entry(
            "blocked",
            f"usd-convert-cad interpreter was not found: {python}. {USD_CONVERT_CAD_INSTALL_HINT}",
            executable=str(python),
            skill_md=skill_md,
        )
        return runtime, steps

    # A self-contained wheel is "ready" when it imports in the resolved
    # interpreter. This is a fast check, so run it even in --check-only mode;
    # preflight never pip-installs the wheel implicitly (see the install hint).
    import_step = _run([str(python), "-c", USD_CONVERT_CAD_IMPORT_CHECK], timeout=120)
    steps.append(import_step)
    installed_versions: dict[str, str | None] = {}
    if import_step.status == "ready":
        version_step, installed_versions = _package_versions_step(
            python,
            ("usd-convert-cad",),
            name="usd_convert_cad_version_gate",
        )
        steps.append(version_step)
    status = "ready" if steps[-1].status == "ready" else "blocked"
    if status == "ready":
        version = installed_versions["usd-convert-cad"]
        message = f"usd-convert-cad wheel matches pinned version {version}"
        if skill_md:
            message = f"{message}; capability/guidance SKILL.md: {skill_md}"
    elif import_step.status == "ready":
        message = steps[-1].message
    else:
        detail = "\n".join(part for part in (import_step.stdout_tail, import_step.stderr_tail) if part)
        message = (
            f"usd-convert-cad wheel is not importable with {python}. {USD_CONVERT_CAD_INSTALL_HINT}"
        )
        if detail:
            message = f"{message} Output: {tail_text(detail)}"
    runtime = _runtime_entry(
        status,
        message,
        executable=str(python),
        skill_md=skill_md,
        version=installed_versions.get("usd-convert-cad"),
        required_version=USD_CONVERT_CAD_VERSION,
        version_manifest=str(UPSTREAM_VERSION_MANIFEST_PATH),
    )
    if status == "blocked":
        runtime["install_hint"] = USD_CONVERT_CAD_INSTALL_HINT
    return runtime, steps


def _check_usd_convert_gsplat(root: Path, *, project_root: Path | None) -> tuple[dict[str, Any], list[Step]]:
    executable = _find_executable("gsplat2USD", project_root=project_root)
    cli_source = root / "source" / "python" / "usd_convert_gsplat" / "cli.py"
    if not root.exists():
        return _runtime_entry("blocked", "usd-convert-gsplat checkout is missing", root=str(root), executable=executable), []
    if not cli_source.is_file():
        return _runtime_entry("blocked", "usd-convert-gsplat CLI source is missing", root=str(root), executable=executable), []
    if executable is None:
        return _runtime_entry("blocked", "gsplat2USD CLI is not on PATH", root=str(root)), []
    return _runtime_entry("ready", "gsplat2USD CLI and upstream capability source are available", root=str(root), executable=executable), []


def _venv_bin_dir(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts" if os.name == "nt" else "bin")


def _simready_executable(venv_dir: Path) -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    return _venv_bin_dir(venv_dir) / f"simready-validate{suffix}"


def _simready_runtime_import_check(venv_dir: Path) -> Step:
    python = _venv_bin_dir(venv_dir) / ("python.exe" if os.name == "nt" else "python")
    step, _ = _package_versions_step(
        python,
        SIMREADY_RUNTIME_PACKAGE_NAMES,
        name="simready_validate_runtime_check",
        import_check="import numpy; import omni.asset_validator; from pxr import Usd",
    )
    return step


def _foundation_requirements_path(root: Path) -> Path:
    candidates = [
        root / "requirements.txt",
        root / "nv_core" / "validator_sample" / "requirements.txt",
    ]
    return next((path for path in candidates if path.is_file()), candidates[0])


def _simready_pip_install_command(python: Path, requirements: list[str], *, uv: str | None) -> list[str]:
    if uv:
        return [uv, "pip", "install", "--python", str(python), *requirements]
    return [str(python), "-m", "pip", "install", "--disable-pip-version-check", *requirements]


def _simready_pip_install_step(python: Path, requirements: list[str], *, uv: str | None) -> Step:
    return _run(_simready_pip_install_command(python, requirements, uv=uv), timeout=1800)


def _install_simready_locked_runtime(python: Path, *, uv: str | None) -> list[Step]:
    steps = [_simready_pip_install_step(python, list(SIMREADY_RUNTIME_REQUIREMENTS), uv=uv)]
    if steps[-1].status != "blocked":
        steps.append(_simready_pip_install_step(python, ["--no-deps", *SIMREADY_NO_DEPS_REQUIREMENTS], uv=uv))
    return steps


def _check_simready(root: Path, venv_root: Path, *, project_root: Path | None, check_only: bool) -> tuple[dict[str, Any], list[Step]]:
    requirements = _foundation_requirements_path(root)
    if not root.exists():
        return _runtime_entry("blocked", "SimReady Foundation checkout is missing", root=str(root)), []
    if not requirements.is_file():
        return _runtime_entry("blocked", "SimReady Foundation requirements.txt is missing", root=str(root)), []
    steps: list[Step] = []
    venv_dir = venv_root / "simready-validate"
    venv_executable = _simready_executable(venv_dir)
    if venv_executable.is_file():
        import_check = _simready_runtime_import_check(venv_dir)
        if import_check.status == "blocked":
            if check_only:
                return _runtime_entry(
                    "blocked",
                    f"existing simready-validate runtime does not match the pinned package set: {import_check.message}",
                    root=str(root),
                    executable=str(venv_executable),
                    venv=str(venv_dir),
                    required_versions=expected_versions(SIMREADY_RUNTIME_PACKAGE_NAMES),
                    version_manifest=str(UPSTREAM_VERSION_MANIFEST_PATH),
                ), [import_check]
            steps.append(
                Step(
                    name="simready_validate_runtime_check",
                    status="ready",
                    message="existing preflight venv is missing runtime dependencies; reinstalling",
                    stderr_tail=import_check.stderr_tail,
                )
            )
        else:
            steps.append(import_check)
            return _runtime_entry(
                "ready",
                "simready-validate pinned runtime is available from the preflight venv",
                root=str(root),
                executable=str(venv_executable),
                venv=str(venv_dir),
                required_versions=expected_versions(SIMREADY_RUNTIME_PACKAGE_NAMES),
                version_manifest=str(UPSTREAM_VERSION_MANIFEST_PATH),
            ), steps
    if check_only:
        return _runtime_entry(
            "blocked",
            "the pinned simready-validate runtime was not installed in check-only mode",
            root=str(root),
            required_versions=expected_versions(SIMREADY_RUNTIME_PACKAGE_NAMES),
            version_manifest=str(UPSTREAM_VERSION_MANIFEST_PATH),
        ), []
    uv = _which("uv")
    if uv:
        venv_command = [uv, "venv", "--python", "3.12"]
        if venv_dir.exists():
            venv_command.append("--clear")
        steps = [_run([*venv_command, str(venv_dir)], timeout=300)]
    else:
        steps = [_run([sys.executable, "-m", "venv", str(venv_dir)], env=_env_with_extra_path(_project_venv_bin(project_root)), timeout=300)]
    python = _venv_bin_dir(venv_dir) / ("python.exe" if os.name == "nt" else "python")
    if steps[-1].status != "blocked":
        steps.extend(_install_simready_locked_runtime(python, uv=uv))
    if steps[-1].status != "blocked":
        steps.append(_simready_runtime_import_check(venv_dir))
    status = "ready" if venv_executable.is_file() and steps[-1].status != "blocked" else "blocked"
    return _runtime_entry(
        status,
        "simready-validate was installed into the preflight venv" if status == "ready" else "simready-validate install failed",
        root=str(root),
        executable=str(venv_executable) if venv_executable.exists() else "",
        venv=str(venv_dir),
        required_versions=expected_versions(SIMREADY_RUNTIME_PACKAGE_NAMES),
        version_manifest=str(UPSTREAM_VERSION_MANIFEST_PATH),
    ), steps


SERVICE_ENV_BY_SERVICE = {
    "ovrtx": ("RENDER_ENDPOINT", "OVRTX_RENDER_ENDPOINT", "CONTENT_AGENTS_RENDER_BASE_URL"),
    "material": ("CONTENT_AGENTS_MATERIAL_AGENT_BASE_URL", "MATERIAL_AGENT_BASE_URL"),
    "physics": ("CONTENT_AGENTS_PHYSICS_AGENT_BASE_URL", "PHYSICS_AGENT_BASE_URL"),
    "texture": ("CONTENT_AGENTS_TEXTURE_AGENT_BASE_URL", "TEXTURE_AGENT_BASE_URL"),
}


def _service_env_url(service: str) -> str:
    for name in SERVICE_ENV_BY_SERVICE[service]:
        value = os.environ.get(name)
        if value:
            return value.rstrip("/")
    return DEFAULT_SERVICE_URLS[service]


def _service_url_was_provided(service: str) -> bool:
    return any(bool(os.environ.get(name)) for name in SERVICE_ENV_BY_SERVICE[service])


def _health_urls(base_url: str) -> list[str]:
    clean = base_url.rstrip("/")
    return [f"{clean}/health", f"{clean}/v2/health/ready"]


def _is_remote_or_invocation_endpoint(base_url: str) -> bool:
    lowered = base_url.lower()
    return lowered.startswith("https://") or "nvcf" in lowered or "invocation" in lowered


def _render_url(base_url: str) -> str:
    clean = base_url.rstrip("/")
    return clean if clean.endswith("/render") else f"{clean}/render"


def _ovrtx_render_smoke_timeout() -> int:
    raw = os.environ.get("OVRTX_PREFLIGHT_SMOKE_TIMEOUT_SECONDS", "120")
    try:
        timeout = int(raw)
    except ValueError:
        timeout = 120
    return max(5, timeout)


def _iter_render_image_candidates(value: Any, parent_key: str | None = None, in_images: bool = False) -> Any:
    candidate_keys = {"image", "png", "image_data", "output_image", "render", "rendered_image", "images", "rgb"}
    if isinstance(value, str) and (in_images or parent_key in candidate_keys):
        yield value
        return
    if isinstance(value, list):
        for item in value:
            yield from _iter_render_image_candidates(item, parent_key, in_images)
        return
    if isinstance(value, dict):
        nested_in_images = in_images or parent_key == "images"
        for key, item in value.items():
            yield from _iter_render_image_candidates(item, key, nested_in_images)


def _render_response_has_png(body: bytes) -> tuple[bool, str]:
    if body.startswith(b"\x89PNG\r\n\x1a\n"):
        return True, "render smoke returned PNG bytes"
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return False, f"render smoke response was not PNG or JSON: {exc}"
    if isinstance(payload, dict) and payload.get("status") == "exception":
        return False, f"render smoke reported exception: {payload.get('error') or 'unknown error'}"
    for candidate in _iter_render_image_candidates(payload):
        data = candidate.split(",", 1)[1] if candidate.startswith("data:image") and "," in candidate else candidate
        try:
            decoded = base64.b64decode(data, validate=True)
        except (ValueError, TypeError):
            continue
        if decoded.startswith(b"\x89PNG\r\n\x1a\n"):
            return True, "render smoke returned base64 PNG"
    if isinstance(payload, dict) and payload.get("images") == {}:
        return False, "render smoke returned success with empty images"
    return False, "render smoke did not return PNG bytes or a base64 PNG field"


def _probe_ovrtx_render_smoke(base_url: str) -> dict[str, Any]:
    payload = {
        "url": "data:application/octet-stream;base64,"
        + base64.b64encode(OVRTX_RENDER_SMOKE_USDA.encode("utf-8")).decode("ascii"),
        "force_render": True,
        "render_settings": {
            "camera_paths": ["/World/Camera"],
            "frame_range": {"start": 0, "end": 0},
            "camera_parameters": {"width": 64, "height": 64},
            "sensors": None,
            "apply_background_mask": False,
            "num_sensor_updates": 1,
        },
    }
    url = _render_url(base_url)
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=_ovrtx_render_smoke_timeout()) as response:
            body = response.read()
            passed, message = _render_response_has_png(body)
            status = "ready" if passed else "blocked"
            return {
                "status": status,
                "render_url": url,
                "message": message,
                "response_status": response.status,
                "response_content_type": response.headers.get("Content-Type", ""),
                "response_bytes": len(body),
            }
    except HTTPError as exc:
        body = exc.read(4096).decode("utf-8", errors="replace")
        return {
            "status": "blocked",
            "render_url": url,
            "message": f"render smoke HTTP {exc.code}: {body}",
        }
    except URLError as exc:
        return {
            "status": "blocked",
            "render_url": url,
            "message": f"render smoke could not reach endpoint: {exc.reason}",
        }
    except TimeoutError:
        return {
            "status": "blocked",
            "render_url": url,
            "message": "render smoke timed out",
        }
    except OSError as exc:
        return {
            "status": "blocked",
            "render_url": url,
            "message": f"render smoke failed: {exc}",
        }


def _missing_api_keys_message(service: str, render_endpoint: str | None) -> str:
    """Explain a Content Agents ``api_keys_configured: false`` health payload.

    A remote render endpoint is the actionable cause worth naming: upstream
    Material/Physics readiness demands a render usage key for any renderer that
    is not on this machine, so say which key is missing and why.
    """
    message = f"{service} health endpoint responded but API keys are not configured"
    if render_endpoint and not _is_local_render_endpoint(render_endpoint):
        host = _endpoint_host(render_endpoint) or render_endpoint
        message += (
            f"; the render endpoint host {host} is remote, so the {service} agent requires"
            " a render usage key in NGC_API_KEY or NVCF_API_KEY"
        )
    return message


def _probe_service(
    service: str,
    base_url: str,
    timeout: int = 8,
    *,
    render_endpoint: str | None = None,
    provided: bool = False,
) -> dict[str, Any]:
    if _is_remote_or_invocation_endpoint(base_url):
        return {
            "status": "ready",
            "base_url": base_url.rstrip("/"),
            "message": f"{service} remote/provided endpoint accepted; generic unauthenticated health probe skipped",
        }
    deadline = time.monotonic() + timeout
    errors: list[str] = []
    while True:
        errors = []
        for url in _health_urls(base_url):
            request = Request(url, method="GET")
            request_timeout = max(1.0, min(5.0, deadline - time.monotonic()))
            try:
                with urlopen(request, timeout=request_timeout) as response:
                    body = response.read(4096).decode("utf-8", errors="replace")
                    try:
                        health_payload = json.loads(body)
                    except json.JSONDecodeError:
                        health_payload = {}
                    if service == "ovrtx":
                        ovrtx_ready = (
                            health_payload.get("status") == "healthy"
                            and health_payload.get("gpu_initialized", True) is not False
                            and health_payload.get("renderer_initialized", True) is not False
                        )
                        if not ovrtx_ready:
                            errors.append(f"{url}: OVRTX renderer is not ready: {body}")
                            continue
                        smoke = _probe_ovrtx_render_smoke(base_url)
                        if smoke["status"] != "ready":
                            return {
                                "status": "blocked",
                                "base_url": base_url.rstrip("/"),
                                "health_url": url,
                                "message": f"ovrtx health endpoint responded but render smoke failed: {smoke['message']}",
                                "health_response": body,
                                "render_smoke": smoke,
                            }
                        return {
                            "status": "ready",
                            "base_url": base_url.rstrip("/"),
                            "health_url": url,
                            "message": f"{service} health endpoint and render smoke responded with HTTP {response.status}",
                            "health_response": body,
                            "render_smoke": smoke,
                        }
                    if service in {"material", "physics", "texture"} and health_payload.get("api_keys_configured") is False:
                        if provided and _upstream_render_gate_rejects_local_endpoint(render_endpoint):
                            # Reuse path: the user owns this already-running
                            # agent, so preflight cannot mirror a render key
                            # into it the way managed deployment does. The
                            # configured renderer is on this machine under a
                            # name upstream's allowlist misses, which is enough
                            # to explain the flag, so do not block a service
                            # that renders fine over HTTP.
                            return {
                                "status": "ready",
                                "base_url": base_url.rstrip("/"),
                                "health_url": url,
                                "message": (
                                    f"{service} provided endpoint accepted; its `api_keys_configured: false` is"
                                    f" explained by the host-local render endpoint {_endpoint_host(render_endpoint)},"
                                    " which upstream readiness does not recognize as local"
                                ),
                                "health_response": body,
                            }
                        return {
                            "status": "blocked",
                            "base_url": base_url.rstrip("/"),
                            "health_url": url,
                            "message": _missing_api_keys_message(service, render_endpoint),
                            "health_response": body,
                        }
                    return {
                        "status": "ready",
                        "base_url": base_url.rstrip("/"),
                        "health_url": url,
                        "message": f"{service} health endpoint responded with HTTP {response.status}",
                        "health_response": body,
                    }
            except HTTPError as exc:
                errors.append(f"{url}: HTTP {exc.code}")
            except URLError as exc:
                errors.append(f"{url}: {exc.reason}")
            except TimeoutError:
                errors.append(f"{url}: timed out")
            except OSError as exc:
                errors.append(f"{url}: {exc}")
        if time.monotonic() >= deadline:
            break
        time.sleep(min(2.0, max(0.1, deadline - time.monotonic())))
    return {
        "status": "blocked",
        "base_url": base_url.rstrip("/"),
        "message": f"{service} health endpoint did not respond",
        "errors": errors,
    }


def _deploy_ovrtx(world_root: Path) -> Step:
    compose_file = world_root / "apps" / "ovrtx_rendering_api" / "docker-compose.yml"
    if not compose_file.is_file():
        return Step(
            name="ovrtx_deploy",
            status="blocked",
            message="upstream OVRTX Docker Compose file was not found",
            command=["docker", "compose", "-f", str(compose_file), "up", "-d", "--build"],
        )
    env = os.environ.copy()
    env.setdefault("BUILDKIT_PROGRESS", "plain")
    env.setdefault("COMPOSE_BAKE", "false")
    env.setdefault("OVRTX_RENDER_MODE", "pt")
    command = ["docker", "compose", "-f", str(compose_file), "up", "-d"]
    if not _docker_image_exists("ovrtx-rendering-api:local"):
        command.append("--build")
    return _run(
        command,
        cwd=world_root,
        env=env,
        timeout=3600,
    )


def _docker_image_exists(image: str) -> bool:
    if _which("docker") is None:
        return False
    try:
        completed = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def _collection_agent_service_name(service: str) -> str:
    return f"{service}-agent-service"


def _collection_agent_image_name(service: str) -> str:
    return f"content-agents-{service}-agent-service"


def _deploy_content_agent_service(
    world_root: Path,
    service: str,
    *,
    include_texture: bool,
    model_options: dict[str, dict[str, str]] | None,
    render_endpoint: str | None,
) -> list[Step]:
    deploy_py = world_root / "deploy" / "collection" / "deploy.py"
    if not deploy_py.is_file():
        return [
            Step(
                name=f"{service}_agent_deploy",
                status="blocked",
                message="upstream Content Agents collection deploy.py was not found",
                command=[sys.executable, str(deploy_py), "up"],
            )
        ]

    config_path, config_step = _prepare_content_agents_collection_config(
        world_root,
        include_texture=include_texture,
        model_options=model_options,
        enabled_agents={service},
        render_endpoint=render_endpoint,
        config_suffix=service,
    )
    steps: list[Step] = []
    if config_step is not None:
        steps.append(config_step)
        if config_step.status == "blocked":
            return steps
    if config_path is None:
        return [
            Step(
                name=f"{service}_agent_deploy",
                status="blocked",
                message="failed to prepare deterministic Content Agents collection config",
                command=[sys.executable, str(deploy_py), "up"],
            )
        ]

    env = _content_agents_deploy_env(model_options, render_endpoint=render_endpoint)
    image_name = _collection_agent_image_name(service)
    service_name = _collection_agent_service_name(service)
    if _docker_image_exists(image_name):
        render_env_command = [sys.executable, str(deploy_py), "-c", str(config_path), "render-env"]
        steps.append(_run(render_env_command, cwd=world_root, env=env, timeout=300))
        steps[-1].name = f"{service}_agent_render_env"
        if steps[-1].status == "blocked":
            return steps

        generated_env = world_root / "deploy" / "collection" / ".collection.generated.env"
        compose_file = world_root / "deploy" / "collection" / "docker-compose.agents.yml"
        command = [
            "docker",
            "compose",
            "--env-file",
            str(generated_env),
            "-p",
            "content-agents",
            "-f",
            str(compose_file),
            "up",
            "-d",
            service_name,
        ]
    else:
        command = [sys.executable, str(deploy_py), "-c", str(config_path), "up"]
    steps.append(
        _run(
            command,
            cwd=world_root,
            env=env,
            timeout=3600,
        )
    )
    steps[-1].name = f"{service}_agent_deploy"
    return steps


def _check_content_agents_deployment_host() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    blockers: list[str] = []

    nvidia_smi = _which("nvidia-smi")
    if nvidia_smi is None:
        message = "nvidia-smi was not found on PATH"
        checks.append({"name": "nvidia_smi", "status": "blocked", "message": message})
        blockers.append(message)
    else:
        step = _run([nvidia_smi, "-L"], timeout=30)
        status = "ready" if step.status == "ready" else "blocked"
        message = "nvidia-smi reported at least one GPU" if status == "ready" else "nvidia-smi could not query GPUs"
        checks.append({"name": "nvidia_smi", "status": status, "message": message, "executable": nvidia_smi})
        if status == "blocked":
            blockers.append(message)

    docker = _which("docker")
    if docker is None:
        message = "docker was not found on PATH"
        checks.append({"name": "docker", "status": "blocked", "message": message})
        blockers.append(message)
    else:
        info = _run([docker, "info", "--format", "{{json .ServerVersion}}"], timeout=30)
        info_status = "ready" if info.status == "ready" else "blocked"
        info_message = "Docker daemon is reachable" if info_status == "ready" else "Docker daemon is not reachable"
        checks.append({"name": "docker_daemon", "status": info_status, "message": info_message, "executable": docker})
        if info_status == "blocked":
            blockers.append(info_message)
        compose = _run([docker, "compose", "version"], timeout=30)
        compose_status = "ready" if compose.status == "ready" else "blocked"
        compose_message = "Docker Compose v2 is available" if compose_status == "ready" else "Docker Compose v2 is not available"
        checks.append({"name": "docker_compose_v2", "status": compose_status, "message": compose_message, "executable": docker})
        if compose_status == "blocked":
            blockers.append(compose_message)

    status = "ready" if not blockers else "blocked"
    return _runtime_entry(
        status,
        "local Content Agents deployment host is ready" if status == "ready" else "; ".join(blockers),
        checks=checks,
    )


def _should_check_content_agents_deployment_host(service_reports: dict[str, Any], *, will_deploy: bool) -> bool:
    if will_deploy:
        return True
    return any(report.get("status") == "blocked" and not _service_url_was_provided(service) for service, report in service_reports.items())


def _content_agent_services(include_texture: bool) -> tuple[str, ...]:
    services = ["material", "physics"]
    if include_texture:
        services.append("texture")
    return tuple(services)


def _agent_needs_render_endpoint(service: str) -> bool:
    return service in {"material", "physics"}


def _check_content_agents(
    world_root: Path,
    *,
    include_texture: bool,
    check_only: bool,
    skip_deploy: bool,
    model_options: dict[str, dict[str, str]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[Step]]:
    agent_services = _content_agent_services(include_texture)
    agents_to_deploy = [service for service in agent_services if not _service_url_was_provided(service)]
    needs_render = any(_agent_needs_render_endpoint(service) for service in agents_to_deploy)
    services = list(agent_services)
    if needs_render or _service_url_was_provided("ovrtx"):
        services.insert(0, "ovrtx")
    configured_render_endpoint = _service_env_url("ovrtx")
    service_reports = {
        service: _probe_service(
            service,
            _service_env_url(service),
            render_endpoint=None if service == "ovrtx" else configured_render_endpoint,
            provided=_service_url_was_provided(service),
        )
        for service in services
    }

    explicit_blockers = [
        service
        for service, report in service_reports.items()
        if _service_url_was_provided(service) and report.get("status") == "blocked"
    ]
    if explicit_blockers:
        message = "explicit Content Agents endpoint(s) are not healthy: " + ", ".join(explicit_blockers)
        if explicit_blockers == ["ovrtx"]:
            message = "explicit renderer endpoint is not healthy"
        return (
            _runtime_entry(
                "blocked",
                message,
                root=str(world_root),
            ),
            service_reports,
            [],
        )

    if not agents_to_deploy and not needs_render and all(report["status"] == "ready" for report in service_reports.values()):
        return _runtime_entry("ready", "Content Agents services are already healthy", root=str(world_root)), service_reports, []

    steps: list[Step] = []
    will_deploy = not (check_only or skip_deploy) and bool(agents_to_deploy or needs_render)
    if _should_check_content_agents_deployment_host(service_reports, will_deploy=will_deploy):
        deployment_host = _check_content_agents_deployment_host()
    else:
        deployment_host = _runtime_entry(
            "skipped",
            "explicit Content Agents endpoints were provided; local deployment host diagnostics were not needed",
        )
    if check_only or skip_deploy:
        return (
            _runtime_entry(
                "blocked",
                "Content Agents services are not healthy and deployment was not requested",
                root=str(world_root),
                deployment_host=deployment_host,
            ),
            service_reports,
            steps,
        )
    if not world_root.exists():
        return _runtime_entry("blocked", "content-agents checkout is missing", root=str(world_root)), service_reports, steps
    if os.name == "nt":
        return (
            _runtime_entry(
                "blocked",
                "Content Agents deployment requires a Linux Docker/GPU host; use WSL2/Linux Docker or provide healthy endpoints",
                root=str(world_root),
            ),
            service_reports,
            steps,
        )
    if _which("docker") is None:
        return _runtime_entry("blocked", "docker was not found on PATH", root=str(world_root), deployment_host=deployment_host), service_reports, steps
    if deployment_host.get("status") == "blocked":
        return _runtime_entry("blocked", "Content Agents local deployment host is not ready", root=str(world_root), deployment_host=deployment_host), service_reports, steps
    if not _content_agents_provider_credentials_available(model_options):
        return (
            _runtime_entry(
                "blocked",
                "a Content Agents provider API key is required for managed local deployment; set one of NVIDIA_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, or GEMINI_API_KEY",
                root=str(world_root),
                deployment_host=deployment_host,
            ),
            service_reports,
            steps,
            )

    if agents_to_deploy:
        steps.append(
            _ensure_content_agents_secret_env(
                world_root,
                model_options=model_options,
                render_endpoint=_service_env_url("ovrtx") if needs_render else None,
            )
        )
        if steps[-1].status == "blocked":
            return _runtime_entry("blocked", "failed to prepare upstream Content Agents credential environment", root=str(world_root)), service_reports, steps

    render_endpoint: str | None = None
    if needs_render and _service_url_was_provided("ovrtx"):
        render_endpoint = service_reports["ovrtx"]["base_url"]
    elif needs_render:
        steps.append(_deploy_ovrtx(world_root))
        if steps[-1].status == "blocked":
            return _runtime_entry("blocked", "failed to deploy upstream OVRTX renderer", root=str(world_root)), service_reports, steps
        service_reports["ovrtx"] = _probe_service("ovrtx", _service_env_url("ovrtx"), timeout=300)
        if service_reports["ovrtx"].get("status") != "ready":
            return (
                _runtime_entry(
                    "blocked",
                    "upstream OVRTX renderer did not become healthy before agent deployment",
                    root=str(world_root),
                ),
                service_reports,
                steps,
            )
        render_endpoint = service_reports["ovrtx"]["base_url"]

    for service in agent_services:
        if _service_url_was_provided(service):
            continue
        service_steps = _deploy_content_agent_service(
            world_root,
            service,
            include_texture=include_texture,
            model_options=model_options,
            render_endpoint=render_endpoint if _agent_needs_render_endpoint(service) else render_endpoint,
        )
        steps.extend(service_steps)
        if any(step.status == "blocked" for step in service_steps):
            return (
                _runtime_entry(
                    "blocked",
                    f"failed to deploy upstream Content Agents {service} service",
                    root=str(world_root),
                ),
                service_reports,
                steps,
            )
        service_reports[service] = _probe_service(
            service,
            _service_env_url(service),
            timeout=300,
            render_endpoint=render_endpoint or configured_render_endpoint,
        )
        if service_reports[service].get("status") != "ready":
            return (
                _runtime_entry(
                    "blocked",
                    f"upstream Content Agents {service} service did not become healthy after deployment",
                    root=str(world_root),
                ),
                service_reports,
                steps,
            )

    status = "ready" if all(report["status"] == "ready" for report in service_reports.values()) else "blocked"
    return _runtime_entry(status, "Content Agents services are healthy" if status == "ready" else "Content Agents deployment did not produce healthy endpoints", root=str(world_root)), service_reports, steps


def _env_payload(manifest_path: Path, manifest: dict[str, Any]) -> dict[str, str]:
    env: dict[str, str] = {
        "PHYSICAL_AI_PREFLIGHT_MANIFEST": str(manifest_path),
        "PHYSICAL_AI_REQUIRE_PREFLIGHT": "1",
        "OMNIVERSE_CAD_TO_SIMREADY_HOME": manifest["paths"]["home"],
        "OMNIVERSE_CAD_TO_SIMREADY_UPSTREAM_ROOT": manifest["paths"]["upstream_root"],
        "OMNIVERSE_CAD_TO_SIMREADY_STATE": manifest["paths"]["state_root"],
    }
    for name, entry in manifest.get("upstreams", {}).items():
        env_name = UPSTREAMS.get(name, {}).get("env")
        if env_name and entry.get("path"):
            env[str(env_name)] = str(entry["path"])
    simready = manifest.get("runtimes", {}).get("simready_validate", {})
    if simready.get("venv"):
        env["PHYSICAL_AI_SIMREADY_VALIDATE_VENV"] = str(simready["venv"])
    repo_python = manifest.get("runtimes", {}).get("repo_python", {})
    if repo_python.get("venv"):
        path_value = _path_env_with_entries(_venv_bin_dir(Path(str(repo_python["venv"]))))
        if path_value:
            env["PATH"] = path_value
    for key, value in manifest.get("env", {}).items():
        env[key] = str(value)
    return env


def _write_env_file(path: Path, env: dict[str, str]) -> None:
    lines = [
        "# Source this file before running cad-to-simready references.",
    ]
    for key in sorted(env):
        value = env[key].replace("'", "'\"'\"'")
        lines.append(f"export {key}='{value}'")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_powershell_env_file(path: Path, env: dict[str, str]) -> None:
    lines = [
        "# Dot-source this file before running cad-to-simready references.",
    ]
    for key in sorted(env):
        value = env[key].replace("'", "''")
        lines.append(f"$env:{key} = '{value}'")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _dotenv_line_key(raw_line: str) -> str | None:
    line = raw_line.strip()
    if not line or line.startswith("#") or "=" not in line:
        return None
    key = line.split("=", 1)[0].strip()
    if key.startswith("export "):
        key = key.removeprefix("export ").strip()
    return key or None


def _dotenv_line_pair(raw_line: str) -> tuple[str, str] | None:
    line = raw_line.strip()
    if not line or line.startswith("#") or "=" not in line:
        return None
    key, value = line.split("=", 1)
    key = key.strip()
    if key.startswith("export "):
        key = key.removeprefix("export ").strip()
    if not key or not (key[0].isalpha() or key[0] == "_"):
        return None
    if not all(character.isalnum() or character == "_" for character in key):
        return None
    value = value.strip()
    if value and value[0] in {"'", '"'} and value[-1:] == value[0]:
        value = value[1:-1]
    elif " #" in value:
        value = value.split(" #", 1)[0].rstrip()
    return key, value


def _read_dotenv_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines() if path.is_file() else []


def _load_content_agents_env_file(path: Path, *, extra_secret_names: tuple[str, ...] = ()) -> dict[str, Any]:
    expanded = path.expanduser().resolve()
    if not expanded.is_file():
        return _runtime_entry(
            "blocked",
            f"Content Agents private env file was not found: {expanded}",
            path=str(expanded),
        )

    loaded: list[str] = []
    lines = _read_dotenv_lines(expanded)
    allowed_names = set(CONTENT_AGENTS_SECRET_ENV_NAMES) | set(CONTENT_AGENTS_MODEL_ENV_NAMES) | set(extra_secret_names)
    for line in lines:
        parsed = _dotenv_line_pair(line)
        if not parsed:
            continue
        key, value = parsed
        if key not in allowed_names or not value:
            continue
        os.environ.setdefault(key, value)
        loaded.append(key)

    custom_secret_names = {
        value.strip()
        for name in CONTENT_AGENTS_API_KEY_ENV_NAME_ENV_NAMES
        for value in [os.environ.get(name, "")]
        if value.strip()
    }
    custom_secret_names.update(name for name in extra_secret_names if name)
    for line in lines:
        parsed = _dotenv_line_pair(line)
        if not parsed:
            continue
        key, value = parsed
        if key not in custom_secret_names or not value:
            continue
        os.environ.setdefault(key, value)
        loaded.append(key)

    return _runtime_entry(
        "ready",
        f"loaded {len(sorted(set(loaded)))} Content Agents environment name(s) from private env file",
        path=str(expanded),
        names=sorted(set(loaded)),
    )


def _content_agents_secret_line(name: str, value: str) -> str:
    return f"{name}={value.replace(chr(10), '')}"


def _content_agents_secret_env(
    *,
    model_options: dict[str, dict[str, str]] | None = None,
    render_endpoint: str | None = None,
) -> dict[str, str]:
    available = {
        name: os.environ[name]
        for name in CONTENT_AGENTS_SECRET_ENV_NAMES
        if os.environ.get(name)
    }
    for name in CONTENT_AGENTS_API_KEY_ENV_NAME_ENV_NAMES:
        secret_name = os.environ.get(name, "").strip()
        if secret_name:
            secret_value = _read_credential_env(secret_name)
            if secret_value:
                available[secret_name] = secret_value
    if model_options:
        for group in model_options.values():
            secret_name = _parse_env_reference(group.get("api_key_env", ""))
            secret_value = _read_credential_env(secret_name)
            if secret_name and secret_value:
                available[secret_name] = secret_value
    if "NGC_API_KEY" not in available and _is_local_render_endpoint(render_endpoint):
        # The upstream local collection uses host.docker.internal for the local
        # renderer, while Material/Physics readiness treats non-local render
        # hostnames as requiring the render usage key. Mirror whichever
        # provider credential is actually configured into NGC_API_KEY so the
        # render gate is satisfied, regardless of which VLM/LLM backend
        # (nim, openai, anthropic, gemini, ...) is in use.
        for name in CONTENT_AGENTS_PROVIDER_SECRET_ENV_NAMES:
            if os.environ.get(name):
                available["NGC_API_KEY"] = os.environ[name]
                break
    for name in CONTENT_AGENTS_MODEL_ENV_NAMES:
        if os.environ.get(name):
            available[name] = os.environ[name]
    return available


def _ensure_content_agents_secret_env(
    world_root: Path,
    *,
    model_options: dict[str, dict[str, str]] | None = None,
    render_endpoint: str | None = None,
) -> Step:
    """Mirror known deployment secrets from the process env into upstream .env."""
    env_path = world_root / ".env"
    available = _content_agents_secret_env(
        model_options=model_options,
        render_endpoint=render_endpoint,
    )
    if not available:
        return Step(
            name="content_agents_secret_env",
            status="skipped",
            message="no known Content Agents credential environment variables were set",
        )
    lines = _read_dotenv_lines(env_path)
    existing_keys: set[str] = set()
    changed = False
    for index, raw_line in enumerate(lines):
        key = _dotenv_line_key(raw_line)
        if not key:
            continue
        existing_keys.add(key)
        if key in available:
            replacement = _content_agents_secret_line(key, available[key])
            if lines[index] != replacement:
                lines[index] = replacement
                changed = True
    appended: list[str] = []
    append_order = [
        *CONTENT_AGENTS_SECRET_ENV_NAMES,
        *sorted(
            set(available)
            - set(CONTENT_AGENTS_SECRET_ENV_NAMES)
            - set(CONTENT_AGENTS_MODEL_ENV_NAMES)
        ),
    ]
    for name in append_order:
        if name in available and name not in existing_keys:
            appended.append(_content_agents_secret_line(name, available[name]))
            existing_keys.add(name)
    if appended:
        if lines and lines[-1].strip():
            lines.append("")
        lines.extend(appended)
        changed = True
    if changed:
        env_path.parent.mkdir(parents=True, exist_ok=True)
        env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        env_path.chmod(0o600)
    except OSError:
        pass
    action = "updated" if changed else "already contains"
    count = len(available)
    return Step(
        name="content_agents_secret_env",
        status="ready",
        message=f"upstream Content Agents .env {action} {count} deployment environment name(s)",
    )


def _write_markdown(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# CAD to SimReady Preflight",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Platform: `{manifest['platform']['system']}`",
        f"- Manifest: `{manifest['manifest_path']}`",
        "",
        "## Runtimes",
        "",
    ]
    for name, entry in sorted(manifest.get("runtimes", {}).items()):
        lines.append(f"- `{name}`: `{entry.get('status')}` - {entry.get('message', '')}")
    lines.extend(["", "## Services", ""])
    for name, entry in sorted(manifest.get("services", {}).items()):
        lines.append(f"- `{name}`: `{entry.get('status')}` - {entry.get('base_url', '')}")
    lines.extend(["", "## Blockers", ""])
    blockers = manifest.get("blockers", [])
    lines.extend(f"- {blocker}" for blocker in blockers)
    if not blockers:
        lines.append("- None")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_manifest(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, str]]:
    home = args.home.expanduser().resolve() if args.home else _default_home().resolve()
    state_root = args.state_root.expanduser().resolve() if args.state_root else _default_state_root(home).resolve()
    upstream_root = args.upstream_root.expanduser().resolve() if args.upstream_root else _default_upstream_root(home).resolve()
    venv_root = args.venv_root.expanduser().resolve() if args.venv_root else _default_venv_root(home).resolve()
    manifest_path = args.report.expanduser().resolve() if args.report else state_root / "cad-to-simready-preflight.json"
    project_root = _find_project_root(args.project_root)
    source_asset = args.source_asset.expanduser().resolve() if args.source_asset else None
    output_root = args.output_root.expanduser().resolve() if args.output_root else None
    targets = _selected_targets(args.targets, skip_content_agents=args.skip_content_agents)
    conversion_tools, route_selection = _selected_conversion_tools(args.conversion_tools, source_asset, args.source_format)
    legacy_options_ignored: list[str] = []

    steps: list[Step] = []
    upstreams: dict[str, Any] = {}
    runtimes: dict[str, Any] = {}
    services: dict[str, Any] = {}
    env: dict[str, str] = {}

    if "content-agents" in targets and args.content_agents_secret_env_file:
        extra_secret_names = tuple(
            name
            for name in (
                args.content_agents_vlm_api_key_env,
                args.content_agents_llm_api_key_env,
                args.content_agents_image_gen_api_key_env,
            )
            if name
        )
        runtimes["content_agents_private_env"] = _load_content_agents_env_file(
            args.content_agents_secret_env_file,
            extra_secret_names=extra_secret_names,
        )

    content_agents_model_options = _content_agents_model_options(args)

    request_runtime = _check_request_inputs(source_asset, output_root, check_only=args.check_only)
    runtimes["request"] = request_runtime

    if not args.check_only:
        home.mkdir(parents=True, exist_ok=True)
        state_root.mkdir(parents=True, exist_ok=True)
        upstream_root.mkdir(parents=True, exist_ok=True)
        venv_root.mkdir(parents=True, exist_ok=True)

    repo_runtime, repo_steps = _check_repo_python(project_root, check_only=args.check_only, skip_uv_sync=args.skip_uv_sync)
    runtimes["repo_python"] = repo_runtime
    steps.extend(repo_steps)
    git_lfs_runtime, git_lfs_steps = _check_git_lfs(install_lfs=args.lfs, check_only=args.check_only)
    runtimes["git_lfs"] = git_lfs_runtime
    steps.extend(git_lfs_steps)

    if "conversion" in targets:
        # usd-convert-cad conversion uses the pip wheel; the checkout is cloned
        # only as its SKILL.md capability/guidance source. usd-convert-gsplat is
        # a git-checkout runtime.
        upstreams_by_tool = {
            "usd-convert-cad": "usd_convert_cad",
            "usd-convert-gsplat": "usd_convert_gsplat",
        }
        for tool_name in sorted(conversion_tools & set(upstreams_by_tool)):
            upstream_name = upstreams_by_tool[tool_name]
            upstreams[upstream_name], upstream_steps = _ensure_upstream(upstream_name, upstream_root, check_only=args.check_only, no_update=args.no_update)
            steps.extend(upstream_steps)
        if "usd-convert-cad" in conversion_tools:
            cad_runtime, cad_steps = _check_usd_convert_cad(
                project_root=project_root,
                check_only=args.check_only,
                skill_source=upstreams.get("usd_convert_cad"),
            )
            runtimes["usd_convert_cad"] = cad_runtime
            steps.extend(cad_steps)
        if "usd-convert-gsplat" in conversion_tools:
            runtimes["usd_convert_gsplat"], gsplat_steps = _check_usd_convert_gsplat(Path(upstreams["usd_convert_gsplat"]["path"]), project_root=project_root)
            steps.extend(gsplat_steps)
        if "repo-python" in conversion_tools:
            runtimes["source_conversion_repo_python"] = _runtime_entry(
                "ready" if repo_runtime.get("status") == "ready" else "blocked",
                "repo Python conversion tools are available" if repo_runtime.get("status") == "ready" else "repo Python conversion tools require repo Python readiness",
                project_root=str(project_root) if project_root else "",
            )

    if "validation" in targets:
        openusd_runtime, openusd_steps = _check_openusd_python(project_root)
        runtimes["openusd_python"] = openusd_runtime
        steps.extend(openusd_steps)
        asset_validator_runtime, asset_validator_steps = _check_asset_validator(project_root)
        runtimes["asset_validator"] = asset_validator_runtime
        steps.extend(asset_validator_steps)
        upstreams["simready_foundation"], simready_upstream_steps = _ensure_upstream("simready_foundation", upstream_root, check_only=args.check_only, no_update=args.no_update)
        steps.extend(simready_upstream_steps)
        simready_runtime, simready_steps = _check_simready(
            Path(upstreams["simready_foundation"]["path"]),
            venv_root,
            project_root=project_root,
            check_only=args.check_only,
        )
        runtimes["simready_validate"] = simready_runtime
        steps.extend(simready_steps)

    if "content-agents" in targets:
        upstreams["content_agents"], world_steps = _ensure_upstream("content_agents", upstream_root, check_only=args.check_only, no_update=args.no_update)
        steps.extend(world_steps)
        content_runtime, services, content_steps = _check_content_agents(
            Path(upstreams["content_agents"]["path"]),
            include_texture=args.include_texture,
            check_only=args.check_only,
            skip_deploy=args.skip_deploy,
            model_options=content_agents_model_options,
        )
        runtimes["content_agents"] = content_runtime
        steps.extend(content_steps)
        if services.get("material", {}).get("status") == "ready":
            env["CONTENT_AGENTS_MATERIAL_AGENT_BASE_URL"] = services["material"]["base_url"]
        if services.get("physics", {}).get("status") == "ready":
            env["CONTENT_AGENTS_PHYSICS_AGENT_BASE_URL"] = services["physics"]["base_url"]
        if services.get("texture", {}).get("status") == "ready":
            env["CONTENT_AGENTS_TEXTURE_AGENT_BASE_URL"] = services["texture"]["base_url"]
        if services.get("ovrtx", {}).get("status") == "ready":
            env["RENDER_ENDPOINT"] = services["ovrtx"]["base_url"]
            env["OVRTX_RENDER_ENDPOINT"] = services["ovrtx"]["base_url"]

    blockers = _manifest_blockers(upstreams, runtimes, services)
    status = "ready" if not blockers else "blocked"
    if args.skip_content_agents and "content-agents" not in targets:
        runtimes["content_agents"] = _runtime_entry("skipped", "Content Agents preflight was explicitly skipped")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "skill": SKILL,
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "preflight_mode": "route-aware-dependency-bootstrap",
        "dependency_policy": "selected-target-dependencies",
        "upstream_version_policy": "pinned-tested-integration",
        "upstream_version_manifest": str(UPSTREAM_VERSION_MANIFEST_PATH),
        "platform": {
            "system": platform.system().lower(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
        },
        "targets": list(targets),
        "conversion_tools": sorted(conversion_tools),
        "legacy_options_ignored": legacy_options_ignored,
        "route_selection": route_selection,
        "paths": {
            "home": str(home),
            "state_root": str(state_root),
            "upstream_root": str(upstream_root),
            "venv_root": str(venv_root),
            "project_root": str(project_root) if project_root else "",
            "output_root": str(output_root) if output_root else "",
        },
        "upstreams": upstreams,
        "runtimes": runtimes,
        "services": services,
        "env": env,
        "steps": [step.to_dict() for step in steps],
        "blockers": blockers,
    }
    return manifest, _env_payload(manifest_path, manifest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install cad-to-simready dependencies, deploy/verify services, and write a preflight manifest.")
    parser.add_argument("--targets", default="", help="Comma-separated targets to prepare: conversion, validation, content-agents.")
    parser.add_argument("--home", type=Path)
    parser.add_argument("--state-root", type=Path)
    parser.add_argument("--upstream-root", type=Path)
    parser.add_argument("--venv-root", type=Path)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--source-asset", type=Path, help="Optional source asset used to infer the conversion route.")
    parser.add_argument("--output-root", type=Path, help="Optional output root to verify or create before downstream stages.")
    parser.add_argument("--source-format", default="", help="Optional source format used to infer the conversion route when no source asset is available.")
    parser.add_argument("--conversion-tools", default="", help="Comma-separated conversion tools to prepare: repo-python, usd-convert-cad, usd-convert-gsplat.")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--powershell-env-file", type=Path)
    parser.add_argument("--markdown-report", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--skip-uv-sync", action="store_true")
    parser.add_argument("--skip-content-agents", action="store_true")
    parser.add_argument("--skip-deploy", action="store_true")
    parser.add_argument("--include-texture", action="store_true")
    parser.add_argument("--content-agents-secret-env-file", type=Path, help="Optional private dotenv file with Content Agents provider credentials such as NVIDIA_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, or GEMINI_API_KEY.")
    parser.add_argument("--content-agents-vlm-backend", default=None, help="Optional Content Agents VLM backend override, for example openai, anthropic, gemini, or nim.")
    parser.add_argument("--content-agents-vlm-model", default=None, help="Optional Content Agents VLM model override, for example gpt-5.5, claude-sonnet-4-20250514, or gemini-3-pro-preview.")
    parser.add_argument("--content-agents-vlm-endpoint", default=None, help="Optional Content Agents VLM OpenAI-compatible /v1 endpoint override; use openai for custom OpenAI-compatible endpoints and nim only for NIM endpoints.")
    parser.add_argument("--content-agents-vlm-api-key-env", default=None, help="Optional environment variable name the VLM backend should read for its API key, for example CUSTOM_VLM_API_KEY.")
    parser.add_argument("--content-agents-llm-backend", default=None, help="Optional Content Agents LLM backend override, for example openai, anthropic, gemini, or nim.")
    parser.add_argument("--content-agents-llm-model", default=None, help="Optional Content Agents LLM model override, for example gpt-5.5, claude-sonnet-4-20250514, or gemini-3-pro-preview.")
    parser.add_argument("--content-agents-llm-endpoint", default=None, help="Optional Content Agents LLM OpenAI-compatible /v1 endpoint override; use openai for custom OpenAI-compatible endpoints and nim only for NIM endpoints.")
    parser.add_argument("--content-agents-llm-api-key-env", default=None, help="Optional environment variable name the LLM backend should read for its API key, for example CUSTOM_LLM_API_KEY.")
    parser.add_argument("--content-agents-image-gen-backend", default=None, help="Optional Content Agents image-generation backend override, for example openai or nim.")
    parser.add_argument("--content-agents-image-gen-model", default=None, help="Optional Content Agents image-generation model override, for example black-forest-labs/flux.2-klein-4b.")
    parser.add_argument("--content-agents-image-gen-endpoint", default=None, help="Optional Content Agents image-generation OpenAI-compatible /v1 endpoint override; use openai for custom OpenAI-compatible endpoints and nim only for NIM endpoints.")
    parser.add_argument("--content-agents-image-gen-api-key-env", default=None, help="Optional environment variable name the image-generation backend should read for its API key, for example MA_IMAGE_GEN_API_KEY or TA_IMAGE_GEN_API_KEY.")
    parser.add_argument("--lfs", action="store_true", help="Run git lfs install and git lfs pull for repo fixtures.")
    parser.add_argument("--no-update", action="store_true", help="Do not update existing upstream checkouts.")
    args = parser.parse_args(argv)

    manifest, env = build_manifest(args)
    manifest_path = Path(manifest["manifest_path"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.env_file:
        _write_env_file(args.env_file.expanduser().resolve(), env)
    if args.powershell_env_file:
        _write_powershell_env_file(args.powershell_env_file.expanduser().resolve(), env)
    if args.markdown_report:
        _write_markdown(args.markdown_report.expanduser().resolve(), manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
