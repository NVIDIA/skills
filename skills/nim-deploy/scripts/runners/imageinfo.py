# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only image metadata for the docker runner: which port does this image serve on?

Never pulls an image and never reads or prints credentials. Sources, in order: `docker image
inspect` (a local image), `docker buildx imagetools inspect` and `skopeo inspect --config` (the
registry, with the user's own registry login). Only the exposed tcp ports and the port variables
of the image environment are kept; the rest of `Env` (which can hold secrets) is dropped.
"""

from __future__ import annotations

import functools
import json
import re
import shutil
import subprocess
from typing import Any

DOCUMENTED_PORT = "8000"  # NIM LLM documents NIM_SERVER_PORT=8000; not verified for any image
PORT_ENV = ("NIM_HTTP_API_PORT", "NIM_SERVER_PORT")  # variables that name the API port
LOOKUP_TIMEOUT = 30


def _config_of(data: Any) -> dict[str, Any] | None:
    """The image Config from `docker image inspect .Config`, skopeo or buildx imagetools output."""
    if not isinstance(data, dict):
        return None
    if "ExposedPorts" in data or "Env" in data:
        return data
    if isinstance(data.get("config"), dict):
        return dict(data["config"])
    for key in ["linux/amd64", *[k for k in data if str(k).startswith("linux/")]]:
        entry = data.get(key)
        if isinstance(entry, dict) and isinstance(entry.get("config"), dict):
            return dict(entry["config"])
    return None


def _lookup(argv: list[str]) -> tuple[dict[str, Any] | None, str]:
    """Run one read-only metadata command; (config, why not). Credentials are never read here."""
    if not shutil.which(argv[0]):
        return None, f"{argv[0]} is not installed"
    try:
        proc = subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=LOOKUP_TIMEOUT,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, "timed out"
    if proc.returncode != 0:
        tail = (proc.stderr.strip().splitlines() or ["failed"])[-1]
        hint = (
            " (registry login needed?)"
            if re.search(
                r"unauthorized|access to the resource is denied|authentication required|\b40[13]\b",
                tail,
                re.I,
            )
            else ""
        )
        return None, tail[:120] + hint
    try:
        config = _config_of(json.loads(proc.stdout))
    except ValueError:
        config = None
    return (config, "") if config is not None else (None, "no image config in the output")


@functools.lru_cache(maxsize=8)
def inspect_image(image: str) -> dict[str, Any]:
    """Read-only image metadata, without pulling: the local image, else the registry.

    Order: `docker image inspect` (local), `docker buildx imagetools inspect`, `skopeo inspect
    --config`. Returns {"via", "ports", "env", "errors"}; env keeps only the port variables."""
    commands = [
        (
            "docker image inspect",
            ["docker", "image", "inspect", "--format", "{{json .Config}}", image],
        ),
        (
            "docker buildx imagetools inspect",
            ["docker", "buildx", "imagetools", "inspect", "--format", "{{json .Image}}", image],
        ),
        ("skopeo inspect", ["skopeo", "inspect", "--config", f"docker://{image}"]),
    ]
    errors: list[str] = []
    for via, argv in commands:
        config, why = _lookup(argv)
        if config is None:
            errors.append(f"{via}: {why}")
            continue
        exposed = config.get("ExposedPorts") or {}
        ports = sorted({key.split("/")[0] for key in exposed if str(key).endswith("/tcp")}, key=int)
        env = dict(item.split("=", 1) for item in config.get("Env") or [] if "=" in item)
        return {
            "via": via,
            "ports": ports,
            "env": {k: env[k] for k in PORT_ENV if k in env},
            "errors": errors,
        }
    return {"via": None, "ports": [], "env": {}, "errors": errors}


def resolve_container_port(image: str) -> dict[str, Any]:
    """{"port", "source", "note", "ports"}; port None means the image has several: ask."""
    info = inspect_image(image)
    via = info["via"]
    if via:
        for name in PORT_ENV:
            value = info["env"].get(name)
            if value and value.isdigit():
                note = f"{name}={value} in the image environment (read with {via})"
                return {
                    "port": value,
                    "source": "image-metadata",
                    "note": note,
                    "ports": info["ports"],
                }
        if len(info["ports"]) == 1:
            note = f"the image exposes {info['ports'][0]}/tcp (read with {via})"
            return {
                "port": info["ports"][0],
                "source": "image-metadata",
                "note": note,
                "ports": info["ports"],
            }
        if len(info["ports"]) > 1:
            return {
                "port": None,
                "source": "image-metadata",
                "note": f"the image exposes {', '.join(info['ports'])}",
                "ports": info["ports"],
            }
        why = f"the image declares no port (read with {via})"
    else:
        why = "image metadata unavailable (" + "; ".join(info["errors"]) + ")"
    note = f"{why}; documented NIM default {DOCUMENTED_PORT}, not verified for this image"
    return {"port": DOCUMENTED_PORT, "source": "documented-default", "note": note, "ports": []}
