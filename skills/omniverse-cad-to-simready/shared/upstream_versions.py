#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load the CAD-to-SimReady tested upstream lock manifest."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Iterable


UPSTREAM_VERSION_MANIFEST_PATH = Path(__file__).resolve().parents[1] / "upstream-versions.lock.json"


@lru_cache(maxsize=1)
def load_upstream_version_manifest() -> dict[str, Any]:
    payload = json.loads(UPSTREAM_VERSION_MANIFEST_PATH.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1.0":
        raise ValueError(f"unsupported upstream version manifest: {UPSTREAM_VERSION_MANIFEST_PATH}")
    return payload


def git_upstreams() -> dict[str, dict[str, str]]:
    return load_upstream_version_manifest()["git_upstreams"]


def package_version(name: str) -> str:
    return str(load_upstream_version_manifest()["python_packages"][name])


def package_requirement(name: str) -> str:
    return f"{name}=={package_version(name)}"


def runtime_package_names(runtime: str, *, no_deps: bool = False) -> tuple[str, ...]:
    key = "no_deps" if no_deps else "packages"
    return tuple(load_upstream_version_manifest()["runtime_sets"][runtime][key])


def runtime_requirements(runtime: str, *, no_deps: bool = False) -> tuple[str, ...]:
    return tuple(package_requirement(name) for name in runtime_package_names(runtime, no_deps=no_deps))


def expected_versions(names: Iterable[str]) -> dict[str, str]:
    return {name: package_version(name) for name in names}


def version_probe_code(names: Iterable[str]) -> str:
    distributions = tuple(names)
    return (
        "import importlib.metadata as m, json; "
        f"names={distributions!r}; "
        "norm=lambda value: value.lower().replace('_', '-').replace('.', '-'); "
        "installed={norm(d.metadata['Name']): d.version for d in m.distributions() if d.metadata['Name']}; "
        "print(json.dumps({name: installed.get(norm(name)) for name in names}, sort_keys=True))"
    )


def version_mismatches(expected: dict[str, str], installed: dict[str, str | None]) -> list[str]:
    return [
        f"requires {name}=={version}, found {installed.get(name) or 'not installed'}"
        for name, version in expected.items()
        if installed.get(name) != version
    ]
