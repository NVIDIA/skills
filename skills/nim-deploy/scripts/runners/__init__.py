# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runners: how the local (`docker`) target starts and stops a NIM container.

The run state machine, the validator and the guard never build a start or stop command
themselves. They ask the run's runner. Another runner, such as a CLI wrapper, implements the same
functions in its own module and is registered below, so the skill text does not change.

A runner module provides:

    required_inputs() -> tuple[str, ...]
        Run inputs that must be set before render (the docker target asks for them).
    render(inputs, run_dir) -> list[Path]
        Write the artifacts into run_dir and return their paths. Raise nd_common.RenderError for
        an input that cannot produce a valid artifact. Never put a secret value in an artifact.
    apply_command(state) -> list[str]
        The argv that starts the deployment (printed as facts.next_command, run alone).
    rollback_command(state) -> list[str]
        The argv that removes it.
    guard_forms(approval) -> list[list[str]]
        Every argv the guard accepts for the approved run: the apply and rollback forms and
        nothing else. `approval.record` holds the inputs bound at approve time and
        `approval.folder` is the run folder.
    ready_url(inputs) -> str
        Where the endpoint answers once the container is up.
    container_name(inputs) -> str
        The container that log triage reads.
    default_inputs(inputs) -> dict[str, str]                           (optional)
        Values for required inputs the user did not give; the run records them with source
        "default" and the approval summary shows them.
    missing_inputs(inputs) -> list[str]                                (optional)
        Further inputs needed for these inputs (docker: `model_path` for a model-free image).
    deny_rules() -> list[dict[str, str]]
        Commands of this runner's tool that are always denied, in the guard-rules.json deny format
        ({"id", "pattern", "reason"}); the guard adds the rules of every registered runner.
    preflight(state, offline) -> tuple[str, list[dict[str, str]]]      (optional)
        Live host checks run by validate.py; returns ("ok" | "failed" | "skipped", errors).

Register a runner in RUNNERS, or at run time with
NIM_DEPLOY_RUNNERS_EXTRA="name=package.module[,name2=module2]". Pick one with the run input
`runner` or the environment variable NIM_DEPLOY_RUNNER (default: docker).
"""

from __future__ import annotations

import importlib
import os
from types import ModuleType
from typing import Any

RUNNERS: dict[str, str] = {"docker": "runners.docker", "nimcli": "runners.nimcli"}


def default_runner() -> str:
    return os.environ.get("NIM_DEPLOY_RUNNER", "docker")


DEFAULT_RUNNER = default_runner()


def registry() -> dict[str, str]:
    """RUNNERS plus any `name=module` pairs from NIM_DEPLOY_RUNNERS_EXTRA."""
    found = dict(RUNNERS)
    for pair in filter(None, os.environ.get("NIM_DEPLOY_RUNNERS_EXTRA", "").split(",")):
        name, _, module = pair.partition("=")
        if name.strip() and module.strip():
            found[name.strip()] = module.strip()
    return found


def name_for(value: Any) -> str:
    """The runner name for a `runner` input value (empty means the default)."""
    return str(value) if value else default_runner()


def load(name: str | None = None) -> ModuleType:
    """Import the runner module called NAME (default: the default runner)."""
    chosen = name_for(name)
    modules = registry()
    if chosen not in modules:
        raise ValueError(f"unknown runner {chosen!r}; known runners: {', '.join(sorted(modules))}")
    return importlib.import_module(modules[chosen])


def deny_rules() -> list[dict[str, str]]:
    """The deny rules of every registered runner."""
    found: list[dict[str, str]] = []
    for name in registry():
        found += load(name).deny_rules()
    return found


def name_of(state: dict[str, Any]) -> str:
    """The runner name of a run: the one saved at creation (`state["runner"]`), else its `runner`
    input, else the default. Old runs without the saved name fall back to the environment."""
    saved = state.get("runner")
    if saved:
        return str(saved)
    entry = state.get("inputs", {}).get("runner")
    return name_for(entry.get("value") if isinstance(entry, dict) else None)


def runner_of(state: dict[str, Any]) -> ModuleType:
    """The runner of a run (see name_of)."""
    return load(name_of(state))
