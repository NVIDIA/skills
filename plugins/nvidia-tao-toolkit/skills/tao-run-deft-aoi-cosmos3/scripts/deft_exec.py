#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Run one external command under the immutable DEFT execution policy."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shlex
import subprocess
import sys
from typing import Any


PACKAGE_TOOLS = {
    "apt", "apt-get", "conda", "dnf", "mamba", "micromamba", "pip",
    "pip3", "uv", "yum",
}
NETWORK_TOOLS = {"aria2c", "curl", "git-lfs", "http", "httpie", "ngc", "scp", "wget"}
OFFLINE_ENV = {
    "AIR_GAPPED": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "PIP_NO_INDEX": "1",
}


def _policy(state_path: pathlib.Path) -> dict[str, Any]:
    state = json.loads(state_path.expanduser().read_text())
    policy = state.get("execution_policy")
    if isinstance(policy, dict):
        return policy
    airgap = os.environ.get("AIR_GAPPED") == "1"
    return {
        "network_mode": "airgap" if airgap else "network-enabled",
        "allow_package_install": not airgap,
        "allow_remote_fetch": not airgap,
        "allow_container_pull": not airgap,
        "allow_registry_login": not airgap,
    }


def _basename(value: str) -> str:
    return pathlib.PurePath(value).name.lower()


def _reject_airgap(command: list[str], policy: dict[str, Any]) -> None:
    if policy.get("network_mode") != "airgap":
        return
    if not command:
        raise ValueError("command is empty")
    tokens = [_basename(token) for token in command]
    if any(token in PACKAGE_TOOLS for token in tokens):
        raise ValueError("package installation is forbidden by air-gap policy")
    if any(token in NETWORK_TOOLS for token in tokens):
        raise ValueError("network/download command is forbidden by air-gap policy")
    for index, token in enumerate(tokens[:-1]):
        if token.startswith("python") and tokens[index + 1 : index + 3] == ["-m", "pip"]:
            raise ValueError("python -m pip is forbidden by air-gap policy")
    if "git" in tokens and any(
        token in {"clone", "fetch", "pull"}
        for token in tokens[tokens.index("git") + 1 :]
    ):
        raise ValueError("remote git operation is forbidden by air-gap policy")
    runtime_indexes = [index for index, token in enumerate(tokens) if token in {"docker", "podman"}]
    if runtime_indexes:
        runtime_index = runtime_indexes[0]
        runtime_tokens = tokens[runtime_index + 1 :]
        if any(token in {"login", "pull"} for token in runtime_tokens):
            raise ValueError("container registry access is forbidden by air-gap policy")
        if "manifest" in runtime_tokens:
            raise ValueError("container manifest lookup is forbidden by air-gap policy")
        pull_values = [token for token in runtime_tokens if token.startswith("--pull=")]
        if any(token != "--pull=never" for token in pull_values):
            raise ValueError("air-gap container runs require --pull=never")
        if "--pull" in runtime_tokens:
            index = runtime_tokens.index("--pull")
            if index + 1 >= len(runtime_tokens) or runtime_tokens[index + 1] != "never":
                raise ValueError("air-gap container runs require --pull never")
        for token in runtime_tokens:
            normalized = token.removeprefix("--env=").removeprefix("-e")
            if "=" not in normalized:
                continue
            name, value = normalized.split("=", 1)
            name = name.upper()
            if name in OFFLINE_ENV and value != OFFLINE_ENV[name]:
                raise ValueError(f"air-gap container cannot override {name}={value}")
        for index, token in enumerate(runtime_tokens[:-1]):
            if token not in {"-e", "--env"} or "=" not in runtime_tokens[index + 1]:
                continue
            name, value = runtime_tokens[index + 1].split("=", 1)
            name = name.upper()
            if name in OFFLINE_ENV and value != OFFLINE_ENV[name]:
                raise ValueError(f"air-gap container cannot override {name}={value}")
    if tokens[0] in {"bash", "sh", "zsh"} and "-c" in tokens:
        script = command[tokens.index("-c") + 1]
        _reject_airgap(shlex.split(script), policy)


def _with_no_pull(command: list[str], policy: dict[str, Any]) -> list[str]:
    if policy.get("network_mode") != "airgap" or not command:
        return command
    tokens = [_basename(token) for token in command]
    runtime_indexes = [index for index, token in enumerate(tokens) if token in {"docker", "podman"}]
    if not runtime_indexes or "run" not in tokens[runtime_indexes[0] + 1 :]:
        return command
    run_index = tokens.index("run", runtime_indexes[0] + 1)
    if any(token == "--pull" or token.startswith("--pull=") for token in tokens):
        return command
    return [*command[: run_index + 1], "--pull=never", *command[run_index + 1 :]]


def _with_offline_container_env(
    command: list[str], policy: dict[str, Any]
) -> list[str]:
    if policy.get("network_mode") != "airgap" or not command:
        return command
    tokens = [_basename(token) for token in command]
    runtime_indexes = [index for index, token in enumerate(tokens) if token in {"docker", "podman"}]
    if not runtime_indexes or "run" not in tokens[runtime_indexes[0] + 1 :]:
        return command
    run_index = tokens.index("run", runtime_indexes[0] + 1)
    options = [f"--env={name}={value}" for name, value in OFFLINE_ENV.items()]
    return [*command[: run_index + 1], *options, *command[run_index + 1 :]]


def run(state_path: pathlib.Path, command: list[str]) -> int:
    policy = _policy(state_path)
    _reject_airgap(command, policy)
    command = _with_no_pull(command, policy)
    command = _with_offline_container_env(command, policy)
    environment = os.environ.copy()
    if policy.get("network_mode") == "airgap":
        environment.update(OFFLINE_ENV)
    return subprocess.run(command, env=environment, check=False).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, type=pathlib.Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    try:
        return run(args.state, command)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"deft_exec: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
