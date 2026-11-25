# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/bench_guard.py: direct benchmark tools and credentials are denied."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import bench_guard  # noqa: E402

DENIED = [
    "aiperf profile --model m --url http://x",
    "gym eval run --benchmark gsm8k",
    "ng eval run --benchmark gsm8k",
    "python3 -m aiperf profile --model m",
    "python -m nemo_gym.cli.main eval run",
    "sudo -u root aiperf profile",
    "env HF_TOKEN=$HF_TOKEN aiperf profile",
    "nohup aiperf profile --model m &",
    "timeout 60 aiperf profile --model m",
    "bash -c 'aiperf profile --model m'",
    'sh -lc "cd /w && aiperf profile"',
    "eval 'aiperf profile'",
    "echo $(aiperf profile --model m)",
    "cd /w && aiperf profile --model m",
    "true; gym eval run",
    "ssh gpu-01 aiperf profile --model m",
    "docker run --rm nvcr.io/nvidia/nemo-gym:1 eval run",
    "docker run --rm python:3.12 sh -c 'pip install aiperf && aiperf profile'",
    "kubectl exec pod -n ns -- aiperf profile --model m",
    "kubectl exec pod -- sh -c 'aiperf profile'",
    "podman run --rm img gym eval run",
    "curl -H 'Authorization: Bearer abcdefghijklmnop1234' http://x",
    "python3 nimbench.py perf --endpoint http://x --api-key sk-abcdefghijklmnopqrstuv",
    "export HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz",
    "tool --api-key secretvalue123",
    "docker run -e API_TOKEN=abc image",
    "ssh -p 22 host aiperf profile",
    "ssh -i ~/.ssh/id -o StrictHostKeyChecking=no host aiperf profile",
    "ssh -p22 -l me host -- aiperf profile",
    "ssh host -p 22 aiperf profile",
    "uv run aiperf profile --model m",
    "uv tool run aiperf profile",
    "uvx aiperf profile --model m",
    "uvx aiperf@0.13.0 profile",
    "pipx run aiperf profile",
    "echo m | xargs aiperf profile --model",
    "echo `aiperf profile --model m`",
    "echo aiperf profile | sh",
    "printf 'aiperf profile' | bash -s",
    "bash <<<'aiperf profile'",
    "bash <<EOF\naiperf profile\nEOF",
]

ALLOWED = [
    "python3 .agents/skills/nim-benchmark/scripts/nimbench.py perf --dry-run --endpoint http://x",
    "python3 scripts/nimbench.py accuracy --api-key-env NGC_API_KEY --gym-image img@sha256:ab",
    "pip install aiperf==0.13.0",
    "pip install aiperf",
    "ls -la runs/a && cat runs/a/report.md",
    "docker ps",
    "docker logs nim-benchmark-abc",
    "kubectl get pods -n ns",
    "echo aiperf is not run here",
    "git status",
    "python3 -m pytest tests",
    'curl -H "Authorization: Bearer $NGC_API_KEY" http://x/v1/models',
    "tool --api-key $MY_KEY",
    "tool --api-key-env MY_KEY",
    "command -v aiperf",
    "which aiperf",
    "type aiperf",
    "ssh -p 22 host nvidia-smi",
    "uv run pytest tests",
    "uv pip install aiperf",
    "echo hello | sh",
    "bash scripts/setup.sh",
]


@pytest.mark.parametrize("command", DENIED)
def test_denies(command: str) -> None:
    assert bench_guard.decide(command), command


@pytest.mark.parametrize("command", ALLOWED)
def test_allows(command: str) -> None:
    assert bench_guard.decide(command) is None, command


def run_hook(payload: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / "bench_guard.py")],
        input=payload,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_hook_exits_2_on_a_denied_command_and_names_the_fix() -> None:
    proc = run_hook(json.dumps({"tool_input": {"command": "aiperf profile --model m"}}))
    assert proc.returncode == 2
    assert "nimbench.py" in proc.stderr


def test_hook_allows_and_ignores_calls_without_a_command() -> None:
    assert run_hook(json.dumps({"tool_input": {"command": "ls"}})).returncode == 0
    assert run_hook(json.dumps({"tool_input": {"file_path": "x"}})).returncode == 0


def test_hook_fails_closed_on_unreadable_input() -> None:
    assert run_hook("not json").returncode == 2


def test_check_prints_one_contract_object() -> None:
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "bench_guard.py"), "--check", "gym eval run"],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )
    payload = json.loads(proc.stdout)
    assert proc.returncode == 2 and payload["errors"][0]["code"] == "denied"


def test_an_unbalanced_quote_is_still_checked() -> None:
    assert bench_guard.decide("echo 'oops && aiperf profile")


def test_deep_nesting_is_refused() -> None:
    assert bench_guard.decide("eval " * 7 + "true") == "the command is nested too deeply to check"
    assert bench_guard.decide("eval eval true") is None
