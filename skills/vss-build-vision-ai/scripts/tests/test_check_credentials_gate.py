# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exit-code contract for the vss-build-vision-ai credential gate.

`SKILL.md` Step 3 and `references/credentials.md` tell callers to branch on the
exit code rather than on the printed lines, so the mapping is the interface:
0 every required credential present and validated where probed, 1 usage error
with nothing checked, 2 gated. These tests pin that mapping, plus the three
rules the exit code alone cannot express -- only a credential named by
`--require` may gate, a service that returns no verdict is not reported as a
rejection, and the presence-only credentials are never probed at all. `curl` is
stubbed, so no probe leaves the host.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest


GATE = Path(__file__).resolve().parents[1] / "check_credentials.sh"

# Real curl writes its `-w` output even when the transfer fails (a refused
# connection prints 000 and exits 7), and the gate reads only stdout, so the
# stub answers on stdout for every case.
# Only authn.nvidia.com answers: NVIDIA_API_KEY and HF_TOKEN are presence-only,
# so a probe for either is a regression, and leaving it unstubbed makes the
# tests below fail loudly rather than silently accept a new request.
CURL_STUB = """\
#!/bin/sh
printf '%s\\n' "$*" >> "$CURL_LOG"
for arg in "$@"; do
  case "$arg" in
    *authn.nvidia.com*) printf '%s' "$STUB_NGC"; exit 0 ;;
  esac
done
printf '000'
exit 7
"""


@dataclass(frozen=True)
class Gate:
    """Runs the gate in an environment that carries no real credentials."""

    bin_dir: Path
    curl_log: Path
    home: Path
    cwd: Path

    def run(
        self,
        *args: str,
        ngc: str = "200",
        **credentials: str,
    ) -> subprocess.CompletedProcess[str]:
        env = {
            "CURL_LOG": str(self.curl_log),
            "HOME": str(self.home),
            "LANG": "C.UTF-8",
            "PATH": f"{self.bin_dir}:/usr/bin:/bin",
            "STUB_NGC": ngc,
        }
        env.update(credentials)
        return subprocess.run(
            ["bash", str(GATE), *args],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=self.cwd,
            env=env,
            check=False,
        )

    @property
    def curl_calls(self) -> list[str]:
        if not self.curl_log.exists():
            return []
        return self.curl_log.read_text(encoding="utf-8").splitlines()


@pytest.fixture
def gate(tmp_path: Path) -> Gate:
    bin_dir = tmp_path / "bin"
    home = tmp_path / "home"
    cwd = tmp_path / "work"
    for directory in (bin_dir, home, cwd):
        directory.mkdir()
    stub = bin_dir / "curl"
    stub.write_text(CURL_STUB, encoding="utf-8")
    stub.chmod(0o755)
    return Gate(
        bin_dir=bin_dir, curl_log=tmp_path / "curl.log", home=home, cwd=cwd
    )


def output(result: subprocess.CompletedProcess[str]) -> str:
    return f"{result.stdout}\n{result.stderr}"


def test_no_requirement_and_no_credential_passes(gate: Gate) -> None:
    """The unused-key case: nothing declared, nothing set, nothing probed."""
    result = gate.run()
    assert result.returncode == 0, output(result)
    assert gate.curl_calls == []


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(("--bogus",), id="unknown-flag"),
        pytest.param(("--require",), id="missing-value"),
        pytest.param(("--require", ""), id="empty-value"),
        pytest.param(("--require=",), id="empty-inline-value"),
        pytest.param(("--require", "nvidia"), id="unknown-name"),
        pytest.param(("--require", "ngc,bogus"), id="unknown-name-in-list"),
    ],
)
def test_usage_error_exits_1_without_probing(gate: Gate, args: tuple[str, ...]) -> None:
    """1, not 2: 2 is reserved for a gate the caller must act on."""
    result = gate.run(*args)
    assert result.returncode == 1, output(result)
    assert "usage" in output(result).lower()
    assert gate.curl_calls == []


@pytest.mark.parametrize("value", ["*", "n?c", "[hn]*"])
def test_requirement_value_is_not_globbed(gate: Gate, value: str) -> None:
    """A requirement set must come from the argument, never from the cwd.

    Run from a directory holding files named after the requirements, unquoted
    word splitting would expand a glob into real requirement names and the
    gate would silently enforce whatever happened to be on disk.
    """
    for name in ("ngc", "nvidia-api", "hf"):
        (gate.cwd / name).touch()
    result = gate.run("--require", value)
    assert result.returncode == 1, output(result)
    assert f"unknown --require value: {value}" in result.stderr


@pytest.mark.parametrize(
    ("require", "label"),
    [
        ("ngc", "NGC"),
        ("nvidia-api", "NVIDIA_API_KEY"),
        ("hf", "HF_TOKEN"),
    ],
)
def test_unset_required_credential_gates(gate: Gate, require: str, label: str) -> None:
    result = gate.run("--require", require)
    assert result.returncode == 2, output(result)
    assert "BLOCKER" in result.stderr
    assert label in result.stderr
    assert gate.curl_calls == []


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(("--require", "ngc,nvidia-api,hf"), id="comma-list"),
        pytest.param(("--require=ngc,nvidia-api,hf",), id="inline-comma-list"),
        pytest.param(
            ("--require", "ngc", "--require", "nvidia-api", "--require", "hf"),
            id="repeated-flag",
        ),
    ],
)
def test_every_named_credential_is_enforced(gate: Gate, args: tuple[str, ...]) -> None:
    result = gate.run(*args)
    assert result.returncode == 2, output(result)
    for label in ("NGC", "NVIDIA_API_KEY", "HF_TOKEN"):
        assert label in result.stderr


@pytest.mark.parametrize("status", ["200", "204"])
@pytest.mark.parametrize("variable", ["NGC_CLI_API_KEY", "NGC_API_KEY"])
def test_validated_ngc_key_passes_under_either_name(
    gate: Gate, variable: str, status: str
) -> None:
    result = gate.run("--require", "ngc", ngc=status, **{variable: "resolved-key"})
    assert result.returncode == 0, output(result)
    assert "NGC key ok" in result.stdout
    assert "$oauthtoken:resolved-key" in gate.curl_calls[0]


def test_rejected_required_credential_gates_as_rejected(gate: Gate) -> None:
    result = gate.run("--require", "ngc", ngc="401", NGC_CLI_API_KEY="stale")
    assert result.returncode == 2, output(result)
    assert "rejected by authn.nvidia.com (HTTP 401)" in output(result)


def test_rejected_unrequired_credential_does_not_gate(gate: Gate) -> None:
    """A stale key for a service this build never calls must not block it."""
    result = gate.run(ngc="401", NGC_CLI_API_KEY="stale")
    assert result.returncode == 0, output(result)
    assert "not required by this build" in result.stdout


@pytest.mark.parametrize("status", ["000", "429", "503", "418"])
def test_absent_verdict_gates_but_is_not_called_a_rejection(
    gate: Gate, status: str
) -> None:
    """No answer, a rate limit, or a server error says nothing about the key.

    It still gates when required -- an unvalidated requirement is not a pass --
    so only the wording differs, and it must not send anyone to rotate a key
    that may be good.
    """
    result = gate.run("--require", "ngc", ngc=status, NGC_CLI_API_KEY="good-key")
    assert result.returncode == 2, output(result)
    assert "not validated" in output(result)
    assert "rejected" not in output(result)


@pytest.mark.parametrize("status", ["000", "503"])
def test_absent_verdict_for_an_unrequired_credential_does_not_gate(
    gate: Gate, status: str
) -> None:
    result = gate.run(ngc=status, NGC_CLI_API_KEY="good-key")
    assert result.returncode == 0, output(result)


def test_ngc_key_name_conflict_gates_without_any_requirement(gate: Gate) -> None:
    """Unconditional: credentials.md says stop and ask which key to use."""
    result = gate.run(NGC_CLI_API_KEY="one", NGC_API_KEY="two")
    assert result.returncode == 2, output(result)
    assert "differ" in result.stderr
    assert gate.curl_calls == []


def test_identical_ngc_key_names_are_not_a_conflict(gate: Gate) -> None:
    result = gate.run("--require", "ngc", NGC_CLI_API_KEY="same", NGC_API_KEY="same")
    assert result.returncode == 0, output(result)
    assert "NGC key ok" in result.stdout


def test_every_probe_is_bounded(gate: Gate) -> None:
    """A gate that promises to fail in seconds cannot sit on the OS timeout.

    With all three credentials set, the NGC key is the only one probed, so this
    also pins that the presence-only pair sends nothing.
    """
    result = gate.run(
        NGC_CLI_API_KEY="key", NVIDIA_API_KEY="key", HF_TOKEN="token"
    )
    assert result.returncode == 0, output(result)
    assert len(gate.curl_calls) == 1
    for call in gate.curl_calls:
        assert "--connect-timeout 5" in call
        assert "--max-time 15" in call


@pytest.mark.parametrize("args", [(), ("--require", "hf")])
def test_set_hf_token_is_reported_unvalidated_and_never_probed(
    gate: Gate, args: tuple[str, ...]
) -> None:
    """Presence is all the gate claims for HF_TOKEN.

    Hugging Face keeps a gated repository's metadata public, so `/api/models`
    answers the same for a good token, a junk token and none; `whoami-v2` does
    discriminate but could not be confirmed to accept a fine-grained read token,
    and passing it still would not prove checkpoint access. So a set token must
    neither be called valid nor gate the build, and no request goes out.
    """
    result = gate.run(*args, HF_TOKEN="whatever-this-is")
    assert result.returncode == 0, output(result)
    assert "not validated here" in result.stdout
    assert "HF_TOKEN ok" not in result.stdout
    assert not [call for call in gate.curl_calls if "huggingface.co" in call]


@pytest.mark.parametrize("args", [(), ("--require", "nvidia-api")])
def test_set_nvidia_key_is_reported_unvalidated_and_never_probed(
    gate: Gate, args: tuple[str, ...]
) -> None:
    """Presence is all the gate claims for NVIDIA_API_KEY.

    `integrate.api.nvidia.com/v1/models` is the public model catalog: it answers
    `200` with no `Authorization` header at all, so a `200` is not a verdict on
    the key and the `401`/`403` arm of `report_status` could never fire for this
    host. Pinning the absent probe is what stops it coming back: a key this
    bogus used to print `NVIDIA_API_KEY ok` and exit 0 under `--require`.
    """
    result = gate.run(*args, NVIDIA_API_KEY="test-not-a-real-key")
    assert result.returncode == 0, output(result)
    assert "not validated here" in result.stdout
    assert "NVIDIA_API_KEY ok" not in result.stdout
    assert not [
        call for call in gate.curl_calls if "integrate.api.nvidia.com" in call
    ]
