# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Put the skill's scripts/ and tests/ folders on sys.path, register the `network` mark, give
the `no_network` fixture, and stop the run at once when pytest's temporary folder fails the
scripts' private-folder check."""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
for folder in (TESTS, TESTS.parent / "scripts"):  # scripts/ ends up first
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import nsc_common as nsc  # noqa: E402


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "network: calls a live endpoint; opt-in, never part of the offline run"
    )


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any name lookup (getaddrinfo, gethostbyname, getnameinfo, a reverse lookup of an
    address other than 127.0.0.1 or ::1) and anything sent out (create_connection, connect
    or sendto on any socket) fails the test: no request leaves this process. A local fake
    server still starts: HTTPServer.server_bind looks up the name of its own 127.0.0.1."""
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("no network in this test")

    def local_name(address: str) -> tuple[str, list[str], list[str]]:
        if address not in ("127.0.0.1", "::1"):
            raise AssertionError("no network in this test")
        return "localhost", [], [address]

    for name in ("getaddrinfo", "gethostbyname", "gethostbyname_ex", "getnameinfo",
                 "create_connection"):
        monkeypatch.setattr(socket, name, refuse)
    monkeypatch.setattr(socket, "gethostbyaddr", local_name)
    for name in ("connect", "connect_ex", "sendto", "sendmsg"):  # sendmsg: POSIX only
        monkeypatch.setattr(socket.socket, name, refuse, raising=False)


def private_temp_problem(base: Path) -> str | None:
    """Why pytest's temporary folder `base` fails the check the scripts make on --out
    (nsc.check_private), as one message for the whole run; None when it passes."""
    try:
        nsc.check_private(base)
    except nsc.ScriptError as exc:
        return (
            f"pytest's temporary folder {base} fails the private-folder check of the scripts "
            f"({exc.code}): {exc.message}. Most tests write a run there and would each fail "
            "with out_not_private. This is the machine, not the skill: a sandbox or a user "
            "namespace often shows /tmp as owned by 'nobody'. Rerun with --basetemp set to a "
            "new folder under one you own, such as ~/.cache/nsc-pytest."
        )
    return None


@pytest.fixture(scope="session", autouse=True)
def private_basetemp(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Stop the run with one clear message when the temporary folder is not private."""
    problem = private_temp_problem(tmp_path_factory.getbasetemp())
    if problem:
        pytest.exit(problem, returncode=pytest.ExitCode.TESTS_FAILED)
