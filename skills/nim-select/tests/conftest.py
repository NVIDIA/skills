# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Make the skill's scripts importable from the tests in this folder, register the `network`
mark of the live tests, and keep the tests off the network: the China NIM distributor catalogs
and Harbor hosts, the location check and the country lookup go to a closed local port unless a
test serves them.
The scripts' endpoints are built in; only test code (ns_testkit) points them elsewhere."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ns_testkit


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "network: calls the live public endpoints (set NIM_SELECT_LIVE=1)"
    )


@pytest.fixture(autouse=True)
def _umask_022() -> Iterator[None]:
    """Each test runs under a umask of 022, as most hosts and CI runners do. Under 002 (Ubuntu's
    default for an account with a private group), a folder a test makes with mkdir is
    group-writable, and --out refuses it; a test that needs 002 sets it itself."""
    old = os.umask(0o022)
    try:
        yield
    finally:
        os.umask(old)


@pytest.fixture(autouse=True)
def _no_cnd_network(monkeypatch: pytest.MonkeyPatch) -> None:
    ns_testkit.keep_offline(monkeypatch)
