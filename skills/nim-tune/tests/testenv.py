# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The environment a test gives the scripts it runs: never a copy of the developer's."""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile

# what a script needs to start (find python3, a temp folder, its locale); read by name, so no
# credential in the developer's environment reaches a test subprocess
KEEP = ("PATH", "TMPDIR", "LANG", "LC_ALL", "SYSTEMROOT")

# a HOME of the tests' own: a subprocess never reads the developer's home folder (Python's
# user site, ~/.docker, ~/.kube), removed when the test run ends
TEST_HOME = tempfile.mkdtemp(prefix="nim-tune-test-home-")
atexit.register(shutil.rmtree, TEST_HOME, ignore_errors=True)


def base_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """A minimal environment for a test subprocess, with a temporary HOME, plus EXTRA."""
    env = {name: os.environ[name] for name in KEEP if name in os.environ}
    env["HOME"] = TEST_HOME
    env.update(extra or {})
    return env
