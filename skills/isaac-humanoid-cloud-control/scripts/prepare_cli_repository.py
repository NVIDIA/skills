#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage the public Isaac ROS APT key and source without privileged writes.

Usage: python3 prepare_cli_repository.py
Output: a private temporary directory containing repository.gpg and repository.list.
Exit codes: 0 ready; 1 failed. Caller installs the reviewed files and removes the
directory after installation. Existing host APT configuration is never modified.
"""

import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import urllib.request

KEY_URL = "https://isaac.download.nvidia.com/isaac-ros/repos.key"
# Public key bundle retrieved from the official HTTPS endpoint on 2026-10-02.
# Rotation requires review of the new bundle, not accepting a runtime checksum.
KEY_SHA256 = "d45fbd909e89914960a66dba17d4c659adf908ab4d6fe2cf55381e9ffe5ce59d"  # gitleaks:allow -- public bundle digest, not a credential
KEYRING = "/usr/share/keyrings/nvidia-isaac-ros.gpg"
SOURCE = (
    f"deb [signed-by={KEYRING}] "
    "https://isaac.download.nvidia.com/isaac-ros/release-4.6 noble main\n"
)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """Do not let the fixed key endpoint redirect to a different source."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def stage_key(payload: bytes, directory: Path, gpg: str) -> None:
    """Verify the exact public bundle before invoking GPG or writing a source."""
    if hashlib.sha256(payload).hexdigest() != KEY_SHA256:
        raise ValueError("public signing-key digest mismatch; review upstream key rotation")
    key = directory / "repository.asc"
    key.write_bytes(payload)
    gnupg = directory / "gnupg"
    gnupg.mkdir(mode=0o700)
    subprocess.run(
        [gpg, "--batch", "--homedir", str(gnupg), "--dearmor", "--output",
         str(directory / "repository.gpg"), str(key)],
        check=True, timeout=30, capture_output=True,
    )
    (directory / "repository.list").write_text(SOURCE, encoding="utf-8")


def main() -> int:
    directory = None
    try:
        gpg = shutil.which("gpg")
        if not gpg:
            raise RuntimeError("install the gnupg prerequisite first")
        directory = Path(tempfile.mkdtemp(prefix="humanoid-apt-"))
        with urllib.request.build_opener(NoRedirect).open(KEY_URL, timeout=30) as response:
            payload = response.read(1024 * 1024 + 1)
        if len(payload) > 1024 * 1024:
            raise ValueError("public signing-key response exceeds size limit")
        stage_key(payload, directory, gpg)
        print(directory)
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        if directory:
            shutil.rmtree(directory)
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
