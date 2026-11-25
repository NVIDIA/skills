# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared SLURM preflight SSH identity and noninteractive connection policy."""

import os
from pathlib import Path


def ssh_command(host: str, remote_command: str) -> list[str]:
    """Use the configured platform identity and the user's SSH config."""
    user = os.environ.get("SLURM_USER")
    if not user:
        raise ValueError("SLURM_USER is required for SLURM probes")
    command = [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", "PasswordAuthentication=no",
        "-o", "PreferredAuthentications=publickey",
        "-o", "ConnectTimeout=15",
        "-o", "StrictHostKeyChecking=yes",
    ]
    key_path = os.environ.get("SSH_KEY_PATH")
    if key_path:
        command.extend(
            ["-i", str(Path(key_path).expanduser()), "-o", "IdentitiesOnly=yes"]
        )
    command.extend([f"{user}@{host}", remote_command])
    return command
