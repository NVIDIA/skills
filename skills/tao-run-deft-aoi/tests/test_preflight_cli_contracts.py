#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pathlib
import re
import subprocess
import sys
import unittest


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
PREFLIGHT = SKILL_ROOT / "references/preflight.md"
VALIDATOR = SKILL_ROOT / "scripts/validate_training_csv.py"
FLAG_PATTERN = re.compile(r"--[a-z][a-z0-9-]*")


def cli_flags(script: pathlib.Path) -> set[str]:
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    return set(FLAG_PATTERN.findall(result.stdout))


class PreflightCliContractTests(unittest.TestCase):
    def test_training_csv_command_supplies_required_flags(self) -> None:
        text = PREFLIGHT.read_text(encoding="utf-8")
        blocks = [
            block
            for block in re.findall(r"```bash\n(.*?)```", text, re.DOTALL)
            if "validate_training_csv.py" in block
        ]

        self.assertEqual(len(blocks), 1)
        documented = set(FLAG_PATTERN.findall(blocks[0]))
        self.assertEqual(documented, {"--csv", "--workspace-root"})
        self.assertLessEqual(documented, cli_flags(VALIDATOR))


if __name__ == "__main__":
    unittest.main()
