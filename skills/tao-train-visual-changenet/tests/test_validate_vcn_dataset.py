#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the Visual ChangeNet dataset validator.

Fixtures are generated in a temp directory at test time: the validator only
checks file existence, so empty placeholder files stand in for images and no
binaries need to live in the repository.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile
import unittest


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts/validate_vcn_dataset.py"
LIGHTS = ("SolderLight", "WhiteLight")


class ValidateVcnDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = pathlib.Path(self._tmp.name)
        self.images_dir = self.root / "images"
        for sample, obj in (("sample_001", "R821@1"), ("golden/boardBOT", "R821@1")):
            directory = self.images_dir / sample
            directory.mkdir(parents=True)
            for light in LIGHTS:
                (directory / f"{obj}_{light}.jpg").touch()

    def _write_csv(self, *rows: str) -> pathlib.Path:
        csv_path = self.root / "dataset.csv"
        header = "input_path,golden_path,label,object_name"
        csv_path.write_text("\n".join((header, *rows)) + "\n")
        return csv_path

    def _run(self, csv_path: pathlib.Path, *extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--csv",
                str(csv_path),
                "--images-dir",
                str(self.images_dir),
                *extra,
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_multi_light_dataset_passes_with_repeated_light(self) -> None:
        csv_path = self._write_csv(
            "sample_001,golden/boardBOT,PASS,R821@1",
            "sample_001,golden/boardBOT,missing,R821@1",
        )
        result = self._run(
            csv_path, "--light", "SolderLight", "--light", "WhiteLight"
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_second_light_is_flagged(self) -> None:
        for path in self.images_dir.rglob("*_WhiteLight.jpg"):
            path.unlink()
        csv_path = self._write_csv(
            "sample_001,golden/boardBOT,PASS,R821@1",
            "sample_001,golden/boardBOT,missing,R821@1",
        )
        result = self._run(
            csv_path, "--light", "SolderLight", "--light", "WhiteLight"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("_WhiteLight.jpg", result.stderr)
        single = self._run(csv_path, "--light", "SolderLight")
        self.assertEqual(single.returncode, 0, single.stderr)

    def test_lowercase_pass_label_is_flagged(self) -> None:
        csv_path = self._write_csv(
            "sample_001,golden/boardBOT,pass,R821@1",
            "sample_001,golden/boardBOT,missing,R821@1",
        )
        result = self._run(csv_path, "--light", "SolderLight")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid label 'pass'", result.stderr)
        self.assertIn("exact case-sensitive 'PASS'", result.stderr)

    def test_label_whitespace_is_flagged(self) -> None:
        csv_path = self._write_csv(
            "sample_001,golden/boardBOT, PASS,R821@1",
            "sample_001,golden/boardBOT,missing,R821@1",
        )
        result = self._run(csv_path, "--light", "SolderLight")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("surrounding whitespace", result.stderr)

    def test_default_light_remains_solderlight(self) -> None:
        csv_path = self._write_csv(
            "sample_001,golden/boardBOT,PASS,R821@1",
            "sample_001,golden/boardBOT,missing,R821@1",
        )
        result = self._run(csv_path)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
