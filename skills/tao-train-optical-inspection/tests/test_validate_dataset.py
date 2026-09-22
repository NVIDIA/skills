#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import unittest


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = SKILL_ROOT / "scripts/validate_dataset.py"
FIXTURES = pathlib.Path(__file__).parent / "fixtures/dataset"
SCHEMA = SKILL_ROOT / "schemas/train.schema.json"


class ValidateDatasetTests(unittest.TestCase):
    def _run(
        self, csv_fixture: str, images_fixture: str = "valid", *extra: str
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--csv",
                str(FIXTURES / csv_fixture),
                "--images-dir",
                str(FIXTURES / images_fixture / "images"),
                *extra,
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_sample_dataset_passes(self) -> None:
        result = self._run("valid/dataset.csv")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("2 row(s), 4 lighting inputs, 16 readable file slots", result.stdout)

    def test_missing_columns_fail(self) -> None:
        result = self._run("missing_columns.csv")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing required column(s): ['golden_path']", result.stderr)

    def test_invalid_pass_label_fails_with_row_number(self) -> None:
        result = self._run("invalid_label.csv")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("row 2: invalid label 'pass'", result.stderr)
        self.assertIn("exact case-sensitive 'PASS'", result.stderr)

    def test_unresolvable_path_fails_with_resolved_directory(self) -> None:
        result = self._run("missing_path.csv")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("row 2: input_path 'does/not/exist'", result.stderr)
        self.assertIn("resolves to missing directory", result.stderr)

    def test_missing_lighting_input_names_file(self) -> None:
        result = self._run("missing_lighting/dataset.csv", "missing_lighting")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("row 2: input_path is missing lighting input(s)", result.stderr)
        self.assertIn("WhiteLight", result.stderr)
        self.assertIn("C42@1_WhiteLight.jpg", result.stderr)

    def test_grid_and_input_map_must_agree(self) -> None:
        result = self._run(
            "valid/dataset.csv", "valid", "--num-input", "3", "--concat-type", "grid"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("num_input=3 but input_map declares 4", result.stderr)
        self.assertIn("grid_map x*y must equal num_input", result.stderr)

    def test_schema_defaults_match_validator_contract(self) -> None:
        schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
        dataset = schema["properties"]["dataset"]
        defaults = dataset["default"]
        self.assertEqual(defaults["num_input"], 4)
        self.assertEqual(defaults["concat_type"], "linear")
        self.assertEqual(defaults["grid_map"], {"x": 2, "y": 2})
        self.assertEqual(
            list(defaults["input_map"].items()),
            [
                ("LowAngleLight", 0),
                ("SolderLight", 1),
                ("UniformLight", 2),
                ("WhiteLight", 3),
            ],
        )
        csv_description = dataset["properties"]["train_dataset"]["properties"][
            "csv_path"
        ]["description"]
        for column in ("input_path", "golden_path", "label", "object_name"):
            self.assertIn(column, csv_description)

    def test_single_class_training_set_fails(self) -> None:
        result = self._run("single_class.csv")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("training set is single-class (PASS=2, non-PASS=0)", result.stderr)
        self.assertIn("ZeroDivisionError", result.stderr)

    def test_single_class_allowed_outside_train_mode(self) -> None:
        result = self._run("single_class.csv", "valid", "--mode", "evaluate")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_batch_size_larger_than_dataset_fails(self) -> None:
        result = self._run("valid/dataset.csv", "valid", "--batch-size", "8")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("batch_size=8 exceeds the dataset limit", result.stderr)
        self.assertIn("Set dataset.batch_size <= 2", result.stderr)

    def test_batch_size_within_dataset_passes(self) -> None:
        result = self._run("valid/dataset.csv", "valid", "--batch-size", "2")
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
