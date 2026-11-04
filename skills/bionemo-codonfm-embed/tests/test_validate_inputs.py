# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Exercise observable CSV verdicts and the read-only command interface."""

import csv
from contextlib import redirect_stdout, redirect_stderr
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest


SKILL = Path(__file__).resolve().parents[1]
SCRIPT = SKILL / "scripts" / "validate_inputs.py"
spec = importlib.util.spec_from_file_location("validate_inputs", SCRIPT)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class ValidateInputsTest(unittest.TestCase):
    def write_csv(self, directory, rows, columns=("id", "ref_seq", "value", "split")):
        path = Path(directory) / "input.csv"
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerows(rows)
        return path

    def test_supplied_clean_rows_and_no_writes(self):
        path = SKILL / "evals" / "files" / "sequences.csv"
        before = path.read_bytes()
        report = checker.validate_csv(path)
        self.assertEqual(report["test_rows"], 2)
        self.assertEqual([row["verdict"] for row in report["rows"]], ["processes"] * 2)
        self.assertTrue(all(not row["issues"] for row in report["rows"]))
        self.assertEqual(path.read_bytes(), before)

    def test_supplied_edge_cases(self):
        path = SKILL / "evals" / "files" / "sequences_edge_cases.csv"
        before = path.read_bytes()
        report = checker.validate_csv(path)
        self.assertEqual(report["test_rows"], 3)
        self.assertEqual(report["excluded_rows"], 1)
        rows = report["rows"]
        self.assertEqual([row["verdict"] for row in rows], ["processes", "excluded", "processes", "truncated"])
        self.assertEqual([row["row"] for row in rows], [1, 2, 3, 4])
        self.assertTrue(any("duplicate id" in issue for issue in rows[0]["issues"]))
        self.assertTrue(any("duplicate id" in issue for issue in rows[2]["issues"]))
        self.assertEqual((rows[3]["codons"], rows[3]["retained_codons"], rows[3]["lost_codons"]), (2050, 2046, 4))
        self.assertEqual(path.read_bytes(), before)

    def test_configurable_context_boundary_and_rna(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv(directory, [
                ["boundary", "AUG" * 6, "0.0", "test"],
                ["over", "ATG" * 7, "0.0", "test"],
            ])
            rows = checker.validate_csv(path, context_length=8)["rows"]
        self.assertEqual(rows[0]["verdict"], "processes")
        self.assertEqual(rows[0]["lost_codons"], 0)
        self.assertEqual(rows[1]["lost_codons"], 1)

    def test_strict_split_filter_including_invalid_excluded_sequence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv(directory, [
                [str(n), "bad-sequence", "not-a-number", split]
                for n, split in enumerate(["", "train", "Test", "test "])
            ])
            report = checker.validate_csv(path)
        self.assertEqual(report["test_rows"], 0)
        self.assertTrue(report["warnings"])
        self.assertTrue(all(row["verdict"] == "excluded" for row in report["rows"]))
        self.assertTrue(all(row["retained_codons"] is None for row in report["rows"]))

    def test_invalid_sequences_and_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv(directory, [
                ["empty", "", "0", "test"],
                ["partial", "ATGA", "0", "test"],
                ["lower", "atg", "0", "test"],
                ["ambiguous", "ANN", "0", "test"],
                ["label", "ATG", "NaN", "test"],
            ])
            report = checker.validate_csv(path)
        self.assertTrue(all(row["verdict"] == "needs_correction" for row in report["rows"]))

    def test_missing_split_and_bad_row_width(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv(directory, [["a", "ATG", "0"]], columns=("id", "ref_seq", "value"))
            with self.assertRaisesRegex(ValueError, "split"):
                checker.validate_csv(path)
            path = self.write_csv(directory, [["a", "ATG", "0", "test", "extra"]])
            with self.assertRaisesRegex(ValueError, "width"):
                checker.validate_csv(path)

    def test_empty_dataset_and_invalid_context(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv(directory, [])
            report = checker.validate_csv(path)
            self.assertEqual(report["rows"], [])
            self.assertTrue(report["warnings"])
            with self.assertRaisesRegex(ValueError, "context length"):
                checker.validate_csv(path, 2)

    def test_cli_exit_codes_and_json(self):
        for name, expected in [("sequences.csv", 0), ("sequences_edge_cases.csv", 1), ("missing.csv", 2)]:
            with self.subTest(name=name):
                stdout, stderr = io.StringIO(), io.StringIO()
                argv = [str(SKILL / "evals" / "files" / name)]
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    code = checker.main(argv)
                self.assertEqual(code, expected)
                report = json.loads(stdout.getvalue())
                self.assertEqual("error" in report, expected == 2)
                self.assertFalse(stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
