# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "smoke_fixture", ROOT / "skills/accelerated-computing-cugraph/scripts/smoke_fixture.py"
)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


class FixtureTests(unittest.TestCase):
    def test_partition_coverage_and_empty_ranks(self):
        for ranks in (1, 2, 4, 16):
            with self.subTest(ranks=ranks), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp) / "input"
                fixture.write_fixture(directory, ranks, 12)
                edges = []
                for rank in range(ranks):
                    with (directory / f"edges-{rank}.csv").open() as file:
                        for row in csv.DictReader(file):
                            src, dst = int(row["src"]), int(row["dst"])
                            self.assertEqual(src % ranks, rank)
                            edges.append((src, dst))
                self.assertEqual(sorted(edges), [(i, (i + 1) % 12) for i in range(12)])

    def write_results(self, directory, rows):
        with (directory / "part-0.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["vertex", "pagerank"])
            writer.writerows(rows)

    def test_exact_solution_and_reordered_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            self.write_results(directory, [(i, 1 / 12) for i in reversed(range(12))])
            fixture.check_results(directory, 12)

    def test_rejects_incomplete_duplicate_nonfinite_and_wrong_results(self):
        good = [(i, 1 / 12) for i in range(12)]
        cases = [good[:-1], good + [good[0]], [(0, float("nan"))] + good[1:],
                 [(0, 0.5)] + good[1:], []]
        for rows in cases:
            with self.subTest(rows=rows), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                self.write_results(directory, rows)
                with self.assertRaises(ValueError):
                    fixture.check_results(directory, 12)

    def test_does_not_overwrite_existing_fixture(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileExistsError):
                fixture.write_fixture(Path(temp), 2, 12)


if __name__ == "__main__":
    unittest.main()
