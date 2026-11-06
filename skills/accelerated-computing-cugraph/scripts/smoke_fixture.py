# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Create a cross-partition directed cycle or check its PageRank CSV output."""
import argparse
import csv
import math
from pathlib import Path


def write_fixture(directory, ranks, vertices):
    if ranks < 1 or vertices < 3:
        raise ValueError("Require ranks >= 1 and vertices >= 3")
    directory.mkdir(parents=True, exist_ok=False)
    for rank in range(ranks):
        with (directory / f"edges-{rank}.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["src", "dst", "weight"])
            for vertex in range(rank, vertices, ranks):
                writer.writerow([vertex, (vertex + 1) % vertices, 1.0])


def check_results(directory, vertices):
    scores = {}
    for path in sorted(directory.glob("*.csv")):
        with path.open(newline="") as file:
            for row in csv.DictReader(file):
                vertex, score = int(row["vertex"]), float(row["pagerank"])
                if vertex in scores:
                    raise ValueError(f"Duplicate output vertex {vertex}")
                if not math.isfinite(score):
                    raise ValueError(f"Nonfinite score for vertex {vertex}")
                scores[vertex] = score
    if set(scores) != set(range(vertices)):
        raise ValueError("Missing or unexpected output vertices")
    if any(not math.isclose(s, 1 / vertices, rel_tol=1e-4, abs_tol=1e-6)
           for s in scores.values()):
        raise ValueError("Cycle PageRank does not match the exact uniform solution")
    if not math.isclose(sum(scores.values()), 1.0, abs_tol=1e-5):
        raise ValueError("PageRank mass does not sum to one")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["create", "check"])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--ranks", type=int, default=2)
    parser.add_argument("--vertices", type=int, default=12)
    args = parser.parse_args()
    if args.vertices < 3:
        parser.error("--vertices must be at least 3")
    if args.mode == "create":
        write_fixture(args.directory, args.ranks, args.vertices)
    else:
        check_results(args.directory, args.vertices)
        print("PASS: all vertices, unique IDs, finite uniform scores, total mass = 1")


if __name__ == "__main__":
    main()
