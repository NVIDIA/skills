# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Speedup geomean and median, overall and per residue bin, into `speedup.json`.

Copy into `$WORKDIR/bench/` and run with `$BIOIR_PYTHON` from `$WORKDIR`, or pass
a role directory for a two-role bench (measurement.md, Speedup aggregates).
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _ok_map(path: Path) -> dict[str, tuple[int, float]]:
    data = json.loads(path.read_text())
    out: dict[str, tuple[int, float]] = {}
    for sid, row in data.get("samples", {}).items():
        if row.get("status") != "ok":
            continue
        if "residues" not in row or not row.get("forward_s"):
            continue
        out[sid] = (int(row["residues"]), float(row["forward_s"][0]))
    return out


def _bin_name(residues: int) -> str:
    if residues < 512:
        return "short"
    if residues <= 1024:
        return "medium"
    return "long"


def _stats(pairs: list[tuple[str, int, float]]) -> dict[str, Any]:
    n = len(pairs)
    ids = [p[0] for p in pairs]
    if n == 0:
        return {"n": 0, "geomean": None, "median": None, "ids": []}
    ratios = [p[2] for p in pairs]
    return {
        "n": n,
        "geomean": statistics.geometric_mean(ratios),
        "median": statistics.median(ratios),
        "ids": ids,
    }


def speedup_block(
    bioir: dict[str, tuple[int, float]],
    other: dict[str, tuple[int, float]],
) -> dict[str, Any]:
    rows: list[tuple[str, int, float]] = []
    for sid, (residues, bioir_s) in bioir.items():
        if sid not in other or bioir_s <= 0:
            continue
        oss_s = other[sid][1]
        if oss_s <= 0:
            continue
        rows.append((sid, residues, oss_s / bioir_s))
    rows.sort(key=lambda t: t[1])
    by_bin = {"short": [], "medium": [], "long": []}
    for sid, residues, ratio in rows:
        by_bin[_bin_name(residues)].append((sid, residues, ratio))
    return {
        "overall": _stats(rows),
        "short": {**_stats(by_bin["short"]), "rule": "residues < 512"},
        "medium": {
            **_stats(by_bin["medium"]),
            "rule": "512 <= residues <= 1024",
        },
        "long": {**_stats(by_bin["long"]), "rule": "residues > 1024"},
    }


def fmt(block: dict[str, Any], label: str) -> str:
    lines = [f"Speedup {label} (oss_s / bioir_s; >1 favors BioIR)"]
    for key, title in (
        ("overall", "overall"),
        ("short", "short < 512"),
        ("medium", "medium 512-1024"),
        ("long", "long > 1024"),
    ):
        item = block[key]
        n = item["n"]
        if n == 0:
            lines.append(f"- {title} (n=0): no speedup")
            continue
        extra = ""
        if key != "overall" and item["ids"]:
            extra = f"  [{', '.join(item['ids'])}]"
        lines.append(f"- {title} (n={n}): geomean {item['geomean']:.3f}, median {item['median']:.3f}{extra}")
    return "\n".join(lines)


def summarize(results: Path) -> dict[str, Any]:
    """Write `results/speedup.json` and return its payload."""
    bioir = _ok_map(results / "bioir_forward.json")
    payload: dict[str, Any] = {
        "vs_eager": speedup_block(bioir, _ok_map(results / "oss_eager.json")),
    }
    compile_path = results / "oss_compile.json"
    if compile_path.exists():
        payload["vs_compile"] = speedup_block(bioir, _ok_map(compile_path))
    (results / "speedup.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("results", nargs="?", type=Path, default=Path("results"), help="results or role directory")
    payload = summarize(parser.parse_args().results)
    print(fmt(payload["vs_eager"], "vs OSS eager"))
    if "vs_compile" in payload:
        print()
        print(fmt(payload["vs_compile"], "vs OSS compile"))


if __name__ == "__main__":
    main()
