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
"""Latency and speedup against residue count, from the result JSONs only.

Copy into `$WORKDIR/bench/` and run with `$BIOIR_PYTHON` from `$WORKDIR`
(measurement.md, Charts). Writes `latency_vs_residues.png` and
`speedup_vs_residues.png` next to the result files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ANNOTATE_MAX_POINTS = 20


def load_ok(path: Path) -> list[tuple[str, int, float]]:
    """`(sample_id, residues, forward_s)` for every `ok` row, sorted by residues."""
    data = json.loads(path.read_text())
    rows = []
    for sid, row in data["samples"].items():
        if row.get("status") != "ok":
            continue
        if "residues" not in row or not row.get("forward_s"):
            continue
        rows.append((sid, int(row["residues"]), float(row["forward_s"][0])))
    rows.sort(key=lambda t: t[1])
    return rows


def speedup_points(
    bioir: list[tuple[str, int, float]], other: list[tuple[str, int, float]]
) -> list[tuple[str, int, float]]:
    """`(sample_id, residues, oss_s / bioir_s)` for samples `ok` on both sides."""
    bioir_s = {sid: s for sid, _, s in bioir}
    return [(sid, n, s / bioir_s[sid]) for sid, n, s in other if bioir_s.get(sid, 0) > 0]


def _plot(ax: Any, rows: list[tuple[str, int, float]], label: str) -> None:
    xs, ys, ids = [r[1] for r in rows], [r[2] for r in rows], [r[0] for r in rows]
    ax.plot(xs, ys, marker="o", label=label)
    if len(xs) <= ANNOTATE_MAX_POINTS:
        for x, y, sid in zip(xs, ys, ids, strict=True):
            ax.annotate(sid, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)


def plot(results: Path, config: Path) -> list[Path]:
    """Write both charts and return their paths.

    Args:
        results: Directory holding `bioir_forward.json`, `oss_eager.json`, and
            optionally `oss_compile.json`.
        config: `bench_config.json`, for the model key and GPU in the titles.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bioir = load_ok(results / "bioir_forward.json")
    eager = load_ok(results / "oss_eager.json")
    compile_path = results / "oss_compile.json"
    compiled = load_ok(compile_path) if compile_path.is_file() else []

    cfg = json.loads(config.read_text())
    title = f"{cfg['model_source']}  {cfg.get('gpu_name', '')}".strip()
    written = []

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for label, rows in (("BioIR", bioir), ("OSS eager", eager), ("OSS compile", compiled)):
        if rows:
            _plot(ax, rows, label)
    ax.set_xlabel("Residues")
    ax.set_ylabel("model.forward() (s)")
    ax.set_title(f"Forward latency vs residues — {title}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    written.append(results / "latency_vs_residues.png")
    fig.savefig(written[-1], dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for label, rows in (("vs eager", eager), ("vs compile", compiled)):
        points = speedup_points(bioir, rows)
        if points:
            _plot(ax, points, label)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Residues")
    ax.set_ylabel("OSS s / BioIR s  (>1 favors BioIR)")
    ax.set_title(f"Speedup vs residues — {title}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    written.append(results / "speedup_vs_residues.png")
    fig.savefig(written[-1], dpi=150)
    plt.close(fig)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("results", nargs="?", type=Path, default=Path("results"), help="results or role directory")
    parser.add_argument("--config", type=Path, default=Path("ref_data/bench_config.json"))
    args = parser.parse_args()
    for path in plot(args.results, args.config):
        print(path)


if __name__ == "__main__":
    main()
