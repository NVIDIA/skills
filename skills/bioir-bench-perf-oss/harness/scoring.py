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
"""OpenStructure lDDT and DockQ on a written CIF, after the timed forward.

Copy into `$WORKDIR/bench/`. Both scorers run as subprocesses from their own
environments (`ost_cmd`, `dockq_cmd` in `bench_config.json`). Every function
returns the fields a result row stores, and a scorer that fails yields `None`
plus its stderr, never an invented number (measurement.md, Quality).

DockQ wants a scorer-only copy of an input when a writer omitted occupancy or
labelled a declared ligand `ATOM` (environment.md, Step 8). Make that copy
before calling `score_dockq` and pass its path; this module does not edit CIFs.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

# DockQ's own quality bands, lower bounds, best first.
DOCKQ_BANDS = ((0.80, "high"), (0.49, "medium"), (0.23, "acceptable"), (0.0, "incorrect"))

# DockQ exits 1 with this on a native that has no interface: the answer, not an error.
NO_NATIVE_INTERFACES = "Could not find interfaces in the native"

DOCKQ_UNSUPPORTED_TYPES = frozenset({"rna", "dna"})

_STDERR_TAIL = 2000


def _scorer_env() -> dict[str, str]:
    """Pass runtime settings without API keys or an OSS Python import path."""
    env = {name: os.environ[name] for name in ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL") if name in os.environ}
    env["PYTHONNOUSERSITE"] = "1"
    return env


def score_lddt(ost_cmd: str, model_cif: str | Path, native_cif: str | Path, out_json: str | Path) -> dict[str, Any]:
    """Global lDDT from `ost compare-structures`.

    Args:
        ost_cmd: Absolute path of `ost` in its dedicated environment.
        model_cif: Predicted structure.
        native_cif: Ground truth.
        out_json: Where `ost` writes its report; kept as the evidence.

    Returns:
        `lddt` and `lddt_json`, plus `lddt_stderr` when the scorer failed.
    """
    cmd = [
        ost_cmd,
        "compare-structures",
        "-m",
        str(model_cif),
        "-r",
        str(native_cif),
        "-o",
        str(out_json),
        "--lddt",
        "--fault-tolerant",
        "--min-pep-length",
        "4",
        "--min-nuc-length",
        "4",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, shell=False, env=_scorer_env())
    row: dict[str, Any] = {"lddt": None, "lddt_json": str(out_json)}
    if proc.returncode != 0:
        row["lddt_stderr"] = proc.stderr[-_STDERR_TAIL:]
        return row
    # A zero exit with an unreadable report, or one without a score, is a
    # failed score, not a crash and not a silent null.
    try:
        lddt = json.loads(Path(out_json).read_text())["lddt"]
        if lddt is None:
            raise ValueError("report has no lddt value")
        row["lddt"] = float(lddt)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        row["lddt"] = None
        row["lddt_stderr"] = f"{proc.stderr[-_STDERR_TAIL:]}\ninvalid lDDT report: {exc}"
    return row


def dockq_skip_status(polymer_types: Iterable[str], n_chains: int, has_gt: bool) -> str | None:
    """The `dockq_status` decided before invoking DockQ, or None when it should run.

    Args:
        polymer_types: `polymer_type` of every polymer in the sample.
        n_chains: Chains in the sample, ligands included.
        has_gt: Whether the sample has a ground-truth structure.
    """
    # RNA/DNA wins: measurement.md marks every such sample unsupported on both sides.
    if DOCKQ_UNSUPPORTED_TYPES & {t.lower() for t in polymer_types}:
        return "unsupported_chain_types"
    if not has_gt:
        return "no_gt"
    if n_chains < 2:
        return "single_chain"
    return None


def score_dockq(
    dockq_cmd: str,
    model_cif: str | Path,
    native_cif: str | Path | None,
    out_json: str | Path,
    args: Sequence[str],
    *,
    polymer_types: Iterable[str],
    n_chains: int,
) -> dict[str, Any]:
    """Interface DockQ, model first and native second.

    Args:
        dockq_cmd: Absolute path of `DockQ` in its dedicated environment.
        model_cif: Predicted structure, or its scorer-only copy.
        native_cif: Ground truth, or its scorer-only copy; None when the
            sample has none.
        out_json: Where DockQ writes `--json`; kept as the evidence.
        args: The locked `dockq_args`, identical on both sides.
        polymer_types: `polymer_type` of every polymer in the sample.
        n_chains: Chains in the sample, ligands included.

    Returns:
        The `dockq*` fields of a result row. `dockq` is `GlobalDockQ`, the mean
        over native interfaces, and is `None` whenever `dockq_status` is not `ok`.
    """
    scope = "all_interfaces" if "--small_molecule" in args else "protein_only"
    row: dict[str, Any] = {
        "dockq": None,
        "dockq_status": dockq_skip_status(polymer_types, n_chains, has_gt=native_cif is not None),
        "dockq_scope": scope,
        "dockq_mapping": None,
        "dockq_interfaces": None,
        "dockq_json": None,
    }
    if row["dockq_status"] == "unsupported_chain_types":
        row["dockq_scope"] = "unsupported"
    if row["dockq_status"] is not None:
        return row

    cmd = [dockq_cmd, str(model_cif), str(native_cif), "--json", str(out_json), *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, shell=False, env=_scorer_env())
    if proc.returncode != 0:
        output = proc.stdout + proc.stderr
        row["dockq_status"] = "no_native_interfaces" if NO_NATIVE_INTERFACES in output else "failed"
        row["dockq_stderr"] = output[-_STDERR_TAIL:]
        return row

    try:
        data = json.loads(Path(out_json).read_text())
        row.update(
            dockq=float(data["GlobalDockQ"]),
            dockq_status="ok",
            dockq_mapping=data.get("best_mapping_str"),
            dockq_interfaces=data.get("best_result"),
            dockq_json=str(out_json),
        )
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        row.update(dockq=None, dockq_status="failed", dockq_json=str(out_json))
        row["dockq_stderr"] = f"{proc.stdout + proc.stderr}\ninvalid DockQ report: {exc}"[-_STDERR_TAIL:]
    return row


def dockq_band(value: float) -> str:
    """DockQ quality band: `incorrect`, `acceptable`, `medium`, or `high`."""
    for lower, name in DOCKQ_BANDS:
        if value >= lower:
            return name
    return "incorrect"
