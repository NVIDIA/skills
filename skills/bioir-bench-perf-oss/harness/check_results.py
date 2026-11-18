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
"""Check result JSONs against the artifact contract before Phase 6 reads them.

Copy into `$WORKDIR/bench/` and run on every result file of one comparison:

    python check_results.py results/bioir_forward.json results/oss_eager.json \
        results/oss_compile.json

Each file is checked alone (measurement.md, Result JSON), then the files are
checked against each other for the fields both sides must agree on. Exits 1 and
lists every problem; a harness that writes a failing file has a bug.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROW_STATUSES = frozenset({"ok", "oom", "recapture", "failed"})
DOCKQ_STATUSES = frozenset({"ok", "single_chain", "no_gt", "no_native_interfaces", "unsupported_chain_types", "failed"})
TEMPLATE_STATUSES = frozenset(
    {
        "none_declared",
        "attached",
        "capped_by_config",
        "synthesized_alignment",
        "attached_via_patch",
        "unsupported",
    }
)
TELEMETRY_FIELDS = ("power_draw_w", "clocks_sm_mhz", "clocks_graphics_mhz", "clocks_memory_mhz")
# Fields that must hold the same value on every side for a given sample.
SHARED_ROW_FIELDS = ("residues", "n_templates_supplied", "n_templates_attached", "dockq_scope")


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _is_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _check_unit_interval(problems: list[str], where: str, key: str, value: Any) -> None:
    if value is not None and not (_is_number(value) and 0.0 <= value <= 1.0):
        problems.append(f"{where}: {key} must be null or in [0, 1], got {value!r}")


def check_row(sid: str, row: dict[str, Any], *, compile_column: bool) -> list[str]:
    """Problems with one sample row."""
    where = f"samples[{sid}]"
    problems: list[str] = []
    status = row.get("status")
    if status not in ROW_STATUSES:
        problems.append(f"{where}: status {status!r} not in {sorted(ROW_STATUSES)}")
        return problems
    residues = row.get("residues")
    if not (isinstance(residues, int) and not isinstance(residues, bool) and residues > 0):
        problems.append(f"{where}: residues must be a positive int (sum over all chains), got {residues!r}")
    if status != "ok":
        return problems

    forward_s = row.get("forward_s")
    if not (isinstance(forward_s, list) and len(forward_s) == 1 and _is_number(forward_s[0]) and forward_s[0] > 0):
        problems.append(f"{where}: forward_s must be one positive time in seconds, got {forward_s!r}")
    for key in ("peak_alloc_gb", "peak_reserved_gb"):
        if not _is_number(row.get(key)):
            problems.append(f"{where}: {key} missing on an ok row")
    telemetry = row.get("gpu_telemetry")
    if not isinstance(telemetry, dict):
        problems.append(f"{where}: gpu_telemetry missing on an ok row")
    else:
        # A sampler that never polled measured nothing; `{}` is not telemetry.
        n_polls, n_busy = telemetry.get("n_polls"), telemetry.get("n_busy")
        if not (_is_count(n_polls) and n_polls > 0):
            problems.append(f"{where}: gpu_telemetry.n_polls must be a positive int, got {n_polls!r}")
        if not (_is_count(n_busy) and (not _is_count(n_polls) or n_busy <= n_polls)):
            problems.append(f"{where}: gpu_telemetry.n_busy must be in [0, n_polls], got {n_busy!r}")
    for key in ("unpaired_msas", "paired_msas", "templates"):
        if not isinstance(row.get(key), list):
            problems.append(f"{where}: {key} must be a list of resolved paths")

    template_status = row.get("template_status")
    if template_status not in TEMPLATE_STATUSES:
        problems.append(f"{where}: template_status {template_status!r} not in {sorted(TEMPLATE_STATUSES)}")
    supplied, attached = row.get("n_templates_supplied"), row.get("n_templates_attached")
    if not (_is_count(supplied) and _is_count(attached) and attached <= supplied):
        problems.append(
            f"{where}: need 0 <= n_templates_attached <= n_templates_supplied, got {attached!r}/{supplied!r}"
        )

    _check_unit_interval(problems, where, "lddt", row.get("lddt"))
    dockq_status = row.get("dockq_status")
    if dockq_status not in DOCKQ_STATUSES:
        problems.append(f"{where}: dockq_status {dockq_status!r} not in {sorted(DOCKQ_STATUSES)}")
    elif dockq_status == "ok":
        if not _is_number(row.get("dockq")):
            problems.append(f"{where}: dockq_status ok needs a numeric dockq")
        _check_unit_interval(problems, where, "dockq", row.get("dockq"))
    elif row.get("dockq") is not None:
        # A monomer has no interface to get wrong: its DockQ is null, never 0.0.
        problems.append(f"{where}: dockq must be null when dockq_status is {dockq_status!r}")

    if compile_column:
        for key in ("warmup_compile_delta", "measure_compile_delta"):
            if not _is_count(row.get(key)):
                problems.append(f"{where}: {key} missing on an oss_compile row")
        if row.get("measure_compile_delta") not in (0, None):
            problems.append(f"{where}: measured forward recompiled; status must be recapture, not ok")
    return problems


def check_file(data: dict[str, Any]) -> list[str]:
    """Problems with one result file."""
    problems: list[str] = []
    backend = data.get("backend")
    if not isinstance(backend, str) or not (backend.startswith("bioir_") or backend in {"oss_eager", "oss_compile"}):
        problems.append(f"backend must be bioir_*, oss_eager, or oss_compile, got {backend!r}")
    for key in ("model_source", "gpu", "config"):
        if key not in data:
            problems.append(f"{key} missing")
    if data.get("warmup") != 1 or data.get("repeats") != 1:
        problems.append(f"warmup and repeats must both be 1, got {data.get('warmup')!r}/{data.get('repeats')!r}")
    samples = data.get("samples")
    if not isinstance(samples, dict) or not samples:
        return [*problems, "samples must be a non-empty object keyed by sample id"]

    compile_column = backend == "oss_compile"
    for sid, row in samples.items():
        if not isinstance(row, dict):
            problems.append(f"samples[{sid}] must be an object, got {type(row).__name__}")
            continue
        problems.extend(check_row(sid, row, compile_column=compile_column))

    n_ok = sum(isinstance(row, dict) and row.get("status") == "ok" for row in samples.values())
    if n_ok:
        telemetry = data.get("gpu_telemetry")
        if not isinstance(telemetry, dict) or not all(f in telemetry for f in TELEMETRY_FIELDS):
            problems.append("file-level gpu_telemetry missing or incomplete")
        elif telemetry.get("n_ok") != n_ok:
            problems.append(f"file-level gpu_telemetry.n_ok must be {n_ok}, the ok rows, got {telemetry.get('n_ok')!r}")
    if compile_column:
        stats = data.get("compile_stats")
        if not isinstance(stats, dict) or stats.get("compile_dynamic") is not None:
            problems.append("oss_compile needs compile_stats with compile_dynamic null (dynamic=None)")
        elif stats.get("measurement_stable") is not True:
            problems.append("oss_compile is published only when compile_stats.measurement_stable is true")
    return problems


def check_pair(files: dict[str, dict[str, Any]]) -> list[str]:
    """Problems between result files that cover the same samples."""
    problems: list[str] = []
    # check_file reports a malformed `samples`; compare only the files that have one.
    samples_by_file = {name: data.get("samples") for name, data in files.items()}
    sample_sets = {name: set(samples) for name, samples in samples_by_file.items() if isinstance(samples, dict)}
    names = sorted(sample_sets)
    for name in names[1:]:
        if sample_sets[name] != sample_sets[names[0]]:
            only = sorted(sample_sets[name] ^ sample_sets[names[0]])
            problems.append(f"{name} and {names[0]} cover different samples: {only}")
    common = set.intersection(*sample_sets.values()) if sample_sets else set()
    for sid in sorted(common):
        rows = {name: samples_by_file[name][sid] for name in names}
        if not all(isinstance(row, dict) for row in rows.values()):
            continue
        for key in SHARED_ROW_FIELDS:
            values = {name: row.get(key) for name, row in rows.items()}
            if len({json.dumps(v) for v in values.values()}) > 1:
                problems.append(f"samples[{sid}].{key} differs across files: {values}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()
    files = {str(path): json.loads(path.read_text()) for path in args.files}
    problems = [f"{name}: {p}" for name, data in files.items() for p in check_file(data)]
    if len(files) > 1:
        problems.extend(check_pair(files))
    for problem in problems:
        print(problem)
    print(f"{len(problems)} problem(s) in {len(files)} file(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
