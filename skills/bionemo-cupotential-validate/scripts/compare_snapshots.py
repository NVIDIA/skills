#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare canonical snapshots. No budgets means diagnostics, never a passing test.

Usage:
    python /path/to/cupotential-validate/scripts/compare_snapshots.py candidate.json reference.json
Arguments:
    Candidate and reference JSON files with identical canonical geometry.
    Optional nonnegative budget flags select checks; see --help and
    references/protocols.md for the snapshot schema, metric definitions and units.
Output:
    JSON on stdout with metrics, budgets, check results and coverage limitations.
    CLI usage errors go to stderr; invalid snapshot input produces JSON on stdout.
Exit codes:
    0: diagnostics without budgets, or all requested checks passed.
    1: at least one requested check failed.
    2: invalid input/arguments, or a requested check could not be evaluated.
"""

import argparse
import json
import math
from pathlib import Path


def number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    try:
        value = float(value)
    except OverflowError as error:
        raise ValueError(f"{label} is outside the supported numeric range") from error
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def vectors(value, label, expected_count=None):
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a nonempty list of 3-vectors")
    if expected_count is not None and len(value) != expected_count:
        raise ValueError(f"{label} has an unexpected vector count")
    result = []
    for index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError(f"{label}[{index}] must have three components")
        result.append([number(v, f"{label}[{index}]") for v in row])
    return result


def validate_snapshot(payload):
    if not isinstance(payload, dict):
        raise ValueError("A snapshot must be a JSON object")
    required = {"positions_nm", "box_vectors_nm", "potential_energy_kj_mol",
                "forces_kj_mol_nm"}
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError("Missing snapshot fields: " + ", ".join(missing))
    positions = vectors(payload["positions_nm"], "positions_nm")
    return {
        "positions_nm": positions,
        "box_vectors_nm": vectors(payload["box_vectors_nm"], "box_vectors_nm", 3),
        "potential_energy_kj_mol": number(payload["potential_energy_kj_mol"], "energy"),
        "forces_kj_mol_nm": vectors(payload["forces_kj_mol_nm"], "forces", len(positions)),
    }


def compare(candidate, reference, budgets=None):
    candidate, reference = validate_snapshot(candidate), validate_snapshot(reference)
    for field in ("positions_nm", "box_vectors_nm"):
        if candidate[field] != reference[field]:
            raise ValueError(f"Mismatched {field}; use the same canonical geometry")
    budgets = {} if budgets is None else dict(budgets)
    allowed = {"energy_atol", "energy_rtol", "force_rms_atol", "force_max_atol",
               "force_relative_l2"}
    if set(budgets) - allowed:
        raise ValueError("Unknown budget name")
    for key, value in budgets.items():
        budgets[key] = number(value, key)
        if budgets[key] < 0:
            raise ValueError(f"{key} must be nonnegative")

    ref_forces = [x for row in reference["forces_kj_mol_nm"] for x in row]
    cand_forces = [x for row in candidate["forces_kj_mol_nm"] for x in row]
    differences = [a - b for a, b in zip(cand_forces, ref_forces)]
    energy_error = abs(candidate["potential_energy_kj_mol"] -
                       reference["potential_energy_kj_mol"])
    error_norm, reference_norm = math.hypot(*differences), math.hypot(*ref_forces)
    if not all(math.isfinite(v) for v in (energy_error, error_norm, reference_norm)):
        raise ValueError("Differences exceed the supported numeric range")
    relative = error_norm / reference_norm if reference_norm else None
    if relative is not None and not math.isfinite(relative):
        relative = None
    metrics = {
        "energy_absolute_kj_mol": energy_error,
        "force_rms_kj_mol_nm": error_norm / math.sqrt(len(candidate["positions_nm"])),
        "force_max_component_kj_mol_nm": max(map(abs, differences)),
        "force_relative_l2": relative,
        "reference_force_l2_kj_mol_nm": reference_norm,
    }
    checks = []

    def check(metric, limit):
        measured = metrics[metric]
        checks.append({"metric": metric, "measured": measured, "limit": limit,
                       "result": ("not_evaluable" if measured is None else
                                  "passed" if measured <= limit else "failed")})

    if "energy_atol" in budgets or "energy_rtol" in budgets:
        limit = budgets.get("energy_atol", 0) + budgets.get("energy_rtol", 0) * abs(
            reference["potential_energy_kj_mol"])
        if not math.isfinite(limit):
            raise ValueError("The combined energy budget exceeds the numeric range")
        check("energy_absolute_kj_mol", limit)
    for budget, metric in (("force_rms_atol", "force_rms_kj_mol_nm"),
                           ("force_max_atol", "force_max_component_kj_mol_nm"),
                           ("force_relative_l2", "force_relative_l2")):
        if budget in budgets:
            check(metric, budgets[budget])
    results = {item["result"] for item in checks}
    status = ("failed" if "failed" in results else
              "inconclusive" if "not_evaluable" in results else
              "passed" if checks else "diagnostic")
    report = {"schema_version": 1, "status": status,
              "particle_count": len(candidate["positions_nm"]),
              "metrics": metrics, "budgets": budgets, "checks": checks,
              "coverage": {
                  "energy_budget_checked": "energy_atol" in budgets or "energy_rtol" in budgets,
                  "force_budget_checked": any(key.startswith("force_") for key in budgets),
                  "model_equivalence_checked": False,
                  "reference_convergence_checked": False,
                  "gpu_execution_checked": False,
              }}
    if relative is None:
        report["relative_force_note"] = "Zero reference norm or unrepresentable normalization."
    return report


def nonnegative(value):
    try:
        result = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use a finite nonnegative number") from error
    if not math.isfinite(result) or result < 0:
        raise argparse.ArgumentTypeError("Use a finite nonnegative number")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("reference", type=Path)
    for flag in ("energy-atol", "energy-rtol", "force-rms-atol", "force-max-atol",
                 "force-relative-l2"):
        parser.add_argument("--" + flag, type=nonnegative,
                            help="Predeclared budget; see references/protocols.md for units")
    args = parser.parse_args()
    budgets = {key: value for key, value in vars(args).items()
               if key not in ("candidate", "reference") and value is not None}
    try:
        candidate = json.loads(args.candidate.read_text())
        reference = json.loads(args.reference.read_text())
        report = compare(candidate, reference, budgets)
    except (OSError, UnicodeError, ValueError) as error:
        report = {"schema_version": 1, "status": "invalid_input", "error": str(error)}
    print(json.dumps(report, indent=2, allow_nan=False))
    return {"diagnostic": 0, "passed": 0, "failed": 1,
            "inconclusive": 2, "invalid_input": 2}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
