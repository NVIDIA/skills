#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate a CSV input for a given KERMT agent workflow.

Mode-dispatched. Emits a single JSON object to stdout that the calling skill
parses to decide whether to proceed (`ok: true`) or surface a structured error
back to the user (`ok: false` with `errors[]`). The JSON shape is stable
across modes; only the contract for what counts as `ok` differs per mode.

Modes
-----
pretrain    Pretrain corpus CSV. Requires a `smiles` column. Other columns
            are ignored. Label columns are not required (and not expected).

finetune    Labeled CSV for a downstream task. Requires `smiles` plus
            >=1 numeric target column. Target columns are specified via
            `--targets <col1> <col2> ...`. If `--targets` is omitted, the
            validator auto-detects numeric non-smiles columns and reports
            them; the skill will then prompt the user to confirm or refine.

inference   CSV to run predictions on. Requires `smiles`. Target columns are
            not required (and not expected — predictions are written out).

embed       CSV to extract embeddings from. Requires `smiles` only.

SMILES validation
-----------------
By default the validator samples up to 20 SMILES (first 10 + last 10) and
checks each one parses with RDKit. Pass `--strict-rdkit` to parse every
SMILES (slow on large corpora). A SMILES is considered "invalid" if RDKit
returns `None` from `MolFromSmiles(smi, sanitize=True)` — empty / null
rows are counted separately.

Duplicate-SMILES detection is always full (cheap).

Output (stdout)
---------------
{
  "ok": true | false,
  "mode": str,
  "csv_path": str,
  "num_rows": int,
  "num_columns": int,
  "columns": [str, ...],
  "has_smiles_column": bool,
  "smiles_column_name": str | null,    // actual header used (may differ in case)
  "num_blank_smiles": int,
  "num_invalid_smiles": int,           // among the parsed sample
  "smiles_check_method": "sampled" | "full",
  "smiles_check_count": int,
  "num_duplicate_smiles": int,
  "target_columns": [str, ...],        // populated only for finetune mode
  "num_missing_per_target": { col: int, ... },
  "auto_detected_targets": [str, ...], // when --targets is omitted in finetune mode
  "errors": [str, ...],
  "warnings": [str, ...]
}

Exit code: 0 on `ok: true`, 1 on `ok: false`. Loader exceptions are caught
and surfaced into `errors[]` with `ok: false` (still exit 1).

CLI
---
    check_data.py --mode <mode> --csv <path>
                  [--targets <col1> <col2> ...]   # finetune only
                  [--strict-rdkit]                # full SMILES parse
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import pandas as pd


CANONICAL_SMILES_COLUMN = "smiles"
SMILES_SAMPLE_PER_END = 10  # how many SMILES from head + how many from tail to sample


def _find_smiles_column(columns: list[str]) -> str | None:
    """Return the actual column header matching 'smiles' case-insensitively, or None."""
    for c in columns:
        if c.lower() == CANONICAL_SMILES_COLUMN:
            return c
    return None


def _parse_smiles_sample(smiles_values: list[str], full: bool) -> tuple[int, int, str]:
    """Run RDKit MolFromSmiles on a sample or all of the SMILES. Returns
    (num_parsed, num_invalid, method)."""
    # Import here so the script can still surface a clean JSON error if RDKit
    # is unavailable in the host env.
    try:
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")  # suppress per-mol parse warnings
    except ImportError as exc:
        raise RuntimeError(
            f"RDKit is not importable in this environment: {exc}. "
            "Run check_data.py inside the kermt container."
        ) from exc

    if full or len(smiles_values) <= 2 * SMILES_SAMPLE_PER_END:
        sample = smiles_values
        method = "full"
    else:
        sample = smiles_values[:SMILES_SAMPLE_PER_END] + smiles_values[-SMILES_SAMPLE_PER_END:]
        method = "sampled"

    invalid = 0
    parsed = 0
    for smi in sample:
        if not smi:  # already counted as blank elsewhere
            continue
        parsed += 1
        mol = Chem.MolFromSmiles(smi, sanitize=True)
        if mol is None:
            invalid += 1
    return parsed, invalid, method


def _autodetect_target_columns(df: pd.DataFrame, smiles_col: str) -> list[str]:
    """Pick columns that look like numeric targets. A column qualifies if it
    is (a) not the smiles column and (b) >=80% of non-null values convert to float.
    Heuristic only — returned for the skill to prompt the user to confirm."""
    candidates: list[str] = []
    for col in df.columns:
        if col == smiles_col:
            continue
        ser = df[col].dropna()
        if len(ser) == 0:
            continue
        try:
            converted = pd.to_numeric(ser, errors="coerce")
        except (TypeError, ValueError):
            continue
        if converted.notna().sum() / max(len(ser), 1) >= 0.8:
            candidates.append(col)
    return candidates


def validate(mode: str, csv_path: str, targets: list[str] | None, strict_rdkit: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "ok": False,
        "mode": mode,
        "csv_path": csv_path,
        "num_rows": 0,
        "num_columns": 0,
        "columns": [],
        "has_smiles_column": False,
        "smiles_column_name": None,
        "num_blank_smiles": 0,
        "num_invalid_smiles": 0,
        "smiles_check_method": "sampled",
        "smiles_check_count": 0,
        "num_duplicate_smiles": 0,
        "target_columns": [],
        "num_missing_per_target": {},
        "auto_detected_targets": [],
        "errors": [],
        "warnings": [],
    }

    # 1. Read the CSV.
    path = Path(csv_path)
    if not path.is_file():
        result["errors"].append(f"CSV not found: {csv_path}")
        return result
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        result["errors"].append(f"CSV is empty (no header): {csv_path}")
        return result
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(f"failed to read CSV {csv_path}: {type(exc).__name__}: {exc}")
        return result

    result["num_rows"] = int(len(df))
    result["num_columns"] = int(len(df.columns))
    result["columns"] = [str(c) for c in df.columns]

    # 2. Locate the SMILES column.
    smiles_col = _find_smiles_column(result["columns"])
    if smiles_col is None:
        result["errors"].append(
            f"no column named 'smiles' (case-insensitive) found in CSV. "
            f"Available columns: {result['columns']}"
        )
        return result
    result["has_smiles_column"] = True
    result["smiles_column_name"] = smiles_col
    if smiles_col != CANONICAL_SMILES_COLUMN:
        result["warnings"].append(
            f"SMILES column is named '{smiles_col}' but downstream code expects '{CANONICAL_SMILES_COLUMN}' "
            f"(lowercase). Rename the column to '{CANONICAL_SMILES_COLUMN}' before running the workflow."
        )

    # 3. Blank-SMILES count + duplicate count + RDKit parse check.
    smi_series = df[smiles_col].astype(str).fillna("").str.strip()
    blank_mask = smi_series.eq("") | smi_series.str.lower().eq("nan")
    result["num_blank_smiles"] = int(blank_mask.sum())

    nonblank = smi_series[~blank_mask]
    result["num_duplicate_smiles"] = int(len(nonblank) - nonblank.nunique())

    if len(nonblank) == 0:
        result["errors"].append("no non-blank SMILES found in the CSV")
        return result

    try:
        parsed, invalid, method = _parse_smiles_sample(nonblank.tolist(), full=strict_rdkit)
    except RuntimeError as exc:
        result["errors"].append(str(exc))
        return result
    result["smiles_check_count"] = parsed
    result["num_invalid_smiles"] = invalid
    result["smiles_check_method"] = method

    if invalid > 0:
        scope = "all rows" if method == "full" else f"the {parsed} sampled rows"
        result["errors"].append(
            f"{invalid} out of {parsed} SMILES in {scope} failed to parse with RDKit. "
            "Either pre-clean the CSV with scripts/clean_smiles.py or pass --strict-rdkit to see "
            "the full count."
        )

    # 4. Target-column handling — finetune mode only.
    if mode == "finetune":
        if targets:
            missing = [t for t in targets if t not in df.columns]
            if missing:
                result["errors"].append(
                    f"target column(s) not found in CSV: {missing}. "
                    f"Available columns: {result['columns']}"
                )
            else:
                result["target_columns"] = list(targets)
                for t in targets:
                    nan_count = int(df[t].isna().sum())
                    result["num_missing_per_target"][t] = nan_count
                    # Confirm numeric-ish.
                    nonnan = df[t].dropna()
                    converted = pd.to_numeric(nonnan, errors="coerce")
                    non_numeric_count = int(converted.isna().sum())
                    if non_numeric_count > 0:
                        result["warnings"].append(
                            f"target column '{t}' has {non_numeric_count} non-numeric value(s) "
                            f"that will be dropped by the finetune runner."
                        )
        else:
            # Auto-detect — surface candidates so the skill can prompt the user.
            result["auto_detected_targets"] = _autodetect_target_columns(df, smiles_col)
            if not result["auto_detected_targets"]:
                result["errors"].append(
                    "no numeric non-smiles columns detected. finetune needs at least one target column; "
                    "specify it explicitly via --targets <col>."
                )
            else:
                result["warnings"].append(
                    f"--targets was not specified; auto-detected candidate target columns "
                    f"{result['auto_detected_targets']}. The skill will prompt the user to confirm."
                )

    # 5. Small-corpus warning — only for pretrain (other modes can be tiny by design).
    if mode == "pretrain" and result["num_rows"] < 100:
        result["warnings"].append(
            f"pretrain corpus is only {result['num_rows']} molecule(s). Pretraining typically "
            f"needs orders of magnitude more — verify this is the intended input."
        )

    result["ok"] = not result["errors"]
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a CSV input for a KERMT agent workflow.")
    parser.add_argument("--mode", required=True, choices=["pretrain", "finetune", "inference", "embed"])
    parser.add_argument("--csv", required=True, help="Path to the input CSV")
    parser.add_argument("--targets", nargs="+", default=None,
                        help="(finetune only) target column names. If omitted, the validator auto-detects "
                             "numeric non-smiles columns and reports them as candidates.")
    parser.add_argument("--strict-rdkit", action="store_true",
                        help="Parse every SMILES with RDKit rather than sampling (slow on large CSVs).")
    args = parser.parse_args(argv)

    try:
        result = validate(args.mode, args.csv, args.targets, args.strict_rdkit)
    except Exception as exc:  # noqa: BLE001
        print(traceback.format_exc(), file=sys.stderr)
        print(json.dumps({
            "ok": False,
            "mode": args.mode,
            "csv_path": args.csv,
            "errors": [f"unhandled exception in validator: {type(exc).__name__}: {exc}"],
            "warnings": [],
        }, indent=2))
        return 1

    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
