#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Read-only CSV preflight for public Encodon embedding extraction.

Usage: python validate_inputs.py sequences.csv [--context-length 2048]
Arguments: input CSV path; optional context length including CLS and SEP.
Output: JSON summary and per-row verdicts on stdout, or an error object.
Exit codes: 0 clean, 1 row findings to review, 2 file/schema/argument error.

Uses only the standard library. Reports input problems without claiming to run
the model or reproducing every pandas/tokenizer coercion. Never writes inputs.
"""

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path


REQUIRED_COLUMNS = {"id", "ref_seq", "value", "split"}
DEFAULT_CONTEXT_LENGTH = 2048
SPECIAL_TOKEN_COUNT = 2
CODON_WIDTH = 3


def validate_csv(path: Path, context_length: int = DEFAULT_CONTEXT_LENGTH) -> dict:
    """Describe split selection, sequence validity, ID ambiguity, and truncation."""
    if context_length <= SPECIAL_TOKEN_COUNT:
        raise ValueError("context length must allow CLS, at least one codon, and SEP")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        columns = reader.fieldnames or []
        missing = sorted(REQUIRED_COLUMNS - set(columns))
        if missing:
            raise ValueError(f"missing required columns for eval: {', '.join(missing)}")
        if len(set(columns)) != len(columns):
            raise ValueError("duplicate CSV column names")
        rows = list(reader)
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError("CSV row width does not match header")

    ids = Counter(row["id"] for row in rows)
    limit = context_length - SPECIAL_TOKEN_COUNT
    verdicts = []
    for number, row in enumerate(rows, start=1):
        selected = row["split"] == "test"
        issues = []
        if not selected:
            issues.append("excluded: split must be exactly 'test' to enter extraction")
        if not row["id"].strip():
            issues.append("blank id: assign a unique nonblank ID")
        elif ids[row["id"]] > 1:
            issues.append("duplicate id: extraction does not reject it, but output-to-source joins are ambiguous; assign unique IDs")

        sequence = row["ref_seq"]
        dna = sequence.replace("U", "T")
        sequence_valid = bool(dna) and set(dna) <= set("ACGT") and len(dna) % CODON_WIDTH == 0
        if not sequence_valid:
            issues.append("invalid sequence for this preflight: require nonempty uppercase A/C/G/T (or U) and a length divisible by three; correct a copy")
        try:
            value_valid = math.isfinite(float(row["value"]))
        except ValueError:
            value_valid = False
        if not value_valid:
            issues.append("invalid value: supply a finite numeric label or 0.0 for extraction-only data")

        codons = len(dna) // CODON_WIDTH if sequence_valid else None
        retained = min(codons, limit) if selected and sequence_valid else None
        lost = codons - retained if retained is not None else None
        if lost:
            issues.append(f"truncated: first {retained} codons retained; last {lost} codons ({CODON_WIDTH * lost} nucleotides) lost")
        if not selected:
            verdict = "excluded"
        elif not (sequence_valid and value_valid):
            verdict = "needs_correction"
        elif lost:
            verdict = "truncated"
        else:
            verdict = "processes"
        verdicts.append({
            "row": number,
            "id": row["id"],
            "split": row["split"],
            "selected_for_test": selected,
            "nucleotides": len(sequence),
            "codons": codons,
            "value": row["value"],
            "sequence_valid": sequence_valid,
            "value_valid": value_valid,
            "retained_codons": retained,
            "lost_codons": lost,
            "verdict": verdict,
            "issues": issues,
        })
    selected_rows = sum(item["selected_for_test"] for item in verdicts)
    return {
        "csv": str(path),
        "context_length": context_length,
        "codon_limit": limit,
        "total_rows": len(rows),
        "test_rows": selected_rows,
        "excluded_rows": len(rows) - selected_rows,
        "warnings": [] if selected_rows else ["no rows have split='test'; no embeddings can be produced"],
        "rows": verdicts,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="input CSV; never modified")
    parser.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH)
    args = parser.parse_args(argv)
    try:
        report = validate_csv(args.csv, args.context_length)
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        print(json.dumps({"error": str(error), "csv": str(args.csv)}))
        return 2
    print(json.dumps(report, indent=2))
    return int(bool(report["warnings"]) or any(row["issues"] for row in report["rows"]))


if __name__ == "__main__":
    raise SystemExit(main())
