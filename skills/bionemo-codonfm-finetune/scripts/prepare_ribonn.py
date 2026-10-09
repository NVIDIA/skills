# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare public human RiboNN TE data for CodonBertDataset, without ML deps."""

import argparse
import csv
import http.client
import io
import json
import math
import time
from collections import Counter
from contextlib import closing
from pathlib import Path


DATA_REVISION = "512fca642b6b7b61ae494ad83cfc2b72831636d2"
DATA_HOST = "raw.githubusercontent.com"
DATA_PATH = (
    f"/CenikLab/TE_classic_ML/{DATA_REVISION}"
    "/data/data_with_human_TE_cellline_all_NA_plain.csv"
)
DATA_URL = f"https://{DATA_HOST}{DATA_PATH}"


def prepare(handle, max_rows_per_split=8, max_codons=2046, deadline=None):
    """Slice CDSs and keep source folds: 0-7 train, 8 val, 9 test.

    The upstream file is TSV despite its .csv suffix. A capped subset is a
    preparation/smoke example, not the notebook's full cross-validation study.
    """
    if max_rows_per_split < 0 or max_codons < 1:
        raise ValueError("Row cap must be non-negative and max_codons must be positive")
    reader = csv.DictReader(handle, delimiter="\t")
    required = {"transcript_id", "tx_sequence", "utr5_size", "cds_size", "mean_te", "fold"}
    if not required <= set(reader.fieldnames or []):
        raise ValueError("Expected RiboNN TSV columns: " + ", ".join(sorted(required)))
    rows, seen_ids, sequence_splits = [], set(), {}
    counts, skipped = Counter(), Counter()
    examined = 0
    for raw in reader:
        examined += 1
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError("Dataset preparation exceeded the network time budget; use --input for an offline file")
        try:
            start, length, fold = int(raw["utr5_size"]), int(raw["cds_size"]), int(raw["fold"])
            value = float(raw["mean_te"])
            transcript = raw["tx_sequence"].upper().replace("U", "T")
            row_id = raw["transcript_id"].strip()
        except (TypeError, ValueError, AttributeError):
            skipped["invalid_metadata"] += 1
            continue
        if not row_id or start < 0 or length <= 0 or start + length > len(transcript) or length % 3:
            skipped["invalid_cds_bounds_or_id"] += 1
            continue
        sequence = transcript[start:start + length]
        if set(sequence) - set("ACGT") or not math.isfinite(value) or fold not in range(10):
            skipped["invalid_sequence_label_or_fold"] += 1
            continue
        if length // 3 > max_codons:
            skipped["cds_exceeds_context"] += 1
            continue
        split = "val" if fold == 8 else "test" if fold == 9 else "train"
        if row_id in seen_ids:
            skipped["duplicate_transcript"] += 1
            continue
        if sequence in sequence_splits and sequence_splits[sequence] != split:
            raise ValueError("Identical CDS appears in different source folds; resolve split leakage before training")
        seen_ids.add(row_id)
        sequence_splits[sequence] = split
        if max_rows_per_split and counts[split] >= max_rows_per_split:
            continue
        rows.append({"id": row_id, "ref_seq": sequence, "value": value, "split": split})
        counts[split] += 1
        if max_rows_per_split and all(counts[s] >= max_rows_per_split for s in ("train", "val", "test")):
            break
    if not all(counts[s] for s in ("train", "val", "test")):
        raise ValueError("Prepared data must have non-empty train, val, and test splits; check source folds 0-9")
    return rows, {
        "rows_examined": examined, "rows_written": len(rows), "split_counts": dict(counts),
        "skipped": dict(skipped), "label": "mean_te (unchanged)",
        "fold_mapping": {"train": list(range(8)), "val": [8], "test": [9]},
        "max_rows_per_split": max_rows_per_split, "max_codons": max_codons,
        "purpose": "input preparation; a capped subset is not a scientific benchmark",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Existing upstream-format TSV; omit to stream the pinned public data")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rows-per-split", type=int, default=8, help="Default 8 for a small example; 0 processes all rows")
    parser.add_argument("--max-codons", type=int, default=2046)
    args = parser.parse_args()
    if args.input and args.input.resolve() == args.output.resolve():
        parser.error("Input and output must be different files")
    try:
        if args.input:
            with args.input.open(encoding="utf-8-sig", newline="") as handle:
                rows, report = prepare(handle, args.max_rows_per_split, args.max_codons)
        else:
            # Fixed HTTPS endpoint with default certificate verification; no redirects or retries.
            deadline = time.monotonic() + 60
            with closing(http.client.HTTPSConnection(DATA_HOST, timeout=10)) as connection:
                connection.request("GET", DATA_PATH)
                with connection.getresponse() as response:
                    if response.status != 200:
                        raise ValueError(f"Dataset download returned HTTP {response.status}; expected 200 without redirects")
                    with io.TextIOWrapper(response, encoding="utf-8-sig", newline="") as handle:
                        rows, report = prepare(handle, args.max_rows_per_split, args.max_codons, deadline)
        report.update({"source": str(args.input) if args.input else DATA_URL, "upstream_url": DATA_URL})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "ref_seq", "value", "split"])
            writer.writeheader()
            writer.writerows(rows)
        args.output.with_suffix(".metadata.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except (ValueError, OSError, http.client.HTTPException) as exc:
        parser.exit(1, f"Preparation failed: {exc}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
