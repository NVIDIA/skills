# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-flight validation for a Visual ChangeNet *classify* dataset CSV.

Runs on a bare host (stdlib only) BEFORE `visual_changenet train|evaluate|
inference`, so a malformed dataset fails in <1s with an actionable message
instead of minutes into a GPU container with a cryptic traceback (or, worse,
after a checkpoint is written and it looks like partial success).

Ported from the DEFT-AOI loop's `validate_training_csv.py` and extended with
the three failure modes reported against the standalone NemoClaw VCN workflow
(NVBugs 6470188):

  1. Absolute host paths in the CSV. `input_path` / `golden_path` are resolved
     RELATIVE to `images_dir`, which is mounted at the container's `/data`. An
     absolute host path (e.g. /home/user/...) does not exist inside the
     container, so every row fails file-not-found.
  2. Flat filenames where a directory-per-sample is required. TAO's siamese
     loader builds `<images_dir>/<input_path>/<object_name>_<light><ext>`, so
     `input_path` must name a DIRECTORY. A flat image filename
     (`input_001.jpg`) makes the loader look for
     `input_001.jpg/<object_name>_...` and die with a confusing
     FileNotFoundError.
  3. Single-class training sets. VCN classify computes
     `pf_ratio = num_pass / len(fail_indices)`; a training CSV with no
     non-PASS rows raises ZeroDivisionError at epoch 1.

Exit code 2 on any validation failure; 0 on success.

CLI:

    python scripts/validate_vcn_dataset.py \
        --csv        ~/tao-workspace/aoi/train.csv \
        --images-dir ~/tao-workspace/aoi/images \
        --mode       train        # train | evaluate | inference
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys

_REQUIRED_COLUMNS = ("input_path", "golden_path", "label", "object_name")
_PATH_COLUMNS = ("input_path", "golden_path")
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _looks_like_flat_file(rel: str) -> bool:
    """True when a path-column value is a flat image filename rather than the
    per-sample directory the siamese loader requires."""
    return pathlib.Path(rel).suffix.lower() in _IMAGE_SUFFIXES


def validate(
    csv_path: pathlib.Path,
    images_dir: pathlib.Path | None,
    mode: str = "train",
    light: str = "SolderLight",
    image_ext: str = ".jpg",
    batch_size: int | None = None,
    num_gpus: int = 1,
) -> list[str]:
    """Return a list of human-readable validation errors (empty == valid).

    Uses stdlib csv so the script runs on bare hosts without pandas.
    """
    errors: list[str] = []

    if not csv_path.is_file():
        return [f"CSV not found: {csv_path}"]

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        rows = list(reader)

    missing_cols = [c for c in _REQUIRED_COLUMNS if c not in columns]
    if missing_cols:
        errors.append(
            f"missing required column(s): {missing_cols}; got {list(columns)}"
        )
    if not rows:
        errors.append("CSV has 0 data rows")
    if missing_cols or not rows:
        return errors  # every check below needs the schema and rows

    # 1. Absolute paths — never valid; they don't exist at the container mount.
    abs_hits: list[tuple[int, str, str]] = []
    for col in _PATH_COLUMNS:
        for i, row in enumerate(rows):
            raw = (row.get(col) or "").strip()
            if raw and pathlib.Path(raw).is_absolute():
                abs_hits.append((i, col, raw))
    if abs_hits:
        sample = ", ".join(f"row {i} {c}={v!r}" for i, c, v in abs_hits[:5])
        errors.append(
            f"{len(abs_hits)} row(s) use ABSOLUTE paths in {list(_PATH_COLUMNS)}. "
            f"Paths must be RELATIVE to images_dir (mounted at /data inside the "
            f"container); absolute host paths do not exist there and every row "
            f"fails file-not-found. First: {sample}"
        )

    # 2. Flat filenames where a per-sample directory is required.
    flagged_pair = {(i, c) for i, c, _ in abs_hits}
    flat_hits: list[tuple[int, str, str]] = []
    for col in _PATH_COLUMNS:
        for i, row in enumerate(rows):
            if (i, col) in flagged_pair:
                continue
            raw = (row.get(col) or "").strip()
            if raw and _looks_like_flat_file(raw):
                flat_hits.append((i, col, raw))
    if flat_hits:
        flagged_pair |= {(i, c) for i, c, _ in flat_hits}
        sample = ", ".join(f"row {i} {c}={v!r}" for i, c, v in flat_hits[:5])
        errors.append(
            f"{len(flat_hits)} row(s) use a FLAT FILENAME in {list(_PATH_COLUMNS)}. "
            f"The siamese loader needs a DIRECTORY per sample and builds "
            f"<images_dir>/<path>/<object_name>_{light}{image_ext}. Restructure "
            f"so input_path/golden_path name a directory (e.g. sample_001/) that "
            f"holds <object_name>_{light}{image_ext}. First: {sample}"
        )

    # 3. Missing images on disk (siamese resolution) — only when images_dir is
    #    supplied and the (row, col) isn't already flagged as absolute/flat.
    if images_dir is not None:
        for col in _PATH_COLUMNS:
            missing: list[tuple[int, str]] = []
            for i, row in enumerate(rows):
                if (i, col) in flagged_pair:
                    continue
                raw = (row.get(col) or "").strip()
                obj = (row.get("object_name") or "").strip()
                if not raw:
                    missing.append((i, f"<empty {col}>"))
                    continue
                if not obj:
                    missing.append((i, "<empty object_name>"))
                    continue
                resolved = images_dir / raw / f"{obj}_{light}{image_ext}"
                if not resolved.is_file():
                    missing.append((i, f"{raw} -> {resolved}"))
            if missing:
                sample = ", ".join(f"row {i}: {p}" for i, p in missing[:5])
                errors.append(
                    f"{len(missing)} row(s) reference a missing {col} image "
                    f"(expected <images_dir>/<{col}>/<object_name>_{light}{image_ext}); "
                    f"first: {sample}"
                )

    # 4. Single-class training set -> ZeroDivisionError in the classify loader.
    if mode == "train":
        labels = [(row.get("label") or "").strip() for row in rows]
        labelled = [l for l in labels if l]
        if not labelled:
            errors.append(
                "training CSV has no labels; VCN classify train needs a 'label' "
                "column with 'PASS' and at least one defect class"
            )
        else:
            num_pass = sum(1 for l in labelled if l == "PASS")
            num_fail = len(labelled) - num_pass
            if num_pass == 0 or num_fail == 0:
                errors.append(
                    f"training set is SINGLE-CLASS (PASS={num_pass}, non-PASS="
                    f"{num_fail}). VCN classify computes "
                    f"pf_ratio = num_pass / len(fail_indices); an all-one-class "
                    f"set raises ZeroDivisionError at epoch 1 (after a checkpoint "
                    f"is written, so it looks like partial success). Provide both "
                    f"PASS and defect (NO_PASS/...) samples."
                )

    # 5. Batch size vs dataset size — the classify loader raises
    #    "Dataset size (N) is smaller than the total batch size" otherwise,
    #    after loading the checkpoint (~60s of GPU time wasted).
    if batch_size is not None:
        if batch_size <= 0 or num_gpus <= 0:
            errors.append(
                f"--batch-size and --num-gpus must be positive "
                f"(got batch_size={batch_size}, num_gpus={num_gpus})"
            )
        else:
            max_per_replica = len(rows) / num_gpus
            if batch_size > max_per_replica:
                suggested = int(max_per_replica) if num_gpus > 1 else len(rows)
                errors.append(
                    f"batch_size={batch_size} exceeds the dataset limit: {len(rows)} "
                    f"row(s) / {num_gpus} GPU(s) = {max_per_replica:g} per replica. "
                    f"The loader raises 'Dataset size ({len(rows)}) is smaller than "
                    f"the total batch size' at launch. Set dataset.classify.batch_size "
                    f"<= {max(1, suggested)} (or reduce num_gpus)."
                )

    return errors


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pre-flight a Visual ChangeNet classify dataset CSV: schema, "
            "relative-path and directory-per-sample layout, on-disk image "
            "existence, PASS-preserving label case, and (for train) a "
            "two-class distribution. Call this before the tao_run GPU launch."
        ),
    )
    parser.add_argument(
        "--csv", required=True, type=pathlib.Path,
        help="Path to the dataset CSV (input_path,golden_path,label,object_name).",
    )
    parser.add_argument(
        "--images-dir", required=False, default=None, type=pathlib.Path,
        help=(
            "images_dir the CSV paths resolve against. When supplied, every "
            "row's <input_path>/<object_name>_<light><ext> is checked on disk. "
            "Omit to run schema/path/label checks only."
        ),
    )
    parser.add_argument(
        "--mode", default="train", choices=("train", "evaluate", "inference"),
        help="Dataset role. 'train' additionally enforces a two-class label set.",
    )
    parser.add_argument(
        "--light", default="SolderLight",
        help="Lighting suffix for siamese resolution. Default: SolderLight.",
    )
    parser.add_argument(
        "--image-ext", default=".jpg",
        help="Image extension for siamese resolution. Default: .jpg.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help=(
            "When set, verify batch_size does not exceed the per-replica dataset "
            "size (rows / num_gpus); an oversized batch crashes the loader after "
            "the checkpoint loads."
        ),
    )
    parser.add_argument(
        "--num-gpus", type=int, default=1,
        help="GPU/shard count for the batch-size check. Default: 1.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    errors = validate(
        args.csv, args.images_dir, mode=args.mode,
        light=args.light, image_ext=args.image_ext,
        batch_size=args.batch_size, num_gpus=args.num_gpus,
    )
    if errors:
        print(
            f"validate_vcn_dataset: FATAL — {len(errors)} issue(s) in {args.csv}",
            file=sys.stderr,
        )
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 2
    print(f"validate_vcn_dataset: ok ({args.csv})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
