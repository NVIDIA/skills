#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Validate a TAO Optical Inspection dataset CSV before GPU launch."""

from __future__ import annotations

import argparse
import csv
import os
import pathlib
import sys


REQUIRED_COLUMNS = ("input_path", "golden_path", "label", "object_name")
DEFAULT_INPUT_MAP = (
    ("LowAngleLight", 0),
    ("SolderLight", 1),
    ("UniformLight", 2),
    ("WhiteLight", 3),
)


def _parse_input_map(values: list[str] | None) -> list[tuple[str, int]]:
    if values is None:
        return list(DEFAULT_INPUT_MAP)
    parsed: list[tuple[str, int]] = []
    for value in values:
        light, separator, raw_index = value.partition("=")
        if not separator or not light or not raw_index:
            raise ValueError(
                f"invalid --input-map {value!r}; expected LIGHT=INDEX, "
                "for example SolderLight=1"
            )
        try:
            index = int(raw_index)
        except ValueError as exc:
            raise ValueError(
                f"invalid --input-map {value!r}; INDEX must be an integer"
            ) from exc
        parsed.append((light, index))
    return parsed


def _config_errors(
    input_map: list[tuple[str, int]],
    num_input: int,
    concat_type: str,
    grid_x: int,
    grid_y: int,
) -> list[str]:
    errors: list[str] = []
    lights = [light for light, _ in input_map]
    indices = [index for _, index in input_map]
    if num_input < 1:
        errors.append(f"num_input must be positive; got {num_input}")
    if len(lights) != len(set(lights)):
        errors.append(f"input_map has duplicate lighting names: {lights}")
    if len(input_map) != num_input:
        errors.append(
            f"num_input={num_input} but input_map declares {len(input_map)} "
            f"lighting inputs: {lights}"
        )
    expected_indices = list(range(len(input_map)))
    if indices != expected_indices:
        errors.append(
            "input_map values must be contiguous zero-based positions in YAML "
            f"key order; expected {expected_indices}, got {indices}"
        )
    if concat_type == "grid":
        if grid_x < 1 or grid_y < 1:
            errors.append(
                f"grid_map x and y must be positive; got x={grid_x}, y={grid_y}"
            )
        if grid_x * grid_y != num_input:
            errors.append(
                f"grid_map x*y must equal num_input; {grid_x}*{grid_y} != {num_input}"
            )
        if num_input % 2:
            errors.append(
                "grid concatenation requires an even num_input; the loader "
                f"would silently use linear concatenation for {num_input} inputs"
            )
    return errors


def _resolve_directory(images_dir: pathlib.Path, raw_path: str) -> pathlib.Path:
    path = pathlib.Path(raw_path)
    return path if path.is_absolute() else images_dir / path


def validate(
    csv_path: pathlib.Path,
    images_dir: pathlib.Path,
    *,
    input_map: list[tuple[str, int]],
    num_input: int,
    concat_type: str,
    grid_x: int,
    grid_y: int,
    image_ext: str,
    mode: str = "train",
    batch_size: int | None = None,
    num_gpus: int = 1,
) -> tuple[list[str], int, int]:
    """Return validation errors, data-row count, and readable file-slot count."""
    errors = _config_errors(input_map, num_input, concat_type, grid_x, grid_y)
    if not image_ext.startswith(".") or len(image_ext) == 1:
        errors.append(f"image_ext must start with '.' and name a suffix; got {image_ext!r}")
    if not csv_path.is_file():
        errors.append(f"CSV not found: {csv_path}")
        return errors, 0, 0
    if not images_dir.is_dir():
        errors.append(f"images_dir not found or not a directory: {images_dir}")
        return errors, 0, 0

    try:
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames or []
            rows = list(reader)
    except (OSError, csv.Error, UnicodeError) as exc:
        errors.append(f"cannot read CSV {csv_path}: {exc}")
        return errors, 0, 0

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in columns]
    if missing_columns:
        errors.append(
            f"missing required column(s): {missing_columns}; got columns {columns}"
        )
    if not rows:
        errors.append("CSV has zero data rows")
    if missing_columns or not rows:
        return errors, len(rows), 0

    lights = [light for light, _ in input_map]
    readable_files = 0
    for row_number, row in enumerate(rows, start=2):
        object_name = row.get("object_name") or ""
        label = row.get("label") or ""
        row_can_resolve = True
        if not object_name:
            errors.append(f"row {row_number}: object_name is empty")
            row_can_resolve = False
        elif (
            pathlib.PurePath(object_name).name != object_name
            or object_name.endswith(image_ext)
            or any(object_name.endswith(f"_{light}") for light in lights)
        ):
            errors.append(
                f"row {row_number}: object_name {object_name!r} must be a filename "
                "stem without directories, lighting suffix, or extension"
            )
            row_can_resolve = False

        if not label:
            errors.append(
                f"row {row_number}: label is empty; use exact 'PASS' or a "
                "non-empty defect name"
            )
        elif label != label.strip():
            errors.append(f"row {row_number}: label {label!r} has surrounding whitespace")
        elif label.casefold() == "pass" and label != "PASS":
            errors.append(
                f"row {row_number}: invalid label {label!r}; non-defective rows "
                "must use exact case-sensitive 'PASS'"
            )

        for column in ("input_path", "golden_path"):
            raw_path = row.get(column) or ""
            if not raw_path:
                errors.append(f"row {row_number}: {column} is empty")
                continue
            directory = _resolve_directory(images_dir, raw_path)
            if not directory.is_dir():
                errors.append(
                    f"row {row_number}: {column} {raw_path!r} resolves to missing "
                    f"directory {directory}"
                )
                continue
            if not row_can_resolve:
                continue
            missing_lights: list[str] = []
            unreadable: list[pathlib.Path] = []
            for light in lights:
                image = directory / f"{object_name}_{light}{image_ext}"
                if not image.is_file():
                    missing_lights.append(f"{light} ({image})")
                elif not os.access(image, os.R_OK):
                    unreadable.append(image)
                else:
                    readable_files += 1
            if missing_lights:
                errors.append(
                    f"row {row_number}: {column} is missing lighting input(s): "
                    + ", ".join(missing_lights)
                )
            if unreadable:
                errors.append(
                    f"row {row_number}: {column} has unreadable image(s): "
                    + ", ".join(str(path) for path in unreadable)
                )
    if mode == "train":
        labels = [(row.get("label") or "").strip() for row in rows]
        labelled = [label for label in labels if label]
        num_pass = sum(1 for label in labelled if label == "PASS")
        num_fail = len(labelled) - num_pass
        if labelled and (num_pass == 0 or num_fail == 0):
            errors.append(
                f"training set is single-class (PASS={num_pass}, "
                f"non-PASS={num_fail}); the classify loader computes "
                "pf_ratio = num_pass / len(fail_indices) and raises "
                "ZeroDivisionError at epoch 1. Provide both PASS and defect rows."
            )

    if batch_size is not None:
        if batch_size <= 0 or num_gpus <= 0:
            errors.append(
                f"batch_size and num_gpus must be positive; got "
                f"batch_size={batch_size}, num_gpus={num_gpus}"
            )
        else:
            per_replica = len(rows) / num_gpus
            if batch_size > per_replica:
                suggested = max(1, int(per_replica) if num_gpus > 1 else len(rows))
                errors.append(
                    f"batch_size={batch_size} exceeds the dataset limit: "
                    f"{len(rows)} row(s) / {num_gpus} GPU(s) = {per_replica:g} "
                    "per replica; the loader raises 'Dataset size "
                    f"({len(rows)}) is smaller than the total batch size' at "
                    f"launch. Set dataset.batch_size <= {suggested} "
                    "(or reduce num_gpus)."
                )

    return errors, len(rows), readable_files


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=pathlib.Path)
    parser.add_argument("--images-dir", required=True, type=pathlib.Path)
    parser.add_argument(
        "--input-map",
        action="append",
        metavar="LIGHT=INDEX",
        help=(
            "Lighting filename suffix and zero-based position. Repeat in YAML "
            "key order. Defaults to the four standard lighting inputs."
        ),
    )
    parser.add_argument("--num-input", type=int, default=4)
    parser.add_argument("--concat-type", choices=("linear", "grid"), default="linear")
    parser.add_argument("--grid-x", type=int, default=2)
    parser.add_argument("--grid-y", type=int, default=2)
    parser.add_argument("--image-ext", default=".jpg")
    parser.add_argument(
        "--mode",
        choices=("train", "evaluate", "inference"),
        default="train",
        help="train additionally requires both PASS and defect rows",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="when set, check the dataset holds at least one batch per replica",
    )
    parser.add_argument("--num-gpus", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        input_map = _parse_input_map(args.input_map)
    except ValueError as exc:
        print(f"validate_dataset: FATAL — {exc}", file=sys.stderr)
        return 2
    errors, row_count, readable_files = validate(
        args.csv,
        args.images_dir,
        input_map=input_map,
        num_input=args.num_input,
        concat_type=args.concat_type,
        grid_x=args.grid_x,
        grid_y=args.grid_y,
        image_ext=args.image_ext,
        mode=args.mode,
        batch_size=args.batch_size,
        num_gpus=args.num_gpus,
    )
    if errors:
        print(
            f"validate_dataset: FATAL — {len(errors)} issue(s) in {args.csv}",
            file=sys.stderr,
        )
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 2
    print(
        f"validate_dataset: ok ({row_count} row(s), {len(input_map)} lighting "
        f"inputs, {readable_files} readable file slots)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
