#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Append a mined ODVG source to a Grounding DINO train spec.

Copies the previous iteration's spec, appends one
``{image_dir, json_file, label_map}`` entry to the ``dataset.train_data_sources``
list, and sets ``train.num_epochs`` and ``train.optim.lr``. Dataset growth is by
list-append: earlier sources are never removed, so iteration N trains on every
source accumulated so far.

This replaces the reference pipeline's ``reformat_train_spec.py``, which exists
only inside an internal container image and cannot be called from a skill.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return data


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, default_flow_style=False)


def append_source(spec: dict[str, Any], image_dir: str, odvg_file: str, label_map: str) -> int:
    dataset = spec.setdefault("dataset", {})
    sources = dataset.get("train_data_sources")
    if sources is None:
        sources = []
    if not isinstance(sources, list):
        raise ValueError(
            "dataset.train_data_sources must be a list for Grounding DINO ODVG training; "
            f"found {type(sources).__name__}."
        )

    entry = {"image_dir": image_dir, "json_file": odvg_file, "label_map": label_map}
    # Re-running an iteration must not double-add the same source.
    if entry in sources:
        print(f"NOTE: source already present, not appending again: {image_dir}")
    else:
        sources.append(entry)
    dataset["train_data_sources"] = sources
    return len(sources)


def set_pretrained(spec: dict[str, Any], checkpoint: str) -> str:
    """Overwrite train.pretrained_model_path with the run's resolved checkpoint.

    Always overwritten, never inherited. The run resolved one checkpoint — the
    user's if they supplied it, otherwise the one pulled from NGC — and that is the
    only correct value. A template carrying some other path would otherwise train
    from weights nobody chose, and every iteration after it would be measured
    against a baseline it does not share.

    Every iteration fine-tunes this same BASE checkpoint, never the previous
    iteration's, so the value is identical on every pass.

    Left unset entirely, training starts from nothing, reports success, and emits a
    model that detects nothing — the most expensive silent failure in the loop,
    because it costs a full training run and only surfaces at KPI.
    """
    train = spec.setdefault("train", {})
    previous = train.get("pretrained_model_path")
    train["pretrained_model_path"] = checkpoint
    if previous and str(previous) != checkpoint:
        print(f"NOTE: replacing train.pretrained_model_path\n"
              f"        template: {previous}\n"
              f"        this run: {checkpoint}")
    return checkpoint


def set_max_labels(spec: dict[str, Any], label_map_file: Path) -> int:
    """Derive dataset.max_labels from the staged label map.

    It caps how many class phrases go into a caption, so it has to match the class
    count. Hardcoding it means every run with a different number of target classes
    is silently wrong: too low and classes are dropped from captions, too high and
    the sampler pads with labels that do not exist.
    """
    label_map = json.loads(label_map_file.read_text(encoding="utf-8"))
    count = len(label_map)
    if count < 1:
        raise ValueError(f"{label_map_file} declares no classes")
    dataset = spec.setdefault("dataset", {})
    previous = dataset.get("max_labels")
    dataset["max_labels"] = count
    if isinstance(previous, int) and previous != count:
        print(f"NOTE: dataset.max_labels {previous} -> {count}, from the staged label map.")
    return count


def set_validation(spec: dict[str, Any], image_dir: str, json_file: str) -> None:
    """Point val (and test) at the pool-derived split.

    Both are set: `grounding_dino train` subscripts val_data_sources unconditionally,
    and leaving test unset is a second way to trip the same schema. The members must
    be real paths — the field is Optional as a whole, but its members are not, so
    `{image_dir: null}` is rejected by Hydra while a bare `null` crashes the loader.
    """
    dataset = spec.setdefault("dataset", {})
    sources = {"image_dir": image_dir, "json_file": json_file}
    dataset["val_data_sources"] = dict(sources)
    dataset["test_data_sources"] = dict(sources)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-spec", required=True, help="Train spec from the previous phase.")
    parser.add_argument("--output-spec", required=True, help="Where to write the new spec.")
    parser.add_argument("--tmm-image-dir", required=True)
    parser.add_argument("--tmm-odvg-file", required=True)
    parser.add_argument("--tmm-label-map-file", required=True)
    parser.add_argument("--pretrained-model-path", required=True,
                        help="Zero-shot checkpoint every iteration fine-tunes FROM — the "
                             "user's if they supplied one, otherwise the NGC download "
                             "(scripts/fetch_gdino_checkpoint.py). Always overwrites whatever "
                             "the template carried; inheriting a stale path would train from "
                             "weights nobody chose.")
    parser.add_argument("--val-image-dir", default=None,
                        help="Validation image directory — the pool images. Required together "
                             "with --val-json-file. `grounding_dino train` cannot run without "
                             "a validation source; see references/grounding-dino.md.")
    parser.add_argument("--val-json-file", default=None,
                        help="Validation COCO from prepare_val_split_for_train.py. Must carry 0-based "
                             "category ids: GDINO's val loader uses category_id verbatim as a "
                             "dense label index, so a conventional 1-based COCO overflows on "
                             "the last class.")
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    args = parser.parse_args()

    try:
        previous = Path(args.previous_spec).expanduser().resolve()
        if not previous.is_file():
            raise FileNotFoundError(f"previous-spec does not exist: {previous}")
        spec = load_yaml(previous)

        for flag, raw in (
            ("--tmm-image-dir", args.tmm_image_dir),
            ("--tmm-odvg-file", args.tmm_odvg_file),
            ("--tmm-label-map-file", args.tmm_label_map_file),
        ):
            path = Path(raw).expanduser()
            if not path.is_absolute():
                raise ValueError(f"{flag} must be absolute: {path}")
            if not path.exists():
                raise FileNotFoundError(f"{flag} does not exist: {path}")

        count = append_source(
            spec,
            str(Path(args.tmm_image_dir).expanduser().resolve()),
            str(Path(args.tmm_odvg_file).expanduser().resolve()),
            str(Path(args.tmm_label_map_file).expanduser().resolve()),
        )

        checkpoint = set_pretrained(spec, args.pretrained_model_path)
        classes = set_max_labels(spec, Path(args.tmm_label_map_file).expanduser().resolve())

        # Validation is mandatory for `grounding_dino train`, so refuse to emit a spec
        # that cannot train. The template ships val_data_sources unset precisely so this
        # has to be answered here rather than silently inherited.
        if bool(args.val_image_dir) != bool(args.val_json_file):
            raise ValueError("--val-image-dir and --val-json-file must be given together")
        if args.val_image_dir:
            for flag, raw in (("--val-image-dir", args.val_image_dir),
                              ("--val-json-file", args.val_json_file)):
                path = Path(raw).expanduser()
                if not path.is_absolute():
                    raise ValueError(f"{flag} must be absolute: {path}")
                if not path.exists():
                    raise FileNotFoundError(f"{flag} does not exist: {path}")
            set_validation(
                spec,
                str(Path(args.val_image_dir).expanduser().resolve()),
                str(Path(args.val_json_file).expanduser().resolve()),
            )
        else:
            existing = (spec.get("dataset") or {}).get("val_data_sources")
            if not (isinstance(existing, dict) and existing.get("json_file")):
                raise ValueError(
                    "no validation source. `grounding_dino train` subscripts "
                    "val_data_sources unconditionally and cannot run without one. Pass "
                    "--val-image-dir/--val-json-file — prepare_val_split_for_train.py builds a "
                    "0-based COCO from the prepared pool."
                )

        if args.num_epochs is not None:
            if args.num_epochs < 1:
                raise ValueError("--num-epochs must be at least 1.")
            train = spec.setdefault("train", {})
            train["num_epochs"] = args.num_epochs
            # TAO fails at config-merge time when checkpoint_interval exceeds num_epochs.
            interval = train.get("checkpoint_interval")
            if isinstance(interval, int) and interval > args.num_epochs:
                print(
                    f"NOTE: lowering train.checkpoint_interval {interval} -> {args.num_epochs} "
                    "to satisfy checkpoint_interval <= num_epochs."
                )
                train["checkpoint_interval"] = args.num_epochs
            val_interval = train.get("validation_interval")
            if isinstance(val_interval, int) and val_interval > args.num_epochs:
                print(
                    f"NOTE: lowering train.validation_interval {val_interval} -> {args.num_epochs}."
                )
                train["validation_interval"] = args.num_epochs

        if args.learning_rate is not None:
            spec.setdefault("train", {}).setdefault("optim", {})["lr"] = args.learning_rate

        output = Path(args.output_spec).expanduser().resolve()
        write_yaml(output, spec)
        print(f"Wrote train spec: {output}")
        print(f"dataset.train_data_sources now has {count} source(s)")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
