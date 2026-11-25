#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate AnomalyGenNext fine-tuning inputs and freeze a canonical recipe."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from PIL import Image


IMAGES = {".jpeg", ".jpg", ".png"}
UNSUPPORTED_IMAGE_EXTS = {".bmp", ".tif", ".tiff", ".webp"}
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
DEFAULT_RECIPE = Path(__file__).resolve().parents[1] / "assets" / "default_recipe.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _images(path: Path) -> list[Path]:
    files = sorted(item for item in path.iterdir() if item.is_file())
    unsupported = [item for item in files
                   if item.suffix.lower() in UNSUPPORTED_IMAGE_EXTS]
    if unsupported:
        raise ValueError(
            f"unsupported AnomalyGenNext image extension: {unsupported[0]}; "
            "use .jpg, .jpeg, or .png"
        )
    return [item for item in files if item.suffix.lower() in IMAGES]


def _validate_extension(path: Path, label: str) -> None:
    if path.suffix.lower() not in IMAGES:
        raise ValueError(
            f"unsupported AnomalyGenNext {label} extension: {path}; "
            "use .jpg, .jpeg, or .png"
        )


def _validate_mask(mask: Path) -> None:
    try:
        with Image.open(mask) as image:
            values = {value for value, count in enumerate(image.convert("L").histogram()) if count}
    except OSError as error:
        raise ValueError(f"cannot read AnomalyGenNext mask: {mask}") from error
    if values != {0, 255}:
        raise ValueError(
            f"mask must contain exactly binary pixel values 0 and 255: {mask}; "
            f"found {sorted(values)}"
        )


def _pairs(root: Path) -> list[list[str]]:
    result = []
    for texture in sorted(item for item in root.iterdir() if item.is_dir()):
        anomaly_root = texture / "anomaly_image"
        if not anomaly_root.is_dir():
            continue
        if not (texture / "clean_image").is_dir() or not _images(texture / "clean_image"):
            raise ValueError(f"texture has no clean images: {texture.name}")
        for defect in sorted(item for item in anomaly_root.iterdir() if item.is_dir()):
            anomalies = _images(defect)
            if not anomalies:
                continue
            mask_root = texture / "mask" / defect.name
            if not mask_root.is_dir():
                raise ValueError(f"missing mask directory: {mask_root}")
            for image in anomalies:
                choices = (mask_root / f"{image.stem}_mask{image.suffix}",
                           mask_root / f"{image.stem}_mask.png", mask_root / image.name)
                mask = next((path for path in choices if path.is_file()), None)
                if mask is None:
                    raise ValueError(f"anomaly lacks matching mask: {image}")
                _validate_mask(mask)
            result.append([texture.name, defect.name])
    if not result:
        raise ValueError(f"no TEXTURE/anomaly_image/DEFECT inputs under {root}")
    return result


def _declared_types(path: Path) -> set[str]:
    result = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        name = str(row.get("defect_type") or row.get("anomaly_type") or "")
        if not name:
            raise ValueError(f"missing defect_type at {path}:{number}")
        if row.get("spatial_dependency") == "text" and not row.get("roi_prompt_defect_location"):
            raise ValueError(f"text-routed {name} lacks roi_prompt_defect_location")
        result.add(name)
    return result


def _validation_path(raw: str, testcase: Path, dataset: Path, name: str) -> Path:
    path = Path(raw)
    choices = [path] if path.is_absolute() else [testcase.parent / path, dataset / path]
    parts = path.parts
    if not path.is_absolute() and len(parts) > 2 and parts[:2] == ("datasets", name):
        choices.append(dataset / Path(*parts[2:]))
    for choice in choices:
        if choice.is_file():
            return choice.resolve()
    raise ValueError(f"cannot resolve validation path: {raw}")


def _validation(path: Path, dataset: Path, name: str) -> tuple[list[dict[str, Any]], Counter]:
    rows, counts = [], Counter()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        anomaly = str(row.get("anomaly_type") or "")
        if not anomaly:
            raise ValueError(f"missing anomaly_type at {path}:{number}")
        for key in ("image_filename", "mask_filename"):
            if not row.get(key):
                raise ValueError(f"missing {key} at {path}:{number}")
            resolved = _validation_path(str(row[key]), path, dataset, name)
            _validate_extension(resolved, "mask" if key == "mask_filename" else "image")
            row[key] = str(resolved)
        _validate_mask(Path(row["mask_filename"]))
        rows.append(row)
        counts[anomaly] += 1
    if not rows:
        raise ValueError("validation testcase is empty")
    return rows, counts


def _validate_checkpoint_tree(root: Path) -> None:
    root = root.expanduser().resolve()
    qwen_assets = root / "hf"
    dinov2 = root / "facebook" / "dinov2-large"
    if not qwen_assets.is_dir():
        raise ValueError(
            f"checkpoint root lacks required Qwen tokenizer assets under hf/: {qwen_assets}"
        )
    if not (dinov2 / "config.json").is_file():
        raise ValueError(f"checkpoint root lacks DINOv2 config: {dinov2}")
    if not ((dinov2 / "model.safetensors").is_file()
            or (dinov2 / "pytorch_model.bin").is_file()):
        raise ValueError(f"checkpoint root lacks DINOv2 weights: {dinov2}")


def _validate_base_checkpoint(root: Path) -> None:
    root = root.expanduser().resolve()
    model = root / "model"
    if (not (root / "checkpoint.json").is_file()
            or not (model / ".metadata").is_file()
            or not any(path.is_file() for path in model.glob("*.distcp"))):
        raise ValueError(
            "Cosmos3-Nano base checkpoint must be a DCP directory containing "
            f"checkpoint.json, model/.metadata, and model/*.distcp: {root}. "
            "Convert it with cosmos_framework.scripts.convert_model_to_dcp in the pinned "
            "AnomalyGenNext container; see references/container-runtime.md."
        )


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    required = (("dataset root", args.dataset_root, True),
                ("validation testcase", args.validation_testcase, False),
                ("base checkpoint", args.base_checkpoint, True),
                ("VAE", args.vae_path, False),
                ("checkpoint root", args.checkpoint_root, True))
    for label, path, directory in required:
        if not (path.is_dir() if directory else path.is_file()):
            raise ValueError(f"{label} is missing: {path}")
    _validate_base_checkpoint(args.base_checkpoint)
    _validate_checkpoint_tree(args.checkpoint_root)
    if not SAFE_NAME.fullmatch(args.dataset_name) or not SAFE_NAME.fullmatch(args.job_name):
        raise ValueError("dataset and job names may contain only letters, numbers, dot, dash, underscore")
    recipe = yaml.safe_load(DEFAULT_RECIPE.read_text())
    if args.recipe_template:
        template = yaml.safe_load(args.recipe_template.read_text())
        if not isinstance(template, dict):
            raise ValueError("recipe template must be a YAML mapping")
        recipe.update(template)
    discovered = _pairs(args.dataset_root.resolve())
    pairs = recipe.get("anomaly_types", discovered)
    if not isinstance(pairs, list) or not pairs or any(
            not isinstance(pair, list) or len(pair) != 2 or not all(isinstance(v, str) and v for v in pair)
            for pair in pairs):
        raise ValueError("anomaly_types must be nonempty [texture, defect] pairs")
    if len({tuple(pair) for pair in pairs}) != len(pairs):
        raise ValueError("anomaly_types contains duplicates")
    missing = sorted(set(map(tuple, pairs)) - set(map(tuple, discovered)))
    if missing:
        raise ValueError(f"recipe types absent from dataset: {missing}")
    types = [f"{texture}+{defect}" for texture, defect in pairs]
    defect_spec = (args.defect_spec or args.dataset_root / "defect_spec.jsonl").resolve()
    if not defect_spec.is_file():
        raise ValueError(f"defect spec is missing: {defect_spec}")
    undefined = sorted(set(types) - _declared_types(defect_spec))
    if undefined:
        raise ValueError(f"recipe types absent from defect spec: {undefined}")
    rows, counts = _validation(args.validation_testcase.resolve(), args.dataset_root.resolve(),
                               args.dataset_name)
    insufficient = {name: counts[name] for name in types if counts[name] < args.min_validation_per_type}
    extra = sorted(set(counts) - set(types))
    if insufficient or extra:
        raise ValueError(f"validation coverage mismatch: insufficient={insufficient}, extra={extra}")
    if args.max_iter is not None:
        recipe["max_iter"] = args.max_iter
    if args.validation_iter is not None:
        recipe["validation_iter"] = args.validation_iter
    if args.save_iter is not None:
        recipe["save_iter"] = args.save_iter
    for key in ("max_iter", "validation_iter", "save_iter"):
        if not isinstance(recipe.get(key), int) or isinstance(recipe[key], bool) or recipe[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if recipe["validation_iter"] % recipe["save_iter"] or recipe["max_iter"] < recipe["validation_iter"]:
        raise ValueError("validation/save intervals must align and max_iter must reach validation")
    normalized = args.output.with_suffix(".validation.jsonl")
    metadata = args.output.with_suffix(".metadata.json")
    if any(path.exists() for path in (args.output, normalized, metadata)):
        raise FileExistsError("refusing to overwrite recipe outputs")
    normalized.parent.mkdir(parents=True, exist_ok=True)
    normalized.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    recipe.update({"task_type": "texture_ft", "experiment": "anomalygen_texture_ft",
                   "job_name": args.job_name, "dataset_name": args.dataset_name,
                   "anomaly_types": pairs, "dataset_path": str(args.dataset_root.resolve()),
                   "testcase_jsonl": str(normalized.resolve()), "model_size": "nano",
                   "base_checkpoint_path": str(args.base_checkpoint.resolve()),
                   "vae_path": str(args.vae_path.resolve()), "run_validation_on_start": True,
                   "early_stop_metric": "nn"})
    args.output.write_text(yaml.safe_dump(recipe, sort_keys=False))
    report = {"status": "COMPLETE", "recipe": str(args.output.resolve()),
              "recipe_sha256": _sha256(args.output), "validation_testcase": str(normalized.resolve()),
              "dataset_root": str(args.dataset_root.resolve()), "defect_spec": str(defect_spec),
              "base_checkpoint": str(args.base_checkpoint.resolve()),
              "base_checkpoint_format": "dcp",
              "checkpoint_root": str(args.checkpoint_root.resolve()),
              "anomaly_types": types,
              "validation_counts": {name: counts[name] for name in types},
              "metric": "Average.nn_score", "direction": "max"}
    metadata.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--validation-testcase", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--vae-path", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--defect-spec", type=Path)
    parser.add_argument("--recipe-template", type=Path)
    parser.add_argument("--job-name", default="anomalygen_texture_ft")
    parser.add_argument("--min-validation-per-type", type=int, default=3)
    parser.add_argument("--max-iter", type=int)
    parser.add_argument("--validation-iter", type=int)
    parser.add_argument("--save-iter", type=int)
    args = parser.parse_args()
    print(json.dumps(prepare(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
