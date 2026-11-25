#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate a TAO Data Services image-embeddings spec.

Fill `assets/default_image_embeddings.yaml` — it carries every default this stage
needs — then run this against the result. The template is the only place a
default value lives, so there is nothing that can disagree with it.

Checks the spec against what `embedding image_embeddings` actually requires:

- `input_parquet`, `output_parquet`, `model` and `model_path` are present, and
  the two paths are absolute.
- `input_parquet` exists and is a file; `output_parquet`'s parent is created and
  probed for writability, so a run does not fail after doing the work.
- `model` is `CLIP` or `SigLIP`.
- `model_config_path` is set when `model_path` is a TAO checkpoint — a `.pth` or
  `.ckpt` cannot be rebuilt without the spec it was trained under.

It also reports the encoder, because embeddings are only comparable to others
produced by the same `model` and `model_path`. A mining step that compares
parquets from different encoders returns neighbours unrelated to its targets,
and nothing in the parquet records which encoder wrote it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

VALID_MODELS = {"CLIP", "SigLIP"}
TAO_CKPT_SUFFIXES = {".pth", ".ckpt"}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return data


def validate_config(config: dict[str, Any]) -> None:
    required = ("input_parquet", "output_parquet", "model", "model_path")
    missing = [
        key
        for key in required
        if not isinstance(config.get(key), str) or not config[key].strip()
    ]
    if missing:
        raise ValueError(f"Missing required field(s): {', '.join(missing)}")

    src = Path(str(config["input_parquet"])).expanduser()
    if not src.is_absolute():
        raise ValueError(f"input_parquet must be absolute: {src}")
    if not src.is_file():
        raise FileNotFoundError(f"input_parquet does not exist or is not a file: {src}")

    out = Path(str(config["output_parquet"])).expanduser()
    if not out.is_absolute():
        raise ValueError(f"output_parquet must be absolute: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        probe = out.parent / ".tao_image_embeddings_write_test"
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        raise PermissionError(f"output_parquet parent is not writable: {out.parent}") from exc

    model = config["model"]
    if model not in VALID_MODELS:
        raise ValueError(f"model must be one of {sorted(VALID_MODELS)}.")
    config["model"] = model

    # `model` selects the loader and `model_path` the weights, and they are otherwise
    # independent — a SigLIP path under model: CLIP passes every other check here and
    # fails inside the container, or worse loads something that embeds badly.
    mp = config["model_path"].lower()
    for name, marker in (("SigLIP", "siglip"), ("CLIP", "clip")):
        if marker in mp and model != name and not (name == "CLIP" and "siglip" in mp):
            raise ValueError(
                f"model is {model!r} but model_path names {name} ({config['model_path']}). "
                "The two must agree — they select the loader and the weights separately."
            )

    # A TAO checkpoint cannot be rebuilt without its training spec.
    model_path = config["model_path"]
    if Path(model_path).suffix in TAO_CKPT_SUFFIXES and not config.get("model_config_path"):
        raise ValueError(
            "model_config_path is required when model_path is a TAO checkpoint "
            f"({Path(model_path).suffix})."
        )

    batch_size = config.get("batch_size", 64)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be an integer of at least 1.")
    config["batch_size"] = batch_size
    model_config_path = config.get("model_config_path", "")
    if not isinstance(model_config_path, str):
        raise ValueError("model_config_path must be a string.")
    config["model_config_path"] = model_config_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", required=True, help="Path to the image-embeddings YAML.")
    return parser.parse_args()


def main() -> int:
    try:
        spec = Path(parse_args().spec).expanduser().resolve()
        config = load_yaml(spec)
        validate_config(config)
        print(f"OK: image-embeddings spec is valid: {spec}")
        print(f"Encoder: {config['model']} @ {config['model_path']}")
        print(f"Output parquet: {config['output_parquet']}")
        out_dir = Path(str(config["output_parquet"])).expanduser().resolve().parent
        if spec.parent != out_dir:
            print(f"WARNING: the spec is outside the output directory ({spec.parent} vs "
                  f"{out_dir}). The run does not copy it, so the embeddings will carry no record "
                  f"of the encoder that produced them. Author it at {out_dir / spec.name} "
                  "instead — embeddings compared against each other must come from the same "
                  "model and model_path, and that is unknowable after the fact otherwise.",
                  file=sys.stderr)
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
