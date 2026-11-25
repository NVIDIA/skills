#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Emit a stage's starting spec from TAO rather than hand-authoring one.

Three sources, chosen by stage:

``annotations`` / ``analytics``
    ``<module> default_specs results_dir=<dir>`` writes ``experiment.yaml``, with
    ``???`` on mandatory fields. ``results_dir`` is itself mandatory. Supported
    modules are analytics, annotations, augmentation, auto_label, image.

``grounding_dino`` / ``codetr``
    ``OmegaConf.to_yaml(OmegaConf.structured(ExperimentConfig))``. No spec files
    ship for these — ``experiment_specs/`` exists only for centerpose and
    visual_changenet — but the Hydra schema carries the defaults.

``gap_analysis`` / ``tmm`` / ``embedding``
    ``assets/*.yaml``. TAO emits nothing for these modules, so they are the only
    hand-maintained specs in the skill.

Output is a starting point: mandatory fields are ``???`` and each stage overlay
documents its overrides. The grounding_dino schema default sets
``dataset.val_data_sources: null``, which cannot train.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

# stage -> how its spec is obtained
DS_MODULES = {
    "kitti_to_coco": "annotations",
    "coco_to_odvg": "annotations",
    "kpi_analyze": "analytics",
}
PYT_MODELS = {
    "grounding_dino_train": "grounding_dino",
    "grounding_dino_inference": "grounding_dino",
    "codetr_inference": "codetr",
}
SHIPPED_ASSETS = {
    "gap_analysis": "gap_analysis_object_detection.yaml",
    "mine": "tmm_unique_neighbor_matching.yaml",
    "embedding": "image_embeddings.yaml",
}
STAGES = sorted({*DS_MODULES, *PYT_MODELS, *SHIPPED_ASSETS})

# Marks where the schema dump begins, past the container's startup banner.
SENTINEL = "---TAO-SPEC-BEGINS---"


def emit_from_ds(module: str, image: str, workdir: Path) -> str:
    """Return the YAML `<module> default_specs` writes into workdir."""
    workdir.mkdir(parents=True, exist_ok=True)
    spec = workdir / "experiment.yaml"
    # Remove any spec from an earlier call: the check below is "did this run write
    # one", and a leftover would otherwise be returned as a fresh success.
    spec.unlink(missing_ok=True)

    # --user with a uid the image has no /etc/passwd entry for leaves the process
    # nameless, and getpass.getuser() then raises before `default_specs` runs. USER is
    # read before pwd is consulted, so any non-empty value settles it. HOME would
    # default to "/", which is not writable; the workdir is already mounted.
    cmd = [
        "docker", "run", "--rm", "--gpus", "all", "--shm-size=8g",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "USER=deft", "-e", f"HOME={workdir}",
        "-v", f"{workdir}:{workdir}", "-w", str(workdir),
        image, module, "default_specs", f"results_dir={workdir}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not spec.is_file():
        tail = (result.stderr or result.stdout or "").strip().splitlines()[-6:]
        raise RuntimeError(
            f"{module} default_specs failed (exit {result.returncode}):\n  " + "\n  ".join(tail)
        )
    return spec.read_text(encoding="utf-8")


def emit_from_pyt_schema(model: str, image: str) -> str:
    """Return the model's Hydra schema rendered as YAML, banner stripped."""
    # The image prints a licence and driver banner on every run, and some of those
    # lines end in a colon, so slicing at "the first top-level YAML key" picks up
    # banner text. Emit an explicit sentinel instead and cut on that.
    code = (
        "from omegaconf import OmegaConf;"
        f"from nvidia_tao_pytorch.config.{model}.default_config import ExperimentConfig;"
        f"print('{SENTINEL}');"
        "print(OmegaConf.to_yaml(OmegaConf.structured(ExperimentConfig)))"
    )
    cmd = ["docker", "run", "--rm", image, "python3", "-c", code]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        tail = (result.stderr or "").strip().splitlines()[-6:]
        raise RuntimeError(f"could not dump the {model} schema:\n  " + "\n  ".join(tail))
    if SENTINEL not in result.stdout:
        raise RuntimeError(
            f"the {model} schema dump produced no output; the container printed:\n  "
            + "\n  ".join(result.stdout.strip().splitlines()[-6:])
        )
    return result.stdout.split(SENTINEL, 1)[1].lstrip("\n")


def _mandatory_summary(text: str) -> list[str]:
    """Describe the ``???`` fields, keyed by the top-level block they sit under.

    A spec covers every action the module supports, so most ``???`` belong to
    blocks this stage never runs -- a grounding_dino spec marks ``evaluate``,
    ``inference`` and ``export`` checkpoints mandatory even when emitting for
    train. Grouping by block keeps that visible instead of presenting one flat
    list that implies all of them need filling.

    Comments are stripped before the check: a value may be followed by one, and
    testing the raw line for a ``???`` suffix silently misses those fields.
    """
    blocks: dict[str, list[str]] = {}
    current = "(top level)"
    for raw in text.splitlines():
        if raw and not raw[0].isspace() and raw.rstrip().endswith(":"):
            current = raw.rstrip()[:-1]
        body = raw.split("#", 1)[0].rstrip()
        if body.endswith("???"):
            blocks.setdefault(current, []).append(body.split(":")[0].strip())

    if not blocks:
        return ["  no mandatory fields"]
    total = sum(len(v) for v in blocks.values())
    out = [f"  {total} mandatory field(s), by block:"]
    for block, fields in blocks.items():
        out.append(f"    {block}: {', '.join(fields)}")
    out.append("  a STARTING point — the stage overlay says which blocks apply and what to override")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", required=True, choices=STAGES,
                        help="Which stage's spec to emit.")
    parser.add_argument("--out", required=True, help="Where to write it.")
    parser.add_argument("--ds-image", default=None,
                        help="Data-services image. Required for the annotations/analytics stages.")
    parser.add_argument("--pyt-image", default=None,
                        help="TAO PyTorch image. Required for the grounding_dino/codetr stages.")
    parser.add_argument("--workdir", default=None,
                        help="Scratch dir for `default_specs`. Defaults to alongside --out.")
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        out = Path(args.out).expanduser().resolve()

        if args.stage in SHIPPED_ASSETS:
            asset = Path(__file__).resolve().parent.parent / "assets" / SHIPPED_ASSETS[args.stage]
            if not asset.is_file():
                raise FileNotFoundError(f"shipped asset missing: {asset}")
            text = asset.read_text(encoding="utf-8")
            source = f"shipped asset {asset.name} (TAO emits no default for this module)"

        elif args.stage in DS_MODULES:
            if not args.ds_image:
                raise ValueError(f"--ds-image is required for stage {args.stage}")
            if shutil.which("docker") is None:
                raise RuntimeError("docker is not on PATH")
            module = DS_MODULES[args.stage]
            workdir = Path(args.workdir).expanduser().resolve() if args.workdir else out.parent / f".defaultspec_{module}"
            text = emit_from_ds(module, args.ds_image, workdir)
            source = f"`{module} default_specs` in {args.ds_image}"

        else:
            if not args.pyt_image:
                raise ValueError(f"--pyt-image is required for stage {args.stage}")
            model = PYT_MODELS[args.stage]
            text = emit_from_pyt_schema(model, args.pyt_image)
            source = f"{model} Hydra schema defaults in {args.pyt_image}"

        # Validate before writing: a banner or a stack trace captured as "the spec"
        # would otherwise only fail later, inside a container, as a merge error.
        try:
            parsed = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise RuntimeError(f"the emitted spec is not valid YAML: {exc}") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError(
                f"the emitted spec is {type(parsed).__name__}, not a mapping — "
                "the container's output was probably captured instead of the spec"
            )

        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"{args.stage} -> {out}")
        print(f"  source: {source}")
        for line in _mandatory_summary(text):
            print(line)
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
