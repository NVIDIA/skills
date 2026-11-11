#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Refuse an embed spec whose encoder is not the run's.

Mining compares each iteration's weak-image vectors with the source pool's by
distance, and that comparison only means anything inside one encoder's space. The
run's encoder is frozen in ``deft_state.json`` at init, where it is also checked
against the encoder the pool report says built the pool. This closes the remaining
gap: the embed spec an iteration actually launches with.

``verify_image_embeddings_spec.py`` checks a spec against itself -- that ``model`` and
``model_path`` agree. It cannot see the run. A spec built with a different encoder
passes it, and then either crashes mining (a different vector size) or, with the same
size, returns confident nearest neighbours that are noise, with nothing reporting it.

Inputs:  --spec (the embed spec), --results-dir (reads config.embedding_model and
         config.embedding_model_path)
Output:  one confirmation line

Exits 1 when the spec's model or model_path differs from the run's, naming both.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from deft_stages import read_state  # noqa: E402


def _as_encoder_path(value: object) -> str:
    """A local snapshot as its resolved absolute path; a HuggingFace id unchanged.

    Matches how init records config.embedding_model_path, so a spec naming the same
    snapshot through a different spelling of the path is not refused.
    """
    text = str(value)
    if text.startswith(("/", "~", ".")) or Path(text).exists():
        return str(Path(text).expanduser().resolve())
    return text


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", required=True, help="The embed stage's spec.")
    parser.add_argument("--results-dir", required=True,
                        help="The run's results dir; its deft_state.json holds the encoder.")
    args = parser.parse_args()
    try:
        spec = yaml.safe_load(Path(args.spec).expanduser().read_text(encoding="utf-8"))
        if not isinstance(spec, dict):
            raise ValueError(f"{args.spec}: not a YAML mapping")
        config = read_state(Path(args.results_dir).expanduser().resolve()).get("config") or {}

        want_family = config.get("embedding_model")
        want_path = config.get("embedding_model_path")
        if not want_family or not want_path:
            raise ValueError(
                "deft_state.json records no embedding_model / embedding_model_path, so "
                "there is no run encoder to check the spec against")

        problems = []
        if str(spec.get("model")) != str(want_family):
            problems.append(f"model is {spec.get('model')!r}, the run's is {want_family!r}")
        if _as_encoder_path(spec.get("model_path")) != _as_encoder_path(want_path):
            problems.append(
                f"model_path is {spec.get('model_path')!r}, the run's is {want_path!r}")
        if problems:
            raise ValueError(
                f"{args.spec} would embed with a different encoder from the one this run, "
                f"and its source pool, use: " + "; ".join(problems) + ". Mining compares "
                f"the two sets of vectors by distance, which is meaningless across "
                f"encoders. Set model and model_path from state.config")

        print(f"verify_embed_encoder: {want_family} @ {want_path} — matches the run")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"verify_embed_encoder: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
