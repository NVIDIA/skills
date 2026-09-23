#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Cosmos Embed terminal-status and output validation."""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import yaml


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from resume_position import resume_position  # noqa: E402
from cosmos_embed_outputs_to_parquet import consolidate_embeddings  # noqa: E402
from validate_cosmos_embed_output import (  # noqa: E402
    check_completion,
    validate_completion,
)


class CosmosEmbedCompletionTests(unittest.TestCase):
    """Exercise accepted teardown and strict output checks."""

    def make_spec(
        self,
        root: pathlib.Path,
        *,
        dataset: str = "kpi",
        mode: str = "text",
        inputs: list[str] | None = None,
        results_dir: pathlib.Path | None = None,
    ) -> pathlib.Path:
        inputs = inputs or (["same question", "same question"] if mode == "text" else ["/data/a.mp4"])
        dataset_dir = root / "cosmos_embed_output" / dataset
        spec_path = dataset_dir / "specs" / f"inference_{mode}.yaml"
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        results_dir = results_dir or dataset_dir / "results" / mode
        query = {
            "input_texts": inputs if mode == "text" else [],
            "input_videos": inputs if mode == "video" else [],
        }
        spec_path.write_text(
            yaml.safe_dump(
                {
                    "results_dir": str(results_dir),
                    "inference": {"mode": mode, "query": query},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        return spec_path

    def write_outputs(
        self,
        spec_path: pathlib.Path,
        *,
        identifiers: list[str],
        finite: bool = True,
    ) -> None:
        spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
        mode = spec["inference"]["mode"]
        inference_dir = pathlib.Path(spec["results_dir"]) / "inference"
        inference_dir.mkdir(parents=True, exist_ok=True)
        embeddings = np.arange(len(identifiers) * 4, dtype=np.float32).reshape(len(identifiers), 4)
        if not finite:
            embeddings[0, 0] = np.nan
        npy_path = inference_dir / f"{mode}_embeddings.npy"
        np.save(npy_path, embeddings)
        identifier_key = "text" if mode == "text" else "video_path"
        metadata = {
            "npy_file": npy_path.name,
            "results": [
                {identifier_key: identifier, "npy_row": index}
                for index, identifier in enumerate(identifiers)
            ],
        }
        (inference_dir / f"{mode}_embeddings.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )

    def test_exit_zero_accepts_repeated_text_occurrences(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            spec = self.make_spec(pathlib.Path(temporary))
            self.write_outputs(spec, identifiers=["same question", "same question"])
            completion_path = validate_completion(spec, 0)
            completion = check_completion(spec)
            self.assertEqual(completion["status"], "ok")
            self.assertEqual(completion["embedding_shape"], [2, 4])
            self.assertTrue(completion_path.is_file())

    def test_exit_130_is_accepted_only_with_complete_video_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            inputs = ["/data/a.mp4", "/data/b.mp4"]
            spec = self.make_spec(pathlib.Path(temporary), mode="video", inputs=inputs)
            self.write_outputs(spec, identifiers=list(reversed(inputs)))
            validate_completion(spec, 130)
            completion = check_completion(spec)
            self.assertEqual(completion["status"], "ok_with_teardown_warning")
            self.assertTrue(completion["accepted_exit_code_130"])

    def test_other_nonzero_exit_is_rejected_even_with_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            spec = self.make_spec(pathlib.Path(temporary))
            self.write_outputs(spec, identifiers=["same question", "same question"])
            with self.assertRaisesRegex(RuntimeError, "only exit 0 or.*130"):
                validate_completion(spec, 1)

    def test_incomplete_or_nonfinite_outputs_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            spec = self.make_spec(root, inputs=["one", "two"])
            self.write_outputs(spec, identifiers=["one"])
            with self.assertRaisesRegex(ValueError, "does not match 2 expected"):
                validate_completion(spec, 130)

            self.write_outputs(spec, identifiers=["one", "two"], finite=False)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                validate_completion(spec, 130)

    def test_identifiers_must_match_the_current_spec(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            spec = self.make_spec(pathlib.Path(temporary), inputs=["one", "two"])
            self.write_outputs(spec, identifiers=["one", "three"])
            with self.assertRaisesRegex(ValueError, "identifiers do not match"):
                validate_completion(spec, 130)

    def test_existing_matching_outputs_are_reusable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            spec = self.make_spec(pathlib.Path(temporary), inputs=["one"])
            self.write_outputs(spec, identifiers=["one"])
            validate_completion(spec, 130)
            self.assertEqual(check_completion(spec)["expected_count"], 1)

    def test_parquet_conversion_uses_results_dir_from_spec(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            custom_results = root / "custom_embed_results"
            spec = self.make_spec(
                root,
                mode="video",
                inputs=["/data/a.mp4"],
                results_dir=custom_results,
            )
            self.write_outputs(spec, identifiers=["/data/a.mp4"])
            output_dir = root / "cosmos_embed_output" / "kpi"
            parquet_dir = root / "embedding_parquets" / "kpi"
            parquet_path = consolidate_embeddings(output_dir, parquet_dir, "video")
            converted = pd.read_parquet(parquet_path)
            self.assertEqual(converted["filepath"].tolist(), ["/data/a.mp4"])
            self.assertEqual(converted["modality"].tolist(), ["video"])

    def test_workflow_forbids_results_dir_command_override(self) -> None:
        mining_loop = (SKILL_ROOT / "references/mining-loop.md").read_text(encoding="utf-8")
        self.assertIn("Do not append a `results_dir=...` CLI override", mining_loop)

    def test_resume_reopens_cosmos_embed_when_validation_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = pathlib.Path(temporary)
            spec = self.make_spec(run_dir, inputs=["one"])
            state = {"max_iterations": 1, "mine_unique_only": False}
            events = [
                {"seq": 1, "iter": "baseline", "stage": "baseline_evaluate", "status": "ok"},
                {
                    "seq": 2,
                    "iter": "initialization",
                    "stage": "prepare_cosmos_embed_inference",
                    "status": "ok",
                },
                {"seq": 3, "iter": "initialization", "stage": "cosmos_embed", "status": "ok"},
            ]
            position = resume_position(state, events, run_dir)
            self.assertEqual(position["next_stage"], "cosmos_embed")

            self.write_outputs(spec, identifiers=["one"])
            validate_completion(spec, 130)
            position = resume_position(state, events, run_dir)
            self.assertEqual(position["next_stage"], "convert_embeddings")


if __name__ == "__main__":
    unittest.main()
