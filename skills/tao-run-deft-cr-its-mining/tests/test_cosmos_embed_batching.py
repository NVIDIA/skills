#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise bounded native commands, coverage, and failure handling without a GPU."""

import json
import os
from pathlib import Path
import subprocess
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from batch_cosmos_embed_text import load_plan, merge_outputs, prepare, run  # noqa: E402
from validate_cosmos_embed_output import check_completion, file_sha256, validate_completion  # noqa: E402


class CosmosEmbedBatchingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.stage = self.root / "results"
        self.spec = self.root / "inference_text.yaml"
        self.queries = ["same question", "different question", "same question", "last question", "same question"]
        self.payload = {
            "results_dir": str(self.stage), "model": {"unchanged": True},
            "inference": {
                "mode": "text", "num_gpus": 4, "checkpoint": "/model/selected",
                "save_dataset_pkl": str(self.stage / "embeddings.pkl"),
                "query": {"input_texts": self.queries, "input_videos": []},
            },
        }
        self.write_spec()

    def write_spec(self):
        self.spec.write_text(yaml.safe_dump(self.payload), encoding="utf-8")

    def native_output(self, command, *, exit_code=0, dimension=7, mutation=None, **kwargs):
        self.assertEqual(command[:3], ["cosmos-embed1", "inference", "-e"])
        self.assertTrue(kwargs["start_new_session"])
        spec = yaml.safe_load(Path(command[3]).read_text())
        texts = spec["inference"]["query"]["input_texts"]
        number = int(Path(command[3]).parent.name.removeprefix("batch_"))
        start = (number - 1) * 2
        matrix = np.arange(start * dimension, (start + len(texts)) * dimension,
                           dtype=np.float32).reshape(len(texts), dimension)
        output = Path(spec["results_dir"]) / "inference"
        output.mkdir(parents=True)
        metadata = {
            "checkpoint": spec["inference"]["checkpoint"], "npy_file": "text_embeddings.npy",
            # Metadata need not be ordered; npy_row is authoritative.
            "results": list(reversed([
                {"text": text, "npy_row": i} for i, text in enumerate(texts)
            ])),
        }
        if mutation:
            mutation(matrix, metadata)
        np.save(output / "text_embeddings.npy", matrix)
        (output / "text_embeddings.json").write_text(json.dumps(metadata), encoding="utf-8")
        return subprocess.CompletedProcess(command, exit_code)

    def test_prepare_preserves_spec_and_exact_ordered_slices(self):
        original = self.spec.read_bytes()
        prepare(self.spec, 2)
        _, entries = load_plan(self.spec)
        self.assertEqual([entry["count"] for entry in entries], [2, 2, 1])
        flattened = []
        for entry in entries:
            partial = yaml.safe_load(Path(entry["spec"]).read_text())
            flattened.extend(partial["inference"]["query"]["input_texts"])
            self.assertEqual(partial["model"], self.payload["model"])
            self.assertEqual(partial["inference"]["num_gpus"], 4)
            self.assertTrue(Path(partial["results_dir"]).is_relative_to(self.stage))
        self.assertEqual(flattened, self.queries)
        self.assertEqual(self.spec.read_bytes(), original)
        with self.assertRaises(FileExistsError):
            prepare(self.spec, 2)

    def test_success_preserves_duplicates_vectors_and_teardown_evidence(self):
        prepare(self.spec, 2)
        with patch("batch_cosmos_embed_text.run_child",
                   side_effect=lambda *a, **kw: self.native_output(*a, exit_code=130, **kw)) as native:
            self.assertEqual(run(self.spec), 0)
        self.assertEqual(native.call_count, 3)
        completion = check_completion(self.spec)
        self.assertEqual(completion["embedding_shape"], [5, 7])
        matrix = np.load(completion["npy_path"])
        np.testing.assert_array_equal(matrix, np.arange(35, dtype=np.float32).reshape(5, 7))
        metadata = json.loads(Path(completion["metadata_path"]).read_text())
        self.assertEqual([row["text"] for row in metadata["results"]], self.queries)
        self.assertEqual([row["npy_row"] for row in metadata["results"]], list(range(5)))
        provenance = json.loads((self.stage / "inference/batch-provenance.json").read_text())
        self.assertEqual([entry["exit_code"] for entry in provenance["batches"]], [130, 130, 130])
        with patch("batch_cosmos_embed_text.run_child") as native:
            with self.assertRaises(FileExistsError):
                run(self.spec)
            native.assert_not_called()

    def test_zero_exit_and_768_dimensional_embeddings(self):
        prepare(self.spec, 2)
        with patch("batch_cosmos_embed_text.run_child",
                   side_effect=lambda *a, **kw: self.native_output(*a, dimension=768, **kw)):
            self.assertEqual(run(self.spec), 0)
        self.assertEqual(check_completion(self.spec)["embedding_shape"], [5, 768])

    def test_failed_child_stops_without_accepting_existing_output(self):
        prepare(self.spec, 2)
        with patch("batch_cosmos_embed_text.run_child",
                   side_effect=lambda *a, **kw: self.native_output(*a, exit_code=1, **kw)) as native:
            self.assertEqual(run(self.spec), 1)
            self.assertEqual(native.call_count, 1)
        self.assertFalse((self.stage / "inference").exists())
        self.assertEqual((self.stage / "batches/batch_001/child-exit-code.txt").read_text(), "1\n")
        with patch("batch_cosmos_embed_text.run_child") as native:
            with self.assertRaises(FileExistsError):
                run(self.spec)
            native.assert_not_called()

    def test_teardown_exit_without_outputs_is_not_success(self):
        prepare(self.spec, 2)
        with patch("batch_cosmos_embed_text.run_child",
                   return_value=subprocess.CompletedProcess([], 130)) as native:
            with self.assertRaises(FileNotFoundError):
                run(self.spec)
            self.assertEqual(native.call_count, 1)
        self.assertFalse((self.stage / "inference").exists())

    def test_nonfinite_child_is_rejected_before_next_launch(self):
        prepare(self.spec, 2)
        def corrupt(matrix, metadata):
            matrix[0, 0] = np.nan
        with patch("batch_cosmos_embed_text.run_child",
                   side_effect=lambda *a, **kw: self.native_output(*a, mutation=corrupt, **kw)) as native:
            with self.assertRaisesRegex(ValueError, "non-finite"):
                run(self.spec)
            self.assertEqual(native.call_count, 1)

    def test_checkpoint_mismatch_prevents_aggregate_completion(self):
        prepare(self.spec, 2)
        def corrupt(matrix, metadata):
            metadata["checkpoint"] = "/wrong/model"
        with patch("batch_cosmos_embed_text.run_child",
                   side_effect=lambda *a, **kw: self.native_output(*a, mutation=corrupt, **kw)):
            with self.assertRaisesRegex(ValueError, "checkpoint mismatch"):
                run(self.spec)
        self.assertFalse((self.stage / "inference").exists())

    def test_full_spec_change_stops_before_launch(self):
        prepare(self.spec, 2)
        self.payload["inference"]["num_gpus"] = 8
        self.write_spec()
        with patch("batch_cosmos_embed_text.run_child") as native:
            with self.assertRaisesRegex(ValueError, "Full spec changed"):
                run(self.spec)
            native.assert_not_called()

    def test_edited_batch_is_rejected_even_with_updated_hash(self):
        plan_path = prepare(self.spec, 2)
        plan = json.loads(plan_path.read_text())
        path = Path(plan["batches"][0]["spec"])
        spec = yaml.safe_load(path.read_text())
        spec["inference"]["query"]["input_texts"].reverse()
        path.write_text(yaml.safe_dump(spec))
        plan["batches"][0]["sha256"] = file_sha256(path)
        plan_path.write_text(json.dumps(plan))
        with patch("batch_cosmos_embed_text.run_child") as native:
            with self.assertRaisesRegex(ValueError, "query slice"):
                run(self.spec)
            native.assert_not_called()

    def test_missing_batch_is_rejected_before_launch(self):
        plan_path = prepare(self.spec, 2)
        plan = json.loads(plan_path.read_text())
        plan["batches"].pop()
        plan_path.write_text(json.dumps(plan))
        with patch("batch_cosmos_embed_text.run_child") as native:
            with self.assertRaisesRegex(ValueError, "cover the full query list"):
                run(self.spec)
            native.assert_not_called()

    def test_invalid_chunk_sizes_make_no_batch_directories(self):
        for value in (0, -1, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                prepare(self.spec, value)
        self.assertFalse((self.stage / "batches").exists())

    def test_video_mode_is_not_batched(self):
        self.payload["inference"]["mode"] = "video"
        self.payload["inference"]["query"]["input_videos"] = ["/data/video.mp4"]
        self.write_spec()
        with self.assertRaisesRegex(ValueError, "mode=text"):
            prepare(self.spec, 2)

    def test_cancellation_during_aggregate_commit_invalidates_completion(self):
        prepare(self.spec, 2)
        previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        def cancel_merge(*args):
            merge_outputs(*args)
            os.kill(os.getpid(), signal.SIGINT)
        with patch("batch_cosmos_embed_text.run_child", side_effect=self.native_output), \
             patch("batch_cosmos_embed_text.merge_outputs", side_effect=cancel_merge):
            self.assertEqual(run(self.spec), 130)
        self.assertFalse((self.stage / "inference/completion_validation.json").exists())
        self.assertTrue((self.stage / "batch-cancellation.json").is_file())
        for code in (0, 130):
            with self.assertRaisesRegex(ValueError, "canceled"):
                validate_completion(self.spec, code)
        with self.assertRaisesRegex(ValueError, "canceled"):
            check_completion(self.spec)
        for sig, handler in previous_handlers.items():
            self.assertEqual(signal.getsignal(sig), handler)


if __name__ == "__main__":
    unittest.main()
