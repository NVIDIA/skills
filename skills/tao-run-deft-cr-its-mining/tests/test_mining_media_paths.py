#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for real-path gap records and symlinked media lookups."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from prepare_nearest_neighbor_mining import (  # noqa: E402
    build_iteration_target,
    media_match_key,
    text_target_dataframe,
    weak_samples,
)


class MiningMediaPathsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.real = self.root / "source" / "clip.mp4"
        self.real.parent.mkdir()
        self.real.touch()
        self.alias = self.root / "workspace-media"
        self.alias.symlink_to(self.real.parent, target_is_directory=True)
        self.video = str(self.alias / self.real.name)
        self.question_path = str(self.root / "questions" / "q_0.txt")
        self.gaps = self.root / "gaps.jsonl"
        self.gaps.write_text(json.dumps({
            "video_id": str(self.real), "question": "Is there a collision?\nAnswer with Yes or No.",
            "ground_truth": "yes", "response": "no", "error_type": "FN",
        }) + "\n")
        self.lookup = self.root / "lookup.parquet"
        pd.DataFrame([{
            "filepath": self.question_path, "video_path": self.video,
            "question": "<video>\nIs there a collision?", "annotation_id": "original-id",
        }]).to_parquet(self.lookup)
        self.embeddings = self.root / "embeddings.parquet"
        pd.DataFrame([
            {"filepath": self.question_path, "modality": "text", "embedding": [1.0, 2.0]},
            {"filepath": self.video, "modality": "video", "embedding": [3.0, 4.0]},
        ]).to_parquet(self.embeddings)

    def test_real_gaps_match_symlinked_text_and_video_without_rewriting(self):
        before = {p: p.read_bytes() for p in (self.gaps, self.lookup, self.embeddings)}
        output = self.root / "target.parquet"
        count = build_iteration_target(
            self.gaps, self.embeddings, self.lookup, ["text", "video"], output,
        )
        self.assertEqual(count, 2)
        target = pd.read_parquet(output)
        self.assertEqual(target.filepath.tolist(), [self.question_path, self.video])
        self.assertEqual([list(v) for v in target.embedding], [[1.0, 2.0], [3.0, 4.0]])
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)

    def test_symlink_gap_matches_real_lookup(self):
        lookup = pd.read_parquet(self.lookup)
        lookup["video_path"] = str(self.real)
        lookup.to_parquet(self.lookup)
        row = json.loads(self.gaps.read_text())
        row["video_id"] = self.video
        self.gaps.write_text(json.dumps(row) + "\n")
        target = text_target_dataframe(self.gaps, self.embeddings, self.lookup)
        self.assertEqual(target.filepath.tolist(), [self.question_path])

    def test_duplicate_path_aliases_do_not_duplicate_weak_samples(self):
        row = json.loads(self.gaps.read_text())
        row["video_id"] = self.video
        with self.gaps.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        self.assertEqual(len(weak_samples(self.gaps)), 1)

    def test_different_files_with_same_basename_do_not_match(self):
        other = self.root / "unrelated" / self.real.name
        other.parent.mkdir()
        other.touch()
        row = json.loads(self.gaps.read_text())
        row["video_id"] = str(other)
        self.gaps.write_text(json.dumps(row) + "\n")
        with self.assertRaisesRegex(RuntimeError, "no KPI text embeddings matched"):
            text_target_dataframe(self.gaps, self.embeddings, self.lookup)

    def test_unavailable_remote_path_keeps_lexical_matching(self):
        missing = str(self.root / "remote-only" / "clip.mp4")
        self.assertEqual(media_match_key(missing), missing)
        self.assertEqual(media_match_key(self.video), str(self.real.resolve()))

    def test_unresolvable_path_keeps_lexical_matching(self):
        path = str(self.root / "restricted" / "clip.mp4")
        for error in (PermissionError("root-squash"), RuntimeError("symlink loop")):
            with self.subTest(error=type(error).__name__), \
                    patch("prepare_nearest_neighbor_mining.Path.resolve", side_effect=error):
                self.assertEqual(media_match_key(path), path)


if __name__ == "__main__":
    unittest.main()
