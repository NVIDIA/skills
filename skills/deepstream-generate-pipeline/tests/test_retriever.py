#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the BM25 pipeline retriever."""

import os
import sys
import unittest

SCRIPT_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")
sys.path.insert(0, SCRIPT_DIR)

from generate_pipeline import (
    BM25Retriever,
    expand_with_synonyms,
    extract_pipeline_metadata,
    normalize_user_params,
    tokenize,
)

DATA_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "data.csv")


class TestTokenizer(unittest.TestCase):
    def test_lowercase(self):
        self.assertEqual(tokenize("Hello World"), ["hello", "world"])

    def test_strips_punctuation(self):
        tokens = tokenize("nvinfer config-file-path=/foo/bar.txt")
        self.assertIn("nvinfer", tokens)
        self.assertIn("config-file-path", tokens)

    def test_preserves_compound_tokens_splits_paths(self):
        """
        The tokenizer keeps compound hyphenated
        tokens (config-file-path) intact but splits on '/' so a path like
        /opt/nvidia/deepstream/config.txt yields useful per-component tokens
        ('opt', 'nvidia', 'deepstream', 'config.txt') for IDF rather than
        collapsing into one giant unique token.
        """
        tokens = tokenize("config-file-path=/opt/nvidia/deepstream/config.txt")
        self.assertIn("config-file-path", tokens)
        # Path components must be split out as separate tokens.
        self.assertIn("opt", tokens)
        self.assertIn("nvidia", tokens)
        self.assertIn("deepstream", tokens)
        self.assertIn("config.txt", tokens)
        # The whole path must NOT appear as a single token.
        self.assertNotIn("/opt/nvidia/deepstream/config.txt", tokens)


class TestSynonymExpansion(unittest.TestCase):
    def test_infer_expands(self):
        expanded = expand_with_synonyms(["infer"])
        self.assertIn("inference", expanded)
        self.assertIn("nvinfer", expanded)
        self.assertIn("detect", expanded)

    def test_no_duplicates(self):
        expanded = expand_with_synonyms(["infer", "inference"])
        counts = {}
        for tok in expanded:
            counts[tok] = counts.get(tok, 0) + 1
        for tok, count in counts.items():
            self.assertEqual(count, 1, f"Duplicate token: {tok}")

    def test_unknown_token_unchanged(self):
        expanded = expand_with_synonyms(["xyzzy123"])
        self.assertEqual(expanded, ["xyzzy123"])


class TestPipelineMetadata(unittest.TestCase):
    def test_dgpu_display(self):
        meta = extract_pipeline_metadata(
            "gst-launch-1.0 filesrc location=a.mp4 ! qtdemux ! h264parse ! "
            "nvv4l2decoder ! nvinfer config-file-path=c.txt ! nveglglessink"
        )
        self.assertEqual(meta["platform"], "dgpu")
        self.assertEqual(meta["sink_type"], "display")
        self.assertEqual(meta["source_type"], "video")
        self.assertEqual(meta["inference"], "primary")
        self.assertEqual(meta["tracker"], "none")

    def test_jetson_display(self):
        meta = extract_pipeline_metadata(
            "gst-launch-1.0 filesrc location=a.mp4 ! nvinfer config-file-path=c.txt ! nv3dsink"
        )
        self.assertEqual(meta["platform"], "jetson")
        self.assertEqual(meta["sink_type"], "display")

    def test_primary_secondary_with_tracker(self):
        meta = extract_pipeline_metadata(
            "nvinfer config-file-path=p.txt ! nvtracker ll-lib-file=t.so ! "
            "nvinfer config-file-path=s.txt infer-on-gie-id=1 ! nveglglessink"
        )
        self.assertEqual(meta["inference"], "primary+secondary")
        self.assertEqual(meta["tracker"], "present")

    def test_multi_source_detection(self):
        meta = extract_pipeline_metadata(
            "filesrc location=a.mp4 ! m.sink_0 nvstreammux name=m batch-size=4 ! "
            "nveglglessink filesrc location=b.mp4 ! m.sink_1 "
            "filesrc location=c.mp4 ! m.sink_2 "
            "filesrc location=d.mp4 ! m.sink_3"
        )
        self.assertEqual(meta["num_sources"], 4)

    def test_rtsp_source(self):
        meta = extract_pipeline_metadata(
            "uridecodebin uri=rtsp://<rtsp-server>:<port>/stream ! nveglglessink"
        )
        self.assertEqual(meta["source_type"], "rtsp")

    def test_save_mp4_sink(self):
        meta = extract_pipeline_metadata(
            "filesrc location=a.mp4 ! nvv4l2h264enc ! h264parse ! qtmux ! filesink location=out.mp4"
        )
        self.assertEqual(meta["sink_type"], "save-mp4")

    def test_image_source(self):
        meta = extract_pipeline_metadata(
            "filesrc location=img.jpg ! jpegparse ! nvjpegdec ! nveglglessink"
        )
        self.assertEqual(meta["source_type"], "image")

    def test_preprocess_inference(self):
        meta = extract_pipeline_metadata(
            "nvdspreprocess config-file=c.txt ! nvinfer config-file-path=p.txt input-tensor-meta=1 ! nveglglessink"
        )
        self.assertEqual(meta["inference"], "primary+preprocess")

    def test_fakesink(self):
        meta = extract_pipeline_metadata("videotestsrc ! fakesink")
        self.assertEqual(meta["source_type"], "test")
        self.assertEqual(meta["sink_type"], "fakesink")


class TestBM25Retriever(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = BM25Retriever()
        cls.retriever.load_csv(DATA_CSV)

    def test_loads_documents(self):
        self.assertGreater(len(self.retriever.documents), 200)

    def test_idf_computed(self):
        self.assertGreater(len(self.retriever.idf), 0)
        self.assertIn("nvinfer", self.retriever.idf)

    def test_avgdl_positive(self):
        self.assertGreater(self.retriever.avgdl, 0)

    def test_retrieve_returns_results(self):
        results, conf = self.retriever.retrieve("flip video 90 degrees")
        self.assertGreater(len(results), 0)

    def test_retrieve_scores_descending(self):
        results, _ = self.retriever.retrieve("primary inference on mp4 video display")
        scores = [r["score"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_retrieve_contains_metadata(self):
        results, _ = self.retriever.retrieve("primary inference")
        for r in results:
            self.assertIn("metadata", r)
            self.assertIn("platform", r["metadata"])

    def test_confidence_levels(self):
        _, conf = self.retriever.retrieve(
            "primary inference on a single mp4 video and display the output"
        )
        self.assertIn(conf, ("high", "medium", "low"))

    def test_structural_boost_prefers_matching_platform(self):
        dgpu_meta = {
            "platform": "dgpu", "source_type": "video", "sink_type": "display",
            "inference": "primary", "tracker": "none", "num_sources": 1,
        }
        jetson_meta = {
            "platform": "jetson", "source_type": "video", "sink_type": "display",
            "inference": "primary", "tracker": "none", "num_sources": 1,
        }
        query = "primary inference on video and display output"
        dgpu_results, _ = self.retriever.retrieve(query, user_meta=dgpu_meta)
        jetson_results, _ = self.retriever.retrieve(query, user_meta=jetson_meta)

        if dgpu_results and jetson_results:
            dgpu_top_platform = dgpu_results[0]["metadata"]["platform"]
            jetson_top_platform = jetson_results[0]["metadata"]["platform"]
            self.assertEqual(dgpu_top_platform, "dgpu")
            self.assertEqual(jetson_top_platform, "jetson")

    def test_empty_query_returns_empty(self):
        results, conf = self.retriever.retrieve("")
        self.assertEqual(conf, "low")


class TestIndexCache(unittest.TestCase):
    """
    Regression: the BM25 index is cached to a sibling
    JSON file keyed on (CSV mtime, internal schema version). A cached
    retriever must (a) reproduce the same indexed state as a fresh one and
    (b) be invalidated when the CSV is touched.
    """

    def setUp(self):
        import shutil
        import tempfile
        self.tmpdir = tempfile.mkdtemp(prefix="ds-pg-cache-test-")
        self.csv_copy = os.path.join(self.tmpdir, "data.csv")
        shutil.copyfile(DATA_CSV, self.csv_copy)
        self.cache_path = self.csv_copy + ".cache.json"

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_cache_written_on_first_load(self):
        self.assertFalse(os.path.exists(self.cache_path))
        retriever = BM25Retriever()
        retriever.load_csv(self.csv_copy)
        self.assertTrue(os.path.exists(self.cache_path))

    def test_second_load_reproduces_state_from_cache(self):
        first = BM25Retriever()
        first.load_csv(self.csv_copy)
        second = BM25Retriever()
        second.load_csv(self.csv_copy)
        # Same documents, same IDF, same average doc length — proves the
        # cached state round-trips through JSON correctly.
        self.assertEqual(len(first.documents), len(second.documents))
        self.assertEqual(first.idf, second.idf)
        self.assertEqual(first.avgdl, second.avgdl)
        self.assertEqual(len(first.doc_tokens), len(second.doc_tokens))

    def test_cache_invalidated_when_csv_mtime_changes(self):
        retriever = BM25Retriever()
        retriever.load_csv(self.csv_copy)
        cache_mtime_before = os.path.getmtime(self.cache_path)

        # Touch the CSV so its mtime advances (sleep ensures the new mtime
        # exceeds filesystem resolution).
        import time
        time.sleep(1.1)
        os.utime(self.csv_copy, None)

        retriever2 = BM25Retriever()
        retriever2.load_csv(self.csv_copy)
        cache_mtime_after = os.path.getmtime(self.cache_path)
        # Cache was rewritten, not reused.
        self.assertGreater(cache_mtime_after, cache_mtime_before)

    def test_cache_invalidated_on_schema_version_bump(self):
        import json
        retriever = BM25Retriever()
        retriever.load_csv(self.csv_copy)
        # Rewrite the cache with an obviously-stale version stamp; the next
        # load must reject it and rebuild from CSV.
        with open(self.cache_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        payload["version"] = -999
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

        retriever2 = BM25Retriever()
        retriever2.load_csv(self.csv_copy)
        # Rebuilt cache must have the current schema version, not the bogus one.
        with open(self.cache_path, "r", encoding="utf-8") as f:
            new_payload = json.load(f)
        self.assertNotEqual(new_payload["version"], -999)


class TestNormalizeUserParams(unittest.TestCase):
    def test_basic_mapping(self):
        class Args:
            source_type = "Local video file"
            num_sources = 1
            inference = "primary"
            tracker = "none"
            sink = "display"
            platform = "dGPU"

        meta = normalize_user_params(Args())
        self.assertEqual(meta["platform"], "dgpu")
        self.assertEqual(meta["source_type"], "video")
        self.assertEqual(meta["sink_type"], "display")
        self.assertEqual(meta["inference"], "primary")
        self.assertEqual(meta["tracker"], "none")

    def test_jetson_mapping(self):
        class Args:
            source_type = "RTSP stream"
            num_sources = 4
            inference = "primary+secondary"
            tracker = "NvDCF"
            sink = "save-mp4"
            platform = "Jetson"

        meta = normalize_user_params(Args())
        self.assertEqual(meta["platform"], "jetson")
        self.assertEqual(meta["source_type"], "rtsp")
        self.assertEqual(meta["tracker"], "present")


if __name__ == "__main__":
    unittest.main()
