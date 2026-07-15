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

"""
Golden test set — regression tests for pipeline retrieval quality.

Each test case specifies a query + user parameters and asserts that the top
results contain expected elements/patterns (or don't contain unwanted ones).
Run this after any change to the retriever or data to ensure quality doesn't
regress.
"""

import os
import sys
import unittest

SCRIPT_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")
sys.path.insert(0, SCRIPT_DIR)

from generate_pipeline import BM25Retriever

DATA_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "data.csv")


def _meta(platform="unknown", source="unknown", sink="unknown",
          inference="unknown", tracker="none", num_sources=1):
    return {
        "platform": platform,
        "source_type": source,
        "sink_type": sink,
        "inference": inference,
        "tracker": tracker,
        "num_sources": num_sources,
    }


class GoldenTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = BM25Retriever()
        cls.retriever.load_csv(DATA_CSV)

    def assert_top_n_contain(self, query, user_meta, expected_in, n=3,
                             not_expected=None):
        """Assert at least one of the top-N results contains all expected_in strings."""
        results, _ = self.retriever.retrieve(query, user_meta=user_meta)
        top = results[:n]
        self.assertGreater(len(top), 0, f"No results for query: {query}")

        for expected in expected_in:
            found = any(expected in r["pipeline"].lower() for r in top)
            self.assertTrue(found, f"Expected '{expected}' in top-{n} results for: {query}")

        if not_expected:
            for bad in not_expected:
                top_pipelines = " ".join(r["pipeline"].lower() for r in top[:1])
                self.assertNotIn(bad, top_pipelines,
                                 f"Did NOT expect '{bad}' in #1 result for: {query}")

    def assert_platform_match(self, query, user_meta, expected_platform, n=3):
        """Assert top-N results prefer the expected platform."""
        results, _ = self.retriever.retrieve(query, user_meta=user_meta)
        top = results[:n]
        platform_matches = sum(
            1 for r in top if r["metadata"]["platform"] == expected_platform
        )
        self.assertGreater(platform_matches, 0,
                           f"No {expected_platform} results in top-{n} for: {query}")


class TestFlipRotate(GoldenTestBase):
    def test_flip_90_ccw(self):
        self.assert_top_n_contain(
            "flip video 90 degrees counter-clockwise",
            _meta(platform="dgpu", source="video", sink="display"),
            ["flip-method=1"],
        )

    def test_rotate_180(self):
        self.assert_top_n_contain(
            "rotate incoming video by 180 degrees",
            _meta(platform="dgpu", source="video", sink="display"),
            ["flip-method=2"],
        )

    def test_flip_horizontal(self):
        self.assert_top_n_contain(
            "flip video horizontally",
            _meta(platform="dgpu", source="video", sink="display"),
            ["flip-method=4"],
        )


class TestColorConversion(GoldenTestBase):
    def test_bgr10a2_to_i420(self):
        self.assert_top_n_contain(
            "convert BGR10A2_LE to I420 and display",
            _meta(platform="dgpu", source="test", sink="display"),
            ["bgr10a2_le", "i420"],
        )

    def test_bgr10a2_to_rgba(self):
        self.assert_top_n_contain(
            "convert BGR10A2_LE to RGBA and display",
            _meta(platform="dgpu", source="test", sink="display"),
            ["bgr10a2_le", "rgba"],
        )


class TestPrimaryInference(GoldenTestBase):
    def test_single_stream_display_dgpu(self):
        self.assert_top_n_contain(
            "primary inference on a single mp4 video and display the output",
            _meta(platform="dgpu", source="video", sink="display",
                  inference="primary", num_sources=1),
            ["nvinfer", "config-file-path", "nveglglessink"],
            not_expected=["nv3dsink"],
        )

    def test_single_stream_display_jetson(self):
        self.assert_top_n_contain(
            "primary inference on a single mp4 video and display the output on Jetson",
            _meta(platform="jetson", source="video", sink="display",
                  inference="primary", num_sources=1),
            ["nvinfer", "nv3dsink"],
        )
        self.assert_platform_match(
            "primary inference on Jetson",
            _meta(platform="jetson"), "jetson",
        )


class TestSecondaryInference(GoldenTestBase):
    def test_primary_secondary_display(self):
        self.assert_top_n_contain(
            "primary and secondary inference on a single stream and display",
            _meta(platform="dgpu", source="video", sink="display",
                  inference="primary+secondary"),
            ["nvinfer", "infer-on-gie-id"],
        )


class TestTracker(GoldenTestBase):
    def test_inference_with_tracker(self):
        self.assert_top_n_contain(
            "primary and secondary inference with tracker on a single stream",
            _meta(platform="dgpu", source="video", sink="display",
                  inference="primary+secondary", tracker="present"),
            ["nvtracker"],
        )


class TestMultiStream(GoldenTestBase):
    def test_4_stream_inference(self):
        self.assert_top_n_contain(
            "primary inference on 4 video streams and display",
            _meta(platform="dgpu", source="video", sink="display",
                  inference="primary", num_sources=4),
            ["m.sink_0", "batch-size=4"],
            n=5,
        )


class TestSaveOutput(GoldenTestBase):
    def test_save_to_mp4(self):
        self.assert_top_n_contain(
            "encode video to h264 and mux into mp4 file",
            _meta(platform="dgpu", source="video", sink="save-mp4"),
            ["filesink", "qtmux"],
            n=5,
        )

    def test_save_h264(self):
        self.assert_top_n_contain(
            "encode test video to h264 and save to file",
            _meta(platform="dgpu", source="test", sink="save-file"),
            ["nvv4l2h264enc", "filesink"],
        )


class TestPreprocessor(GoldenTestBase):
    def test_preprocess_before_primary(self):
        self.assert_top_n_contain(
            "preprocess before primary inference on a single stream",
            _meta(platform="dgpu", source="video", sink="display",
                  inference="primary+preprocess"),
            ["nvdspreprocess", "input-tensor-meta=1"],
        )


class TestResize(GoldenTestBase):
    def test_resize_video(self):
        self.assert_top_n_contain(
            "resize incoming video to specific height and width",
            _meta(platform="dgpu", source="video", sink="display"),
            ["nvvideoconvert"],
        )


class TestCrop(GoldenTestBase):
    def test_crop_video(self):
        self.assert_top_n_contain(
            "crop the incoming video frame",
            _meta(platform="dgpu", source="video", sink="display"),
            ["src-crop"],
        )


class TestDewarp(GoldenTestBase):
    def test_dewarp_dgpu(self):
        self.assert_top_n_contain(
            "dewarp a given mp4 video on dGPU",
            _meta(platform="dgpu", source="video", sink="display"),
            ["nvdewarper"],
        )


class TestPlatformFiltering(GoldenTestBase):
    def test_dgpu_query_prefers_dgpu(self):
        self.assert_platform_match(
            "primary inference on video and display output",
            _meta(platform="dgpu", source="video", sink="display",
                  inference="primary"),
            "dgpu",
        )

    def test_jetson_query_prefers_jetson(self):
        self.assert_platform_match(
            "primary inference on video and display output on Jetson",
            _meta(platform="jetson", source="video", sink="display",
                  inference="primary"),
            "jetson",
        )


if __name__ == "__main__":
    unittest.main()
