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
End-to-end CLI tests for the --format flag on generate_pipeline.py and
validate_pipeline.py. Run the scripts as subprocesses so the full surface
(argparse, formatting branches, exit codes) is exercised.
"""

import json
import os
import subprocess
import sys
import unittest

SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
GENERATE = os.path.join(SCRIPTS_DIR, "generate_pipeline.py")
VALIDATE = os.path.join(SCRIPTS_DIR, "validate_pipeline.py")

GENERATE_BASE_ARGS = [
    "--query", "infer on 3 videos and display",
    "--source-type", "Local video file",
    "--num-sources", "3",
    "--inference", "primary",
    "--tracker", "none",
    "--sink", "display",
    "--platform", "dGPU",
    "--extras", "none",
]

VALID_PIPELINE = (
    "gst-launch-1.0 videotestsrc num-buffers=1 ! "
    "video/x-raw,format=NV12,width=320,height=240 ! fakesink"
)
INVALID_PIPELINE = "gst-launch-1.0 filesrc ! ! fakesink"


def run_generate(extra_args=None):
    args = [sys.executable, GENERATE] + GENERATE_BASE_ARGS + (extra_args or [])
    return subprocess.run(args, capture_output=True, text=True, timeout=30)


def run_validate(pipeline, extra_args=None):
    args = [sys.executable, VALIDATE, pipeline] + (extra_args or [])
    return subprocess.run(args, capture_output=True, text=True, timeout=30)


class TestGenerateFormat(unittest.TestCase):
    """generate_pipeline.py --format {json,compact,summary}"""

    def test_default_is_full_json(self):
        """Default (no --format) must remain full JSON for backward compat."""
        result = run_generate()
        self.assertEqual(result.returncode, 0)
        payload = json.loads(result.stdout)
        # Full JSON includes the retrieved_pipelines list (~10 entries)
        self.assertIn("retrieved_pipelines", payload)
        self.assertIsInstance(payload["retrieved_pipelines"], list)
        self.assertIn("confidence", payload)
        self.assertIn("context", payload)

    def test_format_json_matches_default(self):
        """--format=json explicitly should match the default behavior."""
        default = run_generate()
        explicit = run_generate(["--format", "json"])
        self.assertEqual(default.stdout, explicit.stdout)

    def test_format_compact_is_smaller_json(self):
        """--format=compact returns JSON with top_pipeline only, not retrieved_pipelines."""
        result = run_generate(["--format", "compact"])
        self.assertEqual(result.returncode, 0)
        payload = json.loads(result.stdout)
        self.assertIn("top_pipeline", payload)
        self.assertNotIn("retrieved_pipelines", payload)
        self.assertIn("confidence", payload)
        self.assertIn("context", payload)
        # top_pipeline is either a result dict or null
        if payload["top_pipeline"] is not None:
            self.assertIn("pipeline", payload["top_pipeline"])
            self.assertIn("score", payload["top_pipeline"])

    def test_format_summary_is_single_line(self):
        """--format=summary prints one line: confidence=… retrieved=… top_match=…"""
        result = run_generate(["--format", "summary"])
        self.assertEqual(result.returncode, 0)
        lines = result.stdout.strip().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertIn("confidence=", lines[0])
        self.assertIn("retrieved=", lines[0])
        self.assertIn("top_match=", lines[0])

    def test_compact_smaller_than_json(self):
        """compact JSON output should be substantially smaller than full json."""
        full = run_generate(["--format", "json"])
        compact = run_generate(["--format", "compact"])
        self.assertLess(len(compact.stdout), len(full.stdout))
        # compact should be at most 1/3 the size for typical queries
        self.assertLess(len(compact.stdout) * 3, len(full.stdout) + 200)

    def test_invalid_format_rejected(self):
        """argparse should reject an unknown --format value."""
        result = run_generate(["--format", "bogus"])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid choice", result.stderr.lower())


class TestValidateFormat(unittest.TestCase):
    """validate_pipeline.py --format {json,summary}"""

    def test_default_is_full_json(self):
        """Default (no --format) must remain full JSON for backward compat."""
        result = run_validate(VALID_PIPELINE)
        # videotestsrc → fakesink may emit gst-launch dry-run warnings; valid still True
        payload = json.loads(result.stdout)
        self.assertIn("valid", payload)
        self.assertIn("elements_found", payload)
        self.assertIn("errors", payload)
        self.assertIn("warnings", payload)

    def test_format_json_matches_default(self):
        default = run_validate(VALID_PIPELINE)
        explicit = run_validate(VALID_PIPELINE, ["--format", "json"])
        self.assertEqual(default.stdout, explicit.stdout)

    def test_format_summary_valid_is_one_line(self):
        result = run_validate(VALID_PIPELINE, ["--format", "summary"])
        lines = result.stdout.strip().splitlines()
        # Summary header is exactly one line; warnings would be indented after.
        self.assertGreaterEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith("valid · "))
        self.assertIn("elements", lines[0])
        self.assertIn("warnings", lines[0])

    def test_format_summary_invalid_lists_errors(self):
        result = run_validate(INVALID_PIPELINE, ["--format", "summary"])
        self.assertNotEqual(result.returncode, 0)
        out = result.stdout
        self.assertTrue(out.startswith("invalid · "))
        # Errors are indented under the header
        self.assertIn("\n  error: ", out)

    def test_format_summary_multistream_notes_skipped_live_parse(self):
        """Multi-stream pipelines should mention 'live-parse skipped' in summary."""
        multistream = (
            "gst-launch-1.0 -e "
            "filesrc location=/dev/null ! qtdemux ! h264parse ! nvv4l2decoder ! m.sink_0 "
            "filesrc location=/dev/null ! qtdemux ! h264parse ! nvv4l2decoder ! m.sink_1 "
            "nvstreammux name=m batch-size=2 width=1920 height=1080 ! "
            "nvinfer config-file-path=/dev/null batch-size=2 ! "
            "nvvideoconvert ! nvdsosd ! nveglglessink"
        )
        result = run_validate(multistream, ["--format", "summary"])
        # May be valid or invalid depending on env; either way the summary line
        # should mention live-parse skip when pad refs are present.
        self.assertIn("live-parse skipped", result.stdout)

    def test_invalid_format_rejected(self):
        result = run_validate(VALID_PIPELINE, ["--format", "bogus"])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid choice", result.stderr.lower())

    def test_dash_e_flag_accepted(self):
        """The Step 5 generated script uses 'gst-launch-1.0 -e' — validator must accept."""
        with_e = "gst-launch-1.0 -e videotestsrc num-buffers=1 ! fakesink"
        result = run_validate(with_e, ["--format", "summary"])
        # The -e flag must be stripped cleanly — the validator should report
        # success (returncode == 0). Surface stdout/stderr in the failure
        # message so a regression is debuggable from the test output alone.
        self.assertEqual(
            result.returncode, 0,
            msg=(
                f"validator rejected 'gst-launch-1.0 -e' pipeline "
                f"(returncode={result.returncode}).\n"
                f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
            ),
        )
        # First line of summary must not be the 'invalid · …' header.
        self.assertNotIn("invalid · ", result.stdout.split("\n")[0])
        # The summary should report a non-zero element count.
        self.assertRegex(result.stdout, r"\d+ elements")


if __name__ == "__main__":
    unittest.main()
