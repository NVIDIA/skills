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

"""Unit tests for the pipeline validator."""

import os
import re
import sys
import unittest

SCRIPT_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")
sys.path.insert(0, SCRIPT_DIR)

from validate_pipeline import (
    extract_elements_and_properties,
    validate_element_ordering,
    validate_memory_format,
    validate_pipeline_structure,
    validate_platform_sink,
    validate_syntax,
    validate_with_gst_launch,
)


class TestSyntaxValidation(unittest.TestCase):
    def test_valid_pipeline(self):
        errors, warnings = validate_syntax(
            "gst-launch-1.0 filesrc location=test.mp4 ! nveglglessink"
        )
        self.assertEqual(errors, [])

    def test_unbalanced_single_quotes(self):
        errors, _ = validate_syntax("gst-launch-1.0 filesrc location='test.mp4 ! nveglglessink")
        self.assertTrue(any("single quotes" in e for e in errors))

    def test_unbalanced_double_quotes(self):
        errors, _ = validate_syntax('gst-launch-1.0 filesrc location="test.mp4 ! nveglglessink')
        self.assertTrue(any("double quotes" in e for e in errors))

    def test_empty_pipe_segment(self):
        errors, _ = validate_syntax("gst-launch-1.0 filesrc location=t.mp4 ! ! nveglglessink")
        self.assertTrue(any("Empty pipe" in e for e in errors))

    def test_leading_pipe(self):
        errors, _ = validate_syntax("! filesrc location=t.mp4 ! nveglglessink")
        self.assertTrue(any("starts with" in e for e in errors))

    def test_trailing_pipe(self):
        errors, _ = validate_syntax("gst-launch-1.0 filesrc location=t.mp4 !")
        self.assertTrue(any("ends with" in e for e in errors))

    def test_underscore_property_typo(self):
        errors, _ = validate_syntax(
            "gst-launch-1.0 nvinfer config_file_path=c.txt ! nveglglessink"
        )
        self.assertTrue(any("hyphens" in e for e in errors))

    def test_underscore_typo_not_masked_by_hyphen_form_elsewhere(self):
        """
        Regression: the underscore-typo check must
        scope to the segment that owns nvinfer. A hyphenated config-file-path
        on a different element (here nvdspreprocess) must not silently mask
        an underscore typo on the actual nvinfer.
        """
        errors, _ = validate_syntax(
            "gst-launch-1.0 nvdspreprocess config-file=p.txt ! "
            "nvinfer config_file_path=c.txt ! nveglglessink"
        )
        self.assertTrue(any("hyphens" in e for e in errors))

    def test_streammux_without_name(self):
        _, warnings = validate_syntax(
            "gst-launch-1.0 filesrc location=t.mp4 ! nvstreammux batch-size=1 ! nveglglessink"
        )
        self.assertTrue(any("name=" in w for w in warnings))


class TestElementParser(unittest.TestCase):
    def test_simple_pipeline(self):
        elements, pad_refs = extract_elements_and_properties(
            "gst-launch-1.0 filesrc location=test.mp4 ! nveglglessink"
        )
        names = [e["name"] for e in elements]
        self.assertIn("filesrc", names)
        self.assertIn("nveglglessink", names)
        self.assertEqual(pad_refs, [])

    def test_properties_extracted(self):
        elements, _ = extract_elements_and_properties(
            "nvinfer config-file-path=c.txt batch-size=4 unique-id=1 ! nveglglessink"
        )
        nvinfer = [e for e in elements if e["name"] == "nvinfer"][0]
        self.assertEqual(nvinfer["properties"]["config-file-path"], "c.txt")
        self.assertEqual(nvinfer["properties"]["batch-size"], "4")

    def test_pad_refs_extracted(self):
        elements, pad_refs = extract_elements_and_properties(
            "filesrc location=a.mp4 ! qtdemux ! h264parse ! nvv4l2decoder ! "
            "m.sink_0 nvstreammux name=m batch-size=1 width=1920 height=1080 ! nveglglessink"
        )
        self.assertIn("m.sink_0", pad_refs)
        names = [e["name"] for e in elements]
        self.assertIn("nvstreammux", names)

    def test_multiple_pad_refs(self):
        # Both branches end with a pad ref followed by a ' ! ' separator so
        # the parser sees each pad ref as the leading token of its own
        # segment — making the assertion on m.sink_0 AND m.sink_1
        # deterministic.
        elements, pad_refs = extract_elements_and_properties(
            "filesrc location=a.mp4 ! m.sink_0 "
            "filesrc location=b.mp4 ! m.sink_1 "
            "nvstreammux name=m batch-size=2 width=1920 height=1080 ! nveglglessink"
        )
        self.assertIn("m.sink_0", pad_refs)
        self.assertIn("m.sink_1", pad_refs)

    def test_caps_filter_skipped(self):
        elements, _ = extract_elements_and_properties(
            "videotestsrc ! 'video/x-raw,format=NV12' ! nveglglessink"
        )
        names = [e["name"] for e in elements]
        self.assertNotIn("video/x-raw,format=NV12", names)

    def test_gst_launch_prefix_stripped(self):
        elements, _ = extract_elements_and_properties(
            "gst-launch-1.0 filesrc location=t.mp4 ! nveglglessink"
        )
        self.assertNotIn("gst-launch-1.0", [e["name"] for e in elements])

    def test_flags_stripped(self):
        elements, _ = extract_elements_and_properties(
            "gst-launch-1.0 -e filesrc location=t.mp4 ! nveglglessink"
        )
        self.assertNotIn("-e", [e["name"] for e in elements])

    def test_multiple_flags_stripped(self):
        """
        Regression: a run of leading flags such as
        '-e -v' must be stripped together. Previously only the first flag was
        removed and the second was parsed as a phantom element.
        """
        elements, _ = extract_elements_and_properties(
            "gst-launch-1.0 -e -v filesrc location=t.mp4 ! nveglglessink"
        )
        names = [e["name"] for e in elements]
        self.assertNotIn("-e", names)
        self.assertNotIn("-v", names)
        self.assertIn("filesrc", names)

    def test_live_parse_substitution_anchors_to_segment_start(self):
        """
        Regression: the live-parse source/sink
        replacement must not match the element name when it appears inside a
        property value (e.g. a config-file-path that contains 'filesrc' as
        part of a directory or filename). Previously \\b{real}\\b matched the
        substring inside the path and corrupted the dry-run input. The fix
        anchors to segment-leading positions ((^|!\\s+)).
        """
        original = (
            "nvinfer config-file-path=/foo/filesrc.txt ! "
            "filesrc location=t.mp4 ! fakesink"
        )
        substituted = re.sub(
            r"(^|!\s+)filesrc\b[^!]*",
            r"\g<1>fakesrc num-buffers=1 ",
            original,
        )
        # The 'filesrc' inside the config-file-path must remain untouched.
        self.assertIn("config-file-path=/foo/filesrc.txt", substituted)
        # The actual filesrc element (after !) must be replaced with fakesrc.
        self.assertIn("fakesrc num-buffers=1", substituted)
        # The original filesrc element's location property should be gone.
        self.assertNotIn("location=t.mp4", substituted)


class TestStructureValidation(unittest.TestCase):
    def test_valid_pipeline(self):
        elements, pad_refs = extract_elements_and_properties(
            "filesrc location=t.mp4 ! nvinfer config-file-path=c.txt ! nveglglessink"
        )
        errors, warnings = validate_pipeline_structure(elements, pad_refs)
        self.assertEqual(errors, [])

    def test_missing_source_warning(self):
        elements, pad_refs = extract_elements_and_properties(
            "nvinfer config-file-path=c.txt ! nveglglessink"
        )
        _, warnings = validate_pipeline_structure(elements, pad_refs)
        self.assertTrue(any("source" in w.lower() for w in warnings))

    def test_missing_sink_warning(self):
        elements, pad_refs = extract_elements_and_properties(
            "filesrc location=t.mp4 ! nvinfer config-file-path=c.txt"
        )
        _, warnings = validate_pipeline_structure(elements, pad_refs)
        self.assertTrue(any("sink" in w.lower() for w in warnings))

    def test_nvinfer_missing_config(self):
        elements, pad_refs = extract_elements_and_properties(
            "filesrc location=t.mp4 ! nvinfer batch-size=1 ! nveglglessink"
        )
        errors, _ = validate_pipeline_structure(elements, pad_refs)
        self.assertTrue(any("config-file-path" in e for e in errors))

    def test_filesrc_missing_location(self):
        elements, pad_refs = extract_elements_and_properties(
            "filesrc ! nveglglessink"
        )
        errors, _ = validate_pipeline_structure(elements, pad_refs)
        self.assertTrue(any("location" in e for e in errors))

    def test_empty_pipeline(self):
        errors, _ = validate_pipeline_structure([], [])
        self.assertTrue(any("empty" in e.lower() for e in errors))

    def test_bad_pad_ref_warns(self):
        elements, pad_refs = extract_elements_and_properties(
            "filesrc location=a.mp4 ! x.sink_0 nveglglessink"
        )
        _, warnings = validate_pipeline_structure(elements, pad_refs)
        self.assertTrue(any("x.sink_0" in w for w in warnings))

    def test_valid_pad_ref_no_warning(self):
        elements, pad_refs = extract_elements_and_properties(
            "filesrc location=a.mp4 ! m.sink_0 nvstreammux name=m batch-size=1 ! nveglglessink"
        )
        _, warnings = validate_pipeline_structure(elements, pad_refs)
        pad_warnings = [w for w in warnings if "pad reference" in w.lower()]
        self.assertEqual(pad_warnings, [])

    def test_batch_size_pad_mismatch(self):
        elements, pad_refs = extract_elements_and_properties(
            "filesrc location=a.mp4 ! m.sink_0 nvstreammux name=m batch-size=4 width=1920 height=1080 ! nveglglessink"
        )
        _, warnings = validate_pipeline_structure(elements, pad_refs)
        self.assertTrue(any("batch-size=4" in w and "1 sink" in w for w in warnings))


class TestMemoryFormatValidation(unittest.TestCase):
    """Test memory format mismatch detection between elements."""

    def test_nvjpegdec_direct_to_mux_detected(self):
        pipeline = (
            "gst-launch-1.0 filesrc location=a.jpg ! jpegparse ! nvjpegdec ! "
            "m.sink_0 nvstreammux name=m batch-size=1 width=1920 height=1080 ! nveglglessink"
        )
        elements, pad_refs = extract_elements_and_properties(pipeline)
        errors = validate_memory_format(pipeline, elements, pad_refs)
        self.assertTrue(len(errors) > 0)
        self.assertTrue(any("nvjpegdec" in e and "NVMM" in e for e in errors))

    def test_nvjpegdec_with_videoconvert_passes(self):
        pipeline = (
            "gst-launch-1.0 filesrc location=a.jpg ! jpegparse ! nvjpegdec ! "
            "nvvideoconvert ! m.sink_0 nvstreammux name=m batch-size=1 width=1920 height=1080 ! nveglglessink"
        )
        elements, pad_refs = extract_elements_and_properties(pipeline)
        errors = validate_memory_format(pipeline, elements, pad_refs)
        self.assertEqual(errors, [])

    def test_nvv4l2decoder_to_mux_passes(self):
        pipeline = (
            "gst-launch-1.0 filesrc location=a.mp4 ! qtdemux ! h264parse ! nvv4l2decoder ! "
            "m.sink_0 nvstreammux name=m batch-size=1 width=1920 height=1080 ! nveglglessink"
        )
        elements, pad_refs = extract_elements_and_properties(pipeline)
        errors = validate_memory_format(pipeline, elements, pad_refs)
        self.assertEqual(errors, [])

    def test_nvjpegdec_to_display_no_error(self):
        pipeline = "gst-launch-1.0 filesrc location=a.jpg ! nvjpegdec ! nveglglessink"
        elements, pad_refs = extract_elements_and_properties(pipeline)
        errors = validate_memory_format(pipeline, elements, pad_refs)
        self.assertEqual(errors, [])

    def test_pngdec_direct_to_mux_detected(self):
        pipeline = (
            "gst-launch-1.0 filesrc location=a.png ! pngdec ! "
            "m.sink_0 nvstreammux name=m batch-size=1 ! nveglglessink"
        )
        elements, pad_refs = extract_elements_and_properties(pipeline)
        errors = validate_memory_format(pipeline, elements, pad_refs)
        self.assertTrue(len(errors) > 0)
        self.assertTrue(any("pngdec" in e for e in errors))

    def test_no_duplicate_errors(self):
        pipeline = (
            "gst-launch-1.0 filesrc location=a.jpg ! jpegparse ! nvjpegdec ! "
            "m.sink_0 nvstreammux name=m batch-size=1 width=1920 height=1080 ! nveglglessink"
        )
        elements, pad_refs = extract_elements_and_properties(pipeline)
        errors = validate_memory_format(pipeline, elements, pad_refs)
        self.assertEqual(len(errors), 1, f"Expected 1 error but got {len(errors)}: {errors}")


class TestKnownProperties(unittest.TestCase):
    """Validate that the property dict covers all properties used in data.csv."""

    def test_nvinfer_has_clip_object_outside_roi(self):
        from validate_pipeline import KNOWN_ELEMENT_PROPERTIES
        self.assertIn("clip-object-outside-roi", KNOWN_ELEMENT_PROPERTIES["nvinfer"])

    def test_nvinfer_has_output_tensor_meta(self):
        from validate_pipeline import KNOWN_ELEMENT_PROPERTIES
        self.assertIn("output-tensor-meta", KNOWN_ELEMENT_PROPERTIES["nvinfer"])

    def test_nvdspreprocess_has_operate_on_gie_id(self):
        from validate_pipeline import KNOWN_ELEMENT_PROPERTIES
        self.assertIn("operate-on-gie-id", KNOWN_ELEMENT_PROPERTIES["nvdspreprocess"])

    def test_nvv4l2h264enc_has_qp_range(self):
        from validate_pipeline import KNOWN_ELEMENT_PROPERTIES
        self.assertIn("qp-range", KNOWN_ELEMENT_PROPERTIES["nvv4l2h264enc"])

    def test_nvdewarper_has_num_output_buffers(self):
        from validate_pipeline import KNOWN_ELEMENT_PROPERTIES
        self.assertIn("num-output-buffers", KNOWN_ELEMENT_PROPERTIES["nvdewarper"])


class TestPlatformSinkValidation(unittest.TestCase):
    """Test platform-sink mismatch detection."""

    def test_mixed_dgpu_jetson_sinks_warns(self):
        elements, _ = extract_elements_and_properties(
            "filesrc location=t.mp4 ! nveglglessink ! nv3dsink"
        )
        warnings = validate_platform_sink(elements)
        self.assertTrue(any("mixes" in w.lower() for w in warnings))

    def test_dgpu_only_no_warning(self):
        elements, _ = extract_elements_and_properties(
            "filesrc location=t.mp4 ! nveglglessink"
        )
        warnings = validate_platform_sink(elements)
        self.assertEqual(warnings, [])

    def test_jetson_only_no_warning(self):
        elements, _ = extract_elements_and_properties(
            "filesrc location=t.mp4 ! nv3dsink"
        )
        warnings = validate_platform_sink(elements)
        self.assertEqual(warnings, [])

    def test_fakesink_no_warning(self):
        elements, _ = extract_elements_and_properties(
            "filesrc location=t.mp4 ! fakesink"
        )
        warnings = validate_platform_sink(elements)
        self.assertEqual(warnings, [])


class TestElementOrderingValidation(unittest.TestCase):
    """Test element ordering checks."""

    def test_correct_order_no_warning(self):
        elements, _ = extract_elements_and_properties(
            "filesrc location=t.mp4 ! nvv4l2decoder ! nvinfer config-file-path=c.txt ! nveglglessink"
        )
        warnings = validate_element_ordering(elements)
        self.assertEqual(warnings, [])

    def test_nvinfer_before_decoder_warns(self):
        elements, _ = extract_elements_and_properties(
            "nvinfer config-file-path=c.txt ! nvv4l2decoder ! filesrc location=t.mp4 ! nveglglessink"
        )
        warnings = validate_element_ordering(elements)
        self.assertTrue(len(warnings) > 0)

    def test_sink_before_source_warns(self):
        elements, _ = extract_elements_and_properties(
            "nveglglessink ! filesrc location=t.mp4 ! nvinfer config-file-path=c.txt"
        )
        warnings = validate_element_ordering(elements)
        # nvinfer after nveglglessink is detected via filesrc -> nvinfer ordering check
        self.assertTrue(len(warnings) > 0)

    def test_tracker_before_infer_warns(self):
        elements, _ = extract_elements_and_properties(
            "filesrc location=t.mp4 ! nvtracker ll-lib-file=t.so ! nvinfer config-file-path=c.txt ! nveglglessink"
        )
        warnings = validate_element_ordering(elements)
        self.assertTrue(any("nvtracker" in w and "nvinfer" in w for w in warnings))


class TestExpandedNVMMChecks(unittest.TestCase):
    """Test that NVMM checks cover encoders too."""

    def test_system_mem_to_h264enc_detected(self):
        pipeline = "gst-launch-1.0 filesrc location=a.jpg ! nvjpegdec ! nvv4l2h264enc ! filesink location=out.h264"
        elements, pad_refs = extract_elements_and_properties(pipeline)
        errors = validate_memory_format(pipeline, elements, pad_refs)
        self.assertTrue(any("nvjpegdec" in e and "nvv4l2h264enc" in e for e in errors))

    def test_videoconvert_to_h264enc_detected(self):
        pipeline = "gst-launch-1.0 filesrc location=a.mp4 ! videoconvert ! nvv4l2h264enc ! filesink location=out.h264"
        elements, pad_refs = extract_elements_and_properties(pipeline)
        errors = validate_memory_format(pipeline, elements, pad_refs)
        self.assertTrue(any("videoconvert" in e and "nvv4l2h264enc" in e for e in errors))

    def test_nvvideoconvert_to_h264enc_passes(self):
        pipeline = "gst-launch-1.0 filesrc location=a.mp4 ! nvvideoconvert ! nvv4l2h264enc ! filesink location=out.h264"
        elements, pad_refs = extract_elements_and_properties(pipeline)
        errors = validate_memory_format(pipeline, elements, pad_refs)
        self.assertEqual(errors, [])


class TestNewElements(unittest.TestCase):
    """Test that new elements are in the known properties dict."""

    def test_nvinferserver_in_known(self):
        from validate_pipeline import KNOWN_ELEMENT_PROPERTIES
        self.assertIn("nvinferserver", KNOWN_ELEMENT_PROPERTIES)
        self.assertIn("config-file-path", KNOWN_ELEMENT_PROPERTIES["nvinferserver"])

    def test_nvdsanalytics_in_known(self):
        from validate_pipeline import KNOWN_ELEMENT_PROPERTIES
        self.assertIn("nvdsanalytics", KNOWN_ELEMENT_PROPERTIES)

    def test_nvmsgconv_in_known(self):
        from validate_pipeline import KNOWN_ELEMENT_PROPERTIES
        self.assertIn("nvmsgconv", KNOWN_ELEMENT_PROPERTIES)

    def test_nvmsgbroker_in_known(self):
        from validate_pipeline import KNOWN_ELEMENT_PROPERTIES
        self.assertIn("nvmsgbroker", KNOWN_ELEMENT_PROPERTIES)

    def test_nvinferserver_missing_config_flagged(self):
        elements, pad_refs = extract_elements_and_properties(
            "filesrc location=t.mp4 ! nvinferserver batch-size=1 ! nveglglessink"
        )
        errors, _ = validate_pipeline_structure(elements, pad_refs)
        self.assertTrue(any("nvinferserver" in e and "config-file-path" in e for e in errors))


class TestDryRunMultiStream(unittest.TestCase):
    """Dry-run is skipped for multi-stream pipelines with named pad refs."""

    def test_skipped_when_pad_refs_present(self):
        pipeline = (
            "gst-launch-1.0 filesrc location=a.mp4 ! qtdemux ! h264parse ! nvv4l2decoder ! "
            "m.sink_0 nvstreammux name=m batch-size=2 width=1920 height=1080 ! "
            "nvinfer config-file-path=c.txt batch-size=2 ! nvvideoconvert ! nvdsosd ! "
            "nveglglessink filesrc location=b.mp4 ! qtdemux ! h264parse ! nvv4l2decoder ! m.sink_1"
        )
        pad_refs = ["m.sink_0", "m.sink_1"]
        errors, warnings = validate_with_gst_launch(pipeline, pad_refs)
        self.assertEqual(errors, [])
        self.assertEqual(warnings, [])

    def test_not_skipped_when_no_pad_refs(self):
        """Dry-run should execute (not be skipped) for single-stream pipelines."""
        pipeline = (
            "gst-launch-1.0 filesrc location=a.mp4 ! qtdemux ! h264parse ! "
            "nvv4l2decoder ! nveglglessink"
        )
        errors, warnings = validate_with_gst_launch(pipeline, pad_refs=[])
        self.assertIsInstance(errors, list)
        self.assertIsInstance(warnings, list)

    def test_not_skipped_when_pad_refs_none(self):
        pipeline = "gst-launch-1.0 videotestsrc ! fakesink"
        errors, warnings = validate_with_gst_launch(pipeline, pad_refs=None)
        self.assertIsNotNone(errors)
        self.assertIsNotNone(warnings)
        self.assertIsInstance(errors, list)
        self.assertIsInstance(warnings, list)


if __name__ == "__main__":
    unittest.main()
