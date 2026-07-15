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
DeepStream Pipeline Validator — Syntax, element, property, and structure checks.

Validates a gst-launch-1.0 pipeline string WITHOUT running it:
1. Syntax check — quotes, empty segments, leading/trailing pipes
2. Element check — verifies each element exists via gst-inspect-1.0
3. Property check — validates known properties for DeepStream elements
4. Structure check — source/sink presence, required props, named-pad refs
5. Live parse — optional dry-run via gst-launch-1.0 with fakesrc/fakesink

Zero external dependencies — uses only Python stdlib + system gst-inspect-1.0.

Usage:
    python3 validate_pipeline.py "gst-launch-1.0 filesrc location=test.mp4 ! ..."
    python3 validate_pipeline.py --pipeline "filesrc location=test.mp4 ! ..."
"""

import argparse
import json
import re
import shlex
import subprocess
import sys
from shutil import which

_SAFE_ELEMENT_NAME = re.compile(r'^[a-zA-Z0-9_-]+$')

# ---------------------------------------------------------------------------
# Known DeepStream/GStreamer element properties
# ---------------------------------------------------------------------------
KNOWN_ELEMENT_PROPERTIES = {
    "filesrc": {
        "location", "num-buffers", "typefind", "blocksize", "do-timestamp",
    },
    "nvinfer": {
        "config-file-path", "batch-size", "unique-id", "infer-on-gie-id",
        "input-tensor-meta", "interval", "model-engine-file", "gpu-id",
        "process-mode", "network-type", "cluster-mode", "operate-on-gie-id",
        "operate-on-class-ids", "filter-out-class-ids", "output-tensor-meta",
        "clip-object-outside-roi", "infer-on-class-ids", "raw-output-file-write",
    },
    "nvstreammux": {
        "name", "batch-size", "width", "height", "batched-push-timeout",
        "live-source", "enable-padding", "num-surfaces-per-frame",
        "gpu-id", "nvbuf-memory-type", "buffer-pool-size",
        "compute-hw", "interpolation-method", "adaptive-batching",
        "sync-inputs",
    },
    "nvvideoconvert": {
        "flip-method", "src-crop", "dest-crop", "gpu-id",
        "nvbuf-memory-type", "interpolation-method", "compute-hw",
    },
    "nvtracker": {
        "ll-lib-file", "ll-config-file", "gpu-id", "tracker-width",
        "tracker-height", "display-tracking-id", "compute-hw",
    },
    "nvdsosd": {
        "process-mode", "display-text", "display-bbox", "display-mask",
        "display-clock", "clock-font", "clock-font-size", "gpu-id",
    },
    "nvmultistreamtiler": {
        "rows", "columns", "width", "height", "gpu-id",
        "nvbuf-memory-type", "compute-hw", "show-source",
    },
    "nvdspreprocess": {
        "config-file", "gpu-id", "operate-on-gie-id",
    },
    "nvv4l2decoder": {
        "gpu-id", "num-extra-surfaces", "cudadec-memtype",
        "drop-frame-interval", "disable-dpb",
    },
    "nvv4l2h264enc": {
        "bitrate", "maxperf-enable", "preset-level", "profile",
        "control-rate", "gpu-id", "iframeinterval",
        "qp-range", "constqp", "initqp",
    },
    "nvv4l2h265enc": {
        "bitrate", "maxperf-enable", "preset-level", "profile",
        "control-rate", "gpu-id", "iframeinterval",
        "qp-range", "constqp", "initqp",
    },
    "nveglglessink": {"sync", "qos", "max-lateness", "window-x", "window-y"},
    "nv3dsink": {"sync", "qos"},
    "fakesink": {"sync", "async", "dump", "signal-handoffs", "num-buffers"},
    "filesink": {"location", "sync", "async", "append", "buffer-mode", "buffer-size"},
    "jpegenc": {"quality", "idct-method", "snapshot"},
    "jpegparse": set(),
    "nvjpegdec": {"gpu-id"},
    "pngenc": {"compression-level", "snapshot"},
    "pngdec": set(),
    "h264parse": {"config-interval", "disable-passthrough"},
    "h265parse": {"config-interval", "disable-passthrough"},
    "qtdemux": {"name"},
    "qtmux": {"fragment-duration", "faststart"},
    "rtph264pay": {"config-interval", "pt", "mtu"},
    "rtph265pay": {"config-interval", "pt", "mtu"},
    "udpsink": {"host", "port", "sync", "async"},
    "uridecodebin": {"uri", "caps", "buffer-size", "use-buffering"},
    "v4l2src": {"device", "num-buffers", "io-mode"},
    "videotestsrc": {"pattern", "num-buffers", "is-live"},
    "videoconvert": set(),
    "videoparse": {"width", "height", "format", "framerate"},
    "xvimagesink": {"sync"},
    "decodebin": {"name"},
    "dsexample": {
        "full-frame", "processing-width", "processing-height",
        "gpu-id", "blur-objects", "unique-id",
    },
    "nvdewarper": {
        "config-file", "source-id", "nvbuf-memory-type", "gpu-id",
        "num-output-buffers",
    },
    "nvsegvisual": {"width", "height", "gpu-id"},
    "nvdspostprocess": {"postprocesslib-config-file", "postprocesslib-name", "gpu-id"},
    "queue": {"max-size-buffers", "max-size-bytes", "max-size-time", "leaky"},
    "tee": {"name"},
    "nvstreamdemux": {"name"},
    "nvvideotestsrc": {
        "num-buffers", "is-live", "location", "file-loop",
        "max-jitter", "fixed-jitter",
    },
    "nvurisrcbin": {"uri"},
    "nvegltransform": set(),
    "nvdslogger": {"fps-measurement-interval-sec"},
    "fpsdisplaysink": {"video-sink", "sync", "text-overlay"},
    "identity": {"silent", "dump", "single-segment"},
    "rtspsrc": {"location", "latency", "protocols", "user-id", "user-pw"},
    "rtph264depay": set(),
    "avdec_h264": set(),
    "videoscale": set(),
    "audioconvert": set(),
    "audioresample": set(),
    "aacparse": set(),
    "avdec_aac": set(),
    "nvdsaudiotemplate": {"customlib-name", "customlib-props"},
    "tcpserversink": {"host", "port"},
    "tcpclientsrc": {"host", "port"},
    "rtpgstpay": {"config-interval"},
    "rtpstreampay": set(),
    "rtpstreamdepay": set(),
    "rtpgstdepay": set(),
    "nvinferserver": {
        "config-file-path", "unique-id", "infer-on-gie-id", "batch-size",
        "interval", "gpu-id", "process-mode", "input-tensor-meta",
        "operate-on-gie-id", "operate-on-class-ids", "infer-on-class-ids",
        "output-tensor-meta",
    },
    "nvdsanalytics": {"config-file", "unique-id"},
    "nvmsgconv": {
        "config", "payload-type", "msg2p-lib", "comp-id", "debug-payload-dir",
        "msg2p-newapi",
    },
    "nvmsgbroker": {
        "proto-lib", "conn-str", "topic", "config", "comp-id", "sync",
        "new-api",
    },
    "nvdsmetamux": set(),
}

PASSTHROUGH_ELEMENTS = {
    "capsfilter", "identity", "multiqueue",
    "input-selector", "output-selector",
}

# Elements that output video/x-raw in system (CPU) memory — NOT NVMM.
# Feeding these directly into NVMM-requiring elements causes linking errors.
SYSTEM_MEMORY_OUTPUT = {"nvjpegdec", "pngdec", "jpegdec", "videoconvert", "videoscale"}

# Elements that require video/x-raw(memory:NVMM) on their sink pads.
NVMM_INPUT_REQUIRED = {"nvstreammux", "nvv4l2h264enc", "nvv4l2h265enc"}

# Platform-specific sink elements — using the wrong one causes runtime failures.
DGPU_ONLY_SINKS = {"nveglglessink", "nvegltransform"}
JETSON_ONLY_SINKS = {"nv3dsink"}

# Element ordering: elements that should appear BEFORE others in the pipeline.
# Key = element that must come first, Value = set of elements it must precede.
EXPECTED_ORDER = {
    "filesrc": {"nvv4l2decoder", "decodebin", "jpegparse", "h264parse", "h265parse", "qtdemux"},
    "nvv4l2decoder": {"nvstreammux", "nvinfer", "nvinferserver", "nvvideoconvert"},
    "nvstreammux": {"nvinfer", "nvinferserver", "nvtracker", "nvdsosd", "nvmultistreamtiler"},
    "nvinfer": {"nvtracker", "nvdsosd", "nveglglessink", "nv3dsink", "filesink", "fakesink"},
    "nvinferserver": {"nvtracker", "nvdsosd", "nveglglessink", "nv3dsink", "filesink", "fakesink"},
    "nvtracker": {"nvdsosd", "nveglglessink", "nv3dsink", "filesink"},
}


# ---------------------------------------------------------------------------
# Element existence check via gst-inspect-1.0
# ---------------------------------------------------------------------------
def check_element_exists(element_name):
    """Check if a GStreamer element is registered using gst-inspect-1.0."""
    if not _SAFE_ELEMENT_NAME.match(element_name):
        return None
    try:
        result = subprocess.run(
            ["gst-inspect-1.0", element_name],
            capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


# ---------------------------------------------------------------------------
# Pipeline parser
# ---------------------------------------------------------------------------
def extract_elements_and_properties(pipeline_str):
    """
    Parse a gst-launch-1.0 pipeline string and extract elements with properties.
    Returns list of dicts: [{"name": str, "properties": dict, "raw": str}]
    """
    pipeline_str = re.sub(r"^gst-launch-1\.0\s+", "", pipeline_str.strip())
    pipeline_str = re.sub(r"^(?:-[a-zA-Z]+\s+)+", "", pipeline_str)
    pipeline_str = re.sub(r"\\\s*\n\s*", " ", pipeline_str)

    segments = re.split(r"\s+!\s+", pipeline_str)

    elements = []
    pad_refs = []

    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue

        if segment.startswith(("'", '"', "video/", "audio/", "image/", "application/")):
            continue

        parts = segment.split()
        if not parts:
            continue

        idx = 0
        # A segment may start with pad refs (e.g. "m.sink_0 nvstreammux name=m ...")
        while idx < len(parts):
            tok = parts[idx]
            if "." in tok and not tok.startswith("/") and "=" not in tok:
                pad_refs.append(tok)
                idx += 1
            else:
                break

        if idx >= len(parts):
            continue

        element_name = parts[idx]

        if element_name.startswith(("'", '"', "video/", "audio/", "image/")):
            continue

        properties = {}
        for part in parts[idx + 1:]:
            if part.startswith(("'", '"')):
                break
            if "=" in part:
                key, _, val = part.partition("=")
                properties[key] = val

        elements.append({
            "name": element_name,
            "properties": properties,
            "raw": segment,
        })

    return elements, pad_refs


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------
def validate_syntax(pipeline_str):
    """Check for common syntax errors."""
    errors = []
    warnings = []

    single_quotes = pipeline_str.count("'")
    double_quotes = pipeline_str.count('"')
    if single_quotes % 2 != 0:
        errors.append("Unbalanced single quotes in pipeline")
    if double_quotes % 2 != 0:
        errors.append("Unbalanced double quotes in pipeline")

    if re.search(r"!\s*!", pipeline_str):
        errors.append("Empty pipe segment detected (consecutive '!' without element)")

    cleaned = re.sub(r"^gst-launch-1\.0\s+", "", pipeline_str.strip())
    cleaned = re.sub(r"^(?:-[a-zA-Z]+\s+)+", "", cleaned)
    cleaned = re.sub(r"\\\s*\n\s*", " ", cleaned).strip()
    if cleaned.startswith("!"):
        errors.append("Pipeline starts with '!' — missing source element")
    if cleaned.endswith("!"):
        errors.append("Pipeline ends with '!' — missing sink element")

    # Underscore-typo check: scope to the segment that owns nvinfer/nvinferserver
    # so a hyphenated config-file-path on a different element (or in a comment)
    # does not mask an underscore typo on the actual nvinfer.
    if re.search(r"\bnvinfer(?:server)?\b[^!]*\bconfig_file_path=", pipeline_str):
        errors.append("nvinfer uses 'config-file-path' (hyphens), not 'config_file_path' (underscores)")

    # Note: this only inspects the segment immediately following the
    # 'nvstreammux' token (non-greedy match up to the next '!' or end of
    # string). Properties placed after a '!' separator — rare, but possible
    # via pipeline-continuation patterns — will not be considered. Adequate
    # for the single-line pipelines this skill produces today.
    mux_match = re.search(r"nvstreammux\b(.*?)(?:!|$)", pipeline_str)
    if mux_match and "name=" not in mux_match.group(1):
        warnings.append("nvstreammux without 'name=' — multi-source linking may fail")

    return errors, warnings


def validate_elements(pipeline_str):
    """Validate all elements exist and properties are recognized."""
    elements, pad_refs = extract_elements_and_properties(pipeline_str)
    errors = []
    warnings = []
    checked_elements = set()

    has_gst_inspect = which("gst-inspect-1.0") is not None

    for elem in elements:
        name = elem["name"]

        if name not in checked_elements:
            checked_elements.add(name)
            if has_gst_inspect and name not in KNOWN_ELEMENT_PROPERTIES and name not in PASSTHROUGH_ELEMENTS:
                exists = check_element_exists(name)
                if exists is False:
                    errors.append(f"Unknown element '{name}' — not found by gst-inspect-1.0")
                    continue
                elif exists is None:
                    warnings.append(f"Could not verify element '{name}' — gst-inspect-1.0 timed out")

        if name in KNOWN_ELEMENT_PROPERTIES:
            known_props = KNOWN_ELEMENT_PROPERTIES[name]
            for prop in elem["properties"]:
                if known_props and prop not in known_props and prop != "name":
                    warnings.append(
                        f"Unrecognized property '{prop}' on element '{name}'"
                        f" — verify with gst-inspect-1.0 {name}"
                    )

    return elements, pad_refs, errors, warnings


def validate_memory_format(pipeline_str, elements, pad_refs):
    """
    Detect memory format mismatches — e.g. nvjpegdec (system memory) feeding
    directly into nvstreammux (requires NVMM) without nvvideoconvert in between.

    Handles two patterns:
      1. Consecutive elements:  ... ! nvjpegdec ! nvstreammux ...
      2. Pad-ref branches:      ... ! nvjpegdec ! mux.sink_N  (branch end)
    """
    errors = []
    flagged_pairs = set()

    # Build mux alias set from named nvstreammux elements
    mux_aliases = set()
    for e in elements:
        if e["name"] == "nvstreammux" and "name" in e["properties"]:
            mux_aliases.add(e["properties"]["name"])

    # --- Check 1: consecutive elements in the parsed list ---
    for i in range(len(elements) - 1):
        curr = elements[i]["name"]
        nxt = elements[i + 1]["name"]
        if curr in SYSTEM_MEMORY_OUTPUT and nxt in NVMM_INPUT_REQUIRED:
            pair = (curr, nxt)
            if pair not in flagged_pairs:
                flagged_pairs.add(pair)
                errors.append(
                    f"'{curr}' outputs system memory (video/x-raw) but "
                    f"'{nxt}' requires NVMM — "
                    f"insert 'nvvideoconvert' between them"
                )

    if not mux_aliases:
        return errors

    # --- Check 2: segment before a mux pad ref (catches multi-branch) ---
    clean = re.sub(r"^gst-launch-1\.0\s+", "", pipeline_str.strip())
    clean = re.sub(r"^(?:-[a-zA-Z]+\s+)+", "", clean)
    clean = re.sub(r"\\\s*\n\s*", " ", clean)
    segments = re.split(r"\s+!\s+", clean)

    for i, seg in enumerate(segments):
        seg_stripped = seg.strip()
        has_mux_pad = any(
            seg_stripped.startswith(f"{a}.sink_") or f" {a}.sink_" in seg_stripped
            for a in mux_aliases
        )
        if not has_mux_pad or i == 0:
            continue
        prev_seg = segments[i - 1].strip()
        prev_parts = prev_seg.split()
        if not prev_parts:
            continue
        prev_elem = prev_parts[0]
        if prev_elem in SYSTEM_MEMORY_OUTPUT:
            pair = (prev_elem, "nvstreammux")
            if pair not in flagged_pairs:
                flagged_pairs.add(pair)
                errors.append(
                    f"'{prev_elem}' outputs system memory (video/x-raw) but "
                    f"feeds into 'nvstreammux' via pad ref — "
                    f"insert 'nvvideoconvert' between them"
                )

    return errors


def validate_pipeline_structure(elements, pad_refs):
    """Check source/sink presence, required props, and named-pad consistency."""
    warnings = []

    if not elements:
        return ["Pipeline is empty — no elements found"], warnings

    errors = []
    element_names = [e["name"] for e in elements]
    named_elements = {}
    for e in elements:
        if "name" in e["properties"]:
            named_elements[e["properties"]["name"]] = e["name"]

    source_elements = {
        "filesrc", "uridecodebin", "v4l2src", "videotestsrc", "rtspsrc",
        "multifilesrc", "nvvideotestsrc", "nvurisrcbin", "tcpclientsrc",
    }
    has_source = any(e in source_elements for e in element_names)
    if not has_source:
        warnings.append(
            "No recognized source element (filesrc, uridecodebin, v4l2src, etc.)"
            " — may be intentional if using a named pad"
        )

    sink_elements = {
        "nveglglessink", "nv3dsink", "filesink", "fakesink", "udpsink",
        "xvimagesink", "autovideosink", "fpsdisplaysink", "tcpserversink",
    }
    has_sink = any(e in sink_elements for e in element_names)
    if not has_sink:
        warnings.append("No recognized sink element — pipeline needs a sink to terminate")

    for elem in elements:
        if elem["name"] == "nvinfer" and "config-file-path" not in elem["properties"]:
            errors.append("nvinfer missing required 'config-file-path' property")
        if elem["name"] == "nvinferserver" and "config-file-path" not in elem["properties"]:
            errors.append("nvinferserver missing required 'config-file-path' property")
        if elem["name"] == "filesrc" and "location" not in elem["properties"]:
            errors.append("filesrc missing required 'location' property")

    # Named-pad cross-reference validation
    for ref in pad_refs:
        alias = ref.split(".")[0]
        if alias not in named_elements:
            if not any(alias == e["properties"].get("name", "") for e in elements):
                warnings.append(
                    f"Pad reference '{ref}' uses alias '{alias}'"
                    f" but no element has name='{alias}'"
                )

    # batch-size / sink-pad count consistency
    for elem in elements:
        if elem["name"] == "nvstreammux" and "batch-size" in elem["properties"]:
            try:
                batch = int(elem["properties"]["batch-size"])
            except ValueError:
                continue
            alias = elem["properties"].get("name", "")
            if alias:
                pad_count = sum(
                    1 for r in pad_refs
                    if r.startswith(f"{alias}.sink_")
                )
                if pad_count > 0 and pad_count != batch and batch > 1:
                    warnings.append(
                        f"nvstreammux '{alias}' has batch-size={batch}"
                        f" but {pad_count} sink pad(s) connected"
                    )

    return errors, warnings


def validate_platform_sink(elements):
    """
    Detect platform-sink mismatches:
    - nveglglessink used alongside nv3dsink hints (Jetson pipeline with dGPU sink)
    - Both Jetson-only and dGPU-only sinks in the same pipeline
    """
    warnings = []
    element_names = {e["name"] for e in elements}

    has_dgpu_sink = bool(element_names & DGPU_ONLY_SINKS)
    has_jetson_sink = bool(element_names & JETSON_ONLY_SINKS)

    if has_dgpu_sink and has_jetson_sink:
        warnings.append(
            "Pipeline mixes dGPU-only sinks (nveglglessink) and Jetson-only "
            "sinks (nv3dsink) — pick one for your target platform"
        )

    return warnings


def validate_element_ordering(elements):
    """
    Detect nonsensical element ordering — e.g. an encoder appearing before a
    decoder, or a sink element appearing before inference elements.
    Only flags clear-cut ordering violations for elements in EXPECTED_ORDER.
    """
    warnings = []
    # Build position map: element_name -> first occurrence index
    first_pos = {}
    for i, e in enumerate(elements):
        if e["name"] not in first_pos:
            first_pos[e["name"]] = i

    for before_elem, after_set in EXPECTED_ORDER.items():
        if before_elem not in first_pos:
            continue
        before_idx = first_pos[before_elem]
        for after_elem in after_set:
            if after_elem in first_pos and first_pos[after_elem] < before_idx:
                warnings.append(
                    f"Element '{after_elem}' appears before '{before_elem}' "
                    f"— expected '{before_elem}' first"
                )

    return warnings


def validate_with_gst_launch(pipeline_str, pad_refs=None):
    """
    Use gst-launch-1.0 for a dry-run parse check by substituting
    fakesrc/fakesink and stripping elements that need real data.

    Multi-stream pipelines (those with named pad refs like m.sink_0) are
    skipped — fakesrc cannot negotiate caps through named pads on
    nvstreammux, causing false-positive linking errors.
    """
    errors = []
    warnings = []

    if pad_refs and len(pad_refs) > 0:
        return errors, warnings

    if not which("gst-launch-1.0"):
        warnings.append("gst-launch-1.0 not found — skipping live pipeline parse check")
        return errors, warnings

    test_pipeline = re.sub(r"^gst-launch-1\.0\s+", "", pipeline_str.strip())
    test_pipeline = re.sub(r"^(?:-[a-zA-Z]+\s+)+", "", test_pipeline)
    test_pipeline = re.sub(r"\\\s*\n\s*", " ", test_pipeline)

    source_replacements = {
        "filesrc": "fakesrc",
        "v4l2src": "fakesrc",
        "videotestsrc": "fakesrc",
        "rtspsrc": "fakesrc",
        "uridecodebin": "fakesrc",
        "multifilesrc": "fakesrc",
        "nvvideotestsrc": "fakesrc",
        "nvurisrcbin": "fakesrc",
        "tcpclientsrc": "fakesrc",
    }
    sink_replacements = {
        "nveglglessink": "fakesink",
        "nv3dsink": "fakesink",
        "filesink": "fakesink",
        "udpsink": "fakesink",
        "xvimagesink": "fakesink",
        "autovideosink": "fakesink",
        "fpsdisplaysink": "fakesink",
        "tcpserversink": "fakesink",
    }

    test_str = test_pipeline

    # Anchor replacements to segment-leading positions ((^|!\s+)) so that a
    # property value containing the element name (e.g. config-file-path=
    # /path/with/filesrc.txt) is not mistakenly rewritten — that would corrupt
    # the dry-run input and produce false-positive parse errors.
    for real, fake in source_replacements.items():
        test_str = re.sub(rf"(^|!\s+){real}\b[^!]*", rf"\g<1>{fake} num-buffers=1 ", test_str)

    for real, fake in sink_replacements.items():
        test_str = re.sub(rf"(^|!\s+){real}\b[^!]*", rf"\g<1>{fake} ", test_str)

    test_str = re.sub(
        r"\b(config-file-path|config-file|ll-config-file|ll-lib-file"
        r"|postprocesslib-config-file|postprocesslib-name)=[^\s!]+",
        "", test_str,
    )

    strip_elements = [
        "qtdemux", "h264parse", "h265parse", "nvv4l2decoder", "nvjpegdec",
        "jpegparse", "decodebin", "pngdec", "jpegdec", "videoparse",
        "qtmux", "matroskamux", "mp4mux", "flvmux", "mpegtsmux",
        "rtph264pay", "rtph265pay", "rtph264depay", "avdec_h264",
        "rtpgstpay", "rtpstreampay", "rtpstreamdepay", "rtpgstdepay",
        "aacparse", "avdec_aac", "audioconvert", "audioresample",
    ]
    for elem in strip_elements:
        test_str = re.sub(rf"\b{elem}\b[^!]*!\s*", "", test_str)
        test_str = re.sub(rf"\b{elem}\b[^!]*$", "", test_str)

    test_str = re.sub(r"\s+", " ", test_str).strip()
    test_str = re.sub(r"!\s*!", "!", test_str)
    test_str = re.sub(r"^\s*!\s*", "", test_str)
    test_str = re.sub(r"\s*!\s*$", "", test_str)

    if not test_str.strip():
        warnings.append("Pipeline too simple to perform live parse validation")
        return errors, warnings

    try:
        try:
            gst_args = shlex.split(test_str)
        except ValueError:
            warnings.append("Live parse check skipped — pipeline string could not be tokenized safely")
            return errors, warnings

        result = subprocess.run(
            ["gst-launch-1.0", "--gst-debug-level=0"] + gst_args,
            capture_output=True, text=True, timeout=5,
        )

        stderr = result.stderr.strip()
        if "erroneous pipeline" in stderr.lower():
            for line in stderr.split("\n"):
                if "erroneous pipeline" in line.lower():
                    line = line.strip()[:300]
                    errors.append(f"GStreamer parse error: {line}")
        elif result.returncode != 0 and "error" in stderr.lower():
            noise_patterns = [
                "no such file", "configuration file not provided",
                "doesn't want to preroll", "not negotiated",
                "internal data stream", "gstnvinfer",
                "no element", "could not set property",
            ]
            for line in stderr.split("\n"):
                line = line.strip()
                if not line or "error" not in line.lower():
                    continue
                line_lower = line.lower()
                if any(p in line_lower for p in noise_patterns):
                    continue
                warnings.append(f"GStreamer runtime warning: {line[:300]}")

    except subprocess.TimeoutExpired:
        warnings.append("Live parse check timed out — pipeline may hang or require hardware not present")
    except FileNotFoundError:
        warnings.append("gst-launch-1.0 not found — skipping live parse check")

    return errors, warnings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Validate a GStreamer/DeepStream pipeline")
    parser.add_argument("pipeline", nargs="?", help="Pipeline string to validate")
    parser.add_argument("--pipeline", dest="pipeline_flag", help="Pipeline string (alternative flag)")
    parser.add_argument(
        "--format", dest="output_format",
        choices=("summary", "json"), default="json",
        help=(
            "Output format. 'json' = full verbose output (default — preserves "
            "existing behavior). 'summary' = one-line human status, with errors "
            "or warnings printed below if present."
        ),
    )

    args = parser.parse_args()
    pipeline_str = args.pipeline or args.pipeline_flag
    fmt = args.output_format

    if not pipeline_str:
        print("ERROR: No pipeline provided", file=sys.stderr)
        print('Usage: python3 validate_pipeline.py "gst-launch-1.0 ..."', file=sys.stderr)
        sys.exit(1)

    if len(pipeline_str) > 16384:
        if fmt == "summary":
            print("invalid: Pipeline string exceeds maximum allowed length (16384 chars)")
        else:
            result = {"valid": False, "errors": ["Pipeline string exceeds maximum allowed length (16384 chars)"], "warnings": []}
            print(json.dumps(result, indent=2))
        sys.exit(1)

    all_errors = []
    all_warnings = []

    syntax_errors, syntax_warnings = validate_syntax(pipeline_str)
    all_errors.extend(syntax_errors)
    all_warnings.extend(syntax_warnings)

    elements, pad_refs, elem_errors, elem_warnings = validate_elements(pipeline_str)
    all_errors.extend(elem_errors)
    all_warnings.extend(elem_warnings)

    mem_errors = validate_memory_format(pipeline_str, elements, pad_refs)
    all_errors.extend(mem_errors)

    struct_errors, struct_warnings = validate_pipeline_structure(elements, pad_refs)
    all_errors.extend(struct_errors)
    all_warnings.extend(struct_warnings)

    platform_warnings = validate_platform_sink(elements)
    all_warnings.extend(platform_warnings)

    ordering_warnings = validate_element_ordering(elements)
    all_warnings.extend(ordering_warnings)

    gst_errors, gst_warnings = validate_with_gst_launch(pipeline_str, pad_refs)
    all_errors.extend(gst_errors)
    all_warnings.extend(gst_warnings)

    result = {
        "valid": len(all_errors) == 0,
        "elements_found": [e["name"] for e in elements],
        "num_elements": len(elements),
        "pad_refs": pad_refs,
        "errors": all_errors,
        "warnings": all_warnings,
    }

    if fmt == "summary":
        live_parse_skipped = bool(pad_refs)
        if result["valid"]:
            note = ""
            if live_parse_skipped:
                note = " · live-parse skipped (multi-stream)"
            print(
                "valid · {n} elements · {w} warnings{note}".format(
                    n=len(elements), w=len(all_warnings), note=note,
                )
            )
        else:
            print(
                "invalid · {n} elements · {e} errors · {w} warnings".format(
                    n=len(elements), e=len(all_errors), w=len(all_warnings),
                )
            )
        for err in all_errors:
            print("  error: {}".format(err))
        for warn in all_warnings:
            print("  warning: {}".format(warn))
    else:
        print(json.dumps(result, indent=2))

    sys.exit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
