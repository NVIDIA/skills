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
DeepStream Pipeline Generator — BM25 retrieval with structural boosting.

Loads a CSV dataset of verified DeepStream pipelines and retrieves the most
relevant ones using BM25 scoring (document-length-normalized) combined with
structural metadata boosting so that results match the user's platform, source
type, sink type, and inference mode.

Zero external dependencies — uses only Python stdlib.

Usage:
    python3 generate_pipeline.py \
        --query "pipeline that performs primary inference on a single mp4 video" \
        --source-type "Local video file" \
        --num-sources 1 \
        --inference "primary" \
        --tracker "none" \
        --sink "display" \
        --platform "dGPU" \
        --extras "none"
"""

import argparse
import csv
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter

# Bumped manually whenever the indexed-state schema changes (tokenizer
# rules, metadata fields, structural-boost behavior). Stored inside the
# cache file so that an old cache from a prior code version is rejected
# even when the CSV mtime hasn't changed.
INDEX_CACHE_VERSION = 3

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SKILL_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_DATA_CSV = os.path.join(SKILL_DIR, "data", "data.csv")
TOP_K = 10
MIN_SCORE = 0.5
BM25_K1 = 1.5
BM25_B = 0.75

SYNONYMS = {
    "infer": ["inference", "nvinfer", "detect", "detection", "classify"],
    "inference": ["infer", "nvinfer", "detect", "detection", "classify"],
    "detect": ["detection", "infer", "inference", "nvinfer"],
    "display": ["render", "show", "nveglglessink", "nv3dsink", "sink"],
    "save": ["dump", "store", "write", "filesink", "output"],
    "image": ["jpg", "jpeg", "png", "frame", "picture"],
    "video": ["mp4", "h264", "h265", "stream", "clip"],
    "track": ["tracker", "nvtracker", "tracking"],
    "tracker": ["track", "nvtracker", "tracking"],
    "rtsp": ["stream", "rtspsrc", "uridecodebin"],
    "primary": ["nvinfer", "pgie", "detector"],
    "secondary": ["sgie", "classifier", "classification"],
    "crop": ["src-crop", "dest-crop", "region"],
    "flip": ["rotate", "flip-method", "mirror"],
    "rotate": ["flip", "flip-method"],
    "resize": ["scale", "resolution", "width", "height"],
    "jetson": ["nv3dsink", "nvv4l2"],
    "dgpu": ["nveglglessink", "desktop", "server"],
    "mp4": ["qtmux", "qtdemux", "video", "h264"],
    "jpg": ["jpeg", "jpegenc", "jpegdec", "nvjpegdec", "image"],
    "jpeg": ["jpg", "jpegenc", "jpegdec", "nvjpegdec", "image"],
    "osd": ["nvdsosd", "bounding", "boxes", "labels", "overlay"],
    "preprocess": ["nvdspreprocess", "preprocessor", "roi"],
    "dewarp": ["nvdewarper", "dewarper", "perspective"],
    "encode": ["encoder", "nvv4l2h264enc", "nvv4l2h265enc", "h264", "h265"],
    "decode": ["decoder", "nvv4l2decoder", "decodebin"],
    "segment": ["segmentation", "nvsegvisual", "semantic"],
    "postprocess": ["nvdspostprocess", "postprocessor"],
    "triton": ["nvinferserver", "inference-server", "grpc", "model-repository"],
    "nvinferserver": ["triton", "inference-server"],
    "analytics": ["nvdsanalytics", "line-crossing", "roi", "direction"],
}


# ---------------------------------------------------------------------------
# Pipeline metadata extraction
# ---------------------------------------------------------------------------
def extract_pipeline_metadata(pipeline):
    """Extract structural metadata from a pipeline string for filtering."""
    p = pipeline.lower()

    # Platform
    if "nv3dsink" in p:
        platform = "jetson"
    elif "nveglglessink" in p or "nvegltransform" in p:
        platform = "dgpu"
    else:
        platform = "unknown"

    # Source type
    if "filesrc" in p and ("jpegparse" in p or "nvjpegdec" in p or ".jpg" in p or ".png" in p):
        source_type = "image"
    elif "filesrc" in p:
        source_type = "video"
    elif "uridecodebin" in p or "rtspsrc" in p or "nvurisrcbin" in p:
        if "rtsp://" in p:
            source_type = "rtsp"
        else:
            source_type = "video"
    elif "v4l2src" in p:
        source_type = "usb"
    elif "videotestsrc" in p or "nvvideotestsrc" in p:
        source_type = "test"
    else:
        source_type = "unknown"

    # Sink type
    if "nveglglessink" in p or "nv3dsink" in p or "xvimagesink" in p or "autovideosink" in p:
        sink_type = "display"
    elif "filesink" in p and ("qtmux" in p or "mp4" in p):
        sink_type = "save-mp4"
    elif "filesink" in p and "jpegenc" in p:
        sink_type = "save-jpg"
    elif "filesink" in p and "pngenc" in p:
        sink_type = "save-png"
    elif "filesink" in p:
        sink_type = "save-file"
    elif "udpsink" in p:
        sink_type = "rtsp-out"
    elif "fakesink" in p:
        sink_type = "fakesink"
    else:
        sink_type = "unknown"

    nvinfer_count = len(re.findall(r"\bnvinfer\b", p))
    nvinferserver_count = len(re.findall(r"\bnvinferserver\b", p))
    total_infer = nvinfer_count + nvinferserver_count
    has_preprocess = "nvdspreprocess" in p
    if total_infer == 0:
        inference = "none"
    elif total_infer == 1 and has_preprocess:
        inference = "primary+preprocess"
    elif total_infer == 1:
        inference = "primary"
    elif total_infer >= 2 and has_preprocess:
        inference = "primary+secondary+preprocess"
    else:
        inference = "primary+secondary"

    # Tracker
    has_tracker = "nvtracker" in p
    tracker = "present" if has_tracker else "none"

    # Number of streams
    sink_pads = re.findall(r"m\.sink_(\d+)", p)
    if sink_pads:
        num_sources = max(int(n) for n in sink_pads) + 1
    else:
        num_sources = 1

    return {
        "platform": platform,
        "source_type": source_type,
        "sink_type": sink_type,
        "inference": inference,
        "tracker": tracker,
        "num_sources": num_sources,
    }


def normalize_user_params(args):
    """Map user CLI params to the same vocabulary as pipeline metadata."""
    source_map = {
        "local video file": "video",
        "local image file": "image",
        "rtsp stream": "rtsp",
        "usb camera": "usb",
        "test pattern": "test",
    }
    sink_map = {
        "display": "display",
        "display-jetson": "display",
        "save-jpg": "save-jpg",
        "save-png": "save-png",
        "save-mp4": "save-mp4",
        "save-h264": "save-file",
        "rtsp-out": "rtsp-out",
        "fakesink": "fakesink",
    }
    inference_map = {
        "none": "none",
        "primary": "primary",
        "primary+secondary": "primary+secondary",
        "primary+preprocess": "primary+preprocess",
        "primary+secondary+preprocess": "primary+secondary+preprocess",
        "primary-triton": "primary",
        "primary+secondary-triton": "primary+secondary",
    }
    # SBSA (aarch64 servers, e.g. Grace/GH200) shares Jetson's sink/plugin
    # set (nv3dsink, nvv4l2*), so it maps to the "jetson" code path.
    def _map_platform(value):
        v = value.lower()
        if "jetson" in v or "sbsa" in v or "aarch64" in v or "arm" in v:
            return "jetson"
        if "dgpu" in v or "x86" in v or "gpu" in v:
            return "dgpu"
        return "unknown"

    return {
        "platform": _map_platform(args.platform),
        "source_type": source_map.get(args.source_type.lower(), "unknown"),
        "sink_type": sink_map.get(args.sink.lower(), "unknown"),
        "inference": inference_map.get(args.inference.lower(), "unknown"),
        "tracker": "present" if args.tracker.lower() != "none" else "none",
        "num_sources": args.num_sources,
    }


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
def tokenize(text):
    """
    Lowercase, strip punctuation, split into tokens.

    `/` is intentionally NOT in the kept-char set: a path like
    /opt/nvidia/deepstream/samples/streams/sample.mp4 should split into
    ['opt', 'nvidia', 'deepstream', 'samples', 'streams', 'sample.mp4']
    so each path component contributes useful IDF signal — instead of
    collapsing into one giant unique token that bloats the IDF table and
    matches almost nothing. Compound tokens with hyphens (e.g.
    config-file-path) are preserved because `-` is in the kept set.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9._-]", " ", text)
    return text.split()


def expand_with_synonyms(tokens):
    """Add domain-specific synonyms to boost recall."""
    expanded = list(tokens)
    seen = set(tokens)
    for tok in tokens:
        for syn in SYNONYMS.get(tok, []):
            if syn not in seen:
                expanded.append(syn)
                seen.add(syn)
    return expanded


# ---------------------------------------------------------------------------
# BM25 Engine
# ---------------------------------------------------------------------------
class BM25Retriever:
    """BM25 retriever with structural metadata boosting. Pure stdlib."""

    def __init__(self, k1=BM25_K1, b=BM25_B):
        self.k1 = k1
        self.b = b
        self.documents = []
        self.metadata = []
        self.doc_tokens = []
        self.doc_tf = []
        self.idf = {}
        self.avgdl = 0

    def load_csv(self, csv_path):
        """
        Load pipeline dataset from CSV.

        The fully-indexed retriever state (documents, metadata, tokens,
        TF, IDF, avgdl) is cached to a sibling JSON file
        ``{csv_path}.cache.json`` keyed on the CSV's mtime + an internal
        schema version. Subsequent invocations with an unchanged CSV
        skip the parse + tokenization + IDF passes entirely. The cache
        is invalidated automatically when the CSV is edited or when the
        indexed schema changes.

        JSON is parser-only — loading a cache file cannot execute
        arbitrary code, even if the file has been tampered with. An
        unreadable, mtime-mismatched, or version-mismatched cache is
        rejected and the dataset is re-indexed from scratch.
        """
        if self._load_from_cache(csv_path):
            return

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompt = row.get("Prompt", "").strip()
                pipeline = row.get("Gst launch pipeline", "").strip()
                if prompt and pipeline:
                    self.documents.append((prompt, pipeline))
                    self.metadata.append(extract_pipeline_metadata(pipeline))

        for prompt, pipeline in self.documents:
            prompt_tokens = tokenize(prompt)
            pipeline_tokens = tokenize(pipeline)
            tokens = expand_with_synonyms(prompt_tokens * 2 + pipeline_tokens)
            self.doc_tokens.append(tokens)

        self._compute_idf()
        self._precompute_tf()
        self._save_to_cache(csv_path)

    @staticmethod
    def _cache_path(csv_path):
        return csv_path + ".cache.json"

    def _load_from_cache(self, csv_path):
        cache_path = self._cache_path(csv_path)
        if not os.path.exists(cache_path):
            return False
        try:
            csv_mtime = os.path.getmtime(csv_path)
        except OSError:
            return False
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            return False
        if not isinstance(payload, dict):
            return False
        if payload.get("version") != INDEX_CACHE_VERSION:
            return False
        if payload.get("csv_mtime") != csv_mtime:
            return False
        try:
            self.documents = payload["documents"]
            self.metadata = payload["metadata"]
            self.doc_tokens = payload["doc_tokens"]
            self.doc_tf = payload["doc_tf"]
            self.idf = payload["idf"]
            self.avgdl = payload["avgdl"]
        except KeyError:
            return False
        return True

    def _save_to_cache(self, csv_path):
        cache_path = self._cache_path(csv_path)
        try:
            csv_mtime = os.path.getmtime(csv_path)
        except OSError:
            return
        payload = {
            "version": INDEX_CACHE_VERSION,
            "csv_mtime": csv_mtime,
            "documents": self.documents,
            "metadata": self.metadata,
            "doc_tokens": self.doc_tokens,
            "doc_tf": self.doc_tf,
            "idf": self.idf,
            "avgdl": self.avgdl,
        }
        # Atomic write — temp file in same dir, then rename. Avoids a
        # partial cache file if the process is killed mid-write.
        cache_dir = os.path.dirname(cache_path) or "."
        try:
            fd, tmp_path = tempfile.mkstemp(
                prefix=".cache.", suffix=".json.tmp", dir=cache_dir,
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(payload, f)
                os.replace(tmp_path, cache_path)
            except Exception:
                # Best-effort cleanup; never raise from the cache writer.
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        except OSError:
            # Read-only filesystem or permission denied — fine, just no cache.
            pass

    def _compute_idf(self):
        """Compute IDF with BM25 formula: log((N - df + 0.5) / (df + 0.5) + 1)."""
        n = len(self.doc_tokens)
        df = Counter()
        for tokens in self.doc_tokens:
            for tok in set(tokens):
                df[tok] += 1
        self.idf = {
            tok: math.log((n - count + 0.5) / (count + 0.5) + 1)
            for tok, count in df.items()
        }
        total_len = sum(len(t) for t in self.doc_tokens)
        self.avgdl = total_len / n if n else 1

    def _precompute_tf(self):
        """Precompute term frequency dicts per document."""
        self.doc_tf = [Counter(tokens) for tokens in self.doc_tokens]

    def _bm25_score(self, query_tokens, doc_idx):
        """Score a single document against query tokens."""
        tf = self.doc_tf[doc_idx]
        dl = len(self.doc_tokens[doc_idx])
        score = 0.0
        for tok in query_tokens:
            if tok not in tf:
                continue
            f = tf[tok]
            idf = self.idf.get(tok, 0)
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += idf * numerator / denominator
        return score

    def _structural_boost(self, doc_meta, user_meta):
        """Compute a multiplier based on structural metadata match."""
        boost = 1.0

        # Platform: strong signal
        if user_meta["platform"] != "unknown" and doc_meta["platform"] != "unknown":
            if doc_meta["platform"] == user_meta["platform"]:
                boost *= 1.4
            else:
                boost *= 0.4

        # Source type
        if user_meta["source_type"] != "unknown" and doc_meta["source_type"] != "unknown":
            if doc_meta["source_type"] == user_meta["source_type"]:
                boost *= 1.3
            else:
                boost *= 0.7

        # Sink type
        if user_meta["sink_type"] != "unknown" and doc_meta["sink_type"] != "unknown":
            if doc_meta["sink_type"] == user_meta["sink_type"]:
                boost *= 1.2
            else:
                boost *= 0.7

        # Inference mode
        if user_meta["inference"] != "unknown" and doc_meta["inference"] != "unknown":
            if doc_meta["inference"] == user_meta["inference"]:
                boost *= 1.3
            elif "primary" in doc_meta["inference"] and "primary" in user_meta["inference"]:
                boost *= 1.1

        # Tracker
        if doc_meta["tracker"] == user_meta["tracker"]:
            boost *= 1.1

        # Number of sources (prefer exact match, tolerate close)
        if user_meta["num_sources"] == doc_meta["num_sources"]:
            boost *= 1.2
        elif user_meta["num_sources"] > 1 and doc_meta["num_sources"] > 1:
            boost *= 1.05

        return boost

    def retrieve(self, query, user_meta=None, top_k=TOP_K, min_score=MIN_SCORE):
        """Find top-K most relevant pipelines for a query."""
        query_tokens = expand_with_synonyms(tokenize(query))

        scores = []
        for i in range(len(self.documents)):
            raw = self._bm25_score(query_tokens, i)
            if raw <= 0:
                continue

            if user_meta:
                boost = self._structural_boost(self.metadata[i], user_meta)
                final = raw * boost
            else:
                final = raw

            scores.append((final, raw, i))

        scores.sort(key=lambda x: x[0], reverse=True)

        # Confidence is thresholded against the *raw* BM25 score, not the
        # boosted final score. Boosting can multiply the score by up to
        # ~3.75× (1.4·1.3·1.2·1.3·1.1·1.2 across platform / source / sink /
        # inference / tracker / num-sources matches), so thresholding the
        # boosted score would make the badge inflate the moment the
        # structural metadata aligns — even when the textual match is weak.
        top_raw = scores[0][1] if scores else 0
        results = []
        seen_pipelines = set()
        for final, raw, idx in scores[:top_k * 2]:
            if final < min_score and len(results) >= 3:
                break
            prompt, pipeline = self.documents[idx]
            normalized = re.sub(r"\s+", " ", pipeline.strip())
            if normalized in seen_pipelines:
                continue
            seen_pipelines.add(normalized)
            results.append({
                "score": round(final, 4),
                "raw_bm25": round(raw, 4),
                "prompt": prompt,
                "pipeline": pipeline,
                "metadata": self.metadata[idx],
            })
            if len(results) >= top_k:
                break

        confidence = "low"
        if top_raw >= 6:
            confidence = "high"
        elif top_raw >= 3:
            confidence = "medium"

        return results, confidence


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="DeepStream Pipeline Generator (BM25)")
    parser.add_argument("--query", required=True, help="Natural language pipeline query")
    parser.add_argument("--source-type", default="Local video file")
    parser.add_argument("--num-sources", type=int, default=1)
    parser.add_argument("--inference", default="None")
    parser.add_argument("--tracker", default="none")
    parser.add_argument("--sink", default="display")
    parser.add_argument("--platform", default="dGPU")
    parser.add_argument("--extras", default="none")
    parser.add_argument("--data-csv", default=DEFAULT_DATA_CSV, help="Path to pipeline dataset CSV")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--min-score", type=float, default=MIN_SCORE)
    parser.add_argument(
        "--format", dest="output_format",
        choices=("summary", "compact", "json"), default="json",
        help=(
            "Output format. 'json' = full verbose output (default — preserves "
            "existing behavior). 'compact' = small JSON with confidence + top "
            "result only. 'summary' = one-line human status."
        ),
    )

    args = parser.parse_args()

    if not os.path.exists(args.data_csv):
        print(f"ERROR: Dataset not found at {args.data_csv}", file=sys.stderr)
        sys.exit(1)

    enriched_query = (
        f"{args.query} "
        f"{args.source_type} "
        f"{args.inference} "
        f"{args.tracker} "
        f"{args.sink} "
        f"{args.platform} "
        f"{args.extras}"
    )

    user_meta = normalize_user_params(args)

    retriever = BM25Retriever()
    retriever.load_csv(args.data_csv)
    results, confidence = retriever.retrieve(
        enriched_query, user_meta=user_meta,
        top_k=args.top_k, min_score=args.min_score,
    )

    output = {
        "query": args.query,
        "confidence": confidence,
        "context": {
            "source_type": args.source_type,
            "num_sources": args.num_sources,
            "inference": args.inference,
            "tracker": args.tracker,
            "sink": args.sink,
            "platform": args.platform,
            "extras": args.extras,
        },
        "num_retrieved": len(results),
        "retrieved_pipelines": results,
    }

    fmt = getattr(args, "output_format", "json")
    if fmt == "summary":
        top = results[0]["prompt"] if results else "(no match)"
        if len(top) > 80:
            top = top[:77] + "..."
        print(
            "confidence={conf} retrieved={n} top_match={top!r}".format(
                conf=confidence, n=len(results), top=top,
            )
        )
    elif fmt == "compact":
        compact = {
            "query": args.query,
            "confidence": confidence,
            "context": output["context"],
            "num_retrieved": len(results),
            "top_pipeline": results[0] if results else None,
        }
        print(json.dumps(compact, indent=2))
    else:
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
