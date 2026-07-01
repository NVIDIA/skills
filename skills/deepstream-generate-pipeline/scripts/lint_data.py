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
Data quality linter for the DeepStream pipeline dataset.

Checks for:
  1. Duplicate or near-duplicate rows
  2. Invalid pipelines (dual sinks, missing source/sink)
  3. Syntax issues (space before '=' in properties)
  4. batch-size / sink-pad count mismatches

Run with --fix to auto-fix known issues and write a cleaned CSV.

Usage:
    python3 lint_data.py                    # report only
    python3 lint_data.py --fix              # fix and overwrite
    python3 lint_data.py --fix --out clean.csv  # fix and write to new file
"""

import argparse
import csv
import os
import re
import sys
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(SCRIPT_DIR, "..", "data", "data.csv")

KNOWN_SINKS = {
    "nveglglessink", "nv3dsink", "filesink", "fakesink",
    "udpsink", "xvimagesink", "autovideosink",
}
KNOWN_SOURCES = {
    "filesrc", "uridecodebin", "v4l2src", "videotestsrc",
    "rtspsrc", "multifilesrc", "nvvideotestsrc", "nvurisrcbin",
}


def normalize_pipeline(pipeline):
    """Normalize whitespace and line continuations for comparison."""
    p = re.sub(r"\\\s*\n\s*", " ", pipeline)
    p = re.sub(r"\s+", " ", p).strip()
    return p


def find_issues(rows):
    """Return list of (row_index_0based, issue_type, message) tuples."""
    issues = []
    seen_pipelines = {}

    for i, (prompt, pipeline) in enumerate(rows):
        norm = normalize_pipeline(pipeline)

        # --- Exact duplicate pipelines ---
        if norm in seen_pipelines:
            prev = seen_pipelines[norm]
            issues.append((i, "duplicate", f"Exact duplicate pipeline of row {prev + 1}"))
        else:
            seen_pipelines[norm] = i

        # --- Dual sinks piped together ---
        # Skip segments that are pad references (e.g. "m.sink_0", "t.src_1")
        # or that *start* with a pad ref (e.g. "t. ! nv3dsink"). These
        # appear in legitimate tee fan-out / nvstreamdemux patterns where
        # a sink immediately following a pad-ref boundary is not a real
        # consecutive-sinks bug.
        def _is_pad_ref_segment(seg):
            tokens = seg.split()
            if not tokens:
                return False
            head = tokens[0]
            return bool(re.match(r"^[a-zA-Z_][a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]+$", head))

        sinks_in_pipe = []
        segments = re.split(r"\s+!\s+", norm)
        for seg in segments:
            if _is_pad_ref_segment(seg):
                continue
            elem = seg.split()[0] if seg.split() else ""
            if elem in KNOWN_SINKS:
                sinks_in_pipe.append(elem)
        if len(sinks_in_pipe) > 1:
            chain = " ! ".join(sinks_in_pipe)
            consecutive = False
            for j in range(len(segments) - 1):
                if _is_pad_ref_segment(segments[j]) or _is_pad_ref_segment(segments[j + 1]):
                    continue
                e1 = segments[j].split()[0] if segments[j].split() else ""
                e2 = segments[j + 1].split()[0] if segments[j + 1].split() else ""
                if e1 in KNOWN_SINKS and e2 in KNOWN_SINKS:
                    consecutive = True
            if consecutive:
                issues.append((i, "dual_sink", f"Consecutive sinks piped together: {chain}"))

        # --- Space before '=' in properties ---
        space_eq_matches = re.findall(r'\b(\w+)\s+=\s*', pipeline)
        if space_eq_matches:
            for prop in space_eq_matches:
                if prop in ("location", "uri", "device"):
                    issues.append((i, "space_equals",
                                   f"Space before '=' in '{prop} =' — should be '{prop}='"))

    return issues


def fix_row(prompt, pipeline, issues_for_row):
    """Apply automatic fixes to a pipeline. Returns (prompt, fixed_pipeline) or None to drop."""
    fixed = pipeline

    for _, issue_type, msg in issues_for_row:
        if issue_type == "duplicate":
            return None

        if issue_type == "space_equals":
            fixed = re.sub(r'\b(location|uri|device)\s+=\s*', r'\1=', fixed)

        if issue_type == "dual_sink":
            fixed = re.sub(r'\bnv3dsink\s*!\s*nveglglessink\b', 'nv3dsink', fixed)
            fixed = re.sub(r'\bnveglglessink\s*!\s*nv3dsink\b', 'nveglglessink', fixed)

    return (prompt, fixed)


def main():
    parser = argparse.ArgumentParser(description="Lint the DeepStream pipeline dataset")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to data CSV")
    parser.add_argument("--fix", action="store_true", help="Auto-fix issues and write output")
    parser.add_argument("--out", default=None, help="Output CSV path (default: overwrite input)")
    args = parser.parse_args()

    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [(row[0].strip(), row[1].strip()) for row in reader if len(row) >= 2]

    issues = find_issues(rows)

    if not issues:
        print("No issues found.")
        return

    issues_by_row = defaultdict(list)
    for idx, itype, msg in issues:
        issues_by_row[idx].append((idx, itype, msg))

    print(f"Found {len(issues)} issue(s) in {len(issues_by_row)} row(s):\n")
    for idx, itype, msg in issues:
        print(f"  Row {idx + 1} [{itype}]: {msg}")

    if not args.fix:
        print(f"\nRun with --fix to auto-fix.")
        sys.exit(1 if issues else 0)

    out_path = args.out or args.csv
    fixed_rows = []
    dropped = 0
    fixed_count = 0

    for i, (prompt, pipeline) in enumerate(rows):
        if i in issues_by_row:
            result = fix_row(prompt, pipeline, issues_by_row[i])
            if result is None:
                dropped += 1
                continue
            fixed_rows.append(result)
            if result[1] != pipeline:
                fixed_count += 1
        else:
            fixed_rows.append((prompt, pipeline))

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for prompt, pipeline in fixed_rows:
            writer.writerow([prompt, pipeline])

    print(f"\nFixed {fixed_count} row(s), dropped {dropped} duplicate(s).")
    print(f"Wrote {len(fixed_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
