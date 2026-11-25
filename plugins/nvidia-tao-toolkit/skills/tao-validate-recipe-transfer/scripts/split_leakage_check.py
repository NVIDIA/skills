#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Detect near-duplicate images across dataset splits.

Near-duplicates across train/val/test are the single biggest source of inflated CV
numbers. Video frames sampled at a few fps, fixed-camera captures, burst photos, and
repeated scenes all produce images that are visually near-identical but not byte-identical,
so exact-hash dedup misses them entirely.

Method: difference hash (dHash) at 8x8 -> 64-bit signature, compared by Hamming distance.
Fast, dependency-light, and robust to resize/recompression. Also reports filename-stem
clustering, which catches frame_0001/frame_0002 style sequences that dHash may miss when
the camera pans.

Usage:
    python split_leakage_check.py --splits train=/data/train val=/data/val test=/data/test
    python split_leakage_check.py --splits train=a val=b --threshold 8 --report leakage.md

Exit codes: 0 = clean, 1 = leakage found (usable as a CI gate).
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def dhash(path, size=8):
    """64-bit difference hash. Returns np.uint8 array of 64 bits, or None on failure."""
    try:
        img = Image.open(path).convert("L").resize((size + 1, size), Image.BILINEAR)
    except Exception:
        return None
    a = np.asarray(img, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten().astype(np.uint8)


def collect(root):
    root = Path(root)
    if not root.exists():
        raise SystemExit(f"error: split path does not exist: {root}")
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT)


def stem_group(name):
    """Strip trailing digits so frame_0001 and frame_0002 share a group."""
    return re.sub(r"[_\-]?\d+$", "", Path(name).stem)


def hamming_matrix(a, b):
    """Pairwise Hamming distance between two bit matrices, in chunks to bound memory."""
    out = np.empty((a.shape[0], b.shape[0]), dtype=np.uint8)
    step = max(1, 2_000_000 // max(b.shape[0], 1))
    for i in range(0, a.shape[0], step):
        chunk = a[i : i + step]
        out[i : i + step] = (chunk[:, None, :] != b[None, :, :]).sum(axis=2)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", required=True,
                    help="name=path pairs, e.g. train=/data/train val=/data/val")
    ap.add_argument("--threshold", type=int, default=6,
                    help="max Hamming distance counted as a near-duplicate (0-64). "
                         "0-4 = near identical, 5-10 = same scene, >12 = loose. Default 6")
    ap.add_argument("--report", default=None, help="write a markdown report to this path")
    ap.add_argument("--max-examples", type=int, default=15, help="example pairs to list per split pair")
    args = ap.parse_args()

    splits = {}
    for item in args.splits:
        if "=" not in item:
            raise SystemExit(f"error: --splits entries must be name=path, got {item!r}")
        name, path = item.split("=", 1)
        splits[name] = path

    if len(splits) < 2:
        raise SystemExit("error: need at least two splits to check for leakage")

    print("Hashing images...", file=sys.stderr)
    data = {}
    for name, path in splits.items():
        files = collect(path)
        hashes, kept = [], []
        for f in files:
            h = dhash(f)
            if h is not None:
                hashes.append(h)
                kept.append(f)
        if not kept:
            raise SystemExit(f"error: no readable images found in split {name!r} ({path})")
        data[name] = {"files": kept, "hashes": np.array(hashes, dtype=np.uint8)}
        print(f"  {name}: {len(kept)} images", file=sys.stderr)

    lines = ["# Split leakage report", ""]
    lines.append(f"Threshold: Hamming distance <= {args.threshold} (of 64 bits)")
    lines.append("")
    lines.append("| Split | Images |")
    lines.append("|---|---|")
    for name in splits:
        lines.append(f"| {name} | {len(data[name]['files'])} |")
    lines.append("")

    leak_found = False
    names = list(splits)

    # --- cross-split near-duplicate detection ---
    lines.append("## Cross-split near-duplicates")
    lines.append("")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            print(f"Comparing {a} vs {b}...", file=sys.stderr)
            dist = hamming_matrix(data[a]["hashes"], data[b]["hashes"])
            hits = np.argwhere(dist <= args.threshold)

            n_a = len(set(hits[:, 0].tolist())) if len(hits) else 0
            pct = 100.0 * n_a / len(data[a]["files"])

            if len(hits) == 0:
                lines.append(f"**{a} vs {b}**: clean — no pairs within threshold.")
                lines.append("")
                continue

            leak_found = True
            lines.append(f"**{a} vs {b}**: {len(hits)} near-duplicate pairs, "
                         f"affecting {n_a} `{a}` images ({pct:.1f}% of {a}).")
            lines.append("")
            lines.append("| distance | " + a + " | " + b + " |")
            lines.append("|---|---|---|")
            order = np.argsort(dist[hits[:, 0], hits[:, 1]])
            for k in order[: args.max_examples]:
                ia, ib = hits[k]
                lines.append(f"| {dist[ia, ib]} | `{data[a]['files'][ia].name}` | `{data[b]['files'][ib].name}` |")
            if len(hits) > args.max_examples:
                lines.append(f"| ... | _{len(hits) - args.max_examples} more_ | |")
            lines.append("")

    # --- filename-stem group overlap ---
    lines.append("## Filename group overlap")
    lines.append("")
    lines.append("Sequences like `frame_0001` / `frame_0002` share a group. Groups appearing in "
                 "more than one split indicate a split made without grouping.")
    lines.append("")
    groups = defaultdict(set)
    for name in names:
        for f in data[name]["files"]:
            groups[stem_group(f.name)].add(name)
    shared = {g: s for g, s in groups.items() if len(s) > 1}
    if shared:
        leak_found = True
        lines.append(f"**{len(shared)} group(s) span multiple splits.**")
        lines.append("")
        lines.append("| group | splits |")
        lines.append("|---|---|")
        for g, s in list(shared.items())[: args.max_examples]:
            lines.append(f"| `{g}` | {', '.join(sorted(s))} |")
        if len(shared) > args.max_examples:
            lines.append(f"| ... | _{len(shared) - args.max_examples} more_ |")
    else:
        lines.append("No filename groups span multiple splits.")
    lines.append("")

    # --- verdict ---
    lines.append("## Verdict")
    lines.append("")
    if leak_found:
        lines.append("**LEAKAGE DETECTED.** Metrics from these splits are optimistic and should "
                     "not be reported to a customer.")
        lines.append("")
        lines.append("Re-split by group rather than by image. Group by whichever of these applies: "
                     "source video, capture session, camera/site, subject or patient, production lot, "
                     "or time window. Every image in a group must land in exactly one split.")
        lines.append("")
        lines.append("Note that this check is visual only. It cannot detect leakage from a shared "
                     "subject photographed from different angles, so also confirm the split respects "
                     "whatever grouping exists in the metadata.")
    else:
        lines.append("No leakage detected at this threshold. Consider re-running with a higher "
                     "threshold (e.g. 10-12) to catch same-scene-different-frame cases, and confirm "
                     "the split respects any grouping present in the capture metadata.")
    lines.append("")

    report = "\n".join(lines)
    if args.report:
        Path(args.report).write_text(report)
        print(f"\nReport written to {args.report}", file=sys.stderr)
    print(report)
    return 1 if leak_found else 0


if __name__ == "__main__":
    sys.exit(main())
