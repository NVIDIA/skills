#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Quantify the gap between a source benchmark and a customer dataset, and flag which
recipe fields the gap puts at risk.

The point is to replace "their data is different" with measured axes that each map to a
specific recipe decision. Object size as a fraction of image area matters most: COCO
objects average a few percent of image area, and a target set at 0.1% implicates input
resolution, FPN level assignment, anchor scales, and rules out mosaic-style augmentation.

Input: COCO-format annotation JSON for both sides. (Convert other formats first; conversion
is lossy in specific ways -- see references/failure-atlas.md.)

Usage:
    python domain_gap_report.py --source instances_val2017.json --target customer.json
    python domain_gap_report.py --source a.json --target b.json --out gap_report.md
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def load_coco(path):
    with open(path) as f:
        d = json.load(f)
    images = {im["id"]: im for im in d.get("images", [])}
    anns = [a for a in d.get("annotations", []) if not a.get("iscrowd", 0)]
    cats = {c["id"]: c.get("name", str(c["id"])) for c in d.get("categories", [])}
    return images, anns, cats


def stats(path):
    images, anns, cats = load_coco(path)
    if not images:
        raise SystemExit(f"error: no images found in {path}")

    w = np.array([im.get("width", 0) for im in images.values()], dtype=float)
    h = np.array([im.get("height", 0) for im in images.values()], dtype=float)
    valid = (w > 0) & (h > 0)

    per_img = Counter(a["image_id"] for a in anns)
    counts = np.array([per_img.get(i, 0) for i in images], dtype=float)

    # object area as fraction of image area
    fracs, boxw, boxh = [], [], []
    for a in anns:
        im = images.get(a["image_id"])
        if not im:
            continue
        iw, ih = im.get("width", 0), im.get("height", 0)
        if iw <= 0 or ih <= 0:
            continue
        bb = a.get("bbox")
        if not bb or len(bb) != 4:
            continue
        bw, bh = float(bb[2]), float(bb[3])
        if bw <= 0 or bh <= 0:
            continue
        fracs.append((bw * bh) / (iw * ih))
        boxw.append(bw)
        boxh.append(bh)

    fracs = np.array(fracs) if fracs else np.array([0.0])
    boxw = np.array(boxw) if boxw else np.array([1.0])
    boxh = np.array(boxh) if boxh else np.array([1.0])
    aspect = boxw / np.maximum(boxh, 1e-9)

    cls = Counter(cats.get(a["category_id"], str(a["category_id"])) for a in anns)
    total = sum(cls.values()) or 1
    freqs = np.array(sorted(cls.values(), reverse=True), dtype=float)

    return {
        "n_images": len(images),
        "n_anns": len(anns),
        "n_classes": len(cls),
        "img_w_med": float(np.median(w[valid])) if valid.any() else 0.0,
        "img_h_med": float(np.median(h[valid])) if valid.any() else 0.0,
        "img_aspect_med": float(np.median((w[valid] / np.maximum(h[valid], 1e-9)))) if valid.any() else 0.0,
        "img_aspect_iqr": float(np.subtract(*np.percentile(w[valid] / np.maximum(h[valid], 1e-9), [75, 25]))) if valid.any() else 0.0,
        "objs_per_img_med": float(np.median(counts)),
        "objs_per_img_p95": float(np.percentile(counts, 95)),
        "objs_per_img_max": float(counts.max()) if counts.size else 0.0,
        "empty_img_pct": float(100.0 * (counts == 0).mean()) if counts.size else 0.0,
        "area_frac_med": float(np.median(fracs)),
        "area_frac_p05": float(np.percentile(fracs, 5)),
        "area_frac_p95": float(np.percentile(fracs, 95)),
        "box_aspect_med": float(np.median(aspect)),
        "box_aspect_p05": float(np.percentile(aspect, 5)),
        "box_aspect_p95": float(np.percentile(aspect, 95)),
        "imbalance": float(freqs.max() / max(freqs.min(), 1e-9)) if freqs.size else 1.0,
        "top_class_pct": float(100.0 * freqs.max() / total) if freqs.size else 0.0,
        "classes": cls,
    }


def fmt(v, p=2):
    return f"{v:.{p}f}"


def build_flags(s, t):
    """Each flag: (severity, axis, finding, recipe implication)."""
    flags = []

    # --- object size, the dominant axis ---
    if s["area_frac_med"] > 0 and t["area_frac_med"] > 0:
        ratio = t["area_frac_med"] / s["area_frac_med"]
        if ratio < 0.25:
            flags.append((
                "HIGH", "Object size",
                f"Target objects are {1/ratio:.0f}x smaller than source "
                f"({fmt(100*t['area_frac_med'],3)}% vs {fmt(100*s['area_frac_med'],3)}% of image area).",
                "Raise input resolution, shift FPN level assignment toward finer levels, recompute "
                "anchor scales, and disable mosaic/mixup. Report per-size metric breakdown -- aggregate "
                "mAP will hide near-zero small-object recall."))
        elif ratio > 4:
            flags.append((
                "MED", "Object size",
                f"Target objects are {ratio:.0f}x larger than source.",
                "Consider dropping the finest FPN level for speed, and reducing input resolution. "
                "Anchor scales need recomputing upward."))
        else:
            flags.append((
                "OK", "Object size",
                f"Comparable ({ratio:.2f}x source).",
                "Anchor scales and FPN assignment likely transfer; still recompute anchors from data."))

    if t["area_frac_p05"] < 0.0005:
        flags.append((
            "HIGH", "Tiny objects",
            f"5th percentile object is {fmt(100*t['area_frac_p05'],4)}% of image area.",
            "A meaningful fraction of targets may be below the model's minimum detectable size at the "
            "paper's resolution. Evaluate tiling/sliced inference before assuming the architecture is wrong."))

    # --- box aspect ratio ---
    if s["box_aspect_p95"] > 0:
        if t["box_aspect_p95"] > 3 * s["box_aspect_p95"] or t["box_aspect_p05"] < s["box_aspect_p05"] / 3:
            flags.append((
                "HIGH", "Box aspect ratio",
                f"Target aspect spread p05-p95 = {fmt(t['box_aspect_p05'])}-{fmt(t['box_aspect_p95'])} "
                f"vs source {fmt(s['box_aspect_p05'])}-{fmt(s['box_aspect_p95'])}.",
                "Inherited anchor aspect ratios will not cover the target distribution. Recompute by "
                "clustering target box dimensions; also revisit assigner IoU thresholds, which behave "
                "poorly for extreme aspect ratios."))

    # --- scene density ---
    if t["objs_per_img_p95"] > 100:
        flags.append((
            "HIGH", "Scene density",
            f"p95 is {t['objs_per_img_p95']:.0f} objects/image (max {t['objs_per_img_max']:.0f}).",
            "The default max-detections cap (often 100) will truncate recall. Raise it, and raise the "
            "NMS IoU threshold for crowded scenes."))
    elif s["objs_per_img_med"] > 0 and t["objs_per_img_med"] > 3 * s["objs_per_img_med"]:
        flags.append((
            "MED", "Scene density",
            f"Target median {fmt(t['objs_per_img_med'],1)} vs source {fmt(s['objs_per_img_med'],1)} objects/image.",
            "Check the max-detections cap and NMS settings."))

    if t["empty_img_pct"] > 10:
        flags.append((
            "MED", "Empty images",
            f"{fmt(t['empty_img_pct'],1)}% of target images have no annotations.",
            "Confirm these are true negatives rather than unlabeled images -- unlabeled positives train "
            "the model to suppress real objects. Also confirm the loss handles empty targets; some repos "
            "assume at least one object."))

    # --- dataset scale ---
    if t["n_images"] < s["n_images"] / 20:
        flags.append((
            "HIGH", "Dataset size",
            f"Target has {t['n_images']} images vs source {s['n_images']} ({s['n_images']/max(t['n_images'],1):.0f}x smaller).",
            "The paper's schedule will overfit badly. Fine-tune from the released checkpoint with a much "
            "shorter schedule, LR reduced roughly 10x, shortened warmup, weaker augmentation, and possibly "
            "a frozen stem. Reason in total gradient steps, not epochs."))

    # --- class balance ---
    if t["imbalance"] > 50 and t["imbalance"] > 5 * s["imbalance"]:
        flags.append((
            "HIGH", "Class imbalance",
            f"Target max/min class frequency ratio is {t['imbalance']:.0f}x "
            f"(source {s['imbalance']:.0f}x); top class is {fmt(t['top_class_pct'],1)}% of instances.",
            "Loss will be dominated by the majority class. Consider class-balanced sampling or loss "
            "weighting, and always report per-class metrics rather than aggregate mAP."))

    if t["n_classes"] != s["n_classes"]:
        flags.append((
            "INFO", "Class count",
            f"Target has {t['n_classes']} classes, source {s['n_classes']}.",
            "Head must be reinitialized. Verify the framework's num_classes semantics (N vs N+1 for "
            "background) and check every place a CLASSES list is defined."))

    # --- resolution / aspect ---
    if s["img_w_med"] > 0 and t["img_w_med"] > 0:
        r = (t["img_w_med"] * t["img_h_med"]) / (s["img_w_med"] * s["img_h_med"])
        if r > 4 or r < 0.25:
            flags.append((
                "MED", "Image resolution",
                f"Target median {t['img_w_med']:.0f}x{t['img_h_med']:.0f} vs source "
                f"{s['img_w_med']:.0f}x{s['img_h_med']:.0f}.",
                "Revisit the resize policy. Downscaling high-resolution imagery to the paper's input size "
                "can destroy small objects; consider tiling instead."))

    if t["img_aspect_iqr"] > 0.5:
        flags.append((
            "MED", "Aspect variability",
            f"Target image aspect ratio IQR is {fmt(t['img_aspect_iqr'])}.",
            "Letterbox vs stretch resize matters more here. Fix the policy in one shared preprocessing "
            "artifact and verify export parity across the full aspect range."))

    return flags


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="source benchmark COCO json (what the paper trained on)")
    ap.add_argument("--target", required=True, help="customer dataset COCO json")
    ap.add_argument("--out", default=None, help="write markdown report here")
    args = ap.parse_args()

    s = stats(args.source)
    t = stats(args.target)

    L = ["# Domain gap report", ""]
    L.append(f"Source: `{Path(args.source).name}` — Target: `{Path(args.target).name}`")
    L.append("")
    L.append("## Measured axes")
    L.append("")
    L.append("| Axis | Source | Target |")
    L.append("|---|---|---|")
    rows = [
        ("Images", f"{s['n_images']}", f"{t['n_images']}"),
        ("Annotations", f"{s['n_anns']}", f"{t['n_anns']}"),
        ("Classes", f"{s['n_classes']}", f"{t['n_classes']}"),
        ("Median image size", f"{s['img_w_med']:.0f}x{s['img_h_med']:.0f}", f"{t['img_w_med']:.0f}x{t['img_h_med']:.0f}"),
        ("Objects/image (median)", fmt(s['objs_per_img_med'], 1), fmt(t['objs_per_img_med'], 1)),
        ("Objects/image (p95)", fmt(s['objs_per_img_p95'], 1), fmt(t['objs_per_img_p95'], 1)),
        ("Images with no objects", f"{fmt(s['empty_img_pct'],1)}%", f"{fmt(t['empty_img_pct'],1)}%"),
        ("Object area % (median)", fmt(100 * s['area_frac_med'], 3), fmt(100 * t['area_frac_med'], 3)),
        ("Object area % (p05)", fmt(100 * s['area_frac_p05'], 4), fmt(100 * t['area_frac_p05'], 4)),
        ("Object area % (p95)", fmt(100 * s['area_frac_p95'], 2), fmt(100 * t['area_frac_p95'], 2)),
        ("Box aspect (p05-p95)", f"{fmt(s['box_aspect_p05'])}-{fmt(s['box_aspect_p95'])}",
         f"{fmt(t['box_aspect_p05'])}-{fmt(t['box_aspect_p95'])}"),
        ("Class imbalance (max/min)", f"{s['imbalance']:.0f}x", f"{t['imbalance']:.0f}x"),
    ]
    for r in rows:
        L.append(f"| {r[0]} | {r[1]} | {r[2]} |")
    L.append("")

    flags = build_flags(s, t)
    order = {"HIGH": 0, "MED": 1, "INFO": 2, "OK": 3}
    flags.sort(key=lambda f: order.get(f[0], 9))

    L.append("## Recipe risk flags")
    L.append("")
    n_high = sum(1 for f in flags if f[0] == "HIGH")
    if n_high:
        L.append(f"**{n_high} high-severity flag(s).** Each names a recipe field that cannot be "
                 "inherited from the paper unchanged.")
    else:
        L.append("No high-severity flags. The recipe is more likely to transfer, but still recompute "
                 "all DERIVED fields from the target data.")
    L.append("")
    for sev, axis, finding, action in flags:
        L.append(f"### [{sev}] {axis}")
        L.append("")
        L.append(f"{finding}")
        L.append("")
        L.append(f"**Implication:** {action}")
        L.append("")

    L.append("## Target class distribution (top 15)")
    L.append("")
    L.append("| class | instances | % |")
    L.append("|---|---|---|")
    tot = sum(t["classes"].values()) or 1
    for name, n in t["classes"].most_common(15):
        L.append(f"| {name} | {n} | {fmt(100*n/tot,1)} |")
    L.append("")
    L.append("Classes with very few instances will have unreliable per-class metrics regardless of "
             "model quality. Decide up front whether they are in scope.")
    L.append("")

    L.append("## Next")
    L.append("")
    L.append("Carry these flags into Phase T (`references/recipe-fields.md`). Every deviation from the "
             "paper's recipe should cite one of them as its justification, and every DERIVED field must "
             "be recomputed from the target data rather than copied.")
    L.append("")

    report = "\n".join(L)
    if args.out:
        Path(args.out).write_text(report)
        print(f"Report written to {args.out}", file=sys.stderr)
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
