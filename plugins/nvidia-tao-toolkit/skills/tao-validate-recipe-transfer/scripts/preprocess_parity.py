#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Assert numerically that the training, eval, and export preprocessing paths agree.

Preprocessing mismatch between training and inference is the failure mode that actually
kills deployments: excellent validation mAP, garbage in production, and a long debugging
loop because nothing ever errors. Letterbox vs stretch, BGR/RGB, normalization constants,
resize-before-vs-after augmentation, interpolation mode -- all silent.

Two modes:

1. SPEC MODE (no user code needed). Give it a YAML/JSON preprocessing spec and it renders
   the tensor, reports the fingerprint, and runs a set of self-checks that catch the
   common misconfigurations.

       python preprocess_parity.py --spec preproc.yaml --image sample.jpg

2. COMPARE MODE. Point it at two or more callables that each take a file path and return
   a CHW float array, and it asserts they produce identical tensors.

       python preprocess_parity.py --image sample.jpg \
           --path train=mypkg.data:train_transform \
           --path export=mypkg.deploy:onnx_preprocess

   Each target is 'module:attribute'. The attribute must be callable as fn(path) -> array
   of shape (C,H,W) or (H,W,C).

Spec format (YAML or JSON):
    resize: letterbox        # letterbox | stretch | shortest_side
    size: [640, 640]         # H, W  (or a single int for shortest_side)
    pad_value: 114
    interpolation: bilinear  # bilinear | nearest | bicubic
    channel_order: RGB       # RGB | BGR
    scale: 255.0             # divide by this before normalization
    mean: [0.485, 0.456, 0.406]
    std:  [0.229, 0.224, 0.225]

Exit codes: 0 = paths agree / spec sane, 1 = mismatch or suspicious configuration.
"""

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

INTERP = {"nearest": Image.NEAREST, "bilinear": Image.BILINEAR, "bicubic": Image.BICUBIC}


def load_spec(path):
    text = Path(path).read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        import yaml
        return yaml.safe_load(text)
    except ImportError:
        raise SystemExit("error: spec is not JSON and PyYAML is not installed. "
                         "Install pyyaml or supply the spec as JSON.")


def apply_spec(spec, img_path):
    img = Image.open(img_path).convert("RGB")
    w0, h0 = img.size
    mode = spec.get("resize", "letterbox")
    size = spec.get("size", [640, 640])
    if isinstance(size, int):
        size = [size, size]
    th, tw = int(size[0]), int(size[1])
    interp = INTERP.get(str(spec.get("interpolation", "bilinear")).lower(), Image.BILINEAR)
    pad = float(spec.get("pad_value", 114))

    meta = {"orig": (h0, w0), "mode": mode}

    if mode == "stretch":
        img = img.resize((tw, th), interp)
        arr = np.asarray(img, dtype=np.float32)
        meta["scale"] = (th / h0, tw / w0)
        meta["pad"] = (0, 0, 0, 0)
    elif mode == "shortest_side":
        s = th / min(h0, w0)
        nw, nh = max(1, round(w0 * s)), max(1, round(h0 * s))
        img = img.resize((nw, nh), interp)
        arr = np.asarray(img, dtype=np.float32)
        meta["scale"] = (s, s)
        meta["pad"] = (0, 0, 0, 0)
    else:  # letterbox
        s = min(th / h0, tw / w0)
        nw, nh = max(1, round(w0 * s)), max(1, round(h0 * s))
        img = img.resize((nw, nh), interp)
        canvas = np.full((th, tw, 3), pad, dtype=np.float32)
        top, left = (th - nh) // 2, (tw - nw) // 2
        canvas[top:top + nh, left:left + nw] = np.asarray(img, dtype=np.float32)
        arr = canvas
        meta["scale"] = (s, s)
        meta["pad"] = (top, th - nh - top, left, tw - nw - left)

    if str(spec.get("channel_order", "RGB")).upper() == "BGR":
        arr = arr[:, :, ::-1].copy()

    scale = float(spec.get("scale", 255.0))
    if scale:
        arr = arr / scale
    mean = np.array(spec.get("mean", [0.0, 0.0, 0.0]), dtype=np.float32)
    std = np.array(spec.get("std", [1.0, 1.0, 1.0]), dtype=np.float32)
    arr = (arr - mean) / std
    return np.transpose(arr, (2, 0, 1)).astype(np.float32), meta


def to_chw(a):
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 4 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 3:
        raise ValueError(f"expected 3D array, got shape {a.shape}")
    if a.shape[0] not in (1, 3) and a.shape[2] in (1, 3):
        a = np.transpose(a, (2, 0, 1))
    return a


def fingerprint(a):
    return {"shape": tuple(a.shape), "min": float(a.min()), "max": float(a.max()),
            "mean": float(a.mean()), "std": float(a.std()),
            "per_channel_mean": [round(float(x), 4) for x in a.reshape(a.shape[0], -1).mean(1)]}


def spec_self_checks(arr, spec, meta):
    """Catch the misconfigurations that survive to production.

    Returns (issues, notes). Issues are likely bugs and fail the run. Notes are things
    that are correct-but-worth-verifying-by-hand, and do not fail the run -- keeping
    these separate matters, because a checker that cries wolf on every valid config
    gets ignored exactly when it is right.
    """
    warns = []
    notes = []
    fp = fingerprint(arr)

    mean = spec.get("mean", [0, 0, 0])
    std = spec.get("std", [1, 1, 1])
    scale = float(spec.get("scale", 255.0))

    # scale/mean unit mismatch: ImageNet-style mean with no /255
    if max(mean) < 1.5 and scale in (1.0, 0, None):
        warns.append("mean/std look like 0-1 units but `scale` is not 255 -- the tensor will be "
                     "off by ~255x. This is the classic silent normalization bug.")
    if max(mean) > 1.5 and scale == 255.0:
        warns.append("mean/std look like 0-255 units (e.g. 123.675) but `scale` is 255 -- "
                     "normalization is being applied to already-scaled values.")

    if abs(fp["mean"]) > 3 or fp["std"] > 5:
        warns.append(f"normalized tensor has mean={fp['mean']:.3f} std={fp['std']:.3f}; expected "
                     "roughly mean~0 std~1 for a natural image. Check scale/mean/std units.")

    if str(spec.get("channel_order", "RGB")).upper() == "BGR":
        notes.append("channel_order is BGR. Confirm the pretrained backbone expects BGR -- most "
                     "torchvision/timm weights expect RGB, and OpenCV-based repos often expect BGR. "
                     "A swap costs accuracy without ever erroring.")

    if meta["mode"] == "letterbox":
        top, bottom, left, right = meta["pad"]
        if max(meta["pad"]) > 0:
            notes.append(f"letterbox padding applied (t={top} b={bottom} l={left} r={right}). "
                         "Verify post-processing un-maps predicted coordinates back to original image "
                         "space -- unmapped boxes are the most common 'boxes are shifted' bug.")
    elif meta["mode"] == "stretch":
        h0, w0 = meta["orig"]
        sh, sw = meta["scale"]
        if abs(sh - sw) / max(sh, sw) > 0.1:
            notes.append(f"stretch resize distorts aspect ratio by {abs(sh-sw)/max(sh,sw)*100:.0f}% on "
                         "this image. If the paper used letterbox, anchor priors and learned box "
                         "statistics will not match.")

    if len(set(map(tuple, [mean]))) and len(mean) != 3:
        warns.append(f"mean has {len(mean)} entries; expected 3.")
    if len(std) != 3:
        warns.append(f"std has {len(std)} entries; expected 3.")

    return warns, notes


def _corr(x, y):
    x, y = x.ravel(), y.ravel()
    xs, ys = x.std(), y.std()
    if xs < 1e-9 or ys < 1e-9:
        return 0.0
    return float(((x - x.mean()) * (y - y.mean())).mean() / (xs * ys))


def diagnose(a, b):
    """Rank likely causes of a train/export tensor mismatch.

    Returns an ordered list of hypotheses rather than a single verdict. Several bugs
    produce overlapping signatures -- a missing /255 and wrong mean/std both show up as a
    large std ratio -- so committing to one cause misleads. Ranked evidence lets the
    engineer check the cheap hypothesis first.

    Correlation is used rather than exact equality because the usual bugs happen *before*
    normalization: a BGR swap upstream of a per-channel mean/std leaves the reversed
    tensors highly correlated but not equal, which an allclose check would miss.
    """
    hyps = []  # (score, message)
    d = np.abs(a - b)
    per = d.reshape(d.shape[0], -1).mean(1)
    sa, sb = float(a.std()), float(b.std())
    ratio = sb / sa if sa > 1e-9 else float("inf")
    corr = _corr(a, b)

    # Correlation is the key discriminator. Any per-channel normalization change is affine
    # and preserves correlation (~1.0); a geometric change -- different resize, pad, or crop --
    # misaligns pixels and destroys it. Without this split, a pad mismatch reads as a
    # normalization bug, because both show a std ratio far from 1.
    geometric = corr < 0.8

    # --- channel order ---
    if a.shape[0] == 3:
        direct, rev = _corr(a, b), _corr(a, b[::-1])
        if rev > direct + 0.05:
            msg = (f"RGB/BGR swap: channel-reversed correlation {rev:.3f} exceeds direct {direct:.3f}")
            if per.size == 3 and per[1] < per[[0, 2]].mean() * 0.3:
                msg += "; middle channel matches while outer channels differ, the classic signature"
            hyps.append((10.0 + (rev - direct), msg))

    # --- scaling (a /255 present on one side only) ---
    for f, label in ((255.0, "one path skips the /255 scaling (other is 255x larger)"),
                     (1 / 255.0, "one path applies /255 twice (other is 255x smaller)")):
        if 0.7 < ratio / f < 1.4 and not geometric:
            hyps.append((9.0, f"{label}; std ratio is {ratio:.1f}"))

    # --- geometry: resize / pad / crop ---
    if geometric:
        hyps.append((8.5, f"geometric mismatch: correlation is only {corr:.3f}, so the two paths "
                          "are not looking at the same pixels. Check resize policy (letterbox vs "
                          "stretch vs shortest-side), pad amount and position, crop, and target size"))
    if d.ndim == 3 and d.shape[1] > 16 and d.shape[2] > 16:
        h, w = d.shape[1], d.shape[2]
        bh, bw = h // 8, w // 8
        interior = d[:, bh:-bh, bw:-bw]
        border_sum = d.sum() - interior.sum()
        border_n = d.size - interior.size
        bd = border_sum / max(border_n, 1)
        idd = float(interior.mean()) if interior.size else 0.0
        if idd > 1e-9 and bd / idd > 1.5:
            hyps.append((8.6 * min(bd / idd / 3, 1.0),
                         f"resize/pad/crop policy differs: border difference is {bd/idd:.1f}x the "
                         "interior (letterbox vs stretch, different pad value, or a different crop)"))
        elif idd <= 1e-9 and border_sum > 0:
            hyps.append((8.0, "difference is confined to the border: padding differs while the "
                              "image content matches"))

    # --- normalization constants ---
    if abs(ratio - 1) > 0.15 and not geometric and not any("255" in m for _, m in hyps):
        hyps.append((6.0, f"different mean/std normalization constants; std ratio {ratio:.3f}"))
    if corr > 0.98 and per.max() > 1e-3 and abs(ratio - 1) <= 0.15:
        hyps.append((6.5, "same pixels with an affine offset: mean differs, std matches -- check "
                          "the mean constants"))

    # --- interpolation / ordering ---
    if corr > 0.9 and float(d.mean()) < 0.2 * max(sa, 1e-9):
        hyps.append((4.0, "small diffuse difference: likely a different interpolation mode, or "
                          "resize-then-augment vs augment-then-resize ordering"))

    hyps.sort(key=lambda x: -x[0])
    out = [m for _, m in hyps[:3]]
    if not out:
        out.append(f"unclassified; per-channel mean diff {[round(float(x), 4) for x in per]}, "
                   f"std ratio {ratio:.3f}, correlation {corr:.3f}")
    elif len(hyps) > 1:
        out[0] = "most likely: " + out[0]
        out[1:] = ["also consistent with: " + m for m in out[1:]]
    return out


def resolve(target):
    if ":" not in target:
        raise SystemExit(f"error: --path target must be module:attribute, got {target!r}")
    mod, attr = target.split(":", 1)
    try:
        m = importlib.import_module(mod)
    except Exception as e:
        raise SystemExit(f"error: cannot import {mod!r}: {e}")
    fn = getattr(m, attr, None)
    if fn is None or not callable(fn):
        raise SystemExit(f"error: {target!r} is not callable")
    return fn


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="sample image (use several, including extreme aspect ratios)")
    ap.add_argument("--spec", default=None, help="preprocessing spec YAML/JSON (spec mode)")
    ap.add_argument("--path", action="append", default=[],
                    help="name=module:attribute, repeatable (compare mode)")
    ap.add_argument("--tol", type=float, default=1e-5, help="max allowed absolute difference")
    args = ap.parse_args()

    if not args.spec and len(args.path) < 2:
        raise SystemExit("error: supply --spec, or at least two --path entries to compare")

    if not Path(args.image).exists():
        raise SystemExit(f"error: image not found: {args.image}")

    failed = False
    tensors = {}

    if args.spec:
        spec = load_spec(args.spec)
        arr, meta = apply_spec(spec, args.image)
        tensors["spec"] = arr
        print("=== Spec mode ===")
        print(json.dumps(fingerprint(arr), indent=2))
        print(f"\nresize={meta['mode']} scale={tuple(round(float(x),4) for x in meta['scale'])} "
              f"pad(t,b,l,r)={meta['pad']} orig={meta['orig']}")
        warns, notes = spec_self_checks(arr, spec, meta)
        if warns:
            failed = True
            print(f"\n--- {len(warns)} likely bug(s) ---")
            for w in warns:
                print(f"  [FAIL] {w}")
        else:
            print("\nSpec self-checks passed -- no likely bugs found.")
        if notes:
            print(f"\n--- {len(notes)} item(s) to verify by hand ---")
            for n in notes:
                print(f"  [note] {n}")

    for item in args.path:
        if "=" not in item:
            raise SystemExit(f"error: --path entries must be name=module:attribute, got {item!r}")
        name, target = item.split("=", 1)
        fn = resolve(target)
        try:
            tensors[name] = to_chw(fn(args.image))
        except Exception as e:
            raise SystemExit(f"error: calling {target!r} failed: {e}")

    if len(tensors) >= 2:
        print("\n=== Parity comparison ===")
        names = list(tensors)
        ref = names[0]
        for n in names:
            print(f"{n:>12}: {fingerprint(tensors[n])}")
        print()
        for other in names[1:]:
            a, b = tensors[ref], tensors[other]
            if a.shape != b.shape:
                print(f"  [FAIL] {ref} vs {other}: shape {a.shape} != {b.shape}")
                failed = True
                continue
            d = np.abs(a - b)
            mx = float(d.max())
            if mx <= args.tol:
                print(f"  [PASS] {ref} vs {other}: max abs diff {mx:.3e}")
            else:
                failed = True
                bad = int((d > args.tol).sum())
                print(f"  [FAIL] {ref} vs {other}: max abs diff {mx:.4f} over {bad} elements "
                      f"({100*bad/d.size:.1f}%)")
                for line in diagnose(a, b):
                    print(f"         -> {line}")

    print()
    if failed:
        print("RESULT: mismatch or suspicious configuration. Resolve before training or export -- "
              "this class of bug does not surface until production.")
    else:
        print("RESULT: paths agree. Re-run across several images, including the most extreme aspect "
              "ratios in the dataset, since padding bugs are aspect-dependent.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
