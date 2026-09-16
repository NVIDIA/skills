#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translate one ``classes.yaml`` into the two mappings TAO needs. Nothing else.

The fold from a detector's vocabulary down to the target classes is TAO's job,
in two places, and this script only feeds them.

**Co-DETR does the fold** via ``inference.category_mapping`` — an ordered mapping
from output category to the original classmap names merged into it. From
``category_mapping.py``: unmapped originals are dropped, a name claimed by two
groups keeps the first with a warning, names absent from the classmap are warned
about and ignored, and an empty remap raises. Crucially it then runs
``apply_category_mapping_groupnms`` — per-output-category soft-NMS *after* the
merge. That matters: one object detected as both ``truck`` and ``car`` becomes two
boxes of the same class the moment they fold together, and only a post-fold NMS
removes the duplicate. Renaming labels afterwards cannot.

**annotations convert does the KITTI->COCO fold** via ``kitti.mapping``::

    labels2cat = {label: k for k, v in cat_map.items() for label in v}
    mapped = labels2cat.get(row_p[0], None)
    if not mapped:
        continue

Because Co-DETR already emitted target names, the mapping handed to the converter
is an identity map over the targets. It is still written explicitly rather than
omitted: with no ``mapping`` the converter auto-derives one by scanning the label
directory, so the COCO category IDs would follow filesystem discovery order and a
target absent from this particular pool would vanish from ``categories``. Those
IDs travel into the ODVG labelmap and then into training, so they need to be
stable across runs.

What is left here is a format translation, and it is worth a script only because
doing it by hand is how ``- car: car`` gets written — a bare string where a list
belongs. Both consumers iterate the value, so a string iterates character by
character, yielding class names ``c``, ``a``, ``r``. Nothing matches, every box is
dropped, and ``annotations convert`` still prints ``Execution status: PASS`` and
exits 0.

Whether the fold actually worked is ``validate_pool_coco.py``'s question, asked of
the COCO that comes out.

Stdlib only apart from PyYAML.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def load_classes(path: Path) -> dict[str, list[str]]:
    """Return {target: [source aliases]} in the order the user declared them.

    Order is preserved deliberately: Co-DETR assigns output category IDs
    ``0..K-1`` in exactly this iteration order.
    """
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    classes = data.get("classes", data) if isinstance(data, dict) else data
    if classes is None:
        raise ValueError(f"{path}: no 'classes' key and file is not a bare list.")

    fold: dict[str, list[str]] = {}
    if isinstance(classes, list):
        # Identity form: keep these names as-is, drop everything else.
        for name in classes:
            fold[str(name)] = [str(name)]
    elif isinstance(classes, dict):
        for target, sources in classes.items():
            target = str(target)
            if sources is None:
                sources = [target]
            if isinstance(sources, str):
                sources = [sources]
            fold[target] = [str(s) for s in sources]
    else:
        raise ValueError(f"{path}: 'classes' must be a list or a mapping.")

    if not fold:
        raise ValueError(f"{path}: no target classes defined.")

    # Co-DETR resolves this by keeping the first group and logging a warning, which
    # is easy to miss in a long inference log. Refuse it here instead: which target
    # a shared source belongs to changes the pseudo-labels, so it is the user's call.
    owner: dict[str, str] = {}
    for target, sources in fold.items():
        for source in sources:
            if source in owner and owner[source] != target:
                raise ValueError(
                    f"{path}: source class '{source}' is mapped to both "
                    f"'{owner[source]}' and '{target}'. Co-DETR would silently keep "
                    f"'{owner[source]}' — pick one explicitly."
                )
            owner[source] = target

    # Target names become COCO categories, then ODVG labels, then the Grounding DINO
    # caption list, and finally the class field of KITTI inference labels. KITTI is
    # space-delimited, so a multi-word target makes those labels unparseable.
    spaced = [t for t in fold if len(t.split()) != 1]
    if spaced:
        raise ValueError(
            f"{path}: target class name(s) contain whitespace: {spaced}. "
            "Targets end up as KITTI class names at inference time and KITTI is "
            "space-delimited. Use an underscore, e.g. 'road_sign'. Source names may "
            "contain spaces — COCO has 'traffic light'."
        )
    return fold


def emit_yaml_mapping(fold: dict[str, list[str]], style: str) -> str:
    """Render a mapping in one of the two shapes TAO expects.

    ``convert`` wants a list of single-key dicts; ``codetr`` wants a plain nested
    mapping. Every scalar goes through ``json.dumps``: YAML is a JSON superset, so
    a quoted scalar is always valid and there is no hand-maintained set of
    characters to escape. Unquoted, a class named ``car,truck`` would emit
    ``[car,truck]`` and parse as two aliases — converting against the wrong
    classes without complaint.
    """
    lines = []
    for target, sources in fold.items():
        prefix = "- " if style == "convert" else "  "
        lines.append(f"{prefix}{json.dumps(target)}:")
        indent = "  " if style == "convert" else "  - "
        for source in sources:
            lines.append(f"{indent}- {json.dumps(source)}" if style == "convert"
                         else f"    - {json.dumps(source)}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classes", required=True, help="Class-mapping YAML.")
    parser.add_argument("--emit-codetr-category-mapping", default=None,
                        help="Write the `category_mapping` block for the Co-DETR inference "
                             "spec — the real fold, applied at detection time.")
    parser.add_argument("--emit-kitti-mapping", default=None,
                        help="Write the `kitti.mapping` YAML for `annotations convert`. "
                             "Identity over the targets, because Co-DETR already folded.")
    parser.add_argument("--emit-classmap", default=None,
                        help="Write the target class list as a newline-separated classmap.txt.")
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        if not any((args.emit_codetr_category_mapping, args.emit_kitti_mapping,
                    args.emit_classmap)):
            print("ERROR: nothing to do — pass at least one --emit-* flag.", file=sys.stderr)
            return 2

        fold = load_classes(Path(args.classes).expanduser().resolve())
        targets = list(fold)

        def write(path_str: str, text: str) -> Path:
            path = Path(path_str).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            return path

        if args.emit_codetr_category_mapping:
            body = emit_yaml_mapping(fold, "codetr")
            path = write(args.emit_codetr_category_mapping, f"category_mapping:\n{body}")
            print(f"codetr category_mapping -> {path}")

        if args.emit_kitti_mapping:
            identity = {t: [t] for t in targets}
            path = write(args.emit_kitti_mapping, emit_yaml_mapping(identity, "convert"))
            print(f"convert kitti.mapping   -> {path}  (identity — Co-DETR already folded)")

        if args.emit_classmap:
            path = write(args.emit_classmap, "\n".join(targets) + "\n")
            print(f"target classmap         -> {path}")

        print()
        for target, sources in fold.items():
            print(f"  {target} <- {', '.join(sources)}")
        print(
            "\nBoth consumers match source names EXACTLY, including case: Co-DETR checks "
            "`orig_name not in name_to_id`\nand the converter does `labels2cat.get(row_p[0])`. "
            "Copy names verbatim from the detector's classmap.\nA name that does not match is "
            "a warning in the log, not an error — run validate_pool_coco.py to catch it."
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
