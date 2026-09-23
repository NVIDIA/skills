#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Set values in a spec emitted by ``emit_default_spec.py``.

Completes the pair: ``emit_default_spec.py`` produces the canonical spec with
``???`` on mandatory fields, this fills them, and no spec is ever hand-edited.

    --apply-workflow-defaults assets/overlays/kpi_analyze.yaml --set results_dir=/abs/out

Every value in the final spec comes from one of three places, and the flags exist
to keep them apart:

1. **TAO's own default**, for any field nobody mentions. Left alone.
2. **A workflow default** — a setting this workflow requires that differs from
   TAO's default, the same on every run. These live in version control as a flat
   dotted-key YAML under ``assets/overlays/``, one file per stage, and are applied
   with ``--apply-workflow-defaults FILE`` (repeatable, applied in order, first).
3. **A run-specific value** — paths, checkpoints, GPU counts. ``--set KEY=VALUE``,
   applied last.

The split exists because 1 and 2 are indistinguishable in the finished spec. TAO's
analytics default for ``kpi.ignore_sqwidth`` is 0 where this workflow needs 40; a
run that never names it scores a different set of boxes and nothing reports a
difference. Holding workflow defaults in a file applies them on every run and makes
changing one show up as a diff.

A ``--set`` naming a key a workflow-defaults file already set is an error: a run
quietly overriding a documented workflow setting is exactly the case worth catching.
``--allow-workflow-default-override`` permits it where a run genuinely must differ,
and says so on stderr.

Keys are dotted paths into nested mappings. Values are parsed as YAML scalars, so
``8`` is an int, ``true`` a bool, ``null`` None, ``[a, b]`` a list, and anything
else a string — matching how the same text would behave written into the file.

``--set-from-file KEY=FILE`` is ``--set`` with the value read from a file, so a
mapping another script emitted is never retyped by hand. Unlike a workflow-defaults
file, which holds one dotted key per leaf, this assigns a whole block — which is why
it is a separate flag rather than another file to apply.

Creating a key that is not already present requires ``--allow-new``. Without it a
typo'd path is an error rather than a silently added field the loader ignores,
which is the failure mode this script exists to prevent.

``--require-no-mandatory`` exits non-zero if any ``???`` remains. Prefer
``--require-no-mandatory-under BLOCK`` for a stage that uses one block of a
multi-action spec: ``default_specs`` marks fields mandatory for every action it
supports, so the whole-spec check can never pass for, say, a KITTI→COCO conversion
that never reads the ``odvg`` block.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

MANDATORY = "???"


def parse_value(raw: str):
    """Parse a scalar the way PyYAML would if it were written into the file."""
    return yaml.safe_load(raw)


def set_path(tree: dict, dotted: str, value, allow_new: bool,
             materialised: set[str]) -> str:
    """Set tree[a][b][c] = value. Returns a repr of what was there before.

    ``materialised`` accumulates the dotted paths of blocks this invocation turned
    from ``null`` into a mapping, and is shared across every ``--set``.

    A block the schema declares ``null`` -- ``dataset.infer_data_sources`` is one --
    has no declared members, so no member key can be checked against it. Requiring
    the leaf to pre-exist there would reject every key the caller came to add, and
    checking only the first would reject the second. Once a block is materialised
    it stays open for the rest of the call; blocks the schema really defines still
    reject unknown keys.
    """
    parts = dotted.split(".")
    node = tree
    for i, key in enumerate(parts[:-1]):
        path = ".".join(parts[: i + 1])
        if not isinstance(node, dict) or key not in node:
            if not allow_new:
                raise KeyError(f"{path!r} is not in the spec (pass --allow-new to create it)")
            node[key] = {}
            materialised.add(path)
        elif node[key] is None:
            node[key] = {}
            materialised.add(path)
        node = node[key]

    leaf = parts[-1]
    parent = ".".join(parts[:-1])
    if not isinstance(node, dict):
        raise KeyError(f"{parent!r} is not a mapping")
    if leaf not in node and not allow_new and parent not in materialised:
        raise KeyError(f"{dotted!r} is not in the spec (pass --allow-new to create it)")
    previous = node.get(leaf, "<absent>")
    node[leaf] = value
    return repr(previous)


def load_overlay(path: Path) -> dict[str, object]:
    """Load a flat ``dotted.key: value`` overlay.

    Flat rather than nested so each setting is one line: a nested block would
    diff as a block, and a reviewer checking whether ``kpi.ignore_sqwidth`` is
    still 40 would have to read the surrounding structure to find out.

    A mapping value is rejected for the same reason it would be wrong: writing
    ``kpi: {iou_threshold: 0.5}`` replaces the whole ``kpi`` block, silently
    dropping every sibling the schema had emitted.
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping of dotted keys, got {type(data).__name__}")
    for key, value in data.items():
        if not isinstance(key, str):
            raise ValueError(f"{path}: key {key!r} is not a string")
        if isinstance(value, dict):
            raise ValueError(
                f"{path}: {key!r} has a mapping value. Use one dotted key per leaf "
                f"({key}.<field>: <value>) — assigning a whole block replaces it and "
                "drops the fields the schema emitted alongside."
            )
    return data


def remaining_mandatory(tree, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(tree, dict):
        for k, v in tree.items():
            found += remaining_mandatory(v, f"{prefix}{k}.")
    elif isinstance(tree, list):
        for i, v in enumerate(tree):
            found += remaining_mandatory(v, f"{prefix}{i}.")
    elif tree == MANDATORY:
        found.append(prefix.rstrip("."))
    return found


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", required=True, help="Spec to modify.")
    parser.add_argument("--out", default=None, help="Where to write. Defaults to --spec.")
    parser.add_argument("--apply-workflow-defaults", "--overlay",
                        dest="overlay", action="append", default=[], metavar="FILE",
                        help="Settings this workflow requires that differ from TAO's "
                             "defaults, as a flat dotted-key YAML. Repeatable; applied in "
                             "order, before --set. (--overlay is the former name.)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="Run-specific dotted key and YAML-parsed value. Repeatable.")
    parser.add_argument("--set-from-file", "--set-from-yaml",
                        dest="set_from_yaml", action="append", default=[], metavar="KEY=FILE",
                        help="Like --set, but the value is read from a YAML file, so a "
                             "generated block goes in whole instead of being retyped. A file "
                             "with one top-level key contributes that key's value. Repeatable. "
                             "(--set-from-yaml is the former name.)")
    parser.add_argument("--allow-workflow-default-override", "--allow-overlay-override",
                        dest="allow_overlay_override", action="store_true",
                        help="Permit a --set to change a key a workflow-defaults file "
                             "already set. Without it that collision is an error.")
    parser.add_argument("--allow-new", action="store_true",
                        help="Permit creating keys absent from the spec.")
    parser.add_argument("--require-no-mandatory", action="store_true",
                        help="Exit non-zero if any ??? remains anywhere in the spec.")
    parser.add_argument("--require-no-mandatory-under", action="append", default=[],
                        metavar="PREFIX",
                        help="Exit non-zero only if a ??? remains under this dotted prefix. "
                             "Repeatable. A schema dump carries mandatory fields for every "
                             "action it supports, so the whole-spec check cannot pass for a "
                             "stage that uses one block — scope it to the blocks the stage "
                             "actually reads.")
    parser.add_argument("--report-json", default=None,
                        help="Record every key applied and its source, so a run can be "
                             "audited without re-deriving what the overlay contained.")
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()

        spec_path = Path(args.spec).expanduser().resolve()
        if not spec_path.is_file():
            raise FileNotFoundError(f"--spec does not exist: {spec_path}")
        tree = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
        if not isinstance(tree, dict):
            raise ValueError(f"{spec_path}: expected a mapping at the top level")

        # Collected rather than printed as we go: nothing is written until every
        # override succeeds, so reporting one as applied before then is a lie if a
        # later one raises.
        applied: list[str] = []
        materialised: set[str] = set()
        record: dict[str, dict] = {}

        # Overlays first, so --set carries only what varies per run and a
        # collision between the two is detectable rather than last-write-wins.
        from_overlay: dict[str, str] = {}
        for overlay_arg in args.overlay:
            overlay_path = Path(overlay_arg).expanduser().resolve()
            if not overlay_path.is_file():
                raise FileNotFoundError(
                    f"--apply-workflow-defaults does not exist: {overlay_path}")
            for dotted, value in load_overlay(overlay_path).items():
                try:
                    previous = set_path(tree, dotted.strip(), value,
                                        args.allow_new, materialised)
                except KeyError as exc:
                    raise KeyError(
                        f"{exc.args[0]} — from workflow defaults {overlay_path.name}; no changes "
                        "were written, the spec is unmodified"
                    ) from exc
                from_overlay[dotted.strip()] = overlay_path.name
                record[dotted.strip()] = {"source": overlay_path.name, "value": value}
                applied.append(f"  {dotted.strip()}: {previous} -> {value!r}  [{overlay_path.name}]")

        # A generated block is loaded whole rather than retyped as a --set literal:
        # a fold transcribed by hand is a fold that can be typed wrong. A file with
        # one top-level key contributes that key's value, so an emitted
        # `category_mapping:` file drops straight onto `inference.category_mapping`.
        for item in args.set_from_yaml:
            if "=" not in item:
                raise ValueError(f"--set-from-file expects KEY=FILE, got {item!r}")
            dotted, filename = item.split("=", 1)
            dotted = dotted.strip()
            src = Path(filename.strip()).expanduser().resolve()
            if not src.is_file():
                raise FileNotFoundError(f"--set-from-file does not exist: {src}")
            loaded = yaml.safe_load(src.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and len(loaded) == 1:
                loaded = next(iter(loaded.values()))
            if dotted in from_overlay and not args.allow_overlay_override:
                raise ValueError(
                    f"--set-from-file {dotted} collides with workflow defaults "
                    f"{from_overlay[dotted]}; pass --allow-workflow-default-override "
                    "if this run genuinely must differ."
                )
            try:
                previous = set_path(tree, dotted, loaded, args.allow_new, materialised)
            except KeyError as exc:
                raise KeyError(
                    f"{exc.args[0]} — no changes were written; the spec is unmodified"
                ) from exc
            record[dotted] = {"source": src.name, "value": loaded}
            applied.append(f"  {dotted}: {previous} -> <{src.name}>  [set-from-file]")

        for item in args.set:
            if "=" not in item:
                raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
            dotted, raw = item.split("=", 1)
            dotted = dotted.strip()
            if dotted in from_overlay and not args.allow_overlay_override:
                raise ValueError(
                    f"--set {dotted} collides with workflow defaults "
                    f"{from_overlay[dotted]}, which already set it. That file holds "
                    "settings this workflow requires on every run, so "
                    "a --set that changes one is the drift this guards against. Pass "
                    "--allow-workflow-default-override if this run genuinely must differ."
                )
            if dotted in from_overlay:
                print(f"NOTE: --set {dotted} overrides workflow default "
                      f"{from_overlay[dotted]}",
                      file=sys.stderr)
            try:
                previous = set_path(tree, dotted, parse_value(raw),
                                    args.allow_new, materialised)
            except KeyError as exc:
                raise KeyError(
                    f"{exc.args[0]} — no changes were written; the spec is unmodified"
                ) from exc
            record[dotted] = {"source": "--set", "value": parse_value(raw)}
            applied.append(f"  {dotted}: {previous} -> {raw}")

        out = Path(args.out).expanduser().resolve() if args.out else spec_path
        out.parent.mkdir(parents=True, exist_ok=True)

        # Validate before the file lands. Writing first and checking after left an
        # invalid spec on disk on a non-zero exit, which a resume reads as usable.
        left = remaining_mandatory(tree)
        scoped = [
            key for key in left
            if any(key == prefix or key.startswith(f"{prefix}.")
                   for prefix in args.require_no_mandatory_under)
        ]
        blocking = left if args.require_no_mandatory else scoped
        if blocking:
            where = ("" if args.require_no_mandatory
                     else f" under {', '.join(args.require_no_mandatory_under)}")
            print(f"  {len(blocking)} field(s) still ???{where}: "
                  f"{', '.join(blocking[:8])}", file=sys.stderr)
            print(f"  nothing was written; {out} is unchanged", file=sys.stderr)
            return 1

        with out.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(tree, fh, sort_keys=False, default_flow_style=False)
        if args.report_json:
            rp = Path(args.report_json).expanduser().resolve()
            rp.parent.mkdir(parents=True, exist_ok=True)
            rp.write_text(json.dumps({
                "spec": str(spec_path),
                "out": str(out),
                "overlays": [Path(o).name for o in args.overlay],
                "applied": record,
            }, indent=2, default=str), encoding="utf-8")

        for line in applied:
            print(line)
        print(f"spec -> {out}")

        if left:
            print(f"  {len(left)} field(s) still ???: {', '.join(left[:8])}",
                  file=sys.stderr)
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
