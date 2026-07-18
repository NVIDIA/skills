#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reuse / descent frontier decision core (shipped as deterministic code).

What this IS
------------
The small, deterministic, identity-first **decision core** of the reuse
analyzer. Given the agent's identity-marked candidate units (the named /
``kind`` / semantic scopes where authoring stops, each with a value-hash and its
occurrence sites), it:

* partitions each structural group by full value-hash — **one prototype per
  genuine value-variant** (the 1UTOPASSY ``[17, 1]`` case);
* applies the band-resolved **externalization cutoff** (copy_count + own size)
  to pick a two-axis disposition (externalize_shared / internal_share /
  keep_local) and, for sub-MINP anonymous repeats, suggests an intent-gated
  reduction_route (point_instance) or marks ``kept_inline_for_merge``;
* records **non-double-counted** savings and the **arc-count contrast** (arcs if
  shared at the frontier vs at mesh level — the load-bearing too-deep signal);
* measures remaining reuse (distinct vs total units) and pivots to the disk tier
  when reuse is low;
* emits ``phase4_targets[]`` + a ``frontier`` block that drop straight into the
  apply-restructure manifest and pass ``validate_manifest_structure``.

What this is NOT
----------------
It does **not** walk USD, hash subtrees, or author anything. The pxr-based
hashing engine is the agent-pasted finder (instance-candidate-finder-spec.md);
the nested-library authoring stays agent-driven per the rewrite-tool spec. This
core just turns identity-marked candidates into a checked frontier so the agent
cannot silently over-share at the mesh level. **Identity defines the grain;
the hash only confirms reuse** — a candidate with no identity signal is refused
a shared disposition unless it is the explicit structural-fallback grain.

Usage:
    python3 select_frontier.py <candidates.json>           # or - for stdin
Exit 0 when a valid frontier is produced, 1 on an identity violation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

#: Evidence-seeded band defaults (overridable per target extent via input "band").
#: Evidence-seeded from a reference data-center assembly run; see units-and-tolerances.md for per-band resolution.
DEFAULT_BAND = {
    "minp": 20,                  # min prims with occurrence >= 2 for nested-library membership
    "externalize_copy_count": 4,  # >= 4 copies across distinct parents
    "externalize_own_prims": 32,  # OR meaningful own size
    "externalize_own_meshes": 16,
}

STRONG_IDENTITY = frozenset({"kind", "naming", "semantic"})


def _occurrences(c: dict) -> int:
    occ = c.get("occurrences")
    if isinstance(occ, list) and occ:
        return len(occ)
    cc = c.get("copy_count")
    return int(cc) if isinstance(cc, int) and not isinstance(cc, bool) else 1


def _share_scope(c: dict) -> str:
    parents = c.get("parents")
    if isinstance(parents, list):
        distinct = len({p for p in parents if p})
        if distinct >= 2:
            return "cross_component"
        if distinct == 1:
            return "internal_parent"
    return c.get("share_scope", "none")


def _meets_externalize_size(c: dict, band: dict) -> bool:
    own_prims = c.get("own_prims") or 0
    own_meshes = c.get("own_meshes") or 0
    return own_prims >= band["externalize_own_prims"] or own_meshes >= band["externalize_own_meshes"]


def select_frontier(payload: dict) -> dict:
    """Pure function: candidates JSON -> {frontier, phase4_targets, diagnostics}.

    Raises ValueError on an identity violation (anonymous shared grain, or an
    identity-destroying route forced onto a strong-identity unit without an
    explicit intent confirmation) — the same rules the manifest contract checks,
    enforced at the source so a bad frontier is never authored in the first place.
    """
    band = dict(DEFAULT_BAND)
    band.update(payload.get("band") or {})
    candidates = payload.get("candidates") or []
    if not isinstance(candidates, list):
        raise ValueError("'candidates' must be a list")

    # Partition each structural group by value-hash: one prototype per genuine
    # value-variant. Variant count drives the [N, 1] outlier handling downstream.
    variant_counts: dict[str, set] = {}
    for c in candidates:
        sh = c.get("structure_hash")
        if sh is not None:
            variant_counts.setdefault(sh, set()).add(c.get("value_hash"))

    targets: list[dict] = []
    diagnostics: list[str] = []
    total_units = 0
    distinct_prototypes = 0

    for c in candidates:
        cid = c.get("id") or c.get("path") or "<unknown>"
        copies = _occurrences(c)
        total_units += copies
        distinct_prototypes += 1  # one prototype authored per value-variant candidate
        signal = c.get("identity_signal", "none")
        grain_source = c.get("grain_source") or ("structural_fallback" if signal == "none" and c.get("structural_fallback") else "identity")
        own_prims = c.get("own_prims") or 0
        own_meshes = c.get("own_meshes") or 0
        scope = _share_scope(c)
        below_minp = own_prims < band["minp"]

        # --- Axis A: identity disposition -----------------------------------
        passes_cutoff = (
            copies >= band["externalize_copy_count"]
            and _meets_externalize_size(c, band)
            and scope == "cross_component"
        )
        if passes_cutoff:
            disposition = "externalize_shared"
        elif copies >= 2 and scope == "internal_parent" and not below_minp:
            disposition = "internal_share"
        else:
            disposition = "keep_local"

        # --- Axis B: reduction route ----------------------------------------
        route = c.get("reduction_route", "none")
        kept_inline = False
        below_cutoff = not passes_cutoff and disposition != "internal_share"
        if route == "none" and below_minp and copies >= band["externalize_copy_count"] and signal in ("none", "structural_fallback"):
            # Sub-MINP anonymous high-count repeats: either point-instance them
            # (intent-gated) or keep them inline for a later within-prototype
            # merge. Default to kept_inline_for_merge to preserve merge-ability.
            if c.get("intent_confirmed"):
                route = "point_instance"
            else:
                kept_inline = True
            disposition = "keep_local"
            below_cutoff = True

        # --- Identity guards (refuse a bad grain at the source) -------------
        is_shared = disposition in ("externalize_shared", "internal_share")
        is_destroying = route in ("point_instance", "merge")
        if is_shared and signal == "none" and grain_source != "structural_fallback":
            raise ValueError(
                f"{cid}: cannot assign shared disposition {disposition!r} to an "
                "anonymous unit (identity_signal 'none') — identity defines the "
                "grain; the hash only confirms reuse. Mark the structural-fallback "
                "grain explicitly (grain_source=structural_fallback) or land on a "
                "named/kind/semantic unit."
            )
        if is_destroying and signal in STRONG_IDENTITY and not c.get("intent_confirmed"):
            raise ValueError(
                f"{cid}: identity-destroying reduction_route {route!r} on a "
                f"strong-identity unit (identity_signal {signal!r}) requires an "
                "explicit intent confirmation (intent_confirmed=true); a named / "
                "kind / semantic part must stay addressable by default."
            )

        non_dc_savings = own_prims * max(copies - 1, 0)
        target = {
            "path": c.get("path") or c.get("target_path") or f"/_proto/{cid}",
            "target_class": c.get("target_class", "prototype"),
            "mesh_count": int(c.get("mesh_count", own_meshes)),
            "identity_disposition": disposition,
            "identity_signal": signal if (is_shared or is_destroying) else c.get("identity_signal", signal),
            "grain_source": grain_source,
            "reduction_route": route,
            "copy_count": copies,
            "own_prims": own_prims,
            "own_meshes": own_meshes,
            "non_double_counted_savings": non_dc_savings,
            "below_externalization_cutoff": below_cutoff,
            "kept_inline_for_merge": kept_inline,
            "share_scope": scope,
            # Arc-count contrast: arcs if shared at the frontier (one per copy)
            # vs arcs at the mesh level (every mesh in every copy). The ratio is
            # the primary too-deep guard and must appear in the report.
            "arc_estimate": {
                "frontier": copies,
                "mesh_level": own_meshes * copies,
            },
            "decision_reason": {
                "identity_signal": signal,
                "stop_condition": "below_floor" if below_minp else "min_meaningful_unit",
            },
        }
        if c.get("value_variant_id") is not None:
            target["value_variant_id"] = c["value_variant_id"]
        if c.get("structure_hash") in variant_counts:
            target["value_variant_count"] = len(variant_counts[c["structure_hash"]])
        if c.get("nested_parent_proto"):
            target["nested_parent_proto"] = c["nested_parent_proto"]
        if c.get("package_group"):
            target["package_group"] = c["package_group"]
        if c.get("descent_level"):
            target["descent_level"] = c["descent_level"]
        targets.append(target)

    shared = [t for t in targets if t["identity_disposition"] in ("externalize_shared", "internal_share")]
    measured_ratio = (distinct_prototypes / total_units) if total_units else 1.0
    # Low measured reuse (mostly-unique parametric assets): pivot to the disk
    # tier rather than forcing more sharing. Heuristic: < 5% of units collapse.
    units_saved = total_units - distinct_prototypes
    low_reuse = bool(shared) and units_saved <= max(1, total_units * 0.05)

    # Audit rollups: count the frontier targets by the three decision axes so the
    # report can show how much rests on strong identity, what dispositions were
    # chosen, and where authoring stopped — without re-walking phase4_targets[].
    by_identity_signal: dict[str, int] = {}
    by_disposition: dict[str, int] = {}
    by_stop_condition: dict[str, int] = {}
    for t in targets:
        by_identity_signal[t["identity_signal"]] = by_identity_signal.get(t["identity_signal"], 0) + 1
        by_disposition[t["identity_disposition"]] = by_disposition.get(t["identity_disposition"], 0) + 1
        sc = t["decision_reason"]["stop_condition"]
        by_stop_condition[sc] = by_stop_condition.get(sc, 0) + 1

    frontier = {
        "reuse_measured": True,
        "distinct_prototypes": distinct_prototypes,
        "total_units": total_units,
        "low_reuse_disk_tier_pivot": low_reuse,
        "by_identity_signal": by_identity_signal,
        "by_disposition": by_disposition,
        "by_stop_condition": by_stop_condition,
        "frontier_estimate_basis": payload.get("frontier_estimate_basis", "exact"),
    }
    if payload.get("descent_entry_level"):
        frontier["descent_entry_level"] = payload["descent_entry_level"]
    if low_reuse:
        diagnostics.append(
            "measured reuse is low (mostly-unique units) — pivot to the disk tier "
            "(unused-primvar removal, fit_primitive, decimation) instead of forcing sharing"
        )

    out: dict[str, Any] = {"frontier": frontier, "phase4_targets": targets}
    if diagnostics:
        out["diagnostics"] = diagnostics
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=str, help="candidates JSON path, or - for stdin")
    args = parser.parse_args(argv)

    raw = sys.stdin.read() if args.input == "-" else Path(args.input).read_text(encoding="utf-8")
    payload = json.loads(raw)
    try:
        result = select_frontier(payload)
    except ValueError as exc:
        print(f"frontier identity violation: {exc}", file=sys.stderr)
        return 1
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
