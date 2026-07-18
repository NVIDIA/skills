# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mesh-fragmentation suggester — read-only converter-fan detector.

Implements `mesh-fragmentation-finder-spec.md`. It is the cheap entry point for
the `reduction_route = merge` landing: it **surfaces** parents whose children look
like a CAD/BIM/EDA converter face-explosion (a flat fan of anonymous same-material
`Mesh` prims under a named/`kind`-tagged unit) and **suggests** merging that fan up
to the named boundary, grouped by material. It NEVER modifies the stage and it
NEVER decides the merge — merge is intent/archetype-gated, so the user (or the
archetype default) confirms.

This is deliberately **not** a geometric engine like the instance-candidate
finder. It does **no** vertex-coincidence computation and applies **no** numeric
merge threshold (see OQ12 — proving the weld is O(all-points) and pointless for
detection, since the merge op welds anyway). Its inputs are cheap or already
computed:

  * the Phase-2c ``perf_small_mesh`` validator finding (free — the validator
    already ran), the tininess signal and the entry point; and
  * cheap structural reads per candidate parent: direct child count + type
    uniformity, anonymous child naming (``Mesh_N``) under a named/``kind`` parent,
    and the mesh:material ratio (distinct bound materials across the fan).

The core decision function ``suggest_merges`` takes explicit, pre-computed parent
summaries so it is unit-testable without Kit. The thin ``scan_stage`` builds those
summaries from a live ``pxr`` stage with cheap reads only (no geometry, no points).

Wiring: ``--emit-suggestions`` prints the ``suggestions[]`` packet (JSON). Each
suggestion names the **merge boundary** (the named/``kind`` ancestor to preserve),
the per-material grouping, and the identity disposition the user must confirm, and
routes the ``perf_small_mesh`` population to **merge (re-stitch)** vs
**removeSmallGeometry (delete)** — the one real decision the signals make. The
``mesh-merge-rewrite-spec.md`` route consumes the confirmed group.

    python3 mesh_fragmentation_finder.py <stage.usd> --emit-suggestions \
      --small-mesh-paths small.json
"""
from __future__ import annotations

import json
import os
import re
import sys

# ============================== KNOBS =====================================
# Surfacing heuristics, NOT merge gates. They decide what to SHOW; the merge
# decision is the user's archetype-gated confirmation. All literal for paste-edit.
#
# SUGGEST_MIN_FAN — what counts as a "fan" worth surfacing. A handful of children
#   is not a converter explosion; this is the display floor, not a threshold that
#   gates whether a surfaced fan may merge.
SUGGEST_MIN_FAN = 12
# MESH_MATERIAL_RATIO_HINT — a fan of many meshes binding few distinct materials
#   is the converter-explosion signature (re-stitchable into one mesh per
#   material). Used to surface/rank, not to gate.
MESH_MATERIAL_RATIO_HINT = 4.0
# ANON_CHILD_FRACTION_HINT — fraction of the fan that must be anonymously named
#   (Mesh_N / Xform wrapper) for the "meaningless children under a meaningful
#   unit" pattern to hold.
ANON_CHILD_FRACTION_HINT = 0.75
# MESH_CHILD_FRACTION_HINT — fraction of direct children that are the mesh fan
#   (low nesting / flat). A deep tree is not a flat fan.
MESH_CHILD_FRACTION_HINT = 0.75
TOP_N = 25
# ==========================================================================

# Anonymous = converter-default naming: a generic-noun stem (Mesh, Mesh_12, mesh3,
# Xform_4, node_7), a purely numeric index (7, 0042), or a long hex/uuid-ish token.
# Reference designators (U302, R14, C7) and semantic names are deliberately NOT
# anonymous — a letter-prefixed alphanumeric carries identity, and merging it away
# would dissolve the reference-designator identity a service/BOM twin relies on.
_ANON_RE = re.compile(
    r"^(mesh|xform|node|prim|group|shape|object|obj|geom|part)[_-]?\d*$"
    r"|^\d+$"
    r"|^[0-9a-fA-F]{12,}$",
    re.IGNORECASE,
)

_STRONG_IDENTITY_KINDS = ("assembly", "group", "component", "subcomponent")


def is_anonymous_name(name: str) -> bool:
    """True when a prim name looks converter-generated (no human/semantic identity).

    Conservative: anything that is not clearly a default/numeric token is treated
    as named (identity-bearing), so the suggester errs toward NOT surfacing — it
    never proposes dissolving something that might carry identity.
    """
    if not name:
        return False
    return bool(_ANON_RE.match(name.strip()))


def _parent_has_identity(summary: dict) -> bool:
    """The boundary must carry real identity: a strong `kind`, or a non-anonymous
    (semantic) name. This is what stays addressable after the merge."""
    kind = (summary.get("parent_kind") or "").strip().lower()
    if kind in _STRONG_IDENTITY_KINDS:
        return True
    if summary.get("parent_named") is True:
        return True
    name = summary.get("parent_name")
    if name and not is_anonymous_name(name):
        return True
    return False


def _matches_fragmentation_pattern(s: dict) -> bool:
    """The qualitative converter-explosion pattern — meaningless children under a
    meaningful unit. No magnitude threshold gates the *merge*; these signals only
    decide what to surface."""
    child_count = int(s.get("child_count") or 0)
    mesh_count = int(s.get("child_mesh_count") or 0)
    if mesh_count < SUGGEST_MIN_FAN:
        return False
    # Flat fan: the mesh children dominate (little nesting).
    if child_count and (mesh_count / child_count) < MESH_CHILD_FRACTION_HINT:
        return False
    # Anonymous children under an identity-bearing parent.
    anon_frac = s.get("anonymous_child_fraction")
    if anon_frac is None or anon_frac < ANON_CHILD_FRACTION_HINT:
        return False
    if not _parent_has_identity(s):
        return False
    # High mesh:material ratio — re-stitchable into one mesh per material.
    distinct_materials = max(int(s.get("distinct_materials") or 1), 1)
    if (mesh_count / distinct_materials) < MESH_MATERIAL_RATIO_HINT:
        return False
    return True


def _identity_signal(s: dict) -> str:
    kind = (s.get("parent_kind") or "").strip().lower()
    if kind in _STRONG_IDENTITY_KINDS:
        return "kind"
    name = s.get("parent_name")
    if s.get("parent_named") is True or (name and not is_anonymous_name(name)):
        return "naming"
    return "none"


def suggest_merges(parent_summaries, *, small_mesh_paths=None, instanced_paths=None,
                   knobs=None) -> dict:
    """Pure function: parent summaries -> {suggestions, routed_small_geometry,
    diagnostics}. No stage access, no geometry — unit-testable without Kit.

    ``parent_summaries`` — one dict per candidate parent with cheap reads:
        path, parent_kind, parent_name/parent_named, child_count,
        child_mesh_count, anonymous_child_fraction, distinct_materials.
    ``small_mesh_paths`` — ancestor-or-exact paths flagged ``perf_small_mesh`` by
        the validator (the tininess signal / entry point).
    ``instanced_paths`` — paths the instance-candidate finder already claims as
        repeated subtrees (division of labor; see the spec). A fan that is ALSO a
        repeated subtree is instanced at the component, then merged INSIDE the
        prototype — so it is annotated, not double-claimed here.
    """
    small = set(small_mesh_paths or [])
    instanced = set(instanced_paths or [])
    suggestions = []
    routed_delete = []

    def _under(path, paths):
        return any(path == p or path.startswith(p.rstrip("/") + "/") or p.startswith(path.rstrip("/") + "/")
                   for p in paths)

    for s in parent_summaries:
        path = s.get("path")
        if not path:
            continue
        if not _matches_fragmentation_pattern(s):
            continue
        mesh_count = int(s.get("child_mesh_count") or 0)
        distinct_materials = max(int(s.get("distinct_materials") or 1), 1)
        from_validator = _under(path, small) if small else False
        composes_with_instance = _under(path, instanced) if instanced else False
        suggestions.append({
            "merge_boundary": path,                  # preserved; never merge above it
            "identity_signal": _identity_signal(s),  # what keeps the boundary addressable
            "identity_disposition": "weak",          # the anonymous fan; user confirms
            "reduction_route": "merge",
            "fan_size": mesh_count,
            "distinct_materials": distinct_materials,
            "merge_groups": distinct_materials,      # (scope × material): one mesh per material
            "geomsubset_fallback": distinct_materials > 1,
            "mesh_material_ratio": round(mesh_count / distinct_materials, 2),
            "from_perf_small_mesh": from_validator,
            "composes_with_instance_candidate": composes_with_instance,
            "small_geometry_route": "merge",         # re-stitch real faces, do NOT delete
            "suggestion": (
                "Looks like a converter face-explosion: %d anonymous same-typed meshes "
                "(%d material group(s)) under a named/kind unit. Merge up to '%s', grouped "
                "by material%s? Identity-destroying (weak-identity fan) — confirm per archetype."
                % (mesh_count, distinct_materials, path,
                   " (one mesh + a GeomSubset per material)" if distinct_materials > 1 else "")
            ),
            "note": (
                "Instance at this component FIRST, then merge the fan INSIDE the prototype."
                if composes_with_instance else
                "Render archetype may merge to the named boundary; service/BOM/simulation "
                "keeps component identity and merges only sub-component shards."
            ),
        })

    # perf_small_mesh members NOT part of any surfaced fragmentation fan are the
    # delete candidates (the validator's catalogued removeSmallGeometry remedy):
    # genuinely negligible tiny meshes, not re-stitchable faces.
    surfaced = {x["merge_boundary"] for x in suggestions}
    for p in sorted(small):
        if not any(p == b or p.startswith(b.rstrip("/") + "/") for b in surfaced):
            routed_delete.append(p)

    suggestions.sort(key=lambda x: x["fan_size"], reverse=True)
    return {
        "suggestions": suggestions[: (knobs or {}).get("top_n", TOP_N)],
        "routed_small_geometry": {
            "merge_restitch_boundaries": sorted(surfaced),
            "remove_small_geometry_candidates": routed_delete,
            "note": (
                "perf_small_mesh population split: meshes inside a surfaced fan are real "
                "faces -> route to merge (re-stitch); the rest are negligible -> "
                "removeSmallGeometry (delete). The one decision the signals make."
            ),
        },
        "diagnostics": {
            "parents_examined": len(parent_summaries),
            "fans_surfaced": len(suggestions),
        },
    }


# --------------------------------------------------------------------------
# Thin pxr scan (cheap reads only — no geometry / no points / no coincidence).
# --------------------------------------------------------------------------

def scan_stage(stage, knobs=None) -> list:
    """Build parent summaries from a live USD stage with CHEAP reads only:
    direct child counts, type uniformity, anonymous naming, and the distinct bound
    materials across the fan (a binding-relationship target path read — no
    geometry, no points). Requires ``pxr``; the decision core does not."""
    from pxr import Usd, UsdGeom, UsdShade  # noqa: F401  (import-gated)

    summaries = []
    for prim in stage.Traverse():
        children = list(prim.GetChildren())
        if not children:
            continue
        mesh_children = [c for c in children if c.GetTypeName() == "Mesh"]
        # Xform->single-Mesh wrapper pairs count as mesh-fan members too.
        for c in children:
            if c.GetTypeName() == "Xform":
                gc = list(c.GetChildren())
                if len(gc) == 1 and gc[0].GetTypeName() == "Mesh":
                    mesh_children.append(gc[0])
        if not mesh_children:
            continue
        anon = sum(1 for c in mesh_children if is_anonymous_name(c.GetName()))
        # Distinct bound materials across the fan (cheap rel-target reads).
        mats = set()
        for c in mesh_children:
            try:
                binding = UsdShade.MaterialBindingAPI(c)
                rel = binding.GetDirectBindingRel()
                tgts = rel.GetTargets() if rel else []
                mats.add(str(tgts[0]) if tgts else "<none>")
            except Exception:
                mats.add("<unreadable>")
        kind = ""
        try:
            kind = Usd.ModelAPI(prim).GetKind() or ""
        except Exception:
            kind = ""
        summaries.append({
            "path": str(prim.GetPath()),
            "parent_kind": kind,
            "parent_name": prim.GetName(),
            "child_count": len(children),
            "child_mesh_count": len(mesh_children),
            "anonymous_child_fraction": (anon / len(mesh_children)) if mesh_children else 0.0,
            "distinct_materials": len([m for m in mats if m not in ("<none>", "<unreadable>")]) or len(mats),
        })
    return summaries


def render_report(result: dict) -> str:
    L = ["Mesh-fragmentation suggester — %d parent(s) examined, %d fan(s) surfaced."
         % (result["diagnostics"]["parents_examined"], result["diagnostics"]["fans_surfaced"])]
    for s in result["suggestions"]:
        L.append("  - %s  [fan=%d, materials=%d, ratio=%.1f%s%s]"
                 % (s["merge_boundary"], s["fan_size"], s["distinct_materials"],
                    s["mesh_material_ratio"],
                    ", from perf_small_mesh" if s["from_perf_small_mesh"] else "",
                    ", composes-with-instance" if s["composes_with_instance_candidate"] else ""))
        L.append("      %s" % s["suggestion"])
    rsg = result["routed_small_geometry"]
    if rsg["remove_small_geometry_candidates"]:
        L.append("  removeSmallGeometry (delete) candidates: %d"
                 % len(rsg["remove_small_geometry_candidates"]))
    return "\n".join(L)


def _main(argv) -> int:
    emit = "--emit-suggestions" in argv
    args = [a for a in argv[1:] if not a.startswith("--")]
    small_paths = []
    if "--small-mesh-paths" in argv:
        i = argv.index("--small-mesh-paths")
        if i + 1 < len(argv):
            small_paths = json.loads(open(argv[i + 1]).read())
            args = [a for a in args if a != argv[i + 1]]
    stage_path = args[0] if args else os.environ.get("DTP_STAGE")
    if not stage_path:
        sys.stderr.write("usage: mesh_fragmentation_finder.py <stage.usd> [--emit-suggestions]\n")
        return 2
    from pxr import Usd
    stage = Usd.Stage.Open(stage_path)
    result = suggest_merges(scan_stage(stage), small_mesh_paths=small_paths)
    print(json.dumps(result, indent=2) if emit else render_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
