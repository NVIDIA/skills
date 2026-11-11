# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Composed subtree inventory and model-kind evidence used by assessment."""
from __future__ import annotations

import hashlib

INCLUDE_CLASS_PRIMS = False


KIND_TRUST_MAX_SINGLE_MESH_FRACTION = 0.80
KIND_TRUST_MIN_SAMPLE_COVERAGE = 0.50


KIND_TRUST_KINDS = ("component", "subcomponent", "group")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_xformop(name: str) -> bool:
    return name == "xformOpOrder" or name.startswith("xformOp:")


class _Walker:
    """Collect subtree summaries using caller-supplied nomination tokens."""

    def __init__(self, stage, tokenize, include_class_prims=INCLUDE_CLASS_PRIMS):
        from pxr import Usd
        self._Usd = Usd
        self.stage = stage
        self.include_classes = include_class_prims
        self._self_tokens = tokenize
        self.full = {}        # path -> full hash
        self.cand = {}        # path -> candidate hash
        self.size = {}        # path -> subtree size
        self.mesh = {}        # path -> Mesh-typed prim count in subtree (own_meshes)
        # Kind-trust preflight sample (issue #172): (path, kind, subtree mesh
        # count) of EVERY fully-walked prim whose authored kind is in
        # KIND_TRUST_KINDS. This is the STAGE-WIDE population, not the candidate
        # groups (which only hold repeated subtrees and would bias the
        # single-mesh fraction). kind_by_path supports the fine-grain-authoring
        # exclusion (nearest kind-bearing ancestor lookup) in
        # _filter_kind_trust_sample.
        self.kind_samples = []
        self.kind_by_path = {}
        # Internal-motion signals for the boundary walk-up, bubbled like mesh
        # counts, each with the set of prims carrying the signal directly (so
        # the walk-up can distinguish AT-the-boundary from STRICTLY-below).
        # PHYSICS: subtree RigidBodyAPI tally. Per the UsdPhysics schema,
        # RigidBodyAPI engulfs its subtree (everything below moves rigidly
        # with it), so a body AT a candidate boundary is the ideal instancing
        # cut, while a body STRICTLY BELOW the boundary is an independent
        # mover whose transform would freeze inside the prototype — jointed or
        # free. Keying on bodies makes joint-prim location irrelevant (joints
        # bind bodies via body0/body1 relationships wherever they live).
        # ANIMATION: subtree tally of prims with time-varying xformOp
        # attributes (authored time samples / clips). Same at-root vs
        # strictly-below rule: animation ON the boundary rides the instance
        # prim's writable transform; animation strictly below would collapse
        # into the shared prototype (and level-2/3 hashes cannot see that the
        # samples differ across copies). Works without UsdPhysics.
        self.bodies = {}
        self.body_self = set()
        self.anim = {}
        self.anim_self = set()
        self._physics = _usd_physics()
        self._proxy_pred = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)

    def _authored_kind(self, prim) -> str:
        try:
            return self._Usd.ModelAPI(prim).GetKind() or ""
        except Exception:
            return ""

    # -- eligibility / children ------------------------------------------

    def _children(self, prim):
        from pxr import Usd, Sdf

        if self.include_classes:
            # GetChildren()'s default predicate drops abstract prims, so a class
            # Scope has no children under it. GetAllChildren() sees them.
            kids = prim.GetAllChildren()
        else:
            kids = prim.GetFilteredChildren(self._proxy_pred)
        out = []
        for c in kids:
            if not c.IsActive():
                continue
            if c.GetSpecifier() == Sdf.SpecifierClass or c.IsAbstract():
                if not self.include_classes:
                    continue
            out.append(c)
        return out

    # -- recursion -------------------------------------------------------
    def visit(self, prim):
        path = prim.GetPath().pathString
        kids = self._children(prim)
        for c in kids:
            self.visit(c)
        child_full = [self.full[c.GetPath().pathString] for c in kids]
        size = 1 + sum(self.size[c.GetPath().pathString] for c in kids)
        child_tok = ["C:" + h for h in child_full]
        full_toks = self._self_tokens(prim, True, True) + child_tok
        cand_toks = self._self_tokens(prim, False, False) + child_tok
        self.full[path] = _sha(("\x00".join(full_toks)).encode("utf-8", "replace"))
        self.cand[path] = _sha(("\x00".join(cand_toks)).encode("utf-8", "replace"))
        self.size[path] = size
        # Mesh-typed prim count in this subtree — additive bookkeeping for the
        # frontier adapter's own_meshes; does not affect any hash.
        self.mesh[path] = (1 if str(prim.GetTypeName()) == "Mesh" else 0) + sum(
            self.mesh[c.GetPath().pathString] for c in kids
        )
        # Subtree rigid-body tally (0 when UsdPhysics is unavailable, matching
        # kind_trust.articulation_check='unavailable').
        own_body = 0
        if self._physics is not None:
            try:
                if prim.HasAPI(self._physics.RigidBodyAPI):
                    own_body = 1
                    self.body_self.add(path)
            except Exception:
                own_body = 0
        self.bodies[path] = own_body + sum(
            self.bodies[c.GetPath().pathString] for c in kids
        )
        # Subtree transform-animation tally (pure Usd — no UsdPhysics needed):
        # a prim counts when any authored xformOp attribute is time-varying
        # (samples or value clips). ValueMightBeTimeVarying is the cheap,
        # conservative check designed for exactly this question.
        own_anim = 0
        try:
            for a in prim.GetAttributes():
                if (_is_xformop(a.GetName()) and a.HasAuthoredValue()
                        and a.ValueMightBeTimeVarying()):
                    own_anim = 1
                    self.anim_self.add(path)
                    break
        except Exception:
            own_anim = 0
        self.anim[path] = own_anim + sum(
            self.anim[c.GetPath().pathString] for c in kids
        )
        # Kind-trust sample: tally the subtree mesh count for every walked prim
        # carrying an authored part-ish kind. Opaque instance leaves (early
        # return above) are excluded — their subtree is not walked, so their
        # mesh count is unknown, and they are already shared anyway.
        k = self._authored_kind(prim)
        if k in KIND_TRUST_KINDS:
            self.kind_samples.append((path, k, self.mesh[path]))
            self.kind_by_path[path] = k
        return


def _kind_prims_on_stage(stage, root_prim):
    """Count prims with a part-ish authored kind, instance proxies included.

    The walker's sample can only hold what it walked. When existing instances
    are skipped, every kind-bearing prim inside one is absent from it, and the
    gate then computes a fraction over whatever is left. Measured on a Revit MEP
    export: 82,239 of 85,405 kind-bearing prims sat inside instances, so the
    verdict came from 784 prims and read as a clean result.
    """
    from pxr import Usd

    pred = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
    total = 0
    for prim in Usd.PrimRange(root_prim, pred):
        try:
            kind = Usd.ModelAPI(prim).GetKind() or ""
        except Exception:
            kind = ""
        if kind in KIND_TRUST_KINDS:
            total += 1
    return total


def _kind_trust_stats(kind_mesh_counts, max_single_mesh_fraction):
    """Kind-trust preflight (issue #172): stage-wide meshes-per-kind stats.

    Input is the walker's stage-wide sample (subtree mesh count per authored
    part-ish kind). Kind is UNTRUSTED when ``single_mesh_fraction >=
    max_single_mesh_fraction`` OR ``median_meshes_per_kind <= 1`` — i.e. when
    kind does not partition geometry into multi-mesh parts and the real reuse
    unit is almost certainly a parent prim. An empty sample (no authored
    part-ish kinds) leaves kind trivially trusted: there is nothing to demote.
    """
    counts = sorted(kind_mesh_counts)
    sampled = len(counts)
    if not sampled:
        return {
            "sampled_kind_prims": 0,
            "single_mesh_fraction": 0.0,
            "median_meshes_per_kind": None,
            "max_single_mesh_fraction": max_single_mesh_fraction,
            "kind_trusted": True,
        }
    single = sum(1 for c in counts if c <= 1)
    fraction = single / float(sampled)
    mid = sampled // 2
    if sampled % 2:
        median = float(counts[mid])
    else:
        median = (counts[mid - 1] + counts[mid]) / 2.0
    trusted = not (fraction >= max_single_mesh_fraction or median <= 1)
    return {
        "sampled_kind_prims": sampled,
        "single_mesh_fraction": round(fraction, 4),
        "median_meshes_per_kind": median,
        "max_single_mesh_fraction": max_single_mesh_fraction,
        "kind_trusted": trusted,
    }


def _apply_sample_coverage(kind_trust, kind_prims_on_stage, walked_kind_prims,
                           min_coverage=KIND_TRUST_MIN_SAMPLE_COVERAGE):
    """Withhold the kind verdict when the walk did not cover the stage.

    Coverage is what the walk SAW (`walked_kind_prims`) against what is on the
    stage, not what the gate finally scored. The fine-grain-subcomponent
    exclusion in `_filter_kind_trust_sample` removes prims the walk did see, on
    purpose, and must not count as blindness. What must count is a prim the walk
    never reached: inside a skipped instance, or under a class Scope.

    `kind_trusted` becomes None -- neither trusted nor untrusted -- so callers
    that demote on False leave the candidates alone, and the report says what is
    missing instead of asserting a fraction it cannot support.
    """
    sampled = walked_kind_prims
    kind_trust["kind_prims_on_stage"] = kind_prims_on_stage
    kind_trust["walked_kind_prims"] = walked_kind_prims
    kind_trust["min_sample_coverage"] = min_coverage
    if not kind_prims_on_stage:
        kind_trust["sample_coverage"] = 1.0
        kind_trust["insufficient_sample"] = False
        return kind_trust
    coverage = sampled / float(kind_prims_on_stage)
    kind_trust["sample_coverage"] = round(coverage, 4)
    if coverage >= min_coverage:
        kind_trust["insufficient_sample"] = False
        return kind_trust
    kind_trust["insufficient_sample"] = True
    kind_trust["kind_trusted"] = None
    kind_trust["single_mesh_fraction"] = None
    kind_trust["median_meshes_per_kind"] = None
    kind_trust["reason"] = (
        "sample covered %d of %d kind-bearing prims (%.1f%%, floor %.0f%%); "
        "re-run with skip_existing_instances=False, and with "
        "include_class_prims=True if the prototypes are a class Scope"
        % (sampled, kind_prims_on_stage, 100.0 * coverage, 100.0 * min_coverage)
    )
    return kind_trust


def _filter_kind_trust_sample(kind_samples, kind_by_path, mesh_by_path):
    """Drop intentional fine-grain authoring from the kind-trust sample.

    A single-mesh ``subcomponent`` nested under a multi-mesh kind-bearing
    ancestor is the WELL-AUTHORED fine-grain pattern (individually addressable
    parts for BOM / service selection under a real multi-mesh component), not
    the junk-CAD signature — which tags single-mesh ``component``s with no
    multi-mesh kind structure above them. Counting the former would falsely
    demote kind stage-wide on well-authored assets. Model-hierarchy convention
    keeps the two disjoint (components do not nest inside components), so this
    exclusion cannot mask the junk pattern.

    Returns ``(counts, excluded)``: the filtered mesh-count list for
    ``_kind_trust_stats`` and the number of excluded fine-grain subcomponents
    (reported for audit).
    """
    counts = []
    excluded = 0
    for path, kind, count in kind_samples:
        if kind == "subcomponent" and count <= 1:
            anc = path
            nearest = None
            while True:
                anc = anc.rsplit("/", 1)[0]
                if not anc or anc == "/":
                    break
                if anc in kind_by_path:
                    nearest = anc
                    break
            if nearest is not None and mesh_by_path.get(nearest, 0) > 1:
                excluded += 1
                continue
        counts.append(count)
    return counts, excluded


def _usd_physics():
    """UsdPhysics, or None when unavailable (the articulation guard is then
    skipped gracefully and noted in the kind_trust diagnostics)."""
    try:
        from pxr import UsdPhysics
        return UsdPhysics
    except Exception:
        return None
