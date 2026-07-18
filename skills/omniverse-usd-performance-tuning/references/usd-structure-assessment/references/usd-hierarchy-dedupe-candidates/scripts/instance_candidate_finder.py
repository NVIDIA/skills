# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Instance-candidate finder — read-only duplicate-subtree analyzer.

Implements `instance-candidate-finder-spec.md` (rev 4). Scans a USD
sub-hierarchy, reports sub-hierarchies that recur and could be made
`instanceable`, and classifies how cleanly each group could become a shared
prototype. It NEVER modifies the stage.

Pasteable into the Kit Script Editor: edit the KNOBS block and run. It also
runs standalone for testing — set the stage via the ``DTP_STAGE`` env var or
``argv[1]`` (a USD file path). The core functions (`analyze`, `render_report`,
`validate_knobs`) take an explicit stage + knobs so they are unit-testable
without Kit.

Wiring to the decision core: pass ``--emit-candidates`` to print the
``candidates[]`` packet (JSON) that the co-located ``select_frontier.py`` consumes,
so the finder pipes straight into the frontier disposition step::

    python3 instance_candidate_finder.py <stage.usd> --emit-candidates \
      | python3 select_frontier.py -

The finder confirms reuse (hash + group); ``to_frontier_candidates`` maps each
group to an identity-marked candidate (``kind`` -> identity_signal, else the
explicit ``structural_fallback`` grain); select_frontier decides dispositions.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys

# ============================== KNOBS =====================================
# All values are literal assignments, trivially editable before paste-and-run.
ROOT = "/"
# Default 2 (structure: types + names + attribute names/types, NO values) for the
# broad Phase-2b sweep — it finds repeated NAMED subtrees cheaply without reading
# geometry. Escalate to 3 (folds in attribute VALUES) to CONFIRM a group and split
# near-duplicates; to 4 to split further on time samples / relationship targets.
# Value digesting is byte-based (see _digest_value_token), so level 3 is now
# affordable as a confirmation pass rather than a default whole-stage cost.
HASH_LEVEL = 2
MIN_SUBTREE_PRIMS = 3
MIN_DUPLICATE_COUNT = 2
TOP_N = 25
SHOW_PATHS_PER_GROUP = 8
SKIP_EXISTING_INSTANCES = True
COLLAPSE_NESTED = True
CHECK_INSTANCEABILITY = True
MAX_FINDINGS_PER_GROUP = 6
INCLUDE_ATTRIBUTE_CONNECTIONS = False
# Kind-trust preflight (issue #172). Stage-wide meshes-per-kind sample taken
# during the walk over ALL prims with an authored kind in KIND_TRUST_KINDS
# (never over candidate groups only — those are a biased sample of repeated
# subtrees). Kind is UNTRUSTED for the whole stage when the fraction of
# kind-tagged prims whose subtree holds <= 1 Mesh reaches this threshold, OR
# the median meshes-per-kind is <= 1. Untrusted kind is demoted: candidates
# fall to the structural-fallback grain and boundaries are re-derived at the
# nearest multi-mesh parent / sibling name-base group (monolithic CAD imports
# commonly tag ~every mesh kind=component, which makes kind a non-boundary).
KIND_TRUST_MAX_SINGLE_MESH_FRACTION = 0.80
# ==========================================================================

_DIGEST_REPR_BYTES = 256
_DIGEST_ARRAY_LEN = 16

#: Kinds sampled by the kind-trust preflight: the "part-ish" kinds an exporter
#: over-applies. `assembly` legitimately aggregates and is not sampled.
KIND_TRUST_KINDS = ("component", "subcomponent", "group")

#: One normalizer for all sibling name-base grouping in this tool — the §2.4
#: sibling-name normalization from usd-structure-assessment (strip trailing
#: numeric suffixes / generated copy suffixes: Station_01, Station-2,
#: Station.003, Station001 -> Station). Do not add a second normalizer.
_NAME_BASE_SUFFIX_RE = re.compile(r"(?:[_\-.]?\d+)+$")


def _name_base(name: str) -> str:
    base = _NAME_BASE_SUFFIX_RE.sub("", name)
    return base or name


# numpy gives a C-speed raw-byte view of Vt/array values; without it we fall back
# to repr (correct, just slower). Imported once, guarded.
try:
    import numpy as _np
except Exception:  # pragma: no cover - numpy is normally present in a USD runtime
    _np = None


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _array_bytes(value):
    """Content-stable raw bytes for a NUMERIC array value, or None when the value
    is not one we can safely byte-digest.

    This avoids the O(n) Python ``repr()`` that made ``HASH_LEVEL >= 3`` an
    O(all-geometry) pass on mesh-heavy stages: ``repr`` stringifies every element
    in Python, while ``.tobytes()`` copies the buffer at C speed. Equality is
    preserved exactly — equal arrays produce equal bytes, differing arrays differ —
    so duplicate grouping is unchanged, only faster. String/token/object arrays
    return None (their object-pointer bytes are NOT content, so they MUST keep the
    repr path); such arrays are short, so repr stays cheap there.
    """
    if _np is None:
        return None
    try:
        arr = _np.asarray(value)
    except Exception:
        return None
    if arr.dtype.kind not in ("f", "i", "u", "b", "c"):
        return None
    return arr.tobytes()


def _digest_value_token(value) -> str:
    """§5.1 long-value digesting. Returns a deterministic token for a value.

    Large array values are digested by their raw bytes (C-speed) when numeric;
    everything else keeps the ``repr`` digest. The repr path is the fallback, so
    identical values still produce identical tokens — the byte path is a speed
    optimization, not a semantic change.
    """
    if value is None:
        return "<none>"
    # Array-typed values of length >= 16 are digested by their bytes (numeric) or
    # their repr (fallback).
    try:
        length = len(value)
        is_sized = True
    except TypeError:
        is_sized = False
        length = 0
    if is_sized and length >= _DIGEST_ARRAY_LEN:
        b = _array_bytes(value)
        if b is not None:
            return "<digest:%s>" % _sha(b)
        return "<digest:%s>" % _sha(repr(value).encode("utf-8", "replace"))
    r = repr(value)
    if len(r.encode("utf-8", "replace")) > _DIGEST_REPR_BYTES:
        return "<digest:%s>" % _sha(r.encode("utf-8", "replace"))
    return r


def _attr_value_token(attr) -> str:
    """Default value token for an attribute; <unreadable> if .Get() raises."""
    try:
        return _digest_value_token(attr.Get())
    except Exception:
        return "<unreadable>"


def _attr_sample_token(attr, t) -> str:
    try:
        return _digest_value_token(attr.Get(t))
    except Exception:
        return "<unreadable>"


def _is_xformop(name: str) -> bool:
    return name == "xformOpOrder" or name.startswith("xformOp:")


class _Walker:
    """Encapsulates traversal + hashing for one (stage, knobs) analysis."""

    def __init__(self, stage, hash_level, skip_existing_instances):
        from pxr import Usd  # local import: keep module importable without pxr

        self._Usd = Usd
        self.stage = stage
        self.level = hash_level
        self.skip = skip_existing_instances
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
    def _eligible(self, prim):
        return (
            prim.IsActive()
            and not prim.IsAbstract()
            and prim.GetSpecifier() != self._Usd.SdfSpecifierClass
            if hasattr(self._Usd, "SdfSpecifierClass")
            else (prim.IsActive() and not prim.IsAbstract())
        )

    def _children(self, prim):
        from pxr import Usd, Sdf

        if self.skip:
            kids = prim.GetChildren()  # default predicate: active, defined, !abstract
        else:
            kids = prim.GetFilteredChildren(self._proxy_pred)
        out = []
        for c in kids:
            if not c.IsActive() or c.IsAbstract():
                continue
            if c.GetSpecifier() == Sdf.SpecifierClass:
                continue
            out.append(c)
        return out

    # -- per-prim self tokens --------------------------------------------
    def _self_tokens(self, prim, include_name, include_xformops):
        toks = ["T:" + str(prim.GetTypeName())]
        if include_name:
            toks.append("N:" + prim.GetName())
        if self.level >= 2:
            attrs = [a for a in prim.GetAttributes() if a.HasAuthoredValue()]
            attrs.sort(key=lambda a: a.GetName())  # I8 author-order invariance
            for a in attrs:
                name = a.GetName()
                if not include_xformops and _is_xformop(name):
                    continue
                toks.append("A:%s:%s" % (name, a.GetTypeName()))
                if self.level >= 3:
                    toks.append("V:%s:%s" % (name, _attr_value_token(a)))
                if self.level >= 4:
                    try:
                        times = sorted(a.GetTimeSamples())
                    except Exception:
                        times = []
                    for t in times:
                        toks.append("S:%s:%r:%s" % (name, t, _attr_sample_token(a, t)))
        if self.level >= 4:
            rels = sorted(prim.GetAuthoredRelationships(), key=lambda r: r.GetName())
            for r in rels:
                tgts = r.GetTargets()
                if not tgts:
                    continue
                toks.append("R:%s:%s" % (r.GetName(), ",".join(str(t) for t in tgts)))
        return toks

    def _instance_tokens(self, prim):
        # Opaque-leaf identity: prototype path + prim type. Used identically in
        # full and candidate hashes (name + local opinions ignored).
        proto = prim.GetPrototype() if hasattr(prim, "GetPrototype") else None
        proto_id = str(proto.GetPath()) if proto else "?"
        return ["INST:%s:%s" % (proto_id, prim.GetTypeName())]

    # -- recursion -------------------------------------------------------
    def visit(self, prim):
        path = prim.GetPath().pathString
        if self.skip and prim.IsInstance():
            toks = self._instance_tokens(prim)
            h = _sha(("\x00".join(toks)).encode("utf-8", "replace"))
            self.full[path] = h
            self.cand[path] = h
            self.size[path] = 1
            self.mesh[path] = 0  # opaque instance leaf: not an own authored mesh
            self.bodies[path] = 0  # subtree unwalked: body count unknown
            self.anim[path] = 0  # subtree unwalked: animation unknown
            return
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


def _classify_target(target_prim_path, root_path):
    """INTERNAL if target prim is at/below root; else EXTERNAL. Returns (kind, value)."""
    tp = target_prim_path
    if tp == root_path or tp.startswith(root_path + "/") or root_path == "/":
        if root_path == "/" :
            rel = tp
        else:
            rel = tp[len(root_path):] or "."
        # only INTERNAL when genuinely under root; '/' root makes everything internal-ish
        if root_path == "/":
            return ("INTERNAL", rel)
        return ("INTERNAL", rel)
    return ("EXTERNAL", tp)


def _collect_refs(stage, walker, root_prim, include_connections):
    """Collect outgoing refs inside root subtree, keyed by relative property key.

    Returns dict: key -> list of (kind, value, is_material) tuples preserving
    target order (kind/value computed per target, joined into a per-key seq).
    Actually returns key -> (seq, is_material) where seq is the ordered list of
    (kind, value) for that key's targets on THIS copy.
    """
    from pxr import Sdf, UsdShade

    root_path = root_prim.GetPath().pathString
    result = {}

    def rel_key(prim, prop_name):
        ppath = prim.GetPath().pathString
        if ppath == root_path:
            rel = ""
        else:
            rel = ppath[len(root_path):] if root_path != "/" else ppath
        return rel + "." + prop_name

    def visit(prim):
        if walker.skip and prim.IsInstance() and prim.GetPath().pathString != root_path:
            return
        for r in prim.GetAuthoredRelationships():
            tgts = r.GetTargets()
            if not tgts:
                continue
            seq = []
            for t in tgts:
                kind, val = _classify_target(t.GetPrimPath().pathString, root_path)
                seq.append((kind, val))
            result[rel_key(prim, r.GetName())] = (seq, r.GetName() == "material:binding")
        if include_connections:
            for a in prim.GetAttributes():
                if not a.HasAuthoredValue() and not a.HasAuthoredConnections():
                    continue
                conns = a.GetConnections()
                if not conns:
                    continue
                seq = []
                for t in conns:
                    kind, val = _classify_target(t.GetPrimPath().pathString, root_path)
                    # preserve property suffix verbatim for evidence
                    suffix = "." + t.name if hasattr(t, "name") and t.name else ""
                    seq.append((kind, val + suffix))
                result[rel_key(prim, a.GetName())] = (seq, False)
        for c in walker._children(prim):
            visit(c)

    visit(root_prim)
    return result


def _instanceability(stage, walker, copies, include_connections, max_findings):
    """Classify a group's instanceability. copies = list of prim paths."""
    per_copy = []
    for path in copies:
        prim = stage.GetPrimAtPath(path)
        per_copy.append(_collect_refs(stage, walker, prim, include_connections))
    all_keys = set()
    for c in per_copy:
        all_keys.update(c.keys())

    key_class = {}   # key -> (classification, evidence, is_material)
    for key in all_keys:
        present = [c.get(key) for c in per_copy]
        authored_on_all = all(p is not None for p in present)
        is_material = any(p[1] for p in present if p is not None)
        if not authored_on_all:
            n = sum(1 for p in present if p is not None)
            key_class[key] = ("INCONSISTENT",
                              "%d of %d copies authored" % (n, len(per_copy)), is_material)
            continue
        seqs = [tuple(p[0]) for p in present]
        kinds = set(k for seq in seqs for (k, _v) in seq)
        identical = len(set(seqs)) == 1
        if kinds == {"INTERNAL"} and identical:
            ev = ",".join(v for (_k, v) in seqs[0])
            key_class[key] = ("INTERNAL", ev, is_material)
        elif kinds == {"EXTERNAL"} and identical:
            ev = ",".join(v for (_k, v) in seqs[0])
            key_class[key] = ("CONSISTENT_EXTERNAL", ev, is_material)
        else:
            distinct = len(set(seqs))
            examples = []
            for seq in seqs[:2]:
                examples.append(";".join("%s:%s" % (k, v) for (k, v) in seq))
            key_class[key] = ("INCONSISTENT",
                              "%d distinct target seqs; e.g. %s" % (distinct, " | ".join(examples)),
                              is_material)

    classes = [c for (c, _e, _m) in key_class.values()]
    if all(c == "INTERNAL" for c in classes):  # also true when no keys
        verdict = "GREEN"
    elif any(c == "INCONSISTENT" for c in classes):
        verdict = "RED"
    else:
        verdict = "YELLOW"

    # Findings, prioritized: INCONSISTENT, material CONSISTENT_EXTERNAL, other
    # CONSISTENT_EXTERNAL, INTERNAL. Ascending key within each bucket.
    def bucket(item):
        key, (cls, _ev, ismat) = item
        if cls == "INCONSISTENT":
            return 0
        if cls == "CONSISTENT_EXTERNAL" and ismat:
            return 1
        if cls == "CONSISTENT_EXTERNAL":
            return 2
        return 3

    ordered = sorted(key_class.items(), key=lambda it: (bucket(it), it[0]))
    findings = []
    for key, (cls, ev, ismat) in ordered:
        label = cls
        if cls == "CONSISTENT_EXTERNAL" and ismat:
            label = "CONSISTENT_EXTERNAL (material — inline candidate)"
        findings.append("%s : %s : %s" % (key, label, ev))
    trailer = None
    if len(findings) > max_findings:
        trailer = "... and %d more findings" % (len(findings) - max_findings)
        findings = findings[:max_findings]
    return verdict, findings, trailer


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


def _articulation_boundary(prim, usd_physics, bodies_below=0):
    """True when landing the boundary walk-up ON ``prim`` would freeze internal
    physics motion — instancing makes descendants read-only, so nothing inside
    the prototype may need to move independently per instance.

    PRIMARY signal — ``bodies_below``: RigidBodyAPI prims STRICTLY below
    ``prim``. Per the UsdPhysics schema, RigidBodyAPI engulfs its subtree, so a
    body strictly below the boundary is an independent mover (a jointed link OR
    a free simulated part) whose transform would freeze inside the prototype.
    A body AT the boundary itself is fine — that is the ideal instancing cut
    (VFI factory-structuring: instance at the rigid-body level), because the
    body's motion is expressed on the instance prim's own writable transform.
    Keying on bodies also makes joint-prim LOCATION irrelevant: joints bind
    their bodies via body0/body1 relationships wherever the joint prim lives
    (joint prims are not Xformables; there is no placement convention).

    EXPLICIT signal — ``ArticulationRootAPI`` on the prim: the author declared
    a jointed mechanism here.

    Transform ANIMATION (time samples / clips) is the non-physics analog and
    is checked separately in the walk-up — it needs no UsdPhysics."""
    if usd_physics is None or prim is None or not prim.IsValid():
        return False
    if bodies_below > 0:
        return True
    try:
        return bool(prim.HasAPI(usd_physics.ArticulationRootAPI))
    except Exception:
        return False


def _rederive_boundary_path(stage, walker, path, root_path, usd_physics):
    """Walk UP from a demoted kind-labeled candidate root to the nearest
    multi-mesh ancestor. Returns ``(boundary_path, stopped_by)`` with
    ``stopped_by`` in ``{None, "animation", "articulation", "root"}``.

    NO-COLLATERAL RULE: the walk terminates immediately at any node whose
    subtree holds > 1 mesh — so a component that is ITSELF multi-mesh
    re-derives to itself (it keeps its boundary and only loses the 'kind'
    label as the identity signal, gaining structural_fallback at the SAME
    prim). ARTICULATION STOP: the walk never crosses (or lands on) a prim
    carrying an articulation/edit boundary — it stops BELOW it, so 1-mesh
    links of a robot arm are never re-derived to the arm root.
    """
    current = path
    stopped_by = None
    while walker.mesh.get(current, 0) <= 1:
        parent = current.rsplit("/", 1)[0] or "/"
        if parent == current or parent == root_path or parent not in walker.mesh:
            stopped_by = "root"
            break
        # Animation strictly below the landing target: an animated prim inside
        # the prototype would collapse per-copy motion (and the level-2/3
        # hashes cannot see that samples differ across copies). Animation ON
        # the target itself is fine — it rides the instance prim's writable
        # transform. Pure Usd: works with or without UsdPhysics.
        anim_below = walker.anim.get(parent, 0) - (
            1 if parent in walker.anim_self else 0)
        if anim_below > 0:
            stopped_by = "animation"
            break
        bodies_below = walker.bodies.get(parent, 0) - (
            1 if parent in walker.body_self else 0)
        if _articulation_boundary(stage.GetPrimAtPath(parent), usd_physics,
                                  bodies_below):
            stopped_by = "articulation"
            break
        current = parent
    return current, stopped_by


def _rederive_group(stage, walker, group, root_path, usd_physics):
    """Structural boundary re-derivation for one demoted kind-labeled group.

    Each occurrence walks up to its re-derived boundary
    (`_rederive_boundary_path`). Distinct boundaries are then re-grouped:
    first by candidate hash (the existing reuse confirmation — boundaries that
    genuinely repeat), then hash-singletons by sibling name-base using the one
    §2.4 normalizer (`_name_base`). Returns ``(rederived_groups, stopped_by)``
    where each entry carries the re-derived paths and the walker's stats at
    that boundary.
    """
    boundaries = set()
    stopped_by = None
    for p in group["paths"]:
        b, stop = _rederive_boundary_path(stage, walker, p, root_path, usd_physics)
        boundaries.add(b)
        # Propagate internal-motion stop reasons ("articulation"/"animation");
        # "root" is the ordinary end-of-walk and is not a finding.
        if stop in ("articulation", "animation"):
            stopped_by = stop
    by_hash = {}
    for b in sorted(boundaries):
        by_hash.setdefault(walker.cand.get(b), []).append(b)
    parts = []
    singles = []
    for _h, paths in by_hash.items():
        if len(paths) >= 2:
            parts.append((sorted(paths), "structure_hash"))
        else:
            singles.append(paths[0])
    by_base = {}
    for b in singles:
        parent, _, name = b.rpartition("/")
        by_base.setdefault((parent or "/", _name_base(name)), []).append(b)
    for _key, paths in by_base.items():
        parts.append((sorted(paths), "name_base" if len(paths) >= 2 else "self"))
    out = []
    for paths, basis in parts:
        rep = paths[0]
        out.append({
            "paths": paths,
            "copies": len(paths),
            "hash": walker.cand.get(rep),
            "subtree_prims": walker.size.get(rep, 0),
            "subtree_meshes": walker.mesh.get(rep, 0),
            "group_by": basis,
        })
    out.sort(key=lambda d: d["paths"][0])
    return out, stopped_by


def analyze(stage, *, root=ROOT, hash_level=HASH_LEVEL,
            min_subtree_prims=MIN_SUBTREE_PRIMS,
            min_duplicate_count=MIN_DUPLICATE_COUNT, top_n=TOP_N,
            show_paths_per_group=SHOW_PATHS_PER_GROUP,
            skip_existing_instances=SKIP_EXISTING_INSTANCES,
            collapse_nested=COLLAPSE_NESTED,
            check_instanceability=CHECK_INSTANCEABILITY,
            max_findings_per_group=MAX_FINDINGS_PER_GROUP,
            include_attribute_connections=INCLUDE_ATTRIBUTE_CONNECTIONS,
            kind_trust_max_single_mesh_fraction=KIND_TRUST_MAX_SINGLE_MESH_FRACTION):
    """Read-only analysis. Returns a structured report dict (see render_report)."""
    from pxr import Usd

    root_prim = stage.GetPrimAtPath(root)
    if not root_prim or not root_prim.IsValid():
        return {"error": "ROOT not found: %s" % root}

    walker = _Walker(stage, hash_level, skip_existing_instances)
    walker.visit(root_prim)
    prims_hashed = len(walker.full)

    root_path = root_prim.GetPath().pathString
    # group eligible candidate roots by candidate hash
    groups = {}
    for path, ch in walker.cand.items():
        if path == root_path:
            continue
        prim = stage.GetPrimAtPath(path)
        if skip_existing_instances and prim.IsInstance():
            continue
        if walker.size[path] < min_subtree_prims:
            continue
        groups.setdefault(ch, []).append(path)

    reported = []
    for ch, paths in groups.items():
        if len(paths) < min_duplicate_count:
            continue
        size = walker.size[paths[0]]
        savings = size * (len(paths) - 1)
        reported.append({"hash": ch, "paths": sorted(paths), "subtree_prims": size,
                         "copies": len(paths), "savings": savings})

    reported.sort(key=lambda g: (-g["savings"], -g["subtree_prims"], g["hash"]))

    if collapse_nested:
        kept = []
        kept_paths = set()
        for g in reported:
            if all(any(p == k or p.startswith(k + "/") for k in kept_paths) for p in g["paths"]):
                continue
            kept.append(g)
            kept_paths.update(g["paths"])
        reported = kept

    if check_instanceability:
        for g in reported:
            v, f, tr = _instanceability(stage, walker, g["paths"],
                                        include_attribute_connections, max_findings_per_group)
            g["verdict"] = v
            g["findings"] = f
            g["findings_trailer"] = tr

    # Per-group facts the frontier adapter (to_frontier_candidates) needs. These
    # are additive: render_report ignores them and the §13 acceptance tests do
    # not key off them. ``subtree_meshes`` feeds own_meshes; ``kind`` is the
    # AUTHORED USD kind on the representative root, which sets the candidate's
    # identity_signal so select_frontier.py stays identity-first (the hash only
    # confirms reuse; it never invents strong identity).
    for g in reported:
        rep = g["paths"][0]
        g["subtree_meshes"] = walker.mesh.get(rep, 0)
        kind = ""
        prim = stage.GetPrimAtPath(rep)
        if prim and prim.IsValid():
            try:
                kind = Usd.ModelAPI(prim).GetKind() or ""
            except Exception:
                kind = ""
        g["kind"] = kind

    # Kind-trust preflight (issue #172): decide once, stage-wide, whether the
    # authored kind is a reliable boundary. When it is not, annotate every
    # kind-labeled group as demoted and re-derive its boundary structurally
    # (nearest multi-mesh parent / sibling name-base group). Read-only: the
    # re-derivation only reads the walker's memoized stats and the stage.
    usd_physics = _usd_physics()
    trust_counts, excluded_fine_grain = _filter_kind_trust_sample(
        walker.kind_samples, walker.kind_by_path, walker.mesh)
    kind_trust = _kind_trust_stats(trust_counts,
                                   kind_trust_max_single_mesh_fraction)
    kind_trust["excluded_fine_grain_subcomponents"] = excluded_fine_grain
    kind_trust["articulation_check"] = (
        "enabled" if usd_physics is not None else "unavailable")
    if not kind_trust["kind_trusted"]:
        for g in reported:
            if not g["kind"]:
                continue
            g["kind_untrusted"] = True
            rederived, stopped_by = _rederive_group(
                stage, walker, g, root_path, usd_physics)
            g["rederived"] = rederived
            if stopped_by:
                g["walkup_stopped_by"] = stopped_by

    total_savings = _eliminated_union_size(reported, walker)
    clean_savings = blocked_savings = None
    if check_instanceability:
        clean_savings = _eliminated_union_size(
            [g for g in reported if g["verdict"] != "RED"], walker)
        blocked_savings = total_savings - clean_savings

    return {
        "root": root_path, "hash_level": hash_level, "prims_hashed": prims_hashed,
        "min_subtree_prims": min_subtree_prims, "min_duplicate_count": min_duplicate_count,
        "groups": reported, "total_groups": len(reported),
        "total_savings": total_savings, "clean_savings": clean_savings,
        "blocked_savings": blocked_savings, "check_instanceability": check_instanceability,
        "top_n": top_n, "show_paths_per_group": show_paths_per_group,
        "kind_trust": kind_trust,
    }


def _eliminated_union_size(groups, walker):
    """Unique prims removed if every group collapses to one representative.

    Per-group ``savings`` counts each group in isolation; when a smaller
    repeated subtree lives inside a larger repeated subtree, both groups count
    the same prims, and a plain sum can exceed the stage's own prim count
    (a naive sum can report more "savings" than the stage actually has prims). Union the eliminated copy
    subtrees (every path but each group's representative), drop roots nested
    under an already-counted root, then sum subtree sizes.
    """
    roots = sorted({p for g in groups for p in g["paths"][1:]})
    total = 0
    last = None
    for p in roots:
        if last is not None and (p == last or p.startswith(last + "/")):
            continue
        total += walker.size.get(p, 0)
        last = p
    return total


def validate_knobs(root, hash_level, min_subtree_prims, min_duplicate_count, top_n,
                   show_paths_per_group, max_findings_per_group):
    if not isinstance(hash_level, int) or not (1 <= hash_level <= 4):
        return "HASH_LEVEL out of range (must be int 1..4): %r" % (hash_level,)
    if not isinstance(min_subtree_prims, int) or min_subtree_prims < 1:
        return "MIN_SUBTREE_PRIMS out of range (must be int >= 1): %r" % (min_subtree_prims,)
    if not isinstance(min_duplicate_count, int) or min_duplicate_count < 2:
        return "MIN_DUPLICATE_COUNT out of range (must be int >= 2): %r" % (min_duplicate_count,)
    if not isinstance(top_n, int) or top_n < 1:
        return "TOP_N out of range (must be int >= 1): %r" % (top_n,)
    if not isinstance(show_paths_per_group, int) or show_paths_per_group < 1:
        return "SHOW_PATHS_PER_GROUP out of range (must be int >= 1): %r" % (show_paths_per_group,)
    if not isinstance(max_findings_per_group, int) or max_findings_per_group < 1:
        return "MAX_FINDINGS_PER_GROUP out of range (must be int >= 1): %r" % (max_findings_per_group,)
    if not isinstance(root, str) or not root:
        return "ROOT must be a non-empty USD path string"
    return None


def render_report(report) -> str:
    if "error" in report:
        return report["error"]
    L = []
    L.append("Instance-candidate finder — root=%s HASH_LEVEL=%d" % (report["root"], report["hash_level"]))
    L.append("Hashed %d prims; grouping by candidate hash." % report["prims_hashed"])
    L.append("Duplicate groups: %d (MIN_SUBTREE_PRIMS=%d, MIN_DUPLICATE_COUNT=%d, HASH_LEVEL=%d)"
             % (report["total_groups"], report["min_subtree_prims"],
                report["min_duplicate_count"], report["hash_level"]))
    kt = report.get("kind_trust")
    if kt and not kt.get("kind_trusted", True):
        L.append("KIND-TRUST PREFLIGHT: authored kind does NOT partition geometry "
                 "(single-mesh fraction %.0f%%, median meshes/kind %s, sampled %d) — "
                 "kind-derived identity is demoted to structural_fallback and "
                 "boundaries are re-derived structurally."
                 % (100.0 * (kt.get("single_mesh_fraction") or 0.0),
                    kt.get("median_meshes_per_kind"),
                    kt.get("sampled_kind_prims") or 0))
    for i, g in enumerate(report["groups"][:report["top_n"]]):
        L.append("")
        L.append("[%d] hash=%s subtree_prims=%d copies=%d est_savings=%d"
                 % (i + 1, g["hash"][:16], g["subtree_prims"], g["copies"], g["savings"]))
        if report["check_instanceability"]:
            L.append("    verdict: %s" % g["verdict"])
            for fnd in g.get("findings", []):
                L.append("    - %s" % fnd)
            if g.get("findings_trailer"):
                L.append("    %s" % g["findings_trailer"])
        shown = g["paths"][:report["show_paths_per_group"]]
        for p in shown:
            L.append("    %s" % p)
        extra = len(g["paths"]) - len(shown)
        if extra > 0:
            L.append("    ... and %d more" % extra)
    L.append("")
    L.append("Total potential prim savings (unique prims; nested-group overlap excluded): %d"
             % report["total_savings"])
    if report["check_instanceability"]:
        L.append("Clean savings (GREEN+YELLOW, unique prims): %d" % report["clean_savings"])
        L.append("Blocked savings (RED-only remainder): %d  (re-run at HASH_LEVEL=4 to split RED groups)"
                 % report["blocked_savings"])
    L.append("")
    L.append("Caveats:")
    L.append("  - Advisory only; this tool does not modify the stage.")
    L.append("  - Outgoing references outside a candidate subtree may prevent clean instancing.")
    L.append("  - Material bindings outside a subtree are common; matching local material")
    L.append("    networks should usually be inlined during the rewrite.")
    L.append("  - Default HASH_LEVEL=2 (structure) finds the families; raise to 3 to CONFIRM a")
    L.append("    group and split value-variants, lower to merge near-duplicates into one family.")
    if report["check_instanceability"]:
        L.append("  - GREEN: cleanly instanceable. YELLOW: instanceable after reviewing/inlining")
        L.append("    external deps. RED: not one prototype as-formed; split (HASH_LEVEL=4) or skip.")
    return "\n".join(L)


def to_frontier_candidates(report) -> dict:
    """Map ``analyze()`` groups onto the co-located ``select_frontier.py`` input.

    This is the executable wiring between the finder (which **confirms reuse**
    by hashing) and the decision core (which **decides dispositions**). It is a
    pure function — no ``pxr``, no stage — so it stays unit-testable without USD;
    everything it needs (``subtree_meshes``, authored ``kind``, occurrence paths)
    was attached by ``analyze()`` while it had the stage.

    One finder group becomes one candidate. At the default ``HASH_LEVEL == 2`` the
    hash is STRUCTURAL only (no values), so a group may pool genuine value-variants
    under one structural family — the cheap broad-sweep view. Escalate to
    ``HASH_LEVEL >= 3`` to CONFIRM: the candidate hash then includes attribute
    values, so value-variants split into SEPARATE groups (one prototype per
    variant) — no within-group sub-partition needed, and ``structure_hash ==
    value_hash`` per candidate. (``structure_hash`` and ``value_hash`` are reported
    equal here because one hash is emitted per run; the level chosen decides what
    that hash folds in.) Re-run lower to collapse near-variants, higher to split
    them (§5 / §8.2 footer guidance).

    Identity stays first: ``identity_signal`` comes from the AUTHORED USD
    ``kind`` on the group's root. With no authored identity the candidate is
    emitted as the explicit ``structural_fallback`` grain (the one fallback the
    spec §6.0 allows) — the hash never manufactures a strong identity. The agent
    may refine ``identity_signal`` (e.g. to ``naming`` / ``semantic``) on the
    emitted candidates before piping them to select_frontier.

    KIND-TRUST PREFLIGHT (issue #172): when ``analyze()`` measured stage-wide
    that authored kind does not partition geometry (``report["kind_trust"]
    ["kind_trusted"] is False`` — e.g. a monolithic CAD import where ~every
    kind=component subtree holds one mesh), a kind-labeled group is NOT emitted
    with ``identity_signal='kind'``. It is demoted to the no-identity ladder
    (``identity_signal='none'``, ``grain_source='structural_fallback'``) and
    its boundary is the structurally re-derived one attached by ``analyze()``
    (nearest multi-mesh parent / sibling name-base group), tagged
    ``kind_untrusted: True`` with the sampled stats so the rejection is
    auditable. Reports without a ``kind_trust`` block (synthetic inputs) keep
    the trusted behavior unchanged.
    """
    if "error" in report:
        return {"error": report["error"]}
    kind_trust = report.get("kind_trust") or {}
    kind_demoted = bool(kind_trust) and not kind_trust.get("kind_trusted", True)
    kind_trust_stats = {
        k: kind_trust.get(k)
        for k in ("single_mesh_fraction", "median_meshes_per_kind",
                  "sampled_kind_prims")
    } if kind_trust else {}
    candidates = []
    seen_rederived = set()
    for g in report.get("groups", []):
        paths = list(g.get("paths", []))
        if not paths:
            continue
        kind = g.get("kind") or ""
        if kind and kind_demoted:
            # Demoted kind: emit at the re-derived boundary, never as 'kind'
            # identity. Two source groups under the same parents re-derive to
            # the same boundary set; emit that boundary once.
            for part in (g.get("rederived") or
                         [{"paths": paths, "copies": g["copies"], "hash": g["hash"],
                           "subtree_prims": g["subtree_prims"],
                           "subtree_meshes": g.get("subtree_meshes", 0),
                           "group_by": "self"}]):
                key = tuple(part["paths"])
                if key in seen_rederived:
                    continue
                seen_rederived.add(key)
                boundary_moved = list(part["paths"]) != paths
                cand = _base_candidate(
                    part["paths"], part["hash"], part["copies"],
                    part["subtree_prims"], part["subtree_meshes"],
                    # Verdict was computed for the ORIGINAL grain; it does not
                    # transfer to a moved boundary.
                    None if boundary_moved else g.get("verdict"),
                )
                cand["identity_signal"] = "none"
                cand["grain_source"] = "structural_fallback"
                cand["structural_fallback"] = True
                cand["kind_untrusted"] = True
                cand["kind_untrusted_source_kind"] = kind
                cand["kind_trust"] = dict(kind_trust_stats)
                cand["rederive_group_by"] = part.get("group_by")
                if boundary_moved:
                    cand["rederived_from"] = paths
                if g.get("walkup_stopped_by"):
                    cand["walkup_stopped_by"] = g["walkup_stopped_by"]
                candidates.append(cand)
            continue
        if kind:
            signal, grain = "kind", "identity"
        else:
            signal, grain = "none", "structural_fallback"
        cand = _base_candidate(paths, g["hash"], g["copies"], g["subtree_prims"],
                               g.get("subtree_meshes", 0), g.get("verdict"))
        cand["identity_signal"] = signal
        cand["grain_source"] = grain
        if signal == "none":
            cand["structural_fallback"] = True
        candidates.append(cand)
    out = {"candidates": candidates}
    if kind_trust:
        out["kind_trust"] = kind_trust
    return out


def _base_candidate(paths, group_hash, copies, own_prims, own_meshes, verdict):
    """Shared candidate skeleton for ``to_frontier_candidates`` (identity
    fields are filled in by the caller)."""
    return {
        "id": paths[0],
        "path": paths[0],
        "target_class": "prototype",
        "structure_hash": group_hash,
        "value_hash": group_hash,
        "copy_count": copies,
        "occurrences": list(paths),
        "parents": sorted({(p.rsplit("/", 1)[0] or "/") for p in paths}),
        "own_prims": own_prims,
        "own_meshes": own_meshes,
        "mesh_count": own_meshes,
        "reduction_route": "none",
        # Advisory only: select_frontier decides disposition by identity +
        # cutoff; a RED group still needs external-ref review before the
        # rewrite tool authors a prototype. Carried for the audit/report.
        "instanceability_verdict": verdict,
    }


def _resolve_stage():
    from pxr import Usd
    # Prefer the Kit USD context when running inside Omniverse.
    try:
        import omni.usd  # type: ignore
        st = omni.usd.get_context().get_stage()
        if st:
            return st
    except Exception:
        pass
    # First non-flag argv arg is the stage path (so --emit-candidates etc. do
    # not get mistaken for a USD file path; the value after --output is that
    # flag's argument, not a stage path).
    positional = []
    skip_next = False
    for a in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if a == "--output":
            skip_next = True
            continue
        if a.startswith("--"):
            continue
        positional.append(a)
    path = os.environ.get("DTP_STAGE") or (positional[0] if positional else None)
    if path:
        return Usd.Stage.Open(path)
    return None


def main():
    err = validate_knobs(ROOT, HASH_LEVEL, MIN_SUBTREE_PRIMS, MIN_DUPLICATE_COUNT,
                         TOP_N, SHOW_PATHS_PER_GROUP, MAX_FINDINGS_PER_GROUP)
    if err:
        print(err)
        return
    stage = _resolve_stage()
    if stage is None:
        print("No stage: open a stage in Kit, or set DTP_STAGE / pass a USD path as argv[1].")
        return
    report = analyze(
        stage, root=ROOT, hash_level=HASH_LEVEL, min_subtree_prims=MIN_SUBTREE_PRIMS,
        min_duplicate_count=MIN_DUPLICATE_COUNT, top_n=TOP_N,
        show_paths_per_group=SHOW_PATHS_PER_GROUP,
        skip_existing_instances=SKIP_EXISTING_INSTANCES, collapse_nested=COLLAPSE_NESTED,
        check_instanceability=CHECK_INSTANCEABILITY,
        max_findings_per_group=MAX_FINDINGS_PER_GROUP,
        include_attribute_connections=INCLUDE_ATTRIBUTE_CONNECTIONS,
        kind_trust_max_single_mesh_fraction=KIND_TRUST_MAX_SINGLE_MESH_FRACTION)
    # --output <path>: write the full structured report (analysis groups plus
    # the select_frontier candidate packet) to disk, per the runtime-artifact
    # token-budget policy — keep raw artifacts on disk, read summaries.
    argv = sys.argv[1:]
    if "--output" in argv:
        i = argv.index("--output")
        if i + 1 >= len(argv):
            print("--output requires a file path")
            return
        out_path = argv[i + 1]
        payload = dict(report)
        payload["candidates"] = to_frontier_candidates(report).get("candidates", [])
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print("Report JSON written: %s" % out_path)
    # --emit-candidates: print the select_frontier.py candidate packet (JSON) so
    # the finder pipes straight into the decision core:
    #   python3 instance_candidate_finder.py <stage.usd> --emit-candidates \
    #     | python3 select_frontier.py -
    if "--emit-candidates" in argv:
        print(json.dumps(to_frontier_candidates(report), indent=2, sort_keys=True))
        return
    print(render_report(report))


if __name__ == "__main__":
    main()
