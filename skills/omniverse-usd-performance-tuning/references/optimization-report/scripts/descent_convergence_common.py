#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared descent-convergence gate logic (#186).

The Phase-4 premature-merge / premature-decimate gate was implemented twice —
once in the report validator (``validate_report.py``) and once in the outcome
oracle scorer (``tools/oracle/score_run.py``). The two copies drifted: the
oracle copy triggered dedup on a bare ``authored_prim_delta_pct < 0`` (a
merge/removeSmallMeshes run drops authored prims with NO dedup, so that
false-failed), and both copies scanned the free-text ``result`` field with a
loose ``instance``/``prototype`` marker set that false-matched strings like
"126 instances preserved" and "point-instance aware".

This module is the single source of truth for the gate's pure decision
functions and constants. It has NO file I/O and NO argparse — callers supply an
already-parsed ``report`` dict, a list of manifest dicts, and (for convergence)
a NORMALIZED frontier dict with these keys:

    descent_converged                     : bool
    final_rescan_new_groups_above_floor   : int | None
    convergence_waived_by_user            : bool
    convergence_waiver_reason             : str | None

Each caller keeps its own frontier-source adapter (the validator reads manifest
``frontier`` blocks + the report ``structural_summary`` mirror; the oracle reads
the report mirror only) and maps its source fields onto those normalized keys.
"""
from __future__ import annotations

from typing import Any

#: Operation name/method substrings that mean dedup / instancing / restructure
#: happened THIS RUN — i.e. a de-duplication frontier descent was in play and its
#: convergence is meaningful. TIGHTENED (#186 finding 2): bare "instance" and
#: "prototype" are deliberately EXCLUDED. They false-matched benign strings such
#: as "126 instances preserved" and "point-instance aware" when the free-text
#: result/summary field was scanned, tripping the gate on runs that did no dedup.
#: These markers are unambiguous dedup/instancing/externalize op names.
DEDUP_OP_MARKERS = (
    "deduplicate", "instanceable", "internal_reference", "internal_share",
    "externalize",
)

#: Phase-4 GEOMETRY ops whose presence requires a CONVERGED dedup frontier first.
#: Matched as lowercase substrings against report ``operations[]`` name/method and
#: ``target_coverage.entries[].operations[]``. These are the lossy /
#: identity-affecting reductions that, run before the frontier converges, fuse or
#: coarsen geometry that further sharing would have collapsed (premature-merge /
#: premature-decimate inflation). Within-prototype mesh merge is detected via the
#: declared ``structural_summary.merge_applied`` / ``preservation`` flags, not by a
#: bare "merge" name match (which also hits the lossless probe).
PHASE4_GEOMETRY_OP_MARKERS = (
    "decimat", "fitprimitiv", "quantiz", "remesh", "subdivid",
    "triangulat", "removesmall", "mergevertices", "dicemesh", "sparsemesh",
)

#: Coverage-entry roles that mean "a restructure happened this run", so their
#: presence is a this-run dedup signal.
RESTRUCTURE_ROLES = frozenset(
    {"assembly_root", "prototype", "shared_layer", "loadable_subasset"}
)


def _is_number(x: Any) -> bool:
    """True for a real int/float, excluding bool (which is an int subclass)."""
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def iter_operation_strings(report: Any):
    """Yield the lowercased name+method of each op in ``report['operations']`` and
    each string in ``target_coverage.entries[].operations[]``.

    Deliberately JOINS the op ``name`` and ``method`` but does NOT include the
    free-text ``result``/``summary`` field (#186 finding 2): the result text
    carries prose like "126 instances preserved" / "point-instance aware" that a
    substring marker match mistook for a dedup op. Only the structured
    name/method identify what the op WAS.
    """
    if not isinstance(report, dict):
        return
    for op in report.get("operations", []) or []:
        if isinstance(op, dict):
            parts = [op.get(key) for key in ("name", "method")]
            joined = " ".join(p for p in parts if isinstance(p, str))
            if joined:
                yield joined.lower()
    coverage = report.get("target_coverage")
    if isinstance(coverage, dict):
        for entry in coverage.get("entries", []) or []:
            if isinstance(entry, dict):
                for value in entry.get("operations", []) or []:
                    if isinstance(value, str):
                        yield value.lower()


def did_phase4_geom(report: Any) -> bool:
    """True when a lossy / identity-affecting Phase-4 geometry op ran.

    Signals: ``structural_summary.merge_applied`` is True, OR
    ``preservation.allow_merge`` is True, OR ``preservation.rendered_mesh_merged_count``
    > 0, OR any op name/method contains a PHASE4_GEOMETRY_OP_MARKER.
    """
    if not isinstance(report, dict):
        return False
    ss = report.get("structural_summary") or report.get("structural") or {}
    if isinstance(ss, dict) and ss.get("merge_applied") is True:
        return True
    pres = report.get("preservation")
    if isinstance(pres, dict):
        if pres.get("allow_merge") is True:
            return True
        merged = pres.get("rendered_mesh_merged_count")
        if _is_number(merged) and merged > 0:
            return True
    for text in iter_operation_strings(report):
        if any(marker in text for marker in PHASE4_GEOMETRY_OP_MARKERS):
            return True
    return False


def did_dedup(report: Any, manifests: Any = ()) -> bool:
    """True when the run performed hierarchy dedup / instancing / restructure THIS
    RUN — so a frontier descent was in play and its convergence is meaningful.

    Keyed on THIS-RUN signals ONLY, never on absolute post-state. The triggers:

    (a) an apply-restructure manifest carrying ``phase4_targets`` or ``frontier``;
    (b) ``structural_summary.reuse_measured`` is True (this run measured reuse);
    (c) a ``metrics[]`` entry ``name == "instance_count"`` with BOTH ``before`` and
        ``after`` present as real numbers and ``after > before`` (a real increase);
    (d) a DEDUP_OP_MARKER in ``iter_operation_strings(report)``;
    (e) a ``target_coverage.entries[]`` with ``role`` in RESTRUCTURE_ROLES.

    It MUST NOT trigger on a bare ``authored_prim_delta_pct < 0`` (a merge /
    removeSmallMeshes-only run drops authored prims with no dedup — #186 finding 1),
    nor on a bare absolute ``unique_prototype_count`` / ``instance_ratio`` /
    ``instance_count`` post-state (an asset already instanced at source and
    optimized for disk did no NEW dedup this run).
    """
    for manifest in manifests or []:
        if isinstance(manifest, dict) and (
            manifest.get("phase4_targets") or manifest.get("frontier")
        ):
            return True
    if not isinstance(report, dict):
        return False
    ss = report.get("structural_summary") or report.get("structural") or {}
    if isinstance(ss, dict) and ss.get("reuse_measured") is True:
        return True
    for metric in report.get("metrics", []) or []:
        if isinstance(metric, dict) and metric.get("name") == "instance_count":
            before = metric.get("before")
            after = metric.get("after")
            if _is_number(before) and _is_number(after) and after > before:
                return True
    for text in iter_operation_strings(report):
        if any(marker in text for marker in DEDUP_OP_MARKERS):
            return True
    coverage = report.get("target_coverage")
    if isinstance(coverage, dict):
        for entry in coverage.get("entries", []) or []:
            if isinstance(entry, dict) and entry.get("role") in RESTRUCTURE_ROLES:
                return True
    return False


def is_converged_or_waived(frontier: Any) -> bool:
    """True when a NORMALIZED frontier dict records a converged descent OR a valid
    user waiver. Fails CLOSED on ambiguity (#186 findings 3 & 4).

    Waiver branch — a waiver counts ONLY when ``convergence_waived_by_user`` is
    truthy AND ``convergence_waiver_reason`` is a non-empty string. The schema
    says a waiver "Requires ...waiver_reason. Silence is NOT a waiver", so a
    reasonless waiver does NOT clear the gate.

    Converged branch — counts ONLY when ``descent_converged is True`` AND
    ``final_rescan_new_groups_above_floor`` is an int (not bool) EQUAL to 0. The
    schema says convergence "requires ... == 0"; a None/absent rescan field must
    NOT count as converged (fail closed — absence is not evidence).
    """
    if not isinstance(frontier, dict):
        return False
    if frontier.get("convergence_waived_by_user"):
        reason = frontier.get("convergence_waiver_reason")
        return isinstance(reason, str) and bool(reason.strip())
    if frontier.get("descent_converged") is True:
        rescan = frontier.get("final_rescan_new_groups_above_floor")
        return _is_number(rescan) and isinstance(rescan, int) and rescan == 0
    return False
