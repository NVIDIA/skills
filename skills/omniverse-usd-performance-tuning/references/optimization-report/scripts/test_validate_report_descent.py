#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dependency-free tests for the descent-convergence gate (validate_descent_convergence).

Run: python3 test_validate_report_descent.py   (exit 0 = all pass, 1 = a failure)
No pytest / third-party deps, matching validate_report.py's standalone ethos.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_report as V  # noqa: E402
import descent_convergence_common as dcc  # noqa: E402

CASES = []


def case(name, report, manifests, *, expect_error, marker=None):
    CASES.append((name, report, manifests, expect_error, marker))


# Building blocks
DEDUP_SS = {"unique_prototype_count": 48, "instance_ratio": 0.0016}
INSTANCE_METRIC = [{"name": "instance_count", "before": 0, "after": 126}]
DECIMATE_OP = [{"order": 4, "name": "decimateMeshes", "method": "0.1mm", "result": "faces -55%"}]
MERGE_SS = {"unique_prototype_count": 48, "merge_applied": True}
CONVERGED_SS = {"frontier_descent_converged": True, "frontier_final_rescan_new_groups_above_floor": 0}
CONVERGED_MANIFEST = {"phase4_targets": [{"path": "/p", "target_class": "prototype", "mesh_count": 5}],
                      "frontier": {"reuse_measured": True, "descent_converged": True,
                                   "final_rescan_new_groups_above_floor": 0}}

# 1. monolith decimate, NO dedup -> gate N/A -> PASS
case("monolith_decimate_no_dedup",
     {"operations": DECIMATE_OP, "target_coverage": {"entries": [{"path": "/m", "role": "monolith"}]}},
     [], expect_error=False)

# 2. dedup + decimate, NO convergence recorded -> FAIL (FF1)
case("dedup_decimate_unconverged",
     {"structural_summary": DEDUP_SS, "metrics": INSTANCE_METRIC, "operations": DECIMATE_OP},
     [], expect_error=True, marker="descent-convergence gate")

# 3. dedup + decimate, report mirror converged -> PASS
case("dedup_decimate_converged_report",
     {"structural_summary": {**DEDUP_SS, **CONVERGED_SS}, "metrics": INSTANCE_METRIC, "operations": DECIMATE_OP},
     [], expect_error=False)

# 4. dedup + merge, manifest frontier converged -> PASS
case("dedup_merge_converged_manifest",
     {"structural_summary": MERGE_SS, "operations": [{"order": 8, "name": "merge", "method": "Parent Prim", "result": "-6k"}]},
     [CONVERGED_MANIFEST], expect_error=False)

# 5. dedup + merge, manifest descent_converged true BUT rescan 5 -> FAIL (FF2 contradiction)
case("converged_but_residue_contradiction",
     {"structural_summary": MERGE_SS},
     [{"frontier": {"descent_converged": True, "final_rescan_new_groups_above_floor": 5}}],
     expect_error=True, marker="final_rescan_new_groups_above_floor is 5")

# 6. dedup + decimate, waiver recorded -> PASS
case("dedup_decimate_waived",
     {"structural_summary": {**DEDUP_SS, "frontier_convergence_waived": True,
                             "frontier_convergence_waiver_reason": "user time budget"},
      "metrics": INSTANCE_METRIC, "operations": DECIMATE_OP},
     [], expect_error=False)

# 7. pure lossless dedup, NO phase4 geometry op -> gate does not fire -> PASS
case("dedup_only_no_geometry_op",
     {"structural_summary": DEDUP_SS, "metrics": INSTANCE_METRIC,
      "operations": [{"order": 2, "name": "internal_share(coarse)", "method": "x", "result": "-27%"}]},
     [], expect_error=False)

# 8. THE RETROSPECTIVE LOOPHOLE: role=monolith + dedup metrics + merge, no convergence -> FAIL (FF1)
case("monolith_role_but_dedup_metrics_and_merge",
     {"structural_summary": MERGE_SS, "metrics": INSTANCE_METRIC,
      "preservation": {"allow_merge": True, "rendered_mesh_merged_count": 6803},
      "target_coverage": {"entries": [{"path": "/m", "role": "monolith"}]}},
     [], expect_error=True, marker="descent-convergence gate")

# 9. ALREADY-INSTANCED DISK PIVOT: prototypes already > 0 at source + a geometry op,
#    but NO new instancing this run (instance_count after == before), NO dedup
#    op-marker, NO manifest, NO restructure role -> the dedup trigger must NOT fire
#    on absolute post-state, so FF1 stays N/A -> PASS (no spurious failure).
case("already_instanced_no_new_dedup_disk_pivot",
     {"structural_summary": {"unique_prototype_count": 64, "instance_ratio": 0.42},
      "metrics": [{"name": "instance_count", "before": 126, "after": 126}],
      "operations": DECIMATE_OP,
      "target_coverage": {"entries": [{"path": "/m", "role": "monolith"}]}},
     [], expect_error=False)

# --- #186 finding-specific regressions ----------------------------------------

# F1 (finding 1): a merge/removeSmallMeshes-only run drops authored prims with NO
# dedup. authored_prim_delta_pct is deeply negative, but there is no reuse_measured,
# no dedup op-marker, no manifest, no restructure role, no instance increase ->
# did_dedup must be False -> the gate must NOT fire (the old oracle copy's bare
# authored_prim_delta_pct < 0 trigger false-failed exactly this run).
case("f1_merge_only_authored_prim_drop_is_not_dedup",
     {"structural_summary": {"authored_prim_delta_pct": -18.0, "merge_applied": True},
      "operations": [{"order": 5, "name": "removeSmallMeshes", "method": "0.5mm"},
                     {"order": 6, "name": "merge", "method": "Parent Prim", "result": "-6k prims"}],
      "target_coverage": {"entries": [{"path": "/m", "role": "monolith"}]}},
     [], expect_error=False)

# F2 (finding 2): the free-text result "126 instances preserved" and a
# "point-instance aware" method must NOT be read as a dedup op. Only name+method
# are scanned, and the tightened markers exclude bare "instance". No real dedup ->
# did_dedup False -> gate does NOT fire, even with a decimate op present.
case("f2_instances_preserved_prose_is_not_dedup",
     {"operations": [
         {"order": 3, "name": "removeUnusedData", "method": "point-instance aware",
          "result": "126 instances preserved"},
         {"order": 4, "name": "decimateMeshes", "method": "0.1mm", "result": "faces -55%"}],
      "target_coverage": {"entries": [{"path": "/m", "role": "monolith"}]}},
     [], expect_error=False)

# F3 (finding 3): a real dedup+merge run declaring descent_converged: true but with
# the rescan field OMITTED must fail closed — absent is NOT converged (schema:
# "0 == converged"). The dedup op-marker "deduplicate" + merge_applied make both
# did_dedup and did_phase4_geom True; the missing rescan means no state is
# converged-or-waived -> gate FIRES.
case("f3_absent_rescan_is_not_converged_fail_closed",
     {"structural_summary": {"merge_applied": True, "frontier_descent_converged": True},
      "operations": [{"order": 2, "name": "deduplicateHierarchies", "method": "exact"}]},
     [], expect_error=True, marker="descent-convergence gate")

# F4a (finding 4): dedup+merge with convergence_waived_by_user true but NO reason ->
# not a valid waiver (schema: "Silence is NOT a waiver") -> gate FIRES.
case("f4a_waiver_without_reason_does_not_count",
     {"structural_summary": {"merge_applied": True, "frontier_convergence_waived": True},
      "operations": [{"order": 2, "name": "deduplicateHierarchies", "method": "exact"}]},
     [], expect_error=True, marker="descent-convergence gate")

# F4b (finding 4): the same run WITH a non-empty reason -> valid waiver -> PASS.
case("f4b_waiver_with_reason_passes",
     {"structural_summary": {"merge_applied": True, "frontier_convergence_waived": True,
                             "frontier_convergence_waiver_reason": "user time budget"},
      "operations": [{"order": 2, "name": "deduplicateHierarchies", "method": "exact"}]},
     [], expect_error=False)


def _run_pure_function_checks():
    """Direct assertions on the shared dcc pure functions (findings 1-4)."""
    failures = 0

    def check(label, cond):
        nonlocal failures
        status = "PASS" if cond else "FAIL"
        if not cond:
            failures += 1
        print(f"[{status}] dcc.{label}")

    # F1: bare authored_prim_delta drop is not dedup.
    check("did_dedup F1 merge-only drop is False",
          dcc.did_dedup({"structural_summary": {"authored_prim_delta_pct": -18.0}}) is False)
    # F2: prose in result / method is not a dedup marker; name+method only.
    check("did_dedup F2 'instances preserved' prose is False",
          dcc.did_dedup({"operations": [
              {"name": "removeUnusedData", "method": "point-instance aware",
               "result": "126 instances preserved"}]}) is False)
    # Positive control: an unambiguous dedup op-marker IS dedup.
    check("did_dedup deduplicate op-marker is True",
          dcc.did_dedup({"operations": [{"name": "deduplicateHierarchies", "method": "exact"}]}) is True)
    check("did_dedup reuse_measured is True",
          dcc.did_dedup({"structural_summary": {"reuse_measured": True}}) is True)
    # F3: absent rescan fails closed; explicit 0 converges.
    check("is_converged_or_waived F3 absent rescan is False",
          dcc.is_converged_or_waived({"descent_converged": True}) is False)
    check("is_converged_or_waived converged with rescan 0 is True",
          dcc.is_converged_or_waived(
              {"descent_converged": True, "final_rescan_new_groups_above_floor": 0}) is True)
    # F4: waiver requires a non-empty reason.
    check("is_converged_or_waived F4 reasonless waiver is False",
          dcc.is_converged_or_waived({"convergence_waived_by_user": True}) is False)
    check("is_converged_or_waived F4 empty-string reason is False",
          dcc.is_converged_or_waived(
              {"convergence_waived_by_user": True, "convergence_waiver_reason": "  "}) is False)
    check("is_converged_or_waived waiver with reason is True",
          dcc.is_converged_or_waived(
              {"convergence_waived_by_user": True, "convergence_waiver_reason": "budget"}) is True)
    return failures


def run():
    failures = 0
    for name, report, manifests, expect_error, marker in CASES:
        errs = V.validate_descent_convergence(report, manifests)
        got_error = bool(errs)
        ok = got_error == expect_error
        if ok and expect_error and marker:
            ok = any(marker in e for e in errs)
        status = "PASS" if ok else "FAIL"
        if not ok:
            failures += 1
        print(f"[{status}] {name}: expect_error={expect_error} got={got_error} "
              + (f"errs={errs}" if not ok else ""))
    pure_failures = _run_pure_function_checks()
    total = len(CASES) + 9  # 9 direct dcc checks
    passed = total - failures - pure_failures
    print(f"\n{passed}/{total} passed")
    return 1 if (failures or pure_failures) else 0


if __name__ == "__main__":
    raise SystemExit(run())
