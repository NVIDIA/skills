#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate a USD Performance Tuning report against optimization-report.schema.json.

Deterministic local validation with no third-party runtime dependencies. The
agent (or CI) should run this before treating a report as final, so an
out-of-enum verdict, a missing required field, or an unexpected array-item key
is caught instead of shipping a schema-invalid report.

Implements the JSON Schema draft-07 subset this schema uses: type (including
type unions like ["string", "null"]), enum, required, properties,
additionalProperties=false, items, minimum, and maximum.

Phase-4 target coverage gate
----------------------------
Schema validation alone cannot catch a Phase-4 target that was never enumerated
in the report (the failure mode where an assembly_root remainder is silently
left un-optimized). A report's ``target_coverage.complete`` flag is self-attested
by the report author, so the gate reconciles ``target_coverage`` against the
upstream apply-restructure manifest(s): the report must cover the UNION of every
iteration's ``phase4_targets[]``, every disposition must be resolved, and
``skipped_zero_meshes`` is accepted only when the manifest's authoritative
``mesh_count`` for that target is 0.

Reconciliation is fail-closed, not opt-in: when any coverage entry has a
restructure role (assembly_root | prototype | shared_layer | loadable_subasset)
a manifest is REQUIRED. Manifests are taken from ``--manifest`` and/or the
report's own ``target_coverage.source_manifests[]`` (auto-loaded relative to the
report), so a restructure report cannot pass merely because the operator forgot
the flag. Monolith/diagnosis runs (no restructure roles) stay manifest-free.

Footprint gate (repack-normalization)
-------------------------------------
When the optional ``footprint`` block is present, the gate checks the
raw -> repack_normalized -> optimized arithmetic is consistent and fails closed
on a repack-as-optimization claim: a raw->optimized saving that is almost
entirely the free crate re-encode (structural shrink ~0 off the
repack-normalized baseline) cannot be presented as the optimization win.

Preservation gate (axis-A silent-loss)
--------------------------------------
When the optional ``preservation`` block is present, the gate checks its shape
matches what the run-scoring oracle consumes:
integer ``rendered_mesh_count`` / ``dangling`` and boolean
``distinct_geometry_bytes_preserved`` / ``bounds_preserved``. Whether those
values actually pass (count unchanged, bounds/bytes preserved, dangling 0) is
scored against the asset's oracle, not here.

Usage:
    python3 validate_report.py <report.json> [--schema <schema.json>] \\
        [--manifest <apply-restructure-manifest.json> ...]
Exit code 0 when the report conforms and the coverage gate passes, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_SCHEMA = Path(__file__).resolve().parent / "optimization-report.schema.json"

#: A Phase-4 target is "resolved" only with one of these dispositions. ``blocked``
#: (or a target with no entry at all) keeps ``target_coverage.complete`` false and
#: the report non-final — mirroring the validation report's RESOLVED_STATUSES.
PHASE4_RESOLVED_DISPOSITIONS = frozenset(
    {"optimized", "skipped_zero_meshes", "skipped_user_declined"}
)
RESTRUCTURE_TARGET_CLASSES = frozenset(
    {"prototype", "shared_layer", "loadable_subasset", "assembly_root"}
)
#: Reuse/descent frontier dispositions that SHARE a unit (and therefore make a
#: scene-graph / consolidation claim that must be backed by MEASURED reuse).
SHARED_IDENTITY_DISPOSITIONS = frozenset({"externalize_shared", "internal_share"})
#: Reuse/descent frontier reduction routes that DESTROY identity — allowed only on weak-identity
#: (anonymous / structural-fallback) units, never on a strong-identity part.
IDENTITY_DESTROYING_ROUTES = frozenset({"point_instance", "merge"})
#: Identity signals that mark a unit as a real, addressable part (strong identity).
STRONG_IDENTITY_SIGNALS = frozenset({"kind", "naming", "semantic"})
#: Coverage-entry roles that mean "a restructure happened", so a manifest is
#: mandatory and reconciliation is not optional. The ``monolith`` role (an
#: optimize-as-is N=1 target) and an empty ledger stay manifest-free.
RESTRUCTURE_ROLES = frozenset(
    {"assembly_root", "prototype", "shared_layer", "loadable_subasset"}
)


def _type_ok(instance: Any, type_name: str) -> bool:
    if type_name == "object":
        return isinstance(instance, dict)
    if type_name == "array":
        return isinstance(instance, list)
    if type_name == "string":
        return isinstance(instance, str)
    if type_name == "number":
        return isinstance(instance, (int, float)) and not isinstance(instance, bool)
    if type_name == "integer":
        return isinstance(instance, int) and not isinstance(instance, bool)
    if type_name == "boolean":
        return isinstance(instance, bool)
    if type_name == "null":
        return instance is None
    return True


def _validate(instance: Any, schema: dict, path: str, errors: list[str]) -> None:
    declared_type = schema.get("type")
    if declared_type is not None:
        candidates = declared_type if isinstance(declared_type, list) else [declared_type]
        if not any(_type_ok(instance, name) for name in candidates):
            got = "null" if instance is None else type(instance).__name__
            errors.append(f"{path}: expected type {candidates}, got {got}")
            return

    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: {instance!r} is not one of {schema['enum']}")

    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append(f"{path}: {instance} is below minimum {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            errors.append(f"{path}: {instance} is above maximum {schema['maximum']}")

    if isinstance(instance, dict):
        properties = schema.get("properties", {})
        for required_key in schema.get("required", []):
            if required_key not in instance:
                errors.append(f"{path}: missing required property '{required_key}'")
        allow_additional = schema.get("additionalProperties", True)
        for key, value in instance.items():
            if key in properties:
                _validate(value, properties[key], f"{path}.{key}", errors)
            elif allow_additional is False:
                errors.append(f"{path}: unexpected property '{key}'")

    if isinstance(instance, list) and "items" in schema:
        for index, item in enumerate(instance):
            _validate(item, schema["items"], f"{path}[{index}]", errors)


def validate_report(report: Any, schema: dict | None = None) -> list[str]:
    """Return a list of schema-violation messages; empty list means the report conforms."""
    if schema is None:
        schema = json.loads(DEFAULT_SCHEMA.read_text(encoding="utf-8"))
    errors: list[str] = []
    _validate(report, schema, "$", errors)
    return errors


def _validate_frontier_entry(target: dict, label: str) -> list[str]:
    """Enforce the reuse/descent frontier contract on a single phase4_targets[] entry (C3).

    These checks are conditional on the frontier fields being present, so a
    minimal (path / target_class / mesh_count) manifest is unaffected. When the
    frontier metadata IS authored the contract makes a bad descent FAIL, not
    merely be discouraged in prose:

    * a shared (externalize_shared / internal_share) or identity-destroying
      (point_instance / merge) entry must carry an ``identity_signal`` (the
      queryable boundary basis);
    * a shared entry whose frontier landed on anonymous meshes
      (``identity_signal == "none"``) fails — the descent crossed from parts into
      triangles;
    * an identity-destroying ``reduction_route`` on a STRONG-identity unit
      (kind / naming / semantic) fails — point_instance / merge are the matrix's
      two weak-identity-only rows.
    """
    errors: list[str] = []
    disposition = target.get("identity_disposition")
    route = target.get("reduction_route")
    signal = target.get("identity_signal")

    is_shared = disposition in SHARED_IDENTITY_DISPOSITIONS
    is_destroying = route in IDENTITY_DESTROYING_ROUTES

    if (is_shared or is_destroying) and not signal:
        errors.append(
            f"{label}: identity_signal is required on a shared "
            f"({disposition}) or identity-destroying ({route}) entry "
            "(the boundary basis must be queryable)"
        )

    if is_shared and signal == "none":
        errors.append(
            f"{label}: identity_disposition {disposition!r} landed on anonymous "
            "meshes (identity_signal 'none') — a shared frontier must land on a "
            "named/kind/semantic unit or, at worst, the coarsest repeating subtree "
            "(structural_fallback), never anonymous geometry"
        )

    if is_destroying and signal in STRONG_IDENTITY_SIGNALS:
        errors.append(
            f"{label}: identity-destroying reduction_route {route!r} on a "
            f"strong-identity unit (identity_signal {signal!r}) — point_instance / "
            "merge are reserved for weak-identity (anonymous / structural_fallback) "
            "units; a named/kind/semantic part must stay addressable"
        )

    return errors


def validate_manifest_structure(manifest: Any) -> list[str]:
    """Enforce the load-bearing apply-restructure manifest invariants.

    Independent of the JSON-Schema walker so the rules hold without ``jsonschema``:
    a ``mode=restructure`` manifest must carry a non-empty ``phase4_targets[]``,
    and every target must declare an integer ``mesh_count >= 0`` (the authoritative
    default-predicate count the coverage gate keys on).
    """
    errors: list[str] = []
    if not isinstance(manifest, dict):
        return [f"manifest must be an object, got {type(manifest).__name__}"]
    mode = manifest.get("mode")
    targets = manifest.get("phase4_targets")
    if mode == "restructure" and not targets:
        errors.append(
            "mode=restructure manifest must list a non-empty phase4_targets[] "
            "(do not drop the key; an assembly_root with retained meshes must appear)"
        )
    for index, target in enumerate(targets or []):
        where = f"phase4_targets[{index}]"
        path = target.get("path") if isinstance(target, dict) else None
        label = f"{where} ({path})" if path else where
        if not isinstance(target, dict):
            errors.append(f"{where}: must be an object")
            continue
        if not isinstance(path, str) or not path:
            errors.append(f"{where}: missing required 'path'")
        target_class = target.get("target_class")
        if target_class not in RESTRUCTURE_TARGET_CLASSES:
            errors.append(
                f"{label}: target_class {target_class!r} not in {sorted(RESTRUCTURE_TARGET_CLASSES)}"
            )
        mesh_count = target.get("mesh_count")
        if isinstance(mesh_count, bool) or not isinstance(mesh_count, int) or mesh_count < 0:
            errors.append(
                f"{label}: mesh_count must be an integer >= 0 (authoritative "
                f"default-predicate count), got {mesh_count!r}"
            )
        errors.extend(_validate_frontier_entry(target, label))

    # Measured-reuse-before-consolidation (C3): a scene-graph / consolidation win
    # may only be reported from MEASURED reuse, never estimated. When any target
    # claims a shared disposition (externalize_shared / internal_share), the
    # manifest's top-level ``frontier`` block must record reuse_measured: true.
    shared_entries = [
        t for t in (targets or [])
        if isinstance(t, dict) and t.get("identity_disposition") in SHARED_IDENTITY_DISPOSITIONS
    ]
    if shared_entries:
        frontier = manifest.get("frontier")
        if not isinstance(frontier, dict) or frontier.get("reuse_measured") is not True:
            errors.append(
                "frontier: a shared disposition (externalize_shared/internal_share) is claimed "
                "but frontier.reuse_measured is not true — a scene-graph / consolidation win "
                "must be backed by MEASURED reuse (distinct vs total units), never estimated. "
                "When measured reuse is low, pivot to the disk tier instead of forcing sharing."
            )

    # apply-restructure residual-mesh postcondition (mirrored here for the
    # manifest, enforced live in apply-restructure where the USD stage is open).
    # When extraction leaves > 0 mesh prims on the assembly root, apply-restructure
    # records the authoritative residual count under ``assembly_root`` AND lists the
    # same root in ``phase4_targets[]`` as ``target_class: assembly_root``. Fail loud
    # if a restructure manifest is written with residual meshes but no such target
    # entry: a retained-mesh assembly root must never be silently dropped from Phase 4.
    residual = manifest.get("assembly_root")
    if isinstance(residual, dict):
        residual_count = residual.get("mesh_count")
        residual_path = residual.get("path")
        if (
            isinstance(residual_count, int)
            and not isinstance(residual_count, bool)
            and residual_count > 0
        ):
            root_entry = next(
                (
                    t
                    for t in (targets or [])
                    if isinstance(t, dict)
                    and t.get("target_class") == "assembly_root"
                    and (residual_path is None or t.get("path") == residual_path)
                ),
                None,
            )
            if root_entry is None:
                errors.append(
                    f"assembly_root records residual mesh_count {residual_count} > 0 but no "
                    "phase4_targets[] entry with target_class 'assembly_root'"
                    + (f" and path {residual_path}" if residual_path else "")
                    + " is present (apply-restructure postcondition: a retained-mesh assembly "
                    "root must be a Phase-4 target, never silently dropped)"
                )
            elif root_entry.get("mesh_count") != residual_count:
                errors.append(
                    f"assembly_root residual mesh_count {residual_count} does not match its "
                    f"phase4_targets[] entry mesh_count {root_entry.get('mesh_count')!r} "
                    "(both must echo the same authoritative default-predicate count)"
                )
    return errors


def load_recorded_manifests(
    report: Any, base_dir: Path
) -> tuple[list[tuple[str, Any]], list[str]]:
    """Load the manifests recorded in ``target_coverage.source_manifests[]``.

    Relative paths resolve against ``base_dir`` (the report's directory) so a
    report can carry its own provenance and the gate fails closed without the
    operator having to remember ``--manifest``. Returns ``(labeled_manifests,
    errors)`` where each labeled manifest is ``(source_path, manifest_dict)``.
    """
    labeled: list[tuple[str, Any]] = []
    errors: list[str] = []
    coverage = report.get("target_coverage") if isinstance(report, dict) else None
    if not isinstance(coverage, dict):
        return labeled, errors
    for rel in coverage.get("source_manifests", []) or []:
        path = Path(rel)
        if not path.is_absolute():
            path = base_dir / path
        try:
            labeled.append((rel, json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(
                f"target_coverage.source_manifests entry {rel!r} could not be loaded: {exc}"
            )
    return labeled, errors


def _manifest_targets(manifests: list[Any]) -> dict[str, int | None]:
    """Union of every manifest's phase4_targets path -> authoritative mesh_count.

    Multi-iteration runs must reconcile against the UNION: the exact regression
    that prompted this gate was iteration 1 listing an assembly_root that
    iteration 2's manifest dropped, leaving it uncovered by the final report.
    """
    planned: dict[str, int | None] = {}
    for manifest in manifests:
        for target in manifest.get("phase4_targets", []) or []:
            if isinstance(target, dict) and isinstance(target.get("path"), str):
                planned[target["path"]] = target.get("mesh_count")
    return planned


#: Arithmetic-consistency tolerance (percentage points) for the footprint split.
_FOOTPRINT_TOLERANCE_PP = 0.6
#: A structural change smaller than this magnitude (percent) is treated as "no
#: real structural shrink off the repack-normalized baseline" for the
#: repack-as-optimization fail-closed check.
_FOOTPRINT_STRUCTURAL_EPSILON_PCT = 1.0


def _pct_change(before: float, after: float) -> float | None:
    if before == 0:
        return None
    return (after - before) / before * 100.0


def validate_footprint(report: Any) -> list[str]:
    """Gate the optional repack-normalized footprint block.

    Returns violation messages (empty == passes or no footprint block). When a
    ``footprint`` block is present it must (1) be arithmetically consistent
    across raw -> repack_normalized -> optimized, and (2) fail closed if the
    only reduction is the free crate re-encode while the structural win off the
    repack-normalized baseline is ~0 (repack-as-optimization). A repack — or an
    unshared disaggregation — presented as the optimization win is a fail-closed
    reporting error per the plan.
    """
    errors: list[str] = []
    footprint = report.get("footprint") if isinstance(report, dict) else None
    if footprint is None:
        return errors
    if not isinstance(footprint, dict):
        return ["footprint must be an object"]

    raw = footprint.get("raw_input_bytes")
    normalized = footprint.get("repack_normalized_baseline_bytes")
    optimized = footprint.get("optimized_bytes")
    nums = {
        "raw_input_bytes": raw,
        "repack_normalized_baseline_bytes": normalized,
        "optimized_bytes": optimized,
    }
    for name, value in nums.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            errors.append(f"footprint.{name} must be a number >= 0, got {value!r}")
    if errors:
        return errors

    if footprint.get("scored_against") != "repack_normalized":
        errors.append(
            "footprint.scored_against must be 'repack_normalized' — the storage "
            "dimension and any footprint claim score against the repack-normalized "
            "baseline, not the raw input"
        )

    structural_delta = footprint.get("structural_delta_pct")
    expected_structural = _pct_change(normalized, optimized)
    if isinstance(structural_delta, (int, float)) and not isinstance(structural_delta, bool):
        if expected_structural is not None and abs(structural_delta - expected_structural) > _FOOTPRINT_TOLERANCE_PP:
            errors.append(
                f"footprint.structural_delta_pct ({structural_delta}) does not match "
                f"repack_normalized->optimized ({expected_structural:.2f}%) within "
                f"{_FOOTPRINT_TOLERANCE_PP}pp"
            )
    else:
        errors.append("footprint.structural_delta_pct must be a number")

    repack_delta = footprint.get("repack_delta_pct")
    if isinstance(repack_delta, (int, float)) and not isinstance(repack_delta, bool):
        expected_repack = _pct_change(raw, normalized)
        if expected_repack is not None and abs(repack_delta - expected_repack) > _FOOTPRINT_TOLERANCE_PP:
            errors.append(
                f"footprint.repack_delta_pct ({repack_delta}) does not match "
                f"raw->repack_normalized ({expected_repack:.2f}%) within "
                f"{_FOOTPRINT_TOLERANCE_PP}pp"
            )

    raw_delta = footprint.get("raw_delta_pct")
    if isinstance(raw_delta, (int, float)) and not isinstance(raw_delta, bool):
        expected_raw = _pct_change(raw, optimized)
        if expected_raw is not None and abs(raw_delta - expected_raw) > _FOOTPRINT_TOLERANCE_PP:
            errors.append(
                f"footprint.raw_delta_pct ({raw_delta}) does not match "
                f"raw->optimized ({expected_raw:.2f}%) within {_FOOTPRINT_TOLERANCE_PP}pp"
            )

    # Repack-as-optimization fail-closed: a raw->optimized saving that is almost
    # entirely the free crate re-encode (structural shrink ~0 off the normalized
    # baseline) must not be presented as the optimization win.
    if (
        isinstance(structural_delta, (int, float))
        and not isinstance(structural_delta, bool)
        and structural_delta > -_FOOTPRINT_STRUCTURAL_EPSILON_PCT
    ):
        observed_raw = raw_delta if isinstance(raw_delta, (int, float)) and not isinstance(raw_delta, bool) else _pct_change(raw, optimized)
        if observed_raw is not None and observed_raw < -_FOOTPRINT_STRUCTURAL_EPSILON_PCT:
            errors.append(
                f"footprint: raw->optimized shows {observed_raw:.2f}% but the structural "
                f"reduction off the repack-normalized baseline is only {structural_delta:.2f}% "
                "— the saving is the free crate re-encode, not the optimization. Presenting a "
                "repack (or an unshared disaggregation) as the optimization win is a "
                "fail-closed reporting error."
            )
    return errors


def validate_preservation(report: Any) -> list[str]:
    """Gate the optional axis-A preservation (silent-loss) block.

    Returns violation messages (empty == passes or no preservation block). The
    block is optional at the report level, but when present its shape must match
    exactly what the run-scoring oracle consumes so the
    schema, scorer, and validator agree: an integer ``rendered_mesh_count`` and
    ``dangling`` (both >= 0), and boolean ``distinct_geometry_bytes_preserved``
    and ``bounds_preserved``. This is shape-only — whether the values pass the
    gate (count unchanged, both bools true, dangling 0) is the scorer's job
    against the asset's oracle, not a static check here.
    """
    errors: list[str] = []
    pres = report.get("preservation") if isinstance(report, dict) else None
    if pres is None:
        return errors
    if not isinstance(pres, dict):
        return ["preservation must be an object"]
    for name in ("rendered_mesh_count", "dangling"):
        value = pres.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            errors.append(
                f"preservation.{name} must be an integer >= 0, got {value!r}"
            )
    for name in ("distinct_geometry_bytes_preserved", "bounds_preserved"):
        value = pres.get(name)
        if not isinstance(value, bool):
            errors.append(f"preservation.{name} must be a boolean, got {value!r}")
    return errors


def reconcile_target_coverage(report: Any, manifests: list[Any] | None = None) -> list[str]:
    """Gate the report's Phase-4 target_coverage; reconcile against manifest(s).

    Returns violation messages (empty == the gate passes). Always checks the
    report's internal consistency (resolved dispositions, the
    ``skipped_zero_meshes => mesh_count == 0`` rule, and the ``complete`` flag).
    When ``manifests`` are supplied it also asserts the covered set equals the
    union of every manifest's ``phase4_targets[]`` and cross-checks each
    disposition against the manifest's authoritative ``mesh_count``.
    """
    errors: list[str] = []
    coverage = report.get("target_coverage") if isinstance(report, dict) else None
    if not isinstance(coverage, dict):
        return ["target_coverage missing or not an object"]
    entries = coverage.get("entries", [])
    by_path: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if isinstance(entry, dict) and isinstance(entry.get("path"), str):
            by_path[entry["path"]] = entry

    for entry in entries:
        path = entry.get("path", "<unknown>")
        disposition = entry.get("disposition")
        mesh_count = entry.get("mesh_count")
        if disposition == "skipped_zero_meshes" and mesh_count != 0:
            errors.append(
                f"target_coverage entry {path}: skipped_zero_meshes requires "
                f"mesh_count == 0, got {mesh_count!r} (a non-zero target cannot be skipped)"
            )
        # No-op-masquerading-as-optimized: an entry that claims the mesh op chain
        # ran must have touched at least one mesh when the target has meshes.
        optimized_mesh_count = entry.get("optimized_mesh_count")
        if (
            disposition == "optimized"
            and isinstance(optimized_mesh_count, int)
            and not isinstance(optimized_mesh_count, bool)
            and optimized_mesh_count == 0
            and isinstance(mesh_count, int)
            and not isinstance(mesh_count, bool)
            and mesh_count > 0
        ):
            errors.append(
                f"target_coverage entry {path}: disposition 'optimized' but "
                f"optimized_mesh_count is 0 while mesh_count is {mesh_count} > 0 "
                "(no-op masquerading as optimized — record the meshes actually touched, "
                "or use skipped_user_declined / skipped_zero_meshes)"
            )

    present_restructure_roles = sorted(
        {e.get("role") for e in entries} & RESTRUCTURE_ROLES
    )

    # Once a restructure exists (a manifest is supplied/recorded OR a restructure
    # role appears), 'monolith' is illegal: it is by definition the N=1, no-manifest,
    # non-restructured optimize-as-is path. A monolith entry alongside manifest
    # provenance or restructure roles is a contradiction, not a valid ledger row.
    recorded_manifests = coverage.get("source_manifests") or []
    manifest_context = bool(manifests) or bool(recorded_manifests)
    if manifest_context or present_restructure_roles:
        for entry in entries:
            if entry.get("role") == "monolith":
                errors.append(
                    f"target_coverage entry {entry.get('path', '<unknown>')}: role 'monolith' "
                    "is illegal once a restructure exists (a source manifest is present or a "
                    "restructure role appears in target_coverage). 'monolith' is only the N=1, "
                    "no-manifest, non-restructured optimize-as-is target."
                )

    if present_restructure_roles and not manifests:
        errors.append(
            "target_coverage has restructure role(s) "
            f"{present_restructure_roles} but no source manifest was supplied or recorded; "
            "reconciliation is mandatory once a restructure happened. Record "
            "target_coverage.source_manifests[] (or pass --manifest) so the covered set is "
            "reconciled against the planned phase4_targets[] instead of self-attested."
        )

    all_resolved = all(
        e.get("disposition") in PHASE4_RESOLVED_DISPOSITIONS for e in entries
    )
    if coverage.get("complete") is not True:
        errors.append(
            "target_coverage.complete must be true for a final report "
            "(false => a Phase-4 target is unresolved/blocked)"
        )
    elif not all_resolved:
        errors.append(
            "target_coverage.complete is true but some entries are unresolved "
            "(only optimized | skipped_zero_meshes | skipped_user_declined count as resolved)"
        )

    if manifests:
        planned = _manifest_targets(manifests)
        planned_paths = set(planned)
        covered_paths = set(by_path)
        for path in sorted(planned_paths - covered_paths):
            errors.append(
                f"target_coverage is missing an entry for manifest phase4_target: {path} "
                "(every planned Phase-4 target, across all iterations, must be covered)"
            )
        for path in sorted(covered_paths - planned_paths):
            errors.append(
                f"target_coverage entry {path} is not present in any supplied manifest "
                "phase4_targets[] (unexpected target or a missing manifest)"
            )
        for path in sorted(planned_paths & covered_paths):
            authoritative = planned[path]
            disposition = by_path[path].get("disposition")
            if (
                disposition == "skipped_zero_meshes"
                and isinstance(authoritative, int)
                and authoritative > 0
            ):
                errors.append(
                    f"target_coverage entry {path}: skipped_zero_meshes but the manifest's "
                    f"authoritative mesh_count is {authoritative} > 0 (lying skip)"
                )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Path to the report JSON to validate.")
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument(
        "--manifest",
        type=Path,
        action="append",
        default=[],
        help="apply-restructure manifest(s) to reconcile Phase-4 coverage against; "
        "repeat once per iteration so the union is checked. Manifests recorded in "
        "the report's target_coverage.source_manifests[] are loaded automatically "
        "and merged with these.",
    )
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    errors = validate_report(report, schema)

    labeled: list[tuple[str, Any]] = []
    for manifest_path in args.manifest:
        labeled.append((manifest_path.name, json.loads(manifest_path.read_text(encoding="utf-8"))))
    recorded, load_errors = load_recorded_manifests(report, args.report.resolve().parent)
    errors.extend(load_errors)
    labeled.extend(recorded)

    for label, manifest in labeled:
        errors.extend(f"{label}: {msg}" for msg in validate_manifest_structure(manifest))

    manifests = [manifest for _, manifest in labeled]
    errors.extend(reconcile_target_coverage(report, manifests))
    errors.extend(validate_footprint(report))
    errors.extend(validate_preservation(report))

    if errors:
        print(f"{args.report}: INVALID ({len(errors)} error(s))")
        for error in errors:
            print(f"  {error}")
        return 1
    print(f"{args.report}: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
