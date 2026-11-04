# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared asset structure, inventory and completion checks for Assess and Verify.

This module accepts observations; it does not choose domain boundaries or infer
optimization from changed counters. A final audit supplies fresh inventory and
the written composed stage rather than trusting assessment records alone.
"""
from collections import Counter
import importlib.util
from pathlib import Path

STRUCTURAL_TYPES = {"atomic", "aggregate", "package"}
RESOLVED_OUTCOMES = {"optimized", "verified_unchanged"}


def check_definition_result(result, require_categories=False):
    """Return errors for a native worker's per-definition result."""
    errors = []
    if result.get('disposition') in ('exact_cache_hit', 'normalized_cache_hit', 'product_identity_hit'):
        verification = result.get('verification', {})
        basis = verification.get('reuse_basis', {})
        identity_confirmed = (bool(basis.get('exact_key')) if result['disposition'] == 'exact_cache_hit' else
                              bool(basis.get('normalized_source_key')) and basis.get('authorization') == 'allow_constant_colour_rounding')
        if result['disposition']=='product_identity_hit':
            identity_confirmed=bool(basis.get('product_key') and basis.get('product_identity')) and basis.get('representation_substitution') is True
        if not (result.get('persistent_reuse') is True and verification.get('passed') is True
                and identity_confirmed
                and verification.get('evidence_fingerprint', {}).get('files')
                and isinstance(result.get('output_geometry_paths'), list)):
            errors.append('cache completion evidence is missing')
        return errors
    if result.get("disposition") not in RESOLVED_OUTCOMES:
        errors.append("definition outcome is unresolved")
    verification=result.get("verification")
    if not isinstance(verification,dict) or verification.get("passed") is not True:
        errors.append("written output verification did not pass")
    assessment = result.get("assessment", {})
    if not isinstance(assessment,dict):
        errors.append("operation assessment is not an object")
        return errors
    if assessment.get("complete") is not True:
        errors.append("operation assessment is incomplete")
    if require_categories:
        categories=assessment.get('categories',[])
        if not isinstance(categories,list):
            errors.append('operation categories are not a list')
            return errors
        expected={'cleanup','primitive_fit','reduction','merge','reuse'}
        names=[item.get('category') for item in categories if isinstance(item,dict)]
        if len(categories)!=5 or set(names)!=expected or len(set(names))!=len(names):
            errors.append('operation assessment requires all five unique categories')
        for item in categories:
            if not isinstance(item,dict) or item.get('status') not in ('executed','no_benefit','not_applicable','excluded') or not item.get('reason') or not item.get('evidence'):
                errors.append('operation category is unresolved or lacks evidence')
    if result.get("disposition") == "verified_unchanged":
        operations = assessment.get("operations", [])
        if not isinstance(operations,list) or not operations or any(not isinstance(op, dict) or
                op.get("status") not in ("not_applicable", "no_benefit", "verified_unchanged")
                or not op.get("reason") or not op.get("evidence") for op in operations):
            errors.append("verified_unchanged requires assessed operations with reasons and evidence")
    return errors


def check_manifest(manifest, inventory=None, stage=None):
    """Check one authoritative manifest against observations and composed USD.

    inventory is independently collected on final verification. Its geometry
    entries contain path, schema, and optional definition_id. The assessment
    inventory remains usable before outputs exist. USD imports stay optional.
    """
    errors = []
    def fail(code, path, message):
        errors.append({"code": code, "path": str(path), "message": message})
    definitions = manifest.get("definitions", [])
    ids = [d.get("id") for d in definitions]
    for key, count in Counter(ids).items():
        if not key or count != 1:
            fail("definition_identity", key, "definition IDs must be unique and nonempty")
    by_id = {d.get("id"): d for d in definitions}
    observed = inventory if inventory is not None else manifest.get("inventory", {})
    if observed.get("uninspected"):
        fail("uninspected", "", "required source contexts were not inspected")
    if manifest.get("decisions_required"):
        fail("unresolved_decisions", "", "boundary or operation decisions remain unresolved")
    if manifest.get('boundary_review_version') == 1:
        location = Path(__file__).resolve().parents[2] / 'usd-structure-assessment/scripts/boundary_reasoning.py'
        spec = importlib.util.spec_from_file_location('conformance_boundary_reasoning', location)
        reasoning = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reasoning)
        if not isinstance(manifest.get('boundary_reviews'), list):
            fail('boundary_review', '', 'boundary review records are missing')
        reviews = manifest.get('boundary_reviews') or []
        scan = manifest.get('discovery', {}).get('source_scan', {})
        if (scan.get('boundary_review_groups') != len(reviews) or
                scan.get('boundary_review_paths') != sum(len(row.get('paths', [])) for row in reviews)):
            fail('boundary_review', '', 'review records differ from recorded discovery coverage')
        for review in manifest.get('boundary_reviews', []):
            if review.get('status') != 'reviewed':
                fail('unresolved_decisions', review.get('owner_path', ''), 'equipment interior review remains unresolved')
            else:
                paths = set(review.get('paths', []))
                decided = {row.get('path') for row in review.get('dispositions', [])}
                if paths != decided:
                    fail('boundary_review', review.get('owner_path', ''), 'review dispositions do not cover nominated paths')
                for disposition in review.get('dispositions', []):
                    if disposition.get('disposition') not in ('boundary', 'internal') or reasoning.selection_errors(disposition.get('selection')):
                        fail('boundary_review', review.get('owner_path', ''), 'review disposition requires valid selection evidence')
        for definition in definitions:
            selection = definition.get('selection') or {}
            if reasoning.selection_errors(selection):
                fail('selection', definition.get('id', ''), 'asset selection requires evidence and intended independent use')
    ownership = {}
    for definition in definitions:
        key = definition.get("id")
        typ = definition.get("structural_type")
        rationale = definition.get("rationale") or {}
        if typ not in STRUCTURAL_TYPES:
            fail("structural_type", key, "asset must be atomic, aggregate or package")
        if not isinstance(rationale, dict) or not rationale.get("reason") or not rationale.get("evidence"):
            fail("rationale", key, "structural classification requires a reason and observed evidence")
        children = definition.get("children", [])
        owned = definition.get("owned_geometry", [])
        if typ == "atomic" and children:
            fail("atomic_children", key, "atomic asset contains child geometry definitions")
        if typ == "aggregate" and owned:
            fail("aggregate_geometry", key, "aggregate asset owns geometry")
        if typ in ("aggregate", "package") and not children:
            fail("missing_children", key, "aggregate or package asset has no child definitions")
        for child in children:
            if child not in by_id:
                fail("missing_child", key, "child definition is absent: " + str(child))
        for path in owned:
            if path in ownership:
                fail("overlapping_geometry", path, "geometry belongs to multiple definitions")
            ownership[path] = key
        for dependency in definition.get("material_dependencies", []):
            if not isinstance(dependency, dict):
                fail("dependency_record", key, "material dependency requires typed resolution evidence")
                continue
            if dependency.get("kind", "material") not in ("material", "shader", "texture", "resource"):
                fail("material_exception", key, "geometry dependencies cannot use the material exception")
            if dependency.get("exists") is False or dependency.get("resolved") is False:
                fail("missing_material_dependency", key, "shared material resource does not resolve")
        assessment = definition.get("assessment", {})
        if assessment.get("unassessed_nested"):
            fail("nested_coverage", key, "meaningful nested regions remain unassessed")
    visiting, visited = set(), set()
    def visit(key):
        if key in visiting:
            fail("dependency_cycle", key, "definition graph contains a cycle")
            return
        if key in visited:
            return
        visiting.add(key)
        for child in by_id.get(key, {}).get("children", []):
            visit(child)
        visiting.remove(key)
        visited.add(key)
    for key in by_id:
        visit(key)
    rows = observed.get("geometry", [])
    observed_paths = {row["path"] for row in rows}
    occurrences = manifest.get("occurrences", [])
    # A shared definition owns representative paths; expand each occurrence to
    # reconcile the complete independently collected composed inventory.
    expanded = dict(ownership)
    for occurrence in occurrences:
        definition = by_id.get(occurrence.get("definition_id"))
        if not definition:
            fail("occurrence_definition", occurrence.get("path"), "occurrence definition is absent")
            continue
        base = definition.get("boundary_path", definition.get("source_prim_path", ""))
        path = occurrence.get("path", "")
        inverse={destination:source for source,destination in occurrence.get('source_to_definition',{}).items()}
        for owned in definition.get("owned_geometry", []):
            if owned == base or owned.startswith(base + "/"):
                mapped = inverse.get(owned,path + owned[len(base):])
                if mapped in expanded and expanded[mapped] != definition["id"]:
                    fail("overlapping_geometry", mapped, "occurrence ownership overlaps")
                expanded[mapped] = definition["id"]
    for path in sorted(observed_paths - set(expanded)):
        fail("unowned_geometry", path, "observed geometry has no definition owner")
    for path in sorted(set(expanded) - observed_paths):
        fail("missing_geometry", path, "declared owned geometry is absent from observed inventory")
    for row in rows:
        if row.get("definition_id") and row["definition_id"] != expanded.get(row["path"]):
            fail("inventory_owner", row["path"], "independent inventory owner differs from manifest")
    if stage is not None:
        try:
            from pxr import Kind, Sdf, Usd, UsdGeom, UsdShade
            for error in stage.GetCompositionErrors():
                fail("composition", "", str(error))
            for occurrence in occurrences:
                path = occurrence.get("path", "")
                prim = stage.GetPrimAtPath(path)
                if not prim:
                    fail("missing_occurrence", path, "declared occurrence is absent")
                    continue
                kind = str(Usd.ModelAPI(prim).GetKind())
                expected = occurrence.get("effective_kind")
                if expected is not None and kind != expected:
                    fail("model_kind", path, "composed kind differs from recorded occurrence role")
                ancestor = prim.GetParent()
                while ancestor and not ancestor.IsPseudoRoot():
                    parent_kind = str(Usd.ModelAPI(ancestor).GetKind())
                    if kind and Kind.Registry.IsA(kind, Kind.Tokens.model):
                        if not parent_kind or not Kind.Registry.IsA(parent_kind, Kind.Tokens.group):
                            fail("model_ancestry", path, "model ancestors must be groups or assemblies")
                            break
                    ancestor = ancestor.GetParent()
                if kind == "subcomponent":
                    ancestor = prim.GetParent()
                    while ancestor and not ancestor.IsPseudoRoot() and str(Usd.ModelAPI(ancestor).GetKind()) != "component":
                        ancestor = ancestor.GetParent()
                    if not ancestor or ancestor.IsPseudoRoot():
                        fail("subcomponent_context", path, "subcomponent must be inside a component")
            atomic_roots = []
            for occurrence in occurrences:
                definition = by_id.get(occurrence.get("definition_id"), {})
                if definition.get("structural_type") == "atomic":
                    root = stage.GetPrimAtPath(occurrence.get("path", ""))
                    if root:
                        atomic_roots.append(root.GetPath())
            retained_descendants = {path: [] for path in atomic_roots}
            if retained_descendants:
                for occurrence in occurrences:
                    child_path = Sdf.Path(occurrence.get("path", ""))
                    ancestor = child_path.GetParentPath()
                    while ancestor:
                        if ancestor in retained_descendants:
                            retained_descendants[ancestor].append(child_path)
                        ancestor = ancestor.GetParentPath()
            # References are composition mechanisms, not asset boundaries.
            # Preserve root and child occurrence order, including duplicates.
            for root_path in atomic_roots:
                for child_path in retained_descendants[root_path]:
                    fail("atomic_children", child_path,
                         "atomic asset contains a retained child asset occurrence")
            for row in rows:
                prim = stage.GetPrimAtPath(row["path"])
                if not prim:
                    fail("missing_geometry", row["path"], "inventory geometry does not exist in composed stage")
                    continue
                material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
                for rel in prim.GetRelationships():
                    if rel.GetName().startswith("material:binding"):
                        for target in rel.GetForwardedTargets():
                            if not stage.GetObjectAtPath(target):
                                fail("material_binding", row["path"], "material binding target is missing")
        except Exception as exc:
            fail("checker_error", "", type(exc).__name__ + ": " + str(exc))
    return {"passed": not errors, "errors": errors,
            "checks": {"definitions": len(definitions), "occurrences": len(occurrences),
                       "geometry": len(observed_paths), "independent_inventory": inventory is not None}}
