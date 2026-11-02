#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Share remaining meshes in unpublished category layers.

Run after mesh optimization and before immutable publication. The caller
supplies owned mesh paths and excludes inherited entries. Native references are
internal geometry reuse; they do not add asset boundaries. Failed preservation
checks restore the staging layer without saving the native changes. Native USD Optimize
owns geometric correctness; runtime checks cover scope, outputs and composition.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import time


def _support():
    spec = importlib.util.spec_from_file_location('_sharing_preservation', Path(__file__).with_name('native_reconstruction.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _mesh_paths(stage, root):
    from pxr import Usd, UsdGeom
    prim = stage.GetPrimAtPath(root)
    predicate = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate) if prim and prim.IsAbstract() else Usd.TraverseInstanceProxies()
    return [str(p.GetPath()) for p in Usd.PrimRange(prim, predicate) if p.IsA(UsdGeom.Mesh)] if prim else []


def _has_entry_ancestor(path, entries):
    """Entry ownership is strict ancestry; an entry cannot own itself."""
    parent = path.GetParentPath()
    while parent:
        if parent in entries:
            return True
        parent = parent.GetParentPath()
    return False


def _incoming_mesh_interfaces(targets):
    """Index mesh paths targeted directly or through a descendant prim."""
    from pxr import Sdf
    protected = set(targets)
    for target in targets:
        parent = Sdf.Path(target).GetParentPath()
        # Preserve the original '/' + '/' string-prefix behavior at the root.
        while parent and parent != Sdf.Path.absoluteRootPath:
            value = str(parent)
            if value in protected:
                break
            protected.add(value)
            parent = parent.GetParentPath()
    return protected


def _storage(layer):
    from pxr import Sdf
    arrays = points = 0
    prims = list(layer.rootPrims)
    while prims:
        prim = prims.pop()
        prims.extend(prim.nameChildren.values())
        for variant_set in prim.variantSets.values():
            prims.extend(variant.primSpec for variant in variant_set.variants.values())
        attr = prim.attributes.get('points')
        if attr:
            path = attr.path
            values = ([attr.default] if attr.HasDefaultValue() else []) + [layer.QueryTimeSample(path, t) for t in layer.ListTimeSamplesForPath(path)]
            values = [value for value in values if value is not None and not isinstance(value, Sdf.ValueBlock)]
            arrays += len(values)
            points += sum(len(v) for v in values)
    return {'authored_point_arrays': arrays, 'authored_points': points}


def eligible_meshes(stage, paths, protected_interfaces=None):
    """Select static authored meshes without externally addressed interfaces."""
    from pxr import Usd, UsdGeom
    root = stage.GetRootLayer()
    skipped = []
    eligible = []
    if protected_interfaces is None:
        incoming = {str(t.GetPrimPath()) for prim in stage.TraverseAll() for prop in prim.GetAuthoredProperties()
                    for t in (prop.GetTargets() if isinstance(prop, Usd.Relationship) else prop.GetConnections())}
        protected_interfaces = _incoming_mesh_interfaces(incoming)
    for p in paths:
        prim = stage.GetPrimAtPath(p)
        reason = None
        code = None
        if not prim or not prim.IsA(UsdGeom.Mesh) or prim.IsInstanceProxy():
            reason = 'requires an editable mesh'
        elif not root.GetAttributeAtPath(p + '.points'):
            reason = 'geometry is already referenced or inherited'
        elif prim.HasAuthoredReferences() or prim.HasAuthoredPayloads() or prim.IsInstance():
            reason = 'existing mesh composition arcs remain unchanged'
        elif p in protected_interfaces:
            reason = 'incoming relationship or connection targets the mesh interface'
        elif any(a.GetNumTimeSamples() or a.GetConnections() for q in Usd.PrimRange(prim, Usd.PrimAllPrimsPredicate) for a in q.GetAttributes()):
            reason = 'animated mesh data or connections require separate sharing verification'
        elif any(r.GetTargets() for q in Usd.PrimRange(prim, Usd.PrimAllPrimsPredicate) for r in q.GetRelationships() if not r.GetName().startswith('material:binding')):
            reason = 'mesh relationships require explicit remapping'
        elif any(q.IsA(UsdGeom.Mesh) for q in list(Usd.PrimRange(prim, Usd.PrimAllPrimsPredicate))[1:]):
            reason = 'nested mesh scope must be resolved before sharing'
        elif prim.GetAllChildren():
            # Installed native deduplicateGeometry excludes child-bearing meshes,
            # including material subsets, before matching. No inclusion switch
            # is exposed. Keep their complete authored appearance unchanged.
            reason = 'native deduplicateGeometry excludes meshes with child prims, including material subsets'
            code = 'native_mesh_children_unsupported'
        if reason:
            row = {'path': p, 'reason': reason}
            if code:
                row['code'] = code
            skipped.append(row)
        else:
            eligible.append(p)
    return eligible, skipped


def authored_mesh_sources(stage, paths, entries):
    """Resolve existing asset-local prototypes, once per authored source.

    Arbitrary inherited/external references remain untouched. Return the source
    for each composed occurrence so publication can retain its path mapping.
    """
    from pxr import Sdf, Usd, UsdGeom
    import re
    root = stage.GetRootLayer()
    sources, occurrences = set(), {}
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsA(UsdGeom.Mesh):
            continue
        if root.GetAttributeAtPath(path + '.points') and not prim.IsInstanceProxy():
            sources.add(path)
            occurrences[path] = path
            continue
        for spec in prim.GetPrimStack():
            if spec.layer != root or 'points' not in spec.attributes:
                continue
            source = str(spec.path)
            namespaces = [entry + '/' + source[len(entry) + 1:].split('/')[0]
                          for entry in entries if source.startswith(entry + '/')]
            if any(re.fullmatch(r'Prototypes(?:_\d+)?', Sdf.Path(namespace).name)
                   and root.GetPrimAtPath(namespace).specifier == Sdf.SpecifierClass for namespace in namespaces):
                sources.add(source)
                occurrences[path] = source
                break
        if path not in occurrences:
            # Keep unsupported paths visible in the existing eligibility report.
            sources.add(path)
            occurrences[path] = path
    return sorted(sources), occurrences


def _organize_shared_meshes(core, stage, changed, namespace):
    """Scope native prototype organization to just this operation's outputs."""
    from pxr import Sdf, Usd
    from usd_optimize.core import ExecutionContext
    root = stage.GetRootLayer()
    isolated = Sdf.Layer.CreateAnonymous()
    for path in changed:
        for ancestor in Sdf.Path(path).GetPrefixes()[:-1]:
            spec = Sdf.CreatePrimInLayer(isolated, ancestor)
            spec.specifier = Sdf.SpecifierDef
            spec.typeName = 'Xform'
        Sdf.CopySpec(root, path, isolated, path)
    scratch = Usd.Stage.Open(isolated)
    context = ExecutionContext(); context.set_stage(scratch)
    organize = dict(operation='organizePrototypes', prototypesNamespace=namespace, hierarchyLevels=0)
    outcomes = core.executeConfig(context, [organize])
    if len(outcomes) != 1 or not outcomes[0][0] or not isolated.GetPrimAtPath(namespace):
        raise RuntimeError('native prototype organization failed: ' + str(outcomes))
    with Sdf.ChangeBlock():
        Sdf.CopySpec(isolated, namespace, root, namespace)
        for path in changed:
            Sdf.CopySpec(isolated, path, root, path)


def share_geometry(staging_layer, owned_mesh_paths, *, entry_prim_paths, read_only_entry_paths=(), policy=None, maximum_occurrence_scale=1.0, progress=None):
    """Modify only the caller's unpublished staging layer after native execution and composition checks.

    Returns status shared/unchanged/verification_failure/execution_failure,
    accepted, native configuration, measured authored point-array reduction,
    old-mesh to composed-mesh path_map, and composition checks. These are not
    independent geometric preservation proofs. Unsupported
    meshes remain unchanged and are listed in skipped. No point instancers.
    The caller must never pass an original asset or an immutable library here.
    """
    timings = {}
    last_phase, last_started = None, time.monotonic()
    def phase(name):
        nonlocal last_phase, last_started
        now = time.monotonic()
        if last_phase is not None:
            timings[last_phase] = now - last_started
        last_phase, last_started = name, now
        if progress:
            progress('library_publication', activity=name, sharing_phase_seconds=dict(timings))
    phase('Inspect sharing eligibility')
    from pxr import Sdf, Usd, UsdGeom
    from usd_optimize.core import ExecutionContext, UsdOptimizeCore
    path = Path(staging_layer).resolve()
    stage = Usd.Stage.Open(str(path))
    if not stage or stage.GetCompositionErrors():
        raise ValueError('category staging composition is invalid')
    root = stage.GetRootLayer()
    paths = sorted(set(str(p) for p in owned_mesh_paths))
    entries = sorted(set(str(p) for p in entry_prim_paths))
    readonly = [Sdf.Path(p) for p in read_only_entry_paths]
    if not entries or any(not Sdf.Path(p).IsAbsolutePath() or not stage.GetPrimAtPath(p) for p in entries):
        raise ValueError('existing absolute entry prim paths are required')
    entry_paths = {Sdf.Path(e) for e in entries}
    for p in paths:
        pp = Sdf.Path(p)
        if not pp.IsAbsolutePath() or not pp.IsPrimPath() or not _has_entry_ancestor(pp, entry_paths):
            raise ValueError('owned mesh must be below a declared entry: ' + p)
        if any(pp.HasPrefix(e) for e in readonly):
            raise ValueError('inherited entry is read-only: ' + p)
    tolerance_path = Path(__file__).resolve().parents[2] / 'geometry_tolerance.py'
    spec = importlib.util.spec_from_file_location('sharing_tolerance', tolerance_path)
    tolerance = importlib.util.module_from_spec(spec); spec.loader.exec_module(tolerance)
    budget_mm = tolerance.local_tolerance_mm(policy, maximum_occurrence_scale)
    result = {'schema_version': 1, 'staging_layer': str(path), 'status': 'unchanged', 'accepted': False,
              'input_sha256': _digest(path), 'path_map': {p: [p] for p in paths},
              'submitted_source_count': 0, 'native_processed_source_count': None,
              'changed_source_count': 0, 'known_native_exclusion_count': 0,
              'skipped': [], 'checks': {}, 'internal_references': [], 'physical_tolerance_mm': budget_mm,
              'verification_basis': 'native_execution_and_composition', 'phase_seconds': timings}
    sources, occurrences = authored_mesh_sources(stage, paths, entries)
    incoming = {str(t.GetPrimPath()) for prim in stage.TraverseAll() for prop in prim.GetAuthoredProperties()
                for t in (prop.GetTargets() if isinstance(prop, Usd.Relationship) else prop.GetConnections())}
    interfaces = _incoming_mesh_interfaces(incoming)
    eligible, result['skipped'] = eligible_meshes(stage, sources, interfaces)
    protected_sources = {source for occurrence, source in occurrences.items() if occurrence in interfaces}
    for source in eligible[:]:
        if source in protected_sources:
            eligible.remove(source)
            result['skipped'].append({'path': source, 'reason': 'referencing occurrence has an addressed mesh interface'})
    result['eligible_source_count'] = len(eligible)
    result['known_native_exclusion_count'] = sum(row.get('code') == 'native_mesh_children_unsupported' for row in result['skipped'])
    result['referencing_occurrence_count'] = len(occurrences)
    initial = Sdf.Layer.CreateAnonymous(); initial.TransferContent(root)
    before_storage = _storage(root)
    result['before'] = before_storage
    result['after'] = before_storage
    if len(eligible) < 2:
        phase('Geometry sharing complete')
        result.update(accepted=True, reason='fewer than two eligible owned meshes', output_sha256=_digest(path), authored_point_arrays_removed=0)
        return result
    entry_meshes = {entry: _mesh_paths(stage, entry) for entry in entries}
    # Preserve the installed operation's conservative matching configuration.
    # Native USD Optimize owns equality; the budget caps its stage-unit tolerance.
    op = {'operation': 'deduplicateGeometry', 'meshPrimPaths': eligible,
          'duplicateMethod': 2, 'tolerance': min(1e-8, tolerance.stage_tolerance(policy, UsdGeom.GetStageMetersPerUnit(stage), maximum_occurrence_scale)),
          'fuzzy': False, 'allowScaling': False, 'considerDeepTransforms': False, 'ignoreAttributes': []}
    result['configuration'] = op
    started = time.monotonic()
    try:
        stage.SetEditTarget(root)
        context = ExecutionContext(); context.set_stage(stage)
        phase('Run native geometry sharing')
        # The native return tuple exposes success, but no processed-prim count.
        result['submitted_source_count'] = len(eligible)
        outcomes = UsdOptimizeCore.getInstance().executeConfig(context, [op])
        phase('Check shared geometry paths')
        result['native_result'] = str(outcomes)
        if len(outcomes) != 1 or not outcomes[0][0]:
            raise RuntimeError('native deduplicateGeometry did not succeed')
        changed = [p for p in eligible if not stage.GetPrimAtPath(p).IsA(UsdGeom.Mesh)]
        result['changed_source_count'] = len(changed)
        if changed:
            common = Sdf.Path(entries[0]).GetParentPath()
            for entry in entries[1:]:
                common = common.GetCommonPrefix(Sdf.Path(entry).GetParentPath())
            namespace = prototype_namespace(stage, str(common).rstrip('/'))
            _organize_shared_meshes(UsdOptimizeCore.getInstance(), stage, changed, namespace)
            result['prototypes_namespace'] = namespace
        for p in eligible:
            result['path_map'][p] = _mesh_paths(stage, p)
        for occurrence, source in occurrences.items():
            mapped = result['path_map'].get(source, [source])
            result['path_map'][occurrence] = [occurrence + path[len(source):] for path in mapped]

        checks = result['checks']
        checks['output_mesh_paths'] = all(len(result['path_map'][p]) == 1 for p in eligible)
        checks['entrypoints'] = all(bool(stage.GetPrimAtPath(p)) for p in entries)
        checks['composition'] = not stage.GetCompositionErrors()
        checks['external_entry_references'] = True
        phase('Check referenced library entries')
        for entry in entries:
            check = Usd.Stage.CreateInMemory()
            check.DefinePrim('/Entry').GetReferences().AddReference(str(path), entry)
            expected = []
            for p in entry_meshes[entry]:
                q = p if stage.GetPrimAtPath(p).IsA(UsdGeom.Mesh) else p + '/Geometry'
                expected.append('/Entry' + q[len(entry):])
            checks['external_entry_references'] &= (not check.GetCompositionErrors()
                and bool(check.GetPrimAtPath('/Entry')) and sorted(_mesh_paths(check, '/Entry')) == sorted(expected))
        phase('Record shared geometry storage')
        for p in eligible:
            spec = root.GetPrimAtPath(p)
            if spec:
                result['internal_references'].extend({'path': p, 'target': str(ref.primPath)} for ref in spec.referenceList.GetAppliedItems() if not ref.assetPath)
        result['after'] = _storage(root)
        result['authored_point_arrays_removed'] = before_storage['authored_point_arrays'] - result['after']['authored_point_arrays']
        if all(checks.values()):
            root.Save()
            result.update(accepted=True, status='shared' if result['authored_point_arrays_removed'] > 0 else 'unchanged')
        else:
            result['status'] = 'verification_failure'
    except Exception as error:
        result.update(status='execution_failure', error=str(error))
    finally:
        if not result['accepted']:
            try:
                result['failure_artifacts'] = _support().retain_failure_layers(path, initial, root)
            finally:
                root.TransferContent(initial)
                result['path_map'] = {p: [p] for p in paths}
                result['after'] = before_storage
                result['authored_point_arrays_removed'] = 0
                result['changed_source_count'] = 0
        phase('Geometry sharing complete')
        result['elapsed_seconds'] = time.monotonic() - started
        result['output_sha256'] = _digest(path)
    return result


def share_definitions(staging_layer, eligible_entry_paths, *, policy=None, maximum_occurrence_scale=1.0):
    """Bounded native matching of explicitly instanceable, newly owned definitions.

    Compare normalized static transform stacks in a disposable stage. Commit only
    native-confirmed whole roots; entry and descendant paths remain unchanged.
    Input signatures describe original inputs and are never rewritten here.
    """
    import os
    import tempfile
    from pxr import Sdf, Usd, UsdGeom
    from usd_optimize.core import ExecutionContext, UsdOptimizeCore
    path = Path(staging_layer).resolve()
    evidence = {'accepted': True, 'status': 'unchanged', 'submitted_entry_count': 0,
                'native_processed_entry_count': None, 'shared_entry_count': 0,
                'skipped': [], 'internal_references': [], 'checks': {}}
    candidates = sorted(set(eligible_entry_paths))
    if len(candidates) < 2:
        return evidence
    stage = Usd.Stage.Open(str(path)); root = stage.GetRootLayer()
    if any(Sdf.Path(p).GetParentPath() != Sdf.Path('/Library') for p in candidates):
        raise ValueError('whole-definition sharing requires direct /Library entries')
    incoming = {str(t) for prim in stage.TraverseAll() for prop in prim.GetAuthoredProperties()
                for t in (prop.GetTargets() if isinstance(prop, Usd.Relationship) else prop.GetConnections())}
    selected = []
    for path_string in candidates:
        prim = stage.GetPrimAtPath(path_string)
        reason = None
        if not prim or prim.IsInstance() or prim.HasAuthoredReferences() or prim.HasAuthoredPayloads():
            reason = 'entry already has composition arcs'
        else:
            for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
                if child.GetVariantSets().GetNames() or not child.IsActive():
                    reason = 'variant or inactive content requires separate sharing verification'; break
                if any(a.GetNumTimeSamples() for a in child.GetAttributes()):
                    reason = 'animated definition requires separate sharing verification'; break
                xf = UsdGeom.Xformable(child)
                if xf and any(a.GetConnections() or str(a.GetPath()) in incoming
                              for a in child.GetAttributes()
                              if a.GetName() == 'xformOpOrder' or a.GetName().startswith('xformOp:')):
                    reason = 'addressed transform properties require their authored stacks'; break
                # Internal prototype references are allowed. Cross-entry arcs
                # would make this definition depend on another candidate's state.
                for item in child.GetPrimStack():
                    for ref in item.referenceList.GetAppliedItems():
                        if not ref.assetPath and ref.primPath and not ref.primPath.HasPrefix(Sdf.Path(path_string)):
                            target = stage.GetPrimAtPath(ref.primPath)
                            if not target or not target.IsAbstract():
                                reason = 'cross-entry reference requires separate sharing verification'; break
                    if reason: break
                if reason: break
        if reason:
            evidence['skipped'].append({'path': path_string, 'reason': reason})
        else:
            selected.append(path_string)
    if len(selected) < 2:
        return evidence
    tolerance_path = Path(__file__).resolve().parents[2] / 'geometry_tolerance.py'
    spec = importlib.util.spec_from_file_location('definition_sharing_tolerance', tolerance_path)
    tolerance = importlib.util.module_from_spec(spec); spec.loader.exec_module(tolerance)
    op = dict(operation='deduplicateHierarchies', paths=['/Library'], maxDepth=1,
              tolerance=min(1e-8, tolerance.stage_tolerance(policy, UsdGeom.GetStageMetersPerUnit(stage), maximum_occurrence_scale)),
              ignoreShaderOutputs=False)
    evidence['configuration'] = op
    initial = Sdf.Layer.CreateAnonymous(); initial.TransferContent(root)
    before = _storage(root)
    original_mesh_paths = {entry: _mesh_paths(stage, entry) for entry in selected}
    # A sibling file keeps relative dependencies anchored without rewriting them.
    fd, temporary = tempfile.mkstemp(prefix='.definition-sharing-', suffix='.usdc', dir=path.parent)
    os.close(fd)
    try:
        root.Export(temporary)
        scratch = Sdf.Layer.FindOrOpen(temporary)
        mask = Usd.StagePopulationMask(selected)
        work = Usd.Stage.OpenMasked(scratch, mask)
        work.ExpandPopulationMask()
        # Dependencies outside candidates may be populated, but native matching
        # must see only candidate roots plus abstract prototype containers.
        for child in work.GetPrimAtPath('/Library').GetAllChildren():
            if str(child.GetPath()) not in selected and not child.IsAbstract():
                evidence['skipped'].append({'path': str(child.GetPath()), 'reason': 'candidate has a populated external entry dependency'})
                return evidence
        for entry in selected:
            for prim in Usd.PrimRange(work.GetPrimAtPath(entry), Usd.PrimAllPrimsPredicate):
                xf = UsdGeom.Xformable(prim)
                if not xf: continue
                matrix, reset = xf.GetLocalTransformation(), xf.GetResetXformStack()
                for attr in list(prim.GetAttributes()):
                    if attr.GetName().startswith('xformOp:'):
                        prim.RemoveProperty(attr.GetName())
                xf.MakeMatrixXform().Set(matrix); xf.SetResetXformStack(reset)
        context = ExecutionContext(); context.set_stage(work)
        evidence['submitted_entry_count'] = len(selected)
        try:
            outcomes = UsdOptimizeCore.getInstance().executeConfig(context, [op])
            evidence['native_result'] = str(outcomes)
            if len(outcomes) != 1 or not outcomes[0][0]:
                raise RuntimeError('native whole-definition sharing did not succeed')
            changed, masters = [], set()
            for entry in selected:
                prim = work.GetPrimAtPath(entry)
                refs = list(scratch.GetPrimAtPath(entry).referenceList.GetAppliedItems())
                if prim.IsInstance() and refs:
                    if len(refs) != 1 or refs[0].assetPath or str(refs[0].primPath) not in selected:
                        raise RuntimeError('native whole-definition reference leaves owned roots')
                    changed.append(entry); masters.add(str(refs[0].primPath))
                    evidence['internal_references'].append({'path': entry, 'target': str(refs[0].primPath)})
            if not changed:
                return evidence
            with Sdf.ChangeBlock():
                for entry in set(changed) | masters:
                    Sdf.CopySpec(scratch, entry, root, entry)
            checks = evidence['checks']
            checks['composition'] = not stage.GetCompositionErrors()
            checks['entry_paths'] = all(bool(stage.GetPrimAtPath(entry)) for entry in selected)
            checks['mesh_paths'] = all(_mesh_paths(stage, entry) == original_mesh_paths[entry] for entry in selected)
            checks['external_references'] = True
            for entry in set(changed) | masters:
                consumer = Usd.Stage.CreateInMemory()
                consumer.DefinePrim('/Entry').GetReferences().AddReference(str(path), entry)
                checks['external_references'] &= not consumer.GetCompositionErrors() and bool(consumer.GetPrimAtPath('/Entry'))
            if not all(checks.values()):
                raise RuntimeError('whole-definition output does not compose')
            root.Save()
            evidence.update(status='shared', shared_entry_count=len(changed),
                            authored_point_arrays_removed=before['authored_point_arrays'] - _storage(root)['authored_point_arrays'])
        except Exception as error:
            evidence.update(accepted=False, status='execution_failure', error=str(error))
            evidence['failure_artifacts'] = _support().retain_failure_layers(path, initial, root)
            root.TransferContent(initial)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return evidence


def prototype_namespace(stage, boundary):
    namespace = boundary + '/Prototypes'
    suffix = 1
    while stage.GetPrimAtPath(namespace):
        namespace = boundary + '/Prototypes_' + str(suffix)
        suffix += 1
    return namespace
