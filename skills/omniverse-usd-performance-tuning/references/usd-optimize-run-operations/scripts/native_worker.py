#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run and verify one generated USD layer writer without modifying source assets."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback


def load_helper(name, relative):
    path = Path(__file__).resolve().parents[2] / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def digest(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def peak_rss_kb():
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return value / 1024 if sys.platform == 'darwin' else value
    except ImportError:
        try:
            import psutil
            memory = psutil.Process().memory_info()
            return getattr(memory, 'peak_wset', memory.rss) / 1024
        except ImportError:
            return None


def geometry_paths(target):
    return target.get('operation_scope', {}).get('geometry_paths', target.get('owned_geometry', []))


def measure(stage, paths):
    from pxr import UsdGeom
    result = {'meshes': 0, 'vertices': 0, 'faces': 0, 'analytic_primitives': 0, 'array_bytes': 0}
    for path in set(paths):
        prim = stage.GetPrimAtPath(path)
        if not prim:
            continue
        for attr in prim.GetAttributes():
            for sample in [None] + attr.GetTimeSamples():
                value = attr.Get() if sample is None else attr.Get(sample)
                if value is not None and type(value).__module__ == 'pxr.Vt':
                    import numpy as np
                    array = np.asarray(value)
                    if array.dtype.kind in 'biufc':
                        result['array_bytes'] += array.nbytes
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            result['meshes'] += 1
            result['vertices'] += len(mesh.GetPointsAttr().Get() or [])
            result['faces'] += len(mesh.GetFaceVertexCountsAttr().Get() or [])
        elif prim.IsA(UsdGeom.Gprim):
            result['analytic_primitives'] += 1
    return result


def outside_scope_digest(stage, paths):
    """Compare authored opinions outside editable geometry, without traversing children."""
    from pxr import Sdf
    layer = Sdf.Layer.CreateAnonymous()
    layer.TransferContent(stage.GetRootLayer())
    edits = Sdf.BatchNamespaceEdit()
    for path in sorted(set(paths), key=len, reverse=True):
        if layer.GetPrimAtPath(path):
            edits.Add(path, Sdf.Path.emptyPath)
    if not layer.Apply(edits):
        raise ValueError('cannot isolate authored opinions outside operation scope')
    return hashlib.sha256(layer.ExportToString().encode()).hexdigest()


def select_edit_context(stage, target):
    """Keep operation opinions inside their declared variant branch."""
    stage.SetEditTarget(stage.GetRootLayer())
    for context in target.get('variant_context', []):
        prim = stage.GetPrimAtPath(context['path'])
        if not prim:
            raise ValueError('variant owner does not exist')
        variant = prim.GetVariantSet(context['set'])
        if context['selection'] not in variant.GetVariantNames():
            raise ValueError('declared variant selection does not exist')
        variant.SetVariantSelection(context['selection'])
        stage.SetEditTarget(variant.GetVariantEditTarget(stage.GetRootLayer()))


def physical_tolerance(request):
    return load_helper('worker_geometry_budget', 'geometry_tolerance.py').physical_tolerance_mm(request)


def verification_policy(request, operations):
    """Derive geometric-normal allowance only for an authorized fit attempt."""
    policy = {key:value for key,value in request.items() if not key.startswith('_')}
    policy['tolerance_mm'] = physical_tolerance(request)
    lossless = (request.get('preservation') == 'lossless' or request.get('preservation_profile') == 'lossless'
                or request.get('lossless') is True or request.get('exact_geometry') is True)
    fit = any(row.get('operation') == 'fitPrimitives' for row in operations)
    allowed = fit and request.get('_explicit_fit_primitives',request.get('fit_primitives')) is True and request.get('preserve_normals') is not True and not lossless
    if allowed:
        policy['tolerance_mm'] = physical_tolerance(request)
    policy['allow_surface_derived_primitive_normals'] = bool(allowed)
    return policy


def merge_scopes(target):
    for group in target.get('merge_groups', []):
        if 'operation_scope' in group:
            scoped = dict(group)
        else:
            scoped = {'boundary_path': target['boundary_path'],
                      'protected_paths': target.get('protected_paths', []),
                      'reduction_route': 'merge',
                      'operation_scope': {'type': 'interior', 'mesh_paths': group['mesh_paths'],
                                          'output_path': group['output_path'],
                                          'spatial_merge': group.get('spatial_merge', {})}}
        selected = scoped['operation_scope']['mesh_paths']
        # Earlier groups may already have removed their meshes. Only paths
        # relevant to this exact operation may waive source-kind protection.
        ancestors = set()
        for mesh in selected:
            path = mesh
            while path and path not in ancestors:
                ancestors.add(path)
                path = path.rsplit('/', 1)[0]
        scoped['internal_paths'] = [path for path in target.get('internal_paths', [])
                                    if path in ancestors]
        yield scoped


CATEGORY_OPERATIONS = {
    'cleanup': {'meshCleanup', 'optimizePrimvars', 'removeUnusedUVs', 'removeSmallGeometry'},
    'primitive_fit': {'fitPrimitives'}, 'reduction': {'decimateMeshes'},
    'merge': {'merge'}, 'reuse': set(),
}


def category_selection(target, request, has_mesh):
    """Resolve discovered work without treating absent configuration as permission."""
    discovered = {item['category']: item for item in target.get('assessment', {}).get('operations', [])
                  if isinstance(item, dict) and item.get('category') in CATEGORY_OPERATIONS}
    lossless = (request.get('preservation') == 'lossless' or request.get('preservation_profile') == 'lossless'
                or request.get('lossless') is True or request.get('exact_geometry') is True)
    local = dict(request)
    categories = []
    for name in CATEGORY_OPERATIONS:
        item = discovered.get(name, {})
        category = {'category': name, 'status': 'pending', 'reason': '',
                    'evidence': {'discovery': item, 'has_mesh_geometry': bool(has_mesh)}}
        flag = {'primitive_fit': 'fit_primitives', 'reduction': 'decimate'}.get(name)
        if name == 'reuse':
            if item.get('status') in ('assessed', 'confirmed', 'complete') and item.get('evidence'):
                category.update(status='executed', reason='Discovery assessed exact reuse and recorded its authored definition decisions.')
            elif not item:
                category.update(status='not_applicable', reason='This worker target has no discovery reuse assignment; its caller owns structure discovery.')
            else:
                category.update(status='incomplete', reason='The discovered reuse opportunity has no confirmed result.')
        elif flag:
            nominated = item.get('status') == 'candidate' or request.get(flag) is True
            if request.get(flag) is False or lossless:
                local[flag] = False
                category.update(status='excluded', reason=('The request explicitly disables '+flag+'.') if request.get(flag) is False else 'The requested lossless preservation profile excludes approximate geometry changes.')
                category['evidence']['request_scope'] = {flag: request.get(flag), 'lossless': lossless}
            elif name == 'primitive_fit' and not fitting_paths(target):
                local[flag] = False
                category.update(status='not_applicable', reason='The merged_meshes representation reserves these meshes for interior merging.')
            elif not has_mesh or item.get('status') == 'not_applicable':
                local[flag] = False
                category.update(status='not_applicable', reason='The inspected scope has no applicable Mesh geometry.')
            elif not nominated:
                local[flag] = False
                category.update(status='not_applicable', reason='No operation in this category was nominated by discovery or requested for this worker scope.')
            elif target.get('assessment', {}).get('functional_precision', {}).get('required') and not request.get('functional_tolerance'):
                local[flag] = False
                category.update(status='incomplete', reason='Functional geometry requires an explicit functional tolerance; a visual deviation budget does not authorize this operation.')
            else:
                try:
                    physical_tolerance(request)
                    local[flag] = True
                except ValueError:
                    local[flag] = False
                    category.update(status='incomplete', reason='A positive physical deviation budget is required for the nominated '+name+' operation.')
        elif name == 'merge' and not target.get('merge_groups'):
            if item.get('status') == 'candidate':
                category.update(status='incomplete', reason='Discovery nominated interior merging but supplied no verified merge scope.')
            else:
                category.update(status='not_applicable', reason='Discovery found no compatible anonymous fragment group in this retained boundary.')
        elif name == 'cleanup' and not has_mesh:
            category.update(status='not_applicable', reason='This definition owns no Mesh geometry.')
        categories.append(category)
    return local, categories


def finish_categories(categories, records, remaining):
    decisions = []
    for category in categories:
        name = category['category']
        operations = [record for record in records if record.get('operation') in CATEGORY_OPERATIONS[name]]
        if category['status'] == 'pending':
            if operations:
                category['status'] = ('executed' if any(record.get('status') == 'changed' for record in operations)
                                      else 'not_applicable' if all(record.get('status') == 'not_applicable' for record in operations)
                                      else 'no_benefit')
                category['reason'] = 'The scoped native operations ran and their actual effects were measured.'
                category['evidence']['operation_results'] = [
                    {'operation': record['operation'], 'status': record['status'], 'evidence': record.get('evidence')}
                    for record in operations]
            else:
                category.update(status='incomplete', reason='A nominated operation category was not executed or independently ruled out.')
        if name == 'cleanup' and remaining:
            category.update(status='incomplete', reason='Applicable cleanup findings remain after the bounded optimization pass.')
            category['evidence']['remaining_findings'] = remaining
        if category['status'] == 'incomplete':
            decisions.append({'category': name, 'reason': category['reason'], 'evidence': category['evidence']})
    return decisions


def actionable_findings(evidence, request):
    """Keep source winding/topology intact unless a verified repair is authorized."""
    actionable, exclusions = [], []
    for issue in evidence.get('issues', []):
        concept = issue.get('concept') if isinstance(issue, dict) else None
        if concept in ('normals_winding', 'topology_zero_area_faces'):
            exclusions.append({'concept': concept, 'reason': 'The preservation contract retains source face orientation and topology; blanket winding or degenerate-face repair is outside this verified operation chain.', 'finding': issue})
        elif concept == 'primvar_unused' and request.get('preserve_uvs') is True:
            proof = request.get('_uv_removal_check', {})
            reason = proof.get('reason') if proof and not proof.get('allowed') else 'The request explicitly preserves UV attributes.'
            exclusions.append({'concept': concept, 'reason': reason, 'evidence': proof, 'finding': issue})
        elif concept in ('primvar_unused', 'primvar_indexability', 'vertex_weld', 'topology_isolated_vertices'):
            actionable.append(issue)
    return actionable, exclusions


def fitting_paths(target):
    """Reserve compatible static merge inputs unless analytic prims are requested."""
    representation = target.get('geometry_representation', 'merged_meshes')
    if representation not in ('merged_meshes', 'analytic_primitives'):
        raise ValueError('Invalid geometry_representation')
    paths = target.get('operation_scope', {}).get('mesh_paths', [])
    merging = {path for group in target.get('merge_groups', [])
               for path in group.get('operation_scope', group)['mesh_paths']}
    placement = {e['path'].split('.')[0] for e in target.get('required_occurrence_edits', [])
                 if e.get('edit_class', 'placement') == 'placement'}
    return [path for path in paths if path not in placement and
            (representation == 'analytic_primitives' or path not in merging)]


def operation_config(target, request, stage):
    from pxr import UsdGeom
    paths = target.get('operation_scope', {}).get('mesh_paths', [])
    threshold = request.get('remove_small_geometry_mm')
    if threshold is not None:
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold < 0:
            raise ValueError('remove_small_geometry_mm must be finite and nonnegative')
        request['_small_geometry_threshold'] = threshold/(UsdGeom.GetStageMetersPerUnit(stage)*1000*target.get('maximum_occurrence_scale', 1.0))
        facts = load_helper('small_geometry_protection', 'component_scope.py').merge_protection_facts(
            stage, target.get('internal_paths', []),
            boundary_paths=target.get('operation_scope', {}).get('exclude_paths', []))
        protected = (set(target.get('protected_paths', [])) - {target.get('boundary_path')}) | {
            p for p, reasons in facts.items() if p != target.get('boundary_path') or set(reasons) - {'metadata:kind'}}
        request['_small_geometry_paths'] = [p for p in paths if not any(
            p == q or p.startswith(q + '/') or q.startswith(p + '/') for q in protected)]
    if 'operations' in target:
        eligible = set(fitting_paths(target))
        operations = [dict(op, paths=[p for p in op['paths'] if p in eligible])
                      if op.get('operation') == 'fitPrimitives' else dict(op)
                      for op in target['operations']]
        return [op for op in operations
            if op.get('operation') != 'fitPrimitives' or op['paths']]
    if not paths:
        return []
    ops = [dict(operation='meshCleanup', paths=paths, tolerance=0.0,
                mergeVertices=True, contractDegenerateEdges=False, removeDegenerateFaces=False, removeDuplicateFaces=False,
                removeIsolatedVertices=True, makeManifold=False, coorientFaces=False),
           dict(operation='optimizePrimvars', paths=paths, mode=1, simplify=True)]
    threshold = request.get('remove_small_geometry_mm')
    if threshold is not None:
        selected = request['_small_geometry_paths']
        if selected:
            ops.insert(0, dict(operation='removeSmallGeometry', paths=selected,
                               detectionMethod=0, removeMethod=1,
                               threshold=request['_small_geometry_threshold']))
    fit_paths = fitting_paths(target)
    if request.get('fit_primitives', False) and fit_paths:
        ops.append(dict(operation='fitPrimitives', paths=fit_paths,
                        ignoreNonConstPrimvars=False, ignoreSubsets=False,
                        allowMissingEndcaps=False, generateMeshes=False,
                        fitSphere=True, fitCube=True, fitCone=True, fitCylinder=True,
                        gpuFaceCountThreshold=0))
    if request.get('decimate', False):
        mm = physical_tolerance(request)
        ops.append(dict(operation='decimateMeshes', paths=paths,
                        maxMeanError=mm/(UsdGeom.GetStageMetersPerUnit(stage)*1000),
                        reductionFactor=0.0, pinBoundaries=True))
    return ops


def validate_operations(core, target, operations, request):
    if not isinstance(operations, list):
        raise ValueError('operations must be a list')
    owned = set(geometry_paths(target))
    available = set(core.getOperations())
    for op in operations:
        name = op.get('operation') if isinstance(op, dict) else None
        if name not in available:
            raise ValueError('unsupported operation: ' + str(name))
        supported = set().union(*CATEGORY_OPERATIONS.values())
        if name not in supported:
            raise ValueError('operation requires a separately authorized and verified execution route: ' + name)
        if name == 'removeSmallGeometry':
            if request.get('remove_small_geometry_mm') is None:
                raise ValueError('small geometry removal requires remove_small_geometry_mm')
            if not isinstance(op.get('threshold'), (int, float)) or not 0 <= op['threshold'] <= request.get('_small_geometry_threshold', -1):
                raise ValueError('small geometry threshold exceeds the authorized physical size')
            if not set(op.get('paths', [])).issubset(request.get('_small_geometry_paths', [])):
                raise ValueError('small geometry removal crosses authored animation or an interface')
            if op.get('removeMethod') != 1 or op.get('detectionMethod') != 0:
                raise ValueError('small geometry removal requires world-space deletion')
            if any(p == q or p.startswith(q + '/') or q.startswith(p + '/')
                   for p in op.get('paths', []) for q in target.get('protected_paths', [])):
                raise ValueError('small geometry removal crosses a required interface')
        if name == 'merge':
            scope = load_helper('worker_merge_paths', 'component_scope.py')
            if op.get('mergePoint', 0) == 0:
                resolved = scope.native_merge_config(op['meshPrimPaths'], op['rootPath'])
                op.update(rootPath=resolved['rootPath'], mergePoint=resolved['mergePoint'])
            scope.merge_output_path(op)
        if name == 'meshCleanup':
            # Native defaults contract repeated face corners independently of
            # removeDegenerateFaces. Preserve those authored corner fields too.
            op.setdefault('contractDegenerateEdges',False)
        if name == 'meshCleanup' and any(op.get(key, False) for key in
                ('makeManifold', 'coorientFaces', 'removeDegenerateFaces', 'removeDuplicateFaces','contractDegenerateEdges')):
            raise ValueError('topology repair is outside the worker preservation contract')
        if name == 'fitPrimitives' and any(p not in fitting_paths(target) for p in op.get('paths', [])):
            raise ValueError('fitPrimitives crosses a required placement frame or reserved merge input')
        if name in ('fitPrimitives', 'decimateMeshes', 'removeSmallGeometry') and (
                request.get('preservation') == 'lossless' or request.get('preservation_profile') == 'lossless'
                or request.get('lossless') is True or request.get('exact_geometry') is True):
            raise ValueError('bounded-loss operation conflicts with the requested lossless preservation')
        arguments = {arg['name']: arg for arg in core.getOperationArguments(name)}
        unknown = set(op) - set(arguments) - {'operation'}
        if unknown:
            raise ValueError('unsupported operation arguments: ' + str(sorted(unknown)))
        paths = op.get('paths', op.get('meshPrimPaths'))
        if not isinstance(paths, list) or not paths or not set(paths).issubset(owned):
            raise ValueError('operation must select explicit geometry owned by its definition')
        if name in ('fitPrimitives', 'decimateMeshes'):
            if target.get('assessment', {}).get('functional_precision', {}).get('required') and not request.get('functional_tolerance'):
                raise ValueError('functional geometry requires an explicit functional tolerance')
            physical_tolerance(request)
        for key, value in op.items():
            if key == 'operation':
                continue
            arg = arguments[key]; default = arg.get('defaultValue')
            if isinstance(default, bool) and not isinstance(value, bool):
                raise ValueError(key + ' must be boolean')
            if isinstance(default, (int, float)) and not isinstance(default, bool):
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(key + ' must be a finite number')
                bounds = arg.get('metadata', {})
                if value < bounds.get('min', -math.inf) or value > bounds.get('max', math.inf):
                    raise ValueError(key + ' is outside the supported range')


def assess_operations(source, target, operations):
    """Batch scoped native probes in one stage open per definition."""
    started = time.monotonic()
    names = {op['operation'] for op in operations}
    mapping = {'meshCleanup': ['vertex_weld', 'topology_isolated_vertices'],
               'optimizePrimvars': ['primvar_indexability'], 'fitPrimitives': ['primitive_fit']}
    concepts = sorted({'primvar_unused', 'primvar_indexability', 'vertex_weld',
                       'topology_zero_area_faces', 'normals_winding', 'topology_isolated_vertices'}
                      | {concept for op in names for concept in mapping.get(op, [])})
    paths = target.get('operation_scope', {}).get('mesh_paths', [])
    if not concepts or not paths:
        return {'concepts': [], 'scope': paths, 'issues': [], 'stage_opens': 0, 'elapsed_seconds': 0.0}
    import usd_optimize.validators
    usd_optimize.validators.register_all()
    executor = load_helper('worker_validation_executor', 'usd-validation-runner/scripts/usd_validation_executor.py')
    registry = executor.load_registry()
    identities = {}
    for concept in concepts:
        implementation = executor.resolve_implementation(registry, concept)
        identities[(executor.module_family(implementation.module), implementation.class_name)] = concept
    issues = executor.validate_concepts(str(source), concepts, mask_paths=paths, registry=registry)
    rows = []
    for issue in issues:
        rule = issue.rule
        concept = identities.get((executor.module_family(rule.__module__), rule.__name__))
        if concept is None:
            raise ValueError('validator returned an issue from an unselected rule')
        rows.append({'concept': concept, 'message': issue.message, 'severity': str(issue.severity),
                     'path': str(getattr(issue.at, 'path', ''))})
    return {'concepts': concepts, 'scope': paths, 'issues': rows, 'stage_opens': 1,
            'elapsed_seconds': time.monotonic() - started}


_OPERATION_PROGRESS_PATH = None


def _execute_native(core, context, op):
    started = time.time()
    def publish(state):
        if _OPERATION_PROGRESS_PATH is None:
            return
        path = Path(_OPERATION_PROGRESS_PATH)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps({'operation': op['operation'], 'state': state,
                                             'started_epoch': started}), encoding='utf-8')
            temporary.replace(path)
        except OSError:
            # Observability must not change native execution or its verdict.
            pass
    publish('running')
    try:
        result = core.executeConfig(context, [op])
    except BaseException:
        publish('failed')
        raise
    publish('finished')
    return result


def operation_metrics(stage, paths, accounting):
    """Reuse only the immediately preceding measurement of the same scope.

    The caller invalidates this local record after every native mutation. It
    never reuses another scope's metrics or survives an execution sequence.
    """
    scope = tuple(sorted(set(paths)))
    if accounting.get('stage') != stage or accounting.get('scope') != scope:
        accounting.clear()
        accounting.update(stage=stage, scope=scope, metrics=measure(stage, scope))
    return dict(accounting['metrics'])


def apply_operation(core, context, stage, op, paths, path_map, pass_number, accounting=None):
    current_paths = (sorted(set(op['paths'])) if op['operation'] == 'fitPrimitives'
                     else sorted(set(op['meshPrimPaths'])) if op['operation'] == 'merge'
                     else sorted(set(paths) | set(path_map.values())))
    accounting = {} if accounting is None else accounting
    before_metrics = operation_metrics(stage, current_paths, accounting)
    snapshot = None
    if op['operation'] == 'decimateMeshes':
        from pxr import Sdf
        snapshot = Sdf.Layer.CreateAnonymous()
        snapshot.TransferContent(stage.GetRootLayer())
    empty_meshes = []
    if op['operation'] == 'meshCleanup':
        from pxr import UsdGeom
        empty_meshes = [path for path in op.get('paths', [])
                        if stage.GetPrimAtPath(path) and stage.GetPrimAtPath(path).IsA(UsdGeom.Mesh)
                        and not UsdGeom.Mesh(stage.GetPrimAtPath(path)).GetFaceVertexCountsAttr().Get()]
    if op['operation'] == 'removeSmallGeometry':
        empty_meshes = [p for p in op['paths'] if stage.GetPrimAtPath(p)]
    began = time.monotonic()
    outcomes = _execute_native(core, context, op)
    if len(outcomes) != 1 or not outcomes[0][0]:
        raise RuntimeError(f"{op['operation']} failed: {outcomes}")
    if op['operation'] == 'merge':
        from pxr import UsdGeom
        selected = op['meshPrimPaths']
        output_path = load_helper('worker_merge_paths', 'component_scope.py').merge_output_path(op)
        retained = [path for path in selected if stage.GetPrimAtPath(path) and stage.GetPrimAtPath(path).IsA(UsdGeom.Mesh)]
        removed = [path for path in selected if not stage.GetPrimAtPath(path)]
        returned = list(outcomes[0][2] or [])
        from pxr import Sdf
        destination = Sdf.Path(output_path)
        valid = lambda path: (Sdf.Path(path).GetParentPath() == destination.GetParentPath()
                              and (path == output_path or path.startswith(output_path + '_'))
                              and stage.GetPrimAtPath(path) and stage.GetPrimAtPath(path).IsA(UsdGeom.Mesh))
        if returned and (not all(valid(path) for path in returned) or len(set(returned)) != len(returned)):
            raise RuntimeError('native merge returned unexpected output paths')
        if len(retained) + len(removed) != len(selected) or bool(removed) != bool(returned):
            raise RuntimeError('native merge produced an ambiguous source-to-output mapping')
        if len(returned) == 1:
            path_map.update({path: returned[0] for path in removed})
        # Multiple clusters have group-level correspondence only; native merge
        # does not return membership. Do not invent a per-source path mapping.
        current_paths = sorted(set(retained + returned))
    accounting.clear()
    after_metrics = operation_metrics(stage, current_paths, accounting)
    changed = any(after_metrics[key] < before_metrics[key] for key in ('meshes', 'vertices', 'faces', 'array_bytes'))
    native_after = None
    if snapshot is not None and not changed:
        native_after = {'metrics': after_metrics}
        stage.GetRootLayer().TransferContent(snapshot)
        after_metrics = dict(before_metrics)
        accounting.update(metrics=after_metrics)
    record = {'operation': op['operation'], 'success': True, 'pass': pass_number,
            'status': 'changed' if changed else 'no_benefit',
            'reason': 'Scoped native operation reduced geometry or stored array data.' if changed else 'Scoped native operation produced no measured geometry or array-storage reduction.',
            'evidence': {'scope': op.get('paths', op.get('meshPrimPaths')), 'before': before_metrics,
                         'after': after_metrics},
            'seconds': time.monotonic()-began, 'result': str(outcomes[0][2])[:4000],
            'settings': dict(op), 'evidence_scope': 'native_execution', 'independent_geometry_proof': False,
            'consumed_geometry_paths': [path for path in empty_meshes if not stage.GetPrimAtPath(path)]}
    if op['operation'] == 'fitPrimitives' and not changed:
        record['reason'] = ('USD Optimize retained the Mesh prims under the configured relative RMS, '
                            'volume and primvar/subset constraints. This does not establish that '
                            'the shapes are unsuitable for primitive representation.')
    if op['operation'] == 'merge':
        record['evidence']['actual_output_paths'] = returned
        record['evidence']['removed_source_paths'] = removed
        record['evidence']['retained_source_paths'] = retained
        record['evidence']['path_map'] = {path: path_map[path] for path in selected if path in path_map}
    if native_after is not None:
        record['evidence']['native_after'] = native_after
        record['evidence']['reverted_no_benefit'] = True
        record['reason'] = 'Native decimation produced no measured reduction; restored the pre-operation layer, retaining prior operations.'
    return record


def resolved_geometry_paths(paths, path_map, operations):
    """Track retained geometry and native merge outputs without guessing cluster membership."""
    removed = {path for row in operations if row.get('operation') == 'merge'
               for path in row.get('evidence', {}).get('removed_source_paths', [])}
    removed.update(consumed_geometry_paths(operations))
    outputs = {path for row in operations if row.get('operation') == 'merge'
               for path in row.get('evidence', {}).get('actual_output_paths', [])}
    return sorted({path_map.get(path, path) for path in (set(paths) | outputs) - removed})


def reassess_and_continue(core, context, stage, target, request, operations, records, path_map, baseline, accounting=None):
    """Revisit affected cleanup findings; no-progress remains incomplete."""
    from pxr import UsdGeom
    budget = request.get('max_passes', 3)
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise ValueError('max_passes must be a positive integer')
    scans = [baseline]
    pass_number = 1
    while True:
        paths = resolved_geometry_paths(geometry_paths(target), path_map, records)
        meshes = [p for p in paths if stage.GetPrimAtPath(p) and stage.GetPrimAtPath(p).IsA(UsdGeom.Mesh)]
        scoped = dict(target, owned_geometry=paths,
                      operation_scope=dict(target.get('operation_scope', {}), geometry_paths=paths, mesh_paths=meshes))
        changed = any(r.get('pass', 1) == pass_number and r['status'] == 'changed' for r in records)
        if changed:
            scan = assess_operations(stage.GetRootLayer().identifier, scoped,
                                     [op for op in operations if 'operation' in op])
            scans.append(scan)
        else:
            scan = scans[-1]
        remaining, exclusions = actionable_findings(scan, request)
        # Merge is the final geometry operation. Welding coincident interfaces
        # afterward can erase surfaces that belonged to distinct source parts.
        merged_paths = {path_map.get(p, p) for row in records if row.get('operation') == 'merge'
                        for p in row.get('evidence', {}).get('actual_output_paths', [])}
        unmerged_meshes = sorted(set(meshes) - merged_paths)
        global_cleanup = any(i.get('path') == '/' and i.get('concept') in
                             ('vertex_weld', 'topology_isolated_vertices') for i in remaining)
        if merged_paths and unmerged_meshes and global_cleanup:
            # Native cleanup probes can report only the pseudo-root. Separate
            # untouched meshes before attributing that finding to merged seams.
            unmerged = dict(scoped, operation_scope=dict(scoped['operation_scope'],
                            mesh_paths=unmerged_meshes, geometry_paths=unmerged_meshes))
            other_scan = assess_operations(stage.GetRootLayer().identifier, unmerged, [op for op in operations if 'operation' in op])
            scan['unmerged_cleanup_probe'] = other_scan
            scan['stage_opens'] += other_scan['stage_opens']
            scan['elapsed_seconds'] += other_scan['elapsed_seconds']
            other_remaining, _ = actionable_findings(other_scan, request)
            unresolved = {i['concept'] for i in other_remaining}
        else:
            unresolved = set()
        retained = []
        for issue in remaining:
            if issue.get('concept') in ('vertex_weld', 'topology_isolated_vertices') and (
                    (issue.get('path') == '/' and merged_paths and issue['concept'] not in unresolved) or any(
                    issue.get('path', '') == p or issue.get('path', '').startswith(p + '.')
                    for p in merged_paths)):
                exclusions.append({'concept': issue['concept'], 'finding': issue,
                    'reason': 'Retain merged part interfaces; cleanup already ran before merging.'})
            else:
                retained.append(issue)
        remaining = retained
        scan['preservation_exclusions'] = exclusions
        if not remaining or not changed or pass_number >= budget:
            return scans, remaining
        pass_number += 1
        followup = []
        for original in operations:
            # A cleanup finding is not authority to repeat bounded-loss work.
            required = {'meshCleanup': {'vertex_weld', 'topology_isolated_vertices'},
                        'optimizePrimvars': {'primvar_indexability'},
                        'removeUnusedUVs': {'primvar_unused'}}
            if not required.get(original.get('operation'), set()).intersection(
                    issue['concept'] for issue in remaining):
                continue
            op = dict(original)
            op['paths'] = sorted(set(path_map.get(p, p) for p in op.get('paths', [])).intersection(meshes).difference(merged_paths))
            if op['paths']:
                followup.append(op)
        if any(issue['concept'] == 'primvar_unused' for issue in remaining) and 'operations' not in target:
            if not any(op['operation'] == 'removeUnusedUVs' for op in followup):
                followup.append({'operation': 'removeUnusedUVs', 'paths': meshes, 'mode': 0})
        if not followup:
            return scans, remaining
        validate_operations(core, scoped, followup, request)
        for op in followup:
            records.append(apply_operation(core, context, stage, op, paths, path_map, pass_number, accounting))


def ordered_definitions(definitions):
    if not definitions or len({target['id'] for target in definitions}) != len(definitions):
        raise ValueError('writer requires distinct definitions')
    pending = {target['id']: target for target in definitions}
    ordered = []
    while pending:
        ready = [target for target in pending.values()
                 if not set(target.get('children', [])).intersection(pending)]
        if not ready:
            raise ValueError('cyclic definitions in one writer layer')
        for target in ready:
            ordered.append(target); pending.pop(target['id'])
    return ordered


def consumed_geometry_paths(operations):
    """Paths observed removed by scoped cleanup or authorized small-geometry deletion."""
    return sorted({path for record in operations
                   if record.get('operation') in ('meshCleanup', 'removeSmallGeometry') and record.get('success') is True
                   for path in record.get('consumed_geometry_paths', [])
                   if path in record.get('settings', {}).get('paths', [])})


def native_execution_verification(source, output, target, policy, operations, evidence_files, animation):
    """Check runtime output composition/accounting; make no surface-proof claim."""
    from pxr import Usd, UsdGeom
    mapping = policy.get('path_map', {})
    consumed = set(consumed_geometry_paths(operations))
    consumed_outputs = {mapping.get(path,path) for path in consumed}
    final_paths = resolved_geometry_paths(geometry_paths(target), mapping, operations)
    missing = [path for path in final_paths if path not in consumed and path not in consumed_outputs
               and (not output.GetPrimAtPath(path) or not output.GetPrimAtPath(path).IsA(UsdGeom.Gprim))]
    motion = animation.verify(source, output, paths=geometry_paths(target), path_map=mapping,
                              requested_times=policy.get('time_samples', []))
    checks = [
        {'name': 'written composition', 'passed': not output.GetCompositionErrors()},
        {'name': 'accounted native output paths', 'passed': not missing, 'evidence': missing},
        {'name': 'consumed geometry preserves protected interfaces',
         'passed': not any(keep == path or keep.startswith(path + '/')
                           for path in consumed for keep in target.get('protected_paths', []))},
        {'name': 'native operation outcomes', 'passed': all(row.get('success', True) for row in operations)},
        {'name': 'authored animation preservation', 'passed': motion['passed'], 'evidence': motion},
    ]
    return {'passed': all(row['passed'] for row in checks),
            'method': 'native_execution_and_authored_composition',
            'independent_geometry_proof': False,
            'geometry_evidence': 'USD Optimize settings and successful execution; no independently measured surface deviation.',
            'checks': checks, 'errors': [],
            'evidence_fingerprint': {'files': dict(evidence_files), 'runtime': list(Usd.GetVersion()),
                'native_worker': digest(__file__),
                'geometry_tolerance_checker': digest(Path(__file__).resolve().parents[2]/'geometry_tolerance.py'),
                'policy': hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()}}


def movable_interface_paths(stage, target, protection_facts):
    """Static editable frames may move in namespace; addressed or animated prims may not."""
    from pxr import UsdGeom
    result = set()
    for edit in target.get('required_occurrence_edits', []):
        path = edit['path']
        prim = stage.GetPrimAtPath(path)
        if (path != target.get('boundary_path') and prim and prim.IsA(UsdGeom.Xform)
                and edit.get('edit_class', 'placement') in ('placement', 'visibility')
                and not (set(protection_facts.get(path, [])) - {'metadata:kind'})):
            result.add(path)
    return result


def hierarchy_scopes(stage, target, *, flatten=False, protection_facts=None):
    """Select native scopes without crossing retained or authored interfaces."""
    from pxr import Sdf, UsdGeom
    boundary = target.get('boundary_path')
    if target.get('structural_type') not in ('atomic', 'package') or not boundary:
        return []
    root = stage.GetPrimAtPath(boundary)
    if not root:
        raise ValueError('missing asset boundary: ' + boundary)
    if protection_facts is None:
        helper = load_helper('component_scope', 'component_scope.py')
        protection_facts = helper.merge_protection_facts(stage, target.get('internal_paths', []),
            boundary_paths=target.get('operation_scope', {}).get('exclude_paths', []))
    facts = {path: reasons for path, reasons in protection_facts.items()
             if path.startswith(boundary + '/') and stage.GetPrimAtPath(path)}
    movable = movable_interface_paths(stage, target, protection_facts) if flatten else set()
    protected = {Sdf.Path(p).GetPrimPath() for p in target.get('protected_paths', []) if p != boundary and p not in movable}
    for path, reasons in facts.items():
        if not path.startswith(boundary + '/'):
            continue
        # Native fitting authors static tessellation settings on Gprims. Native
        # flattening moves the entire prim with those settings; they do not make
        # its organizational ancestors independently addressable. Static purpose
        # also stays on the moved Gprim. Merge keeps
        # its stricter protection rules.
        if flatten and stage.GetPrimAtPath(path).IsA(UsdGeom.Gprim):
            reasons = set(reasons) - {'customAttribute:refinementLevel',
                                      'customAttribute:refinementEnableOverride',
                                      'attribute:purpose'}
        if reasons and path not in movable:
            protected.add(Sdf.Path(path))
    ancestors = set()
    for path in protected:
        ancestors.update(path.GetPrefixes()[:-1])
    scopes = []
    def select(prim):
        path = prim.GetPath()
        # A native flatten scope root can absorb its child transform. Keep an
        # editable frame below the scope root, or flatten only its interiors.
        if path in protected or str(path) in movable:
            if prim.IsA(UsdGeom.Gprim): return
            if any(reason.startswith('composition:') for reason in facts.get(str(path), [])):
                return
            for child in prim.GetChildren(): select(child)
            return
        # Native flattening may author a baked transform on its scope root.
        # An untyped prim or Scope cannot apply that transform.
        if path in ancestors or (flatten and not prim.IsA(UsdGeom.Xform)):
            for child in prim.GetChildren(): select(child)
        else: scopes.append(str(path))
    if flatten:
        for child in root.GetChildren(): select(child)
    else:
        select(root)
    return scopes


def flatten_asset_groups(core, stage, target, protection_facts=None):
    """Run native flattening and capture geometry paths without comparing arrays."""
    import uuid
    from pxr import Usd, UsdGeom
    from usd_optimize.core import ExecutionContext
    if protection_facts is None:
        protection_facts = load_helper('flatten_protection', 'component_scope.py').merge_protection_facts(
            stage, target.get('internal_paths', []),
            boundary_paths=target.get('operation_scope', {}).get('exclude_paths', []))
    movable = movable_interface_paths(stage, target, protection_facts)
    scopes = hierarchy_scopes(stage, target, flatten=True, protection_facts=protection_facts)
    if not scopes: return None, {}
    root = stage.GetPrimAtPath(target['boundary_path'])
    before = sum(p.IsA(UsdGeom.Xform) for p in Usd.PrimRange(root))
    key = 'usdTuningPath_' + uuid.uuid4().hex
    # Unique temporary metadata follows native namespace edits. Never export it.
    originals = {}
    for selected in scopes:
        for prim in Usd.PrimRange(stage.GetPrimAtPath(selected)):
            if prim.IsA(UsdGeom.Gprim) or str(prim.GetPath()) in movable:
                path = str(prim.GetPath())
                prim.SetCustomDataByKey(key, path)
                if path in movable:
                    # Native flattening preserves prims with relationships while
                    # still collapsing their redundant ancestors.
                    prim.CreateRelationship(key).SetTargets([target['boundary_path']])
                originals[path] = path
    op = {'operation': 'flattenHierarchy', 'paths': scopes, 'identity': False}
    context = ExecutionContext(); context.set_stage(stage)
    started = time.monotonic()
    try:
        outcomes = _execute_native(core, context, op)
        if len(outcomes) != 1 or not outcomes[0][0]:
            raise RuntimeError('flattenHierarchy failed: ' + str(outcomes))
        mapping = {}
        for prim in Usd.PrimRange(stage.GetPrimAtPath(target['boundary_path'])):
            old = prim.GetCustomDataByKey(key)
            if old: mapping[old] = str(prim.GetPath())
        if set(mapping) != set(originals):
            raise RuntimeError('flattenHierarchy did not retain geometry path records')
    finally:
        for prim in Usd.PrimRange(stage.GetPrimAtPath(target['boundary_path'])):
            if prim.HasCustomDataKey(key): prim.ClearCustomDataByKey(key)
            if prim.HasRelationship(key): prim.RemoveProperty(key)
    after = sum(p.IsA(UsdGeom.Xform) for p in Usd.PrimRange(stage.GetPrimAtPath(target['boundary_path'])))
    return {'operation': 'flattenHierarchy', 'success': True,
            'status': 'changed' if after < before else 'no_benefit',
            'reason': 'Native removal of intermediate static Xforms owned by this asset.',
            'settings': op, 'elapsed_seconds': time.monotonic() - started,
            'evidence': {'before_xforms': before, 'after_xforms': after, 'removed_xforms': before - after}}, {
                old: new for old, new in mapping.items() if old != new}


def prune_asset_groups(core, stage, target, protection_facts=None):
    """Prune empty organizational prims with the native operation after geometry work."""
    from pxr import Sdf, Usd
    from usd_optimize.core import ExecutionContext
    boundary = target.get('boundary_path')
    scopes = hierarchy_scopes(stage, target, protection_facts=protection_facts)
    root = stage.GetPrimAtPath(boundary) if boundary else None
    if not scopes:
        return None
    before = sum(1 for _ in Usd.PrimRange(root))
    op = {'operation': 'pruneLeaves', 'paths': scopes, 'pruneMode': 1,
          'preserveUnloadedPayloads': True}
    context = ExecutionContext(); context.set_stage(stage)
    started = time.monotonic()
    outcomes = _execute_native(core, context, op)
    if len(outcomes) != 1 or not outcomes[0][0]:
        raise RuntimeError('pruneLeaves failed: ' + str(outcomes))
    after = sum(1 for _ in Usd.PrimRange(stage.GetPrimAtPath(boundary)))
    return {'operation': 'pruneLeaves', 'success': True,
            'status': 'changed' if after < before else 'no_benefit',
            'reason': 'Native cleanup of empty organizational prims owned by this asset.',
            'settings': op, 'elapsed_seconds': time.monotonic() - started,
            'evidence': {'before_prims': before, 'after_prims': after, 'removed_prims': before - after}}


def run(job, summary_path):
    started = time.monotonic()
    result = {'definition_results': [], 'stage_opens': 0, 'validation_seconds': 0.0, 'status': 'failed'}
    pending = None
    source_hash = None
    try:
        from pxr import Sdf, Usd, UsdGeom, UsdUtils
        from usd_optimize.core import ExecutionContext, UsdOptimizeCore
        source = Path(job['source_layer']).resolve()
        output = Path(job['output_layer']).resolve()
        result.update(source=str(source), output=str(output))
        if source == output:
            raise ValueError('worker output must differ from its immutable input')
        source_hash = digest(source)
        original = Usd.Stage.Open(str(source)); result['stage_opens'] += 1
        if not original or original.GetCompositionErrors():
            raise ValueError('definition input has composition errors')
        layer = Sdf.Layer.CreateAnonymous('worker.usda')
        layer.TransferContent(original.GetRootLayer())
        UsdUtils.ModifyAssetPaths(layer, lambda asset: Sdf.ComputeAssetPathRelativeToLayer(original.GetRootLayer(), asset))
        stage = Usd.Stage.Open(layer); result['stage_opens'] += 1
        if not stage or stage.GetCompositionErrors():
            raise ValueError('copied definition has composition errors')
        core = UsdOptimizeCore.getInstance()
        verifier = load_helper('definition_verification', 'optimization-report/scripts/definition_verification.py')
        uv_cache = {}
        request = job.get('request', {})
        definitions = ordered_definitions(job['definitions'])
        owned = [p for target in definitions for p in geometry_paths(target)]
        if len(owned) != len(set(owned)):
            raise ValueError('definition operation scopes overlap')
        prepared = []
        requests = []
        categories_by_target = []
        for target in definitions:
            select_edit_context(stage, target)
            for path in geometry_paths(target):
                prim = stage.GetPrimAtPath(path)
                if not prim or prim.IsInstanceProxy() or not layer.GetPrimAtPath(stage.GetEditTarget().MapToSpecPath(path)):
                    raise ValueError('geometry must be authored in the editable layer: ' + path)
            local_request = dict(request)
            local_request['tolerance_mm'] = physical_tolerance(request)
            local_request['_explicit_fit_primitives'] = request.get('fit_primitives') is True
            scale = target.get('maximum_occurrence_scale', 1.0)
            if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
                raise ValueError('maximum_occurrence_scale must be finite and positive')
            for key in ('tolerance_mm', 'deviation_tolerance_mm', 'max_deviation_mm'):
                if key in local_request:
                    local_request[key] = physical_tolerance(request) / scale
            uv_proof = verifier.unused_uv_removal_check(stage, geometry_paths(target),
                time_samples=local_request.get('time_samples'), cache=uv_cache)
            local_request['_uv_removal_check'] = uv_proof
            if not uv_proof['allowed']:
                local_request['preserve_uvs'] = True
            requests.append(local_request)
            local_request, categories = category_selection(target, local_request, measure(stage, geometry_paths(target))['meshes'] > 0)
            requests[-1] = local_request
            categories_by_target.append(categories)
            placement_meshes = sorted(set(target.get('operation_scope', {}).get('mesh_paths', [])) &
                {e['path'].split('.')[0] for e in target.get('required_occurrence_edits', []) if e.get('edit_class', 'placement') == 'placement'})
            for category in categories:
                if category['category'] == 'primitive_fit' and placement_meshes:
                    category.setdefault('evidence', {})['placement_frame_exclusions'] = placement_meshes
            ops = operation_config(target, local_request, stage)
            validate_operations(core, target, ops, local_request)
            prepared.append(ops)
        # Snapshot only this writer's authored layer. Referenced child geometry
        # has its own verified writer; no parent repeats those triangle checks.
        authored_scopes = []
        for target in definitions:
            select_edit_context(stage, target)
            authored_scopes.extend(str(stage.GetEditTarget().MapToSpecPath(p)) for p in geometry_paths(target))
            authored_scopes.extend(str(stage.GetEditTarget().MapToSpecPath(g['operation_scope']['output_path'])) for g in merge_scopes(target))
        outside_before = outside_scope_digest(stage, authored_scopes)
        mappings = []
        for target, operations, local_request, categories in zip(definitions, prepared, requests, categories_by_target):
            select_edit_context(stage, target)
            paths = geometry_paths(target)
            accounting = {}
            before = operation_metrics(stage, paths, accounting)
            evidence = assess_operations(source, target, operations)
            result['stage_opens'] += evidence['stage_opens']
            result['validation_seconds'] += evidence['elapsed_seconds']
            context = ExecutionContext(); context.set_stage(stage)
            records = []; path_map = {}
            operations = list(operations)
            actionable, exclusions = actionable_findings(evidence, local_request)
            if any(issue['concept'] == 'primvar_unused' for issue in actionable) and 'operations' not in target:
                corrective = {'operation': 'removeUnusedUVs', 'paths': target['operation_scope']['mesh_paths'], 'mode': 0}
                validate_operations(core, target, [corrective], local_request)
                operations.insert(next((i for i, op in enumerate(operations) if op.get('operation') in ('fitPrimitives', 'decimateMeshes')), len(operations)), corrective)
            evidence['preservation_exclusions'] = exclusions
            operations.extend({'merge_scope': group} for group in merge_scopes(target))
            merge_protection = None
            for op in operations:
                if 'merge_scope' in op:
                    group = op['merge_scope']
                    remaining = [p for p in group['operation_scope']['mesh_paths']
                                 if stage.GetPrimAtPath(p) and stage.GetPrimAtPath(p).IsA(UsdGeom.Mesh)]
                    if len(remaining) < 2:
                        records.append({'operation': 'merge', 'status': 'not_applicable',
                            'reason': 'Fewer than two eligible Mesh prims remain.',
                            'evidence': {'remaining_mesh_paths': remaining}})
                        continue
                    group = dict(group, operation_scope=dict(group['operation_scope'], mesh_paths=remaining))
                    scope = load_helper('component_scope', 'component_scope.py')
                    if merge_protection is None:
                        # Geometry optimization has finished. Disjoint merges do
                        # not add motion, composition arcs or addressed interfaces;
                        # inspect those facts once, retaining exact kind exemptions.
                        merge_protection = scope.merge_protection_facts(
                            stage, target.get('internal_paths', []), retain_internal_facts=True,
                            boundary_paths=target.get('operation_scope', {}).get('exclude_paths', []))
                    op = scope.check_stage_scope(stage, group, protection_facts=merge_protection)
                    validate_operations(core, target, [op], local_request)
                if op.get('operation') == 'merge':
                    remaining = [p for p in op['meshPrimPaths'] if stage.GetPrimAtPath(p)
                                 and stage.GetPrimAtPath(p).IsA(UsdGeom.Mesh)
                                 and not stage.GetPrimAtPath(p).IsInstanceProxy()]
                    if len(remaining) < 2:
                        records.append({'operation': 'merge', 'status': 'not_applicable',
                                        'reason': 'Fewer than two eligible Mesh prims remain.',
                                        'evidence': {'remaining_mesh_paths': remaining}})
                        continue
                    op = dict(op, meshPrimPaths=remaining)
                if 'paths' in op:
                    op = dict(op, paths=[p for p in op['paths'] if stage.GetPrimAtPath(p)])
                    if not op['paths']:
                        continue
                # Native settings and outcomes define runtime geometry execution.
                # Independent surface qualification remains a developer tool.
                records.append(apply_operation(core, context, stage, op,
                    resolved_geometry_paths(paths, path_map, records), path_map, 1, accounting))
            if not records:
                records.append({'operation': 'geometry applicability', 'status': 'not_applicable',
                                'reason': ('No native operation was supplied for the owned Mesh geometry; category coverage remains incomplete.'
                                           if before['meshes'] else 'This definition owns no Mesh geometry; referenced child definitions have separate results.'),
                                'evidence': {'geometry_counts': before, 'child_definitions': target.get('children', [])}})
            scans, remaining = reassess_and_continue(core, context, stage, target, local_request,
                                                     operations, records, path_map, evidence, accounting)
            for scan in scans[1:]:
                result['stage_opens'] += scan['stage_opens']
                result['validation_seconds'] += scan['elapsed_seconds']
            decisions = finish_categories(categories, records, remaining)
            for category in categories:
                if category['category'] == 'cleanup':
                    category['evidence']['preservation_exclusions'] = [item for scan in scans for item in scan.get('preservation_exclusions', [])]
            result.setdefault('decisions_required', []).extend(dict(item, definition_id=target['id']) for item in decisions)
            after_paths = resolved_geometry_paths(paths, path_map, records)
            result['definition_results'].append({'definition_id': target['id'], 'geometry_representation': target.get('geometry_representation', 'merged_meshes'), 'before': before,
                'after': operation_metrics(stage, after_paths, accounting), 'operations': records, 'path_map': dict(path_map),
                'consumed_geometry_paths': consumed_geometry_paths(records),
                'assessment': {'complete': False, 'operations': records, 'validation': evidence,
                               'categories': categories, 'passes': scans, 'decisions_required': decisions,
                               'uv_removal': local_request.get('_uv_removal_check')},
                'disposition': 'incomplete', 'verification': {'passed': False},
                'verification_context': {'request_sha256': hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest(),
                                         'target_sha256': hashlib.sha256(json.dumps(target, sort_keys=True).encode()).hexdigest()}})
            # Native compatibility buckets may create additional sibling meshes.
            # Account only recorded outputs absent from the immutable input; an
            # existing unrelated prim must remain covered by the outside check.
            for record in records:
                for path in record.get('evidence', {}).get('actual_output_paths', []):
                    spec_path = stage.GetEditTarget().MapToSpecPath(path)
                    if not original.GetRootLayer().GetPrimAtPath(spec_path):
                        authored_scopes.append(str(spec_path))
            mappings.append(path_map)
        # Geometry writes must stay scoped. Native structural cleanup then owns
        # eligible interiors inside each asset boundary, not just its mesh paths.
        unowned_unchanged = outside_scope_digest(stage, authored_scopes) == outside_before
        for target, record, record_mapping in zip(definitions, result['definition_results'], mappings):
            select_edit_context(stage, target)
            # Merging leaves empty siblings that make otherwise linear chains
            # look branched to native flattening. Remove them before flattening.
            scope_started = time.monotonic()
            helper = load_helper('hierarchy_protection', 'component_scope.py')
            # Capture incoming targets across the stage once. Pruning only deletes
            # unprotected empty prims; flattening reports moved geometry paths below.
            protection_facts = helper.merge_protection_facts(stage, target.get('internal_paths', []),
                boundary_paths=target.get('operation_scope', {}).get('exclude_paths', []))
            scope_seconds = time.monotonic() - scope_started
            print(f'Hierarchy protection inspection ({scope_seconds:.3f}s)', flush=True)
            cleanup = prune_asset_groups(core, stage, target, protection_facts)
            if cleanup:
                record['operations'].append(cleanup)
            flattened, moved = flatten_asset_groups(core, stage, target, protection_facts)
            record['hierarchy_protection_seconds'] = scope_seconds
            if flattened:
                record['operations'].append(flattened)
                mapping = record_mapping
                for path in set(geometry_paths(target)) | set(resolved_geometry_paths(geometry_paths(target), mapping, record['operations'])):
                    current = mapping.get(path, path)
                    if current in moved: mapping[path] = moved[current]
                mapping.update(moved)
                record['path_map'] = dict(mapping)
                final_paths = resolved_geometry_paths(geometry_paths(target), mapping, record['operations'])
                record['after'] = measure(stage, final_paths)
            protection_facts = {moved.get(path, path): reasons for path, reasons in protection_facts.items()}
            cleanup_target = dict(target, protected_paths=[moved.get(p, p) for p in target.get('protected_paths', [])])
            cleanup = prune_asset_groups(core, stage, cleanup_target, protection_facts)
            if cleanup:
                record['operations'].append(cleanup)
        output.parent.mkdir(parents=True, exist_ok=True)
        handle, filename = tempfile.mkstemp(prefix=output.stem+'.pending-', suffix='.usdc', dir=output.parent)
        os.close(handle); pending = Path(filename)
        if not layer.Export(str(pending)):
            raise RuntimeError('output layer export failed')
        written = Usd.Stage.Open(str(pending)); result['stage_opens'] += 1
        if not written:
            raise RuntimeError('cannot open written native output')
        evidence_files = verifier._evidence_files(original, written)
        animation = load_helper('native_execution_animation', '../scripts/animation_context.py')
        for target, record, mapping, local_request in zip(definitions, result['definition_results'], mappings, requests):
            policy = verification_policy(local_request, record['operations'])
            policy['path_map'] = mapping
            if any(r.get('operation') in ('fitPrimitives', 'decimateMeshes') for r in record['operations']):
                policy['tolerance_mm'] = physical_tolerance(local_request)
            policy['allow_reduction'] = any(r.get('operation') == 'decimateMeshes' for r in record['operations'])
            began = time.monotonic()
            record['verification'] = native_execution_verification(original, written, target, policy,
                record['operations'], evidence_files, animation)
            record['verification_seconds'] = time.monotonic() - began
            result['validation_seconds'] += record['verification_seconds']
            record['assessment']['complete'] = record['verification']['passed'] and not record['assessment']['decisions_required']
            if record['assessment']['complete']:
                changed = any(r['status'] == 'changed' for r in record['operations'])
                record['disposition'] = 'optimized' if changed else 'verified_unchanged'
        result['unowned_verification'] = {'passed': unowned_unchanged,
            'method': 'authored opinions outside geometry before scoped native empty-group cleanup; child geometry verified by its writer'}
        result['source_unchanged'] = digest(source) == source_hash
        if result['source_unchanged'] and all(r['verification']['passed'] for r in result['definition_results']) and result.get('unowned_verification', {}).get('passed', True):
            pending_path = str(pending.resolve())
            verified_hash = digest(pending)
            os.replace(pending, output); pending = None
            result['output_sha256'] = digest(output)
            if result['output_sha256'] != verified_hash:
                raise RuntimeError('output content changed during publication')
            for record in result['definition_results']:
                files = record['verification'].get('evidence_fingerprint', {}).get('files', {})
                if pending_path in files:
                    files[str(output)] = files.pop(pending_path)
            files = result.get('unowned_verification', {}).get('evidence_fingerprint', {}).get('files', {})
            if pending_path in files:
                files[str(output)] = files.pop(pending_path)
            result['status'] = 'done' if all(r['assessment']['complete'] for r in result['definition_results']) else 'incomplete'
            if result['status'] == 'incomplete':
                result['error'] = 'Unresolved optimization categories; inspect decisions_required.'

        else:
            result['error'] = 'written output failed execution or composition checks; existing output retained'
    except Exception as error:
        result['error'] = str(error)
        result['traceback'] = traceback.format_exc()
    finally:
        if pending is not None:
            if pending.is_file():
                handle, filename = tempfile.mkstemp(prefix=output.stem+'.rejected-', suffix='.usdc', dir=output.parent)
                os.close(handle)
                rejected = Path(filename)
                old_path = str(pending.resolve())
                os.replace(pending,rejected)
                result['rejected_output_layer'] = str(rejected)
                result['rejected_output_sha256'] = digest(rejected)
                result['rejected_output_publishable'] = False
                for record in result.get('definition_results',[]):
                    files = record.get('verification',{}).get('evidence_fingerprint',{}).get('files',{})
                    if old_path in files:
                        files[str(rejected)] = files.pop(old_path)
        if source_hash is not None:
            result['source_unchanged'] = digest(source) == source_hash
    result['seconds'] = time.monotonic()-started
    result['peak_rss_kb'] = peak_rss_kb()
    summary = Path(summary_path); summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def run_batch(payload):
    """Reuse completed member receipts after an interrupted bounded batch."""
    started = time.monotonic()
    resumed = []
    scheduler = load_helper('native_member_resume', 'usd-optimize-run-operations/scripts/run_batch.py')
    cache = scheduler._FingerprintCache()
    results = []
    for item in payload['jobs']:
        spec = item['target']
        checkpoint = Path(item['summary_path']).with_suffix('.resume.json')
        fingerprint = scheduler._resume_fingerprint(spec, False, cache)
        previous = {}
        if payload.get('resume') and checkpoint.is_file():
            try:
                previous = json.loads(checkpoint.read_text())
            except (OSError, ValueError):
                pass
        outputs = scheduler._output_fingerprint(spec, cache)
        _, errors = scheduler._definition_results(item['summary_path'], spec['definition_ids'])
        if (fingerprint is not None and outputs is not None and not errors
                and previous.get('resume_fingerprint') == fingerprint
                and previous.get('output_fingerprint') == outputs):
            results.append(json.loads(Path(item['summary_path']).read_text()))
            resumed.extend(spec['definition_ids'])
            continue
        result = run(item['job'], item['summary_path'])
        results.append(result)
        _, errors = scheduler._definition_results(item['summary_path'], spec['definition_ids'])
        outputs = scheduler._output_fingerprint(spec, cache)
        if result['status'] == 'done' and result.get('source_unchanged') and not errors and fingerprint and outputs:
            pending = checkpoint.with_suffix('.pending')
            pending.write_text(json.dumps({'resume_fingerprint': fingerprint, 'output_fingerprint': outputs}))
            os.replace(pending, checkpoint)
        if result['status'] != 'done':
            break
    return {'status': 'done' if len(results) == len(payload['jobs']) and all(row['status'] == 'done' for row in results) else 'incomplete',
            'definition_results': [record for row in results for record in row['definition_results']],
            'seconds': time.monotonic() - started, 'peak_rss_kb': peak_rss_kb(),
            'resumed_definition_ids': resumed,
            'source_unchanged': all(row.get('source_unchanged') for row in results)}


def main(argv=None):
    global _OPERATION_PROGRESS_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True)
    parser.add_argument('--summary', required=True)
    args = parser.parse_args(argv)
    _OPERATION_PROGRESS_PATH = Path(args.summary).with_suffix('.progress.json')
    payload = json.loads(Path(args.job).read_text())
    if 'jobs' in payload:
        result = run_batch(payload)
        Path(args.summary).write_text(json.dumps(result, indent=2, allow_nan=False))
    else:
        result = run(payload, args.summary)
    print(json.dumps({'status': result['status'], 'summary': args.summary,
                      'definitions': len(result['definition_results'])}))
    return 0 if result['status'] == 'done' else 1


if __name__ == '__main__':
    raise SystemExit(main())
