#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scoped native mesh splitting with one-to-many output accounting.

This adapter separates existing faces only. It does not assign semantic ownership.
Animated meshes and interfaces with relationships require a separate reconstruction
path and are rejected before dispatch, rather than silently baked or disconnected.
"""
from __future__ import annotations

import hashlib
import uuid
import json
from pathlib import Path
import time


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def retain_failure_layers(output_path, initial, rejected):
    """Retain raw layers beside their dependencies; never flatten for diagnostics."""
    prefix = Path(output_path).with_name(Path(output_path).stem + '.rejected-' + uuid.uuid4().hex[:12])
    result = {}
    for label, layer in [('before', initial), ('rejected', rejected)]:
        path = Path(str(prefix) + '.' + label + '.usdc')
        if not layer.Export(str(path)):
            raise RuntimeError('cannot retain native failure layer: ' + str(path))
        result[label + '.usdc'] = {'path': str(path), 'sha256': _digest(path)}
    return result


def split(source_layer, output_layer, paths, *, split_on='vertices'):
    """Copy the selected composition, run native splitting, and check output resolution.

    Native USD Optimize owns geometric correctness; no runtime surface comparison.

    Status is split, unchanged, unsupported, execution_failure or
    verification_failure. A split must create at least two accounted mesh outputs;
    successful execution alone is insufficient. Never overwrite an existing file.
    """
    from pxr import Sdf, Usd, UsdGeom
    from usd_optimize.core import ExecutionContext, UsdOptimizeCore
    source_layer, output_layer = Path(source_layer).resolve(), Path(output_layer).resolve()
    if source_layer == output_layer or output_layer.exists():
        raise ValueError('output must be a new path distinct from the source')
    if split_on not in ('vertices', 'geom_subsets'):
        raise ValueError('split_on must be vertices or geom_subsets')
    paths = list(dict.fromkeys(paths))
    if not paths or any(not Sdf.Path(path).IsAbsolutePath() or not Sdf.Path(path).IsPrimPath() for path in paths):
        raise ValueError('explicit absolute mesh prim paths are required')
    source = Usd.Stage.Open(str(source_layer))
    if not source or source.GetCompositionErrors():
        raise ValueError('source composition is invalid')
    started = time.monotonic()
    result = {'schema_version': 1, 'source_layer': str(source_layer), 'output_layer': str(output_layer), 'status': 'unsupported', 'operations': [], 'path_map': {}, 'verification': {}, 'verification_basis': 'native_execution_and_composition'}
    incoming_targets = []
    for prim in source.Traverse():
        for prop in prim.GetProperties():
            incoming_targets.extend(prop.GetTargets() if isinstance(prop, Usd.Relationship) else prop.GetConnections())
    for path in paths:
        prim = source.GetPrimAtPath(path)
        if not prim or not prim.IsA(UsdGeom.Mesh) or prim.IsInstanceProxy():
            raise ValueError('scope requires editable mesh prims: ' + path)
        if UsdGeom.Mesh(prim).GetSubdivisionSchemeAttr().Get() != 'none':
            result['reason'] = 'subdivision surface reconstruction requires a limit-surface preservation check'
            return result
        # The native operation does not promise channel/relationship remapping.
        subtree = list(Usd.PrimRange(prim))
        if any(a.GetNumTimeSamples() or a.GetConnections() for p in subtree for a in p.GetAttributes()) or any(p.IsA(UsdGeom.Mesh) for p in subtree[1:]):
            result['reason'] = 'animated mesh data or nested mesh interfaces require preserved reconstruction'
            return result
        if any(r.GetTargets() for p in subtree for r in p.GetRelationships() if not r.GetName().startswith('material:binding')):
            result['reason'] = 'mesh relationships require an explicit remapping decision'
            return result
        if any(target.HasPrefix(prim.GetPath()) for target in incoming_targets):
            result['reason'] = 'incoming relationships or connections require explicit remapping'
            return result
    output_layer.parent.mkdir(parents=True, exist_ok=True)
    source.Flatten().Export(str(output_layer))
    stage = Usd.Stage.Open(str(output_layer))
    source_hash = _digest(source_layer)
    result['input_sha256'] = source_hash
    initial = Sdf.Layer.CreateAnonymous(); initial.TransferContent(stage.GetRootLayer())
    context = ExecutionContext(); context.set_stage(stage)
    core = UsdOptimizeCore.getInstance()
    for path in paths:
        before_paths = {str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)}
        op = {'operation': 'splitMeshes', 'paths': [path], 'splitOn': 0 if split_on == 'vertices' else 1,
              'originalGeomOption': 1, 'mergePoint': 9, 'spatialMode': 0, 'considerAllAttributes': True}
        row = {'configuration': op, 'source_paths': [path]}
        result['operations'].append(row)
        try:
            outcomes = core.executeConfig(context, [op])
            row['native_result'] = str(outcomes)
            if len(outcomes) != 1 or not outcomes[0][0]:
                result['status'] = 'execution_failure'; row['status'] = 'failed'; break
            after_paths = {str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)}
            outputs = sorted((after_paths - before_paths) | ({path} & after_paths))
            result['path_map'][path] = outputs
            row['output_paths'] = outputs
            row['output_paths_resolve'] = bool(outputs) and all(stage.GetPrimAtPath(p).IsA(UsdGeom.Mesh) for p in outputs)
            row['status'] = 'split' if len(outputs) > 1 else 'unchanged'
            if not row['output_paths_resolve'] or stage.GetCompositionErrors():
                result['status'] = 'verification_failure'; break
        except Exception as error:
            row['error'] = str(error); result['status'] = 'execution_failure'; break
    else:
        result['status'] = 'split' if any(row['status'] == 'split' for row in result['operations']) else 'unchanged'
    result['verification'] = {'source_unchanged': _digest(source_layer) == source_hash,
                              'composition': not bool(stage.GetCompositionErrors()),
                              'output_paths_resolve': all(row.get('output_paths_resolve', False) for row in result['operations'])}
    if not all(result['verification'].values()) and result['status'] != 'execution_failure':
        result['status'] = 'verification_failure'
    if result['status'] in ('split', 'unchanged'):
        stage.GetRootLayer().Save()
    else:
        try:
            result['failure_artifacts'] = retain_failure_layers(output_layer, initial, stage.GetRootLayer())
        finally:
            stage.GetRootLayer().TransferContent(initial)
            stage.GetRootLayer().Save()
            result['path_map'] = {path: [path] for path in paths}
    result['elapsed_seconds'] = time.monotonic() - started
    result['output_sha256'] = _digest(output_layer)
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_layer'); parser.add_argument('output_layer')
    parser.add_argument('--path', action='append', required=True)
    parser.add_argument('--split-on', choices=('vertices', 'geom_subsets'), default='vertices')
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    result = split(args.source_layer, args.output_layer, args.path, split_on=args.split_on)
    Path(args.report).write_text(json.dumps(result, indent=2) + '\n')
    return 0 if result['status'] in ('split', 'unchanged') else 1


if __name__ == '__main__':
    raise SystemExit(main())
