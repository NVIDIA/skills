#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Comparable fresh-process USD open/traversal measurements.

The first open is not guaranteed OS-cold. Renderer, RAM and GPU metrics are not
measured. The parent process retains the original baseline JSON for later runs.

Usage:
    python stage_profile.py STAGE [--output JSON] [--required-artifact PATH]
Arguments:
    Stage path; optional new output JSON path; repeat --required-artifact for
    delivery files. --sample selects one child measurement; --inventory adds
    inventory to that sample. Normal profiling uses fresh child processes.
Output:
    Prints profile JSON and optionally creates the output file without overwriting.
    Records open/traversal timings and delivery metrics, not render performance.
Exit codes:
    0 on success, 1 for an uncaught profiling/file error, 2 for invalid CLI arguments.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import os
import platform
import subprocess
import sys
import time
import tempfile


def sample(path, inventory=False):
    from pxr import Sdf, Usd, UsdGeom
    started = time.perf_counter()
    stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadAll)
    elapsed = (time.perf_counter() - started) * 1000
    if not stage or stage.GetCompositionErrors():
        raise ValueError('stage could not be fully composed')
    started = time.perf_counter()
    for prim in stage.Traverse():
        pass
    traversal = (time.perf_counter() - started) * 1000
    result = {'open_ms': elapsed, 'traversal_ms': traversal, 'usd_version': list(Usd.GetVersion()),
              'process_id': os.getpid(), 'python_version': platform.python_version(),
              'platform': platform.platform(), 'traversal_scope': 'ordinary composed prims; instance roots counted once'}
    if not inventory:
        return result
    geometry = Counter()
    selections, resources, unresolved = {}, set(), set()
    regular_count = sum(1 for _ in stage.Traverse())
    occurrence_count = 0
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        occurrence_count += 1
        if prim.IsA(UsdGeom.Gprim) or prim.IsA(UsdGeom.PointInstancer):
            geometry[prim.GetTypeName()] += 1
        variants = prim.GetVariantSets()
        if variants.GetNames():
            selections[str(prim.GetPath())] = {name: variants.GetVariantSelection(name) for name in variants.GetNames()}
        for attr in prim.GetAttributes():
            if attr.GetTypeName() not in (Sdf.ValueTypeNames.Asset, Sdf.ValueTypeNames.AssetArray):
                continue
            for at in [Usd.TimeCode.Default()] + [Usd.TimeCode(t) for t in attr.GetTimeSamples()]:
                value = attr.Get(at)
                values = value if attr.GetTypeName() == Sdf.ValueTypeNames.AssetArray else [value]
                for asset in values or []:
                    if asset and asset.path:
                        if asset.resolvedPath and Path(asset.resolvedPath).is_file():
                            resources.add(str(Path(asset.resolvedPath).resolve()))
                        else:
                            unresolved.add(asset.path)
    layers = {str(Path(layer.realPath).resolve()) for layer in stage.GetUsedLayers() if layer.realPath}
    # USD asset attributes reveal the module, not its imported modules or textures.
    # Counting that file alone cannot establish the complete referenced footprint.
    opaque_mdl_resources = sorted(p for p in resources if Path(p).suffix.lower() == '.mdl')
    files = layers | resources
    dependencies = {}
    for filename in sorted(files):
        with open(filename, 'rb') as stream:
            dependencies[filename] = {'bytes': Path(filename).stat().st_size,
                                      'sha256': hashlib.file_digest(stream, 'sha256').hexdigest()}
    result.update(ordinary_prim_count=regular_count, occurrence_prim_count=occurrence_count,
                  geometry_occurrences=dict(geometry), dependencies=dependencies,
                  referenced_storage_bytes=sum(row['bytes'] for row in dependencies.values()),
                  used_layer_paths=sorted(layers),
                  storage_scope='composed used layers and resolved authored asset attributes',
                  unresolved_resources=sorted(unresolved), selected_variants=selections,
                  inventory_complete=not unresolved and not opaque_mdl_resources,
                  unqualified_resource_dependencies=opaque_mdl_resources,
                  resource_closure_note=('MDL transitive dependencies require an MDL-aware collector; only USD-authored asset paths were inspected.' if opaque_mdl_resources else ''),
                  inventory_scope='selected variants, loaded payloads and hidden geometry; inactive prims and alternate variants excluded',
                  geometry_count_scope='composed geometry prim occurrences including instance proxies; point instancer instances are not expanded',
                  metersPerUnit=UsdGeom.GetStageMetersPerUnit(stage), upAxis=str(UsdGeom.GetStageUpAxis(stage)),
                  startTimeCode=stage.GetStartTimeCode(), endTimeCode=stage.GetEndTimeCode(),
                  timeCodesPerSecond=stage.GetTimeCodesPerSecond(), load_policy='all')
    return result


def repack_storage(inventory):
    """Measure lossless per-layer crate encoding without flattening composition.

    Each used file is counted once. Referenced resources retain their raw bytes;
    USD layers are exported independently with authored arcs/specs unchanged.
    Temporary exports are evidence measurements, never replacements for input.
    """
    from pxr import Sdf
    files=inventory['dependencies'];layers=set(inventory.get('used_layer_paths',[]))
    if not layers:
        return {'status':'partial','reason':'Used-layer inventory is unavailable; normalization was not inferred from file extensions.'}
    def fingerprint(path):
        with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
    changed=[path for path,row in files.items() if not Path(path).is_file() or fingerprint(path)!=row['sha256']]
    if changed:raise ValueError('source dependencies changed before storage normalization: '+str(changed))
    records=[];errors=[]
    with tempfile.TemporaryDirectory(prefix='usd-repack-baseline-') as directory:
        for index,path in enumerate(sorted(layers)):
            try:
                layer=Sdf.Layer.OpenAsAnonymous(path)
                if not layer:raise ValueError('layer could not be opened')
                output=Path(directory)/(str(index)+'.usdc')
                if not layer.Export(str(output)):raise ValueError('crate export failed')
                written=Sdf.Layer.FindOrOpen(str(output))
                if not written:
                    raise ValueError('crate export could not be reopened')
                records.append({'path':path,'raw_bytes':files[path]['bytes'],'repacked_bytes':output.stat().st_size,
                                'exported_without_flattening':True,'preservation_basis':'Sdf.Layer.Export',
                                'independent_content_comparison':False,'source_sha256':files[path]['sha256']})
            except Exception as exc:
                errors.append({'path':path,'error':str(exc)})
    if any(fingerprint(path)!=row['sha256'] for path,row in files.items()):
        raise ValueError('source dependencies changed during storage normalization')
    resource_bytes=sum(row['bytes'] for path,row in files.items() if path not in layers)
    complete=not errors and inventory.get('inventory_complete',False)
    return {'status':'complete' if complete else 'partial','raw_input_bytes':inventory['referenced_storage_bytes'],
            'repack_normalized_baseline_bytes':resource_bytes+sum(row['repacked_bytes'] for row in records) if not errors else None,
            'resource_bytes':resource_bytes,'layer_measurements':records,'errors':errors,
            'method':'Independent lossless USDC re-encoding of each composed used layer; no flattening, geometry changes or cross-layer deduplication.',
            'scope':inventory.get('storage_scope'),'source_unchanged':True}


def _include_required_artifacts(inventory, paths):
    """Count explicitly required delivery files once with USD dependencies."""
    artifacts = {}
    for path in sorted({str(Path(value).resolve()) for value in paths}):
        with Path(path).open('rb') as stream:
            artifacts[path] = {'bytes': Path(path).stat().st_size,
                               'sha256': hashlib.file_digest(stream, 'sha256').hexdigest()}
    additional = sum(row['bytes'] for path, row in artifacts.items()
                     if path not in inventory['dependencies'])
    scene_bytes = inventory.get('scene_storage_bytes', inventory['referenced_storage_bytes'])
    inventory['dependencies'].update(artifacts)
    inventory['required_artifacts'] = artifacts
    inventory['additional_required_artifact_bytes'] = additional
    inventory['scene_storage_bytes'] = scene_bytes
    inventory['reuse_overhead_bytes'] = additional
    inventory['total_delivery_bytes'] = scene_bytes + additional
    inventory['referenced_storage_bytes'] = sum(row['bytes'] for row in inventory['dependencies'].values())
    if artifacts:
        inventory['storage_scope'] += '; explicitly required delivery artifacts'
    return inventory


def profile(path, *, warm_samples=5, timeout_seconds=300, required_artifacts=()):
    if type(warm_samples) is not int or warm_samples < 5:
        raise ValueError('at least five warm samples are required')
    path = str(Path(path).resolve())
    def run(inventory=False):
        args = [sys.executable, str(Path(__file__).resolve()), path, '--sample']
        if inventory:
            args.append('--inventory')
        completed = subprocess.run(args, check=True, text=True, capture_output=True, timeout=timeout_seconds)
        # Native runtime messages may precede the one-line JSON result.
        result = json.loads(completed.stdout.splitlines()[-1])
        return _include_required_artifacts(result, required_artifacts) if inventory else result
    first = run(inventory=True)
    run()  # unreported warmup; each open has its own USD process/cache
    warm = [run() for _ in range(warm_samples)]
    measured = run(inventory=True)
    for key in ('dependencies', 'selected_variants', 'metersPerUnit', 'upAxis',
                'startTimeCode', 'endTimeCode', 'timeCodesPerSecond', 'load_policy',
                'usd_version', 'python_version', 'platform'):
        if first[key] != measured[key]:
            raise ValueError('source or profiling context changed during measurement: ' + key)
    opens = [value['open_ms'] for value in warm]
    median = statistics.median(opens)
    spread = max(opens) - min(opens)
    normalization=repack_storage(measured)
    return {'schema_version': 1, 'stage': path, 'first_open_ms': first['open_ms'],
            'warm_open_ms': median, 'warm_open_samples_ms': opens,
            'warm_open_sample_count': warm_samples,
            'warm_open_spread_pct': spread / median * 100 if median else 0,
            'warm_open_spread_ms': spread,
            'timing_conclusive': bool(median and spread / median <= .15),
            'traversal_ms': statistics.median(value['traversal_ms'] for value in warm),
            'traversal_samples_ms': [value['traversal_ms'] for value in warm],
            'sample_process_ids': [first['process_id']] + [value['process_id'] for value in warm] + [measured['process_id']],
            'open_timing_context': 'fresh process per open; first sample not guaranteed OS-cold; one warmup',
            'inventory': measured, 'storage_normalization':normalization, 'renderer_metrics': None,
            'status': 'complete' if measured['inventory_complete'] else 'partial'}


def compare_profiles(before, after):
    """Summarize observed deltas without attributing repacking to geometry work.

    Timing differences no larger than either run's sample range are neutral.
    This does not replace USD preservation verification or the report schema.
    """
    left, right = before['inventory'], after['inventory']
    mismatches = [key for key in ('usd_version', 'python_version', 'platform', 'load_policy',
                                  'metersPerUnit', 'upAxis', 'startTimeCode', 'endTimeCode', 'timeCodesPerSecond')
                  if left.get(key) != right.get(key)]
    def timing(key, samples):
        baseline, result = before[key], after[key]
        delta = result - baseline
        before_spread = max(before[samples]) - min(before[samples])
        after_spread = max(after[samples]) - min(after[samples])
        noise = max(before_spread, after_spread)
        excessive_spread = (baseline <= 0 or result <= 0 or before_spread / baseline > .15 or after_spread / result > .15)
        direction = 'incomparable' if mismatches else ('neutral' if excessive_spread or abs(delta) <= max(noise,abs(baseline)*.05) else 'decreased' if delta < 0 else 'increased')
        return {'before_ms': baseline, 'after_ms': result, 'delta_ms': delta,
                'delta_pct': delta / baseline * 100 if baseline else None,
                'noise_range_ms': noise, 'neutral_band_pct':5, 'excessive_sample_spread': excessive_spread, 'interpretation': direction}
    left_files, right_files = left['dependencies'], right['dependencies']
    complete = left.get('inventory_complete', False) and right.get('inventory_complete', False)
    return {'schema_version': 1, 'timing_context_mismatches': mismatches,
            'warm_open': timing('warm_open_ms', 'warm_open_samples_ms'),
            'traversal': timing('traversal_ms', 'traversal_samples_ms'),
            'storage': {'before_bytes': left['referenced_storage_bytes'], 'after_bytes': right['referenced_storage_bytes'],
                        'delta_bytes': right['referenced_storage_bytes'] - left['referenced_storage_bytes'],
                        'scene_before_bytes': left.get('scene_storage_bytes', left['referenced_storage_bytes']),
                        'scene_after_bytes': right.get('scene_storage_bytes', right['referenced_storage_bytes']),
                        'reuse_overhead_before_bytes': left.get('reuse_overhead_bytes', 0),
                        'reuse_overhead_after_bytes': right.get('reuse_overhead_bytes', 0),
                        'coverage_complete': complete,
                        'interpretation': 'combined geometry, composition, packing and encoding effects' if complete else 'partial resolved dependencies only',
                        'layer_file_count_before': sum(Path(p).suffix.lower() in ('.usd', '.usda', '.usdc') for p in left_files),
                        'layer_file_count_after': sum(Path(p).suffix.lower() in ('.usd', '.usda', '.usdc') for p in right_files)},
            'unmeasured': ['RAM', 'GPU memory', 'render frame rate']}


def write_profile(path, result):
    """Create a baseline once; never overwrite a source or earlier measurement."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage')
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--inventory', action='store_true')
    parser.add_argument('--output')
    parser.add_argument('--required-artifact', action='append', default=[])
    args = parser.parse_args()
    result = sample(args.stage, args.inventory) if args.sample else profile(args.stage, required_artifacts=args.required_artifact)
    text = json.dumps(result, allow_nan=False)
    if args.output:
        write_profile(args.output, result)
    print(text)
