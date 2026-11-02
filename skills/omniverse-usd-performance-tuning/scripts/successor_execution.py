#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compute isolated definitions using the existing native worker and scheduler.

Category publication and composed-consumer verification are separate phases.
Reference children contribute immutable inputs, not artificial compute barriers.

Usage:
    python successor_execution.py MANIFEST DIRECTORY [--max-workers N] [--resume]
    The full customer workflow invokes compute_definitions() from successor_workflow.
Arguments:
    Assessed/authored manifest JSON, output directory, worker limit (default 4),
    and optional resume of unchanged inputs, runtime, parameters and outputs.
Output:
    Writes the batch plan, worker status and compute-results.json in DIRECTORY.
    Publication and composed-consumer verification remain separate workflow steps.
Exit codes:
    0 for passed computation; 1 for incomplete computation or an uncaught error;
    2 for invalid command-line arguments.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import hashlib
import os
import tempfile
from pathlib import Path
import time


def _tuning():
    spec = importlib.util.spec_from_file_location('successor_tuning_helpers', Path(__file__).with_name('usd_tuning.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def geometry_paths(definition):
    target = definition.get('worker_target', definition)
    return target.get('operation_scope', {}).get('geometry_paths', target.get('owned_geometry', []))


def _assembly_result(definition, request, dependencies, directory, tuning, runner, resume, fingerprint_cache):
    """Publish an unchanged authored assembly without starting USD Optimize."""
    from pxr import Sdf, Usd, UsdUtils
    source = Path(definition['input_layer']).resolve()
    output = Path(definition['output_layer']).resolve()
    target = definition.get('worker_target', definition)
    context = {'request_sha256': hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest(),
               'target_sha256': hashlib.sha256(json.dumps(target, sort_keys=True).encode()).hexdigest()}
    receipt = directory/'assemblies'/(hashlib.sha256(definition['id'].encode()).hexdigest()+'.json')
    files = sorted(set(dependencies) | {str(source), str(output), str(Path(__file__).resolve())})
    def fingerprints():
        values = runner._file_hashes(files, fingerprint_cache)
        return dict(values) if values is not None else None
    if resume and receipt.is_file():
        previous = json.loads(receipt.read_text())
        expected = previous.get('verification', {}).get('evidence_fingerprint', {}).get('files')
        if (previous.get('verification_context') == context and expected
                and previous.get('assessment', {}).get('complete') and expected == fingerprints()):
            return previous
    layer = Sdf.Layer.FindOrOpen(str(source))
    copied = Sdf.Layer.CreateAnonymous('assembly.usda')
    copied.TransferContent(layer)
    UsdUtils.ModifyAssetPaths(copied, lambda asset: Sdf.ComputeAssetPathRelativeToLayer(layer, asset))
    output.parent.mkdir(parents=True, exist_ok=True)
    handle, filename = tempfile.mkstemp(prefix=output.stem+'.pending-', suffix='.usdc', dir=output.parent)
    os.close(handle)
    pending = Path(filename)
    try:
        if not copied.Export(str(pending)):
            raise ValueError('assembly export failed: '+definition['id'])
        written = Usd.Stage.Open(str(pending))
        if not written or written.GetCompositionErrors():
            raise ValueError('authored assembly output does not compose: '+definition['id'])
        os.replace(pending, output)
    finally:
        pending.unlink(missing_ok=True)
    empty = {'meshes': 0, 'vertices': 0, 'faces': 0, 'analytic_primitives': 0, 'array_bytes': 0}
    worker = tuning.helper('assembly_category_assessment', 'usd-optimize-run-operations/scripts/native_worker.py')
    _, categories = worker.category_selection(target, request, False)
    applicability = {'operation': 'geometry applicability', 'status': 'not_applicable',
                     'reason': 'Definition owns no geometry; retained child assets have separate results.',
                     'evidence': {'geometry_counts': empty, 'child_definitions': target.get('children', [])}}
    decisions = worker.finish_categories(categories, [], [])
    result = {'definition_id': definition['id'], 'disposition': 'incomplete' if decisions else 'verified_unchanged',
              'work_type': 'assembly_authoring', 'before': empty, 'after': dict(empty),
              'operations': [], 'path_map': {}, 'consumed_geometry_paths': [],
              'assessment': {'complete': not decisions, 'operations': [applicability], 'categories': categories,
                             'decisions_required': decisions, 'reason': 'Definition owns no geometry; authored composition retained.'},
              'verification_context': context,
              'verification': {'passed': True, 'method': 'authored_assembly_composition',
                               'independent_geometry_proof': False,
                               'checks': [{'name': 'written composition', 'passed': True}],
                               'evidence_fingerprint': {'files': fingerprints()}}}
    if result['verification']['evidence_fingerprint']['files'] is None:
        raise ValueError('cannot fingerprint authored assembly output')
    tuning.write(receipt, result)
    return result


def _batch_small_targets(plan, definitions, directory, max_workers, tuning, resume):
    """Amortize worker startup for small independent layers, keeping each writer."""
    targets = plan['targets']
    depended_on = {identifier for target in targets for identifier in target['depends_on']}
    by_id = {definition['id']: definition for definition in definitions}
    small = [target for target in targets if not target['depends_on'] and target['id'] not in depended_on
             and all(len(geometry_paths(by_id[identifier])) <= 16
                     and Path(by_id[identifier]['input_layer']).stat().st_size <= 256 * 1024
                     for identifier in target['definition_ids'])]
    # Keep all available workers useful. A batch holds at most four distinct
    # layers; larger scopes and dependency chains retain their original jobs.
    size = min(4, (len(small) + max_workers - 1) // max_workers)
    if size < 2:
        return
    selected = {target['id'] for target in small}
    result = [target for target in targets if target['id'] not in selected]
    for offset in range(0, len(small), size):
        members = small[offset:offset + size]
        if len(members) == 1:
            result.extend(members)
            continue
        identifier = 'batch-' + hashlib.sha256('|'.join(t['id'] for t in members).encode()).hexdigest()[:16]
        job_path = directory/'jobs'/(identifier+'.json')
        summary = directory/'jobs'/(identifier+'.result.json')
        jobs = [{'job': json.loads(Path(t['command'][3]).read_text()), 'summary_path': t['summary_path'], 'target': t}
                for t in members]
        tuning.write(job_path, {'jobs': jobs, 'resume': resume})
        target = copy.deepcopy(members[0])
        target.update(id=identifier, definition_ids=[i for t in members for i in t['definition_ids']],
                      command=[target['command'][0], target['command'][1], '--job', str(job_path), '--summary', str(summary)],
                      summary_path=str(summary), log_path=str(directory/'jobs'/(identifier+'.log')),
                      timeout_sec=sum(t['timeout_sec'] for t in members),
                      input_files=sorted({str(job_path)} | {p for t in members for p in t['input_files']}),
                      runtime_files=sorted({p for t in members for p in t['runtime_files']}),
                      output_files=sorted({str(summary)} | {p for t in members for p in t['output_files']}))
        result.append(target)
    plan['targets'] = result


def compute_definitions(manifest, directory, max_workers=4, resume=False, on_progress=None):
    """Optimize owned geometry and retain assembly coverage without native jobs.

    The authored manifest must opt into successor_policy and provide disjoint
    input/output layers. Only compute_requires creates scheduling dependencies.
    Native results and untouched-input checks gate success, not worker exit alone.
    """
    from pxr import Usd
    started = time.monotonic()
    if manifest.get('assessment_phase') == 'boundary_review':
        raise ValueError('Complete detailed assessment before computation')
    if not manifest.get('request', {}).get('successor_policy'):
        raise ValueError('successor_policy must be enabled before authoring')
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
        raise ValueError('max_workers must be a positive integer')
    definitions = manifest.get('definitions', [])
    if not definitions:
        raise ValueError('no authored definitions to execute')
    ids = {definition['id'] for definition in definitions}
    if len(ids) != len(definitions):
        raise ValueError('definition ids must be unique')
    inputs = [str(Path(d['input_layer']).resolve()) for d in definitions]
    outputs = [str(Path(d['output_layer']).resolve()) for d in definitions]
    if len(set(inputs)) != len(inputs) or len(set(outputs)) != len(outputs) or set(inputs) & set(outputs):
        raise ValueError('successor computation requires distinct immutable inputs and one output writer per definition')
    directory = Path(directory).resolve(); directory.mkdir(parents=True, exist_ok=True)
    tuning = _tuning()
    working = copy.deepcopy(manifest)
    working.setdefault('request', {})['material_source_paths'] = copy.deepcopy(manifest.get('material_source_paths', {}))
    dependencies = {}
    for definition in working['definitions']:
        requires = definition.get('compute_requires', [])
        if not isinstance(requires, list) or any(identifier not in ids or identifier == definition['id'] for identifier in requires):
            raise ValueError('compute_requires must name other known definitions')
        stage = Usd.Stage.Open(definition['input_layer'])
        if not stage or stage.GetCompositionErrors():
            raise ValueError('immutable definition input does not compose: ' + definition['id'])
        layers = {str(Path(layer.realPath).resolve()) for layer in stage.GetUsedLayers() if layer.realPath}
        if layers & set(outputs):
            raise ValueError('working references must target immutable inputs rather than compute outputs')
        dependencies[definition['id']] = layers
        # Reuse the scheduler-plan builder's dependency mechanism while leaving
        # worker_target and the caller's structural children untouched.
        definition['children'] = requires
    # Explicit compute dependencies remain valid through geometry-free assemblies.
    by_id = {definition['id']: definition for definition in working['definitions']}
    native_ids = {identifier for identifier, definition in by_id.items() if geometry_paths(definition)}
    resolved, visiting = {}, set()
    def native_requirements(identifier):
        if identifier in visiting:
            raise ValueError('compute_requires dependency cycle')
        if identifier in resolved:
            return resolved[identifier]
        visiting.add(identifier)
        result = set()
        for child in by_id[identifier]['children']:
            result.update(native_requirements(child))
            if child in native_ids:
                result.add(child)
        visiting.remove(identifier)
        resolved[identifier] = result
        return result
    for identifier in by_id:
        native_requirements(identifier)
    assemblies = [definition for definition in working['definitions'] if definition['id'] not in native_ids]
    working['definitions'] = [definition for definition in working['definitions'] if definition['id'] in native_ids]
    for definition in working['definitions']:
        definition['children'] = sorted(resolved[definition['id']])
    tuning.record_numerical_runtime(directory, manifest.get('request', {}))
    plan = tuning.batch_plan(working, directory)
    plan['run_id'] = 'successor-definition-compute'
    plan['phase'] = 'definition_computation'
    for target in plan['targets']:
        definition_id = target['definition_ids'][0]
        target['input_files'] = sorted(set(target['input_files']) | dependencies[definition_id])
        target['runtime_files'] = sorted(set(target['runtime_files']) | {
            str(Path(__file__).resolve()), str(Path(tuning.__file__).resolve()),
            str(Path(__file__).resolve().with_name('animation_context.py'))})
    _batch_small_targets(plan, working['definitions'], directory, max_workers, tuning, resume)
    tuning.write(directory/'batch-plan.json', plan)
    # The existing scheduler validates dependency cycles and verified-result
    # requirements before dispatching any worker process.
    runner = tuning.helper('successor_batch_scheduler', 'usd-optimize-run-operations/scripts/run_batch.py')
    immutable_files = sorted(set(inputs).union(*dependencies.values()))
    fingerprint_cache = runner._FingerprintCache()
    before = runner._file_hashes(immutable_files, fingerprint_cache)
    if before is None:
        raise ValueError('cannot fingerprint immutable definition inputs and dependencies')
    assembly_rows = [_assembly_result(definition, working['request'], dependencies[definition['id']],
                                     directory, tuning, runner, resume, fingerprint_cache) for definition in assemblies]
    prepared = time.monotonic()
    if on_progress:
        on_progress('definition_compute', activity='Run native geometry jobs',
                    definition_count=len(definitions), assembly_definition_count=len(assemblies),
                    native_definition_count=len(native_ids), native_job_count=len(plan['targets']))
    status = runner.BatchScheduler(plan, directory/'status.json', max_workers=max_workers, resume=resume).run()
    rows = assembly_rows + [row for target in status.get('targets', []) for row in target.get('definition_results', [])]
    untouched = before == runner._file_hashes(immutable_files)
    returned_ids = [row.get('definition_id') for row in rows]
    passed = (all(target['state'] == 'done' for target in status.get('targets', []))
              and all(row.get('assessment', {}).get('complete') for row in rows)
              and set(returned_ids) == ids and len(returned_ids) == len(ids) and untouched)
    result = {'schema_version': 1, 'phase': 'definition_computation', 'status': 'verified' if passed else 'incomplete',
              'passed': passed, 'definition_results': rows, 'immutable_inputs_unchanged': untouched,
              'definition_count': len(definitions), 'assembly_definition_count': len(assemblies),
              'native_job_count': len(plan['targets']), 'native_definition_count': len(native_ids),
              'scheduler': status, 'elapsed_seconds': time.monotonic()-started,
              'preparation_seconds': prepared-started, 'immutable_files': immutable_files,
              'publication_required': True, 'composed_consumer_verification_required': True}
    tuning.write(directory/'compute-results.json', result)
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest'); parser.add_argument('directory')
    parser.add_argument('--max-workers', type=int, default=4); parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    result = compute_definitions(json.loads(Path(args.manifest).read_text()), args.directory, args.max_workers, args.resume)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
