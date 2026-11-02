#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared assessment, planning and report helpers for the successor workflow."""
from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import sys
import time

REFERENCES = Path(__file__).resolve().parents[1] / 'references'


def helper(name, path):
    spec = importlib.util.spec_from_file_location(name, REFERENCES / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read(path):
    return json.loads(Path(path).read_text()) if path else {}


def safety_checks(path):
    import usd_optimize.validators
    usd_optimize.validators.register_all()
    executor = helper('tuning_safety_executor', 'usd-validation-runner/scripts/usd_validation_executor.py')
    registry = executor.load_registry()
    names = [c['canonical_name'] for c in registry['concepts']
             if c.get('role') == 'safety_gate' and c.get('scope_policy') == 'whole_stage']
    began = time.monotonic()
    issues = executor.validate_concepts(str(path), names, registry=registry)
    rows = [{'message': issue.message, 'severity': str(issue.severity),
             'rule': issue.rule.__name__, 'path': str(getattr(issue.at, 'path', ''))} for issue in issues]
    return {'passed': not any(r['severity'].endswith(('FAILURE', 'ERROR')) for r in rows),
            'concepts': names, 'issues': rows, 'seconds': time.monotonic()-began}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    large = (value.get('phase') == 'discovery_preview' or value.get('assessment_phase') == 'boundary_review' or len(value.get('definitions', [])) > 500
             or len(value.get('targets', [])) > 500)
    temporary.write_text(json.dumps(value, indent=None if large else 2, allow_nan=False))
    temporary.replace(path)


def record_numerical_runtime(directory, request):
    """Fingerprint numerical dependencies used by ordinary native/composition work."""
    versions = {'numpy': importlib.import_module('numpy').__version__}
    files = set()
    for name in versions:
        if versions.get(name) is None:
            continue
        module = importlib.import_module(name)
        root = Path(module.__file__).resolve().parent
        files.update(p for p in root.rglob('*') if p.is_file()
                     and (p.suffix in ('.py', '.pyd', '.dll') or '.so' in p.name))
        libraries = root.parent / (name + '.libs')
        if libraries.is_dir():
            files.update(p for p in libraries.iterdir() if p.is_file())
    fingerprints = {}
    for path in sorted(files):
        with path.open('rb') as stream:
            fingerprints[str(path)] = hashlib.file_digest(stream, 'sha256').hexdigest()
    write(Path(directory)/'verification-runtime.json', {'versions': versions, 'files': fingerprints})


def assessment_policy_fingerprint():
    """Boundary evidence is reusable only under the discovery policy that produced it."""
    paths = [Path(__file__), Path(__file__).with_name('animation_context.py'),
             REFERENCES/'component_scope.py', REFERENCES/'geometry_tolerance.py',
             REFERENCES/'usd-structure-assessment/scripts/structure_assessment.py',
             REFERENCES/'usd-structure-assessment/scripts/atomic_placement.py',
             REFERENCES/'usd-structure-assessment/scripts/boundary_reasoning.py',
             REFERENCES/'usd-structure-assessment/scripts/discovery_preview.py',
             REFERENCES/'usd-structure-assessment/references/usd-hierarchy-dedupe-candidates/scripts/instance_candidate_finder.py',
             REFERENCES/'usd-structure-assessment/references/usd-mesh-fragmentation-candidates/scripts/mesh_fragmentation_finder.py']
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def assessment(source, directory, request=None, decisions=None, force_rescan=False):
    spec = importlib.util.spec_from_file_location('assessment_live_progress', Path(__file__).with_name('successor_progress.py'))
    live = importlib.util.module_from_spec(spec); spec.loader.exec_module(live)
    with live.Progress({'source': {'path': str(source)}, 'request': request or {}}, directory, assessment=True) as report:
        report.event('assessment')
        manifest = _assessment_impl(source, directory, request, decisions, force_rescan, report.event)
        report.update_decisions(manifest)
        ready = manifest.get('structure_check', {}).get('passed', False)
        report.finish(ready, 'assessed' if ready else 'boundary review required')
        return manifest


def _assessment_impl(source, directory, request=None, decisions=None, force_rescan=False, on_progress=None):
    saved = Path(directory)/'manifest.json'
    cached = read(saved) if saved.exists() else {}
    same_source = cached.get('source', {}).get('path') == str(Path(source).resolve())
    # A boundary decision must not erase an already recorded preservation request.
    request = dict(cached.get('request', {}) if request is None and same_source else (request or {}))
    request['tolerance_mm'] = helper('assessment_geometry_budget', 'geometry_tolerance.py').physical_tolerance_mm(request)
    discovery = helper('structure_assessment', 'usd-structure-assessment/scripts/structure_assessment.py')
    from pxr import Ar, Sdf
    resolver_context = repr(Ar.GetResolver().GetCurrentContext())
    same_context = (all(request.get(k, default) == cached.get('request', {}).get(k, default)
                        for k, default in [('load','all'),('variants',{}),('time_samples',[None])])
                    and cached.get('resolver_context') == resolver_context)
    current_policy = assessment_policy_fingerprint()
    invalidated = []
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    assessment_pass = cached.get('assessment_pass', 0) + 1
    progress_path = directory/'assessment-progress.jsonl'
    progress_path.write_text('')
    def progress(phase, **counts):
        counts['assessment_pass'] = assessment_pass
        with progress_path.open('a') as log:
            log.write(json.dumps({'phase': phase, 'elapsed_seconds': time.monotonic()-began, 'counts': counts})+'\n')
        if on_progress:
            on_progress(phase, **counts)
    # Validate even when no decisions were supplied, and before choosing the fast path.
    dependencies_unchanged = same_context
    changed_layers = []
    for dependency in cached.get('inventory', {}).get('dependencies', []) if same_source else []:
        path = Path(dependency['path'])
        unchanged = path.is_file()
        if unchanged:
            with path.open('rb') as stream:
                unchanged = hashlib.file_digest(stream, 'sha256').hexdigest() == dependency['fingerprint']
        if not unchanged:
            dependencies_unchanged = False
            layer = Sdf.Layer.Find(str(path))
            if layer:
                changed_layers.append(layer)
    # A still-live stage can keep disk-backed layers cached across invocations.
    for layer in changed_layers:
        layer.Reload()
    previous = ({item['path']: dict(item) for item in cached['boundary_decisions']}
                if 'boundary_decisions' in cached else discovery._decisions_by_path(cached.get('decisions'))) if same_source else {}
    if previous and not dependencies_unchanged:
        invalidated = [choice for choice in previous.values()
                       if choice.get('selection', {}).get('basis') != 'user_specified']
        previous = {path: choice for path, choice in previous.items()
                    if choice.get('selection', {}).get('basis') == 'user_specified'}
    previous.update(discovery._decisions_by_path(decisions))
    grouped = decisions.get('groups', {}) if isinstance(decisions, dict) else {}
    effective_decisions = {'boundaries': list(previous.values()), 'groups': grouped}
    unresolved_dependencies = any(row.get('reason') in ('composition_error', 'unreadable_material_resource')
                                  for row in cached.get('inventory', {}).get('uninspected', []))
    reusable = (same_source and dependencies_unchanged and not unresolved_dependencies and not force_rescan
                and discovery.source_evidence_request(request) == discovery.source_evidence_request(cached.get('request', {}))
                and cached.get('assessment_policy_sha256') == current_policy)
    if reusable and cached.get('assessment_phase') == 'boundary_review' and cached.get('review_index', {}).get('source_evidence_only'):
        manifest = discovery.assess(str(Path(source).resolve()), request=request,
                                   decisions=effective_decisions, progress=progress, cached_review=cached)
    elif reusable and request == cached.get('request', {}) and decisions:
        manifest = discovery.apply_decisions(cached, decisions, progress=progress)
    else:
        manifest = discovery.assess(str(Path(source).resolve()), request=request,
                                    decisions=effective_decisions, progress=progress)
    manifest['assessment_pass'] = assessment_pass
    manifest['resolver_context'] = resolver_context
    manifest['assessment_policy_sha256'] = current_policy
    manifest['request'] = request or {}
    # boundary_decisions is canonical; do not serialize the same large records twice.
    manifest.pop('decisions', None)
    if invalidated:
        manifest['invalidated_boundary_decisions'] = invalidated
        manifest['selection_invalidation_reason'] = 'Source dependencies or composition context changed; inferred judgments require review.'
    manifest['workspace'] = str(Path(directory).resolve())
    conformance = helper('structure_conformance', 'optimization-report/scripts/structure_conformance.py')
    if manifest.get('assessment_phase') == 'boundary_review':
        manifest['structure_check'] = {'passed': False, 'errors': [
            'Boundary review is pending; detailed geometry ownership and structure conformance have not run.']}
    else:
        manifest['structure_check'] = conformance.check_manifest(manifest)
    write(Path(directory) / 'manifest.json', manifest)
    return manifest


def batch_plan(manifest, directory):
    """Generate writer jobs; one file owner can contain several definitions."""
    directory = Path(directory).resolve()
    worker = REFERENCES / 'usd-optimize-run-operations/scripts/native_worker.py'
    groups = {}
    for definition in manifest['definitions']:
        groups.setdefault(definition['output_layer'], []).append(definition)
    owner = {d['id']: layer for layer, definitions in groups.items() for d in definitions}
    ids = {layer: 'writer-' + hashlib.sha256(layer.encode()).hexdigest()[:16] for layer in groups}
    targets = []
    request = manifest.get('request', {})
    dependencies = {str(Path(manifest['source']['path']).resolve())}
    dependencies.update(d['path'] for d in manifest.get('inventory', {}).get('dependencies', []))
    for definition in manifest['definitions']:
        dependencies.update(d['resolved_path'] for d in definition.get('material_dependencies', [])
                            if d.get('resolved_path') and Path(d['resolved_path']).is_file())
    material = manifest.get('output', {}).get('material_library')
    if material:
        dependencies.add(material)
    runtime_files = sorted(str(p) for p in REFERENCES.rglob('*') if p.suffix in ('.py', '.json'))
    numerical_snapshot = directory / 'verification-runtime.json'
    if numerical_snapshot.is_file():
        runtime_files.append(str(numerical_snapshot))
    runtime = os.environ.get('USD_OPTIMIZE_ROOT')
    if runtime:
        runtime_files += sorted(str(p) for p in Path(runtime).rglob('*')
                                if p.is_file() and (p.suffix in ('.py', '.json', '.dll', '.pyd') or '.so' in p.name))
    else:
        raise ValueError('USD_OPTIMIZE_ROOT must identify the verified runtime for resumable execution')
    for layer, definitions in groups.items():
        job_id = ids[layer]
        source_layers = {d['input_layer'] for d in definitions}
        if len(source_layers) != 1:
            raise ValueError('definitions sharing a writer must share its input layer')
        source_layer = next(iter(source_layers))
        job_dependencies = sorted({ids[owner[c]] for d in definitions for c in d.get('children', [])
                               if owner[c] != layer})
        job_path = directory / 'jobs' / (job_id + '.json')
        summary = directory / 'jobs' / (job_id + '.result.json')
        payload = {'source_layer': source_layer, 'output_layer': layer,
                   'definitions': [d.get('worker_target', d) for d in definitions],
                   'request': manifest.get('request', {})}
        write(job_path, payload)
        targets.append({'id': job_id, 'path': layer, 'role': 'loadable_subasset',
                        'depends_on': job_dependencies, 'definition_ids': [d['id'] for d in definitions],
                        'require_verified_results': True,
                        'command': [sys.executable, str(worker), '--job', str(job_path), '--summary', str(summary)],
                        'summary_path': str(summary), 'timeout_sec': manifest.get('request', {}).get('timeout_seconds', 600),
                        'log_path': str(directory/'jobs'/(job_id+'.log')),
                        'resources': {'memory_mb': request.get('worker_memory_mb', 512), 'gpu_slots': 0},
                        'input_files': sorted(dependencies | {source_layer, str(job_path)} | {owner[c] for d in definitions for c in d.get('children', []) if owner[c] != layer}),
                        'runtime_files': runtime_files,
                        'output_files': [layer, str(summary)]})
    plan = {'run_id': 'asset-definitions', 'targets': targets,
            'resource_limits': {'memory_mb': request.get('memory_budget_mb', 4096), 'gpu_slots': 0}}
    write(directory / 'batch-plan.json', plan)
    return plan






def report_assessment(assessment, definition_id):
    """Keep decisions in the report; retain detailed native evidence once on disk."""
    if not isinstance(assessment, dict):
        return assessment
    compact = {key: assessment[key] for key in ('complete', 'decisions_required') if key in assessment}
    for key in ('categories', 'operations'):
        rows = assessment.get(key)
        if not isinstance(rows, list):
            if key in assessment:
                compact[key] = rows
            continue
        compact[key] = []
        for row in rows:
            if not isinstance(row, dict):
                compact[key].append(row)
                continue
            entry = {field: row[field] for field in ('category', 'operation', 'name', 'status', 'reason') if field in row}
            entry['evidence'] = ({'file': 'definition-results.json', 'definition_id': definition_id,
                                  'assessment_section': key, 'entry_index': len(compact[key])}
                                 if row.get('evidence') else row.get('evidence'))
            compact[key].append(entry)
    return compact


def build_report(manifest, results, verification, directory, selected_profiles=None, *, persist_evidence=True):
    """Build report data and coverage artifacts before workflow-specific measurements."""
    directory = Path(directory)
    validator = helper('validate_report', 'optimization-report/scripts/validate_report.py')
    counts = {'optimized': 0, 'verified_unchanged': 0, 'incomplete': 0}
    for result in results:
        key = result.get('disposition', 'incomplete')
        counts[key if key in counts else 'incomplete'] += 1
    counts['incomplete'] += len({d['id'] for d in manifest['definitions']} - {r['definition_id'] for r in results})
    report = {
        'asset_name': Path(manifest['source']['path']).stem,
        'input_path': manifest['source']['path'], 'output_path': manifest['output']['path'],
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'verdict': 'neutral', 'workflow_mode': 'full',
        'reasoning': ('Definition work and assembly verification passed.' if verification['passed'] else
                      'Required work or verification is incomplete; inspect verification.json.'),
        'optimization_score': 5.0, 'score_scope': 'stage_optimization',
        'score_label': 'neutral', 'metric_groups': [{'id': 'other', 'display_name': 'Comparative performance',
            'score': 5.0, 'status': 'not_measured', 'weight': 1,
            'summary': 'Neutral pending comparable performance measurements.'}], 'metrics': [],
        'operations': [{'order': i+1, 'name': op['operation'], 'method': 'Usd Optimize',
                        'result': op.get('status', 'executed').replace('_', ' ') + (': ' + op['reason'] if op.get('reason') else '')} for i, op in enumerate(
                            op for result in results for op in result.get('operations', []) if op.get('success') is True)],
        'validators': [{'name': 'Definition and assembly verification',
                        'issues': len(verification.get('errors', [])),
                        'notes': 'passed' if verification['passed'] else 'incomplete'}],
        'measurement_context': {'runtime': 'standalone USD Python', 'profile_mode': 'definition operations'},
        'runtime_profiling': {'status': 'not_run', 'summary': 'No renderer profiling was performed.'},
        'artifacts': {key: str(directory / ('report.' + ext)) for key, ext in [('json','json'),('markdown','md'),('html','html')]},
        'notes': '',
        'target_coverage': {'complete': bool(verification['passed']),
                            'source_manifests': [str(directory/'manifest.json')], 'entries': []}}
    baseline_connections = manifest.get('inventory', {}).get('connection_findings', [])
    if baseline_connections:
        report['validators'].append({'name': 'Pre-existing shader connection failures',
                                     'issues': len(baseline_connections),
                                     'notes': 'Recorded in the source inventory; preservation checks reject new failures.'})
        report['notes'] += (f' The source has {len(baseline_connections)} recorded shader connection failures. '
                            'These are findings in the original source; no renderer validation was performed.')
    if verification['passed']:
        from pxr import UsdUtils
        def referenced_bytes(path):
            layers, assets, missing = UsdUtils.ComputeAllDependencies(path)
            if missing:
                raise ValueError('storage measurement has unresolved dependencies')
            paths = {str(Path(p).resolve()) for p in [layer.realPath for layer in layers] + list(assets) if p}
            return sum(Path(p).stat().st_size for p in paths if Path(p).is_file())
        if selected_profiles is not None:
            inventories = [profile['inventory'] for profile in selected_profiles]
            if len(inventories) != 2:
                raise ValueError('selected-composition storage requires before and after inventories')
            storage_complete=all(item.get('inventory_complete') for item in inventories)
            storage = [item['referenced_storage_bytes'] for item in inventories]
            report['measurement_context']['storage_scope'] = inventories[0]['storage_scope']
            report['measurement_context']['storage_coverage']='complete' if storage_complete else 'partial'
        else:
            storage = [referenced_bytes(manifest['source']['path']), referenced_bytes(manifest['output']['path'])]
            storage_complete=True
        observations = [('referenced_storage_bytes', 'Referenced storage', 'storage_proxy', *storage)]
        for key, label in [('meshes', 'Unique-definition meshes'), ('vertices', 'Unique-definition mesh vertices'),
                           ('faces', 'Unique-definition mesh faces')]:
            observations.append((key, label, 'structure_proxy',
                                 sum(r.get('before', {}).get(key, 0) for r in results),
                                 sum(r.get('after', {}).get(key, 0) for r in results)))
        for key, label, category, before, after in observations:
            delta = 100*(after-before)/before if before else 0.0
            report['metrics'].append({'name': key, 'display_name': label, 'category': category,
                                      'before': before, 'after': after, 'change_pct': delta,
                                      'verdict': 'improved' if delta < -5 else 'regressed' if delta > 5 else 'neutral'})
        measured = report['metrics'][0]
        if not storage_complete:
            measured.update(verdict='neutral',notes='Resolved dependencies only; storage coverage incomplete and improvement unscored.')
        score = validator.score_from_change(measured['change_pct'])
        if not storage_complete:score=5.0
        report['optimization_score'] = score
        report['score_label'] = validator.score_band(score)
        report['metric_groups'] = [{'id': 'storage_proxy', 'display_name': 'Referenced storage', 'score': score,
                                   'status': 'proxy' if storage_complete else 'not_measured', 'weight': 1,
                                   'summary': 'Referenced files counted once; this includes packaging and encoding effects.',
                                   'caveat': 'Storage is not a rendering or load-time measurement. '+validator.SCORE_INTERPOLATION}]
        verdicts = {m['verdict'] for m in report['metrics']} - {'neutral'}
        report['verdict'] = 'mixed' if len(verdicts)>1 else next(iter(verdicts), 'neutral')
    by_id = {r['definition_id']: r for r in results}
    manifest['phase4_targets'] = []
    for definition in manifest['definitions']:
        result = by_id.get(definition['id'], {})
        count = result.get('before', {}).get('meshes', definition.get('geometry_counts', {}).get('Mesh', 0))
        row = {'path': definition.get('output_layer', manifest['source']['path']), 'definition_id': definition['id'], 'role': 'loadable_subasset', 'mesh_count': count,
               'disposition': result.get('disposition') if result.get('disposition') in ('optimized','verified_unchanged','exact_cache_hit','normalized_cache_hit','product_identity_hit') else 'blocked',
               'geometry_count': len(definition.get('owned_geometry', [])),
               'operations': [op['operation'] for op in result.get('operations', [])],
               'assessment': report_assessment(result.get('assessment', {'complete': False, 'operations': []}), definition['id']),
               'verification': result.get('verification', {'passed': False})}
        report['target_coverage']['entries'].append(row)
        manifest['phase4_targets'].append({'path': row['path'], 'definition_id': definition['id'],
                                            'role': row['role'], 'target_class': row['role'], 'mesh_count': count})
    if persist_evidence:
        write(directory/'manifest.json', manifest)
        write(directory/'definition-results.json', {'definition_results': results})
    return report


def finalize_report(report, manifest, results, verification, directory):
    """Validate and render the completed report once, after all workflow facts."""
    directory = Path(directory)
    helper('asset_report_results', 'optimization-report/scripts/asset_results.py').summarize_report(
        report, manifest, results, verification)
    validator = helper('validate_report', 'optimization-report/scripts/validate_report.py')
    errors = (validator.validate_completed_report(report, [manifest]) if verification['passed']
              else validator.validate_report(report))
    if errors:
        write(directory / 'report-rejected.json', {'report': report, 'errors': errors})
        raise ValueError('report validation failed: ' + '; '.join(errors[:10]))
    write(directory / 'report.json', report)
    renderer = helper('tuning_report_renderer', 'report-templates/render_preview.py')
    context = renderer.build_context(report)
    for suffix in ('html', 'md'):
        template = (REFERENCES/'report-templates'/('optimization-report.'+suffix+'.template')).read_text()
        (directory/('report.'+suffix)).write_text(renderer.render_template(template, context, autoescape=suffix == 'html'))
    return {'passed': True, 'json': str(directory/'report.json'),
            'markdown': str(directory/'report.md'), 'html': str(directory/'report.html')}


def generate_report(manifest, results, verification, directory, selected_profiles=None):
    """Build and finalize a report for callers without additional measurements."""
    report = build_report(manifest, results, verification, directory, selected_profiles)
    finalize_report(report, manifest, results, verification, directory)
    return report
