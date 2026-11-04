#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execute assessed successor definitions and verify their published consumer.

Assessment supplies engineering decisions. This tool authors immutable inputs,
computes independent definitions, publishes semantic libraries, then performs
one check of authored composition in the published assembly.

Usage:
    python successor_workflow.py --manifest JSON --workspace DIRECTORY [--max-workers N]
Arguments:
    Completed assessment manifest, fresh output workspace and worker limit (default 4).
Output:
    Writes prepared definitions, native-operation receipts, published libraries,
    consumer USD, composition results, profiles and JSON/Markdown/HTML reports.
    Failed runs retain incomplete results and reports when the workspace exists.
Exit codes:
    0 for a passed workflow; 1 for incomplete execution or an uncaught error;
    2 for invalid command-line arguments.
"""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import time


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def _child_boundaries(definition, occurrences, root):
    """Map retained child occurrences beneath this representative into its local frame."""
    boundary = definition['boundary_path']
    return sorted({root + occurrence['path'][len(boundary):]
                   for occurrence in occurrences
                   if occurrence['path'].startswith(boundary + '/')})


def _hash_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _paths(value, old, new):
    from pxr import Sdf
    if isinstance(value, str) and (value == old or value.startswith(old+'/') or value.startswith(old+'.')):
        return str(Sdf.Path(value).ReplacePrefix(Sdf.Path(old), Sdf.Path(new)))
    if isinstance(value, list):
        return [_paths(item, old, new) for item in value]
    if isinstance(value, dict):
        return {_paths(key, old, new): _paths(item, old, new) for key,item in value.items()}
    return value


def _baseline_profile(profiler, source, cached):
    """Reuse a baseline only when its source content and measurement context match."""
    if cached and Path(cached).is_file():
        value = json.loads(Path(cached).read_text())
        required = ('traversal_samples_ms', 'warm_open_samples_ms', 'warm_open_spread_ms', 'sample_process_ids')
        if all(key in value for key in required) and value.get('warm_open_sample_count', 0) >= 5:
            current = profiler.sample(source, inventory=True)
            keys = ('dependencies', 'selected_variants', 'metersPerUnit', 'upAxis', 'startTimeCode',
                    'endTimeCode', 'timeCodesPerSecond', 'load_policy', 'usd_version', 'python_version', 'platform')
            if all(current.get(key) == value.get('inventory', {}).get(key) for key in keys):
                if not value.get('storage_normalization'):
                    value['inventory']['used_layer_paths']=current['used_layer_paths']
                    value['storage_normalization']=profiler.repack_storage(current)
                return value, {'reused': True, 'source': str(cached)}
    return profiler.profile(source), {'reused': False, 'reason': 'No complete matching baseline was available.'}


def _verify_reconstruction(manifest, directory, tuning):
    """Check operation records, file integrity and composed output paths.

    Regroup preservation belongs in its tests, not runtime evidence or replay.
    """
    from pxr import Sdf, Usd, UsdGeom
    receipts = manifest.get('reconstruction_results', [])
    original = manifest.get('original_source')
    if not original and not receipts:
        return {'passed': True, 'required': False, 'checks': []}
    if not original or not receipts:
        raise ValueError('Reconstruction requires an original source and a complete receipt chain')
    if manifest.get('unresolved_requirements') or manifest.get('reconstruction_requests'):
        raise ValueError('Reconstruction still has unresolved requirements')
    bridge = _load('workflow_reconstruction_bridge', Path(__file__).with_name('successor_reconstruction.py'))
    adapter = bridge._adapter()
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    def digest(path):
        path = str(Path(path).resolve())
        if path in files:
            return files[path]
        with Path(path).open('rb') as stream:
            files[path] = hashlib.file_digest(stream, 'sha256').hexdigest()
            return files[path]
    stages={}
    def composed(path):
        path=str(Path(path).resolve())
        if path not in stages:
            stage=Usd.Stage.Open(path)
            if not stage or stage.GetCompositionErrors():
                raise ValueError('Reconstruction layer does not compose: '+path)
            stages[path]=stage
            # Reuse nearby inputs without retaining every large intermediate.
            if len(stages)>3:
                del stages[next(iter(stages))]
        return stages[path]
    _, discovery, reasoning = adapter._tools()
    if discovery._digest(original['dependencies']) != original['fingerprint']:
        raise ValueError('Original reconstruction dependency inventory changed')
    for dependency in original['dependencies']:
        if digest(dependency['path']) != dependency['fingerprint']:
            raise ValueError('Original reconstruction source dependency changed')
    current = str(Path(original['path']).resolve())
    mapping = []
    checks = []
    for number, receipt in enumerate(receipts):
        if not receipt.get('accepted'):
            raise ValueError('Unaccepted reconstruction receipt')
        for path,expected in receipt['source_context']['dependency_fingerprints'].items():
            if digest(path)!=expected:
                raise ValueError('Reconstruction source dependency fingerprint changed: '+path)
        context_stage = composed(receipt['source_context']['stage_identifier'])
        if reasoning.decision_source_context(context_stage,receipt['source_context']['dependency_fingerprints']) != receipt['source_context']:
            raise ValueError('Reconstruction source composition context changed')
        input_path, output_path = receipt['source_layer'], receipt['output_layer']
        if str(Path(input_path).resolve()) != current:
            raise ValueError('Reconstruction receipt chain is missing or out of order')
        if digest(input_path) != receipt['input_sha256'] or digest(output_path) != receipt['output_sha256']:
            raise ValueError('Reconstruction input or output fingerprint changed')
        before, after = composed(input_path), composed(output_path)
        action = receipt['request']['action']
        changed = receipt['changed_path_map']
        if action == 'repair_material_targets':
            module=tuning.helper('workflow_material_target_repairs','usd-optimize-run-operations/scripts/material_publication.py')
            with Usd.EditContext(before,before.GetSessionLayer()):
                confirmed=module.repair_shader_targets(before)
            declared=receipt.get('material_target_repairs',{})
            replay={'exact_repairs':confirmed==declared and bool(confirmed['repaired']),
                    'written_targets':all(after.GetAttributeAtPath(row['property_path']) and
                        list(map(str,after.GetAttributeAtPath(row['property_path']).GetConnections()))==[row['target']]
                        for row in confirmed['repaired']),
                    'unchanged_paths':not changed}
            passed=all(replay.values())
        elif action == 'split':
            module = tuning.helper('workflow_replay_split','usd-optimize-run-operations/scripts/native_reconstruction.py')
            digest(module.__file__)
            paths = sorted({str(prim.GetPath()) for path in receipt['source_prim_paths']
                            for prim in Usd.PrimRange(before.GetPrimAtPath(path)) if prim.IsA(UsdGeom.Mesh)})
            if set(paths) != set(changed) or any(not targets for targets in changed.values()):
                raise ValueError('Split path map does not account for selected meshes')
            replay={'native_execution':receipt.get('accepted') is True,
                    'output_paths':all(after.GetPrimAtPath(p) for values in changed.values() for p in values)}
            passed=all(replay.values())
        elif action == 'regroup':
            requests=receipt['request'].get('requests')
            if requests is not None:
                memberships,created=bridge._regroup_batch_memberships(requests,mapping,before)
            else:
                owner=bridge._regroup_owner(receipt['request'],mapping,before)
                memberships={path:owner for path in receipt['source_prim_paths']}
                created=[owner] if receipt['request'].get('create_owner',False) else []
            roots=receipt.get('root_path_map')
            if not isinstance(roots,dict):
                raise ValueError('Old regroup record; reconstruct with the current API')
            if set(roots)!=set(memberships) or changed!={p:[q] for p,q in roots.items()}:
                raise ValueError('Regroup root mappings do not match requested scopes')
            if sorted(receipt.get('created_owner_paths',[]))!=sorted(created):
                raise ValueError('Regroup record does not match requested owners')
            replay={'operation_completed':receipt.get('status')=='regrouped',
                    'output_paths':all(after.GetPrimAtPath(q) and not after.GetPrimAtPath(p)
                        and Sdf.Path(q)!=Sdf.Path(memberships[p])
                        and Sdf.Path(q).HasPrefix(Sdf.Path(memberships[p])) for p,q in roots.items()),
                    'created_owners':all(after.GetPrimAtPath(p).IsA(UsdGeom.Xform) for p in created)}
            passed=all(replay.values())
        elif action == 'keep':
            replay={'same_input':Path(input_path).resolve()==Path(output_path).resolve() and not changed}
            passed=replay['same_input']
        else:
            raise ValueError('Unsupported reconstruction receipt action')
        if not passed:
            raise ValueError('Reconstruction operation record or output paths are invalid')
        mapping = bridge._compose(mapping, changed)
        current = str(Path(output_path).resolve())
        checks.append({'request_id':receipt['request_id'], 'passed':True, 'operation':replay})
    if current != str(Path(manifest['source']['path']).resolve()) or mapping != manifest.get('reconstruction_path_map'):
        raise ValueError('Reconstruction endpoint or composed path map changed')
    final = composed(current)
    for requirement in manifest.get('required_occurrence_edits', []):
        if not final.GetObjectAtPath(requirement['path']):
            raise ValueError('Required occurrence interface disappeared: '+requirement['path'])
    digest(__file__)
    return {'passed':True, 'required':True, 'checks':checks, 'files':files,
            'runtime':list(Usd.GetVersion()),
            'verification_script_sha256':digest(__file__),
            'method':'Operation integrity, compact root mappings and composed output paths; regroup preservation is covered by tests and native splitting is trusted.'}


def _report(tuning, manifest, rows, verification, directory, comparison, profiles, work_counts=None):
    validator = tuning.helper('successor_workflow_report_validation', 'optimization-report/scripts/validate_report.py')
    report = tuning.build_report(manifest, rows, verification, directory, selected_profiles=profiles)
    if work_counts:
        report['measurement_context']['execution_counts'] = dict(work_counts)
        report['notes'] += '\n\nSupplied-library reuse: ' + str(work_counts.get('reused_definition_count', 0)) + ' definition lookups confirmed, referencing ' + str(work_counts.get('reused_library_entry_count', 0)) + ' distinct library entries.'
    if manifest.get('within_run_reuse'):
        report['notes'] += '\n\nWithin-run reuse: ' + str(len(manifest['within_run_reuse'])) + ' duplicate definition computations avoided; original/representative IDs and accepted colour differences are recorded in the manifest.'
    report['input_path'] = manifest.get('original_source',manifest['source'])['path']
    if manifest.get('output', {}).get('material_library'):
        report['notes'] += '\n\nShared material library: ' + manifest['output']['material_library'] + '.'
    if manifest.get('material_optimization'):
        material = manifest['material_optimization']
        report['operations'].append({'order':len(report['operations'])+1,'name':'optimizeMaterials','method':'Usd Optimize',
                                     'result':material['status']+'; '+str(material.get('before'))+' -> '+str(material.get('after'))})
    repair_batches=[manifest.get('material_target_repairs', {})]+manifest.get('reconstruction_material_target_repairs', [])
    repaired=sum(len(batch.get('repaired', [])) for batch in repair_batches)
    unresolved=sum(len(batch.get('unresolved', [])) for batch in repair_batches)
    if repaired or unresolved:
        report['operations'].append({'order':len(report['operations'])+1, 'name':'repairMaterialTargets',
            'method':'USD authoring', 'result':f'{repaired} uniquely resolved shader targets repaired; {unresolved} targets unresolved'})
        report['notes']+=f' Material target repair: {repaired} confirmed namespace corrections; {unresolved} unresolved targets. Exact paths and reasons are in the manifest.'
    report['measurement_context'].update(profile_mode='fresh-process stage open and traversal', warm_open_samples=5,
        first_open_policy='not guaranteed OS-cold', original_baseline=str(directory/'before-profile.json'),
        optimized_profile=str(directory/'after-profile.json'))
    normalization=profiles[0].get('storage_normalization',{})
    before_inventory,after_inventory=[p['inventory'] for p in profiles]
    complete=(normalization.get('status')=='complete' and before_inventory.get('inventory_complete')
              and after_inventory.get('inventory_complete'))
    raw=before_inventory['referenced_storage_bytes'];optimized=after_inventory['referenced_storage_bytes']
    storage=next((m for m in report['metrics'] if m['name']=='referenced_storage_bytes'),None)
    if complete and storage is not None:
        normalized=normalization['repack_normalized_baseline_bytes']
        def percent(before,after):return 100*(after-before)/before if before else 0.
        delta=percent(normalized,optimized)
        storage.update(display_name='Total delivery against lossless USDC baseline',before=normalized,after=optimized,
                       change_pct=delta,verdict='improved' if delta < -5 else 'regressed' if delta > 5 else 'neutral',
                       notes='Each original used layer was independently re-encoded to USDC without flattening; referenced resource bytes remain included once.')
        report['metrics'].append({'name':'raw_referenced_storage_bytes','display_name':'Raw referenced footprint (encoding included)',
            'category':'storage_proxy','before':raw,'after':optimized,'change_pct':percent(raw,optimized),'verdict':'neutral',
            'notes':'Observed raw bytes only; encoding savings are separated below and do not determine the optimization score.'})
        scene_bytes=after_inventory.get('scene_storage_bytes', optimized-after_inventory.get('additional_required_artifact_bytes',0))
        overhead=optimized-scene_bytes
        report['metrics'].extend([
            {'name':'scene_storage_bytes','display_name':'Scene USD and referenced resources','category':'storage_proxy',
             'unit':'bytes','before':normalized,'after':scene_bytes,'change_pct':percent(normalized,scene_bytes),
             'verdict':'improved' if percent(normalized,scene_bytes)<-5 else 'regressed' if percent(normalized,scene_bytes)>5 else 'neutral',
             'notes':'Files required to open the scene, counted once; excludes the persistent reuse index.'},
            {'name':'reuse_overhead_bytes','display_name':'Persistent reuse index overhead','category':'storage_proxy',
             'unit':'bytes','before':0,'after':overhead,'change_pct':0.,'verdict':'neutral',
             'notes':'Additional delivery artifacts used for subsequent optimization runs; not stage-open or rendering cost. Included in total delivery.'}])
        changed={op['operation'] for row in rows for op in row.get('operations',[]) if op.get('status')=='changed'}
        lever=next((label for operation,label in [('fitPrimitives','fit_primitive'),('decimateMeshes','decimation'),
                    ('removeUnusedUVs','unused_primvar_removal'),('optimizePrimvars','primvar_indexing'),('meshCleanup','dead_data_removal')]
                    if operation in changed),'none')
        if lever=='none' and len(before_inventory.get('used_layer_paths',[]))>len(after_inventory.get('used_layer_paths',[])):
            lever='layer_consolidation'
        report['footprint']={'raw_input_bytes':raw,'repack_normalized_baseline_bytes':normalized,'optimized_bytes':optimized,
            'repack_delta_pct':percent(raw,normalized),'structural_delta_pct':delta,'raw_delta_pct':percent(raw,optimized),
            'scored_against':'repack_normalized','disk_win_source':lever,
            'notes':normalization['method']+' Resources are counted once unchanged. '+
                    ('Listed byte-reduction operation ran; its isolated contribution was not measured.' if lever!='none' else 'No specific byte-reduction operation was identified.')}
        score=validator.score_from_change(delta)
        report['optimization_score']=score
        report['score_label']=validator.score_band(score)
        report['metric_groups']=[{'id':'storage_proxy','display_name':'Total delivery storage','score':score,
            'status':'proxy','weight':1,'summary':'Compared with the independently re-crated original layer graph; encoding savings do not count as structural optimization.',
            'caveat':'Storage is not a rendering, RAM or GPU measurement. '+validator.SCORE_INTERPOLATION}]
        report['measurement_context']['storage_baseline']='lossless per-used-layer USDC re-encoding'
    else:
        if storage is not None:
            storage.update(verdict='neutral',notes='Resolved footprint only; complete repack-normalized storage could not be measured. Unscored.')
        report['optimization_score']=5.;report['score_label']='neutral'
        report['metric_groups']=[{'id':'storage_proxy','display_name':'Referenced storage','score':5.,'status':'not_measured','weight':1,
                                 'summary':'Storage coverage or lossless normalization is incomplete; no improvement is claimed.'}]
        report['measurement_context']['storage_coverage']='partial'
    for key, label in [('warm_open', 'Warm stage open'), ('traversal', 'Stage traversal')]:
        metric = comparison[key]
        verdict = {'decreased': 'improved', 'increased': 'regressed'}.get(metric['interpretation'], 'neutral')
        report['metrics'].append({'name': key+'_ms', 'display_name': label, 'category': 'load_time' if key == 'warm_open' else 'composition',
            'unit': 'ms', 'direction': 'lower_is_better', 'evidence_type': 'direct', 'before': metric['before_ms'],
            'after': metric['after_ms'], 'change_pct': metric['delta_pct'] or 0.0, 'verdict': verdict,
            'notes': 'Measured in fresh processes. Interpretation: '+metric['interpretation']+'. See profile-comparison.json for variability.'})
    measured_verdicts = {row['verdict'] for row in report['metrics']} - {'neutral'}
    report['verdict'] = 'mixed' if len(measured_verdicts) > 1 else next(iter(measured_verdicts), 'neutral')
    if not complete:
        report['verdict']='neutral'
        report['reasoning']+=' Storage measurement or normalization is incomplete; the required numeric score is a not-measured placeholder, not evidence of optimization.'
    return tuning.finalize_report(report, manifest, rows, verification, directory)


def _preauthor_product_reuse(working, reuse, lookup, progress):
    """Confirm complete product families before any worker inputs are written.

    Ordinary retained parents still need prepared source-content matching. Their
    children stay in that existing transaction so later parent reconciliation
    cannot invalidate references already authored to an early selection.
    """
    indexes = working['request'].get('persistent_indexes', [])
    if not indexes or working.get('context', {}).get('variants'):
        # Variant-context selection is applied by normal input authoring. The
        # source-only shortcut must not inspect a different composed asset.
        return {}
    roots = {o['definition_id'] for o in working['occurrences'] if not o.get('parent_path')}
    definitions = {d['id']: d for d in working['definitions'] if d['id'] not in roots}
    parents = {identifier: set() for identifier in definitions}
    children = {identifier: set() for identifier in definitions}
    by_path = {o['path']: o for o in working['occurrences']}
    for occurrence in working['occurrences']:
        child = occurrence['definition_id']
        parent = by_path.get(occurrence.get('parent_path'), {}).get('definition_id')
        if child in definitions and parent in definitions:
            parents[child].add(parent)
            children[parent].add(child)
    eligible = {identifier for identifier, d in definitions.items() if d.get('product_identity')}
    while True:
        filtered = {identifier for identifier in eligible
                    if (parents[identifier] | children[identifier]) <= eligible}
        if filtered == eligible:
            break
        eligible = filtered
    if not eligible:
        return {}
    context = lookup.LookupContext(indexes)
    candidates = []
    prepared = {}
    progress('product_confirmation', definition_count=len(eligible))
    from pxr import Usd
    # Repeated product confirmations read one immutable source stage. Reuse its
    # composed stage within this call instead of reopening the whole station.
    with Usd.StageCacheContext(Usd.StageCache()):
        for identifier in sorted(eligible):
            definition = copy.deepcopy(definitions[identifier])
            definition['source_layer'] = working['source']['path']
            definition['source_prim_path'] = definition['boundary_path']
            definition['child_boundaries'] = _child_boundaries(definition, working['occurrences'], definition['boundary_path'])
            candidates.append(definition)
            prepared[identifier] = reuse.confirm_reuse(definition, working['request'], indexes, context, product_only=True)
        selected = reuse.reconcile_parent_reuse(candidates, working['occurrences'], prepared, context)
    # A declined family stays wholly on ordinary preparation; its future parent
    # selection must not change an already-written child reference.
    accepted = {identifier for identifier, row in selected.items() if row.get('entry')}
    while True:
        filtered = {identifier for identifier in accepted
                    if (parents[identifier] | children[identifier]) <= accepted}
        if filtered == accepted:
            break
        accepted = filtered
    for identifier in accepted:
        definitions[identifier]['persistent_entry'] = selected[identifier]['entry']
    progress('product_confirmation_complete', confirmed_definition_count=len(accepted),
             deferred_definition_count=len(definitions)-len(accepted))
    return {identifier: selected[identifier] for identifier in accepted}


def verify_published(working, records, index, published_dir, directory, max_workers=4, progress=None):
    """Rebind completed definitions to published entrypoints; no geometry replay."""
    from pxr import Sdf, Usd, UsdGeom
    worker = _load('publication_output_paths', Path(__file__).resolve().parents[1]/'references/usd-optimize-run-operations/scripts/native_worker.py')
    published_dir = Path(published_dir)
    final_consumer = published_dir/index['consumer_layer']
    roots = {row['definition_id'] for row in working['occurrences'] if not row.get('parent_path')}
    after = Usd.Stage.Open(str(final_consumer))
    reverified = []
    files = {str((published_dir/path).resolve()): digest for path,digest in index.get('dependencies',{}).items()}
    expected = dict(files)
    conflicts = []
    for row in records.values():
        for path,digest in row.get('verification',{}).get('evidence_fingerprint',{}).get('files',{}).items():
            if path in expected and expected[path]!=digest:conflicts.append(path)
            expected[path]=digest
    stale = conflicts + [path for path,digest in expected.items()
                         if not Path(path).is_file() or hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest]
    current = bool(files) and not stale
    stages = {}
    scripts=Path(__file__).resolve().parent
    composition=_load('publication_material_check',scripts/'composition_verification.py')
    discovery=_load('publication_material_facts',scripts.parent/'references/usd-structure-assessment/scripts/structure_assessment.py')
    material_caches={}
    for definition in working['definitions']:
        old_target = copy.deepcopy(definition.get('worker_target', definition))
        old_root = old_target.get('boundary_path', '/Asset')
        record = records[definition['id']]
        if definition['id'] in roots:
            occurrence = next(row for row in working['occurrences'] if row['definition_id'] == definition['id'] and not row.get('parent_path'))
            refs = after.GetPrimAtPath(occurrence['path']).GetMetadata('references').GetAddedOrExplicitItems()
            if len(refs) != 1:
                raise ValueError('root occurrence requires one published assembly reference')
            output_layer = Sdf.ComputeAssetPathRelativeToLayer(after.GetRootLayer(), refs[0].assetPath)
            output_root = str(refs[0].primPath) or str(Usd.Stage.Open(output_layer).GetDefaultPrim().GetPath())
        else:
            entry = index['entries'][definition['id']]
            output_layer = str(published_dir/entry['layer']); output_root = entry['prim_path']
        persistent=definition.get('persistent_entry',{})
        native_layer=persistent.get('layer',definition.get('output_layer'))
        native_root=persistent.get('prim_path',old_root)
        native_mapping=record.get('path_map', {})
        original_mapping = native_mapping
        if definition['id'] not in roots:
            original_mapping = entry.get('matching',{}).get('operation_path_map',original_mapping)
        scope = old_target.get('operation_scope', {}).get('geometry_paths', old_target.get('owned_geometry', []))
        consumed = set(record.get('consumed_geometry_paths', []))
        clustered = [row.get('evidence', {}) for row in record.get('operations', [])
                     if row.get('operation') == 'merge' and len(row.get('evidence', {}).get('actual_output_paths', [])) > 1]
        removed = {path for row in clustered for path in row.get('removed_source_paths', [])}
        scope = [path for path in scope if path not in removed] + [path for row in clustered for path in row['actual_output_paths']]
        scope = [path for path in scope if path not in consumed and original_mapping.get(path,path) not in consumed] + old_target.get('protected_paths', [])
        if record.get('disposition') in ('exact_cache_hit', 'normalized_cache_hit', 'product_identity_hit'):
            scope = [_paths(p, record['cached_root'], old_root) for p in record['output_geometry_paths']] + old_target.get('protected_paths', [])
            original_mapping = {}; native_mapping = {}
        output_mapping = {path: _paths(original_mapping.get(path,path),old_root,output_root) for path in scope}
        record['compute_verification'] = record['verification']
        record['publication_verification'] = {'source_layer': definition['input_layer'], 'output_layer': output_layer,
                                               'path_map': output_mapping}
        definition['output_layer'] = output_layer; definition['output_prim_path'] = output_root
        definition['worker_target'] = _paths(old_target, old_root, output_root)
        record['output_geometry_paths'] = [_paths(path, old_root, output_root) for path in
            worker.resolved_geometry_paths(old_target.get('operation_scope', {}).get('geometry_paths', old_target.get('owned_geometry', [])),
                                           original_mapping, record.get('operations', []))
            if path not in consumed]
        if record.get('disposition') in ('exact_cache_hit', 'normalized_cache_hit', 'product_identity_hit'):
            cached_stage = Usd.Stage.Open(native_layer)
            record['output_geometry_paths'] = [published for local, published in output_mapping.items()
                if cached_stage.GetPrimAtPath(_paths(local, old_root, native_root)).IsA(UsdGeom.Gprim)]
        record['path_map'] = _paths(original_mapping, old_root, output_root)
        record['consumed_geometry_paths'] = _paths(record.get('consumed_geometry_paths', []), old_root, output_root)
        if output_layer not in stages:
            stages[output_layer] = Usd.Stage.Open(output_layer)
        stage = stages[output_layer]
        accepted = bool(current and stage and not stage.GetCompositionErrors() and stage.GetPrimAtPath(output_root))
        native_error=None
        if not native_layer:
            native_error='Completed native output layer is missing'
            stages[native_layer]=None
        elif native_layer not in stages:
            try:
                stages[native_layer]=Usd.Stage.Open(native_layer)
            except Exception as error:
                stages[native_layer]=None
                native_error=str(error)
        native_stage=stages[native_layer]
        if not native_stage:
            native_error=native_error or 'Completed native output cannot be opened'
        bindings={};material_errors=[];compared=set()
        for original,published in output_mapping.items():
            native_path=_paths(native_mapping.get(original,original),old_root,native_root)
            pair=(native_path,published)
            if pair in compared:continue
            compared.add(pair)
            before_prim=native_stage.GetPrimAtPath(native_path) if native_stage else None
            if not before_prim or not before_prim.IsA(UsdGeom.Gprim):continue
            expected_materials=composition.material_fingerprints(before_prim,discovery,
                material_caches.setdefault(native_layer,{}),working.get('material_source_paths',{}))
            if not stage.GetPrimAtPath(published):
                material_errors.append(published)
                continue
            actual_materials=composition.material_fingerprints(stage.GetPrimAtPath(published),discovery,
                material_caches.setdefault(output_layer,{}),working.get('material_source_paths',{}))
            bindings[published]=sorted(expected_materials)
            if expected_materials != actual_materials:material_errors.append(published)
        # A fitted geometry root has a new local frame (for example Mesh -> Cube).
        # Preserve that native frame when checking our occurrence placement.
        geometry_root_transform = None
        native_prim = native_stage.GetPrimAtPath(native_root) if native_stage else None
        if native_prim and native_prim.IsA(UsdGeom.Gprim):
            input_layer = definition['input_layer']
            if input_layer not in stages:
                stages[input_layer] = Usd.Stage.Open(input_layer)
            input_prim = stages[input_layer].GetPrimAtPath(old_root)
            if input_prim and input_prim.IsA(UsdGeom.Gprim):
                before_frame = UsdGeom.Xformable(input_prim).GetLocalTransformation()
                after_frame = UsdGeom.Xformable(native_prim).GetLocalTransformation()
                geometry_root_transform = [list(row) for row in after_frame * before_frame.GetInverse()]
        record['publication_verification']['geometry_root_transform'] = geometry_root_transform
        accepted=accepted and bool(native_stage) and not material_errors
        record['publication_verification'].update(passed=accepted, method='entrypoint_and_publication_integrity',
            native_output_layer=native_layer,native_output_error=native_error,material_bindings=bindings,material_errors=material_errors)
        record['verification'] = {'passed':record['compute_verification']['passed'] and accepted,
            'method':'native_execution_and_publication_composition', 'independent_geometry_proof':False,
            'evidence_fingerprint':{'files':files},
            'errors':([] if accepted else [native_error or ('Published material outcome changed' if material_errors
                else 'Published entry or recorded dependency changed: '+', '.join(stale))])}
        if record.get('disposition') in ('exact_cache_hit', 'normalized_cache_hit', 'product_identity_hit'):
            record['verification']['reuse_basis'] = record['compute_verification']['reuse_basis']
        record['verification_context'] = {'request_sha256': _hash_json(working['request']),
                                          'target_sha256': _hash_json(definition['worker_target'])}
        reverified.append(record)
    execution = {'passed': all(r['publication_verification']['passed'] for r in reverified),
                 'method':'publication_composition_checks', 'definition_count':len(reverified),
                 'independent_geometry_proof':False}

    return reverified, execution


def _execute(manifest, directory, max_workers=4, on_progress=None, common_library=None):
    from pxr import Sdf, Usd
    if manifest.get('assessment_phase') == 'boundary_review':
        raise ValueError('Resolve boundary review and finalize assessment before execution')
    directory = Path(directory).resolve(); directory.mkdir(parents=True, exist_ok=True)
    if (directory/'workflow-results.json').exists():
        raise ValueError('use a fresh workflow output directory')
    started = time.monotonic()
    def progress(phase, **facts):
        event = {'phase': phase, 'elapsed_seconds': time.monotonic()-started, **facts}
        with (directory/'workflow-progress.jsonl').open('a') as stream:
            stream.write(json.dumps(event)+'\n')
        if on_progress:
            on_progress(phase, **facts)
    progress('source_validation')
    scripts = Path(__file__).resolve().parent
    tuning = _load('successor_workflow_helpers', scripts/'usd_tuning.py')
    compute = _load('successor_workflow_compute', scripts/'successor_execution.py')
    working = copy.deepcopy(manifest)
    if not working.get('request', {}).get('successor_policy'):
        raise ValueError('manifest must explicitly select successor_policy')
    if working.get('request', {}).get('variants'):
        raise ValueError('Use the source stage current variant selections; request overrides are not supported')
    if working.get('decisions_required') or not working.get('structure_check', working.get('conformance', {})).get('passed'):
        raise ValueError('resolve structural decisions before execution')
    animation = _load('successor_animation_context',scripts/'animation_context.py')
    selected_source = Usd.Stage.Open(working['source']['path'])
    if not selected_source or selected_source.GetCompositionErrors():
        raise ValueError('Source selected composition is invalid')
    capabilities = animation.schema_availability(selected_source)
    tuning.write(directory/'schema-availability.json',capabilities)
    if not capabilities['passed']:
        return {'passed':False,'status':'unavailable_source_schema','capabilities':capabilities,
                'errors':capabilities['findings']}
    source_animation = animation.collect(selected_source,requested_times=working.get('context',{}).get('time_samples',[]))
    working.setdefault('context',{})['time_samples'] = source_animation['time_samples']
    working['animation_context'] = source_animation
    tuning.write(directory/'animation-context.json',source_animation)
    reconstruction = _verify_reconstruction(working, directory/'reconstruction-verification', tuning)
    tuning.write(directory/'reconstruction-verification.json', reconstruction)
    safety = tuning.safety_checks(working['source']['path'])
    working['source_safety'] = safety
    unresolved = [row for row in safety.get('issues', []) if row['severity'].endswith(('FAILURE', 'ERROR')) and row['rule'] not in ('KindChecker', 'TypeChecker')]
    if unresolved:
        result = {'schema_version': 1, 'passed': False, 'status': 'blocked_source_safety', 'source_safety': safety, 'errors': unresolved}
        tuning.write(directory/'workflow-results.json', result)
        return result
    profiler = _load('successor_workflow_profile', scripts/'stage_profile.py')
    baseline_source = working.get('original_source', working['source'])['path']
    progress('source_profile')
    baseline, baseline_context = _baseline_profile(profiler, baseline_source, working.get('request', {}).get('baseline_profile'))
    tuning.write(directory/'before-profile.json', baseline)
    before = Usd.Stage.Open(baseline_source)
    if not before or before.GetCompositionErrors():
        raise ValueError('source does not compose')
    before.Flatten().Export(str(directory/'before.usdc'))
    reuse = tuning.helper('successor_workflow_reuse', 'usd-optimize-run-operations/scripts/persistent_reuse.py')
    lookup = tuning.helper('successor_early_product_lookup', 'usd-optimize-run-operations/scripts/persistent_library.py')
    early_reuse = (_preauthor_product_reuse(working, reuse, lookup, progress)
                   if not working.get('output') else {})
    progress('author_inputs', reused_authored_inputs=bool(working.get('output')))
    if not working.get('output'):
        author = tuning.helper('successor_workflow_author', 'usd-optimize-run-operations/scripts/asset_authoring.py')
        author.author(working, directory/'working', progress=progress)
    working['request']['material_source_paths'] = copy.deepcopy(working.get('material_source_paths', {}))
    tuning.write(directory/'authored-manifest.json', working)
    if working.get('material_optimization',{}).get('passed') is False:
        result={'passed':False,'status':'material_preservation_failure','material_optimization':working['material_optimization']}
        tuning.write(directory/'workflow-results.json',result)
        return result
    roots = {row['definition_id'] for row in working['occurrences'] if not row.get('parent_path')}
    nonroots = [definition for definition in working['definitions'] if definition['id'] not in roots]
    reuse = tuning.helper('successor_workflow_reuse', 'usd-optimize-run-operations/scripts/persistent_reuse.py')
    worker = tuning.helper('successor_workflow_native_measure', 'usd-optimize-run-operations/scripts/native_worker.py')
    reused_results, matching, reuse_findings = {}, {}, {}
    indexes = working['request'].get('persistent_indexes', [])
    progress('persistent_matching', definition_count=len(nonroots))
    for definition in nonroots:
        target = definition.get('worker_target', definition)
        boundary = definition['boundary_path']
        local = target.get('boundary_path', '/Asset')
        target['child_boundaries'] = _child_boundaries(definition, working['occurrences'], local)
    prepared_reuse = dict(early_reuse)
    remaining = [definition for definition in nonroots if definition['id'] not in early_reuse]
    if remaining:
        prepared_reuse.update(reuse.prepare_definitions(remaining, working['request'], indexes,
            directory/'persistent-matching', max_workers=max_workers, progress=progress))
    if indexes:
        lookup = tuning.helper('successor_parent_lookup', 'usd-optimize-run-operations/scripts/persistent_library.py')
        prepared_reuse = reuse.reconcile_parent_reuse(nonroots, working['occurrences'], prepared_reuse,
                                                     lookup.LookupContext(indexes))
    current_reuse = tuning.helper('successor_current_reuse', 'usd-optimize-run-operations/scripts/current_run_reuse.py')
    within_run_reuse = current_reuse.consolidate(working, prepared_reuse, worker)
    nonroots = [definition for definition in working['definitions'] if definition['id'] not in roots]
    for definition in nonroots:
        identifier = definition['id']
        if indexes:
            confirmation = prepared_reuse[identifier]
            matching[identifier] = confirmation['matching']; reuse_findings[identifier] = confirmation['findings']
            if confirmation['entry']:
                entry = confirmation['entry']
                target = definition.get('worker_target', definition)
                # Early hits already reference the library; their original source
                # is accounting evidence, not a generated input-layer alias.
                entry['reference_layers'] = [] if definition.get('reuse_before_preparation') else [
                    {'layer': definition['input_layer'], 'prim_path': target.get('boundary_path', '/Asset')}]
                definition['persistent_entry'] = entry
                if entry.get('normalized_placement'):
                    from pxr import Gf
                    delta = Gf.Matrix4d(*[x for values in entry['normalized_placement']['matrix'] for x in values])
                    for occurrence in working['occurrences']:
                        if occurrence['definition_id'] != identifier: continue
                        existing = occurrence.get('internal_placement', {}).get('matrix')
                        if existing:
                            correction = delta * Gf.Matrix4d(*[x for values in existing for x in values])
                        else:
                            correction = delta
                        occurrence['internal_placement'] = {'matrix': [list(values) for values in correction],
                            'basis': 'normalized source identity with authorized constant colour rounding'}
                        occurrence.setdefault('source_to_definition', {occurrence['path']: definition['boundary_path']})
                row = confirmation['result']
                paths = target.get('operation_scope', {}).get('geometry_paths', target.get('owned_geometry', []))
                row['before'] = worker.measure(Usd.Stage.Open(definition['input_layer']), paths)
                output_paths = row['output_geometry_paths']
                row['after'] = worker.measure(Usd.Stage.Open(entry['layer']), output_paths)
                reused_results[identifier] = row
        else:
            matching[identifier] = prepared_reuse[identifier]['matching']
    compute_manifest = copy.deepcopy(working)
    compute_manifest['definitions'] = [definition for definition in compute_manifest['definitions'] if definition['id'] not in reused_results]
    for definition in compute_manifest['definitions']:
        definition['compute_requires'] = [identifier for identifier in definition.get('compute_requires', []) if identifier not in reused_results]
    progress('definition_compute', definition_count=len(compute_manifest['definitions']), max_workers=max_workers)
    computation = compute.compute_definitions(compute_manifest, directory/'compute', max_workers=max_workers, on_progress=progress)
    work_counts = {key: computation[key] for key in ('definition_count', 'assembly_definition_count', 'native_definition_count', 'native_job_count')}
    distinct_library_entries = {(d['persistent_entry']['layer'], d['persistent_entry']['prim_path'])
                                for d in nonroots if d.get('persistent_entry')}
    work_counts.update(asset_definition_count=len(working['definitions']), reused_definition_count=len(reused_results),
                       reused_library_entry_count=len(distinct_library_entries),
                       within_run_reused_definition_count=len(within_run_reuse))
    progress('definition_compute_complete', passed=computation['passed'], **work_counts)
    result = {'schema_version': 1, 'status': 'incomplete', 'passed': False,
              'scope': 'assessment decisions supplied by caller; optimization, publication and preservation verification',
              'compute': computation, 'before_layer': str(directory/'before.usdc')}
    result['baseline_profile'] = baseline_context
    result['within_run_reuse'] = within_run_reuse
    result['persistent_reuse'] = {'indexes': indexes, 'confirmed_definition_ids': sorted(reused_results), 'distinct_library_entry_count': len(distinct_library_entries), 'findings': reuse_findings}
    result['coverage'] = {'definition_computation': 'executed', 'persistent_cross_scene_lookup': 'executed' if indexes else 'no_indexes_supplied',
                          'independent_agent_qualification': 'not_claimed', 'renderer_verification': 'not_run'}
    if not computation['passed']:
        tuning.write(directory/'workflow-results.json', result)
        return result
    current_reuse.redirect_outputs(working)
    placement_author = tuning.helper('successor_workflow_placement_author', 'usd-optimize-run-operations/scripts/asset_authoring.py')
    result['placement_reconciliation'] = placement_author.reconcile_internal_placements(working)
    reconciled_owners={row['owner_definition_id'] for row in result['placement_reconciliation'] if row['owner_definition_id']}
    if reconciled_owners:
        for record in computation['definition_results']:
            if record['definition_id'] not in reconciled_owners:continue
            definition=next(row for row in working['definitions'] if row['id']==record['definition_id'])
            target=definition.get('worker_target',definition)
            policy=worker.verification_policy(working['request'],record.get('operations',[]))
            policy['path_map']=record.get('path_map',{})
            policy['allow_reduction']=any(row.get('operation')=='decimateMeshes' for row in record.get('operations',[]))
            for key in ('tolerance_mm','deviation_tolerance_mm','max_deviation_mm'):
                if key in policy:policy[key]/=target.get('maximum_occurrence_scale',1.0)
            files=dict(record['verification'].get('evidence_fingerprint',{}).get('files',{}))
            files[definition['output_layer']]=hashlib.sha256(Path(definition['output_layer']).read_bytes()).hexdigest()
            proof=worker.native_execution_verification(Usd.Stage.Open(definition['input_layer']),
                Usd.Stage.Open(definition['output_layer']),target,policy,record.get('operations',[]),files,animation)
            record['placement_reconciliation_verification']={'previous_verification':record['verification'],'passed':proof['passed']}
            record['verification']=proof
            if not proof['passed']:
                result['status']='placement_reconciliation_failure'
                tuning.write(directory/'workflow-results.json',result)
                return result
    planner = tuning.helper('successor_workflow_category_plan', 'usd-optimize-run-operations/scripts/category_publication.py')
    publisher = tuning.helper('successor_workflow_publisher', 'usd-optimize-run-operations/scripts/semantic_library.py')
    if not nonroots:
        raise ValueError('category publication requires a root aggregate with asset definitions')
    plan = planner.publication_plan(nonroots, directory/'publication-plan')
    records = {row['definition_id']: copy.deepcopy(row) for row in computation['definition_results']}
    records.update(copy.deepcopy(reused_results))
    outputs = {definition['id']: {'layer': definition['output_layer'],
               'prim_path': definition.get('output_prim_path', '/Asset'), 'verified': records[definition['id']]['verification']['passed'],
               'fingerprint': publisher.file_fingerprint(definition['output_layer']),
               'reference_layers': [definition['input_layer']] + definition.get('reference_layers', []),
               'whole_definition_reuse_eligible': (definition.get('structural_type') == 'atomic'
                    and definition.get('instanceable') is True
                    and not definition.get('worker_target', definition).get('required_occurrence_edits')
                    and not definition.get('worker_target', definition).get('protected_paths')),
               'maximum_occurrence_scale': definition.get('worker_target', definition).get('maximum_occurrence_scale', 1.0),
               'geometry_paths': [path for path in worker.resolved_geometry_paths(
                    definition.get('worker_target', definition).get('operation_scope', {}).get('geometry_paths', []),
                    records[definition['id']].get('path_map', {}), records[definition['id']].get('operations', []))
                    if path not in records[definition['id']].get('consumed_geometry_paths', [])],
               'matching': dict(matching[definition['id']], operation_assessment=records[definition['id']]['assessment'],
                                native_operations=records[definition['id']].get('operations',[]),
                                operation_path_map=records[definition['id']].get('path_map', {}))}
               for definition in nonroots if definition['id'] not in reused_results}
    published_dir = Path(common_library).resolve() if common_library else directory/'published'
    progress('library_publication')
    index = publisher.publish_libraries(plan, outputs, published_dir, consumer_layer=working['output']['path'], portable=working['request'].get('portable_output', False), geometry_reuse=True, geometry_policy=working['request'], progress=progress, material_library=working['output'].get('material_library'), common_library=bool(common_library), consumer_destination=directory/'after.usdc' if common_library else None)
    progress('library_publication_complete')
    final_consumer = published_dir/index['consumer_layer']
    working['output']['path'] = str(final_consumer)
    working['output']['material_library'] = (str(published_dir / index['material_library'])
                                              if index.get('material_library') else None)
    reverified, publication_execution = verify_published(working, records, index, published_dir,
        directory/'publication-verification', max_workers=max_workers, progress=progress)
    result['publication_verification_execution'] = {key: value for key, value in publication_execution.items() if key != 'results'}
    progress('composed_verification')
    composition = _load('successor_composition', scripts/'composition_verification.py')
    final_verification = composition.verify(working['source']['path'], str(final_consumer), working, reverified)
    progress('composed_verification_complete', passed=final_verification['passed'])
    stale=[]
    for path,expected in reconstruction.get('files',{}).items():
        if not Path(path).is_file() or hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:
            stale.append(path)
    reconstruction['passed'] = reconstruction['passed'] and not stale
    reconstruction['changed_since_verification'] = stale
    final_verification['checks'].append({'name':'original source through reconstructed source', 'passed':reconstruction['passed'], 'evidence':reconstruction})
    final_safety = tuning.safety_checks(str(final_consumer))
    final_verification['checks'].append({'name': 'published stage safety gates', 'passed': final_safety['passed'], 'evidence': final_safety})
    final_verification['passed'] = final_verification['passed'] and final_safety['passed'] and reconstruction['passed']
    passed = all(row['verification']['passed'] for row in reverified) and final_verification['passed']
    progress('output_profile')
    after_profile = profiler.profile(final_consumer, required_artifacts=[published_dir/'index.json'])
    comparison = profiler.compare_profiles(baseline, after_profile)
    tuning.write(directory/'after-profile.json', after_profile)
    tuning.write(directory/'profile-comparison.json', comparison)
    tuning.write(directory/'published-manifest.json', working)
    tuning.write(directory/'verification.json', final_verification)
    progress('report')
    report = _report(tuning, working, reverified, final_verification, directory, comparison, (baseline, after_profile), work_counts)
    passed = passed and report['passed']
    result.update(status='verified' if passed else 'verification_failure', passed=passed,
                  publication=index, manifest=working, definition_results=reverified,
                  verification=final_verification, report=report, profile_comparison=comparison,
                  after_layer=str(final_consumer), elapsed_seconds=time.monotonic()-started)
    tuning.write(directory/'workflow-results.json', result)
    progress('complete', passed=passed, status=result['status'])
    return result


def execute(manifest,directory,max_workers=4,common_library=None):
    """Preserve rejection exceptions while always reporting started attempts."""
    directory=Path(directory).resolve()
    if manifest.get('assessment_phase')=='boundary_review':
        raise ValueError('Resolve boundary review and finalize assessment before execution')
    if (directory/'workflow-results.json').exists():
        raise ValueError('use a fresh workflow output directory')
    scripts=Path(__file__).resolve().parent
    tuning=_load('workflow_failure_helpers',scripts/'usd_tuning.py')
    reporting=_load('workflow_failure_reporting',scripts/'failure_reporting.py')
    try:
        live=_load('successor_live_progress',scripts/'successor_progress.py')
        with live.Progress(manifest,directory,common_library=common_library) as progress_report:
            result=_execute(manifest,directory,max_workers,on_progress=progress_report.event,common_library=common_library)
            progress_report.finish(result.get('passed',False),result.get('status','incomplete'))
    except Exception as error:
        if directory.exists():
            retained_manifest = directory/'manifest.json'
            retained_results = directory/'definition-results.json'
            failure_manifest = (json.loads(retained_manifest.read_text())
                                if retained_manifest.is_file() else manifest)
            if not retained_results.is_file():
                retained_results = directory/'compute'/'compute-results.json'
            definition_results = (json.loads(retained_results.read_text()).get('definition_results', [])
                                  if retained_results.is_file() else [])
            reporting.finalize(tuning,failure_manifest,directory,{'passed':False,'status':'execution_exception',
                'definition_results':definition_results,
                'after_layer':failure_manifest.get('output',{}).get('path'),
                'errors':[type(error).__name__+': '+str(error)]})
        raise
    if not result['passed']:
        return reporting.finalize(tuning,result.get('manifest',manifest),directory,result)
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True); parser.add_argument('--workspace', required=True)
    parser.add_argument('--max-workers', type=int, default=4)
    parser.add_argument('--common-library', help='Shared publication directory; use one publisher at a time')
    args = parser.parse_args()
    result = execute(json.loads(Path(args.manifest).read_text()), args.workspace, args.max_workers, common_library=args.common_library)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
