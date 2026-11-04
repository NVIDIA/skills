# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Consolidate compatible exact or normalized leaf inputs before native computation."""
import json
from pathlib import Path
from pxr import Gf, Sdf, Usd


def consolidate(manifest, prepared, worker):
    identity = worker.load_helper('current_run_identity', 'usd-optimize-run-operations/scripts/persistent_library.py')
    definitions = {d['id']: d for d in manifest['definitions']}
    occurrences = manifest['occurrences']
    by_path = {o['path']: o for o in occurrences}
    pinned = set()
    for occurrence in occurrences:
        parent = by_path.get(occurrence.get('parent_path'))
        if parent and prepared.get(parent['definition_id'], {}).get('entry'):
            pinned.add(occurrence['definition_id'])
    representatives, aliases, receipts, selected_stages = {}, {}, [], {}
    for identifier, confirmation in prepared.items():
        definition = definitions[identifier]
        target = definition.get('worker_target', definition)
        matching = confirmation['matching']
        root = target.get('boundary_path', '/Asset')
        if (identifier in pinned or confirmation.get('entry') or target.get('structural_type') != 'atomic'
            or target.get('children') or target.get('child_boundaries')
            or not matching.get('resource_closure_complete')):
            continue
        info = json.loads(matching['normalized_source_info']) if matching.get('normalized_source_key') else None
        local, categories = worker.category_selection(target, manifest['request'], bool(target.get('geometry_counts', {}).get('Mesh')))
        def policy_for(mode):
            def canonical(value):
                if isinstance(value, str):
                    return value.replace(root + info['child'], '/Asset/Content') if mode == 'normalized' else value.replace(root, '/Asset', 1)
                if isinstance(value, dict): return {k:canonical(v) for k,v in value.items()}
                if isinstance(value, list): return [canonical(v) for v in value]
                return value
            return [target.get('geometry_representation', 'merged_meshes'),
                    [(c['category'], c['status']) for c in categories],
                    canonical([{k:v for k,v in group.items() if k not in ('output_path','reason')} for group in target.get('merge_groups', [])]),
                    canonical(target.get('operation_scope', {})), bool(local.get('decimate')), bool(local.get('fit_primitives')),
                    sorted(canonical(target.get('protected_paths', []))),
                    sorted((canonical(e['path']),e.get('edit_class','placement')) for e in target.get('required_occurrence_edits', [])
                           if e['path'] != root or e.get('edit_class','placement') != 'placement')]
        options = [('product', matching.get('product_key')), ('exact', matching.get('source_signature'))]
        if info and not any(p != root for p in target.get('protected_paths', [])) and not any(
                e.get('path') != root or e.get('edit_class', 'placement') != 'placement' for e in target.get('required_occurrence_edits', [])):
            options.append(('normalized', matching['normalized_source_key']))
        representative = None
        keys=[]
        for mode, signature in options:
            if not signature: continue
            policy = policy_for(mode)
            if mode == 'product': policy = [policy[0],policy[-2],policy[-1]]
            key = (mode, '' if mode=='product' else root, signature, '' if mode=='product' else matching['root_transform_signature'], matching['metersPerUnit'], json.dumps(policy, sort_keys=True))
            keys.append(key)
            if key in representatives:
                representative=representatives[key]
                break
        if representative is None:
            for key in keys: representatives[key]=identifier
            continue
        previous_matching=prepared[representative]['matching']
        if identity.source_identity_kind(matching,previous_matching) != mode:
            continue
        selected = definitions[representative]
        selected_target=selected.get('worker_target',selected)
        selected_root=selected_target.get('boundary_path','/Asset')
        required=set(target.get('protected_paths',[])) | {e['path'] for e in target.get('required_occurrence_edits',[])}
        if required:
            if representative not in selected_stages:
                selected_stages[representative]=Usd.Stage.Open(selected['input_layer'])
            selected_stage=selected_stages[representative]
            if not selected_stage or selected_stage.GetCompositionErrors() or any(
                    not selected_stage.GetPrimAtPath(selected_root+p[len(root):]) for p in required):
                continue
        if mode == 'product':
            animation=worker.load_helper('current_product_animation','../scripts/animation_context.py')
            incoming_stage=Usd.Stage.Open(definition['input_layer'])
            selected_stage=selected_stages.setdefault(representative,Usd.Stage.Open(selected['input_layer']))
            if not animation.verify(incoming_stage,selected_stage,paths=[root],path_map={root:selected_root})['passed']:
                continue
        # Occurrence scale controls the local error budget, not asset identity.
        # Optimize the shared definition for the largest retained occurrence.
        scale = max(selected_target.get('maximum_occurrence_scale', 1.0),
                    target.get('maximum_occurrence_scale', 1.0))
        selected['maximum_occurrence_scale'] = scale
        selected_target['maximum_occurrence_scale'] = scale
        previous_matching['tolerance_mm'] = min(previous_matching['tolerance_mm'], matching['tolerance_mm'])
        previous = json.loads(previous_matching['normalized_source_info']) if mode == 'normalized' else None
        matrix = lambda rows: Gf.Matrix4d(*[v for row in rows for v in row])
        delta = matrix(previous['matrix']).GetInverse() * matrix(info['matrix']) if mode == 'normalized' else None
        for occurrence in occurrences:
            if occurrence['definition_id'] != identifier: continue
            occurrence['definition_id'] = representative
            if mode == 'normalized':
                existing = occurrence.get('internal_placement', {}).get('matrix')
                correction = delta * matrix(existing) if existing else delta
                occurrence['internal_placement'] = {'matrix':[list(row) for row in correction], 'basis':'within-run normalized source identity'}
            original_mapping = occurrence.get('source_to_definition', {})
            inverse_mapping = {value:key for key,value in original_mapping.items()}
            if mode == 'product':
                # Product substitution need not preserve the source mesh names or
                # tessellation. Keep its assessed source ownership separately.
                occurrence['source_owned_geometry'] = [
                    inverse_mapping.get(path, occurrence['path'] + path[len(definition['boundary_path']):])
                    for path in definition.get('owned_geometry', [])]
            mapping = {}
            for source_path, path in list(original_mapping.items()) + [
                (inverse_mapping.get(path, occurrence['path'] + path[len(definition['boundary_path']):]), path)
                for path in definition.get('owned_geometry', [])]:
                relative = path[len(definition['boundary_path']):]
                if mode == 'product' and root + relative not in required:
                    continue
                canonical_relative = previous['child'] + relative[len(info['child']):] if mode == 'normalized' and relative.startswith(info['child'] + '/') else relative
                mapping[source_path] = selected['boundary_path'] + canonical_relative
            mapping[occurrence['path']] = selected['boundary_path']
            occurrence['source_to_definition'] = mapping
            selected.setdefault('occurrences', []).append(occurrence['path'])
        selected.setdefault('reference_layers', []).append(definition['input_layer'])
        aliases[identifier] = representative
        receipt={'definition_id':identifier, 'representative_definition_id':representative,
                 'disposition':'within_run_reuse', 'identity':mode}
        if mode == 'exact':
            receipt['exact_key']=identity.exact_key(matching)
        elif mode == 'product':
            receipt.update(product_key=matching['product_key'],product_identity=json.loads(matching['product_identity_json']),representation_substitution=True)
        else:
            receipt.update(normalized_source_key=matching['normalized_source_key'], authorization=info['authorization'],
                           incoming_colour_changes=info['colour_changes'], representative_colour_changes=previous['colour_changes'])
        receipts.append(receipt)
    if not aliases: return []
    manifest['definitions'] = [d for d in manifest['definitions'] if d['id'] not in aliases]
    for definition in manifest['definitions']:
        for target in [definition, definition.get('worker_target', {})]:
            for field in ('children', 'compute_requires'):
                if field in target: target[field] = list(dict.fromkeys(aliases.get(v,v) for v in target[field]))
    for geometry in manifest.get('inventory', {}).get('geometry', []):
        if geometry.get('definition_id') in aliases:
            geometry['definition_id'] = aliases[geometry['definition_id']]
    for group in manifest.get('reuse_candidates', []):
        if 'definition_ids' in group:
            group['definition_ids'] = list(dict.fromkeys(aliases.get(v,v) for v in group['definition_ids']))
    if 'phase4_targets' in manifest:
        manifest['phase4_targets'] = [target for target in manifest['phase4_targets'] if target.get('id') not in aliases]
    manifest['within_run_reuse'] = receipts
    manifest['within_run_reference_replacements'] = {str(Path(definitions[a]['output_layer']).resolve()):definitions[b]['output_layer'] for a,b in aliases.items()}
    return receipts


def redirect_outputs(manifest):
    # Authoring already created parent references. Redirect only generated output
    # layers; immutable inputs remain available for preservation verification.
    replacements = manifest.get('within_run_reference_replacements', {})
    if not replacements: return
    files = {d['output_layer'] for d in manifest['definitions'] if not d.get('persistent_entry')} | {manifest['output']['path']}
    for filename in files:
        layer=Sdf.Layer.FindOrOpen(filename);changed=[False]
        def visit(path):
            prim=layer.GetPrimAtPath(path)
            if not prim:return
            for editor, constructor in [(prim.referenceList,Sdf.Reference),(prim.payloadList,Sdf.Payload)]:
                for field in ('explicitItems','prependedItems','appendedItems','addedItems','deletedItems','orderedItems'):
                    values=list(getattr(editor,field));mapped=[]
                    for value in values:
                        resolved=str(Path(Sdf.ComputeAssetPathRelativeToLayer(layer,value.assetPath)).resolve()) if value.assetPath else ''
                        replacement=replacements.get(resolved)
                        if replacement:
                            mapped.append(constructor(replacement,value.primPath,value.layerOffset,value.customData) if constructor is Sdf.Reference else constructor(replacement,value.primPath,value.layerOffset));changed[0]=True
                        else:mapped.append(value)
                    if mapped!=values:setattr(editor,field,mapped)
        layer.Traverse(Sdf.Path.absoluteRootPath,visit)
        if changed[0]:layer.Save()
