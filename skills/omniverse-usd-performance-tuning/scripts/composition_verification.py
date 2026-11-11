# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check authored asset composition once; native geometry is trusted execution.

Usage:
    verify(source_layer, output_layer, manifest, results), called by the workflow.
    This module has no command-line interface.
Arguments:
    Source and output layer paths, the assessed manifest, and definition results.
Output:
    A dictionary with passed, checks and errors. Uncomposable stages, changed
    dependencies, invalid definition results and missing interfaces fail checks.
Exit codes:
    Not applicable to this imported module. Verification failures are returned
    as findings; unexpected USD or file-access exceptions propagate.
"""
import copy
import hashlib
import json
import importlib.util
from pathlib import Path


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def material_fingerprints(prim, discovery, cache, material_source_paths):
    from pxr import Sdf, UsdGeom, UsdShade
    def material_target_map(prim):
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if not material:return {}
        current = str(material.GetPath())
        for spec in material.GetPrim().GetPrimStack():
            authored = str(spec.path)
            original = material_source_paths.get(authored)
            if original is None:continue
            root = str(Sdf.Path(authored).GetPrefixes()[0])
            suffix = authored[len(root):]
            # Authoring wrapped the original namespace under the material library.
            # Derive its composed prefix from the recorded mapping and prim stack.
            if suffix == original and current.endswith(suffix):
                prefix = current[:-len(suffix)]
                if prefix:return {prefix: ''}
        return {}
    def materials(prim):
        if not prim:return set()
        consumers=[prim]+[q for q in prim.GetChildren() if q.IsA(UsdGeom.Subset)]
        return {discovery._effective_appearance(q,cache,material_target_map(q))[0].get('network_fingerprint')
                for q in consumers}
    return materials(prim) - {None}

def verify(source_layer, output_layer, manifest, results):
    from pxr import Sdf, Usd, UsdGeom, UsdShade
    import numpy as np
    root = Path(__file__).resolve().parents[1]
    conformance = _load('composition_structure', root/'references/optimization-report/scripts/structure_conformance.py')
    animation = _load('composition_animation', root/'scripts/animation_context.py')
    budget = _load('composition_budget', root/'references/geometry_tolerance.py').physical_tolerance_mm(manifest.get('request'))
    checks = []
    def check(name, passed, evidence=None):
        checks.append({'name': name, 'passed': bool(passed), 'evidence': evidence})
    source = Usd.Stage.Open(str(source_layer)); output = Usd.Stage.Open(str(output_layer))
    check('composed output', source and output and not output.GetCompositionErrors())
    if not checks[-1]['passed']:
        return {'passed': False, 'checks': checks, 'errors': ['Source or output cannot compose']}
    changed=[]
    for dependency in manifest.get('inventory', {}).get('dependencies', []):
        file=Path(dependency['path'])
        if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest()!=dependency['fingerprint']:
            changed.append(str(file))
    check('original source dependencies unchanged', not changed, changed)
    repairs=manifest.get('material_target_repairs', {}).get('repaired', [])
    if repairs:
        # Recompute the narrow repair from original composition. The manifest
        # cannot authorize an arbitrary replacement shader or new appearance.
        helper=_load('composition_material_repair', root/'references/usd-optimize-run-operations/scripts/material_publication.py')
        with Usd.EditContext(source, source.GetSessionLayer()):
            confirmed=helper.repair_shader_targets(source)
        check('recorded material target repairs', confirmed['repaired']==repairs, confirmed)

    check('stage conventions', UsdGeom.GetStageMetersPerUnit(source)==UsdGeom.GetStageMetersPerUnit(output)
          and UsdGeom.GetStageUpAxis(source)==UsdGeom.GetStageUpAxis(output))
    definitions={d['id']:d for d in manifest['definitions']}
    records={row['definition_id']:row for row in results}
    check('complete unique definition results', len(records)==len(results) and set(records)==set(definitions))
    request_hash=hashlib.sha256(json.dumps(manifest.get('request',{}),sort_keys=True).encode()).hexdigest()
    for identifier,row in records.items():
        errors=conformance.check_definition_result(row, require_categories=manifest.get('schema_version')==2)
        if manifest.get('schema_version')==2:
            target=definitions.get(identifier,{}).get('worker_target',definitions.get(identifier,{}))
            target_hash=hashlib.sha256(json.dumps(target,sort_keys=True).encode()).hexdigest()
            expected={'request_sha256':request_hash,'target_sha256':target_hash}
            if row.get('verification_context')!=expected:
                errors.append('execution request or scope changed')
        check('definition outcome '+identifier, not errors, errors)
    mapping={}
    optimized_materials={}
    written=copy.deepcopy(manifest)
    for definition in written['definitions']:
        boundary=definition['boundary_path']; local=definition.get('output_prim_path','/Asset')
        operation_map=records.get(definition['id'],{}).get('path_map',{})
        for defined_source in definition.get('protected_paths', []):
            old = local + defined_source[len(boundary):]
            destination = operation_map.get(old, old)
            if not old.startswith(local + '/') or not destination.startswith(local + '/'):
                continue
            for occurrence in manifest['occurrences']:
                if occurrence['definition_id'] != definition['id']:
                    continue
                inverse = {v:k for k,v in occurrence.get('source_to_definition', {}).items()}
                original = inverse.get(defined_source, occurrence['path'] + old[len(local):])
                mapping[original] = occurrence['path'] + destination[len(local):]
        consumed=set(records.get(definition['id'],{}).get('consumed_geometry_paths',[]))
        final_geometry=records.get(definition['id'],{}).get('output_geometry_paths')
        remapped=[]
        for owned in definition.get('owned_geometry',[]):
            local_source=local+owned[len(boundary):]
            destination=operation_map.get(local_source,local_source)
            if local_source in consumed or destination in consumed:
                continue
            if final_geometry is not None and destination not in final_geometry:
                # Native clustered merges record output membership as a group.
                # Removed static inputs have no individual output correspondence.
                continue
            if not (destination==local or destination.startswith(local+'/')):
                check('owned output remains within definition',False,owned)
                continue
            remapped.append(boundary+destination[len(local):])
            for occurrence in manifest['occurrences']:
                if occurrence['definition_id']!=definition['id']:continue
                inverse={v:k for k,v in occurrence.get('source_to_definition',{}).items()}
                original=inverse.get(owned)
                if original is None and 'source_owned_geometry' not in occurrence:
                    original=occurrence['path']+owned[len(boundary):]
                if original is not None:
                    mapping[original]=occurrence['path']+destination[len(local):]
                publication=records.get(definition['id'],{}).get('publication_verification',{})
                if 'material_bindings' in publication and final_geometry is None:
                    bindings=publication['material_bindings']
                    if destination not in bindings:
                        check('published material baseline covers owned geometry',False,destination)
                    optimized_materials[occurrence['path']+destination[len(local):]]=set(bindings.get(destination,[]))
        if final_geometry is not None:
            remapped=[]
            publication=records[definition['id']].get('publication_verification',{})
            bindings=publication.get('material_bindings',{})
            for destination in final_geometry:
                if not destination.startswith(local+'/') and destination != local:
                    check('owned output remains within definition',False,destination)
                    continue
                remapped.append(boundary+destination[len(local):])
                check('published material baseline covers owned geometry',destination in bindings,destination)
                for occurrence in manifest['occurrences']:
                    if occurrence['definition_id']==definition['id']:
                        optimized_materials[occurrence['path']+destination[len(local):]]=set(bindings.get(destination,[]))
        definition['owned_geometry']=sorted(set(remapped))
    for occurrence in written['occurrences']:
        occurrence.pop('source_to_definition',None)
    references=[]
    for occurrence in manifest['occurrences']:
        definition=definitions[occurrence['definition_id']]
        if 'output_layer' not in definition:continue
        prim=output.GetPrimAtPath(occurrence['path'])
        expected_layer=str(Path(definition['output_layer']).resolve())
        expected_path=definition.get('output_prim_path','/Asset')
        if not prim or not any(spec.layer.realPath and str(Path(spec.layer.realPath).resolve())==expected_layer
                               and str(spec.path)==expected_path for spec in prim.GetPrimStack()):
            references.append(occurrence['path'])
    check('occurrences reference their published definitions',not references,references)
    geometry=[{'path':str(p.GetPath()),'schema':p.GetTypeName()} for p in Usd.PrimRange.Stage(output,Usd.TraverseInstanceProxies())
              if p.IsA(UsdGeom.Gprim) or p.IsA(UsdGeom.PointInstancer)]
    structure=conformance.check_manifest(written, inventory={'geometry':geometry,'uninspected':[]}, stage=output)
    check('written structure and geometry ownership', structure['passed'], structure)
    motion=animation.verify(source, output, path_map=mapping)
    check('authored animation',motion['passed'],motion)
    # We author occurrence frames; native fitting/merging may legitimately author
    # geometry frames. Check only retained asset and application interfaces here.
    interfaces={o['path'] for o in manifest['occurrences']}
    for d in manifest['definitions']:
        interfaces.update(d.get('protected_paths',[]))
    requirements = manifest.get('required_occurrence_edits', [])
    placement_paths = {str(Sdf.Path(q['path']).GetPrimPath()) for q in requirements if q.get('edit_class', 'placement') == 'placement'}
    nonplacement = {str(Sdf.Path(q['path']).GetPrimPath()) for q in requirements if q.get('edit_class') in ('identification', 'visibility')}
    animated_paths = {str(Sdf.Path(row['property_path']).GetPrimPath()) for row in motion['checks']
                      if row.get('property_path') and Sdf.Path(row['property_path']).name.startswith('xformOp')}
    interfaces.difference_update(nonplacement - placement_paths - {o['path'] for o in manifest['occurrences']} - animated_paths)
    interfaces.update(placement_paths)
    occurrence_frames = {}
    for occurrence in manifest['occurrences']:
        publication = records.get(occurrence['definition_id'], {}).get('publication_verification', {})
        frame = publication.get('geometry_root_transform')
        prim = source.GetPrimAtPath(occurrence['path'])
        if publication.get('passed') and frame is not None and prim and prim.IsA(UsdGeom.Gprim):
            occurrence_frames[occurrence['path']] = np.asarray(frame)
    failures=[]
    for sample in motion['time_samples']:
        time=Usd.TimeCode.Default() if sample is None else Usd.TimeCode(sample)
        left_cache=UsdGeom.XformCache(time);right_cache=UsdGeom.XformCache(time)
        for path in sorted(interfaces):
            a=source.GetPrimAtPath(path); b=output.GetPrimAtPath(mapping.get(path,path))
            if not a or not b:
                failures.append({'path':path,'reason':'missing interface'});continue
            left=np.asarray(left_cache.GetLocalToWorldTransform(a));right=np.asarray(right_cache.GetLocalToWorldTransform(b))
            if path in occurrence_frames:
                left = occurrence_frames[path] @ left
            mm=UsdGeom.GetStageMetersPerUnit(source)*1000
            if not np.allclose(left[:3,:3],right[:3,:3],rtol=0,atol=1e-9) or np.linalg.norm(left[3,:3]-right[3,:3])*mm>budget:
                failures.append({'path':path,'reason':'authored occurrence placement changed','time':sample})
            for relation in a.GetAuthoredRelationships():
                if relation.GetName().startswith('material:binding'):continue
                other=b.GetRelationship(relation.GetName())
                if not other or list(map(str,other.GetTargets()))!=list(map(str,relation.GetTargets())):
                    failures.append({'path':path,'reason':'interface relationship changed'})
    check('authored occurrence placement and relationships',not failures,failures)
    requirements_helper = _load('composition_required_uses', root/'references/usd-structure-assessment/scripts/boundary_reasoning.py')
    edits=[]
    for requirement in requirements:
        path = Sdf.Path(requirement['path'])
        destination = path.ReplacePrefix(path.GetPrimPath(), Sdf.Path(mapping.get(str(path.GetPrimPath()), str(path.GetPrimPath()))))
        valid = requirements_helper.valid_required_use(output, dict(requirement, path=str(destination)), output=True)
        before = source.GetObjectAtPath(path); after = output.GetObjectAtPath(destination)
        kind = requirement.get('edit_class', 'placement')
        if valid and kind == 'identification':
            if path.IsPrimPath():
                valid = all(before.GetMetadata(key) == after.GetMetadata(key)
                            for key in ('displayName', 'assetInfo') if before.HasAuthoredMetadata(key))
            elif isinstance(before, Usd.Attribute):
                valid = isinstance(after, Usd.Attribute) and all(before.Get(t) == after.Get(t)
                    for t in [Usd.TimeCode.Default()] + [Usd.TimeCode(t) for t in before.GetTimeSamples()])
            elif isinstance(before, Usd.Relationship):
                valid = isinstance(after, Usd.Relationship) and before.GetTargets() == after.GetTargets()
        if valid and kind == 'visibility':
            a = UsdGeom.Imageable(source.GetPrimAtPath(path.GetPrimPath())).GetVisibilityAttr()
            b = UsdGeom.Imageable(output.GetPrimAtPath(destination.GetPrimPath())).GetVisibilityAttr()
            valid = all(a.Get(t) == b.Get(t) for t in [Usd.TimeCode.Default()] + [Usd.TimeCode(t) for t in a.GetTimeSamples()])
        edits.append({'path':requirement['path'],'passed':bool(valid)})
    check('required editable interfaces',all(q['passed'] for q in edits),edits)
    discovery = _load('composition_materials', root/'references/usd-structure-assessment/scripts/structure_assessment.py')
    caches = ({}, {})
    # Binding checks concern our remapping, not native face redistribution.
    required=dict(optimized_materials)
    for original,destination in mapping.items():
        if destination not in optimized_materials:
            required.setdefault(destination,set()).update(material_fingerprints(source.GetPrimAtPath(original),discovery,caches[0],{}))
    missing=[]
    for destination,expected in required.items():
        actual=material_fingerprints(output.GetPrimAtPath(destination),discovery,caches[1],manifest.get('material_source_paths',{}))
        if expected != actual:
            missing.append(destination)
    check('material binding remaps',not missing,missing)
    targets=[]
    for row in manifest.get('occurrence_interfaces',[]):
        prop=output.GetObjectAtPath(row['property_path'])
        actual=(prop.GetTargets() if row['property_type']=='relationship' else prop.GetConnections()) if prop else []
        if list(map(str,actual))!=row['targets'] or any(not output.GetObjectAtPath(p) for p in actual):
            targets.append(row['property_path'])
    check('authored relationship and connection targets',not targets,targets)

    return {'passed':all(row['passed'] for row in checks),'checks':checks,'errors':[],
            'method':'authored_composition_and_native_execution',
            'independent_geometry_proof':False,
            'limitations':['Native geometry operations are trusted; configured tolerances are not independently measured deviation.',
                           'Detailed geometry and edit-isolation comparisons belong to development qualification.']}
