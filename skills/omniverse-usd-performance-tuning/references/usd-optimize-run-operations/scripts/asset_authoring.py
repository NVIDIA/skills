# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Author definitions and reference sites from an assessed manifest."""
from __future__ import annotations

import copy
from collections import defaultdict
import hashlib
import importlib.util
from pathlib import Path


def _mapped(path, root):
    from pxr import Sdf
    return str(Sdf.Path(path).ReplacePrefix(Sdf.Path(root), Sdf.Path('/Asset')))


def _nearest_occurrence_owner(path, boundaries):
    """Find the deepest selected ancestor in a set of prebuilt Sdf.Paths."""
    while path:
        if path in boundaries:
            return str(path)
        path = path.GetParentPath()
    return None


def _pack_helpers():
    spec = importlib.util.spec_from_file_location(
        '_usd_optimize_pack_definitions', Path(__file__).with_name('pack_definitions.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _placement(prim):
    """Keep occurrence-authored transform properties outside the shared definition."""
    return [(a.GetName(), a.GetTypeName(), a.Get(),
             [(t, a.Get(t)) for t in a.GetTimeSamples()])
            for a in prim.GetAttributes() if a.GetName().startswith('xformOp')]


def _child_occurrence_placements(stage, source, occurrences):
    """Keep retained body frames on the consumer, including shared-parent children."""
    from pxr import Sdf, UsdGeom
    paths = set()
    for occurrence in occurrences:
        parent = occurrence.get('parent_path')
        if not parent:
            continue
        path = Sdf.Path(occurrence['path'])
        while str(path) != parent and path != Sdf.Path.absoluteRootPath:
            paths.add(str(path))
            path = path.GetParentPath()
    for path in sorted(paths, key=lambda value: value.count('/')):
        original = source.GetPrimAtPath(path)
        # Native fitting can encode geometry dimensions and offset in the root
        # transform. Restore equipment/body frames, not geometry transforms.
        if not UsdGeom.Xformable(original) or original.IsA(UsdGeom.Gprim):
            continue
        written = stage.OverridePrim(path)
        # An explicit order also clears a cached transform on an identity frame.
        written.CreateAttribute('xformOpOrder', Sdf.ValueTypeNames.TokenArray).Set(
            UsdGeom.Xformable(original).GetXformOpOrderAttr().Get() or [])
        for name, typ, default, samples in _placement(original):
            attr = written.CreateAttribute(name, typ)
            if default is not None:
                attr.Set(default)
            for time, value in samples:
                attr.Set(value, time)


def _internal_placement(stage, occurrence, source, definition_root, target_root,
                        canonical_stage=None, canonical_root=None):
    """Put the verified common delta above surviving geometry, preserving the root."""
    from pxr import Sdf, Usd, UsdGeom, Gf
    if 'internal_placement' not in occurrence:
        return
    canonical_stage = canonical_stage or source
    canonical_root = canonical_root or definition_root
    stage.GetPrimAtPath(target_root).SetInstanceable(False)
    delta=Gf.Matrix4d(*[value for row in occurrence['internal_placement']['matrix'] for value in row])
    def placement_op(xform,suffix):
        name=suffix;index=1
        while xform.GetPrim().GetAttribute('xformOp:transform:'+name):
            index+=1;name=suffix+str(index)
        return xform.AddTransformOp(opSuffix=name)
    opinions=[];placed=[]
    for child in canonical_stage.GetPrimAtPath(canonical_root).GetChildren():
        if not any(prim.IsA(UsdGeom.Gprim) for prim in Usd.PrimRange(child)):
            continue
        xform=UsdGeom.Xformable(child)
        if not xform or xform.GetResetXformStack():
            raise ValueError('common placement requires transformable geometry branches')
        path=str(child.GetPath().ReplacePrefix(Sdf.Path(canonical_root),Sdf.Path(target_root)))
        written=UsdGeom.Xformable(stage.OverridePrim(path))
        if not written:
            # The verified merged path may only compose after publication rewrites references.
            stage.OverridePrim(path).SetTypeName(child.GetTypeName())
            written=UsdGeom.Xformable(stage.GetPrimAtPath(path))
        op=placement_op(written,'internalPlacement')
        op.Set(xform.GetLocalTransformation()*delta)
        written.SetXformOpOrder([op])
        opinions.append({'path':path,'attributes':[op.GetOpName(),'xformOpOrder'],'metadata':[]})
        placed.append(child.GetPath())
    retained=[]
    for original,canonical in occurrence['source_to_definition'].items():
        if original==occurrence['path']:continue
        local=str(Sdf.Path(canonical).ReplacePrefix(Sdf.Path(definition_root),Sdf.Path(canonical_root)))
        current=canonical_stage.GetPrimAtPath(local)
        if not current:continue
        prim=source.GetPrimAtPath(original)
        if prim.GetMetadata('displayName')==current.GetMetadata('displayName'):continue
        path=str(Sdf.Path(local).ReplacePrefix(Sdf.Path(canonical_root),Sdf.Path(target_root)))
        stage.OverridePrim(path).SetMetadata('displayName',prim.GetMetadata('displayName'))
        identity_attributes=[]
        if UsdGeom.Xformable(current) and not any(p.IsA(UsdGeom.Gprim) for p in Usd.PrimRange(current)):
            # A surviving metadata-only interface has no geometry to place after
            # merging. Preserve its own source frame without reviving any mesh.
            source_cache=UsdGeom.XformCache();canonical_cache=UsdGeom.XformCache()
            desired=source_cache.GetLocalToWorldTransform(prim)*source_cache.GetLocalToWorldTransform(source.GetPrimAtPath(occurrence['path'])).GetInverse()
            parent_frame=canonical_cache.GetLocalToWorldTransform(current.GetParent())*canonical_cache.GetLocalToWorldTransform(canonical_stage.GetPrimAtPath(canonical_root)).GetInverse()
            if any(current.GetParent().GetPath().HasPrefix(branch) for branch in placed):parent_frame=parent_frame*delta
            written=UsdGeom.Xformable(stage.GetPrimAtPath(path))
            op=placement_op(written,'internalIdentity');op.Set(desired*parent_frame.GetInverse())
            written.SetXformOpOrder([op]);identity_attributes=[op.GetOpName(),'xformOpOrder']
        opinions.append({'path':path,'attributes':identity_attributes,'metadata':['displayName']})
        retained.append(original)
    occurrence['internal_placement_opinions']=opinions
    occurrence['retained_internal_metadata_paths']=retained


def reconcile_internal_placements(manifest):
    """Rebuild occurrence opinions against verified native output, after merging.

    Ordinary CAD paths are not protected. Old placement opinions are removed before
    targeting surviving output branches; immutable inputs and inherited libraries
    remain read-only.
    """
    from pxr import Usd, Sdf
    if not any('internal_placement' in row for row in manifest.get('occurrences',[])):return []
    session=Sdf.Layer.CreateAnonymous()
    source=Usd.Stage.Open(Sdf.Layer.FindOrOpen(manifest['source']['path']),session)
    source.SetEditTarget(session)
    for path,selections in manifest.get('context',{}).get('variants',{}).items():
        for name,selection in selections.items():
            variant=source.GetPrimAtPath(path).GetVariantSet(name)
            if selection not in variant.GetVariantNames():raise ValueError('missing placement variant context')
            variant.SetVariantSelection(selection)
    definitions={row['id']:row for row in manifest['definitions']}
    occurrences={row['path']:row for row in manifest['occurrences']}
    stages={};evidence=[]
    for occurrence in manifest['occurrences']:
        if 'internal_placement' not in occurrence:continue
        definition=definitions[occurrence['definition_id']]
        parent=occurrences.get(occurrence.get('parent_path'))
        if parent:
            owner=definitions[parent['definition_id']]
            if owner.get('persistent_entry'):
                raise ValueError('inherited package placement requires a separate local occurrence layer')
            layer=owner['output_layer']
            target_root=str(Sdf.Path(occurrence['path']).ReplacePrefix(Sdf.Path(parent['path']),Sdf.Path(owner.get('output_prim_path','/Asset'))))
        else:
            layer=manifest['output']['path'];target_root=occurrence['path']
        if layer not in stages:stages[layer]=Usd.Stage.Open(layer)
        stage=stages[layer]
        old=occurrence.get('internal_placement_opinions',[])
        for opinion in sorted(old,key=lambda row:-row['path'].count('/')):
            prim=stage.GetPrimAtPath(opinion['path'])
            if not prim:continue
            for name in opinion['attributes']:prim.RemoveProperty(name)
            for name in opinion['metadata']:prim.ClearMetadata(name)
            spec=stage.GetRootLayer().GetPrimAtPath(opinion['path'])
            if spec and not spec.properties and not spec.nameChildren and set(spec.ListInfoKeys())<={'specifier','typeName'}:
                stage.RemovePrim(opinion['path'])
        entry=definition.get('persistent_entry') or {}
        canonical=Usd.Stage.Open(entry.get('layer',definition['output_layer']))
        canonical_root=entry.get('prim_path',definition.get('output_prim_path','/Asset'))
        _internal_placement(stage,occurrence,source,definition['boundary_path'],target_root,canonical,canonical_root)
        evidence.append({'occurrence':occurrence['path'],'owner_definition_id':parent['definition_id'] if parent else None,'previous_opinion_sites':len({row['path'] for row in old}),
                         'output_opinion_sites':len({row['path'] for row in occurrence['internal_placement_opinions']})})
    for stage in stages.values():stage.GetRootLayer().Save()
    return evidence


def _reference(stage, path, layer, placement, kind, interfaces=(), source_root=None, target_root=None, payload=False, source_prim=None, instanceable=False, target_prim_path='/Asset'):
    from pxr import UsdGeom, Sdf
    stage.RemovePrim(path)
    prim = stage.DefinePrim(path)
    if payload:
        prim.GetPayloads().AddPayload(str(layer), target_prim_path)
    else:
        prim.GetReferences().AddReference(str(layer), target_prim_path)
    for name, typ, default, samples in placement:
        attr = prim.CreateAttribute(name, typ)
        if default is not None:
            attr.Set(default)
        for time, value in samples:
            attr.Set(value, time)
    if kind:
        prim.SetMetadata('kind', kind)
    if source_prim:
        for name in ('customData', 'assetInfo', 'displayName', 'documentation', 'hidden'):
            if source_prim.HasAuthoredMetadata(name):
                prim.SetMetadata(name, source_prim.GetMetadata(name))
        for original in source_prim.GetAuthoredAttributes():
            name = original.GetName()
            if not (original.IsCustom() or name.startswith(('physics:', 'physx', 'semantic', 'asset:', 'userProperties:'))):
                continue
            attr = prim.CreateAttribute(name, original.GetTypeName(), custom=original.IsCustom())
            value = original.Get()
            if value is not None:
                attr.Set(value)
            for sample in original.GetTimeSamples():
                attr.Set(original.Get(sample), sample)
    for name, targets in interfaces:
        mapped = [Sdf.Path(p).ReplacePrefix(Sdf.Path(source_root), Sdf.Path(target_root))
                  if source_root and target_root and Sdf.Path(p).HasPrefix(Sdf.Path(source_root)) else Sdf.Path(p)
                  for p in targets]
        prim.CreateRelationship(name).SetTargets(mapped)
    if instanceable:
        prim.SetInstanceable(True)
    return prim


def _interfaces(prim):
    return [(r.GetName(), [str(p) for p in r.GetTargets()]) for r in prim.GetAuthoredRelationships()
            if not r.GetName().startswith("material:binding")]


def _type_material_scopes(stage, paths):
    """Type authored material namespaces without promoting inherited overs."""
    from pxr import Sdf
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        spec = stage.GetRootLayer().GetPrimAtPath(path)
        if prim and not prim.GetTypeName() and spec and spec.specifier == Sdf.SpecifierDef:
            prim.SetTypeName('Scope')


def check_material_policy(manifest):
    """This author implements shared-library extraction, not legacy policy modes.

    An omitted policy selects the shared-library path. An explicit historical
    policy must use its owning extraction tool; never silently reinterpret it.
    """
    records = [('manifest', manifest), ('request', manifest.get('request') or {})]
    for collection in ('definitions', 'targets'):
        for record in manifest.get(collection, []):
            records.append((record.get('id', collection), record))
            for nested in ('request', 'worker_target', 'extraction'):
                if isinstance(record.get(nested), dict):
                    records.append((record.get('id', collection) + '.' + nested, record[nested]))
    for location, record in records:
        if 'material_policy' in record and record['material_policy'] != 'shared_library':
            raise ValueError(f'unsupported explicit material_policy at {location}: '
                             f'{record["material_policy"]!r}; this author supports shared_library only')


def _unmodified_material_aliases(source):
    """Reuse an existing base material only through a pure specialization wrapper."""
    from pxr import Usd,UsdShade
    aliases,identities,closed={},{},{}
    def identity(spec):
        return (spec.layer.identifier,str(spec.path))
    for prim in Usd.PrimRange.Stage(source,Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdShade.Material):
            continue
        candidate=prim.GetPrimInPrototype() if prim.IsInstanceProxy() else prim
        key=str(candidate.GetPath())
        if key not in identities:
            base=UsdShade.Material(candidate).GetBaseMaterialPath()
            target=source.GetPrimAtPath(base) if base else None
            chosen=None
            if target and target.IsA(UsdShade.Material) and not target.IsInstanceProxy() and not target.IsInPrototype():
                stack=candidate.GetPrimStack();base_stack=target.GetPrimStack()
                if (len(stack)==len(base_stack)+1 and
                        list(map(identity,stack[1:]))==list(map(identity,base_stack))):
                    wrapper=stack[0]
                    pure=(set(wrapper.ListInfoKeys())<={'specifier','specializes'} and
                          not wrapper.properties and not wrapper.nameChildren and
                          list(wrapper.specializesList.GetAppliedItems())==[base])
                    if pure:
                        if base not in closed:
                            closed[base]=all(
                                (node==target or not node.IsA(UsdShade.Material)) and all(path.HasPrefix(base)
                                    for prop in node.GetProperties()
                                    for path in (prop.GetTargets() if isinstance(prop,Usd.Relationship) else prop.GetConnections()))
                                for node in Usd.PrimRange(target))
                        if closed[base]:
                            chosen=str(base)
            identities[key]=chosen
        if identities[key]:
            aliases[str(prim.GetPath())]=identities[key]
    return aliases



def _owned_material_binding_paths(stage, child_roots, metrics=None):
    """Read this definition's bindings without visiting retained child content."""
    from pxr import Usd
    excluded=set(child_roots)
    iterator=iter(Usd.PrimRange.Stage(stage))
    paths=[]
    for prim in iterator:
        if metrics is not None:
            metrics['visited_prims']=metrics.get('visited_prims',0)+1
        if prim.GetPath() in excluded:
            iterator.PruneChildren()
            continue
        paths.extend(relationship.GetPath() for relationship in prim.GetAuthoredRelationships()
                     if relationship.GetName().startswith('material:binding'))
    return paths


def _clear_root_placement(prim):
    """Leave occurrence placement to the consuming reference site."""
    for attr in list(prim.GetAttributes()):
        if attr.GetName().startswith('xformOp'):
            prim.RemoveProperty(attr.GetName())


def _preserve_variant_context(source, root, editable, standalone_kind, successor, copy_stage_metadata):
    """Retain unoptimized variant branches around the selected worker input."""
    from pxr import Sdf, Usd
    prim = source.GetPrimAtPath(root)
    asset = editable.GetDefaultPrim()
    variant_context = []
    variant_sets = prim.GetVariantSets().GetNames()
    if len(variant_sets) > 1 and not successor:
        raise ValueError('Multiple interacting variant sets require separate inspected contexts: ' + root)
    if variant_sets and not successor:
        name = variant_sets[0]
        source_variants = prim.GetVariantSet(name)
        selection = source_variants.GetVariantSelection()
        selected_layer = editable.GetRootLayer()
        wrapper = Usd.Stage.CreateInMemory()
        wrapper_asset = wrapper.DefinePrim('/Asset', asset.GetTypeName())
        wrapper_asset.SetMetadata('kind', standalone_kind)
        wrapper.SetDefaultPrim(wrapper_asset)
        copy_stage_metadata(wrapper)
        variants = wrapper_asset.GetVariantSets().AddVariantSet(name)
        for choice in source_variants.GetVariantNames():
            variants.AddVariant(choice)
            destination = Sdf.Path('/Asset').AppendVariantSelection(name, choice)
            if choice == selection:
                Sdf.CopySpec(selected_layer, '/Asset', wrapper.GetRootLayer(), destination)
            else:
                # Retain the alternate composed branch. It remains outside
                # this declared optimization context and is never reported
                # as optimized merely because its selected sibling passed.
                source_variants.SetVariantSelection(choice)
                alternate = source.Flatten()
                alternative_stage = Usd.Stage.CreateInMemory()
                Sdf.CopySpec(alternate, root, alternative_stage.GetRootLayer(), '/Asset')
                # Retain load/unload behavior in unoptimized alternatives.
                # Their original payloads remain declared read-only assets.
                for payload_prim in Usd.PrimRange(source.GetPrimAtPath(root)):
                    if not payload_prim.HasAuthoredPayloads():
                        continue
                    local_path = _mapped(str(payload_prim.GetPath()), root)
                    snapshot = _placement(payload_prim)
                    payloads = []
                    for spec in payload_prim.GetPrimStack():
                        for payload in spec.payloadList.GetAddedOrExplicitItems():
                            if payload.assetPath:
                                resolved = Sdf.ComputeAssetPathRelativeToLayer(spec.layer, payload.assetPath)
                                payloads.append(Sdf.Payload(resolved, payload.primPath, payload.layerOffset))
                        if payloads:
                            break
                    if not payloads:
                        raise ValueError('Cannot preserve payload dependency: ' + str(payload_prim.GetPath()))
                    alternative_stage.RemovePrim(local_path)
                    payload_site = alternative_stage.DefinePrim(local_path)
                    for payload in payloads:
                        payload_site.GetPayloads().AddPayload(payload)
                    for attr_name, typ, default, samples in snapshot:
                        attr = payload_site.CreateAttribute(attr_name, typ)
                        if default is not None:
                            attr.Set(default)
                        for time, value in samples:
                            attr.Set(value, time)
                # Occurrence placement is authored at the consuming site.
                alt_root = alternative_stage.GetPrimAtPath('/Asset')
                _clear_root_placement(alt_root)
                Sdf.CopySpec(alternative_stage.GetRootLayer(), '/Asset', wrapper.GetRootLayer(), destination)
                variant_context.append({'unoptimized_selection': choice})
        source_variants.SetVariantSelection(selection)
        variants.SetVariantSelection(selection)
        editable = wrapper
        variant_context = [{'path': '/Asset', 'set': name, 'selection': selection,
                            'preserved_unoptimized_selections': [v['unoptimized_selection'] for v in variant_context]}]
    return editable, variant_context


def _prepare_worker_target(definition, editable, variant_context, scope):
    """Map the manifest to /Asset and replan merges against prepared composition."""
    from pxr import Sdf, UsdGeom
    root = definition['boundary_path']
    target = copy.deepcopy(definition)
    target['variant_context'] = variant_context
    target['source_boundary_path'] = root
    target['required_occurrence_edits'] = [dict(e, path=str(Sdf.Path('/Asset').AppendPath(Sdf.Path(e['path']).MakeRelativePath(Sdf.Path(root))))) for e in definition.get('required_occurrence_edits', [])]
    target['boundary_path'] = '/Asset'
    target['owned_geometry'] = [_mapped(p, root) for p in definition.get('owned_geometry', [])]
    target['protected_paths'] = [_mapped(p, root) for p in definition.get('protected_paths', []) if p.startswith(root+'/')]
    target['internal_paths'] = [_mapped(p, root) for p in definition.get('internal_paths', []) if p == root or p.startswith(root+'/')]
    target['internal_kind_overrides'] = {_mapped(p, root): kind for p, kind in definition.get('internal_kind_overrides', {}).items()}
    target['operation_scope'] = {key: [_mapped(p, root) for p in values]
                                 for key, values in definition.get('operation_scope', {}).items()
                                 if isinstance(values, list)}
    # Source composition is resolved here. Recompute eligibility using the
    # actual editable inputs, while retained child references remain opaque.
    mesh_paths = [p for p in target['owned_geometry']
                  if editable.GetPrimAtPath(p).IsA(UsdGeom.Mesh)]
    facts = scope.merge_protection_facts(editable, target['internal_paths'],
        boundary_paths=target['operation_scope'].get('exclude_paths', []))
    allowed = set(target['operation_scope'].get('mesh_paths', mesh_paths))
    excluded = target['operation_scope'].get('exclude_paths', [])
    remaining = {p for p in mesh_paths if p in allowed
                 and not any(p == child or p.startswith(child + '/') for child in excluded)}
    groups, exclusions = [], []
    reserved_outputs = {_mapped(group['output_path'], root) for group in definition.get('merge_groups', [])}
    # Existing groups retain their explicit scope and native spatial policy.
    # Reconsider previously excluded geometry separately, without broadening
    # a requested cluster or combining separately specified merge scopes.
    for original in definition.get('merge_groups', []):
        selected = {_mapped(p, root) for p in original.get('mesh_paths', [])} & remaining
        remaining.difference_update(selected)
        planned, rejected = scope.plan_merge_groups(editable, '/Asset', sorted(selected),
            set(target['protected_paths']), protection_facts=facts, reserved_outputs=reserved_outputs)
        exclusions.extend(rejected)
        for group in planned:
            preserved = copy.deepcopy(original)
            preserved['mesh_paths'] = group['mesh_paths']
            preserved['output_path'] = (_mapped(original['output_path'], root)
                if len(planned) == 1 else group['output_path'])
            groups.append(preserved)
    planned, rejected = scope.plan_merge_groups(editable, '/Asset', sorted(remaining),
        set(target['protected_paths']), protection_facts=facts, reserved_outputs=reserved_outputs)
    groups.extend(planned)
    exclusions.extend(rejected)
    target['merge_groups'] = groups
    for operation in target.get('assessment', {}).get('operations', []):
        if operation.get('category') == 'merge':
            operation.update(status='candidate' if groups else 'not_applicable',
                requires_scoped_validation=bool(groups),
                reason=('Compatible prepared interior meshes can merge within the retained boundary.'
                        if groups else 'No compatible prepared interior mesh group.'),
                evidence={'groups': len(groups), 'retained_mesh_interfaces': exclusions})
    return target


def author(manifest, directory, progress=None):
    """Create immutable worker inputs and a consumer assembly; sources are read-only.

    Flatten each selected definition's declared composed context before editing it.
    Material networks remain referenced from one shared library. Successor policy
    writes only the recorded selected variant configuration and preserves animation.
    """
    from pxr import Sdf, Usd, UsdGeom, UsdShade
    import numpy as np
    completed_definitions=set()
    def report(step, written=()):
        completed_definitions.update(written)
        if progress:
            progress('author_inputs',step=step,completed_count=len(completed_definitions),
                     definition_count=len(manifest['definitions']))
    report('selected_composition')
    if manifest.get('assessment_phase') == 'boundary_review':
        raise ValueError('Resolve boundary review and finalize assessment before authoring')
    check_material_policy(manifest)
    request = manifest.get('request') or {}
    if type(request.get('successor_policy', False)) is not bool:
        raise ValueError('successor_policy must be a boolean')
    successor = request.get('successor_policy', False)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    source_path = manifest['source']['path']
    session = Sdf.Layer.CreateAnonymous('authoring-context.usda')
    source = Usd.Stage.Open(Sdf.Layer.FindOrOpen(source_path), session)
    correspondence_path=Path(__file__).resolve().parents[2]/'usd-structure-assessment/scripts/structure_assessment.py'
    correspondence_spec=importlib.util.spec_from_file_location('author_atomic_correspondence',correspondence_path)
    correspondence_module=importlib.util.module_from_spec(correspondence_spec);correspondence_spec.loader.exec_module(correspondence_module)
    source.SetEditTarget(session)
    for path, selections in manifest.get('context', {}).get('variants', {}).items():
        prim = source.GetPrimAtPath(path)
        if not prim:
            raise ValueError(f'missing declared variant path {path}')
        for name, selection in selections.items():
            variant = prim.GetVariantSet(name)
            if selection not in variant.GetVariantNames():
                raise ValueError(f'missing declared variant {path}:{name}={selection}')
            variant.SetVariantSelection(selection)
    correspondence_module.validate_atomic_correspondences(source,manifest)
    publication_spec=importlib.util.spec_from_file_location('material_publication',Path(__file__).with_name('material_publication.py'))
    publication=importlib.util.module_from_spec(publication_spec);publication_spec.loader.exec_module(publication)
    manifest['material_target_repairs']=publication.repair_shader_targets(source) if successor else {'repaired':[], 'unresolved':[]}
    material_aliases = _unmodified_material_aliases(source)
    # Make proxy content addressable in a disposable session layer. Source
    # authored layers remain unchanged, and the flat snapshot preserves the
    # selected occurrence-specific material and geometry opinions.
    while True:
        instances = [prim for prim in source.Traverse() if prim.IsInstance()]
        if not instances:
            break
        # Author the whole level before recomposing. Updating each of tens of
        # thousands of instances through UsdPrim would recompose the stage for
        # every occurrence. Only the disposable session layer is edited here.
        paths = [prim.GetPath() for prim in instances]
        with Sdf.ChangeBlock():
            for path in paths:
                Sdf.CreatePrimInLayer(session, path).SetInfo('instanceable', False)
    if not source or source.GetCompositionErrors():
        raise ValueError('source composition is incomplete')
    definitions = {d['id']: d for d in manifest['definitions']}
    occurrences = manifest['occurrences']
    by_path = {o['path']: o for o in occurrences}
    occurrence_interfaces = []
    external_properties = set()
    if successor:
        # Composition arcs cannot carry targets outside their referenced root.
        # Keep these occurrence-specific opinions on the consumer, where both
        # equipment occurrences and the targets they address exist.
        boundaries = {Sdf.Path(path) for path in by_path}
        for prim in source.Traverse():
            owner = _nearest_occurrence_owner(prim.GetPath(), boundaries)
            if owner is None or prim.IsA(UsdShade.Shader) or prim.IsA(UsdShade.Material):
                continue
            for prop in prim.GetProperties():
                if prop.GetName().startswith('material:binding'):
                    continue
                targets = prop.GetTargets() if isinstance(prop, Usd.Relationship) else prop.GetConnections()
                if targets and any(not target.HasPrefix(Sdf.Path(owner)) for target in targets):
                    occurrence_interfaces.append({'property_path':str(prop.GetPath()),'owner_prim_path':owner,
                        'property_type':'relationship' if isinstance(prop,Usd.Relationship) else 'connection',
                        'targets':list(map(str,targets))})
                    external_properties.add(str(prop.GetPath()))
        manifest['occurrence_interfaces'] = occurrence_interfaces
    def local_interfaces(prim):
        return [(name,targets) for name,targets in _interfaces(prim)
                if str(prim.GetPath().AppendProperty(name)) not in external_properties]
    by_definition = defaultdict(list)
    direct_children = defaultdict(list)
    for occurrence in occurrences:
        by_definition[occurrence['definition_id']].append(occurrence)
        direct_children[occurrence.get('parent_path')].append(occurrence)
    samples = manifest.get('context', {}).get('time_samples', [None])
    transform_caches = [UsdGeom.XformCache(Usd.TimeCode.Default() if sample is None else Usd.TimeCode(sample))
                        for sample in samples]

    for occurrence in occurrences:
        definition = definitions[occurrence['definition_id']]
        original_kind = occurrence.get('effective_kind') or ''
        occurrence.setdefault('source_kind', original_kind)
        if successor:
            parent = by_path.get(occurrence.get('parent_path'))
            parent_kind = parent.get('effective_kind') if parent else None
            kind = original_kind or ('subcomponent' if parent_kind in ('component', 'subcomponent')
                                     else 'assembly' if parent is None and definition.get('children')
                                     else 'component')
            ancestor = parent
            while ancestor:
                if ancestor.get('effective_kind') in ('component', 'subcomponent') and kind != 'subcomponent':
                    raise ValueError('A retained child inside a component requires subcomponent kind: ' + occurrence['path'])
                ancestor = by_path.get(ancestor.get('parent_path'))
            occurrence['effective_kind'] = kind
            occurrence['kind_override'] = kind
            definition.setdefault('standalone_kind', 'component' if kind == 'subcomponent' else kind)
            continue
        if not original_kind:
            parent = by_path.get(occurrence.get('parent_path'))
            parent_definition = definitions.get(parent['definition_id']) if parent else None
            kind = 'assembly' if definition['structural_type'] == 'aggregate' else 'component'
            if parent_definition and parent_definition['structural_type'] == 'package' and parent.get('effective_kind') == 'component':
                kind = 'subcomponent'
            occurrence['effective_kind'] = kind
        parent = by_path.get(occurrence.get('parent_path'))
        while parent:
            if parent.get('effective_kind') == 'component':
                occurrence['effective_kind'] = 'subcomponent'
                occurrence['kind_override'] = 'subcomponent'
                break
            parent = by_path.get(parent.get('parent_path'))
        if definition['structural_type'] == 'package' and original_kind == 'assembly':
            definition['standalone_kind'] = 'assembly'
        else:
            definition.setdefault('standalone_kind', 'assembly' if definition['structural_type'] == 'aggregate' else 'component')
    report('material_setup')
    material_file = directory / 'materials.usdc'
    materials = Usd.Stage.CreateInMemory()
    material_root = UsdGeom.Scope.Define(materials, '/Materials').GetPrim()
    materials.SetDefaultPrim(material_root)
    UsdGeom.SetStageMetersPerUnit(materials, UsdGeom.GetStageMetersPerUnit(source))
    UsdGeom.SetStageUpAxis(materials, UsdGeom.GetStageUpAxis(source))
    materials.SetTimeCodesPerSecond(source.GetTimeCodesPerSecond())
    materials.SetFramesPerSecond(source.GetFramesPerSecond())
    materials.SetStartTimeCode(source.GetStartTimeCode())
    materials.SetEndTimeCode(source.GetEndTimeCode())
    material_paths = {}
    # Flatten once for material networks and assembly metadata, preserving source input.
    flat = source.Flatten()
    shading_roots = set()
    alias_roots = {Sdf.Path(path) for path in material_aliases}
    material_containers = set()
    for prim in source.Traverse():
        if prim.IsA(UsdShade.Material) or prim.IsA(UsdShade.Shader) or prim.IsA(UsdShade.NodeGraph):
            path = str(prim.GetPath())
            if path in material_aliases:
                material_paths[path] = '/Materials' + material_aliases[path]
                continue
            if _nearest_occurrence_owner(prim.GetPath(), alias_roots) is not None:
                continue
            material_containers.update(str(p) for p in prim.GetPath().GetParentPath().GetPrefixes()
                                       if source.GetPrimAtPath(p) and not source.GetPrimAtPath(p).GetTypeName())
            dest = '/Materials' + path
            if prim.IsA(UsdShade.Material):
                material_paths[path] = dest
            if _nearest_occurrence_owner(prim.GetPath().GetParentPath(), shading_roots) is None:
                shading_roots.add(prim.GetPath())
                ancestors = list(Sdf.Path(dest).GetParentPath().GetPrefixes())
                for ancestor in ancestors:
                    UsdGeom.Scope.Define(materials, ancestor)
                if not Sdf.CopySpec(flat, path, materials.GetRootLayer(), dest):
                    raise ValueError(f'cannot copy shading network {path}')
    # Provenance belongs in the manifest, not in authored material identity.
    manifest['material_source_paths']={copied:material_aliases.get(original,original)
                                       for original,copied in material_paths.items()}
    # Connections outside an individual Material remain inside the shared library.
    for prim in materials.TraverseAll():
        for attr in prim.GetAttributes():
            connections = attr.GetConnections()
            if connections:
                attr.SetConnections([p if str(p).startswith('/Materials/') else Sdf.Path('/Materials'+str(p))
                                     for p in connections])
    if successor:
        manifest['pruned_material_networks']=publication.prune_unused_material_networks(materials,source,material_paths)
        material_paths={original:path for original,path in material_paths.items() if materials.GetPrimAtPath(path)}
    if material_paths:
        if successor:
            material_input = directory/'materials.input.usdc'
            materials.GetRootLayer().Export(str(material_input))
            path = Path(__file__).with_name('material_optimization.py')
            spec = importlib.util.spec_from_file_location('successor_material_optimization',path)
            optimizer = importlib.util.module_from_spec(spec);spec.loader.exec_module(optimizer)
            manifest['material_optimization'] = optimizer.optimize(material_input,material_file)
            remapped = manifest['material_optimization']['material_path_map']
            material_paths = {original:remapped[destination] for original,destination in material_paths.items()}
        else:
            materials.GetRootLayer().Export(str(material_file))
    report('materials_ready')
    material_stage=Usd.Stage.Open(str(material_file)) if material_paths else None
    pack_helpers = _pack_helpers()
    max_definitions_per_library = int(request.get('definitions_per_library', 32))
    max_input_bytes_per_library = int(request.get('max_input_bytes_per_library', 8*1024*1024))
    if max_definitions_per_library < 1 or max_input_bytes_per_library < 1:
        raise ValueError('library batching limits must be positive')
    direct_authoring = request.get('direct_library_authoring', not successor) is not False

    for definition in definitions.values():
        stem = hashlib.sha256(definition['id'].encode()).hexdigest()[:20]
        entry = definition.get('persistent_entry')
        if entry:
            # A confirmed library asset uses the original read-only source for
            # input accounting; no disposable worker input is needed.
            definition['input_layer'] = source_path
            definition['output_layer'] = entry['layer']
            definition['output_prim_path'] = entry['prim_path']
            definition['reuse_before_preparation'] = True
        else:
            definition['input_layer'] = str(directory / (stem + '.input.usdc'))
            definition['output_layer'] = str(directory / (stem + '.usdc'))
            definition['output_prim_path'] = '/Asset'
        boundary = definition['boundary_path']
        definition['instanceable'] = (definition['structural_type'] == 'atomic'
            and not any(path.startswith(boundary + '/') for path in definition.get('protected_paths', [])))
        if any(row['owner_prim_path'] == boundary and str(Sdf.Path(row['property_path']).GetPrimPath()) != boundary
               for row in occurrence_interfaces):
            definition['instanceable'] = False

        scales = [float(np.linalg.norm(np.asarray(cache.GetLocalToWorldTransform(source.GetPrimAtPath(o['path'])))[:3, :3], 2))
                  for o in by_definition[definition['id']] for cache in transform_caches]
        definition['maximum_occurrence_scale'] = max(scales, default=1.0)

    direct_batches = []
    if direct_authoring:
        candidates = [d for d in definitions.values() if d.get('structural_type') == 'atomic' and not d.get('children') and not d.get('persistent_entry')]
        current = []
        for definition in sorted(candidates, key=lambda d: d['id']):
            if len(current) >= max_definitions_per_library:
                if len(current) > 1:
                    direct_batches.append(current)
                current = []
            current.append(definition)
        if len(current) > 1:
            direct_batches.append(current)
        for number, batch in enumerate(direct_batches):
            input_path = directory / f'library-{number:05d}.input.usdc'
            output_path = directory / f'library-{number:05d}.usdc'
            for definition in batch:
                definition['input_layer'] = str(input_path)
                definition['output_layer'] = str(output_path)
                definition['output_prim_path'] = pack_helpers.packed_prim_path(definition['id'])

    direct_definition_ids = {definition['id'] for batch in direct_batches for definition in batch}

    def _stage_metadata(stage):
        UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(source))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(source))
        stage.SetTimeCodesPerSecond(source.GetTimeCodesPerSecond())
        stage.SetFramesPerSecond(source.GetFramesPerSecond())
        stage.SetStartTimeCode(source.GetStartTimeCode())
        stage.SetEndTimeCode(source.GetEndTimeCode())

    scope_spec = importlib.util.spec_from_file_location(
        'prepared_merge_scope', Path(__file__).resolve().parents[2] / 'component_scope.py')
    scope = importlib.util.module_from_spec(scope_spec)
    scope_spec.loader.exec_module(scope)

    def _build_definition(definition):
        root = definition['boundary_path']
        prim = source.GetPrimAtPath(root)
        if not prim:
            raise ValueError(f'missing source boundary {root}')
        editable = Usd.Stage.CreateInMemory()
        if not Sdf.CopySpec(flat, root, editable.GetRootLayer(), '/Asset'):
            raise ValueError(f'cannot extract definition {root}')
        asset = editable.GetPrimAtPath('/Asset')
        for row in occurrence_interfaces:
            path = Sdf.Path(row['property_path'])
            if path.HasPrefix(Sdf.Path(root)):
                local = path.ReplacePrefix(Sdf.Path(root),Sdf.Path('/Asset'))
                subject = editable.GetPrimAtPath(local.GetPrimPath())
                if subject:
                    if row['property_type'] == 'relationship':
                        subject.RemoveProperty(local.name)
                    else:
                        subject.GetAttribute(local.name).ClearConnections()
        _clear_root_placement(asset)
        # Copied CAD model labels are not retained asset or interface decisions.
        child_roots = {_mapped(o['path'], root) for o in direct_children[root]}
        child_descendants = set()
        for internal in editable.Traverse():
            local = str(internal.GetPath())
            parent = str(internal.GetPath().GetParentPath())
            if (local in child_roots or parent in child_descendants
                    or internal.HasAuthoredReferences() or internal.HasAuthoredPayloads()
                    or internal.HasAuthoredInherits() or internal.HasAuthoredSpecializes()):
                child_descendants.add(local)
                continue
            # Retaining a coordinate frame does not retain its imported model kind.
            if local != '/Asset' and not internal.IsInstanceProxy():
                internal.ClearMetadata('kind')
        asset.SetMetadata('kind', definition['standalone_kind'])
        for internal_path, kind in definition.get('internal_kind_overrides', {}).items():
            internal = editable.GetPrimAtPath(_mapped(internal_path, root))
            if not internal:
                raise ValueError('Missing retained internal interface: ' + internal_path)
            internal.SetMetadata('kind', kind)
        editable.SetDefaultPrim(asset)
        _stage_metadata(editable)
        for shading in reversed(list(editable.Traverse())):
            if shading.IsA(UsdShade.Material) or shading.IsA(UsdShade.Shader) or shading.IsA(UsdShade.NodeGraph):
                editable.RemovePrim(shading.GetPath())
        mounts=publication.MaterialMounts(editable,material_file,material_stage) if material_paths else None
        def shared_material_path(original_path):
            if original_path not in material_paths:
                raise ValueError(f'unrecorded material network {original_path}')
            return mounts.mount(material_paths[original_path])
        # CopySpec retains declared ancestor/purpose bindings even when a leaf
        # has a stronger effective binding. Every target must remain inside the
        # referenceable definition, not just the material selected at a leaf.
        # Mounting a material reference can resync its descendants.
        # Keep property paths, not prim/property handles, across that mutation.
        child_roots=[Sdf.Path(_mapped(o['path'],root)) for o in direct_children[root]]
        binding_paths = _owned_material_binding_paths(editable,child_roots)
        for binding_path in binding_paths:
            relationship = editable.GetRelationshipAtPath(binding_path)
            if not relationship or relationship.GetPrim().IsInstanceProxy():
                raise ValueError('declared binding is no longer editable: '+str(binding_path))
            mapped=[]
            # Resolve target identity from its source property. A source can
            # itself be rooted at /Asset, so a copied /Asset/Looks target
            # is not necessarily local to this extracted child definition.
            source_property=relationship.GetPath().ReplacePrefix(Sdf.Path('/Asset'),Sdf.Path(root))
            original_relationship=source.GetRelationshipAtPath(source_property)
            if not original_relationship:
                raise ValueError('material binding has no source correspondence: '+str(relationship.GetPath()))
            for source_target in original_relationship.GetTargets():
                original=str(source_target)
                target=source_target.ReplacePrefix(Sdf.Path(root),Sdf.Path('/Asset'))
                if original in material_paths:
                    mapped.append(shared_material_path(original))
                elif target.IsPropertyPath() and target.HasPrefix(Sdf.Path('/Asset')) and editable.GetObjectAtPath(target):
                    # Collection bindings may retain an internal collection.
                    mapped.append(target)
                else:
                    raise ValueError(f'material binding target is outside the extracted definition: {relationship.GetPath()} -> {target}')
            relationship = editable.GetRelationshipAtPath(binding_path)
            if not relationship or relationship.GetPrim().IsInstanceProxy():
                raise ValueError('declared binding became uneditable after material mounting: '+str(binding_path))
            relationship.SetTargets(mapped)
        # Rebind inherited materials explicitly in the extracted occurrence context.
        material_subjects = list(definition.get('owned_geometry', []))
        for geometry_path in definition.get('owned_geometry', []):
            geometry_prim = source.GetPrimAtPath(geometry_path)
            material_subjects.extend(str(child.GetPath()) for child in geometry_prim.GetChildren()
                                     if child.IsA(UsdGeom.Subset))
        for path in material_subjects:
            original = source.GetPrimAtPath(path)
            written = editable.GetPrimAtPath(_mapped(path, root))
            if not original or not written:
                raise ValueError(f'missing owned geometry {path}')
            material, _ = UsdShade.MaterialBindingAPI(original).ComputeBoundMaterial()
            if material:
                material_path = str(material.GetPath())
                local = shared_material_path(material_path)
                mat = UsdShade.Material(editable.GetPrimAtPath(local))
                UsdShade.MaterialBindingAPI.Apply(written).Bind(mat)
        _type_material_scopes(editable, [_mapped(p, root) for p in material_containers if p == root or p.startswith(root + '/')])
        # A parent's child occurrences reference already allocated outputs.
        for occurrence in direct_children[root]:
            path = occurrence['path']
            child_id = occurrence['definition_id']
            child = definitions[child_id]
            original = source.GetPrimAtPath(path)
            dependency_layer = child['output_layer'] if child.get('persistent_entry') else child['input_layer'] if successor else child['output_layer']
            _reference(editable, _mapped(path, root), dependency_layer,
                       _placement(original), occurrence.get('effective_kind'),
                       local_interfaces(original), root, '/Asset', original.HasAuthoredPayloads(), original, child['instanceable'],
                       target_prim_path=child.get('output_prim_path', '/Asset'))
            _internal_placement(editable,occurrence,source,child['boundary_path'],_mapped(path,root))
            if successor and occurrence.get('effective_kind') in ('component', 'assembly', 'group'):
                # Organizational ancestors participate in the model hierarchy
                # without becoming independently published asset boundaries.
                ancestor = Sdf.Path(_mapped(path, root)).GetParentPath()
                while ancestor != Sdf.Path('/Asset') and ancestor != Sdf.Path.absoluteRootPath:
                    editable.GetPrimAtPath(ancestor).SetMetadata('kind', 'group')
                    ancestor = ancestor.GetParentPath()
        editable, variant_context = _preserve_variant_context(
            source, root, editable, definition['standalone_kind'], successor, _stage_metadata)
        target = _prepare_worker_target(definition, editable, variant_context, scope)
        definition['pruned_abstract_roots']=publication.prune_unused_abstract_roots(editable) if successor else []
        return editable, target

    written_batches = []

    def _write_bounded_library(batch, library_stage, input_path, output_path):
        library_stage.GetRootLayer().Export(str(input_path))
        byte_count = input_path.stat().st_size
        if byte_count > max_input_bytes_per_library and len(batch) > 1:
            # Split only generated scratch layers. No per-definition files are
            # needed for ordinary small leaves, and oversized leaves stay alone.
            input_path.unlink()
            midpoint = len(batch) // 2
            for suffix, subset in (('a', batch[:midpoint]), ('b', batch[midpoint:])):
                split = Usd.Stage.CreateInMemory()
                split.GetRootLayer().TransferContent(library_stage.GetRootLayer())
                keep = {d['output_prim_path'] for d in subset}
                for child in list(split.GetDefaultPrim().GetChildren()):
                    if str(child.GetPath()) not in keep:
                        split.RemovePrim(child.GetPath())
                _write_bounded_library(subset, split,
                    input_path.with_name(input_path.stem + suffix + input_path.suffix),
                    output_path.with_name(output_path.stem + suffix + output_path.suffix))
            return
        for definition in batch:
            pack_helpers.apply_packed_assignment(
                definition, input_path, output_path, definition['output_prim_path'],
                'Author independent definitions in bounded writer libraries.')
            definition['packaging']['input_bytes'] = byte_count
            definition['packaging']['oversized_single_definition'] = byte_count > max_input_bytes_per_library
        written_batches.append(batch)
        report('definition_inputs',[definition['id'] for definition in batch])

    for batch in direct_batches:
        input_path = Path(batch[0]['input_layer'])
        output_path = Path(batch[0]['output_layer'])
        library_stage = Usd.Stage.CreateInMemory()
        library = UsdGeom.Scope.Define(library_stage, '/Library').GetPrim()
        library.SetMetadata('kind', 'group')
        library_stage.SetDefaultPrim(library)
        _stage_metadata(library_stage)
        for definition in batch:
            editable, target = _build_definition(definition)
            prim_path = definition['output_prim_path']
            if not Sdf.CopySpec(editable.GetRootLayer(), '/Asset', library_stage.GetRootLayer(), prim_path):
                raise ValueError('cannot copy atomic definition ' + definition['id'])
            definition['worker_target'] = target
        _write_bounded_library(batch, library_stage, input_path, output_path)

    # Immutable child inputs must exist before a parent input composes them.
    # Geometry workers can then run concurrently without reading child outputs.
    independent_order = sorted(definitions.values(), key=lambda d: -d['boundary_path'].count('/')) if successor else definitions.values()
    for definition in independent_order:
        if definition['id'] in direct_definition_ids:
            continue
        if definition.get('persistent_entry'):
            definition['worker_target'] = {key: value for key, value in definition.items()
                if key not in ('worker_target', 'persistent_entry')}
            report('reused_definition_inputs', [definition['id']])
            continue
        editable, target = _build_definition(definition)
        # Worker source needs dependency outputs, generated in dependency order.
        editable.GetRootLayer().Export(definition['input_layer'])
        definition['worker_target'] = target
        report('definition_inputs',[definition['id']])

    if direct_batches:
        manifest['packaging_metrics'] = {'definitions': len(manifest['definitions']),
            'writer_layers': len({d['output_layer'] for d in manifest['definitions']}),
            'max_definitions_per_library': max_definitions_per_library,
            'max_input_bytes_per_library': max_input_bytes_per_library,
            'direct_authoring': True,
            'direct_libraries': len(written_batches),
            'direct_definitions': len(direct_definition_ids)}
    manifest['preparation_counts'] = {
        'definition_count': len(definitions),
        'prepared_definition_count': sum(not bool(d.get('persistent_entry')) for d in definitions.values()),
        'reused_without_input_count': sum(bool(d.get('persistent_entry')) for d in definitions.values()),
    }
    report('assembly_write')
    interface_specs = Sdf.Layer.CreateAnonymous('occurrence-interfaces.usda')
    for row in occurrence_interfaces:
        path = Sdf.Path(row['property_path'])
        Sdf.CreatePrimInLayer(interface_specs,path.GetPrimPath())
        if not Sdf.CopySpec(flat,path,interface_specs,path):
            raise ValueError('cannot record occurrence interface '+str(path))
    assembly = Usd.Stage.Open(flat)
    roots = []
    for occurrence in sorted(occurrences, key=lambda o: len(o['path'])):
        path = occurrence['path']
        if any(path.startswith(p + '/') for p in roots):
            continue
        roots.append(path)
        definition = definitions[occurrence['definition_id']]
        _reference(assembly, path, definition['output_layer'],
                   _placement(source.GetPrimAtPath(path)), occurrence.get('effective_kind'),
                   local_interfaces(source.GetPrimAtPath(path)), payload=source.GetPrimAtPath(path).HasAuthoredPayloads(),
                   source_prim=source.GetPrimAtPath(path), instanceable=definition['instanceable'],
                   target_prim_path=definition.get('output_prim_path', '/Asset'))
        _internal_placement(assembly,occurrence,source,definition['boundary_path'],path)
    _child_occurrence_placements(assembly, source, occurrences)
    _type_material_scopes(assembly, material_containers)
    for row in occurrence_interfaces:
        path = Sdf.Path(row['property_path'])
        assembly.OverridePrim(path.GetPrimPath())
        if not Sdf.CopySpec(interface_specs,path,assembly.GetRootLayer(),path):
            raise ValueError('cannot preserve occurrence interface '+str(path))
    root = source.GetDefaultPrim()
    if root:
        assembly.SetDefaultPrim(assembly.GetPrimAtPath(root.GetPath()))
    manifest['pruned_abstract_roots']=publication.prune_unused_abstract_roots(assembly) if successor else []
    output = directory / 'assembly.usdc'
    assembly.GetRootLayer().Export(str(output))
    report('assembly_written')
    manifest['output'] = {'path': str(output), 'directory': str(directory),
                          'material_library': str(material_file) if material_paths else None}
    return manifest
