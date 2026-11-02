# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate explicit mesh selections inside a retained USD component.

Manifest checks need only Python. Stage checks additionally require OpenUSD and
must run immediately before a merge against the editable output definition.
"""
from __future__ import annotations


def prim_path(value):
    return (isinstance(value, str) and value.startswith('/') and value != '/'
            and all(part.isidentifier() for part in value[1:].split('/')))


def beneath(path, parent):
    return path.startswith(parent + '/')


def overlaps(a, b):
    return a == b or beneath(a, b) or beneath(b, a)


def validate_scope(target):
    """Return errors for additive boundary fields; old manifests remain valid."""
    errors = []
    boundary = target.get('boundary_path')
    protected = target.get('protected_paths', [])
    internal = target.get('internal_paths', [])
    scope = target.get('operation_scope')
    if boundary is not None and not prim_path(boundary):
        errors.append('boundary_path must be an absolute prim path')
    if not isinstance(protected, list) or any(not prim_path(p) for p in protected):
        errors.append('protected_paths must be absolute prim paths')
    if (not isinstance(internal, list) or any(not prim_path(p) for p in internal)
            or len(set(internal)) != len(internal)):
        errors.append('internal_paths must be unique absolute prim paths')
    elif internal:
        if not prim_path(boundary) or any(not beneath(p, boundary) for p in internal):
            errors.append('internal_paths must be strictly inside boundary_path')
    occurrences = target.get('occurrence_mappings', [])
    if not isinstance(occurrences, list) or any(
            not isinstance(item, dict) or set(item) != {'source_path', 'output_path'}
            or any(not prim_path(p) for p in item.values()) for item in occurrences):
        errors.append('occurrence_mappings require absolute source_path and output_path')
    authorization = target.get('identity_change_authorization')
    if authorization is not None and (
            not isinstance(authorization, dict)
            or set(authorization) != {'operation', 'paths', 'reason'}
            or authorization.get('operation') != target.get('reduction_route')
            or not isinstance(authorization.get('paths'), list)
            or not authorization['paths']
            or any(not prim_path(p) for p in authorization['paths'])
            or not isinstance(authorization.get('reason'), str)
            or not authorization['reason'].strip()):
        errors.append('identity_change_authorization requires the operation, exact paths and user instruction')
    if scope is None:
        return errors
    if not isinstance(scope, dict) or (set(scope) - {'spatial_merge'}) != {'type', 'mesh_paths', 'output_path'} or scope.get('type') != 'interior':
        return errors + ['operation_scope requires type=interior, mesh_paths and output_path']
    if not isinstance(scope.get('spatial_merge', {}), dict):
        return errors + ['spatial_merge must contain native merge settings']
    if not prim_path(boundary) or 'protected_paths' not in target:
        errors.append('interior operations require boundary_path and explicit protected_paths')
    paths = scope.get('mesh_paths')
    output = scope.get('output_path')
    if not isinstance(paths, list) or not paths or any(not prim_path(p) for p in paths):
        errors.append('mesh_paths must be a nonempty list of absolute prim paths')
        return errors
    if len(set(paths)) != len(paths):
        errors.append('mesh_paths must not repeat a mesh')
    if target.get('reduction_route') == 'merge' and len(paths) < 2:
        errors.append('interior merge requires at least two meshes')
    if not prim_path(output):
        errors.append('output_path must be an absolute prim path')
        return errors
    if prim_path(boundary):
        if any(not beneath(p, boundary) for p in paths + [output]):
            errors.append('mesh_paths and output_path must be strictly inside boundary_path')
    if isinstance(protected, list) and all(prim_path(p) for p in protected):
        if any(p == keep or beneath(keep, p) for p in paths + [output] for keep in protected) or any(
                beneath(p, keep) != beneath(output, keep) for p in paths for keep in protected):
            errors.append('operation scope crosses a protected prim')
    if any(overlaps(output, p) for p in paths):
        errors.append('output_path must be separate from selected meshes')
    return errors


def identity_change_authorized(target):
    authorization = target.get('identity_change_authorization')
    return (bool(authorization) and not validate_scope(target)
            and prim_path(target.get('boundary_path'))
            and target['boundary_path'] in authorization['paths'])


def retains_boundary(target):
    return (target.get('reduction_route') == 'merge'
            and isinstance(target.get('operation_scope'), dict)
            and target['operation_scope'].get('type') == 'interior'
            and not validate_scope(target))


def merge_protection_facts(stage, internal_paths=(), *, retain_internal_facts=False, boundary_paths=()):
    """Return concrete authored properties that constrain native structural edits.

    Accepted internal roots include descendants. Descriptive CAD metadata and
    source kinds do not establish an editable interface inside these scopes.
    """
    from pxr import UsdGeom
    internal = set(internal_paths)
    boundaries = set(boundary_paths)
    internal_descendants = set()
    facts = {}
    material_names = {f'omni:{converter}:material:name' for converter in ('jt', 'hoops')}
    descriptive = {'omni:jt:nodeType'} | {
        f'omni:{converter}:material:{field}'
        for converter in ('jt', 'hoops') for field in ('diffuseColor', 'opacity', 'name')}
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        inside = path not in boundaries and (path in internal or str(prim.GetPath().GetParentPath()) in internal_descendants)
        if inside: internal_descendants.add(path)
        reasons = set()
        if prim.GetMetadata('kind'):
            if not inside: reasons.add('metadata:kind')
            elif retain_internal_facts: reasons.add('internal_kind')
        for name, present in [('references', prim.HasAuthoredReferences()),
                              ('payload', prim.HasAuthoredPayloads()),
                              ('inherits', prim.HasAuthoredInherits()),
                              ('specializes', prim.HasAuthoredSpecializes()),
                              ('variantSets', bool(prim.GetVariantSets().GetNames())),
                              ('instanceable', prim.IsInstance())]:
            if present: reasons.add('composition:' + name)
        for schema in prim.GetAppliedSchemas():
            if any(word in str(schema) for word in ('Physics', 'Physx', 'Semantic')):
                reasons.add('apiSchema:' + str(schema))
        for attr in prim.GetAttributes():
            name = attr.GetName()
            if attr.GetNumTimeSamples(): reasons.add('timeSamples:' + name)
            if name in ('visibility', 'purpose') and attr.HasAuthoredValueOpinion():
                if name == 'visibility' and inside and attr.Get() == UsdGeom.Tokens.inherited and not attr.HasAuthoredConnections():
                    if retain_internal_facts: reasons.add('internal_inherited_visibility')
                else: reasons.add('attribute:' + name)
            if attr.IsCustom() and (attr.HasAuthoredValueOpinion() or attr.HasAuthoredConnections()):
                descriptive_metadata = inside and name.startswith(('omni:hoops:metadata:', 'omni:jt:metadata:'))
                if not descriptive_metadata and (name not in descriptive or (name in material_names and not inside)):
                    reasons.add('customAttribute:' + name)
            connections = attr.GetConnections()
            if connections: reasons.add('connection:' + name)
            for dst in connections:
                facts.setdefault(str(dst.GetPrimPath()), set()).add('addressed_target')
        for rel in prim.GetRelationships():
            if rel.GetName().startswith('material:binding'): continue
            if rel.HasAuthoredTargets(): reasons.add('relationship:' + rel.GetName())
            for dst in rel.GetForwardedTargets():
                facts.setdefault(str(dst.GetPrimPath()), set()).add('addressed_target')
        if reasons: facts.setdefault(path, set()).update(reasons)
    return {path: sorted(reasons) for path, reasons in facts.items()}


def merge_children_supported(prim, protection_facts=None):
    """Native merge preserves unaddressed face material subsets, not arbitrary children."""
    from pxr import UsdGeom
    for child in prim.GetAllChildren():
        if (not child.IsA(UsdGeom.Subset) or child.GetAllChildren()
                or (protection_facts or {}).get(str(child.GetPath()))):
            return False
        subset = UsdGeom.Subset(child)
        if (subset.GetElementTypeAttr().Get() != 'face'
                or subset.GetFamilyNameAttr().Get() != 'materialBind'):
            return False
    return True


def plan_merge_groups(stage, boundary_path, mesh_paths, protected, *, protection_facts, reserved_outputs=None):
    """Group compatible meshes in the supplied composition; preparation is authoritative."""
    from collections import defaultdict
    import hashlib
    import json
    from pxr import Sdf
    reserved_outputs = set() if reserved_outputs is None else reserved_outputs
    merge_facts = protection_facts
    merge_candidates = defaultdict(list)
    merge_exclusions = []
    for mesh_path in mesh_paths:
        mesh_prim = stage.GetPrimAtPath(mesh_path)
        if not merge_children_supported(mesh_prim, merge_facts) or mesh_path in protected:
            continue
        current = mesh_path
        blockers = []
        anchors = []
        while current != boundary_path:
            facts = merge_facts.get(current, [])
            if current in protected or facts:
                anchors.append(current)
            if facts and (current == mesh_path or any(f.startswith('composition:') for f in facts)):
                blockers.append({'path': current, 'facts': facts})
            current = current.rsplit('/', 1)[0]
        if blockers:
            merge_exclusions.append({'path': mesh_path, 'reason': 'Preserve authored merge-scope interfaces and state.', 'evidence': blockers})
            continue
        merge_candidates[hashlib.sha256(json.dumps(anchors, sort_keys=True, default=str, separators=(',', ':')).encode()).hexdigest()].append(mesh_path)
    merge_groups = []
    for signature, paths in sorted(merge_candidates.items()):
        if len(paths) < 2:
            continue
        # Keep the existing common frame: baking distant parent transforms
        # into float32 mesh points can violate lossless preservation. This
        # changes only operation placement, not asset ownership or eligibility.
        output_parent = Sdf.Path(paths[0]).GetParentPath()
        for mesh_path in paths[1:]:
            output_parent = output_parent.GetCommonPrefix(Sdf.Path(mesh_path).GetParentPath())
        output_path = str(output_parent) + '/Merged_' + signature[:12]
        suffix = 1
        while output_path in reserved_outputs or stage.GetPrimAtPath(output_path):
            output_path = str(output_parent) + '/Merged_' + signature[:12] + '_' + str(suffix)
            suffix += 1
        reserved_outputs.add(output_path)
        merge_groups.append({'mesh_paths': sorted(paths), 'output_path': output_path,
                             'reason': 'Internal meshes share a protected scope; native merge determines mesh compatibility and preserves primvars and material assignments.'})
    exclusion_groups = {}
    for exclusion in merge_exclusions:
        key = json.dumps(exclusion['evidence'], sort_keys=True)
        group = exclusion_groups.setdefault(key, {'evidence': exclusion['evidence'], 'mesh_count': 0, 'example_mesh_paths': []})
        group['mesh_count'] += 1
        if len(group['example_mesh_paths']) < 3: group['example_mesh_paths'].append(exclusion['path'])
    return merge_groups, list(exclusion_groups.values())


def native_merge_config(mesh_paths, output_path):
    """Use an explicit root boundary and a name relative to it, even with references."""
    from pxr import Sdf
    roots = {Sdf.Path(path).GetPrefixes()[0] for path in mesh_paths}
    if len(roots) != 1:
        raise ValueError('native merge requires one top-level source root')
    root = next(iter(roots))
    output = Sdf.Path(output_path)
    if not output.IsAbsolutePath() or output == root or not output.HasPrefix(root):
        raise ValueError('merge output must be beneath the selected source root')
    return {'operation': 'merge', 'meshPrimPaths': list(mesh_paths),
            'rootPath': str(output.MakeRelativePath(root)),
            'considerMaterials': False, 'mergePoint': 7}


def merge_output_path(config):
    """Resolve the supported Root Prim output name to its absolute stage path."""
    from pxr import Sdf
    roots = {Sdf.Path(path).GetPrefixes()[0] for path in config['meshPrimPaths']}
    name = Sdf.Path(config['rootPath'])
    if (config.get('mergePoint') != 7 or len(roots) != 1 or name.IsAbsolutePath()
            or not name.IsPrimPath() or '..' in str(name).split('/')):
        raise ValueError('merge requires Root Prim mode and a relative descendant output name')
    return str(next(iter(roots)).AppendPath(name))


def check_stage_scope(stage, target, *, protection_facts=None):
    """Resolve selected meshes and reject unlisted child boundaries.

Explicit protected paths cover semantic identities the USD schemas cannot infer.
Authored kind is protected outside accepted internal subtrees. Composition arcs, animation, physics and addressed targets remain protected. Instance proxies and referenced definitions need separate output
targets; this helper does not make instances editable or flatten composition.
"""
    from pxr import Sdf, Usd, UsdGeom
    errors = validate_scope(target)
    if errors or not retains_boundary(target):
        raise ValueError('; '.join(errors or ['a retained interior merge scope is required']))
    boundary_path = target['boundary_path']
    boundary = stage.GetPrimAtPath(boundary_path)
    if not boundary or boundary.IsInstance() or boundary.IsInstanceProxy() or boundary.IsInPrototype():
        raise ValueError('boundary_path must resolve to an editable definition')
    if not (boundary.IsA(UsdGeom.Xform) or boundary.IsA(UsdGeom.Scope)):
        raise ValueError('retained merge boundary must be an Xform or Scope')
    for path in target['protected_paths']:
        if not stage.GetPrimAtPath(path):
            raise ValueError('protected prim does not exist: ' + path)
    scope = target['operation_scope']
    if stage.GetPrimAtPath(scope['output_path']):
        raise ValueError('output_path already exists')
    parent = stage.GetPrimAtPath(Sdf.Path(scope['output_path']).GetParentPath())
    if not parent or parent.IsInstance() or parent.IsInstanceProxy():
        raise ValueError('output parent must exist and be editable')
    internal = set(target.get('internal_paths', []))
    for path in internal:
        if not stage.GetPrimAtPath(path):
            raise ValueError('internal prim does not exist: ' + path)
    implicit = set(target['protected_paths'])
    # A caller may retain facts across the final sequence of disjoint merges.
    # Internal kind/visibility exemptions still belong to each exact scope.
    facts = (merge_protection_facts(stage, internal) if protection_facts is None
             else protection_facts)
    internal_reasons = {'internal_kind', 'internal_inherited_visibility'}
    for path, reasons in facts.items():
        if not beneath(path, boundary_path): continue
        ancestor = path
        while ancestor and ancestor not in internal:
            ancestor = ancestor.rsplit('/', 1)[0]
        if not ancestor or set(reasons) - internal_reasons:
            implicit.add(path)
    # Protection retains a prim, not every mesh beneath it. Composition arcs
    # remain opaque; native edits cannot delete their weaker-layer contents.
    output = scope['output_path']
    for keep in implicit:
        reasons = facts.get(keep, ['required internal prim'])
        opaque = any(reason.startswith('composition:') for reason in reasons)
        for path in scope['mesh_paths'] + [output]:
            removes = path == keep or beneath(keep, path)
            crosses = path != output and beneath(path, keep) != beneath(output, keep)
            if removes or crosses or (opaque and beneath(path, keep)):
                raise ValueError('operation crosses a retained child or addressed prim: '+path+
                                 '; blocked by '+keep+' ('+', '.join(reasons)+')')
    for path in scope['mesh_paths']:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsA(UsdGeom.Mesh) or prim.IsInstanceProxy() or prim.IsInPrototype():
            raise ValueError('selected path is not an editable Mesh: ' + path)
        if not merge_children_supported(prim):
            raise ValueError('selected Mesh has unsupported children that a merge could remove: ' + path)
        # The caller opens a copied authoritative definition. A stronger override
        # alone is not enough: the merge must be able to remove the original mesh.
        if not stage.GetEditTarget().GetLayer().GetPrimAtPath(stage.GetEditTarget().MapToSpecPath(prim.GetPath())):
            raise ValueError('selected mesh is not authored in the edit layer: ' + path)
    spatial = scope.get('spatial_merge', {})
    if set(spatial) - {'spatialMode', 'spatialThreshold', 'spatialMaxSize', 'spatialVertexCount',
                       'boundaryTolerance', 'boundaryMinSharedVertices'}:
        raise ValueError('unsupported spatial merge setting')
    output = scope['output_path']
    if spatial.get('spatialMode', 0):
        # Native clusters receive suffixed sibling names. Keep all of them
        # beneath the declared output scope for ownership and write isolation.
        output += '/Mesh'
    return dict(native_merge_config(scope['mesh_paths'], output), **spatial)
