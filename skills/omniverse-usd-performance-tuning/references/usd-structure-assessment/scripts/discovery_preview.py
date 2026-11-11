# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cheap nominations only. Never a manifest or authorization to share geometry."""
from collections import defaultdict
import time


def review_index(stage, protected_paths=(), progress=None):
    """Complete occurrence topology and authored interfaces without mesh values.

    This is the input to boundary review, not confirmation of reuse or material
    compatibility. Instance proxies are visited so nested equipment and hidden
    geometry remain addressable in the composed stage.
    """
    from pxr import Usd, UsdGeom
    started = time.monotonic()
    predicate = Usd.PrimIsActive & Usd.PrimIsDefined & ~Usd.PrimIsAbstract
    prims, children, geometry, facts = {}, defaultdict(list), [], {}
    identifying = {}
    from importlib.util import spec_from_file_location, module_from_spec
    from pathlib import Path
    spec = spec_from_file_location('review_labels', Path(__file__).with_name('boundary_reasoning.py'))
    labels = module_from_spec(spec); spec.loader.exec_module(labels)
    protected, variants, unloaded, materials = set(protected_paths), [], [], []
    schemas = {}
    unknown_schemas = set()
    authored_samples = any(layer.ListAllTimeSamples() for layer in stage.GetUsedLayers())
    for index, prim in enumerate(Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies(predicate))):
        path = str(prim.GetPath()); schema = str(prim.GetTypeName())
        prims[path] = prim; children[path.rsplit('/', 1)[0] or '/'].append(path)
        if schema not in schemas:
            schemas[schema] = prim.IsA(UsdGeom.Gprim) or prim.IsA(UsdGeom.PointInstancer)
            if schema and Usd.SchemaRegistry.GetTypeFromSchemaTypeName(schema).isUnknown:
                unknown_schemas.add(schema)
        if not schemas[schema] and schema not in ('Shader', 'Material', 'NodeGraph'):
            identifying[path] = labels.identifying_facts(prim)
        if schemas[schema]:
            geometry.append({'path': path, 'schema': schema})
        if schema == 'Material': materials.append(prim)
        if prim.HasPayload() and not prim.IsLoaded(): unloaded.append(path)
        row = facts[path] = {}
        for name in prim.GetVariantSets().GetNames() if prim.HasVariantSets() else ():
            variant = prim.GetVariantSet(name)
            variants.append({'path': path, 'set': name, 'selection': variant.GetVariantSelection(),
                             'unselected': [v for v in variant.GetVariantNames() if v != variant.GetVariantSelection()]})
            row['variants'] = path
        apis = prim.GetAppliedSchemas()
        if any('RigidBody' in api or 'ArticulationRoot' in api for api in apis):
            protected.add(path); row['motion'] = path
        has_clips = prim.HasAuthoredMetadata('clips')
        if has_clips or (authored_samples and prim.IsA(UsdGeom.Xformable) and UsdGeom.Xformable(prim).TransformMightBeTimeVarying()):
            protected.add(path); row['motion'] = path
        if (path in protected or prim.GetAttribute('asset:interface') or
                prim.GetAttribute('interface:editable').Get() is True or
                any('Semantic' in api for api in apis) or prim.GetAssetInfo().get('identifier')):
            protected.add(path); row['interface'] = path
        for rel in prim.GetAuthoredRelationships():
            if rel.GetName().startswith('material:'): continue
            row['nonmaterial_relationships'] = path
            protected.update(str(target.GetPrimPath()) for target in rel.GetTargets())
        # Geometry attribute connections are resolved in mandatory detailed
        # assessment, which may reopen reviews for newly protected interfaces.
        # Do not inspect millions of mesh attributes before boundaries exist.
        if not schemas[schema] and schema not in ('Shader', 'Material', 'NodeGraph'):
            for attr in prim.GetAuthoredAttributes():
                if attr.HasAuthoredConnections():
                    protected.update(str(target.GetPrimPath()) for target in attr.GetConnections())
        if progress and index % 16384 == 0:
            progress('scan_stage', prims_done=index+1, prims_total=None, geometry_done=len(geometry))
    if progress:
        progress('scan_stage', prims_done=len(prims), prims_total=len(prims), geometry_done=len(geometry))
    protected.intersection_update(prims)
    for path in protected: facts[path]['interface'] = path
    geometry_under = defaultdict(list)
    for row in geometry:
        path = row['path']
        while path:
            geometry_under[path].append(row['path']); path = path.rsplit('/', 1)[0]
    return {'prims': prims, 'children': children, 'geometry': geometry, 'geometry_under': geometry_under,
            'identifying': identifying, 'facts': facts, 'protected': protected, 'variant_contexts': variants,
            'unloaded': unloaded, 'materials': materials, 'unknown_schemas': sorted(unknown_schemas),
            'metrics': {'prim_count': len(prims), 'geometry_count': len(geometry),
                        'scan_count': 1, 'geometry_values_read': False, 'elapsed_seconds': time.monotonic()-started},
            'deferred_checks': ['geometry_attribute_connections', 'effective_geometry_appearance',
                                'exact_reuse_confirmation', 'scoped_geometry_operations']}


def freeze_review_index(index):
    """Lossless compact topology for later review rounds; no geometry values."""
    paths = list(index['prims'])
    ids = {path: i for i, path in enumerate(paths)}
    types, type_ids = [], {}
    nodes = []
    for path in paths:
        schema = str(index['prims'][path].GetTypeName())
        if schema not in type_ids:
            type_ids[schema] = len(types); types.append(schema)
        parent, _, name = path.rpartition('/')
        nodes.append([ids.get(parent, -1), name, type_ids[schema]])
    return {'schema_version': 1, 'identifying': index.get('identifying', {}), 'types': types, 'nodes': nodes,
            'geometry_nodes': [ids[row['path']] for row in index['geometry']],
            'facts': [[ids[path], sorted(row)] for path, row in index['facts'].items() if row],
            'protected_nodes': sorted(ids[path] for path in index['protected']),
            'variant_contexts': index['variant_contexts'], 'unloaded': index['unloaded'],
            'unknown_schemas': index.get('unknown_schemas', []),
            'deferred_checks': list(index['deferred_checks']), 'metrics': dict(index['metrics'])}


def thaw_review_index(cache):
    """Restore topology facts only. Caller must validate dependencies/context."""
    from pxr import UsdGeom
    if cache.get('schema_version') != 1: raise ValueError('unsupported review index schema')
    started = time.monotonic()
    geometry_ids = set(cache['geometry_nodes'])
    class PrimSummary:
        __slots__ = ('schema', 'geometry')
        def __init__(self, schema, geometry): self.schema, self.geometry = schema, geometry
        def GetTypeName(self): return self.schema
        def IsA(self, schema): return schema == UsdGeom.Gprim and self.geometry and self.schema != 'PointInstancer'
    paths, prims, children, geometry = [], {}, defaultdict(list), []
    for i, (parent, name, schema_id) in enumerate(cache['nodes']):
        if parent >= i or parent < -1 or '/' in name: raise ValueError('invalid cached topology')
        path = (paths[parent] if parent >= 0 else '') + '/' + name
        if path in prims: raise ValueError('duplicate cached prim path')
        schema = cache['types'][schema_id]
        paths.append(path); prims[path] = PrimSummary(schema, i in geometry_ids)
        children[path.rsplit('/', 1)[0] or '/'].append(path)
        if i in geometry_ids: geometry.append({'path': path, 'schema': schema})
    facts = {paths[i]: {name: paths[i] for name in names} for i, names in cache['facts']}
    geometry_under = defaultdict(list)
    for row in geometry:
        path = row['path']
        while path:
            geometry_under[path].append(row['path']); path = path.rsplit('/', 1)[0]
    return {'prims': prims, 'children': children, 'geometry': geometry, 'geometry_under': geometry_under,
            'identifying': cache.get('identifying', {}), 'facts': facts, 'protected': {paths[i] for i in cache['protected_nodes']},
            'variant_contexts': cache['variant_contexts'], 'unloaded': cache['unloaded'], 'materials': [],
            'unknown_schemas': cache.get('unknown_schemas', []),
            'deferred_checks': cache['deferred_checks'],
            'metrics': dict(cache['metrics'], review_index_cache_hit=True,
                            index_restore_seconds=time.monotonic()-started)}
