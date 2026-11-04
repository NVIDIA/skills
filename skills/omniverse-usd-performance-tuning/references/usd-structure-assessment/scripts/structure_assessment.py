# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read-only USD assessment used by the tuning workflow.

Paths in ownership and occurrence records are absolute *composed prim paths* in
the declared source context. Definition IDs use confirmed effective content or occurrence-specific source
identity, never generated prototype paths. The independent inventory deliberately
does not accept a manifest: a second scan can expose a jointly omitted target.
"""
from collections import Counter, defaultdict, OrderedDict
from contextvars import ContextVar
import hashlib
import importlib.util
import json
import copy
import math
import struct
from pathlib import Path
import time
import sys


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str,
                                    separators=(',', ':')).encode()).hexdigest()


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _within(path, root):
    return path == root or path.startswith(root.rstrip('/') + '/')


def _protected_path_index(paths, boundaries, protected):
    """Index retained ancestors once; preserve strict-descendant protection."""
    descendants = defaultdict(list)
    for path in sorted(protected):
        if path != '/' and '/' in boundaries:
            descendants['/'].append(path)
        ancestor = path.rsplit('/', 1)[0]
        while ancestor:
            if ancestor in boundaries:
                descendants[ancestor].append(path)
            ancestor = ancestor.rsplit('/', 1)[0]
    nearest = {'': '/' if '/' in protected else None}
    for path in paths:
        pending = []
        ancestor = path
        while ancestor not in nearest:
            pending.append(ancestor)
            ancestor = ancestor.rsplit('/', 1)[0]
        for current in reversed(pending):
            parent = current.rsplit('/', 1)[0]
            nearest[current] = current if current in protected else nearest[parent]
    return descendants, nearest


def _authored_identity(prim):
    """The strongest defining spec, without transient prototype identities."""
    from pxr import Sdf
    stack = prim.GetPrimStack()
    defining = [s for s in stack if s.specifier == Sdf.SpecifierDef]
    chosen = (defining or stack)[-1] if stack else None
    if chosen:
        return str(chosen.layer.realPath or chosen.layer.identifier), str(chosen.path)
    return '', str(prim.GetPath())


def _file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _traverse(stage):
    from pxr import Usd
    # The default predicate excludes unloaded payload roots. Keep those roots
    # visible so that missing coverage cannot turn into an empty success.
    predicate = Usd.PrimIsActive & Usd.PrimIsDefined & ~Usd.PrimIsAbstract
    return Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies(predicate))


def _value(value):
    # Hash typed contiguous arrays directly. Converting millions of numeric
    # elements to decimal text wastes time and can abbreviate array reprs.
    if value is not None and type(value).__module__ == 'pxr.Vt':
        import numpy as np
        array = np.asarray(value)
        if array.dtype.kind in 'biufc':
            array = np.ascontiguousarray(array)
            return {'type': type(value).__name__, 'shape': list(array.shape),
                    'dtype': array.dtype.str,
                    'sha256': hashlib.sha256(memoryview(array).cast('B') if array.size else b'').hexdigest()}
    return str(value)


_property_cache_context = ContextVar('assessment_property_cache', default=None)


def _assessment_property_cache(function):
    """Reuse exact reads only during one read-only assessment; always invalidate on exit."""
    from functools import wraps
    @wraps(function)
    def run(*args, **kwargs):
        token = _property_cache_context.set({'facts': OrderedDict(), 'hits': 0, 'reads': 0, 'geometry_reads': 0,
                                             'bytes': 0, 'peak_bytes': 0, 'evictions': 0, 'oversized': 0})
        try:
            return function(*args, **kwargs)
        finally:
            _property_cache_context.reset(token)
    return run


_PROPERTY_CACHE_BYTES = 64 * 1024 * 1024
_PROPERTY_ITEM_BYTES = 8 * 1024 * 1024


def _property_fact_size(facts):
    # Conservative retained Python object budget; stop counting oversized
    # entries early. Include list elements and dict keys/values, not just arrays.
    pending = [facts]
    size = 512  # cache key, prim handle, tuple and LRU entry overhead
    while pending and size <= _PROPERTY_ITEM_BYTES:
        value = pending.pop()
        size += sys.getsizeof(value)
        if isinstance(value, dict):
            pending.extend(value.keys()); pending.extend(value.values())
        elif isinstance(value, (list, tuple)):
            pending.extend(value)
    return size


def _property_facts(prim, ignore_placement=False, root=None):
    """Return exact facts, shared read-only inside an assessment cache context."""
    cache = _property_cache_context.get()
    if cache is None:
        return _read_property_facts(prim, ignore_placement, root)
    # Root affects connection/relationship normalization. Composed prim paths
    # retain inherited context; do not key on authored identity alone.
    key = (prim, root or str(prim.GetPath()))
    if key in cache['facts']:
        cache['hits'] += 1
        facts, size = cache['facts'].pop(key)
        cache['facts'][key] = facts, size
    else:
        from pxr import UsdGeom
        facts = _read_property_facts(prim, root=root)
        cache['reads'] += 1
        cache['geometry_reads'] += int(prim.IsA(UsdGeom.Gprim))
        size = _property_fact_size(facts)
        if size <= _PROPERTY_ITEM_BYTES:
            while cache['facts'] and cache['bytes'] + size > _PROPERTY_CACHE_BYTES:
                _, (_, removed) = cache['facts'].popitem(last=False)
                cache['bytes'] -= removed
                cache['evictions'] += 1
            cache['facts'][key] = facts, size
            cache['bytes'] += size
            cache['peak_bytes'] = max(cache['peak_bytes'], cache['bytes'])
        else:
            cache['oversized'] += 1
    # Assessment consumers treat these facts as read-only. Lists are shared
    # only inside this private, bounded, read-only assessment context.
    if ignore_placement:
        return facts[:3] + [row for row in facts[3:]
                           if not (row[0].startswith('xformOp:') or row[0] == 'xformOpOrder')]
    return facts


def _read_property_facts(prim, ignore_placement=False, root=None):
    """Exact authored values, samples, metadata, connections and relationships."""
    root = root or str(prim.GetPath())
    def normalized(path):
        path = str(path)
        if _within(path.split('.')[0], root):
            return '.' + path[len(root):]
        if path.startswith('/__Prototype_'):
            # Prototype names depend on stage population order. Resolve their
            # authored identity before values enter persistent fingerprints.
            target = prim.GetStage().GetPrimAtPath(path.split('.')[0])
            if target:
                return str(_authored_identity(target)) + path[len(path.split('.')[0]):]
        return path
    metadata = {key: value for key, value in prim.GetAllAuthoredMetadata().items()
                if key in ('customData', 'assetInfo', 'documentation', 'displayName', 'hidden')}
    facts = [str(prim.GetTypeName()), sorted(prim.GetAppliedSchemas()), metadata]
    for attr in sorted(prim.GetAuthoredAttributes(), key=lambda a: a.GetName()):
        name = attr.GetName()
        if ignore_placement and (name.startswith('xformOp:') or name == 'xformOpOrder'):
            continue
        if not attr.HasAuthoredValue() and not attr.HasAuthoredConnections():
            continue
        samples = [(t, _value(attr.Get(t))) for t in attr.GetTimeSamples()]
        connections = attr.GetConnections() if attr.HasAuthoredConnections() else []
        interpolation = attr.GetMetadata('interpolation') if name.startswith('primvars:') or name == 'normals' else ''
        element_size = attr.GetMetadata('elementSize') if name.startswith('primvars:') else ''
        facts.append([name, str(attr.GetTypeName()), _value(attr.Get()), samples,
                      sorted(normalized(p) for p in connections),
                      str(interpolation or ''), str(element_size or '')])
    for rel in sorted(prim.GetAuthoredRelationships(), key=lambda r: r.GetName()):
        if rel.GetName().startswith('material:binding'):
            # Binding namespace differs per occurrence. Effective material
            # networks are compared separately in the consumer context.
            continue
        facts.append([rel.GetName(), [normalized(p) for p in rel.GetTargets()]])
    return facts


def _connected_network_facts(material_prim, network):
    """Exact rooted, edge-labelled graph independent of shader prim names.

    Stable attribute and connection order assigns node IDs without erasing shared
    upstream nodes or cycles. Disconnected or unresolved graphs retain the older
    conservative key; arbitrary path/string-valued attributes are never renamed.
    """
    from pxr import Sdf
    stage=material_prim.GetStage();identities={};nodes=[];resource_hashes={}
    def value(item):
        if isinstance(item,Sdf.AssetPath):
            if not item.path:return ['asset','']
            resolved=item.resolvedPath
            if not resolved or not Path(resolved).is_file():raise ValueError('unresolved resource')
            if resolved not in resource_hashes:resource_hashes[resolved]=_file_hash(resolved)
            return ['asset',resolved,resource_hashes[resolved]]
        if type(item).__name__=='AssetPathArray':return [value(entry) for entry in item]
        return _value(item)
    def target(path):
        if not stage.GetObjectAtPath(path):raise ValueError('unresolved graph target')
        node=stage.GetPrimAtPath(path.GetPrimPath())
        if str(node.GetPath()) not in network:raise ValueError('target outside inspected graph')
        return [visit(node),path.name if path.IsPropertyPath() else None]
    def visit(node):
        path=str(node.GetPath())
        if path in identities:return identities[path]
        identifier=len(nodes);identities[path]=identifier;nodes.append(None)
        attrs=[]
        for attr in sorted(node.GetAuthoredAttributes(),key=lambda entry:entry.GetName()):
            metadata={key:str(item) for key,item in attr.GetAllAuthoredMetadata().items()
                      if key not in ('default','timeSamples','connectionPaths')}
            attrs.append([attr.GetName(),str(attr.GetTypeName()),metadata,value(attr.Get()),
                          [[time,value(attr.Get(time))] for time in attr.GetTimeSamples()],
                          [target(path) for path in attr.GetConnections()]])
        relationships=[[rel.GetName(),[target(path) for path in rel.GetTargets()]]
                       for rel in sorted(node.GetAuthoredRelationships(),key=lambda entry:entry.GetName())]
        metadata={key:item for key,item in node.GetAllAuthoredMetadata().items()
                  if key in ('customData','assetInfo','documentation','displayName','hidden')}
        nodes[identifier]=[str(node.GetTypeName()),sorted(node.GetAppliedSchemas()),metadata,attrs,relationships]
        return identifier
    try:
        visit(material_prim)
    except (ValueError,OSError,RecursionError):
        return None
    if set(identities)!=set(network):return None
    return ['connected_shader_graph_v1',nodes]


def _addressed_property_targets(stage, prims):
    """Collect composed authored addressing without querying schema-only properties."""
    addressed = set()
    reads = 0
    for prim in prims:
        # Targets and connections require authored property opinions, including
        # those composed from references, inherits and selected variants.
        for prop in prim.GetAuthoredProperties():
            reads += 1
            if prop.GetName().startswith('material:binding'):
                continue
            targets = prop.GetTargets() if hasattr(prop, 'GetTargets') else prop.GetConnections()
            for target in targets:
                target_prim = stage.GetPrimAtPath(target.GetPrimPath())
                if target_prim and target_prim.GetTypeName() in ('Material', 'Shader', 'NodeGraph'):
                    continue
                addressed.add(str(target.GetPrimPath()))
    return addressed, reads


def atomic_correspondence(stage, source, representative, required=(), _cache=None, common_placement=False, policy=None):
    """Exact static content correspondence, optionally factoring common placement.

    Display names may differ only in the placement route, where authoring restores
    their source opinions; authored presence and all other metadata remain exact.
    """
    from pxr import UsdGeom,UsdShade
    if any(p != boundary and _within(p,boundary) for p in required for boundary in (source,representative)):
        return None
    _cache={} if _cache is None else _cache
    signatures=_cache.setdefault("signatures",{});parts=_cache.setdefault("parts",{});materials=_cache.setdefault("materials",{})
    def visit(prim,root):
        path=str(prim.GetPath());cache_key=(path,path==root,common_placement)
        if cache_key in signatures:return signatures[cache_key]
        if prim.IsInstance() or prim.IsInstanceProxy():raise ValueError('instance correspondence uses existing reuse route')
        if prim.IsA(UsdShade.Material):
            signatures[cache_key]=_digest(['material',_effective_appearance(prim,materials)[0]['network_fingerprint']])
            parts[(path,common_placement)]=[]
            return signatures[cache_key]
        if prim.GetTypeName() not in ('','Xform','Scope','Mesh','Cube','Cylinder'):
            raise ValueError('unsupported internal schema')
        if any('Physics' in api or 'Physx' in api for api in prim.GetAppliedSchemas()):
            raise ValueError('physics interface')
        if any(not rel.GetName().startswith('material:binding') for rel in prim.GetAuthoredRelationships()):
            raise ValueError('relationship interface')
        ancestor=prim
        while ancestor:
            if any(rel.GetName().startswith('material:binding') and rel.GetName()!='material:binding'
                   for rel in ancestor.GetAuthoredRelationships()):
                raise ValueError('purpose or collection binding')
            ancestor=ancestor.GetParent()
        if path!=root and any(attr.GetName().startswith('interface:') and attr.Get() for attr in prim.GetAuthoredAttributes()):
            raise ValueError('required internal interface')
        if any(attr.GetConnections() or attr.GetTimeSamples() for attr in prim.GetAuthoredAttributes()):
            raise ValueError('connected or animated interface')
        nested=sorted((visit(child,root),str(child.GetPath())) for child in prim.GetChildren())
        facts=_property_facts(prim,ignore_placement=common_placement or path==root,root=path)
        if common_placement:
            facts=list(facts)
            facts[2]={key:value for key,value in facts[2].items() if key!='displayName'}
            facts.append(['occurrence_displayName_authored',prim.HasAuthoredMetadata('displayName')])
        appearance=_effective_appearance(prim,materials)[0]['network_fingerprint'] if prim.IsA(UsdGeom.Gprim) else None
        binding_metadata=[[rel.GetName(),{key:value for key,value in rel.GetAllAuthoredMetadata().items() if key!='targetPaths'}]
                          for rel in sorted(prim.GetAuthoredRelationships(),key=lambda rel:rel.GetName())
                          if rel.GetName().startswith('material:binding')]
        signatures[cache_key]=_digest([facts,appearance,binding_metadata,[key for key,_ in nested]])
        parts[(path,common_placement)]=nested
        return signatures[cache_key]
    try:
        if visit(stage.GetPrimAtPath(source),source)!=visit(stage.GetPrimAtPath(representative),representative):return None
    except (ValueError,RecursionError):return None
    mapping={}
    def pair(left,right):
        mapping[left]=right
        for (_,child),(_,other) in zip(parts[(left,common_placement)],parts[(right,common_placement)]):pair(child,other)
    pair(source,representative)
    if common_placement:
        helper=_module("atomic_placement",Path(__file__).with_name("atomic_placement.py"))
        if helper.common_placement(stage,source,representative,mapping,policy) is None:return None
    return mapping


def validate_atomic_correspondences(stage,manifest):
    if not any('source_to_definition' in occurrence for occurrence in manifest.get('occurrences',[])):return True
    definitions={d['id']:d for d in manifest.get('definitions',[])}
    cache={}
    required={edit['path'] for edit in manifest.get('required_occurrence_edits',[])}
    required.update(p for d in definitions.values() for p in d.get('protected_paths',[]))
    # Any authored property targeting an internal prim establishes addressability.
    for prim in _traverse(stage):
        for prop in prim.GetProperties():
            if prop.GetName().startswith('material:binding'):continue
            targets=prop.GetTargets() if hasattr(prop,'GetTargets') else prop.GetConnections()
            for target in targets:
                target_prim=stage.GetPrimAtPath(target.GetPrimPath())
                if target_prim and target_prim.GetTypeName() in ('Material','Shader','NodeGraph'):continue
                required.add(str(target.GetPrimPath()))
    for occurrence in manifest.get('occurrences',[]):
        mapping=occurrence.get('source_to_definition')
        if mapping is None:continue
        definition=definitions[occurrence['definition_id']]
        if definition.get('structural_type')!='atomic' or definition.get('children'):
            raise ValueError('correspondence requires a childless atomic asset')
        expected=atomic_correspondence(stage,occurrence['path'],definition['boundary_path'],required,_cache=cache,common_placement='internal_placement' in occurrence,policy=manifest.get('request',{}))
        if not expected or expected!=mapping:
            raise ValueError('atomic correspondence is stale, incomplete or violates required identity')
        if 'internal_placement' in occurrence:
            helper=_module('atomic_placement',Path(__file__).with_name('atomic_placement.py'))
            proof=helper.common_placement(stage,occurrence['path'],definition['boundary_path'],expected,manifest.get('request',{}))
            if proof!=occurrence['internal_placement']:raise ValueError('stale internal placement proof')
    return True


def _effective_appearance(prim, material_cache, target_path_map=None):
    from pxr import UsdShade, Usd
    if prim.IsA(UsdShade.Material):
        material = UsdShade.Material(prim)
    else:
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    if not material:
        return {'material': None, 'network_fingerprint': None}, []
    material_prim = material.GetPrim()
    material_path = str(material_prim.GetPath())
    proxy_key = str(material_prim.GetPrimInPrototype().GetPath()) if material_prim.IsInstanceProxy() else material_path
    cache_key = proxy_key if proxy_key in material_cache else material_path
    if cache_key not in material_cache:
        facts, dependencies, connection_findings = [], [], []
        network = {str(child.GetPath()): child for child in Usd.PrimRange(material_prim)}
        pending = list(network.values())
        while pending:
            child = pending.pop()
            for attr in sorted(child.GetAuthoredAttributes(), key=lambda a: a.GetName()):
                for connection in attr.GetConnections():
                    path = str(connection.GetPrimPath())
                    if path in network:
                        connected = network[path]
                    else:
                        connected = prim.GetStage().GetPrimAtPath(path)
                    reason = None
                    if not connected:
                        reason = 'missing_shader_prim'
                    elif not connection.IsPropertyPath() or not prim.GetStage().GetAttributeAtPath(connection):
                        reason = 'missing_shader_property'
                    if reason:
                        connection_findings.append({'owner_path': str(child.GetPath()),
                            'property_path': str(attr.GetPath()), 'target_path': str(connection),
                            'reason': reason})
                    if not connected:
                        continue
                    if path not in network:
                        network[path] = connected
                        pending.append(connected)
        aliases = {path: ('.' + path[len(material_path):] if _within(path, material_path)
                          else 'connected_' + str(i)) for i, path in enumerate(network)}
        def normalize(value):
            if isinstance(value, str):
                for node_path, alias in aliases.items():
                    if value == node_path or value.startswith(node_path + '.'):
                        return alias + value[len(node_path):]
            if isinstance(value, list):
                return [normalize(item) for item in value]
            if isinstance(value, dict):
                return {key: normalize(item) for key, item in value.items()}
            return value
        for node_path, child in network.items():
            properties = _property_facts(child, root=material_path)
            if target_path_map:
                # Remap addressing only; shader values and metadata stay exact.
                def remap_target(path):
                    for prefix, replacement in target_path_map.items():
                        if path == prefix or path.startswith(prefix + '/') or path.startswith(prefix + '.'):
                            return replacement + path[len(prefix):]
                    return path
                properties = properties[:3] + [
                    row[:4] + [[remap_target(path) for path in row[4]]] + row[5:]
                    if len(row) > 2 else [row[0], [remap_target(path) for path in row[1]]]
                    for row in properties[3:]]
            facts.append([aliases[node_path], normalize(properties)])
            layer, _ = _authored_identity(child)
            if layer:
                dependencies.append({'asset_path': layer, 'resolved_path': layer,
                                     'kind': 'material' if child == material_prim else 'shader',
                                     'exists': Path(layer).is_file()})
            for attr in child.GetAuthoredAttributes():
                if str(attr.GetTypeName()) not in ('asset', 'asset[]'):
                    continue
                values = attr.Get()
                if values is None:
                    continue
                if str(attr.GetTypeName()) == 'asset':
                    values = [values]
                for value in values:
                    if not value.path:
                        continue
                    resolved = value.resolvedPath
                    dependencies.append({'asset_path': value.path, 'resolved_path': resolved,
                                         'kind': 'texture', 'exists': bool(resolved)})
        # Instance-local shading networks are immutable across occurrences.
        # An external consumer shader may carry overrides, so keep its context.
        safe_proxy = material_prim.IsInstanceProxy() and all(child.IsInstanceProxy() for child in network.values())
        cache_key = proxy_key if safe_proxy else material_path
        connected_facts = _connected_network_facts(material_prim, network)
        material_cache[cache_key] = (_digest(connected_facts if connected_facts is not None else facts), dependencies, connection_findings)
    fingerprint, dependencies, connection_findings = material_cache[cache_key]
    return {'material': material_path, 'network_fingerprint': fingerprint,
            'connection_findings': connection_findings}, dependencies


def _authored_extent(prim):
    """Six local-space scalars for conservative candidate eviction, never proof."""
    attr = prim.GetAttribute('extent')
    if not attr or not attr.HasAuthoredValueOpinion() or attr.ValueMightBeTimeVarying():
        return None
    value = attr.Get()
    if value is None or len(value) != 2:
        return None
    try:
        bounds = struct.unpack('=6f', value)
    except (TypeError, BufferError, struct.error):
        return None
    if (not all(map(math.isfinite, bounds)) or
            any(bounds[i] > bounds[i + 3] for i in range(3))):
        return None
    return bounds


def scan_inventory(stage, request=None, progress=None, defer_geometry=False):
    """Observe all composed geometry, including instance proxies, independently.

    Local geometry arrays are fingerprinted once per prototype prim unless
    defer_geometry requests nomination without exact values. Occurrence
    placement and effective material are inspected at each composed site.
    Unsupported/unloaded data remains explicit. Callers may pass a path or stage.
    """
    from pxr import Usd, UsdGeom
    request = request or {}
    started = time.monotonic()
    if isinstance(stage, (str, Path)):
        stage = _open_stage(stage, request)
    if stage is None:
        raise ValueError('Could not open source stage')
    geometry, uninspected, materials, variant_contexts = [], [], {}, []
    inventory_only = bool(request.get('inventory_only'))
    geometry_cache, extent_cache, xform_caches = {}, {}, {}
    times = request.get('time_samples', [None])
    if not times:
        raise ValueError('time_samples must not be empty')
    for sample in times:
        tc = Usd.TimeCode.Default() if sample is None else Usd.TimeCode(float(sample))
        xform_caches[sample] = UsdGeom.XformCache(tc)
    prims = list(_traverse(stage))
    for error in stage.GetCompositionErrors():
        uninspected.append({'path': '/', 'reason': 'composition_error', 'error': str(error)})
    for index, prim in enumerate(prims):
        if progress and index % 2048 == 0:
            progress("inventory", prims_done=index, prims_total=len(prims), geometry_done=len(geometry))
        path = str(prim.GetPath())
        if prim.HasPayload() and not prim.IsLoaded():
            uninspected.append({'path': path, 'reason': 'unloaded_payload'})
        for name in prim.GetVariantSets().GetNames():
            variant = prim.GetVariantSets().GetVariantSet(name)
            selection = variant.GetVariantSelection()
            alternatives = [v for v in variant.GetVariantNames() if v != selection]
            variant_contexts.append({'path': path, 'set': name, 'selection': selection,
                                     'unselected': alternatives})
            if alternatives and request.get('require_all_variants', not request.get('successor_policy', False)):
                uninspected.append({'path': path, 'reason': 'uninspected_variants',
                                    'variant_set': name, 'selections': alternatives})
        if not (prim.IsA(UsdGeom.Gprim) or prim.IsA(UsdGeom.PointInstancer)):
            continue
        if inventory_only:
            # The final audit checks appearance and placement itself using
            # shared caches. This pass only independently enumerates geometry.
            geometry.append({'path': path, 'schema': str(prim.GetTypeName())})
            continue
        try:
            content_prim = prim.GetPrimInPrototype() if prim.IsInstanceProxy() else prim
            key = str(content_prim.GetPath())
            if not defer_geometry and key not in geometry_cache:
                geometry_cache[key] = _digest(_property_facts(content_prim, ignore_placement=True))
            if defer_geometry and key not in extent_cache:
                extent_cache[key] = _authored_extent(content_prim)
            appearance, dependencies = _effective_appearance(prim, materials)
            transforms = {str(sample): str(cache.GetLocalToWorldTransform(prim))
                          for sample, cache in xform_caches.items()}
            layer, authored_path = _authored_identity(prim)
            geometry.append({'path': path, 'schema': str(prim.GetTypeName()),
                             'source_layer': layer, 'source_prim_path': authored_path,
                             'content_fingerprint': geometry_cache.get(key),
                             'coarse_extent': extent_cache.get(key),
                             'appearance': appearance, 'world_transforms': transforms,
                             'material_dependencies': dependencies,
                             'context_fingerprint': _digest([appearance, transforms])})
        except Exception as exc:
            uninspected.append({'path': path, 'reason': 'geometry_read_failed', 'error': str(exc)})
    root = stage.GetRootLayer()
    source = root.realPath or root.identifier
    dependency_paths = set()
    for layer in sorted(stage.GetUsedLayers(), key=lambda item: item.identifier):
        path = layer.realPath
        if path and Path(path).is_file():
            dependency_paths.add(path)
    connection_findings = {}
    for fingerprint, material_dependencies, failures in materials.values():
        for finding in failures:
            record = {**finding, 'network_fingerprint': fingerprint}
            connection_findings[json.dumps(record, sort_keys=True)] = record
        for dependency in material_dependencies:
            resolved = dependency['resolved_path']
            if resolved and Path(resolved).is_file():
                dependency_paths.add(resolved)
            elif dependency['kind'] in ('texture', 'resource'):
                uninspected.append({'path': dependency['asset_path'],
                                    'reason': 'unreadable_material_resource'})
    dependencies = [{'path': path, 'fingerprint': _file_hash(path)} for path in sorted(dependency_paths)]
    return {'geometry': geometry, 'uninspected': uninspected, 'variant_contexts': variant_contexts,
            'connection_findings': list(connection_findings.values()),
            'source_fingerprint': _digest(dependencies), 'dependencies': dependencies,
            'source_path': source, 'geometry_values_read': not inventory_only and not defer_geometry,
            'appearance_observed': not inventory_only,
            'dependency_scope': 'usd_layers' if inventory_only else 'usd_layers_and_material_resources',
            'meters_per_unit': UsdGeom.GetStageMetersPerUnit(stage),
            'up_axis': str(UsdGeom.GetStageUpAxis(stage)),
            'metrics': {'scan_count': 1, 'prim_count': len(prims),
                        'geometry_count': len(geometry), 'unique_geometry_reads': len(geometry_cache),
                        'elapsed_seconds': time.monotonic() - started}}


def source_evidence_request(request):
    """Only requested protection can change without invalidating source evidence."""
    return {key: value for key, value in request.items() if key not in ('protected_paths', 'required_occurrence_edits')}


def _open_stage(source, request):
    from pxr import Usd, UsdGeom
    load = request.get('load', 'all')
    if load not in ('all', 'none'):
        raise ValueError('load must be all or none')
    stage = Usd.Stage.Open(str(source), load=Usd.Stage.LoadAll if load == 'all' else Usd.Stage.LoadNone)
    if not stage:
        raise ValueError('Could not open source stage: ' + str(source))
    # All assessment opinions are isolated in an anonymous session layer.
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for path, selections in request.get('variants', {}).items():
            prim = stage.GetPrimAtPath(path)
            if not prim:
                raise ValueError('Variant prim does not exist: ' + path)
            for name, selection in selections.items():
                variant = prim.GetVariantSets().GetVariantSet(name)
                if selection not in variant.GetVariantNames():
                    raise ValueError('Unknown variant selection: ' + path + ':' + name + '=' + selection)
                variant.SetVariantSelection(selection)
    return stage


def _decisions_by_path(decisions):
    if decisions is None:
        return {}
    if isinstance(decisions, dict) and ('boundaries' in decisions or 'groups' in decisions):
        decisions = decisions.get('boundaries', [])
    if isinstance(decisions, dict):
        decisions = [dict(value, path=path) for path, value in decisions.items()]
    if not isinstance(decisions, list):
        raise ValueError('Boundary decisions must be a list or path mapping')
    result = {}
    for decision in decisions:
        if not isinstance(decision, dict) or not str(decision.get('path', '')).startswith('/'):
            raise ValueError('Boundary decision needs an absolute prim path')
        unknown = set(decision) - {'product_identity', 'path', 'structural_type', 'reason', 'boundary', 'packaging', 'selection', 'kind', 'standalone_kind', 'equipment_family', 'geometry_representation', 'semantic_category', 'decision_scope'}
        if unknown:
            raise ValueError('Unknown boundary decision fields: ' + ', '.join(sorted(unknown)))
        if 'geometry_representation' in decision and decision['geometry_representation'] not in ('merged_meshes', 'analytic_primitives'):
            raise ValueError('Invalid geometry_representation')
        if 'equipment_family' in decision and (not isinstance(decision['equipment_family'], str) or not decision['equipment_family'].strip()):
            raise ValueError('equipment_family requires a nonempty label')
        if 'decision_scope' in decision and (not isinstance(decision.get('boundary'), bool) or decision['decision_scope'] not in ('prim', 'subtree')):
            raise ValueError('decision_scope requires a retained or internal prim or subtree decision')
        if decision.get('structural_type') not in (None, 'atomic', 'aggregate', 'package'):
            raise ValueError('Unknown structural type')
        if decision.get('structural_type') and not str(decision.get('reason', '')).strip():
            raise ValueError('Structural decision requires a reason')
        for field in ('kind', 'standalone_kind'):
            if field in decision and decision[field] not in ('component', 'subcomponent', 'assembly', 'group'):
                raise ValueError('Unsupported model kind in ' + field)
        if decision['path'] in result:
            raise ValueError('Repeated boundary decision: ' + decision['path'])
        result[decision['path']] = decision
    return result


def _pending_boundary_manifest(stage, source_path, request, choices, group_choices, progress, review_state=None, cached_review=None):
    """Review complete topology before evaluating geometry inside chosen assets.

    A pending manifest cannot supply jobs or bypass the later authoritative
    assessment. Dependencies include material resources even though effective
    binding and compatibility analysis waits for the reviewed scopes.
    """
    from pxr import Usd, UsdGeom
    started = time.monotonic()
    preview = _module('successor_review_index', Path(__file__).with_name('discovery_preview.py'))
    reasoning = _module('successor_review_reasoning', Path(__file__).with_name('boundary_reasoning.py'))
    if cached_review is not None:
        if (cached_review.get('assessment_phase') != 'boundary_review' or source_evidence_request(cached_review.get('request', {})) != source_evidence_request(request)
                or cached_review.get('source', {}).get('path') != str(Path(source_path).resolve())):
            raise ValueError('review index source or request differs')
        frozen = cached_review.get('review_index')
        if not frozen or _digest(frozen) != cached_review.get('review_index_sha256'):
            raise ValueError('cached review index evidence changed')
        for dependency in cached_review['inventory']['dependencies']:
            if not Path(dependency['path']).is_file() or _file_hash(dependency['path']) != dependency['fingerprint']:
                raise ValueError('cached review dependency changed')
        if not frozen.get('source_evidence_only'):
            return _pending_boundary_manifest(stage, source_path, request, choices, group_choices, progress, review_state)
        index = preview.thaw_review_index(frozen)
        if 'identifying' not in frozen:
            index['identifying'] = {p: reasoning.identifying_facts(stage.GetPrimAtPath(p)) for p, prim in index['prims'].items()
                                   if not prim.IsA(UsdGeom.Gprim) and prim.GetTypeName() not in ('Shader', 'Material', 'NodeGraph')}
            frozen = preview.freeze_review_index(index)
    else:
        index = preview.review_index(stage, (), progress)
        frozen = preview.freeze_review_index(index)
        frozen['source_evidence_only'] = True
    if cached_review is not None:
        progress('source_evidence_reused', prims_done=len(index['prims']), prims_total=len(index['prims']))
        index['metrics']['scan_count'] = 0
    # Request protection is an overlay, never persisted as authored source evidence.
    for path in request.get('protected_paths', ()):
        if path in index['prims']:
            index['protected'].add(path)
            index['facts'].setdefault(path, {})['interface'] = path
    prims, children, geometry_under = index['prims'], index['children'], index['geometry_under']
    unknown = set(choices) - set(prims)
    if unknown: raise ValueError('Decision paths do not exist: ' + ', '.join(sorted(unknown)))
    for choice in choices.values():
        if 'selection' in choice: reasoning.validate_selection(choice['selection'], stage)
    cache = {}
    for material in index['materials']: _effective_appearance(material, cache)
    dependencies = {layer.realPath for layer in stage.GetUsedLayers() if layer.realPath and Path(layer.realPath).is_file()}
    uninspected = [{'path': p, 'reason': 'unloaded_payload'} for p in index['unloaded']]
    for path, prim in prims.items() if index.get('unknown_schemas') else ():
        if prim.GetTypeName() in index.get('unknown_schemas', ()):
            uninspected.append({'path': path, 'schema': prim.GetTypeName(), 'reason': 'unavailable_schema_requires_capability_review'})
    uninspected += [{'path': '/', 'reason': 'composition_error', 'error': str(e)} for e in stage.GetCompositionErrors()]
    findings = {}
    for fingerprint, resources, failures in cache.values():
        for resource in resources:
            resolved = resource['resolved_path']
            if resolved and Path(resolved).is_file(): dependencies.add(resolved)
            elif resource['kind'] in ('texture', 'resource'):
                uninspected.append({'path': resource['asset_path'], 'reason': 'unreadable_material_resource'})
        for failure in failures:
            row = dict(failure, network_fingerprint=fingerprint)
            findings[json.dumps(row, sort_keys=True)] = row
    if cached_review is not None:
        deps = cached_review['inventory']['dependencies']
        findings = {json.dumps(row,sort_keys=True): row for row in cached_review['inventory']['connection_findings']}
        uninspected = cached_review['inventory']['uninspected']
    else:
        deps = [{'path': path, 'fingerprint': _file_hash(path)} for path in sorted(dependencies)]
    source_fingerprint = _digest(deps)
    required = {}
    for path, prim in prims.items():
        if not geometry_under[path]: continue
        reasons = []
        if path in children['/']: reasons.append('root_asset')
        if path in index['protected'] and choices.get(path, {}).get('boundary') is not False:
            reasons.append('protected_interface')
        if choices.get(path, {}).get('boundary') or choices.get(path, {}).get('structural_type'):
            reasons.append('declared_boundary')
        if choices.get(path, {}).get('boundary') is False:
            if path in children['/']: raise ValueError('Cannot remove a stage root boundary: ' + path)
            reasons = []
        if reasons: required[path] = reasons
    progress('boundary_review', **index['metrics'])
    reviews, unresolved, used = reasoning.collect_reviews(stage, prims, children, geometry_under, {},
        required, choices, group_choices, source_fingerprint, index['facts'], index.get('identifying'))
    for row in reviews + unresolved:
        row.setdefault('evidence', {})['deferred_checks'] = index['deferred_checks']
        if 'all_members' in row['evidence']:
            row['evidence']['all_members']['signal_scope'] = 'container motion/interfaces and relationships; geometry attribute connections deferred'
    # All boundary judgments are present: continue into the existing detailed
    # assessment. No frozen cheap record can be promoted directly to executable.
    if not unresolved:
        if review_state is not None: review_state['protected'] = set(index['protected'])
        return None
    unused = set(group_choices) - used
    if unused: raise ValueError('Grouped decision is stale or unknown: ' + ', '.join(sorted(unused)))
    source = {'path': str(Path(source_path).resolve()), 'fingerprint': source_fingerprint,
              'meters_per_unit': UsdGeom.GetStageMetersPerUnit(stage),
              'up_axis': str(UsdGeom.GetStageUpAxis(stage))}
    owned = defaultdict(list)
    identifiers = {path: 'definition-' + _digest([source_fingerprint, path])[:24] for path in required}
    for row in index['geometry']:
        owner = row['path']
        while owner and owner not in required: owner = owner.rsplit('/', 1)[0]
        row['definition_id'] = identifiers.get(owner)
        if owner: owned[owner].append(row['path'])
    definitions, occurrences = [], []
    geometry_schema = {row['path']: row['schema'] for row in index['geometry']}
    for path, reasons in sorted(required.items()):
        parent = path.rsplit('/', 1)[0]
        while parent and parent not in required: parent = parent.rsplit('/', 1)[0]
        choice = choices.get(path, {})
        definitions.append({'id': identifiers[path], 'boundary_path': path,
            'structural_type': choice.get('structural_type'), 'rationale': {'reason': 'Boundary review is pending.', 'evidence': reasons},
            'selection': reasoning.selection_for(path, reasons, choice), 'occurrences': [path],
            'children': [], 'protected_paths': sorted(p for p in index['protected'] if _within(p,path)),
            'assessment': {'status': 'decision_required', 'operations': []},
            'geometry_counts': dict(Counter(geometry_schema[p] for p in owned[path])),
            'fingerprint_method': 'pending_detailed_assessment', 'content_fingerprint': None})
        for field in ('product_identity', 'standalone_kind', 'equipment_family', 'geometry_representation', 'semantic_category', 'packaging'):
            if field in choice: definitions[-1][field] = copy.deepcopy(choice[field])
        occurrences.append({'path': path, 'definition_id': identifiers[path], 'parent_path': parent or None,
                            'effective_kind': choice.get('kind', ''), 'source_kind': str(Usd.ModelAPI(stage.GetPrimAtPath(path)).GetKind() or '')})
    for definition in definitions:
        definition['children'] = [row['definition_id'] for row in occurrences if row['parent_path'] == definition['boundary_path']]
    variants = {}
    for row in index['variant_contexts']: variants.setdefault(row['path'], {})[row['set']] = row['selection']
    inventory = {'geometry': index['geometry'], 'dependencies': deps, 'source_fingerprint': source_fingerprint,
                 'source_path': source['path'], 'uninspected': uninspected, 'variant_contexts': index['variant_contexts'],
                 'connection_findings': list(findings.values()), 'geometry_values_read': False, 'appearance_observed': False,
                 'dependency_scope': 'usd_layers_and_material_resources', 'metrics': index['metrics'],
                 'meters_per_unit': source['meters_per_unit'], 'up_axis': source['up_axis']}
    unresolved.append({'type': 'detailed_assessment_pending', 'reason': 'After boundary review, inspect effective appearance and confirm geometry before authoring or scheduling.'})
    manifest = {'schema_version': 2, 'mode': 'restructure', 'input_stage': source['path'], 'source': source,
        'request': copy.deepcopy(request), 'context': {'load': request.get('load','all'), 'variants': variants,
        'time_samples': request.get('time_samples',[None])}, 'status': 'incomplete', 'assessment_phase': 'boundary_review',
        'review_index': frozen, 'review_index_sha256': _digest(frozen),
        'definitions': definitions, 'occurrences': occurrences, 'inventory': inventory, 'boundary_reviews': reviews,
        'boundary_review_version': 1, 'boundary_decisions': list(choices.values()), 'decisions_required': unresolved,
        'reuse_candidates': [], 'repetition_nominations': [n for row in reviews for n in row['evidence'].get('repetition_nominations', [])], 'fragmentation_candidates': [], 'phase4_targets': [],
        'conformance': {'passed': False, 'reason': 'Detailed assessment has not run.'},
        'discovery': {'method': 'exhaustive_recursive_inventory', 'source_scan': {
            'prim_count': len(prims), 'geometry_count': len(index['geometry']), 'instance_proxy_descent': True,
            'bottom_up_confirmation': False, 'geometry_paths_fingerprint': _digest(sorted(row['path'] for row in index['geometry'])),
            'confirmed_reuse_groups': 0}, 'kind_trust': {'kind_trusted': False, 'reason': 'Source kind does not determine successor asset boundaries.'}}}
    inventory['metrics']['assessment_seconds'] = time.monotonic()-started
    compact_manifest(manifest)
    progress('boundary_review_complete', elapsed_seconds=time.monotonic()-started, review_groups=len(reviews))
    return manifest


def _group_decisions(decisions):
    """Read either supported grouped-decision representation."""
    group_choices = decisions.get('groups', {}) if isinstance(decisions, dict) else {}
    if isinstance(group_choices, list):
        group_choices = {item['group_id']: item for item in group_choices}
    return group_choices


def _collect_interface_facts(prims, walker, request, successor):
    """Collect addressed interfaces and authored motion independently of boundaries."""
    protected = set(request.get('protected_paths', []))
    review_facts = {}
    for path, prim in prims.items():
        facts = review_facts[path] = {}
        if path in walker.anim_self or path in walker.body_self:
            facts['motion'] = path
        if prim.GetVariantSets().GetNames():
            facts['variants'] = path
        if (path in protected or prim.GetAttribute('asset:interface') or
                any('Semantic' in api for api in prim.GetAppliedSchemas()) or prim.GetAssetInfo().get('identifier')):
            facts['interface'] = path
        if (path in walker.body_self or path in walker.anim_self or
                any('ArticulationRoot' in schema for schema in prim.GetAppliedSchemas())):
            protected.add(path)
        if successor and prim.GetAttribute('interface:editable').Get() is True:
            protected.add(path)
        for rel in prim.GetAuthoredRelationships():
            if rel.GetName().startswith('material:'):
                continue
            facts['nonmaterial_relationships'] = path
            for target in rel.GetTargets():
                target = str(target.GetPrimPath())
                if target in prims:
                    protected.add(target)
        for attr in prim.GetAuthoredAttributes():
            if prim.GetTypeName() in ('Shader', 'Material', 'NodeGraph'):
                continue
            for target in attr.GetConnections():
                target = str(target.GetPrimPath())
                if target in prims and prims[target].GetTypeName() not in ('Shader', 'Material', 'NodeGraph'):
                    protected.add(target)

    for path in protected:
        if path in review_facts:
            review_facts[path]['interface'] = path

    return protected, review_facts


def _scope_reuse_nominees(prims, children, geometry_by_path, geometry_under, nominees,
                          candidate_hashes, required, choices, metrics):
    """Restrict structural nominations to asset scopes and filter by known extents."""
    def boundary_path(path):
        current = path
        while current and current not in required:
            current = current.rsplit('/', 1)[0]
        return current or path
    # Reject converter fragments as independent comparison scopes before reading
    # their arrays. Nomination remains conservative: structure never proves reuse.
    scoped_nominees = defaultdict(set)
    for paths in nominees.values():
        if len(paths) < 2:
            continue
        for path in paths:
            candidate = boundary_path(path)
            scoped_nominees[candidate_hashes.get(candidate, ('scope', candidate))].add(candidate)
    nominees = {key: sorted(paths) for key, paths in scoped_nominees.items() if len(paths) > 1}
    # Bottom-up extent keys include descendant names, but exclude root placement.
    # Any unknown extent retains its entire structural bucket for exact comparison.
    extent_keys, extent_intern = {}, {}
    for path in sorted(prims, key=lambda p: (-p.count('/'), p)):
        row = geometry_by_path.get(path)
        child_keys = [(prims[child].GetName(), extent_keys[child])
                      for child in children[path] if geometry_under[child]]
        bounds = row.get('coarse_extent') if row is not None else None
        if ((row is not None and bounds is None) or any(key is None for _, key in child_keys)):
            extent_keys[path] = None
        else:
            key = (bounds, tuple(child_keys))
            extent_keys[path] = extent_intern.setdefault(key, len(extent_intern) + 1)
    filtered_nominees, rejected_scopes, unknown_buckets = {}, 0, 0
    for signature, paths in nominees.items():
        if any(extent_keys[path] is None for path in paths):
            unknown_buckets += 1
            filtered_nominees[signature] = paths
            continue
        matches = defaultdict(list)
        for path in paths:
            matches[extent_keys[path]].append(path)
        for key, members in matches.items():
            if len(members) > 1:
                filtered_nominees[(signature, key)] = members
            else:
                rejected_scopes += 1
    product_roots = [path for path, choice in choices.items() if choice.get('product_identity')]
    nominees = {key: [path for path in paths if not any(_within(path, root) or _within(root, path) for root in product_roots)]
                for key, paths in filtered_nominees.items()}
    nominees = {key: paths for key, paths in nominees.items() if len(paths)>1}
    metrics['bounds_rejected_scopes'] = rejected_scopes
    metrics['bounds_unknown_buckets'] = unknown_buckets
    return nominees, product_roots


def _confirm_reuse_content(prims, children, geometry_by_path, nominees, inventory, progress):
    """Read nominated content once, computing exact identities from children to parents."""
    confirmation_paths = set()
    for paths in nominees.values():
        for path in paths:
            stack = [path]
            while stack:
                child = stack.pop()
                if child in confirmation_paths:
                    continue
                confirmation_paths.add(child)
                stack.extend(children[child])

    # Confirm with exact values/metadata, effective appearance, child names and
    # transforms. Compute these Merkle digests once, children before parents;
    # rescanning each ancestor repeats arrays and grows with hierarchy depth.
    exact_full, confirmation_cache, proxy_facts, geometry_facts = {}, {}, {}, {}
    progress("exact_confirmation", candidate_buckets=len(nominees))
    confirmation_start = time.monotonic()
    for index, path in enumerate(sorted(confirmation_paths, key=lambda p: (-p.count('/'), p))):
        if index % 2048 == 0:
            progress('exact_confirmation', prims_done=index, prims_total=len(confirmation_paths),
                     unique_geometry_reads=len(geometry_facts))
        prim = prims[path]
        row = geometry_by_path.get(path)
        if row is not None:
            content_prim = prim.GetPrimInPrototype() if prim.IsInstanceProxy() else prim
            key = str(content_prim.GetPath())
            if key not in geometry_facts:
                geometry_facts[key] = _digest(_property_facts(content_prim, ignore_placement=True))
            row['content_fingerprint'] = geometry_facts[key]
            facts = [row['content_fingerprint']]
            root_facts = facts
            transforms = [[a.GetName(), _value(a.Get()),
                           [(t, _value(a.Get(t))) for t in a.GetTimeSamples()]]
                          for a in prim.GetAuthoredAttributes() if a.GetName().startswith('xformOp')]
            facts = facts + transforms
        else:
            if prim.IsInstanceProxy():
                prototype_prim = prim.GetPrimInPrototype()
                key = str(prototype_prim.GetPath())
                if key not in proxy_facts:
                    proxy_facts[key] = _property_facts(prototype_prim, root=key)
                facts = proxy_facts[key]
            else:
                facts = _property_facts(prim, root=path)
            root_facts = facts[:3] + [entry for entry in facts[3:]
                                      if not str(entry[0]).startswith('xformOp')]
        appearance = row['appearance']['network_fingerprint'] if row is not None else None
        descendants = [(prims[p].GetName(), exact_full[p]) for p in children[path]]
        exact_full[path] = _digest([facts, appearance, descendants])
        confirmation_cache[path] = _digest([root_facts, appearance, descendants])
    inventory['metrics']['confirmation_seconds'] = time.monotonic() - confirmation_start
    inventory['metrics']['exact_geometry_reads'] = len(geometry_facts)
    inventory['metrics']['unique_geometry_reads'] = len(geometry_facts)
    inventory['metrics']['exact_prim_reads'] = len(confirmation_paths)
    inventory['geometry_values_read'] = bool(geometry_facts)
    return confirmation_cache


@_assessment_property_cache
def assess(source_path, request=None, decisions=None, progress=None, cached_review=None):
    """Build the authoritative definition manifest without authoring source USD."""
    from pxr import Sdf, Usd, UsdGeom
    request = dict(request or {})
    request['tolerance_mm'] = _module('assessment_geometry_budget', Path(__file__).resolve().parents[2]/'geometry_tolerance.py').physical_tolerance_mm(request)
    if type(request.get('successor_policy', False)) is not bool:
        raise ValueError('successor_policy must be a boolean')
    successor = request.get('successor_policy', False)
    choices = _decisions_by_path(decisions)
    group_choices = _group_decisions(decisions)
    if not isinstance(group_choices, dict):
        raise ValueError('Grouped decisions must be a group-ID mapping or list')
    for group_id, choice in group_choices.items():
        if isinstance(choice, dict) and 'disposition' in choice:
            continue
        if (not isinstance(choice, dict) or choice.get('structural_type') not in ('atomic', 'aggregate', 'package')
                or not str(choice.get('reason', '')).strip()):
            raise ValueError('Grouped decision requires structural_type and reason: ' + str(group_id))
        if set(choice) - {'group_id', 'structural_type', 'reason', 'evidence_fingerprint'}:
            raise ValueError('Unknown grouped-decision fields')
    started = time.monotonic()
    report_progress = progress or (lambda phase, **counts: None)
    def progress(phase, **counts):
        cache = _property_cache_context.get()
        report_progress(phase, property_reads=cache['reads'],
                        geometry_property_reads=cache['geometry_reads'],
                        property_cache_hits=cache['hits'], **counts)
    progress("stage_open")
    stage = _open_stage(source_path, request)
    open_seconds = time.monotonic() - started
    products = {}
    for path, choice in choices.items():
        if 'product_identity' not in choice: continue
        if not choice.get('boundary'): raise ValueError('product_identity requires a retained asset')
        value=choice['product_identity']; token=_digest(value)
        products.setdefault(token, [value, []])[1].append(path)
    if products:
        clues=_module('assessment_product_identity',Path(__file__).resolve().parents[3]/'scripts/product_clues.py')
        for value, paths in products.values():clues.validate_product_identity(value,stage,paths)
    review_state = {}
    if successor:
        pending = _pending_boundary_manifest(stage, source_path, request, choices, group_choices, progress, review_state, cached_review)
        if pending is not None:
            pending['inventory']['metrics']['stage_open_seconds'] = open_seconds
            return pending
    progress("inventory", stage_open_seconds=open_seconds)
    animation = None
    inventory_request = request
    if successor:
        animation_helper = _module('successor_animation_context', Path(__file__).resolve().parents[3] / 'scripts' / 'animation_context.py')
        animation = animation_helper.collect(stage, requested_times=request.get('time_samples', ()))
        inventory_request = dict(request, time_samples=animation['time_samples'])
    inventory = scan_inventory(stage, inventory_request, progress=progress, defer_geometry=True)
    if animation is not None:
        inventory['animation'] = animation
    progress("candidate_nomination", **inventory["metrics"])
    inventory['metrics']['stage_open_seconds'] = open_seconds
    reasoning = _module('workflow_boundary_reasoning', Path(__file__).with_name('boundary_reasoning.py'))
    for choice in choices.values():
        if 'selection' in choice:
            reasoning.validate_selection(choice['selection'], stage)
    geometry_by_path = {g['path']: g for g in inventory['geometry']}
    prims = {str(p.GetPath()): p for p in _traverse(stage)}
    children = defaultdict(list)
    for path in prims:
        children[path.rsplit('/', 1)[0] or '/'].append(path)
    geometry_under = defaultdict(list)
    for path in geometry_by_path:
        parent = path
        while parent:
            geometry_under[parent].append(path)
            parent = parent.rsplit('/', 1)[0]
    unknown_paths = set(choices) - set(prims)
    if unknown_paths:
        raise ValueError('Decision paths do not exist: ' + ', '.join(sorted(unknown_paths)))

    base = Path(__file__).resolve().parents[1]
    finder = _module('workflow_instance_candidates', base / 'references/usd-hierarchy-dedupe-candidates/scripts/instance_candidate_finder.py')
    fragmenter = _module('workflow_fragmentation', base / 'references/usd-mesh-fragmentation-candidates/scripts/mesh_fragmentation_finder.py')
    # Retain every nomination, including those inside a matched parent and a
    # pre-existing instance. Exact confirmation below is distinct from hashes.
    nomination_visits = [0]
    def inventory_tokens(prim, include_name, include_xformops):
        # Cheap nomination deliberately ignores values and transforms. This
        # broadens candidate buckets; exact confirmation rejects false matches.
        nomination_visits[0] += 1
        if nomination_visits[0] % 2048 == 0:
            progress('candidate_nomination', prims_done=nomination_visits[0] // 2,
                     prims_total=len(prims))
        tokens = ['T:' + str(prim.GetTypeName())]
        if include_name:
            tokens.append('N:' + prim.GetName())
        return tokens
    walker = finder._Walker(stage, inventory_tokens)
    nomination_start = time.monotonic()
    for path in children['/']:
        walker.visit(prims[path])
    inventory['metrics']['nomination_seconds'] = time.monotonic() - nomination_start
    # Share the candidate finder's stage-wide trust gate, including its exception
    # for deliberately authored subcomponents and its full-descent coverage gate.
    counts, excluded = finder._filter_kind_trust_sample(
        walker.kind_samples, walker.kind_by_path, walker.mesh)
    kind_trust = finder._kind_trust_stats(counts, finder.KIND_TRUST_MAX_SINGLE_MESH_FRACTION)
    kind_trust = finder._apply_sample_coverage(
        kind_trust, finder._kind_prims_on_stage(stage, stage.GetPseudoRoot()),
        len(walker.kind_samples))
    kind_trust['excluded_fine_grain_subcomponents'] = excluded
    kind_trust['articulation_check'] = 'enabled' if walker._physics is not None else 'unavailable'
    inventory['kind_trust'] = kind_trust
    if kind_trust['insufficient_sample']:
        inventory['uninspected'].append({'path': '/', 'reason': 'kind_trust_sample_incomplete'})
    nominees = defaultdict(list)
    for path, candidate_hash in walker.cand.items():
        if geometry_under[path] and prims[path].GetTypeName() == 'Xform':
            nominees[candidate_hash].append(path)

    progress("protected_interfaces", nomination_seconds=inventory["metrics"]["nomination_seconds"])
    protected, review_facts = _collect_interface_facts(prims, walker, request, successor)

    # Required identity is distinct from a reuse candidate or a safe merge stop.
    # A CAD solid's mesh count, name or untrusted kind does not make it an asset.
    required = {}
    for path, prim in prims.items():
        if not geometry_under[path]:
            continue
        evidence = []
        if path in children['/']:
            evidence.append('root_asset')
        if path in protected and not (successor and choices.get(path, {}).get('boundary') is False):
            evidence.append('protected_interface')
        if choices.get(path, {}).get('boundary') or choices.get(path, {}).get('structural_type'):
            evidence.append('declared_boundary')
        kind = str(Usd.ModelAPI(prim).GetKind() or '')
        if not successor and kind_trust['kind_trusted'] is True and kind in ('component', 'subcomponent', 'assembly'):
            evidence.append('trusted_model_kind')
        # An externally authored asset entry is useful composition evidence.
        # Internal CAD instancing remains inside its chosen equipment owner.
        for key in ('references', 'payload'):
            arcs = prim.GetMetadata(key)
            if not successor and arcs and any(arc.assetPath for arc in arcs.GetAddedOrExplicitItems()):
                evidence.append('external_asset_reference')
                break
        if (prim.GetAttribute('asset:interface') or
                any('Semantic' in api for api in prim.GetAppliedSchemas()) or
                prim.GetAssetInfo().get('identifier')):
            evidence.append('declared_interface')
        if choices.get(path, {}).get('boundary') is False:
            if (path in protected and not successor) or path in children['/']:
                raise ValueError('Cannot remove a protected interface or stage root boundary: ' + path)
            evidence = []
        if evidence:
            required[path] = evidence

    boundary_reviews, boundary_unresolved, review_groups_used = reasoning.collect_reviews(
        stage, prims, children, geometry_under, walker.cand, required, choices, group_choices, inventory['source_fingerprint'], review_facts)
    inventory['metrics']['boundary_review_groups'] = len(boundary_reviews)
    inventory['metrics']['boundary_review_paths'] = sum(len(r['paths']) for r in boundary_reviews)

    rederived = {}
    nominees, product_roots = _scope_reuse_nominees(
        prims, children, geometry_by_path, geometry_under, nominees,
        walker.cand, required, choices, inventory['metrics'])
    confirmation_cache = _confirm_reuse_content(
        prims, children, geometry_by_path, nominees, inventory, progress)
    def confirmation(path):
        if path in confirmation_cache:
            return confirmation_cache[path]
        # Identity is intentionally occurrence-specific until equality is proved.
        # Dependency hashes invalidate it when source bytes change; composed path
        # prevents overrides of one referenced asset from sharing a definition.
        return _digest(['unshared_source_identity', inventory['source_fingerprint'],
                        request.get('variants', {}), request.get('time_samples', [None]), path])
    repeated, confirmations = set(), []
    for paths in nominees.values():
        if len(paths) < 2:
            continue
        exact = []
        for path in sorted(paths):
            facts = confirmation(path)
            for group in exact:
                if facts == confirmation(group[0]):
                    group.append(path)
                    break
            else:
                exact.append([path])
        for group in exact:
            if len(group) > 1:
                repeated.update(group)
                confirmations.append({'paths': group, 'method': 'exact_scoped_properties_and_appearance',
                                      'fingerprint': _digest(confirmation(group[0]))})

    progress("boundary_selection", confirmed_groups=len(confirmations))
    boundaries = set(required)
    reasons = required
    # Sharing never creates identity. Every unselected geometry region remains
    # owned by its closest required boundary, including named CAD solids.

    parent_boundary = {}
    for path in boundaries:
        parent = path.rsplit('/', 1)[0]
        while parent and parent not in boundaries:
            parent = parent.rsplit('/', 1)[0]
        parent_boundary[path] = parent or None
    owned = defaultdict(list)
    owner_by_geometry = {}
    for path in geometry_by_path:
        owner = path
        while owner and owner not in boundaries:
            owner = owner.rsplit('/', 1)[0]
        if not owner:
            inventory['uninspected'].append({'path': path, 'reason': 'no_geometry_owner'})
            continue
        owned[owner].append(path)
        owner_by_geometry[path] = owner
    child_boundaries = defaultdict(list)
    for path, parent in parent_boundary.items():
        if parent:
            child_boundaries[parent].append(path)

    functional_by_boundary = defaultdict(list)
    # Motion and a selected variant describe composition, not geometric accuracy.
    # Physics requirements belong to the nearest retained geometry owner; a
    # sibling mechanism cannot impose a tolerance on a whole factory assembly.
    for signal_path, signal_prim in prims.items():
        if not (signal_path in walker.body_self or
                any('Physics' in api for api in signal_prim.GetAppliedSchemas())):
            continue
        owner=signal_path
        while owner and owner not in boundaries:
            owner=owner.rsplit('/',1)[0]
        if owner and owned[owner]:
            functional_by_boundary[owner].append({'path':signal_path,'reason':'physics'})

    progress("classification", boundaries=len(boundaries))
    internal_kinds_by_boundary = defaultdict(dict)
    if successor:
        for internal_path, internal_choice in choices.items():
            if internal_choice.get('boundary') is not False or not internal_choice.get('kind'):
                continue
            ancestor = internal_path
            while ancestor and ancestor not in boundaries:
                ancestor = ancestor.rsplit('/', 1)[0]
            if ancestor:
                internal_kinds_by_boundary[ancestor][internal_path] = internal_choice['kind']
    merge_scope = _module('assessment_merge_scope', Path(__file__).resolve().parents[2] / 'component_scope.py')
    progress("merge_protection", prims_total=len(prims))
    accepted_internal_roots = {p for p, choice in choices.items() if choice.get('boundary') is False
                               and (choice.get('decision_scope') == 'subtree' or not boundary_unresolved)}
    if not successor: accepted_internal_roots = set(prims) - set(boundaries) - protected
    # Resolve once per accepted prim, stopping at the nearest retained asset.
    # Ancestors outside that asset cannot authorize its internal edits.
    internal_by_boundary = defaultdict(set)
    nearest_boundary = {}
    for prim_path in prims:
        parent = prim_path.rsplit('/', 1)[0]
        nearest_boundary[prim_path] = prim_path if prim_path in boundaries else nearest_boundary.get(parent)
    # A retained subtree decision reviews its own interior, stopping at child assets.
    for prim_path, boundary in nearest_boundary.items():
        if prim_path != boundary and choices.get(boundary, {}).get('decision_scope') == 'subtree':
            internal_by_boundary[boundary].add(prim_path)
    for internal_path in accepted_internal_roots:
        boundary = nearest_boundary.get(internal_path)
        if not boundary:
            continue
        local = internal_by_boundary[boundary]
        current = internal_path
        while current != boundary and current not in local:
            local.add(current)
            current = current.rsplit('/', 1)[0]
    merge_facts = merge_scope.merge_protection_facts(
        stage, set().union(*internal_by_boundary.values()), boundary_paths=boundaries)
    progress("boundary_records", boundaries=len(boundaries))
    unresolved, records, unresolved_groups, used_group_choices = list(boundary_unresolved), {}, {}, set(review_groups_used)
    protected_by_boundary, nearest_protected = _protected_path_index(prims, boundaries, protected)
    additional_protected = protected - review_state.get('protected', set())
    for path in sorted(boundaries, key=lambda p: (-p.count('/'), p)):
        prim = prims[path]
        child_paths = sorted(child_boundaries[path])
        protected_descendants = protected_by_boundary[path]
        choice = choices.get(path, {})
        role_evidence = [confirmation(path), [p[len(path):] for p in owned[path]],
                         [p[len(path):] for p in child_paths],
                         [p[len(path):] for p in protected_descendants]]
        evidence_fingerprint = _digest(role_evidence)
        group_id = 'role-' + evidence_fingerprint[:24]
        if group_id in group_choices:
            group_choice = group_choices[group_id]
            if group_choice.get('evidence_fingerprint', evidence_fingerprint) != evidence_fingerprint:
                raise ValueError('Grouped decision evidence has changed: ' + group_id)
            used_group_choices.add(group_id)
            choice = {**group_choice, **choice}
            choices[path] = {key: value for key, value in dict(choice, path=path).items()
                             if key in ('path', 'structural_type', 'reason', 'boundary', 'packaging', 'selection', 'kind', 'standalone_kind', 'equipment_family', 'geometry_representation', 'semantic_category')}
        if choice.get('structural_type'):
            structural_type = choice['structural_type']
            reason = choice['reason']
        elif not child_paths:
            structural_type = 'atomic'
            reason = 'Leaf asset owns geometry and has no child geometry assets.'
        elif successor:
            structural_type = 'package' if owned[path] else 'aggregate'
            reason = ('Retained asset owns geometry and contains retained child assets.' if owned[path]
                      else 'Retained asset composes child assets and owns no geometry.')
        elif protected_descendants and (
                any(not rel.GetName().startswith('material:') for rel in prim.GetAuthoredRelationships())
                or any('ArticulationRoot' in schema for schema in prim.GetAppliedSchemas())
                or bool(prim.GetAttribute('asset:interface'))):
            structural_type = 'package'
            reason = 'Equipment contains child assets with protected motion or interface targets.'
        elif not owned[path]:
            structural_type = 'aggregate'
            reason = 'Layout contains child assets and owns no geometry.'
        else:
            structural_type = None
            reason = 'Child assets and directly owned geometry require an equipment or layout decision.'
            if group_id not in unresolved_groups:
                unresolved_groups[group_id] = {'id': group_id, 'group_id': group_id,
                    'type': 'structural_role', 'paths': [], 'choices': ['package', 'aggregate'],
                    'evidence_fingerprint': evidence_fingerprint,
                    'evidence': {'representative': path, 'children': child_paths,
                                 'owned_geometry': owned[path]},
                    'consequence': 'Aggregate geometry requires a meaningful child boundary.'}
                unresolved.append(unresolved_groups[group_id])
            unresolved_groups[group_id]['paths'].append(path)
        if (successor and path in additional_protected
                and 'boundary' not in choice and not choice.get('structural_type')):
            unresolved.append({'id': group_id, 'group_id': group_id, 'type': 'structural_role',
                'paths': [path], 'evidence_fingerprint': evidence_fingerprint,
                'choices': ['atomic', 'aggregate', 'package'],
                'evidence': {'representative': path, 'protected_interface': path,
                             'reason': 'Detailed geometry connection analysis found an additional protected interface.'},
                'consequence': 'Confirm this interface before authoring; earlier subtree decisions do not approve new connection evidence.'})
        layer, authored_path = _authored_identity(prim)
        # Context determines identity when occurrence geometry or appearance differs.
        effective = _digest(confirmation(path))
        required_uses = [e for e in request.get('required_occurrence_edits', [])
                         if Sdf.Path(e['path']).GetPrimPath().HasPrefix(Sdf.Path(path))
                         and not any(Sdf.Path(e['path']).GetPrimPath().HasPrefix(Sdf.Path(child)) for child in child_paths)]
        use_policy = sorted((e['path'][len(path):], e.get('edit_class', 'placement')) for e in required_uses)
        identity = [layer, authored_path, effective, use_policy] if use_policy else [layer, authored_path, effective]
        if choice.get('geometry_representation') == 'analytic_primitives':
            identity.append('analytic_primitives')
        identifier = 'definition-' + _digest(identity)[:24]
        own_geometry = sorted(owned[path])
        counts = dict(Counter(geometry_by_path[p]['schema'] for p in own_geometry))
        materials = {json.dumps(d, sort_keys=True): d for p in geometry_under[path]
                     for d in geometry_by_path[p]['material_dependencies']}
        mesh_paths = [p for p in own_geometry if geometry_by_path[p]['schema'] == 'Mesh']
        operations = [{'category': category, 'status': 'candidate' if mesh_paths else 'not_applicable',
                       'scope': 'owned_geometry', 'requires_scoped_validation': bool(mesh_paths),
                       'reason': 'Evaluate the owned meshes with scoped validation.' if mesh_paths else 'This definition owns no mesh geometry.',
                       'evidence': {'mesh_count': len(mesh_paths)}}
                      for category in ('cleanup', 'primitive_fit', 'reduction')]
        merge_groups, merge_exclusions = merge_scope.plan_merge_groups(
            stage, path, mesh_paths, protected, protection_facts=merge_facts)
        operations.append({'category': 'merge', 'status': 'candidate' if len(mesh_paths) > 1 else 'not_applicable',
                           'scope': 'merge_groups',
                           'requires_scoped_validation': bool(merge_groups),
                           'reason': ('Compatible internal meshes can merge within the retained boundary.' if merge_groups else
                                      'Required interfaces or authored state exclude meshes from merging.' if merge_exclusions else
                                      'No compatible internal mesh group.'),
                           'evidence': {'groups': len(merge_groups), 'retained_mesh_interfaces': merge_exclusions}})
        operations[-1]['status'] = 'candidate' if merge_groups else 'not_applicable'
        operations.append({'category': 'reuse', 'status': 'confirmed' if path in repeated else 'assessed',
                           'scope': 'definition', 'requires_scoped_validation': False,
                           'reason': ('Exact scoped content and appearance confirmation completed.' if path in confirmation_cache
                                      else 'Structural comparison found no repeated semantic scope; retained as a unique definition.'),
                           'evidence': {'confirmed_repeated': path in repeated}})
        internal_paths = internal_by_boundary[path]
        records[path] = {'id': identifier, 'source_layer': layer, 'source_prim_path': authored_path,
                         'boundary_path': path, 'structural_type': structural_type,
                         'rationale': {'reason': reason, 'evidence': reasons[path]},
                         'selection': reasoning.selection_for(path, reasons[path], choice),
                         'owned_geometry': own_geometry, 'protected_paths': protected_descendants,
                         'required_occurrence_edits': required_uses,
                         'internal_paths': sorted(internal_paths),
                         'children': child_paths, 'occurrences': [path],
                         'operation_scope': {'type': 'definition', 'geometry_paths': own_geometry, 'mesh_paths': mesh_paths,
                                             'exclude_paths': child_paths},
                         'geometry_counts': counts, 'material_dependencies': list(materials.values()),
                         'packaging': choice.get('packaging', {'layer': layer, 'prim_path': authored_path}),
                         'assessment': {'functional_precision': {'required': bool(functional_by_boundary[path]),
                                                               'evidence': functional_by_boundary[path]},
                                        'status': 'ready' if structural_type else 'decision_required',
                                        'operations': operations, 'candidates': []},
                         'merge_groups': merge_groups,
                         'content_fingerprint': effective,
                         'fingerprint_method': ('exact_scoped_properties_and_appearance' if path in confirmation_cache
                                                else 'unshared_source_identity')}
        if choice.get('standalone_kind'):
            records[path]['standalone_kind'] = choice['standalone_kind']
        if choice.get('semantic_category'):
            records[path]['semantic_category'] = choice['semantic_category']
        records[path]['geometry_representation'] = choice.get('geometry_representation', 'merged_meshes')
        if choice.get('product_identity'):
            records[path]['product_identity'] = copy.deepcopy(choice['product_identity'])
        if choice.get('equipment_family'):
            records[path]['equipment_family'] = choice['equipment_family']
        records[path]['internal_kind_overrides'] = internal_kinds_by_boundary[path]

    unused_group_choices = set(group_choices) - used_group_choices
    if unused_group_choices:
        raise ValueError('Grouped decision is stale or unknown: ' + ', '.join(sorted(unused_group_choices)))

    # Reuse must also work between identical locally authored copies whose spec
    # paths differ. A verified group chooses a deterministic authored identity.
    aliases = {}
    for group in confirmations:
        selected = [p for p in group['paths'] if p in records]
        if not selected:
            continue
        representative = selected[0]
        for path in selected:
            # A different explicit role or boundary policy is a separate job.
            if (records[path]['structural_type'] == records[representative]['structural_type'] and
                    records[path].get('standalone_kind') == records[representative].get('standalone_kind') and
                    records[path].get('semantic_category') == records[representative].get('semantic_category') and
                    records[path]['geometry_representation'] == records[representative]['geometry_representation'] and
                    sorted((e['path'][len(path):], e.get('edit_class', 'placement')) for e in records[path]['required_occurrence_edits']) ==
                    sorted((e['path'][len(representative):], e.get('edit_class', 'placement')) for e in records[representative]['required_occurrence_edits']) and
                    [(p[len(path):], k) for p, k in sorted(records[path]['internal_kind_overrides'].items())] ==
                    [(p[len(representative):], k) for p, k in sorted(records[representative]['internal_kind_overrides'].items())] and
                    [p[len(path):] for p in records[path]['owned_geometry']] ==
                    [p[len(representative):] for p in records[representative]['owned_geometry']] and
                    [(p[len(path):], records[p]['structural_type']) for p in records[path]['children']] ==
                    [(p[len(representative):], records[p]['structural_type']) for p in records[representative]['children']] and
                    [p[len(path):] for p in records[path]['protected_paths']] ==
                    [p[len(representative):] for p in records[representative]['protected_paths']]):
                aliases[path] = representative
                precision = records[representative]['assessment']['functional_precision']
                other = records[path]['assessment']['functional_precision']
                precision['required'] = precision['required'] or other['required']
                known = {(e['path'], e['reason']) for e in precision['evidence']}
                precision['evidence'].extend(e for e in other['evidence']
                                            if (e['path'], e['reason']) not in known)
    progress("property_addressing", prims_total=len(prims))
    correspondences={}
    addressed=set(p for record in records.values() for p in record.get('protected_paths',[]))
    addressed.update(edit['path'] for choice in choices.values() for edit in choice.get('required_occurrence_edits',[]))
    property_targets, address_property_reads = _addressed_property_targets(stage, prims.values())
    addressed.update(property_targets)
    eligible=[path for path in sorted(records) if records[path]['structural_type']=='atomic'
              and not records[path]['children'] and not records[path].get('internal_kind_overrides')]
    # Cheap name-independent shapes/extents nominate before reading mesh arrays.
    progress("structural_shapes", prims_total=len(prims), address_property_reads=address_property_reads)
    cheap_shapes={};cheap_extents={}
    for node in sorted(prims,key=lambda path:(-path.count('/'),path)):
        descendants=children[node]
        cheap_shapes[node]=_digest([str(prims[node].GetTypeName()),sorted(cheap_shapes[child] for child in descendants)])
        geometry=geometry_by_path.get(node)
        bounds=geometry.get('coarse_extent') if geometry is not None else ()
        cheap_extents[node]=(None if bounds is None or any(cheap_extents[child] is None for child in descendants)
                             else _digest([bounds,sorted(cheap_extents[child] for child in descendants)]))
    shape_buckets={}
    for path in eligible:shape_buckets.setdefault(cheap_shapes[path],[]).append(path)
    nominated=[]
    for members in shape_buckets.values():
        extent_buckets={}
        if any(cheap_extents[path] is None for path in members):extent_buckets[None]=members
        else:
            for path in members:extent_buckets.setdefault(cheap_extents[path],[]).append(path)
        for group in extent_buckets.values():
            if len({aliases.get(path,path) for path in group})>1:nominated.extend(group)
    eligible=sorted(path for path in nominated if not any(_within(path, root) or _within(root, path) for root in product_roots))
    # Resolve addressability once, rather than scanning all requirements per pair.
    eligible_set=set(eligible);blocked=set()
    for target in addressed:
        parent=target.rsplit('/',1)[0]
        while parent:
            if parent in eligible_set:blocked.add(parent)
            parent=parent.rsplit('/',1)[0]
    progress("atomic_qualification", candidates=len(eligible), blocked=len(blocked))
    correspondence_cache={};qualified={};placements={}
    for path in eligible:
        if path in blocked:continue
        if atomic_correspondence(stage,path,path,_cache=correspondence_cache,common_placement=True,policy=request):
            qualified[path]=correspondence_cache['signatures'][(path,True,True)]
    progress("atomic_pairing", qualified=len(qualified))
    old_groups={}
    for path in records:old_groups.setdefault(aliases.get(path,path),[]).append(path)
    buckets={}
    for members in old_groups.values():
        # Never split an existing alias group while redirecting its representative.
        if any(path not in qualified for path in members):continue
        keys={(qualified[path],records[path].get('standalone_kind'),records[path].get('semantic_category'),records[path]['geometry_representation']) for path in members}
        if len(keys)!=1:continue
        buckets.setdefault(next(iter(keys)),[]).extend(members)
    for members in buckets.values():
        if len({aliases.get(path,path) for path in members})<2:continue
        representative=min(members)
        mappings={member:atomic_correspondence(stage,member,representative,_cache=correspondence_cache,common_placement=True,policy=request)
                  for member in members if member!=representative}
        if any(mapping is None for mapping in mappings.values()):continue
        for member,mapping in sorted(mappings.items()):
            if atomic_correspondence(stage,member,representative) is None:
                helper=_module('atomic_placement',Path(__file__).with_name('atomic_placement.py'))
                placements[member]=helper.common_placement(stage,member,representative,mapping,request)
            aliases[member]=representative;correspondences[member]=mapping
            precision=records[representative]['assessment']['functional_precision']
            other=records[member]['assessment']['functional_precision']
            precision['required']=precision['required'] or other['required']
            known={(e['path'],e['reason']) for e in precision['evidence']}
            precision['evidence'].extend(e for e in other['evidence'] if (e['path'],e['reason']) not in known)
        records[representative]['fingerprint_method']='exact_atomic_subtree_correspondence'
    def canonical(path):
        seen=set()
        while aliases.get(path,path)!=path:
            if path in seen:raise ValueError('cyclic definition aliases')
            seen.add(path);path=aliases[path]
        return path
    definitions, by_id, occurrences = [], {}, []
    for path in sorted(records):
        record = records[path]
        representative = canonical(path)
        identifier = records[representative]['id']
        record['id'] = identifier
        record['children'] = sorted(set(records[canonical(p)]['id'] for p in record['children']))
        prim = prims[path]
        source_kind = str(Usd.ModelAPI(prim).GetKind() or '')
        kind = choices.get(path, {}).get('kind', '' if successor else source_kind)
        occurrence = {'path': path, 'definition_id': identifier, 'effective_kind': kind,
                      'source_kind': source_kind, 'kind_override': choices.get(path, {}).get('kind'),
                      'context_fingerprint': _digest([geometry_by_path[p]['context_fingerprint'] for p in geometry_under[path]]),
                      'parent_path': parent_boundary[path]}
        if path in correspondences:occurrence["source_to_definition"]=correspondences[path]
        if path in placements:occurrence["internal_placement"]=placements[path]
        occurrences.append(occurrence)
        if identifier not in by_id:
            # Ensure canonical geometry paths, even if iteration order changes.
            selected = records[representative]
            selected['occurrences'] = []
            by_id[identifier] = selected
            definitions.append(selected)
        by_id[identifier]['occurrences'].append(path)
    for row in inventory['geometry']:
        owner = owner_by_geometry.get(row['path'])
        row['definition_id'] = records[canonical(owner)]['id'] if owner else None
        row['owner_path'] = owner

    def candidate_owners(group):
        owners = set()
        for path in group['paths']:
            while path and path not in records:
                path = path.rsplit('/', 1)[0]
            if path:
                owners.add(records[path]['id'])
        return sorted(owners)

    progress("fragmentation", definitions=len(definitions))
    fragmentation = fragmenter.scan_stage(stage)
    fragmentation = fragmenter.suggest_merges(fragmentation, knobs={'top_n': len(fragmentation)})['suggestions']
    source = {'path': str(Path(source_path).resolve()), 'fingerprint': inventory['source_fingerprint'],
              'meters_per_unit': inventory['meters_per_unit'], 'up_axis': inventory['up_axis']}
    selected_variants = {}
    for context in inventory.get('variant_contexts', []):
        selected_variants.setdefault(context['path'], {})[context['set']] = context['selection']
    manifest = {'schema_version': 2, 'mode': 'restructure', 'input_stage': source['path'], 'request': copy.deepcopy(request),
                'source': source, 'context': {'load': request.get('load', 'all'),
                                            'variants': selected_variants if successor else request.get('variants', {}),
                                            'time_samples': animation['time_samples'] if animation is not None else request.get('time_samples', [None])},
                'definitions': definitions, 'occurrences': occurrences, 'inventory': inventory,
                'decisions_required': unresolved, 'boundary_reviews': boundary_reviews,
                'boundary_review_version': 1, 'boundary_decisions': list(choices.values()),
                'reuse_candidates': [{'representative': group['paths'][0],
                                      'member_roots': group['paths'], 'status': 'confirmed',
                                      'occurrence_count': len(group['paths']),
                                      'definition_ids': candidate_owners(group),
                                      'disposition': 'asset_reuse' if any(p in records for p in group['paths']) else 'retained_within_asset',
                                      'method': group['method'], 'fingerprint': group['fingerprint']}
                                     for group in confirmations],
                'fragmentation_candidates': fragmentation,
                'phase4_targets': [{'id': d['id'], 'definition_id': d['id'], 'path': source['path'],
                                    'target_class': 'prototype', 'mesh_count': d['geometry_counts'].get('Mesh', 0)}
                                   for d in definitions]}
    if successor:
        manifest['assessment_phase'] = 'detailed'
        manifest['required_occurrence_edits'] = copy.deepcopy(request.get('required_occurrence_edits', []))
    manifest['discovery'] = {'method': 'exhaustive_recursive_inventory',
                             'kind_trust': kind_trust,
                             'boundary_rederivation': rederived,
                             'source_scan': {'prim_count': len(prims),
                                             'geometry_count': len(geometry_by_path),
                                             'boundary_review_groups': len(boundary_reviews),
                                             'boundary_review_paths': sum(len(row['paths']) for row in boundary_reviews),
                                             'geometry_paths_fingerprint': _digest(sorted(geometry_by_path)),
                                             'confirmed_reuse_groups': len(confirmations),
                                             'instance_proxy_descent': True,
                                             'bottom_up_confirmation': True,
                                             'provenance': 'assessment_traversal'}}
    manifest['revision'] = _digest({key: value for key, value in manifest.items() if key != 'inventory'})
    manifest['status'] = 'incomplete' if unresolved or inventory['uninspected'] else 'assessed'
    inventory['metrics']['assessment_seconds'] = time.monotonic() - started
    checker_path = base.parent / 'optimization-report/scripts/structure_conformance.py'
    if checker_path.exists():
        checker = _module('workflow_structure_conformance', checker_path)
        manifest['conformance'] = checker.check_manifest(manifest)
        if not manifest['conformance']['passed']:
            manifest['status'] = 'incomplete'
    # Keep the persisted inventory for independent coverage reconciliation.
    # Array/property hashes live on unique definitions; occurrence context
    # hashes retain placement/appearance evidence without repeating matrices,
    # material dependency lists and authored paths for every geometry prim.
    compact_manifest(manifest)
    property_cache = _property_cache_context.get()
    inventory["metrics"]["property_cache"] = {key: property_cache[key] for key in ("reads", "hits", "geometry_reads", "peak_bytes", "evictions", "oversized")}
    progress("assessment_complete", definitions=len(definitions), elapsed_seconds=time.monotonic()-started)
    return manifest


def refresh_discovery_evidence(manifest):
    """Reconcile recursive discovery facts with the current definition worklist.

    This proves assessment coverage, not successful optimization. Final output
    still requires an independent inventory and verified definition results.
    There is no terminal candidate rescan in this algorithm.
    """
    inventory = manifest.get('inventory', {})
    discovery = manifest.setdefault('discovery', {'method': 'exhaustive_recursive_inventory'})
    paths = [row['path'] for row in inventory.get('geometry', [])]
    source_scan = discovery.get('source_scan', {})
    definitions = {d['id']: d for d in manifest.get('definitions', [])}
    occurrences = manifest.get('occurrences', [])
    owned = Counter()
    owner_ids = {}
    invalid_occurrences = 0
    for occurrence in occurrences:
        definition = definitions.get(occurrence.get('definition_id'))
        if definition is None:
            invalid_occurrences += 1
            continue
        root = definition['boundary_path']
        inverse={destination:source for source,destination in occurrence.get('source_to_definition',{}).items()}
        if 'source_owned_geometry' in occurrence:
            # The replacement product can have different internal geometry.
            # Coverage still accounts for every originally assessed source prim.
            for path in occurrence['source_owned_geometry']:
                if path != occurrence['path'] and not path.startswith(occurrence['path'] + '/'):
                    invalid_occurrences += 1
                    continue
                owned[path] += 1
                owner_ids[path] = definition['id']
            continue
        for geometry in definition.get('owned_geometry', []):
            if geometry != root and not geometry.startswith(root + '/'):
                invalid_occurrences += 1
                continue
            path = inverse.get(geometry,occurrence['path'] + geometry[len(root):])
            owned[path] += 1
            owner_ids[path] = definition['id']
    inventory_paths = set(paths)
    absent = inventory_paths - set(owned)
    extra = set(owned) - inventory_paths
    duplicate = {path for path, count in owned.items() if count != 1}
    mismatched = sum(row.get('definition_id') != owner_ids.get(row['path'])
                     for row in inventory.get('geometry', []))
    candidates = manifest.get('reuse_candidates', [])
    unaccounted = sum(not candidate.get('definition_ids') or
                      not set(candidate['definition_ids']).issubset(definitions)
                      for candidate in candidates)
    observed = (discovery.get('method') == 'exhaustive_recursive_inventory' and
                source_scan.get('prim_count') == inventory.get('metrics', {}).get('prim_count') and
                isinstance(source_scan.get('prim_count'), int) and
                source_scan['prim_count'] >= len(paths) and
                source_scan.get('instance_proxy_descent') is True and
                source_scan.get('bottom_up_confirmation') is True and
                source_scan.get('geometry_count') == len(paths) and
                source_scan.get('geometry_paths_fingerprint') == _digest(sorted(paths)) and
                source_scan.get('confirmed_reuse_groups') == len(candidates))
    unresolved = len(manifest.get('decisions_required', [])) + sum(
        d.get('structural_type') not in ('atomic', 'aggregate', 'package') or
        d.get('assessment', {}).get('status') == 'decision_required' for d in definitions.values())
    discovery.update({'inventory_geometry_count': len(paths),
                      'owned_geometry_count': len(owned),
                      'unassigned_geometry_count': len(absent),
                      'unexpected_geometry_count': len(extra),
                      'duplicate_geometry_count': len(duplicate) + len(paths) - len(inventory_paths),
                      'ownership_mismatch_count': mismatched,
                      'invalid_occurrence_count': invalid_occurrences,
                      'confirmed_reuse_groups': len(candidates),
                      'unaccounted_reuse_groups': unaccounted,
                      'uninspected_contexts': len(inventory.get('uninspected', [])),
                      'unresolved_decisions': unresolved,
                      'source_scan_evidence_matches': observed,
                      'worklist_complete': bool(observed and not (absent or extra or duplicate or
                           len(paths) != len(inventory_paths) or mismatched or invalid_occurrences or
                           unaccounted or inventory.get('uninspected') or unresolved) and
                           manifest.get('conformance', {}).get('passed') is True)})
    return discovery


def compact_manifest(manifest):
    """Remove generated duplicate facts in-place; retain authoritative records."""
    if manifest.get('schema_version') != 2:
        return manifest
    refresh_discovery_evidence(manifest)
    for definition in manifest['definitions']:
        for operation in definition.get('assessment', {}).get('operations', []):
            operation.pop('geometry_paths', None)
            operation.setdefault('scope', 'merge_groups' if operation.get('category') == 'merge'
                                 else 'definition' if operation.get('category') == 'reuse' else 'owned_geometry')
    occurrences_by_path = {o['path']: o['definition_id'] for o in manifest['occurrences']}
    for occurrence in manifest['occurrences']:
        for key in ('source_layer', 'source_prim_path', 'owned_geometry'):
            occurrence.pop(key, None)
    manifest['inventory']['geometry'] = [
        {key: row[key] for key in ('path', 'schema', 'definition_id') if key in row}
        for row in manifest['inventory']['geometry']]
    manifest['phase4_targets'] = [
        {key: value for key, value in target.items()
         if key in ('id', 'definition_id', 'path', 'target_class', 'mesh_count', 'role')}
        for target in manifest['phase4_targets']]
    for candidate in manifest.get('reuse_candidates', []):
        if 'paths' not in candidate:
            continue
        paths = candidate.pop('paths')
        candidate.update({'representative': paths[0], 'occurrence_count': len(paths),
                          'definition_ids': sorted({occurrences_by_path[p] for p in paths if p in occurrences_by_path})})
    fragmentation = manifest.get('fragmentation_candidates', [])
    if fragmentation and 'path' in fragmentation[0]:
        helper = _module('workflow_compact_fragmentation', Path(__file__).resolve().parents[1] /
                         'references/usd-mesh-fragmentation-candidates/scripts/mesh_fragmentation_finder.py')
        manifest['fragmentation_candidates'] = helper.suggest_merges(
            fragmentation, knobs={'top_n': len(fragmentation)})['suggestions']
    manifest['revision'] = _digest({key: value for key, value in manifest.items()
                                  if key not in ('inventory', 'conformance', 'revision', 'status')})
    return manifest


def check_discovery_evidence(manifest):
    """Check recursive assessment coverage without USD or source file access."""
    working = dict(manifest, discovery=copy.deepcopy(manifest.get('discovery', {})))
    evidence = refresh_discovery_evidence(working)
    errors = []
    if not evidence['source_scan_evidence_matches']:
        errors.append({'code': 'discovery_scan_evidence',
                       'message': 'Recorded recursive scan does not match the retained inventory and candidates.'})
    for key in ('unassigned_geometry_count', 'unexpected_geometry_count', 'duplicate_geometry_count',
                'ownership_mismatch_count', 'invalid_occurrence_count', 'unaccounted_reuse_groups',
                'uninspected_contexts', 'unresolved_decisions'):
        if evidence[key]:
            errors.append({'code': key, 'count': evidence[key]})
    if manifest.get('conformance', {}).get('passed') is not True:
        errors.append({'code': 'structure_conformance', 'message': 'Current structure conformance has not passed.'})
    return {'passed': evidence['worklist_complete'], 'errors': errors, 'evidence': evidence}


def _refresh_classification(manifest, choices, progress=None):
    """Update labels on resolved definitions without changing their proven geometry."""
    if manifest.get('status') != 'assessed' or not choices:
        return None
    labels = ('semantic_category', 'equipment_family')
    metadata = {*labels, 'reason', 'selection'}
    previous = {row['path']: row for row in manifest.get('boundary_decisions', [])}
    occurrences = {row['path']: row['definition_id'] for row in manifest['occurrences']}
    definitions = {row['id']: row for row in manifest['definitions']}
    affected = set()
    for path, choice in choices.items():
        if path not in previous or path not in occurrences:
            return None
        if any(key not in metadata and value != previous[path].get(key)
               for key, value in choice.items()):
            return None
        definition = definitions[occurrences[path]]
        if definition.get('input_layer'):
            return None
        affected.add(definition['id'])
    updates = {}
    for identifier in affected:
        definition = definitions[identifier]
        updates[identifier] = {}
        for field in labels:
            values = {choices.get(path, {}).get(field, previous.get(path, {}).get(field, definition.get(field)))
                      for path in definition['occurrences']}
            if len(values) != 1:
                return None
            value = values.pop()
            if value is not None:
                updates[identifier][field] = value
    result = copy.deepcopy(manifest)
    for definition in result['definitions']:
        if definition['id'] in updates:
            definition.update(updates[definition['id']])
            representative = definition['source_prim_path']
            if 'selection' in choices.get(representative, {}):
                definition['selection'] = copy.deepcopy(choices[representative]['selection'])
    for row in result['boundary_decisions']:
        row.update(choices.get(row['path'], {}))
    result['decision_refresh'] = {'geometry_rescanned': False, 'classification_only': True,
                                  'affected_definitions': sorted(affected)}
    compact_manifest(result)
    if progress:
        progress('classification_refresh', definitions=len(affected), geometry_rescanned=False)
    return result


def apply_decisions(manifest, decisions, progress=None):
    """Resolve recorded role choices without scanning unchanged geometry again.

    Structural type/rationale and packaging decisions reuse the frozen facts.
    A new identity boundary requires assessment because it changes ownership.
    """
    for dependency in manifest.get('inventory', {}).get('dependencies', []):
        path = dependency['path']
        if not Path(path).is_file() or _file_hash(path) != dependency['fingerprint']:
            raise ValueError('Assessment dependency changed; reassess before applying decisions: ' + path)
    choices = _decisions_by_path(decisions)
    if any('selection' in choice for choice in choices.values()):
        reasoning = _module('selection_refresh', Path(__file__).with_name('boundary_reasoning.py'))
        source_stage = _open_stage(manifest['source']['path'], manifest.get('context', {}))
        for choice in choices.values():
            if 'selection' in choice:
                reasoning.validate_selection(choice['selection'], source_stage)
    group_choices = _group_decisions(decisions)
    if not group_choices:
        refreshed = _refresh_classification(manifest, choices, progress)
        if refreshed is not None:
            return refreshed
    if manifest.get('assessment_phase') == 'boundary_review':
        prior = {choice['path']: choice for choice in manifest.get('boundary_decisions', [])}
        prior.update(choices)
        return assess(manifest['source']['path'], request=manifest.get('request', {}),
                      decisions={'boundaries': list(prior.values()), 'groups': group_choices}, cached_review=manifest, progress=progress)
    review_choices = {key: value for key, value in group_choices.items() if 'disposition' in value}
    occurrence_paths = {row['path'] for row in manifest['occurrences']}
    if any('boundary' in choice or path not in occurrence_paths for path, choice in choices.items()):
        prior = {choice['path']: choice for choice in manifest.get('boundary_decisions', [])}
        prior.update(choices)
        return assess(manifest['source']['path'], request=manifest.get('request', manifest.get('context', {})),
                      decisions={'boundaries': list(prior.values()), 'groups': group_choices}, progress=progress)
    if review_choices:
        recorded = {row['group_id']: row for row in manifest.get('boundary_reviews', [])}
        reasoning = _module('decision_boundary_reasoning', Path(__file__).with_name('boundary_reasoning.py'))
        stage = source_stage if any('selection' in choice for choice in choices.values()) else _open_stage(manifest['source']['path'], manifest.get('context', {}))
        for key, choice in review_choices.items():
            if key not in recorded or choice.get('disposition') not in ('boundary', 'internal'):
                raise ValueError('Unknown boundary review or invalid disposition: ' + key)
            if choice.get('evidence_fingerprint', recorded[key]['evidence_fingerprint']) != recorded[key]['evidence_fingerprint']:
                raise ValueError('Grouped decision evidence has changed: ' + key)
            reasoning.validate_selection(choice.get('selection'), stage)
        if any(choice['disposition'] == 'boundary' for choice in review_choices.values()) or any('boundary' in c for c in choices.values()):
            prior = {choice['path']: choice for choice in manifest.get('boundary_decisions', [])}
            prior.update(choices)
            return assess(manifest['source']['path'], request=manifest.get('request', manifest.get('context', {})),
                          decisions={'boundaries': list(prior.values()), 'groups': group_choices}, progress=progress)
        result = copy.deepcopy(manifest)
        for review in result.get('boundary_reviews', []):
            choice = review_choices.get(review['group_id'])
            if choice is None:
                continue
            review.update(status='reviewed', disposition='internal', selection=choice['selection'])
            review['dispositions'] = [{'path': path, 'disposition': 'internal', 'selection': choice['selection']}
                                      for path in review['paths']]
            result.setdefault('boundary_decisions', []).extend(
                {'path': path, 'boundary': False, 'selection': choice['selection']} for path in review['paths'])
        result['decisions_required'] = [row for row in result['decisions_required'] if row.get('group_id') not in review_choices]
        remaining = {key: value for key, value in group_choices.items() if key not in review_choices}
        if remaining or choices:
            return apply_decisions(result, {'boundaries': list(choices.values()), 'groups': remaining})
        checker = _module('review_conformance', Path(__file__).resolve().parents[2] / 'optimization-report/scripts/structure_conformance.py')
        result['conformance'] = checker.check_manifest(result)
        if 'structure_check' in result:
            result['structure_check'] = result['conformance']
        result['status'] = 'assessed' if result['conformance']['passed'] else 'incomplete'
        result['revision'] = _digest({key: value for key, value in result.items() if key not in ('inventory', 'revision')})
        result['decision_refresh'] = {'geometry_rescanned': False, 'reviewed_groups': sorted(review_choices)}
        compact_manifest(result)
        return result
    recorded_groups = {group['group_id']: group for group in manifest.get('decisions_required', [])
                       if group.get('group_id')}
    for group_id, choice in group_choices.items():
        if group_id not in recorded_groups:
            raise ValueError('Grouped decision is stale or unknown: ' + group_id)
        group = recorded_groups[group_id]
        if choice.get('evidence_fingerprint', group['evidence_fingerprint']) != group['evidence_fingerprint']:
            raise ValueError('Grouped decision evidence has changed: ' + group_id)
        if (choice.get('structural_type') not in ('atomic', 'aggregate', 'package')
                or not str(choice.get('reason', '')).strip()):
            raise ValueError('Grouped decision requires structural_type and reason')
        if set(choice) - {'group_id', 'structural_type', 'reason', 'evidence_fingerprint'}:
            raise ValueError('Unknown grouped-decision fields')
        for path in group['paths']:
            choices.setdefault(path, dict(choice, path=path))
    occurrences = {o['path']: o for o in manifest['occurrences']}
    for path, choice in choices.items():
        if path not in occurrences or 'boundary' in choice or any(field in choice for field in ('kind', 'standalone_kind', 'equipment_family', 'geometry_representation', 'semantic_category')):
            raise ValueError('Boundary edits require reassessment: ' + path)
    result = copy.deepcopy(manifest)
    by_id = {d['id']: d for d in result['definitions']}
    affected = set()
    for path, choice in choices.items():
        identifier = occurrences[path]['definition_id']
        definition = by_id[identifier]
        if choice.get('structural_type'):
            if any(other not in choices or choices[other].get('structural_type') != choice['structural_type']
                   for other in definition['occurrences']):
                raise ValueError('A shared definition needs consistent decisions for all occurrences: ' + path)
            definition['structural_type'] = choice['structural_type']
            definition['rationale'] = {'reason': choice['reason'],
                                       'evidence': definition['rationale']['evidence']}
            definition['assessment']['status'] = 'ready'
            affected.add(identifier)
        if 'selection' in choice:
            definition['selection'] = dict(choice['selection'])
        saved = {item['path']: item for item in result.get('boundary_decisions', [])}
        saved[path] = {key: value for key, value in choice.items() if key in ('path', 'structural_type', 'reason', 'boundary', 'packaging', 'selection')}
        result['boundary_decisions'] = list(saved.values())
        if 'packaging' in choice:
            definition['packaging'] = choice['packaging']
    for group in result.get('decisions_required', []):
        if group.get('type') == 'boundary_review':
            continue
        group['paths'] = [path for path in group['paths']
                          if occurrences[path]['definition_id'] not in affected]
    result['decisions_required'] = [group for group in result.get('decisions_required', []) if group['paths']]
    for target in result['phase4_targets']:
        definition = by_id[target['definition_id']]
        if 'structural_type' in target:
            target['structural_type'] = definition['structural_type']
        if 'rationale' in target:
            target['rationale'] = definition['rationale']
    checker_path = Path(__file__).resolve().parents[2] / 'optimization-report/scripts/structure_conformance.py'
    checker = _module('workflow_decision_conformance', checker_path)
    result['conformance'] = checker.check_manifest(result)
    if 'structure_check' in result:
        result['structure_check'] = result['conformance']
    result['status'] = 'assessed' if result['conformance']['passed'] else 'incomplete'
    result['revision'] = _digest({key: value for key, value in result.items()
                                 if key not in ('inventory', 'conformance', 'revision', 'status')})
    result['decision_refresh'] = {'geometry_rescanned': False, 'affected_definitions': sorted(affected)}
    compact_manifest(result)
    return result
