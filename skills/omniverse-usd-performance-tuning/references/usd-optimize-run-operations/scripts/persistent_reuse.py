# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Confirm persistent entries against current immutable definition inputs."""
import hashlib
import importlib.util
import json
from pathlib import Path



def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module




def _content_signature(stage, root):
    """Hash composed definition content, independent of its source file identity."""
    from pxr import Usd, UsdGeom, UsdShade
    assessment = _load('reuse_content', Path(__file__).resolve().parents[2] /
                       'usd-structure-assessment/scripts/structure_assessment.py')
    boundary = stage.GetPrimAtPath(root)
    if not boundary:
        raise ValueError('immutable definition root is missing')
    facts, cache = [], {}
    for prim in Usd.PrimRange(boundary, Usd.TraverseInstanceProxies()):
        # Material placement is packaging; effective networks are compared on
        # their consumers, including GeomSubsets with separate bindings.
        if prim.IsA(UsdShade.Material) or prim.IsA(UsdShade.Shader) or prim.IsA(UsdShade.NodeGraph):
            continue
        appearance, dependencies = assessment._effective_appearance(prim, cache)
        resources = []
        for dependency in dependencies:
            if dependency.get('kind') == 'texture':
                path = dependency.get('resolved_path')
                if not path or not Path(path).is_file():
                    raise ValueError('persistent matching requires resolved material resources')
                resources.append(assessment._file_hash(path))
        properties = assessment._property_facts(prim, ignore_placement=prim == boundary, root=root)
        # HOOPS retains the JT occurrence label for identification. Its export
        # suffix does not change asset content; keep the authored data in USD.
        properties = properties[:3] + [row for row in properties[3:]
            if row[0] != 'omni:hoops:metadata:JT_PROP_NAME']
        facts.append([str(prim.GetPath())[len(root):], properties,
                      appearance.get('network_fingerprint'), sorted(resources)])
    return assessment._digest({'content': facts, 'upAxis': str(UsdGeom.GetStageUpAxis(stage)),
                               'timeCodesPerSecond': stage.GetTimeCodesPerSecond(),
                               'framesPerSecond': stage.GetFramesPerSecond(),
                               'interpolation': str(stage.GetInterpolationType())})


def _normalized_source(stage, target, request):
    """Exact source identity in a single static enclosing-child frame.

    Constant colour quantization is opt-in workload authorization. All geometry,
    internal transforms and other appearance remain exact through the existing
    content signature; no surface comparison or hierarchy remapping occurs.
    """
    if request.get('allow_constant_colour_rounding') is not True:
        return {}
    from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf, Vt
    import math
    root_path = target.get('boundary_path', '/Asset')
    if target.get('child_boundaries') or any(p != root_path for p in target.get('protected_paths', [])):
        return {}
    if any(e.get('path') != root_path or e.get('edit_class', 'placement') != 'placement'
           for e in target.get('required_occurrence_edits', [])):
        return {}
    root = stage.GetPrimAtPath(root_path)
    if root.IsA(UsdGeom.Gprim): return {}
    looks = stage.GetPrimAtPath(root_path + '/Looks')
    if looks and any(p.IsA(UsdGeom.Gprim) for p in Usd.PrimRange(looks, Usd.TraverseInstanceProxies())): return {}
    children = [p for p in root.GetChildren() if UsdGeom.Xformable(p)]
    if len(children) != 1 or any(p != children[0] and p.GetName() != 'Looks' for p in root.GetChildren()):
        return {}
    child = children[0]
    if child.IsA(UsdGeom.Gprim) or UsdGeom.Xformable(child).GetResetXformStack():
        return {}
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if prim.GetVariantSets().GetNames() or any(a.GetNumTimeSamples() for a in prim.GetAttributes()):
            return {}
        if not (prim.IsA(UsdShade.Material) or prim.IsA(UsdShade.Shader) or prim.IsA(UsdShade.NodeGraph)):
            if any(a.GetConnections() for a in prim.GetAttributes()) or any(r.GetTargets() for r in prim.GetRelationships() if not r.GetName().startswith('material:binding')):
                return {}
    frame = UsdGeom.Xformable(child).GetLocalTransformation()
    if abs(frame.GetDeterminant()) < 1e-12:
        return {}
    # Session edits expand composition on a disposable stage; supplied inputs stay immutable.
    scratch = Usd.Stage.Open(stage.GetRootLayer(), Sdf.Layer.CreateAnonymous())
    scratch.SetEditTarget(scratch.GetSessionLayer())
    while True:
        instances = [p for p in scratch.Traverse() if p.IsInstance()]
        if not instances: break
        for prim in instances: prim.SetInstanceable(False)
    scratch = Usd.Stage.Open(scratch.Flatten())
    colors = []
    for prim in scratch.Traverse():
        if prim.IsA(UsdShade.Material): prim.ClearMetadata('displayName')
        for attr in list(prim.GetAuthoredAttributes()):
            if attr.GetName().startswith('omni:hoops:material:'):
                prim.RemoveProperty(attr.GetName()); continue
            array = attr.GetName() == 'primvars:displayColor'
            if array:
                value = attr.Get()
                if attr.GetMetadata('interpolation') not in (None, 'constant') or value is None or len(value) != 1: continue
                value = value[0]
            elif (prim.IsA(UsdShade.Shader) or prim.IsA(UsdShade.Material) or prim.IsA(UsdShade.NodeGraph)) and attr.GetName().startswith('inputs:') and str(attr.GetTypeName()) == 'color3f':
                value = attr.Get()
                if value is None or attr.GetConnections(): continue
            else: continue
            if any(not math.isfinite(float(x)) or x < 0 or x > 1 for x in value): continue
            canonical = Gf.Vec3f(*[min(255, math.floor(float(x) * 255 + 1e-5)) / 255 for x in value])
            if list(value) != list(canonical):
                colors.append({'path': str(attr.GetPath()), 'original': list(value), 'canonical': list(canonical)})
                attr.Set(Vt.Vec3fArray([canonical]) if array else canonical)
    local_child = scratch.GetPrimAtPath(child.GetPath())
    for attr in list(local_child.GetAttributes()):
        if attr.GetName().startswith('xformOp:'): local_child.RemoveProperty(attr.GetName())
    UsdGeom.Xformable(local_child).MakeMatrixXform().Set(Gf.Matrix4d(1))
    local_child.ClearMetadata('displayName')
    edit = Sdf.BatchNamespaceEdit(); edit.Add(child.GetPath(), root_path + '/Content')
    if not scratch.GetRootLayer().Apply(edit): return {}
    # Root displayName identifies this occurrence, retained in the delivered USD.
    scratch.GetPrimAtPath(root_path).ClearMetadata('displayName')
    # Exporters number material subsets differently. Canonicalize only their
    # labels; face membership and effective material binding remain exact.
    subsets = [p for p in scratch.Traverse() if p.IsA(UsdGeom.Subset)]
    edits = Sdf.BatchNamespaceEdit()
    destinations = set()
    for prim in subsets:
        subset = UsdGeom.Subset(prim)
        facts = [list(subset.GetIndicesAttr().Get() or []),
                 str(subset.GetFamilyNameAttr().Get()), str(subset.GetElementTypeAttr().Get())]
        suffix = hashlib.sha256(json.dumps(facts).encode()).hexdigest()
        destination = prim.GetPath().GetParentPath().AppendChild('Subset_' + suffix)
        if destination in destinations: return {}
        destinations.add(destination)
        edits.Add(prim.GetPath(), destination)
    if subsets and not scratch.GetRootLayer().Apply(edits): return {}
    key = _content_signature(scratch, root_path)
    unique_colours = {}
    for colour in colors:
        identity = json.dumps([colour['original'], colour['canonical']])
        value = unique_colours.setdefault(identity, {'original': colour['original'], 'canonical': colour['canonical'], 'property_count': 0})
        value['property_count'] += 1
    info = {'version': 2, 'child': str(child.GetPath())[len(root_path):],
            'matrix': [list(row) for row in frame], 'colour_changes': list(unique_colours.values()),
            'authorization': 'allow_constant_colour_rounding',
            'colour_rule': 'floor(channel * 255 + 1e-5) / 255 for constant RGB values in [0,1]'}
    return {'normalized_source_key': key, 'normalized_source_info': json.dumps(info, sort_keys=True)}


def matching_record(definition, request):
    """Identify exact pre-optimization content and its input placement.

    Discovery fingerprints may deliberately contain source identity to avoid
    expensive scans. Persistent matching instead hashes composed input content.
    """
    from pxr import Sdf, Usd, UsdGeom
    stage = Usd.Stage.Open(definition.get('input_layer', definition.get('source_layer')))
    if not stage or stage.GetCompositionErrors():
        raise ValueError('immutable definition input does not compose')
    target = dict(definition.get('worker_target', definition))
    if 'required_occurrence_edits' in target:
        target['required_occurrence_edits'] = sorted(
            ({'path': e['path'], 'edit_class': e.get('edit_class', 'placement')}
             for e in target['required_occurrence_edits']), key=lambda e: (e['path'], e['edit_class']))
    product = definition.get('product_identity') or target.get('product_identity')
    signature = '' if product else _content_signature(stage, target.get('boundary_path', '/Asset'))
    # USD dependency APIs do not inspect imports or texture paths inside MDL.
    # A module hash cannot prove that its opaque transitive resources are stable.
    mdl_resources=set()
    def resource_prims():
        if not product:
            yield from Usd.PrimRange.Stage(stage,Usd.TraverseInstanceProxies())
            return
        from pxr import UsdShade
        material_roots={}
        for prim in Usd.PrimRange(stage.GetPrimAtPath(target.get('boundary_path','/Asset')),Usd.TraverseInstanceProxies()):
            yield prim
            if prim.IsA(UsdGeom.Gprim) or prim.IsA(UsdGeom.Subset):
                material,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
                if material:material_roots[str(material.GetPath())]=material.GetPrim()
        for material in material_roots.values():
            yield from Usd.PrimRange(material,Usd.TraverseInstanceProxies())
    for prim in resource_prims():
        for attr in prim.GetAttributes():
            if attr.GetTypeName() not in (Sdf.ValueTypeNames.Asset,Sdf.ValueTypeNames.AssetArray):continue
            for time in [Usd.TimeCode.Default()]+[Usd.TimeCode(t) for t in attr.GetTimeSamples()]:
                value=attr.Get(time)
                for asset in [value] if isinstance(value,Sdf.AssetPath) else value or []:
                    if Path(asset.path).suffix.lower()=='.mdl':mdl_resources.add(asset.resolvedPath or asset.path)
    root = stage.GetPrimAtPath(target.get('boundary_path', '/Asset'))
    # Content identity intentionally omits occurrence placement. Record the
    # authored transform stack separately, including ancestor samples, so an
    # arbitrary immutable input cannot reuse an entry from a different frame.
    assessment = _load('reuse_root_facts', Path(__file__).resolve().parents[2] /
                       'usd-structure-assessment/scripts/structure_assessment.py')
    transforms = []
    current = root
    while current and not current.IsPseudoRoot():
        xform = UsdGeom.Xformable(current)
        if xform:
            transforms.append({
                'reset': xform.GetResetXformStack(),
                'order': list(xform.GetXformOpOrderAttr().Get() or []),
                'ops': [[op.GetOpName(), str(op.GetAttr().GetTypeName()),
                         assessment._value(op.GetAttr().Get()),
                         [(t, assessment._value(op.GetAttr().Get(t))) for t in op.GetAttr().GetTimeSamples()]]
                        for op in xform.GetOrderedXformOps()]})
            if xform.GetResetXformStack():
                break
        current = current.GetParent()
    root_transform_signature = hashlib.sha256(json.dumps(transforms, sort_keys=True).encode()).hexdigest()
    scale = target.get('maximum_occurrence_scale', 1.0)
    lookup = _load('reuse_identifiers', Path(__file__).with_name('persistent_library.py'))
    reasoning = _load('reuse_labels', Path(__file__).resolve().parents[2] / 'usd-structure-assessment/scripts/boundary_reasoning.py')
    identifiers = lookup.identifying_values(reasoning.identifying_facts(stage.GetPrimAtPath(target.get('boundary_path', '/Asset'))))
    return {'source_signature': signature, 'root_transform_signature': root_transform_signature,
            'identifiers': identifiers,
            'metersPerUnit': UsdGeom.GetStageMetersPerUnit(stage), 'upAxis': str(UsdGeom.GetStageUpAxis(stage)),
            'resource_closure_complete':not mdl_resources,'opaque_mdl_resources':sorted(mdl_resources),
            'tolerance_mm': _load('reuse_local_budget', Path(__file__).resolve().parents[2]/'geometry_tolerance.py').local_tolerance_mm(request,scale),
            'source_prim_path': target.get('boundary_path', '/Asset'),
            **({'product_key': product['key'], 'product_identity_json': json.dumps(product,sort_keys=True),
                'structural_type':target.get('structural_type','atomic')} if product else _normalized_source(stage, target, request))}


def confirm_reuse(definition, request, index_paths, lookup_context=None, *, product_only=False):
    """Confirm accepted product identity, then existing prepared content identity.

    Early source lookup requests only product confirmation. Prepared inputs retain
    exact/normalized matching when an accepted product key has no confirmed entry.
    """
    from pxr import Usd
    matching = matching_record(definition, request)
    if not matching['resource_closure_complete']:
        return {'entry':None,'matching':matching,'findings':[{'reason':'Persistent reuse cannot confirm opaque MDL dependency closure; module-internal imports/textures are uninspected.','modules':matching['opaque_mdl_resources']}]}
    lookup = _load('reuse_lookup', Path(__file__).with_name('persistent_library.py'))
    findings = []
    selected = json.loads(matching['product_identity_json']).get('library_entry') if matching.get('product_key') else None
    def confirm():
        nominated = lookup.nominate(index_paths, matching, context=lookup_context)
        findings.extend(nominated['findings'])
        for candidate in nominated['candidates']:
            if selected and (candidate['id']!=selected['id'] or Path(candidate['index']).resolve()!=Path(selected['index']).resolve()):
                continue
            result = _confirm_candidate(definition, matching, candidate, lookup, findings)
            if result:
                return result
        return None
    result = confirm()
    if result:
        return result
    if matching.get('product_key') and not product_only:
        stage = Usd.Stage.Open(definition.get('input_layer', definition.get('source_layer')))
        target = definition.get('worker_target', definition)
        matching['source_signature'] = _content_signature(stage, target.get('boundary_path', '/Asset'))
        matching.update(_normalized_source(stage, target, request))
        result = confirm()
        if result:
            return result
    return {'entry': None, 'matching': matching, 'findings': findings}


def _confirm_candidate(definition, matching, candidate, lookup, findings):
    """Build the same cache receipt for nominated and parent-referenced entries."""
    from pxr import Usd
    if not matching.get('resource_closure_complete'):
        findings.append({'entry': candidate['id'], 'reason': 'current resource closure incomplete'})
        return None
    target = dict(definition.get('worker_target', definition))
    source_root = target.get('boundary_path', '/Asset')
    previous = candidate.get('matching', {})
    compatibility = previous if previous.get('record_version') == 4 else lookup.compact_matching(previous)
    product = lookup.source_identity_kind(matching, dict(compatibility, metersPerUnit=previous.get('metersPerUnit'), upAxis=previous.get('upAxis'))) == 'product'
    normalized = bool(matching.get('normalized_source_key') and compatibility.get('normalized_source_key') == matching['normalized_source_key']
                      and compatibility.get('root_transform_signature') == matching['root_transform_signature'])
    if (compatibility.get('exact_key') != lookup.exact_key(matching) and not normalized and not product) or not compatibility.get('completed'):
        findings.append({'entry': candidate['id'], 'reason': 'completed source-identity publication unavailable'})
        return None
    if not compatibility.get('resource_closure_complete'):
        findings.append({'entry': candidate['id'], 'reason': 'persistent resource closure incomplete'})
        return None
    stage = Usd.Stage.Open(candidate['layer'])
    root = candidate['prim_path']
    required = set(target.get('protected_paths', [])) | {e['path'] for e in target.get('required_occurrence_edits', [])}
    paths = [root + p[len(source_root):] for p in required]
    missing = [p for p in paths if not stage or not stage.GetObjectAtPath(p)]
    if not stage or stage.GetCompositionErrors() or missing:
        findings.append({'entry': candidate['id'], 'reason': 'cached composition or required relative paths unavailable', 'paths': missing})
        return None
    if product and compatibility.get('structural_type', target.get('structural_type')) != target.get('structural_type'):
        findings.append({'entry':candidate['id'],'reason':'product structure differs'});return None
    if product:
        animation=_load('product_animation',Path(__file__).resolve().parents[3]/'scripts/animation_context.py')
        incoming_stage=Usd.Stage.Open(definition.get('input_layer',definition.get('source_layer')))
        motion=animation.verify(incoming_stage,stage,paths=[source_root],path_map={source_root:root})
        if not motion['passed']:
            findings.append({'entry':candidate['id'],'reason':'product representation does not preserve authored animation','animation':motion})
            return None
    children = [root + p[len(source_root):] for p in target.get('child_boundaries', [])]
    geometry = lookup.output_geometry(stage, root, children)
    files = dict(candidate.get('dependency_fingerprints', {}))
    files[candidate['layer']] = candidate['fingerprint']
    verification = {'passed': True, 'independent_geometry_proof': False, 'checks': [
        {'name': 'completed source-identity publication', 'passed': True},
        {'name': 'cached composition and required paths', 'passed': True}],
        'evidence_fingerprint': {'files': files},
        'reuse_basis': {'exact_key': lookup.exact_key(matching), 'native_operations_reexecuted': False}}
    proof = hashlib.sha256(json.dumps(verification, sort_keys=True).encode()).hexdigest()
    entry = dict(candidate, confirmed=True, confirmation_fingerprint=proof)
    entry.pop('status', None)
    if normalized and compatibility.get('exact_key') != lookup.exact_key(matching):
        from pxr import Gf
        incoming = json.loads(matching['normalized_source_info'])
        published = json.loads(compatibility['normalized_source_info'])
        matrix = lambda value: Gf.Matrix4d(*[x for row in value for x in row])
        delta = matrix(published['matrix']).GetInverse() * matrix(incoming['matrix'])
        entry['normalized_placement'] = {'matrix': [list(row) for row in delta], 'source_child': incoming['child'], 'published_child': published['child']}
        verification['reuse_basis'] = {'normalized_source_key': matching['normalized_source_key'], 'native_operations_reexecuted': False,
            'authorization': incoming['authorization'], 'incoming_colour_changes': incoming['colour_changes'],
            'published_colour_changes': published['colour_changes'], 'colour_rule': incoming['colour_rule']}
        entry['confirmation_fingerprint'] = hashlib.sha256(json.dumps(verification, sort_keys=True).encode()).hexdigest()
    result = {'definition_id': definition['id'], 'disposition': 'exact_cache_hit',
              'verification': verification, 'operations': [], 'persistent_reuse': True,
              'output_geometry_paths': geometry, 'cached_root': root,
              'assessment': {'complete': True, 'operations': [], 'reuse': 'exact_cache_hit'}}
    if product:
        verification['reuse_basis']={'product_key':matching['product_key'], 'product_identity':json.loads(matching['product_identity_json']),
            'representation_substitution':True,'native_operations_reexecuted':False}
        verification['checks'][0]['name']='accepted product identity and completed publication'
        entry['confirmation_fingerprint']=hashlib.sha256(json.dumps(verification,sort_keys=True).encode()).hexdigest()
        result['disposition']='product_identity_hit';result['assessment']['reuse']='product_identity_hit'
    if entry.get('normalized_placement'):
        result['disposition'] = 'normalized_cache_hit'
        result['assessment']['reuse'] = 'normalized_cache_hit'
    return {'entry': entry, 'result': result, 'matching': matching, 'findings': findings}


def reconcile_parent_reuse(definitions, occurrences, prepared_reuse, lookup_context):
    """Select child entries actually referenced by reused parents, atomically.

    Constraints follow retained occurrence paths. A conflict declines the parent
    cache proposals that require it, leaving independent child lookup available.
    """
    import copy
    from pxr import Usd
    lookup = _load('parent_reuse_lookup', Path(__file__).with_name('persistent_library.py'))
    definitions = {d['id']: d for d in definitions}
    by_path = {o['path']: o for o in occurrences if o['definition_id'] in definitions}
    edges = {identifier: set() for identifier in definitions}
    incoming = {identifier: set() for identifier in definitions}
    for path, occurrence in by_path.items():
        ancestor = path.rsplit('/', 1)[0]
        while ancestor and ancestor not in by_path:
            ancestor = ancestor.rsplit('/', 1)[0]
        if ancestor:
            parent = by_path[ancestor]['definition_id']; child = occurrence['definition_id']
            edges[parent].add((child, path[len(ancestor):]))
            incoming[child].add(parent)
    pending = set(definitions); order = []
    while pending:
        ready = sorted(identifier for identifier in pending if not (incoming[identifier] & pending))
        if not ready:
            raise ValueError('retained definition hierarchy is cyclic')
        order.extend(ready); pending.difference_update(ready)

    def entrypoint(confirmation):
        entry = confirmation['entry']
        return str(Path(entry['layer']).resolve()), entry['prim_path']

    def child_confirmation(parent, relative, child):
        stage = Usd.Stage.Open(parent['entry']['layer'])
        prim = stage.GetPrimAtPath(parent['entry']['prim_path'] + relative) if stage else None
        if not prim or stage.GetCompositionErrors():
            raise ValueError('cached child relative path does not compose: ' + relative)
        findings = []
        for spec in prim.GetPrimStack():
            if not spec.layer.realPath:
                continue
            key = (str(Path(spec.layer.realPath).resolve()), str(spec.path))
            for index, identifier, entry in lookup_context.by_entrypoint.get(key, []):
                try:
                    matching = prepared_reuse[child]['matching']
                    candidate = lookup.validate_entry(lookup_context, index, identifier, entry, matching)
                    confirmed = _confirm_candidate(definitions[child], matching, candidate, lookup, findings)
                    if confirmed:
                        return confirmed
                except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
                    findings.append({'reason': str(error)})
        reason = findings[-1]['reason'] if findings else 'no indexed child reference'
        raise ValueError('cached child ' + relative + ': ' + reason)

    declined = {}
    while True:
        selected = copy.deepcopy(prepared_reuse)
        constraints = {identifier: [] for identifier in definitions}
        failure = None
        for identifier in order:
            demands = constraints[identifier]
            if demands and (identifier in declined or len({entrypoint(row) for _, row in demands}) > 1):
                failure = ({parent for parent, _ in demands}, 'cached parents require conflicting child entrypoints: ' + identifier)
                break
            if demands:
                selected[identifier] = demands[0][1]
            elif identifier in declined:
                selected[identifier] = {'entry': None, 'matching': prepared_reuse[identifier]['matching'],
                                        'findings': list(prepared_reuse[identifier].get('findings', []))}
            if not selected[identifier].get('entry'):
                continue
            proposed = []
            try:
                for child, relative in sorted(edges[identifier]):
                    proposed.append((child, child_confirmation(selected[identifier], relative, child)))
            except ValueError as error:
                failure = ({identifier}, str(error))
                break
            for child, confirmation in proposed:
                constraints[child].append((identifier, confirmation))
        if failure is None:
            for identifier, reason in declined.items():
                selected[identifier]['findings'].append({'reason': reason})
            return selected
        rejected, reason = failure
        # Declining a nested selection also declines any enclosing cached parent
        # that requires that selection. Recompute from untouched independent hits.
        pending = list(rejected)
        while pending:
            child = pending.pop()
            for parent, _ in constraints[child]:
                if parent not in rejected:
                    rejected.add(parent); pending.append(parent)
        if not rejected - declined.keys():
            raise ValueError('parent reuse constraints cannot be resolved')
        declined.update({identifier: reason for identifier in rejected})


def prepare_definitions(definitions, request, index_paths, directory, max_workers=4, progress=None):
    """Prepare exact matching independently through the existing worker scheduler.

    Inputs and inherited indexes are read-only. Each process writes a bounded batch;
    failures retain logs/status and prevent dependent workflow execution.
    """
    import sys
    import threading
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
        raise ValueError('max_workers must be a positive integer')
    ids = [definition['id'] for definition in definitions]
    if len(ids) != len(set(ids)):
        raise ValueError('persistent preparation definition ids must be unique')
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    shared = directory / 'request.json'
    shared.write_text(json.dumps({'request': request,
                                  'index_paths': [str(Path(p).resolve()) for p in index_paths]}))
    targets, outputs = [], {}
    # Amortize runtime imports and status snapshots while retaining concurrency.
    batch_size = min(64, max(1, (len(definitions) + max_workers - 1) // max_workers))
    for index, offset in enumerate(range(0, len(definitions), batch_size)):
        members = definitions[offset:offset + batch_size]
        name = str(index).zfill(5)
        task = directory / (name + '.task.json')
        output = directory / (name + '.result.json')
        task.write_text(json.dumps({'definitions': members, 'shared_request': str(shared)}))
        identifier = 'preparation-' + name
        outputs[identifier] = output
        targets.append({'id': identifier, 'path': str(output),
                        'command': [sys.executable, str(Path(__file__).resolve()), str(task), str(output)],
                        'log_path': str(directory / (name + '.log')),
                        'output_files': [str(output)], 'depends_on': []})
    scheduler = _load('persistent_preparation_scheduler', Path(__file__).with_name('run_batch.py'))
    callback_lock = threading.Lock()
    completed = 0
    class PreparationScheduler(scheduler.BatchScheduler):
        def _run_one(self, target):
            try:
                super()._run_one(target)
            finally:
                if progress:
                    with callback_lock:
                        nonlocal completed
                        path = outputs[target.id]
                        evidence = json.loads(path.read_text()) if path.is_file() else {}
                        completed += len(evidence.get('results', {}))
                        progress('persistent_matching', completed_count=completed,
                                 definition_count=len(definitions), state=target.state)
    plan = {'run_id': 'persistent-matching', 'targets': targets}
    (directory / 'plan.json').write_text(json.dumps(plan, indent=2))
    if progress:
        progress('persistent_matching', completed_count=0, definition_count=len(definitions))
    status = PreparationScheduler(plan, directory / 'status.json', max_workers=max_workers).run()
    failed = [row['id'] for row in status['targets'] if row['state'] != scheduler.DONE]
    if failed:
        raise ValueError('persistent preparation failed; see ' + str(directory / 'status.json') + ': ' + ', '.join(failed))
    results = {}
    for path in outputs.values():
        results.update(json.loads(path.read_text())['results'])
    if set(results) != set(ids):
        raise ValueError('persistent preparation result coverage differs from requested definitions')
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task')
    parser.add_argument('output')
    args = parser.parse_args()
    task = json.loads(Path(args.task).read_text())
    shared = json.loads(Path(task['shared_request']).read_text())
    task.update(shared)
    lookup = _load('reuse_batch_lookup', Path(__file__).with_name('persistent_library.py'))
    lookup_context = lookup.LookupContext(task['index_paths'])
    def prepare(definition):
        return confirm_reuse(definition, task['request'], task['index_paths'], lookup_context)
    if 'definitions' not in task:
        # Preserve the original positional CLI and single-definition task shape.
        Path(args.output).write_text(json.dumps(prepare(task['definition']), indent=2))
        return 0
    import traceback
    results, errors = {}, {}
    for definition in task['definitions']:
        try:
            results[definition['id']] = prepare(definition)
        except Exception as exc:
            errors[definition['id']] = {'type': type(exc).__name__, 'message': str(exc)}
            traceback.print_exc()
    Path(args.output).write_text(json.dumps({'results': results, 'errors': errors}, indent=2))
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
