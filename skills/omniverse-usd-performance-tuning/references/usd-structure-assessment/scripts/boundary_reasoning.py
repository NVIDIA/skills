# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Selection evidence and grouped interior review; no geometry-value reads."""
import hashlib
import json
from pxr import Sdf, Usd, UsdGeom
import copy


def valid_required_use(stage, requirement, *, output=False):
    """Check the USD capability requested by a decision, also at delivery."""
    path = Sdf.Path(requirement['path'])
    obj = stage.GetObjectAtPath(path)
    prim = stage.GetPrimAtPath(path.GetPrimPath())
    if not obj or not prim:
        return False
    kind = requirement.get('edit_class', 'placement')
    if kind == 'identification':
        return True
    if output and prim.IsInstanceProxy():
        return False
    if kind == 'placement':
        return bool(UsdGeom.Xformable(prim)) and (path.IsPrimPath() or
            isinstance(obj, Usd.Attribute) and (obj.GetName().startswith('xformOp:') or obj.GetName() == 'xformOpOrder'))
    if kind == 'visibility':
        return bool(UsdGeom.Imageable(prim)) and (path.IsPrimPath() or
            isinstance(obj, Usd.Attribute) and obj.GetName() == 'visibility')
    return False


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def decision_source_context(stage, dependency_fingerprints, selected_variants=None):
    """Describe the selected composition; callers supply cached dependency hashes.

    This reads no geometry arrays and never opens alternative variant branches.
    Dependency fingerprints must include relevant session opinions and resolver
    dependencies, not just the root layer. Their computation belongs to inspection.
    """
    if not isinstance(dependency_fingerprints, dict) or not dependency_fingerprints:
        raise ValueError('source context requires dependency_fingerprints')
    if any(not isinstance(k, str) or not k or not isinstance(v, str) or not v
           for k, v in dependency_fingerprints.items()):
        raise ValueError('dependency_fingerprints requires layer identifiers and fingerprints')
    if selected_variants is None:
        selected_variants = {}
        for prim in stage.TraverseAll():
            variants = prim.GetVariantSets()
            selections = {name: variants.GetVariantSelection(name) for name in variants.GetNames()}
            if selections:
                selected_variants[str(prim.GetPath())] = selections
    return {'stage_identifier': stage.GetRootLayer().identifier,
            'selected_variants': copy.deepcopy(selected_variants),
            'load_policy': [[str(path), str(rule)] for path, rule in stage.GetLoadRules().GetRules()],
            'metersPerUnit': UsdGeom.GetStageMetersPerUnit(stage),
            'upAxis': str(UsdGeom.GetStageUpAxis(stage)),
            'startTimeCode': stage.GetStartTimeCode(), 'endTimeCode': stage.GetEndTimeCode(),
            'timeCodesPerSecond': stage.GetTimeCodesPerSecond(),
            'framesPerSecond': stage.GetFramesPerSecond(),
            'dependency_fingerprints': copy.deepcopy(dependency_fingerprints)}


def _decision_path(value, stage, label, prim_only=False):
    if not isinstance(value, str) or not Sdf.Path.IsValidPathString(value):
        raise ValueError(label + ' requires a valid Sdf.Path')
    path = Sdf.Path(value)
    if not path.IsAbsolutePath() or not (path.IsPrimPath() if prim_only else
                                        path.IsPrimPath() or path.IsPropertyPath()):
        raise ValueError(label + ' requires an absolute prim' + ('' if prim_only else '/property') + ' path')
    if not stage.GetObjectAtPath(path):
        raise ValueError(label + ' does not exist in composed stage: ' + value)
    return path


def regroup_owner_path(value, stage, create_owner=False):
    """Validate an existing owner or an explicitly requested single new Xform."""
    if type(create_owner) is not bool:
        raise ValueError('create_owner must be a boolean')
    if not create_owner:
        return _decision_path(value, stage, 'regroup owner', prim_only=True)
    if not isinstance(value, str) or not Sdf.Path.IsValidPathString(value):
        raise ValueError('new regroup owner requires a valid Sdf.Path')
    path = Sdf.Path(value)
    if not path.IsAbsolutePath() or not path.IsPrimPath() or path.ContainsPrimVariantSelection():
        raise ValueError('new regroup owner requires an absolute prim path')
    if stage.GetObjectAtPath(path):
        raise ValueError('create_owner cannot overwrite an existing prim')
    parent = stage.GetPrimAtPath(path.GetParentPath())
    if not parent or parent.IsPseudoRoot() or not parent.IsDefined() or not parent.IsActive():
        raise ValueError('new regroup owner requires one existing defined parent')
    if parent.IsInstance() or parent.IsInstanceProxy() or parent.IsInPrototype():
        raise ValueError('new regroup owner parent must be editable outside an instance')
    return path


def validate_successor_decisions(request, stage, candidates, source_context, *, dependency_validated_variants=None):
    """Validate compact judgments against trusted, current tool evidence.

    ``candidates`` maps candidate_id to prim_paths and evidence_fingerprint. It
    remains tool-owned; geometry membership and jobs are not agent input. This
    separate entrypoint leaves the existing assessment decision API compatible.
    Return normalized copies. Partial review is allowed; unmentioned candidates
    remain pending in the caller. Validation never certifies semantic judgment.
    """
    if not isinstance(request, dict) or set(request) != {'schema_version', 'source_context', 'decisions'}:
        raise ValueError('request requires schema_version, source_context and decisions only')
    if type(request['schema_version']) is not int or request['schema_version'] != 1:
        raise ValueError('unsupported schema_version')
    if not isinstance(source_context, dict) or request['source_context'] != source_context:
        raise ValueError('source context has changed; inspect selected variants and dependencies again')
    # Recheck inexpensive live composition metadata, including variant selections.
    selections = None
    if dependency_validated_variants is not None:
        selections = {}
        for path in dependency_validated_variants:
            prim = stage.GetPrimAtPath(path)
            if not prim:
                raise ValueError('Cached variant owner is missing: ' + path)
            sets = prim.GetVariantSets()
            selections[path] = {name: sets.GetVariantSelection(name) for name in sets.GetNames()}
        # Only file-backed callers which revalidated every dependency may use
        # this index; new variant owners require changed dependency contents.
        if stage.GetSessionLayer() and not stage.GetSessionLayer().empty:
            raise ValueError('Cached variant context cannot validate authored session opinions')
    if decision_source_context(stage, source_context.get('dependency_fingerprints'), selections) != source_context:
        raise ValueError('source context differs from current composed stage')
    if not isinstance(request['decisions'], list):
        raise ValueError('decisions must be an array')
    if not isinstance(candidates, dict):
        raise ValueError('candidates must be a tool-owned mapping')
    required = {'candidate_id', 'evidence_fingerprint', 'disposition', 'reason',
                'independent_use', 'basis', 'evidence_paths', 'semantic_category',
                'required_occurrence_edits'}
    optional = {'product_identity', 'prim_path', 'source_prim_paths', 'reconstruction', 'structural_type',
                'kind', 'structural_reason', 'kind_reason', 'decision_scope', 'equipment_family', 'geometry_representation'}
    result, seen = [], {}
    for decision in request['decisions']:
        if not isinstance(decision, dict) or required - set(decision) or set(decision) - required - optional:
            raise ValueError('decision has missing required or unknown fields')
        identifier = decision['candidate_id']
        if not isinstance(identifier, str) or identifier not in candidates:
            raise ValueError('candidate_id must name one unreviewed tool candidate')
        candidate = candidates[identifier]
        if not candidate.get('evidence_fingerprint') or decision['evidence_fingerprint'] != candidate['evidence_fingerprint']:
            raise ValueError('candidate evidence fingerprint has changed: ' + identifier)
        if ('prim_path' in decision) == ('source_prim_paths' in decision):
            raise ValueError('decision requires prim_path or source_prim_paths, exclusively')
        paths = [decision['prim_path']] if 'prim_path' in decision else decision['source_prim_paths']
        if not isinstance(paths, list) or not paths or any(not isinstance(p, str) for p in paths) or len(set(paths)) != len(paths):
            raise ValueError('source prim paths must be a nonempty distinct array')
        for path in paths:
            _decision_path(path, stage, 'source', prim_only=True)
            if any(Sdf.Path(path).HasPrefix(Sdf.Path(previous)) or Sdf.Path(previous).HasPrefix(Sdf.Path(path))
                   for previous in seen.get(identifier, [])):
                raise ValueError('candidate_id has overlapping source decisions')
        seen.setdefault(identifier, []).extend(paths)
        allowed = set(candidate.get('prim_paths', []))
        if decision['disposition'] == 'retain':
            allowed.update(row['prim_path'] for row in candidate.get('evidence', {}).get('enclosing_boundaries', []))
        if not set(paths).issubset(allowed):
            raise ValueError('source prim paths are outside candidate scope')
        if decision['disposition'] not in ('retain', 'internal', 'reconstruct'):
            raise ValueError('disposition must be retain, internal or reconstruct')
        if 'decision_scope' in decision and (decision['disposition'] not in ('retain', 'internal') or decision['decision_scope'] not in ('prim', 'subtree')):
            raise ValueError('decision_scope requires retain or internal disposition and prim or subtree')
        selection = {key: decision[key] for key in ('reason', 'independent_use', 'basis')}
        selection['evidence'] = decision['evidence_paths']
        errors = selection_errors(selection)
        if errors:
            raise ValueError('; '.join(errors))
        for path in decision['evidence_paths']:
            _decision_path(path, stage, 'evidence')
        category = decision['semantic_category']
        if not isinstance(category, str) or not category.strip():
            raise ValueError('semantic_category requires a category or unknown')
        if 'geometry_representation' in decision and (decision['disposition'] != 'retain' or
                decision['geometry_representation'] not in ('merged_meshes', 'analytic_primitives')):
            raise ValueError('geometry_representation requires merged_meshes or analytic_primitives on a retained asset')
        if 'equipment_family' in decision and (decision['disposition'] != 'retain' or
                not isinstance(decision['equipment_family'], str) or not decision['equipment_family'].strip()):
            raise ValueError('equipment_family requires a nonempty label on a retained asset')
        if 'product_identity' in decision:
            if decision['disposition'] != 'retain': raise ValueError('product_identity requires retain')
            import importlib.util
            from pathlib import Path
            location=Path(__file__).resolve().parents[3]/'scripts/product_clues.py'
            spec=importlib.util.spec_from_file_location('accepted_product_identity',location)
            clues=importlib.util.module_from_spec(spec);spec.loader.exec_module(clues)
            clues.validate_product_identity(decision['product_identity'],stage,paths)
        edits = decision['required_occurrence_edits']
        if not isinstance(edits, list):
            raise ValueError('required_occurrence_edits must be an array')
        for edit in edits:
            if not isinstance(edit, dict) or not {'path', 'use'}.issubset(edit) or set(edit)-{'path','use','edit_class'} or not isinstance(edit['use'], str) or not edit['use'].strip():
                raise ValueError('occurrence edit requires path and use')
            if edit.get('edit_class') not in (None,'placement','visibility','identification'):
                raise ValueError('occurrence edit_class must be placement, visibility or identification')
            path = _decision_path(edit['path'], stage, 'occurrence edit')
            if not valid_required_use(stage, edit):
                raise ValueError('Invalid '+edit.get('edit_class', 'placement')+' requirement: '+edit['path'])
            if not any(path.GetPrimPath().HasPrefix(Sdf.Path(p)) for p in paths):
                # Empty coordinate frames are absent from geometry review groups.
                # Permit their explicit preservation within the tool-owned asset
                # scope, without treating a sibling's geometry as reviewed.
                owner = candidate.get('owner_prim_path')
                prim = stage.GetPrimAtPath(path.GetPrimPath())
                empty_interface = (owner and path.GetPrimPath().HasPrefix(Sdf.Path(owner))
                    and all(Sdf.Path(p).HasPrefix(Sdf.Path(owner)) for p in paths)
                    and edit['path'] in decision['evidence_paths']
                    and bool(UsdGeom.Xformable(prim))
                    and not any(p.IsA(UsdGeom.Boundable) or (p.HasPayload() and not p.IsLoaded())
                                for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate))))
                if not empty_interface:
                    raise ValueError('occurrence edit is outside reviewed source scope; '
                                     'empty coordinate frames in the owning asset require explicit evidence')
        for field, allowed in (('structural_type', ('atomic', 'aggregate', 'package')),
                               ('kind', ('component', 'subcomponent', 'assembly', 'group'))):
            if field in decision:
                if decision['disposition'] != 'retain' or decision[field] not in allowed:
                    raise ValueError(field + ' requires a retained asset and a supported value')
                reason = decision.get('structural_reason' if field == 'structural_type' else 'kind_reason')
                if not isinstance(reason, str) or not reason.strip():
                    raise ValueError(field + ' requires a separate rationale')
        if ('structural_reason' in decision and 'structural_type' not in decision or
                'kind_reason' in decision and 'kind' not in decision):
            raise ValueError('structural/kind rationale requires its corresponding judgment')
        reconstruction = decision.get('reconstruction')
        if decision['disposition'] == 'reconstruct':
            if not isinstance(reconstruction, dict) or reconstruction.get('action') not in ('keep', 'regroup', 'split'):
                raise ValueError('reconstruction requires keep, regroup or split action')
            action = reconstruction['action']
            fields = {'action'} | ({'owner_prim_path'} if action == 'regroup' else
                                   {'operation', 'parameters'} if action == 'split' else set())
            if set(reconstruction) - ({'create_owner'} if action == 'regroup' else set()) != fields:
                raise ValueError('reconstruction fields do not match action')
            if action == 'regroup':
                owner = regroup_owner_path(reconstruction['owner_prim_path'], stage, reconstruction.get('create_owner', False))
                if any(owner.HasPrefix(Sdf.Path(p)) for p in paths):
                    raise ValueError('regroup owner cannot be a source prim or its descendant')
            if action == 'split':
                if reconstruction['operation'] != 'splitMeshes' or not isinstance(reconstruction['parameters'], dict):
                    raise ValueError('split requires native splitMeshes and parameters')
                if 'paths' in reconstruction['parameters']:
                    raise ValueError('split paths are derived from the reviewed source scope')
                # Installed argument validation and capability checks belong to
                # the native adapter; this contract does not invent its schema.
        elif reconstruction is not None:
            raise ValueError('reconstruction requires reconstruct disposition')
        normalized = copy.deepcopy(decision)
        normalized['semantic_category'] = 'general' if category.strip() == 'unknown' else category.strip()
        if 'equipment_family' in normalized:
            normalized['equipment_family'] = normalized['equipment_family'].strip()
        result.append(normalized)
    return result


def selection_errors(selection):
    if not isinstance(selection, dict):
        return ['selection evidence must be an object']
    errors = []
    if selection.get('basis') not in ('observed', 'user_specified', 'inferred'):
        errors.append('selection basis must be observed, user_specified or inferred')
    for key in ('reason', 'independent_use'):
        if not isinstance(selection.get(key), str) or not selection[key].strip():
            errors.append('selection requires ' + key)
    evidence = selection.get('evidence')
    if not isinstance(evidence, list) or not evidence or any(not isinstance(p, str) or not p.startswith('/') for p in evidence):
        errors.append('selection requires absolute prim/property evidence paths')
    return errors


def validate_selection(selection, stage):
    errors = selection_errors(selection)
    if errors:
        raise ValueError('; '.join(errors))
    for path in selection['evidence']:
        if not stage.GetObjectAtPath(path):
            raise ValueError('Selection evidence does not exist in composed stage: ' + path)
    return dict(selection)


def selection_for(path, signals, choice):
    if choice.get('selection'):
        return dict(choice['selection'])
    if choice:
        return {'basis': 'inferred', 'reason': choice.get('reason') or 'Legacy explicit boundary disposition.',
                'independent_use': 'Legacy boundary decision; intended use was not independently established.',
                'evidence': [path], 'legacy': True}
    actual = any(s in signals for s in ('protected_interface', 'declared_interface'))
    root = 'root_asset' in signals
    return {'basis': 'observed' if actual or root else 'inferred',
            'reason': ('Authored interface or independent behavior requires retained identity.' if actual else
                       'The stage entry point is retained.' if root else
                       'Composition or trusted model metadata nominates an asset boundary.'),
            'independent_use': ('Preserve the authored interface and independently addressed behavior.' if actual else
                                'Open and compose the supplied stage.' if root else
                                'Retain the existing authored model or externally composed asset entry.'),
            'evidence': [path]}


def identifying_facts(prim):
    """Small authored labels only; never infer a CAD field's meaning."""
    import re
    row = {'prim_path': str(prim.GetPath()), 'prim_name': prim.GetName()}
    if prim.GetDisplayName(): row['displayName'] = prim.GetDisplayName()
    info = prim.GetAssetInfo()
    if info: row['assetInfo'] = {str(k): str(v) if isinstance(v,Sdf.AssetPath) else v for k,v in info.items() if isinstance(v,(str,Sdf.AssetPath,int,float,bool))}
    labels = []
    for attr in prim.GetAuthoredAttributes():
        if 'material' in attr.GetName().lower() or 'shader' in attr.GetName().lower(): continue
        if not attr.IsCustom() or not re.search(r'name|label|description|product|partnumber|category|manufacturer|supplier|model|hersteller|lieferant|artikelnummer', attr.GetName(), re.I): continue
        if str(attr.GetTypeName()) not in ('string', 'token', 'asset'): continue
        value = attr.Get()
        if value: labels.append({'name': attr.GetName(), 'type': str(attr.GetTypeName()), 'value': str(value)[:256]})
    if labels:
        row['custom_attributes'] = labels[:12]
        row['custom_attributes_omitted'] = max(0, len(labels)-12)
    data=[]
    def descriptive_data(value, keys=()):
        for key,item in value.items():
            path=keys+(key,)
            if isinstance(item,dict): descriptive_data(item,path)
            elif isinstance(item,str) and re.search(r'name|label|description|product|partnumber|category|manufacturer|supplier|model|hersteller|lieferant|artikelnummer',key,re.I):
                if not any('material' in k.lower() or 'shader' in k.lower() for k in path):
                    data.append({'key_path':list(path),'value':item[:256]})
    descriptive_data(prim.GetCustomData())
    if data:
        row['customData_labels']=data[:12]
        row['customData_labels_omitted']=max(0,len(data)-12)
    # Only authored local extents. Missing extents never cause a point-array scan.
    extent = prim.GetAttribute('extent')
    if extent and extent.HasAuthoredValueOpinion() and not extent.ValueMightBeTimeVarying():
        value = extent.Get()
        if value is not None and len(value) == 2:
            row['authored_extent'] = {'minimum': list(value[0]), 'maximum': list(value[1]),
                'dimensions': [float(value[1][i]-value[0][i]) for i in range(3)],
                'coordinate_space': 'prim_local', 'time_code': 'default', 'purpose': str(UsdGeom.Imageable(prim).ComputePurpose())}
    row['label_scope']='Selected scalar descriptive metadata; values limited to 256 characters.'
    return row


def review_evidence(prims, children, geometry_under, identity):
    """Children-first topology hints; no exact content or appearance comparison."""
    from collections import Counter, defaultdict
    import re
    topology, labels, nearest = {}, {}, {}
    for path in sorted(prims, key=lambda p: p.count('/'), reverse=True):
        geometric = [p for p in children[path] if geometry_under[p]]
        child_tokens = [topology[p] for p in geometric]
        topology[path] = child_tokens[0] if len(child_tokens) == 1 else digest([prims[path].GetTypeName(), sorted(child_tokens)])
        own = identity.get(path, {})
        name=own.get('prim_name', '')
        descriptive_name = name if not re.fullmatch(r'n?[0-9a-fA-F]{6,}|(?:Xform|Scope|Mesh|Geo|Geometry|Wrapper|Part)\d*',name) else ''
        values = [own.get('displayName', ''), descriptive_name] + [v for v in own.get('assetInfo', {}).values() if isinstance(v,str)]
        values += [a['value'] for a in own.get('custom_attributes', [])]
        values += [a['value'] for a in own.get('customData_labels', [])]
        labels[path] = list(dict.fromkeys(v for v in values if v))[:12]
    for path in sorted(prims, key=lambda p: p.count('/')):
        parent = path.rsplit('/', 1)[0]
        nearest[path] = parent if labels.get(parent) else nearest.get(parent)
    def summarize(paths):
        members=[]; buckets=defaultdict(list)
        for path in paths:
            row=dict(identity.get(path, {'prim_path': path, 'prim_name': path.rsplit('/',1)[-1]}))
            row['bounds_status']='authored_extent' if 'authored_extent' in row else 'unknown'
            row['geometry_count']=len(geometry_under[path])
            geometric=[p for p in children[path] if geometry_under[p]]
            row['child_count']=len(geometric)
            row['child_schema_counts']=dict(Counter(prims[p].GetTypeName() for p in geometric))
            row['children']=[dict(identity.get(p, {'prim_path':p}), geometry_count=len(geometry_under[p])) for p in geometric[:8]]
            row['children_omitted']=max(0,len(geometric)-8)
            ancestor=nearest.get(path)
            row['nearest_descriptive_ancestor']=identity.get(ancestor) if ancestor else None
            row['sibling_parent_path']=path.rsplit('/',1)[0]
            members.append(row)
            buckets[('summarized_topology', topology[path])].append(path)
            extent=row.get('authored_extent')
            if extent:
                buckets[('authored_local_extent_dimensions',json.dumps([extent['dimensions'],extent['purpose'],extent['time_code']]))].append(path)
            for label in labels[path] + [v for p in geometric[:8] for v in labels[p]]:
                key=re.sub(r'^Copy\s*\(\d+\)\s+of\s+', '', label, flags=re.I).casefold()
                buckets[('related_authored_label', key)].append(path)
                # Equipment occurrence designations differ from product/model numbers.
                if re.fullmatch(r'\+\+[^+]+\+[A-Za-z]+\d+', label):
                    designation=re.sub(r'\d+$', '', label).casefold()
                    buckets[('equipment_designation_prefix', designation)].append(path)
        nominations=[]
        for (clue,key),group in buckets.items():
            group=sorted(set(group))
            if len(group)<2: continue
            nominations.append({'id':'nomination-'+digest([clue,key,group])[:20], 'status':'nominated',
                'member_roots':group,'occurrence_count':len(group),'representative':group[0],
                'method':'boundary_reasoning.review_evidence', 'clues':[clue],
                'comparison_scope':'authored_labels_and_wrapper_insensitive_topology',
                'geometry_compared':False,'appearance_compared':False,
                'interpretation':'Candidate repetition only; unequal geometry counts are allowed.'})
        combined={}
        for nomination in nominations:
            key=tuple(nomination['member_roots'])
            if key not in combined: combined[key]=nomination
            else: combined[key]['clues']=sorted(set(combined[key]['clues']+nomination['clues']))
        for key, nomination in combined.items(): nomination['id']='nomination-'+digest([key,nomination['clues']])[:20]
        return {'members':members,'repetition_nominations':list(combined.values())}
    return summarize


def _member_facts(prims, children, geometry_under, authored):
    """Memoized authored signals and wrapper topology; never asset identity."""
    cached = {}
    def inspect(path):
        if path not in cached:
            signals = dict(authored.get(path, {}))
            geometric = [p for p in children[path] if geometry_under[p]]
            direct = bool(geometric) and all(prims[p].IsA(UsdGeom.Gprim) for p in geometric)
            child_facts = {p: inspect(p) for p in children[path]}
            for child in child_facts.values():
                for name, witness in child[2].items():
                    signals.setdefault(name, witness)
            chain = direct or (len(geometric) == 1 and child_facts[geometric[0]][1])
            cached[path] = direct, chain, signals
        return cached[path]
    def summarize(paths):
        members = [inspect(path) for path in paths]
        return {'members_inspected': len(paths),
                'direct_geometry_wrapper_count': sum(item[0] for item in members),
                'chain_geometry_wrapper_count': sum(item[1] for item in members),
                'signals': {name: {'member_count': sum(name in item[2] for item in members),
                                  'witness_path': next((item[2][name] for item in members if name in item[2]), None)}
                            for name in ('motion', 'variants', 'nonmaterial_relationships', 'interface')},
                'motion_scope': 'existing_animation_and_rigid_body_detection',
                'signal_scope': 'member_and_all_cached_descendants',
                'interpretation': 'structural_and_authored_facts_not_asset_identity'}
    summarize.inspect = inspect
    return summarize


def collect_reviews(stage, prims, children, geometry_under, signatures, required, choices, group_choices, source_fingerprint, authored_facts=None, identifying=None):
    """Inspect a top-down review frontier; a nomination covers its whole interior.

    Single-child wrappers supply context, not independent review tasks. Selecting
    a boundary opens its next frontier. Internal branching containers also open
    their children for review; terminal geometry wrappers need no extra judgment.
    """
    summarize_members = _member_facts(prims, children, geometry_under, authored_facts or {})
    identity = identifying if identifying is not None else {p: identifying_facts(stage.GetPrimAtPath(p)) for p in prims if geometry_under[p] and not prims[p].IsA(UsdGeom.Gprim)}
    summarize_evidence = review_evidence(prims, children, geometry_under, identity)
    records, unresolved, used = [], [], set()
    owners = list(sorted(required))
    processed = set()
    while owners:
        owner = owners.pop(0)
        if owner in processed:
            continue
        processed.add(owner)
        if choices.get(owner, {}).get('decision_scope') == 'subtree':
            continue
        asset_owner = owner
        while asset_owner and asset_owner not in required:
            asset_owner = str(Sdf.Path(asset_owner).GetParentPath())
        aggregate_owner = choices.get(asset_owner, {}).get('structural_type') == 'aggregate'
        nominated = {}
        def nominate(path, skipped):
            if path != owner and path not in required:
                nominated[path] = list(skipped)
        def descend(path, skipped):
            if path != owner and path in required:
                return
            if path != owner and choices.get(path, {}).get('boundary') is False:
                nominate(path, skipped)
                return
            geometric = [child for child in children[path] if geometry_under[child]]
            containers = [child for child in geometric if prims[child].GetTypeName() in ('Xform', 'Scope', '')]
            if aggregate_owner and (path == owner or containers):
                for child in geometric:
                    if prims[child].IsA(UsdGeom.Gprim):
                        nominate(child, skipped)
            if len(geometric) == 1 and len(containers) == 1:
                descend(containers[0], skipped + ([path] if path != owner else []))
                return
            if containers:
                for child in containers:
                    if child not in required:
                        nominate(child, skipped + ([path] if path != owner else []))
                return
            if path != owner and prims[path].GetTypeName() in ('Xform', 'Scope', ''):
                nominate(path, skipped)
        descend(owner, [])
        grouped = {}
        for path in sorted(nominated):
            shallow = tuple(sorted({prims[child].GetTypeName() for child in children[path] if geometry_under[child]}))
            grouped.setdefault(shallow, []).append(path)
        owner_evidence=summarize_evidence(sorted(nominated))
        member_evidence={row['prim_path']:row for row in owner_evidence['members']}
        for p, row in member_evidence.items():
            row['authored_signal_witnesses']=summarize_members.inspect(p)[2]
        for signature, paths in sorted(grouped.items()):
            path_set=set(paths)
            evidence={'members':[member_evidence[p] for p in paths],
                      'repetition_nominations':[n for n in owner_evidence['repetition_nominations'] if path_set.intersection(n['member_roots'])],
                      'meters_per_unit':UsdGeom.GetStageMetersPerUnit(stage)}
            enclosing = {}
            for member in paths:
                alternatives = nominated[member]
                if choices.get(owner, {}).get('boundary') is False:
                    alternatives = [owner] + alternatives
                for alternative in alternatives:
                    enclosing.setdefault(alternative, []).append(member)
            evidence['enclosing_boundaries'] = [
                dict(identity[path], member_roots=members) for path, members in sorted(enclosing.items())]
            fingerprint = digest([source_fingerprint, owner, signature, paths])
            identifier = 'boundary-' + fingerprint[:24]
            grouped_choice = group_choices.get(identifier)
            if grouped_choice is not None:
                if set(grouped_choice) - {'group_id', 'disposition', 'selection', 'evidence_fingerprint'}:
                    raise ValueError('Unknown grouped boundary review fields')
                if grouped_choice.get('evidence_fingerprint', fingerprint) != fingerprint:
                    raise ValueError('Grouped decision evidence has changed: ' + identifier)
                if grouped_choice.get('disposition') not in ('boundary', 'internal'):
                    raise ValueError('Boundary review requires boundary or internal disposition')
                selected = validate_selection(grouped_choice.get('selection'), stage)
                used.add(identifier)
                for path in paths:
                    choices.setdefault(path, {'path': path, 'boundary': grouped_choice['disposition'] == 'boundary',
                                              'selection': selected})
            pending = [path for path in paths if 'boundary' not in choices.get(path, {})
                       and not choices.get(path, {}).get('structural_type')]
            dispositions = []
            for path in paths:
                choice = choices.get(path)
                if choice is None or path in pending:
                    continue
                selection = selection_for(path, [], choice)
                validate_selection(selection, stage)
                retain = choice.get('boundary', bool(choice.get('structural_type')))
                dispositions.append({'path': path, 'disposition': 'boundary' if retain else 'internal', 'selection': selection})
                if retain:
                    required.setdefault(path, []).append('reviewed_boundary')
                    owners.append(path)
                elif choice.get('decision_scope') != 'subtree':
                    # A prim-only decision must also expose a single child container.
                    # Direct geometry ends naturally in descend without mesh judgments.
                    owners.append(path)
            record = {'id': identifier, 'group_id': identifier, 'type': 'boundary_review', 'paths': paths,
                      'owner_path': owner, 'evidence_fingerprint': fingerprint, 'dispositions': dispositions,
                      'grouping': 'review_batch_only_not_equivalence', 'path_count': len(paths),
                      'coverage': 'subtrees_except_independently_required_owners',
                      'evidence': {**evidence, 'all_members': summarize_members(paths), 'representative': paths[0], 'geometry_count': len(geometry_under[paths[0]]),
                                   'skipped_wrapper_paths': nominated[paths[0]],
                                   'children': [{'path': p, 'schema': prims[p].GetTypeName(),
                                                 'geometry_count': len(geometry_under[p])}
                                                for p in [c for c in children[paths[0]] if geometry_under[c]][:8]],
                                   'children_omitted':max(0, sum(bool(geometry_under[c]) for c in children[paths[0]])-8)},
                      'status': 'decision_required' if pending else 'reviewed'}
            if not pending and dispositions:
                kinds = {item['disposition'] for item in dispositions}
                record['disposition'] = next(iter(kinds)) if len(kinds) == 1 else 'mixed'
                if all(item['selection'] == dispositions[0]['selection'] for item in dispositions):
                    record['selection'] = dispositions[0]['selection']
            records.append(record)
            if pending:
                unresolved.append({**record, 'paths': pending, 'choices': ['boundary', 'internal'],
                    'consequence': 'Review each region and its child summary. Similar structure only batches review. Internal branching containers expose child groups for review; terminal geometry wrappers stay within their owner.',
                    'response_shape': {'groups': {identifier: {'disposition': 'boundary|internal',
                        'evidence_fingerprint': fingerprint,
                        'selection': {'basis': 'observed|user_specified|inferred',
                                      'reason': '<supporting facts and judgment>',
                                      'independent_use': '<intended edit/replacement/motion, or why none>',
                                      'evidence': [paths[0]]}}}}})
    return records, unresolved, used
