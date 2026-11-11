#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inspect selected USD composition and apply compact, evidence-bound judgments.

Usage:
    python successor_assessment.py inspect --source STAGE --workspace DIRECTORY
    python successor_assessment.py decide --workspace DIRECTORY --decisions JSON
Arguments:
    inspect requires a source stage and workspace; --request optionally supplies
    workload/preservation JSON. decide reads completed decisions from JSON.
Output:
    Prints assessment status JSON and writes candidates.json, decision-template.json,
    decision-schema.json, decision-help.json, assessment-state.json and manifest.json.
Exit codes:
    0 when the command returns, 1 for an uncaught error, 2 for invalid CLI arguments.
    Inspect the returned assessment status; exit 0 does not mean review is complete.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]


def _load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def _tools():
    tuning=_load('successor_assessment_tuning',Path(__file__).with_name('usd_tuning.py'))
    discovery=tuning.helper('successor_discovery','usd-structure-assessment/scripts/structure_assessment.py')
    reasoning=tuning.helper('successor_reasoning','usd-structure-assessment/scripts/boundary_reasoning.py')
    return tuning,discovery,reasoning


def _read(value):
    if isinstance(value,dict):return copy.deepcopy(value)
    return json.loads(Path(value).read_text())


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def _file_hash(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def _dependencies_current(context):
    for path,fingerprint in context['dependency_fingerprints'].items():
        if not Path(path).is_file() or _file_hash(path)!=fingerprint:
            raise ValueError('Source dependency changed; inspect into a new workspace: '+path)


def _candidates(manifest,reconstruction_requests=()):
    """Only pending members of a review group are actionable."""
    blocked={path for row in reconstruction_requests for path in row['source_prim_paths']}
    result={}
    for row in manifest.get('decisions_required',[]):
        if row.get('type') not in ('boundary_review','structural_role'):continue
        paths=[path for path in row.get('paths',[]) if path not in blocked]
        if not paths:continue
        identifier=row.get('group_id') or row.get('id')
        if not identifier or not row.get('evidence_fingerprint'):
            raise ValueError('Assessment candidate is missing its identity or evidence fingerprint')
        if identifier in result:raise ValueError('Duplicate assessment candidate group')
        result[identifier]={'candidate_id':identifier,'prim_paths':paths,
                            'evidence_fingerprint':row['evidence_fingerprint'],
                            'grouping':row.get('grouping'),
                            'type':row['type'],'owner_prim_path':row.get('owner_path'),
                            'evidence':copy.deepcopy(row.get('evidence',{})),
                            'consequence':row.get('consequence','')}
    for candidate in result.values():
        paths=set(candidate['prim_paths'])
        candidate['evidence']['confirmed_reuse']=[copy.deepcopy(r) for r in manifest.get('reuse_candidates',[]) if paths.intersection(r.get('member_roots',[]))]
    # Protected interfaces and the stage root can be selected mechanically.
    # Their role/category still needs an addressable decision, rather than
    # requiring the agent to edit a generated definition record by hand.
    reviewed={row['path'] for row in manifest.get('boundary_decisions',[])
              if row.get('boundary') is True and 'semantic_category' in row}
    already={path for candidate in result.values() for path in candidate['prim_paths']}
    excluded=reviewed|blocked|already
    for definition in manifest.get('definitions',[]):
        paths=[p for p in definition.get('occurrences',[]) if p not in excluded]
        if not paths:continue
        evidence={'definition_id':definition['id'],'structural_type':definition['structural_type'],
                  'owned_geometry_count':(len(definition['owned_geometry']) if 'owned_geometry' in definition
                                          else sum(definition.get('geometry_counts',{}).values())),
                  'child_definition_ids':definition.get('children',[]),
                  'content_fingerprint':definition.get('content_fingerprint'),
                  'source_fingerprint':manifest['source']['fingerprint'],'prim_paths':paths}
        fingerprint=_digest(evidence);identifier='asset-role-'+fingerprint[:24]
        result[identifier]={'candidate_id':identifier,'prim_paths':paths,'evidence_fingerprint':fingerprint,
                            'type':'asset_role','owner_prim_path':None,'evidence':evidence,
                            'consequence':'Confirm the selected asset role and semantic category in its composed context.'}
    return result


def _validate_candidates(candidates):
    from pxr import Sdf
    seen=set()
    for identifier,row in candidates.items():
        if identifier!=row.get('candidate_id') or not isinstance(row.get('prim_paths'),list) or not row['prim_paths']:
            raise ValueError('Malformed candidate group')
        members=[]
        for value in row['prim_paths']:
            if not isinstance(value,str) or not Sdf.Path.IsValidPathString(value):raise ValueError('Malformed candidate prim path')
            path=Sdf.Path(value)
            if not path.IsAbsolutePath() or not path.IsPrimPath():raise ValueError('Candidate requires absolute prim paths')
            if value in seen:
                raise ValueError('Candidate groups have duplicate or overlapping members')
            members.append(path);seen.add(value)
        # Sdf path ordering keeps each ancestor immediately before its subtree.
        # Adjacent prefixes detect the same within-group overlaps without a
        # quadratic scan. Cross-group overlaps remain allowed; duplicates do not.
        members.sort()
        if any(right.HasPrefix(left) or left.HasPrefix(right)
               for left,right in zip(members,members[1:])):
            raise ValueError('Candidate groups have duplicate or overlapping members')


def decision_schema(source_context):
    """Emit the compact response contract beside each current candidate set."""
    text={'type':'string','minLength':1}
    properties={name:dict(text) for name in ('candidate_id','evidence_fingerprint','reason','independent_use','semantic_category','equipment_family','prim_path','structural_reason','kind_reason')}
    properties.update(geometry_representation={'enum':['merged_meshes','analytic_primitives']},decision_scope={'enum':['prim','subtree']},disposition={'enum':['retain','internal','reconstruct']},basis={'enum':['observed','user_specified','inferred']},
                      source_prim_paths={'type':'array','minItems':1,'uniqueItems':True,'items':text},
                      evidence_paths={'type':'array','minItems':1,'uniqueItems':True,'items':text},
                      required_occurrence_edits={'type':'array','items':{'type':'object','additionalProperties':False,'required':['path','use'],'properties':{'path':text,'use':text,'edit_class':{'enum':['placement','visibility','identification']}}}},
                      structural_type={'enum':['atomic','aggregate','package']},kind={'enum':['component','subcomponent','assembly','group']},
                      reconstruction={'oneOf':[{'type':'object','additionalProperties':False,'required':['action'], 'properties':{'action':{'const':'keep'}}},
                                               {'type':'object','additionalProperties':False,'required':['action','owner_prim_path'],'properties':{'action':{'const':'regroup'},'owner_prim_path':text,'create_owner':{'type':'boolean','description':'Explicitly create one new Xform beneath an existing parent; existing owners must omit this flag or use false.'}}},
                                               {'type':'object','additionalProperties':False,'required':['action','operation','parameters'],'properties':{'action':{'const':'split'},'operation':{'const':'splitMeshes'},'parameters':{'type':'object'}}}]})
    properties['product_identity']={'type':'object','required':['key','reason','frame','evidence']}
    item={'type':'object','additionalProperties':False,'properties':properties,
          'required':['candidate_id','evidence_fingerprint','disposition','reason','independent_use','basis','evidence_paths','semantic_category','required_occurrence_edits'],
          'oneOf':[{'required':['prim_path'],'not':{'required':['source_prim_paths']}},{'required':['source_prim_paths'],'not':{'required':['prim_path']}}],
          'dependentRequired':{'kind':['kind_reason'],'kind_reason':['kind'],'structural_type':['structural_reason'],'structural_reason':['structural_type']},
          'allOf':[{'if':{'required':['decision_scope']},'then':{'properties':{'disposition':{'enum':['retain','internal']}}}}, {'if':{'properties':{'disposition':{'const':'reconstruct'}}},'then':{'required':['reconstruction']},'else':{'not':{'required':['reconstruction']}}},
                   {'if':{'anyOf':[{'required':['kind']},{'required':['structural_type']},{'required':['geometry_representation']}]},'then':{'properties':{'disposition':{'const':'retain'}}}}]}
    return {'$schema':'https://json-schema.org/draft/2020-12/schema','title':'Selected USD composition decision response','type':'object',
            'additionalProperties':False,'required':['schema_version','source_context','decisions'],
            'properties':{'schema_version':{'const':1},'source_context':{'const':source_context},'decisions':{'type':'array','items':item}}}


def selection_evidence(selection, manifest):
    """Resolve the complete cited paths; never treat a representative as full evidence."""
    reference=selection.get('evidence_reference')
    if reference is None:return selection.get('evidence',[])
    paths=manifest.get('selection_evidence',{}).get(reference)
    if paths is None or reference!=_digest(paths):
        raise ValueError('Missing or modified shared selection evidence: '+reference)
    return paths


def _library_nominations(candidates, index_paths, identifying=None):
    """Derived index evidence; never modify the frozen USD evidence cache."""
    lookup = _load('assessment_library_lookup', ROOT / 'references/usd-optimize-run-operations/scripts/persistent_library.py')
    context = lookup.LookupContext(index_paths)
    queried = {}
    for candidate in candidates.values():
        rows = []
        members = list(candidate['evidence'].get('members', []))
        present = {row['prim_path'] for row in members}
        members += [identifying[path] for path in candidate.get('prim_paths', [])
                    if path not in present and identifying and path in identifying]
        candidate['evidence']['members'] = members
        pending = set(candidate.get('prim_paths', [m['prim_path'] for m in members]))
        candidate['evidence']['product_review_groups'] = lookup.product_clues.review_groups(
            [m for m in members if m['prim_path'] in pending])
        seen = set()
        for member in members:
            # Child evidence retains its own path: a common connector does not
            # imply that the containing equipment matches the library entry.
            for facts in [member] + member.get('children', []):
                values = tuple(lookup.identifying_values(facts))
                if values not in queried:
                    queried[values] = context.identifiers(values) if values else []
                for match in queried[values]:
                    key = (facts['prim_path'], match['index'], match['id'])
                    if key not in seen:
                        rows.append(dict(match, source_prim_path=facts['prim_path']))
                        seen.add(key)
        candidate['evidence']['persistent_nominations'] = rows
    return context.findings


def _publish(workspace,state,manifest,tuning):
    manifest=copy.deepcopy(manifest)
    if state.get('selection_evidence'):
        manifest['selection_evidence']=copy.deepcopy(state['selection_evidence'])
    for field in ('original_source','reconstruction_material_target_repairs','reconstruction_results','reconstruction_path_map','inherited_decisions','invalidated_decisions','unresolved_requirements','required_occurrence_edits'):
        if field in state:manifest[field]=copy.deepcopy(state[field])
    edits=list(state.get('required_occurrence_edits',[]))
    edits.extend(edit for decision in state.get('accepted_decisions',[]) for edit in decision.get('required_occurrence_edits',[]))
    manifest['required_occurrence_edits']=list({(edit['path'],edit['use'],edit.get('edit_class')):copy.deepcopy(edit) for edit in edits}.values())
    pending=state.get('reconstruction_requests',[])
    candidates=_candidates(manifest,pending)
    reviewed={row['path'] for row in manifest.get('boundary_decisions',[]) if 'boundary' in row}
    present={path for candidate in candidates.values() for path in candidate['prim_paths']}
    for review in state.get('required_reviews',[]):
        paths=[p for p in review['paths'] if p not in reviewed and p not in present]
        if not paths:continue
        fingerprint=_digest([manifest['source']['fingerprint'],review,paths])
        identifier='reconstruction-review-'+fingerprint[:24]
        candidates[identifier]={'candidate_id':identifier,'prim_paths':paths,'evidence_fingerprint':fingerprint,
                               'type':'reconstruction_review','owner_prim_path':None,'evidence':review,
                               'consequence':'Reassess prior intent against the reconstructed geometry.'}
        present.update(paths)
    _validate_candidates(candidates)
    manifest['persistent_lookup_findings'] = _library_nominations(candidates, state.get('request', {}).get('persistent_indexes', []), manifest.get('review_index', {}).get('identifying', {}))
    state['candidates']=candidates
    state['candidates_sha256']=_digest(candidates)
    state['status']='reconstruction_required' if pending else 'decision_required' if candidates else 'incomplete' if state.get('unresolved_requirements') else manifest['status']
    if candidates or pending or state.get('unresolved_requirements'):
        manifest['status']='incomplete'
    manifest['successor_decisions_required']=list(candidates)
    if pending:
        manifest['reconstruction_requests']=copy.deepcopy(pending)
        manifest['decisions_required']=[row for row in manifest['decisions_required'] if row.get('type')!='reconstruction_pending']
        manifest['decisions_required'] += [{'type':'reconstruction_pending','id':row['id'],'paths':row['source_prim_paths'],
                                         'reason':'Accepted reconstruction has not been executed or verified.'} for row in pending]
    tuning.write(workspace/'manifest.json',manifest)
    state['manifest_path']=str(workspace/'manifest.json')
    tuning.write(workspace/'source-context.json',state['source_context'])
    tuning.write(workspace/'candidates.json',candidates)
    tuning.write(workspace/'decision-template.json',{'schema_version':1,'source_context':state['source_context'],'decisions':[]})
    tuning.write(workspace/'decision-schema.json',decision_schema(state['source_context']))
    example=None
    if candidates:
        candidate=next((row for row in candidates.values() if len(row['prim_paths'])>1),next(iter(candidates.values())))
        example_paths=candidate['prim_paths'][:2]
        example={'candidate_id':candidate['candidate_id'],'evidence_fingerprint':candidate['evidence_fingerprint'],
                 'source_prim_paths':example_paths,'disposition':'retain','decision_scope':'subtree',
                 'reason':'REPLACE with the common boundary decision and why the supplied evidence resolves these interiors.',
                 'independent_use':'REPLACE with the required placement, replacement, motion or other use.',
                 'basis':'inferred','evidence_paths':example_paths,'semantic_category':'unknown','required_occurrence_edits':[]}
    tuning.write(workspace/'decision-help.json',{'purpose':'Grouped example format only; select members sharing the reviewed decision and replace judgment fields. Use subtree only when supplied evidence resolves their interiors; otherwise use prim.',
                 'checklist':'references/structure.md#boundary-decision-checklist',
                 'geometry_representation':'Select the operation order per retained asset using references/operations.md#conditional-execution-order; record the choice and its reason. Omission uses merged_meshes.',
                 'evidence_fields':['members','enclosing_boundaries','product_review_groups','repetition_nominations','persistent_nominations','confirmed_reuse','all_members'],
                 'review_groups':'Read product_review_groups and member evidence together. Submit one source_prim_paths row for members sharing the decision, rather than repeating the same reason per prim. Split exceptions into disjoint rows; keep differing required edits and product identities distinct, and retain evidence for every member of an accepted product group. Product groups and schema groups nominate review, not equivalence.',
                 'product_identity':'Optional on retain: {key, reason, frame: asset_root, evidence: [{path, field, value}], library_entry: {index, id} (optional)}. Record interchangeable product identity and compatible asset-root frame for the workload. Reuse the nominated library product key when accepting that product. Cite exact name/displayName, assetInfo:<key>, attribute:<name> or dimensions_m facts for every grouped asset. Explain which actual model, part or configuration values distinguish the product; a genuine identifier may be in a name. A generic description echoed in exporter metadata remains one clue. Preserve relevant revision, dimensions, configuration, handedness and cut length in the key. Manufacturer alone, family membership, bounds alone, or a category or description do not establish product identity; omit product_identity in those cases; existing content matching remains available. Accepted identity authorizes reuse of a published representation, not identical source geometry.',
                 'example_decision':example,'dispositions':{'retain':'Select this prim as an asset boundary for extraction and publication; its internal hierarchy may be optimized.','internal':'Keep the region within its owning asset while preserving declared edits.','reconstruct':'Stage keep, regroup or native split work; no reconstruction runs during decide.'},
                 'decision_scope':'For retain or internal, prim (default) exposes child groups; subtree records completed interior review. Separately retained and protected boundaries still require their own decisions.',
                 'partial_review':'A response may review a disjoint subset of a candidate group. Remaining members stay pending. If children were reviewed first, review their enclosing equipment using the linked boundary checklist. Select its exact path from evidence.enclosing_boundaries when it represents a complete unit required by the workload. Organizational ancestors do not require separate asset decisions.',
                 'regroup_owner':'Use reconstruction {action: regroup, owner_prim_path: /existing/owner}. To create one absent Xform beneath an existing defined parent, add create_owner: true. Creation cannot overwrite a prim, introduce missing ancestors, enter an instance, or move the owner ancestry. Cite existing source prims and parent as evidence; the new owner is reassessed after reconstruction.',
                 'reconstruction_scope':'For reconstruct, source prims may be descendants of the candidate members; scopes must be disjoint.',
                 'basis_values':['observed','user_specified','inferred'],'unknown_category':'unknown becomes general',
                 'structural_and_kind_judgments':'Optional on retain; structural_reason and kind_reason accompany explicit overrides.',
                 'required_occurrence_edits':'Each entry names an existing prim/property path, its required use, and edit_class placement, visibility or identification. Identification preserves the named object for a stated selection/lookup use; placement and visibility request editable USD capabilities. An empty list is valid. An empty coordinate frame outside the reviewed members may be preserved within owner_prim_path when its exact edit path is included in evidence_paths; geometry-bearing siblings still require their own review.'})
    tuning.write(workspace/'assessment-state.json',state)
    return {'schema_version':1,'status':state['status'],'workspace':str(workspace),
            'source_context':state['source_context'],'candidates':candidates,
            'reconstruction_requests':pending,'manifest_path':state['manifest_path'],
            'decision_template_path':str(workspace/'decision-template.json'),'decision_schema_path':str(workspace/'decision-schema.json'),
            'decision_help_path':str(workspace/'decision-help.json'),
            'accepted_decision_count':len(state.get('accepted_decisions',[]))}


def inspect(source,workspace,request=None):
    """Write selected source context and grouped candidates; never edit source USD."""
    tuning,discovery,reasoning=_tools()
    workspace=Path(workspace).resolve();source=Path(source).resolve()
    policy=_read(request) if request is not None else {}
    if policy.get('variants'):
        raise ValueError('Variant overrides are outside successor scope; supply the input with the intended current selections authored.')
    if policy.get('successor_policy') is False:raise ValueError('Successor inspection requires successor_policy')
    policy['successor_policy']=True
    policy.setdefault('direct_library_authoring',False)
    workspace.mkdir(parents=True,exist_ok=True)
    saved=workspace/'assessment-state.json'
    if saved.exists():
        state=_read(saved)
        if state.get('superseded_by'):raise ValueError('Continue in '+state['superseded_by'])
        if state['source']!=str(source) or state['initial_request']!=policy:
            raise ValueError('Workspace already belongs to another source or request; use a new workspace')
        _dependencies_current(state['source_context'])
        if state['assessment_policy_sha256']!=tuning.assessment_policy_fingerprint():
            raise ValueError('Assessment implementation changed; inspect into a new workspace')
        return _publish(workspace,state,_read(workspace/'manifest.json'),tuning)
    if (workspace/'manifest.json').exists():
        raise ValueError('Workspace contains an assessment without successor state; use a new workspace')
    # Refresh USD's layer cache so a new workspace sees edited source bytes.
    stage=discovery._open_stage(source,policy);stage.Reload()
    manifest=tuning.assessment(source,workspace,request=policy,force_rescan=True)
    stage=discovery._open_stage(source,policy)
    dependencies={row['path']:row['fingerprint'] for row in manifest['inventory']['dependencies']}
    selections={}
    for row in manifest['inventory'].get('variant_contexts',[]):
        selections.setdefault(row['path'],{})[row['set']]=row['selection']
    context=reasoning.decision_source_context(stage,dependencies,selections)
    state={'schema_version':1,'source':str(source),'initial_request':policy,'request':copy.deepcopy(policy),
           'source_context':context,'assessment_policy_sha256':tuning.assessment_policy_fingerprint(),
           'accepted_decisions':[],'reconstruction_requests':[]}
    return _publish(workspace,state,manifest,tuning)


def accept_decisions(workspace,request_json):
    """Apply disjoint reviewed members and retain earlier accepted decisions.

    Reconstruction requests are staged for a controller. They never run native
    geometry operations here and cannot make a stage ready for publication.
    """
    from pxr import Sdf
    tuning,discovery,reasoning=_tools()
    workspace=Path(workspace).resolve()
    state=_read(workspace/'assessment-state.json')
    if state.get('superseded_by'):raise ValueError('Continue in '+state['superseded_by'])
    _dependencies_current(state['source_context'])
    if state['assessment_policy_sha256']!=tuning.assessment_policy_fingerprint():
        raise ValueError('Assessment implementation changed; inspect into a new workspace')
    candidates=state['candidates']
    _validate_candidates(candidates)
    if state.get('candidates_sha256')!=_digest(candidates) or _read(workspace/'candidates.json')!=candidates:
        raise ValueError('Tool-owned candidate evidence was modified; inspect again')
    stage=discovery._open_stage(state['source'],state['request'])
    submitted=_read(request_json)
    validation_candidates=copy.deepcopy(candidates)
    # A reconstruction can address disjoint members inside a reviewed organizational prim
    # (for example bolts with different owners). Retain/internal judgments still
    # address only the nominated boundary members. The context fingerprint binds
    # all descendant evidence to the same unchanged selected source.
    if isinstance(submitted,dict) and isinstance(submitted.get('decisions'),list):
        for item in submitted['decisions']:
            if not isinstance(item,dict) or item.get('disposition')!='reconstruct':continue
            candidate=validation_candidates.get(item.get('candidate_id'))
            if candidate is None:continue
            paths=[item['prim_path']] if 'prim_path' in item else item.get('source_prim_paths',[])
            if not isinstance(paths,list):continue
            original_roots=candidates[item['candidate_id']]['prim_paths']
            for path in paths:
                if isinstance(path,str) and Sdf.Path.IsValidPathString(path) and any(Sdf.Path(path).HasPrefix(Sdf.Path(root)) for root in original_roots):
                    if path not in candidate['prim_paths']:candidate['prim_paths'].append(path)
    decisions=reasoning.validate_successor_decisions(submitted,stage,validation_candidates,state['source_context'],
        dependency_validated_variants=state['source_context']['selected_variants'])
    if not decisions:
        return _publish(workspace,state,_read(workspace/'manifest.json'),tuning)
    boundaries=[]
    protected=set(state['request'].get('protected_paths',[]))
    staged=list(state['reconstruction_requests'])
    for decision in decisions:
        paths=[decision['prim_path']] if 'prim_path' in decision else decision['source_prim_paths']
        for edit in decision['required_occurrence_edits']:
            protected.add(str(Sdf.Path(edit['path']).GetPrimPath()))
        if decision['disposition']=='reconstruct':
            staged.append({'id':'reconstruct-'+_digest(decision)[:24],
                           'source_prim_paths':paths,'request':decision['reconstruction'],
                           'reason':decision['reason'],'evidence_fingerprint':decision['evidence_fingerprint'],
                           'source_context':state['source_context'],'status':'pending','executed':False})
            continue
        selection={'reason':decision['reason'],'independent_use':decision['independent_use'],
                   'basis':decision['basis'],'evidence':decision['evidence_paths']}
        # A grouped judgment is validated once above. Repeating its complete
        # citations in every member, review and definition makes JSON quadratic.
        # Keep one original citation for existing selection readers, and retain
        # the complete list once for evidence-aware consumers and inspection.
        if len(paths)>1 and len(decision['evidence_paths'])>1:
            reference=_digest(decision['evidence_paths'])
            state.setdefault('selection_evidence',{})[reference]=decision['evidence_paths']
            selection['evidence']=decision['evidence_paths'][:1]
            selection['evidence_reference']=reference
        for path in paths:
            boundary={'path':path,'boundary':decision['disposition']=='retain','reason':decision['reason'],
                      'selection':selection,'semantic_category':decision['semantic_category']}
            if 'product_identity' in decision:boundary['product_identity']=copy.deepcopy(decision['product_identity'])
            if 'equipment_family' in decision:boundary['equipment_family']=decision['equipment_family']
            if 'geometry_representation' in decision:boundary['geometry_representation']=decision['geometry_representation']
            if 'decision_scope' in decision:boundary['decision_scope']=decision['decision_scope']
            if 'structural_type' in decision:boundary['structural_type']=decision['structural_type']
            if 'kind' in decision:
                boundary['kind']=decision['kind']
                boundary['standalone_kind']='component' if decision['kind']=='subcomponent' else decision['kind']
            boundaries.append(boundary)
        # A required edit remains authored in the owning composition even when
        # its prim is explicitly internal. Its geometry may be selected later.
        if decision['disposition']=='internal':
            for edit in decision['required_occurrence_edits']:
                path=str(Sdf.Path(edit['path']).GetPrimPath())
                if path not in paths:
                    boundary={'path':path,'boundary':False,'reason':decision['reason'],'selection':selection}
                    if decision.get('decision_scope')=='subtree' and any(Sdf.Path(path).HasPrefix(Sdf.Path(root)) for root in paths):
                        boundary['decision_scope']='subtree'
                    boundaries.append(boundary)
    policy=copy.deepcopy(state['request'])
    requirements = list(state.get('required_occurrence_edits', []))
    requirements.extend(edit for row in state['accepted_decisions'] + decisions for edit in row.get('required_occurrence_edits', []))
    policy['required_occurrence_edits'] = list({(e['path'], e.get('edit_class', 'placement')): e for e in requirements}.values())
    if protected or 'protected_paths' in policy: policy['protected_paths']=sorted(protected)
    if boundaries:
        manifest=tuning.assessment(state['source'],workspace,request=policy,decisions={'boundaries':boundaries})
    else:
        manifest=_read(workspace/'manifest.json')
    state['request']=policy
    state['accepted_decisions'].extend(decisions)
    state['reconstruction_requests']=staged
    return _publish(workspace,state,manifest,tuning)


def cli_summary(result):
    """Keep CLI handoffs small; the Python result and saved evidence stay complete."""
    summary = {key: value for key, value in result.items()
               if key not in ('candidates', 'reconstruction_requests')}
    summary['pending_candidate_count'] = len(result.get('candidates', {}))
    summary['pending_reconstruction_count'] = len(result.get('reconstruction_requests', []))
    workspace = Path(result['workspace'])
    summary['candidates_path'] = str(workspace / 'candidates.json')
    summary['assessment_state_path'] = str(workspace / 'assessment-state.json')
    return summary


def show_candidates(workspace, candidate_id=None, field=None, offset=0, limit=10):
    """Page existing evidence without changing decisions or their fingerprints."""
    if offset < 0 or not 1 <= limit <= 100:
        raise ValueError('offset must be nonnegative and limit between 1 and 100')
    candidates = _read(Path(workspace) / 'candidates.json')
    if candidate_id is None:
        rows = [{'candidate_id': key, 'type': row.get('type'),
                 'owner_prim_path': row.get('owner_prim_path'),
                 'prim_count': len(row.get('prim_paths', [])),
                 'product_review_group_count': len(row.get('evidence', {}).get('product_review_groups', [])),
                 'evidence_fingerprint': row['evidence_fingerprint']}
                for key, row in candidates.items()]
        value = rows
    else:
        value = candidates[candidate_id]
        if not field:
            return {'candidate_id': candidate_id, 'fields': list(value),
                    'evidence_fields': list(value.get('evidence', {})),
                    'prim_count': len(value.get('prim_paths', [])),
                    'evidence_fingerprint': value['evidence_fingerprint']}
        for key in field.split('.'):
            value = value[int(key)] if isinstance(value, list) else value[key]
    if isinstance(value, list):
        end = min(len(value), offset + limit)
        return {'candidate_id': candidate_id, 'field': field, 'total': len(value),
                'offset': offset, 'next_offset': end if end < len(value) else None,
                'items': value[offset:end]}
    return {'candidate_id': candidate_id, 'field': field, 'value': value}


def main():
    parser=argparse.ArgumentParser(description=__doc__,epilog='Inspect writes complete candidates.json and decision templates. Use show to page candidate groups, then --candidate ID --field evidence.members to read members. Fill the decision template from the evidence and pass it to decide.')
    commands=parser.add_subparsers(dest='command',required=True)
    command=commands.add_parser('inspect');command.add_argument('--source',required=True);command.add_argument('--workspace',required=True);command.add_argument('--request')
    command=commands.add_parser('decide');command.add_argument('--workspace',required=True);command.add_argument('--decisions',required=True)
    command=commands.add_parser('show');command.add_argument('--workspace',required=True);command.add_argument('--candidate');command.add_argument('--field');command.add_argument('--offset',type=int,default=0);command.add_argument('--limit',type=int,default=10)
    args=parser.parse_args()
    if args.command == 'show':
        result=show_candidates(args.workspace,args.candidate,args.field,args.offset,args.limit)
    else:
        result=inspect(args.source,args.workspace,args.request) if args.command=='inspect' else accept_decisions(args.workspace,args.decisions)
        result=cli_summary(result)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
