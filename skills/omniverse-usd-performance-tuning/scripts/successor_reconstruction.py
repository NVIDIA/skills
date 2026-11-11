#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execute accepted reconstruction requests, retain receipts, then reassess USD.

Usage:
    python successor_reconstruction.py --workspace DIRECTORY [--request-id ID]
Arguments:
    Existing assessment workspace; repeat --request-id to select accepted requests.
Output:
    Retains reconstruction attempts, generated USD and reassessment artifacts;
    prints the result as JSON. Successful reconstruction supersedes the prior
    assessment workspace and records its successor.
Exit codes:
    0 when the command returns, 1 for an uncaught error, 2 for invalid CLI arguments.
    Read advanced/status and unresolved requirements; exit 0 is not completion.
"""
from __future__ import annotations
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys


def _adapter():
    spec=importlib.util.spec_from_file_location('reconstruction_assessment_adapter',Path(__file__).with_name('successor_assessment.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def _mapped(path,mapping,stage=None):
    """Apply ordered root substitutions to prim or property paths on demand."""
    from pxr import Sdf
    paths=[str(path)]
    for step in mapping if isinstance(mapping,list) else [mapping]:
        targets=[]
        for value in paths:
            current=Sdf.Path(value);ancestor=current.GetPrimPath()
            while ancestor!=Sdf.Path.absoluteRootPath and str(ancestor) not in step:
                ancestor=ancestor.GetParentPath()
            replacements=step.get(str(ancestor))
            if replacements is None:
                targets.append(value)
            else:
                targets.extend(str(current.ReplacePrefix(ancestor,Sdf.Path(p))) for p in replacements)
        paths=sorted(set(targets))
    return [p for p in paths if stage.GetObjectAtPath(p)] if stage is not None else paths


def _compose(first,second):
    """Retain ordered substitutions instead of enumerating unchanged prims."""
    steps=[second] if isinstance(second,dict) else second
    return first+[step for step in steps if step]


def _new_run(workspace):
    root=workspace/'reconstruction';root.mkdir(exist_ok=True)
    index=1
    while (root/f'run-{index:04d}').exists():index+=1
    result=root/f'run-{index:04d}';result.mkdir()
    return result


def _require_single(path,mapping,stage,label):
    candidates=_mapped(path,mapping,stage)
    if len(candidates)!=1:raise ValueError(label+' requires one current prim: '+path)
    return candidates[0]


def _regroup_owner(operation, mapping, stage):
    from pxr import Sdf
    if not operation.get('create_owner', False):
        return _require_single(operation['owner_prim_path'], mapping, stage, 'Regroup owner')
    path = Sdf.Path(operation['owner_prim_path'])
    parent = _require_single(str(path.GetParentPath()), mapping, stage, 'New regroup owner parent')
    return str(Sdf.Path(parent).AppendChild(path.name))


def _request_batches(pending, requested):
    """Combine adjacent independent regroup requests without reordering work."""
    from pxr import Sdf
    group=[]; touched=[]
    for index,row in enumerate(pending):
        if row['id'] not in requested:continue
        operation=row['request']; is_regroup=operation['action']=='regroup'
        paths=[Sdf.Path(p) for p in row['source_prim_paths']+[operation['owner_prim_path']]] if is_regroup else []
        overlap=any(a.HasPrefix(b) or b.HasPrefix(a) for a in paths for b in touched)
        if group and (not is_regroup or overlap):
            yield group; group=[]; touched=[]
        if not is_regroup:
            yield [(index,row)]; continue
        group.append((index,row));touched.extend(paths)
    if group:yield group


def _regroup_batch_memberships(rows, mapping, stage):
    memberships={};created=[]
    for row in rows:
        operation=row['request'];owner=_regroup_owner(operation,mapping,stage)
        if operation.get('create_owner',False):created.append(owner)
        paths=sorted({p for source in row['source_prim_paths'] for p in _mapped(source,mapping,stage)})
        if not paths:raise ValueError('Reviewed source scope no longer resolves')
        if set(paths)&set(memberships):raise ValueError('Batch source scopes overlap')
        memberships.update({p:owner for p in paths})
    return memberships,sorted(set(created))


def _transfer_intent(manifest,state,mapping,stage,changed_sources):
    """Preserve unaffected judgments; retain every invalidated intent for review."""
    from pxr import Sdf
    boundaries=[];inherited=[];invalidated=[];reviews=[];unresolved=[]
    adapter=_adapter();mapped_evidence={}
    state.setdefault('selection_evidence',{}).update(manifest.get('selection_evidence',{}))
    for original in manifest.get('boundary_decisions',[]):
        choice=copy.deepcopy(original);path=choice['path'];destinations=_mapped(path,mapping,stage)
        affected=any(Sdf.Path(path).HasPrefix(Sdf.Path(p)) or Sdf.Path(p).HasPrefix(Sdf.Path(path)) for p in changed_sources)
        basis=choice.get('selection',{}).get('basis')
        reason=None
        if len(destinations)!=1:reason='The former prim now maps to zero or multiple prims.'
        elif affected and basis!='user_specified':reason='Reconstruction changed the evidence supporting this judgment.'
        selection=choice.get('selection',{})
        reference=selection.get('evidence_reference')
        key=reference or tuple(selection.get('evidence',[]))
        if key not in mapped_evidence:
            evidence=[];missing=False
            for p in adapter.selection_evidence(selection,manifest):
                mapped=_mapped(p,mapping,stage);evidence.extend(mapped)
                if not mapped:missing=True
            evidence=sorted(set(evidence))
            updated_reference=adapter._digest(evidence) if reference else None
            if updated_reference:state["selection_evidence"][updated_reference]=evidence
            mapped_evidence[key]=(evidence,missing,updated_reference)
        evidence,missing,updated_reference=mapped_evidence[key]
        if missing:reason='A cited evidence property or prim no longer exists.'
        if reason:
            invalidated.append({'decision':original,'mapped_prim_paths':destinations,'reason':reason})
            if destinations:reviews.append({'paths':destinations,'prior_decision':original,'reason':reason})
            else:unresolved.append({'type':'decision_target_removed','decision':original,'reason':reason})
            continue
        choice['path']=destinations[0]
        if 'selection' in choice:
            if reference:
                choice['selection']['evidence_reference']=updated_reference
                choice['selection']['evidence']=evidence[:1]
            else:choice['selection']['evidence']=evidence
        removed=choice.pop('structural_type',None) if affected else None
        boundaries.append(choice)
        inherited.append({'source_prim_path':path,'prim_path':choice['path'],'decision':choice,
                          'structural_override_reassessed':removed})
    protected=[]
    for path in state['request'].get('protected_paths',[]):
        mapped=_mapped(path,mapping,stage);protected.extend(mapped)
        if not mapped:unresolved.append({'type':'protected_interface_removed','path':path,'reason':'Required interface has no current prim.'})
    required_edits=[]
    previous_edits=list(state.get('required_occurrence_edits',[]))
    previous_edits.extend(edit for d in state.get('accepted_decisions',[]) for edit in d['required_occurrence_edits'])
    for edit in previous_edits:
        targets=_mapped(edit['path'],mapping,stage)
        if len(targets)==1:required_edits.append({**edit,'path':targets[0]})
        else:
            unresolved.append({'type':'occurrence_edit_needs_remapping','requirement':edit,'mapped_paths':targets,
                               'reason':'Required edit needs one unambiguous existing target.'})
    return boundaries,inherited,invalidated,reviews,unresolved,sorted(set(protected)),required_edits


def reconstruct(workspace,request_ids=None):
    """Consume selected staged requests; return the next assessment workspace.

    An unchanged split, unsupported operation, or failed operation stays
    pending. Successful steps are retained even when another step cannot finish.
    """
    from pxr import Usd,UsdGeom
    adapter=_adapter();tuning,discovery,reasoning=adapter._tools()
    workspace=Path(workspace).resolve()
    state=adapter._read(workspace/'assessment-state.json')
    if state.get('superseded_by'):raise ValueError('Continue reconstruction in '+state['superseded_by'])
    adapter._dependencies_current(state['source_context'])
    if state['assessment_policy_sha256']!=tuning.assessment_policy_fingerprint():raise ValueError('Assessment implementation changed; inspect again')
    manifest=adapter._read(workspace/'manifest.json')
    pending=state['reconstruction_requests']
    requested=set(request_ids or [row['id'] for row in pending])
    if not requested or requested-{row['id'] for row in pending}:raise ValueError('Select existing pending reconstruction request IDs')
    if manifest['context'].get('load','all')!='all':raise ValueError('Native reconstruction requires fully loaded selected composition')
    initial=discovery._open_stage(state['source'],state['request'])
    if reasoning.decision_source_context(initial,state['source_context']['dependency_fingerprints'])!=state['source_context']:
        raise ValueError('Source composition context changed')
    run=_new_run(workspace)
    current=Path(state['source'])
    material_tools=tuning.helper('reconstruction_material_publication','usd-optimize-run-operations/scripts/material_publication.py')
    with Usd.EditContext(initial,initial.GetSessionLayer()):
        material_repairs=material_tools.repair_shader_targets(initial)
    # Resolve diagnosed source namespace defects before flattening can discard
    # their out-of-reference connection opinions. Input layers stay untouched.
    if state['request'].get('variants') or material_repairs['repaired']:
        current=run/'selected-input.usdc';initial.Flatten().Export(str(current))
    tuning.write(run/'material-target-repairs.json',material_repairs)
    mapping=[]
    original_source=copy.deepcopy(manifest.get('original_source') or
                   {'path':manifest['source']['path'],'fingerprint':manifest['source']['fingerprint'],
                    'dependencies':manifest['inventory']['dependencies'],'context':manifest['context']})
    receipts=[];attempts=[];completed=set();changed_sources=[];created_owners=[]
    if material_repairs['repaired']:
        receipts.append({'request_id':'repair-material-targets',
            'request':{'action':'repair_material_targets'},'source_context':state['source_context'],
            'original_source':original_source,'source_layer':state['source'],'output_layer':str(current),
            'input_sha256':adapter._file_hash(state['source']),'output_sha256':adapter._file_hash(current),
            'source_prim_paths':[],'changed_path_map':{},
            'material_target_repairs':material_repairs,'accepted':True,'executed':True})

    splitter=tuning.helper('successor_native_split','usd-optimize-run-operations/scripts/native_reconstruction.py')
    regrouper=tuning.helper('successor_native_regroup','usd-optimize-run-operations/scripts/prim_regroup.py')
    for batch in _request_batches(pending,requested):
        index,row=batch[0]
        batch_rows=[item for _,item in batch]
        before=Usd.Stage.Open(str(current));output=run/f'step-{index+1:04d}.usdc'
        paths=sorted({p for source in row['source_prim_paths'] for p in _mapped(source,mapping,before)})
        operation=row['request'];action=operation['action']
        if len(batch)>1:
            operation={'action':'regroup','requests':copy.deepcopy(batch_rows)}
        receipt={'request_id':row['id'],'request':copy.deepcopy(operation),'source_context':state['source_context'],
                 'original_source':original_source,'source_prim_paths':paths,'executed':False}
        try:
            if not paths:raise ValueError('Reviewed source scope no longer resolves')
            if action=='split':
                parameters=operation['parameters']
                supported={'originalGeomOption':1,'mergePoint':9,'spatialMode':0,'considerAllAttributes':True}
                if set(parameters)-set(supported)-{'splitOn'} or any(k in parameters and parameters[k]!=v for k,v in supported.items()):
                    raise ValueError('Requested split parameters are not supported by the verified native adapter')
                mode=parameters.get('splitOn',0)
                if type(mode) is not int or mode not in (0,1):raise ValueError('splitOn must select vertices (0) or GeomSubsets (1)')
                meshes=sorted({str(p.GetPath()) for root in paths for p in Usd.PrimRange(before.GetPrimAtPath(root)) if p.IsA(UsdGeom.Mesh)})
                if not meshes:raise ValueError('Reviewed split scope contains no meshes')
                result=splitter.split(current,output,meshes,split_on='vertices' if mode==0 else 'geom_subsets')
                successful=result['status']=='split' and all(result.get('verification',{}).values())
            elif action=='regroup':
                if len(batch)>1:
                    memberships,owners=_regroup_batch_memberships(batch_rows,mapping,before)
                    paths=sorted(memberships);receipt['source_prim_paths']=paths
                    result=regrouper.regroup(current,output,memberships,create_owners=owners)
                else:
                    owner=_regroup_owner(operation,mapping,before)
                    result=regrouper.regroup(current,output,{p:owner for p in paths},create_owners=[owner] if operation.get('create_owner',False) else [])
                successful=result['status']=='regrouped'
            elif action=='keep':
                output=current
                result={'status':'kept','source_layer':str(current),'output_layer':str(current),'path_map':{}}
                successful=True
            else:raise ValueError('Unknown reconstruction action')
            receipt.update(result)
            receipt.update(input_sha256=adapter._file_hash(current),output_sha256=adapter._file_hash(output))
            receipt['executed']=bool(result.get('operations')) or action in ('regroup','keep')
            receipt['accepted']=successful
            if successful:
                after=Usd.Stage.Open(str(output))
                changed={p:targets if isinstance(targets,list) else [targets]
                         for p,targets in result.get('root_path_map',result.get('path_map',{})).items()}
                receipt.pop('path_map',None)
                receipt['changed_path_map']=changed
                created_owners=[target for owner in created_owners for target in _mapped(owner,changed,after)]
                created_owners.extend(result.get('created_owner_paths',[]))
                mapping=_compose(mapping,changed)
                current=output;completed.update(item['id'] for item in batch_rows);receipts.append(receipt)
                changed_sources.extend(p for item in batch_rows for p in item['source_prim_paths'] if action!='keep')
            elif result['status']=='unchanged':receipt['reason']='Native split preserved the input but produced no separate pieces; membership remains unresolved.'
        except Exception as error:
            receipt.update(status='unsupported_or_failed',reason=str(error),accepted=False)
        if len(batch)>1:
            receipt['request_outcomes']=[{'request_id':item['id'],'request':copy.deepcopy(item['request']),
                'source_prim_paths':item['source_prim_paths'],'accepted':receipt['accepted'],
                'executed':receipt['executed'],'status':receipt['status']} for item in batch_rows]
        attempts.append(receipt)
        tuning.write(run/f'request-{index+1:04d}.json',receipt)
        for other_index,item in batch[1:]:
            outcome=next(r for r in receipt['request_outcomes'] if r['request_id']==item['id'])
            tuning.write(run/f'request-{other_index+1:04d}.json',{**outcome,'batch_receipt':str(run/f'request-{index+1:04d}.json')})
    tuning.write(run/'attempts.json',{'requests':attempts})
    if not completed:
        state.setdefault('reconstruction_attempts',[]).append(str(run/'attempts.json'))
        result=adapter._publish(workspace,state,manifest,tuning)
        result['attempts']=attempts;result['advanced']=False
        return result
    after=Usd.Stage.Open(str(current))
    boundaries,inherited,invalidated,reviews,unresolved,protected,edits=_transfer_intent(manifest,state,mapping,after,changed_sources)
    reviews.extend({'paths':[owner],'reason':'Classify the explicitly created equipment owner after regrouping.'} for owner in sorted(set(created_owners)))
    policy=copy.deepcopy(state['request']);policy['variants']={};policy['protected_paths']=protected
    following=run/'assessment'
    adapter.inspect(current,following,policy)
    new_manifest=tuning.assessment(current,following,request=policy,decisions={'boundaries':boundaries},force_rescan=True)
    new_state=adapter._read(following/'assessment-state.json')
    new_context=new_state['source_context']
    remaining=[]
    for row in pending:
        if row['id'] in completed:continue
        record=copy.deepcopy(row)
        record['source_prim_paths']=sorted({p for source in row['source_prim_paths'] for p in _mapped(source,mapping,after)})
        if record['request']['action']=='regroup':
            try:record['request']['owner_prim_path']=_regroup_owner(record['request'],mapping,after)
            except ValueError as error:unresolved.append({'type':'pending_reconstruction_owner','request':row,'reason':str(error)})
        if not record['source_prim_paths']:
            unresolved.append({'type':'pending_reconstruction_scope','request':row,'reason':'No current source scope remains'})
        record['source_context']=new_context;remaining.append(record)
    new_state.update(original_source=original_source,
                     selection_evidence=state.get('selection_evidence',{}),
                     reconstruction_material_target_repairs=copy.deepcopy(manifest.get('reconstruction_material_target_repairs',[]))+[material_repairs],
                     reconstruction_results=copy.deepcopy(manifest.get('reconstruction_results',[]))+receipts,
                     reconstruction_path_map=_compose(manifest['reconstruction_path_map'],mapping) if 'reconstruction_path_map' in manifest else mapping,
                     inherited_decisions=copy.deepcopy(manifest.get('inherited_decisions',[]))+inherited,
                     invalidated_decisions=copy.deepcopy(manifest.get('invalidated_decisions',[]))+invalidated,
                     required_reviews=reviews,unresolved_requirements=copy.deepcopy(state.get('unresolved_requirements',[]))+unresolved,
                     required_occurrence_edits=edits,reconstruction_requests=remaining,
                     reconstruction_attempts=[str(run/'attempts.json')],previous_workspace=str(workspace))
    result=adapter._publish(following,new_state,new_manifest,tuning)
    state['superseded_by']=str(following);state['status']='superseded'
    tuning.write(workspace/'assessment-state.json',state)
    result.update(advanced=True,attempts=attempts,original_source=original_source)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--workspace',required=True);parser.add_argument('--request-id',action='append')
    args=parser.parse_args();print(json.dumps(reconstruct(args.workspace,args.request_id),indent=2,allow_nan=False))


if __name__=='__main__':main()
