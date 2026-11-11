# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-writer native material deduplication with verified binding remapping."""
from pathlib import Path
import hashlib


def _fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _spec(stage,path,flat=None):
    from pxr import Sdf,Usd
    layer=Sdf.Layer.CreateAnonymous()
    if not Sdf.CopySpec(flat if flat is not None else stage.Flatten(),path,layer,'/Material'):
        raise ValueError('Cannot inspect material '+path)
    return layer.ExportToString()


def optimize(source_layer,output_layer):
    """Return a checked old-material -> retained-material map; source is immutable."""
    from pxr import Sdf,Usd,UsdShade
    from usd_optimize.core import ExecutionContext,UsdOptimizeCore
    source_layer=Path(source_layer).resolve();output_layer=Path(output_layer).resolve()
    if source_layer==output_layer or output_layer.exists():
        raise ValueError('Material optimization requires a new output layer')
    source=Usd.Stage.Open(str(source_layer))
    if not source or source.GetCompositionErrors():raise ValueError('Invalid material library')
    source_hash=_fingerprint(source_layer)
    paths=[str(p.GetPath()) for p in source.Traverse() if p.IsA(UsdShade.Material)]
    result={'operation':'optimizeMaterials','configuration':{'operation':'optimizeMaterials','materialPrimPaths':paths,
        'optimizeMaterialsMode':0,'materialsPath':'/Materials'},'source_layer':str(source_layer),'output_layer':str(output_layer),
        'input_sha256':source_hash,'verification_method':'native_execution_and_material_remapping',
        'independent_material_proof':False,'passed':False,'status':'incomplete','material_path_map':{},'checks':[]}
    flat=source.Flatten();stage=Usd.Stage.Open(flat)
    expected={path:_spec(stage,path,flat) for path in paths}
    groups={}
    for path,signature in expected.items():groups.setdefault(signature,[]).append(path)
    compatible=[group for group in groups.values() if len(group)>1]
    nominated=[path for group in compatible for path in group]
    result['configurations']=[dict(result['configuration'],materialPrimPaths=nominated)] if nominated else []
    result['nomination']={'compatible_groups':compatible,
        'distinct_materials':[group[0] for group in groups.values() if len(group)==1],
        'reason':'Only identical authored networks and required metadata may share; distinct material specifications are retained.'}
    try:
        context=ExecutionContext();context.set_stage(stage)
        outcomes=UsdOptimizeCore.getInstance().executeConfig(context,result['configurations']) if compatible else []
        if len(outcomes)!=len(result['configurations']) or any(not row[0] for row in outcomes):raise ValueError('Native material optimization failed: '+str(outcomes))
        result['native_result']=str(outcomes)
        # Mode 0 retains an existing member of each exact authored-network group.
        # Resolve only unambiguous surviving paths; do not invent a remap from
        # native success alone, or recreate geometry to observe native bindings.
        for group in groups.values():
            retained=[path for path in group if (prim:=stage.GetPrimAtPath(path)) and prim.IsA(UsdShade.Material)]
            missing=set(group)-set(retained)
            if missing and len(retained)!=1:
                raise ValueError('Native material remap has no unique surviving target: '+str(group))
            for path in group:
                result['material_path_map'][path]=retained[0] if path in missing else path
        result['checks'].append({'name':'complete material remapping',
            'passed':set(result['material_path_map'])==set(paths) and all(
                stage.GetPrimAtPath(path).IsA(UsdShade.Material)
                for path in set(result['material_path_map'].values()))})
        result['checks'].append({'name':'source unchanged','passed':_fingerprint(source_layer)==source_hash})
        result['checks'].append({'name':'composition','passed':not stage.GetCompositionErrors()})
        result['passed']=all(row['passed'] for row in result['checks'])
        unique=len(set(result['material_path_map'].values()))
        result['before']={'materials':len(paths)};result['after']={'materials':unique}
        result['status']='optimized' if result['passed'] and unique<len(paths) else 'verified_unchanged' if result['passed'] else 'rejected'
    except Exception as error:
        result['error']=str(error);result['status']='rejected'
    output_layer.parent.mkdir(parents=True,exist_ok=True)
    if result['passed']:
        stage.GetRootLayer().Export(str(output_layer))
    else:
        source.Flatten().Export(str(output_layer))
        result['material_path_map']={path:path for path in paths}
        result['rolled_back']=True
    result['output_sha256']=_fingerprint(output_layer)
    result['runtime']=list(Usd.GetVersion())
    return result
