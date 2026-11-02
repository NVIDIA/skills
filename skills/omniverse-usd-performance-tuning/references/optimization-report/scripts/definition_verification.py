# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime UV-consumer checks and immutable evidence fingerprints."""
import hashlib
from pathlib import Path
from pxr import Usd, UsdGeom, UsdShade


def unused_uv_evidence(prim, name, time=Usd.TimeCode.Default(), cache=None):
    """Prove a texcoord has no known USD consumer; unknown shader semantics block.

    This deliberately does not infer anything from the optimizer's own result.
    A broken graph, opaque shader, dynamic primvar name, or property target is
    insufficient evidence to discard source-authored data.
    """
    cache={} if cache is None else cache
    var=UsdGeom.PrimvarsAPI(prim).FindPrimvarWithInheritance(name)
    if not var or str(var.GetTypeName()) not in ('texCoord2f[]','texCoord2d[]','texCoord2h[]'):
        return {'unused':False,'reason':'Deleted value is not a typed texture-coordinate primvar.'}
    material,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    material_prim=material.GetPrim() if material else None
    identity=(material_prim.GetPrimInPrototype() if material_prim and material_prim.IsInstanceProxy() else material_prim)
    key=('shader_consumers',str(identity.GetPath()) if identity else '',str(time))
    if key not in cache:
        used=set();unknown=[];seen=set()
        def visit(node):
            if str(node.GetPath()) in seen:return
            seen.add(str(node.GetPath()))
            if node.IsA(UsdShade.Shader):
                shader_id=UsdShade.Shader(node).GetIdAttr().Get(time)
                if not shader_id or not (shader_id in ('UsdPreviewSurface','UsdUVTexture','UsdTransform2d') or str(shader_id).startswith('UsdPrimvarReader_')):
                    unknown.append('opaque shader '+str(node.GetPath()))
                if str(shader_id).startswith('UsdPrimvarReader_'):
                    attr=node.GetAttribute('inputs:varname')
                    if not attr or attr.GetConnections() or not isinstance(attr.Get(time),str):
                        unknown.append('dynamic primvar reader '+str(node.GetPath()))
                    else:used.add(attr.Get(time))
            for attr in node.GetAttributes():
                if not attr.GetName().startswith(('inputs:','outputs:')):continue
                for connection in attr.GetConnections():
                    target=prim.GetStage().GetObjectAtPath(connection)
                    if not target:
                        unknown.append('unresolved shader connection '+str(connection))
                    else:visit(target.GetPrim())
        if material:visit(material_prim)
        cache[key]=(used,unknown)
    used,unknown=cache[key]
    if unknown:return {'unused':False,'reason':'Shader consumers could not be established.','unknown':unknown}
    if name in used:return {'unused':False,'reason':'Material graph reads this primvar.'}
    # Check explicit USD consumers once per source stage, including non-shader
    # relationships and attribute connections outside the bound material.
    if 'property_consumers' not in cache:
        targets=set()
        for node in Usd.PrimRange.Stage(prim.GetStage(),Usd.TraverseInstanceProxies()):
            for rel in node.GetAuthoredRelationships():targets.update(map(str,rel.GetTargets()))
            for attr in node.GetAuthoredAttributes():targets.update(map(str,attr.GetConnections()))
        cache['property_consumers']=targets
    paths={str(var.GetAttr().GetPath()),str(var.GetIndicesAttr().GetPath())}
    if paths & cache['property_consumers']:
        return {'unused':False,'reason':'A USD property explicitly targets this primvar.'}
    # Custom source metadata may declare consumers outside standard shader APIs.
    node=prim
    while node and not node.IsPseudoRoot():
        for attr in node.GetAuthoredAttributes():
            if attr.IsCustom() and str(attr.Get(time)) in (name,'primvars:'+name):
                return {'unused':False,'reason':'Custom metadata names this primvar.'}
        def names_primvar(value):
            if isinstance(value, dict):
                return any(names_primvar(key) or names_primvar(item) for key, item in value.items())
            if isinstance(value, str):
                return value in (name, 'primvars:'+name) or value in paths
            if isinstance(value, (list, tuple)):
                return any(names_primvar(item) for item in value)
            return False
        if names_primvar(node.GetCustomData()):
            return {'unused':False,'reason':'Custom data may consume this primvar.'}
        node=node.GetParent()
    return {'unused':True,'reason':'No bound shader reader or explicit USD property consumes this texcoord.',
            'material':str(material.GetPath()) if material else None,'primvar':name}



def unused_uv_removal_check(stage, paths, time_samples=None, cache=None):
    """Read-only operation preflight shared with the worker; uncertain means skip."""
    cache={} if cache is None else cache
    evidence=[]
    for path in paths:
        prim=stage.GetPrimAtPath(path)
        if not prim:return {'allowed':False,'reason':'Missing source geometry.','evidence':{'path':path}}
        for var in UsdGeom.PrimvarsAPI(prim).FindPrimvarsWithInheritance():
            if not var.HasAuthoredValue() or str(var.GetTypeName()) not in ('texCoord2f[]','texCoord2d[]','texCoord2h[]'):continue
            for sample in [None]+[v for v in (time_samples or []) if v is not None]:
                time=Usd.TimeCode.Default() if sample is None else Usd.TimeCode(sample)
                proof=unused_uv_evidence(prim,var.GetPrimvarName(),time,cache)
                evidence.append({'path':path,'time':sample,**proof})
                if not proof['unused']:
                    return {'allowed':False,'reason':proof['reason'],'evidence':evidence}
    return {'allowed':True,'reason':'Every authored texcoord was independently checked for consumers.','evidence':evidence}



def _stat_key(stat):
    return (stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)



def _file_fingerprint(path, cache=None, metrics=None):
    path=str(Path(path).resolve())
    stat=Path(path).stat()
    key=(path,*_stat_key(stat))
    if cache is not None and key in cache:
        if metrics is not None:
            metrics['file_hash_hits']=metrics.get('file_hash_hits',0)+1
        return cache[key]
    with open(path,"rb") as stream:
        result=hashlib.file_digest(stream,"sha256").hexdigest()
    if _stat_key(stat)!=_stat_key(Path(path).stat()):
        raise OSError("file changed while fingerprinting: "+path)
    if metrics is not None:
        metrics['file_hash_reads']=metrics.get('file_hash_reads',0)+1
    if cache is not None:
        if len(cache)>=4096:
            cache.clear()
        cache[key]=result
    return result



def _evidence_files(*stages, resources=(), cache=None, metrics=None):
    files={str(Path(path).resolve()) for path in resources}
    for stage in stages:
        for layer in stage.GetUsedLayers():
            if layer.realPath:
                files.add(str(Path(layer.realPath).resolve()))
    return {path:_file_fingerprint(path,cache,metrics) for path in sorted(files)}
