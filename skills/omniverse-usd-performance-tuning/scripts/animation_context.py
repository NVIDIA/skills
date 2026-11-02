# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Selected-composition animation evidence and explicit geometry capabilities.

Call during detailed assessment or verification, never the topology-only boundary
review. Authored values are hashed; the manifest does not duplicate animation arrays.

Usage:
    Import collect(), verify() and schema_availability() from assessment or
    composition verification. This module has no command-line interface.
Arguments:
    Open Usd.Stage objects; optional prim paths, path mappings and requested times.
Output:
    Dictionaries containing authored animation evidence or verification findings.
    No animation arrays are copied into the manifest.
Exit codes:
    Not applicable to this imported module. Callers inspect returned findings;
    USD and data-access exceptions propagate to the caller.
"""
import hashlib
import math


def _digest(value):
    from pxr import Sdf
    import numpy as np
    if isinstance(value,Sdf.AssetPath):
        return hashlib.sha256((value.resolvedPath or value.path).encode()).hexdigest()
    if value is None:return hashlib.sha256(b'None').hexdigest()
    array=np.asarray(value)
    if array.dtype.kind in 'biufc' and array.dtype!=object:
        data=(str(array.dtype)+str(array.shape)).encode()+array.tobytes()
    else:
        data=repr(value).encode()
    return hashlib.sha256(data).hexdigest()


def _prims(stage,paths):
    from pxr import Sdf,Usd,UsdGeom,UsdShade
    if paths is None:
        return list(Usd.PrimRange.Stage(stage,Usd.TraverseInstanceProxies()))
    found={}
    for path in paths:
        root=stage.GetPrimAtPath(Sdf.Path(path).GetPrimPath())
        if not root:continue
        for prim in Usd.PrimRange(root,Usd.TraverseInstanceProxies()):found[str(prim.GetPath())]=prim
        while root and not root.IsPseudoRoot():
            found[str(root.GetPath())]=root;root=root.GetParent()
    # Bound networks contribute time samples even when the mesh itself is static.
    materials=[]
    for prim in list(found.values()):
        if prim.IsA(UsdGeom.Gprim) or prim.IsA(UsdGeom.Subset):
            material,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            if material:materials.append(material.GetPrim())
    for material in materials:
        for prim in Usd.PrimRange(material,Usd.TraverseInstanceProxies()):found[str(prim.GetPath())]=prim
    pending=list(found.values())
    while pending:
        prim=pending.pop()
        for attr in prim.GetAttributes():
            for path in attr.GetConnections():
                target=stage.GetPrimAtPath(path.GetPrimPath())
                if target and str(target.GetPath()) not in found:
                    found[str(target.GetPath())]=target;pending.append(target)
    return list(found.values())


def collect(stage,paths=None,requested_times=()):
    """Collect every resolved authored sample and interval midpoint in scope."""
    from pxr import Usd
    times=set();attributes={}
    for value in requested_times or ():
        if value is not None:
            value=float(value)
            if not math.isfinite(value):raise ValueError('Animation sample times must be finite')
            times.add(value)
    for prim in _prims(stage,paths):
        sampled=[(attr,attr.GetTimeSamples()) for attr in prim.GetAttributes()]
        animated_transform=any(attr.GetName().startswith('xformOp:') and samples for attr,samples in sampled)
        for attr,authored in sampled:
            times.update(authored)
            if authored or (animated_transform and attr.GetName()=='xformOpOrder' and attr.HasAuthoredValueOpinion()):
                attributes[str(attr.GetPath())]={'type':str(attr.GetTypeName()),'time_samples':authored,
                    'default':_digest(attr.Get(Usd.TimeCode.Default())),
                    'values':{str(t):_digest(attr.Get(t)) for t in authored},
                    'interpolation':attr.GetMetadata('interpolation'),
                    'connections':list(map(str,attr.GetConnections()))}
    authored=sorted(times)
    checks=sorted(times|{(a+b)/2 for a,b in zip(authored,authored[1:])})
    return {'authored_times':authored,'time_samples':[None]+checks,'attributes':attributes,
            'stage':{'startTimeCode':stage.GetStartTimeCode(),'endTimeCode':stage.GetEndTimeCode(),
                     'timeCodesPerSecond':stage.GetTimeCodesPerSecond(),'framesPerSecond':stage.GetFramesPerSecond(),
                     'interpolation':str(stage.GetInterpolationType())},
            'sampling':'All selected-composition authored times and interval midpoints; samples alone are not a proof between times.'}


def verify(source,output,paths=None,path_map=None,requested_times=()):
    """Require authored animated values/curves and transform order to survive."""
    from pxr import Sdf
    mapping=path_map or {}
    before=collect(source,paths,requested_times)
    after=collect(output,requested_times=requested_times)
    checks=[]
    for path,expected in before['attributes'].items():
        candidate=Sdf.Path(path)
        destinations = [candidate.ReplacePrefix(Sdf.Path(old),Sdf.Path(new))
                         for old,new in sorted(mapping.items(),key=lambda row:len(row[0]),reverse=True)
                         if candidate.HasPrefix(Sdf.Path(old))]
        destinations.append(candidate)
        actual=None;destination=None
        for possible in destinations:
            attr=output.GetAttributeAtPath(possible)
            if attr:
                destination=str(possible)
                actual=after['attributes'].get(destination)
                break
        if actual is not None:
            actual=dict(actual)
            # Connections are checked independently in the composed consumer.
            actual.pop('connections',None)
        expected=dict(expected);expected.pop('connections',None)
        checks.append({'property_path':path,'output_property_path':destination,'passed':expected==actual})
    checks.append({'name':'authored animation stage metadata','passed':before['stage']==after['stage']})
    times=set(before['time_samples']+after['time_samples'])
    return {'passed':all(row['passed'] for row in checks),'checks':checks,
            'time_samples':[None]+sorted(t for t in times if t is not None),
            'authored_attribute_count':len(before['attributes']),'sampling':before['sampling']}


def schema_availability(stage):
    """Require loaded authored schemas, without a geometry-proof whitelist."""
    from pxr import Usd
    findings=[{'path':str(prim.GetPath()),'schema':prim.GetTypeName(),'reason':'Authored schema is unavailable'}
              for prim in Usd.PrimRange.Stage(stage,Usd.TraverseInstanceProxies())
              if prim.GetTypeName() and Usd.SchemaRegistry.GetTypeFromSchemaTypeName(prim.GetTypeName()).isUnknown]
    return {'passed':not findings,'findings':findings}
