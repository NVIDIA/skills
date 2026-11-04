# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Selective shared material composition and conservative source cleanup."""
import hashlib
import re
from pxr import Sdf, Usd, UsdShade


def repair_shader_targets(stage):
    """Repair a stale namespace only when the owning material identifies one output.

    The source is opened with a disposable edit target by the caller. Never infer
    a material from proximity, color or a shader name elsewhere in the scene.
    """
    repaired=[]; unresolved=[]
    for prim in stage.Traverse():
        if not prim.IsA(UsdShade.Material):
            continue
        for name in ('outputs:surface', 'outputs:displacement', 'outputs:volume'):
            attr=prim.GetAttribute(name)
            if not attr:
                continue
            targets=attr.GetConnections()
            if not targets:
                # USD suppresses a target that escapes a referenced material's
                # source root. Read its authored single-target opinion without
                # treating an intentional stronger disconnect as a defect.
                opinions=[spec.GetInfo('connectionPaths') for spec in attr.GetPropertyStack()
                          if spec.HasInfo('connectionPaths')]
                if opinions and (len(opinions)==1 or opinions[0].isExplicit):
                    authored=list(opinions[0].GetAddedOrExplicitItems())
                    if len(authored)==1:
                        targets=authored
            replacements=[]
            for target in targets:
                replacement=target
                if not stage.GetObjectAtPath(target):
                    marker='/'+prim.GetName()+'/'
                    text=str(target)
                    candidates=[]
                    if text.count(marker)==1:
                        suffix=text.split(marker,1)[1]
                        candidate=prim.GetPath().AppendPath(suffix)
                        obj=stage.GetAttributeAtPath(candidate)
                        if (obj and obj.GetPrim().IsA(UsdShade.Shader)
                                and obj.GetName()==target.name
                                and obj.GetTypeName()==attr.GetTypeName()):
                            candidates.append(candidate)
                    if len(candidates)==1:
                        replacement=candidates[0]
                        repaired.append({'property_path':str(attr.GetPath()),
                                         'old_target':text,'target':str(replacement)})
                    else:
                        unresolved.append({'property_path':str(attr.GetPath()),'target':text})
                replacements.append(replacement)
            if replacements!=targets:
                attr.SetConnections(replacements)
    return {'repaired':repaired,'unresolved':unresolved}


class MaterialMounts:
    """Reference only used networks, with external shader connections kept in scope."""
    def __init__(self, stage, library_file, library_stage=None):
        self.stage=stage
        self.library_file=str(library_file)
        self.library=library_stage or Usd.Stage.Open(self.library_file)
        self.mounted={}

    def mount(self, path):
        root=Sdf.Path(path)
        if root in self.mounted:
            return self.mounted[root]
        source=self.library.GetPrimAtPath(root)
        if not source:
            raise ValueError('Missing material network: '+str(root))
        local=Sdf.Path('/Asset/Looks/m_'+hashlib.sha256(str(root).encode()).hexdigest()[:20])
        self.mounted[root]=local
        self.stage.DefinePrim('/Asset/Looks','Scope')
        site=self.stage.DefinePrim(local)
        site.GetReferences().AddReference(self.library_file,root)
        # Most materials are closed networks and can share one instance. External
        # shader/nodegraph connections need explicit mounts and local overrides.
        external=False
        for prim in Usd.PrimRange(source):
            for attr in prim.GetAuthoredAttributes():
                targets=attr.GetConnections()
                if not targets or all(t.HasPrefix(root) for t in targets):
                    continue
                mapped=[]
                for target in targets:
                    if target.HasPrefix(root):
                        mapped.append(target.ReplacePrefix(root,local)); continue
                    obj=self.library.GetObjectAtPath(target)
                    if not obj:
                        # Preserve diagnosed unresolved paths as authored opinions.
                        mapped.append(target); continue
                    dependency=obj.GetPrim()
                    while dependency.GetParent() and (dependency.GetParent().IsA(UsdShade.Material)
                            or dependency.GetParent().IsA(UsdShade.NodeGraph)):
                        dependency=dependency.GetParent()
                    mounted=self.mount(str(dependency.GetPath()))
                    mapped.append(target.ReplacePrefix(dependency.GetPath(),mounted))
                target_attr=attr.GetPath().ReplacePrefix(root,local)
                self.stage.GetAttributeAtPath(target_attr).SetConnections(mapped)
                external=True
        if not external:
            site.SetInstanceable(True)
        return local


def prune_unused_abstract_roots(stage, candidate_paths=None):
    """Drop unreachable abstract roots, optionally limited to specified paths."""
    allowed = None if candidate_paths is None else {Sdf.Path(path) for path in candidate_paths}
    layer=stage.GetRootLayer()
    candidates=[]
    def collect(path):
        spec=layer.GetObjectAtPath(path)
        if not isinstance(spec,Sdf.PrimSpec):
            return
        # Flatten emits bare root overs for shared prototypes. They cease to
        # contribute once every occurrence has been replaced by asset references.
        # Ordinary overs, definitions and roots with additional metadata stay.
        flattened_prototype = (path.IsRootPrimPath()
            and spec.specifier == Sdf.SpecifierOver
            and re.fullmatch(r'Flattened_Prototype_[0-9]+', spec.name)
            and set(spec.ListInfoKeys()) == {'specifier'})
        if (spec.specifier==Sdf.SpecifierClass or flattened_prototype) and (allowed is None or path in allowed):
            candidates.append(path)
    layer.Traverse(Sdf.Path.absoluteRootPath,collect)
    roots=set()
    for path in sorted(candidates,key=lambda p:len(str(p))):
        if not any(prefix in roots for prefix in path.GetPrefixes()): roots.add(path)
    if not roots:
        return []
    edges={root:set() for root in roots}; live=set()
    def owner(path):
        return next((prefix for prefix in reversed(path.GetPrimPath().GetPrefixes()) if prefix in roots),None)
    def visit(path):
        spec=layer.GetObjectAtPath(path)
        if not spec:
            return
        origin=owner(path)
        for key in ('targetPaths','connectionPaths','inheritPaths','specializes','references','payload'):
            if not spec.HasInfo(key):
                continue
            for item in spec.GetInfo(key).GetAddedOrExplicitItems():
                if key in ('references','payload'):
                    if item.assetPath:
                        continue
                    item=item.primPath
                if item.isEmpty:
                    continue
                if not item.IsAbsolutePath():
                    item=item.MakeAbsolutePath(path.GetPrimPath())
                target=owner(item)
                if target is not None:
                    if origin is None: live.add(target)
                    else: edges[origin].add(target)
    layer.Traverse(Sdf.Path.absoluteRootPath,visit)
    pending=list(live)
    while pending:
        for target in edges[pending.pop()]-live:
            live.add(target);pending.append(target)
    removed=sorted(roots-live)
    for path in removed:
        stage.RemovePrim(path)
    return list(map(str,removed))


def prune_unused_material_networks(materials, source, material_paths):
    """Keep bound networks and their shader dependencies in the shared library."""
    roots=set()
    def owner(path):
        for prefix in reversed(path.GetPrimPath().GetPrefixes()):
            if prefix in roots: return prefix
        return None
    for prim in materials.Traverse():
        if (prim.IsA(UsdShade.Material) or prim.IsA(UsdShade.Shader) or prim.IsA(UsdShade.NodeGraph)) and owner(prim.GetPath()) is None:
            roots.add(prim.GetPath())
    live=set()
    for prim in source.Traverse():
        if prim.IsA(UsdShade.Material) or prim.IsA(UsdShade.Shader) or prim.IsA(UsdShade.NodeGraph):
            continue
        for prop in prim.GetAuthoredProperties():
            targets=prop.GetTargets() if isinstance(prop,Usd.Relationship) else prop.GetConnections()
            for target in targets:
                mapped=Sdf.Path(material_paths.get(str(target),'/Materials'+str(target)))
                root=owner(mapped)
                if root: live.add(root)
    pending=list(live)
    while pending:
        for prim in Usd.PrimRange(materials.GetPrimAtPath(pending.pop())):
            for prop in prim.GetAuthoredProperties():
                targets=prop.GetTargets() if isinstance(prop,Usd.Relationship) else prop.GetConnections()
                for target in targets:
                    root=owner(target)
                    if root and root not in live:
                        live.add(root);pending.append(root)
    removed=sorted(roots-live)
    for path in removed: materials.RemovePrim(path)
    return list(map(str,removed))
