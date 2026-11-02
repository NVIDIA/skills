#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Move caller-selected USD prims beneath owners while retaining placement.

Only the selected local layer stack is copied. Reference and payload dependencies
remain external; native namespace editing updates paths and authors relocates.
"""
from __future__ import annotations

import json
from pathlib import Path


def remap_path(path, root_path_map):
    """Apply a moved-root prefix to a prim or property path, returning Sdf.Path."""
    from pxr import Sdf
    path = Sdf.Path(path)
    for prefix in path.GetPrefixes():
        destination = root_path_map.get(str(prefix))
        if destination is not None:
            return path.ReplacePrefix(prefix, Sdf.Path(destination))
    return path


def _chain(prim):
    """Contributing transform ancestors in root-to-leaf order."""
    from pxr import UsdGeom
    chain = []
    while prim and not prim.IsPseudoRoot():
        xform = UsdGeom.Xformable(prim)
        if xform:
            chain.append(prim)
            if xform.GetResetXformStack():
                break
        prim = prim.GetParent()
    return list(reversed(chain))


def _ancestors(prim):
    while prim and not prim.IsPseudoRoot():
        yield prim
        prim = prim.GetParent()


def _copy_attribute(attribute, target):
    # Native property flattening resolves only this inherited property, including
    # layer offsets and authored samples; it does not flatten composed geometry.
    attribute.FlattenTo(target)
    # USD 25.11 may return a stale property handle on a referenced prim.
    if not target.GetAttribute(attribute.GetName()):
        raise RuntimeError(f'Copy inherited attribute {attribute.GetPath()} to {target.GetPath()} failed')


def _copy_transform(stage, source, path, inverse=False):
    from pxr import UsdGeom
    target = UsdGeom.Xform.Define(stage, path)
    order = [str(token) for token in UsdGeom.Xformable(source).GetXformOpOrderAttr().Get() or []
             if str(token) != '!resetXformStack!']
    for attribute in source.GetAttributes():
        if attribute.GetName().startswith('xformOp:'):
            _copy_attribute(attribute, target.GetPrim())
    if inverse:
        order = [token[len('!invert!'):] if token.startswith('!invert!') else '!invert!' + token
                 for token in reversed(order)]
    target.GetXformOpOrderAttr().Set(order)


def _check_destination(original, destination):
    """Reject inheritance that USD cannot cancel beneath the chosen owner."""
    from pxr import Usd, UsdGeom, UsdShade
    original_names = {str(var.GetPrimvarName()) for var in
                      UsdGeom.PrimvarsAPI(original).FindPrimvarsWithInheritance() if var.HasAuthoredValue()}
    destination_names = {str(var.GetPrimvarName()) for var in
                         UsdGeom.PrimvarsAPI(destination).FindPrimvarsWithInheritance() if var.HasAuthoredValue()}
    if destination_names - original_names:
        raise ValueError(f'Destination {destination.GetPath()} adds inherited primvars absent from {original.GetPath()}')
    for purpose in ('', 'preview', 'full'):
        original_material, _ = UsdShade.MaterialBindingAPI(original).ComputeBoundMaterial(purpose)
        destination_material, _ = UsdShade.MaterialBindingAPI(destination).ComputeBoundMaterial(purpose)
        if destination_material and not original_material:
            raise ValueError(f'Destination adds an inherited material absent from {original.GetPath()}')
    source_ancestors = {prim.GetPath() for prim in _ancestors(original.GetParent())}
    for ancestor in _ancestors(destination):
        if ancestor.GetPath() in source_ancestors:
            continue
        visibility = ancestor.GetAttribute('visibility')
        if visibility and visibility.HasAuthoredValueOpinion():
            if any(visibility.Get(time) == 'invisible' for time in [Usd.TimeCode.Default()] + visibility.GetTimeSamples()):
                raise ValueError(f'Destination visibility cannot be cancelled beneath {ancestor.GetPath()}')
        for relation in ancestor.GetRelationships():
            if relation.GetName().startswith('material:binding:collection:') and relation.HasAuthoredTargets():
                raise ValueError(f'Destination collection binding cannot be cancelled beneath {ancestor.GetPath()}')
            if (relation.GetName().startswith('material:binding') and relation.HasAuthoredTargets()
                    and UsdShade.MaterialBindingAPI.GetMaterialBindingStrength(relation) == 'strongerThanDescendants'):
                raise ValueError(f'Destination material binding cannot be cancelled beneath {ancestor.GetPath()}')


def _placement(stage, original, owner):
    from pxr import Gf, UsdGeom
    holder_path = owner.GetPath().AppendChild('__placement_' + original.GetName())
    holder = UsdGeom.Xform.Define(stage, holder_path)
    holder.GetPrim().SetCustomDataByKey('usdTuning:placement', True)
    parent_path = holder_path
    # Each visibility opinion keeps its own ancestor so USD combines animated
    # invisibility exactly as before the move.
    for index, ancestor in enumerate(_ancestors(original.GetParent())):
        attribute = ancestor.GetAttribute('visibility')
        if attribute and attribute.HasAuthoredValueOpinion():
            parent_path = parent_path.AppendChild('visibility' + str(index))
            target = UsdGeom.Xform.Define(stage, parent_path).GetPrim()
            _copy_attribute(attribute, target)
    xform = UsdGeom.Xformable(original)
    source_chain = [] if xform and xform.GetResetXformStack() else _chain(original.GetParent())
    owner_chain = _chain(owner)
    animated = any(attr.GetNumTimeSamples() for ancestor in source_chain + owner_chain
                   for attr in ancestor.GetAttributes() if attr.GetName().startswith('xformOp:'))
    if animated:
        for label, chain, inverse in [('destination', list(reversed(owner_chain)), True),
                                      ('source', source_chain, False)]:
            for index, ancestor in enumerate(chain):
                parent_path = parent_path.AppendChild(label + str(index))
                _copy_transform(stage, ancestor, parent_path, inverse)
    else:
        cache = UsdGeom.XformCache()
        source_world = cache.GetLocalToWorldTransform(original.GetParent()) if source_chain else Gf.Matrix4d(1)
        owner_world = cache.GetLocalToWorldTransform(owner)
        if owner_world.GetDeterminant() == 0:
            raise ValueError(f'Destination transform is singular: {owner.GetPath()}')
        UsdGeom.Xformable(stage.GetPrimAtPath(parent_path)).AddTransformOp().Set(source_world * owner_world.GetInverse())
    return parent_path.AppendChild(original.GetName())


def _inherit_state(original, moved):
    from pxr import UsdGeom, UsdShade
    for var in UsdGeom.PrimvarsAPI(original).FindPrimvarsWithInheritance():
        for attribute in (var.GetAttr(), var.GetIndicesAttr()):
            if attribute and attribute.HasAuthoredValueOpinion():
                _copy_attribute(attribute, moved)
    imageable = UsdGeom.Imageable(original)
    if imageable:
        UsdGeom.Imageable(moved).CreatePurposeAttr(imageable.ComputePurpose())
    _inherit_materials(original, moved.GetStage())
    if UsdGeom.Xformable(moved):
        UsdGeom.Xformable(moved).SetResetXformStack(False)


def _inherit_materials(original, stage):
    from pxr import Usd, UsdShade
    # Only bindings inherited across the removed ancestry need materializing.
    # Collection bindings can select descendants without binding the root.
    has_collections = any(relation.GetName().startswith('material:binding:collection:')
                          and relation.HasAuthoredTargets()
                          for ancestor in _ancestors(original.GetParent())
                          for relation in ancestor.GetRelationships())
    prims = Usd.PrimRange(original, Usd.TraverseInstanceProxies()) if has_collections else (original,)
    for prim in prims:
        for purpose in ('', 'preview', 'full'):
            material, relation = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(purpose)
            if not material or relation.GetPrim().GetPath().HasPrefix(original.GetPath()):
                continue
            target = stage.GetPrimAtPath(prim.GetPath())
            if target.IsInstanceProxy():
                root_material, _ = UsdShade.MaterialBindingAPI(original).ComputeBoundMaterial(purpose)
                if not root_material or root_material.GetPath() != material.GetPath():
                    raise ValueError(f'Inherited collection binding on instance proxy cannot be authored: {prim.GetPath()}')
                continue
            UsdShade.MaterialBindingAPI.Apply(target).Bind(
                UsdShade.Material(stage.GetPrimAtPath(material.GetPath())),
                UsdShade.MaterialBindingAPI.GetMaterialBindingStrength(relation), purpose)


def regroup(source_layer, output_layer, memberships, *, create_owners=()):
    """Move disjoint source roots to existing owners or explicitly created Xforms.

    Instance proxies and destination-only inherited primvars, invisibility or
    stronger material bindings are unsupported. Native namespace edit failures
    identify the moved root and never produce a successful result.
    """
    from pxr import Sdf, Tf, Usd, UsdGeom, UsdUtils
    source_layer, output_layer = Path(source_layer).resolve(), Path(output_layer).resolve()
    if not isinstance(memberships, dict) or not memberships:
        raise ValueError('A nonempty memberships object is required')
    if not source_layer.is_file():
        raise ValueError(f'Source layer does not exist: {source_layer}')
    if output_layer.exists() or source_layer == output_layer:
        raise ValueError('A new output path distinct from the source is required')
    if not isinstance(create_owners, (list, tuple)) or any(not isinstance(p, str) for p in create_owners):
        raise ValueError('create_owners must be an array of destination paths')
    for path, owner in memberships.items():
        if not all(isinstance(p, str) and Sdf.Path(p).IsAbsolutePath() and Sdf.Path(p).IsPrimPath()
                   for p in (path, owner)):
            raise ValueError('Membership paths must be absolute prim paths')
    created = set(create_owners)
    if len(created) != len(create_owners) or not created.issubset(memberships.values()):
        raise ValueError('Created owners must be unique membership destinations')
    source = Usd.Stage.Open(str(source_layer))
    if not source or source.GetCompositionErrors():
        raise ValueError('Invalid source composition')
    holders = set()
    source_paths = {Sdf.Path(path) for path in memberships}
    for owner in set(memberships.values()):
        prim = source.GetPrimAtPath(owner)
        if owner in created:
            if prim:
                raise ValueError(f'Created owner already exists: {owner}')
            prim = source.GetPrimAtPath(Sdf.Path(owner).GetParentPath())
        if (not prim or prim.IsPseudoRoot() or not prim.IsActive() or prim.IsInstance()
                or prim.IsInstanceProxy() or prim.IsInPrototype()):
            raise ValueError(f'Owner requires an existing editable parent or prim: {owner}')
    for path, owner in memberships.items():
        original = source.GetPrimAtPath(path)
        if not original or not original.IsActive() or original.IsInstanceProxy() or original.IsInPrototype():
            raise ValueError(f'Source must be an active editable prim, not an instance proxy: {path}')
        if (any(prefix in source_paths for prefix in Sdf.Path(owner).GetPrefixes())
                or any(prefix in source_paths for prefix in Sdf.Path(path).GetParentPath().GetPrefixes())):
            raise ValueError('Source subtrees must be disjoint and owners must not move')
        holder = Sdf.Path(owner).AppendChild('__placement_' + original.GetName())
        if holder in holders or source.GetPrimAtPath(holder):
            raise ValueError(f'Placement path already exists: {holder}')
        holders.add(holder)
        destination = source.GetPrimAtPath(Sdf.Path(owner).GetParentPath() if owner in created else owner)
        _check_destination(original, destination)
    # FlattenLayerStack merges only local layer opinions and anchors asset paths.
    # References, payloads, variants and instances remain composition arcs. Only
    # this anonymous layer is editable; no dependent stages are registered.
    working_layer = UsdUtils.FlattenLayerStack(source)
    stage = Usd.Stage.Open(working_layer)
    stage.SetInterpolationType(source.GetInterpolationType())
    for owner in sorted(created):
        UsdGeom.Xform.Define(stage, owner)
    mappings = {}
    for path, owner_path in memberships.items():
        original = source.GetPrimAtPath(path)
        destination = _placement(stage, original, stage.GetPrimAtPath(owner_path))
        # Author inherited opinions at the old path before the namespace edit so
        # native remapping also sees their relationship and connection targets.
        _inherit_state(original, stage.OverridePrim(path))
        mappings[path] = str(destination)
    for path, destination in mappings.items():
        editor = Usd.NamespaceEditor(stage)
        try:
            if not editor.MovePrimAtPath(path, destination):
                raise RuntimeError(f'NamespaceEditor.MovePrimAtPath failed for {path}')
            can_apply = editor.CanApplyEdits()
            if not can_apply:
                raise ValueError(f'NamespaceEditor cannot move {path}: {can_apply.whyNot}')
            if not editor.ApplyEdits():
                raise RuntimeError(f'NamespaceEditor.ApplyEdits failed for {path}')
        except Tf.ErrorException as error:
            error.add_note(f'NamespaceEditor move {path} to {destination}')
            raise
    output_layer.parent.mkdir(parents=True, exist_ok=True)
    if not working_layer.Export(str(output_layer)):
        raise RuntimeError(f'Export regrouped layer failed: {output_layer}')
    return {'status': 'regrouped', 'source_layer': str(source_layer),
            'output_layer': str(output_layer), 'root_path_map': mappings,
            'created_owner_paths': sorted(created)}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_layer')
    parser.add_argument('output_layer')
    parser.add_argument('--memberships', required=True, help='JSON file mapping source prim paths to owners')
    parser.add_argument('--create-owners', help='JSON file containing an array of new Xform owner paths')
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    result = regroup(args.source_layer, args.output_layer,
                     json.loads(Path(args.memberships).read_text()),
                     create_owners=json.loads(Path(args.create_owners).read_text()) if args.create_owners else ())
    Path(args.report).write_text(json.dumps(result, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
