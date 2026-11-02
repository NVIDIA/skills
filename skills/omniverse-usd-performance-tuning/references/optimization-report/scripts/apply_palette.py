#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Apply an explicitly requested RGB palette to editable output layers.

Run after merging, before instancing, or on copied library definitions. Material
bindings remain inside their asset; undefined over holders retain their specifier.
Superseded unbound materials are removed after binding remaps. Display-color-only
meshes retain that representation and map their values to the requested palette.

--canonical takes {"colours": [[r,g,b], ...], "map": [[[source], [target]], ...]}.
Pinned mappings take precedence; other RGB values map to the nearest palette
entry. --check reports authored shader/display colors without writing. This helper
supports static diffuse-color materials; textured or animated appearance needs a
separate, explicit conversion rather than a claim of preserved appearance.
"""
from __future__ import annotations

import argparse
import json
import hashlib
import math
import os
import sys

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

ROUND = 4
# All prims, not the default predicate. A shared master parked in an undefined
# `over` and referenced from the sites that place it composes everywhere those
# sites are -- including its Looks scope. A default-predicate pass does not
# reach the master, reports the layer clean, and leaves the original CAD
# colours authored: measured, 8 colours reported against 19 present.
ALL_PRIMS = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)


def walk(stage):
    for root in stage.GetPseudoRoot().GetFilteredChildren(ALL_PRIMS):
        for prim in Usd.PrimRange(root, ALL_PRIMS):
            yield prim


def _key(rgb):
    return tuple(round(float(c), ROUND) for c in rgb)


def _diffuse_of_shader(prim):
    shader = UsdShade.Shader(prim)
    if not shader:
        return None
    inp = shader.GetInput("diffuseColor")
    if not inp:
        return None
    val = inp.Get()
    return None if val is None else _key(val)


def _material_colour(material_prim):
    """The diffuse a Material resolves to through its surface source."""
    material = UsdShade.Material(material_prim)
    if not material:
        return None
    src = material.ComputeSurfaceSource()
    if isinstance(src, tuple):
        src = src[0] if src else None
    return _diffuse_of_shader(src) if src else None


def _nearest(colour, canonical):
    return min(canonical, key=lambda c: sum((a - b) ** 2 for a, b in zip(colour, c)))


def _looks_scope(material_prim):
    """The ancestor the canonical sibling belongs in: the material's parent."""
    return material_prim.GetParent().GetPath()


def _author_material(layer, scope_path, colour):
    """A UsdPreviewSurface material of `colour` under `scope_path`.

    Authored through Sdf, never `UsdShade.Material.Define`. `Define` turns every
    ancestor on the way down into a `def`, and a material sitting inside a
    shared master parked as an `over` would therefore DEFINE that master --
    composing 425 meshes at their parked location, 6.0 m outside the station,
    and taking the stage from 3,006 composed meshes to 3,353. `CreatePrimInLayer`
    creates missing ancestors as `over` and leaves existing specifiers alone.
    """
    name = "Canon_" + hashlib.sha256(repr(_key(colour)).encode()).hexdigest()[:16]
    path = scope_path.AppendChild(name)
    if layer.GetPrimAtPath(path) is not None:
        return path
    material = Sdf.CreatePrimInLayer(layer, path)
    material.specifier = Sdf.SpecifierDef
    material.typeName = "Material"
    shader = Sdf.CreatePrimInLayer(layer, path.AppendChild("PreviewSurface"))
    shader.specifier = Sdf.SpecifierDef
    shader.typeName = "Shader"
    Sdf.AttributeSpec(shader, "info:id", Sdf.ValueTypeNames.Token).default = (
        "UsdPreviewSurface")
    Sdf.AttributeSpec(
        shader, "inputs:diffuseColor", Sdf.ValueTypeNames.Color3f
    ).default = Gf.Vec3f(*colour)
    out = Sdf.AttributeSpec(material, "outputs:surface", Sdf.ValueTypeNames.Token)
    out.connectionPathList.explicitItems.append(
        path.AppendChild("PreviewSurface").AppendProperty("outputs:surface"))
    Sdf.AttributeSpec(shader, "outputs:surface", Sdf.ValueTypeNames.Token)
    return path


def collapse(path, canonical, pinned, write=True):
    """Collapse one layer onto the canonical palette. Returns a per-layer result dict."""
    stage = Usd.Stage.Open(path)
    if not stage:
        raise SystemExit(f"cannot open layer: {path}")
    layer = stage.GetRootLayer()
    if not canonical or any(len(c) != 3 or any(not math.isfinite(v) for v in c) for c in canonical):
        raise ValueError('canonical palette requires finite RGB triples')
    canon_set = {_key(c) for c in canonical}
    if any(value not in canon_set for value in pinned.values()):
        raise ValueError('pinned target is outside the canonical palette')
    # Resolve every unsupported case before editing this layer.
    for prim in walk(stage):
        if layer.GetPrimAtPath(prim.GetPath()) is None:
            continue
        if any(attr.GetNumTimeSamples() for attr in prim.GetAttributes()
               if attr.GetName().startswith(('inputs:', 'primvars:displayColor'))):
            raise ValueError('animated appearance is unsupported: ' + str(prim.GetPath()))
        if prim.IsA(UsdShade.Material):
            if _material_colour(prim) is None:
                raise ValueError('material has no supported diffuse color: ' + str(prim.GetPath()))
        if prim.IsA(UsdShade.Shader):
            shader = UsdShade.Shader(prim)
            inp = shader.GetInput('diffuseColor')
            if inp and inp.GetAttr().GetConnections():
                raise ValueError('connected diffuse color needs an explicit conversion: ' + str(prim.GetPath()))
    display_only = {}
    for prim in walk(stage):
        if not prim.IsA(UsdGeom.Mesh) or layer.GetPrimAtPath(prim.GetPath()) is None:
            continue
        material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        if not material:
            values = UsdGeom.Gprim(prim).GetDisplayColorAttr().Get()
            if values is not None:
                display_only[str(prim.GetPath())] = [pinned.get(_key(c)) or _nearest(_key(c), canon_set) for c in values]

    # 1. Every material authored in THIS layer, and the colour it resolves to.
    materials = {}
    for prim in walk(stage):
        if not prim.IsA(UsdShade.Material):
            continue
        if layer.GetPrimAtPath(prim.GetPath()) is None:
            continue  # defined elsewhere; that layer owns it
        colour = _material_colour(prim)
        if colour is not None:
            materials[prim.GetPath()] = colour

    superseded = {p: c for p, c in materials.items() if c not in canon_set}

    # What binds what, before any edit. A superseded material nothing binds is
    # deleted outright; authoring a replacement for it would add a prim to a
    # scope that needed one fewer.
    bound = set()
    for prim in walk(stage):
        material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        if material:
            bound.add(material.GetPrim().GetPath())

    replacement = {}
    if write:
        for mat_path, colour in superseded.items():
            if mat_path not in bound:
                continue
            target = pinned.get(colour) or _nearest(colour, canon_set)
            replacement[mat_path] = _author_material(
                layer, _looks_scope(stage.GetPrimAtPath(mat_path)), target)

    # Retarget locally authored direct, purpose and collection bindings.
    retargeted = 0
    if write and replacement:
        for prim in walk(stage):
            for rel in prim.GetRelationships():
                if not rel.GetName().startswith('material:binding'):
                    continue
                if layer.GetPropertyAtPath(rel.GetPath()) is None:
                    continue
                targets = rel.GetTargets()
                new = [replacement.get(t, t) for t in targets]
                if new != list(targets):
                    rel.SetTargets(new)
                    retargeted += 1
        bound = set()
        for prim in walk(stage):
            for rel in prim.GetRelationships():
                if rel.GetName().startswith('material:binding'):
                    bound.update(rel.GetForwardedTargets())

    # 3. Delete the superseded materials, now that nothing targets them.
    removed = []
    if write:
        for mat_path in superseded:
            if mat_path in bound:
                continue
            spec = layer.GetPrimAtPath(mat_path)
            if spec is not None:
                _remove(layer, mat_path)
                removed.append(str(mat_path))

    # 4. displayColor follows the bound material, one constant element.
    recoloured = cleared = 0
    if write:
        for prim in walk(stage):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            spec = layer.GetPrimAtPath(prim.GetPath())
            if spec is None:
                continue
            material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
            colour = _material_colour(material.GetPrim()) if material else None
            if colour is None:
                values = display_only.get(str(prim.GetPath()))
                if values is not None:
                    attr = spec.attributes.get('primvars:displayColor')
                    if attr is None:
                        # Inherited color is converted on its authored definition.
                        continue
                    attr.default = [Gf.Vec3f(*c) for c in values]
                    recoloured += 1
                continue
            _set_display_colour(spec, colour)
            recoloured += 1

    if write:
        tmp = path + ".palette.tmp" + os.path.splitext(path)[1]
        layer.Export(tmp)
        os.replace(tmp, path)

    after = Usd.Stage.Open(path)
    if write:
        after.GetRootLayer().Reload()
    authored = set()
    for prim in walk(after):
        colour = _diffuse_of_shader(prim)
        if colour is not None:
            authored.add(colour)
        if prim.IsA(UsdGeom.Mesh):
            values = UsdGeom.Gprim(prim).GetDisplayColorAttr().Get()
            if values is not None:
                authored.update(_key(c) for c in values)
    return {
        "layer": path,
        "materials_authored_here": len(materials),
        "superseded": len(superseded),
        "bindings_retargeted": retargeted,
        "materials_removed": len(removed),
        "meshes_recoloured": recoloured,
        "display_colour_cleared": cleared,
        "authored_colours_after": sorted(authored),
        "outside_canonical_after": sorted(c for c in authored if c not in canon_set),
    }


def _remove(layer, prim_path):
    """Remove a prim spec and its descendants from `layer`."""
    parent = layer.GetPrimAtPath(prim_path.GetParentPath())
    spec = layer.GetPrimAtPath(prim_path)
    if spec is None:
        return
    if parent is None:
        del layer.rootPrims[prim_path.name]
    else:
        del parent.nameChildren[prim_path.name]


def _set_display_colour(spec, colour):
    attr = spec.attributes.get("primvars:displayColor")
    if attr is None:
        attr = Sdf.AttributeSpec(
            spec, "primvars:displayColor", Sdf.ValueTypeNames.Color3fArray
        )
    attr.default = [Gf.Vec3f(*colour)]
    attr.SetInfo('interpolation', 'constant')
    legacy = spec.attributes.get('primvars:displayColor:interpolation')
    if legacy is not None:
        spec.RemoveProperty(legacy)
    # An indexed primvar keeps the old per-vertex values addressable.
    for leftover in ("primvars:displayColor:indices",):
        if spec.attributes.get(leftover) is not None:
            spec.RemoveProperty(spec.attributes[leftover])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("layers", nargs="+")
    ap.add_argument("--canonical", required=True,
                    help="JSON with `colours` and optional `map`")
    ap.add_argument("--check", action="store_true", help="report only, write nothing")
    ap.add_argument("--json", dest="json_out")
    args = ap.parse_args(argv)

    spec = json.load(open(args.canonical))
    canonical = [_key(c) for c in spec["colours"]]
    pinned = {_key(s): _key(d) for s, d in spec.get("map", [])}

    results = [collapse(path, canonical, pinned, write=not args.check)
               for path in args.layers]

    orphan_total = set()
    for r in results:
        orphan_total.update(tuple(c) for c in r["outside_canonical_after"])
        print(f"{os.path.basename(r['layer']):44} "
              f"authored={len(r['authored_colours_after']):3} "
              f"outside={len(r['outside_canonical_after']):3} "
              + (f"removed={r['materials_removed']:4} "
                 f"retargeted={r['bindings_retargeted']:5} "
                 f"recoloured={r['meshes_recoloured']:6}" if "materials_removed" in r else ""))
    print(f"\ndistinct colours outside the canonical set, all layers: {len(orphan_total)}")
    for colour in sorted(orphan_total):
        print(f"   {colour}")

    if args.json_out:
        json.dump({"schema": "apply-palette/1", "layers": results,
                   "outside_canonical": sorted(orphan_total)},
                  open(args.json_out, "w"), indent=1)
    return 1 if orphan_total else 0


if __name__ == "__main__":
    sys.exit(main())
