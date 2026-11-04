# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cheap product nominations from collected scalar labels; no geometry claims."""
import re
from collections import defaultdict


_ID = re.compile(r"product|model|partnumber|artikelnummer|identifier", re.I)
_LABEL = re.compile(r"name|label", re.I)
_CONTEXT = re.compile(r"manufacturer|supplier|category|hersteller|lieferant", re.I)
_GENERIC = re.compile(r"n?[0-9a-f]{6,}|(?:xform|scope|mesh|geo|geometry|wrapper|part)\d*", re.I)


def identifying_clues(facts):
    """Keep full authored values beside distinctly typed nomination keys."""
    fields = [("prim_name", facts.get("prim_name")), ("displayName", facts.get("displayName"))]
    fields += [("assetInfo." + str(k), v) for k, v in facts.get("assetInfo", {}).items()]
    fields += [(a.get("name", ""), a.get("value")) for a in facts.get("custom_attributes", [])]
    fields += [(".".join(a.get("key_path", [])), a.get("value")) for a in facts.get("customData_labels", [])]
    result = []
    for field, value in fields:
        if not isinstance(value, str) or not value.strip() or len(value) > 256:
            continue
        value = value.strip()
        leaf = field.rsplit(".", 1)[-1]
        explicit = bool(_ID.search(leaf))
        if not explicit and (not _LABEL.search(leaf) or _CONTEXT.search(leaf)):
            continue
        if not explicit and _GENERIC.fullmatch(value):
            continue
        result.append(dict(key=value if explicit else "label:" + value.casefold(),
                           kind="identifier" if explicit else "authored_label", field=field, value=value))
        # Namespace tokens often encode converter identity, not product identity.
        # Keep the complete prim name for exact-label nomination and agent review.
        if field == "prim_name":
            continue
        # Separators nominate related labels; no token is removed from evidence.
        # In particular, digit suffixes are never interpreted as occurrence IDs.
        for token in sorted(set(re.findall(r"[^\W_]+", value.casefold()))):
            if len(token) >= 4 and any(c.isalpha() for c in token) and any(c.isdigit() for c in token) and not _GENERIC.fullmatch(token):
                result.append(dict(key="token:" + token, kind="label_token", field=field, value=value))
    return result


def identifying_values(facts):
    return sorted({row["key"] for row in identifying_clues(facts)})


def review_groups(members):
    """Group shared clues; full authored facts remain in evidence.members."""
    buckets = defaultdict(set)
    kinds = {}
    for member in members:
        for clue in identifying_clues(member):
            buckets[clue["key"]].add(member["prim_path"])
            kinds[clue["key"]] = clue["kind"]
    groups = {}
    for key, paths in sorted(buckets.items()):
        paths = tuple(sorted(paths))
        if len(paths) < 2:
            continue
        row = groups.setdefault(paths, dict(source_prim_paths=list(paths), clues=[],
                                            status="nomination_only", geometry_compared=False))
        row["clues"].append(dict(key=key, kind=kinds[key]))
    return sorted(groups.values(), key=lambda row: (-len(row["source_prim_paths"]), row["source_prim_paths"]))


def validate_product_identity(value, stage, roots):
    """Validate cited facts; interchangeability is the recorded agent judgment."""
    import json
    from pxr import Sdf, Usd, UsdGeom
    required = {'key', 'reason', 'frame', 'evidence'}
    if not isinstance(value, dict) or not required <= value.keys() or set(value)-required-{'library_entry'}:
        raise ValueError('product_identity requires key, reason, frame and evidence')
    if any(not isinstance(value[k], str) or not value[k].strip() for k in ('key','reason')) or value['frame'] != 'asset_root':
        raise ValueError('product_identity requires an explicit key, substitution reason and asset_root frame')
    if not value['key'].isprintable():
        invalid=next(char for char in value['key'] if not char.isprintable())
        raise ValueError(f'product_identity.key must contain only printable characters (found U+{ord(invalid):04X})')
    evidence=value['evidence']
    if not isinstance(evidence,list) or not evidence:
        raise ValueError('product_identity requires source evidence')
    covered=set()
    for row in evidence:
        if not isinstance(row,dict) or set(row)!={'path','field','value'}:
            raise ValueError('product evidence requires path, field and value')
        path,field=row['path'],row['field']
        prim=stage.GetPrimAtPath(path)
        owners=[root for root in roots if Sdf.Path(path).HasPrefix(Sdf.Path(root)) or Sdf.Path(root).HasPrefix(Sdf.Path(path))]
        if not prim or not owners: raise ValueError('product evidence is outside selected assets')
        if field=='name': actual=prim.GetName()
        elif field=='displayName': actual=prim.GetDisplayName()
        elif field.startswith('assetInfo:'): actual=prim.GetAssetInfoByKey(field[len('assetInfo:'):])
        elif field.startswith('attribute:'):
            attr=prim.GetAttribute(field[len('attribute:'):]);actual=attr.Get() if attr else None
        elif field=='dimensions_m':
            box=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_,UsdGeom.Tokens.render]).ComputeRelativeBound(prim,prim)
            actual=[v*UsdGeom.GetStageMetersPerUnit(stage) for v in box.ComputeAlignedRange().GetSize()]
        else: raise ValueError('Unsupported product evidence field: '+str(field))
        if actual is None or str(actual).strip()=='' or json.dumps(actual,default=list,sort_keys=True)!=json.dumps(row['value'],sort_keys=True):
            raise ValueError('Product evidence value differs from source: '+path+' '+field)
        covered.update(owners)
    if set(roots)-covered:
        raise ValueError('Product identity requires cited source evidence for every selected asset')
    entry=value.get('library_entry')
    if entry is not None and (not isinstance(entry,dict) or set(entry)!={'index','id'} or not all(isinstance(v,str) and v for v in entry.values())):
        raise ValueError('product library_entry requires index and id')
    return value
