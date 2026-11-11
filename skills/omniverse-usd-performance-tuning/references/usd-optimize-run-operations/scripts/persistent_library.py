# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Look up persistent library nominations and write library metadata and indexes.

Lookup leaves published assets unchanged. write_index() authors metadata and
saves layers allowed by writable_layers, then writes index.json. rebuild_index()
reconstructs the index from published USD metadata.

Matching metadata describes the source before optimization. A nomination is not
geometry-preservation evidence; the caller verifies it against the new source.
"""
import copy
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path


METADATA_KEY = 'usdPerformance:reuse'


def exact_key(matching):
    fields = ('source_signature', 'root_transform_signature')
    return hashlib.sha256(json.dumps([2] + [matching.get(k, '') for k in fields]).encode()).hexdigest()


def compact_matching(matching):
    """Compatibility metadata; operation reports never enter persistent delivery."""
    assessment = matching.get('operation_assessment', {})
    complete = matching.get('completed', False)
    if assessment:
        rows = assessment.get('categories', [])
        complete = (assessment.get('complete') is True and
                    {r.get('category') for r in rows} == {'cleanup', 'primitive_fit', 'reduction', 'merge', 'reuse'} and
                    all(r.get('status') in ('executed', 'no_benefit', 'not_applicable', 'excluded') for r in rows))
    return dict(record_version=4, exact_key=exact_key(matching) if matching.get('source_signature') else '',
                source_signature=matching.get('source_signature', ''),
                root_transform_signature=matching.get('root_transform_signature', ''),
                tolerance_mm=matching.get('tolerance_mm', .1), completed=bool(complete),
                resource_closure_complete=matching.get('resource_closure_complete', False),
                identifiers=list(matching.get('identifiers', [])),
                **{k: matching[k] for k in ('normalized_source_key', 'normalized_source_info', 'product_key', 'product_identity_json', 'structural_type', 'upAxis') if matching.get(k)})


def output_geometry(stage, root, child_paths=()):
    """Inventory the delivered definition, excluding retained child assets."""
    from pxr import Usd, UsdGeom
    children = tuple(child_paths)
    return [str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath(root), Usd.TraverseInstanceProxies())
            if p.IsA(UsdGeom.Gprim) and not any(str(p.GetPath()) == c or str(p.GetPath()).startswith(c + '/') for c in children)]


def expand_index(index):
    """Adapt compact addresses to the existing in-memory publication contract."""
    if index.get('schema_version') != 2:
        return index
    result = dict(index, schema_version=1)
    result['dependencies'] = {v['location']: v['fingerprint'] for v in index.get('dependencies', {}).values()}
    result['entries'] = {}
    for identifier, row in index['entries'].items():
        lib = index['libraries'][row['library']]
        result['entries'][identifier] = dict(row, layer=lib['layer'], fingerprint=lib['fingerprint'],
            matching={'exact_key': row.get('exact_key'), 'identifiers': row.get('identifiers', []),
                      'normalized_source_key': row.get('normalized_source_key'), 'product_key': row.get('product_key')},
            library_units=lib.get('metersPerUnit'), library_axis=lib.get('upAxis'),
            library_time=lib.get('timeCodesPerSecond'), library_fps=lib.get('framesPerSecond'), compact=True)
    return result


def write_index(directory, index, writable_layers=None, stages=None, finalize_layers=()):
    """Author metadata on owned definitions, then project a rebuildable index."""
    from pxr import Usd, UsdGeom
    directory = Path(directory)
    writable = set(writable_layers) if writable_layers is not None else None
    stages, metadata = ({} if stages is None else stages), {}
    for identifier, entry in index['entries'].items():
        path = str((directory / entry['layer']).resolve())
        if path not in stages:
            stages[path] = Usd.Stage.Open(path)
        stage = stages[path]
        prim = stage.GetPrimAtPath(entry['prim_path'])
        if not prim: raise ValueError('index entry is missing: ' + identifier)
        value = prim.GetCustomDataByKey(METADATA_KEY)
        if writable is None or path in writable:
            value = compact_matching(value if entry.get('compact') and value else entry.get('matching', {}))
            value.update(entry_id=identifier, semantic_category=entry.get('semantic_category', (prim.GetCustomDataByKey(METADATA_KEY) or {}).get('semantic_category', '')))
            # Empty lists need an explicit USD array type in a dictionary.
            from pxr import Vt
            value['identifiers'] = Vt.StringArray(value['identifiers'])
            prim.SetCustomDataByKey(METADATA_KEY, value)
        if not value:
            raise ValueError('prior definition has no compact metadata; republish it before compact delivery')
        metadata[identifier] = dict(value)
    finalize = {str(Path(path).resolve()) for path in finalize_layers}
    for path, stage in stages.items():
        if writable is not None and path not in writable:
            continue
        layer = stage.GetRootLayer()
        if path not in finalize:
            layer.Save()
            continue
        fd, temporary = tempfile.mkstemp(prefix='.final-', suffix='.usdc', dir=Path(path).parent)
        os.close(fd)
        try:
            if not layer.Export(temporary):
                raise ValueError('final category export failed: ' + path)
            os.replace(temporary, path)
            layer.Reload(force=True)
        finally:
            Path(temporary).unlink(missing_ok=True)
    libraries, entries, hashes = {}, {}, {}
    def fingerprint(path):
        if path not in hashes:
            with Path(path).open("rb") as stream:
                hashes[path] = hashlib.file_digest(stream, "sha256").hexdigest()
        return hashes[path]
    for identifier, entry in index['entries'].items():
        path = str((directory / entry['layer']).resolve()); stage = stages[path]
        lid = hashlib.sha256(entry['layer'].encode()).hexdigest()[:20]
        digest = fingerprint(path)
        libraries[lid] = dict(layer=entry['layer'], fingerprint=digest,
            metersPerUnit=UsdGeom.GetStageMetersPerUnit(stage), upAxis=str(UsdGeom.GetStageUpAxis(stage)),
            timeCodesPerSecond=stage.GetTimeCodesPerSecond(), framesPerSecond=stage.GetFramesPerSecond())
        m = metadata[identifier]
        entries[identifier] = dict(library=lid, prim_path=entry['prim_path'], exact_key=m['exact_key'],
                                  identifiers=list(m.get('identifiers', [])))
        if m.get('product_key'): entries[identifier]['product_key'] = m['product_key']
        if m.get('normalized_source_key'): entries[identifier]['normalized_source_key'] = m['normalized_source_key']
        entry['fingerprint'] = digest
        entry.setdefault('matching', {}).update(dict(m, identifiers=list(m.get('identifiers', []))))
    # Existing publication computes a closure once; refresh owned files after metadata writes.
    deps = {}
    for location in index.get('dependencies', {}):
        path = directory / location
        digest = fingerprint(str(path.resolve()))
        index['dependencies'][location] = digest
        deps[hashlib.sha256(location.encode()).hexdigest()[:20]] = dict(location=location, fingerprint=digest)
    for row in libraries.values():
        row['dependencies'] = list(deps)
    result = dict(schema_version=2, status='published', entries=entries, libraries=libraries, dependencies=deps)
    for key in ('consumer_layer', 'material_library', 'portable', 'resource_dependency_closure'):
        if key in index: result[key] = index[key]
    (directory/'index.json').write_text(json.dumps(result, separators=(',', ':')) + '\n')
    return result


def rebuild_index(library_paths, directory):
    """Rebuild lookup from published USD metadata without run artifacts."""
    from pxr import Usd, UsdUtils
    directory = Path(directory).resolve()
    index = {'entries': {}, 'dependencies': {}, 'status': 'published'}
    for file in sorted(set(str(Path(p).resolve()) for p in library_paths)):
        stage = Usd.Stage.Open(file)
        if not stage or stage.GetCompositionErrors(): raise ValueError('library cannot compose: ' + file)
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(file)
        if unresolved: raise ValueError('unresolved library dependencies: ' + str(unresolved))
        for location in [str(p.realPath) for p in layers] + list(assets):
            path = Path(location).resolve()
            try: relative = str(path.relative_to(directory))
            except ValueError: relative = os.path.relpath(path, directory).replace(os.sep, '/')
            index['dependencies'][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        for prim in stage.TraverseAll():
            value = prim.GetCustomDataByKey(METADATA_KEY)
            if not value: continue
            # Only definitions whose metadata is locally authored are entries;
            # inherited metadata below a reference does not create new assets.
            if not stage.GetRootLayer().GetPrimAtPath(prim.GetPath()): continue
            local = stage.GetRootLayer().GetPrimAtPath(prim.GetPath()).customData
            if 'reuse' not in local.get('usdPerformance', {}): continue
            try: relative = str(Path(file).relative_to(directory))
            except ValueError: relative = os.path.relpath(file, directory).replace(os.sep, '/')
            index['entries'][value['entry_id']] = dict(layer=relative, prim_path=str(prim.GetPath()),
                semantic_category=value.get('semantic_category', ''), matching=dict(value))
    return write_index(directory, index, set())


def validate_matching(value):
    required = {'source_signature', 'root_transform_signature', 'metersPerUnit', 'tolerance_mm'}
    if not isinstance(value, dict) or not required.issubset(value):
        raise ValueError('matching metadata is incomplete')
    for key in ('source_signature', 'root_transform_signature'):
        if not isinstance(value[key], str) or (not value[key] and not (key=='source_signature' and value.get('product_key'))):
            raise ValueError('matching requires ' + key)
    for key in ('metersPerUnit', 'tolerance_mm'):
        item = value[key]
        if isinstance(item, bool) or not isinstance(item, (float, int)) or not math.isfinite(item) or item <= 0:
            raise ValueError(key + ' must be finite and positive')
    return value


class LookupContext:
    """One immutable input snapshot per lookup batch; never persisted as truth.

    Assessment uses identifiers without reading geometry or hashing libraries.
    Confirmation lazily validates dependencies, once per context. Start a new
    context after publication or index changes.
    """
    def __init__(self, index_paths):
        self.indexes, self.findings, self.hashes = [], [], {}
        self.by_identifier = {}
        self.by_exact = {}
        self.by_normalized = {}
        self.by_product = {}
        self.by_entrypoint, self.metadata, self.stages = {}, {}, {}
        for value in sorted(set(str(Path(p).resolve()) for p in index_paths)):
            path = Path(value)
            try:
                index = expand_index(json.loads(path.read_text()))
                if index.get('schema_version') != 1 or index.get('status') != 'published' or not isinstance(index.get('entries'), dict):
                    raise ValueError('unsupported or unpublished index')
                if not isinstance(index.get('dependencies', {}), dict):
                    raise ValueError('invalid dependency fingerprint record')
                self.indexes.append((path, index))
                for identifier, entry in sorted(index['entries'].items()):
                    if not isinstance(entry, dict) or not isinstance(entry.get('layer'), str) or not isinstance(entry.get('prim_path'), str) or not entry['prim_path'].startswith('/'):
                        continue
                    key = (str((path.parent / entry['layer']).resolve()), entry['prim_path'])
                    self.by_entrypoint.setdefault(key, []).append((path, identifier, entry))
                    if entry.get('compact'):
                        self.by_exact.setdefault((str(path), entry.get('exact_key')), []).append((identifier, entry))
                    matching = entry.get('matching', {})
                    if matching.get('product_key'):
                        self.by_product.setdefault((str(path), matching['product_key']), []).append((identifier, entry))
                    if matching.get('normalized_source_key'):
                        self.by_normalized.setdefault((str(path), matching['normalized_source_key']), []).append((identifier, entry))
                    values = matching.get('identifiers', []) if isinstance(matching, dict) else []
                    if not isinstance(values, list):
                        continue
                    for value in set(v for v in values if isinstance(v, str)):
                        self.by_identifier.setdefault(value, []).append((path, identifier, entry))
            except (OSError, ValueError, TypeError) as error:
                self.findings.append({'index': value, 'reason': str(error)})

    def stage(self, path):
        from pxr import Usd
        path=str(path)
        if path not in self.stages:self.stages[path]=Usd.Stage.Open(path)
        return self.stages[path]

    def digest(self, path):
        if path not in self.hashes:
            with path.open('rb') as stream:
                self.hashes[path] = hashlib.file_digest(stream, 'sha256').hexdigest()
        return self.hashes[path]

    def identifiers(self, values):
        """Stored identification-clue nominations, never confirmed reuse."""
        matches = {}
        for value in sorted(set(values)):
            for path, identifier, entry in self.by_identifier.get(value, []):
                key = (str(path), identifier)
                if key not in matches:
                    matches[key] = {'id': identifier, 'index': str(path),
                                    'layer': str((path.parent / entry['layer']).resolve()),
                                    'prim_path': entry['prim_path'], 'identifiers': [],
                                    'status': 'confirmation_required', 'confirmed': False}
                matches[key]['identifiers'].append(value)
        for nomination in matches.values():
            key=(nomination['layer'],nomination['prim_path'])
            if key not in self.metadata and Path(key[0]).is_file():
                stage=self.stage(key[0]);prim=stage.GetPrimAtPath(key[1]) if stage else None
                if prim:self.metadata[key]=dict(prim.GetCustomDataByKey(METADATA_KEY) or {})
            metadata=self.metadata.get(key,{})
            for field in ('semantic_category','structural_type'):
                if metadata.get(field):nomination[field]=metadata[field]
            if metadata.get('product_identity_json'):
                nomination['product_identity']=json.loads(metadata['product_identity_json'])
        return list(matches.values())



# Shared nomination extraction also feeds boundary review; no USD reads here.
import importlib.util as _importlib_util
_product_spec = _importlib_util.spec_from_file_location(
    'usd_product_clues', Path(__file__).resolve().parents[3] / 'scripts/product_clues.py')
product_clues = _importlib_util.module_from_spec(_product_spec)
_product_spec.loader.exec_module(product_clues)
identifying_values = product_clues.identifying_values


def source_identity_kind(matching, other):
    """Compare validated composed-source identity, frame and physical units."""
    if other.get('metersPerUnit') != matching.get('metersPerUnit'):
        return None
    if matching.get('product_key') and other.get('product_key') == matching['product_key']:
        return 'product' if matching.get('upAxis')==other.get('upAxis') else None
    if other.get('root_transform_signature') != matching.get('root_transform_signature'):
        return None
    if matching.get('source_signature') and other.get('source_signature') == matching['source_signature']:
        return 'exact'
    if matching.get('normalized_source_key') and other.get('normalized_source_key') == matching['normalized_source_key']:
        return 'normalized'
    return None


def validate_entry(context, index_file, identifier, entry, matching):
    """Validate one indexed entry using the batch's immutable metadata/digest cache."""
    validate_matching(matching)
    index_file = Path(index_file).resolve()
    index = next(index for path, index in context.indexes if path == index_file)
    layer = (index_file.parent / entry['layer']).resolve()
    stored = entry.get('matching', {})
    if entry.get('compact'):
        if stored.get('exact_key') != exact_key(matching) and not (matching.get('normalized_source_key') and stored.get('normalized_source_key') == matching['normalized_source_key']) and not (matching.get('product_key') and stored.get('product_key') == matching['product_key']):
            raise ValueError('published source identity differs')
        key = (str(layer), entry['prim_path'])
        if key not in context.metadata:
            stage = context.stage(layer)
            prim = stage.GetPrimAtPath(entry['prim_path']) if stage else None
            if not prim:
                raise ValueError('published entry prim is missing')
            context.metadata[key] = dict(prim.GetCustomDataByKey(METADATA_KEY) or {})
        stored = dict(context.metadata[key], metersPerUnit=entry['library_units'],upAxis=entry.get('library_axis'))
        stored['identifiers'] = list(stored.get('identifiers', []))
    else:
        stored = validate_matching(stored)
    if not source_identity_kind(matching, stored):
        raise ValueError('published source identity or units differ')
    if context.digest(layer) != entry.get('fingerprint'):
        raise ValueError('published layer fingerprint changed')
    if not isinstance(entry.get('prim_path'), str) or not entry['prim_path'].startswith('/'):
        raise ValueError('entry has no absolute prim path')
    dependencies = index.get('dependencies', {})
    for path, fingerprint in dependencies.items():
        dependency = (index_file.parent / path).resolve()
        if context.digest(dependency) != fingerprint:
            raise ValueError('published dependency fingerprint changed: ' + path)
    return {**entry, 'matching': stored, 'layer': str(layer), 'id': identifier,
            'index': str(index_file), 'status': 'confirmation_required',
            'read_only': True, 'confirmed': False,
            'dependency_fingerprints': {str((index_file.parent / path).resolve()): fingerprint
                                      for path, fingerprint in dependencies.items()},
            'dependency_verification': 'published_closure' if dependencies else 'entry_layer_only'}


def nominate(index_paths, matching, context=None):
    """Nominate accepted product, exact or normalized source identity across categories."""
    validate_matching(matching)
    context = context if context is not None else LookupContext(index_paths)
    candidates, findings = [], list(context.findings)
    for index_file, index in context.indexes:
        compact = any(entry.get('compact') for entry in index['entries'].values())
        rows = context.by_exact.get((str(index_file), exact_key(matching)), []) if compact else sorted(index['entries'].items())
        if matching.get('normalized_source_key'):
            rows = list(rows) + [row for row in context.by_normalized.get((str(index_file), matching['normalized_source_key']), []) if row not in rows]
        if matching.get('product_key'):
            rows = list(rows) + [row for row in context.by_product.get((str(index_file), matching['product_key']), []) if row not in rows]
        for identifier, entry in rows:
            try:
                candidates.append(validate_entry(context, index_file, identifier, entry, matching))
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
                findings.append({'index': str(index_file), 'entry': identifier, 'reason': str(error)})
    return {'schema_version': 1, 'candidates': candidates, 'findings': findings,
            'coverage': 'product_exact_and_normalized_source_identity', 'layers_hashed': len(context.hashes)}
