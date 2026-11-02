# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Publish verified definitions with immutable assets and index-last common delivery.

Standalone delivery uses an atomic directory rename. Common-library delivery
installs uniquely named files and atomically replaces its cumulative index last.
"""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid

from pxr import Sdf, Usd, UsdGeom


def file_fingerprint(path):
    value = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def _check_dependency_fingerprints(fingerprints, base=None, digest=None):
    if not isinstance(fingerprints, dict):
        raise ValueError('dependency fingerprints must be a mapping')
    digest = digest or file_fingerprint
    resolved = {}
    for path, fingerprint in fingerprints.items():
        file = (Path(base) / path if base else Path(path)).resolve()
        if not file.is_file() or digest(file) != fingerprint:
            raise ValueError('inherited dependency changed before publication: ' + str(file))
        resolved[str(file)] = fingerprint
    return resolved


def _read_inherited_index(path, digest=None):
    index_file = Path(path).resolve()
    index = json.loads(index_file.read_text())
    spec = importlib.util.spec_from_file_location('inherited_reader', Path(__file__).with_name('persistent_library.py'))
    reader = importlib.util.module_from_spec(spec); spec.loader.exec_module(reader)
    index = reader.expand_index(index)
    if index.get('schema_version') != 1 or index.get('status') != 'published':
        raise ValueError('existing library index is not published')
    return index, _check_dependency_fingerprints(index.get('dependencies', {}), index_file.parent, digest)


_LIST_FIELDS = ('explicitItems', 'prependedItems', 'appendedItems', 'addedItems', 'deletedItems', 'orderedItems')


def _map_list(operation, function):
    for field in _LIST_FIELDS:
        items = list(getattr(operation, field))
        if items:
            setattr(operation, field, [function(item) for item in items])
    return operation


def _prim_path(value):
    if not isinstance(value, str) or not Sdf.Path.IsValidPathString(value):
        raise ValueError('entry requires valid prim_path')
    path = Sdf.Path(value)
    if not path.IsAbsolutePath() or not path.IsPrimPath():
        raise ValueError('entry requires absolute prim_path')
    return path


class _Writer:
    def __init__(self, staging, replacements, portable=False, publication_id=None, material_library=None):
        self.staging = staging
        self.publication_id = publication_id
        self.material_library = Path(material_library).resolve() if material_library else None
        self.replacements = replacements
        self.localized = {}
        self.external = set()
        self.portable = portable
        self.opaque_material_resources = set()

    def register_reference_aliases(self, file, old, new, target_file, aliases):
        """Register non-overlapping source entries for reference rewriting."""
        for alias in [str(file)] + aliases:
            if isinstance(alias, str):
                alias_file, alias_prim = alias, old
            elif isinstance(alias, dict) and set(alias) == {'layer', 'prim_path'}:
                alias_file, alias_prim = alias['layer'], _prim_path(alias['prim_path'])
            else:
                raise ValueError('reference alias requires layer and prim_path')
            alias_file = str(Path(alias_file).resolve())
            rows = self.replacements.setdefault(alias_file, [])
            # Several current definitions can reuse the same published entry.
            # Register their identical source-to-destination mapping only once.
            if (alias_prim, new, target_file) in rows:
                continue
            if any(alias_prim.HasPrefix(row[0]) or row[0].HasPrefix(alias_prim) for row in rows):
                raise ValueError('duplicate or overlapping definition reference alias')
            if not Path(alias_file).is_file():
                raise ValueError('reference alias layer does not exist')
            alias_layer = Sdf.Layer.FindOrOpen(alias_file)
            if not alias_layer or not alias_layer.GetPrimAtPath(alias_prim):
                raise ValueError('reference alias prim does not exist')
            rows.append((alias_prim, new, target_file))

    def preserve_dependencies(self, source):
        """Retain an inherited dependency closure without copying or rewriting it."""
        source = Path(source).resolve()
        if source in self.external:
            return
        self.external.add(source)
        if source.suffix.lower()=='.mdl':
            self.opaque_material_resources.add(source)
        if source.suffix.lower() not in ('.usd', '.usda', '.usdc'):
            return
        layer = Sdf.Layer.FindOrOpen(str(source))
        if not layer:
            raise ValueError('persistent dependency cannot be opened')
        for asset in layer.GetExternalReferences():
            if asset:
                self.preserve_dependencies(self.resolve(layer, asset))
        def inspect(value):
            if isinstance(value, Sdf.AssetPath) and value.path:
                self.preserve_dependencies(self.resolve(layer, value.path))
            elif isinstance(value, dict):
                for item in value.values():
                    inspect(item)
            elif isinstance(value, (list, tuple)) or type(value).__name__ == 'AssetPathArray':
                for item in value:
                    inspect(item)
        def visit(path):
            spec = layer.GetObjectAtPath(path)
            if spec:
                for key in spec.ListInfoKeys():
                    inspect(spec.GetInfo(key))
        layer.Traverse(Sdf.Path.absoluteRootPath, visit)

    def resolve(self, source_layer, asset):
        resolved = Sdf.ComputeAssetPathRelativeToLayer(source_layer, asset)
        path = Path(resolved)
        if not path.is_file():
            raise ValueError('dependency must be resolved/localized before publication: ' + asset)
        return path.resolve()

    def relative(self, target, output):
        try:
            return os.path.relpath(target, output.parent).replace(os.sep, '/')
        except ValueError as error:
            raise ValueError(f'Cannot reference {target} relative to {output}; use portable_output for cross-volume delivery') from error

    def asset(self, source_layer, asset, output):
        if not asset:
            return asset
        source = self.resolve(source_layer, asset)
        target = self.localize(source)
        return self.relative(target, output)

    def arc(self, entry, source_layer, output, remap):
        target = entry.primPath
        asset = entry.assetPath
        if asset:
            source = self.resolve(source_layer, asset)
            candidates = self.replacements.get(str(source), [])
            if target.isEmpty:
                dependency = Sdf.Layer.FindOrOpen(str(source))
                if dependency and dependency.defaultPrim:
                    target = Sdf.Path('/' + dependency.defaultPrim)
            match = next((row for row in candidates if target.HasPrefix(row[0])), None)
            if match:
                old, new, file = match
                asset = self.relative(file, output)
                target = target.ReplacePrefix(old, new)
            else:
                asset = self.asset(source_layer, asset, output)
        else:
            target = remap(target)
        if isinstance(entry, Sdf.Reference):
            return Sdf.Reference(asset, target, entry.layerOffset, entry.customData)
        return Sdf.Payload(asset, target, entry.layerOffset)

    def rewrite(self, layer, original, output, scope=Sdf.Path.absoluteRootPath, mappings=()):
        def remap(path):
            for old, new in sorted(mappings, key=lambda row: len(str(row[0])), reverse=True):
                if path.HasPrefix(old):
                    return path.ReplacePrefix(old, new)
            return path

        def values(value):
            if isinstance(value, Sdf.AssetPath):
                return Sdf.AssetPath(self.asset(original, value.path, output))
            if isinstance(value, dict):
                return {key: values(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)) or type(value).__name__ == 'AssetPathArray':
                return type(value)([values(item) for item in value])
            return value

        def visit(path):
            spec = layer.GetObjectAtPath(path)
            if not spec:
                return
            for key in spec.ListInfoKeys():
                value = spec.GetInfo(key)
                if key in ('references', 'payload'):
                    value = _map_list(value, lambda item: self.arc(item, original, output, remap))
                elif key in ('targetPaths', 'connectionPaths', 'inheritPaths', 'specializes'):
                    def map_path(item):
                        if item.IsAbsolutePath():
                            return remap(item)
                        original_path = path
                        for old, new in sorted(mappings, key=lambda row: len(str(row[1])), reverse=True):
                            if path.HasPrefix(new):
                                original_path = path.ReplacePrefix(new, old)
                                break
                        return remap(item.MakeAbsolutePath(original_path.GetPrimPath()))
                    value = _map_list(value, map_path)
                    proxy = getattr(spec, {'targetPaths': 'targetPathList', 'connectionPaths': 'connectionPathList',
                                           'inheritPaths': 'inheritPathList', 'specializes': 'specializesList'}[key])
                    for field in _LIST_FIELDS:
                        if list(getattr(proxy, field)) != list(getattr(value, field)):
                            setattr(proxy, field, list(getattr(value, field)))
                    continue
                elif key == 'timeSamples':
                    for time, sample in value.items():
                        converted = values(sample)
                        if converted != sample:
                            layer.SetTimeSample(path, time, converted)
                    continue
                else:
                    converted = values(value)
                    if converted == value:
                        continue
                    value = converted
                spec.SetInfo(key, value)
        layer.Traverse(scope, visit)

    def local_path(self, source):
        name = hashlib.sha256(str(source).encode()).hexdigest()[:20] + source.suffix
        if self.publication_id:
            name = self.publication_id + '_' + name
        folder = 'materials' if source == self.material_library else 'dependencies'
        if folder == 'materials':
            name = 'materials_' + name
        return self.staging / folder / name

    def localize(self, source):
        key = str(source)
        if source.suffix.lower()=='.mdl':
            if self.portable:
                raise ValueError('portable MDL dependency closure is unqualified; an MDL-aware collector is required: '+key)
            self.opaque_material_resources.add(source)
            self.external.add(source)
        if key in self.localized:
            return self.localized[key]
        if source in self.external:
            self.localized[key] = source
            return source
        target = self.local_path(source)
        target.parent.mkdir(exist_ok=True)
        self.localized[key] = target  # Break external dependency cycles.
        if source.suffix.lower() not in ('.usd', '.usda', '.usdc'):
            shutil.copy2(source, target)
            return target
        original = Sdf.Layer.FindOrOpen(key)
        layer = Sdf.Layer.OpenAsAnonymous(key)
        if not layer:
            raise ValueError('cannot read dependency layer: ' + key)
        layer.subLayerPaths = [self.asset(original, item, target) for item in original.subLayerPaths]
        self.rewrite(layer, original, target)
        if not layer.Export(str(target)):
            raise ValueError('cannot write dependency layer')
        return target


def _copy_definition(writer, source, source_path, layer, target, output):
    """Copy the definition plus only its referenced local auxiliary roots."""
    mappings = [(source_path, target)]
    pending = [source_path]
    copied = set()
    while pending:
        root = pending.pop()
        if root in copied:
            continue
        copied.add(root)
        def inspect(path):
            spec = source.GetObjectAtPath(path)
            if not spec:
                return
            targets = []
            for key in ('targetPaths', 'connectionPaths', 'inheritPaths', 'specializes', 'references', 'payload'):
                if not spec.HasInfo(key):
                    continue
                for field in _LIST_FIELDS:
                    for item in getattr(spec.GetInfo(key), field):
                        if key in ('references', 'payload'):
                            if item.assetPath:
                                continue
                            item = item.primPath
                        if item.isEmpty:
                            continue
                        if not item.IsAbsolutePath():
                            item = item.MakeAbsolutePath(path.GetPrimPath())
                        targets.append(item)
            for item in targets:
                if any(item.HasPrefix(old) for old, _ in mappings):
                    continue
                # Copy the addressed root, not unrelated siblings or geometry.
                prim = item.GetPrimPath()
                if not source.GetPrimAtPath(prim):
                    raise ValueError('local dependency has no authored spec: ' + str(item))
                top = prim.GetPrefixes()[0]
                if source_path.HasPrefix(top):
                    raise ValueError('local dependency points above the definition boundary')
                if not any(top.HasPrefix(old) for old, _ in mappings):
                    destination = target.AppendChild('__dependencies').AppendChild(top.name)
                    if source.GetPrimAtPath(source_path.AppendChild('__dependencies')):
                        raise ValueError('definition uses reserved dependency destination; choose another entry scope')
                    mappings.append((top, destination))
                    pending.append(top)
        source.Traverse(root, inspect)
    for old, new in mappings:
        Sdf.CreatePrimInLayer(layer, new.GetParentPath())
        if not Sdf.CopySpec(source, old, layer, new):
            raise ValueError('definition copy failed: ' + str(old))
    writer.rewrite(layer, source, output, target, mappings)


def _unresolved(stage, root=None):
    """Exact property/target pairs, including instance proxies, for baseline evidence."""
    prims=Usd.PrimRange(stage.GetPrimAtPath(root),Usd.TraverseInstanceProxies()) if root else Usd.PrimRange.Stage(stage,Usd.TraverseInstanceProxies())
    result=set()
    for prim in prims:
        for prop in prim.GetAuthoredProperties():
            targets=prop.GetTargets() if isinstance(prop,Usd.Relationship) else prop.GetConnections()
            for target in targets:
                if not stage.GetObjectAtPath(target):result.add((str(prop.GetPath()),str(target)))
    return result


def _check_unresolved(source, published, old=None, new=None):
    baseline=_unresolved(source,old)
    if old is not None:
        baseline={(str(Sdf.Path(prop).ReplacePrefix(old,new)),str(Sdf.Path(target).ReplacePrefix(old,new))) for prop,target in baseline}
    actual=_unresolved(published,new)
    introduced=actual-baseline
    if introduced:raise ValueError('published property has new unresolved target: '+str(sorted(introduced)[0]))
    return [{'property_path':prop,'target_path':target,'status':'preexisting_unresolved'} for prop,target in sorted(actual)]


def _stage_units_and_rates(stage):
    layer = stage.GetRootLayer()
    return (UsdGeom.GetStageMetersPerUnit(stage), str(UsdGeom.GetStageUpAxis(stage)),
            layer.timeCodesPerSecond, layer.framesPerSecond)


def _build_category_layers(writer, categories, existing_libraries, entries, records, staging):
    """Compose category contents in memory, preserving units and inherited entries."""
    layers, metadata = {}, {}
    for category, filename in categories.items():
        output = staging / filename
        existing = existing_libraries.get(category)
        if existing:
            prior_index = Path(existing).resolve().parent / 'index.json'
            if prior_index.is_file():
                _read_inherited_index(prior_index)
            original = Sdf.Layer.FindOrOpen(str(Path(existing).resolve()))
            if not original:
                raise ValueError('existing category cannot be opened')
            layer = Sdf.Layer.OpenAsAnonymous(str(Path(existing).resolve()))
            layer.subLayerPaths = [writer.asset(original, item, output) for item in original.subLayerPaths]
            writer.rewrite(layer, original, output)
        else:
            layer = Sdf.Layer.CreateAnonymous()
            root = Sdf.CreatePrimInLayer(layer, '/Library')
            root.specifier = Sdf.SpecifierDef
            root.typeName = 'Scope'
            root.SetInfo('kind', 'group')
            layer.defaultPrim = 'Library'
        layers[category] = layer
    for identifier, (file, source, old, new, record) in records.items():
        entry = entries[identifier]
        if entry.get('read_only'):
            writer.localize(file)
            continue
        category = entry['semantic_category']
        layer = layers[category]
        if layer.GetPrimAtPath(new):
            raise ValueError('entry collision; existing definitions are immutable: ' + str(new))
        source_stage = Usd.Stage.Open(source)
        value = _stage_units_and_rates(source_stage)
        if category in metadata and metadata[category] != value:
            raise ValueError('category definitions have incompatible units, axis or time rates')
        if category not in metadata and category in existing_libraries:
            prior = Usd.Stage.Open(layer)
            prior_value = _stage_units_and_rates(prior)
            if prior_value != value:
                raise ValueError('existing category has incompatible units, axis or time rates')
        has_previous = category in metadata or category in existing_libraries
        metadata[category] = value
        layer.pseudoRoot.SetInfo('metersPerUnit', value[0])
        layer.pseudoRoot.SetInfo('upAxis', value[1])
        layer.timeCodesPerSecond, layer.framesPerSecond = value[2:]
        layer.startTimeCode = min(layer.startTimeCode, source.startTimeCode) if has_previous else source.startTimeCode
        layer.endTimeCode = max(layer.endTimeCode, source.endTimeCode) if has_previous else source.endTimeCode
        _copy_definition(writer, source, old, layer, new, staging / categories[category])
        if entry.get('equipment_family'):
            prim_spec = layer.GetPrimAtPath(new)
            data = dict(prim_spec.customData)
            data['equipment_family'] = entry['equipment_family']
            prim_spec.customData = data
    return layers


def publish_libraries(plan, verified_outputs, destination, existing_libraries=None, consumer_layer=None, portable=False, geometry_reuse=False, geometry_policy=None, progress=None, material_library=None, common_library=False, consumer_destination=None):
    """Return published entries only after all layers reopen successfully.

    verified_outputs[id] contains layer, prim_path, verified=True and fingerprint
    (SHA256 of the exact file). Existing category layers are cloned into a new
    version before appending, retaining entry paths. Atomic directory publication
    does not update a global cross-scene index or mutate an earlier version.
    The root consumer is separate from library definitions; /Library is an
    organizational default prim. Optional consumer_layer is copied to after.usdc
    and its dependencies remapped in the same transaction. Its root assembly
    remains outside the category plan.
    reference_layers on verified outputs lists immutable input-layer aliases
    used by parallel parent computation, mapped to the verified output entry.
    For read-only definitions put reference_layers on the planned entry. Aliases
    are {layer, prim_path} objects, or strings using the record's source prim path.
    Resolver-backed and tiled assets must be localized by the caller first.
    Confirmed inherited entries remain external and read-only by default. Setting
    portable=True snapshots those dependencies too. New working dependencies are
    always copied into the delivered version.
    geometry_reuse requires explicit geometry_paths on each new output record;
    native sharing is confined to those owned meshes before the version is published.
    common_library installs new assets without changing earlier files, updates the
    cumulative index last, and writes the consumer at consumer_destination.
    material_library identifies the authored shared library; record its delivered
    path only when referenced by the published content.
    """
    destination = Path(destination).resolve()
    if type(portable) is not bool or type(geometry_reuse) is not bool:
        raise ValueError('portable and geometry_reuse must be booleans')
    if common_library and (not consumer_layer or not consumer_destination):
        raise ValueError('common library publication requires a separate consumer destination')
    if common_library and existing_libraries:
        raise ValueError('common library assets are immutable; nominate existing entries instead of extending category files')
    if destination.exists() and not common_library:
        raise ValueError('publication destination must be a new immutable version')
    entries = plan.get('entries', {})
    if plan.get('schema_version') != 1 or not entries:
        raise ValueError('publication requires a versioned plan with entries')
    existing_libraries = existing_libraries or {}
    publication_id = uuid.uuid4().hex if common_library else None
    records, categories, replacements = {}, {}, {}
    # Supplied files are immutable during this invocation. Keep all expected-hash
    # comparisons, but read each file/index and compare its layer snapshot once.
    input_hashes, input_indexes, input_layers = {}, {}, {}
    def input_digest(path):
        path = Path(path).resolve()
        if path not in input_hashes:
            input_hashes[path] = file_fingerprint(path)
        return input_hashes[path]
    def input_index(path):
        path = Path(path).resolve()
        if path not in input_indexes:
            input_indexes[path] = _read_inherited_index(path, input_digest)
        return input_indexes[path]
    def input_layer(path, identifier):
        if path not in input_layers:
            source = Sdf.Layer.FindOrOpen(str(path))
            fresh = Sdf.Layer.OpenAsAnonymous(str(path))
            if source and (source.dirty or source.ExportToString() != fresh.ExportToString()):
                raise ValueError('source layer has unsaved or stale cached opinions: ' + identifier)
            input_layers[path] = source
        return input_layers[path]
    for identifier, entry in entries.items():
        if entry.get('read_only'):
            if entry.get('confirmed') is not True or not entry.get('confirmation_fingerprint') or entry.get('status') == 'confirmation_required':
                raise ValueError('persistent reuse requires confirmed verification evidence')
            _check_dependency_fingerprints(entry.get('dependency_fingerprints', {}), digest=input_digest)
            if entry.get('index'):
                input_index(entry['index'])
            record = {'layer': entry['layer'], 'prim_path': entry['prim_path'],
                      'fingerprint': entry['fingerprint'], 'verified': True,
                      'matching': entry.get('matching', {})}
            for field in ('assessment', 'operation_assessment', 'path_map', 'reference_layers'):
                if field in entry:
                    record[field] = entry[field]
        else:
            record = copy.deepcopy(verified_outputs.get(identifier, {}))
        if record.get('verified') is not True or not record.get('fingerprint'):
            raise ValueError('definition requires verified output: ' + identifier)
        file = Path(record['layer']).resolve()
        if not file.is_file() or input_digest(file) != record['fingerprint']:
            raise ValueError('verified output fingerprint changed: ' + identifier)
        source = input_layer(file, identifier)
        old, new = _prim_path(record['prim_path']), _prim_path(entry['prim_path'])
        if not source or not source.GetPrimAtPath(old):
            raise ValueError('definition entry does not exist: ' + identifier)
        if source.subLayerPaths and not entry.get('read_only'):
            raise ValueError('working definition must contain its authored opinions without sublayers')
        records[identifier] = (file, source, old, new, record)
        if not entry.get('read_only'):
            category = entry['semantic_category']
            filename = Path(entry['layer']).name
            if filename in ('after.usda', 'after.usdc', 'index.json', 'dependencies') or Path(filename).suffix not in ('.usd', '.usda', '.usdc'):
                raise ValueError('category layer filename is reserved or unsupported')
            if category in categories and categories[category] != filename:
                raise ValueError('category must have one writer layer')
            if filename in categories.values() and categories.get(category) != filename:
                raise ValueError('different categories cannot share a layer')
            categories[category] = filename
    if common_library:
        categories = {category: 'assets/' + Path(filename).stem + '_' + publication_id + Path(filename).suffix
                      for category, filename in categories.items()}
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.' + destination.name + '-', dir=destination.parent))
    retained_failure = False
    try:
        writer = _Writer(staging, replacements, portable=portable, publication_id=publication_id, material_library=material_library)
        if common_library:
            (staging / 'assets').mkdir()
        migrate_legacy = {identifier for identifier, (_, source, old, _, _) in records.items()
                          if entries[identifier].get('read_only') and not source.GetPrimAtPath(old).customData.get('usdPerformance', {}).get('reuse')}
        if not portable:
            for identifier, (file, source, old, new, record) in records.items():
                if entries[identifier].get('read_only') and identifier not in migrate_legacy:
                    writer.preserve_dependencies(file)
        for identifier, (file, source, old, new, record) in records.items():
            entry = entries[identifier]
            aliases = record.get('reference_layers', [])
            if not isinstance(aliases, list):
                raise ValueError('reference_layers must be an array of layer aliases')
            if entry.get('read_only'):
                target_file = file if not portable and identifier not in migrate_legacy else writer.local_path(file)
            else:
                target_file = staging / categories[entry['semantic_category']]
            writer.register_reference_aliases(file, old, new, target_file, aliases)
        layers = _build_category_layers(
            writer, categories, existing_libraries, entries, records, staging)
        for category, layer in layers.items():
            if not layer.Export(str(staging / categories[category])):
                raise ValueError('category layer export failed')
        sharing = {}
        if geometry_reuse:
            spec = importlib.util.spec_from_file_location('publication_geometry_reuse', Path(__file__).with_name('shared_geometry.py'))
            helper = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(helper)
            for category_number, (category, filename) in enumerate(categories.items(), 1):
                members = {identifier: values for identifier, values in records.items()
                           if not entries[identifier].get('read_only') and entries[identifier]['semantic_category'] == category}
                if progress:
                    progress('library_publication', activity='Sharing eligible whole definitions',
                             operation='deduplicateHierarchies', category=category,
                             candidate_entry_count=sum(values[4].get('whole_definition_reuse_eligible') is True for values in members.values()))
                whole = helper.share_definitions(staging / filename,
                    [str(values[3]) for values in members.values()
                     if values[4].get('whole_definition_reuse_eligible') is True],
                    policy=geometry_policy,
                    maximum_occurrence_scale=max(values[4].get('maximum_occurrence_scale', 1.0) for values in members.values()))
                if progress:
                    progress('library_publication', activity='Whole-definition sharing complete',
                             operation='deduplicateHierarchies', category=category,
                             submitted_entry_count=whole['submitted_entry_count'],
                             shared_entry_count=whole['shared_entry_count'])
                if not whole['accepted']:
                    retained_failure = True
                    failure = destination.parent / (destination.name + '.definition-reuse-failure.json')
                    whole['retained_staging_directory'] = str(staging)
                    failure.write_text(json.dumps(whole, indent=2) + '\n')
                    raise ValueError('whole-definition sharing failed preservation; evidence: ' + str(failure))
                shared_roots = {row[key] for row in whole['internal_references'] for key in ('path', 'target')}
                meshes = []
                stage = Usd.Stage.Open(str(staging / filename))
                for identifier, (file, source, old, new, record) in members.items():
                    if not isinstance(record.get('geometry_paths'), list):
                        raise ValueError('geometry reuse requires explicit owned output paths: ' + identifier)
                    for path in record['geometry_paths']:
                        original = _prim_path(path)
                        if not original.HasPrefix(old):
                            raise ValueError('geometry reuse scope leaves its definition')
                        published_path = original.ReplacePrefix(old, new)
                        prim = stage.GetPrimAtPath(published_path)
                        if not prim:
                            raise ValueError('geometry reuse scope is missing: ' + str(published_path))
                        if prim.IsA(UsdGeom.Mesh) and published_path != new and str(new) not in shared_roots:
                            meshes.append(str(published_path))
                if progress:
                    progress('library_publication', activity='Sharing category geometry', operation='deduplicateGeometry',
                             category=category, category_number=category_number, category_count=len(categories),
                             definition_count=len(members), candidate_mesh_count=len(meshes))
                evidence = helper.share_geometry(staging / filename, meshes,
                    entry_prim_paths=[str(values[3]) for values in members.values()], policy=geometry_policy,
                    maximum_occurrence_scale=max(values[4].get('maximum_occurrence_scale', 1.0) for values in members.values()),
                    progress=(lambda phase, **facts: progress(phase, category=category, **facts)) if progress else None)
                if progress:
                    progress('library_publication', activity='Category geometry sharing complete', category=category,
                             submitted_source_count=evidence['submitted_source_count'],
                             native_processed_source_count=evidence['native_processed_source_count'],
                             changed_source_count=evidence['changed_source_count'],
                             known_native_exclusion_count=evidence['known_native_exclusion_count'],
                             referencing_occurrence_count=evidence.get('referencing_occurrence_count', 0),
                             authored_point_arrays_removed=evidence.get('authored_point_arrays_removed', 0))
                evidence['whole_definition_reuse'] = whole
                evidence['staging_layer'] = filename
                sharing[category] = evidence
                if not evidence['accepted']:
                    retained_failure = True
                    evidence['retained_staging_directory'] = str(staging)
                    failure = destination.parent / (destination.name + '.geometry-reuse-failure.json')
                    failure.write_text(json.dumps(sharing, indent=2) + '\n')
                    raise ValueError('category geometry sharing failed preservation; evidence: ' + str(failure))
                for identifier, (file, source, old, new, record) in members.items():
                    matching = record.setdefault('matching', {})
                    mapping = dict(record.get('path_map', {}))
                    mapping.update(matching.get('operation_path_map', {}))
                    # Include identity mappings, so newly introduced Geometry
                    # children remain traceable to each pre-optimization mesh.
                    for path in record['geometry_paths']:
                        mapping.setdefault(path, path)
                    for original, optimized in list(mapping.items()):
                        published_path = str(_prim_path(optimized).ReplacePrefix(old, new))
                        final = evidence['path_map'].get(published_path, [published_path])
                        if len(final) != 1:
                            raise ValueError('native sharing requires one geometry target per owned mesh')
                        mapping[original] = str(_prim_path(final[0]).ReplacePrefix(new, old))
                    matching['operation_path_map'] = mapping
                    record['path_map'] = mapping
        library_files = {p for p in staging.rglob('*') if p.is_file()}
        if consumer_layer:
            consumer_source = Path(consumer_layer).resolve()
            original = Sdf.Layer.FindOrOpen(str(consumer_source))
            consumer = Sdf.Layer.OpenAsAnonymous(str(consumer_source))
            if not original or not consumer or original.dirty:
                raise ValueError('consumer requires a saved readable layer')
            output = staging / 'after.usdc'
            consumer.subLayerPaths = [writer.asset(original, item, output) for item in original.subLayerPaths]
            writer.rewrite(consumer, original, output)
            if not consumer.Export(str(output)):
                raise ValueError('consumer export failed')
            composed = Usd.Stage.Open(str(output))
            if not composed or not composed.GetDefaultPrim() or composed.GetCompositionErrors():
                raise ValueError('consumer failed composed reopening')
            if progress:
                progress('library_publication', activity='Checking whole-consumer references and connections',
                         operation='consumer_reference_checks', entry_count=len(records))
            _check_unresolved(Usd.Stage.Open(original),composed)
        consumer_files = {p for p in staging.rglob('*') if p.is_file()} - library_files if common_library else set()
        published = {}
        # All category/dependency writes are complete. Reuse immutable files only
        # for this validation/index pass; entry scopes are still checked separately.
        published_stages, published_fingerprints = {}, {}
        def published_stage(path):
            key = str(Path(path).resolve())
            if key not in published_stages:
                published_stages[key] = Usd.Stage.Open(key)
            return published_stages[key]
        def published_fingerprint(path):
            key = str(Path(path).resolve())
            if key not in published_fingerprints:
                published_fingerprints[key] = file_fingerprint(key)
            return published_fingerprints[key]
        # Carry forward matching metadata and entry identity from the previous
        # version's index when extending a category. No second index is invented.
        for category, previous_file in existing_libraries.items():
            previous_file = Path(previous_file).resolve()
            prior_index = previous_file.parent / 'index.json'
            if category not in categories or not prior_index.is_file():
                continue
            prior, _ = _read_inherited_index(prior_index)
            for identifier, row in prior.get('entries', {}).items():
                if (prior_index.parent / row['layer']).resolve() != previous_file:
                    continue
                if identifier in entries:
                    raise ValueError('existing index identity cannot be replaced: ' + identifier)
                if file_fingerprint(previous_file) != row['fingerprint']:
                    raise ValueError('existing library no longer matches its index')
                output = staging / categories[category]
                stage = published_stage(output)
                if not stage.GetPrimAtPath(row['prim_path']) or stage.GetCompositionErrors():
                    raise ValueError('existing entry failed reopening in new version')
                published[identifier] = {**row, 'layer': categories[category],
                                         'fingerprint': published_fingerprint(output)}
        if progress:
            progress('library_publication', activity='Validating published entries', operation='entry_validation',
                     completed_count=0, entry_count=len(records))
        for entry_number, (identifier, (file, source, old, new, record)) in enumerate(records.items(), 1):
            entry = entries[identifier]
            output = writer.localized[str(file)] if entry.get('read_only') else staging / categories[entry['semantic_category']]
            path = old if entry.get('read_only') else new
            stage = published_stage(output)
            if not stage or not stage.GetPrimAtPath(path) or stage.GetCompositionErrors():
                raise ValueError('published definition failed composed reopening: ' + identifier)
            unresolved=_check_unresolved(Usd.Stage.Open(source),stage,old,path)
            matching = dict(record.get('matching', {}))
            if 'operation_path_map' not in matching and 'path_map' in record:
                matching['operation_path_map'] = record['path_map']
            published[identifier] = {'layer': writer.relative(output, staging / 'index.json'), 'prim_path': str(path),
                                    'semantic_category': entry['semantic_category'],
                                    'fingerprint': published_fingerprint(output),
                                    'source_fingerprint': record['fingerprint'],
                                    'matching': matching,
                                    'preexisting_unresolved_properties':unresolved}
            for field in ('assessment', 'operation_assessment'):
                if field in record:
                    published[identifier][field] = record[field]
            if progress and (entry_number == 1 or entry_number % 25 == 0 or entry_number == len(records)):
                progress('library_publication', activity='Validating published entries', operation='entry_validation',
                         category=entry['semantic_category'], completed_count=entry_number, entry_count=len(records))
        index = {'schema_version': 1, 'entries': published, 'status': 'published',
                 'publication': 'immutable_version', 'verification': 'composition_and_dependency_resolution'}
        index['portable'] = portable
        index['resource_dependency_closure'] = {
            'complete': not writer.opaque_material_resources,
            'external_mdl_modules': sorted(map(str,writer.opaque_material_resources)),
            'limitation': 'MDL imports and resources declared inside modules are not inspected; original module locations are retained.' if writer.opaque_material_resources else None}
        if geometry_reuse:
            index['geometry_reuse'] = sharing
        if consumer_layer and not common_library:
            index['consumer_layer'] = 'after.usdc'
        if material_library:
            delivered_materials = writer.localized.get(str(Path(material_library).resolve()))
            if delivered_materials is not None:
                index['material_library'] = writer.relative(delivered_materials, staging / 'index.json')
        index['dependencies'] = {str(path.relative_to(staging)): published_fingerprint(path)
                                 for path in sorted(staging.rglob('*')) if path.is_file() and path not in consumer_files}
        index['dependencies'].update({writer.relative(path, staging / 'index.json'): published_fingerprint(path) for path in sorted(writer.external)})
        from importlib.util import spec_from_file_location, module_from_spec
        spec = spec_from_file_location('published_compact_index', Path(__file__).with_name('persistent_library.py'))
        compact = module_from_spec(spec); spec.loader.exec_module(compact)
        compact.write_index(staging, index, {str(p.resolve()) for p in staging.rglob('*') if p.suffix in ('.usd', '.usda', '.usdc')}, stages=published_stages,
                            finalize_layers=[staging / filename for filename in categories.values()])
        if common_library:
            _publish_common(staging, destination, consumer_destination, index, writer, consumer_files)
        else:
            # Rename is the only publication mutation; failure leaves no visible index.
            staging.rename(destination)
        return index
    finally:
        if staging.exists() and not retained_failure:
            shutil.rmtree(staging)


def _publish_common(staging, destination, consumer_destination, index, writer, consumer_files):
    """Install unique files, then commit the combined index last (one publisher)."""
    destination.mkdir(parents=True, exist_ok=True)
    consumer_destination = Path(consumer_destination).resolve()
    if consumer_destination.is_relative_to(destination):
        raise ValueError('station consumer must be outside the common library')
    if consumer_destination.exists():
        raise ValueError('station consumer destination already exists')
    current = json.loads((staging / 'index.json').read_text())
    previous_path = destination / 'index.json'
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else None
    if previous and (previous.get('schema_version') != 2 or previous.get('status') != 'published'):
        raise ValueError('common library requires a published compact index')
    # Rebase current references to previously published files from staging to final.
    for row in current['libraries'].values():
        source = (staging / row['layer']).resolve()
        if not source.is_relative_to(staging):
            row['layer'] = os.path.relpath(source, destination).replace(os.sep, '/')
    for row in current['dependencies'].values():
        source = (staging / row['location']).resolve()
        if not source.is_relative_to(staging):
            row['location'] = os.path.relpath(source, destination).replace(os.sep, '/')
    # Compact identifiers derive from final relative paths, including reused files.
    library_ids = {key: hashlib.sha256(row['layer'].encode()).hexdigest()[:20]
                   for key, row in current['libraries'].items()}
    dependency_ids = {key: hashlib.sha256(row['location'].encode()).hexdigest()[:20]
                      for key, row in current['dependencies'].items()}
    for row in current['entries'].values(): row['library'] = library_ids[row['library']]
    for row in current['libraries'].values():
        row['dependencies'] = [dependency_ids[key] for key in row.get('dependencies', [])]
    current['libraries'] = {library_ids[key]: row for key, row in current['libraries'].items()}
    current['dependencies'] = {dependency_ids[key]: row for key, row in current['dependencies'].items()}
    combined = copy.deepcopy(previous or current)
    if previous:
        combined['libraries'].update(current['libraries'])
        combined['dependencies'].update(current['dependencies'])
        addresses = {(combined['libraries'][r['library']]['layer'], r['prim_path']) for r in combined['entries'].values()}
        for identifier, row in current['entries'].items():
            address = (current['libraries'][row['library']]['layer'], row['prim_path'])
            if address in addresses:
                continue
            if identifier in combined['entries']:
                identifier += '_' + hashlib.sha256(repr(address).encode()).hexdigest()[:16]
            combined['entries'][identifier] = row
            addresses.add(address)
    combined.pop('consumer_layer', None)
    combined.pop('material_library', None)
    if previous:
        combined['portable'] = previous.get('portable', False) and current.get('portable', False)
        closure = combined.setdefault('resource_dependency_closure', {})
        recent = current.get('resource_dependency_closure', {})
        closure['complete'] = closure.get('complete', False) and recent.get('complete', False)
        closure['external_mdl_modules'] = sorted(set(closure.get('external_mdl_modules', [])) | set(recent.get('external_mdl_modules', [])))
        if recent.get('limitation'): closure['limitation'] = recent['limitation']
    files = [p for p in staging.rglob('*') if p.is_file() and p.name not in ('index.json', 'after.usdc')]
    files = [p for p in files if p not in consumer_files]
    locations = {source: destination / source.relative_to(staging) for source in files}
    locations.update({source: (consumer_destination if source.name == 'after.usdc' else
                              consumer_destination.parent / source.relative_to(staging)) for source in consumer_files})
    for source, target in locations.items():
        writer.localized[str(source.resolve())] = target
    # Relocate station-only assembly layers along with the consumer. Library files
    # retain their staged relative layout; no finalized category crate is rewritten.
    for source in consumer_files:
        output = locations[source]
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists(): raise ValueError('station output already exists: ' + str(output))
        if source.suffix in ('.usd', '.usda', '.usdc'):
            original = Sdf.Layer.FindOrOpen(str(source))
            layer = Sdf.Layer.OpenAsAnonymous(str(source))
            layer.subLayerPaths = [writer.asset(original, value, output) for value in original.subLayerPaths]
            writer.rewrite(layer, original, output)
            if not layer.Export(str(output)): raise ValueError('station layer export failed')
        else:
            shutil.copyfile(source, output)
    for source in files:
        target = locations[source]
        target.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation avoids replacing an existing published asset.
        with target.open('xb') as output, source.open('rb') as input:
            shutil.copyfileobj(input, output)
    stage = Usd.Stage.Open(str(consumer_destination))
    if not stage or stage.GetCompositionErrors():
        raise ValueError('station consumer failed final-location reopening')
    fd, temporary = tempfile.mkstemp(prefix='.index-', suffix='.json', dir=destination)
    try:
        with os.fdopen(fd, 'w') as output:
            output.write(json.dumps(combined, separators=(',', ':')) + '\n')
        os.replace(temporary, destination / 'index.json')
    finally:
        Path(temporary).unlink(missing_ok=True)
    # The workflow receipt is station-specific; the persistent index is cumulative.
    index['consumer_layer'] = os.path.relpath(consumer_destination, destination).replace(os.sep, '/')
    index['publication'] = 'common_library'
    for source in consumer_files:
        output = locations[source]
        index['dependencies'][os.path.relpath(output, destination).replace(os.sep, '/')] = file_fingerprint(output)
