# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plan isolated definition computation and single-writer category publication.

This module constructs dependencies. It neither infers asset boundaries nor
claims that a computed definition or published layer has passed verification.
"""
from collections import defaultdict
from pathlib import Path
import hashlib
import re


def _token(value):
    readable = re.sub(r'[^a-zA-Z0-9_]', '_', value).strip('_')[:48] or 'entry'
    return readable + '_' + hashlib.sha256(value.encode()).hexdigest()[:12]


def category_name(value):
    if value is None or isinstance(value, str) and value.strip() == 'unknown':
        return 'general'
    if not isinstance(value, str) or not value.strip():
        raise ValueError('semantic_category must be a nonempty string or unknown')
    return value.strip()


def publication_plan(definitions, directory):
    """Return tool-owned paths and an acyclic graph without modifying inputs.

    Children describe composition, not automatic compute prerequisites. Only
    compute_requires adds those prerequisites. Existing entries are read-only
    and have no compute/publication job. Publication copies authored specs and
    remaps stable references; it must not read composed child geometry.
    The root consumer is authored separately from these library definitions.
    """
    directory = Path(directory)
    indexed = {}
    for definition in definitions:
        identifier = definition.get('id')
        if not isinstance(identifier, str) or not identifier:
            raise ValueError('definition requires id')
        if identifier in indexed:
            raise ValueError('duplicate definition id: ' + identifier)
        indexed[identifier] = definition
    if not indexed:
        raise ValueError('publication requires definitions')
    for identifier, definition in indexed.items():
        for field in ('children', 'compute_requires'):
            values = definition.get(field, [])
            if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
                raise ValueError(field + ' must contain definition IDs')
            if len(set(values)) != len(values) or any(v not in indexed for v in values):
                raise ValueError('duplicate or missing ' + field + ' for ' + identifier)
    # Cyclic composition cannot be made valid by the scheduler.
    _check_graph({key: set(value.get('children', [])) for key, value in indexed.items()})
    entries, categories, jobs = {}, defaultdict(list), []
    for identifier, definition in sorted(indexed.items()):
        category = category_name(definition.get('semantic_category'))
        token = _token(identifier)
        reused = definition.get('persistent_entry')
        if reused is not None:
            if not isinstance(reused, dict) or any(not reused.get(k) for k in ('layer', 'prim_path', 'fingerprint')):
                raise ValueError('persistent_entry requires layer, prim_path and fingerprint')
            if reused.get('confirmed') is not True or not reused.get('confirmation_fingerprint') or reused.get('status') == 'confirmation_required':
                raise ValueError('persistent_entry requires confirmed reuse and confirmation_fingerprint')
            entries[identifier] = {**reused, 'semantic_category': category, 'read_only': True}
            if 'persistent_matching' in definition:
                entries[identifier]['matching'] = definition['persistent_matching']
            continue
        categories[category].append(identifier)
        entries[identifier] = {
            'semantic_category': category, 'read_only': False,
            'working_layer': str(directory / 'working' / (token + '.usdc')),
            'layer': str(directory / 'libraries' / (_token(category) + '.usdc')),
            'prim_path': '/Library/D_' + token,
        }
    for identifier, definition in sorted(indexed.items()):
        entry = entries[identifier]
        if definition.get('equipment_family') and not entry['read_only']:
            entry['equipment_family'] = definition['equipment_family']
        if entry['read_only']:
            continue
        jobs.append({'id': 'compute:' + identifier, 'kind': 'compute',
                     'definition_ids': [identifier],
                     'depends_on': sorted('compute:' + c for c in definition.get('compute_requires', [])
                                          if not entries[c]['read_only']),
                     'write_layers': [entry['working_layer']]})
    for category, members in sorted(categories.items()):
        jobs.append({'id': 'publish:' + category, 'kind': 'publish',
                     'definition_ids': members,
                     'depends_on': ['compute:' + c for c in members],
                     'write_layers': [entries[members[0]]['layer']]})
    # All libraries must be available before opening a cross-category consumer.
    publication_ids = ['publish:' + category for category in sorted(categories)]
    for identifier in sorted(indexed):
        jobs.append({'id': 'verify:' + identifier, 'kind': 'verify',
                     'definition_ids': [identifier],
                     'depends_on': publication_ids,
                     'write_layers': []})
    jobs.append({'id': 'verify:consumer', 'kind': 'verify_consumer',
                 'definition_ids': [],
                 'depends_on': ['verify:' + identifier for identifier in sorted(indexed)],
                 'write_layers': []})
    if len({job['id'] for job in jobs}) != len(jobs):
        raise ValueError('definition ID conflicts with reserved consumer job')
    _check_graph({job['id']: set(job['depends_on']) for job in jobs})
    writes = [path for job in jobs for path in job['write_layers']]
    if len(set(writes)) != len(writes):
        raise ValueError('multiple writers for one layer')
    return {'schema_version': 1, 'entries': entries, 'jobs': jobs}


def _check_graph(dependencies):
    remaining = {key: set(value) for key, value in dependencies.items()}
    while remaining:
        ready = {key for key, value in remaining.items() if not value}
        if not ready:
            raise ValueError('cyclic or missing dependencies')
        remaining = {key: value - ready for key, value in remaining.items() if key not in ready}
