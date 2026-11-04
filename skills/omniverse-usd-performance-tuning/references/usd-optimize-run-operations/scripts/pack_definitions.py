# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Batch small independent definitions without changing their logical boundaries."""


PACK_REMAP_FIELDS = ('boundary_path', 'owned_geometry', 'protected_paths', 'internal_paths', 'operation_scope',
    'merge_groups', 'variant_context', 'operations')


def packed_prim_path(definition_id):
    return '/Library/D_' + definition_id.replace('definition-', '').replace('-', '_')


def _remap_record(value, prefix):
    if isinstance(value, str):
        return prefix + value[6:] if value == '/Asset' or value.startswith('/Asset/') else value
    if isinstance(value, list):
        return [_remap_record(item, prefix) for item in value]
    if isinstance(value, dict):
        return {_remap_record(key, prefix): _remap_record(item, prefix) for key, item in value.items()}
    return value


def remap_worker_target(target, prim_path):
    target = dict(target)
    for field in PACK_REMAP_FIELDS:
        if field in target:
            target[field] = _remap_record(target[field], prim_path)
    return target


def apply_packed_assignment(definition, input_path, output_path, prim_path, reason):
    definition.update(input_layer=str(input_path), output_layer=str(output_path), output_prim_path=prim_path)
    definition['packaging'] = {'layer': str(output_path), 'prim_path': prim_path, 'reason': reason}
    target = remap_worker_target(definition['worker_target'], prim_path)
    target.update(input_layer=str(input_path), output_layer=str(output_path), output_prim_path=prim_path)
    definition['worker_target'] = target
