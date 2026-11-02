# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Join recorded definition decisions and execution receipts; never inspect USD."""
from collections import Counter, defaultdict
import importlib.util
from pathlib import Path


def _operations(receipts):
    # Preserve receipt order, including failed and unchanged attempts.
    return [{'order': order, 'name': operation.get('operation', 'Unknown operation'),
             'status': operation.get('status', 'executed' if operation.get('success') else 'failed'),
             'reason': operation.get('reason', ''), 'seconds': operation.get('seconds')}
            for order, operation in enumerate(receipts, 1)]


def build_asset_results(manifest, results):
    records = {row['definition_id']: row for row in results}
    occurrences = defaultdict(list)
    for row in manifest.get('occurrences', []):
        occurrences[row['definition_id']].append(row['path'])
    assets = []
    for definition in manifest.get('definitions', []):
        identifier = definition['id']
        record = records.get(identifier, {})
        path = definition.get('source_prim_path', definition.get('boundary_path', ''))
        selection = definition.get('selection', {})
        operations = _operations(record.get('operations', []))
        assessment = record.get('assessment', {})
        retained = [{'category': row.get('category', ''), 'reason': row.get('reason', '')}
                    for row in assessment.get('categories', [])
                    if row.get('status') not in ('executed',) and row.get('reason')]
        assets.append({
            **({'equipment_family': definition['equipment_family']} if definition.get('equipment_family') else {}),
            **({'product_identity':definition['product_identity']} if definition.get('product_identity') else {}),
            'definition_id': identifier,
            'name': definition.get('display_name') or path.rsplit('/', 1)[-1] or identifier,
            'source_prim_path': path,
            'semantic_category': definition.get('semantic_category', 'unclassified'),
            'structural_type': definition.get('structural_type', 'not recorded'),
            'boundary_reason': selection.get('reason', ''),
            'structure_reason': definition.get('rationale', {}).get('reason', ''),
            'independent_use': selection.get('independent_use', ''),
            'occurrences': sorted(set(occurrences.get(identifier, definition.get('occurrences', [])))),
            'children': definition.get('children', []),
            'reused_from_library': bool(record.get('persistent_reuse') or definition.get('persistent_entry')),
            'disposition': record.get('disposition', 'incomplete'),
            'before': record.get('before', {}), 'after': record.get('after', {}),
            'operations': operations, 'retained_geometry': retained,
            'prior_operations': _operations(record.get('prior_native_operations', [])),
        })
    return assets


def summarize_report(report, manifest, results, verification):
    """Call after measurements are finalized. Unknown measurements stay unknown."""
    report['asset_results'] = build_asset_results(manifest, results)
    counts = Counter(asset['disposition'] for asset in report['asset_results'])
    spec = importlib.util.spec_from_file_location('summary_validation', Path(__file__).with_name('validate_report.py'))
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    resolved = validator.PHASE4_RESOLVED_DISPOSITIONS
    passed = bool(verification.get('passed'))
    completion = ('Processing completed; assembly verification passed.' if passed else
                  'Processing or verification is incomplete.')
    report['executive_summary'] = completion
    failure_reason = report.get('reasoning', '') if not passed else ''
    report['reasoning'] = (f"{len(report['asset_results']):,} asset definitions: "
        f"{counts['optimized']:,} optimized, {counts['verified_unchanged']:,} verified unchanged, "
        f"{counts['exact_cache_hit']:,} exact reuse, {counts['normalized_cache_hit']:,} normalized reuse, {counts['product_identity_hit']:,} product reuse, "
        f"{counts['skipped_zero_meshes'] + counts['skipped_user_declined']:,} skipped, "
        f"{sum(n for state, n in counts.items() if state not in resolved):,} incomplete. "
        'Each asset below includes the recorded boundary decision and operation outcomes.')
    if failure_reason:
        report['reasoning'] = failure_reason + ' ' + report['reasoning']
    pending = verification.get('decisions_required', [])
    instructions = []
    for decision in pending:
        reason = decision.get('reason', '')
        identifiers = ', '.join(decision.get('definition_ids', []))
        if reason:
            instructions.append(f"Review {decision.get('category', 'the recorded finding')} for {identifiers}: {reason}")
    report['report_summary'] = {
        'completion': completion,
        'checks': ['Definition and assembly verification: ' + ('passed.' if passed else 'incomplete; see verification.json.')],
        'limitations': ['Native USD Optimize operations were trusted within their configured scopes and settings. '
                        'No independent surface-distance or rendered-appearance comparison was performed.',
                        'RAM, GPU memory and rendering performance were not measured.'],
        'next_run_instructions': instructions,
    }

    if report.get('measurement_context', {}).get('storage_coverage') == 'partial':
        report['report_summary']['limitations'].append('Storage coverage is incomplete; the numeric storage score is an unmeasured placeholder.')
    if any(metric.get('name') == 'warm_open_ms' for metric in report.get('metrics', [])):
        report['report_summary']['limitations'].append('Stage-open timings use standalone USD and warm-open samples; the first open is not guaranteed OS-cold.')
