# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render incomplete workflow evidence without inventing an accepted output.

Usage:
    finalize(tuning, manifest, directory, result), called on workflow failure.
    This module has no command-line interface.
Arguments:
    Loaded usd_tuning helpers, the assessed manifest, output directory and failed
    workflow result, including available operation outcomes and errors.
Output:
    Writes report.json, report.md, report.html and workflow-results.json. Returns
    the result with report paths and completion explicitly marked false.
Exit codes:
    Not applicable to this imported module. Invalid report content raises
    ValueError; rendering and file-access errors propagate.
"""
import copy
from pathlib import Path


def finalize(tuning, manifest, directory, result):
    directory = Path(directory)
    working = copy.deepcopy(manifest)
    working.setdefault('definitions', [])
    working['output'] = {'path': 'Unavailable: no accepted optimized USD was produced'}
    rows = result.get('definition_results', result.get('compute', {}).get('definition_results', []))
    status = result.get('status', 'incomplete')
    verification = {'passed': False, 'errors': result.get('errors', [status]),
                    'checks': [{'name': 'workflow completion', 'passed': False, 'evidence': status}]}
    report = tuning.build_report(working, rows, verification, directory, persist_evidence=False)
    candidate = result.get('after_layer')
    if candidate and Path(candidate).is_file():
        report['reasoning'] = 'Incomplete: ' + status.replace('_', ' ') + '. A candidate output was written.'
        report['notes'] += ' Candidate available for inspection: ' + str(candidate) + '. See verification.json for completed checks.'
    else:
        report['reasoning'] = 'Incomplete: ' + status.replace('_', ' ') + '. No validated optimized output is available.'
    report['input_path'] = working.get('original_source', working['source'])['path']
    report['notes'] += (' Candidate operation results are diagnostic trials, not accepted savings. '
                        'See workflow-results.json for exact errors, scopes and retained candidate outputs. '
                        'After-stage performance and preservation were not qualified.')
    for operation in report['operations']:
        operation['result'] = str(operation.get('result') or 'executed') + '; candidate result; final validation incomplete'
    report['target_coverage']['complete'] = False
    result['report'] = dict(tuning.finalize_report(report, working, rows, verification, directory), complete=False)
    tuning.write(directory / 'workflow-results.json', result)
    return result
