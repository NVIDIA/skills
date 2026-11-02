#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render inexpensive progress and accepted decision summaries; never inspect USD.

Usage:
    Use Progress(manifest, directory) as a context manager in successor_workflow.
    Send phase events to event() and the final outcome to finish(). No CLI.
Arguments:
    The assessed manifest, output directory and workflow event dictionaries.
Output:
    Writes progress summaries and steering.md in the run directory; preserves
    existing user edits to steering.md. Reads worker receipts without opening USD.
Exit codes:
    Not applicable. Progress snapshot I/O failures are logged and retried on the next update.
"""
import collections
import json
import importlib.util
import logging
from pathlib import Path
import threading
import time



# One presentation map for Markdown, snapshots and the browser. Unlisted tool
# substeps remain within the current workflow phase.
STAGES = [('assessment', 'Assess asset structure'), ('source_validation', 'Validate source'),
          ('source_profile', 'Profile source'), ('author_inputs', 'Prepare assets'),
          ('persistent_matching', 'Confirm library reuse'), ('definition_compute', 'Optimize assets'),
          ('library_publication', 'Publish libraries'), ('composed_verification', 'Verify assembly'),
          ('output_profile', 'Profile output'), ('report', 'Report results')]
PHASE_STAGE = {key: key for key, _ in STAGES}
PHASE_STAGE.update(starting='assessment', product_confirmation='author_inputs',
    product_confirmation_complete='author_inputs', definition_compute_complete='definition_compute',
    library_publication_complete='library_publication', composed_verification_complete='composed_verification',
    complete='report')
ACTIVITIES = {
    'assessment': 'Starting assessment', 'stage_open': 'Opening the selected source stage',
    'scan_stage': 'Collecting source hierarchy and boundary evidence',
    'source_evidence_reused': 'Reusing collected source evidence',
    'boundary_review': 'Preparing candidate boundary groups',
    'boundary_review_complete': 'Candidate groups ready for agent review',
    'inventory': 'Inspecting geometry and material assignments',
    'exact_confirmation': 'Confirming candidate reuse',
    'assessment_complete': 'Asset assessment complete',
    'complete': 'Execution finished',
}
PHASES = {key: (title, 'Continue the recorded workflow') for key, title in STAGES}
PHASES.update({key: (title, 'Continue the recorded workflow') for key, title in ACTIVITIES.items()})


def read_state(path):
    """Read only a compact snapshot; never load a manifest or USD for progress."""
    try:
        path = Path(path)
        if path.stat().st_size > 8 * 1024 * 1024:
            return {}
        value = json.loads(path.read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def counter(phase, facts, workers):
    if phase == 'source_evidence_reused':
        return None
    for done, total, unit in [('prims_done', 'prims_total', 'prims visited in this activity'),
                             ('completed_count', 'definition_count', 'assets processed'),
                             ('completed', 'total', 'items processed')]:
        if isinstance(facts.get(done), (int, float)):
            return {'done': facts[done], 'total': facts.get(total), 'unit': unit}
    overall = workers.get('overall', {})
    if phase in ('persistent_matching', 'definition_compute', 'definition_compute_complete') and overall:
        return {'done': sum(overall.get(k, 0) for k in ('done', 'failed', 'timeout', 'blocked')),
                'total': overall.get('total'), 'unit': 'jobs finished'}
    return None


def decision_rows(manifest):
    groups = collections.Counter()
    examples = {}
    for definition in manifest.get('definitions', []):
        selection, rationale = definition.get('selection', {}), definition.get('rationale', {})
        key = (definition.get('semantic_category') or 'general',
               definition.get('structural_type') or 'Structure not yet assigned',
               selection.get('reason') or rationale.get('reason') or 'Reason not recorded',
               selection.get('independent_use') or '', definition.get('equipment_family') or '')
        groups[key] += 1
        examples.setdefault(key, definition.get('boundary_path', definition.get('id', '')))
    return [dict(category=k[0], structure=k[1], reason=k[2], independent_use=k[3], family=k[4],
                 count=n, example=examples[k]) for k, n in groups.most_common()]


def decision_summary(manifest, limit=12):
    groups = collections.Counter()
    examples = {}
    for definition in manifest.get('definitions', []):
        selection = definition.get('selection', {})
        rationale = definition.get('rationale', {})
        reason = selection.get('reason') or rationale.get('reason') or 'Reason not recorded; review required.'
        key = (definition.get('semantic_category', 'general'), definition.get('structural_type', 'unclassified'),
               selection.get('basis', 'recorded'), reason, selection.get('independent_use', ''), definition.get('equipment_family'))
        groups[key] += 1
        examples.setdefault(key, definition.get('boundary_path', definition.get('id', '')))
    lines = []
    for key, count in groups.most_common(limit):
        category, structure, basis, reason, use, family = key
        lines.append(f'- **{category}: {structure}** — {count} definition(s); {basis}. {reason}'
                     + (f' Family: {family}.' if family else '')
                     + (f' Independent use: {use}.' if use else '')
                     + f' Example: `{examples[key]}`.')
    if len(groups) > limit:
        lines.append(f'- {len(groups)-limit} further decision groups are recorded in the assessment manifest.')
    return '\n'.join(lines) or '- No accepted definition decisions are available yet.'


def operation_decisions(manifest, limit=8):
    choices = collections.Counter()
    for definition in manifest.get('definitions', []):
        for row in definition.get('assessment', {}).get('operations', []):
            if isinstance(row, dict):
                choices[(row.get('category', 'operation'), row.get('status', 'unrecorded'), row.get('reason', 'See the recorded operation scope.'))] += 1
    lines = [f'- {category} ({status}): {reason} ({count} definition assessment(s)).'
             for (category, status, reason), count in choices.most_common(limit)]
    if len(choices) > limit:
        lines.append(f'- {len(choices)-limit} further operation groups are recorded in the authored manifest.')
    return '\n'.join(lines)


class Progress:
    def __init__(self, manifest, directory, *, assessment=False, common_library=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        self.started_epoch = time.time()
        self.assessing = assessment
        self.stage = 'assessment' if assessment else 'source_validation'
        self.phase_snapshots = {}
        self.events = []
        self.rounds = []
        self.rows = []
        self.common_library = str(Path(common_library).resolve()) if common_library else None
        self.phase_started = self.started
        self.assessment_pass = None
        self.source_evidence_reused = False
        self.phase = 'starting'
        self.state = 'running'
        self.facts = {}
        self.workers = {}
        self.active_operations = []
        self.stop = threading.Event()
        self.lock = threading.RLock()
        self.decisions = decision_summary(manifest)
        self.operation_choices = operation_decisions(manifest)
        self.source = manifest.get('original_source', manifest.get('source', {})).get('path', 'unspecified')
        previous = read_state(self.directory/'progress.json')
        if not previous and manifest.get('workspace'):
            previous = read_state(Path(manifest['workspace'])/'progress.json')
        if previous.get('source') == self.source:
            self.started_epoch = previous.get('started_at', self.started_epoch)
            self.phase_snapshots = previous.get('phases', {})
            self.events = previous.get('events', [])
            self.rounds = previous.get('assessment_rounds', [])
            self.rows = previous.get('decisions', [])
            self.decisions = previous.get('decisions_markdown', self.decisions)
            if assessment:
                self.phase_started = self.started - max(0, time.time()-self.started_epoch)
        if manifest.get('definitions'):
            self.rows = decision_rows(manifest)
            self.decisions = decision_summary(manifest, limit=len(manifest['definitions']))
        request = manifest.get('request', {})
        path = Path(__file__).resolve().parents[1]/'references/geometry_tolerance.py'
        spec = importlib.util.spec_from_file_location('progress_geometry_budget', path)
        budget = importlib.util.module_from_spec(spec); spec.loader.exec_module(budget)
        self.requirements = f"Workload: {request.get('workload', 'factory visualization')}. Native tolerance: {budget.physical_tolerance_mm(request)} mm."
        choices = [f'{key}={request[key]}' for key in ('preservation', 'preservation_profile', 'lossless',
                   'exact_geometry', 'fit_primitives', 'decimate', 'preserve_normals', 'preserve_uvs') if key in request]
        if choices:
            self.requirements += ' Recorded constraints: '+', '.join(choices)+'.'
        protected = set(request.get('protected_paths', []))
        protected.update(path for definition in manifest.get('definitions', []) for path in definition.get('protected_paths', []))
        self.scope = f"Accepted scope: {len(manifest.get('definitions', []))} unique definitions; {len(protected)} protected interface paths."
        if protected:
            self.scope += ' Examples: '+', '.join('`'+path+'`' for path in sorted(protected)[:3])+'.'
        self.scope += ' Keep the complete scope and constraints in [the authored manifest](authored-manifest.json) unless explicitly revised.'
        self.steering = self.directory/'steering.md'
        try:
            with self.steering.open('x', encoding='utf-8') as stream:
                stream.write('# Steer the next run\n\n'
                    f'Optimize `{self.source}`. {self.requirements}\n'+self.scope+'\n'
                    'Use the selected variant configuration. Preserve authored animation, materials, '
                    'required interfaces, source layers and inherited libraries.\n\n'
                    '## Decisions to keep or revise\n\n'+self.decisions+'\n\n'
                    '## Changes for the next run\n\n'
                    '- [Edit: name the asset group, desired change and reason.]\n\n'
                    'Review this text and submit it as the next prompt. Edits here are preserved; '
                    'the running workflow does not execute this file.\n')
        except FileExistsError:
            pass
        self.render()
        self.thread = threading.Thread(target=self._watch, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, kind, error, traceback):
        if error is not None:
            self.finish(False, 'interrupted' if issubclass(kind, (KeyboardInterrupt, SystemExit)) else 'failed', str(error) or kind.__name__)
        self.stop.set()
        self.thread.join(timeout=2)
        self.render()

    def event(self, phase, **facts):
        with self.lock:
            if 'assessment_pass' in facts:
                self.assessment_pass = facts['assessment_pass']
            if phase == 'source_evidence_reused':
                self.source_evidence_reused = True
            if phase != self.phase:
                if phase in ('persistent_matching', 'definition_compute'):
                    self.workers = {}
                    self.active_operations = []
            if not self.assessing and phase in PHASE_STAGE and PHASE_STAGE[phase] != self.stage:
                self.snapshot()  # Close the preceding phase at this transition.
            self.phase, self.facts = phase, facts
            if not self.assessing and phase in PHASE_STAGE:
                stage = PHASE_STAGE[phase]
                if stage != self.stage:
                    self.phase_started = time.monotonic()
                self.stage = stage
            row = {'phase': phase, 'activity': ACTIVITIES.get(phase, phase.replace('_', ' ')),
                   'time': time.time(), 'facts': facts}
            if self.events and self.events[-1]['phase'] == phase:
                self.events[-1] = row
            else:
                self.events.append(row)
                self.events = self.events[-100:]
            if self.assessing and self.assessment_pass is not None:
                if not self.rounds or self.rounds[-1]['number'] != self.assessment_pass:
                    self.rounds.append({'number': self.assessment_pass, 'steps': [], 'status': 'running'})
                steps = self.rounds[-1]['steps']
                item = dict(row, counter=counter(phase, facts, {}))
                if steps and steps[-1]['phase'] == phase:
                    steps[-1] = item
                else:
                    steps.append(item)
            self.render()

    def finish(self, passed, status, error=''):
        with self.lock:
            self.state = ('assessed' if passed else 'waiting') if self.assessing and status in ('assessed', 'boundary review required') else ('passed' if passed else status or 'incomplete')
            if self.assessing and self.rounds:
                self.rounds[-1]['status'] = self.state
            self.facts = {**self.facts, 'passed': passed, 'status': status}
            if error:
                self.facts['error'] = error
            self.render()

    def _watch(self):
        stamp = None
        last = time.monotonic()
        while not self.stop.wait(1):
            try:
                folder = 'persistent-matching' if self.stage == 'persistent_matching' else 'compute'
                summary = self.directory/folder/'status.summary.json'
                current = (folder, summary.stat().st_mtime_ns) if summary.exists() else None
                changed = current != stamp
                if changed:
                    data = json.loads(summary.read_text())
                    with self.lock:
                        self.workers = data
                    stamp = current
                operations = []
                for job in self.workers.get('running_jobs', []):
                    try:
                        operation = json.loads(Path(job['progress_path']).read_text())
                        if operation.get('state') == 'running':
                            operations.append(dict(operation, definition_id=job['id']))
                    except (OSError, ValueError):
                        continue
                with self.lock:
                    changed = changed or operations != self.active_operations
                    self.active_operations = operations
                if changed or time.monotonic()-last >= 30:
                    self.render()
                    last = time.monotonic()
            except (OSError, ValueError):
                # An incomplete status write must not change the execution verdict.
                continue

    def update_decisions(self, manifest):
        with self.lock:
            self.rows = decision_rows(manifest)
            self.decisions = decision_summary(manifest, limit=max(1, len(manifest.get('definitions', []))))
            self.operation_choices = operation_decisions(manifest)

    def snapshot(self):
        now = time.time()
        workers = self.workers if self.stage in ('persistent_matching', 'definition_compute') else {}
        result = {'schema_version': 1, 'updated_at': now, 'started_at': self.started_epoch,
                  'source': self.source, 'state': self.state, 'phase': self.phase, 'stage': self.stage,
                  'activity': ACTIVITIES.get(self.phase, PHASES.get(self.phase, (self.phase.replace('_', ' '),))[0]),
                  'requirements': self.requirements, 'elapsed_seconds': now-self.started_epoch,
                  'tool_elapsed_seconds': time.monotonic()-self.started,
                  'phase_elapsed_seconds': time.monotonic()-self.phase_started,
                  'facts': self.facts, 'counter': counter(self.stage if workers else self.phase, self.facts, workers),
                  'workers': workers, 'active_operations': self.active_operations if workers else [],
                  'assessment_pass': self.assessment_pass, 'assessment_rounds': self.rounds,
                  'decisions': self.rows, 'decisions_markdown': self.decisions,
                  'events': self.events, 'common_library': self.common_library}
        # Phase summaries contain facts, never copies of history or decisions.
        self.phase_snapshots[self.stage] = {key: result[key] for key in
            ('stage', 'phase', 'state', 'activity', 'updated_at', 'elapsed_seconds',
             'phase_elapsed_seconds', 'facts', 'counter', 'workers', 'active_operations')}
        result['phases'] = self.phase_snapshots
        return result

    def render(self):
        # Dashboard readers can temporarily lock a snapshot on Windows. Keep
        # the previous complete snapshot and retry at the next normal update.
        with self.lock:
            try:
                self._render()
            except OSError as error:
                if not getattr(self, '_snapshot_write_failed', False):
                    logging.getLogger(__name__).warning('Progress snapshot update failed; workflow continues: %s', error)
                self._snapshot_write_failed = True
            else:
                self._snapshot_write_failed = False

    def _render(self):
        with self.lock:
            snapshot = self.snapshot()
            target = self.directory/'progress.json'
            temporary = target.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(snapshot, ensure_ascii=False), encoding='utf-8')
            temporary.replace(target)
            now = time.monotonic()
            activity, following = snapshot['activity'], 'Continue the recorded workflow'
            if self.phase == 'author_inputs' and self.facts.get('reused_authored_inputs'):
                activity = 'Reusing authored materials and definition inputs'
            if self.state not in ('running', 'passed', 'waiting', 'assessed'):
                following = 'Review the unresolved work and edit steering.md before retrying'
            elif self.state in ('passed', 'assessed'):
                following = 'Review the recorded result before continuing'
            lines = ['# Optimization progress', '', f'**Status: {self.state}.** {activity}.',
                f"Elapsed including assessment and agent review: {snapshot['elapsed_seconds']:.1f} s; current phase: {snapshot['phase_elapsed_seconds']:.1f} s.",
                f'Next: {following}.', '', self.requirements, '', f'Source: `{self.source}`.']
            if self.assessment_pass is not None:
                lines += ['', f'Assessment pass: {self.assessment_pass}.']
            if self.source_evidence_reused:
                lines += ['Source evidence reused; boundary-dependent results are recomputed.']
            if self.phase == 'scan_stage' and 'prims_done' in self.facts:
                total = self.facts.get('prims_total')
                lines += ['', f'Stage prims visited: {self.facts["prims_done"]:,}'
                          + (f' / {total:,}.' if total is not None else '; total not yet known.')]
            if self.facts:
                lines += ['', 'Current result: '+ '; '.join(f"{key.replace('_', ' ')}: {str(value)[:400] if value is not None else 'not reported'}" for key,value in self.facts.items())+'.']
            counts = snapshot['workers'].get('overall', {})
            if counts:
                lines += ['', '## Definition '+('preparation' if self.phase == 'persistent_matching' else 'optimization')+' jobs', '',
                    '; '.join(f'{name}: {counts.get(name, 0)}' for name in ('total', 'done', 'running', 'queued', 'failed', 'timeout', 'blocked'))+'.']
            if self.active_operations and self.state == 'running' and self.stage == 'definition_compute':
                lines += ['', 'Native operations currently running:', '']
                lines += [f"- {row['operation']}: {max(0, time.time()-row['started_epoch']):.1f} s elapsed; definition `{row['definition_id']}`."
                          for row in self.active_operations]
            last = self.workers.get('last_operation')
            if last:
                lines += ['', f'Last completed native operation: {last}.']
            outcomes = self.workers.get('operation_outcomes', {})
            if outcomes:
                lines += ['', 'Observed native operation outcomes (completed jobs):', '']
                lines += [f'- {operation}: '+', '.join(f'{state} {count}' for state,count in states.items())+'.'
                          for operation,states in sorted(outcomes.items())]
            reasons = self.workers.get('unchanged_reasons', {})
            if reasons:
                lines += ['', 'Recorded reasons for unchanged work:', '']+[f'- {reason} ({count} definition outcome(s)).' for reason,count in reasons.items()]
            failures = self.workers.get('failures', [])
            if failures:
                lines += ['', 'Failures requiring attention:', '']+[f'- {row}' for row in failures[:5]]
            if self.operation_choices:
                lines += ['', '## Planned operation choices', '', self.operation_choices]
            lines += ['', '## Accepted asset decisions', '', self.decisions, '',
                'These are recorded requirements and engineering judgments; operation counts above are observed results.', '',
                '## Steer another run', '', '[Edit the next-run prompt](steering.md). Progress updates preserve your edits.', '',
                'Details: [assessment events](assessment-progress.jsonl), [execution events](workflow-progress.jsonl), [worker status](compute/status.json), '
                '[authored manifest](authored-manifest.json), [final report](report.md). '
                'Files appear as their phases complete.', '']
            target = self.directory/'progress.md'
            temporary = target.with_suffix('.md.tmp')
            temporary.write_text('\n'.join(lines), encoding='utf-8')
            temporary.replace(target)
