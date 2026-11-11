#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Serve read-only recorded optimization progress on localhost; no USD imports.

Usage: python progress_dashboard.py --workspace RUN [--execution RUN/execution]
       [--assessment RUN/assessment] [--common-library DIRECTORY] [--port 8765]
The server reads compact progress snapshots and existing logs. It never starts,
changes or stops optimization. Ctrl-C closes only the dashboard.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import threading
import time
from urllib.parse import urlsplit

from successor_progress import ACTIVITIES, PHASE_STAGE, STAGES, counter

ASSETS = Path(__file__).resolve().parents[1]/'assets/progress-dashboard'
THEME = Path(__file__).resolve().parents[1]/'references/report-templates/theme.css'
TERMINAL = {'passed', 'failed', 'interrupted', 'incomplete', 'execution_exception'}


class RunView:
    def __init__(self, workspace, execution=None, assessment=None, common_library=None):
        self.workspace = Path(workspace).resolve()
        self.execution = Path(execution).resolve() if execution else None
        self.assessment = Path(assessment).resolve() if assessment else None
        self.common_library = Path(common_library).resolve() if common_library else None
        self.cache = {}
        self.files = {}
        self.lock = threading.RLock()

    def read(self, path, kind='json'):
        path = Path(path)
        try:
            stat = path.stat()
            if kind != 'jsonl' and stat.st_size > 8*1024*1024:
                return None
            stamp = (stat.st_mtime_ns, stat.st_size)
            if path not in self.cache or self.cache[path][0] != stamp:
                if kind == 'jsonl':
                    # Large old logs must still expose their final completion.
                    # Keep only a bounded tail; detailed history stays on disk.
                    with path.open('rb') as stream:
                        if stat.st_size > 1024*1024:
                            stream.seek(-1024*1024, 2)
                            stream.readline()
                        text = stream.read().decode('utf-8', errors='replace')
                else:
                    text = path.read_text(encoding='utf-8')
                if kind == 'jsonl':
                    value = []
                    for line in text.splitlines():
                        try: value.append(json.loads(line))
                        except ValueError: continue  # A final event can still be being written.
                elif kind == 'text': value = text
                else: value = json.loads(text)
                self.cache[path] = (stamp, value)
            return self.cache[path][1]
        except (OSError, ValueError):
            return None

    def _recorded_state(self, directory):
        """Read pre-dashboard outputs through their existing compact event log.

        This compatibility reader never writes a snapshot or infers completion
        from a report's presence. The workflow's completion event owns its verdict.
        """
        path = directory/'workflow-progress.jsonl'
        events = self.read(path, 'jsonl')
        if not events:
            return None
        stage, phases = 'source_validation', {}
        for event in events:
            phase = event['phase']
            stage = PHASE_STAGE.get(phase, stage)
            row = {'stage': stage, 'phase': phase, 'activity': ACTIVITIES.get(phase, phase.replace('_', ' ')),
                   'facts': {k: v for k, v in event.items() if k not in ('phase', 'elapsed_seconds')},
                   'elapsed_seconds': event.get('elapsed_seconds'), 'state': 'recorded'}
            phases[stage] = row
        last = events[-1]
        state = ('passed' if last.get('passed') else last.get('status', 'incomplete')) if last['phase']=='complete' else 'recorded'
        notes = self.read(directory/'progress.md', 'text') or ''
        source = re.search(r'^Source: `(.*?)`', notes, re.M)
        decisions = notes.split('## Accepted asset decisions\n', 1)[-1].split('\nThese are recorded', 1)[0].strip() if '## Accepted asset decisions' in notes else ''
        if decisions:
            phases['assessment'] = {'stage': 'assessment', 'phase': 'assessment_complete',
                'state': 'recorded', 'activity': 'Recorded accepted asset decisions'}
        return dict(phases[stage], schema_version=1, state=state, source=source.group(1) if source else '',
                    updated_at=path.stat().st_mtime, phases=phases, events=[{'phase': e['phase'],
                    'activity': ACTIVITIES.get(e['phase'], e['phase'].replace('_', ' ')),
                    'facts': {k:v for k,v in e.items() if k not in ('phase','elapsed_seconds')},
                    'elapsed_seconds': e.get('elapsed_seconds')} for e in events[-100:]],
                    decisions_markdown=decisions, decisions=[], assessment_rounds=[], legacy=True)

    def _artifact(self, label, path, *, openable=False, root=None):
        path = Path(path).resolve()
        if not path.exists() or (root and not path.is_relative_to(Path(root).resolve())):
            return None
        identifier = hashlib.sha256(str(path).encode()).hexdigest()[:20]
        href = None
        if openable and path.is_file():
            self.files[identifier] = path
            href = '/artifact/'+identifier
        return {'label': label, 'path': str(path), 'href': href}

    def snapshot(self, now=None):
        with self.lock:
            now = time.time() if now is None else now
            nested = self.workspace/'execution'
            execution = self.execution or (nested if nested.is_dir() else self.workspace)
            assessment = self.assessment or (self.workspace/'assessment' if (self.workspace/'assessment').is_dir() else self.workspace)
            current = self.read(execution/'progress.json') or self._recorded_state(execution)
            assessed = self.read(assessment/'progress.json')
            # Execution is authoritative once it starts. An older assessment
            # snapshot must not pull a running/completed workflow back to Assess.
            state = dict(current or assessed or {})
            if current and assessed and assessment != execution:
                state['assessment_rounds'] = state.get('assessment_rounds') or assessed.get('assessment_rounds', [])
                state['decisions'] = state.get('decisions') or assessed.get('decisions', [])
                state['decisions_markdown'] = state.get('decisions_markdown') or assessed.get('decisions_markdown', '')
                state['phases'] = {**assessed.get('phases', {}), **state.get('phases', {})}
            age = max(0, now-state.get('updated_at', now))
            freshness = ('unavailable' if not state else 'recorded' if state.get('state') in TERMINAL or state.get('legacy')
                         else 'stale' if age > 90 else 'current')
            artifacts = []
            artifact_paths = set()
            self.files = {}
            def add(label, path, **kwargs):
                item = self._artifact(label, path, **kwargs)
                if item and item['path'] not in artifact_paths:
                    artifacts.append(item)
                    artifact_paths.add(item['path'])
            add('Run directory', self.workspace)
            for label, name in [('HTML report','report.html'), ('Markdown report','report.md'),
                                ('Report data','report.json'), ('Text progress','progress.md'),
                                ('Workflow log','workflow-progress.jsonl'),
                                ('Worker summary','compute/status.summary.json')]:
                add(label, execution/name, openable=True, root=execution)
            add('Worker logs', execution/'compute/jobs', root=execution)
            add('Published consumer', execution/'after.usdc', root=execution)
            add('Assessment log', assessment/'assessment-progress.jsonl', openable=True, root=assessment)
            add('Assessment manifest', assessment/'manifest.json', root=assessment)
            library = self.common_library or (Path(state['common_library']) if state.get('common_library') else execution/'published')
            add('Library directory', library)
            add('Library index', library/'index.json', openable=True, root=library)
            index = self.read(library/'index.json') or {}
            if not (execution/'after.usdc').is_file() and index.get('consumer_layer'):
                add('Published consumer', library/index['consumer_layer'], root=library)
            if index.get('material_library'):
                add('Material library', library/index['material_library'], root=library)
            for key, row in index.get('libraries', {}).items():
                value = row.get('layer') if isinstance(row, dict) else row
                if value: add('Library '+key, library/value, root=library)
            for row in index.get('dependencies', {}).values():
                value = row.get('location') if isinstance(row, dict) else None
                if value: add('Library dependency', library/value, root=library)
            return {'state': state or None, 'stages': [{'id': k, 'title': v} for k,v in STAGES],
                    'freshness': freshness, 'age_seconds': age, 'artifacts': artifacts,
                    'server_time': now}


def make_server(view, port=8765):
    template = (THEME.parent/'optimization-report.html.template').read_text()
    # Permit only the shipped report's static filter script, not arbitrary
    # scripts embedded in an output file. The report remains self-contained.
    script_hashes = ' '.join("'sha256-"+base64.b64encode(hashlib.sha256(code.encode()).digest()).decode()+"'"
                            for code in re.findall(r'<script>(.*?)</script>', template, re.S))
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_GET(self):
            host = self.headers.get('Host', '')
            if host not in (f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'):
                self.send_error(403); return
            route = urlsplit(self.path).path
            if route == '/api/progress':
                body = json.dumps(view.snapshot(), ensure_ascii=False).encode()
                self.respond(body, 'application/json; charset=utf-8'); return
            static = {'/': ASSETS/'index.html', '/app.js': ASSETS/'app.js',
                      '/styles.css': ASSETS/'styles.css', '/theme.css': THEME}
            if route.startswith('/artifact/'):
                with view.lock:
                    path = view.files.get(route.removeprefix('/artifact/'))
            else:
                path = static.get(route)
            if path is None or not path.is_file():
                self.send_error(404); return
            kind = {'.html':'text/html', '.css':'text/css', '.js':'text/javascript',
                    '.json':'application/json'}.get(path.suffix, 'text/plain')
            self.respond(path.read_bytes(), kind+'; charset=utf-8')

        def respond(self, body, kind):
            self.send_response(200)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', f"default-src 'self'; script-src 'self' {script_hashes}; style-src 'self' 'unsafe-inline'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # The browser closed its read-only connection.
    return ThreadingHTTPServer(('127.0.0.1', port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True, type=Path)
    parser.add_argument('--execution', type=Path)
    parser.add_argument('--assessment', type=Path)
    parser.add_argument('--common-library', type=Path)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    view = RunView(args.workspace, args.execution, args.assessment, args.common_library)
    server = make_server(view, args.port)
    print(f'USD progress: http://127.0.0.1:{server.server_port}/ — read-only {view.workspace}', flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()


if __name__ == '__main__': main()
