# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline contract tests: python3 -m unittest discover -s <skill>/tests -v."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

SKILL = Path(__file__).resolve().parents[1]


class DependencyContract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.skill = self.root / 'parent skill'
        shutil.copytree(SKILL, self.skill, ignore=shutil.ignore_patterns('__pycache__'))
        self.upstream = self.root / 'upstream'
        self.upstream.mkdir()
        self.env = {
            'PATH': os.defpath,
            'HOME': str(self.root / 'home'),
            'TMPDIR': str(self.root),
            'LANG': 'C.UTF-8',
            'GIT_CONFIG_NOSYSTEM': '1',
            'GIT_CONFIG_GLOBAL': os.devnull,
        }
        self.env['PYTHONDONTWRITEBYTECODE'] = '1'
        self.env['HUMANOID_CLOUD_CONTROL_HOME'] = str(self.root / 'state')
        self.env['MC_WORKSPACE'] = str(self.root / 'overlay with spaces')
        self.lock = json.loads((self.skill / 'upstream-versions.lock.json').read_text())
        for name, spec in self.lock['skills'].items():
            folder = self.upstream / 'skills' / name
            folder.mkdir(parents=True)
            (folder / 'SKILL.md').write_text(f'# {name}\n')
            for relative in spec.get('resources', []):
                target = folder / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text('fixture\n')
            for relative in spec['entrypoints'].values():
                target = folder / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.suffix == '.py':
                    target.write_text('import json, os, sys\nprint(json.dumps({"cwd": os.getcwd(), "args": sys.argv[1:], "workspace": os.environ["MC_WORKSPACE"]}))\nsys.exit(7)\n')
                else:
                    target.write_text('#!/bin/sh\nprintf "%s\\n" "$PWD" "$MC_WORKSPACE" "$@"\nexit 7\n')
                    target.chmod(0o755)
        self.git('init', '-q')
        self.git('add', '.')
        self.git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.com', 'commit', '-qm', 'fixture')
        for spec in self.lock['upstreams'].values():
            spec['commit'] = self.git('rev-parse', 'HEAD').strip()
            self.env[spec['root_env']] = str(self.upstream)
        (self.skill / 'upstream-versions.lock.json').write_text(json.dumps(self.lock))

    def git(self, *args):
        return subprocess.check_output(['git', *args], cwd=self.upstream, text=True, env=self.env)

    def run_script(self, relative, *args):
        return subprocess.run([sys.executable, str(self.skill / relative), *args],
                              cwd=self.root, env=self.env, text=True, capture_output=True, timeout=30)

    def resolve(self, *args):
        result = self.run_script('scripts/doctor.py', 'dependencies', *args)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result

    def adapter(self, name, *args):
        return self.run_script(f'references/{name}/scripts/run.py', *args)

    def test_unrelated_environment_is_not_forwarded(self):
        from unittest.mock import patch
        with patch.dict(os.environ, {'UNRELATED_TEST_SECRET': 'sentinel'}):
            result = subprocess.run(
                [sys.executable, '-c',
                 'import os; print(os.getenv("UNRELATED_TEST_SECRET", "absent"))'],
                env=self.env, text=True, capture_output=True, check=True,
            )
        self.assertEqual(result.stdout.strip(), 'absent')

    def test_all_adapters_forward_cwd_arguments_environment_and_exit(self):
        self.resolve()
        for name, spec in self.lock['skills'].items():
            if not spec.get('adapter'):
                continue
            with self.subTest(skill=name):
                result = self.adapter(name, 'argument with spaces', '--flag=value')
                self.assertEqual(result.returncode, 7, result.stderr)
                self.assertIn(str(self.upstream / 'skills' / name), result.stdout)
                self.assertIn('argument with spaces', result.stdout)
                self.assertIn(self.env['MC_WORKSPACE'], result.stdout)

    def test_verification_handoff_resolves_without_submission_entrypoint(self):
        result = self.resolve('--json')
        entry = json.loads(result.stdout)['skills']['submit-and-monitor-mission']
        self.assertEqual(entry['status'], 'ready')
        self.assertTrue(Path(entry['skill_md']).is_file())
        self.assertTrue(entry['agent_only'])
        self.assertEqual(entry['entrypoints'], {})
        self.assertIsNone(entry['adapter'])

    def test_missing_manifest_blocks_without_execution(self):
        result = self.adapter('isaac-sim-remote')
        self.assertEqual(result.returncode, 3)
        self.assertIn('BLOCKED [isaac-sim-remote]', result.stderr)
        self.assertEqual(result.stdout, '')

    def test_changed_checkout_blocks_after_resolution(self):
        self.resolve()
        (self.upstream / 'skills/isaac-sim-remote/SKILL.md').write_text('changed\n')
        result = self.adapter('isaac-sim-remote')
        self.assertEqual(result.returncode, 3)
        self.assertIn('checkout changed', result.stderr)

    def test_diagnostic_drift_is_never_executable(self):
        (self.upstream / 'skills/isaac-sim-remote/SKILL.md').write_text('changed\n')
        self.resolve('--allow-commit-drift')
        result = self.adapter('isaac-sim-remote')
        self.assertEqual(result.returncode, 3)
        self.assertIn('--allow-commit-drift', result.stderr)

    def test_check_only_writes_no_manifest(self):
        self.resolve('--check-only')
        self.assertFalse((self.root / 'state').exists())

    def test_wrong_pin_and_missing_entrypoint_block_resolution(self):
        for spec in self.lock['upstreams'].values():
            spec['commit'] = '0' * 40
        (self.skill / 'upstream-versions.lock.json').write_text(json.dumps(self.lock))
        result = self.run_script('scripts/doctor.py', 'dependencies')
        self.assertEqual(result.returncode, 3)
        (self.upstream / 'skills/isaac-sim-remote/scripts/isaacsim_send.py').unlink()
        result = self.run_script('scripts/doctor.py', 'dependencies')
        self.assertEqual(result.returncode, 3)
        self.assertIn('missing entrypoint', result.stderr)

    def test_prepare_preserves_explicit_upstream_overrides(self):
        before = self.git('rev-parse', 'HEAD')
        result = self.resolve('--prepare')
        self.assertIn('skip mission_control', result.stdout)
        self.assertEqual(self.git('rev-parse', 'HEAD'), before)


if __name__ == '__main__':
    unittest.main()
