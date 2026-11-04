# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regressions for inherited repositories and different launcher environments."""

import json
import os
from pathlib import Path
import shlex
import signal
import time
import subprocess
import sys
import tempfile
import unittest
import venv
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from launch_environment import build_environment, normalize_environment
import supervisor


class LaunchEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.uv = self.root / "tools/uv"
        self.uv.parent.mkdir()
        self.uv.write_text("#!/bin/sh\nexit 0\n")
        self.uv.chmod(0o755)

    def test_child_imports_selected_venv_despite_shadow_pythonpath(self):
        venv.EnvBuilder(with_pip=False, symlinks=True).create(self.root / ".venv")
        site = next((self.root / ".venv/lib").glob("python*/site-packages"))
        (site / "isolation_probe.py").write_text("origin='selected'\n")
        shadow = self.root / "old-checkout"
        shadow.mkdir()
        (shadow / "isolation_probe.py").write_text("origin='old'\n")
        parent = {"PATH": str(shadow), "PYTHONPATH": str(shadow),
                  "LD_LIBRARY_PATH": str(shadow), "LD_PRELOAD": str(shadow / "lib.so"),
                  "VIRTUAL_ENV": str(shadow), "UV_PROJECT_ENVIRONMENT": str(shadow),
                  "GITLAB_TOKEN": "secret", "CUDA_VISIBLE_DEVICES": "2"}
        config = normalize_environment({"uv_executable": str(self.uv)}, parent=parent)
        env = build_environment(self.root, config, eula_accepted=False)
        result = subprocess.run([str(self.root / '.venv/bin/python'), '-c',
                                 'import isolation_probe; print(isolation_probe.origin)'],
                                cwd=self.root, env=env, capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.strip(), 'selected')
        for name in ('PYTHONPATH', 'LD_LIBRARY_PATH', 'LD_PRELOAD', 'VIRTUAL_ENV',
                     'UV_PROJECT_ENVIRONMENT', 'GITLAB_TOKEN', 'OMNI_KIT_ACCEPT_EULA'):
            self.assertNotIn(name, env)
        self.assertNotIn(str(shadow), env['PATH'].split(os.pathsep))
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '2')

    def test_explicit_overrides_and_caches_survive_changed_parent(self):
        config = normalize_environment({'uv_executable': str(self.uv), 'variables': {
            'LD_PRELOAD': '/opt/approved/libgomp.so', 'ISAACSIM_ASSET_ROOT': '/assets/selected',
            'UV_CACHE_DIR': str(self.root / 'cache'),
            'UV_PYTHON_INSTALL_DIR': str(self.root / 'python')}},
            parent={'DISPLAY': ':5', 'CUDA_VISIBLE_DEVICES': '1'})
        with patch.dict(os.environ, {'DISPLAY': ':9', 'CUDA_VISIBLE_DEVICES': '9',
                                     'LD_LIBRARY_PATH': '/old/lib'}, clear=True):
            env = build_environment(self.root, config, eula_accepted=True)
        self.assertEqual(env['DISPLAY'], ':5')
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '1')
        self.assertEqual(env['LD_PRELOAD'], '/opt/approved/libgomp.so')
        self.assertEqual(env['UV_CACHE_DIR'], str(self.root / 'cache'))
        self.assertEqual(env['UV_PYTHON_INSTALL_DIR'], str(self.root / 'python'))
        self.assertNotIn('LD_LIBRARY_PATH', env)

    def test_rejects_implicit_cwd_search_and_unsupported_overrides(self):
        for variables in ({'LD_LIBRARY_PATH': ':/opt/lib'}, {'LD_PRELOAD': 'lib.so'},
                          {'PYTHONPATH': '.'}, {'PATH': '/old/bin'}, {'UV_PROJECT_ENVIRONMENT': '/old'}):
            with self.subTest(variables=variables), self.assertRaises(ValueError):
                normalize_environment({'uv_executable': str(self.uv), 'variables': variables})

    def test_tmux_launch_resets_environment_before_supervisor_python(self):
        config = normalize_environment({'uv_executable': str(self.uv)}, parent={'DISPLAY': ':7'})
        plan = {'training_root': str(self.root), 'environment': config, 'eula_accepted': True}
        state = {'status': 'created'}
        session = self.root / 'session'
        with patch.object(supervisor, 'session_state', return_value=(plan, state)), \
             patch.object(supervisor.Path, 'cwd', return_value=self.root), \
             patch.object(supervisor.shutil, 'which', return_value='/usr/bin/tmux'), \
             patch.object(supervisor, 'require_eula_record'), \
             patch.object(supervisor, 'emit'), \
             patch.object(supervisor.subprocess, 'run') as run:
            supervisor.launcher(session, 'tmux')
        shell = run.call_args.args[0][-1]
        tokens = shlex.split(shell)
        start = tokens.index('/usr/bin/env')
        end = tokens.index('>>')
        command = tokens[start:end]
        python_index = command.index(str(self.root / '.venv/bin/python'))
        # Execute the same reset prefix with a harmless child instead of a campaign.
        probe = [*command[:python_index], sys.executable, '-c',
                 'import os,json; names=("LD_LIBRARY_PATH", "PYTHONPATH", "DISPLAY"); '
                 'print(json.dumps({name:os.environ[name] for name in names if name in os.environ}))']
        result = subprocess.run(probe, env={'PATH': os.defpath, 'LD_LIBRARY_PATH': '/old/lib',
                                           'PYTHONPATH': '/old/repo', 'DISPLAY': ':9'},
                                capture_output=True, text=True, check=True)
        actual = json.loads(result.stdout)
        self.assertNotIn('LD_LIBRARY_PATH', actual)
        self.assertNotIn('PYTHONPATH', actual)
        self.assertEqual(actual['DISPLAY'], ':7')

    def test_command_preserves_exit_code_and_literal_arguments(self):
        script = ("import sys; from pathlib import Path; "
                  "from launch_environment import run_command; "
                  "raise SystemExit(run_command(sys.argv[2:], Path(sys.argv[1]), "
                  "{'PATH': '/usr/bin:/bin', 'PYTHONNOUSERSITE': '1'}))")
        marker = self.root / 'must-not-exist'
        literal = f'$(touch {marker})'
        result = subprocess.run([sys.executable, '-c', script, str(self.root), sys.executable,
                                 '-c', 'import sys; print(sys.argv[1]); sys.exit(7)', literal],
                                env={'PATH': os.defpath, 'PYTHONPATH': str(Path(__file__).resolve().parents[1] / 'scripts')},
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertEqual(result.stdout.strip(), literal)
        self.assertFalse(marker.exists())

    def test_cancellation_reaches_child_and_preserves_signal_status(self):
        marker = self.root / 'child-ready'
        script = ("import sys; from pathlib import Path; "
                  "from launch_environment import run_command; "
                  "raise SystemExit(run_command(sys.argv[2:], Path(sys.argv[1]), "
                  "{'PATH': '/usr/bin:/bin', 'PYTHONNOUSERSITE': '1'}))")
        child = "import os,sys,time; from pathlib import Path; Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(30)"
        process = subprocess.Popen([sys.executable, '-c', script, str(self.root), sys.executable,
                                    '-c', child, str(marker)],
                                   env={'PATH': os.defpath, 'PYTHONPATH': str(Path(__file__).resolve().parents[1] / 'scripts')},
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            deadline = time.monotonic() + 10
            while not marker.exists() and time.monotonic() < deadline and process.poll() is None:
                time.sleep(.02)
            self.assertTrue(marker.exists())
            process.send_signal(signal.SIGTERM)
            self.assertEqual(process.wait(timeout=5), 128 + signal.SIGTERM)
            with self.assertRaises(ProcessLookupError):
                os.kill(int(marker.read_text()), 0)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)


if __name__ == '__main__':
    unittest.main()
