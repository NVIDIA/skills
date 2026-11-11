# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise installation provenance in a real isolated venv without simulator packages."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import venv

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/runtime_checks.py"
REV = "a8b4da3c29ae528b39d4b3c9444d782ce58d886d"
LAB_URL = "https://github.com/isaac-sim/IsaacLab.git"


class StandaloneInstallationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "isaac_ros_manipulation/isaaclab_training"
        self.root.mkdir(parents=True)
        self.environment = self.root / ".venv"
        # Symlink the base interpreter so uv-managed Python can locate libpython.
        venv.EnvBuilder(with_pip=False, symlinks=True).create(self.environment)
        self.python = self.environment / "bin/python"
        self.site = next((self.environment / "lib").glob("python*/site-packages"))
        self.package = self.root / "src/isaaclab_training"
        self.package.mkdir(parents=True)
        (self.package / "__init__.py").touch()
        (self.site / "training.pth").write_text(str(self.root / "src") + "\n")
        (self.site / "isaaclab").mkdir()
        (self.site / "isaaclab/__init__.py").touch()
        self.root.joinpath("pyproject.toml").write_text(f'''[project]
name = "isaaclab-training"
version = "0.1.0"
[tool.uv.sources]
isaaclab = {{git = "{LAB_URL}", rev = "{REV}", subdirectory = "tools/wheel_builder"}}
''')
        self.lock = self.root / "uv.lock"
        self.lock.write_text(f'''version = 1
[[package]]
name = "isaaclab"
version = "3.0.0"
source = {{git = "{LAB_URL}?subdirectory=tools%2Fwheel_builder&rev={REV}#{REV}"}}
[[package]]
name = "isaaclab-training"
version = "0.1.0"
source = {{editable = "."}}
[[package]]
name = "isaacsim"
version = "6.1.0.0"
''')
        self.lab_direct = self.distribution("isaaclab", "3.0.0", {
            "url": LAB_URL, "subdirectory": "tools/wheel_builder",
            "vcs_info": {"vcs": "git", "commit_id": REV, "requested_revision": REV},
        })
        self.training_direct = self.distribution("isaaclab-training", "0.1.0", {
            "url": self.root.as_uri(), "dir_info": {"editable": True},
        })
        self.distribution("isaacsim", "6.1.0.0")

    def distribution(self, name, version, direct=None):
        folder = self.site / f"{name.replace('-', '_')}-{version}.dist-info"
        folder.mkdir()
        (folder / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
        path = folder / "direct_url.json"
        if direct:
            path.write_text(json.dumps(direct))
        return path

    def probe(self, *, python=None, pythonpath=None):
        env = {"PATH": os.defpath, "PYTHONNOUSERSITE": "1"}
        if pythonpath:
            env["PYTHONPATH"] = str(pythonpath)
        result = subprocess.run([str(python or self.python), str(SCRIPT), "installation", "--root", str(self.root)],
                                cwd=self.root, env=env, capture_output=True, text=True, timeout=15)
        self.assertTrue(result.stdout.strip(), result.stderr)
        return result.returncode, json.loads(result.stdout)

    def assert_rejected(self, message, **kwargs):
        code, value = self.probe(**kwargs)
        self.assertEqual(code, 2, value)
        self.assertIn(message, value["error"])

    def test_standalone_without_source_checkout(self):
        code, value = self.probe()
        self.assertEqual(code, 0, value)
        self.assertEqual(value["training_root"], str(self.root))
        self.assertEqual(value["isaaclab_commit"], REV)
        self.assertEqual(value["isaaclab_training_source"], str(self.package / "__init__.py"))
        self.assertEqual(len(value["lock_sha256"]), 64)

    def test_wrong_environment(self):
        self.assert_rejected("training_root/.venv/bin/python", python=sys.executable)

    def test_installed_commit_drift(self):
        value = json.loads(self.lab_direct.read_text())
        value["vcs_info"]["commit_id"] = "b" * 40
        self.lab_direct.write_text(json.dumps(value))
        self.assert_rejected("does not match the locked wheel-builder")

    def test_lock_drift(self):
        self.lock.write_text(self.lock.read_text().replace(REV, "b" * 40))
        self.assert_rejected("pin and uv.lock disagree")

    def test_source_editable_lab(self):
        self.lab_direct.write_text(json.dumps({"url": "file:///tmp/IsaacLab", "dir_info": {"editable": True}}))
        self.assert_rejected("does not match the locked wheel-builder")

    def test_source_lab_import_shadow(self):
        shadow = Path(self.temp.name) / "source/isaaclab"
        shadow.mkdir(parents=True)
        (shadow / "__init__.py").touch()
        self.assert_rejected("outside the standalone .venv", pythonpath=shadow.parent)

    def test_external_training_editable(self):
        self.training_direct.write_text(json.dumps({"url": "file:///tmp/IsaacLabTraining",
                                                   "dir_info": {"editable": True}}))
        self.assert_rejected("editable from training_root")

    def test_missing_lock(self):
        self.lock.unlink()
        self.assert_rejected("uv.lock")

    def test_sim_version_drift(self):
        metadata = self.site / "isaacsim-6.1.0.0.dist-info/METADATA"
        metadata.write_text(metadata.read_text().replace("6.1.0.0", "6.2.0.0"))
        self.assert_rejected("Isaac Sim does not match uv.lock")

    def test_training_lock_must_be_this_project(self):
        self.lock.write_text(self.lock.read_text().replace('editable = "."', 'editable = "../IsaacLabTraining"'))
        self.assert_rejected("editable project in uv.lock")


if __name__ == "__main__":
    unittest.main()
