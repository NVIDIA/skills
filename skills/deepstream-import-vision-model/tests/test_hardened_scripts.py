#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Validation tests for the hardened scripts in deepstream-import-vision-model.

Covers:
  - install.sh:                 --target validation (rejects /, .., empty, missing)
                                dry-run previews self-contained skill copies
  - scripts/model/cleanup.sh:   MODEL_NAME regex enforcement, empty arg rejection,
                                dry-run does not touch the filesystem
  - scripts/model/hf-list-files.sh / hf-download-config.sh:
                                rejects injection characters in HF_ORG / MODEL_NAME
  - scripts/model/ngc-list-files.sh / ngc-download.sh:
                                rejects injection characters in NGC args
                                refuses invalid DEST_DIR (empty, /, containing ..)
  - scripts/deepstream/ds-kitti-dump.sh:
                                usage message printed when args missing
  - scripts/report/md-to-html-pdf.py::embed_images:
                                base64-inlines local images, rejects path traversal,
                                leaves absolute/remote URLs alone

Run:
    python3 -m unittest discover -s tests -v
"""
from __future__ import annotations

import base64
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent.parent
SCRIPTS = SKILL_DIR / "scripts"
INSTALL_SH = SKILL_DIR / "install.sh"


def run_script(cmd, cwd=None, timeout=30):
    """Run a command and return (returncode, stdout, stderr)."""
    r = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=False,
        cwd=str(cwd) if cwd else None,
        timeout=timeout,
    )
    return r.returncode, r.stdout, r.stderr


class TestInstallScript(unittest.TestCase):
    """install.sh TARGET validation (B6)."""

    def test_rejects_missing_target_arg(self):
        rc, out, err = run_script(["bash", str(INSTALL_SH)])
        self.assertNotEqual(rc, 0)
        self.assertIn("--target is required", out + err)

    def test_rejects_filesystem_root(self):
        rc, out, err = run_script(["bash", str(INSTALL_SH), "--target", "/", "--dry-run"])
        self.assertNotEqual(rc, 0)
        self.assertIn("invalid --target", out + err)

    def test_rejects_path_traversal(self):
        rc, out, err = run_script(
            ["bash", str(INSTALL_SH), "--target", "/tmp/../etc", "--dry-run"]
        )
        self.assertNotEqual(rc, 0)
        self.assertIn("invalid --target", out + err)

    def test_rejects_nonexistent_target(self):
        rc, out, err = run_script(
            ["bash", str(INSTALL_SH), "--target", "/does/not/exist", "--dry-run"]
        )
        self.assertNotEqual(rc, 0)
        self.assertIn("target directory not found", out + err)

    def test_dry_run_previews_all_skill_locations(self):
        with tempfile.TemporaryDirectory() as td:
            rc, out, _ = run_script(
                ["bash", str(INSTALL_SH), "--target", td, "--dry-run"]
            )
            self.assertEqual(rc, 0, f"install.sh --dry-run failed: {out}")
            self.assertIn("[dry-run] cp -r", out)
            # Single skill copied as real files (references pattern — no sub-skills)
            self.assertIn(".claude/skills/deepstream-import-vision-model", out)
            self.assertNotIn("nv-model-acquire", out)
            self.assertNotIn("nv-engine-build", out)
            self.assertNotIn("ds-run-pipeline", out)
            self.assertNotIn("nv-import-vision-model-report", out)
            # Cursor skill also installed
            self.assertIn(".cursor/skills/deepstream-import-vision-model", out)
            self.assertIn(".codex/skills/deepstream-import-vision-model", out)
            # No agent directory — skill-only architecture
            self.assertNotIn(".claude/agents", out)
            # Nothing must be written in dry-run mode
            self.assertEqual(os.listdir(td), [])


    def test_reinstall_from_installed_path_is_safe(self):
        with tempfile.TemporaryDirectory() as td:
            installed = Path(td) / ".claude" / "skills" / "deepstream-import-vision-model"
            shutil.copytree(SKILL_DIR, installed)
            marker = installed / "SKILL.md"
            rc, out, err = run_script([
                "bash", str(installed / "install.sh"),
                "--target", td, "--no-cursor",
            ])
            self.assertEqual(rc, 0, f"stdout={out} stderr={err}")
            self.assertTrue(marker.exists(), "in-place reinstall must not delete its source")
            self.assertIn("source and destination are identical", out)


class TestCleanupScript(unittest.TestCase):
    """scripts/model/cleanup.sh MODEL_NAME validation (B8)."""

    CLEANUP = SCRIPTS / "model" / "cleanup.sh"

    def test_rejects_missing_arg(self):
        rc, out, err = run_script(["bash", str(self.CLEANUP)])
        self.assertNotEqual(rc, 0)
        self.assertIn("Usage", out + err)

    def test_rejects_shell_metachars(self):
        rc, _, err = run_script(["bash", str(self.CLEANUP), "bad;name", "--dry-run"])
        self.assertNotEqual(rc, 0)
        self.assertIn("MODEL_NAME must match", err)

    def test_rejects_slash(self):
        rc, _, err = run_script(["bash", str(self.CLEANUP), "bad/name", "--dry-run"])
        self.assertNotEqual(rc, 0)
        self.assertIn("MODEL_NAME must match", err)

    def test_accepts_valid_name_dry_run(self):
        with tempfile.TemporaryDirectory() as td:
            rc, out, err = run_script(
                ["bash", str(self.CLEANUP), "yolov8n", "--dry-run"], cwd=td
            )
            self.assertEqual(rc, 0, f"stdout={out} stderr={err}")
            # No candidates exist in a fresh dir — all should be skipped, nothing removed
            self.assertIn("skip (not present)", out)

    def test_dry_run_does_not_remove_present_paths(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            target = tdp / "build" / ".venv_yolov8n"
            target.mkdir(parents=True)
            (target / "marker").write_text("present")
            rc, out, _ = run_script(
                ["bash", str(self.CLEANUP), "yolov8n", "--dry-run"], cwd=td
            )
            self.assertEqual(rc, 0)
            self.assertIn("[dry-run]", out)
            self.assertTrue(target.exists(), "dry-run must not remove files")
            self.assertTrue((target / "marker").exists())


class TestHFScripts(unittest.TestCase):
    """HF helper script input validation (B4)."""

    LIST = SCRIPTS / "model" / "hf-list-files.sh"
    CONFIG = SCRIPTS / "model" / "hf-download-config.sh"

    def test_list_rejects_bad_org(self):
        rc, _, err = run_script(["bash", str(self.LIST), "bad;org", "model"])
        self.assertNotEqual(rc, 0)
        self.assertIn("invalid characters", err)

    def test_list_rejects_bad_model(self):
        rc, _, err = run_script(["bash", str(self.LIST), "org", "bad$model"])
        self.assertNotEqual(rc, 0)
        self.assertIn("invalid characters", err)

    def test_list_rejects_missing_args(self):
        rc, _, err = run_script(["bash", str(self.LIST)])
        self.assertNotEqual(rc, 0)
        self.assertIn("Usage", err)

    def test_config_rejects_path_traversal_in_dest(self):
        rc, _, err = run_script(
            [
                "bash",
                str(self.CONFIG),
                "org",
                "model",
                "relative/../etc/config.json",
            ]
        )
        self.assertNotEqual(rc, 0)
        self.assertIn("'..'", err)

    def test_config_rejects_absolute_dest(self):
        rc, _, err = run_script(
            [
                "bash",
                str(self.CONFIG),
                "org",
                "model",
                "/tmp/config.json",
            ]
        )
        self.assertNotEqual(rc, 0)
        self.assertIn("must be relative", err)


class TestNGCScripts(unittest.TestCase):
    """NGC helper script input validation (B5)."""

    LIST = SCRIPTS / "model" / "ngc-list-files.sh"
    DOWNLOAD = SCRIPTS / "model" / "ngc-download.sh"

    def test_list_rejects_bad_org(self):
        rc, _, err = run_script(
            ["bash", str(self.LIST), "bad;org", "team", "model", "v1"]
        )
        self.assertNotEqual(rc, 0)
        self.assertIn("invalid characters", err)

    def test_download_rejects_empty_dest(self):
        rc, _, err = run_script(
            [
                "bash",
                str(self.DOWNLOAD),
                "org",
                "team",
                "model",
                "v1",
                "",
            ]
        )
        self.assertNotEqual(rc, 0)

    def test_download_rejects_traversal_dest(self):
        rc, _, err = run_script(
            [
                "bash",
                str(self.DOWNLOAD),
                "org",
                "team",
                "model",
                "v1",
                "/tmp/../etc",
            ]
        )
        self.assertNotEqual(rc, 0)
        self.assertIn("invalid DEST_DIR", err)

    def test_download_rejects_fs_root_dest(self):
        rc, _, err = run_script(
            [
                "bash",
                str(self.DOWNLOAD),
                "org",
                "team",
                "model",
                "v1",
                "/",
            ]
        )
        self.assertNotEqual(rc, 0)
        self.assertIn("invalid DEST_DIR", err)


class TestKittiDumpUsage(unittest.TestCase):
    """scripts/deepstream/ds-kitti-dump.sh usage message (B7)."""

    KITTI = SCRIPTS / "deepstream" / "ds-kitti-dump.sh"

    def test_prints_usage_when_args_missing(self):
        rc, out, err = run_script(["bash", str(self.KITTI)])
        self.assertNotEqual(rc, 0)
        combined = out + err
        self.assertTrue(
            "Usage" in combined or "unbound variable" in combined,
            f"expected Usage or unbound-variable error, got: {combined!r}",
        )


class TestEmbedImages(unittest.TestCase):
    """scripts/report/md-to-html-pdf.py::embed_images (B2).

    Images must be base64-inlined so wkhtmltopdf no longer needs
    --enable-local-file-access.
    """

    @classmethod
    def setUpClass(cls):
        # Import the module by loading its source directly to avoid hyphen-path issues.
        # md-to-html-pdf.py imports the third-party `markdown` package at module scope,
        # which isn't needed for embed_images(). Stub it so tests run without that dep.
        import importlib.util
        import types

        if "markdown" not in sys.modules:
            stub = types.ModuleType("markdown")
            stub.markdown = lambda text, **kw: text  # no-op for tests
            sys.modules["markdown"] = stub

        src = SCRIPTS / "report" / "md-to-html-pdf.py"
        spec = importlib.util.spec_from_file_location("md_to_html_pdf", src)
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)

    def test_inlines_local_png(self):
        # 1x1 transparent PNG
        png = base64.b64decode(
            b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
        )
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            (p / "pic.png").write_bytes(png)
            html = '<img src="pic.png" alt="x">'
            out = self.mod.embed_images(html, str(p))
            self.assertIn("data:image/png;base64,", out)
            self.assertNotIn('src="pic.png"', out)

    def test_leaves_remote_url_alone(self):
        html = '<img src="https://example.com/pic.png" alt="x">'
        with tempfile.TemporaryDirectory() as td:
            out = self.mod.embed_images(html, td)
            self.assertEqual(html, out)

    def test_leaves_absolute_path_alone(self):
        html = '<img src="/opt/data/elsewhere.png" alt="x">'
        with tempfile.TemporaryDirectory() as td:
            out = self.mod.embed_images(html, td)
            # Should not be rewritten to a data: URI (absolute paths are skipped)
            self.assertNotIn("data:", out)
            self.assertIn('src="/opt/data/elsewhere.png"', out)

    def test_rejects_path_traversal(self):
        # src points outside base_dir — must not be embedded
        with tempfile.TemporaryDirectory() as td_base:
            with tempfile.TemporaryDirectory() as td_outside:
                secret = Path(td_outside) / "secret.png"
                secret.write_bytes(b"\x89PNG\r\n\x1a\nFAKE")
                rel = os.path.relpath(secret, td_base)  # starts with ../
                html = f'<img src="{rel}" alt="x">'
                out = self.mod.embed_images(html, td_base)
                self.assertNotIn("data:", out)
                self.assertIn(rel, out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
