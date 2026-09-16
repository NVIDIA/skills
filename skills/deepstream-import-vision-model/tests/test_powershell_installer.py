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

"""Regression tests for the Windows PowerShell installers of BOTH DeepStream skills.

Why this exists
---------------
`install.ps1` once shipped with:

    Copy-Item -LiteralPath $SkillDir -Destination $dest -Recurse -Force

which is not portable. Whether it copies a folder's CONTENTS or the folder ITSELF depends on
whether the destination already exists, and on Windows PowerShell 5.1 it created the directory
tree without copying any leaf files -- installing an empty skill directory with no SKILL.md, so
the agent runtime silently refused to load the skill. Nothing caught it, because the test suites
only ever exercised install.sh.

The central assertion here is therefore a strict **file-count match** against the source, not just
"SKILL.md exists": a partial copy is exactly the failure mode that shipped.

Runner
------
Needs a PowerShell. Discovery order: `pwsh`, `powershell`, then the `mcr.microsoft.com/powershell`
container if docker has it locally (so this runs on Linux dev boxes and CI too). Skips cleanly when
none is available -- never a false failure on a machine without PowerShell.

Caveat: the container/dev-box runner is PowerShell 7. The bug that motivated this was 5.1-specific,
so this pins the copy *contract* rather than reproducing every 5.1 quirk. Run it on Windows for
full fidelity.

Run:
    python3 -m unittest discover -s tests -v
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_SKILLS = Path(__file__).resolve().parents[2]
PS_IMAGE = "mcr.microsoft.com/powershell:latest"

# (skill dir name, expected file count key) -- both installers share the copy logic under test.
SKILLS = ["deepstream-import-vision-model", "deepstream-eval-and-finetune"]

# Machine-local build artifacts each installer strips from the copy; excluded when computing the
# expected file count. Mirrors the strip rules inside the two install.ps1 scripts.
STRIP_SUFFIXES = (".pyc", ".so", ".o")
STRIP_NAMES = ("ds_image_eval",)


def _docker_has_ps_image() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        out = subprocess.run(["docker", "images", "-q", PS_IMAGE],
                             capture_output=True, text=True, timeout=30)
        return bool(out.stdout.strip())
    except Exception:
        return False


def _runner():
    """Return a callable(script, args, cwd) -> CompletedProcess, or None if no PowerShell."""
    for exe in ("pwsh", "powershell"):
        if shutil.which(exe):
            def run_native(script, args, cwd, _exe=exe):
                return subprocess.run(
                    [_exe, "-NoProfile", "-NonInteractive", "-File", str(script), *args],
                    cwd=str(cwd), capture_output=True, text=True, timeout=300)
            return run_native
    if _docker_has_ps_image():
        def run_docker(script, args, cwd):
            # Mount both trees at their real absolute paths so in-container paths match the host's
            # and the assertions below can read results back directly. `-File` (not `-Command`) so
            # that -Target/-NoCursor bind as named parameters rather than positional strings.
            return subprocess.run(
                ["docker", "run", "--rm",
                 # Write as the invoking user, or the installed tree lands root-owned and the
                 # caller cannot clean up its own temp dir.
                 "--user", f"{os.getuid()}:{os.getgid()}",
                 "-e", "HOME=/tmp",
                 "-v", f"{REPO_SKILLS}:{REPO_SKILLS}:ro",
                 "-v", f"{cwd}:{cwd}",
                 "-w", str(cwd), PS_IMAGE,
                 "pwsh", "-NoProfile", "-NonInteractive",
                 "-File", str(script), *args],
                capture_output=True, text=True, timeout=600)
        return run_docker
    return None


RUN = _runner()


def _expected_files(src: Path) -> set[str]:
    """Files the installer should land, relative to the skill root, after its strip rules."""
    out = set()
    for p in src.rglob("*"):
        if not p.is_file():
            continue
        if "__pycache__" in p.parts:
            continue
        if p.suffix in STRIP_SUFFIXES or p.name in STRIP_NAMES:
            continue
        out.add(p.relative_to(src).as_posix())
    return out


@unittest.skipIf(RUN is None,
                 "no PowerShell available (need pwsh/powershell on PATH, or the "
                 f"{PS_IMAGE} docker image pulled)")
class TestPowerShellInstaller(unittest.TestCase):
    def _install(self, skill, target, extra=()):
        # Copy the skill out of the repo first: the installer resolves $PSScriptRoot, and the repo
        # checkout is mounted read-only under the docker runner.
        src = Path(target) / "_src" / skill
        src.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(REPO_SKILLS / skill, src)
        proj = Path(target) / "proj"
        proj.mkdir()
        # --no-plugin equivalent: never touch the host's real ~/.claude during a test.
        args = ["-Target", str(proj), "-NoCursor", *extra]
        if skill == "deepstream-eval-and-finetune":
            args.append("-NoPlugin")
        res = RUN(src / "install.ps1", args, target)
        return res, proj, src

    def test_installs_every_source_file_including_skill_md(self):
        for skill in SKILLS:
            with self.subTest(skill=skill), tempfile.TemporaryDirectory() as td:
                res, proj, src = self._install(skill, td)
                self.assertEqual(res.returncode, 0,
                                 f"installer failed\nstdout={res.stdout}\nstderr={res.stderr}")
                dest = proj / ".claude" / "skills" / skill
                self.assertTrue((dest / "SKILL.md").is_file(),
                                f"SKILL.md missing -> the skill would not load. stdout={res.stdout}")
                # The regression: an empty/partial tree used to pass silently.
                got = {p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()}
                self.assertEqual(got, _expected_files(src),
                                 "installed file set does not match the source")

    def test_installs_for_codex_as_well_as_claude(self):
        for skill in SKILLS:
            with self.subTest(skill=skill), tempfile.TemporaryDirectory() as td:
                _, proj, _ = self._install(skill, td)
                for agent in (".claude", ".codex"):
                    self.assertTrue((proj / agent / "skills" / skill / "SKILL.md").is_file(),
                                    f"{agent} install is missing SKILL.md")

    def test_no_cursor_flag_is_honoured(self):
        skill = SKILLS[0]
        with tempfile.TemporaryDirectory() as td:
            _, proj, _ = self._install(skill, td)
            self.assertFalse((proj / ".cursor").exists(), "-NoCursor still wrote .cursor/")

    def test_reinstall_is_idempotent_and_does_not_nest(self):
        for skill in SKILLS:
            with self.subTest(skill=skill), tempfile.TemporaryDirectory() as td:
                _, proj, src = self._install(skill, td)
                dest = proj / ".claude" / "skills" / skill
                first = {p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()}
                args = ["-Target", str(proj), "-NoCursor"]
                if skill == "deepstream-eval-and-finetune":
                    args.append("-NoPlugin")
                res = RUN(src / "install.ps1", args, td)
                self.assertEqual(res.returncode, 0, f"reinstall failed: {res.stderr}")
                again = {p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()}
                self.assertEqual(first, again, "reinstall changed the installed file set")
                self.assertFalse((dest / skill).exists(),
                                 "reinstall nested the skill inside itself")

    def test_install_fails_loudly_when_skill_md_would_be_missing(self):
        """The guard must fire -- a silent partial install is the failure mode that shipped."""
        skill = SKILLS[0]
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "_src" / skill
            src.parent.mkdir(parents=True)
            shutil.copytree(REPO_SKILLS / skill, src)
            (src / "SKILL.md").unlink()
            proj = Path(td) / "proj"
            proj.mkdir()
            res = RUN(src / "install.ps1", ["-Target", str(proj), "-NoCursor"], td)
            self.assertNotEqual(res.returncode, 0,
                                "installer reported success with SKILL.md absent")
            self.assertIn("SKILL.md", res.stdout + res.stderr)


if __name__ == "__main__":
    unittest.main()
