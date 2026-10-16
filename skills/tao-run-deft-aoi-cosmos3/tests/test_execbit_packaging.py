#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import unittest


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
APPLICATIONS_ROOT = SKILL_ROOT.parent
REPO_ROOT = SKILL_ROOT.parents[2]
SKILL_NAMES = (
    "tao-run-deft-aoi",
    "tao-run-deft-aoi-cosmos3",
)
DIRECT_CONTROL_SCRIPTS = {
    "deft_context.py",
    "deft_exec.py",
    "deft_python.sh",
    "finalize_run.py",
}
SCRIPT_REFERENCE_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])scripts/(?P<name>[A-Za-z0-9_.-]+\.(?:py|sh))"
)
SHELL_CONTINUATION_RE = re.compile(r"\\\n[ \t]*")
BARE_SCRIPT_EXEC_RE = re.compile(
    r"(?m)(?:^[ \t]*|(?:&&|\|\||;|\$\()[ \t]*|bash[ \t]+-lc[ \t]+[\"'][ \t]*)"
    r"(?:[A-Z_][A-Z0-9_]*=\$\([ \t]*)?"
    r"[\"']?(?P<command>(?:(?:\$[A-Z_]+|<[^>]+>)[^ \t\n]*/|\./)?"
    r"scripts/(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.(?:py|sh))"
    r"(?=[\"' \t;)])"
)
INLINE_BARE_SCRIPT_EXEC_RE = re.compile(
    r"\b(?:call|execute|invoke|run|through|use|using|with)\s+`"
    r"(?P<command>(?:(?:\$[A-Z_]+|<[^>]+>)[^ `]*/|\./)?"
    r"scripts/(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.(?:py|sh))(?=[ `])",
    re.IGNORECASE,
)


def _documented_direct_control_scripts() -> dict[pathlib.Path, set[pathlib.Path]]:
    """Find DEFT control programs that app docs instruct agents to execute."""
    found: dict[pathlib.Path, set[pathlib.Path]] = {}
    for name in SKILL_NAMES:
        skill_root = APPLICATIONS_ROOT / name
        docs = [skill_root / "SKILL.md"]
        references = skill_root / "references"
        if references.is_dir():
            docs.extend(sorted(references.rglob("*.md")))
        for doc in docs:
            if not doc.is_file():
                continue
            names = {
                match.group("name")
                for match in SCRIPT_REFERENCE_RE.finditer(
                    doc.read_text(encoding="utf-8")
                )
                if match.group("name") in DIRECT_CONTROL_SCRIPTS
            }
            for name in names:
                found.setdefault(skill_root, set()).add(skill_root / "scripts" / name)
    return found


def _documented_bare_invocations(skill_root: pathlib.Path) -> list[str]:
    """Return shell command positions that execute a script without an interpreter."""
    docs = [skill_root / "SKILL.md"]
    docs.extend(sorted((skill_root / "references").glob("*.md")))
    found: list[str] = []
    for doc in docs:
        text = SHELL_CONTINUATION_RE.sub(" ", doc.read_text(encoding="utf-8"))
        for pattern in (BARE_SCRIPT_EXEC_RE, INLINE_BARE_SCRIPT_EXEC_RE):
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                found.append(
                    f"{doc.relative_to(skill_root)}:{line}: {match.group('command')}"
                )
    return found


def _strip_file_exec_bits(root: pathlib.Path) -> None:
    for path in root.rglob("*"):
        if path.is_file():
            path.chmod(0o644)


def _companion_source(category: str, name: str) -> pathlib.Path:
    candidates = (
        SKILL_ROOT.parent / name,
        REPO_ROOT / "skills" / category / name,
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"required companion skill {name!r} not found in: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


class ExecBitAndPackagingTests(unittest.TestCase):
    def test_deft_python_toml_probe_handles_python_310_and_311(self) -> None:
        launcher_source = (SKILL_ROOT / "scripts/deft_python.sh").read_text(
            encoding="utf-8"
        )
        isolated_source, replacements = re.subn(
            r"candidates=\(\n.*?\n\)",
            'candidates=(\n  "${DEFT_PYTHON:-}"\n)',
            launcher_source,
            count=1,
            flags=re.DOTALL,
        )
        self.assertEqual(replacements, 1)

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            launcher = root / "deft_python.sh"
            launcher.write_text(isolated_source, encoding="utf-8")
            launcher.chmod(0o755)

            candidate = root / "simulated-python"
            candidate.write_text(
                f"#!{sys.executable}\n"
                "import builtins, os, sys, types\n"
                "version = tuple(int(part) for part in os.environ['SIM_VERSION'].split('.'))\n"
                "available = set(filter(None, os.environ.get('SIM_MODULES', '').split(',')))\n"
                "real_import = builtins.__import__\n"
                "def controlled_import(name, globals=None, locals=None, fromlist=(), level=0):\n"
                "    root = name.split('.', 1)[0]\n"
                "    if root in {'pyarrow', 'yaml', 'tomli', 'tomllib'}:\n"
                "        if root not in available:\n"
                "            raise ModuleNotFoundError(f\"No module named '{root}'\", name=root)\n"
                "        return types.ModuleType(root)\n"
                "    return real_import(name, globals, locals, fromlist, level)\n"
                "sys.version_info = version\n"
                "builtins.__import__ = controlled_import\n"
                "exec(sys.argv[2], {})\n",
                encoding="utf-8",
            )
            candidate.chmod(0o755)

            cases = (
                ("3.10 without tomli", "3.10", "pyarrow,yaml", False),
                ("3.10 with tomli", "3.10", "pyarrow,yaml,tomli", True),
                ("3.11+ with tomllib", "3.11", "pyarrow,yaml,tomllib", True),
            )
            for label, version, modules, succeeds in cases:
                with self.subTest(label=label):
                    environment = os.environ.copy()
                    environment.update(
                        {
                            "DEFT_PYTHON": str(candidate),
                            "SIM_VERSION": version,
                            "SIM_MODULES": modules,
                        }
                    )
                    result = subprocess.run(
                        ["bash", str(launcher)],
                        env=environment,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    if succeeds:
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertEqual(result.stdout.strip(), str(candidate))
                    else:
                        self.assertEqual(result.returncode, 2)
                        self.assertIn(
                            "Python 3.11+ tomllib or Python 3.10 with tomli",
                            result.stderr,
                        )
                        self.assertNotIn("Traceback", result.stderr)

    def test_documented_invocations_survive_exec_bit_stripped_install(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            skills_root = pathlib.Path(temporary) / "skills-only"
            skills_root.mkdir(parents=True)
            installed = []
            for name in SKILL_NAMES:
                destination = skills_root / name
                shutil.copytree(
                    APPLICATIONS_ROOT / name,
                    destination,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                )
                _strip_file_exec_bits(destination)
                installed.append(destination)

            bare = [
                invocation
                for skill_root in installed
                for invocation in _documented_bare_invocations(skill_root)
            ]
            self.assertFalse(
                bare,
                "documented commands depend on an executable script bit: "
                + ", ".join(bare),
            )

            environment = os.environ.copy()
            environment["DEFT_PYTHON"] = "/bin/true"
            for skill_root in installed:
                launcher = skill_root / "scripts" / "deft_python.sh"
                self.assertEqual(launcher.stat().st_mode & 0o111, 0)
                result = subprocess.run(
                    ["bash", str(launcher)],
                    cwd=skill_root,
                    env=environment,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), "/bin/true")

    def test_control_scripts_remain_executable_in_repo(self) -> None:
        documented = _documented_direct_control_scripts()
        self.assertTrue(documented, "no directly invoked DEFT control scripts found")
        self.assertTrue(
            {
                "tao-run-deft-aoi",
                "tao-run-deft-aoi-cosmos3",
            }.issubset(skill.name for skill in documented),
            "both DEFT AOI variants must be covered",
        )

        missing = sorted(
            str(script)
            for scripts in documented.values()
            for script in scripts
            if not script.is_file()
        )
        self.assertFalse(missing, f"documented scripts do not exist: {missing}")

        not_executable = sorted(
            str(script)
            for scripts in documented.values()
            for script in scripts
            if not script.stat().st_mode & 0o111
        )
        self.assertFalse(
            not_executable,
            f"documented direct commands are not executable: {not_executable}",
        )

        if not (REPO_ROOT / ".git").exists():
            return
        relative = sorted(
            str(script.relative_to(REPO_ROOT))
            for scripts in documented.values()
            for script in scripts
        )
        result = subprocess.run(
            ["git", "ls-files", "-s", "--", *relative],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return
        recorded = {
            line.split("\t")[-1]: line.split()[0]
            for line in result.stdout.splitlines()
            if "\t" in line
        }
        stale = sorted(path for path in relative if recorded.get(path) != "100755")
        self.assertFalse(stale, f"executable bit is missing from git: {stale}")

    def test_cosmos3_suite_discovers_from_flattened_plugin_layout(self) -> None:
        sources = (
            ("applications", "tao-run-deft-aoi-cosmos3"),
            ("applications", "tao-run-deft-aoi"),
            ("models", "tao-finetune-cosmos-reason"),
            ("data", "tao-mine-aoi-images"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            skills_root = pathlib.Path(temporary) / "skills-only"
            skills_root.mkdir(parents=True)
            for category, name in sources:
                shutil.copytree(
                    _companion_source(category, name),
                    skills_root / name,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                )

            environment = os.environ.copy()
            environment.pop("PYTHONPATH", None)
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "unittest",
                    "discover",
                    "-s",
                    "tests",
                    "-p",
                    "test_cosmos3_*.py",
                ],
                cwd=skills_root / "tao-run-deft-aoi-cosmos3",
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            output = result.stdout + result.stderr
            self.assertEqual(result.returncode, 0, output)
            self.assertIn("OK", output)


if __name__ == "__main__":
    unittest.main()
