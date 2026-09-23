# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the DEFT workflow host-Python prerequisite contract."""

from __future__ import annotations

import ast
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
REQUIRED_IMPORT_PACKAGES = {
    "huggingface_hub": "huggingface_hub",
    "numpy": "numpy",
    "pandas": "pandas",
    "pyarrow": "pyarrow",
    "yaml": "pyyaml",
}
INSTALL_PACKAGES = "numpy pandas pyarrow pyyaml huggingface_hub"
KNOWN_STDLIB_IMPORTS = {
    "__future__",
    "argparse",
    "collections",
    "copy",
    "datetime",
    "hashlib",
    "json",
    "os",
    "pathlib",
    "re",
    "shutil",
    "subprocess",
    "sys",
    "tempfile",
    "tomllib",
    "typing",
}


def third_party_script_imports() -> set[str]:
    """Return non-stdlib imports used by the workflow's Python helpers."""
    local_modules = {path.stem for path in SCRIPTS.glob("*.py")}
    imported: set[str] = set()
    for path in SCRIPTS.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.partition(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.partition(".")[0])
    stdlib_modules = getattr(sys, "stdlib_module_names", KNOWN_STDLIB_IMPORTS)
    return imported - stdlib_modules - local_modules - {"tomli"}


class PythonSelectorTests(unittest.TestCase):
    """Verify selection and failure guidance without installing packages."""

    def test_prefers_workspace_virtualenv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = pathlib.Path(temporary)
            interpreter = workspace / ".venv/bin/python"
            interpreter.parent.mkdir(parents=True)
            interpreter.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            interpreter.chmod(0o755)

            result = subprocess.run(
                ["/bin/bash", str(SCRIPTS / "deft_python.sh")],
                check=True,
                capture_output=True,
                text=True,
                env={**os.environ, "WORKSPACE_DIR": str(workspace), "DEFT_PYTHON": ""},
            )

            self.assertEqual(result.stdout.strip(), str(interpreter))

    def test_missing_dependencies_prints_workspace_install(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            workspace = root / "workspace"
            bin_dir = root / "bin"
            bin_dir.mkdir()
            for name in ("dirname", "cat"):
                source = shutil.which(name)
                self.assertIsNotNone(source)
                os.symlink(source, bin_dir / name)
            failing_python = bin_dir / "python3"
            failing_python.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
            failing_python.chmod(0o755)

            result = subprocess.run(
                ["/bin/bash", str(SCRIPTS / "deft_python.sh")],
                check=False,
                capture_output=True,
                text=True,
                env={
                    "PATH": str(bin_dir),
                    "WORKSPACE_DIR": str(workspace),
                    "DEFT_PYTHON": str(failing_python),
                },
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn(f'-m venv "{workspace}/.venv"', result.stderr)
            self.assertIn("-m pip --version", result.stderr)
            self.assertIn("-m ensurepip --upgrade", result.stderr)
            self.assertIn(f"-m pip install {INSTALL_PACKAGES}", result.stderr)
            self.assertIn("a global pip executable is not required", result.stderr)
            self.assertNotIn("requirements.txt", result.stderr)

    def test_probe_covers_every_third_party_script_import(self) -> None:
        selector = (SCRIPTS / "deft_python.sh").read_text(encoding="utf-8")
        self.assertEqual(third_party_script_imports(), set(REQUIRED_IMPORT_PACKAGES))
        for import_name in REQUIRED_IMPORT_PACKAGES:
            self.assertIn(import_name, selector)


class DocumentationTests(unittest.TestCase):
    """Keep documented helper invocations on the selected interpreter."""

    def test_documented_helpers_use_selected_python(self) -> None:
        documentation = "\n".join(
            (SKILL_ROOT / path).read_text(encoding="utf-8")
            for path in (
                "SKILL.md",
                "references/host-prerequisites.md",
                "references/mining-loop.md",
            )
        )
        self.assertNotIn('python3 "$DEFT_SKILL_ROOT/scripts/', documentation)
        self.assertIn('"$DEFT_PYTHON" "$DEFT_SKILL_ROOT/scripts/', documentation)

    def test_host_dependencies_do_not_use_a_skill_manifest(self) -> None:
        self.assertFalse((SKILL_ROOT / "requirements.txt").exists())
        skill = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
        prerequisites = (SKILL_ROOT / "references/host-prerequisites.md").read_text(
            encoding="utf-8"
        )
        documentation = skill + prerequisites
        self.assertNotIn("requirements.txt", documentation)
        self.assertIn(f"pip install {INSTALL_PACKAGES}", documentation)
        self.assertIn("ensurepip", documentation)
        self.assertIn("a globally installed `pip` executable is\nnot required", documentation)
        for package_name in REQUIRED_IMPORT_PACKAGES.values():
            self.assertIn(package_name, documentation)
        self.assertIn("references/host-prerequisites.md", skill)


if __name__ == "__main__":
    unittest.main()
