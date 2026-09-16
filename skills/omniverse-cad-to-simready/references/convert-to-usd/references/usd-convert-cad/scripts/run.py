#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "shared"))

from preflight_manifest import (
    load_preflight_manifest,
    preflight_required,
    preflight_status_check,
    ready_executable_from_runtime,
    ready_path_from_upstream,
)
from script_utils import emit_json_report, subprocess_output, tail_text
from upstream_versions import UPSTREAM_VERSION_MANIFEST_PATH, package_requirement, package_version


SKILL = "usd-convert-cad"
TOOL = "usd-convert-cad"
NEXT_STEP = "validate-usd-minimum"
UPSTREAM_REPO_URL = "https://github.com/NVIDIA-Omniverse/usd-convert-cad"
UPSTREAM_SKILL_URL = "https://github.com/NVIDIA-Omniverse/usd-convert-cad/blob/v0.2.0/skills/omniverse-cad-to-usd/SKILL.md"

# usd-convert-cad 0.2 ships as a self-contained Python wheel. It exposes the
# console command `usd-convert-cad` (equivalently `python -m usd_convert_cad`)
# with `-i <input> -o <output>` and bundles its own OpenUSD (`pxr`) runtime, so
# it must be installed into an isolated Python 3.12 interpreter. There is no Kit
# app, no extension-registry pull, and no upstream `convert.py`/`validate.py`.
WHEEL_MODULE = "usd_convert_cad"
PIP_PACKAGE = package_requirement("usd-convert-cad")
EXPECTED_WHEEL_VERSION = package_version("usd-convert-cad")
CONVERTER_PYTHON_ENV = "USD_CONVERT_CAD_PYTHON"
INSTALL_HINT = (
    "Install the usd-convert-cad wheel into an isolated Python 3.12 environment, then point this "
    f"reference at that interpreter with --usd-convert-cad-python or {CONVERTER_PYTHON_ENV}: "
    'python3.12 -m venv "$HOME/.omniverse-cad-to-simready/venvs/usd-convert-cad" '
    '&& "$HOME/.omniverse-cad-to-simready/venvs/usd-convert-cad/bin/python" -m pip install '
    f"{PIP_PACKAGE} --extra-index-url https://pypi.nvidia.com "
    f'&& export {CONVERTER_PYTHON_ENV}="$HOME/.omniverse-cad-to-simready/venvs/usd-convert-cad/bin/python"'
)

USD_OUTPUT_SUFFIXES = {".usd", ".usda", ".usdc"}
IMPORT_CHECK_TIMEOUT_SECONDS = 120

# The authoritative source-format list lives in the usd-convert-cad SKILL.md
# Supported Formats table. When a checkout of that repo is available, the probe
# parses the table live so upstream format changes flow through automatically;
# see resolve_skill_md / parse_supported_suffixes_from_skill below. These env
# vars/paths locate that checkout (they never affect the conversion runtime,
# which is always the installed wheel).
CONVERTER_ROOT_ENV = "USD_CONVERT_CAD_ROOT"
UPSTREAM_ROOT_ENV = "OMNIVERSE_CAD_TO_SIMREADY_UPSTREAM_ROOT"
SKILL_MD_ENV = "USD_CONVERT_CAD_SKILL"
# Candidate SKILL.md locations within a usd-convert-cad checkout, most specific
# first. A glob fallback keeps this working if the upstream layout moves.
SKILL_MD_CANDIDATES = (
    Path("skills") / "omniverse-cad-to-usd" / "SKILL.md",
    Path(".agents") / "skills" / "usd-convert-cad" / "SKILL.md",
)

# Fallback snapshot of the usd-convert-cad Supported Formats table, used only
# when no SKILL.md checkout is reachable (for example, fully offline runs).
# Keep it in sync with the upstream SKILL.md.
SUPPORTED_CAD_SUFFIXES = frozenset(
    {
        ".jt",
        ".dgn",
        ".catpart",
        ".catproduct",
        ".cgr",
        ".3dxml",
        ".ifc",
        ".ifczip",
        ".prt",
        ".asm",
        ".xmt",
        ".x_t",
        ".x_b",
        ".xmt_txt",
        ".sldprt",
        ".sldasm",
        ".stl",
        ".ipt",
        ".iam",
        ".dwg",
        ".dxf",
        ".rvt",
        ".rfa",
        ".par",
        ".pwd",
        ".psm",
        ".stp",
        ".step",
        ".igs",
        ".iges",
        ".3dm",
        ".dae",
        ".fbx",
        ".obj",
        ".3ds",
        ".3mf",
        ".gltf",
        ".glb",
        ".sat",
        ".sab",
    }
)
LINUX_AARCH64_UNSUPPORTED_SUFFIXES = frozenset({".dwg", ".dxf", ".rvt", ".rfa"})


@dataclass(frozen=True)
class ConversionReport:
    source_asset_path: str
    source_format: str
    converter_skill: str
    converter_tool: str
    converter_command: list[str]
    output_directory: str
    output_usd_path: str
    generated_files: list[str]
    sidecar_inputs: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    install_hint: str = ""
    next_step: str = NEXT_STEP

    @property
    def passed(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "source_asset_path": self.source_asset_path,
            "source_format": self.source_format,
            "converter_skill": self.converter_skill,
            "converter_tool": self.converter_tool,
            "converter_command": self.converter_command,
            "output_directory": self.output_directory,
            "output_usd_path": self.output_usd_path,
            "generated_files": self.generated_files,
            "sidecar_inputs": self.sidecar_inputs,
            "warnings": self.warnings,
            "errors": self.errors,
            "next_step": self.next_step,
        }
        if self.diagnostics:
            payload["diagnostics"] = self.diagnostics
        if self.install_hint:
            payload["install_hint"] = self.install_hint
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_markdown(self) -> str:
        lines = [
            "# Conversion Report",
            "",
            f"- Source asset: `{self.source_asset_path}`",
            f"- Source format: `{self.source_format}`",
            f"- Converter skill: `{self.converter_skill}`",
            f"- Converter tool: `{self.converter_tool}`",
            f"- Converter command: `{' '.join(self.converter_command)}`",
            f"- Output directory: `{self.output_directory}`",
            f"- Output USD: `{self.output_usd_path}`",
            f"- Next step: `{self.next_step}`",
            "",
            "## Generated Files",
            "",
        ]
        lines.extend(f"- `{path}`" for path in self.generated_files)
        if not self.generated_files:
            lines.append("- None")
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in self.warnings)
        if not self.warnings:
            lines.append("- None")
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in self.errors)
        if not self.errors:
            lines.append("- None")
        if self.install_hint:
            lines.extend(["", "## Install Hint", "", self.install_hint])
        lines.append("")
        return "\n".join(lines)


def _absolute_without_resolving_symlinks(path: Path) -> Path:
    """Return an absolute path while preserving a virtualenv Python symlink."""
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def resolve_converter_python(explicit: Path | None) -> Path:
    """Resolve the Python interpreter that has the usd-convert-cad wheel.

    Because the wheel bundles its own ``pxr`` runtime it is expected to live in
    a dedicated interpreter rather than the orchestrator's Python. Resolution
    order: explicit ``--usd-convert-cad-python`` -> ``USD_CONVERT_CAD_PYTHON``
    env -> preflight manifest ``usd_convert_cad`` runtime executable -> the
    current interpreter as a last resort.
    """
    if explicit is not None:
        return _absolute_without_resolving_symlinks(explicit)
    env_python = os.environ.get(CONVERTER_PYTHON_ENV)
    if env_python:
        return _absolute_without_resolving_symlinks(Path(env_python))
    manifest, _, _ = load_preflight_manifest()
    manifest_python = ready_executable_from_runtime(manifest, "usd_convert_cad")
    if manifest_python:
        return _absolute_without_resolving_symlinks(Path(manifest_python))
    return _absolute_without_resolving_symlinks(Path(sys.executable))


def wheel_version(python: Path) -> tuple[str | None, str]:
    """Return the installed wheel version, or ``None`` with a failure detail."""
    command = [
        str(python),
        "-c",
        (
            "import usd_convert_cad as m; "
            "print(getattr(m, '__version__', None) or "
            "(m.get_version() if hasattr(m, 'get_version') else 'unknown'))"
        ),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=IMPORT_CHECK_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"could not run {python}: {exc}"
    if completed.returncode == 0:
        return completed.stdout.strip() or "unknown", ""
    detail = subprocess_output(completed.stdout, completed.stderr) or f"exit {completed.returncode}"
    return None, tail_text(detail)


def real_suffix(source_asset: Path) -> str:
    suffix = source_asset.suffix.lower()
    if suffix.lstrip(".").isdigit():
        return Path(source_asset.stem).suffix.lower()
    return suffix


def default_upstream_checkout() -> Path:
    root = os.environ.get(UPSTREAM_ROOT_ENV)
    if root:
        return Path(root).expanduser() / "usd-convert-cad"
    return Path.home() / ".omniverse-cad-to-simready" / "upstreams" / "usd-convert-cad"


def _skill_md_in_checkout(root: Path) -> Path | None:
    root = root.expanduser()
    if root.is_file():
        return root
    for relative in SKILL_MD_CANDIDATES:
        candidate = root / relative
        if candidate.is_file():
            return candidate
    if root.exists():
        matches = sorted(root.glob("**/SKILL.md"))
        if matches:
            return matches[0]
    return None


def resolve_skill_md(explicit_skill: Path | None, explicit_root: Path | None) -> Path | None:
    """Locate the usd-convert-cad SKILL.md that defines the Supported Formats.

    Order: explicit ``--usd-convert-cad-skill`` -> ``USD_CONVERT_CAD_SKILL`` env
    -> explicit ``--usd-convert-cad-root`` -> preflight manifest upstream path
    -> ``USD_CONVERT_CAD_ROOT`` env -> the default shared upstream checkout.
    Returns ``None`` when no checkout is reachable (the probe then falls back to
    the built-in snapshot).
    """
    for candidate_root in (explicit_skill, os.environ.get(SKILL_MD_ENV), explicit_root):
        if candidate_root:
            found = _skill_md_in_checkout(Path(candidate_root))
            if found is not None:
                return found
    manifest, _, _ = load_preflight_manifest()
    upstream_path = ready_path_from_upstream(manifest, "usd_convert_cad")
    if upstream_path is not None:
        found = _skill_md_in_checkout(upstream_path)
        if found is not None:
            return found
    env_root = os.environ.get(CONVERTER_ROOT_ENV)
    if env_root:
        found = _skill_md_in_checkout(Path(env_root))
        if found is not None:
            return found
    return _skill_md_in_checkout(default_upstream_checkout())


def parse_supported_suffixes_from_skill(skill_md: Path) -> set[str] | None:
    """Parse the Supported Formats markdown table from a usd-convert-cad SKILL.md."""
    try:
        text = skill_md.read_text(encoding="utf-8")
    except OSError:
        return None
    collecting = False
    suffixes: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("## "):
            collecting = stripped.lower() == "## supported formats"
            continue
        if not collecting or not stripped.startswith("|"):
            continue
        first_column = stripped.strip("|").split("|")[0]
        for token in re.findall(r"`([^`]+)`", first_column):
            token = token.strip().lower()
            if token.startswith("."):
                suffixes.add(token)
    return suffixes or None


def supported_cad_suffixes(skill_md: Path | None) -> tuple[frozenset[str], str]:
    """Return (suffixes, human-readable source) from SKILL.md or the snapshot."""
    if skill_md is not None:
        parsed = parse_supported_suffixes_from_skill(skill_md)
        if parsed:
            return frozenset(parsed), f"usd-convert-cad SKILL.md ({skill_md})"
    return SUPPORTED_CAD_SUFFIXES, "built-in snapshot of the usd-convert-cad Supported Formats table"


def platform_limitation(suffix: str) -> str | None:
    machine = platform.machine().lower()
    if (
        platform.system().lower() == "linux"
        and machine in {"aarch64", "arm64"}
        and suffix in LINUX_AARCH64_UNSUPPORTED_SUFFIXES
    ):
        return (
            f"usd-convert-cad does not support {suffix} on linux-aarch64; "
            "the upstream DWG/Revit readers are unavailable on this platform"
        )
    return None


def probe_source(
    source_asset: Path,
    *,
    converter_python: Path | None = None,
    skill_md: Path | None = None,
) -> dict[str, Any]:
    source_asset = source_asset.resolve()
    if preflight_required() and converter_python is None:
        preflight_check = preflight_status_check("usd-convert-cad", "usd_convert_cad")
        if not preflight_check["passed"]:
            return {
                "source_asset_path": str(source_asset),
                "source_format": "unknown",
                "converter_skill": SKILL,
                "converter_tool": TOOL,
                "supported": False,
                "sidecar_inputs": [],
                "warnings": [],
                "errors": [preflight_check["message"]],
                "install_hint": preflight_check["message"],
            }
    suffix = real_suffix(source_asset)
    suffixes, source_label = supported_cad_suffixes(skill_md)
    limitation = platform_limitation(suffix)
    supported = suffix in suffixes and limitation is None
    warnings = [f"Capability lookup uses the {source_label}."]
    if limitation:
        warnings.append(limitation)
    elif not supported:
        warnings.append(f"usd-convert-cad does not list source suffix: {suffix or 'unknown'}")
    return {
        "source_asset_path": str(source_asset),
        "source_format": "cad" if supported else "unknown",
        "converter_skill": SKILL,
        "converter_tool": TOOL,
        "supported": supported,
        "sidecar_inputs": [],
        "warnings": warnings,
        "errors": [],
        "install_hint": "",
    }


def cad_to_usd(
    source_asset: Path,
    output_directory: Path,
    *,
    converter_python: Path | None = None,
    skill_md: Path | None = None,
    output_extension: str = ".usd",
    timeout: int = 1800,
    extra_args: list[str] | None = None,
    legacy_backend: str = "auto",
) -> ConversionReport:
    source_asset = source_asset.resolve()
    output_directory = output_directory.resolve()
    output_extension = output_extension if output_extension.startswith(".") else f".{output_extension}"
    output_usd_path = output_directory / f"{source_asset.stem}{output_extension}"
    extra_args = list(extra_args or [])

    python = resolve_converter_python(converter_python)
    command = [
        str(python),
        "-m",
        WHEEL_MODULE,
        "-i",
        str(source_asset),
        "-o",
        str(output_usd_path),
        *extra_args,
    ]
    warnings = [
        f"Delegating CAD conversion to the {TOOL} wheel: {UPSTREAM_REPO_URL}.",
        f"Upstream agent skill reference: {UPSTREAM_SKILL_URL}.",
        f"Converter interpreter: {python}.",
    ]
    if legacy_backend != "auto":
        warnings.append(
            "usd-convert-cad 0.2 no longer exposes backend selection; "
            f"ignored legacy --backend {legacy_backend}."
        )
    errors: list[str] = []

    if preflight_required() and converter_python is None:
        preflight_check = preflight_status_check("usd-convert-cad", "usd_convert_cad")
        if not preflight_check["passed"]:
            errors.append(preflight_check["message"])

    suffixes, source_label = supported_cad_suffixes(skill_md)
    warnings.append(f"Supported-format source: {source_label}.")
    source_suffix = real_suffix(source_asset)
    if source_suffix not in suffixes:
        errors.append(f"unsupported CAD source format: {source_suffix or 'unknown'}")
    limitation = platform_limitation(source_suffix)
    if limitation:
        errors.append(limitation)
    if not source_asset.exists():
        errors.append(f"source asset does not exist: {source_asset}")
    if output_extension.lower() not in USD_OUTPUT_SUFFIXES:
        errors.append(f"unsupported USD output extension: {output_extension}")

    version, version_error = wheel_version(python)
    if version is None:
        errors.append(
            f"usd-convert-cad wheel is not importable with {python}: {version_error}"
        )
    elif version != EXPECTED_WHEEL_VERSION:
        errors.append(
            f"usd-convert-cad requires version {EXPECTED_WHEEL_VERSION}, found {version}; "
            f"see {UPSTREAM_VERSION_MANIFEST_PATH}"
        )
    else:
        warnings.append(f"usd-convert-cad wheel version: {version}.")

    if errors:
        install_hint = INSTALL_HINT if version != EXPECTED_WHEEL_VERSION else ""
        return _report(source_asset, output_directory, output_usd_path, command, python, warnings, errors, install_hint)

    output_directory.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        detail = subprocess_output(getattr(exc, "stdout", ""), getattr(exc, "stderr", ""))
        errors.append(
            f"usd-convert-cad timed out after {timeout}s."
            + (f" Output: {tail_text(detail)}" if detail else "")
        )
        return _report(source_asset, output_directory, output_usd_path, command, python, warnings, errors)

    if completed.returncode != 0:
        detail = subprocess_output(completed.stdout, completed.stderr) or f"{TOOL} exited with {completed.returncode}"
        errors.append(tail_text(detail))
    if not output_usd_path.exists() and not errors:
        errors.append(f"converter did not produce expected USD output: {output_usd_path}")
    return _report(source_asset, output_directory, output_usd_path, command, python, warnings, errors)


def discover_generated_files(output_directory: Path) -> list[str]:
    if not output_directory.exists():
        return []
    return sorted(
        str(path.relative_to(output_directory))
        for path in output_directory.rglob("*")
        if path.is_file()
    )


def _report(
    source_asset: Path,
    output_directory: Path,
    output_usd_path: Path,
    command: list[str],
    converter_python: Path,
    warnings: list[str],
    errors: list[str],
    install_hint: str = "",
) -> ConversionReport:
    return ConversionReport(
        source_asset_path=str(source_asset),
        source_format="cad",
        converter_skill=SKILL,
        converter_tool=TOOL,
        converter_command=command,
        output_directory=str(output_directory),
        output_usd_path=str(output_usd_path) if output_usd_path.exists() else "",
        generated_files=discover_generated_files(output_directory),
        sidecar_inputs=[str(converter_python)],
        warnings=warnings,
        errors=errors,
        install_hint=install_hint,
    )


def emit_report(
    report: ConversionReport,
    *,
    report_path: Path | None = None,
    markdown_report_path: Path | None = None,
) -> None:
    report_json = report.to_json()
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report_json, encoding="utf-8")
    if markdown_report_path is not None:
        markdown_report_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_report_path.write_text(report.to_markdown(), encoding="utf-8")
    print(report_json, end="")


def emit_probe(payload: dict[str, Any], *, report_path: Path | None = None) -> None:
    emit_json_report(payload, report_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert supported source assets to OpenUSD with the usd-convert-cad wheel."
    )
    parser.add_argument("source_asset", type=Path)
    parser.add_argument("output_directory", type=Path, nargs="?")
    parser.add_argument("--probe", action="store_true", help="Report whether usd-convert-cad claims this source format.")
    parser.add_argument(
        "--usd-convert-cad-python",
        type=Path,
        help=(
            "Python interpreter that has the usd-convert-cad wheel installed. "
            f"Defaults to {CONVERTER_PYTHON_ENV}, the preflight manifest, then this interpreter."
        ),
    )
    parser.add_argument("--backend", default="auto", help=argparse.SUPPRESS)
    parser.add_argument(
        "--usd-convert-cad-root",
        type=Path,
        help=(
            "Path to a usd-convert-cad checkout used only to read its SKILL.md Supported Formats "
            f"table. Defaults to {SKILL_MD_ENV}/{CONVERTER_ROOT_ENV}, the preflight manifest, then "
            "the shared upstream checkout. Falls back to a built-in snapshot when unavailable."
        ),
    )
    parser.add_argument(
        "--usd-convert-cad-skill",
        type=Path,
        help="Explicit path to the usd-convert-cad SKILL.md (overrides --usd-convert-cad-root discovery).",
    )
    parser.add_argument("--output-extension", default=".usd", choices=sorted(USD_OUTPUT_SUFFIXES))
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--report", type=Path, help="Write a JSON report to this path.")
    parser.add_argument("--markdown-report", type=Path, help="Write a Markdown report to this path.")
    args, passthrough = parser.parse_known_args(argv)

    skill_md = resolve_skill_md(args.usd_convert_cad_skill, args.usd_convert_cad_root)

    if args.probe:
        payload = probe_source(
            args.source_asset,
            converter_python=args.usd_convert_cad_python,
            skill_md=skill_md,
        )
        emit_probe(payload, report_path=args.report)
        return 0 if payload["supported"] else 1
    if args.output_directory is None:
        parser.error("output_directory is required unless --probe is used")

    report = cad_to_usd(
        args.source_asset,
        args.output_directory,
        converter_python=args.usd_convert_cad_python,
        skill_md=skill_md,
        output_extension=args.output_extension,
        timeout=args.timeout,
        extra_args=passthrough,
        legacy_backend=args.backend,
    )
    emit_report(report, report_path=args.report, markdown_report_path=args.markdown_report)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
