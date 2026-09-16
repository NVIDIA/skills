#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


UNRESOLVED_TEXTURE_EXTENSIONS = {
    ".bmp",
    ".dds",
    ".exr",
    ".gif",
    ".hdr",
    ".jpeg",
    ".jpg",
    ".png",
    ".tga",
    ".tif",
    ".tiff",
    ".tx",
    ".webp",
}


class OpenUSDUnavailableError(RuntimeError):
    pass


class OpenUSDInspectionError(RuntimeError):
    pass


class AlternateOpenUSDRuntimeUnavailable(RuntimeError):
    pass


class UploadPreparationError(RuntimeError):
    def __init__(self, message: str, upload_info: dict[str, Any]) -> None:
        super().__init__(message)
        self.upload_info = upload_info


def _usd_utils() -> Any:
    try:
        from pxr import UsdUtils
    except Exception as exc:
        raise OpenUSDUnavailableError(str(exc)) from exc
    return UsdUtils


def _base_upload_info(asset_path: Path) -> dict[str, Any]:
    return {
        "asset_path": str(asset_path),
        "dependency_layers": [],
        "dependency_assets": [],
        "dependency_count": 0,
        "inspection_runtime": None,
        "inspection_executable": None,
        "inspection_error": None,
        "packaging": "none",
        "package_size_bytes": None,
        "path": str(asset_path),
        "unresolved_paths": [],
        "warning": None,
    }


def _layer_identifier(layer: Any) -> str:
    return str(getattr(layer, "realPath", None) or getattr(layer, "identifier", "") or "")


def _is_texture_dependency(path: str) -> bool:
    clean_path = path.strip().strip("@").split("?", 1)[0].split("#", 1)[0]
    return Path(clean_path).suffix.lower() in UNRESOLVED_TEXTURE_EXTENSIONS


def _preparation_error(upload_info: dict[str, Any], message: str) -> UploadPreparationError:
    upload_info["inspection_error"] = message
    upload_info["packaging"] = "failed"
    return UploadPreparationError(message, upload_info)


def prepare_upload_asset(
    asset_path: Path,
    output_directory: Path,
    *,
    allow_missing_textures: bool = False,
) -> tuple[Path, dict[str, Any]]:
    upload_info = _base_upload_info(asset_path)
    if asset_path.suffix.lower() == ".usdz":
        upload_info["inspection_runtime"] = "not-required-already-usdz"
        upload_info["packaging"] = "already_usdz"
        return asset_path, upload_info

    usd_utils = _usd_utils()
    upload_info["inspection_runtime"] = "process-python"
    upload_info["inspection_executable"] = sys.executable
    try:
        layers, assets, unresolved_paths = usd_utils.ComputeAllDependencies(str(asset_path))
    except Exception as exc:
        raise OpenUSDInspectionError(f"Could not inspect USD dependencies: {exc}") from exc

    root_path = asset_path.resolve()
    dependency_layers: list[str] = []
    for layer in layers:
        identifier = _layer_identifier(layer)
        if not identifier:
            continue
        try:
            if Path(identifier).resolve() == root_path:
                continue
        except OSError:
            pass
        dependency_layers.append(identifier)

    dependency_assets = [str(asset) for asset in assets]
    unresolved = [str(path) for path in unresolved_paths]
    upload_info["dependency_layers"] = dependency_layers
    upload_info["dependency_assets"] = dependency_assets
    upload_info["unresolved_paths"] = unresolved
    upload_info["dependency_count"] = len(dependency_layers) + len(dependency_assets)

    non_texture_unresolved = [path for path in unresolved if not _is_texture_dependency(path)]
    if unresolved and (not allow_missing_textures or non_texture_unresolved):
        blocked = non_texture_unresolved or unresolved
        raise _preparation_error(
            upload_info,
            "Cannot package USD for Content Agents upload; unresolved dependencies: " + ", ".join(blocked),
        )
    if unresolved:
        upload_info["warning"] = (
            "Continuing Material Agent upload despite unresolved texture dependencies: " + ", ".join(unresolved)
        )
    if upload_info["dependency_count"] == 0:
        if unresolved:
            upload_info["packaging"] = "missing_textures_passthrough"
        return asset_path, upload_info

    output_directory.mkdir(parents=True, exist_ok=True)
    package_path = output_directory / f"{asset_path.stem}_content_agents_upload.usdz"
    package_path.unlink(missing_ok=True)
    package_error: Exception | None = None
    try:
        ok = usd_utils.CreateNewUsdzPackage(str(asset_path), str(package_path))
    except Exception as exc:
        if not unresolved:
            raise _preparation_error(
                upload_info,
                f"OpenUSD failed to package USD dependencies for Content Agents upload: {exc}",
            ) from exc
        ok = False
        package_error = exc
    if not ok or not package_path.exists() or package_path.stat().st_size == 0:
        if unresolved:
            package_path.unlink(missing_ok=True)
            message = (
                "OpenUSD could not package the incomplete texture set; refusing to upload the original USD "
                "because it would omit resolved dependencies"
            )
            if package_error is not None:
                message += f": {package_error}"
            raise _preparation_error(upload_info, message)
        raise _preparation_error(
            upload_info,
            f"OpenUSD failed to package USD dependencies for Content Agents upload: {package_path}",
        )

    upload_info["packaging"] = "usdz"
    upload_info["path"] = str(package_path.resolve())
    upload_info["package_size_bytes"] = package_path.stat().st_size
    return package_path, upload_info


def _project_root() -> Path | None:
    # Only trust project metadata discovered from the shipped helper. The
    # caller's working directory may be an untrusted asset checkout.
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return None


def openusd_python_commands() -> list[tuple[list[str], str]]:
    commands: list[tuple[list[str], str]] = []
    seen_pythons: set[str] = set()

    def add_python(raw_path: str | Path | None, label: str) -> None:
        if not raw_path:
            return
        path = Path(raw_path).expanduser()
        if not path.is_file():
            return
        # Preserve venv interpreter symlinks so the executed Python retains its
        # environment-specific site-packages.
        executable = str(path.absolute())
        if executable in seen_pythons:
            return
        seen_pythons.add(executable)
        commands.append(([executable], label))

    add_python(os.getenv("CONTENT_AGENTS_OPENUSD_PYTHON"), "configured-openusd-python")
    add_python(os.getenv("USD_CONVERT_CAD_PYTHON"), "usd-convert-cad-python")

    project_root = _project_root()
    if project_root is not None:
        add_python(project_root / ".venv" / "bin" / "python", "project-venv")
        add_python(project_root / ".venv" / "Scripts" / "python.exe", "project-venv")

    if uv := shutil.which("uv"):
        command = [uv, "run"]
        if project_root is not None:
            command.extend(["--project", str(project_root)])
        else:
            command.append("--no-project")
        command.extend(["--python", "3.12", "python"])
        commands.append((command, "uv-python-3.12"))
    return commands


def openusd_runtime_commands() -> list[tuple[list[str], str]]:
    helper = Path(__file__).resolve()
    return [([*prefix, str(helper)], label) for prefix, label in openusd_python_commands()]


def prepare_upload_asset_external(
    asset_path: Path,
    output_directory: Path,
    *,
    allow_missing_textures: bool,
) -> tuple[Path, dict[str, Any]]:
    errors: list[str] = []
    for prefix, runtime_label in openusd_runtime_commands():
        command = [*prefix, str(asset_path), str(output_directory)]
        if allow_missing_textures:
            command.append("--allow-missing-textures")
        try:
            completed = subprocess.run(
                command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=1800,
                check=False,
            )
        except Exception as exc:
            errors.append(f"{runtime_label}: failed to launch: {exc}")
            continue
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            errors.append(f"{runtime_label}: {detail[:500]}")
            continue

        output_lines = completed.stdout.strip().splitlines()
        if not output_lines:
            errors.append(f"{runtime_label}: returned no result")
            continue
        try:
            payload = json.loads(output_lines[-1])
        except json.JSONDecodeError as exc:
            errors.append(f"{runtime_label}: returned invalid JSON: {exc}")
            continue
        if not isinstance(payload, dict):
            errors.append(f"{runtime_label}: returned a non-object payload")
            continue
        if not payload.get("ok"):
            error = str(payload.get("error") or "OpenUSD dependency preparation failed")
            if payload.get("error_kind") == "preparation_failed":
                info = payload.get("info")
                if not isinstance(info, dict):
                    info = _base_upload_info(asset_path)
                info["inspection_runtime"] = runtime_label
                info["inspection_executable"] = prefix[0]
                raise _preparation_error(info, error)
            errors.append(f"{runtime_label}: {error}")
            continue

        path_value = payload.get("path")
        info = payload.get("info")
        if not isinstance(path_value, str) or not isinstance(info, dict):
            errors.append(f"{runtime_label}: returned an invalid result")
            continue
        prepared_path = Path(path_value)
        if not prepared_path.is_file():
            errors.append(f"{runtime_label}: prepared upload does not exist: {prepared_path}")
            continue
        info["inspection_runtime"] = runtime_label
        info["inspection_executable"] = prefix[0]
        return prepared_path, info

    detail = "; ".join(errors) if errors else "no alternate OpenUSD runtime found"
    raise AlternateOpenUSDRuntimeUnavailable(detail)


def prepare_upload_asset_portable(
    asset_path: Path,
    output_directory: Path,
    *,
    allow_missing_textures: bool = False,
) -> tuple[Path, dict[str, Any]]:
    try:
        return prepare_upload_asset(
            asset_path,
            output_directory,
            allow_missing_textures=allow_missing_textures,
        )
    except (OpenUSDUnavailableError, OpenUSDInspectionError) as exc:
        try:
            return prepare_upload_asset_external(
                asset_path,
                output_directory,
                allow_missing_textures=allow_missing_textures,
            )
        except AlternateOpenUSDRuntimeUnavailable as external_exc:
            message = (
                "OpenUSD dependency inspection is required before Content Agents upload, but no working runtime "
                f"was available: {exc}. {external_exc}. Provide an already self-contained .usdz input or set "
                "CONTENT_AGENTS_OPENUSD_PYTHON to a Python 3.12 interpreter with pxr available."
            )
            info = _base_upload_info(asset_path)
            info["inspection_runtime"] = "unavailable"
            info["runtime_attempts"] = str(external_exc)
            raise _preparation_error(info, message) from external_exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("asset_path", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--allow-missing-textures", action="store_true")
    args = parser.parse_args()

    try:
        path, info = prepare_upload_asset(
            args.asset_path,
            args.output_directory,
            allow_missing_textures=args.allow_missing_textures,
        )
    except OpenUSDUnavailableError as exc:
        payload = {"ok": False, "error_kind": "runtime_unavailable", "error": str(exc)}
    except OpenUSDInspectionError as exc:
        payload = {"ok": False, "error_kind": "inspection_failed", "error": str(exc)}
    except UploadPreparationError as exc:
        payload = {
            "ok": False,
            "error_kind": "preparation_failed",
            "error": str(exc),
            "info": exc.upload_info,
        }
    except Exception as exc:
        payload = {"ok": False, "error_kind": "preparation_failed", "error": str(exc)}
    else:
        payload = {"ok": True, "path": str(path), "info": info}

    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
