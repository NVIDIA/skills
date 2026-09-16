#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the DEFT CR ITS mining workflow scripts."""

from __future__ import annotations

import datetime
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    try:
        import tomli as tomllib
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on host env
        raise SystemExit("ERROR: tomli is required on Python versions without tomllib.") from exc

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only on missing dependency
    raise SystemExit("ERROR: PyYAML is required. Install pyyaml before running this script.") from exc


MODALITY_CHOICES = ("text", "video", "both")


def absolute_path(path: str | Path) -> Path:
    """Expand a user path to an absolute path without resolving symlinks."""
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def path_in_workspace(path: Path, workspace: Path, label: str) -> None:
    """Require a path to live inside the configured DEFT workspace."""
    try:
        path.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(f"{label} must be inside the DEFT workspace: {path}") from exc


def load_json_array(path: Path) -> list[dict[str, Any]]:
    """Read a JSON file and require a list of JSON objects."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"{path}: expected a JSON array")
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"{path}: item {index} is not a JSON object")
    return payload


def write_json_array(path: Path, records: list[dict[str, Any]]) -> None:
    """Write a list of JSON objects with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write a JSON value atomically so readers never observe partial output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def atomic_write_parquet(frame: Any, path: Path) -> None:
    """Write a pandas-compatible dataframe to Parquet atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".parquet", dir=path.parent)
    os.close(fd)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file and require each non-empty row to be an object."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(record)
    return records


def load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML and require a mapping at the document root."""
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a YAML object")
    return payload


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    """Write a YAML mapping in stable block style."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def load_toml(path: Path) -> dict[str, Any]:
    """Load TOML and require a mapping at the document root."""
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a TOML object")
    return payload


def toml_value(value: Any) -> str:
    """Serialize a TOML scalar or simple list."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if value is None:
        return '""'
    if isinstance(value, list):
        return "[ " + ", ".join(toml_value(item) for item in value) + " ]"
    raise TypeError(f"unsupported TOML value: {value!r}")


def dump_toml(config: dict[str, Any]) -> str:
    """Serialize the nested TOML table shape used by Cosmos Reason configs."""
    lines: list[str] = []

    def emit_table(prefix: str, table: dict[str, Any]) -> None:
        scalars = [(key, value) for key, value in table.items() if not isinstance(value, dict)]
        children = [(key, value) for key, value in table.items() if isinstance(value, dict)]
        if prefix:
            if lines:
                lines.append("")
            lines.append(f"[{prefix}]")
        for key, value in scalars:
            lines.append(f"{key} = {toml_value(value)}")
        for key, value in children:
            child_prefix = f"{prefix}.{key}" if prefix else key
            emit_table(child_prefix, value)

    emit_table("", config)
    return "\n".join(lines) + "\n"


def require_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    """Return a required child mapping from a workflow config."""
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"missing required object: {key}")
    return value


def require_string(section: dict[str, Any], dotted_key: str) -> str:
    """Return a required non-empty string from a config section."""
    key = dotted_key.rsplit(".", 1)[-1]
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing required string: {dotted_key}")
    return value


def optional_bool(section: dict[str, Any], dotted_key: str, default: bool) -> bool:
    """Return an optional boolean from a config section."""
    key = dotted_key.rsplit(".", 1)[-1]
    value = section.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{dotted_key} must be true or false")
    return value


def existing_absolute_path(value: str, workspace: Path, label: str, expected: str) -> Path:
    """Validate an absolute workflow path under the workspace."""
    configured_path = Path(os.path.expanduser(value))
    if not configured_path.is_absolute():
        raise ValueError(f"{label} must be absolute: {value}")
    path = Path(os.path.normpath(str(configured_path)))
    path_in_workspace(path, workspace, label)
    if expected == "file" and not path.is_file():
        raise FileNotFoundError(f"{label} must be an existing file: {path}")
    if expected == "dir" and not path.is_dir():
        raise NotADirectoryError(f"{label} must be an existing directory: {path}")
    if expected == "path" and not path.exists():
        raise FileNotFoundError(f"{label} must exist: {path}")
    return path


def modality_list(embedding_modality: str) -> list[str]:
    """Expand `both` into the ordered modalities used by the workflow."""
    if embedding_modality == "both":
        return ["text", "video"]
    if embedding_modality not in ("text", "video"):
        raise ValueError("embedding modality must be text, video, or both")
    return [embedding_modality]


def dataset_modalities(dataset: str, embedding_modality: str) -> list[str]:
    """Return the complete modality set required for one workflow dataset."""
    if dataset == "train":
        return ["text", "video"]
    if dataset == "kpi":
        return modality_list(embedding_modality)
    raise ValueError(f"unsupported dataset: {dataset}")


def validate_embedding_parquet(
    path: Path,
    required_modalities: list[str],
    label: str,
) -> list[str]:
    """Validate one combined mining-ready embedding Parquet and its modalities."""
    import pandas as pd
    import pyarrow.parquet as pq

    required_columns = {"filepath", "embedding", "modality"}
    missing = required_columns - set(pq.read_schema(path).names)
    if missing:
        raise ValueError(f"{label} is missing required columns: {sorted(missing)}")
    modality_values = pd.read_parquet(path, columns=["modality"])["modality"]
    if modality_values.isna().any():
        raise ValueError(f"{label} contains null modality values")
    actual = set(modality_values.astype(str))
    expected = set(required_modalities)
    if actual != expected:
        raise ValueError(
            f"{label} must contain exactly modalities {sorted(expected)}, found {sorted(actual)}"
        )
    return sorted(actual)


def optional_embedding_parquets(
    mining: dict[str, Any],
    workspace: Path,
    embedding_modality: str,
) -> dict[str, Path | None]:
    """Validate optional combined KPI/train embedding Parquets from workflow.yaml."""
    stale_fields = [key for key in ("text_embeddings", "video_embeddings") if key in mining]
    if stale_fields:
        raise ValueError(
            "removed per-modality mining fields are present: "
            f"{', '.join(stale_fields)}; use mining.embedding_parquets.kpi and .train"
        )
    section = mining.get("embedding_parquets", {})
    if section is None:
        section = {}
    if not isinstance(section, dict):
        raise ValueError("mining.embedding_parquets must be an object when provided")

    resolved: dict[str, Path | None] = {}
    for dataset in ("kpi", "train"):
        value = section.get(dataset)
        if value in (None, ""):
            resolved[dataset] = None
            continue
        if not isinstance(value, str):
            raise ValueError(f"mining.embedding_parquets.{dataset} must be a string or null")
        path = existing_absolute_path(
            value,
            workspace,
            f"mining.embedding_parquets.{dataset}",
            "file",
        )
        if path.suffix != ".parquet":
            raise ValueError(
                f"mining.embedding_parquets.{dataset} must point to a .parquet file: {path}"
            )
        validate_embedding_parquet(
            path,
            dataset_modalities(dataset, embedding_modality),
            f"mining.embedding_parquets.{dataset}",
        )
        resolved[dataset] = path
    return resolved


def workflow_run_dir(config: dict[str, Any], workspace: Path) -> Path:
    """Return the run directory derived from workflow.yaml."""
    run = require_mapping(config, "run")
    run_name = run.get("name")
    if run_name is not None and (not isinstance(run_name, str) or not run_name.strip()):
        raise ValueError("run.name must be a non-empty string or null")
    if run_name:
        return workspace / "results" / run_name
    return workspace / "results" / datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")


def copy_workflow_yaml_to_run_dir(workflow_yaml: Path, run_dir: Path) -> Path:
    """Copy workflow.yaml into the run directory as the reproducibility snapshot."""
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "workflow.yaml"
    shutil.copy2(workflow_yaml, output_path)
    return output_path


def clean_question(text: str) -> str:
    """Normalize a LLaVA question before matching text embeddings."""
    text = text.replace("<video>", " ")
    text = re.sub(r"(?i)\s*answer with yes or no\.?\s*$", "", text)
    return " ".join(text.split())


def normalize_media_path(path: str) -> str:
    """Normalize a media path string for joins while preserving absolute paths."""
    return os.path.normpath(os.path.expanduser(path))


def find_results_json(output_dir: Path) -> Path:
    """Find exactly one results.json under a completed evaluation output directory."""
    matches = sorted(output_dir.rglob("results.json"))
    if not matches:
        raise FileNotFoundError(f"no results.json found under {output_dir}")
    if len(matches) > 1:
        joined = ", ".join(str(path) for path in matches[:5])
        raise ValueError(f"multiple results.json files found under {output_dir}: {joined}")
    return matches[0]
