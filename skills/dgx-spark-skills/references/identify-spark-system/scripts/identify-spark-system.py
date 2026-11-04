#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Identify a DGX Spark system and report read-only hardware capacity data."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any


DEFAULT_IDENTITY_FILE = "/opt/nvidia/dgx-oobe-customizations/customize.json"
DEFAULT_PROC_ROOT = "/proc"
INVENTORY_TIMEOUT_SECONDS = 10


class ArgumentError(ValueError):
    """Report an invalid command-line argument and whether usage is needed."""

    def __init__(self, message: str, *, show_usage: bool) -> None:
        super().__init__(message)
        self.show_usage = show_usage


def usage() -> str:
    """Return command usage text."""
    return """Usage: identify-spark-system.py [--identity-file PATH] [--systems-file PATH]

Read the DGX Spark customization identity and Linux capacity interfaces, then
emit JSON. The helper performs no writes and reports partial data with warnings.

Options:
  --identity-file PATH  Override the default customization file.
  --systems-file PATH   Override the bundled YAML catalog.
  -h, --help            Show this help."""


def parse_arguments(argv: list[str]) -> tuple[str, str] | None:
    """Parse the small stable command-line interface without dependencies."""
    script_dir = Path(__file__).resolve().parent
    identity_file = DEFAULT_IDENTITY_FILE
    systems_file = str(script_dir / "../references/spark-systems.yaml")

    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument in {"-h", "--help"}:
            return None
        if argument in {"--identity-file", "--systems-file"}:
            if index + 1 >= len(argv):
                raise ArgumentError(f"{argument} requires a path", show_usage=False)
            if argument == "--identity-file":
                identity_file = argv[index + 1]
            else:
                systems_file = argv[index + 1]
            index += 2
            continue
        raise ArgumentError(f"unknown argument: {argument}", show_usage=True)

    return identity_file, systems_file


def scalar(value: str) -> Any:
    """Parse the scalar subset used by the bundled YAML catalog."""
    value = value.strip()
    if value in {"null", "~"}:
        return None
    if value == "[]":
        return []
    if len(value) >= 2 and value[0] == value[-1] == '"':
        return json.loads(value)
    if len(value) >= 2 and value[0] == value[-1] == "'":
        return value[1:-1].replace("''", "'")
    return value


def load_catalog(filename: str) -> tuple[Any, list[dict[str, Any]]]:
    """Load and validate the dependency-free catalog subset."""
    systems: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    in_match = False
    schema_version: Any = None

    with open(filename, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            indent = len(line) - len(line.lstrip(" "))

            if indent == 0 and stripped.startswith("schema_version:"):
                schema_version = scalar(stripped.split(":", 1)[1])
                continue

            if indent == 2 and stripped.startswith("- id:"):
                current = {"id": scalar(stripped.split(":", 1)[1])}
                systems.append(current)
                in_match = False
                continue

            if current is None:
                continue

            if indent == 4 and ":" in stripped:
                key, value = stripped.split(":", 1)
                in_match = key == "match" and not value.strip()
                if key == "match":
                    current["match"] = {} if in_match else scalar(value)
                elif key in {"classification", "canonical_name", "vendor", "catalog_status"}:
                    current[key] = scalar(value)
                continue

            if indent == 6 and in_match and ":" in stripped:
                key, value = stripped.split(":", 1)
                current["match"][key] = scalar(value)

    if str(schema_version) != "1":
        raise ValueError(f"unsupported catalog schema_version: {schema_version!r}")
    if not systems:
        raise ValueError("catalog contains no systems")

    seen_ids: set[str] = set()
    allowed_statuses = {"nvidia-product", "partner-oem", "unresolved"}
    for system in systems:
        missing = [
            key
            for key in ("id", "canonical_name", "vendor", "catalog_status")
            if not system.get(key)
        ]
        if missing:
            raise ValueError(f"catalog entry is missing {', '.join(missing)}")
        if system["id"] in seen_ids:
            raise ValueError(f"duplicate system id: {system['id']}")
        seen_ids.add(system["id"])
        if system["catalog_status"] not in allowed_statuses:
            raise ValueError(f"invalid catalog_status for {system['id']}")
        if system["id"] != "nvidia-dgx-spark-founders-edition":
            if "match" in system or "classification" in system:
                raise ValueError("reference-only entries cannot define automatic identification")
            continue
        if (system["catalog_status"] != "nvidia-product"
                or system.get("classification") != "nvidia-dgx-spark-founders-edition"):
            raise ValueError("invalid NVIDIA identification entry")
        matcher = system.get("match")
        if not isinstance(matcher, dict):
            raise ValueError("NVIDIA identification requires a matcher")
        if set(matcher) != {"company", "deviceType"}:
            raise ValueError(f"matcher for {system['id']} must define company and deviceType")
        if not all(isinstance(value, str) and value for value in matcher.values()):
            raise ValueError(f"matcher for {system['id']} must use non-empty string literals")

    return schema_version, systems


def run_command(
    command: list[str], warnings: list[str]
) -> subprocess.CompletedProcess[str] | None:
    """Run a read-only inventory command with a per-command timeout."""
    # Inventory tools need executable lookup and deterministic output, not the
    # caller's credentials, shell startup settings, or loader overrides.
    environment = {"PATH": os.environ.get("PATH", os.defpath), "LC_ALL": "C"}
    executable = shutil.which(command[0], path=environment["PATH"])
    if executable is None:
        warnings.append(f"{command[0]} is unavailable.")
        return None
    try:
        completed = subprocess.run(
            [executable, *command[1:]],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=environment,
            timeout=INVENTORY_TIMEOUT_SECONDS,
        )
        if completed.returncode != 0:
            warnings.append(f"{command[0]} exited with code {completed.returncode}.")
        return completed
    except subprocess.TimeoutExpired:
        warnings.append(
            f"{command[0]} timed out after {INVENTORY_TIMEOUT_SECONDS} seconds."
        )
    except OSError as error:
        warnings.append(f"{command[0]} could not run: {error}")
    return None


def command_output(command: list[str], warnings: list[str]) -> str:
    """Return stripped stdout only for a successful command."""
    completed = run_command(command, warnings)
    if completed is None or completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def decimal_integer(value: str) -> int | None:
    """Return a base-10 integer, or None for invalid output."""
    if not value or not value.isascii() or not value.isdigit():
        return None
    return int(value)


def read_logical_cores(warnings: list[str]) -> int | None:
    """Read configured logical-core capacity."""
    cores = decimal_integer(command_output(["getconf", "_NPROCESSORS_CONF"], warnings))
    if cores is not None:
        return cores
    return decimal_integer(command_output(["nproc", "--all"], warnings))


def read_memory_bytes(proc_root: str) -> int | None:
    """Read total system-memory capacity from procfs."""
    meminfo = Path(proc_root) / "meminfo"
    try:
        content = meminfo.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None
    match = re.search(r"^MemTotal:\s+(\d+)\s+kB\s*$", content, re.MULTILINE)
    return int(match.group(1)) * 1024 if match else None


def collect_json_command(
    command: list[str], failed_warning: str, warnings: list[str]
) -> str:
    """Collect JSON text from an optional read-only inventory command."""
    completed = run_command(command, warnings)
    if completed is None:
        return ""
    if completed.returncode != 0:
        warnings.append(failed_warning)
        return ""
    return completed.stdout.strip()


def parse_identity(filename: str, warnings: list[str]) -> dict[str, Any]:
    """Read the identity JSON without inferring missing fields."""
    result: dict[str, Any] = {
        "file": filename,
        "status": "unknown",
        "present_fields": [],
        "company": None,
        "deviceType": None,
        "infoURL": None,
    }
    if not os.path.exists(filename):
        result["status"] = "missing"
        warnings.append(f"Identity file is missing: {filename}")
        return result
    if not os.path.isfile(filename) or not os.access(filename, os.R_OK):
        result["status"] = "unreadable"
        warnings.append(f"Identity file is not a readable regular file: {filename}")
        return result
    try:
        with open(filename, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        result["status"] = "invalid"
        warnings.append(f"Identity file is not valid readable JSON: {error}")
        return result
    if not isinstance(payload, dict):
        result["status"] = "invalid"
        warnings.append("Identity JSON root must be an object.")
        return result
    result["status"] = "ok"
    for field in ("company", "deviceType", "infoURL"):
        if field in payload:
            result["present_fields"].append(field)
            result[field] = payload[field]
    return result


def classify(identity: dict[str, Any], systems: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify only an exact, unique catalog matcher."""
    if identity["status"] != "ok":
        return {
            "kind": "unknown",
            "system_id": None,
            "canonical_name": None,
            "vendor": None,
            "matched_fields": [],
            "reason": f"identity-{identity['status']}",
        }

    matches = []
    for system in systems:
        if system["id"] != "nvidia-dgx-spark-founders-edition":
            continue
        matcher = system.get("match")
        if matcher and all(identity.get(key) == value for key, value in matcher.items()):
            matches.append((system, matcher))

    if len(matches) != 1:
        reason = "ambiguous-source-backed-matchers" if len(matches) > 1 else "no-source-backed-exact-matcher"
        return {
            "kind": "unknown",
            "system_id": None,
            "canonical_name": None,
            "vendor": None,
            "matched_fields": [],
            "reason": reason,
        }

    system, matcher = matches[0]
    return {
        "kind": system["classification"],
        "system_id": system["id"],
        "canonical_name": system["canonical_name"],
        "vendor": system["vendor"],
        "matched_fields": sorted(matcher),
        "reason": "all-source-backed-matcher-fields-exact",
    }


def integer_or_none(value: Any) -> int | None:
    """Coerce an inventory value to an integer when possible."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_block_devices(raw: str, warnings: list[str]) -> list[dict[str, Any]]:
    """Normalize lsblk JSON into the stable capacity schema."""
    if not raw:
        return []
    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("root is not an object")
        devices = payload.get("blockdevices", [])
        if not isinstance(devices, list):
            raise ValueError("blockdevices is not a list")
        return [
            {
                "name": item.get("name"),
                "path": item.get("path"),
                "type": item.get("type"),
                "capacity_bytes": integer_or_none(item.get("size")),
                "model": item.get("model"),
            }
            for item in devices
            if isinstance(item, dict)
        ]
    except (json.JSONDecodeError, ValueError) as error:
        warnings.append(f"lsblk returned invalid JSON: {error}")
        return []


def parse_filesystems(raw: str, warnings: list[str]) -> list[dict[str, Any]]:
    """Flatten findmnt JSON into the stable capacity schema."""
    if not raw:
        return []
    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("root is not an object")
        roots = payload.get("filesystems", [])
        if not isinstance(roots, list):
            raise ValueError("filesystems is not a list")
        flattened: list[dict[str, Any]] = []

        def visit(item: Any) -> None:
            if not isinstance(item, dict):
                return
            flattened.append(
                {
                    "target": item.get("target"),
                    "source": item.get("source"),
                    "filesystem_type": item.get("fstype"),
                    "capacity_bytes": integer_or_none(item.get("size")),
                }
            )
            for child in item.get("children", []) or []:
                visit(child)

        for root in roots:
            visit(root)
        return flattened
    except (json.JSONDecodeError, ValueError) as error:
        warnings.append(f"findmnt returned invalid JSON: {error}")
        return []


def collect_capacity(proc_root: str, warnings: list[str]) -> dict[str, Any]:
    """Collect read-only CPU, memory, block-device, and filesystem capacity."""
    architecture = command_output(["uname", "-m"], warnings)
    if not architecture:
        warnings.append("CPU architecture could not be read with uname.")

    logical_cores = read_logical_cores(warnings)
    if logical_cores is None:
        warnings.append("Logical-core capacity could not be read.")

    memory_bytes = read_memory_bytes(proc_root)
    if memory_bytes is None:
        warnings.append(f"Total system-memory capacity could not be read from {proc_root}/meminfo.")

    block_devices_raw = collect_json_command(
        ["lsblk", "--bytes", "--json", "--nodeps", "--exclude", "7", "--output", "NAME,PATH,TYPE,SIZE,MODEL"],
        "Block-device capacity could not be read with lsblk.",
        warnings,
    )
    filesystems_raw = collect_json_command(
        ["findmnt", "--bytes", "--json", "--real", "--output", "TARGET,SOURCE,FSTYPE,SIZE"],
        "Mounted-filesystem capacity could not be read with findmnt.",
        warnings,
    )

    return {
        "cpu": {
            "architecture": architecture or None,
            "logical_cores": logical_cores,
            "logical_cores_kind": "configured-capacity",
        },
        "system_memory_bytes": memory_bytes,
        "block_devices_raw": block_devices_raw,
        "filesystems_raw": filesystems_raw,
    }


def main(argv: list[str]) -> int:
    """Run the read-only inventory and emit stable JSON."""
    try:
        arguments = parse_arguments(argv)
    except ArgumentError as error:
        print(f"error: {error}", file=sys.stderr)
        if error.show_usage:
            print(usage(), file=sys.stderr)
        return 2
    if arguments is None:
        print(usage())
        return 0

    identity_file, systems_file = arguments
    if not os.access(systems_file, os.R_OK):
        print(f"error: systems catalog is not readable: {systems_file}", file=sys.stderr)
        return 2

    try:
        catalog_schema, systems = load_catalog(systems_file)
    except (OSError, UnicodeError, ValueError) as error:
        print(f"error: invalid systems catalog: {error}", file=sys.stderr)
        return 2

    warnings: list[str] = []
    identity = parse_identity(identity_file, warnings)
    classification = classify(identity, systems)
    recognized_gb10 = classification["kind"] != "unknown"

    proc_root = os.environ.get("IDENTIFY_SPARK_SYSTEM_PROC_ROOT", DEFAULT_PROC_ROOT)
    capacity = collect_capacity(proc_root, warnings)

    result = {
        "schema_version": 1,
        "identity": identity,
        "classification": classification,
        "capacity": {
            "cpu": capacity["cpu"],
            "system_memory": {
                "capacity_bytes": capacity["system_memory_bytes"],
                "capacity_kind": (
                    "unified-system-memory"
                    if recognized_gb10
                    else "system-memory-unverified-for-gb10"
                ),
                "utilization_reported": False,
                "separate_gpu_memory_capacity_reported": False,
            },
            "block_devices": parse_block_devices(capacity["block_devices_raw"], warnings),
            "mounted_filesystems": parse_filesystems(capacity["filesystems_raw"], warnings),
            "values_are_capacities_not_utilization": True,
        },
        "catalog": {
            "file": systems_file,
            "schema_version": integer_or_none(catalog_schema),
            "systems": len(systems),
            "match_semantics": "all-fields-exact-case-sensitive",
        },
        "warnings": warnings,
    }

    json.dump(result, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
