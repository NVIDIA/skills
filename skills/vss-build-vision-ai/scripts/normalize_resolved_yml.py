#!/usr/bin/env -S uv run --quiet --script
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.9"
# dependencies = ["pyyaml"]
# ///
"""Make a profile-filtered Compose model directly deployable.

Profile filtering can omit an inactive service while retaining a
``depends_on`` entry that explicitly marks that service ``required: false``.
Compose rejects the resulting model. This script removes only those optional
references and removes service profile gates after filtering. A missing
required dependency remains an error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def normalize(path: Path) -> int:
    try:
        with path.open() as resolved_file:
            document = yaml.safe_load(resolved_file) or {}
    except FileNotFoundError:
        print(f"ERROR: {path} not found", file=sys.stderr)
        return 1
    except yaml.YAMLError as error:
        print(f"ERROR: failed to parse {path}: {error}", file=sys.stderr)
        return 1

    services = document.get("services") or {}
    defined = set(services)
    removed: list[tuple[str, str]] = []
    profile_gates_removed = 0
    errors: list[tuple[str, str]] = []

    for service_name, service in services.items():
        if not isinstance(service, dict):
            continue

        if service.pop("profiles", None) is not None:
            profile_gates_removed += 1

        dependencies = service.get("depends_on")
        if not dependencies:
            continue

        if isinstance(dependencies, dict):
            kept = {}
            for dependency_name, dependency in dependencies.items():
                if dependency_name in defined:
                    kept[dependency_name] = dependency
                elif (
                    isinstance(dependency, dict)
                    and dependency.get("required") is False
                ):
                    removed.append((service_name, dependency_name))
                else:
                    kept[dependency_name] = dependency
                    errors.append((service_name, dependency_name))

            if kept:
                service["depends_on"] = kept
            else:
                service.pop("depends_on", None)
        elif isinstance(dependencies, list):
            for dependency_name in dependencies:
                if dependency_name not in defined:
                    errors.append((service_name, dependency_name))

    if errors:
        print(
            f"ERROR: {path} has {len(errors)} dangling required "
            f"dependenc{'y' if len(errors) == 1 else 'ies'}:",
            file=sys.stderr,
        )
        for service_name, dependency_name in errors:
            print(
                f"  - {service_name} -> {dependency_name}",
                file=sys.stderr,
            )
        return 2

    if removed or profile_gates_removed:
        with path.open("w") as resolved_file:
            yaml.safe_dump(document, resolved_file, sort_keys=False)

    print(
        f"Normalized {path}: removed {len(removed)} dangling optional "
        f"dependenc{'y' if len(removed) == 1 else 'ies'} and "
        f"{profile_gates_removed} service profile "
        f"gate{'s' if profile_gates_removed != 1 else ''}"
    )
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Normalize a profile-filtered Compose model for deployment."
    )
    parser.add_argument(
        "resolved_yml",
        nargs="?",
        type=Path,
        default=Path("resolved.yml"),
        help="resolved Compose YAML to normalize (default: resolved.yml)",
    )
    args = parser.parse_args(argv)
    raise SystemExit(normalize(args.resolved_yml))


if __name__ == "__main__":
    main()
