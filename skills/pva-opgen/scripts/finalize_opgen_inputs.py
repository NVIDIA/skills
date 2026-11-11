#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create or verify the structured Spec-to-OpGen handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any


class DuplicateKeyError(ValueError):
    """Report a duplicate key in a JSON object."""


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def reject_json_constant(value: str) -> None:
    raise ValueError(f"nonstandard JSON constant: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        type=Path,
        metavar="OPGEN_INPUTS",
        help="verify an existing handoff without writing",
    )
    parser.add_argument("--operator-spec", type=Path)
    parser.add_argument("--test-matrix", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--doctor-report", type=Path)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_json_constant,
        )
    except DuplicateKeyError as error:
        raise ValueError(f"{label} contains {error}: {path}") from error
    except (OSError, ValueError) as error:
        raise ValueError(f"{label} is not valid JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain one JSON object: {path}")
    return value


def existing_file(path: Path | None, label: str) -> Path:
    if path is None:
        raise ValueError(f"{label} is required")
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path: {path}")
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"{label} is missing or not a file: {resolved}")
    return resolved


def existing_directory(path: Path | None, label: str) -> Path:
    if path is None:
        raise ValueError(f"{label} is required")
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path: {path}")
    resolved = path.resolve()
    if not resolved.is_dir():
        raise ValueError(f"{label} is missing or not a directory: {resolved}")
    return resolved


def output_path(path: Path | None) -> Path:
    if path is None:
        raise ValueError("output is required")
    if not path.is_absolute():
        raise ValueError(f"output must be an absolute path: {path}")
    parent = path.parent.resolve()
    if not parent.is_dir():
        raise ValueError(f"output directory does not exist: {parent}")
    return parent / path.name


def reject_input_output_collision(destination: Path, inputs: list[Path]) -> None:
    resolved = destination.resolve()
    for path in inputs:
        if resolved == path or (destination.exists() and destination.samefile(path)):
            raise ValueError(
                f"output must not replace an input or evidence file: {destination} aliases {path}"
            )


def path_record(path: Path) -> dict[str, Any]:
    return {
        "absolute_path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def normalized_relative(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} is unsafe: {value}")
    return path.as_posix()


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def required_object(parent: dict[str, Any], key: str, label: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"{label}.{key} must be an object")
    return value


def required_list(parent: dict[str, Any], key: str, label: str) -> list[Any]:
    value = parent.get(key)
    if not isinstance(value, list):
        raise TypeError(f"{label}.{key} must be a list")
    return value


def required_string(parent: dict[str, Any], key: str, label: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value:
        raise TypeError(f"{label}.{key} must be a nonempty string")
    return value


def unique_ids(items: list[Any], label: str) -> set[str]:
    identifiers: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise TypeError(f"{label}[{index}] must be an object")
        identifier = required_string(item, "id", f"{label}[{index}]")
        if identifier in identifiers:
            raise ValueError(f"{label} contains duplicate ID {identifier!r}")
        identifiers.add(identifier)
    return identifiers


def validate_spec(spec: dict[str, Any]) -> tuple[str, str, set[str], int]:
    if spec.get("schema_version") != 2:
        raise ValueError("operator spec schema_version must be 2")
    if spec.get("schema_authority") != "skill_workflow":
        raise ValueError("operator spec schema_authority must be skill_workflow")
    if spec.get("status") != "CONFIGURED":
        raise ValueError("operator spec status must be CONFIGURED")

    operator = required_object(spec, "operator", "operator spec")
    operator_name = required_string(operator, "name", "operator spec.operator")
    task_mode = required_string(operator, "task_mode", "operator spec.operator")
    if task_mode not in {"new", "change", "repair"}:
        raise ValueError(
            f"unsupported operator task_mode: {task_mode}; expected new, change, or repair"
        )

    required_object(operator, "public_api", "operator spec.operator")
    platform = required_object(spec, "platform", "operator spec")
    emulator = required_object(platform, "emulator", "operator spec.platform")
    emu_policy = required_string(emulator, "policy", "operator spec.platform.emulator")
    if emu_policy not in {"optional", "required"}:
        raise ValueError("operator spec EMU policy must be optional or required")
    requirements = required_list(spec, "requirements", "operator spec")
    requirement_ids = unique_ids(requirements, "operator spec.requirements")
    required_object(spec, "resources", "operator spec")
    required_list(spec, "parameters", "operator spec")
    required_object(spec, "behavior", "operator spec")
    required_list(spec, "design_constraints", "operator spec")
    required_list(spec, "correctness_contracts", "operator spec")
    required_object(spec, "performance_contract", "operator spec")
    unresolved = required_list(spec, "unresolved_requirements", "operator spec")

    blocking = 0
    for index, item in enumerate(unresolved):
        if not isinstance(item, dict):
            raise TypeError(f"operator spec.unresolved_requirements[{index}] must be an object")
        if not isinstance(item.get("blocking"), bool):
            raise TypeError(
                f"operator spec.unresolved_requirements[{index}].blocking must be boolean"
            )
        blocking += int(item["blocking"])
    if blocking:
        raise ValueError(f"operator spec has {blocking} blocking unresolved requirements")
    return operator_name, emu_policy, requirement_ids, len(unresolved)


def validate_matrix(
    matrix: dict[str, Any],
    operator_name: str,
    emu_policy: str,
    requirement_ids: set[str],
) -> tuple[int, int]:
    if matrix.get("schema_version") != 1:
        raise ValueError("test matrix schema_version must be 1")
    if matrix.get("schema_authority") != "skill_workflow":
        raise ValueError("test matrix schema_authority must be skill_workflow")
    if matrix.get("operator") != operator_name:
        raise ValueError("test matrix operator does not match the operator spec")
    if matrix.get("specification") != "operator-spec.json":
        raise ValueError("test matrix specification must be operator-spec.json")

    execution_policy = required_object(matrix, "execution_policy", "test matrix")
    if execution_policy.get("emulator") != emu_policy:
        raise ValueError("test matrix EMU policy does not match the operator spec")
    if execution_policy.get("hardware") != "required":
        raise ValueError("test matrix hardware policy must be required")

    correctness = required_list(matrix, "correctness_cases", "test matrix")
    if not correctness:
        raise ValueError("test matrix requires at least one correctness case")
    performance = required_list(matrix, "performance_cases", "test matrix")
    case_ids = unique_ids(correctness + performance, "test matrix cases")
    if len(case_ids) != len(correctness) + len(performance):
        raise ValueError("test matrix case IDs must be unique across both matrices")

    for index, case in enumerate(correctness + performance):
        links = required_list(case, "requirement_ids", f"test matrix case[{index}]")
        if not links or not all(isinstance(item, str) for item in links):
            raise ValueError(f"test matrix case[{index}] needs requirement_ids")
        unknown = sorted(set(links) - requirement_ids)
        if unknown:
            raise ValueError(
                f"test matrix case[{index}] has unknown requirement IDs: {', '.join(unknown)}"
            )

    coverage = required_object(matrix, "coverage", "test matrix")
    covered = required_list(coverage, "requirement_ids", "test matrix.coverage")
    if not all(isinstance(item, str) for item in covered):
        raise TypeError("test matrix.coverage.requirement_ids must be a string list")
    if set(covered) != requirement_ids:
        raise ValueError("test matrix coverage does not match the operator requirements")
    uncovered = required_list(coverage, "uncovered_requirement_ids", "test matrix.coverage")
    if uncovered:
        raise ValueError("test matrix has uncovered requirement IDs")
    unresolved = required_list(coverage, "unresolved_cases", "test matrix.coverage")
    if unresolved:
        raise ValueError("test matrix has unresolved cases")
    linked = {
        requirement_id
        for case in correctness + performance
        for requirement_id in case["requirement_ids"]
    }
    if linked != requirement_ids:
        raise ValueError("test cases do not cover every operator requirement")
    return len(correctness), len(performance)


def validate_evidence(evidence: dict[str, Any]) -> None:
    if evidence.get("schema_version") != 2:
        raise ValueError("evidence schema_version must be 2")
    if evidence.get("schema_authority") != "skill_workflow":
        raise ValueError("evidence schema_authority must be skill_workflow")
    required_list(evidence, "additional_evidence_used", "evidence")
    required_list(evidence, "public_sources", "evidence")
    required_list(evidence, "user_sources", "evidence")
    required_list(evidence, "assumptions", "evidence")
    required_list(evidence, "conflicts", "evidence")


def evidence_file_record(
    raw: Any,
    workspace: Path,
    label: str,
    known_requirement_ids: set[str],
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise TypeError(f"{label} must be an object")
    relative = normalized_relative(raw.get("relative_path"), f"{label}.relative_path")
    path = (workspace / relative).resolve()
    if not is_relative_to(path, workspace) or not path.is_file():
        raise ValueError(f"{label} is missing from the workspace: {relative}")
    category = required_string(raw, "category", label)
    digest = sha256_file(path)
    requirement_ids = raw.get("requirement_ids", [])
    if not isinstance(requirement_ids, list) or not all(
        isinstance(item, str) for item in requirement_ids
    ):
        raise TypeError(f"{label}.requirement_ids must be a string list")
    unknown = sorted(set(requirement_ids) - known_requirement_ids)
    if unknown:
        raise ValueError(f"{label} has unknown requirement IDs: {', '.join(unknown)}")
    used_for = required_string(raw, "used_for", label)
    return {
        "category": category,
        "relative_path": relative,
        "absolute_path": str(path),
        "sha256": digest,
        "size_bytes": path.stat().st_size,
        "requirement_ids": requirement_ids,
        "used_for": used_for,
    }


def additional_records(
    workspace: Path, evidence: dict[str, Any], requirement_ids: set[str]
) -> list[dict[str, Any]]:
    items = required_list(evidence, "additional_evidence_used", "evidence")
    records: list[dict[str, Any]] = []
    paths: set[str] = set()
    for index, item in enumerate(items):
        record = evidence_file_record(
            item,
            workspace,
            f"evidence.additional_evidence_used[{index}]",
            requirement_ids,
        )
        relative = str(record["relative_path"])
        if relative in paths:
            raise ValueError(f"evidence repeats additional evidence: {relative}")
        paths.add(relative)
        records.append(record)
    return sorted(records, key=lambda item: str(item["relative_path"]))


def build_payload(
    spec_path: Path,
    matrix_path: Path,
    evidence_path: Path,
    doctor_path: Path,
    workspace: Path,
) -> dict[str, Any]:
    spec = strict_json_object(spec_path, "operator spec")
    matrix = strict_json_object(matrix_path, "test matrix")
    evidence = strict_json_object(evidence_path, "evidence manifest")
    (
        operator_name,
        emu_policy,
        requirement_ids,
        unresolved_count,
    ) = validate_spec(spec)
    correctness_count, performance_count = validate_matrix(
        matrix, operator_name, emu_policy, requirement_ids
    )
    validate_evidence(evidence)

    operator = required_object(spec, "operator", "operator spec")
    platform = required_object(spec, "platform", "operator spec")
    payload: dict[str, Any] = {
        "schema_version": 3,
        "schema_authority": "skill_workflow",
        "stage": "spec",
        "result": "CONFIGURED",
        "operator": {
            "name": operator_name,
            "task_mode": operator["task_mode"],
        },
        "target": platform,
        "doctor_report": path_record(doctor_path),
        "workspace": {"absolute_path": str(workspace)},
        "additional_evidence_used": additional_records(workspace, evidence, requirement_ids),
        "specification": path_record(spec_path),
        "test_matrix": path_record(matrix_path),
        "evidence_manifest": path_record(evidence_path),
        "counts": {
            "requirements": len(requirement_ids),
            "correctness_cases": correctness_count,
            "performance_cases": performance_count,
            "blocking_questions": 0,
            "non_blocking_unresolved": unresolved_count,
            "additional_evidence_files": len(evidence["additional_evidence_used"]),
        },
    }
    return payload


def write_atomic(path: Path, payload: dict[str, Any]) -> None:
    content = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def record_path(payload: dict[str, Any], key: str) -> Path:
    record = required_object(payload, key, "OpGen input manifest")
    value = required_string(record, "absolute_path", f"OpGen input manifest.{key}")
    return existing_file(Path(value), f"OpGen input manifest.{key}.absolute_path")


def check_handoff(path: Path) -> dict[str, Any]:
    current = strict_json_object(path, "OpGen input manifest")
    if current.get("schema_version") != 3:
        raise ValueError("OpGen input manifest schema_version must be 3")
    workspace_record = required_object(current, "workspace", "OpGen input manifest")
    workspace = existing_directory(
        Path(required_string(workspace_record, "absolute_path", "workspace")),
        "workspace.absolute_path",
    )
    expected = build_payload(
        record_path(current, "specification"),
        record_path(current, "test_matrix"),
        record_path(current, "evidence_manifest"),
        record_path(current, "doctor_report"),
        workspace,
    )
    if current != expected:
        changed = sorted(
            key for key in set(current) | set(expected) if current.get(key) != expected.get(key)
        )
        raise ValueError(
            "OpGen input manifest is stale or was changed; differing fields: " + ", ".join(changed)
        )
    return current


def main() -> int:
    args = parse_args()
    if args.check is not None:
        supplied_creation_arg = any(
            value is not None
            for value in (
                args.operator_spec,
                args.test_matrix,
                args.evidence,
                args.doctor_report,
                args.workspace,
                args.output,
            )
        )
        if supplied_creation_arg:
            raise ValueError("--check cannot be combined with creation arguments")
        handoff = existing_file(args.check, "OpGen input manifest")
        payload = check_handoff(handoff)
        mode = "check"
        output = handoff
    else:
        spec_path = existing_file(args.operator_spec, "operator spec")
        matrix_path = existing_file(args.test_matrix, "test matrix")
        evidence_path = existing_file(args.evidence, "evidence manifest")
        doctor_path = existing_file(args.doctor_report, "Doctor report")
        workspace = existing_directory(args.workspace, "workspace")
        destination = output_path(args.output)
        payload = build_payload(
            spec_path,
            matrix_path,
            evidence_path,
            doctor_path,
            workspace,
        )
        reject_input_output_collision(
            destination,
            [spec_path, matrix_path, evidence_path, doctor_path]
            + [Path(record["absolute_path"]) for record in payload["additional_evidence_used"]],
        )
        write_atomic(destination, payload)
        payload = check_handoff(destination)
        mode = "create"
        output = destination

    print(
        json.dumps(
            {
                "result": "VALID",
                "mode": mode,
                "output": str(output),
                "requirements": payload["counts"]["requirements"],
                "correctness_cases": payload["counts"]["correctness_cases"],
                "performance_cases": payload["counts"]["performance_cases"],
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (TypeError, ValueError) as error:
        raise SystemExit(f"invalid Spec-to-OpGen handoff: {error}") from error
