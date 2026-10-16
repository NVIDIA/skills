#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Materialize the four DEFT OD AOI roles from a dataset source manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


CATEGORY = [{"id": 1, "name": "defect"}]
INPUT_TO_ROLE = {"benchmark": "kpi", "test": "test", "mining": "real", "clean": "clean"}


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _resolve(raw: Any, base: Path, label: str) -> Path:
    text = str(raw or "").strip()
    if not text:
        raise ValueError(f"{label} is required")
    path = Path(text).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _entries(manifest: dict[str, Any], name: str) -> list[dict[str, Any]]:
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("inputs must be an object")
    entries = inputs.get(name)
    allow_empty = name == "clean"
    if (not isinstance(entries, list) or (not entries and not allow_empty)
            or not all(isinstance(entry, dict) for entry in entries)):
        qualifier = "an array" if allow_empty else "a non-empty array"
        raise ValueError(f"inputs.{name} must be {qualifier} of objects")
    return entries


def _coco(path: Path) -> dict[str, Any]:
    value = _read_object(path)
    for key in ("images", "annotations", "categories"):
        if not isinstance(value.get(key), list):
            raise ValueError(f"COCO {path} lacks array {key!r}")
    category_ids = {row.get("id") for row in value["categories"] if isinstance(row, dict)}
    if value["annotations"] and not category_ids:
        raise ValueError(f"COCO {path} has annotations but no categories")
    return value


def _source(image: dict[str, Any], images: Path, coco: Path) -> Path:
    raw = image.get("source_path") or image.get("file_name")
    if not str(raw or "").strip():
        raise ValueError(f"COCO image in {coco} needs source_path or file_name")
    candidate = Path(str(raw)).expanduser()
    if candidate.is_absolute():
        options = [candidate]
    else:
        options = [images / candidate, coco.parent / candidate]
        if candidate.parent == Path("."):
            options.append(coco.parent / "images" / candidate)
    resolved = next((path.resolve() for path in options if path.is_file()), options[0].resolve())
    if not resolved.is_file():
        raise FileNotFoundError(f"COCO image is missing: {resolved}")
    return resolved


def _output_name(source: Path) -> str:
    digest = hashlib.sha256(str(source).encode()).hexdigest()[:20]
    return digest + source.suffix.lower()


def _annotation(value: dict[str, Any], image_id: int, annotation_id: int,
                categories: set[Any], source: Path, width: int, height: int) -> dict[str, Any]:
    if value.get("category_id") not in categories:
        raise ValueError(f"annotation in {source} references an unknown category")
    bbox = value.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"annotation in {source} needs a four-value bbox")
    box = [float(item) for item in bbox]
    if min(box[:2]) < 0 or box[2] <= 0 or box[3] <= 0:
        raise ValueError(f"annotation in {source} has an invalid bbox")
    if box[0] + box[2] > width or box[1] + box[3] > height:
        raise ValueError(f"annotation in {source} exceeds normalized COCO image bounds")
    output = dict(value)
    output.update(id=annotation_id, image_id=image_id, category_id=1, bbox=box,
                  area=float(value.get("area", box[2] * box[3])),
                  iscrowd=int(value.get("iscrowd", 0)))
    return output


def prepare(manifest_path: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    manifest = _read_object(manifest_path)
    if manifest.get("schema_version") != 1:
        raise ValueError("dataset source manifest schema_version must be 1")
    base = manifest_path.parent
    documents: dict[str, list[dict[str, Any]]] = {
        role: [] for role in INPUT_TO_ROLE.values()}
    identities: dict[str, str] = {}
    source_reports = []
    for input_name, role in INPUT_TO_ROLE.items():
        for entry in _entries(manifest, input_name):
            coco_paths = entry.get("coco")
            coco_values = coco_paths if isinstance(coco_paths, list) else [coco_paths]
            if not coco_values or any(not str(value or "").strip() for value in coco_values):
                raise ValueError(f"inputs.{input_name} entry needs coco")
            for raw_coco in coco_values:
                coco_path = _resolve(raw_coco, base, f"inputs.{input_name}.coco")
                if not coco_path.is_file():
                    raise FileNotFoundError(coco_path)
                images_root = _resolve(entry.get("images_dir") or coco_path.parent, base,
                                       f"inputs.{input_name}.images_dir")
                if not images_root.is_dir():
                    raise FileNotFoundError(images_root)
                source = _coco(coco_path)
                image_rows = source["images"]
                image_ids = [row.get("id") for row in image_rows]
                if ((not image_rows and role != "clean")
                        or len(image_ids) != len(set(image_ids))):
                    raise ValueError(f"COCO {coco_path} has no images or duplicate image ids")
                by_image: defaultdict[Any, list[dict[str, Any]]] = defaultdict(list)
                known_images = set(image_ids)
                for row in source["annotations"]:
                    if row.get("image_id") not in known_images:
                        raise ValueError(f"annotation in {coco_path} references an unknown image")
                    by_image[row.get("image_id")].append(row)
                if role == "real" and any(not by_image[image_id] for image_id in image_ids):
                    raise ValueError("inputs.mining must contain only boxed images")
                if role == "clean" and source["annotations"]:
                    raise ValueError("inputs.clean COCO must have zero annotations")
                categories = {row.get("id") for row in source["categories"]}
                document = {"images": [], "annotations": [], "categories": CATEGORY}
                for image in image_rows:
                    path = _source(image, images_root, coco_path)
                    identity = str(path)
                    if identity in identities:
                        raise ValueError(
                            f"image overlaps {identities[identity]} and {role}: {identity}"
                        )
                    identities[identity] = role
                    width, height = int(image.get("width", 0)), int(image.get("height", 0))
                    if width <= 0 or height <= 0:
                        raise ValueError(f"image in {coco_path} needs positive width and height")
                    image_id = len(document["images"]) + 1
                    output_image = dict(image)
                    output_image.update(id=image_id, file_name=_output_name(path),
                                        source_path=identity, width=width, height=height)
                    document["images"].append(output_image)
                    for row in by_image[image.get("id")]:
                        document["annotations"].append(_annotation(
                            row, image_id, len(document["annotations"]) + 1,
                            categories, coco_path, width, height
                        ))
                documents[role].append(document)
                source_reports.append({"input": input_name, "role": role,
                                       "coco": str(coco_path),
                                       "images": len(image_rows),
                                       "annotations": len(source["annotations"])})
    role_counts = {name: {
        "images": sum(len(shard["images"]) for shard in shards),
        "annotations": sum(len(shard["annotations"]) for shard in shards),
    } for name, shards in documents.items()}
    warnings = [{
        "code": "empty_retrieval_source_role",
        "role": role,
        "message": f"{role} retrieval source role is empty; that producer starts exhausted",
    } for role in ("clean",) if role_counts[role]["images"] == 0]
    report = {
        "status": "VALID",
        "manifest": str(manifest_path),
        "roles": role_counts,
        "capabilities": {"retrieval": {
            role: {"status": "AVAILABLE" if role_counts[role]["images"] else "UNAVAILABLE",
                   "reason": None if role_counts[role]["images"] else "empty_source_role",
                   "source_image_count": role_counts[role]["images"]}
            for role in ("real", "clean")
        }},
        "warnings": warnings,
        "sources": source_reports,
        "overlaps": {},
    }
    return documents, report


def _merge_with_data_services(inputs: list[Path], work: Path) -> Path:
    work.mkdir(parents=True)
    spec = work / "merge.yaml"
    spec.write_text(json.dumps({
        "data": {"format": "COCO", "annotations": [str(path) for path in inputs],
                 "same_categories": True, "on_duplicate": "error"},
        "results_dir": str(work),
    }, indent=2) + "\n")
    try:
        subprocess.run(["annotations", "merge", "-e", str(spec)], check=True)
    except FileNotFoundError as error:
        raise RuntimeError(
            "materialization requires the TAO Data Services container with "
            "the `annotations merge` command"
        ) from error
    merged = work / "output.json"
    if not merged.is_file():
        raise RuntimeError(f"Data Services merge did not produce {merged}")
    return merged


def _validate_merged(role: str, document: dict[str, Any], expected: dict[str, int]) -> None:
    if document.get("categories") != CATEGORY:
        raise ValueError(f"Data Services changed the canonical {role} categories")
    images, annotations = document.get("images"), document.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError(f"Data Services emitted an invalid {role} COCO document")
    if len(images) != expected["images"] or len(annotations) != expected["annotations"]:
        raise ValueError(f"Data Services changed the expected {role} counts")
    image_ids = [row.get("id") for row in images]
    annotation_ids = [row.get("id") for row in annotations]
    if len(image_ids) != len(set(image_ids)) or len(annotation_ids) != len(set(annotation_ids)):
        raise ValueError(f"Data Services emitted duplicate {role} identifiers")
    known = set(image_ids)
    boxed = set()
    for row in annotations:
        if row.get("image_id") not in known or row.get("category_id") != 1:
            raise ValueError(f"Data Services emitted an invalid {role} annotation reference")
        boxed.add(row["image_id"])
    if role == "real" and boxed != known:
        raise ValueError("Data Services emitted a boxless real image")
    if role == "clean" and annotations:
        raise ValueError("Data Services emitted an annotated clean image")


def materialize(manifest_path: Path, documents: dict[str, list[dict[str, Any]]],
                report: dict[str, Any], output: Path, link_mode: str,
                merge_action: Callable[[list[Path], Path], Path] | None = None
                ) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    output.mkdir(parents=True)
    sources = {}
    shutil.copyfile(manifest_path, output / "dataset_sources.json")
    merge_action = merge_action or _merge_with_data_services
    merge_inputs = output / "merge_inputs"
    merge_inputs.mkdir()
    merge_evidence = {}
    for role, shards in documents.items():
        images = output / f"{role}_images"
        images.mkdir()
        inputs = []
        for index, shard in enumerate(shards, start=1):
            materialized = {
                "images": [dict(row) for row in shard["images"]],
                "annotations": [dict(row) for row in shard["annotations"]],
                "categories": [dict(row) for row in CATEGORY],
            }
            for row in materialized["images"]:
                source = Path(str(row["source_path"]))
                target = images / row["file_name"]
                if link_mode == "symlink":
                    target.symlink_to(source)
                else:
                    shutil.copy2(source, target)
                # The normalized target is canonical in both materialization modes.
                row["original_source_path"] = str(source)
                row["source_path"] = str(target.absolute())
            shard_path = merge_inputs / f"{role}-{index:04d}.json"
            shard_path.write_text(json.dumps(materialized, indent=2) + "\n")
            inputs.append(shard_path)
        coco = output / f"{role}.json"
        if report["roles"][role]["images"]:
            merged = merge_action(inputs, output / "merge_actions" / role)
            shutil.copyfile(merged, coco)
            merge_evidence[role] = {"status": "COMPLETE",
                                    "inputs": [str(path.resolve()) for path in inputs],
                                    "output": str(merged.resolve())}
        else:
            coco.write_text(json.dumps({"images": [], "annotations": [],
                                        "categories": CATEGORY}, indent=2) + "\n")
            merge_evidence[role] = {"status": "SKIPPED", "reason": "empty_source_role",
                                    "inputs": [], "output": str(coco.resolve())}
        _validate_merged(role, _read_object(coco), report["roles"][role])
        sources[role] = {"images": str(images.resolve()), "coco": str(coco.resolve())}
    report = {**report, "merger": {"command": "annotations merge",
                                     "roles": merge_evidence}}
    handoff = {"schema_version": 1, "sources": sources,
               "source_preparation_report": str((output / "source_preparation_report.json").resolve())}
    (output / "sources.json").write_text(json.dumps(handoff, indent=2) + "\n")
    (output / "source_preparation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return handoff


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output-dir", type=Path)
    group.add_argument("--check-only", action="store_true")
    parser.add_argument("--link-mode", choices=("symlink", "copy"), default="symlink")
    args = parser.parse_args()
    try:
        manifest = args.manifest.expanduser().resolve()
        documents, report = prepare(manifest)
        result = (report if args.check_only else materialize(
            manifest, documents, report, args.output_dir.expanduser().resolve(), args.link_mode
        ))
        for warning in report.get("warnings", []):
            print(f"WARNING: {warning['message']}", file=sys.stderr)
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as error:  # noqa: BLE001 - CLI reports contract failures without a traceback.
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
