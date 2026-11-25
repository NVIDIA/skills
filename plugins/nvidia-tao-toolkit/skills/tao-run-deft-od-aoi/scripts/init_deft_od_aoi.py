#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate normalized DEFT OD AOI roles and freeze a real-only run contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deft_od_aoi_synthesis_contract import validate_synthesis_contract


DEFAULTS = Path(__file__).resolve().parents[1] / "assets" / "default_policy.yaml"
# Normalized handoff roles: KPI/test are held out, ``real`` is the
# defective-real mining pool, and ``clean`` is the verified-clean mining pool.
ROLES = ("kpi", "test", "real", "clean")
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
ROUTED_SYNTHESIS_FIELDS = ("texture_id", "defect_class", "fn_mask_source")


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        result[key] = _merge(result.get(key, {}), value) if isinstance(value, dict) else value
    return result


def _image_path(images: Path, row: dict[str, Any]) -> Path:
    source = str(row.get("source_path") or "").strip()
    path = Path(source) if source else images / str(row.get("file_name") or "")
    return path.expanduser().resolve()


def _validate_bbox(value: Any, image: dict[str, Any], role: str) -> None:
    x, y, width, height = map(float, value)
    image_width = float(image.get("width") or 0)
    image_height = float(image.get("height") or 0)
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"{role} normalized COCO image dimensions must be positive")
    if width <= 0 or height <= 0:
        raise ValueError(f"{role} normalized COCO bbox dimensions must be positive")
    if x < 0 or y < 0 or x + width > image_width or y + height > image_height:
        raise ValueError(f"{role} normalized COCO bbox exceeds image bounds")


def _has_synthesis_clean_reference(pool: Path) -> bool:
    return any(
        image.is_file() and image.suffix.lower() in IMAGE_SUFFIXES
        for texture in pool.iterdir() if texture.is_dir()
        for clean_dir in (texture / "clean_image",) if clean_dir.is_dir()
        for image in clean_dir.iterdir()
    )


def _synthesis_metadata(image: dict[str, Any], annotation: dict[str, Any]) -> dict[str, Any]:
    """Resolve synthesis metadata with the same image-to-annotation precedence as routing."""
    metadata = {}
    for owner, label in ((image, "image"), (annotation, "annotation")):
        nested = owner.get("deft_od_aoi", {})
        if not isinstance(nested, dict):
            raise ValueError(f"KPI {label} deft_od_aoi metadata must be an object")
        for source in (owner, nested):
            metadata.update({key: source[key] for key in
                             ("dataset_id", *ROUTED_SYNTHESIS_FIELDS) if key in source})
    return metadata


def _role(name: str, value: dict[str, Any], require_dataset_id: bool = False) -> tuple[
        dict[str, Any], dict[str, Any]]:
    images = Path(str(value.get("images") or "")).expanduser().resolve()
    coco_path = Path(str(value.get("coco") or "")).expanduser().resolve()
    if not images.is_dir() or not coco_path.is_file():
        raise ValueError(f"{name} images/COCO are missing")
    coco = json.loads(coco_path.read_text())
    categories = coco.get("categories", [])
    if not categories or {str(row.get("name")) for row in categories} != {"defect"}:
        raise ValueError(f"{name} must declare only the defect category")
    category_ids = {int(row["id"]) for row in categories}
    image_rows = coco.get("images", [])
    images_by_id = {int(row["id"]): row for row in image_rows}
    image_ids = set(images_by_id)
    if not image_rows and name != "clean":
        raise ValueError(f"{name} has no images")
    if len(image_ids) != len(image_rows):
        raise ValueError(f"{name} has duplicate image ids")
    counts = {image_id: 0 for image_id in image_ids}
    missing_dataset_ids = []
    for annotation in coco.get("annotations", []):
        image_id, category = int(annotation["image_id"]), int(annotation["category_id"])
        if image_id not in counts or category not in category_ids:
            raise ValueError(f"{name} annotation references unknown image/category")
        _validate_bbox(annotation["bbox"], images_by_id[image_id], name)
        counts[image_id] += 1
        if require_dataset_id:
            metadata = _synthesis_metadata(images_by_id[image_id], annotation)
            if not str(metadata.get("dataset_id") or "").strip():
                image = images_by_id[image_id]
                missing_dataset_ids.append(
                    f"annotation_id={annotation.get('id')} image_id={image_id} "
                    f"file_name={image.get('file_name')!r}"
                )
    if missing_dataset_ids:
        details = ", ".join(missing_dataset_ids[:5])
        remainder = len(missing_dataset_ids) - 5
        if remainder > 0:
            details += f", and {remainder} more"
        raise ValueError(
            "enabled synthesis requires every KPI annotation to resolve a nonempty "
            f"dataset_id; missing for {details}. synthesis.routes is an allowlist: "
            "a nonempty unconfigured dataset_id is valid and skips synthesis"
        )
    paths = [_image_path(images, row) for row in image_rows]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"{name} references missing images; first={missing[0]}")
    if name == "clean" and any(counts.values()):
        raise ValueError("clean role must have zero annotations")
    if name == "real" and any(count == 0 for count in counts.values()):
        raise ValueError("defective-real role contains a boxless image")
    report = {"images": str(images), "coco": str(coco_path),
              "coco_sha256": _sha(coco_path), "image_count": len(paths),
              "annotation_count": sum(counts.values()),
              "identities": {str(path) for path in paths}}
    return report, coco


def _validate_kpi_retrieval_metadata(document: dict[str, Any]) -> None:
    boxed_image_ids = {
        int(annotation["image_id"]) for annotation in document.get("annotations", [])
    }
    fields = ("dataset_id", "texture_id", "defect_class")
    for row in document.get("images", []):
        if int(row["id"]) not in boxed_image_ids:
            continue
        metadata = row.get("deft_od_aoi") or {}
        missing = [
            key for key in fields
            if not str(metadata.get(key) or row.get(key) or "").strip()
        ]
        if missing:
            raise ValueError(
                f"KPI image {row['id']} lacks canonical retrieval metadata: {missing}"
            )


def _validate_routed_kpi_metadata(
        policy: dict[str, Any], coco: dict[str, Any]) -> None:
    images = {int(row["id"]): row for row in coco["images"]}
    routes = policy["synthesis"]["routes"]
    issues = []
    for annotation in coco.get("annotations", []):
        image_id = int(annotation["image_id"])
        image = images[image_id]
        metadata = _synthesis_metadata(image, annotation)
        dataset = str(metadata.get("dataset_id") or "").strip()
        if not dataset or dataset not in routes:
            continue
        identity = (f"annotation_id={annotation.get('id')} image_id={image_id} "
                    f"file_name={image.get('file_name')!r}")
        missing = [key for key in ROUTED_SYNTHESIS_FIELDS
                   if not str(metadata.get(key) or "").strip()]
        if missing:
            issues.append(f"{identity} missing={missing}")
            continue
        mask = Path(str(metadata["fn_mask_source"])).expanduser().resolve()
        if not mask.is_file():
            issues.append(f"{identity} fn_mask_source is not an existing file: {mask}")
    if issues:
        details = "; ".join(issues[:5])
        remainder = len(issues) - 5
        if remainder > 0:
            details += f"; and {remainder} more"
        raise ValueError(f"routed KPI synthesis metadata is incomplete: {details}")


def initialize(config_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    user = yaml.safe_load(config_path.read_text())
    if not isinstance(user, dict):
        raise ValueError("config must be a YAML mapping")
    if "near_miss_real_cap" in (user.get("routing") or {}):
        raise ValueError(
            "routing.near_miss_real_cap is unsupported; use "
            "routing.near_miss_real_cap_per_pocket"
        )
    policy = _merge(yaml.safe_load(DEFAULTS.read_text()), user)
    user_synthesis = user.get("synthesis", {})
    user_retrieval = user.get("retrieval", {})
    inherited_synthesis_candidate_topn = False
    if (isinstance(user_synthesis, dict)
            and "candidate_topn" not in user_synthesis
            and isinstance(user_retrieval, dict)
            and "candidate_overfetch" in user_retrieval):
        # Map an explicit pre-split shared depth to the new synthesis field.
        policy["synthesis"]["candidate_topn"] = user_retrieval["candidate_overfetch"]
        inherited_synthesis_candidate_topn = True
    if (isinstance(user_synthesis, dict)
            and "cumulative_fraction_of_real_defects" in user_synthesis
            and "cumulative_fraction_of_total_defects" not in user_synthesis):
        policy["synthesis"].pop("cumulative_fraction_of_total_defects", None)
        if "fn_selection" not in user_synthesis:
            policy["synthesis"]["fn_selection"] = {"mode": "all_eligible"}
    if not isinstance(policy.get("max_iterations"), int) or policy["max_iterations"] < 1:
        raise ValueError("max_iterations must be a positive integer")
    if not str(policy.get("platform") or "").strip():
        raise ValueError("platform must be selected before initialization")
    checkpoint = Path(str(policy.get("base_checkpoint") or "")).expanduser().resolve()
    if not checkpoint.is_file():
        raise ValueError("base_checkpoint must be a trainable RT-DETR file")
    baseline_mode = str(policy.get("baseline_mode") or "")
    if baseline_mode not in {"cold_start", "checkpoint"}:
        raise ValueError("baseline_mode must be cold_start or checkpoint")
    synthesis = policy.get("synthesis", {})
    synthesis_enabled = bool(synthesis.get("enabled"))
    role_reports = {}
    role_documents = {}
    for name in ROLES:
        report, document = _role(
            name, policy["sources"][name], synthesis_enabled and name == "kpi"
        )
        role_reports[name] = report
        role_documents[name] = document
    _validate_kpi_retrieval_metadata(role_documents["kpi"])
    owners: dict[str, str] = {}
    for name, report in role_reports.items():
        for identity in report.pop("identities"):
            if identity in owners:
                raise ValueError(f"image overlaps {owners[identity]} and {name}: {identity}")
            owners[identity] = name
    gap = policy["gap"]
    if not (0 <= gap["inference_confidence"] <= gap["loose_confidence"]
            <= gap["strict_confidence"] <= 1):
        raise ValueError("inference/loose/strict confidence thresholds are inconsistent")
    if not (0 <= gap["background_iou_upper"] < gap["near_miss_iou_upper"] <= 1):
        raise ValueError("background and near-miss IoU thresholds are inconsistent")
    if policy["class_name"] != "defect":
        raise ValueError("DEFT OD AOI has one foreground class named defect")
    profile = str(((policy.get("retrieval") or {}).get("preprocessing") or {}).get(
        "profile", ""
    ))
    if profile not in {"tight_context", "square_context"}:
        raise ValueError(
            "retrieval.preprocessing.profile must be tight_context or square_context"
        )
    if (profile == "square_context"
            and int(policy["retrieval"].get("output_size", 0)) < 1):
        raise ValueError("retrieval.output_size must be positive")
    strategy = ((policy.get("retrieval") or {}).get("selection") or {}).get(
        "strategy", ""
    )
    if strategy not in {"max_similarity", "round_robin_similarity"}:
        raise ValueError(f"unsupported retrieval selection strategy: {strategy}")
    if int(policy["retrieval"].get("audit_top_k_per_query", 0)) < 1:
        raise ValueError("retrieval.audit_top_k_per_query must be positive")
    if strategy == "round_robin_similarity":
        routing = policy["routing"]
        minimum = int(routing["real_mine_factor_min"])
        maximum = int(routing["real_mine_factor_max"])
        default = int(routing["round_robin_real_factor_default"])
        if not 1 <= minimum <= default <= maximum:
            raise ValueError(
                "round-robin real factor default must be within the frozen bounds"
            )
    if synthesis_enabled:
        try:
            validate_synthesis_contract(synthesis)
        except ValueError as error:
            if (inherited_synthesis_candidate_topn
                    and "candidate_topn" in str(error)):
                raise ValueError(
                    "retrieval.candidate_overfetch was inherited as "
                    f"synthesis.candidate_topn; {error}"
                ) from error
            raise
        pool = Path(str(synthesis.get("pool_dataset_root") or "")).expanduser().resolve()
        if not pool.is_dir():
            raise ValueError("enabled synthesis needs pool_dataset_root")
        if not _has_synthesis_clean_reference(pool):
            raise ValueError("enabled synthesis needs at least one clean reference image")
        if not Path(str(synthesis.get("defect_spec") or "")).is_file():
            raise ValueError("enabled synthesis needs defect_spec")
        if not synthesis.get("routes"):
            raise ValueError("enabled synthesis needs at least one dataset route")
        for name, route in synthesis["routes"].items():
            ready = Path(str(route.get("checkpoint") or "")).is_file() and Path(
                str(route.get("recipe") or "")).is_file()
            finetune = route.get("finetune") or {}
            if not ready and any(not str(finetune.get(key) or "").strip() for key in
                                 ("dataset_root", "validation_testcase", "base_checkpoint",
                                  "vae_path", "checkpoint_root", "result_handoff")):
                raise ValueError(f"synthesis route {name} needs checkpoint/recipe or finetune inputs")
        # Reuse the document parsed from the resolved, existence-checked KPI path.
        _validate_routed_kpi_metadata(policy, role_documents["kpi"])
    output.mkdir(parents=True)
    policy["base_checkpoint"] = str(checkpoint)
    policy["sources"] = {name: {"images": role_reports[name]["images"],
                                 "coco": role_reports[name]["coco"]} for name in ROLES}
    frozen = output / "deft_od_aoi_policy.yaml"
    frozen.write_text(yaml.safe_dump(policy, sort_keys=False))
    classmap = output / "inference_classmap.txt"
    classmap.write_text("background\ndefect\n")
    bootstrap_required = synthesis_enabled and any(
        not (Path(str(route.get("checkpoint") or "")).is_file()
             and Path(str(route.get("recipe") or "")).is_file())
        for route in policy.get("synthesis", {}).get("routes", {}).values()
    )
    retrieval_capabilities = {
        role: {"status": "AVAILABLE" if role_reports[role]["image_count"] else "UNAVAILABLE",
               "reason": None if role_reports[role]["image_count"] else "empty_source_role",
               "source_image_count": role_reports[role]["image_count"]}
        for role in ("real", "clean")
    }
    warnings = [{
        "code": "empty_retrieval_source_role",
        "role": role,
        "message": f"{role} retrieval source role is empty; that producer starts exhausted",
    } for role, evidence in retrieval_capabilities.items()
        if evidence["status"] == "UNAVAILABLE"]
    if inherited_synthesis_candidate_topn and synthesis_enabled:
        warnings.append({
            "code": "inherited_synthesis_candidate_topn",
            "source": "retrieval.candidate_overfetch",
            "target": "synthesis.candidate_topn",
            "value": policy["synthesis"]["candidate_topn"],
            "message": (
                "synthesis.candidate_topn was omitted; inherited the explicit "
                "retrieval.candidate_overfetch value for backward compatibility"
            ),
        })
    state = {"schema_version": 1, "status": "READY",
             "mode": "rtdetr_with_synthesis" if synthesis_enabled else "rtdetr_real_only",
             "baseline_mode": baseline_mode,
             "synthesis_enabled": synthesis_enabled,
             "synthesis_bootstrap_required": bootstrap_required,
             "current_iteration": 0,
             "next_stage": "synthesis_bootstrap" if bootstrap_required else "candidate_cache",
             "max_iterations": policy["max_iterations"], "platform": policy["platform"],
             "base_checkpoint": str(checkpoint), "policy": str(frozen.resolve()),
             "policy_sha256": _sha(frozen), "classmap": str(classmap.resolve()),
             "roles": role_reports,
             "capabilities": {"retrieval": retrieval_capabilities},
             "warnings": warnings, "iterations": {}}
    _json(output / "deft_state.json", state)
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    state = initialize(args.config.resolve(), args.output_dir.resolve())
    for warning in state.get("warnings", []):
        print(f"WARNING: {warning['message']}", file=sys.stderr)
    print(json.dumps(state, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
