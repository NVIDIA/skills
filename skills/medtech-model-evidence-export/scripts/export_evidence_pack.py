#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export a sanitized medical-inference run summary to MLflow."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable

SKILL_NAME = "medtech_model_evidence_export"
MODES = ("dry-run", "local", "databricks")
ARTIFACT_POLICIES = ("metadata", "preview", "all")
PACK_DOCUMENTS = (
    "manifest.json",
    "validation_summary.json",
    "runtime_profile.json",
    "cost_profile.json",
    "integrity_check.json",
)
PRIVATE_KEY = re.compile(
    r"(^|[_.-])(api[_-]?key|authorization|credential|password|secret|token|"
    r"patient|subject|medical[_-]?record|mrn|accession|birth|dob|"
    r"study[_-]?instance[_-]?uid|series[_-]?instance[_-]?uid)($|[_.-])",
    re.IGNORECASE,
)
PATH_KEY = re.compile(r"(^|_)(dir|directory|file|path|root)($|_)", re.IGNORECASE)
NIFTI_SUFFIXES = (".nii", ".nii.gz")
VISUAL_SUFFIXES = (".html", ".png", ".jpg", ".jpeg")


def _load_json(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.is_file():
        if required:
            raise ValueError(f"missing JSON file: {path}")
        return {}
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"cannot read JSON object from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("mlflow", "nibabel", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def _safe_key(value: str) -> str:
    return (re.sub(r"[^A-Za-z0-9_. /-]+", "_", value).strip(" ./_-") or "value")[:250]


def _safe_path(value: str) -> str:
    if value.startswith(("http://", "https://", "hf://")):
        return value[:500]
    return (Path(value).name or "<path>")[:500]


def _sanitize(value: Any, *, key: str = "", depth: int = 0) -> Any:
    """Redact likely secrets/identifiers and remove raw logs and commands."""
    if depth > 8:
        return "<max-depth>"
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for name, child in value.items():
            name = str(name)
            if name.lower() in {
                "argv",
                "command",
                "logs",
                "stderr",
                "stderr_tail",
                "stdout",
                "stdout_tail",
            }:
                continue
            output[name] = (
                "<redacted>"
                if PRIVATE_KEY.search(name)
                else _sanitize(child, key=name, depth=depth + 1)
            )
        return output
    if isinstance(value, list):
        return [_sanitize(item, key=key, depth=depth + 1) for item in value[:200]]
    if isinstance(value, str):
        if "prompt" in key.lower():
            return f"sha256:{hashlib.sha256(value.encode()).hexdigest()}"
        if PATH_KEY.search(key) or value.startswith(("/", "~/")):
            return _safe_path(value)
        return value[:1000]
    return value


def _flatten_params(value: Any, prefix: str, output: dict[str, str], depth: int = 0) -> None:
    if len(output) >= 100 or depth > 5:
        return
    if isinstance(value, dict):
        for key in sorted(value, key=str):
            name = _safe_key(f"{prefix}.{key}" if prefix else str(key))
            if PRIVATE_KEY.search(name) or "prompt" in name.lower():
                continue
            child = value[key]
            if isinstance(child, dict):
                _flatten_params(child, name, output, depth + 1)
            else:
                rendered = _sanitize(child, key=name)
                if isinstance(rendered, (list, dict)):
                    rendered = json.dumps(rendered, sort_keys=True, separators=(",", ":"))
                elif rendered is None:
                    rendered = "null"
                elif isinstance(rendered, bool):
                    rendered = str(rendered).lower()
                elif not isinstance(rendered, (str, int, float)):
                    continue
                output[name] = str(rendered)[:500]
            if len(output) >= 100:
                return


def _numeric_metrics(value: Any, prefix: str = "") -> dict[str, float]:
    output: dict[str, float] = {}
    if not isinstance(value, dict):
        return output
    for key, child in value.items():
        name = _safe_key(f"{prefix}.{key}" if prefix else str(key))
        if isinstance(child, (int, float)) and not isinstance(child, bool):
            number = float(child)
            if math.isfinite(number):
                output[name] = number
        elif isinstance(child, dict) and name.count(".") < 4:
            output.update(_numeric_metrics(child, name))
    return output


def _find_text(value: Any, key_part: str) -> str | None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key_part in str(key).lower() and isinstance(child, str):
                return child
            found = _find_text(child, key_part)
            if found:
                return found
    if isinstance(value, list):
        for child in value:
            found = _find_text(child, key_part)
            if found:
                return found
    return None


def _source_payload(source: Path) -> dict[str, Any]:
    resolved = source.expanduser().resolve()
    if resolved.is_file():
        return {
            "pack": None,
            "result": _load_json(resolved),
            "manifest": {},
            "validation": {},
            "runtime": {},
            "trust": {},
            "documents": {},
            "kind": "direct_run",
        }

    if (resolved / "manifest.json").is_file():
        pack, root = resolved, resolved
    elif (resolved / "skill_run" / "manifest.json").is_file():
        pack, root = resolved / "skill_run", resolved
    else:
        raise ValueError(
            f"{resolved} is neither an evidence pack nor a trusted-run directory with skill_run/"
        )
    manifest = _load_json(pack / "manifest.json")
    documents: dict[str, Any] = {}
    for name in PACK_DOCUMENTS:
        document = _load_json(pack / name, required=False)
        if document:
            documents[name] = _sanitize(document)
    trust = _load_json(root / "trust_summary.json", required=False)
    if trust:
        documents["trust_summary.json"] = _sanitize(trust)
    return {
        "pack": pack,
        "result": _load_json(pack / "output.json", required=False),
        "manifest": manifest,
        "validation": _load_json(pack / "validation_summary.json"),
        "runtime": _load_json(pack / "runtime_profile.json"),
        "trust": trust,
        "documents": documents,
        "kind": str(manifest.get("pack_kind", "unknown")),
    }


def _collect_params(
    result: dict[str, Any], config: dict[str, Any], seed: int | None
) -> tuple[dict[str, str], int | None, str | None]:
    params: dict[str, str] = {}
    for root in ("input", "parameters", "params", "config"):
        if root in result:
            _flatten_params(result[root], root, params)
    invocation = result.get("invocation") if isinstance(result.get("invocation"), dict) else {}
    _flatten_params(invocation.get("rendered_infer_config", {}), "config", params)
    if invocation.get("upstream_commit"):
        params["model.upstream_commit"] = str(invocation["upstream_commit"])[:500]
    inventory = invocation.get("model_inventory")
    files = inventory.get("files", []) if isinstance(inventory, dict) else []
    for index, item in enumerate(files[:20] if isinstance(files, list) else []):
        if not isinstance(item, dict):
            continue
        if item.get("path"):
            params[f"model.checkpoint_{index}.name"] = Path(str(item["path"])).name
        if item.get("sha256"):
            params[f"model.checkpoint_{index}.sha256"] = str(item["sha256"])[:500]
    for key in ("model", "model_repo", "model_weights_repo", "version"):
        if result.get(key) is not None:
            params[f"source.{key}"] = str(result[key])[:500]
    if config:
        _flatten_params(config, "config_file", params)

    input_data = result.get("input") if isinstance(result.get("input"), dict) else {}
    detected_seed = seed
    if detected_seed is None:
        candidate = input_data.get("random_seed", input_data.get("seed", result.get("seed")))
        if isinstance(candidate, int) and not isinstance(candidate, bool):
            detected_seed = candidate
    if detected_seed is not None:
        params["reproducibility.seed"] = str(detected_seed)

    recipe = {
        "input": result.get("input"),
        "parameters": result.get("parameters") or result.get("params"),
        "config": result.get("config"),
        "rendered_config": invocation.get("rendered_infer_config"),
        "config_file": config or None,
    }
    recipe = {key: value for key, value in recipe.items() if value not in (None, {}, [])}
    return params, detected_seed, _sha256_json(recipe) if recipe else None


def _embedded_metrics(result: dict[str, Any]) -> dict[str, float]:
    metrics = _numeric_metrics(result.get("metrics"))
    metrics.update(_numeric_metrics(result.get("runtime"), "runtime"))
    output = result.get("output") if isinstance(result.get("output"), dict) else {}
    samples = output.get("samples") if isinstance(output.get("samples"), list) else []
    keys = {
        "image_hu_min": "hu_min",
        "image_hu_max": "hu_max",
        "image_hu_mean": "hu_mean",
        "image_hu_std": "hu_std",
    }
    for index, sample in enumerate(samples[:3]):
        if not isinstance(sample, dict):
            continue
        for source_key, metric_key in keys.items():
            value = sample.get(source_key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics[f"quality.sample_{index}.{metric_key}"] = float(value)
        foreground, background = (
            sample.get("label_foreground_voxels"),
            sample.get("label_background_voxels"),
        )
        if isinstance(foreground, int) and isinstance(background, int) and foreground + background:
            metrics[f"quality.sample_{index}.mask_foreground_pct"] = (
                100.0 * foreground / (foreground + background)
            )
    return metrics


def _artifact_kind(key: str, path: Path) -> str | None:
    name = path.name.lower()
    if name.endswith(NIFTI_SUFFIXES):
        return "mask" if any(word in key.lower() for word in ("label", "mask", "seg")) else "image"
    return "visual" if name.endswith(VISUAL_SUFFIXES) else None


def _discover_artifacts(result: dict[str, Any], base: Path) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []

    def visit(value: Any, key: str = "") -> None:
        if isinstance(value, dict):
            for child_key, child in value.items():
                visit(child, str(child_key))
        elif isinstance(value, list):
            for child in value:
                visit(child, key)
        elif isinstance(value, str) and ("path" in key.lower() or key.lower().endswith("html")):
            path = Path(value).expanduser()
            if path.is_absolute():
                path = path.resolve()
            else:
                beside_source = (base / path).resolve()
                path = beside_source if beside_source.exists() else path.resolve()
            kind = _artifact_kind(key, path)
            if kind:
                found.append({"kind": kind, "path": path})

    visit(result.get("output", {}), "output")
    return list({str(item["path"]): item for item in found}.values())


def _sample_volume(image: Any, numpy: Any, max_voxels: int = 1_000_000) -> Any:
    shape = tuple(int(value) for value in image.shape[:3])
    stride = max(1, math.ceil((math.prod(shape) / max_voxels) ** (1 / 3)))
    selector: list[Any] = [slice(None, None, stride) for _ in shape]
    selector.extend(0 for _ in image.shape[3:])
    return numpy.asarray(image.dataobj[tuple(selector)], dtype=numpy.float32)


def _nifti_metrics(
    image_path: Path,
    mask_path: Path | None,
    index: int,
    ct_intensity: bool,
    label_mapping: list[dict[str, Any]],
) -> tuple[dict[str, float], str | None]:
    try:
        nibabel = importlib.import_module("nibabel")
        numpy = importlib.import_module("numpy")
        values = _sample_volume(nibabel.load(str(image_path)), numpy)
        finite = values[numpy.isfinite(values)]
        if not finite.size:
            return {}, f"{image_path.name}: no finite voxels"
        prefix = f"quality.sample_{index}"
        unit = "hu" if ct_intensity else "intensity"
        mean, std = float(finite.mean()), float(finite.std())
        metrics = {
            f"{prefix}.{unit}_min": float(finite.min()),
            f"{prefix}.{unit}_max": float(finite.max()),
            f"{prefix}.{unit}_mean": mean,
            f"{prefix}.{unit}_std": std,
            f"{prefix}.snr_abs_mean_over_std": abs(mean) / std if std else 0.0,
        }
        if mask_path and mask_path.is_file():
            labels = _sample_volume(nibabel.load(str(mask_path)), numpy)
            total = int(labels.size)
            if total:
                metrics[f"{prefix}.mask_foreground_pct"] = (
                    100.0 * float(numpy.count_nonzero(labels)) / total
                )
                for item in label_mapping:
                    if (
                        not isinstance(item, dict)
                        or "tumor" not in str(item.get("anatomy", "")).lower()
                    ):
                        continue
                    label_id = item.get("output_label_id", item.get("label_id"))
                    if isinstance(label_id, int):
                        name = re.sub(
                            r"[^a-z0-9]+", "_", str(item.get("anatomy", "tumor")).lower()
                        ).strip("_")
                        metrics[f"{prefix}.{name}_volume_pct"] = (
                            100.0 * float(numpy.count_nonzero(labels == label_id)) / total
                        )
        return metrics, None
    except Exception as exc:
        return {}, f"{image_path.name}: {type(exc).__name__}: {exc}"


def _preview(image_path: Path, mask_path: Path | None) -> Any:
    nibabel = importlib.import_module("nibabel")
    numpy = importlib.import_module("numpy")
    image = nibabel.load(str(image_path))
    z_index = int(image.shape[2] // 2)
    selector: list[Any] = [slice(None), slice(None), z_index]
    selector.extend(0 for _ in image.shape[3:])
    plane = numpy.asarray(image.dataobj[tuple(selector)], dtype=numpy.float32)
    finite = plane[numpy.isfinite(plane)]
    if not finite.size:
        raise ValueError("preview source has no finite voxels")
    low, high = numpy.percentile(finite, (1.0, 99.0))
    high = high if high > low else low + 1.0
    gray = numpy.clip((numpy.nan_to_num(plane, nan=low) - low) / (high - low), 0, 1)
    rgb = numpy.repeat((gray * 255).astype(numpy.uint8)[..., None], 3, axis=2)
    if mask_path and mask_path.is_file():
        mask = nibabel.load(str(mask_path))
        mask_selector: list[Any] = [
            slice(None),
            slice(None),
            min(z_index, int(mask.shape[2]) - 1),
        ]
        mask_selector.extend(0 for _ in mask.shape[3:])
        overlay = numpy.asarray(mask.dataobj[tuple(mask_selector)]) != 0
        if overlay.shape == gray.shape:
            rgb[overlay, 0] = 255
            rgb[overlay, 1:] = (rgb[overlay, 1:] * 0.35).astype(numpy.uint8)
    return rgb


def collect_summary(
    source: Path,
    *,
    artifact_policy: str = "metadata",
    config_path: Path | None = None,
    seed: int | None = None,
    source_ref: str | None = None,
    image_paths: Iterable[Path] = (),
    mask_paths: Iterable[Path] = (),
    max_artifact_mb: float = 256.0,
    note: str | None = None,
) -> dict[str, Any]:
    """Collect MLflow-ready facts without contacting a tracking server."""
    source_data = _source_payload(source)
    result = source_data["result"]
    manifest = source_data["manifest"]
    validation = source_data["validation"]
    runtime = source_data["runtime"]
    trust = source_data["trust"]

    config: dict[str, Any] = {}
    config_name = config_sha256 = None
    if config_path:
        config_file = config_path.expanduser().resolve()
        if not config_file.is_file():
            raise ValueError(f"config file not found: {config_file}")
        config_name, config_sha256 = config_file.name, _sha256_file(config_file)
        try:
            config = _load_json(config_file)
        except ValueError:
            pass

    skill_id = (
        manifest.get("skill_id") or result.get("skill_id") or result.get("skill") or "unknown"
    )
    skill_version = (
        manifest.get("skill_version") or result.get("skill_version") or result.get("version")
    )
    params, detected_seed, recipe_sha256 = _collect_params(result, config, seed)
    extra_params = {
        "source.config_name": config_name,
        "source.config_sha256": config_sha256,
        "source.recipe_ref": source_ref,
        "source.repo_git_sha": manifest.get("repo_git_sha"),
    }
    params.update(
        {key: str(value)[:500] for key, value in extra_params.items() if value is not None}
    )

    metrics = _embedded_metrics(result)
    metrics.update(_numeric_metrics(runtime, "runtime"))
    base = source_data["pack"] or source.expanduser().resolve().parent
    artifacts = _discover_artifacts(result, base)
    artifacts.extend(
        {"kind": "image", "path": Path(path).expanduser().resolve()} for path in image_paths
    )
    artifacts.extend(
        {"kind": "mask", "path": Path(path).expanduser().resolve()} for path in mask_paths
    )
    artifacts = list({str(item["path"]): item for item in artifacts}.values())
    images = [item for item in artifacts if item["kind"] == "image"]
    masks = [item for item in artifacts if item["kind"] == "mask"]
    output = result.get("output") if isinstance(result.get("output"), dict) else {}
    label_mapping = output.get("output_label_mapping", [])
    input_data = result.get("input") if isinstance(result.get("input"), dict) else {}
    modality = str(input_data.get("modality") or result.get("modality") or "").upper()
    skill_name = str(skill_id).lower()
    is_ct = modality == "CT" or (not modality and "ct" in skill_name and "mr" not in skill_name)
    warnings: list[str] = []
    for index, image in enumerate(images[:3]):
        if not image["path"].is_file():
            warnings.append(f"referenced image not found: {image['path'].name}")
            continue
        mask_path = masks[index]["path"] if index < len(masks) else None
        derived, warning = _nifti_metrics(
            image["path"],
            mask_path,
            index,
            is_ct,
            label_mapping if isinstance(label_mapping, list) else [],
        )
        metrics.update(derived)
        if warning:
            warnings.append(warning)

    max_bytes = int(max_artifact_mb * 1024 * 1024)
    plan_items: list[dict[str, Any]] = []
    for item in artifacts:
        path, kind = item["path"], item["kind"]
        size = path.stat().st_size if path.is_file() else None
        selected = (kind == "visual" and artifact_policy in {"preview", "all"}) or (
            kind in {"image", "mask"} and artifact_policy == "all"
        )
        status = "not_selected"
        if selected:
            status = (
                "missing" if size is None else "over_size_limit" if size > max_bytes else "ready"
            )
        item["status"] = status
        plan_items.append({"name": path.name, "kind": kind, "bytes": size, "status": status})

    documents = source_data["documents"]
    metadata_names = [
        "evidence_summary.json",
        "parameters.json",
        "quality_metrics.json",
        "artifact_manifest.json",
        *sorted(documents),
    ]
    artifact_plan = {
        "policy": artifact_policy,
        "metadata": metadata_names,
        "metadata_count": len(metadata_names),
        "preview_count": (
            sum(item["path"].is_file() for item in images[:3])
            if artifact_policy in {"preview", "all"}
            else 0
        ),
        "items": plan_items,
        "requires_confirmation": artifact_policy in {"preview", "all"},
        "max_artifact_mb": max_artifact_mb,
    }

    tags = {
        "medical_ai_skills.not_clinical": "true",
        "medical_ai_skills.intended_use": "engineering_verification",
        "medical_ai_skills.pack_kind": source_data["kind"],
        "medical_ai_skills.skill_id": str(skill_id),
        "medical_ai_skills.artifact_policy": artifact_policy,
    }
    optional_tags = {
        "medical_ai_skills.run_id": manifest.get("run_id") or result.get("run_id"),
        "medical_ai_skills.skill_version": skill_version,
        "medical_ai_skills.repo_git_sha": manifest.get("repo_git_sha"),
        "medical_ai_skills.validation_overall": validation.get("overall_status")
        or result.get("validation_overall"),
        "medical_ai_skills.trust_overall": trust.get("overall"),
        "medical_ai_skills.recipe_sha256": recipe_sha256,
        "medical_ai_skills.source_config_sha256": config_sha256,
        "medical_ai_skills.source_ref": source_ref,
    }
    tags.update(
        {key: str(value)[:500] for key, value in optional_tags.items() if value is not None}
    )
    tags[
        (
            "medical_ai_skills.reproducibility_seed"
            if detected_seed is not None
            else "medical_ai_skills.reproducibility_seed_missing"
        )
    ] = (str(detected_seed) if detected_seed is not None else "true")
    prompt = _find_text(result, "prompt")
    if prompt:
        tags["medical_ai_skills.source_prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()

    validation_status = tags.get("medical_ai_skills.validation_overall", "unknown")
    human_note = note or (
        f"Post-hoc MLflow evidence export for {skill_id}; validation={validation_status}; "
        f"metrics={len(metrics)}; artifact_policy={artifact_policy}. Engineering use only."
    )
    tags["mlflow.note.content"] = human_note[:500]
    logged_plan = {
        **artifact_plan,
        "items": [
            {
                **item,
                "name": f"{item['kind']}_{index}{''.join(Path(item['name']).suffixes)}",
            }
            for index, item in enumerate(plan_items)
        ],
    }
    logged_summary = {
        "pack_kind": source_data["kind"],
        "skill_id": skill_id,
        "skill_version": skill_version,
        "validation_overall": tags.get("medical_ai_skills.validation_overall"),
        "trust_overall": tags.get("medical_ai_skills.trust_overall"),
        "params": params,
        "metrics": metrics,
        "artifact_plan": logged_plan,
        "note": human_note,
    }
    return {
        "source": {
            "pack_path": str(source_data["pack"] or source.expanduser().resolve()),
            "pack_kind": source_data["kind"],
            "skill_id": skill_id,
            "skill_version": skill_version,
        },
        "tags": tags,
        "params": params,
        "metrics": metrics,
        "artifact_plan": artifact_plan,
        "note": human_note,
        "warnings": warnings,
        "logged_summary": logged_summary,
        "_documents": documents,
        "_artifacts": artifacts,
        "_images": images[:3],
        "_masks": masks[:3],
    }


def _mlflow_available() -> bool:
    try:
        return importlib.util.find_spec("mlflow") is not None
    except (ImportError, ValueError):
        return False


def _tracking_uri(mode: str, requested: str | None) -> str:
    if requested:
        return requested
    if os.environ.get("MLFLOW_TRACKING_URI"):
        return os.environ["MLFLOW_TRACKING_URI"]
    return "databricks" if mode == "databricks" else (Path.cwd() / "mlruns").resolve().as_uri()


def log_summary(
    summary: dict[str, Any],
    *,
    mode: str,
    tracking_uri: str | None,
    experiment_name: str | None,
    run_name: str | None,
    mlflow_module: Any | None = None,
) -> dict[str, Any]:
    """Log one collected summary; return MLflow errors as structured data."""
    uri = _tracking_uri(mode, tracking_uri)
    try:
        if mode == "local" and uri.startswith("file:"):
            os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        mlflow = mlflow_module or importlib.import_module("mlflow")
        mlflow.set_tracking_uri(uri)
        if experiment_name:
            mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name) as active_run:
            mlflow.set_tags(summary["tags"])
            if summary["params"]:
                mlflow.log_params(summary["params"])
            if summary["metrics"]:
                mlflow.log_metrics(summary["metrics"])
            metadata = {
                "evidence_summary.json": summary["logged_summary"],
                "parameters.json": summary["params"],
                "quality_metrics.json": summary["metrics"],
                "artifact_manifest.json": summary["logged_summary"]["artifact_plan"],
            }
            for name, document in metadata.items():
                mlflow.log_dict(document, f"medical_ai_skills/{name}")
            for name, document in summary["_documents"].items():
                mlflow.log_dict(document, f"medical_ai_skills/evidence/{name}")

            policy = summary["artifact_plan"]["policy"]
            if policy in {"preview", "all"}:
                for index, image in enumerate(summary["_images"]):
                    if not image["path"].is_file():
                        continue
                    mask = summary["_masks"][index] if index < len(summary["_masks"]) else None
                    mlflow.log_image(
                        _preview(image["path"], mask["path"] if mask else None),
                        artifact_file=f"medical_ai_skills/previews/sample_{index}.png",
                    )
                for item in summary["_artifacts"]:
                    if item["kind"] == "visual" and item["status"] == "ready":
                        mlflow.log_artifact(
                            str(item["path"]), artifact_path="medical_ai_skills/review"
                        )
            if policy == "all":
                for item in summary["_artifacts"]:
                    if item["kind"] in {"image", "mask"} and item["status"] == "ready":
                        mlflow.log_artifact(
                            str(item["path"]), artifact_path="medical_ai_skills/raw"
                        )
            run_id = active_run.info.run_id
    except Exception as exc:
        return {
            "available": mlflow_module is not None or _mlflow_available(),
            "tracking_uri": uri,
            "experiment_name": experiment_name,
            "run_id": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "available": True,
        "tracking_uri": uri,
        "experiment_name": experiment_name,
        "run_id": run_id,
        "error": None,
    }


def _failure(args: argparse.Namespace, error: str) -> dict[str, Any]:
    return {
        "skill": SKILL_NAME,
        "status": "failed",
        "mode": args.mode,
        "source": {
            "pack_path": str(args.source),
            "pack_kind": "unknown",
            "skill_id": None,
            "skill_version": None,
        },
        "tags": {},
        "params": {},
        "metrics": {},
        "artifact_plan": {
            "policy": args.artifact_policy,
            "metadata": [],
            "metadata_count": 0,
            "preview_count": 0,
            "items": [],
            "requires_confirmation": args.artifact_policy in {"preview", "all"},
            "max_artifact_mb": max(args.max_artifact_mb, 0.000001),
        },
        "note": "Export failed before an MLflow run was created.",
        "warnings": [],
        "mlflow": {
            "available": _mlflow_available(),
            "tracking_uri": args.tracking_uri,
            "experiment_name": args.experiment_name,
            "run_id": None,
            "error": error,
        },
        "environment": {"packages": _package_versions()},
        "intended_use_disclaimer": "Engineering verification only.",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", type=Path, help="Evidence pack, trusted-run directory, or result JSON"
    )
    parser.add_argument("--mode", choices=MODES, default="dry-run")
    parser.add_argument("--tracking-uri")
    parser.add_argument("--experiment-name")
    parser.add_argument("--run-name")
    parser.add_argument("--artifact-policy", choices=ARTIFACT_POLICIES, default="metadata")
    parser.add_argument("--confirm-medical-artifact-upload", action="store_true")
    parser.add_argument("--max-artifact-mb", type=float, default=256.0)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--source-ref")
    parser.add_argument("--note")
    parser.add_argument("--image", type=Path, action="append", default=[])
    parser.add_argument("--mask", type=Path, action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_artifact_mb <= 0:
        print(json.dumps(_failure(args, "--max-artifact-mb must be positive"), indent=2))
        return 2
    try:
        summary = collect_summary(
            args.source,
            artifact_policy=args.artifact_policy,
            config_path=args.config,
            seed=args.seed,
            source_ref=args.source_ref,
            image_paths=args.image,
            mask_paths=args.mask,
            max_artifact_mb=args.max_artifact_mb,
            note=args.note,
        )
    except ValueError as exc:
        print(json.dumps(_failure(args, str(exc)), indent=2))
        return 2

    public = {
        key: value
        for key, value in summary.items()
        if not key.startswith("_") and key != "logged_summary"
    }
    payload = {
        "skill": SKILL_NAME,
        "status": "dry_run",
        "mode": args.mode,
        **public,
        "mlflow": {
            "available": _mlflow_available(),
            "tracking_uri": args.tracking_uri,
            "experiment_name": args.experiment_name,
            "run_id": None,
            "error": None,
        },
        "environment": {"packages": _package_versions()},
        "intended_use_disclaimer": (
            "Engineering verification only. Medical-image artifacts are uploaded only "
            "after explicit policy selection and confirmation."
        ),
    }
    if (
        args.mode != "dry-run"
        and args.artifact_policy in {"preview", "all"}
        and not args.confirm_medical_artifact_upload
    ):
        payload["status"] = "failed"
        payload["mlflow"][
            "error"
        ] = "preview/all requires --confirm-medical-artifact-upload in live modes"
        print(json.dumps(payload, indent=2))
        return 2
    if args.mode != "dry-run":
        payload["mlflow"] = log_summary(
            summary,
            mode=args.mode,
            tracking_uri=args.tracking_uri,
            experiment_name=args.experiment_name,
            run_name=args.run_name or f"evidence:{Path(summary['source']['pack_path']).name}",
        )
        payload["status"] = "logged" if payload["mlflow"]["error"] is None else "failed"
    print(json.dumps(payload, indent=2))
    return 0 if payload["status"] != "failed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
