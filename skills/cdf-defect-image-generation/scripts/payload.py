#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check and preview an SMT DIG payload using the product's actual compiler.

Run with the product Airflow Python environment. No API, storage, or cluster
operations are performed. The checkout passed with --repo-root must be trusted.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from urllib.parse import urlsplit

COMPONENTS = {
    "golden_generation": {"image_generation", "simulation_output", "augmentation"},
    "defect_generation": {"anomaly_input", "anomaly_generation"},
}
# The payload model accepts, then ignores, every other cosmos and service_lifecycle field.
COSMOS_FIELDS = {
    "image_edit_service_url",
    "image_edit_model",
    "base_config_path",
    "external_services",
    "output_directory",
}
LIFECYCLE_FIELDS = {"image_edit_service": {"replicas", "enabled"}}
# Keys the workflow sets itself, so an override is either discarded or breaks staging.
MANAGED = {
    "image_generation": {
        ("scene", "usd"): "use input_scene",
        ("output_dir",): "use output_directory",
    },
    "augmentation": {("data",): "the workflow writes one input/output pair per image"},
    "anomaly_generation": {
        ("input",): "the workflow stages inputs inside the task",
        ("output",): "the workflow stages outputs inside the task",
        ("recipe",): "set staging.fetch.recipe",
        ("checkpoint",): "set staging.fetch.checkpoint",
        ("staging", "publish"): "the workflow publishes to the run output",
    },
}
REQUIRED_STAGES = {"prepare", "generate", "export", "publish"}


class PayloadError(ValueError):
    """A payload is not ready for local compilation."""


def location(value: object, field: str, schemes: set[str]) -> None:
    if not isinstance(value, str) or not value.strip() or any(c in value for c in "<>\n\r"):
        raise PayloadError(f"{field} requires an explicit location without placeholders")
    parsed = urlsplit(value)
    if parsed.scheme not in schemes or not parsed.netloc:
        raise PayloadError(f"{field} requires a URL using {', '.join(sorted(schemes))}")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise PayloadError(f"{field} must not embed credentials, query parameters or fragments")


def supported(mapping: object, allowed: set[str], field: str) -> None:
    extra = sorted(set(mapping or {}) - allowed) if isinstance(mapping, dict) else []
    if extra:
        raise PayloadError(
            f"{field} has unsupported keys {', '.join(extra)}; use {', '.join(sorted(allowed))}"
        )


def has_path(value: object, path: tuple[str, ...]) -> bool:
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return False
        value = value[key]
    return True


def check_overrides(payload: dict, model) -> None:
    supported(payload.get("cosmos"), COSMOS_FIELDS, "cosmos")
    lifecycle = payload.get("service_lifecycle")
    supported(lifecycle, set(LIFECYCLE_FIELDS), "service_lifecycle")
    for name, fields in LIFECYCLE_FIELDS.items():
        supported((lifecycle or {}).get(name), fields, f"service_lifecycle.{name}")
    collections = {
        "golden_generation": model.golden_generation,
        "defect_generation": model.defect_generation,
    }
    for name, allowed in COMPONENTS.items():
        collection = collections[name]
        if collection is None:
            continue
        sources = [(f"{name}.defaults", collection.defaults)]
        sources += [
            (f"{name}.{item.id}", item.model_extra or {}) for item in collection.configurations
        ]
        for label, overrides in sources:
            supported(overrides, allowed, label)
            for component, override in overrides.items():
                for path, hint in MANAGED.get(component, {}).items():
                    if has_path(override, path):
                        raise PayloadError(
                            f"{label}.{component}.{'.'.join(path)} is set by the workflow; {hint}"
                        )
            endpoints = overrides.get("augmentation", {}).get("endpoints")
            if endpoints is not None and not (
                isinstance(endpoints, list)
                and any(
                    isinstance(item, dict) and "image_edit" in (item.get("id"), item.get("role"))
                    for item in endpoints
                )
            ):
                raise PayloadError(
                    f"{label}.augmentation.endpoints replaces the list and must keep the image_edit entry"
                )
            stages = overrides.get("anomaly_generation", {}).get("stages")
            if stages is not None and not (
                isinstance(stages, list)
                and REQUIRED_STAGES <= {item for item in stages if isinstance(item, str)}
            ):
                raise PayloadError(
                    f"{label}.anomaly_generation.stages must keep {', '.join(sorted(REQUIRED_STAGES))}"
                )


def build_plan(payload: dict, repo_root: Path, run_id: str = "preview") -> dict:
    source = repo_root.resolve() / "airflow"
    compiler = (
        source / "dags/workflows/defect_image_generation_dag/callables/smt/config_generation.py"
    )
    if not compiler.is_file():
        raise PayloadError("--repo-root must name a trusted checkout containing the SMT DAG")
    sys.path.insert(0, str(source))
    # Airflow initialization can log to stdout; keep the command's JSON output clean.
    with contextlib.redirect_stdout(sys.stderr):
        from dags.workflows.defect_image_generation_dag.callables.config_overrides import (
            apply_component_overrides,
        )
        from dags.workflows.defect_image_generation_dag.callables.smt.config_generation import (
            build_work_items,
        )
        from dags.workflows.defect_image_generation_dag.callables.smt.models import (
            SmtDefectImageGenerationPayload,
        )

        model = SmtDefectImageGenerationPayload.model_validate(payload)
        check_overrides(payload, model)
        location(model.output_directory, "output_directory", {"s3"})
        if model.pipeline_mode != "defect":
            location(model.input_scene, "input_scene", {"s3"})
        if model.external_services and model.pipeline_mode in {"sim2real", "full"}:
            location(
                model.cosmos.image_edit_service_url,
                "cosmos.image_edit_service_url",
                {"http", "https"},
            )
        if model.pipeline_mode in {"full", "defect"}:
            collection = model.defect_generation
            for variant in collection.configurations:
                # Start from an empty template so the checked-in example locations cannot silently pass.
                explicit = apply_component_overrides(
                    {},
                    defaults=collection.defaults,
                    configuration=variant,
                    component_names=("anomaly_generation",),
                )
                fetch = explicit.get("staging", {}).get("fetch", {})
                inputs = fetch.get("input")
                if not isinstance(inputs, list) or not inputs:
                    raise PayloadError(f"{variant.id}: supply staging.fetch.input explicitly")
                for item in inputs:
                    location(item, f"{variant.id}.staging.fetch.input", {"s3"})
                for key in ("recipe", "checkpoint"):
                    location(fetch.get(key), f"{variant.id}.staging.fetch.{key}", {"s3"})
        work = build_work_items(payload, run_id=run_id)

    return {
        "dag_id": "defect_image_generation_smt_dag_k8s",
        "pipeline_mode": model.pipeline_mode,
        "output_directory": work["output_directory"],
        "simulation_jobs": len(work["image_generation_environments"]),
        "anomalygen_jobs": len(work["anomaly_package_kwargs"]),
        "goldens": [
            {key: item[key] for key in ("golden_id", "simulation_directory", "simulation_pass")}
            for item in work["golden_work_items"]
        ],
        "defects": [
            {
                "golden_id": item.get("golden_id"),
                "defect_id": item["defect_id"],
                "output_directory": item["output_directory"],
                "per_defect_counts": item["anomaly_generation_config"].get("per_defect_counts"),
                "num_sdg": item["anomaly_generation_config"].get("num_sdg"),
                "staging_fetch": item["anomaly_generation_config"].get("staging", {}).get("fetch"),
            }
            for item in work["anomaly_package_kwargs"]
        ],
        "validation_scope": (
            "Payload schema, override keys, explicit storage/service locations and work-item "
            "compilation only. Components validate their own settings when their tasks start."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument(
        "--payload", required=True, type=Path, help="Plain payload JSON, not conf.payload"
    )
    parser.add_argument(
        "--run-id", default="preview", help="Output-path preview only; does not submit a run"
    )
    args = parser.parse_args()
    try:
        payload = json.loads(args.payload.read_text())
        result = build_plan(payload, args.repo_root, args.run_id)
    except ImportError as exc:
        missing = getattr(exc, "name", None) or "a runtime dependency"
        print(
            f"Missing module {missing!r}. Run with the checkout's Airflow environment: "
            "`uv sync --frozen --all-packages --inexact` in --repo-root, then use its "
            ".venv/bin/python.",
            file=sys.stderr,
        )
        return 2
    except Exception as exc:
        # Pydantic/Airflow exception text may include user input: emit field names only.
        if isinstance(exc, PayloadError):
            message = str(exc)
        elif callable(getattr(exc, "errors", None)):
            fields = [".".join(map(str, item["loc"])) or "payload" for item in exc.errors()]
            message = "Invalid payload fields: " + ", ".join(fields)
        else:
            message = f"Local validation failed ({type(exc).__name__}); check JSON, component overrides and file paths."
        print(message, file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
