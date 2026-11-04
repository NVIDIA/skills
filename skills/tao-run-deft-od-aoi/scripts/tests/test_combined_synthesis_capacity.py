# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from PIL import Image


def _load(name):
    spec = importlib.util.spec_from_file_location(
        "combined_" + name, Path(__file__).with_name(name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


ADMISSION = _load("test_admit_deft_od_aoi_coco")
PLANNING = _load("test_prepare_deft_od_aoi_synthesis")


def _admit(root: Path, basis: str, prior_synthetic: int):
    policy, candidates, retrieval = ADMISSION._fixture(root)
    value = yaml.safe_load(policy.read_text())
    synthesis = {"enabled": True}
    if basis in {"total", "both"}:
        synthesis["cumulative_fraction_of_total_defects"] = 0.25
    if basis in {"real", "both"}:
        synthesis["cumulative_fraction_of_real_defects"] = 0.25 if basis == "real" else 9
    value["synthesis"] = synthesis
    policy.write_text(yaml.safe_dump(value))
    for role in ("real", "clean"):
        pd.DataFrame({"filepath": []}).to_parquet(
            retrieval / f"mine_{role}/final_unique_files.parquet", index=False
        )
    images, annotations = [], []
    # Clean negatives must not enlarge the real-plus-synthetic defect cap.
    kinds = ["real_defect"] * 3 + ["clean_negative"] * 5 + ["synthetic_defect"] * prior_synthetic
    for image_id, kind in enumerate(kinds, 1):
        image = root / f"prior-{image_id}.png"
        Image.new("RGB", (16, 16)).save(image)
        images.append({"id": image_id, "file_name": image.name, "source_path": str(image),
                       "width": 16, "height": 16, "deft_kind": kind})
        if kind != "clean_negative":
            annotations.append({"id": image_id, "image_id": image_id, "category_id": 1,
                                "bbox": [1, 1, 4, 4]})
    previous = root / "previous.json"
    previous.write_text(json.dumps({"images": images, "annotations": annotations,
                                    "categories": [{"id": 1, "name": "defect"}]}))
    output = root / "admitted"
    report = ADMISSION.MODULE.admit(policy, candidates, retrieval, output, previous, "copy")
    state = root / "state.json"
    state.write_text(json.dumps({"status": "RUNNING", "next_stage": "iteration_admission",
                                 "current_iteration": 1, "synthesis_enabled": True}))
    result = ADMISSION.COMMIT_MODULE.commit(state, "iteration_admission", 1, [
        f"admission_report={output / 'admission_report.json'}",
    ])
    return output, state, report, result


@pytest.mark.parametrize("basis", ["real", "total", "both"])
@pytest.mark.parametrize("prior_synthetic", [0, 1])
def test_initial_capacity_uses_configured_cap_before_generation(
        tmp_path: Path, basis: str, prior_synthetic: int) -> None:
    _, _, report, result = _admit(tmp_path, basis, prior_synthetic)
    limit = 0 if basis == "real" else 1
    room = max(0, limit - prior_synthetic)
    evidence = report["synthetic_admission"]
    assert evidence["fraction_basis"] == ("fraction_of_real" if basis == "real" else "fraction_of_total")
    assert evidence["configured_fraction"] == 0.25
    assert evidence["cumulative_real_images"] == 3
    assert evidence["cumulative_limit"] == evidence["cumulative_synthetic_limit"] == limit
    assert evidence["available_room_before_admission"] == room
    assert report["new_training_images"] == 0
    assert result["next_stage"] == ("iteration_synthesis" if room else None)
    if not room:
        assert result["events"][-1]["synthesis_decision"]["reason"] == "no_synthetic_budget"


def test_actual_default_planner_budget_skip_uses_initial_admission(tmp_path: Path) -> None:
    output, state, _, _ = _admit(tmp_path, "total", 0)
    fixture = tmp_path / "planner"
    fixture.mkdir()
    PLANNING.test_runtime_generated_plan_selects_deterministically_and_is_reported(fixture)
    request_root = fixture / "one-slot"
    skipped = PLANNING.MODULE.prepare(
        fixture / "policy.yaml", fixture / "strict.parquet", request_root,
        iteration=1, real_coco=output / "train.json",
    )
    assert skipped["status"] == "SKIPPED" and skipped["reason"] == "no_synthetic_budget"
    assert skipped["planning"]["new_image_budget"] == 1
    assert skipped["planning"]["images_per_fn"] == 2
    assert not (request_root / "anomalygen_filtering.yaml").exists()
    result = ADMISSION.COMMIT_MODULE.commit(state, "iteration_synthesis", 1, [
        f"synthesis_request={request_root / 'synthesis_request.json'}",
        f"admission_report={output / 'admission_report.json'}",
    ])
    assert result["status"] == "COMPLETE" and result["next_stage"] is None
    assert result["events"][-1]["synthesis_decision"]["reason"] == "no_synthetic_budget"
