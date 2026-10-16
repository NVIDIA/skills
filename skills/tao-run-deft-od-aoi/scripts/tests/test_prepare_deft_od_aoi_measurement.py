# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path

import pytest
import yaml


SCRIPT = Path(__file__).parents[1] / "prepare_deft_od_aoi_measurement.py"
SPEC = importlib.util.spec_from_file_location("prepare_deft_od_aoi_measurement", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_measurement_freezes_binary_inference_and_dual_gap_specs(tmp_path: Path) -> None:
    sources = {}
    for role in ("kpi", "test"):
        images = tmp_path / role
        images.mkdir()
        (images / f"{role}.png").write_bytes(b"image")
        coco = tmp_path / f"{role}.json"
        coco.write_text(json.dumps({"images": [{"id": 1, "file_name": f"{role}.png",
                                                 "width": 10, "height": 10}],
                                    "annotations": [{"id": 1, "image_id": 1,
                                                     "category_id": 1, "bbox": [1, 2, 3, 4]}],
                                    "categories": [{"id": 1, "name": "defect"}]}))
        sources[role] = {"images": str(images), "coco": str(coco)}
    policy = tmp_path / "policy.yaml"
    policy.write_text(yaml.safe_dump({"sources": sources, "baseline_mode": "checkpoint",
                                      "gap": {"inference_confidence": 0.001,
                                              "loose_confidence": 0.3,
                                              "strict_confidence": 0.8,
                                              "match_iou": 0.5}}))
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"model")
    report = MODULE.prepare(policy, checkpoint, tmp_path / "measure/kpi/inference/labels",
                            tmp_path / "measure", tmp_path / "specs", baseline=True)
    assert report["status"] == "COMPLETE"
    assert report["checkpoint_sha256"] == MODULE._sha(checkpoint)
    assert report["inference_roles"] == {
        "kpi": {"expected_images": 1,
                "predictions": str((tmp_path / "measure/kpi/inference/labels").resolve())},
        "test": {"expected_images": 1,
                 "predictions": str((tmp_path / "measure/test/inference/labels").resolve())},
    }
    inference = yaml.safe_load((tmp_path / "specs/kpi_inference.yaml").read_text())
    assert inference["dataset"]["num_classes"] == 2
    assert Path(inference["dataset"]["infer_data_sources"]["classmap"]).read_text() == (
        "background\ndefect\n"
    )
    loose = yaml.safe_load((tmp_path / "specs/gap_loose.yaml").read_text())
    strict = yaml.safe_load((tmp_path / "specs/gap_strict.yaml").read_text())
    assert loose["conf_threshold"] == 0.3 and strict["conf_threshold"] == 0.8
    label = tmp_path / "specs/kpi_ground_truth_kitti/kpi.txt"
    assert label.read_text().startswith("defect 0.0 0 0.0 1.000000 2.000000 4.000000 6.000000")


def test_kitti_rejects_bbox_outside_image(tmp_path: Path) -> None:
    coco = tmp_path / "kpi.json"
    coco.write_text(json.dumps({
        "images": [{"id": 1, "file_name": "kpi.png", "width": 10, "height": 8}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                         "bbox": [8, 2, 3, 4]}],
        "categories": [{"id": 1, "name": "defect"}],
    }))

    with pytest.raises(ValueError, match="normalized COCO bbox exceeds image bounds"):
        MODULE._kitti(coco, tmp_path / "labels")


def test_cold_start_baseline_emits_empty_kpi_predictions_without_inference(
    tmp_path: Path,
) -> None:
    sources = {}
    for role in ("kpi", "test"):
        images = tmp_path / role
        images.mkdir()
        (images / f"{role}.png").write_bytes(b"image")
        coco = tmp_path / f"{role}.json"
        coco.write_text(json.dumps({
            "images": [{"id": 1, "file_name": f"{role}.png", "width": 10, "height": 10}],
            "annotations": ([{"id": 1, "image_id": 1, "category_id": 1,
                              "bbox": [1, 2, 3, 4]}] if role == "kpi" else []),
            "categories": [{"id": 1, "name": "defect"}],
        }))
        sources[role] = {"images": str(images), "coco": str(coco)}
    policy = tmp_path / "policy.yaml"
    policy.write_text(yaml.safe_dump({
        "sources": sources, "baseline_mode": "cold_start",
        "gap": {"inference_confidence": 0.001, "loose_confidence": 0.3,
                "strict_confidence": 0.8, "match_iou": 0.5},
    }))
    checkpoint = tmp_path / "warehouse.pth"
    checkpoint.write_bytes(b"seven-class-model")

    report = MODULE.prepare(
        policy, checkpoint, tmp_path / "unused/labels", tmp_path / "measure",
        tmp_path / "specs", baseline=True,
    )

    assert report["cold_start"] is True
    assert report["checkpoint_sha256"] == MODULE._sha(checkpoint)
    assert set(report["specs"]) == {"gap_loose.yaml", "gap_strict.yaml"}
    assert not (tmp_path / "specs/kpi_inference.yaml").exists()
    assert not (tmp_path / "specs/test_inference.yaml").exists()
    assert report["inference_roles"] == {
        "kpi": {
            "expected_images": 1,
            "predictions": str((tmp_path / "specs/cold_start_predictions/kpi").resolve()),
        },
        "test": {
            "expected_images": 1,
            "predictions": str((tmp_path / "specs/cold_start_predictions/test").resolve()),
        },
    }
    for role in ("kpi", "test"):
        assert (tmp_path / f"specs/cold_start_predictions/{role}/{role}.txt").read_text() == ""
    strict = yaml.safe_load((tmp_path / "specs/gap_strict.yaml").read_text())
    assert strict["inference_ann_path"] == str(tmp_path / "specs/cold_start_predictions/kpi")

    later = MODULE.prepare(
        policy, checkpoint, tmp_path / "later/kpi/inference/labels", tmp_path / "later",
        tmp_path / "later_specs",
    )
    assert later["cold_start"] is False
    assert (tmp_path / "later_specs/kpi_inference.yaml").is_file()
    assert (tmp_path / "later_specs/test_inference.yaml").is_file()
    assert later["checkpoint_sha256"] == MODULE._sha(checkpoint)
    assert set(later["inference_roles"]) == {"kpi", "test"}
