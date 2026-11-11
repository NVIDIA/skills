# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path

import yaml


SCRIPT = Path(__file__).parents[1] / "prepare_deft_od_aoi_training.py"
SPEC = importlib.util.spec_from_file_location("prepare_deft_od_aoi_training", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def _fixture(root: Path, iteration: int, size: int) -> tuple[Path, Path, Path]:
    policy = yaml.safe_load((SCRIPT.parents[1] / "assets/default_policy.yaml").read_text())
    checkpoint = root / "base.pth"
    checkpoint.write_bytes(b"model")
    kpi_images = root / "kpi"
    kpi_images.mkdir()
    kpi_coco = root / "kpi.json"
    kpi_coco.write_text("{}")
    policy.update(max_iterations=5, base_checkpoint=str(checkpoint))
    policy["sources"]["kpi"] = {"images": str(kpi_images), "coco": str(kpi_coco)}
    policy_path = root / "policy.yaml"
    policy_path.write_text(yaml.safe_dump(policy))
    train_images = root / "images"
    train_images.mkdir()
    train_coco = root / "train.json"
    train_coco.write_text(json.dumps({"images": [{"id": value} for value in range(size)]}))
    return policy_path, train_coco, train_images


def test_early_iteration_emits_direct_frozen_base_spec(tmp_path: Path) -> None:
    policy, coco, images = _fixture(tmp_path, 1, 20)
    report = MODULE.prepare(policy, 1, coco, images, tmp_path / "runs",
                            tmp_path / "specs", None, None)
    assert report["planned_epochs"] == 36 and report["probes_enabled"] is False
    spec = yaml.safe_load((tmp_path / "specs/train.yaml").read_text())
    assert spec["train"]["num_epochs"] == 36
    assert spec["train"]["pretrained_model_path"].endswith("base.pth")
    assert spec["dataset"]["num_classes"] == 2


def test_configured_first_iteration_emits_all_probe_specs(tmp_path: Path) -> None:
    policy, coco, images = _fixture(tmp_path, 1, 20)
    value = yaml.safe_load(policy.read_text())
    value["training"]["probes_start_iteration"] = 1
    policy.write_text(yaml.safe_dump(value))

    report = MODULE.prepare(
        policy, 1, coco, images, tmp_path / "runs", tmp_path / "specs", None, None
    )

    assert report["probes_enabled"] is True
    assert len(report["probes"]) == 3
    assert all((tmp_path / f"specs/probe{index}.yaml").is_file() for index in range(3))


def test_later_iteration_emits_three_deterministic_probe_specs(tmp_path: Path) -> None:
    policy, coco, images = _fixture(tmp_path, 3, 10)
    history = tmp_path / "history.json"
    history.write_text(json.dumps([{"train_size": 5}]))
    report = MODULE.prepare(policy, 3, coco, images, tmp_path / "runs",
                            tmp_path / "specs", history, None)
    assert report["probes_enabled"] is True and len(report["probes"]) == 3
    assert report["planned_epochs"] == 48
    assert not (tmp_path / "specs/train.yaml").exists()
    specs = [yaml.safe_load((tmp_path / f"specs/probe{index}.yaml").read_text())
             for index in range(3)]
    assert {spec["results_dir"] for spec in specs} == {
        str(tmp_path / f"runs/probes/p{index}") for index in range(3)
    }
    assert all(spec["train"]["num_epochs"] == 10 for spec in specs)
