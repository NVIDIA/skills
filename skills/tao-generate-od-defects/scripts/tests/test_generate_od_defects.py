# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import jsonschema
import pytest
import yaml


SCRIPT = Path(__file__).parents[1] / "generate_od_defects.py"
SPEC = importlib.util.spec_from_file_location("generate_od_defects", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def _inputs(root: Path) -> argparse.Namespace:
    clean, mask = root / "clean.png", root / "mask.png"
    clean.write_bytes(b"image")
    mask.write_bytes(b"mask")
    testcase = root / "testcase.jsonl"
    testcase.write_text(json.dumps({"image_filename": str(clean), "mask_filename": str(mask),
                                    "anomaly_type": "texture+defect"}) + "\n")
    checkpoint, recipe = root / "adapter.pt", root / "recipe.yaml"
    checkpoint.write_bytes(b"adapter")
    recipe.write_text(yaml.safe_dump({"anomaly_types": [["texture", "defect"]]}))
    return argparse.Namespace(inputs_dir=None, input_data_path=testcase, checkpoint=checkpoint,
                              recipe=recipe, anomaly_types="", dataset_id="default", datasets=None)


def _checkpoint_root(repo: Path) -> Path:
    root = repo / "checkpoints"
    for name in MODULE.OFFLINE_HF_REPOS:
        directory = root / "hf/hub" / f"models--{name.replace('/', '--')}"
        (directory / "blobs").mkdir(parents=True)
        (directory / "snapshots").mkdir()
    dinov2 = root / "facebook/dinov2-large"
    dinov2.mkdir(parents=True)
    (dinov2 / "config.json").write_text("{}\n")
    (dinov2 / "model.safetensors").write_bytes(b"weights")
    return root


def test_native_contract_infers_and_validates_types(tmp_path: Path) -> None:
    selected = MODULE.groups(_inputs(tmp_path))
    assert selected[0]["anomaly_types"] == ["texture+defect"]
    assert selected[0]["requested_rows"] == 1


def test_contract_rejects_type_absent_from_recipe(tmp_path: Path) -> None:
    args = _inputs(tmp_path)
    args.recipe.write_text(yaml.safe_dump({"anomaly_types": [["texture", "other"]]}))
    with pytest.raises(ValueError, match="absent from the recipe"):
        MODULE.groups(args)


@pytest.mark.parametrize("guardrail_enabled", [True, False])
def test_native_logs_are_kept_off_machine_readable_stdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    guardrail_enabled: bool,
) -> None:
    calls = []

    def fake_run(command, *, check, stdout):
        assert check is True and stdout is sys.stderr
        calls.append(command)
        if str(command[1]).endswith("pseudo_label.py"):
            labels = tmp_path / "out/pseudo_labels"
            labels.mkdir(parents=True)
            (labels / "coco_annotations.json").write_text(json.dumps({
                "images": [], "annotations": [], "categories": [],
            }))

    monkeypatch.setattr(MODULE.subprocess, "run", fake_run)
    group = {
        "dataset_id": "d", "checkpoint": "adapter.pt", "recipe": "recipe.yaml",
        "testcase": "testcase.jsonl", "anomaly_types": ["texture+defect"],
        "requested_rows": 0,
    }
    args = argparse.Namespace(repo=tmp_path, num_gpus=1, base_checkpoint=tmp_path,
                              guardrail=guardrail_enabled)
    result = MODULE._run_group(group, tmp_path / "out", args)

    assert len(calls) == 2
    if guardrail_enabled:
        assert "--guardrail" not in calls[0] and "--no-guardrail" not in calls[0]
    else:
        assert "--no-guardrail" in calls[0]
    assert "--guardrail" not in calls[1] and "--no-guardrail" not in calls[1]
    assert result["generated"] == result["blocked"] == 0


def test_disabled_guardrails_reject_blocked_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(command, *, check, stdout):
        if str(command[1]).endswith("pseudo_label.py"):
            labels = tmp_path / "out/pseudo_labels"
            labels.mkdir(parents=True)
            (labels / "coco_annotations.json").write_text(json.dumps({
                "images": [], "annotations": [], "categories": [],
            }))

    monkeypatch.setattr(MODULE.subprocess, "run", fake_run)
    monkeypatch.setattr(
        MODULE, "_csv_count",
        lambda path: 1 if path.name == "guardrail_blocked.csv" else 0,
    )
    group = {
        "dataset_id": "d", "checkpoint": "adapter.pt", "recipe": "recipe.yaml",
        "testcase": "testcase.jsonl", "anomaly_types": ["texture+defect"],
        "requested_rows": 1,
    }
    args = argparse.Namespace(repo=tmp_path, num_gpus=1, base_checkpoint=tmp_path,
                              guardrail=False)

    with pytest.raises(ValueError, match="guardrails are disabled"):
        MODULE._run_group(group, tmp_path / "out", args)


def test_generate_action_contract_exposes_guardrail_mode() -> None:
    action = yaml.safe_load(
        (SCRIPT.parents[1] / "references/skill_info.yaml").read_text()
    )["actions"]["generate"]

    assert action["inputs"]["guardrail"] == {"type": "bool"}
    assert action["args"]["guardrail"] == "--guardrail/--no-guardrail {guardrail}"
    assert action["defaults"]["guardrail"] is True


def test_generation_metadata_uses_persistent_output_paths(tmp_path: Path) -> None:
    output = tmp_path / "temporary"
    output.mkdir()
    metadata = output / "report.json"
    metadata.write_text(json.dumps({"image": str(output / "generated.png")}))
    published = tmp_path / "persistent"

    MODULE._publish_paths(output, published)

    assert json.loads(metadata.read_text())["image"] == str(published / "generated.png")


def test_offline_cache_requires_all_pinned_repositories(tmp_path: Path) -> None:
    assert set(MODULE.CORE_OFFLINE_HF_REPOS) == {
        "Qwen/Qwen3-VL-8B-Instruct",
        "nvidia/Cosmos3-Edge",
    }
    assert set(MODULE.GUARDRAIL_OFFLINE_HF_REPOS) == {
        "Qwen/Qwen3Guard-Gen-0.6B",
        "nvidia/Cosmos-Guardrail1",
    }
    for repo in MODULE.CORE_OFFLINE_HF_REPOS:
        directory = tmp_path / "hub" / f"models--{repo.replace('/', '--')}"
        (directory / "blobs").mkdir(parents=True)
        (directory / "snapshots").mkdir()
    MODULE._validate_offline_hf_cache(tmp_path, guardrail_enabled=False)
    with pytest.raises(FileNotFoundError, match="Qwen3Guard-Gen-0.6B"):
        MODULE._validate_offline_hf_cache(tmp_path, guardrail_enabled=True)
    for repo in MODULE.GUARDRAIL_OFFLINE_HF_REPOS:
        directory = tmp_path / "hub" / f"models--{repo.replace('/', '--')}"
        (directory / "blobs").mkdir(parents=True)
        (directory / "snapshots").mkdir()
    MODULE._validate_offline_hf_cache(tmp_path, guardrail_enabled=True)
    (tmp_path / "hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots").rmdir()
    with pytest.raises(FileNotFoundError, match="Qwen3-VL-8B-Instruct"):
        MODULE._validate_offline_hf_cache(tmp_path, guardrail_enabled=False)


def test_checkpoint_root_requires_canonical_mount_and_dinov2(tmp_path: Path) -> None:
    external = _checkpoint_root(tmp_path / "external")
    with pytest.raises(ValueError, match="must be mounted"):
        MODULE._validate_checkpoint_root(external, tmp_path / "repo")

    repo = tmp_path / "repo"
    root = _checkpoint_root(repo)
    assert MODULE._validate_checkpoint_root(root, repo) == root / "hf"
    for name in MODULE.GUARDRAIL_OFFLINE_HF_REPOS:
        snapshots = (
            root / "hf/hub" / f"models--{name.replace('/', '--')}" / "snapshots"
        )
        snapshots.rmdir()
    assert MODULE._validate_checkpoint_root(
        root, repo, guardrail_enabled=False
    ) == root / "hf"
    with pytest.raises(FileNotFoundError, match="Qwen3Guard-Gen-0.6B"):
        MODULE._validate_checkpoint_root(root, repo, guardrail_enabled=True)
    (root / "facebook/dinov2-large/model.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="DINOv2 weights"):
        MODULE._validate_checkpoint_root(root, repo, guardrail_enabled=False)


def test_generate_contract_mounts_complete_checkpoint_root() -> None:
    contract = yaml.safe_load(
        (SCRIPT.parents[1] / "references/skill_info.yaml").read_text()
    )["actions"]["generate"]
    inputs = contract["inputs"]
    assert inputs["checkpoint_root"]["container_path"] == (
        "/workspace/paidf-anomalygen/checkpoints"
    )
    assert "hf_cache" not in inputs
    assert contract["args"]["checkpoint_root"] == "--checkpoint-root {checkpoint_root}"


def test_base_checkpoint_requires_parent_of_model_directory(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (tmp_path / "checkpoint.json").write_text("{}\n")
    (model / ".metadata").write_bytes(b"metadata")
    (model / "__0_0.distcp").write_bytes(b"shard")
    MODULE._validate_base_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="parent"):
        MODULE._validate_base_checkpoint(model)


def test_merge_validates_boxes_and_writes_binary_projection(tmp_path: Path) -> None:
    image = tmp_path / "generated.png"
    image.write_bytes(b"image")
    result = {"group": {"dataset_id": "d", "requested_rows": 1,
                        "anomaly_types": ["texture+defect"]},
              "generated": 1, "blocked": 0, "image_root": tmp_path,
              "coco": {"images": [{"id": 7, "file_name": str(image),
                                      "width": 20, "height": 10}],
                       "annotations": [{"id": 9, "image_id": 7, "category_id": 4,
                                        "bbox": [1, 2, 5, 6], "area": 30}],
                       "categories": [{"id": 4, "name": "texture+defect"}]}}
    report = MODULE._merge([result], tmp_path / "out", guardrail_enabled=True)
    assert report["status"] == "COMPLETE" and report["training_pool_mutated"] is False
    assert report["guardrail_enabled"] is True
    binary = json.loads((tmp_path / "out/pseudo_labels/coco_annotations_od_defect.json").read_text())
    assert binary["categories"] == [{"id": 1, "name": "defect"}]
    assert binary["annotations"][0]["category_id"] == 1
    info = yaml.safe_load((Path(__file__).parents[2] / "references/skill_info.yaml").read_text())
    outputs = info["actions"]["generate"]["outputs"]
    assert outputs["native_coco"]["relative_path"] == "pseudo_labels/coco_annotations.json"
    assert outputs["binary_coco"]["relative_path"] == (
        "pseudo_labels/coco_annotations_od_defect.json"
    )
    for name in ("native_coco", "binary_coco"):
        assert (tmp_path / "out" / outputs[name]["relative_path"]).is_file()
    schema = json.loads((Path(__file__).parents[4]
                         / "core/tao-artifacts/references/spec_bundle.schema.json").read_text())
    jsonschema.validate({
        "network_arch": "tao-generate-od-defects", "action": "generate",
        "image": "nvcr.io/nvidia/paidf-anomalygen:1.1.0", "mode": "args",
        "command": "scripts/generate_od_defects.py", "args": [],
        "declared_inputs": [],
        "declared_outputs": [{"spec_key": name, **contract}
                             for name, contract in outputs.items()],
        "compute_shape": {"gpus": 1, "nodes": 1},
    }, schema)
