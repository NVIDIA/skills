# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path

import pytest
import yaml


SCRIPT = Path(__file__).parents[1] / "prepare_deft_od_aoi_sources.py"
SPEC = importlib.util.spec_from_file_location("prepare_deft_od_aoi_sources", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)
INIT_SCRIPT = Path(__file__).parents[1] / "init_deft_od_aoi.py"
INIT_SPEC = importlib.util.spec_from_file_location("init_deft_od_aoi_for_sources", INIT_SCRIPT)
INIT_MODULE = importlib.util.module_from_spec(INIT_SPEC)
assert INIT_SPEC.loader
INIT_SPEC.loader.exec_module(INIT_MODULE)


def _source(root: Path, name: str, *, boxed: bool, category: int = 7,
            file_name: str | None = None) -> dict:
    directory = root / name / "images"
    directory.mkdir(parents=True)
    image = directory / (file_name or f"{name}.png")
    image.write_bytes(b"image")
    annotations = ([{"id": 9, "image_id": 4, "category_id": category,
                     "bbox": [1, 2, 3, 4], "label": "scratch"}] if boxed else [])
    coco = root / name / "source.json"
    image_row = {"id": 4, "file_name": image.name, "width": 12, "height": 10,
                 "customer_field": name}
    if name == "kpi" and boxed:
        image_row["deft_od_aoi"] = {
            "dataset_id": "route", "texture_id": "texture", "defect_class": "scratch"
        }
    coco.write_text(json.dumps({
        "images": [image_row],
        "annotations": annotations,
        "categories": [{"id": category, "name": "customer-defect"}],
    }))
    return {"coco": str(coco), "images_dir": str(directory)}


def _data_services_merge(inputs: list[Path], work: Path) -> Path:
    """Small test double for Data Services COCOMerger's public behavior."""
    work.mkdir(parents=True, exist_ok=True)
    documents = [json.loads(path.read_text()) for path in inputs]
    assert all(row["categories"] == documents[0]["categories"] for row in documents)
    output = {"images": [], "annotations": [], "categories": documents[0]["categories"]}
    for document in documents:
        image_ids = {}
        for image in document["images"]:
            image_ids[image["id"]] = len(output["images"]) + 1
            output["images"].append({**image, "id": image_ids[image["id"]]})
        for annotation in document["annotations"]:
            output["annotations"].append({
                **annotation, "id": len(output["annotations"]) + 1,
                "image_id": image_ids[annotation["image_id"]],
            })
    merged = work / "output.json"
    merged.write_text(json.dumps(output))
    return merged


def _manifest(root: Path) -> Path:
    path = root / "dataset_sources.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "inputs": {
            "benchmark": [_source(root, "kpi", boxed=True)],
            "test": [_source(root, "test", boxed=False)],
            "mining": [_source(root, "mine", boxed=True)],
            "clean": [_source(root, "clean", boxed=False)],
        },
    }))
    return path


def test_prepares_binary_roles_and_customer_handoff(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    documents, report = MODULE.prepare(manifest)

    assert set(documents) == {"kpi", "test", "real", "clean"}
    assert documents["real"][0]["categories"] == [{"id": 1, "name": "defect"}]
    assert documents["real"][0]["annotations"][0]["category_id"] == 1
    assert documents["real"][0]["images"][0]["customer_field"] == "mine"
    assert documents["kpi"][0]["images"][0]["deft_od_aoi"]["dataset_id"] == "route"
    assert not documents["clean"][0]["annotations"]
    assert report["roles"]["real"] == {"images": 1, "annotations": 1}
    assert report["sources"][0]["input"] == "benchmark"
    assert report["sources"][0]["role"] == "kpi"

    output = tmp_path / "normalized"
    handoff = MODULE.materialize(
        manifest, documents, report, output, "copy", _data_services_merge)
    assert json.loads((output / "sources.json").read_text()) == handoff
    assert set(handoff["sources"]) == {"kpi", "test", "real", "clean"}
    for role, source in handoff["sources"].items():
        coco = json.loads(Path(source["coco"]).read_text())
        image = Path(source["images"]) / coco["images"][0]["file_name"]
        assert image.read_bytes() == b"image", role
        assert Path(coco["images"][0]["source_path"]) == image
        assert Path(coco["images"][0]["original_source_path"]).is_file()

    checkpoint = tmp_path / "base.pth"
    checkpoint.write_bytes(b"checkpoint")
    policy = tmp_path / "policy.yaml"
    policy.write_text(yaml.safe_dump({"platform": "slurm", "max_iterations": 1,
                                      "base_checkpoint": str(checkpoint),
                                      "sources": handoff["sources"]}))
    state = INIT_MODULE.initialize(policy, tmp_path / "contract")
    assert state["roles"]["real"]["annotation_count"] == 1
    assert state["roles"]["clean"]["annotation_count"] == 0


def test_empty_clean_role_is_typed_and_materialized(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    value = json.loads(manifest.read_text())
    value["inputs"]["clean"] = []
    manifest.write_text(json.dumps(value))

    documents, report = MODULE.prepare(manifest)

    role = "clean"
    assert documents[role] == []
    assert report["roles"][role] == {"images": 0, "annotations": 0}
    assert report["capabilities"]["retrieval"][role] == {
        "status": "UNAVAILABLE", "reason": "empty_source_role", "source_image_count": 0}
    assert report["warnings"][0]["code"] == "empty_retrieval_source_role"

    output = tmp_path / "normalized"
    handoff = MODULE.materialize(
        manifest, documents, report, output, "copy", _data_services_merge)
    coco = json.loads(Path(handoff["sources"][role]["coco"]).read_text())
    assert coco == {"images": [], "annotations": [],
                    "categories": [{"id": 1, "name": "defect"}]}
    evidence = json.loads((output / "source_preparation_report.json").read_text())
    assert evidence["merger"]["roles"][role]["status"] == "SKIPPED"

    checkpoint = tmp_path / "base.pth"
    checkpoint.write_bytes(b"checkpoint")
    policy = tmp_path / "policy.yaml"
    policy.write_text(yaml.safe_dump({"platform": "slurm", "max_iterations": 1,
                                      "base_checkpoint": str(checkpoint),
                                      "sources": handoff["sources"]}))
    state = INIT_MODULE.initialize(policy, tmp_path / "contract")
    assert state["roles"][role]["image_count"] == 0
    assert state["capabilities"]["retrieval"][role]["status"] == "UNAVAILABLE"
    assert any(warning["role"] == role for warning in state["warnings"])


def test_rejects_empty_mining_role(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    value = json.loads(manifest.read_text())
    value["inputs"]["mining"] = []
    manifest.write_text(json.dumps(value))

    with pytest.raises(ValueError, match=r"inputs\.mining must be a non-empty array"):
        MODULE.prepare(manifest)


def test_rejects_empty_mining_coco(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    value = json.loads(manifest.read_text())
    coco = Path(value["inputs"]["mining"][0]["coco"])
    document = json.loads(coco.read_text())
    document["images"] = []
    document["annotations"] = []
    coco.write_text(json.dumps(document))

    with pytest.raises(ValueError, match="has no images"):
        MODULE.prepare(manifest)


def test_rejects_internal_kpi_name_at_user_manifest_boundary(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    value = json.loads(manifest.read_text())
    value["inputs"]["kpi"] = value["inputs"].pop("benchmark")
    manifest.write_text(json.dumps(value))

    with pytest.raises(ValueError, match=r"inputs\.benchmark"):
        MODULE.prepare(manifest)


def test_accepts_multiple_coco_shards_in_one_role(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    value = json.loads(manifest.read_text())
    second = _source(tmp_path, "mine_second", boxed=True, category=23,
                     file_name="mine.png")
    value["inputs"]["mining"].append(second)
    manifest.write_text(json.dumps(value))

    documents, report = MODULE.prepare(manifest)

    assert len(documents["real"]) == 2
    assert sum(len(shard["images"]) for shard in documents["real"]) == 2
    assert sum(len(shard["annotations"]) for shard in documents["real"]) == 2
    assert len(report["sources"]) == 5

    output = tmp_path / "normalized"
    handoff = MODULE.materialize(
        manifest, documents, report, output, "copy", _data_services_merge)
    merged = json.loads(Path(handoff["sources"]["real"]["coco"]).read_text())
    assert merged["categories"] == [{"id": 1, "name": "defect"}]
    assert [row["id"] for row in merged["images"]] == [1, 2]
    assert [row["image_id"] for row in merged["annotations"]] == [1, 2]
    assert len({row["file_name"] for row in merged["images"]}) == 2


def test_accepts_coco_list_with_one_shared_image_root(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    value = json.loads(manifest.read_text())
    entry = value["inputs"]["mining"][0]
    images = Path(entry["images_dir"])
    extra_image = images / "extra.png"
    extra_image.write_bytes(b"extra")
    extra_coco = images.parent / "extra.json"
    extra_coco.write_text(json.dumps({
        "images": [{"id": 81, "file_name": extra_image.name,
                    "width": 8, "height": 9}],
        "annotations": [{"id": 91, "image_id": 81, "category_id": 31,
                         "bbox": [1, 1, 2, 2]}],
        "categories": [{"id": 31, "name": "dent"}],
    }))
    entry["coco"] = [entry["coco"], str(extra_coco)]
    manifest.write_text(json.dumps(value))

    documents, report = MODULE.prepare(manifest)

    assert len(documents["real"]) == 2
    assert report["roles"]["real"] == {"images": 2, "annotations": 2}


@pytest.mark.parametrize("bbox", [[10, 2, 3, 4], [1, 8, 3, 4]])
def test_rejects_source_bbox_beyond_image_bounds(tmp_path: Path, bbox: list[int]) -> None:
    manifest = _manifest(tmp_path)
    value = json.loads(manifest.read_text())
    source = Path(value["inputs"]["benchmark"][0]["coco"])
    coco = json.loads(source.read_text())
    coco["annotations"][0]["bbox"] = bbox
    source.write_text(json.dumps(coco))

    with pytest.raises(ValueError, match="exceeds normalized COCO image bounds"):
        MODULE.prepare(manifest)


def test_rejects_boxless_mining_until_it_is_explicitly_routed(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    value = json.loads(manifest.read_text())
    value["inputs"]["mining"] = [_source(tmp_path, "boxless_mine", boxed=False)]
    manifest.write_text(json.dumps(value))

    with pytest.raises(ValueError, match="only boxed images"):
        MODULE.prepare(manifest)


def test_rejects_annotations_in_verified_clean_source(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    value = json.loads(manifest.read_text())
    value["inputs"]["clean"] = [_source(tmp_path, "boxed_clean", boxed=True)]
    manifest.write_text(json.dumps(value))

    with pytest.raises(ValueError, match="zero annotations"):
        MODULE.prepare(manifest)


def test_rejects_cross_role_image_overlap(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    value = json.loads(manifest.read_text())
    value["inputs"]["test"] = value["inputs"]["benchmark"]
    manifest.write_text(json.dumps(value))

    with pytest.raises(ValueError, match="overlaps kpi and test"):
        MODULE.prepare(manifest)


def test_refuses_to_overwrite_materialized_output(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    documents, report = MODULE.prepare(manifest)
    output = tmp_path / "normalized"
    output.mkdir()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        MODULE.materialize(manifest, documents, report, output, "symlink")


def test_data_services_command_uses_generated_merge_spec(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"images": [], "annotations": [],
                                  "categories": [{"id": 1, "name": "defect"}]}))

    def run(command: list[str], *, check: bool) -> None:
        assert command[:3] == ["annotations", "merge", "-e"]
        assert check is True
        spec = json.loads(Path(command[3]).read_text())
        assert spec["data"]["annotations"] == [str(source)]
        _data_services_merge([source], Path(spec["results_dir"]))

    monkeypatch.setattr(MODULE.subprocess, "run", run)
    assert MODULE._merge_with_data_services([source], tmp_path / "action").is_file()


def test_rejects_data_services_output_count_drift(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    documents, report = MODULE.prepare(manifest)

    def merge_with_loss(inputs: list[Path], work: Path) -> Path:
        merged = _data_services_merge(inputs, work)
        value = json.loads(merged.read_text())
        value["images"] = []
        merged.write_text(json.dumps(value))
        return merged

    with pytest.raises(ValueError, match="changed the expected kpi counts"):
        MODULE.materialize(
            manifest, documents, report, tmp_path / "normalized", "copy", merge_with_loss)
