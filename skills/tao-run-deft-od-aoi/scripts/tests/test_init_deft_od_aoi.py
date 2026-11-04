# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path

import pytest
import yaml


SCRIPT = Path(__file__).parents[1] / "init_deft_od_aoi.py"
SPEC = importlib.util.spec_from_file_location("init_deft_od_aoi", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def _role(root: Path, name: str, boxed: bool) -> dict:
    images = root / name / "images"
    images.mkdir(parents=True)
    image = images / f"{name}.png"
    image.write_bytes(b"image")
    annotations = ([{"id": 1, "image_id": 1, "category_id": 1,
                     "bbox": [1, 1, 4, 4], "area": 16}] if boxed else [])
    image_row = {"id": 1, "file_name": image.name, "width": 10, "height": 10}
    if name == "kpi" and boxed:
        image_row["deft_od_aoi"] = {
            "dataset_id": "route", "texture_id": "texture", "defect_class": "defect"
        }
    coco = root / name / "coco.json"
    coco.write_text(json.dumps({"images": [image_row],
                                "annotations": annotations,
                                "categories": [{"id": 1, "name": "defect"}]}))
    return {"images": str(images), "coco": str(coco)}


def _config(root: Path) -> Path:
    checkpoint = root / "base.pth"
    checkpoint.write_bytes(b"model")
    path = root / "policy.yaml"
    path.write_text(yaml.safe_dump({"platform": "slurm", "max_iterations": 2,
                                    "base_checkpoint": str(checkpoint),
                                    "sources": {name: _role(root, name, name != "clean")
                                                for name in MODULE.ROLES}}))
    return path


def _enable_synthesis(root: Path, config: Path) -> dict:
    value = yaml.safe_load(config.read_text())
    pool = root / "pool"
    pool.mkdir()
    clean = pool / "texture/clean_image/clean.png"
    clean.parent.mkdir(parents=True)
    clean.write_bytes(b"clean")
    defect, checkpoint, recipe = (
        root / "defect.jsonl", root / "adapter.pt", root / "recipe.yaml"
    )
    defect.write_text("{}\n")
    checkpoint.write_bytes(b"adapter")
    recipe.write_text("anomaly_types: []\n")
    value["synthesis"] = {
        "enabled": True,
        "pool_dataset_root": str(pool),
        "defect_spec": str(defect),
        "routes": {"route": {"checkpoint": str(checkpoint), "recipe": str(recipe)}},
    }
    return value


def _append_boxless_image(role: dict, name: str) -> None:
    images = Path(role["images"])
    image = images / f"{name}_boxless.png"
    image.write_bytes(b"boxless-image")
    coco = Path(role["coco"])
    data = json.loads(coco.read_text())
    row = {"id": 2, "file_name": image.name, "width": 10, "height": 10}
    if name == "kpi":
        row.update({"dataset_id": "line-a", "texture_id": "board",
                    "defect_class": "bridge"})
    data["images"].append(row)
    coco.write_text(json.dumps(data))


def _set_kpi_dataset_id(config: dict, dataset_id: str, *, nested: bool = False) -> None:
    coco = Path(config["sources"]["kpi"]["coco"])
    data = json.loads(coco.read_text())
    target = data["annotations"][0]
    if nested:
        target["deft_od_aoi"] = {"dataset_id": dataset_id}
    else:
        target["dataset_id"] = dataset_id
    coco.write_text(json.dumps(data))


def _enable_synthesis(root: Path, config: dict) -> None:
    pool, dataset, base, checkpoints = (
        root / "pool", root / "ft_dataset", root / "ft_base", root / "checkpoints"
    )
    for path in (pool, dataset, base, checkpoints):
        path.mkdir()
    clean = pool / "texture/clean_image/clean.png"
    clean.parent.mkdir(parents=True)
    clean.write_bytes(b"clean")
    defect = root / "defect.jsonl"
    validation = root / "validation.jsonl"
    vae = root / "vae.pth"
    defect.write_text("{}\n")
    validation.write_text("{}\n")
    vae.write_bytes(b"vae")
    config["synthesis"] = {
        "enabled": True,
        "pool_dataset_root": str(pool),
        "defect_spec": str(defect),
        "routes": {"route": {"finetune": {
            "dataset_root": str(dataset),
            "validation_testcase": str(validation),
            "base_checkpoint": str(base),
            "vae_path": str(vae),
            "checkpoint_root": str(checkpoints),
            "result_handoff": str(root / "future/handoff.json"),
        }}},
    }


def _set_kpi_synthesis_metadata(config: dict, **metadata: str) -> None:
    coco = Path(config["sources"]["kpi"]["coco"])
    data = json.loads(coco.read_text())
    data["annotations"][0]["deft_od_aoi"] = metadata
    coco.write_text(json.dumps(data))


def _set_valid_routed_metadata(root: Path, config: dict,
                               mask_contents: bytes = b"mask") -> Path:
    mask = root / "mask.png"
    mask.write_bytes(mask_contents)
    _set_kpi_synthesis_metadata(
        config, dataset_id="route", texture_id="texture", defect_class="scratch",
        fn_mask_source=str(mask)
    )
    return mask


def test_initialize_freezes_real_only_disjoint_contract(tmp_path: Path) -> None:
    state = MODULE.initialize(_config(tmp_path), tmp_path / "results")
    assert state["mode"] == "rtdetr_real_only"
    assert state["baseline_mode"] == "cold_start"
    assert state["next_stage"] == "candidate_cache"
    assert Path(state["policy"]).is_file()
    assert Path(state["classmap"]).read_text() == "background\ndefect\n"
    assert state["roles"]["clean"]["annotation_count"] == 0
    policy = yaml.safe_load(Path(state["policy"]).read_text())
    assert policy["retrieval"]["preprocessing"]["profile"] == "square_context"
    assert policy["retrieval"]["selection"]["strategy"] == "round_robin_similarity"
    assert policy["retrieval"]["output_size"] == 224
    assert policy["retrieval"]["candidate_overfetch"] == 15
    assert policy["retrieval"]["round_robin_refill_overfetch"] == [
        5, 15, 50, 200, 100000,
    ]
    assert policy["synthesis"]["candidate_topn"] == 3
    assert policy["synthesis"]["max_neighbors_per_fn"] == 1
    assert policy["routing"]["round_robin_real_factor_default"] == 3
    assert policy["routing"]["near_miss_real_cap_per_pocket"] == 20
    assert "near_miss_real_cap" not in policy["routing"]


def test_initialize_accepts_tight_context_preprocessing(tmp_path: Path) -> None:
    config = _config(tmp_path)
    value = yaml.safe_load(config.read_text())
    value["retrieval"] = {
        "preprocessing": {"profile": "tight_context"},
        "output_size": "unused-by-tight-context",
    }
    config.write_text(yaml.safe_dump(value))

    state = MODULE.initialize(config, tmp_path / "results")

    policy = yaml.safe_load(Path(state["policy"]).read_text())
    assert policy["retrieval"]["preprocessing"]["profile"] == "tight_context"


def test_initialize_rejects_invalid_square_output_size(tmp_path: Path) -> None:
    config = _config(tmp_path)
    value = yaml.safe_load(config.read_text())
    value["retrieval"] = {
        "preprocessing": {"profile": "square_context"}, "output_size": 0,
    }
    config.write_text(yaml.safe_dump(value))

    with pytest.raises(ValueError, match="output_size must be positive"):
        MODULE.initialize(config, tmp_path / "results")


def test_initialize_rejects_unknown_preprocessing_profile(tmp_path: Path) -> None:
    config = _config(tmp_path)
    value = yaml.safe_load(config.read_text())
    value["retrieval"] = {"preprocessing": {"profile": "unsupported_profile"}}
    config.write_text(yaml.safe_dump(value))

    with pytest.raises(ValueError, match="tight_context or square_context"):
        MODULE.initialize(config, tmp_path / "results")


def test_initialize_accepts_max_similarity_selection(tmp_path: Path) -> None:
    config = _config(tmp_path)
    value = yaml.safe_load(config.read_text())
    value["retrieval"] = {"selection": {"strategy": "max_similarity"}}
    config.write_text(yaml.safe_dump(value))

    state = MODULE.initialize(config, tmp_path / "results")

    policy = yaml.safe_load(Path(state["policy"]).read_text())
    assert policy["retrieval"]["selection"]["strategy"] == "max_similarity"


def test_initialize_rejects_unknown_selection_strategy(tmp_path: Path) -> None:
    config = _config(tmp_path)
    value = yaml.safe_load(config.read_text())
    value["retrieval"] = {"selection": {"strategy": "nearest"}}
    config.write_text(yaml.safe_dump(value))

    with pytest.raises(ValueError, match="unsupported retrieval selection strategy"):
        MODULE.initialize(config, tmp_path / "results")


def test_initialize_rejects_round_robin_default_outside_factor_bounds(
        tmp_path: Path) -> None:
    config = _config(tmp_path)
    value = yaml.safe_load(config.read_text())
    value["routing"] = {"round_robin_real_factor_default": 7}
    config.write_text(yaml.safe_dump(value))

    with pytest.raises(ValueError, match="default must be within the frozen bounds"):
        MODULE.initialize(config, tmp_path / "results")


def test_initialize_rejects_obsolete_near_miss_cap_name(tmp_path: Path) -> None:
    config = _config(tmp_path)
    value = yaml.safe_load(config.read_text())
    value["routing"] = {"near_miss_real_cap": 20}
    config.write_text(yaml.safe_dump(value))

    with pytest.raises(ValueError, match="near_miss_real_cap_per_pocket"):
        MODULE.initialize(config, tmp_path / "results")


def test_initialize_accepts_explicit_checkpoint_baseline(tmp_path: Path) -> None:
    config = _config(tmp_path)
    value = yaml.safe_load(config.read_text())
    value["baseline_mode"] = "checkpoint"
    config.write_text(yaml.safe_dump(value))
    state = MODULE.initialize(config, tmp_path / "results")
    assert state["baseline_mode"] == "checkpoint"


def test_initialize_preserves_legacy_synthetic_fraction_override(tmp_path: Path) -> None:
    config = _config(tmp_path)
    value = yaml.safe_load(config.read_text())
    _enable_synthesis(tmp_path, value)
    _set_valid_routed_metadata(tmp_path, value)
    value["synthesis"]["cumulative_fraction_of_real_defects"] = 0.4
    config.write_text(yaml.safe_dump(value))

    state = MODULE.initialize(config, tmp_path / "results")
    policy = yaml.safe_load(Path(state["policy"]).read_text())

    assert policy["synthesis"]["cumulative_fraction_of_real_defects"] == 0.4
    assert "cumulative_fraction_of_total_defects" not in policy["synthesis"]
    assert policy["synthesis"]["fn_selection"] == {"mode": "all_eligible"}


def test_initialize_keeps_synthesis_topn_independent_from_retrieval(
        tmp_path: Path) -> None:
    config = _config(tmp_path)
    value = yaml.safe_load(config.read_text())
    _enable_synthesis(tmp_path, value)
    _set_valid_routed_metadata(tmp_path, value)
    value["retrieval"] = {"candidate_overfetch": 9}
    value["synthesis"]["candidate_topn"] = 3
    config.write_text(yaml.safe_dump(value))

    state = MODULE.initialize(config, tmp_path / "results")
    policy = yaml.safe_load(Path(state["policy"]).read_text())

    assert policy["retrieval"]["candidate_overfetch"] == 9
    assert policy["synthesis"]["candidate_topn"] == 3
    assert not any(
        warning["code"] == "inherited_synthesis_candidate_topn"
        for warning in state["warnings"]
    )


def test_initialize_migrates_legacy_shared_candidate_overfetch(tmp_path: Path) -> None:
    config = _config(tmp_path)
    value = yaml.safe_load(config.read_text())
    _enable_synthesis(tmp_path, value)
    _set_valid_routed_metadata(tmp_path, value)
    value["retrieval"] = {"candidate_overfetch": 7}
    config.write_text(yaml.safe_dump(value))

    state = MODULE.initialize(config, tmp_path / "results")
    policy = yaml.safe_load(Path(state["policy"]).read_text())

    assert policy["retrieval"]["candidate_overfetch"] == 7
    assert policy["synthesis"]["candidate_topn"] == 7
    assert state["warnings"][-1] == {
        "code": "inherited_synthesis_candidate_topn",
        "source": "retrieval.candidate_overfetch",
        "target": "synthesis.candidate_topn",
        "value": 7,
        "message": (
            "synthesis.candidate_topn was omitted; inherited the explicit "
            "retrieval.candidate_overfetch value for backward compatibility"
        ),
    }


def test_initialize_names_invalid_legacy_candidate_overfetch(tmp_path: Path) -> None:
    config = _config(tmp_path)
    value = yaml.safe_load(config.read_text())
    _enable_synthesis(tmp_path, value)
    _set_valid_routed_metadata(tmp_path, value)
    value["retrieval"] = {"candidate_overfetch": 0}
    config.write_text(yaml.safe_dump(value))

    with pytest.raises(ValueError, match="retrieval.candidate_overfetch.*inherited"):
        MODULE.initialize(config, tmp_path / "results")


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"fn_selection": {"mode": "external_plan"}}, "unsupported.*mode"),
        ({"max_neighbors_per_fn": 0}, "max_neighbors_per_fn.*positive integer"),
        ({"max_neighbors_per_fn": True}, "max_neighbors_per_fn.*positive integer"),
        ({"max_neighbors_per_fn": 1.5}, "max_neighbors_per_fn.*positive integer"),
        ({"candidate_topn": 0}, "candidate_topn.*positive integer"),
        ({"candidate_topn": True}, "candidate_topn.*positive integer"),
        ({"candidate_topn": 1.5}, "candidate_topn.*positive integer"),
        ({"candidate_topn": 2, "max_neighbors_per_fn": 3},
         "candidate_topn.*at least max_neighbors_per_fn"),
        ({"fn_selection": {"mode": "generated_per_type_plan", "images_per_fn": 2}},
         "images_per_fn is derived"),
        ({"cumulative_fraction_of_total_defects": -0.1}, "must be in.*0, 1"),
        ({"cumulative_fraction_of_total_defects": 1.0}, "must be in.*0, 1"),
        ({"cumulative_fraction_of_total_defects": float("nan")}, "must be in.*0, 1"),
    ],
)
def test_initialize_rejects_invalid_bounded_synthesis_contract(
        tmp_path: Path, override: dict, message: str) -> None:
    config = _config(tmp_path)
    value = yaml.safe_load(config.read_text())
    _enable_synthesis(tmp_path, value)
    value["synthesis"] = MODULE._merge(value["synthesis"], override)
    config.write_text(yaml.safe_dump(value))

    with pytest.raises(ValueError, match=message):
        MODULE.initialize(config, tmp_path / "results")


def test_initialize_rejects_automatic_baseline_selection(tmp_path: Path) -> None:
    config = _config(tmp_path)
    value = yaml.safe_load(config.read_text())
    value["baseline_mode"] = "auto"
    config.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match="cold_start or checkpoint"):
        MODULE.initialize(config, tmp_path / "results")


def test_initialize_rejects_role_overlap(tmp_path: Path) -> None:
    config = _config(tmp_path)
    value = yaml.safe_load(config.read_text())
    kpi = Path(value["sources"]["kpi"]["coco"])
    clean = Path(value["sources"]["clean"]["coco"])
    clean_data = json.loads(clean.read_text())
    kpi_image = Path(value["sources"]["kpi"]["images"]) / "kpi.png"
    clean_data["images"][0]["source_path"] = str(kpi_image)
    clean.write_text(json.dumps(clean_data))
    with pytest.raises(ValueError, match="overlaps"):
        MODULE.initialize(config, tmp_path / "results")


def test_initialize_rejects_boxed_clean_role(tmp_path: Path) -> None:
    config = _config(tmp_path)
    value = yaml.safe_load(config.read_text())
    clean = Path(value["sources"]["clean"]["coco"])
    data = json.loads(clean.read_text())
    data["annotations"] = [{"id": 1, "image_id": 1, "category_id": 1,
                            "bbox": [0, 0, 2, 2]}]
    clean.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="clean role"):
        MODULE.initialize(config, tmp_path / "results")


@pytest.mark.parametrize("bbox", [
    [-1, 1, 4, 4],
    [1, -1, 4, 4],
    [7, 1, 4, 4],
    [1, 7, 4, 4],
])
def test_initialize_rejects_bbox_outside_image(tmp_path: Path, bbox: list[int]) -> None:
    config = _config(tmp_path)
    value = yaml.safe_load(config.read_text())
    kpi = Path(value["sources"]["kpi"]["coco"])
    data = json.loads(kpi.read_text())
    data["annotations"][0]["bbox"] = bbox
    kpi.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="normalized COCO bbox exceeds image bounds"):
        MODULE.initialize(config, tmp_path / "results")


def test_initialize_accepts_mixed_boxed_and_boxless_heldout_roles(tmp_path: Path) -> None:
    config = _config(tmp_path)
    value = yaml.safe_load(config.read_text())
    for name in ("kpi", "test"):
        _append_boxless_image(value["sources"][name], name)

    state = MODULE.initialize(config, tmp_path / "results")

    for name in ("kpi", "test"):
        assert state["roles"][name]["image_count"] == 2
        assert state["roles"][name]["annotation_count"] == 1


def test_initialize_accepts_direct_canonical_kpi_metadata(tmp_path: Path) -> None:
    config = _config(tmp_path)
    value = yaml.safe_load(config.read_text())
    kpi = Path(value["sources"]["kpi"]["coco"])
    data = json.loads(kpi.read_text())
    metadata = data["images"][0].pop("deft_od_aoi")
    data["images"][0].update(metadata)
    kpi.write_text(json.dumps(data))

    state = MODULE.initialize(config, tmp_path / "results")

    assert state["status"] == "READY"


def test_initialize_rejects_missing_metadata_on_boxed_kpi_image(tmp_path: Path) -> None:
    config = _config(tmp_path)
    value = yaml.safe_load(config.read_text())
    kpi = Path(value["sources"]["kpi"]["coco"])
    data = json.loads(kpi.read_text())
    data["images"][0].pop("deft_od_aoi")
    kpi.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="lacks canonical retrieval metadata"):
        MODULE.initialize(config, tmp_path / "results")


def test_initialize_accepts_legacy_aliases_when_canonical_metadata_exists(
        tmp_path: Path) -> None:
    config = _config(tmp_path)
    value = yaml.safe_load(config.read_text())
    kpi = Path(value["sources"]["kpi"]["coco"])
    data = json.loads(kpi.read_text())
    data["images"][0]["deft_od_aoi"].update({
        "benchmark": "legacy-dataset",
        "texture": "legacy-texture",
        "defect_type": "legacy-defect",
    })
    kpi.write_text(json.dumps(data))

    state = MODULE.initialize(config, tmp_path / "results")

    assert state["status"] == "READY"


def test_initialize_rejects_legacy_only_kpi_metadata(tmp_path: Path) -> None:
    config = _config(tmp_path)
    value = yaml.safe_load(config.read_text())
    kpi = Path(value["sources"]["kpi"]["coco"])
    data = json.loads(kpi.read_text())
    data["images"][0]["deft_od_aoi"] = {
        "benchmark": "route", "texture": "texture", "defect_type": "defect"
    }
    kpi.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="lacks canonical retrieval metadata"):
        MODULE.initialize(config, tmp_path / "results")


def test_initialize_expands_and_resolves_coco_paths(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    config = _config(fake_home)
    value = yaml.safe_load(config.read_text())
    expected = {}
    for name, source in value["sources"].items():
        coco = Path(source["coco"])
        expected[name] = str(coco.resolve())
        source["coco"] = f"~/{coco.relative_to(fake_home)}"
    config.write_text(yaml.safe_dump(value))
    monkeypatch.setenv("HOME", str(fake_home))

    state = MODULE.initialize(config, tmp_path / "results")
    frozen = yaml.safe_load(Path(state["policy"]).read_text())

    assert {name: role["coco"] for name, role in state["roles"].items()} == expected
    assert {name: role["coco"] for name, role in frozen["sources"].items()} == expected


def test_initialize_rejects_boxless_defective_real_role(tmp_path: Path) -> None:
    config = _config(tmp_path)
    value = yaml.safe_load(config.read_text())
    _append_boxless_image(value["sources"]["real"], "real")

    with pytest.raises(ValueError, match="defective-real role contains a boxless image"):
        MODULE.initialize(config, tmp_path / "results")


def test_initialize_accepts_empty_clean_role_with_capability_evidence(tmp_path: Path) -> None:
    config = _config(tmp_path)
    value = yaml.safe_load(config.read_text())
    role = "clean"
    coco = Path(value["sources"][role]["coco"])
    data = json.loads(coco.read_text())
    data["images"] = []
    data["annotations"] = []
    coco.write_text(json.dumps(data))

    state = MODULE.initialize(config, tmp_path / "results")

    assert state["roles"][role]["image_count"] == 0
    assert state["capabilities"]["retrieval"][role] == {
        "status": "UNAVAILABLE", "reason": "empty_source_role", "source_image_count": 0}
    assert state["warnings"][0]["code"] == "empty_retrieval_source_role"


def test_initialize_rejects_empty_real_role(tmp_path: Path) -> None:
    config = _config(tmp_path)
    value = yaml.safe_load(config.read_text())
    coco = Path(value["sources"]["real"]["coco"])
    data = json.loads(coco.read_text())
    data["images"] = []
    data["annotations"] = []
    coco.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="real has no images"):
        MODULE.initialize(config, tmp_path / "results")


@pytest.mark.parametrize("role", ("kpi", "test"))
def test_initialize_rejects_empty_heldout_role_with_specific_error(
        tmp_path: Path, role: str) -> None:
    config = _config(tmp_path)
    value = yaml.safe_load(config.read_text())
    coco = Path(value["sources"][role]["coco"])
    data = json.loads(coco.read_text())
    data["images"] = []
    data["annotations"] = []
    coco.write_text(json.dumps(data))

    with pytest.raises(ValueError, match=rf"{role} has no images"):
        MODULE.initialize(config, tmp_path / "results")


def test_initialize_rejects_duplicate_image_ids_with_specific_error(tmp_path: Path) -> None:
    config = _config(tmp_path)
    value = yaml.safe_load(config.read_text())
    coco = Path(value["sources"]["real"]["coco"])
    data = json.loads(coco.read_text())
    data["images"].append(dict(data["images"][0]))
    coco.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="real has duplicate image ids"):
        MODULE.initialize(config, tmp_path / "results")


def test_initialize_routes_missing_synthesis_weights_to_bootstrap(tmp_path: Path) -> None:
    config = _config(tmp_path)
    value = yaml.safe_load(config.read_text())
    _set_valid_routed_metadata(tmp_path, value)
    _enable_synthesis(tmp_path, value)
    config.write_text(yaml.safe_dump(value))
    state = MODULE.initialize(config, tmp_path / "results")
    assert state["synthesis_bootstrap_required"] is True
    assert state["next_stage"] == "synthesis_bootstrap"


@pytest.mark.parametrize("pool_shape", ("empty", "missing_clean_dir", "unsupported_file"))
def test_initialize_rejects_globally_empty_synthesis_clean_pool(
        tmp_path: Path, pool_shape: str) -> None:
    config = _config(tmp_path)
    value = yaml.safe_load(config.read_text())
    pool = tmp_path / "pool"
    pool.mkdir()
    if pool_shape == "missing_clean_dir":
        (pool / "texture_1").mkdir()
    elif pool_shape == "unsupported_file":
        clean = pool / "texture_1/clean_image/readme.txt"
        clean.parent.mkdir(parents=True)
        clean.write_text("not an image")
    defect = tmp_path / "defect.jsonl"
    defect.write_text("{}\n")
    checkpoint = tmp_path / "adapter.pt"
    checkpoint.write_bytes(b"checkpoint")
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text("anomaly_types: []\n")
    value["synthesis"] = {
        "enabled": True,
        "pool_dataset_root": str(pool),
        "defect_spec": str(defect),
        "routes": {"route": {"checkpoint": str(checkpoint), "recipe": str(recipe)}},
    }
    config.write_text(yaml.safe_dump(value))

    with pytest.raises(ValueError, match="at least one clean reference image"):
        MODULE.initialize(config, tmp_path / "results")


def test_initialize_rejects_missing_kpi_dataset_id_for_synthesis(tmp_path: Path) -> None:
    config = _config(tmp_path)
    value = yaml.safe_load(config.read_text())
    _set_kpi_dataset_id(value, "")
    _enable_synthesis(tmp_path, value)
    config.write_text(yaml.safe_dump(value))

    with pytest.raises(ValueError, match=(
            "annotation_id=1 image_id=1 file_name='kpi.png'.*allowlist")):
        MODULE.initialize(config, tmp_path / "results")
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("missing", MODULE.ROUTED_SYNTHESIS_FIELDS)
def test_initialize_rejects_missing_routed_kpi_metadata(
        tmp_path: Path, missing: str) -> None:
    config = _config(tmp_path)
    value = yaml.safe_load(config.read_text())
    mask = _set_valid_routed_metadata(tmp_path, value)
    metadata = {"dataset_id": "route", "texture_id": "texture",
                "defect_class": "scratch", "fn_mask_source": str(mask)}
    metadata[missing] = ""
    _set_kpi_synthesis_metadata(value, **metadata)
    _enable_synthesis(tmp_path, value)
    config.write_text(yaml.safe_dump(value))

    with pytest.raises(ValueError, match=rf"annotation_id=1.*missing=\['{missing}'\]"):
        MODULE.initialize(config, tmp_path / "results")
    assert not (tmp_path / "results").exists()


def test_initialize_rejects_missing_routed_kpi_mask_file(tmp_path: Path) -> None:
    config = _config(tmp_path)
    value = yaml.safe_load(config.read_text())
    missing_mask = tmp_path / "missing-mask.png"
    _set_kpi_synthesis_metadata(
        value, dataset_id="route", texture_id="texture", defect_class="scratch",
        fn_mask_source=str(missing_mask)
    )
    _enable_synthesis(tmp_path, value)
    config.write_text(yaml.safe_dump(value))

    with pytest.raises(ValueError, match="fn_mask_source is not an existing file"):
        MODULE.initialize(config, tmp_path / "results")


def test_initialize_accepts_unconfigured_nested_kpi_dataset_id(tmp_path: Path) -> None:
    config = _config(tmp_path)
    value = yaml.safe_load(config.read_text())
    _set_kpi_dataset_id(value, "normal_real_data", nested=True)
    _enable_synthesis(tmp_path, value)
    config.write_text(yaml.safe_dump(value))

    state = MODULE.initialize(config, tmp_path / "results")

    assert state["synthesis_enabled"] is True


def test_initialize_resolves_tilde_kpi_coco_before_routed_validation(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    home.mkdir()
    config = _config(home)
    value = yaml.safe_load(config.read_text())
    _set_valid_routed_metadata(home, value)
    _enable_synthesis(home, value)
    kpi_coco = Path(value["sources"]["kpi"]["coco"])
    value["sources"]["kpi"]["coco"] = f"~/{kpi_coco.relative_to(home)}"
    config.write_text(yaml.safe_dump(value))
    monkeypatch.setenv("HOME", str(home))

    state = MODULE.initialize(config, tmp_path / "results")

    assert state["roles"]["kpi"]["coco"] == str(kpi_coco.resolve())


def test_initialize_accepts_mixed_routed_and_unrouted_kpi_metadata(
        tmp_path: Path) -> None:
    config = _config(tmp_path)
    value = yaml.safe_load(config.read_text())
    _set_valid_routed_metadata(tmp_path, value)
    kpi_images = Path(value["sources"]["kpi"]["images"])
    unrouted_image = kpi_images / "unrouted.png"
    unrouted_image.write_bytes(b"unrouted-image")
    kpi_coco = Path(value["sources"]["kpi"]["coco"])
    data = json.loads(kpi_coco.read_text())
    data["images"].append({
        "id": 2, "file_name": unrouted_image.name, "width": 10, "height": 10,
        "deft_od_aoi": {
            "dataset_id": "normal_real_data", "texture_id": "texture",
            "defect_class": "scratch",
        },
    })
    data["annotations"].append({
        "id": 2,
        "image_id": 2,
        "category_id": 1,
        "bbox": [1, 1, 4, 4],
        "area": 16,
        "deft_od_aoi": {"dataset_id": "normal_real_data"},
    })
    kpi_coco.write_text(json.dumps(data))
    _enable_synthesis(tmp_path, value)
    config.write_text(yaml.safe_dump(value))

    state = MODULE.initialize(config, tmp_path / "results")

    assert state["synthesis_enabled"] is True
    assert state["roles"]["kpi"]["annotation_count"] == 2


def test_annotation_dataset_id_overrides_image_metadata(tmp_path: Path) -> None:
    config = _config(tmp_path)
    value = yaml.safe_load(config.read_text())
    coco = Path(value["sources"]["kpi"]["coco"])
    data = json.loads(coco.read_text())
    data["images"][0]["dataset_id"] = "route"
    data["annotations"][0]["dataset_id"] = ""
    coco.write_text(json.dumps(data))
    _enable_synthesis(tmp_path, value)
    config.write_text(yaml.safe_dump(value))

    with pytest.raises(ValueError, match="nonempty dataset_id"):
        MODULE.initialize(config, tmp_path / "results")


def test_synthesis_allows_boxless_kpi_image_without_dataset_id(tmp_path: Path) -> None:
    config = _config(tmp_path)
    value = yaml.safe_load(config.read_text())
    _set_valid_routed_metadata(tmp_path, value)
    _append_boxless_image(value["sources"]["kpi"], "kpi")
    _enable_synthesis(tmp_path, value)
    config.write_text(yaml.safe_dump(value))

    state = MODULE.initialize(config, tmp_path / "results")

    assert state["roles"]["kpi"]["image_count"] == 2


def test_initialize_defers_routed_mask_content_validation(tmp_path: Path) -> None:
    config = _config(tmp_path)
    value = yaml.safe_load(config.read_text())
    _set_valid_routed_metadata(tmp_path, value, b"invalid mask contents")
    _enable_synthesis(tmp_path, value)
    config.write_text(yaml.safe_dump(value))

    state = MODULE.initialize(config, tmp_path / "results")

    assert state["synthesis_enabled"] is True
