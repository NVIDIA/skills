# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image


SCRIPT = Path(__file__).parents[1] / "prepare_anomalygennext_inputs.py"
SPEC = importlib.util.spec_from_file_location("prepare_anomalygennext_inputs", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)
AMP_SCRIPT = Path(__file__).parents[1] / "run_anomalygennext_amp.py"
AMP_SPEC = importlib.util.spec_from_file_location("run_anomalygennext_amp_for_prepare", AMP_SCRIPT)
AMP_MODULE = importlib.util.module_from_spec(AMP_SPEC)
assert AMP_SPEC.loader
AMP_SPEC.loader.exec_module(AMP_MODULE)


def image(path: Path, value: int = 80) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((32, 32, 3), value, dtype=np.uint8)).save(path)


def mask(path: Path, offset: int = 0) -> None:
    value = np.zeros((32, 32), dtype=np.uint8)
    value[4 + offset:20 + offset, 4:20] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(value).save(path)


def fixture(tmp_path: Path) -> tuple[Path, Path]:
    defective = tmp_path / "defect.png"
    source_mask = tmp_path / "defect_mask.png"
    image(defective)
    mask(source_mask)
    pool = tmp_path / "pool" / "texture_1"
    image(pool / "clean_image" / "clean1.png", 10)
    image(pool / "clean_image" / "clean2.png", 20)
    mask(pool / "mask" / "crack" / "mask1.png")
    mask(pool / "mask" / "crack" / "mask2.png", 1)
    gaps = tmp_path / "gaps.parquet"
    pd.DataFrame([
        {"image_id": 1, "filepath": str(defective), "gap_type": "FN",
         "bbox": [4, 4, 20, 20], "class": "defect", "split": "kpi",
         "dataset_id": "example", "texture_id": "texture_1",
         "defect_class": "crack", "anomaly_type": "texture_1+crack",
         "fn_mask_source": str(source_mask)},
        {"image_id": 1, "filepath": str(defective), "gap_type": "FN",
         "bbox": [6, 6, 18, 18], "class": "defect", "split": "kpi",
         "dataset_id": "example", "texture_id": "texture_1",
         "defect_class": "crack", "anomaly_type": "texture_1+crack",
         "fn_mask_source": str(source_mask)},
    ]).to_parquet(gaps)
    checkpoint = tmp_path / "adapter.pt"
    checkpoint.write_bytes(b"checkpoint")
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(yaml.safe_dump({"anomaly_types": [["texture_1", "crack"]]}))
    defect_spec = tmp_path / "defect_spec.jsonl"
    defect_spec.write_text(json.dumps({"defect_type": "texture_1+crack",
                                       "spatial_dependency": "free"}) + "\n")
    config = tmp_path / "filtering.yaml"
    config.write_text(yaml.safe_dump({
        "source_tag": "test", "gap_parquet": str(gaps),
        "pool_dataset_root": str(tmp_path / "pool"), "defect_spec": str(defect_spec),
        "datasets": {"example": {"checkpoint": str(checkpoint), "recipe": str(recipe)}},
        "selection": {"mode": "all_eligible", "datasets": ["example"],
                      "mask_sample_seed": 7},
        "embedding": {"model": "SigLIP", "model_path": "local/siglip", "batch_size": 8},
        "retrieval": {"metric": "cosine", "candidate_topn": 2,
                      "max_neighbors_per_fn": 1, "min_similarity": -1.0,
                      "prior_clean_exclusion_manifest": ""},
    }, sort_keys=False))
    return config, gaps


def test_prepare_preserves_box_level_queries_and_unique_embedding_input(tmp_path: Path) -> None:
    config, _ = fixture(tmp_path)
    output = tmp_path / "output"
    report = MODULE.prepare(config, output)
    assert report["status"] == "COMPLETE"
    assert report["source_tag"] == "test"
    assert report["selected_fn_count"] == report["eligible_fn_count"] == 2
    assert report["skipped_fn_count"] == 0
    assert report["clean_image_count"] == 2 and report["source_mask_count"] == 4
    assert report["training_pool_mutated"] is False
    queries = pd.read_parquet(output / "manifests" / "selected_fn_queries.parquet")
    embedding = pd.read_parquet(output / "manifests" / "fn_embedding_inputs.parquet")
    masks = pd.read_parquet(output / "manifests" / "mask_selection.parquet")
    assert len(queries) == 2 and queries.fn_id.nunique() == 2
    assert len(embedding) == 1
    assert masks.groupby("fn_id").branch.nunique().eq(2).all()
    assert all(Path(path).is_file() for path in masks.mask_path)
    clean_spec = yaml.safe_load((output / "specs" / "clean_embeddings.yaml").read_text())
    fn_spec = yaml.safe_load((output / "specs" / "fn_embeddings.yaml").read_text())
    assert clean_spec["model_path"] == fn_spec["model_path"] == "local/siglip"


def test_typed_pool_root_is_frozen_for_both_embedding_specs(tmp_path: Path) -> None:
    config, _ = fixture(tmp_path)
    configured_pool = tmp_path / "pool"
    staged_pool = tmp_path / "staged/pool"
    staged_pool.parent.mkdir()
    configured_pool.rename(staged_pool)
    output = tmp_path / "output"

    MODULE.prepare(config, output, staged_pool)

    frozen = yaml.safe_load(
        (output / "prepared_anomalygennext_inputs/filtering_config.yaml").read_text()
    )
    assert frozen["pool_dataset_root"] == str(staged_pool.resolve())
    clean = pd.read_parquet(output / "manifests/clean_pool.parquet")
    assert all(Path(path).is_relative_to(staged_pool.resolve()) for path in clean.filepath)
    contract = json.loads(
        (output / "prepared_anomalygennext_inputs/input_contract.json").read_text()
    )
    expected_mount = {
        "type": "folder", "compute_path": str(staged_pool.resolve()), "read_only": True
    }
    assert contract["downstream_inputs"] == {
        "clean_embeddings": {"pool_dataset_root": expected_mount},
        "run_amp": {"pool_dataset_root": expected_mount},
    }
    for name in ("clean", "fn"):
        spec = yaml.safe_load((output / f"specs/{name}_embeddings.yaml").read_text())
        assert "pool_dataset_root" not in spec
        assert Path(spec["input_parquet"]).is_relative_to(output.resolve())
        assert Path(spec["output_parquet"]).is_relative_to(output.resolve())
        inputs = pd.read_parquet(spec["input_parquet"])
        assert all(Path(path).is_file() for path in inputs.filepath)


def test_action_contract_declares_pool_folder_for_prepare_and_amp() -> None:
    info = yaml.safe_load(
        (Path(__file__).parents[2] / "references/skill_info.yaml").read_text()
    )
    for action in ("prepare_plan", "run_amp"):
        assert info["actions"][action]["inputs"]["pool_dataset_root"] == {"type": "folder"}
        assert "{pool_dataset_root}" in info["actions"][action]["args"]["pool_dataset_root"]
    assert "config" not in info["actions"]["run_amp"]["inputs"]
    assert "config" not in info["actions"]["run_amp"]["args"]


def test_prepare_requires_normalized_identity(tmp_path: Path) -> None:
    config, gaps = fixture(tmp_path)
    frame = pd.read_parquet(gaps).drop(columns=["fn_mask_source"])
    frame.to_parquet(gaps)
    with pytest.raises(ValueError, match="normalized columns"):
        MODULE.prepare(config, tmp_path / "output")


def test_prepare_refuses_output_reuse(tmp_path: Path) -> None:
    config, _ = fixture(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(FileExistsError):
        MODULE.prepare(config, output)


def test_prepare_skips_one_invalid_fn_and_amp_plans_only_the_valid_fn(tmp_path: Path) -> None:
    config, gaps_path = fixture(tmp_path)
    gaps = pd.read_parquet(gaps_path)
    invalid = tmp_path / "invalid-empty.png"
    Image.fromarray(np.zeros((32, 32), dtype=np.uint8)).save(invalid)
    gaps.loc[1, "fn_mask_source"] = str(invalid)
    gaps.to_parquet(gaps_path, index=False)

    output = tmp_path / "output"
    report = MODULE.prepare(config, output)

    assert report["eligible_fn_count"] == 1 and report["skipped_fn_count"] == 1
    assert report["skip_counts"] == {"empty_fn_mask": 1}
    assert report["skipped_fns"][0]["reason"] == "empty_fn_mask"
    eligible_id = report["eligible_fn_ids"][0]
    for name in ("fn_queries", "selected_fn_queries", "fn_embedding_inputs"):
        frame = pd.read_parquet(output / "manifests" / f"{name}.parquet")
        assert set(frame.fn_id) == {eligible_id}
    masks = pd.read_parquet(output / "manifests/mask_selection.parquet")
    assert set(masks.fn_id) == {eligible_id} and len(masks) == 2

    clean = pd.read_parquet(output / "manifests/clean_pool.parquet")
    clean["embedding"] = [[1.0, 0.0]] * len(clean)
    (output / "embeddings").mkdir()
    clean.to_parquet(output / "embeddings/clean_embeddings.parquet", index=False)
    embedded = pd.read_parquet(output / "manifests/fn_embedding_inputs.parquet")
    embedded["embedding"] = [[1.0, 0.0]] * len(embedded)
    embedded.to_parquet(output / "embeddings/fn_embeddings.parquet", index=False)
    amp_report = AMP_MODULE.plan(output, yaml.safe_load(config.read_text()))
    assert amp_report["amp_rows"] == 4
    candidates = pd.read_parquet(output / "manifests/knn_candidates.parquet")
    assert set(candidates.fn_id) == {eligible_id}


@pytest.mark.parametrize("missing_directory", (False, True))
def test_prepare_types_all_missing_clean_references_as_skipped(
        tmp_path: Path, missing_directory: bool) -> None:
    config, _ = fixture(tmp_path)
    clean_dir = tmp_path / "pool/texture_1/clean_image"
    for path in clean_dir.iterdir():
        path.unlink()
    if missing_directory:
        clean_dir.rmdir()

    output = tmp_path / "output"
    report = MODULE.prepare(config, output)

    assert report["status"] == "SKIPPED"
    assert report["reason"] == "no_eligible_false_negatives"
    assert report["eligible_fn_count"] == 0 and report["skipped_fn_count"] == 2
    assert report["skip_counts"] == {"no_clean_reference_images": 2}
    assert {row["reason"] for row in report["skipped_fns"]} == {
        "no_clean_reference_images"}
    assert report["warnings"] == [{
        "code": "no_clean_reference_images", "fn_count": 2,
        "message": ("Skipped 2 routed false negatives because their AnomalyGen "
                    "clean-reference pools contain no images"),
    }]
    assert (output / "prepared_anomalygennext_inputs/input_contract.json").is_file()
    assert not list((output / "specs").glob("*_embeddings.yaml"))
    assert not (output / "manifests/selected_fn_queries.parquet").exists()


def test_prepare_skips_missing_clean_type_while_eligible_type_continues(
        tmp_path: Path) -> None:
    config, gaps_path = fixture(tmp_path)
    gaps = pd.read_parquet(gaps_path)
    gaps.loc[1, ["texture_id", "defect_class", "anomaly_type"]] = [
        "texture_2", "dent", "texture_2+dent"
    ]
    gaps.to_parquet(gaps_path, index=False)
    donor = tmp_path / "pool/texture_2/mask/dent/donor.png"
    mask(donor)
    value = yaml.safe_load(config.read_text())
    recipe = Path(value["datasets"]["example"]["recipe"])
    recipe.write_text(yaml.safe_dump({"anomaly_types": [["texture_1", "crack"],
                                                         ["texture_2", "dent"]]}))
    defect_spec = Path(value["defect_spec"])
    with defect_spec.open("a") as stream:
        stream.write(json.dumps({"defect_type": "texture_2+dent",
                                 "spatial_dependency": "free"}) + "\n")

    output = tmp_path / "output"
    report = MODULE.prepare(config, output)

    assert report["status"] == "COMPLETE"
    assert report["eligible_fn_count"] == 1 and report["skipped_fn_count"] == 1
    assert report["skip_counts"] == {"no_clean_reference_images": 1}
    assert report["warnings"][0]["code"] == "no_clean_reference_images"
    queries = pd.read_parquet(output / "manifests/selected_fn_queries.parquet")
    assert list(queries.anomaly_type) == ["texture_1+crack"]
    assert (output / "specs/clean_embeddings.yaml").is_file()
    assert (output / "specs/fn_embeddings.yaml").is_file()


@pytest.mark.parametrize(
    ("kind", "reason"),
    [
        ("empty", "empty_fn_mask"),
        ("full", "full_frame_fn_mask"),
        ("full_extent", "full_tight_extent_fn_mask"),
        ("dimension", "dimension_mismatch"),
        ("outside_bbox", "bbox_mask_mismatch"),
    ],
)
def test_prepare_reports_invalid_fn_masks(tmp_path: Path, kind: str, reason: str) -> None:
    config, gaps_path = fixture(tmp_path)
    gaps = pd.read_parquet(gaps_path).iloc[[0]].copy()
    values = np.zeros((16, 16) if kind == "dimension" else (32, 32), dtype=np.uint8)
    if kind == "full":
        values[:] = 255
    elif kind == "full_extent":
        values[0, 0] = values[-1, -1] = 255
    elif kind == "dimension":
        values[2:8, 2:8] = 255
    elif kind == "outside_bbox":
        values[24:30, 24:30] = 255
    invalid = tmp_path / f"{kind}.png"
    Image.fromarray(values).save(invalid)
    gaps.loc[gaps.index[0], "fn_mask_source"] = str(invalid)
    gaps.to_parquet(gaps_path, index=False)

    report = MODULE.prepare(config, tmp_path / "output")

    assert report["status"] == "SKIPPED"
    assert report["reason"] == "no_eligible_false_negatives"
    assert report["eligible_fn_count"] == 0 and report["skipped_fn_count"] == 1
    assert report["skip_counts"] == {reason: 1}
    assert not (tmp_path / "output/specs/fn_embeddings.yaml").exists()


def test_prepare_falls_through_invalid_donor_deterministically(tmp_path: Path) -> None:
    config, gaps_path = fixture(tmp_path)
    gaps = pd.read_parquet(gaps_path).iloc[[0]].copy()
    gaps.to_parquet(gaps_path, index=False)
    row = gaps.iloc[0]
    bbox_json = json.dumps(np.asarray(row.bbox).reshape(-1).tolist(), separators=(",", ":"))
    fn_id = "fn-" + MODULE._stable_id("example", row.image_id,
                                      Path(row.filepath).resolve(), bbox_json)
    donors = MODULE._images(tmp_path / "pool/texture_1/mask/crack")
    ordered = MODULE._ordered_donors(donors, set(), 7, fn_id)
    Image.fromarray(np.zeros((32, 32), dtype=np.uint8)).save(ordered[0])
    values = np.zeros((32, 32), dtype=np.uint8)
    values[3:9, 5:12] = 255
    Image.fromarray(values).save(ordered[1])

    report = MODULE.prepare(config, tmp_path / "output")

    assert report["eligible_fn_ids"] == [fn_id]
    donor = pd.read_parquet(tmp_path / "output/manifests/mask_selection.parquet")
    donor = donor[donor.branch == "same_type_sampled_mask"].iloc[0]
    assert donor.foreground_pixels == 42


def test_prepare_skips_only_fn_whose_same_type_donors_are_invalid(tmp_path: Path) -> None:
    config, gaps_path = fixture(tmp_path)
    gaps = pd.read_parquet(gaps_path)
    gaps.loc[1, ["texture_id", "defect_class", "anomaly_type"]] = [
        "texture_2", "dent", "texture_2+dent"
    ]
    gaps.to_parquet(gaps_path, index=False)
    pool = tmp_path / "pool/texture_2"
    image(pool / "clean_image/clean.png")
    bad_donor = pool / "mask/dent/bad.png"
    bad_donor.parent.mkdir(parents=True)
    Image.fromarray(np.zeros((32, 32), dtype=np.uint8)).save(bad_donor)
    value = yaml.safe_load(config.read_text())
    recipe = Path(value["datasets"]["example"]["recipe"])
    recipe.write_text(yaml.safe_dump({"anomaly_types": [["texture_1", "crack"],
                                                         ["texture_2", "dent"]]}))
    defect_spec = Path(value["defect_spec"])
    with defect_spec.open("a") as stream:
        stream.write(json.dumps({"defect_type": "texture_2+dent",
                                 "spatial_dependency": "free"}) + "\n")

    report = MODULE.prepare(config, tmp_path / "output")

    assert report["eligible_fn_count"] == 1 and report["skipped_fn_count"] == 1
    assert report["skip_counts"] == {"no_valid_same_type_donor": 1}
    queries = pd.read_parquet(tmp_path / "output/manifests/selected_fn_queries.parquet")
    assert list(queries.anomaly_type) == ["texture_1+crack"]
