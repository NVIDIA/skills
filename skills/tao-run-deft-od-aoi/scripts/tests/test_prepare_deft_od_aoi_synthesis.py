# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import pytest
import yaml


SCRIPT = Path(__file__).parents[1] / "prepare_deft_od_aoi_synthesis.py"
SPEC = importlib.util.spec_from_file_location("prepare_deft_od_aoi_synthesis", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_synthesis_normalizes_exact_kpi_false_negative(tmp_path: Path) -> None:
    images = tmp_path / "kpi"
    images.mkdir()
    image, mask = images / "image.png", tmp_path / "mask.png"
    image.write_bytes(b"image")
    mask.write_bytes(b"mask")
    coco = tmp_path / "kpi.json"
    coco.write_text(json.dumps({
        "images": [{"id": 7, "file_name": image.name, "dataset_id": "route",
                    "texture_id": "texture"}],
        "annotations": [{"id": 3, "image_id": 7, "category_id": 1,
                         "bbox": [4, 5, 10, 12], "defect_class": "crack",
                         "fn_mask_source": str(mask)}],
        "categories": [{"id": 1, "name": "defect"}]}))
    pool = tmp_path / "pool"
    pool.mkdir()
    defect_spec, checkpoint, recipe = tmp_path / "defect.jsonl", tmp_path / "adapter.pt", tmp_path / "recipe.yaml"
    defect_spec.write_text(json.dumps({"defect_type": "texture+crack"}) + "\n")
    checkpoint.write_bytes(b"adapter")
    recipe.write_text("anomaly_types: [[texture, crack]]\n")
    policy = tmp_path / "policy.yaml"
    policy.write_text(yaml.safe_dump({"sources": {"kpi": {"images": str(images),
                                                             "coco": str(coco)}},
                                      "retrieval": {"model": "SigLIP", "model_path": "siglip",
                                                    "candidate_overfetch": 15},
                                      "synthesis": {"enabled": True,
                                                    "fn_selection": {"mode": "all_eligible"},
                                                    "pool_dataset_root": str(pool),
                                                    "defect_spec": str(defect_spec),
                                                    "routes": {"route": {"checkpoint": str(checkpoint),
                                                                           "recipe": str(recipe)}},
                                                    "candidate_topn": 3,
                                                    "max_neighbors_per_fn": 1,
                                                    "min_similarity": 0.9,
                                                    "amp_model_id": "nvidia/Cosmos3-Nano"}}))
    gaps = tmp_path / "strict.parquet"
    pd.DataFrame([{"image_id": 7, "filepath": str(image), "gap_type": "FN",
                   "bbox": [4, 5, 14, 17], "class": "defect"}]).to_parquet(gaps)
    report = MODULE.prepare(policy, gaps, tmp_path / "out")
    assert report["fn_count"] == 1
    assert report["eligible_fn_count"] == 1
    assert report["selection_mode"] == "all_eligible"
    assert report["candidate_topn"] == 3
    assert report["candidate_topn_source"] == "synthesis.candidate_topn"
    normalized = pd.read_parquet(tmp_path / "out/normalized_fn_gaps.parquet").iloc[0]
    assert normalized.anomaly_type == "texture+crack"
    config = yaml.safe_load((tmp_path / "out/anomalygen_filtering.yaml").read_text())
    assert config["datasets"]["route"]["checkpoint"] == str(checkpoint.resolve())
    assert config["retrieval"]["candidate_topn"] == 3
    assert config["retrieval"]["max_neighbors_per_fn"] == 1


def test_synthesis_resolves_gap_filename_stem_to_coco_id(tmp_path: Path) -> None:
    images = tmp_path / "kpi"
    images.mkdir()
    image, mask = images / "stable-image-key.png", tmp_path / "mask.png"
    image.write_bytes(b"image")
    mask.write_bytes(b"mask")
    coco = tmp_path / "kpi.json"
    coco.write_text(json.dumps({
        "images": [{"id": 17, "file_name": image.name, "dataset_id": "route",
                    "texture_id": "texture"}],
        "annotations": [{"id": 3, "image_id": 17, "category_id": 1,
                         "bbox": [4, 5, 10, 12], "defect_class": "crack",
                         "fn_mask_source": str(mask)}],
        "categories": [{"id": 1, "name": "defect"}]}))
    pool = tmp_path / "pool"
    pool.mkdir()
    defect_spec = tmp_path / "defect.jsonl"
    checkpoint = tmp_path / "adapter.pt"
    recipe = tmp_path / "recipe.yaml"
    defect_spec.write_text(json.dumps({"defect_type": "texture+crack"}) + "\n")
    checkpoint.write_bytes(b"adapter")
    recipe.write_text("anomaly_types: [[texture, crack]]\n")
    policy = tmp_path / "policy.yaml"
    policy.write_text(yaml.safe_dump({
        "sources": {"kpi": {"images": str(images), "coco": str(coco)}},
        "retrieval": {"model": "SigLIP", "model_path": "siglip",
                      "candidate_overfetch": 15},
        "synthesis": {"enabled": True, "pool_dataset_root": str(pool),
                      "fn_selection": {"mode": "all_eligible"},
                      "defect_spec": str(defect_spec),
                      "routes": {"route": {"checkpoint": str(checkpoint),
                                             "recipe": str(recipe)}},
                      "max_neighbors_per_fn": 5, "min_similarity": 0.9,
                      "amp_model_id": "nvidia/Cosmos3-Nano"}}))
    gaps = tmp_path / "strict.parquet"
    pd.DataFrame([{"image_id": image.stem, "filepath": str(image), "gap_type": "FN",
                   "bbox": [4, 5, 14, 17], "class": "defect"}]).to_parquet(gaps)

    report = MODULE.prepare(policy, gaps, tmp_path / "out")

    assert report["fn_count"] == 1
    assert report["candidate_topn"] == 15
    assert report["candidate_topn_source"] == "retrieval.candidate_overfetch"
    normalized = pd.read_parquet(tmp_path / "out/normalized_fn_gaps.parquet").iloc[0]
    assert normalized.image_id == image.stem
    config = yaml.safe_load((tmp_path / "out/anomalygen_filtering.yaml").read_text())
    assert config["retrieval"]["candidate_topn"] == 15


def test_synthesis_accepts_hardlinked_normalized_kpi_view(tmp_path: Path) -> None:
    images = tmp_path / "kpi"
    images.mkdir()
    original = tmp_path / "original.png"
    normalized = images / "stable-image-key.png"
    mask = tmp_path / "mask.png"
    original.write_bytes(b"image")
    os.link(original, normalized)
    mask.write_bytes(b"mask")
    coco = tmp_path / "kpi.json"
    coco.write_text(json.dumps({
        "images": [{"id": 17, "file_name": normalized.name,
                    "source_path": str(original), "dataset_id": "route",
                    "texture_id": "texture"}],
        "annotations": [{"id": 3, "image_id": 17, "category_id": 1,
                         "bbox": [4, 5, 10, 12], "defect_class": "crack",
                         "fn_mask_source": str(mask)}],
        "categories": [{"id": 1, "name": "defect"}],
    }))
    pool = tmp_path / "pool"
    pool.mkdir()
    defect_spec = tmp_path / "defect.jsonl"
    checkpoint = tmp_path / "adapter.pt"
    recipe = tmp_path / "recipe.yaml"
    defect_spec.write_text(json.dumps({"defect_type": "texture+crack"}) + "\n")
    checkpoint.write_bytes(b"adapter")
    recipe.write_text("anomaly_types: [[texture, crack]]\n")
    policy = tmp_path / "policy.yaml"
    policy.write_text(yaml.safe_dump({
        "sources": {"kpi": {"images": str(images), "coco": str(coco)}},
        "retrieval": {"model": "SigLIP", "model_path": "siglip",
                      "candidate_overfetch": 15},
        "synthesis": {"enabled": True, "pool_dataset_root": str(pool),
                      "fn_selection": {"mode": "all_eligible"},
                      "defect_spec": str(defect_spec),
                      "routes": {"route": {"checkpoint": str(checkpoint),
                                             "recipe": str(recipe)}},
                      "max_neighbors_per_fn": 5, "min_similarity": 0.9,
                      "amp_model_id": "nvidia/Cosmos3-Nano"},
    }))
    gaps = tmp_path / "strict.parquet"
    pd.DataFrame([{"image_id": normalized.stem, "filepath": str(normalized),
                   "gap_type": "FN", "bbox": [4, 5, 14, 17],
                   "class": "defect"}]).to_parquet(gaps)

    report = MODULE.prepare(policy, gaps, tmp_path / "out")

    assert report["fn_count"] == 1
    output = pd.read_parquet(tmp_path / "out/normalized_fn_gaps.parquet").iloc[0]
    assert Path(output.filepath) == original.resolve()

    unrelated = tmp_path / "unrelated.png"
    unrelated.write_bytes(b"different")
    pd.DataFrame([{"image_id": normalized.stem, "filepath": str(unrelated),
                   "gap_type": "FN", "bbox": [4, 5, 14, 17],
                   "class": "defect"}]).to_parquet(gaps)
    with pytest.raises(ValueError, match="filepath does not match"):
        MODULE.prepare(policy, gaps, tmp_path / "out-unrelated")


def test_synthesis_skips_unrouted_dataset_without_weakening_routed_masks(
    tmp_path: Path,
) -> None:
    images = tmp_path / "kpi"
    images.mkdir()
    routed_image = images / "routed.png"
    unrouted_image = images / "unrouted.png"
    mask = tmp_path / "mask.png"
    for path in (routed_image, unrouted_image, mask):
        path.write_bytes(b"input")
    coco = tmp_path / "kpi.json"
    coco.write_text(json.dumps({
        "images": [
            {"id": 1, "file_name": routed_image.name, "dataset_id": "route",
             "texture_id": "texture"},
            {"id": 2, "file_name": unrouted_image.name, "dataset_id": "boxes_only",
             "texture_id": "board"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 2, 3, 4],
             "defect_class": "crack", "fn_mask_source": str(mask)},
            {"id": 2, "image_id": 2, "category_id": 1, "bbox": [5, 6, 7, 8],
             "defect_class": "open"},
        ],
        "categories": [{"id": 1, "name": "defect"}],
    }))
    pool = tmp_path / "pool"
    pool.mkdir()
    defect_spec = tmp_path / "defect.jsonl"
    checkpoint = tmp_path / "adapter.pt"
    recipe = tmp_path / "recipe.yaml"
    defect_spec.write_text(json.dumps({"defect_type": "texture+crack"}) + "\n")
    checkpoint.write_bytes(b"adapter")
    recipe.write_text("anomaly_types: [[texture, crack]]\n")
    policy = tmp_path / "policy.yaml"
    policy.write_text(yaml.safe_dump({
        "sources": {"kpi": {"images": str(images), "coco": str(coco)}},
        "retrieval": {"model": "SigLIP", "model_path": "siglip",
                      "candidate_overfetch": 15},
        "synthesis": {"enabled": True, "pool_dataset_root": str(pool),
                      "fn_selection": {"mode": "all_eligible"},
                      "defect_spec": str(defect_spec),
                      "routes": {"route": {"checkpoint": str(checkpoint),
                                             "recipe": str(recipe)}},
                      "max_neighbors_per_fn": 5, "min_similarity": 0.9,
                      "amp_model_id": "nvidia/Cosmos3-Nano"},
    }))
    gaps = tmp_path / "strict.parquet"
    pd.DataFrame([
        {"image_id": 1, "filepath": str(routed_image), "gap_type": "FN",
         "bbox": [1, 2, 4, 6], "class": "defect"},
        {"image_id": 2, "filepath": str(unrouted_image), "gap_type": "FN",
         "bbox": [5, 6, 12, 14], "class": "defect"},
    ]).to_parquet(gaps)

    report = MODULE.prepare(policy, gaps, tmp_path / "out")

    assert report["fn_count"] == 1
    assert report["skipped_unrouted_by_dataset"] == {"boxes_only": 1}
    normalized = pd.read_parquet(tmp_path / "out/normalized_fn_gaps.parquet")
    assert list(normalized.dataset_id) == ["route"]

    coco_value = json.loads(coco.read_text())
    coco_value["annotations"][0].pop("fn_mask_source")
    coco.write_text(json.dumps(coco_value))
    with pytest.raises(ValueError, match="fn_mask_source"):
        MODULE.prepare(policy, gaps, tmp_path / "out-missing-routed-mask")

    coco_value["annotations"][0]["fn_mask_source"] = str(mask)
    coco_value["images"][0].pop("dataset_id")
    coco.write_text(json.dumps(coco_value))
    with pytest.raises(ValueError, match="real nonempty dataset_id.*allowlist"):
        MODULE.prepare(policy, gaps, tmp_path / "out-missing-dataset-id")

    coco_value["images"][0]["dataset_id"] = "boxes_only"
    coco.write_text(json.dumps(coco_value))
    skipped = MODULE.prepare(policy, gaps, tmp_path / "out-all-unrouted")
    assert skipped == {
        "status": "SKIPPED", "reason": "no_routed_false_negatives", "fn_count": 0,
        "candidate_topn": 15,
        "candidate_topn_source": "retrieval.candidate_overfetch",
        "skipped_unrouted_fn_count": 2,
        "skipped_unrouted_by_dataset": {"boxes_only": 2},
        "observed_dataset_ids": ["boxes_only"], "configured_route_keys": ["route"],
        "message": ("Skipped synthesis: all 2 strict false negatives use unrouted "
                    "dataset IDs ['boxes_only']; configured synthesis routes are ['route']."),
        "config": "",
    }


def test_image_index_rejects_duplicate_filename_stems() -> None:
    with pytest.raises(ValueError, match="duplicate KPI image identity: shared"):
        MODULE._image_index([
            {"id": 1, "file_name": "line-a/shared.png"},
            {"id": 2, "file_name": "line-b/shared.jpg"},
        ])


def test_generated_plan_uses_fraction_of_combined_total(tmp_path: Path) -> None:
    real_coco = tmp_path / "real.json"
    real_coco.write_text(json.dumps({
        "images": [{"id": index, "deft_kind": "real_defect"}
                   for index in range(2828)]
    }))
    rows = ([{"anomaly_type": "metal+crack"}] * 500
            + [{"anomaly_type": "metal+dent"}] * 500)
    output = tmp_path / "out"
    output.mkdir()

    plan, contract, evidence = MODULE._generated_plan(
        {"cumulative_fraction_of_total_defects": 0.25,
         "fn_selection": {"mode": "generated_per_type_plan"},
         "max_neighbors_per_fn": 1},
        rows, real_coco, output, 1,
    )

    assert plan == {"metal+crack": 472, "metal+dent": 470}
    assert evidence["cumulative_synthetic_limit"] == 942
    assert evidence["planned_images"] == 942
    assert evidence["selected_fn_count"] == 471
    assert contract is not None and contract["generated"] is True
    assert contract["sha256"] == hashlib.sha256(
        (output / "synthetic_plan.json").read_bytes()
    ).hexdigest()


def test_runtime_generated_plan_selects_deterministically_and_is_reported(
    tmp_path: Path,
) -> None:
    images = tmp_path / "kpi"
    images.mkdir()
    mask = tmp_path / "mask.png"
    mask.write_bytes(b"mask")
    coco_images, annotations, gaps = [], [], []
    cases = [("10", "crack"), ("2", "crack"), ("1", "crack"), ("3", "dent")]
    for index, (image_id, defect_class) in enumerate(cases, start=1):
        image = images / f"image-{image_id}.png"
        image.write_bytes(b"image")
        coco_images.append({"id": image_id, "file_name": image.name,
                            "dataset_id": "route", "texture_id": "metal"})
        annotations.append({"id": index, "image_id": image_id, "category_id": 1,
                            "bbox": [index, 2, 4, 5], "defect_class": defect_class,
                            "fn_mask_source": str(mask)})
        gaps.append({"image_id": image_id, "filepath": str(image), "gap_type": "FN",
                     "bbox": [index, 2, index + 4, 7], "class": "defect"})
    coco = tmp_path / "kpi.json"
    coco.write_text(json.dumps({"images": coco_images, "annotations": annotations,
                                "categories": [{"id": 1, "name": "defect"}]}))
    real_coco = tmp_path / "real.json"
    real_coco.write_text(json.dumps({
        "images": [{"id": index, "deft_kind": "real_defect"}
                   for index in range(18)]
    }))
    pool = tmp_path / "pool"
    pool.mkdir()
    defect_spec = tmp_path / "defect.jsonl"
    checkpoint = tmp_path / "adapter.pt"
    recipe = tmp_path / "recipe.yaml"
    defect_spec.write_text("{}\n")
    checkpoint.write_bytes(b"adapter")
    recipe.write_text("anomaly_types: []\n")
    policy = tmp_path / "policy.yaml"
    policy.write_text(yaml.safe_dump({
        "sources": {"kpi": {"images": str(images), "coco": str(coco)}},
        "retrieval": {"model": "SigLIP", "model_path": "siglip",
                      "candidate_overfetch": 3},
        "synthesis": {"enabled": True, "pool_dataset_root": str(pool),
                      "defect_spec": str(defect_spec),
                      "routes": {"route": {"checkpoint": str(checkpoint),
                                             "recipe": str(recipe)}},
                      "max_neighbors_per_fn": 1, "min_similarity": 0.9,
                      "amp_model_id": "nvidia/Cosmos3-Nano",
                      "cumulative_fraction_of_total_defects": 0.25,
                      },
    }))
    strict = tmp_path / "strict.parquet"
    pd.DataFrame(gaps).to_parquet(strict)

    report = MODULE.prepare(
        policy, strict, tmp_path / "out", iteration=1, real_coco=real_coco
    )

    selected = pd.read_parquet(tmp_path / "out/normalized_fn_gaps.parquet")
    assert selected.image_id.astype(str).tolist() == ["1", "10", "3"]
    assert report["eligible_fn_count"] == 4
    assert report["fn_count"] == 3
    assert report["requested_images"] == 6
    assert report["frozen_generator_rows"] == 6
    assert "bounded_shortfall" not in report
    assert all("bounded_shortfall" not in row for row in report["per_type"].values())
    assert report["synthetic_plan"]["counts"] == {
        "metal+crack": 4, "metal+dent": 2,
    }
    config = yaml.safe_load((tmp_path / "out/anomalygen_filtering.yaml").read_text())
    assert config["synthetic_plan"] == report["synthetic_plan"]


def test_runtime_generated_plan_emits_typed_no_budget_skip(tmp_path: Path) -> None:
    images = tmp_path / "kpi"
    images.mkdir()
    image, mask = images / "image.png", tmp_path / "mask.png"
    image.write_bytes(b"image")
    mask.write_bytes(b"mask")
    coco = tmp_path / "kpi.json"
    coco.write_text(json.dumps({
        "images": [{"id": 1, "file_name": image.name, "dataset_id": "route",
                    "texture_id": "metal"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                         "bbox": [1, 2, 4, 5], "defect_class": "crack",
                         "fn_mask_source": str(mask)}],
        "categories": [{"id": 1, "name": "defect"}],
    }))
    real_coco = tmp_path / "real.json"
    real_coco.write_text(json.dumps({
        "images": [{"id": 1, "deft_kind": "real_defect"}],
    }))
    pool = tmp_path / "pool"
    pool.mkdir()
    defect_spec, checkpoint = tmp_path / "defect.jsonl", tmp_path / "adapter.pt"
    recipe = tmp_path / "recipe.yaml"
    defect_spec.write_text("{}\n")
    checkpoint.write_bytes(b"adapter")
    recipe.write_text("anomaly_types: []\n")
    policy = tmp_path / "policy.yaml"
    policy.write_text(yaml.safe_dump({
        "sources": {"kpi": {"images": str(images), "coco": str(coco)}},
        "retrieval": {"model": "SigLIP", "model_path": "siglip",
                      "candidate_overfetch": 3},
        "synthesis": {
            "enabled": True, "pool_dataset_root": str(pool),
            "defect_spec": str(defect_spec),
            "routes": {"route": {"checkpoint": str(checkpoint),
                                  "recipe": str(recipe)}},
            "fn_selection": {"mode": "generated_per_type_plan"},
            "max_neighbors_per_fn": 1, "min_similarity": 0.9,
            "amp_model_id": "nvidia/Cosmos3-Nano",
            "cumulative_fraction_of_total_defects": 0.25,
        },
    }))
    strict = tmp_path / "strict.parquet"
    pd.DataFrame([{"image_id": 1, "filepath": str(image), "gap_type": "FN",
                   "bbox": [1, 2, 5, 7], "class": "defect"}]).to_parquet(strict)

    report = MODULE.prepare(
        policy, strict, tmp_path / "out", iteration=1, real_coco=real_coco
    )

    assert report["status"] == "SKIPPED"
    assert report["reason"] == "no_synthetic_budget"
    assert report["eligible_fn_count"] == 1
    assert report["planning"]["images_per_fn"] == 2
    assert report["planning"]["planned_images"] == 0
    assert not (tmp_path / "out/anomalygen_filtering.yaml").exists()


def test_selection_contract_rejects_unknown_modes() -> None:
    with pytest.raises(ValueError, match="unsupported synthesis.fn_selection.mode"):
        MODULE._selection_contract({"fn_selection": {"mode": "external_plan"}}, 1)


def test_generated_plan_derives_yield_from_retained_neighbors(tmp_path: Path) -> None:
    real_coco = tmp_path / "real.json"
    real_coco.write_text(json.dumps({
        "images": [{"id": index, "deft_kind": "real_defect"}
                   for index in range(24)]
    }))
    output = tmp_path / "out"
    output.mkdir()

    plan, contract, evidence = MODULE._generated_plan(
        {"cumulative_fraction_of_total_defects": 0.2,
         "fn_selection": {"mode": "generated_per_type_plan"},
         "max_neighbors_per_fn": 3},
        [{"anomaly_type": "metal+crack"}] * 4,
        real_coco, output, 1,
    )

    assert plan == {"metal+crack": 6}
    assert contract is not None and contract["images_per_fn"] == 6
    assert evidence["images_per_fn"] == 6
    assert evidence["selected_fn_count"] == 1


def test_default_policy_uses_generated_per_type_plan() -> None:
    policy = yaml.safe_load(
        (Path(__file__).parents[2] / "assets" / "default_policy.yaml").read_text()
    )

    assert policy["synthesis"]["fn_selection"] == {
        "mode": "generated_per_type_plan",
    }
    assert policy["retrieval"]["candidate_overfetch"] == 15
    assert policy["retrieval"]["round_robin_refill_overfetch"] == [
        5, 15, 50, 200, 100000,
    ]
    assert policy["synthesis"]["candidate_topn"] == 3
    assert policy["synthesis"]["max_neighbors_per_fn"] == 1
    assert policy["synthesis"]["cumulative_fraction_of_total_defects"] == 0.25
    assert "cumulative_fraction_of_real_defects" not in policy["synthesis"]
