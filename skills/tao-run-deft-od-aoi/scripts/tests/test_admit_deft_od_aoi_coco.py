# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image


SCRIPT = Path(__file__).parents[1] / "admit_deft_od_aoi_coco.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("admit_deft_od_aoi_coco", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)
SELECTION_SCRIPT = SCRIPT.parent / "deft_od_aoi_round_robin_selection.py"
SELECTION_SPEC = importlib.util.spec_from_file_location(
    "deft_od_aoi_round_robin_selection_test", SELECTION_SCRIPT
)
SELECTION_MODULE = importlib.util.module_from_spec(SELECTION_SPEC)
assert SELECTION_SPEC.loader
SELECTION_SPEC.loader.exec_module(SELECTION_MODULE)
COMMIT_SPEC = importlib.util.spec_from_file_location(
    "commit_deft_od_aoi_stage_test", SCRIPT.parent / "commit_deft_od_aoi_stage.py"
)
COMMIT_MODULE = importlib.util.module_from_spec(COMMIT_SPEC)
assert COMMIT_SPEC.loader
COMMIT_SPEC.loader.exec_module(COMMIT_MODULE)


def _fixture(root: Path, similarity: float = 1.0) -> tuple[Path, Path, Path]:
    candidate_root, retrieval_root = root / "candidates", root / "retrieval"
    candidate_root.mkdir()
    retrieval_root.mkdir()
    sources = {}
    for role in ("real", "clean"):
        image = root / f"{role}.png"
        Image.fromarray(np.full((16, 16), 80, dtype=np.uint8)).save(image)
        annotations = ([{"id": 5, "image_id": 1, "category_id": 1,
                         "bbox": [1, 1, 4, 4], "area": 16}] if role == "real" else [])
        coco = root / f"{role}.json"
        coco.write_text(json.dumps({"images": [{"id": 1, "file_name": image.name,
                                                 "source_path": str(image)}],
                                    "annotations": annotations,
                                    "categories": [{"id": 1, "name": "defect"}]}))
        sources[role] = {"images": str(root), "coco": str(coco)}
        crop = f"/{role}-crop.png"
        pd.DataFrame([{"filepath": crop, "source_filepath": str(image),
                       "source_image_id": 1, "embedding": [similarity, 1.0 - similarity]}]).to_parquet(
            candidate_root / f"{role}_candidate_embeddings.parquet"
        )
        reason = "fn" if role == "real" else "background_fp"
        pd.DataFrame([{"filepath": f"/{role}-query.png", "embedding": [1.0, 0.0],
                       "reason": reason}]).to_parquet(
            retrieval_root / f"{role}_query_embeddings.parquet"
        )
        mine = retrieval_root / f"mine_{role}"
        mine.mkdir()
        pd.DataFrame([{"filepath": crop}]).to_parquet(mine / "final_unique_files.parquet")
    policy = root / "policy.yaml"
    policy.write_text(yaml.safe_dump({"sources": sources,
                                      "retrieval": {
                                          "selection": {"strategy": "max_similarity"},
                                          "minimum_similarity": 0.5,
                                      },
                                      "routing": {"clean_cumulative_cap_per_real": 1.0},
                                      "admission": {"minimum_box_area_px": 4,
                                                    "maximum_box_aspect": 25.0},
                                      "synthesis": {"cumulative_fraction_of_real_defects": 1.0}}))
    (retrieval_root / "query_manifest.json").write_text(
        json.dumps({"status": "COMPLETE", "iteration": 1,
                    "query_counts": {"real": 1, "clean": 1},
                    "enabled_roles": ["real", "clean"],
                    "admission_targets": {"real": {"fn": 1, "near_miss_fp": 0},
                                          "clean": {"background_fp": 1}},
                    "requested_crop_counts": {"real": 15, "clean": 15}})
    )
    return policy, candidate_root, retrieval_root


def test_admission_deduplicates_sources_and_preserves_explicit_clean(tmp_path: Path) -> None:
    policy, candidates, retrieval = _fixture(tmp_path)
    report = MODULE.admit(policy, candidates, retrieval, tmp_path / "out", None, "copy")
    assert report["admitted"] == {"real": 1, "clean": 1, "synthetic": 0}
    assert report["retrieval_admission"] == {
        "real": {"requested": 1, "deduplicated": 1, "capped": 1},
        "clean": {"requested": 1, "deduplicated": 1, "capped": 1},
    }
    assert report["by_kind"] == {"real_defect": 1, "clean_negative": 1,
                                 "synthetic_defect": 0}
    preview = json.loads((tmp_path / "out/admission_preview.json").read_text())
    assert preview["roles"]["real"]["branches"]["fn"] == {
        "desired_parents": 1, "mined_crops": 1, "unique_parents": 1,
        "novel_parents": 1, "selected_parents": 1, "shortfall_parents": 0,
        "quota_met": True,
    }
    assert preview["roles"]["real"]["per_dataset"] == {"unknown": 1}
    coco = json.loads((tmp_path / "out/train.json").read_text())
    assert len(coco["images"]) == 2 and len(coco["annotations"]) == 1
    clean_id = next(row["id"] for row in coco["images"] if row["deft_kind"] == "clean_negative")
    assert all(row["image_id"] != clean_id for row in coco["annotations"])


def test_clean_admission_uses_cumulative_real_capacity_after_real_mining_exhausts(
        tmp_path: Path) -> None:
    policy, candidates, retrieval = _fixture(tmp_path)
    extra_clean = tmp_path / "clean-2.png"
    extra_clean.write_bytes(b"clean-2")
    clean_coco = tmp_path / "clean.json"
    clean_document = json.loads(clean_coco.read_text())
    clean_document["images"].append({
        "id": 2, "file_name": extra_clean.name, "source_path": str(extra_clean)})
    clean_coco.write_text(json.dumps(clean_document))
    clean_embeddings = candidates / "clean_candidate_embeddings.parquet"
    embedded = pd.read_parquet(clean_embeddings)
    embedded.loc[len(embedded)] = {
        "filepath": "/clean-crop-2.png", "source_filepath": str(extra_clean),
        "source_image_id": 2, "embedding": [1.0, 0.0],
    }
    embedded.to_parquet(clean_embeddings, index=False)
    mined = retrieval / "mine_clean/final_unique_files.parquet"
    pd.DataFrame({"filepath": ["/clean-crop.png", "/clean-crop-2.png"]}).to_parquet(
        mined, index=False)
    (retrieval / "query_manifest.json").write_text(
        json.dumps({"iteration": 2, "enabled_roles": ["clean"]})
    )
    previous = tmp_path / "previous.json"
    previous.write_text(json.dumps({
        "images": [{"id": 1, "file_name": "real.png",
                    "source_path": str(tmp_path / "real.png"),
                    "width": 16, "height": 16, "deft_kind": "real_defect"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                         "bbox": [1, 1, 4, 4], "area": 16}],
        "categories": [{"id": 1, "name": "defect"}],
    }))

    report = MODULE.admit(
        policy, candidates, retrieval, tmp_path / "out", previous, "copy")

    assert len(pd.read_parquet(mined)) == 2
    assert report["admitted"] == {"real": 0, "clean": 1, "synthetic": 0}
    assert report["retrieval_admission"] == {
        "real": {"requested": 0, "deduplicated": 0, "capped": 0},
        "clean": {"requested": 1, "deduplicated": 1, "capped": 1},
    }
    assert report["by_kind"] == {"real_defect": 1, "clean_negative": 1,
                                 "synthetic_defect": 0}


def test_admission_reports_requested_deduplicated_and_capped_role_counts(
        tmp_path: Path) -> None:
    policy, candidates, retrieval = _fixture(tmp_path)
    clean_coco = tmp_path / "clean.json"
    clean_document = json.loads(clean_coco.read_text())
    embedded = pd.read_parquet(candidates / "clean_candidate_embeddings.parquet")
    mined = pd.read_parquet(retrieval / "mine_clean/final_unique_files.parquet")
    for image_id in (2, 3):
        image = tmp_path / f"clean-{image_id}.png"
        image.write_bytes(f"clean-{image_id}".encode())
        clean_document["images"].append({
            "id": image_id, "file_name": image.name, "source_path": str(image),
        })
        embedded.loc[len(embedded)] = {
            "filepath": f"/clean-crop-{image_id}.png",
            "source_filepath": str(image), "source_image_id": image_id,
            "embedding": [1.0, 0.0],
        }
        mined.loc[len(mined)] = {"filepath": f"/clean-crop-{image_id}.png"}
    clean_coco.write_text(json.dumps(clean_document))
    embedded.to_parquet(candidates / "clean_candidate_embeddings.parquet", index=False)
    mined.to_parquet(retrieval / "mine_clean/final_unique_files.parquet", index=False)
    manifest = json.loads((retrieval / "query_manifest.json").read_text())
    manifest["admission_targets"]["clean"]["background_fp"] = 3
    (retrieval / "query_manifest.json").write_text(json.dumps(manifest))

    report = MODULE.admit(policy, candidates, retrieval, tmp_path / "out", None, "copy")

    assert report["retrieval_admission"] == {
        "real": {"requested": 1, "deduplicated": 1, "capped": 1},
        "clean": {"requested": 3, "deduplicated": 3, "capped": 1},
    }


def test_real_only_policy_without_synthesis_reports_zero_capacity(tmp_path: Path) -> None:
    policy, candidates, retrieval = _fixture(tmp_path)
    value = yaml.safe_load(policy.read_text())
    value.pop("synthesis")
    policy.write_text(yaml.safe_dump(value))

    report = MODULE.admit(policy, candidates, retrieval, tmp_path / "out", None, "copy")

    assert report["admitted"] == {"real": 1, "clean": 1, "synthetic": 0}
    assert report["synthetic_admission"]["fraction_basis"] == "disabled"
    assert report["synthetic_admission"]["cumulative_synthetic_limit"] == 0
    assert report["synthetic_admission"]["available_room_before_admission"] == 0
    assert report["warnings"] == []


def test_round_robin_admission_consumes_materialized_selection(tmp_path: Path) -> None:
    policy, candidates, retrieval = _fixture(tmp_path)
    value = yaml.safe_load(policy.read_text())
    value["retrieval"].update({
        "selection": {"strategy": "round_robin_similarity"},
        "candidate_overfetch": 2, "audit_top_k_per_query": 20,
    })
    value["routing"].update({
        "real_mine_factor_min": 1, "near_miss_real_factor": 2,
        "near_miss_real_cap_per_pocket": 20, "clean_factor": 2,
    })
    policy.write_text(yaml.safe_dump(value))
    for role in ("real", "clean"):
        candidate_path = candidates / f"{role}_candidate_embeddings.parquet"
        frame = pd.read_parquet(candidate_path)
        frame["candidate_id"] = f"{role}-candidate"
        frame.to_parquet(candidate_path, index=False)
    real_queries = pd.read_parquet(retrieval / "real_query_embeddings.parquet")
    real_queries = real_queries.assign(
        query_id="real-query", reason="fn", dataset_id="line-a", texture_id="board",
        defect_class="bridge", real_factor=1,
    )
    real_queries.to_parquet(retrieval / "real_query_embeddings.parquet", index=False)
    clean_queries = pd.read_parquet(retrieval / "clean_query_embeddings.parquet")
    clean_queries = clean_queries.assign(query_id="clean-query", reason="background_fp")
    clean_queries.to_parquet(retrieval / "clean_query_embeddings.parquet", index=False)
    for role in ("real", "clean"):
        (retrieval / f"mine_{role}/final_unique_files.parquet").unlink()
    selection = SELECTION_MODULE.materialize(policy, candidates, retrieval)

    report = MODULE.admit(
        policy, candidates, retrieval, tmp_path / "out", None, "copy"
    )

    assert report["selection_strategy"] == "round_robin_similarity"
    assert report["admitted"] == {"real": 1, "clean": 1, "synthetic": 0}
    assert [row["admitted"] for row in report["selection_audit"]["branches"]] == [1, 1]
    assert selection["selected_counts"] == {"real": 1, "clean": 1}
    assert report["retrieval_admission"] == {
        "real": {"requested": 1, "deduplicated": 1, "capped": 1},
        "clean": {"requested": 1, "deduplicated": 1, "capped": 1},
    }
    assert (tmp_path / "out/admission_index.npy").is_file()
    assert report["selection_admission_counters"]["admitted"] == 2


def test_empty_round_robin_selection_converges_after_admission(tmp_path: Path) -> None:
    policy, candidates, retrieval = _fixture(tmp_path)
    value = yaml.safe_load(policy.read_text())
    value["retrieval"].update({
        "selection": {"strategy": "round_robin_similarity"},
        "minimum_similarity": 1.1,
        "candidate_overfetch": 2,
        "audit_top_k_per_query": 20,
    })
    value["routing"].update({
        "real_mine_factor_min": 1,
        "near_miss_real_factor": 2,
        "near_miss_real_cap_per_pocket": 20,
        "clean_factor": 2,
    })
    policy.write_text(yaml.safe_dump(value))
    manifest = json.loads((retrieval / "query_manifest.json").read_text())
    manifest["selection_strategy"] = "round_robin_similarity"
    manifest["role_status"] = {
        role: {
            "status": "READY", "query_count": 1, "candidate_count": 1,
            "excluded_count": 0, "remaining_candidate_count": 1,
        }
        for role in ("real", "clean")
    }
    manifest["converged"] = False
    manifest["synthesis_pending"] = False
    (retrieval / "query_manifest.json").write_text(json.dumps(manifest))
    for role in ("real", "clean"):
        pd.DataFrame({"filepath": pd.Series(dtype="str")}).to_parquet(
            retrieval / f"exclude_{role}_candidates.parquet", index=False
        )
        candidate_path = candidates / f"{role}_candidate_embeddings.parquet"
        frame = pd.read_parquet(candidate_path)
        frame["candidate_id"] = f"{role}-candidate"
        frame.to_parquet(candidate_path, index=False)
        embeddings = retrieval / f"{role}_query_embeddings.parquet"
        pd.read_parquet(embeddings)[["filepath"]].to_parquet(
            retrieval / f"{role}_queries.parquet", index=False
        )
    real_queries = pd.read_parquet(retrieval / "real_query_embeddings.parquet")
    real_queries = real_queries.assign(
        query_id="real-query", reason="fn", dataset_id="line-a", texture_id="board",
        defect_class="bridge", real_factor=1,
    )
    real_queries.to_parquet(retrieval / "real_query_embeddings.parquet", index=False)
    clean_queries = pd.read_parquet(retrieval / "clean_query_embeddings.parquet")
    clean_queries = clean_queries.assign(query_id="clean-query", reason="background_fp")
    clean_queries.to_parquet(retrieval / "clean_query_embeddings.parquet", index=False)
    for role in ("real", "clean"):
        (retrieval / f"mine_{role}/final_unique_files.parquet").unlink()

    selection = SELECTION_MODULE.materialize(policy, candidates, retrieval)
    assert selection["selected_counts"] == {"real": 0, "clean": 0}
    artifacts = [
        f"query_manifest={retrieval / 'query_manifest.json'}",
        f"selection_report={retrieval / 'round_robin_selection_report.json'}",
        f"admission_index={retrieval / 'round_robin_admission_index.npy'}",
    ]
    for role in ("real", "clean"):
        exclusions = retrieval / f"exclude_{role}_candidates.parquet"
        artifacts.extend((
            f"{role}_queries={retrieval / f'{role}_queries.parquet'}",
            f"{role}_exclusions={exclusions}",
            f"{role}_query_embeddings={retrieval / f'{role}_query_embeddings.parquet'}",
            f"{role}_mined={retrieval / f'mine_{role}/final_unique_files.parquet'}",
        ))
    state = tmp_path / "deft_state.json"
    state.write_text(json.dumps({
        "status": "RUNNING", "next_stage": "iteration_retrieval",
        "current_iteration": 0, "last_stage": "baseline_gaps",
        "max_iterations": 2, "synthesis_enabled": False,
    }))
    result = COMMIT_MODULE.commit(state, "iteration_retrieval", 1, artifacts)
    assert result["next_stage"] == "iteration_admission"

    output = tmp_path / "empty-admission"
    report = MODULE.admit(policy, candidates, retrieval, output, None, "copy")
    assert report["role_status"] == {
        role: {"status": "NO_MATCHES", "selected_count": 0}
        for role in ("real", "clean")
    }
    assert report["new_training_images"] == 0
    result = COMMIT_MODULE.commit(state, "iteration_admission", 1, [
        f"admission_report={output / 'admission_report.json'}",
        f"admission_index={output / 'admission_index.npy'}",
    ])

    assert result["status"] == "COMPLETE"
    assert result["next_stage"] is None
    assert result["completion_reason"] == "retrieval_no_matches"
    assert all(event["stage"] != "iteration_training" for event in result["events"])


def test_admission_reports_parent_shortfall_without_failing(tmp_path: Path) -> None:
    policy, candidates, retrieval = _fixture(tmp_path)
    manifest = json.loads((retrieval / "query_manifest.json").read_text())
    manifest["admission_targets"]["real"]["fn"] = 2
    (retrieval / "query_manifest.json").write_text(json.dumps(manifest))

    report = MODULE.admit(policy, candidates, retrieval, tmp_path / "out", None, "copy")

    assert report["admitted"]["real"] == 1
    preview = json.loads((tmp_path / "out/admission_preview.json").read_text())
    assert preview["roles"]["real"]["branches"]["fn"] == {
        "desired_parents": 2, "mined_crops": 1, "unique_parents": 1,
        "novel_parents": 1, "selected_parents": 1, "shortfall_parents": 1,
        "quota_met": False,
    }


def test_admission_uses_overfetch_to_replace_a_previously_used_parent(tmp_path: Path) -> None:
    policy, candidates, retrieval = _fixture(tmp_path)
    first, second = tmp_path / "real.png", tmp_path / "real-second.png"
    second.write_bytes(b"second")
    document = json.loads((tmp_path / "real.json").read_text())
    document["images"].append({"id": 2, "file_name": second.name, "source_path": str(second),
                               "deft_od_aoi": {
                                   "dataset_id": "canonical-dataset-b",
                                    "benchmark": "source-dataset-b",
                               }})
    document["annotations"].append({"id": 6, "image_id": 2, "category_id": 1,
                                    "bbox": [1, 1, 4, 4], "area": 16})
    (tmp_path / "real.json").write_text(json.dumps(document))
    pd.DataFrame([
        {"filepath": "/real-crop-a.png", "source_filepath": str(first),
         "source_image_id": 1, "embedding": [1.0, 0.0]},
        {"filepath": "/real-crop-b.png", "source_filepath": str(second),
         "source_image_id": 2, "embedding": [0.9, 0.1]},
    ]).to_parquet(candidates / "real_candidate_embeddings.parquet")
    pd.DataFrame([{"filepath": "/real-crop-a.png"},
                  {"filepath": "/real-crop-b.png"}]).to_parquet(
        retrieval / "mine_real/final_unique_files.parquet"
    )
    manifest = json.loads((retrieval / "query_manifest.json").read_text())
    manifest["admission_targets"]["real"]["fn"] = 2
    (retrieval / "query_manifest.json").write_text(json.dumps(manifest))
    previous = tmp_path / "previous.json"
    previous.write_text(json.dumps({
        "images": [{"id": 1, "file_name": first.name, "source_path": str(first),
                    "deft_kind": "real_defect"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                         "bbox": [1, 1, 4, 4]}],
        "categories": [{"id": 1, "name": "defect"}],
    }))

    report = MODULE.admit(policy, candidates, retrieval, tmp_path / "out", previous, "copy")

    assert report["admitted"]["real"] == 1
    assert report["retrieval_admission"]["real"] == {
        "requested": 2, "deduplicated": 1, "capped": 1,
    }
    preview = json.loads((tmp_path / "out/admission_preview.json").read_text())
    assert preview["roles"]["real"]["branches"]["fn"]["unique_parents"] == 2
    assert preview["roles"]["real"]["branches"]["fn"]["novel_parents"] == 1
    assert preview["roles"]["real"]["per_dataset"] == {"canonical-dataset-b": 1}


def test_round_robin_rejects_empty_selection_without_required_schema(
        tmp_path: Path) -> None:
    mine = tmp_path / "mine_real"
    mine.mkdir()
    pd.DataFrame({"wrong_column": pd.Series(dtype="str")}).to_parquet(
        mine / "final_unique_files.parquet", index=False
    )

    with pytest.raises(ValueError, match="round-robin selection output is invalid"):
        MODULE._round_robin_selected("real", tmp_path)


@pytest.mark.parametrize(("synthesis_enabled", "prior_real"), [(False, 0), (True, 0), (True, 1)])
def test_empty_max_similarity_mining_routes_through_admission(
        tmp_path: Path, synthesis_enabled: bool, prior_real: int) -> None:
    policy, candidates, retrieval = _fixture(tmp_path)
    value = yaml.safe_load(policy.read_text())
    value["synthesis"]["enabled"] = synthesis_enabled
    policy.write_text(yaml.safe_dump(value))
    manifest = retrieval / "query_manifest.json"
    evidence = json.loads(manifest.read_text())
    evidence["selection_strategy"] = "max_similarity"
    evidence["role_status"] = {
        role: {
            "status": "READY", "query_count": 1, "candidate_count": 1,
            "excluded_count": 0, "remaining_candidate_count": 1,
        }
        for role in ("real", "clean")
    }
    evidence["converged"] = False
    evidence["synthesis_pending"] = synthesis_enabled
    manifest.write_text(json.dumps(evidence))
    artifacts = [f"query_manifest={manifest}"]
    for role in ("real", "clean"):
        queries = retrieval / f"{role}_queries.parquet"
        embeddings = retrieval / f"{role}_query_embeddings.parquet"
        pd.read_parquet(embeddings)[["filepath"]].to_parquet(queries, index=False)
        mined = retrieval / f"mine_{role}/final_unique_files.parquet"
        pd.DataFrame({"filepath": pd.Series(dtype="str")}).to_parquet(mined, index=False)
        exclusions = retrieval / f"{role}_exclusions.parquet"
        pd.DataFrame({"filepath": pd.Series(dtype="str")}).to_parquet(
            exclusions, index=False
        )
        artifacts.extend((f"{role}_queries={queries}",
                          f"{role}_exclusions={exclusions}",
                          f"{role}_query_embeddings={embeddings}", f"{role}_mined={mined}"))
    state = tmp_path / "deft_state.json"
    state.write_text(json.dumps({
        "status": "RUNNING", "next_stage": "iteration_retrieval",
        "current_iteration": 0, "last_stage": "baseline_gaps",
        "max_iterations": 2, "synthesis_enabled": synthesis_enabled,
    }))

    result = COMMIT_MODULE.commit(state, "iteration_retrieval", 1, artifacts)
    assert result["next_stage"] == "iteration_admission"
    output = tmp_path / "admission"
    previous = None
    if prior_real:
        previous = tmp_path / "previous.json"
        document = json.loads((tmp_path / "real.json").read_text())
        document["images"][0]["deft_kind"] = "real_defect"
        previous.write_text(json.dumps(document))
    report = MODULE.admit(policy, candidates, retrieval, output, previous, "copy")
    assert report["new_training_images"] == 0
    assert report["role_status"] == {
        role: {"status": "NO_MATCHES", "selected_count": 0} for role in ("real", "clean")
    }
    result = COMMIT_MODULE.commit(
        state, "iteration_admission", 1, [f"admission_report={output / 'admission_report.json'}"]
    )
    if synthesis_enabled and prior_real:
        assert result["status"] == "RUNNING"
        assert result["next_stage"] == "iteration_synthesis"
        generated = tmp_path / "generated.json"
        generated.write_text(json.dumps({
            "images": [], "annotations": [], "categories": [{"id": 1, "name": "defect"}],
        }))
        post_synthesis = tmp_path / "post_synthesis"
        MODULE.admit(policy, candidates, retrieval, post_synthesis, previous, "copy",
                     generated, tmp_path)
        generation_report = tmp_path / "generation_report.json"
        generation_report.write_text(json.dumps({
            "status": "COMPLETE", "generated": 0,
            "groups": [{"requested": 1, "generated": 0, "guardrail_blocked": 1}],
        }))
        result = COMMIT_MODULE.commit(state, "iteration_synthesis", 1, [
            f"generation_report={generation_report}",
            f"admission_report={post_synthesis / 'admission_report.json'}",
        ])
    assert result["status"] == "COMPLETE"
    assert result["next_stage"] is None
    assert result["completion_reason"] == "retrieval_no_matches"
    assert all(event["stage"] != "iteration_training" for event in result["events"])


def test_max_similarity_rejects_empty_mining_without_filepath(tmp_path: Path) -> None:
    policy, candidates, retrieval = _fixture(tmp_path)
    for role in ("real", "clean"):
        pd.DataFrame({"wrong_column": []}).to_parquet(
            retrieval / f"mine_{role}/final_unique_files.parquet", index=False
        )
    with pytest.raises(ValueError, match="lacks filepath"):
        MODULE.admit(policy, candidates, retrieval, tmp_path / "out", None, "copy")


@pytest.mark.parametrize(("prior_real", "prior_synthetic", "new_real", "room"), [
    (1, 1, False, 0), (1, 0, False, 1), (0, 0, True, 1), (0, 0, False, 0),
])
def test_capacity_gate_uses_actual_cumulative_admission(
        tmp_path: Path, prior_real: int, prior_synthetic: int, new_real: bool, room: int) -> None:
    policy, candidates, retrieval = _fixture(tmp_path)
    value = yaml.safe_load(policy.read_text())
    value["synthesis"]["enabled"] = True
    policy.write_text(yaml.safe_dump(value))
    for role in ("real", "clean"):
        if role == "clean" or not new_real:
            pd.DataFrame({"filepath": []}).to_parquet(
                retrieval / f"mine_{role}/final_unique_files.parquet", index=False
            )
    images = []
    for kind, count in (("real_defect", prior_real), ("synthetic_defect", prior_synthetic)):
        if count:
            source = tmp_path / f"previous-{kind}.png"
            Image.new("RGB", (16, 16)).save(source)
            images.append({"id": len(images) + 1, "file_name": source.name,
                           "source_path": str(source), "deft_kind": kind})
    previous = tmp_path / "previous.json"
    previous.write_text(json.dumps({
        "images": images, "annotations": [], "categories": [{"id": 1, "name": "defect"}],
    }))
    output = tmp_path / "admitted"
    report = MODULE.admit(policy, candidates, retrieval, output, previous, "copy")
    assert report["synthetic_admission"]["available_room_before_admission"] == room
    assert report["new_training_images"] == int(new_real)
    state = tmp_path / "state.json"
    state.write_text(json.dumps({
        "status": "RUNNING", "next_stage": "iteration_admission",
        "current_iteration": 1, "synthesis_enabled": True,
    }))
    result = COMMIT_MODULE.commit(
        state, "iteration_admission", 1, [f"admission_report={output / 'admission_report.json'}"]
    )
    assert result["next_stage"] == ("iteration_synthesis" if room else None)


def test_max_similarity_records_no_matches_without_empty_pool_error(tmp_path: Path) -> None:
    policy, candidates, retrieval = _fixture(tmp_path, similarity=0.0)
    report = MODULE.admit(policy, candidates, retrieval, tmp_path / "out", None, "copy")

    assert report["role_status"] == {
        "real": {"status": "NO_MATCHES", "selected_count": 0},
        "clean": {"status": "NO_MATCHES", "selected_count": 0},
    }
    assert report["new_training_images"] == 0
    assert report["total_images"] == 0


def test_max_similarity_no_matches_does_not_count_retained_data_as_new(tmp_path: Path) -> None:
    policy, candidates, retrieval = _fixture(tmp_path, similarity=0.0)
    previous = tmp_path / "previous.json"
    previous.write_text(json.dumps({
        "images": [{
            "id": 1, "file_name": "real.png", "source_path": str(tmp_path / "real.png"),
            "width": 16, "height": 16, "deft_kind": "real_defect",
        }],
        "annotations": [{
            "id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 4, 4],
        }],
        "categories": [{"id": 1, "name": "defect"}],
    }))

    report = MODULE.admit(
        policy, candidates, retrieval, tmp_path / "out", previous, "copy"
    )

    assert report["retained_previous_images"] == 1
    assert report["total_images"] == 1
    assert report["new_training_images"] == 0


def test_no_real_data_warns_before_synthesis_generation(tmp_path: Path) -> None:
    policy, candidates, retrieval = _fixture(tmp_path, similarity=0.0)
    value = yaml.safe_load(policy.read_text())
    value["synthesis"]["enabled"] = True
    value["synthesis"]["cumulative_fraction_of_real_defects"] = 0.25
    policy.write_text(yaml.safe_dump(value))

    report = MODULE.admit(
        policy, candidates, retrieval, tmp_path / "out", None, "copy"
    )

    assert report["warnings"] == [{
        "code": "SYNTHETIC_ADMISSION_CAP_ZERO",
        "message": (
            "synthetic admission currently has zero room; generation may complete "
            "without admitting any synthetic images"
        ),
        "cumulative_real_images": 0,
        "configured_fraction": 0.25,
        "cumulative_synthetic_limit": 0,
        "synthetic_images_before_admission": 0,
    }]


def test_admission_folds_capped_synthetic_categories_to_defect(tmp_path: Path) -> None:
    policy, candidates, retrieval = _fixture(tmp_path)
    generated = tmp_path / "generated"
    generated.mkdir()
    image = generated / "synthetic.png"
    image.write_bytes(b"synthetic")
    coco = tmp_path / "synthetic.json"
    coco.write_text(json.dumps({"images": [{"id": 2, "file_name": image.name,
                                             "width": 16, "height": 16}],
                                "annotations": [{"id": 8, "image_id": 2, "category_id": 4,
                                                 "bbox": [1, 1, 3, 3]}],
                                "categories": [{"id": 4, "name": "texture+defect"}]}))
    report = MODULE.admit(policy, candidates, retrieval, tmp_path / "out", None, "copy",
                          coco, generated)
    assert report["admitted"]["synthetic"] == 1
    output = json.loads((tmp_path / "out/train.json").read_text())
    assert {row["category_id"] for row in output["annotations"]} == {1}


def test_flat_install_cli_accepts_explicit_synthetic_artifacts(tmp_path: Path) -> None:
    policy, candidates, retrieval = _fixture(tmp_path)
    generated = tmp_path / "generated"
    generated.mkdir()
    image = generated / "synthetic.png"
    image.write_bytes(b"synthetic")
    coco = generated / "synthetic.json"
    coco.write_text(json.dumps({
        "images": [{"id": 2, "file_name": image.name, "width": 16, "height": 16}],
        "annotations": [{"id": 8, "image_id": 2, "category_id": 4,
                         "bbox": [1, 1, 3, 3]}],
        "categories": [{"id": 4, "name": "texture+defect"}],
    }))
    flat_script = (
        tmp_path / "flat/skills/tao-run-deft-od-aoi/scripts/admit_deft_od_aoi_coco.py"
    )
    flat_script.parent.mkdir(parents=True)
    shutil.copy2(SCRIPT, flat_script)
    help_result = subprocess.run(
        [sys.executable, str(flat_script), "--help"],
        check=True, capture_output=True, text=True,
    )
    assert "--generation-root" not in help_result.stdout

    output = tmp_path / "out"
    completed = subprocess.run([
        sys.executable, str(flat_script),
        "--policy", str(policy),
        "--candidate-root", str(candidates),
        "--retrieval-root", str(retrieval),
        "--output-dir", str(output),
        "--synthetic-coco", str(coco),
        "--synthetic-images", str(generated),
    ], check=True, capture_output=True, text=True)

    assert json.loads(completed.stdout)["status"] == "COMPLETE"
    report = json.loads((output / "admission_report.json").read_text())
    assert report["admitted"] == {"clean": 1, "real": 1, "synthetic": 1}


def test_synthetic_quality_filter_and_proportional_allocation(tmp_path: Path) -> None:
    policy, candidates, retrieval = _fixture(tmp_path)
    extra = tmp_path / "extra-real.png"
    extra.write_bytes(b"extra")
    previous = tmp_path / "previous.json"
    previous.write_text(json.dumps({
        "images": [
            {"id": 1, "file_name": "real.png", "source_path": str(tmp_path / "real.png"),
             "width": 16, "height": 16, "deft_kind": "real_defect"},
            {"id": 2, "file_name": extra.name, "source_path": str(extra),
             "width": 16, "height": 16, "deft_kind": "real_defect"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 4, 4]},
            {"id": 2, "image_id": 2, "category_id": 1, "bbox": [1, 1, 4, 4]},
        ],
        "categories": [{"id": 1, "name": "defect"}],
    }))
    generated = tmp_path / "generated"
    generated.mkdir()
    images, annotations = [], []
    for index in range(6):
        name = f"synthetic-{index}.png"
        (generated / name).write_bytes(name.encode())
        images.append({"id": index + 1, "file_name": name, "width": 20, "height": 20,
                       "dataset_id": "line-a" if index < 3 else "line-b"})
        bbox = ([0, 0, 20, 20] if index == 0 else
                [1, 1, 1, 1] if index == 3 else [2, 2, 8, 8])
        annotations.append({"id": index + 1, "image_id": index + 1,
                            "category_id": 7, "bbox": bbox})
    synthetic = tmp_path / "synthetic.json"
    synthetic.write_text(json.dumps({"images": images, "annotations": annotations,
                                     "categories": [{"id": 7, "name": "defect-variant"}]}))

    report = MODULE.admit(policy, candidates, retrieval, tmp_path / "out", previous, "copy",
                          synthetic, generated, synthetic_only=True)

    admission = report["synthetic_admission"]
    assert admission["quality_filter"]["rejected_annotations_full_frame"] == 1
    assert admission["quality_filter"]["rejected_annotations_small"] == 1
    assert admission["requested_new"] == 4
    assert admission["admitted_new"] == 2
    assert admission["excluded_by_cap"] == 2
    assert admission["cumulative_real_images"] == 2
    assert admission["cumulative_synthetic_limit"] == 2
    assert admission["admitted_by_stratum"] == {"line-a": 1, "line-b": 1}
    preview = json.loads((tmp_path / "out/admission_preview.json").read_text())
    assert preview["mining_admission"] == "skipped" and preview["roles"] == {}
    assert report["warnings"] == [{
        "code": "SYNTHETIC_ADMISSION_CAPPED",
        "message": (
            "2 eligible generated synthetic images were excluded by "
            "the configured cumulative synthetic-admission cap"
        ),
        "cumulative_real_images": 2,
        "configured_fraction": 1.0,
        "cumulative_synthetic_limit": 2,
        "synthetic_images_before_admission": 0,
    }]


def test_synthetic_inputs_do_not_implicitly_disable_mining(tmp_path: Path) -> None:
    policy, candidates, retrieval = _fixture(tmp_path)
    previous = tmp_path / "previous.json"
    previous.write_text(json.dumps({
        "images": [{"id": 1, "file_name": "old.png", "source_path": str(tmp_path / "old.png"),
                    "deft_kind": "synthetic_defect"}],
        "annotations": [], "categories": [{"id": 1, "name": "defect"}],
    }))
    (tmp_path / "old.png").write_bytes(b"old")
    generated = tmp_path / "generated"
    generated.mkdir()
    image = generated / "synthetic.png"
    image.write_bytes(b"synthetic")
    synthetic = tmp_path / "synthetic.json"
    synthetic.write_text(json.dumps({
        "images": [{"id": 2, "file_name": image.name, "width": 16, "height": 16}],
        "annotations": [{"id": 2, "image_id": 2, "category_id": 1,
                         "bbox": [1, 1, 3, 3]}],
        "categories": [{"id": 1, "name": "defect"}],
    }))

    report = MODULE.admit(policy, candidates, retrieval, tmp_path / "out", previous, "copy",
                          synthetic, generated)

    assert report["admitted"]["real"] == 1
    preview = json.loads((tmp_path / "out/admission_preview.json").read_text())
    assert preview["mining_admission"] == "evaluated"


def test_admission_reports_missing_selected_source_cleanly(tmp_path: Path) -> None:
    policy, candidates, retrieval = _fixture(tmp_path)
    frame = pd.read_parquet(candidates / "real_candidate_embeddings.parquet")
    frame["source_filepath"] = str(tmp_path / "not-in-frozen-coco.png")
    frame.to_parquet(candidates / "real_candidate_embeddings.parquet", index=False)

    with pytest.raises(ValueError, match="selected real source is absent from its frozen COCO"):
        MODULE.admit(policy, candidates, retrieval, tmp_path / "out", None, "copy")


def test_synthetic_cap_selection_is_independent_of_coco_order(tmp_path: Path) -> None:
    policy, candidates, retrieval = _fixture(tmp_path)
    generated = tmp_path / "generated"
    generated.mkdir()
    images, annotations = [], []
    for index, name in enumerate(("z.png", "a.png", "m.png"), start=1):
        (generated / name).write_bytes(name.encode())
        images.append({"id": index, "file_name": name, "width": 16, "height": 16})
        annotations.append({"id": index, "image_id": index, "category_id": 1,
                            "bbox": [2, 2, 8, 8]})
    synthetic = tmp_path / "synthetic.json"

    def run(order: list[int], output: Path) -> str:
        synthetic.write_text(json.dumps({
            "images": [images[index] for index in order],
            "annotations": annotations,
            "categories": [{"id": 1, "name": "defect"}],
        }))
        MODULE.admit(policy, candidates, retrieval, output, None, "copy", synthetic, generated)
        coco = json.loads((output / "train.json").read_text())
        return next(row["source_path"] for row in coco["images"]
                    if row["deft_kind"] == "synthetic_defect")

    assert run([0, 1, 2], tmp_path / "out-a") == run([2, 0, 1], tmp_path / "out-b")


def test_total_fraction_limit_matches_historical_definition() -> None:
    assert MODULE._synthetic_limit(
        2828, {"cumulative_fraction_of_total_defects": 0.25}
    ) == (942, "fraction_of_total", 0.25)
    assert MODULE._synthetic_limit(
        2828, {"cumulative_fraction_of_real_defects": 0.25}
    ) == (707, "fraction_of_real", 0.25)
