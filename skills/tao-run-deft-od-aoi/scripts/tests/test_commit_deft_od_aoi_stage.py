# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT = Path(__file__).parents[1] / "commit_deft_od_aoi_stage.py"
SPEC = importlib.util.spec_from_file_location("commit_deft_od_aoi_stage", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def _state(root: Path) -> tuple[Path, Path]:
    state = root / "deft_state.json"
    state.write_text(json.dumps({"status": "READY", "next_stage": "candidate_cache",
                                 "current_iteration": 0, "max_iterations": 1}))
    artifact = root / "done.json"
    artifact.write_text("{}")
    return state, artifact


def _retrieval(root: Path, sparse: bool = False, exhausted: bool = False) -> list[str]:
    manifest = root / "query_manifest.json"
    counts = {"real": 1} if sparse else {"real": 1, "clean": 1}
    enabled = [] if exhausted else list(counts)
    role_status = {
        role: ({"status": "EXHAUSTED" if exhausted else "READY",
                "query_count": count, "candidate_count": 1,
                "excluded_count": 1 if exhausted else 0,
                "remaining_candidate_count": 0 if exhausted else 1}
               if count else {"status": "NO_QUERIES", "query_count": 0,
                              "candidate_count": 0, "excluded_count": 0,
                              "remaining_candidate_count": 0})
        for role, count in {"real": counts.get("real", 0),
                            "clean": counts.get("clean", 0)}.items()
    }
    manifest.write_text(json.dumps({"status": "COMPLETE", "iteration": 1,
                                    "query_counts": counts, "enabled_roles": enabled,
                                    "role_status": role_status,
                                    "converged": exhausted, "synthesis_pending": False}))
    artifacts = [f"query_manifest={manifest}"]
    for role, count in counts.items():
        queries = root / f"{role}_queries.parquet"
        exclusions = root / f"{role}_exclusions.parquet"
        embeddings = root / f"{role}_query_embeddings.parquet"
        mined = root / f"{role}_mined.parquet"
        pd.DataFrame({"filepath": [f"/{role}-query"] * count}).to_parquet(queries)
        pd.DataFrame({"filepath": ([f"/{role}-candidate"] if exhausted else [])}).to_parquet(
            exclusions, index=False
        )
        artifacts.extend((f"{role}_queries={queries}", f"{role}_exclusions={exclusions}"))
        if not exhausted:
            pd.DataFrame({"filepath": [f"/{role}-query"] * count,
                          "embedding": [[1.0, 0.0]] * count}).to_parquet(embeddings)
            pd.DataFrame({"filepath": [f"/{role}-candidate"]}).to_parquet(mined)
            artifacts.extend((f"{role}_query_embeddings={embeddings}",
                              f"{role}_mined={mined}"))
    return artifacts


def _iteration_artifacts(root: Path) -> dict[str, list[str]]:
    admission = root / "admission_report.json"
    admission.write_text(json.dumps({"status": "COMPLETE", "iteration": 1,
                                     "role_status": {
                                         "real": {"status": "SELECTED", "selected_count": 1},
                                     },
                                     "admitted": {"real": 1, "clean": 0, "synthetic": 0},
                                     "new_training_images": 1}))
    checkpoint = root / "model_epoch_001.pth"
    checkpoint.write_bytes(b"checkpoint")
    status = root / "train_status.json"
    status.write_text('{"epoch": 1, "kpi": {"val_mAP50": 0.75}}\n')
    selection = root / "checkpoint_selection.json"
    selection.write_text(json.dumps({
        "status": "COMPLETE", "action": "select", "best_epoch": 1,
        "best_kpi_mAP50": 0.75, "selected_checkpoint": str(checkpoint.resolve()),
        "status_files": [str(status.resolve())],
    }))
    roles = {}
    for role in ("kpi", "test"):
        predictions = root / role / "labels"
        predictions.mkdir(parents=True)
        (predictions / "image.txt").write_text("")
        roles[role] = {"expected_images": 1, "predictions": str(predictions)}
    measurement = root / "measurement_manifest.json"
    measurement.write_text(json.dumps({
        "status": "COMPLETE", "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": MODULE._sha(checkpoint), "inference_roles": roles,
    }))
    gap_artifacts = []
    for kind, confidence in (("loose", 0.3), ("strict", 0.8)):
        report = root / f"{kind}_gap_report.json"
        report.write_text(json.dumps({
            "kpi": f"kpi_{kind}", "counts_by_type": {"FN": 1},
            "counts_by_class": {"defect": {"FN": 1}},
            "settings": {"iou_threshold": 0.5, "conf_threshold": confidence,
                         "min_area": 0},
        }))
        boxes = root / f"{kind}_box_gaps.parquet"
        pd.DataFrame({"gap_type": ["FN"]}).to_parquet(boxes)
        gap_artifacts.extend((f"{kind}_gap_report={report}", f"{kind}_box_gaps={boxes}"))
    return {
        "iteration_admission": [f"admission_report={admission}"],
        "iteration_training": [f"checkpoint_selection={selection}"],
        "iteration_measurement": [f"measurement_manifest={measurement}"],
        "iteration_gaps": gap_artifacts,
    }


def _advance_to_synthesis(root: Path, *, retrieval_enabled: bool) -> Path:
    state, _ = _state(root)
    value = json.loads(state.read_text())
    value.update(status="RUNNING", synthesis_enabled=True,
                 next_stage="iteration_retrieval", current_iteration=0,
                 last_stage="baseline_gaps", events=[])
    state.write_text(json.dumps(value))
    if retrieval_enabled:
        retrieval = _retrieval(root, sparse=True)
    else:
        manifest = root / "query_manifest.json"
        no_queries = {
            "status": "NO_QUERIES", "query_count": 0, "candidate_count": 1,
            "excluded_count": 0, "remaining_candidate_count": 1,
        }
        manifest.write_text(json.dumps({
            "status": "COMPLETE", "iteration": 1,
            "query_counts": {"real": 0, "clean": 0}, "enabled_roles": [],
            "role_status": {"real": no_queries, "clean": no_queries},
            "converged": False, "synthesis_pending": True,
        }))
        retrieval = [f"query_manifest={manifest}"]
    MODULE.commit(state, "iteration_retrieval", 1, retrieval)
    admission = root / "admission_report.json"
    admission.write_text(json.dumps({"status": "COMPLETE", "iteration": 1,
                                     "role_status": {},
                                     "admitted": {"real": 0, "clean": 0,
                                                  "synthetic": 0},
                                     "synthetic_admission": {
                                         "available_room_before_admission": 1,
                                     },
                                     "new_training_images": 0}))
    MODULE.commit(state, "iteration_admission", 1, [f"admission_report={admission}"])
    return state


def _budget_skip(
        root: Path, admitted_real: int, *, round_robin: bool = False
) -> tuple[Path, list[str]]:
    state, _ = _state(root)
    value = json.loads(state.read_text())
    value.update(status="RUNNING", synthesis_enabled=True,
                 next_stage="iteration_admission", current_iteration=1,
                 last_stage="iteration_retrieval")
    state.write_text(json.dumps(value))
    admission = root / "admission_report.json"
    admission_report = {
        "status": "COMPLETE", "iteration": 1,
        "admitted": {"real": admitted_real, "clean": 0, "synthetic": 0},
        "new_training_images": admitted_real,
        "synthetic_admission": {"available_room_before_admission": 1},
        "role_status": {
            "real": {
                "status": "SELECTED" if admitted_real else "NO_MATCHES",
                "selected_count": admitted_real,
            },
        },
    }
    admission_artifacts = [f"admission_report={admission}"]
    if round_robin:
        index = root / "admission_index.npy"
        np.save(index, np.zeros((0, MODULE.ADMISSION_INDEX_WIDTH)))
        admission_report.update(
            selection_strategy="round_robin_similarity",
            admission_index=str(index),
        )
        admission_artifacts.append(f"admission_index={index}")
    admission.write_text(json.dumps(admission_report))
    MODULE.commit(state, "iteration_admission", 1, admission_artifacts)
    request = root / "synthesis_request.json"
    request.write_text(json.dumps({
        "status": "SKIPPED", "reason": "no_synthetic_budget",
        "selection_mode": "generated_per_type_plan", "fn_count": 0,
        "eligible_fn_count": 2,
        "planning": {
            "new_image_budget": 1, "images_per_fn": 2,
            "eligible_fn_count": 2, "selected_fn_count": 0,
            "planned_images": 0, "unplanned_budget": 1,
        },
    }))
    return state, [f"synthesis_request={request}", *admission_artifacts]


def test_commit_enforces_semantic_evidence_and_completes(tmp_path: Path) -> None:
    state, artifact = _state(tmp_path)
    iteration = _iteration_artifacts(tmp_path)
    stages = [("candidate_cache", 0), ("baseline_measurement", 0),
              ("baseline_gaps", 0), ("iteration_retrieval", 1),
              ("iteration_admission", 1), ("iteration_training", 1),
              ("iteration_measurement", 1), ("iteration_gaps", 1)]
    value = None
    for stage, number in stages:
        artifacts = (_retrieval(tmp_path) if stage == "iteration_retrieval" else
                     iteration.get(stage, [f"done={artifact}"]))
        value = MODULE.commit(state, stage, number, artifacts)
    assert value["status"] == "COMPLETE" and value["next_stage"] is None


def test_retrieval_accepts_omitted_zero_count_role(tmp_path: Path) -> None:
    state, _ = _state(tmp_path)
    value = json.loads(state.read_text())
    value.update(status="RUNNING", next_stage="iteration_retrieval",
                 current_iteration=0, last_stage="baseline_gaps")
    state.write_text(json.dumps(value))
    result = MODULE.commit(state, "iteration_retrieval", 1, _retrieval(tmp_path, sparse=True))
    assert result["next_stage"] == "iteration_admission"


def test_retrieval_commits_all_role_exhaustion_as_convergence(tmp_path: Path) -> None:
    state, _ = _state(tmp_path)
    value = json.loads(state.read_text())
    value.update(status="RUNNING", next_stage="iteration_retrieval",
                 current_iteration=0, last_stage="baseline_gaps")
    state.write_text(json.dumps(value))

    result = MODULE.commit(state, "iteration_retrieval", 1,
                           _retrieval(tmp_path, exhausted=True))

    assert result["status"] == "COMPLETE" and result["next_stage"] is None
    assert result["completion_reason"] == "mining_exhausted"


@pytest.mark.parametrize("invalid", [
    "missing_file", "corrupt_file", "empty_without_filepath", "rows_without_filepath",
    "failed_manifest", "query_count_mismatch",
])
def test_max_similarity_rejects_invalid_mining_evidence(tmp_path: Path, invalid: str) -> None:
    state, _ = _state(tmp_path)
    value = json.loads(state.read_text())
    value.update(status="RUNNING", next_stage="iteration_retrieval",
                 current_iteration=0, last_stage="baseline_gaps")
    state.write_text(json.dumps(value))
    artifacts = _retrieval(tmp_path, sparse=True)
    mined = tmp_path / "real_mined.parquet"
    message = "lacks filepath"
    if invalid == "missing_file":
        mined.unlink()
        message = "existing-file"
    elif invalid == "corrupt_file":
        mined.write_bytes(b"not a parquet file")
        message = "Parquet|parquet"
    elif invalid in {"empty_without_filepath", "rows_without_filepath"}:
        rows = [] if invalid == "empty_without_filepath" else ["/candidate"]
        pd.DataFrame({"wrong_column": rows}).to_parquet(mined, index=False)
    elif invalid == "failed_manifest":
        manifest = tmp_path / "query_manifest.json"
        report = json.loads(manifest.read_text())
        report["status"] = "ERROR"
        manifest.write_text(json.dumps(report))
        message = "incomplete"
    else:
        pd.DataFrame({"filepath": []}).to_parquet(tmp_path / "real_queries.parquet")
        message = "query count"
    original_state = state.read_bytes()
    with pytest.raises(ValueError, match=message):
        MODULE.commit(state, "iteration_retrieval", 1, artifacts)
    assert state.read_bytes() == original_state
    assert not (tmp_path / "loop_log.jsonl").exists()


def test_admission_index_uses_shared_shape_contract(tmp_path: Path) -> None:
    valid = tmp_path / "valid.npy"
    np.save(valid, np.zeros((0, MODULE.ADMISSION_INDEX_WIDTH)))
    MODULE._validate_admission_index({"path": valid})

    for name, index in (
        ("one-dimensional", np.zeros(MODULE.ADMISSION_INDEX_WIDTH)),
        ("wrong-width", np.zeros((1, MODULE.ADMISSION_INDEX_WIDTH - 1))),
    ):
        path = tmp_path / f"{name}.npy"
        np.save(path, index)
        with pytest.raises(ValueError, match="admission index shape"):
            MODULE._validate_admission_index({"path": path})


@pytest.mark.parametrize(
    ("synthesis_enabled", "expected_status", "expected_stage"),
    [(True, "RUNNING", "iteration_synthesis"), (False, "COMPLETE", None)],
)
def test_no_matches_routes_to_synthesis_or_converges(
        tmp_path: Path, synthesis_enabled: bool,
        expected_status: str, expected_stage: str | None) -> None:
    state, _ = _state(tmp_path)
    value = json.loads(state.read_text())
    value.update(status="RUNNING", next_stage="iteration_admission",
                 current_iteration=1, last_stage="iteration_retrieval",
                 synthesis_enabled=synthesis_enabled)
    state.write_text(json.dumps(value))
    report = tmp_path / "admission_report.json"
    report.write_text(json.dumps({
        "status": "COMPLETE", "iteration": 1,
        "role_status": {
            "real": {"status": "NO_MATCHES", "selected_count": 0},
            "clean": {"status": "NO_MATCHES", "selected_count": 0},
        },
        "admitted": {"real": 0, "clean": 0, "synthetic": 0},
        "new_training_images": 0,
        "synthetic_admission": {"available_room_before_admission": 1},
    }))

    result = MODULE.commit(
        state, "iteration_admission", 1, [f"admission_report={report}"]
    )

    assert result["status"] == expected_status
    assert result["next_stage"] == expected_stage
    if not synthesis_enabled:
        assert result["completion_reason"] == "retrieval_no_matches"


def test_main_prints_completion_reason(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    state, _ = _state(tmp_path)
    value = json.loads(state.read_text())
    value.update(status="RUNNING", next_stage="iteration_admission",
                 current_iteration=1, last_stage="iteration_retrieval",
                 synthesis_enabled=False)
    state.write_text(json.dumps(value))
    report = tmp_path / "admission_report.json"
    report.write_text(json.dumps({
        "status": "COMPLETE", "iteration": 1,
        "role_status": {
            "real": {"status": "NO_MATCHES", "selected_count": 0},
        },
        "admitted": {"real": 0, "clean": 0, "synthetic": 0},
        "new_training_images": 0,
    }))
    monkeypatch.setattr(sys, "argv", [
        str(SCRIPT), "--state", str(state), "--stage", "iteration_admission",
        "--iteration", "1", "--artifact", f"admission_report={report}",
    ])

    assert MODULE.main() == 0

    assert json.loads(capsys.readouterr().out) == {
        "status": "COMPLETE", "next_stage": None,
        "completion_reason": "retrieval_no_matches",
    }


def test_mixed_role_outcomes_continue_with_selected_data(tmp_path: Path) -> None:
    state, _ = _state(tmp_path)
    value = json.loads(state.read_text())
    value.update(status="RUNNING", next_stage="iteration_admission",
                 current_iteration=1, last_stage="iteration_retrieval",
                 synthesis_enabled=False)
    state.write_text(json.dumps(value))
    report = tmp_path / "admission_report.json"
    report.write_text(json.dumps({
        "status": "COMPLETE", "iteration": 1,
        "role_status": {
            "real": {"status": "SELECTED", "selected_count": 1},
            "clean": {"status": "NO_MATCHES", "selected_count": 0},
        },
        "admitted": {"real": 1, "clean": 0, "synthetic": 0},
        "new_training_images": 1,
    }))

    result = MODULE.commit(
        state, "iteration_admission", 1, [f"admission_report={report}"]
    )

    assert result["status"] == "RUNNING"
    assert result["next_stage"] == "iteration_training"
    assert "completion_reason" not in result


def test_measurement_rejects_incomplete_inference(tmp_path: Path) -> None:
    state, _ = _state(tmp_path)
    artifacts = _iteration_artifacts(tmp_path)
    value = json.loads(state.read_text())
    value.update(status="RUNNING", next_stage="iteration_training", current_iteration=1,
                 last_stage="iteration_admission", events=[])
    state.write_text(json.dumps(value))
    MODULE.commit(state, "iteration_training", 1, artifacts["iteration_training"])
    manifest = tmp_path / "measurement_manifest.json"
    report = json.loads(manifest.read_text())
    report["inference_roles"]["kpi"]["expected_images"] = 2
    manifest.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="inference cardinality mismatch"):
        MODULE.commit(state, "iteration_measurement", 1, artifacts["iteration_measurement"])


def test_training_rejects_checkpoint_that_does_not_match_epoch(tmp_path: Path) -> None:
    state, _ = _state(tmp_path)
    artifacts = _iteration_artifacts(tmp_path)
    value = json.loads(state.read_text())
    value.update(status="RUNNING", next_stage="iteration_training", current_iteration=1,
                 last_stage="iteration_admission")
    state.write_text(json.dumps(value))
    selection = tmp_path / "checkpoint_selection.json"
    report = json.loads(selection.read_text())
    report["best_epoch"] = 2
    selection.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="KPI-best checkpoint"):
        MODULE.commit(state, "iteration_training", 1, artifacts["iteration_training"])


def test_gaps_reject_report_parquet_count_disagreement(tmp_path: Path) -> None:
    state, _ = _state(tmp_path)
    artifacts = _iteration_artifacts(tmp_path)
    value = json.loads(state.read_text())
    value.update(status="RUNNING", next_stage="iteration_gaps", current_iteration=1,
                 last_stage="iteration_measurement")
    state.write_text(json.dumps(value))
    pd.DataFrame({"gap_type": []}).to_parquet(tmp_path / "strict_box_gaps.parquet")
    with pytest.raises(ValueError, match="parquet and report counts disagree"):
        MODULE.commit(state, "iteration_gaps", 1, artifacts["iteration_gaps"])


def test_synthesis_reconciles_generated_and_blocked_counts(tmp_path: Path) -> None:
    state, _ = _state(tmp_path)
    value = json.loads(state.read_text())
    value.update(status="RUNNING", synthesis_enabled=True, next_stage="iteration_synthesis",
                 current_iteration=1, last_stage="iteration_admission")
    state.write_text(json.dumps(value))
    generation = tmp_path / "generation_report.json"
    generation.write_text(json.dumps({
        "status": "COMPLETE", "generated": 2,
        "groups": [{"requested": 3, "generated": 2, "guardrail_blocked": 1}],
    }))
    admission = tmp_path / "admission_report.json"
    admission.write_text(json.dumps({"status": "COMPLETE", "iteration": 1,
                                     "role_status": {
                                         "real": {"status": "NO_MATCHES",
                                                  "selected_count": 0},
                                     },
                                     "admitted": {"real": 0, "clean": 0,
                                                  "synthetic": 2},
                                     "new_training_images": 2}))
    result = MODULE.commit(state, "iteration_synthesis", 1,
                           [f"generation_report={generation}",
                            f"admission_report={admission}"])
    assert result["next_stage"] == "iteration_training"


def test_synthesis_accepts_typed_no_eligible_skip(tmp_path: Path) -> None:
    state, _ = _state(tmp_path)
    value = json.loads(state.read_text())
    value.update(status="RUNNING", synthesis_enabled=True, next_stage="iteration_synthesis",
                 current_iteration=1, last_stage="iteration_admission")
    state.write_text(json.dumps(value))
    preparation = tmp_path / "input_contract.json"
    preparation.write_text(json.dumps({
        "status": "SKIPPED", "reason": "no_eligible_false_negatives",
        "selection_candidate_fn_count": 2, "eligible_fn_count": 0,
        "skipped_fn_count": 2,
        "skipped_fns": [{"fn_id": "fn-1"}, {"fn_id": "fn-2"}],
        "skip_counts": {"empty_fn_mask": 2},
    }))

    result = MODULE.commit(
        state, "iteration_synthesis", 1,
        [f"synthesis_preparation={preparation}"],
    )

    assert result["next_stage"] == "iteration_training"


def test_synthesis_accepts_typed_no_routed_skip(tmp_path: Path) -> None:
    state, _ = _state(tmp_path)
    value = json.loads(state.read_text())
    value.update(status="RUNNING", synthesis_enabled=True, next_stage="iteration_synthesis",
                 current_iteration=1, last_stage="iteration_admission")
    state.write_text(json.dumps(value))
    request = tmp_path / "synthesis_request.json"
    request.write_text(json.dumps({
        "status": "SKIPPED", "reason": "no_routed_false_negatives", "fn_count": 0,
        "skipped_unrouted_fn_count": 2,
        "skipped_unrouted_by_dataset": {"boxes_only": 2},
        "observed_dataset_ids": ["boxes_only"],
        "configured_route_keys": ["route"],
        "message": "Skipped synthesis because every FN is unrouted.",
    }))

    result = MODULE.commit(
        state, "iteration_synthesis", 1, [f"synthesis_request={request}"],
    )

    assert result["next_stage"] == "iteration_training"


def test_no_clean_reference_skip_allows_retrieval_result_to_train(tmp_path: Path) -> None:
    state = _advance_to_synthesis(tmp_path, retrieval_enabled=True)
    preparation = tmp_path / "input_contract.json"
    preparation.write_text(json.dumps({
        "status": "SKIPPED", "reason": "no_eligible_false_negatives",
        "selection_candidate_fn_count": 2, "eligible_fn_count": 0,
        "skipped_fn_count": 2,
        "skipped_fns": [{"fn_id": "fn-1", "reason": "no_clean_reference_images"},
                        {"fn_id": "fn-2", "reason": "no_clean_reference_images"}],
        "skip_counts": {"no_clean_reference_images": 2},
        "warnings": [{"code": "no_clean_reference_images", "fn_count": 2,
                      "message": "No synthesis clean references are available."}],
    }))

    result = MODULE.commit(
        state, "iteration_synthesis", 1,
        [f"synthesis_preparation={preparation}"],
    )

    assert result["status"] == "RUNNING"
    assert result["next_stage"] == "iteration_training"


def test_no_clean_reference_skip_converges_when_retrieval_has_no_output(
        tmp_path: Path) -> None:
    state = _advance_to_synthesis(tmp_path, retrieval_enabled=False)
    preparation = tmp_path / "input_contract.json"
    preparation.write_text(json.dumps({
        "status": "SKIPPED", "reason": "no_eligible_false_negatives",
        "selection_candidate_fn_count": 1, "eligible_fn_count": 0,
        "skipped_fn_count": 1,
        "skipped_fns": [{"fn_id": "fn-1", "reason": "no_clean_reference_images"}],
        "skip_counts": {"no_clean_reference_images": 1},
        "warnings": [{"code": "no_clean_reference_images", "fn_count": 1,
                      "message": "No synthesis clean references are available."}],
    }))

    result = MODULE.commit(
        state, "iteration_synthesis", 1,
        [f"synthesis_preparation={preparation}"],
    )

    assert result["status"] == "COMPLETE" and result["next_stage"] is None
    assert result["completion_reason"] == "all_producers_exhausted"


def test_no_eligible_skip_accepts_mixed_detailed_reasons(tmp_path: Path) -> None:
    state = _advance_to_synthesis(tmp_path, retrieval_enabled=True)
    preparation = tmp_path / "input_contract.json"
    preparation.write_text(json.dumps({
        "status": "SKIPPED", "reason": "no_eligible_false_negatives",
        "selection_candidate_fn_count": 2, "eligible_fn_count": 0,
        "skipped_fn_count": 2,
        "skipped_fns": [
            {"fn_id": "fn-1", "reason": "no_clean_reference_images"},
            {"fn_id": "fn-2", "reason": "empty_fn_mask"},
        ],
        "skip_counts": {"empty_fn_mask": 1, "no_clean_reference_images": 1},
        "warnings": [{"code": "no_clean_reference_images", "fn_count": 1,
                      "message": "One synthesis clean reference is unavailable."}],
    }))

    result = MODULE.commit(
        state, "iteration_synthesis", 1,
        [f"synthesis_preparation={preparation}"],
    )

    assert result["status"] == "RUNNING"
    assert result["next_stage"] == "iteration_training"


@pytest.mark.parametrize(
    ("admitted_real", "expected_status", "expected_stage"),
    [(1, "RUNNING", "iteration_training"), (0, "COMPLETE", None)],
)
def test_synthesis_budget_skip_trains_only_with_new_admission(
        tmp_path: Path, admitted_real: int,
        expected_status: str, expected_stage: str | None) -> None:
    state, artifacts = _budget_skip(tmp_path, admitted_real)

    result = MODULE.commit(state, "iteration_synthesis", 1, artifacts)

    assert result["status"] == expected_status
    assert result["next_stage"] == expected_stage
    decision = result["events"][-1]["synthesis_decision"]
    assert decision == {
        "status": "SKIPPED", "reason": "no_synthetic_budget",
        "available_room_before_admission": 1,
    }
    assert set(result["events"][-1]["artifacts"]) == {
        "synthesis_request", "admission_report",
    }
    if admitted_real == 0:
        assert result["completion_reason"] == "retrieval_no_matches"


def test_synthesis_budget_skip_accepts_round_robin_admission_index(
        tmp_path: Path) -> None:
    state, artifacts = _budget_skip(tmp_path, 1, round_robin=True)

    result = MODULE.commit(state, "iteration_synthesis", 1, artifacts)

    assert result["next_stage"] == "iteration_training"
    assert set(result["events"][-1]["artifacts"]) == {
        "synthesis_request", "admission_report", "admission_index",
    }


def test_synthesis_budget_skip_requires_exact_committed_admission(
        tmp_path: Path) -> None:
    state, artifacts = _budget_skip(tmp_path, 1)
    original = Path(artifacts[1].split("=", 1)[1])
    replacement = tmp_path / "replacement_admission_report.json"
    replacement.write_bytes(original.read_bytes())
    artifacts[1] = f"admission_report={replacement}"

    with pytest.raises(ValueError, match="committed admission report"):
        MODULE.commit(state, "iteration_synthesis", 1, artifacts)


def test_synthesis_budget_skip_rejects_all_eligible_mode(tmp_path: Path) -> None:
    state, artifacts = _budget_skip(tmp_path, 1)
    request = Path(artifacts[0].split("=", 1)[1])
    value = json.loads(request.read_text())
    value["selection_mode"] = "all_eligible"
    request.write_text(json.dumps(value))

    with pytest.raises(
        ValueError,
        match="no_synthetic_budget requires selection_mode=generated_per_type_plan",
    ):
        MODULE.commit(state, "iteration_synthesis", 1, artifacts)


def test_commit_rejects_out_of_order_stage(tmp_path: Path) -> None:
    state, artifact = _state(tmp_path)
    with pytest.raises(ValueError, match="expected stage candidate_cache"):
        MODULE.commit(state, "baseline_measurement", 0, [f"done={artifact}"])
