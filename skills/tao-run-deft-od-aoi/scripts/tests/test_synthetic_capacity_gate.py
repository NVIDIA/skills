# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "commit_deft_od_aoi_stage.py"
SPEC = importlib.util.spec_from_file_location("capacity_gate_commit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def _admission(root: Path, room: int, new: int, enabled: bool = True) -> tuple[Path, Path]:
    state = root / "state.json"
    state.write_text(json.dumps({
        "status": "RUNNING", "next_stage": "iteration_admission",
        "current_iteration": 1, "last_stage": "iteration_retrieval",
        "synthesis_enabled": enabled,
    }))
    report = root / "admission.json"
    report.write_text(json.dumps({
        "status": "COMPLETE", "iteration": 1,
        "role_status": {"real": {"status": "SELECTED" if new else "NO_MATCHES",
                                  "selected_count": new}},
        "admitted": {"real": new, "clean": 0, "synthetic": 0},
        "new_training_images": new, "retained_previous_images": 10,
        "synthetic_admission": {"available_room_before_admission": room},
    }))
    return state, report


@pytest.mark.parametrize(("room", "new", "enabled", "next_stage"), [
    (0, 0, True, None), (0, 1, True, "iteration_training"),
    (1, 0, True, "iteration_synthesis"), (2, 1, True, "iteration_synthesis"),
    (2, 0, False, None), (2, 1, False, "iteration_training"),
])
def test_capacity_gate(tmp_path: Path, room: int, new: int,
                       enabled: bool, next_stage: str | None) -> None:
    state, report = _admission(tmp_path, room, new, enabled)
    result = MODULE.commit(state, "iteration_admission", 1, [f"admission_report={report}"])
    assert result["next_stage"] == next_stage
    assert result["status"] == ("RUNNING" if next_stage else "COMPLETE")
    decision = result["events"][-1].get("synthesis_decision")
    if enabled and room == 0:
        assert decision == {"status": "SKIPPED", "reason": "no_synthetic_budget",
                            "available_room_before_admission": 0}
    else:
        assert decision is None
    assert not (tmp_path / "generation_report.json").exists()


@pytest.mark.parametrize("capacity", [None, -1, True, "0", 0.5])
def test_invalid_capacity_does_not_advance(tmp_path: Path, capacity) -> None:
    state, report = _admission(tmp_path, 0, 0)
    value = json.loads(report.read_text())
    value["synthetic_admission"]["available_room_before_admission"] = capacity
    report.write_text(json.dumps(value))
    before = state.read_bytes()
    with pytest.raises(ValueError, match="synthetic capacity"):
        MODULE.commit(state, "iteration_admission", 1, [f"admission_report={report}"])
    assert state.read_bytes() == before


def test_clean_admission_can_train_when_synthetic_capacity_is_full(tmp_path: Path) -> None:
    state, report = _admission(tmp_path, 0, 1)
    value = json.loads(report.read_text())
    value["admitted"] = {"real": 0, "clean": 1, "synthetic": 0}
    report.write_text(json.dumps(value))
    result = MODULE.commit(state, "iteration_admission", 1, [f"admission_report={report}"])
    assert result["next_stage"] == "iteration_training"
    assert result["events"][-1]["synthesis_decision"]["status"] == "SKIPPED"


def _budget_skip(root: Path, new: int) -> tuple[Path, Path, Path]:
    state, admission = _admission(root, 1, new)
    MODULE.commit(state, "iteration_admission", 1, [f"admission_report={admission}"])
    request = root / "request.json"
    request.write_text(json.dumps({
        "status": "SKIPPED", "reason": "no_synthetic_budget", "fn_count": 0,
        "eligible_fn_count": 1, "selection_mode": "generated_per_type_plan",
        "planning": {"new_image_budget": 1, "images_per_fn": 2,
                     "eligible_fn_count": 1, "selected_fn_count": 0,
                     "planned_images": 0, "unplanned_budget": 1},
    }))
    return state, admission, request


@pytest.mark.parametrize("new", [0, 1])
def test_whole_fn_budget_skip_trains_only_with_new_data(tmp_path: Path, new: int) -> None:
    state, admission, request = _budget_skip(tmp_path, new)
    result = MODULE.commit(state, "iteration_synthesis", 1, [
        f"synthesis_request={request}", f"admission_report={admission}",
    ])
    assert result["next_stage"] == ("iteration_training" if new else None)
    assert result["status"] == ("RUNNING" if new else "COMPLETE")
    assert result["events"][-1]["synthesis_decision"]["reason"] == "no_synthetic_budget"


@pytest.mark.parametrize(("field", "value"), [
    ("new_image_budget", 0), ("new_image_budget", True), ("new_image_budget", -1),
    ("images_per_fn", 1), ("images_per_fn", 0), ("planned_images", 2),
    ("selected_fn_count", 1), ("eligible_fn_count", 0),
])
def test_budget_skip_rejects_invalid_evidence(tmp_path: Path, field: str, value) -> None:
    state, admission, request = _budget_skip(tmp_path, 0)
    evidence = json.loads(request.read_text())
    evidence["planning"][field] = value
    request.write_text(json.dumps(evidence))
    before = state.read_bytes()
    with pytest.raises(ValueError, match="budget skip"):
        MODULE.commit(state, "iteration_synthesis", 1, [
            f"synthesis_request={request}", f"admission_report={admission}",
        ])
    assert state.read_bytes() == before


@pytest.mark.parametrize("invalid", ["changed_admission", "generation", "missing_request"])
def test_budget_skip_requires_bound_evidence(tmp_path: Path, invalid: str) -> None:
    state, admission, request = _budget_skip(tmp_path, 0)
    artifacts = [f"synthesis_request={request}", f"admission_report={admission}"]
    if invalid == "changed_admission":
        admission.write_text(admission.read_text() + "\n")
    elif invalid == "generation":
        artifacts.append(f"generation_report={request}")
    else:
        artifacts = [f"admission_report={admission}"]
    before = state.read_bytes()
    with pytest.raises(ValueError):
        MODULE.commit(state, "iteration_synthesis", 1, artifacts)
    assert state.read_bytes() == before


@pytest.mark.parametrize("new", [0, 1])
def test_zero_synthetic_admission_uses_whole_iteration_growth(tmp_path: Path, new: int) -> None:
    state, admission = _admission(tmp_path, 2, new)
    MODULE.commit(state, "iteration_admission", 1, [f"admission_report={admission}"])
    post = tmp_path / "post_admission.json"
    value = json.loads(admission.read_text())
    value.update(admitted={"real": 0, "clean": 0, "synthetic": 0},
                 new_training_images=0, role_status={})
    post.write_text(json.dumps(value))
    generation = tmp_path / "generation.json"
    generation.write_text(json.dumps({
        "status": "COMPLETE", "generated": 0,
        "groups": [{"requested": 2, "generated": 0, "guardrail_blocked": 2}],
    }))
    result = MODULE.commit(state, "iteration_synthesis", 1, [
        f"generation_report={generation}", f"admission_report={post}",
    ])
    assert result["next_stage"] == ("iteration_training" if new else None)
    if not new:
        assert result["completion_reason"] == "retrieval_no_matches"


def test_changed_initial_admission_cannot_justify_training(tmp_path: Path) -> None:
    state, admission, request = _budget_skip(tmp_path, 1)
    admission.write_text(admission.read_text() + "\n")
    with pytest.raises(ValueError, match="initial admission evidence changed"):
        MODULE._committed_admission(json.loads(state.read_text()), 1)
