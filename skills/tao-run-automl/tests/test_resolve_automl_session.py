# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "resolve_automl_session.py"
SPEC = importlib.util.spec_from_file_location("resolve_automl_session", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_controller(workspace: Path, session_id: str, state=None) -> Path:
    path = workspace / ".automl" / "controller" / f"{session_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([] if state is None else state), encoding="utf-8")
    return path


def test_new_session_id_matches_wheel_default_shape():
    first = MODULE.new_session_id()
    second = MODULE.new_session_id()
    assert first != second
    assert len(first) == 12
    assert MODULE.SESSION_ID_RE.fullmatch(first)


def test_resolve_unique_controller(tmp_path):
    _write_controller(tmp_path, "abc123def456", [{"id": 0, "status": "success"}])
    assert MODULE.resolve_session_id(tmp_path) == "abc123def456"


def test_missing_controller_state_fails_closed(tmp_path):
    with pytest.raises(MODULE.SessionResolutionError, match="resume state is missing"):
        MODULE.resolve_session_id(tmp_path)


def test_ambiguous_controller_state_fails_closed(tmp_path):
    _write_controller(tmp_path, "111111111111")
    _write_controller(tmp_path, "222222222222")
    with pytest.raises(MODULE.SessionResolutionError, match="resume state is ambiguous"):
        MODULE.resolve_session_id(tmp_path)


def test_explicit_session_resolves_legacy_ambiguous_workspace(tmp_path):
    _write_controller(tmp_path, "111111111111")
    _write_controller(tmp_path, "222222222222")
    assert MODULE.resolve_session_id(tmp_path, "222222222222") == "222222222222"


def test_malformed_controller_state_is_rejected(tmp_path):
    path = _write_controller(tmp_path, "abc123def456")
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(MODULE.SessionResolutionError, match="unexpected shape"):
        MODULE.resolve_session_id(tmp_path)


def test_runner_settings_without_explicit_session_fail_before_launch():
    with pytest.raises(
        MODULE.SessionResolutionError,
        match="automl_settings.session_id",
    ):
        MODULE.validate_session_settings(
            {"algorithm": "bayesian"},
            resume=False,
        )


def test_resume_settings_must_bind_existing_controller(tmp_path):
    _write_controller(tmp_path, "abc123def456")

    with pytest.raises(MODULE.SessionResolutionError, match="has no controller state"):
        MODULE.validate_session_settings(
            {"algorithm": "bayesian", "session_id": "different123"},
            resume=True,
            workspace=tmp_path,
        )

    assert (
        MODULE.validate_session_settings(
            {"algorithm": "bayesian", "session_id": "abc123def456"},
            resume=True,
            workspace=tmp_path,
        )
        == "abc123def456"
    )


def test_cli_returns_nonzero_instead_of_silently_starting_fresh(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "resolve", "--workspace", str(tmp_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "refuse to start a fresh search with resume=True" in result.stderr
    assert result.stdout == ""


def test_explicit_identity_resumes_real_automl_controller(tmp_path):
    tao_automl = pytest.importorskip("tao_automl")
    schema_path = (
        Path(__file__).resolve().parents[3]
        / "models/tao-finetune-nv-tesseract-forecasting/schemas/train.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    session_id = MODULE.new_session_id()
    workspace = tmp_path / "run_resume"
    settings = {
        "algorithm": "bayesian",
        "metric": "val_mse",
        "direction": "minimize",
        "automl_max_recommendations": 1,
        "session_id": session_id,
    }
    assert MODULE.validate_session_settings(settings, resume=False) == session_id
    fresh = tao_automl.AutoML(
        workspace=str(workspace),
        network="nv_tesseract_forecasting",
        train_specs=schema["default"],
        settings=settings,
        action="train",
        search_schema=schema,
    )
    rec = fresh.next_recommendation()[0]
    fresh.report_result(rec.id, 0.5)
    fresh.finish()

    assert MODULE.resolve_session_id(workspace) == session_id
    assert (
        MODULE.validate_session_settings(
            settings,
            resume=True,
            workspace=workspace,
        )
        == session_id
    )
    resumed = tao_automl.AutoML(
        workspace=str(workspace),
        network="nv_tesseract_forecasting",
        train_specs=schema["default"],
        settings={**settings, "automl_max_recommendations": 2},
        action="train",
        search_schema=schema,
        resume=True,
    )
    assert resumed.get_status()["progress"]["completed"] == 1
    assert [item.id for item in resumed.next_recommendation()] == [1]
    assert [path.name for path in (workspace / ".automl/controller").glob("*.json")] == [
        f"{session_id}.json"
    ]
