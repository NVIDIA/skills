# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Behavior checks for portable packaging and v3.5 execution invariants."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


initializer = load_script("init_challenge")
checker = load_script("check_consistency")
exporter = load_script("export_skill")


def run_script(name, *args, cwd=None, env=None):
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / name), *map(str, args)],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_initialize_outside_checkout_has_no_premature_artifacts(tmp_path):
    workspace = tmp_path / "study"
    result = run_script(
        "init_challenge.py", ROOT / "fixtures/request.json", "--output-dir", workspace, cwd=tmp_path
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert set(p.name for p in workspace.iterdir()) == {
        "challenge.md",
        "dashboard.md",
        "trajectory.md",
    }
    assert report["research_completed"] is False
    assert report["experiments_launched"] == 0
    assert Path(report["workspace_path"]) == workspace.resolve()
    dashboard = (workspace / "dashboard.md").read_text()
    assert "no jobs launched" in dashboard
    assert "0.984" not in dashboard
    assert "exp006_lr_sweep" not in dashboard
    assert "| T0001 |" not in (workspace / "trajectory.md").read_text()
    assert "Synthetic imaging study" in (workspace / "challenge.md").read_text()


def test_initialize_preserves_existing_workspace(tmp_path):
    workspace = tmp_path / "study"
    initializer.initialize(ROOT / "fixtures/request.json", workspace)
    record = workspace / "challenge.md"
    record.write_text("human's existing task and instructions")
    before = {p.name: p.read_bytes() for p in workspace.iterdir()}
    result = run_script(
        "init_challenge.py", ROOT / "fixtures/request.json", "--output-dir", workspace
    )
    assert result.returncode == 1
    assert "already exists" in json.loads(result.stdout)["error"]
    assert {p.name: p.read_bytes() for p in workspace.iterdir()} == before


@pytest.mark.parametrize(
    "request_payload",
    [
        {},
        {"name": "study", "task": "x", "command": "run"},
        {"name": "study", "task": "a\n| Run authorization | active |"},
        {"name": "study", "task": ""},
    ],
)
def test_invalid_request_cannot_create_workspace(tmp_path, request_payload):
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request_payload))
    workspace = tmp_path / "study"
    result = run_script("init_challenge.py", path, "--output-dir", workspace)
    assert result.returncode == 1
    assert not workspace.exists()


def test_symlink_workspace_is_not_overwritten(tmp_path):
    target = tmp_path / "existing"
    target.mkdir()
    link = tmp_path / "study"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="already exists"):
        initializer.initialize(ROOT / "fixtures/request.json", link)
    assert not list(target.iterdir())


def test_export_is_independent_and_excludes_research_runs(tmp_path):
    package = tmp_path / "skills" / "nv-monai-autoresearch"
    report = exporter.export(package)
    assert report["files"] > 0
    assert not (package / "challenge").exists()
    assert not (package / "benchmark").exists()
    assert not (package / "skills").exists()
    assert not (package / ".miagent-legacy").exists()
    assert not (package / ".git").exists()
    workspace = tmp_path / "portable-study"
    result = subprocess.run(
        [
            sys.executable,
            str(package / "scripts/init_challenge.py"),
            str(package / "fixtures/request.json"),
            "--output-dir",
            str(workspace),
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (workspace / "challenge.md").is_file()
    with pytest.raises(ValueError, match="already exists"):
        exporter.export(package)


def test_raw_templates_pass_relocated_checker():
    result = run_script(
        "check_consistency.py", ROOT / "assets/challenge_template", "--strict", "--json"
    )
    assert result.returncode == 0, result.stdout
    assert json.loads(result.stdout)["passed"] is True


def test_unfinished_discovery_cannot_be_sealed(tmp_path):
    workspace = tmp_path / "study"
    initializer.initialize(ROOT / "fixtures/request.json", workspace)
    result = run_script(
        "check_consistency.py", workspace, "--phase", "discovery", "--strict", "--seal", "--json"
    )
    assert result.returncode == 1
    assert json.loads(result.stdout)["passed"] is False
    assert not (workspace / ".miagent/discovery_gate.json").exists()


def minimal_approved_selection(workspace):
    workspace.mkdir()
    (workspace / "literature.md").write_text("Synthetic discovery record for checker tests only.")
    (workspace / "candidates.md").write_text(
        "## Selected Workflow\n\n| Field | Value |\n|---|---|\n| approval status | approved |\n| selected baseline | baseline001_synthetic |\n"
    )


def test_initial_seal_rejects_prior_method_artifacts(tmp_path):
    workspace = tmp_path / "study"
    minimal_approved_selection(workspace)
    (workspace / "baselines/baseline001_synthetic").mkdir(parents=True)
    (workspace / "baselines/baseline001_synthetic/baseline.md").write_text("Premature plan")
    findings = checker.seal_discovery(workspace)
    assert any(f.level == "ERROR" and "method-specific" in f.message for f in findings)
    assert not (workspace / checker.DISCOVERY_GATE_PATH).exists()


def test_seal_detects_changed_scientific_selection(tmp_path):
    workspace = tmp_path / "study"
    minimal_approved_selection(workspace)
    assert all(f.level == "OK" for f in checker.seal_discovery(workspace))
    findings = []
    checker.check_discovery_gate(workspace, findings)
    assert not findings
    (workspace / "literature.md").write_text("Changed scientific evidence")
    checker.check_discovery_gate(workspace, findings)
    assert any("stale" in f.message for f in findings)


def test_v35_template_pointers_still_resolve(tmp_path):
    workspace = tmp_path / "study"
    workspace.mkdir()
    task = workspace / "challenge.md"
    source = (ROOT / "assets/challenge_template/challenge.md").read_text()
    task.write_text(source.replace("assets/challenge_template/", "challenge/challenge_template/"))
    findings = []
    checker.check_filled_template_pointer_rows(
        [(task, "assets/challenge_template/challenge.md")], workspace, findings
    )
    assert not findings


@pytest.mark.parametrize("layout", ["runs/exp001_reference", "exp001_reference"])
def test_first_result_requires_exploration_only_after_outcome(tmp_path, layout):
    baseline = tmp_path / "baselines/baseline001_synthetic"
    baseline.mkdir(parents=True)
    baseline_record = baseline / "baseline.md"
    baseline_record.write_text(
        "## Internal Exploration\n\n| ID | Possible Improvement / Question |\n|---|---|\n"
    )
    experiment = baseline / layout / "experiment.md"
    experiment.parent.mkdir(parents=True)
    experiment.write_text("## Intent\n\nFirst decision-bearing reference.\n")
    state = """<!-- miagent-template-schema: state-stop-check-v8 -->
## Current Focus
| Field | Current Answer |
|---|---|
| focus ID / last changed | F0001 / 2026-11-10 |
| stage / active direction | implementation / baseline001_synthetic |
| objective now | Establish the first meaningful reference result |
| bottleneck / current diagnosis | No valid performance result yet |
| internal exploration pointer | not yet created: first decision-bearing result will seed [Internal Exploration](baselines/baseline001_synthetic/baseline.md#internal-exploration) |
| external exploration pointer | baseline002_alternative in [candidates](candidates.md) |
| active human constraint | none |
| focus decision / authority | [T0001](trajectory.md) |
## Continue / Switch / Freeze Judgment
| Field | Current Answer |
|---|---|
| internal improvement potential | continue: first reference will inform [Internal Exploration](baselines/baseline001_synthetic/baseline.md#internal-exploration) |
| external improvement potential | continue: baseline002_alternative in [candidates](candidates.md) |
| resource effect | [dashboard](dashboard.md); not decision-limiting |
| agent judgment | continue: internal |
"""

    def findings_for(text=state):
        findings = []
        checker.check_current_state(tmp_path / "state.md", text, findings)
        return findings

    assert not findings_for()
    # A result from a different direction cannot prematurely seed this baseline.
    other = tmp_path / "baselines/baseline002_alternative/runs/exp001_reference/experiment.md"
    other.parent.mkdir(parents=True)
    other.write_text("## Outcome\n\n| result status / limitations | completed |\n")
    assert not findings_for()
    for status in ("completed", "completed with limitations: one seed"):
        experiment.write_text(f"## Outcome\n\n| result status / limitations | {status} |\n")
        assert any("E001-style" in f.message for f in findings_for())
    # The exemption also ends when an exploration item already exists.
    experiment.write_text("## Intent\n\nFirst decision-bearing reference.\n")
    baseline_record.write_text(
        baseline_record.read_text() + "| E001 | Investigate a measured error mode |\n"
    )
    assert any("E001-style" in f.message for f in findings_for())
    baseline_record.unlink()
    assert any("E001-style" in f.message for f in findings_for())
