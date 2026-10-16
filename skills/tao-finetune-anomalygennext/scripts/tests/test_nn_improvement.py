# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).parents[1]


def _kpi(path: Path, score: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["kpi", "texture+defect", "Average"])
        writer.writerow(["nn_score", score, score])


def _run(root: Path, baseline: float, trained: float) -> subprocess.CompletedProcess:
    run = root / "run"
    _kpi(run / "valid" / "0" / "valid_kpi.csv", baseline)
    _kpi(run / "valid" / "1000" / "valid_kpi.csv", trained)
    model = run / "checkpoints" / "model" / "iter_000001000.pt"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"adapter")
    (run / "checkpoints" / "best_checkpoint.txt").write_text("iter_000001000.pt\n")
    recipe = root / "recipe.yaml"
    recipe.write_text(yaml.safe_dump({"dataset_name": "fixture",
                                      "anomaly_types": [["texture", "defect"]]}))
    return subprocess.run([sys.executable, str(ROOT / "validate_nn_improvement.py"),
                           "--run-dir", str(run), "--recipe", str(recipe),
                           "--results-dir", str(root / "results")],
                          capture_output=True, text=True)


def test_nn_gate_publishes_hash_bound_handoff(tmp_path: Path) -> None:
    result = _run(tmp_path, 0.2, 0.3)
    assert result.returncode == 0, result.stderr
    output = tmp_path / "results"
    handoff = json.loads((output / "training_handoff.json").read_text())
    assert handoff["status"] == "COMPLETE"
    assert handoff["anomaly_types"] == ["texture+defect"]
    assert handoff["checkpoint_sha256"] == hashlib.sha256(b"adapter").hexdigest()
    assert Path(handoff["checkpoint"]).read_bytes() == b"adapter"


def test_nn_gate_rejects_regression(tmp_path: Path) -> None:
    result = _run(tmp_path, 0.4, 0.3)
    assert result.returncode == 1
    report = json.loads((tmp_path / "results" / "nn_improvement_summary.json").read_text())
    assert report["status"] == "ERROR"
    assert any("did not improve" in error for error in report["errors"])
    assert not (tmp_path / "results" / "best_model_checkpoint.pt").exists()


def test_training_wrapper_exposes_container_inputs() -> None:
    result = subprocess.run(["bash", str(ROOT / "finetune_anomalygennext.sh"), "--help"],
                            capture_output=True, text=True)
    assert result.returncode == 0
    assert "--checkpoint-root PATH" in result.stdout


def test_training_contract_mounts_complete_checkpoint_tree() -> None:
    contract = yaml.safe_load((ROOT.parent / "references" / "skill_info.yaml").read_text())
    inputs = contract["actions"]["train"]["inputs"]
    assert inputs["checkpoint_root"]["container_path"] == (
        "/workspace/paidf-anomalygen/checkpoints"
    )
    assert "nn_backbone" not in inputs
    assert "hf_cache" not in inputs
