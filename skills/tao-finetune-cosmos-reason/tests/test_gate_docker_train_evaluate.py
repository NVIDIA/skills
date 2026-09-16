# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import os
from pathlib import Path
import struct

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "gate_docker_train_evaluate.py"
SPEC = importlib.util.spec_from_file_location("gate_docker_train_evaluate", SCRIPT)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def write_adapter(root: Path, config: dict) -> Path:
    checkpoint = root / "checkpoints" / "stamp" / "safetensors" / "epoch_3"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    header = json.dumps({"layer.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    (checkpoint / "adapter_model.safetensors").write_bytes(
        struct.pack("<Q", len(header)) + header + b"\0\0\0\0"
    )
    return checkpoint


def write_dense(root: Path) -> Path:
    checkpoint = root / "checkpoints" / "stamp" / "safetensors" / "epoch_1"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text(
        json.dumps({"model_type": "qwen3_vl"}), encoding="utf-8"
    )
    header = json.dumps(
        {"layer.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    ).encode()
    (checkpoint / "00000.safetensors").write_bytes(
        struct.pack("<Q", len(header)) + header + b"\0\0\0\0"
    )
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer.weight": "00000.safetensors"}}),
        encoding="utf-8",
    )
    return checkpoint


def test_validates_unique_adapter_checkpoint(tmp_path: Path) -> None:
    expected = {"r": 64, "use_rslora": False}
    expected_path = tmp_path / "expected.json"
    expected_path.write_text(json.dumps(expected), encoding="utf-8")
    checkpoint = write_adapter(tmp_path / "results", expected)

    assert (
        gate.find_and_validate_checkpoint(
            tmp_path / "results", 3, "adapter", expected_path
        )
        == checkpoint
    )


def test_rejects_adapter_metadata_mismatch(tmp_path: Path) -> None:
    expected_path = tmp_path / "expected.json"
    expected_path.write_text(json.dumps({"r": 64}), encoding="utf-8")
    write_adapter(tmp_path / "results", {"r": 32})

    with pytest.raises(ValueError, match="adapter config mismatch"):
        gate.find_and_validate_checkpoint(
            tmp_path / "results", 3, "adapter", expected_path
        )


def test_validates_dense_checkpoint_and_index(tmp_path: Path) -> None:
    checkpoint = write_dense(tmp_path / "results")

    assert (
        gate.find_and_validate_checkpoint(tmp_path / "results", 1, "dense")
        == checkpoint
    )


def test_requires_structured_training_success(tmp_path: Path) -> None:
    status_path = tmp_path / "status.json"
    status_path.write_text(
        "\n".join(
            [
                json.dumps({"status": "STARTED"}),
                json.dumps({"status": "SUCCESS"}),
            ]
        ),
        encoding="utf-8",
    )
    gate.require_structured_success(status_path)

    status_path.write_text(json.dumps({"status": "FAILURE"}), encoding="utf-8")
    with pytest.raises(ValueError, match="expected SUCCESS"):
        gate.require_structured_success(status_path)


def test_atomic_json_replaces_complete_document(tmp_path: Path) -> None:
    destination = tmp_path / "state.json"
    gate.atomic_json(destination, {"phase": "training"})
    gate.atomic_json(destination, {"phase": "evaluation", "job": "job-1"})

    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "phase": "evaluation",
        "job": "job-1",
    }
    assert list(tmp_path.glob(".*.tmp")) == []


def test_summarizes_row_oriented_evaluation_results(tmp_path: Path) -> None:
    results_file = tmp_path / "evaluation" / "adapter" / "letter" / "general" / "results.json"
    results_file.parent.mkdir(parents=True)
    results_file.write_text(
        json.dumps(
            [
                {"response": "A", "gt": "A"},
                {"response": " B ", "gt": "B"},
                {"response": "C", "gt": "D"},
            ]
        ),
        encoding="utf-8",
    )

    assert gate.find_results(tmp_path) == {
        "accuracy": pytest.approx(2 / 3),
        "correct_samples": 2,
        "total_samples": 3,
        "results_file": str(results_file),
    }


def test_cli_identity_defaults_follow_invoking_account(monkeypatch: pytest.MonkeyPatch) -> None:
    required = {
        "--train-job": "train-1",
        "--train-container": "train-1",
        "--train-results": "/tmp/train",
        "--checkpoint-epoch": "3",
        "--expected-adapter-config": "/tmp/adapter.json",
        "--eval-spec": "/tmp/evaluate.toml",
        "--eval-results-root": "/tmp/eval-results",
        "--eval-cache": "/tmp/cache",
        "--dataset-root": "/tmp/data",
        "--base-model": "/tmp/model",
        "--image": "example/image:test",
        "--job-helper": "/tmp/job.py",
        "--redactor": "/tmp/redact.py",
        "--state-file": "/tmp/state.json",
        "--summary-file": "/tmp/summary.json",
        "--log-file": "/tmp/gate.log",
        "--pid-file": "/tmp/gate.pid",
        "--lock-file": "/tmp/gate.lock",
    }
    argv = [str(SCRIPT)]
    for key, value in required.items():
        argv.extend([key, value])
    monkeypatch.setattr(gate.sys, "argv", argv)

    args = gate.parse_args()

    assert args.user == f"{os.getuid()}:{os.getgid()}"
    assert args.runtime_user
    assert args.group_add == []
    assert args.dataset_mount_destination == Path("/tmp/data")
