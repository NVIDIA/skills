# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path

import yaml


SCRIPT = Path(__file__).parents[1] / "select_deft_od_aoi_training.py"
SPEC = importlib.util.spec_from_file_location("select_deft_od_aoi_training", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def _status(path: Path, epoch: int, score: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"epoch": epoch, "kpi": {"val_mAP50": score}}) + "\n")


def test_metrics_carries_epoch_to_later_kpi_row(tmp_path: Path) -> None:
    status = tmp_path / "status.json"
    status.write_text("\n".join((
        json.dumps({"epoch": 7, "step": 120}),
        json.dumps({"kpi": {"val_mAP50": 0.61}}),
    )) + "\n")
    assert MODULE._metrics([status]) == [(7, 0.61)]


def test_metrics_prefers_epoch_row_over_preceding_untagged_copy(
        tmp_path: Path) -> None:
    status = tmp_path / "status.json"
    status.write_text("\n".join((
        json.dumps({"epoch": 28, "kpi": {"val_mAP50": 0.77}}),
        json.dumps({"message": "Eval metrics generated",
                    "kpi": {"val_mAP50": 0.7808247013944787}}),
        json.dumps({"epoch": 29,
                    "kpi": {"val_mAP50": 0.7808247013944787}}),
    )) + "\n")
    assert MODULE._metrics([status]) == [
        (28, 0.77),
        (29, 0.7808247013944787),
    ]


def test_probe_selection_patches_winner_and_updates_history(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"iteration": 3, "train_size": 120,
                                    "probes": [{"index": index, "name": str(index),
                                                "overrides": {"train.optim.lr": 0.1 + index}}
                                               for index in range(3)]}))
    template = tmp_path / "template.yaml"
    template.write_text(yaml.safe_dump({"train": {"optim": {"lr": 0.01}}}))
    statuses = []
    for index, score in enumerate((0.4, 0.6, 0.5)):
        status = tmp_path / f"p{index}.jsonl"
        _status(status, 9, score)
        statuses.append(status)
    report = MODULE.probes(manifest, template, statuses, tmp_path / "selected", None)
    assert report["winner"]["index"] == 1
    spec = yaml.safe_load((tmp_path / "selected/train.yaml").read_text())
    assert spec["train"]["optim"]["lr"] == 1.1
    assert json.loads((tmp_path / "selected/history.json").read_text()) == [
        {"iteration": 3, "train_size": 120}
    ]


def test_checkpoint_selection_emits_one_terminal_resume_extension(tmp_path: Path) -> None:
    policy = tmp_path / "policy.yaml"
    policy.write_text(yaml.safe_dump({"training": {"late_best_window": 3,
                                                    "extension_epochs": 12}}))
    spec = tmp_path / "train.yaml"
    spec.write_text(yaml.safe_dump({"results_dir": "/run", "train": {
        "num_epochs": 36, "pretrained_model_path": "/base.pth"}}))
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    for epoch in (34, 35):
        (checkpoints / f"model_epoch_{epoch:03d}.pth").write_bytes(b"model")
    status = tmp_path / "status.jsonl"
    _status(status, 34, 0.7)
    report = MODULE.checkpoint(policy, spec, [status], checkpoints, 36, False,
                               tmp_path / "selection.json")
    assert report["action"] == "extend" and report["extended_num_epochs"] == 48
    extension = yaml.safe_load((tmp_path / "extension.yaml").read_text())
    assert extension["train"]["num_epochs"] == 48
    assert extension["train"]["resume_training_checkpoint_path"].endswith("model_epoch_035.pth")
    assert "pretrained_model_path" not in extension["train"]


def test_checkpoint_selection_counts_resumed_epochs_once(tmp_path: Path) -> None:
    policy = tmp_path / "policy.yaml"
    policy.write_text(yaml.safe_dump({"training": {"late_best_window": 3,
                                                    "extension_epochs": 12}}))
    spec = tmp_path / "train.yaml"
    spec.write_text(yaml.safe_dump({"train": {"num_epochs": 4}}))
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    for epoch in range(4):
        (checkpoints / f"model_epoch_{epoch:03d}.pth").write_bytes(b"model")
    rows = {0: 0.5, 1: 0.6, 2: 0.65, 3: 0.68}
    phase_a = tmp_path / "status_phaseA.json"
    phase_a.write_text("".join(json.dumps({"epoch": e, "kpi": {"val_mAP50": rows[e]}}) + "\n"
                               for e in (0, 1)))
    # The resumed run appends to the same status.json, so phase B repeats 0 and 1.
    phase_b = tmp_path / "status_phaseB.json"
    phase_b.write_text("".join(json.dumps({"epoch": e, "kpi": {"val_mAP50": rows[e]}}) + "\n"
                               for e in range(4)))
    assert MODULE._metrics([phase_a, phase_b]) == sorted(rows.items())
    report = MODULE.checkpoint(policy, spec, [phase_a, phase_b], checkpoints, 4, True,
                               tmp_path / "selection.json")
    assert report["best_epoch"] == 3
    assert report["epochs_covered"] == [0, 1, 2, 3]
    assert report["duplicate_epochs"] == [0, 1]


def test_metrics_keeps_latest_row_for_a_rerun_epoch(tmp_path: Path) -> None:
    first = tmp_path / "before_crash.json"
    first.write_text(json.dumps({"epoch": 2, "kpi": {"val_mAP50": 0.9}}) + "\n")
    rerun = tmp_path / "after_resume.json"
    rerun.write_text(json.dumps({"epoch": 2, "kpi": {"val_mAP50": 0.4}}) + "\n")
    assert MODULE._metrics([first, rerun]) == [(2, 0.4)]
