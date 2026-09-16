# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the AutoML -> best_rec.json adapter.

The two load-bearing behaviors (both red-team must-fixes on the contract): budget
keys are moved OUT of `specs` into `observed_budget` (so a rung's num_epochs can't
overwrite DEFT's baseline), and metric_name+direction are mandatory. Output must
validate against the M0 best_rec schema.
"""

import copy
import json
import sys
from pathlib import Path

import jsonschema
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import best_rec_adapter as bra  # noqa: E402

SCHEMA = json.loads(bra.SCHEMA_PATH.read_text())

WINNING_SPECS = {
    "train": {"num_epochs": 10, "optim": {"lr": 0.00013}},   # num_epochs = a rung budget
    "dataset": {"batch_size": 4},
}


def build(**over):
    kw = dict(experiment_id="automl-exp-7", metric_name="far_at_100_recall", direction="minimize",
              best_rec_id="rec_003", best_score=0.0012, best_specs=WINNING_SPECS,
              checkpoint_uri="/lustre/results/exp7/rec_003/model_epoch_009_step_01200.pth")
    kw.update(over)
    return bra.build_best_rec(**kw)


def test_output_validates_against_schema():
    jsonschema.validate(build(), SCHEMA)


def test_budget_stripped_from_specs_into_observed_budget():
    rec = build()
    assert "num_epochs" not in rec["best"]["specs"]["train"]        # moved out of specs
    assert rec["best"]["observed_budget"]["num_epochs"] == 10       # preserved separately
    assert rec["best"]["specs"]["train"]["optim"]["lr"] == 0.00013  # non-budget HPs kept


def test_original_specs_not_mutated():
    original = copy.deepcopy(WINNING_SPECS)
    build()
    assert WINNING_SPECS == original                                # deepcopy — caller's dict intact


def test_metric_identity_required():
    with pytest.raises(ValueError, match="metric_name"):
        build(metric_name="")
    with pytest.raises(ValueError, match="direction"):
        build(direction="down")


def test_metric_name_and_direction_carried():
    rec = build()
    assert rec["metric_name"] == "far_at_100_recall" and rec["direction"] == "minimize"


def test_checkpoint_epoch_step_optional_and_valid():
    rec = build(checkpoint_epoch=9, checkpoint_step=1200)
    jsonschema.validate(rec, SCHEMA)
    assert rec["best"]["checkpoint_epoch"] == 9 and rec["best"]["checkpoint_step"] == 1200


def test_top_level_num_epochs_also_stripped():
    rec = bra.build_best_rec(experiment_id="e", metric_name="m", direction="maximize",
                             best_rec_id="r1", best_score=1.0, checkpoint_uri="/x",
                             best_specs={"num_epochs": 20, "train": {"lr": 0.1}})
    assert "num_epochs" not in rec["best"]["specs"]
    assert rec["best"]["observed_budget"]["num_epochs"] == 20
    jsonschema.validate(rec, SCHEMA)


def test_all_recs_passthrough():
    rec = build(all_recs=[{"rec_id": "rec_001", "score": 0.004, "job_id": "dino-train-x1"},
                          {"rec_id": "rec_002", "score": None}])
    jsonschema.validate(rec, SCHEMA)
    assert len(rec["all_recs"]) == 2


def test_cli_stdin_to_stdout(capsys, monkeypatch):
    payload = {"experiment_id": "e", "metric_name": "acc", "direction": "maximize",
               "best_rec_id": "r1", "best_score": 0.9, "checkpoint_uri": "/ckpt.pth",
               "best_specs": {"train": {"num_epochs": 12, "lr": 0.1}}}
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO(json.dumps(payload)))
    assert bra.main([]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["best"]["observed_budget"]["num_epochs"] == 12
    assert "num_epochs" not in out["best"]["specs"]["train"]
