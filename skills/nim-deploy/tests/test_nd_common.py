# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/nd_common.py: strict inputs, run state, transitions and the emit contract."""

from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Any

import nd_common as nd
import pytest
from _testkit import FAKE_HF, FAKE_NVAPI, assert_contract, make_run
from _testkit import runs_home as runs_home

pytestmark = pytest.mark.usefixtures("runs_home")


@pytest.mark.parametrize(
    ("value", "expected"),
    [(True, True), ("YES", True), ("0", False), (0, False), ("off", False), (" true ", True)],
)
def test_flag_accepts_strict_booleans(value: Any, expected: bool) -> None:
    assert nd.flag(value) is expected


@pytest.mark.parametrize("value", ["maybe", 2, None, "falsee"])
def test_flag_rejects_everything_else(value: Any) -> None:
    with pytest.raises(ValueError, match="expected true or false"):
        nd.flag(value)


@pytest.mark.parametrize(
    ("key", "value", "secret"),
    [
        ("ngc_api_key", FAKE_NVAPI, True),
        ("hf_token", FAKE_HF, True),
        ("ngc_secret", "ngc-api-secret", False),  # a Secret NAME is fine
        ("pass_hf_token", "true", False),  # a switch is fine
        ("password", "hunter2", True),
        ("model", "meta/llama", False),
        ("note", "Bearer abc", True),
    ],
)
def test_looks_secret(key: str, value: str, secret: bool) -> None:
    assert nd.looks_secret(key, value) is secret


def test_emit_prints_one_contract_object(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        nd.emit(False, "render", "ask_user", {"a": 1}, [nd.error("bad_thing", "m", "f")])
    assert exit_info.value.code == nd.EXIT_INPUT
    payload = json.loads(capsys.readouterr().out)
    assert_contract(payload)
    assert payload["errors"][0]["code"] == "bad_thing"


def test_emit_ok_exits_zero() -> None:
    with pytest.raises(SystemExit) as exit_info:
        nd.emit(True, "done", "done")
    assert exit_info.value.code == 0


def test_runs_root_is_private_and_git_ignored(runs_home: Path) -> None:
    root = nd.runs_root()
    assert root == runs_home
    assert (root / ".gitignore").read_text() == "*\n"
    assert stat.S_IMODE(root.stat().st_mode) == 0o700


def test_run_id_must_be_safe() -> None:
    with pytest.raises(ValueError, match="invalid run id"):
        nd.run_dir("../escape")


def test_write_atomic_sets_mode_and_leaves_no_temp_file(tmp_path: Path) -> None:
    target = tmp_path / "out" / "f.txt"
    nd.write_atomic(target, "hi", mode=0o640)
    assert target.read_text() == "hi"
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    assert [p.name for p in target.parent.iterdir()] == ["f.txt"]


def test_transition_follows_edges_and_counts_visits() -> None:
    state = make_run({}, state="render")
    nd.transition(state, "validate")
    assert state["state"] == "validate"
    assert state["attempts"]["validate"] == 1
    assert nd.load_state("run1")["state"] == "validate"
    with pytest.raises(ValueError, match="illegal transition"):
        nd.transition(state, "apply")


def test_transition_stops_after_three_visits() -> None:
    state = make_run({}, state="render")
    for _ in range(nd.MAX_VISITS):
        nd.transition(state, "validate")
        nd.transition(state, "render")
    with pytest.raises(ValueError, match="ask the user before retrying"):
        nd.transition(state, "validate")


def test_transition_hashes_artifacts(tmp_path: Path) -> None:
    artifact = tmp_path / "a.yaml"
    artifact.write_text("x: 1\n")
    state = make_run({}, state="render")
    nd.transition(state, "validate", [str(artifact)])
    assert state["artifacts"][str(artifact)] == nd.sha256_file(artifact)


def test_missing_inputs_by_target_and_runner() -> None:
    assert nd.missing_inputs(make_run({}, target="docker")) == ["image"]  # name, port, gpus default
    assert nd.missing_inputs(make_run({"image": "i"}, target="docker")) == []
    model_free = make_run({"image": "nvcr.io/nim/nvidia/model-free-nim:2"}, target="docker")
    assert nd.missing_inputs(model_free) == ["model_path"]
    assert nd.missing_inputs(make_run({}, target="helm")) == list(nd.REQUIRED["helm"])
    assert nd.missing_inputs(make_run({}, target="nim-operator"))[:2] == ["image", "namespace"]


def test_approval_bindings_carry_scope() -> None:
    state = make_run({"name": "rel", "namespace": "ns", "context": "ctx"}, target="helm")
    bound = nd.approval_bindings(state)
    assert (bound["release"], bound["namespace"], bound["context"]) == ("rel", "ns", "ctx")


def test_to_yaml_is_deterministic_and_quotes_when_needed() -> None:
    text = nd.to_yaml({"a": [{"b": 1, "c": "yes"}, "x y"], "d": {}, "e": None, "f": True})
    assert text == 'a:\n- b: 1\n  c: "yes"\n- "x y"\nd: {}\ne: null\nf: true\n'


def test_whole_number_and_flag_input_raise_render_errors() -> None:
    assert nd.whole_number({"n": "3"}, "n") == 3
    for bad in ("0", "1.5", "x", True, "-2"):
        with pytest.raises(nd.RenderError):
            nd.whole_number({"n": bad}, "n")
    with pytest.raises(nd.RenderError, match="expected true or false"):
        nd.flag_input({"x": "maybe"}, "x")


def test_open_run_reports_missing_inputs(capsys: pytest.CaptureFixture[str]) -> None:
    make_run({}, target="docker")
    with pytest.raises(SystemExit) as exit_info:
        nd.open_run("run1", ("docker",), preview=False)
    assert exit_info.value.code == nd.EXIT_PRECONDITION
    payload = json.loads(capsys.readouterr().out)
    assert payload["facts"]["missing"] == ["image"]


def test_open_run_refuses_wrong_state_and_target(capsys: pytest.CaptureFixture[str]) -> None:
    make_run({}, target="docker", state="validate")
    with pytest.raises(SystemExit) as exit_info:
        nd.open_run("run1", ("nim-operator",), preview=False)
    assert exit_info.value.code == nd.EXIT_INPUT
    assert json.loads(capsys.readouterr().out)["errors"][0]["code"] == "wrong_target"
    with pytest.raises(SystemExit) as exit_info:
        nd.open_run("run1", ("docker",), preview=False)
    assert exit_info.value.code == nd.EXIT_PRECONDITION
    assert json.loads(capsys.readouterr().out)["errors"][0]["code"] == "wrong_state"
