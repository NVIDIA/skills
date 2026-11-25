# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/handoff.py and `nimdeploy.py new --from` (nim-select, nim-tune results)."""

from __future__ import annotations

import copy
import json
from datetime import timedelta
from pathlib import Path
from types import ModuleType
from typing import Any

import guard
import handoff
import nd_common as nd
import nimdeploy
import pytest
import render_docker
import render_k8s
import validate
from _testkit import TESTS, assert_contract, run_cli, with_docs
from _testkit import runs_home as runs_home

pytestmark = pytest.mark.usefixtures("runs_home")

SELECT = json.loads((TESTS / "fixtures" / "select-host.json").read_text())
ENV = json.loads((TESTS / "fixtures" / "env-host.json").read_text())
PROFILE = SELECT["handoff"]["model_profile"]
IMAGE = SELECT["handoff"]["image"]
DIGEST = "sha256:" + "ab" * 32


def now_text(days_ago: float = 0) -> str:
    return (nd.utc_now() - timedelta(days=days_ago)).isoformat()


def select_data(**handoff_change: Any) -> dict[str, Any]:
    data: dict[str, Any] = copy.deepcopy(SELECT)
    data["environment_summary"]["captured_at"] = now_text()
    data["handoff"].update(handoff_change)
    return data


LABEL_PLACEMENT = {
    "selector": {"example.com/pool": "gpu-a"},
    "nodes_matched": ["node-a1", "node-a2"],
    "source": "node-labels",
}
HOSTNAME_PLACEMENT = {
    "selector": None,
    "hostnames": ["node-a1", "node-a2"],
    "nodes_matched": ["node-a1", "node-a2"],
    "source": "hostname-fallback",
}


def k8s_select(placement: dict[str, Any] | None = LABEL_PLACEMENT) -> dict[str, Any]:
    data = select_data(gpus=1, gpu_ids=None)
    gpu = data["handoff"]["gpu_name"]
    group: dict[str, Any] = {"name": gpu}
    if placement:
        group["placement"] = placement
    data["environment_summary"].update(
        target="k8s", context="prod-a", namespace="nim-prod", groups=[{"name": "OTHER-GPU"}, group]
    )
    return data


def new_k8s(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, data: dict[str, Any], *more: str,
    target: str = "helm",
) -> tuple[int, dict[str, Any]]:
    return new(
        capsys, "--target", target, "--from", write(tmp_path, "select.json", data),
        "--set", "name=gb300", "--set", "ngc_secret=ngc", *more,
    )


@pytest.mark.parametrize("target", ["helm", "nim-operator"])
@pytest.mark.parametrize("verdict", ["fits", "unknown"])
def test_select_placement_becomes_the_selector_and_shows_in_approval(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, target: str, verdict: str
) -> None:
    data = k8s_select()
    data["handoff"]["verdict"] = verdict
    code, payload = new_k8s(capsys, tmp_path, data, target=target)
    assert code == 0
    facts = payload["facts"]
    assert facts["inputs"]["node_selector"] == LABEL_PLACEMENT["selector"]
    assert facts["inputs"]["placement_nodes"] == LABEL_PLACEMENT["nodes_matched"]
    summary = render_and_approve(capsys, facts["run_id"])
    assert summary["placement"]["node_selector"] == LABEL_PLACEMENT["selector"]
    assert summary["placement"]["source"] == "select"
    assert summary["placement"]["found_by"] == "node-labels"
    assert summary["handoff"]["sources"]["node_selector"] == "select"


@pytest.mark.parametrize(
    "nodes_field",
    [{}, {"nodes_matched": None}, {"nodes_matched": "node-a1"}, {"nodes_matched": {}}],
)
def test_selected_placement_requires_a_nodes_matched_list(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, nodes_field: dict[str, Any]
) -> None:
    placement = {key: value for key, value in LABEL_PLACEMENT.items() if key != "nodes_matched"}
    placement.update(nodes_field)
    code, payload = new_k8s(capsys, tmp_path, k8s_select(placement))
    assert code == 2
    error = payload["errors"][0]
    assert error["code"] == "handoff_invalid"
    assert "nodes_matched" in error["message"]
    assert error["fix"] == handoff.SELECT_FIX


def test_hostname_fallback_is_taken_with_a_warning(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    _, payload = new_k8s(capsys, tmp_path, k8s_select(HOSTNAME_PLACEMENT))
    facts = payload["facts"]
    assert facts["inputs"]["node_hostnames"] == ["node-a1", "node-a2"]
    assert "node_selector" not in facts["inputs"]
    assert any("breaks when those nodes are replaced" in w for w in facts["handoff"]["warnings"])


def test_no_placement_in_the_select_file_asks_the_user(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    _, payload = new_k8s(capsys, tmp_path, k8s_select(None), "--set", "confirm_handoff=yes",
                         target="nim-operator")
    assert "placement" in payload["facts"]["missing"]
    code, _ = call(capsys, render_k8s, "--run", payload["facts"]["run_id"])
    assert code == 3


@pytest.mark.parametrize(
    "answer", ["node_selector={\"team\":\"a\"}", "node_hostnames=[\"n1\"]", "placement=none"]
)
def test_the_user_can_answer_the_placement_question(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, answer: str
) -> None:
    _, payload = new_k8s(capsys, tmp_path, k8s_select(None), "--set", "confirm_handoff=yes")
    _, payload = call(capsys, nimdeploy, "set", payload["facts"]["run_id"], answer)
    assert "placement" not in payload["facts"]["missing"]


def test_changing_cluster_asks_for_placement_again(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    _, payload = new_k8s(capsys, tmp_path, k8s_select())
    _, payload = call(capsys, nimdeploy, "set", payload["facts"]["run_id"], "context=another")
    assert "placement" in payload["facts"]["missing"]


@pytest.mark.parametrize("change_at_intake", [True, False])
def test_changing_namespace_keeps_select_placement(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, change_at_intake: bool
) -> None:
    flags = ["--set", "namespace=another-namespace"] if change_at_intake else []
    _, payload = new_k8s(capsys, tmp_path, k8s_select(), *flags)
    if not change_at_intake:
        _, payload = call(capsys, nimdeploy, "set", payload["facts"]["run_id"],
                          "namespace=another-namespace")
    assert "placement" not in payload["facts"]["missing"]
    assert payload["facts"]["inputs"]["node_selector"] == LABEL_PLACEMENT["selector"]


def test_a_user_selector_replaces_the_select_placement_and_its_node_list(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    _, payload = new_k8s(capsys, tmp_path, k8s_select())
    _, payload = call(capsys, nimdeploy, "set", payload["facts"]["run_id"],
                      'node_selector={"team":"a"}')
    inputs = payload["facts"]["inputs"]
    assert inputs["node_selector"] == {"team": "a"} or inputs["node_selector"] == '{"team":"a"}'
    assert "placement_nodes" not in inputs


def test_review_same_name_gpu_groups_match_selected_memory(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    data = k8s_select()
    group = data["environment_summary"]["groups"][-1]
    group["memory_mib"] = 80000
    other = copy.deepcopy(group)
    other["memory_mib"] = 40000
    other["placement"]["selector"] = {"pool": "wrong"}
    data["environment_summary"]["groups"].insert(0, other)
    data["handoff"]["gpu_memory_mib"] = 80000
    _, payload = new_k8s(capsys, tmp_path, data)
    assert payload["facts"]["inputs"]["node_selector"] == LABEL_PLACEMENT["selector"]


def test_legacy_same_name_group_without_placement_still_makes_identity_ambiguous(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    data = k8s_select()
    assert "gpu_memory_mib" not in data["handoff"]
    data["environment_summary"]["groups"].append({"name": data["handoff"]["gpu_name"]})
    _, payload = new_k8s(capsys, tmp_path, data)
    assert "placement" in payload["facts"]["missing"]


@pytest.mark.parametrize(
    "answer",
    [
        "node_selector={}",
        "node_selector=[]",
        "node_hostnames=[]",
        "node_hostnames={}",
        "placement=yes",
        'node_selector={"pool":"bad value"}',
    ],
)
def test_review_invalid_placement_keeps_question_open(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, answer: str
) -> None:
    _, payload = new_k8s(capsys, tmp_path, k8s_select(None))
    _, payload = call(capsys, nimdeploy, "set", payload["facts"]["run_id"], answer)
    assert "placement" in payload["facts"]["missing"]


@pytest.mark.parametrize(
    "answer",
    ['node_selector={"pool":"a"}', 'node_hostnames=["worker-a"]', "placement=none"],
)
@pytest.mark.parametrize("same_command", [False, True])
def test_review_context_change_invalidates_old_user_placement(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, answer: str, same_command: bool
) -> None:
    _, payload = new_k8s(capsys, tmp_path, k8s_select())
    run = payload["facts"]["run_id"]
    call(capsys, nimdeploy, "set", run, answer)
    values = ["context=another", *([answer] if same_command else [])]
    _, payload = call(capsys, nimdeploy, "set", run, *values)
    assert ("placement" in payload["facts"]["missing"]) is not same_command


def tune_data(**change: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "apiVersion": "nim-tune/v1",
        "kind": "RecommendedConfig",
        "created_at": now_text(),
        "campaign": "qwen-c7",
        "model": SELECT["handoff"]["model"],
        "image": IMAGE,
        "image_digest": DIGEST,
        "model_profile": PROFILE,
        "target": "docker",
        "gpus": 1,
        "gpu_name": SELECT["handoff"]["gpu_name"],
        "replicas": 1,
        "args": ["--max-num-seqs", "64"],
        "env": {"NIM_KV_CACHE_PERCENT": "0.8"},
        "env_from_host": ["NGC_API_KEY", "HF_TOKEN"],
        "baseline_won": False,
        "verdict": "seqs64: 18% more throughput at the latency target",
    }
    data.update(change)
    return data


def write(folder: Path, name: str, data: Any) -> str:
    path = folder / name
    path.write_text(json.dumps(data) if not isinstance(data, str) else data)
    return str(path)


def call(
    capsys: pytest.CaptureFixture[str], module: ModuleType, *argv: str
) -> tuple[int, dict[str, Any]]:
    code = 0
    try:
        module.main(list(argv))
    except SystemExit as exit_info:
        code = int(exit_info.code or 0)
    payload = json.loads(capsys.readouterr().out)
    assert_contract(payload)
    return code, payload


def new(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict[str, Any]]:
    return call(capsys, nimdeploy, "new", *argv)


def froms(*paths: str) -> list[str]:
    return [item for path in paths for item in ("--from", path)]


def render_and_approve(capsys: pytest.CaptureFixture[str], run: str) -> dict[str, Any]:
    state = nd.load_state(run)
    if state["target"] == "docker":
        call(capsys, render_docker, "--run", run)
    else:
        call(capsys, render_k8s, "--run", run)
    with_docs(nd.load_state(run))
    code, payload = call(capsys, validate, "--run", run, "--offline")
    assert code == 0, payload
    facts: dict[str, Any] = payload["facts"]
    return facts


# --- no files: unchanged --------------------------------------------------------------------


def test_without_files_or_target_new_is_an_input_error(capsys: pytest.CaptureFixture[str]) -> None:
    code, payload = new(capsys, "--set", f"image={IMAGE}")
    assert (code, payload["errors"][0]["code"]) == (2, "bad_input")


def test_without_files_a_run_has_no_handoff_block(capsys: pytest.CaptureFixture[str]) -> None:
    code, payload = new(capsys, "--target", "docker", "--set", f"image={IMAGE}")
    assert code == 0 and "handoff" not in payload["facts"]
    assert "handoff" not in nd.load_state(payload["facts"]["run_id"])


# --- select only ----------------------------------------------------------------------------


def test_select_only_pins_the_profile_and_says_not_tuned(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    code, payload = new(capsys, *froms(write(tmp_path, "select.json", select_data())))
    assert (code, payload["state"]) == (0, "render")  # a host select.json means docker
    facts = payload["facts"]
    assert facts["target"] == "docker"
    assert facts["inputs"]["model_profile"] == PROFILE
    assert facts["inputs"]["gpus"] == "device=0"
    assert facts["inputs"]["image"] == IMAGE
    assert facts["handoff"]["tuning"].startswith("not tuned")
    assert facts["handoff"]["sources"]["model_profile"] == "select"
    summary = render_and_approve(capsys, facts["run_id"])
    script = (nd.run_dir(facts["run_id"]) / "docker-run.sh").read_text()
    assert f"-e NIM_MODEL_PROFILE={PROFILE}" in script
    assert "--gpus '\"device=0\"'" in script
    profile = summary["handoff"]["profile"]
    assert (profile["id"], profile["name"], profile["chosen_in"]) == (
        PROFILE,
        "vllm-fp8-tp1-pp1",
        "nim-select",
    )
    assert "chosen in nim-select" in summary["instruction"]
    [entry] = summary["handoff"]["files"]
    assert entry["kind"] == "select" and entry["sha256"] == nd.sha256_file(entry["path"])


def test_the_approval_record_binds_the_files(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    path = write(tmp_path, "select.json", select_data())
    _, payload = new(capsys, *froms(path))
    run = payload["facts"]["run_id"]
    summary = render_and_approve(capsys, run)
    code, _ = call(capsys, nimdeploy, "approve", run, "--token", summary["token"])
    assert code == 0
    [record] = nd.load_state(run)["approvals"]
    assert record["handoff_files"][0]["sha256"] == nd.sha256_file(path)
    assert record["model_profile"] == PROFILE


def test_select_with_an_unknown_verdict_is_flagged(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    _, payload = new(capsys, *froms(write(tmp_path, "s.json", select_data(verdict="unknown"))))
    assert any("unknown" in w for w in payload["facts"]["handoff"]["warnings"])


# --- select and tune ------------------------------------------------------------------------


def test_select_and_tune_render_args_env_digest_and_profile(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    paths = write(tmp_path, "s.json", select_data()), write(tmp_path, "t.json", tune_data())
    code, payload = new(capsys, "--target", "docker", *froms(*paths))
    assert (code, payload["state"]) == (0, "render")
    run = payload["facts"]["run_id"]
    summary = render_and_approve(capsys, run)
    script = (nd.run_dir(run) / "docker-run.sh").read_text()
    assert f"{IMAGE}@{DIGEST} \\\n  --max-num-seqs \\\n  64\n" in script
    assert "-e NIM_KV_CACHE_PERCENT=0.8" in script and "-e HF_TOKEN \\" in script
    assert f"NIM_MODEL_PROFILE={PROFILE}" in script
    assert "NIM_PASSTHROUGH_ARGS" not in script
    block = summary["handoff"]
    assert block["tuning"].startswith("tuned in campaign qwen-c7: --max-num-seqs 64")
    assert block["sources"]["image"] == "tune" and block["sources"]["extra_args"] == "tune"
    assert block["sources"]["env.NIM_KV_CACHE_PERCENT"] == "tune"
    assert [f["run_id"] for f in block["files"]] == [None, "qwen-c7"]
    assert summary["extra_args"] == ["--max-num-seqs", "64"]
    # the guard allows the approved script in its exact form, extra args included
    call(capsys, nimdeploy, "approve", run, "--token", summary["token"])
    folder = nd.run_dir(run)
    assert guard.evaluate(f"bash {folder}/docker-run.sh", tmp_path, nd.utc_now())[0] == "allow"
    assert guard.evaluate(f"bash {folder}/docker-run.sh -x", tmp_path, nd.utc_now())[0] == "deny"


def test_tuned_args_reach_helm_and_the_operator(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    sel = write(tmp_path, "s.json", k8s_select())
    tune = write(tmp_path, "t.json", tune_data(target="nim-operator", replicas=2))
    common = ["--set", "name=llama", "--set", "ngc_secret=ngc-api", "--set", "hf_token_secret=hf"]
    _, payload = new(capsys, "--target", "helm", *common, *froms(sel, tune))
    assert payload["state"] == "render", payload
    run = payload["facts"]["run_id"]
    summary = render_and_approve(capsys, run)
    values = (nd.run_dir(run) / "values.yaml").read_text()
    assert 'customArgs:\n- "--max-num-seqs"\n- "64"\n' in values
    assert f'value: "{PROFILE}"' in values and "replicaCount: 2" in values
    assert any("measured on nim-operator" in w for w in summary["handoff"]["warnings"])
    assert any("not pinned" in w for w in summary["handoff"]["warnings"])  # digest on k8s
    assert summary["context"] == "prod-a" and summary["namespace"] == "nim-prod"
    again = ["--set", "confirm_handoff=yes"]  # the same files were used by the helm run
    _, payload = new(capsys, "--target", "nim-operator", *common, *again, *froms(sel, tune))
    run = payload["facts"]["run_id"]
    render_and_approve(capsys, run)
    manifest = (nd.run_dir(run) / "nimservice.yaml").read_text()
    assert f'profiles:\n        - "{PROFILE}"' in manifest
    assert f'nimCache:\n      name: llama\n      profile: "{PROFILE}"' in manifest
    assert 'args:\n  - "--max-num-seqs"\n  - "64"' in manifest


def test_k8s_hf_token_from_tune_asks_for_the_secret_name(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    sel = write(tmp_path, "s.json", k8s_select())
    tune = write(tmp_path, "t.json", tune_data())
    flags = ["--set", "name=llama", "--set", "ngc_secret=ngc-api"]
    _, payload = new(capsys, "--target", "helm", *flags, *froms(sel, tune))
    assert (payload["state"], payload["next"]) == ("plan", "ask_user")
    assert payload["facts"]["missing"] == ["hf_token_secret"]
    assert "HF_TOKEN" in payload["facts"]["questions"][0]["question"]


def test_tune_baseline_won_deploys_the_selected_config_without_args(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    tune = tune_data(baseline_won=True, args=[], env={}, verdict="the untuned NIM was best")
    paths = write(tmp_path, "s.json", select_data()), write(tmp_path, "t.json", tune)
    _, payload = new(capsys, *froms(*paths))
    run = payload["facts"]["run_id"]
    summary = render_and_approve(capsys, run)
    script = (nd.run_dir(run) / "docker-run.sh").read_text()
    assert script.rstrip().endswith(f"{IMAGE}@{DIGEST}")
    assert summary["handoff"]["tuning"].startswith("baseline won in campaign qwen-c7")
    assert "no extra arguments" in summary["handoff"]["tuning"]


def test_baseline_won_with_a_base_env_still_means_no_tuned_change(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    env = {"NIM_MAX_MODEL_LEN": "8192"}
    tune = tune_data(baseline_won=True, args=[], env=env, verdict="the untuned NIM was best")
    paths = write(tmp_path, "s.json", select_data()), write(tmp_path, "t.json", tune)
    _, payload = new(capsys, *froms(*paths))
    facts = payload["facts"]
    assert facts["inputs"]["env"] == env and "extra_args" not in facts["inputs"]
    assert "no tuned change" in facts["handoff"]["tuning"]
    summary = render_and_approve(capsys, facts["run_id"])
    text = (nd.run_dir(facts["run_id"]) / "docker-run.sh").read_text()
    assert "NIM_MAX_MODEL_LEN=8192" in text and summary["handoff"]["tuning"].endswith(
        "(the untuned NIM was best)"
    )


def test_a_secret_in_the_base_env_of_a_baseline_win_is_still_refused(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    tune = tune_data(baseline_won=True, args=[], env={"HF_TOKEN": "hf_abc"})
    paths = write(tmp_path, "s.json", select_data()), write(tmp_path, "t.json", tune)
    code, payload = new(capsys, *froms(*paths))
    assert (code, payload["errors"][0]["code"]) == (2, "handoff_invalid")


def test_baseline_won_with_args_is_contradictory(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    tune = write(tmp_path, "t.json", tune_data(baseline_won=True))
    code, payload = new(capsys, *froms(write(tmp_path, "s.json", select_data()), tune))
    assert (code, payload["errors"][0]["code"]) == (2, "handoff_invalid")
    assert "baseline_won" in payload["errors"][0]["message"]


def test_tuned_on_the_operator_deploys_on_docker_with_a_flag(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    paths = (
        write(tmp_path, "s.json", select_data()),
        write(tmp_path, "t.json", tune_data(target="nim-operator")),
    )
    _, payload = new(capsys, *froms(*paths))
    assert payload["facts"]["inputs"]["extra_args"] == ["--max-num-seqs", "64"]
    assert payload["facts"]["handoff"]["measured_on"] == "nim-operator"
    assert any("measured on nim-operator" in w for w in payload["facts"]["handoff"]["warnings"])


# --- tune without select --------------------------------------------------------------------


def test_tune_without_select_asks_for_the_gpu_ids(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    code, payload = new(capsys, *froms(write(tmp_path, "t.json", tune_data(gpus=2))))
    assert (code, payload["state"], payload["next"]) == (0, "plan", "ask_user")
    facts = payload["facts"]
    assert facts["missing"] == ["gpus"]
    assert "2 GPU" in facts["questions"][0]["question"]
    assert facts["handoff"]["profile"]["chosen_in"].startswith("nim-tune")
    assert facts["inputs"]["model_profile"] == PROFILE
    code, payload = call(capsys, nimdeploy, "set", facts["run_id"], "gpus=device=2,3")
    assert (code, payload["state"]) == (0, "render")


def test_tune_without_select_on_kubernetes_asks_for_context_and_namespace(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    tune = write(tmp_path, "t.json", tune_data(target="nim-operator", env_from_host=[]))
    _, payload = new(capsys, *froms(tune))
    facts = payload["facts"]
    assert (facts["target"], payload["next"]) == ("nim-operator", "ask_user")
    assert set(facts["missing"]) == {"namespace", "context", "name", "ngc_secret"}
    assert facts["inputs"]["gpus"] == "1"


def test_select_without_gpu_ids_asks_which_gpus_instead_of_using_all(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    select = write(tmp_path, "s.json", select_data(gpu_ids=None, gpus=None, gpu_count=2))
    code, payload = new(capsys, *froms(select))
    assert (code, payload["state"], payload["next"]) == (0, "plan", "ask_user")
    facts = payload["facts"]
    assert facts["missing"] == ["gpus"]
    assert "did not say which GPUs" in facts["questions"][0]["question"]
    assert "2 GPU" in facts["questions"][0]["question"]
    code, payload = call(capsys, nimdeploy, "set", facts["run_id"], "gpus=device=2,3")
    assert (code, payload["state"]) == (0, "render")


def test_env_json_alone_is_not_enough(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    code, payload = new(capsys, *froms(write(tmp_path, "env.json", ENV)))
    assert (code, payload["errors"][0]["code"]) == (2, "handoff_env_alone")


# --- refusals ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("change", "code_name"),
    [
        ({"verdict": "does_not_fit", "handoff": None}, "select_does_not_fit"),
        ({"handoff": None}, "select_no_profile"),
    ],
)
def test_select_without_a_profile_is_refused(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, change: dict[str, Any], code_name: str
) -> None:
    data = {**select_data(), **change}
    code, payload = new(capsys, *froms(write(tmp_path, "s.json", data)))
    assert (code, payload["next"], payload["errors"][0]["code"]) == (3, "ask_user", code_name)
    assert list((nd.runs_root() / "runs").glob("*")) == []


def test_select_of_no_metadata_is_refused(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    data = {"model": "x", "images_considered": [], "warnings": [], "notes": []}
    code, payload = new(capsys, *froms(write(tmp_path, "s.json", data)))
    assert (code, payload["errors"][0]["code"]) == (3, "select_does_not_fit")


@pytest.mark.parametrize(
    ("change", "shown"),
    [
        ({"model": "meta/llama-3.3-70b-instruct"}, "meta/llama-3.3-70b-instruct"),
        ({"image": "nvcr.io/nim/meta/llama-3.1-8b-instruct:2.0.13"}, ":2.0.13"),
        ({"model_profile": "f" * 64}, "f" * 64),
        ({"gpus": 2}, "tune 2"),
        ({"gpu_name": "NVIDIA L40S"}, "NVIDIA L40S"),
    ],
)
def test_files_that_disagree_are_refused_with_both_values(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, change: dict[str, Any], shown: str
) -> None:
    paths = write(tmp_path, "s.json", select_data()), write(tmp_path, "t.json", tune_data(**change))
    code, payload = new(capsys, *froms(*paths))
    assert (code, payload["next"], payload["errors"][0]["code"]) == (
        3,
        "ask_user",
        "handoff_disagree",
    )
    message = payload["errors"][0]["message"]
    assert shown in message and "select" in message and "tune" in message


def test_a_gpu_name_that_differs_only_in_case_or_prefix_is_accepted(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    tune = tune_data(gpu_name="h100  nvl")
    paths = write(tmp_path, "s.json", select_data()), write(tmp_path, "t.json", tune)
    code, _ = new(capsys, *froms(*paths))
    assert code == 0


def test_a_tune_result_without_a_gpu_name_asks_before_it_is_used(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    paths = (
        write(tmp_path, "s.json", select_data()),
        write(tmp_path, "t.json", tune_data(gpu_name=None)),
    )
    code, payload = new(capsys, *froms(*paths))
    facts = payload["facts"]
    assert (code, facts["missing"]) == (0, ["confirm_handoff"])
    assert "has no gpu_name" in facts["questions"][0]["question"]
    assert any("does not say which GPU" in w for w in facts["handoff"]["warnings"])


def test_image_digests_that_differ_are_refused_with_both_values(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    other = "sha256:" + "c" * 64
    select = select_data(image=f"{IMAGE}@{other}")
    paths = write(tmp_path, "s.json", select), write(tmp_path, "t.json", tune_data())
    code, payload = new(capsys, *froms(*paths))
    assert (code, payload["errors"][0]["code"]) == (3, "handoff_disagree")
    message = payload["errors"][0]["message"]
    assert other in message and DIGEST in message


def test_a_tag_only_select_and_a_tune_digest_pin_the_digest_and_say_the_tag_was_not_checked(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    # what nim-select really writes: nvcr.io/...:tag, no digest
    assert "@" not in SELECT["handoff"]["image"]
    paths = write(tmp_path, "s.json", select_data()), write(tmp_path, "t.json", tune_data())
    code, payload = new(capsys, *froms(*paths))
    facts = payload["facts"]
    assert (code, facts["inputs"]["image"]) == (0, f"{IMAGE}@{DIGEST}")
    notes = [w for w in facts["handoff"]["warnings"] if "did not check" in w]
    assert len(notes) == 1
    assert DIGEST in notes[0] and "from the tune result" in notes[0] and "2.1.4" in notes[0]


def test_a_selected_digest_is_kept_when_the_tune_result_has_none(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    pinned = f"{IMAGE}@{DIGEST}"
    tune = tune_data(image_digest=None)
    paths = write(tmp_path, "s.json", select_data(image=pinned)), write(tmp_path, "t.json", tune)
    code, payload = new(capsys, *froms(*paths))
    facts = payload["facts"]
    assert (code, facts["inputs"]["image"]) == (0, pinned)
    assert not any("did not check" in w for w in facts["handoff"]["warnings"])
    render_and_approve(capsys, facts["run_id"])
    script = (nd.run_dir(facts["run_id"]) / "docker-run.sh").read_text()
    assert f"  {pinned} \\\n" in script


def test_the_same_image_digest_in_both_files_is_accepted(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    select = select_data(image=f"{IMAGE}@{DIGEST}")
    paths = write(tmp_path, "s.json", select), write(tmp_path, "t.json", tune_data())
    code, payload = new(capsys, *froms(*paths))
    assert code == 0
    assert not any("did not check" in w for w in payload["facts"]["handoff"]["warnings"])


@pytest.mark.parametrize(
    ("data", "code_name", "part"),
    [
        ("{not json", "handoff_unreadable", "not JSON"),
        ({"hello": 1}, "handoff_unknown", "neither"),
        (tune_data(apiVersion="nim-tune/v2"), "handoff_version", "nim-tune/v2"),
        ({k: v for k, v in tune_data().items() if k != "campaign"}, "handoff_invalid", "campaign"),
        (tune_data(gpus="2"), "handoff_invalid", "$.gpus: expected integer"),
        (tune_data(args=["--x", ""]), "handoff_invalid", "$.args[1]"),
        (tune_data(image="nvcr.io/nim/x"), "handoff_invalid", "$.image"),
        (tune_data(extra=1), "handoff_invalid", "unknown field 'extra'"),
        (tune_data(created_at="yesterday"), "handoff_invalid", "created_at"),
        ({**ENV, "schema": "nim-select/environment/v9"}, "handoff_version", "v9"),
        ({"environment_summary": {"target": "moon"}, "verdict": "fits"}, "handoff_invalid", ""),
    ],
)
def test_malformed_files_are_input_errors_with_a_fix(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, data: Any, code_name: str, part: str
) -> None:
    code, payload = new(capsys, "--target", "docker", *froms(write(tmp_path, "f.json", data)))
    error = payload["errors"][0]
    assert (code, error["code"]) == (2, code_name)
    assert part in error["message"] and error["fix"]


def test_a_missing_file_is_an_input_error(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    code, payload = new(capsys, *froms(str(tmp_path / "nope.json")))
    assert (code, payload["errors"][0]["code"]) == (2, "handoff_missing")


@pytest.mark.parametrize(
    ("change", "code_name"),
    [
        ({"env": {"NIM_PASSTHROUGH_ARGS": "--x 1"}}, "handoff_invalid"),
        ({"env": {"MY_API_KEY": "abc"}}, "handoff_invalid"),
        ({"env_from_host": ["WANDB_TOKEN"]}, "secret_unsupported"),
    ],
)
def test_env_rules(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, change: dict[str, Any], code_name: str
) -> None:
    paths = write(tmp_path, "s.json", select_data()), write(tmp_path, "t.json", tune_data(**change))
    code, payload = new(capsys, *froms(*paths))
    assert (code, payload["errors"][0]["code"]) == (2, code_name)


def test_model_path_from_tuned_env_becomes_the_input(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    env = {"NIM_MODEL_PATH": "hf://org/m", "NIM_MODEL_PROFILE": PROFILE}
    paths = write(tmp_path, "s.json", select_data()), write(tmp_path, "t.json", tune_data(env=env))
    _, payload = new(capsys, *froms(*paths))
    inputs = payload["facts"]["inputs"]
    assert inputs["model_path"] == "hf://org/m" and "env" not in inputs


def test_a_tuned_env_profile_that_differs_is_refused(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    tune = tune_data(env={"NIM_MODEL_PROFILE": "f" * 64})
    paths = write(tmp_path, "s.json", select_data()), write(tmp_path, "t.json", tune)
    code, payload = new(capsys, *froms(*paths))
    assert (code, payload["errors"][0]["code"]) == (3, "handoff_disagree")


def test_several_replicas_cannot_run_on_docker(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    paths = (
        write(tmp_path, "s.json", select_data()),
        write(tmp_path, "t.json", tune_data(replicas=3)),
    )
    code, payload = new(capsys, *froms(*paths))
    assert (code, payload["errors"][0]["code"]) == (3, "replicas_unsupported")


def test_two_files_of_one_kind_are_refused(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    path = write(tmp_path, "s.json", select_data())
    code, payload = new(capsys, *froms(path, path))
    assert (code, payload["errors"][0]["code"]) == (2, "handoff_duplicate")


def test_kubernetes_select_needs_the_target_named(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    code, payload = new(capsys, *froms(write(tmp_path, "s.json", k8s_select())))
    assert (code, payload["errors"][0]["code"]) == (2, "target_required")


# --- questions before use ---------------------------------------------------------------------


def test_a_stale_file_must_be_confirmed(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    data = select_data()
    data["environment_summary"]["captured_at"] = now_text(days_ago=9)
    code, payload = new(capsys, *froms(write(tmp_path, "s.json", data)))
    facts = payload["facts"]
    assert (code, payload["state"], payload["next"]) == (0, "plan", "ask_user")
    assert facts["missing"] == ["confirm_handoff"]
    assert any("days old" in w for w in facts["handoff"]["warnings"])
    assert data["environment_summary"]["captured_at"] in facts["handoff"]["warnings"][0]
    code, payload = call(capsys, nimdeploy, "set", facts["run_id"], "confirm_handoff=yes")
    assert (code, payload["state"]) == (0, "render")


def test_a_file_used_by_another_run_must_be_confirmed(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    path = write(tmp_path, "s.json", select_data())
    _, first = new(capsys, *froms(path))
    _, payload = new(capsys, *froms(path))
    assert payload["facts"]["missing"] == ["confirm_handoff"]
    assert first["facts"]["run_id"] in payload["facts"]["questions"][0]["question"]


def test_an_env_json_from_another_inspect_must_be_confirmed(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    sel = write(tmp_path, "s.json", select_data())
    _, payload = new(capsys, *froms(sel, write(tmp_path, "env.json", ENV)))
    assert payload["facts"]["missing"] == ["confirm_handoff"]
    same = {
        **ENV,
        "captured_at": json.loads(Path(sel).read_text())["environment_summary"]["captured_at"],
    }
    _, payload = new(capsys, *froms(sel, write(tmp_path, "env2.json", same)))
    assert payload["facts"]["missing"] == ["confirm_handoff"]  # select.json was used just above
    assert "env.json" not in json.dumps(payload["facts"]["handoff"]["warnings"])


def test_a_host_select_for_a_kubernetes_target_warns_and_asks(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    flags = [
        "--set",
        "name=l",
        "--set",
        "ngc_secret=s",
        "--set",
        "context=c",
        "--set",
        "namespace=n",
    ]
    sel = write(tmp_path, "s.json", select_data())
    _, payload = new(capsys, "--target", "helm", *flags, *froms(sel))
    assert payload["facts"]["missing"] == ["placement", "confirm_handoff"]
    assert any("inspected this GPU host" in w for w in payload["facts"]["handoff"]["warnings"])
    assert payload["facts"]["inputs"]["gpus"] == "1"


def test_another_context_than_select_checked_warns_and_asks(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    flags = ["--set", "name=l", "--set", "ngc_secret=s", "--set", "context=other"]
    _, payload = new(
        capsys, "--target", "helm", *flags, *froms(write(tmp_path, "s.json", k8s_select()))
    )
    facts = payload["facts"]
    assert facts["missing"] == ["placement", "confirm_handoff"]
    assert facts["inputs"]["context"] == "other"  # the user's value wins
    assert facts["handoff"]["sources"]["context"] == "user (overrides select)"
    assert any("'prod-a'" in w and "'other'" in w for w in facts["handoff"]["warnings"])


# --- user overrides ---------------------------------------------------------------------------


def test_a_user_set_wins_and_is_marked(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    paths = write(tmp_path, "s.json", select_data()), write(tmp_path, "t.json", tune_data())
    _, payload = new(
        capsys, "--set", "gpus=device=4", "--set", "extra_args=--max-num-seqs 32", *froms(*paths)
    )
    facts = payload["facts"]
    assert facts["inputs"]["gpus"] == "device=4"
    assert facts["handoff"]["sources"]["gpus"] == "user (overrides select)"
    assert facts["handoff"]["sources"]["extra_args"] == "user (overrides tune)"
    _, payload = call(
        capsys, nimdeploy, "set", facts["run_id"], f"image={IMAGE.replace('2.1.4', '2.0.13')}"
    )
    block = payload["facts"]["handoff"]
    assert block["sources"]["image"] == "user (overrides tune)"
    assert any("profile was chosen for" in w for w in block["warnings"])


def test_an_image_change_after_from_asks_for_the_profile_too(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    other = IMAGE.replace("2.1.4", "2.0.13")
    code, payload = new(
        capsys, "--set", f"image={other}", *froms(write(tmp_path, "s.json", select_data()))
    )
    facts = payload["facts"]
    assert (code, payload["state"], payload["next"]) == (0, "plan", "ask_user")
    assert facts["missing"] == ["model_profile"]
    assert "profile ids belong to one image version" in facts["questions"][0]["question"]
    code, payload = call(capsys, nimdeploy, "set", facts["run_id"], f"model_profile={'f' * 64}")
    assert (code, payload["state"]) == (0, "render")
    assert "questions" not in payload["facts"]


def test_an_image_change_with_a_profile_in_the_same_command_is_accepted(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    other = IMAGE.replace("2.1.4", "2.0.13")
    flags = ("--set", f"image={other}", "--set", f"model_profile={'f' * 64}")
    code, payload = new(capsys, *flags, *froms(write(tmp_path, "s.json", select_data())))
    assert (code, payload["state"]) == (0, "render")


def test_fewer_gpus_than_the_profile_needs_asks_for_a_yes(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    select = write(tmp_path, "s.json", select_data(gpu_count=2, gpu_ids=[0, 1]))
    code, payload = new(capsys, "--set", "gpus=device=3", *froms(select))
    facts = payload["facts"]
    assert (code, payload["next"], facts["missing"]) == (0, "ask_user", ["confirm_handoff"])
    assert "1 GPU(s), but the profile needs 2" in facts["questions"][0]["question"]
    code, payload = call(capsys, nimdeploy, "set", facts["run_id"], "confirm_handoff=yes")
    assert (code, payload["state"]) == (0, "render")


def test_enough_gpus_after_an_override_asks_nothing(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    select = write(tmp_path, "s.json", select_data(gpu_count=2, gpu_ids=[0, 1]))
    code, payload = new(capsys, "--set", "gpus=device=3,4", *froms(select))
    assert (code, payload["state"]) == (0, "render")


@pytest.mark.parametrize("blank", ["", "   "])
def test_a_blank_profile_is_refused_and_the_pin_stays(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, blank: str
) -> None:
    select = froms(write(tmp_path, "s.json", select_data()))
    code, payload = new(capsys, "--set", f"model_profile={blank}", *select)
    assert (code, payload["errors"][0]["code"]) == (2, "profile_not_pinned")
    assert "--set model_profile=ID" in payload["errors"][0]["fix"]
    _, payload = new(capsys, *select)
    run = payload["facts"]["run_id"]
    code, payload = call(capsys, nimdeploy, "set", run, f"model_profile={blank}")
    error = payload["errors"][0]
    assert (code, error["code"]) == (2, "profile_not_pinned")
    assert "Never drop model_profile" in error["fix"]
    assert nd.input_value(nd.load_state(run), "model_profile") == PROFILE


def test_a_user_profile_override_is_shown(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    _, payload = new(
        capsys, "--set", "model_profile=abc", *froms(write(tmp_path, "s.json", select_data()))
    )
    block = payload["facts"]["handoff"]
    assert block["profile"]["overridden_by_user"] == "abc"
    assert block["sources"]["model_profile"] == "user (overrides select)"


# --- the nim CLI runner -----------------------------------------------------------------------


def test_the_nim_cli_runner_pins_the_profile_and_device_ids(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    tune = tune_data(args=[], env={})
    paths = write(tmp_path, "s.json", select_data()), write(tmp_path, "t.json", tune)
    _, payload = new(capsys, "--set", "runner=nimcli", "--set", "port=8000", *froms(*paths))
    facts = payload["facts"]
    assert (facts["inputs"]["gpus"], facts["inputs"]["model"]) == ("0", IMAGE)
    assert facts["handoff"]["profile"]["pinned_as"] == "--profile"
    from runners import nimcli

    script = nimcli.render_text(nd.run_inputs(nd.load_state(facts["run_id"])))
    assert f"--profile {PROFILE}" in script and "--gpus 0" in script


def test_the_nim_cli_runner_refuses_tuned_args(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    paths = write(tmp_path, "s.json", select_data()), write(tmp_path, "t.json", tune_data())
    code, payload = new(capsys, "--set", "runner=nimcli", "--set", "port=8000", *froms(*paths))
    assert (code, payload["errors"][0]["code"]) == (3, "runner_cannot_pass_args")


# --- the module on its own --------------------------------------------------------------------


def test_schema_subset() -> None:
    schema = {
        "type": "object",
        "required": ["a"],
        "additionalProperties": False,
        "properties": {
            "a": {"type": "integer", "minimum": 1},
            "b": {"type": ["string", "null"], "pattern": "^x"},
            "c": {"enum": ["p", "q"]},
            "d": {"type": "object", "additionalProperties": {"type": "string"}},
        },
    }
    assert handoff.schema_errors({"a": 1, "b": None, "c": "p", "d": {"k": "v"}}, schema) == []
    found = handoff.schema_errors({"a": True, "b": "y", "c": "z", "d": {"k": 1}, "e": 0}, schema)
    assert found == [
        "$.a: expected integer, got true",
        "$.b: 'y' does not match ^x",
        '$.c: must be one of "p", "q", got "z"',
        "$.d.k: expected string, got 1",
        "$: unknown field 'e'",
    ]
    assert handoff.schema_errors({}, schema) == ["$: missing 'a'"]


def test_the_shipped_schema_accepts_a_full_result() -> None:
    schema = json.loads(handoff.SCHEMA_FILE.read_text())
    assert handoff.schema_errors(tune_data(), schema) == []
    assert handoff.schema_errors(tune_data(image_digest=None), schema) == []
    assert set(schema["required"]) == set(tune_data()) - {"image_digest"}


def test_cli_prints_what_the_files_give(tmp_path: Path) -> None:
    code, payload = run_cli("handoff.py", write(tmp_path, "s.json", select_data()))
    assert code == 0 and payload["facts"]["profile"]["id"] == PROFILE
    code, payload = run_cli("handoff.py", write(tmp_path, "x.json", {"x": 1}))
    assert code == 2 and payload["errors"][0]["code"] == "handoff_unknown"


# --- select.json of nim-select-cli (a China registry image, the nim CLI runner) ---------------
# nim-select-cli runs the same nimselect.py, so the file has the same shape; the fixture is the
# real `select --out` facts of a lichan image (nim-select's test_select_china_ref_checks_...).

CHINA = json.loads((TESTS / "fixtures" / "select-china.json").read_text())
CHINA_IMAGE = CHINA["handoff"]["image"]
NIMCLI = ["--set", "runner=nimcli", "--set", "port=8000"]


def china_data(**change: Any) -> dict[str, Any]:
    data: dict[str, Any] = copy.deepcopy(CHINA)
    data["environment_summary"]["captured_at"] = now_text()
    data.update(change)
    return data


def china_tune(**change: Any) -> dict[str, Any]:
    return tune_data(
        model=CHINA["handoff"]["model"], image=CHINA_IMAGE, model_profile="fits-here", **change
    )


def test_the_cli_select_shape_maps_model_profile_and_device_ids(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    _, payload = new(capsys, *NIMCLI, *froms(write(tmp_path, "s.json", china_data())))
    facts = payload["facts"]
    assert (facts["target"], payload["state"]) == ("docker", "render")
    inputs = facts["inputs"]
    assert (inputs["model"], inputs["model_profile"], inputs["gpus"]) == (
        CHINA_IMAGE,
        "fits-here",
        "0",
    )
    assert facts["handoff"]["sources"]["model"] == "select"
    assert facts["handoff"]["profile"]["pinned_as"] == "--profile"
    from runners import nimcli

    script = nimcli.render_text(nd.run_inputs(nd.load_state(facts["run_id"])))
    assert f"nim run {CHINA_IMAGE} -p 8000 --gpus 0" in script
    assert "--profile fits-here" in script


def test_a_china_registry_image_is_kept_as_given(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    _, payload = new(capsys, *froms(write(tmp_path, "s.json", china_data())))  # docker runner
    inputs = payload["facts"]["inputs"]
    assert inputs["image"] == CHINA_IMAGE and inputs["gpus"] == "device=0"
    assert "nvcr.io" not in json.dumps(inputs)  # handoff.ngc_image is never used
    run = payload["facts"]["run_id"]
    render_and_approve(capsys, run)
    script = (nd.run_dir(run) / "docker-run.sh").read_text()
    assert f"  {CHINA_IMAGE}\n" in script and "-e NIM_MODEL_PROFILE=fits-here" in script


def test_the_cli_select_shape_refuses_without_a_profile(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    data = china_data(recommended=None, shortlist=[])
    del data["handoff"]
    code, payload = new(capsys, *NIMCLI, *froms(write(tmp_path, "s.json", data)))
    assert (code, payload["next"]) == (3, "ask_user")
    assert payload["errors"][0]["code"] == "select_no_profile"
    data = china_data(verdict="does_not_fit", recommended=None)
    del data["handoff"]
    code, payload = new(capsys, *NIMCLI, *froms(write(tmp_path, "s2.json", data)))
    assert (code, payload["errors"][0]["code"]) == (3, "select_does_not_fit")


def test_the_cli_select_shape_still_checks_the_tune_result(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    tune = china_tune(args=[], env={})
    tune["model_profile"] = "other-profile"
    paths = write(tmp_path, "s.json", china_data()), write(tmp_path, "t.json", tune)
    code, payload = new(capsys, *NIMCLI, *froms(*paths))
    message = payload["errors"][0]["message"]
    assert (code, payload["errors"][0]["code"]) == (3, "handoff_disagree")
    assert "'fits-here'" in message and "other-profile" in message


def test_a_user_set_overrides_the_cli_select_values(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    flags = ["--set", "gpus=2,3", "--set", "model_profile=abc"]
    _, payload = new(capsys, *NIMCLI, *flags, *froms(write(tmp_path, "s.json", china_data())))
    block = payload["facts"]["handoff"]
    assert payload["facts"]["inputs"]["gpus"] == "2,3"
    assert block["sources"]["gpus"] == "user (overrides select)"
    assert block["sources"]["model_profile"] == "user (overrides select)"
    assert block["profile"]["overridden_by_user"] == "abc"


def test_a_user_image_with_the_nim_cli_runner_moves_model_too(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    other = "nvcr.io/nim/other/model:9.9"
    flags = ["--set", f"image={other}", "--set", "model_profile=abc"]
    _, payload = new(capsys, *NIMCLI, *flags, *froms(write(tmp_path, "s.json", china_data())))
    inputs = payload["facts"]["inputs"]
    assert (inputs["image"], inputs["model"]) == (other, other)
    from runners import nimcli

    script = nimcli.render_text(nd.run_inputs(nd.load_state(payload["facts"]["run_id"])))
    assert f"nim run {other} " in script and CHINA_IMAGE not in script


def test_tuned_engine_args_with_the_nim_cli_runner_say_to_use_the_docker_runner(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    paths = write(tmp_path, "s.json", china_data()), write(tmp_path, "t.json", china_tune())
    code, payload = new(capsys, *NIMCLI, *froms(*paths))
    error = payload["errors"][0]
    assert (code, error["code"]) == (3, "runner_cannot_pass_args")
    assert "--max-num-seqs 64" in error["message"] and "docker runner" in error["fix"]
    # the same files work with the docker runner, which passes the arguments after the image
    _, payload = new(capsys, "--set", "confirm_handoff=yes", *froms(*paths))
    assert payload["facts"]["inputs"]["extra_args"] == ["--max-num-seqs", "64"]
