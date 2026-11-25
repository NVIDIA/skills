# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""scripts/write_recommended.py on the results of a real (fake-docker) campaign run."""

from __future__ import annotations

import importlib.util
import json
import re
import secrets
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from testenv import base_env

TESTS = Path(__file__).resolve().parent
SCRIPTS = TESTS.parent / "scripts"
SCRIPT = SCRIPTS / "write_recommended.py"
sys.path.insert(0, str(SCRIPTS))
import write_recommended as wr  # noqa: E402


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# the docker campaign test already holds the fake docker host and the AIPerf stand-in
docker_tests = _load("docker_campaign_fixture", TESTS / "test_docker_target.py")
DIGEST = "sha256:" + "ab" * 32
PROFILE = "tensorrt_llm-h100-bf16-tp2-pp1-throughput"
STAMP = "2026-10-29T08:00:00Z"


DOCKER_BLOCK: dict[str, Any] = {
    "image": "nvcr.io/nim/nvidia/vllm-model-free-nim:2.1.4",
    "gpus": "2,3",
    "env": {
        "NIM_MODEL_PATH": "hf://org/m",
        "NIM_SERVED_MODEL_NAME": "m",
        "NIM_MODEL_PROFILE": PROFILE,
    },
    "env_from_host": ["NGC_API_KEY"],
}
UNPINNED_BLOCK: dict[str, Any] = {
    **DOCKER_BLOCK,
    "env": {k: v for k, v in DOCKER_BLOCK["env"].items() if k != "NIM_MODEL_PROFILE"},
}


def measure(tmp: Path, **over: Any) -> Path:
    """Run a two-candidate docker campaign to the end: the untuned NIM and a tuned one."""
    fields: dict[str, Any] = {
        "docker": DOCKER_BLOCK,
        "model": "meta/llama-3.1-8b-instruct",
        "model_profile": PROFILE,
        "image_digest": DIGEST,
        "sla": {"metric": "ttft_p99", "max_ms": 500},
        "candidates": [
            {"id": "before", "gpus": 2},
            {"id": "mnt4096", "gpus": 2, "args": ["--max-num-batched-tokens", "4096"]},
        ],
    }
    fields.update(over)
    fields = {k: v for k, v in fields.items() if v is not None}  # None leaves the key out
    path, env = docker_tests.setup(tmp, **fields)
    code, _ = docker_tests.go(tmp, path, env)
    assert code == 0
    return tmp


@pytest.fixture(scope="module")
def finished(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A finished two-candidate docker campaign: the untuned NIM and --max-num-batched-tokens."""
    return measure(tmp_path_factory.mktemp("campaign"))


@pytest.fixture
def run_dir(finished: Path, tmp_path: Path) -> Path:
    shutil.copytree(finished, tmp_path / "run")
    return tmp_path / "run"


def env_file(tmp_path: Path) -> Path:
    """An env file holding one dummy secret, its value made at run time (no committed literal)."""
    path = tmp_path.parent / (tmp_path.name + "-nim.env")
    path.write_text("HF_TOKEN=" + secrets.token_hex(8) + "\n")
    return path


def edit_campaign(run_dir: Path, **changes: Any) -> None:
    path = run_dir / "campaign.json"
    c = json.loads(path.read_text())
    for key, value in changes.items():
        if value is None:
            c.pop(key, None)
        else:
            c[key] = value
    path.write_text(json.dumps(c))


def write(
    run_dir: Path, *extra: str, fake: dict[str, str] | None = None
) -> tuple[int, dict[str, Any]]:
    """Run the writer on RUN_DIR; FAKE holds switches for the fake docker (FAKE_NO_DIGEST, ...)."""
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(run_dir / "campaign.json"),
            "--out",
            str(run_dir / "r"),
            "--created-at",
            STAMP,
            *extra,
        ],
        env=base_env(
            {
                "NIM_TUNE_DOCKER": str(run_dir / "docker"),
                "FAKE_DB": str(run_dir / "db.json"),
                **(fake or {}),
            }
        ),
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=120,
        check=False,
    )
    out = json.loads(proc.stdout)
    assert set(out) == {"ok", "state", "next", "facts", "errors"}
    return proc.returncode, out


def written(run_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = json.loads((run_dir / "r/final/recommended_config.json").read_text())
    return result


def codes(out: dict[str, Any]) -> list[str]:
    return [e["code"] for e in out["errors"]]


def test_a_win_where_the_untuned_nim_overloaded_at_every_load_is_written(run_dir: Path) -> None:
    # the untuned NIM failed requests on every repeat, the candidate holds the target there:
    # no load to compare, but clearly better, never "measure again (more repeats)"
    runs = sorted(p for p in (run_dir / "r" / "before").glob("c*-r*") if p.is_dir())
    for f in (f for p in runs for f in p.rglob("*") if f.is_file()):
        if f.suffix in (".csv", ".log", ".json"):
            text = f.read_text(errors="replace")
            f.write_text(re.sub(r"(Request Error Rate \(%\)[,│ ]+)[0-9.]+", r"\g<1>4.50", text))
    state_file = run_dir / "r" / "campaign_state.json"
    st = json.loads(state_file.read_text())
    st["candidates"]["before"]["failed_runs"] = [p.name for p in runs]
    st["candidates"]["before"]["overloaded_runs"] = [p.name for p in runs]
    state_file.write_text(json.dumps(st))
    code, out = write(run_dir)
    assert code == 0 and out["state"] == "written", codes(out)
    assert written(run_dir)["baseline_won"] is False


def test_a_higher_boundary_alone_is_no_separation_when_the_untuned_nim_meets_the_target() -> None:
    same = [{"case": "t", "concurrency": 8, "metric": "ttft_p99", "verdict": "same"}]
    over = [["before", 16]]
    # the untuned NIM holds the target at 8 and overloads at 16, the tuned case holds 16:
    # within the spread at 8, so not promoted on the boundary alone
    boundary = {"before": {"max_concurrency": 8, "first_failing": 16},
                "t": {"max_concurrency": 16}}  # fmt: skip
    assert wr.separated(same, "t", "ttft_p99", boundary, "before", over) is not None
    # it never meets the target and overloads where the tuned case was measured to hold it
    boundary["before"] = {"max_concurrency": None, "first_failing": 16}
    picks = [{"concurrency": 16, "sla": {"meets": ["t"]}}]
    assert wr.separated(same, "t", "ttft_p99", boundary, "before", over, picks) is None
    # a boundary at or above 16 without a passing measurement at 16 itself is not enough
    picks = [{"concurrency": 8, "sla": {"meets": ["t"]}}, {"concurrency": 32, "sla": {"meets": ["t"]}}]
    boundary["t"] = {"max_concurrency": 32}
    assert wr.separated(same, "t", "ttft_p99", boundary, "before", over, picks) is not None


def test_winner_is_written_and_matches_the_schema(run_dir: Path) -> None:
    edit_campaign(run_dir, gpu_name="NVIDIA H100 NVL")
    code, out = write(run_dir)
    assert code == 0 and out["state"] == "written" and out["next"] == "done"
    record = written(run_dir)
    assert record == {
        "apiVersion": "nim-tune/v1",
        "kind": "RecommendedConfig",
        "created_at": STAMP,
        "campaign": "d1",
        "model": "meta/llama-3.1-8b-instruct",
        "image": "nvcr.io/nim/nvidia/vllm-model-free-nim:2.1.4",
        "image_digest": DIGEST,
        "model_profile": PROFILE,
        "target": "docker",
        "gpus": 2,
        "gpu_name": "NVIDIA H100 NVL",
        "replicas": 1,
        "args": ["--max-num-batched-tokens", "4096"],
        "env": DOCKER_BLOCK["env"],
        "env_from_host": ["NGC_API_KEY"],
        "baseline_won": False,
        "verdict": record["verdict"],
    }
    assert 0 < len(record["verdict"]) <= 300
    schema = json.loads(wr.SCHEMA.read_text())
    assert wr.schema_errors(record, schema) == []
    assert set(record) == set(schema["required"]) | {"image_digest"}


def test_an_env_file_is_refused_because_its_contents_are_never_read(tmp_path: Path) -> None:
    run_dir = measure(tmp_path, docker={**DOCKER_BLOCK, "env_file": str(env_file(tmp_path))})
    code, out = write(run_dir)
    assert code == 3 and codes(out) == ["env_file_unrecorded"]
    fix = out["errors"][0]["fix"]
    assert "docker.env" in fix and "env_from_host" in fix and "--env-file-secrets-only" in fix
    assert not (run_dir / "r/final/recommended_config.json").exists()


def test_a_secrets_only_env_file_is_written_with_a_warning(tmp_path: Path) -> None:
    run_dir = measure(tmp_path, docker={**DOCKER_BLOCK, "env_file": str(env_file(tmp_path))})
    code, out = write(run_dir, "--env-file-secrets-only")
    assert code == 0
    assert any("env_file" in w and "not recorded" in w for w in out["facts"]["warnings"])
    assert "HF_TOKEN" not in json.dumps(written(run_dir))


def test_a_campaign_changed_after_the_measurement_is_refused(run_dir: Path) -> None:
    c = json.loads((run_dir / "campaign.json").read_text())
    c["docker"]["env"]["NIM_KV_CACHE_PERCENT"] = "0.9"
    c["candidates"][1]["args"] = ["--max-num-batched-tokens", "8192"]
    edit_campaign(run_dir, docker=c["docker"], candidates=c["candidates"])
    code, out = write(run_dir)
    assert code == 3 and codes(out) == ["campaign_changed"]
    message = out["errors"][0]["message"]
    assert "mnt4096" in message and "args" in message and "docker.env" in message
    assert not (run_dir / "r/final/recommended_config.json").exists()


def test_a_workload_changed_after_the_measurement_is_refused(run_dir: Path) -> None:
    c = json.loads((run_dir / "campaign.json").read_text())
    c["workload"]["osl"] = c["workload"].get("osl", 256) * 2
    edit_campaign(run_dir, workload=c["workload"])
    code, out = write(run_dir)
    assert code == 3 and codes(out) == ["campaign_changed"]
    assert "workload" in out["errors"][0]["message"]


def test_the_baseline_is_bound_to_its_measurement_too(run_dir: Path) -> None:
    for src in (run_dir / "r/before").glob("c*-r*"):
        for csv in src.iterdir():
            shutil.copy(csv, run_dir / "r/mnt4096" / src.name / csv.name)
    c = json.loads((run_dir / "campaign.json").read_text())
    c["candidates"][0]["gpus"] = 1
    edit_campaign(run_dir, candidates=c["candidates"])
    code, out = write(run_dir)
    assert (code, codes(out)) == (3, ["campaign_changed"])
    assert "before" in out["errors"][0]["message"]


def test_results_without_the_recorded_settings_are_refused(run_dir: Path) -> None:
    state_path = run_dir / "r/campaign_state.json"
    state = json.loads(state_path.read_text())
    del state["candidates"]["mnt4096"]["measured"]
    state_path.write_text(json.dumps(state))
    code, out = write(run_dir)
    assert code == 3 and codes(out) == ["measured_config_unrecorded"]


def test_a_ranked_baseline_without_its_recorded_layout_is_refused(run_dir: Path) -> None:
    # without the record, a GPU edit to the baseline would rescale its tokens/s per GPU unseen
    state_path = run_dir / "r/campaign_state.json"
    state = json.loads(state_path.read_text())
    del state["candidates"]["before"]["measured"]
    state_path.write_text(json.dumps(state))
    c = json.loads((run_dir / "campaign.json").read_text())
    c["candidates"][0]["gpus"] = 1
    edit_campaign(run_dir, candidates=c["candidates"])
    code, out = write(run_dir)
    assert code == 3 and codes(out) == ["measured_config_unrecorded"]
    assert "before" in out["errors"][0]["message"]


def test_a_campaign_without_gpu_name_writes_null_and_warns(run_dir: Path) -> None:
    code, out = write(run_dir)
    assert code == 0 and written(run_dir)["gpu_name"] is None
    assert any("gpu_name is null" in w for w in out["facts"]["warnings"])


def test_the_runner_refuses_an_empty_gpu_name(run_dir: Path) -> None:
    edit_campaign(run_dir, gpu_name=" ")
    code, out = write(run_dir)
    assert code == 2 and codes(out) == ["bad_field_type"]


def test_the_two_schema_copies_are_identical() -> None:
    consumer = TESTS.parents[1] / "nim-deploy" / "references" / "tune-result.schema.json"
    if not consumer.exists():
        pytest.skip("nim-deploy is not installed next to nim-tune")
    assert consumer.read_bytes() == wr.SCHEMA.read_bytes()


def test_the_schema_catches_a_bad_file() -> None:
    schema = json.loads(wr.SCHEMA.read_text())
    problems = wr.schema_errors(
        {"apiVersion": "nim-tune/v2", "args": "x", "gpus": 0, "extra": 1}, schema
    )
    text = " ".join(problems)
    assert "apiVersion" in text and "$.args" in text and "$.gpus" in text
    assert "unknown field 'extra'" in text and "missing 'model'" in text


def test_dry_run_writes_nothing(run_dir: Path) -> None:
    code, out = write(run_dir, "--dry-run")
    assert code == 0 and out["facts"]["written"] is False
    assert not (run_dir / "r/final/recommended_config.json").exists()


def test_baseline_win_has_no_args_and_the_base_env(run_dir: Path) -> None:
    # the candidate's runs become the untuned NIM's numbers: nothing is separable, so it loses
    for src in (run_dir / "r/before").glob("c*-r*"):
        for csv in src.iterdir():
            shutil.copy(csv, run_dir / "r/mnt4096" / src.name / csv.name)
    code, out = write(run_dir)
    assert code == 0 and out["facts"]["baseline_won"] is True
    record = written(run_dir)
    # the same kind of env as a candidate win (test above): the base env it was measured with
    base_env = DOCKER_BLOCK["env"]
    assert record["args"] == [] and record["baseline_won"] is True
    assert (
        record["env"]
        == base_env
        == json.loads((run_dir / "campaign.json").read_text())["docker"]["env"]
    )
    assert wr.schema_errors(record, json.loads(wr.SCHEMA.read_text())) == []


def test_a_winner_inside_the_noise_is_not_promoted(run_dir: Path) -> None:
    comparisons = [
        {"case": "x", "metric": "ttft_p99", "verdict": "within_noise"},
        {"case": "x", "metric": "otps_avg", "verdict": "within_noise"},
    ]
    assert wr.separated(comparisons, "x", "ttft_p99") == (
        "x is not better than the untuned NIM outside the repeat spread"
    )
    comparisons.append({"case": "x", "metric": "ttft_p99", "verdict": "worse"})
    assert wr.separated(comparisons, "x", "ttft_p99") == (
        "x is worse than the untuned NIM on ttft_p99"
    )
    comparisons[-1]["verdict"] = "better"
    assert wr.separated(comparisons, "x", "ttft_p99") is None


def test_no_profile_is_an_error_never_a_guess(tmp_path: Path) -> None:
    run_dir = measure(tmp_path, docker=UNPINNED_BLOCK, model_profile=None)
    code, out = write(run_dir)
    assert code == 3 and codes(out) == ["profile_unknown"]
    assert "model_profile" in out["errors"][0]["fix"]
    assert not (run_dir / "r/final/recommended_config.json").exists()


def test_a_profile_label_without_a_pin_in_the_measured_env_is_refused(tmp_path: Path) -> None:
    # model_profile is only a label: the runner does not use it, so the NIM may have auto-selected
    run_dir = measure(tmp_path, docker=UNPINNED_BLOCK)
    code, out = write(run_dir)
    assert code == 3 and codes(out) == ["profile_not_pinned"]
    assert "NIM_MODEL_PROFILE" in out["errors"][0]["fix"] and "label" in out["errors"][0]["message"]
    assert not (run_dir / "r/final/recommended_config.json").exists()


def test_a_candidate_env_pin_counts_as_measured_evidence(tmp_path: Path) -> None:
    pinned = {"id": "mnt4096", "gpus": 2, "args": ["--max-num-batched-tokens", "4096"]}
    pinned["env"] = {"NIM_MODEL_PROFILE": PROFILE}
    pinned["engagement_pattern"] = "max-num-batched-tokens"
    candidates = [{"id": "before", "gpus": 2}, pinned]
    run_dir = measure(tmp_path, docker=UNPINNED_BLOCK, candidates=candidates)
    code, _ = write(run_dir)
    assert code == 0 and written(run_dir)["model_profile"] == PROFILE


def test_the_profile_comes_from_the_pinned_env(tmp_path: Path) -> None:
    run_dir = measure(tmp_path)
    edit_campaign(run_dir, model_profile=None)  # a label, not part of what was measured
    code, _ = write(run_dir)
    assert code == 0 and written(run_dir)["model_profile"] == PROFILE


def test_two_profiles_conflict(tmp_path: Path) -> None:
    other = {**DOCKER_BLOCK, "env": {**DOCKER_BLOCK["env"], "NIM_MODEL_PROFILE": "other-profile"}}
    run_dir = measure(tmp_path, docker=other)
    code, out = write(run_dir)
    assert code == 3 and codes(out) == ["profile_conflict"]
    assert "other-profile" in out["errors"][0]["message"] and PROFILE in out["errors"][0]["message"]


def test_no_digest_is_an_error_unless_the_user_allows_it(run_dir: Path) -> None:
    no_digest = {"FAKE_NO_DIGEST": "1"}
    code, out = write(run_dir, fake=no_digest)
    # campaign.json names a digest, but no measured container shows it: never pinned from there
    assert code == 3 and codes(out) == ["image_digest_incomplete"]
    assert DIGEST in out["errors"][0]["message"] and "mirror" in out["errors"][0]["fix"]
    code, out = write(run_dir, "--allow-unpinned-image", fake=no_digest)
    assert code == 0 and written(run_dir)["image_digest"] is None
    assert out["facts"]["warnings"]


def test_a_digest_in_the_image_reference_matches_the_measured_image(tmp_path: Path) -> None:
    pinned = {**DOCKER_BLOCK, "image": DOCKER_BLOCK["image"] + "@" + DIGEST}
    run_dir = measure(tmp_path, docker=pinned, image_digest=None)
    code, _ = write(run_dir)
    record = written(run_dir)
    assert code == 0 and record["image_digest"] == DIGEST and "@" not in record["image"]


def test_two_digests_conflict(tmp_path: Path) -> None:
    pinned = {**DOCKER_BLOCK, "image": DOCKER_BLOCK["image"] + "@sha256:" + "cd" * 32}
    run_dir = measure(tmp_path, docker=pinned)
    code, out = write(run_dir)
    assert code == 3 and codes(out) == ["image_digest_conflict"]


@pytest.mark.parametrize("source", ["campaign", "reference"])
@pytest.mark.parametrize("flag", [(), ("--allow-unpinned-image",)])
def test_docker_inspection_failure_never_uses_a_supplied_digest(
    tmp_path: Path, source: str, flag: tuple[str, ...]
) -> None:
    # An unavailable measured image must refuse the hand-off, even with a supplied digest.
    docker = {**DOCKER_BLOCK, "image": DOCKER_BLOCK["image"] + "@" + DIGEST}
    run_dir = measure(
        tmp_path,
        docker=docker if source == "reference" else DOCKER_BLOCK,
        image_digest=DIGEST if source == "campaign" else None,
    )
    code, out = write(run_dir, *flag, fake={"FAKE_IMAGE_INSPECT_FAIL": "1"})
    assert code == 3 and codes(out) == ["image_digest_unavailable"]
    assert not (run_dir / "r/final/recommended_config.json").exists()


def test_docker_digest_is_obtained_without_campaign_metadata(run_dir: Path) -> None:
    # Only the immutable image ID from the saved container may be inspected.
    edit_campaign(run_dir, image_digest=None)
    code, _ = write(run_dir)
    assert code == 0 and written(run_dir)["image_digest"] == DIGEST
    calls = json.loads((run_dir / "db.json").read_text())["calls"]
    assert ["image", "inspect"] in calls
    assert ["inspect", "sha256:" + "ef" * 32] in calls


def test_campaign_digest_must_match_the_measured_container(run_dir: Path) -> None:
    # A supplied digest remains a consistency check, never the source of the pin.
    edit_campaign(run_dir, image_digest="sha256:" + "cd" * 32)
    code, out = write(run_dir)
    assert code == 3 and codes(out) == ["image_digest_conflict"]


@pytest.mark.parametrize("contents", [None, "[]", '[{"Config": {"Image": "tag:1"}}]'])
def test_missing_container_evidence_refuses_the_docker_handoff(
    run_dir: Path, contents: str | None
) -> None:
    # Campaign metadata cannot stand in for missing measured container evidence.
    for path in (run_dir / "r/mnt4096").glob("*.pod.json"):
        if contents is None:
            path.unlink()
        else:
            path.write_text(contents)
    code, out = write(run_dir, "--allow-unpinned-image")
    # a deleted record is a measured replica without evidence; an empty one names no image
    want = "image_digest_incomplete" if contents is None else "image_digest_unavailable"
    assert code == 3 and codes(out) == [want]


def test_the_cluster_reported_digest_is_found(tmp_path: Path) -> None:
    pod = {"status": {"containerStatuses": [{"imageID": f"nvcr.io/nim/x@{DIGEST}"}]}}
    (tmp_path / "c1").mkdir()
    (tmp_path / "c1" / "p.pod.json").write_text(json.dumps(pod))
    assert wr.pod_digests(tmp_path, "c1") == {DIGEST}
    pod["status"]["containerStatuses"][0]["imageID"] = "sha256:abc"  # an id, not a digest
    (tmp_path / "c1" / "p.pod.json").write_text(json.dumps(pod))
    assert wr.pod_digests(tmp_path, "c1") == set()


def test_the_model_must_be_named(run_dir: Path) -> None:
    edit_campaign(run_dir, model=None)
    code, out = write(run_dir)
    assert code == 3 and codes(out) == ["model_unknown"]


def test_an_untagged_image_is_refused(tmp_path: Path) -> None:
    untagged = {**DOCKER_BLOCK, "image": "nvcr.io/nim/nvidia/vllm-model-free-nim"}
    run_dir = measure(tmp_path, docker=untagged)
    code, out = write(run_dir)
    assert code == 3 and codes(out) == ["image_untagged"]


def test_no_case_within_the_target_gives_no_recommendation(run_dir: Path) -> None:
    edit_campaign(run_dir, sla={"metric": "ttft_p99", "max_ms": 1})
    code, out = write(run_dir)
    assert code == 3 and codes(out) == ["no_recommendation"]


def test_a_campaign_without_a_target_is_refused(run_dir: Path) -> None:
    edit_campaign(run_dir, sla=None)
    code, out = write(run_dir)
    assert code == 3 and codes(out) == ["sla_missing"]


def test_missing_results_say_how_to_get_them(run_dir: Path) -> None:
    shutil.rmtree(run_dir / "r")
    code, out = write(run_dir)
    assert code == 3 and codes(out) == ["results_unreadable"]


def test_unengaged_or_unchecked_winners_are_not_promoted(run_dir: Path) -> None:
    state_path = run_dir / "r/campaign_state.json"
    state = json.loads(state_path.read_text())
    state["candidates"]["mnt4096"]["needs_accuracy_check"] = True
    state_path.write_text(json.dumps(state))
    code, out = write(run_dir)
    assert code == 3 and codes(out) == ["accuracy_check_needed"]
    code, _ = write(run_dir, "--accuracy-checked")
    assert code == 0
    state["candidates"]["mnt4096"]["engagement"] = {"p": {"state": "unconfirmed"}}
    state_path.write_text(json.dumps(state))
    code, out = write(run_dir, "--accuracy-checked")
    assert code == 3 and codes(out) == ["winner_not_engaged"]


def test_the_runner_refuses_a_bad_digest_or_profile_field(run_dir: Path) -> None:
    edit_campaign(run_dir, image_digest="latest", model_profile="")
    code, out = write(run_dir)
    assert code == 2 and set(codes(out)) == {"bad_image_digest", "bad_field_type"}


def test_the_nim_operator_record_gives_image_env_and_secret_names(tmp_path: Path) -> None:
    # the rendered NIMService the runner saves as deploy.json (targets.render)
    deployed = {
        "spec": {
            "image": {"repository": "nvcr.io/nim/meta/llama-3.1-8b-instruct", "tag": "1.8.0"},
            "env": [
                {"name": "NIM_SERVED_MODEL_NAME", "value": "m"},
                {"name": "HF_TOKEN", "valueFrom": {"secretKeyRef": {"name": "hf", "key": "t"}}},
            ],
        }
    }
    (tmp_path / "c1").mkdir()
    (tmp_path / "c1" / "deploy.json").write_text(json.dumps(deployed))
    image, env, secrets = wr.operator_config(tmp_path, "c1")
    assert image == "nvcr.io/nim/meta/llama-3.1-8b-instruct:1.8.0"
    assert env == {"NIM_SERVED_MODEL_NAME": "m"} and secrets == ["HF_TOKEN"]


def test_a_failed_candidate_is_never_hidden_behind_a_baseline_win(run_dir: Path) -> None:
    state_path = run_dir / "r/campaign_state.json"
    state = json.loads(state_path.read_text())
    state["candidates"]["mnt4096"].update(status="failed", kind="config", reason="x")
    state_path.write_text(json.dumps(state))
    code, out = write(run_dir)
    assert code == 3 and codes(out) == ["candidate_not_measured"]
    assert "mnt4096 (failed)" in out["errors"][0]["message"]


def test_a_refused_rerun_removes_the_earlier_result(run_dir: Path) -> None:
    assert write(run_dir)[0] == 0 and (run_dir / "r/final/recommended_config.json").exists()
    edit_campaign(run_dir, sla={"metric": "ttft_p99", "max_ms": 1})
    code, out = write(run_dir)
    assert code == 3 and codes(out) == ["no_recommendation"]
    # nim-deploy --from must not find the result this campaign no longer supports
    assert not (run_dir / "r/final/recommended_config.json").exists()


def test_an_earlier_result_that_cannot_be_removed_is_a_json_refusal(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert write(run_dir)[0] == 0
    # in process, so the failure is injected whatever the file-system privileges (root ignores
    # a read-only folder)
    def denied(self: Path, *a: Any, **k: Any) -> None:
        raise PermissionError(f"cannot remove {self.name}")

    monkeypatch.setattr(wr.Path, "unlink", denied)
    monkeypatch.setenv("NIM_TUNE_DOCKER", str(run_dir / "docker"))
    monkeypatch.setenv("FAKE_DB", str(run_dir / "db.json"))
    argv = [str(run_dir / "campaign.json"), "--out", str(run_dir / "r"), "--created-at", STAMP]
    with pytest.raises(SystemExit) as exc:
        wr.main(argv)
    out = json.loads(capsys.readouterr().out)
    assert exc.value.code == 3 and out["errors"][0]["code"] == "write_failed"
    assert (run_dir / "r/final/recommended_config.json").exists()  # nothing else happened


def test_an_invalid_campaign_also_removes_the_earlier_result(run_dir: Path) -> None:
    assert write(run_dir)[0] == 0 and (run_dir / "r/final/recommended_config.json").exists()
    (run_dir / "campaign.json").write_text("{not json")
    code, out = write(run_dir)
    assert code == 2 and out["next"] == "fix_errors"
    assert not (run_dir / "r/final/recommended_config.json").exists()


@pytest.mark.parametrize("state", ["null", "[]", '{"candidates": []}', '{"candidates": {"a": 1}}'])
def test_a_malformed_campaign_state_is_a_contract_error(run_dir: Path, state: str) -> None:
    (run_dir / "r/campaign_state.json").write_text(state)
    code, out = write(run_dir)
    assert code == 3 and codes(out) == ["results_unreadable"]


@pytest.mark.parametrize(
    "field", [{"failed_runs": None}, {"overloaded_runs": None}, {"failed_runs": "c8-r1"},
              {"overloaded_runs": [["c8-r1"]]}, {"engagement": []}, {"engagement": {"p": 1}}]
)  # fmt: skip
def test_a_malformed_run_list_or_engagement_is_a_contract_error(
    run_dir: Path, field: dict[str, Any]
) -> None:
    state_path = run_dir / "r/campaign_state.json"
    state = json.loads(state_path.read_text())
    state["candidates"]["mnt4096"].update(field)
    state_path.write_text(json.dumps(state))
    code, out = write(run_dir)  # a traceback would not parse as the JSON contract
    assert code == 3 and codes(out) == ["results_unreadable"]


def test_a_baseline_gpu_edit_is_refused_when_a_candidate_wins(run_dir: Path) -> None:
    # the baseline's GPU count sets its tokens/s per GPU in the ranking, so it is bound too
    c = json.loads((run_dir / "campaign.json").read_text())
    c["candidates"][0]["gpus"] = 1
    edit_campaign(run_dir, candidates=c["candidates"])
    code, out = write(run_dir)
    assert (code, codes(out)) == (3, ["campaign_changed"])
    assert "before.gpus" in out["errors"][0]["message"]


def test_a_baseline_that_misses_the_target_is_never_handed_off(run_dir: Path) -> None:
    # The baseline gets the candidate's numbers with TTFT p99 1% higher: the two are not
    # separable, yet only the candidate meets a target set at its own p99 (80 ms at 8 concurrent).
    for src in (run_dir / "r/mnt4096").glob("c*-r*"):
        log = (src / "aiperf.log").read_text().splitlines()
        for i, line in enumerate(log):
            cells = line.split("\u2502")
            if len(cells) == 10 and cells[1] == "Time to First Token (ms)":
                cells[5] = f"{float(cells[5].replace(',', '')) * 1.01:,.2f}"
                log[i] = "\u2502".join(cells)
        (run_dir / "r/before" / src.name / "aiperf.log").write_text("\n".join(log) + "\n")
    edit_campaign(run_dir, sla={"metric": "ttft_p99", "max_ms": 80})
    code, out = write(run_dir)
    assert code == 3 and codes(out) == ["no_recommendation"]
    assert "mnt4096 is not promoted" in out["errors"][0]["message"]
    assert not (run_dir / "r/final/recommended_config.json").exists()


def test_a_digest_of_another_repository_is_never_used(tmp_path: Path) -> None:
    # the fake host reports a digest for nvcr.io/nim/nvidia/vllm-model-free-nim only
    mirror = {**DOCKER_BLOCK, "image": "registry.example.com/nim/vllm-model-free-nim:2.1.4"}
    run_dir = measure(tmp_path, docker=mirror, image_digest=None)
    code, out = write(run_dir)
    assert code == 3 and codes(out) == ["image_digest_incomplete"]
    code, _ = write(run_dir, "--allow-unpinned-image")
    assert code == 0 and written(run_dir)["image_digest"] is None


def test_repository_names_are_normalized() -> None:
    assert wr.repository("nginx:1.25") == wr.repository("docker.io/library/nginx@" + DIGEST)
    assert wr.repository("localhost:5000/x:1") == "localhost:5000/x"
    assert wr.repository("docker-pullable://nvcr.io/nim/x@" + DIGEST) == "nvcr.io/nim/x"
    assert wr.repository("nvcr.io/nim/x:1.8.0") == "nvcr.io/nim/x"
    # Docker's legacy registry name is the same repository; another repository never is
    assert wr.repository("index.docker.io/library/nginx:1.25") == "docker.io/library/nginx"
    assert wr.repository("index.docker.io/nginx") == wr.repository("nginx")
    assert wr.repository("index.docker.io/library/nginx") != wr.repository("docker.io/x/nginx")


def test_a_record_left_by_an_earlier_deploy_is_not_the_measured_replicas(tmp_path: Path) -> None:
    old = "sha256:" + "ef" * 32
    (tmp_path / "c1").mkdir()
    for name, digest in (("p-old", old), ("p-new", DIGEST)):
        status = {"image": "nvcr.io/nim/x:1.8.0", "imageID": f"nvcr.io/nim/x@{digest}"}
        pod = {"status": {"containerStatuses": [status]}}
        (tmp_path / "c1" / f"{name}.pod.json").write_text(json.dumps(pod))
    # only the measured replica's record counts: the leftover never adds a conflicting digest
    assert wr.pod_digests(tmp_path, "c1", "nvcr.io/nim/x:1.8.0", ["p-new"]) == {DIGEST}
    src = (Path(wr.__file__)).read_text()
    assert "pod_digests(out, chosen, image, list(engaged))" in src  # the writer names them


def test_only_the_nim_containers_digest_counts(tmp_path: Path) -> None:
    side = "sha256:" + "cd" * 32
    nim = {"image": "nvcr.io/nim/x:1.8.0", "imageID": f"nvcr.io/nim/x@{DIGEST}"}
    sidecar = {"image": "docker.io/istio/proxyv2:1.2", "imageID": f"docker.io/istio/proxyv2@{side}"}
    (tmp_path / "c1").mkdir()
    pod = {"status": {"containerStatuses": [nim, sidecar]}}
    (tmp_path / "c1" / "p.pod.json").write_text(json.dumps(pod))
    assert wr.pod_digests(tmp_path, "c1", "nvcr.io/nim/x:1.8.0") == {DIGEST}
    # a NIM without a pullable digest never borrows the sidecar's
    pod["status"]["containerStatuses"][0]["imageID"] = "sha256:abc"
    (tmp_path / "c1" / "p.pod.json").write_text(json.dumps(pod))
    assert wr.pod_digests(tmp_path, "c1", "nvcr.io/nim/x:1.8.0") == set()


def test_a_helper_of_the_same_repository_never_lends_its_digest(tmp_path: Path) -> None:
    helper_digest = "sha256:" + "ef" * 32
    pod = {
        "spec": {"containers": [{"name": "nim"}, {"name": "helper"}]},
        "status": {
            "containerStatuses": [
                {"name": "nim", "image": "nvcr.io/nim/x:1.8.0", "imageID": "sha256:abc"},
                {"name": "helper", "image": "nvcr.io/nim/x:tools",
                 "imageID": f"nvcr.io/nim/x@{helper_digest}"},
            ]
        },
    }  # fmt: skip
    (tmp_path / "c1").mkdir()
    (tmp_path / "c1" / "p.pod.json").write_text(json.dumps(pod))
    # the NIM (first container) has no pullable digest: the helper's is never used instead
    assert wr.pod_digests(tmp_path, "c1", "nvcr.io/nim/x:1.8.0") == set()
    pod["metadata"] = {"annotations": {"kubectl.kubernetes.io/default-container": "helper"}}
    (tmp_path / "c1" / "p.pod.json").write_text(json.dumps(pod))
    assert wr.pod_digests(tmp_path, "c1", "nvcr.io/nim/x:1.8.0") == {helper_digest}


def test_two_same_repository_statuses_of_an_unnamed_container_lend_no_digest(
    tmp_path: Path,
) -> None:
    # no spec and no default-container annotation: neither status is known to be the NIM
    helper = "sha256:" + "ef" * 32
    pod = {
        "status": {
            "containerStatuses": [
                {"image": "nvcr.io/nim/x:1.8.0", "imageID": f"nvcr.io/nim/x@{DIGEST}"},
                {"image": "nvcr.io/nim/x:tools", "imageID": f"nvcr.io/nim/x@{helper}"},
            ]
        }
    }
    (tmp_path / "c1").mkdir()
    (tmp_path / "c1" / "p.pod.json").write_text(json.dumps(pod))
    assert wr.pod_digests(tmp_path, "c1", "nvcr.io/nim/x:1.8.0") == set()


def test_a_measured_digest_of_another_repository_is_refused(tmp_path: Path, capsys: Any) -> None:
    # status.image names the repository and only the imageID another name: containerd reports
    # the first name the node stored the same image under (a mirror pre-pull). No digest of the
    # repository, so missing evidence, as before; never pinned and never a hard refusal
    other = {"image": "nvcr.io/nim/x:1.8.0", "imageID": f"mirror.example.com/x@{DIGEST}"}
    (tmp_path / "c1").mkdir()
    pod: dict[str, Any] = {"status": {"containerStatuses": [other]}}
    (tmp_path / "c1" / "p.pod.json").write_text(json.dumps(pod))
    call = lambda: wr.pod_digests(tmp_path, "c1", "nvcr.io/nim/x:1.8.0")  # noqa: E731
    assert call() == set()
    # the named NIM container without a status.image: nothing says the repository ran
    pod = {
        "spec": {"containers": [{"name": "nim"}]},
        "status": {"containerStatuses": [{"name": "nim",
                                          "imageID": f"mirror.example.com/x@{DIGEST}"}]},
    }  # fmt: skip
    (tmp_path / "c1" / "p.pod.json").write_text(json.dumps(pod))
    assert refused(capsys, call) == ["image_mismatch"]
    # the named NIM container runs another repository's image altogether
    pod = {
        "spec": {"containers": [{"name": "nim"}]},
        "status": {"containerStatuses": [{"name": "nim", "image": "mirror.example.com/x:1.8.0",
                                          "imageID": f"mirror.example.com/x@{DIGEST}"}]},
    }  # fmt: skip
    (tmp_path / "c1" / "p.pod.json").write_text(json.dumps(pod))
    assert refused(capsys, call) == ["image_mismatch"]
    # a NIM status without a pullable digest is missing evidence, not a contradiction
    pod["status"]["containerStatuses"][0]["imageID"] = "sha256:abc"
    (tmp_path / "c1" / "p.pod.json").write_text(json.dumps(pod))
    assert wr.pod_digests(tmp_path, "c1", "nvcr.io/nim/x:1.8.0") == set()


def test_an_annotated_init_container_nims_digest_is_its_own(tmp_path: Path, capsys: Any) -> None:
    # the NIM runs as a restartable init container named by the default-container annotation:
    # its digest comes from initContainerStatuses, never from the app container beside it
    helper = "sha256:" + "ef" * 32
    pod: dict[str, Any] = {
        "metadata": {"annotations": {"kubectl.kubernetes.io/default-container": "nim"}},
        "spec": {"containers": [{"name": "app"}], "initContainers": [{"name": "nim"}]},
        "status": {
            "containerStatuses": [{"name": "app", "image": "nvcr.io/nim/x:tools",
                                   "imageID": f"nvcr.io/nim/x@{helper}"}],
            "initContainerStatuses": [{"name": "nim", "image": "nvcr.io/nim/x:1.8.0",
                                       "imageID": f"nvcr.io/nim/x@{DIGEST}"}],
        },
    }  # fmt: skip
    (tmp_path / "c1").mkdir()
    (tmp_path / "c1" / "p.pod.json").write_text(json.dumps(pod))
    assert wr.pod_digests(tmp_path, "c1", "nvcr.io/nim/x:1.8.0") == {DIGEST}
    pod["spec"]["initContainers"] = "nim"  # malformed: refused, never read as no container
    (tmp_path / "c1" / "p.pod.json").write_text(json.dumps(pod))
    call = lambda: wr.pod_digests(tmp_path, "c1", "nvcr.io/nim/x:1.8.0")  # noqa: E731
    assert refused(capsys, call) == ["pod_record_malformed"]


def refused(capsys: Any, call: Any) -> list[str]:
    with pytest.raises(SystemExit):
        call()
    out = json.loads(capsys.readouterr().out)
    assert out["state"] == "refused"
    return [e["code"] for e in out["errors"]]


@pytest.mark.parametrize("second", ["sha256:abc", None])  # a bare image id, an unreadable record
def test_every_measured_replica_must_show_the_digest(
    tmp_path: Path, capsys: Any, second: str | None
) -> None:
    def pod(image_id: str) -> str:
        status = {"image": "nvcr.io/nim/x:1.8.0", "imageID": image_id}
        return json.dumps({"status": {"containerStatuses": [status]}})

    (tmp_path / "c1").mkdir()
    (tmp_path / "c1" / "p1.pod.json").write_text(pod(f"nvcr.io/nim/x@{DIGEST}"))
    (tmp_path / "c1" / "p2.pod.json").write_text(pod(f"nvcr.io/nim/x@{DIGEST}"))
    assert wr.pod_digests(tmp_path, "c1", "nvcr.io/nim/x:1.8.0") == {DIGEST}
    (tmp_path / "c1" / "p2.pod.json").write_text(pod(second) if second else "{broken")
    call = lambda: wr.pod_digests(tmp_path, "c1", "nvcr.io/nim/x:1.8.0")  # noqa: E731
    assert refused(capsys, call) == ["image_digest_incomplete"]


def test_every_measured_docker_container_must_show_the_digest(
    tmp_path: Path, capsys: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    pulled, local = "sha256:" + "11" * 32, "sha256:" + "22" * 32
    repo_digests = {pulled: [f"nvcr.io/nim/x@{DIGEST}"], local: []}

    class Host:
        def __init__(self, *_: Any) -> None:
            pass

        def docker(self, *a: str) -> Any:
            out = json.dumps([{"RepoDigests": repo_digests[a[-1]]}])
            return subprocess.CompletedProcess(a, 0, out, "")

    monkeypatch.setattr(wr.targets, "DockerTarget", Host)
    c = {"docker": {"image": "nvcr.io/nim/x:1.8.0"}}
    (tmp_path / "c1").mkdir()
    (tmp_path / "c1" / "a.pod.json").write_text(json.dumps([{"Image": pulled}]))
    assert wr.docker_digests(c, tmp_path, "c1") == {DIGEST}
    (tmp_path / "c1" / "b.pod.json").write_text(json.dumps([{"Image": local}]))
    call = lambda: wr.docker_digests(c, tmp_path, "c1")  # noqa: E731
    assert refused(capsys, call) == ["image_digest_incomplete"]


@pytest.mark.parametrize(
    "pod",
    [
        {"spec": "broken"},
        {"spec": []},  # falsey, still the wrong type
        {"spec": {"containers": False}},
        {"status": {"containerStatuses": "x"}},
        {"status": {"containerStatuses": ["x"]}},
    ],
)
def test_a_malformed_pod_record_is_a_json_refusal(
    tmp_path: Path, capsys: Any, pod: dict[str, Any]
) -> None:
    (tmp_path / "c1").mkdir()
    (tmp_path / "c1" / "p.pod.json").write_text(json.dumps(pod))
    with pytest.raises(SystemExit):
        wr.pod_digests(tmp_path, "c1")
    out = json.loads(capsys.readouterr().out)
    assert out["state"] == "refused"
    assert [e["code"] for e in out["errors"]] == ["pod_record_malformed"]


@pytest.mark.parametrize(
    "pod", [{}, {"spec": None, "status": None}, {"spec": {}, "status": {"containerStatuses": []}}]
)
def test_missing_null_and_empty_pod_parts_are_not_malformed(
    tmp_path: Path, pod: dict[str, Any]
) -> None:
    (tmp_path / "c1").mkdir()
    (tmp_path / "c1" / "p.pod.json").write_text(json.dumps(pod))
    assert wr.pod_digests(tmp_path, "c1") == set()


def test_an_overloaded_load_does_not_block_promoting_the_winner() -> None:
    state = {
        "status": "measured",
        "failed_runs": ["c128-r1"],
        "overloaded_runs": ["c128-r1"],
        "engagement": {"p-0": {"state": "engaged"}},
    }
    assert wr.promotion_problem("c1", state, accuracy_checked=False) is None
    state["failed_runs"].append("c8-r2")  # a run that never produced a result is still missing
    problem = wr.promotion_problem("c1", state, accuracy_checked=False)
    assert problem is not None and problem[0] == "winner_not_measured"


def test_a_short_named_credential_never_reads_as_a_changed_campaign(monkeypatch: Any) -> None:
    import targets

    short = "seqs"[1:3]  # fixed, and part of the settings below, so masking would change them
    monkeypatch.setenv("SHORT_PASS", short)  # made at test time; placed inside the settings
    sc = targets.Scrubber()
    sc.add_env(["SHORT_PASS"])
    workload = {"model": f"m{short}", "tokenizer": "t", "concurrency": [8]}
    c = {"target": "docker", "docker": {"image": "nim:1", "env": {"NIM_X": f"n{short}"}},
         "workload": workload}  # fmt: skip
    cand = {"id": "a", "gpus": 1, "replicas": 1, "args": [f"--max-{short}-seqs", "64"],
            "env": {"NIM_MAX_MODEL_LEN": f"4096{short}"}}  # fmt: skip
    recorded = {k: cand[k] for k in ("gpus", "replicas", "args", "env")}
    recorded["workload"] = workload
    recorded["docker"] = {"image": "nim:1", "env": {"NIM_X": f"n{short}"}, "env_from_host": None,
                          "env_file": None}  # fmt: skip
    state = json.loads(sc.dumps({"candidates": {"a": {"measured": recorded}}}, indent=1))
    assert wr.changed_since_measured(state["candidates"]["a"]["measured"], c, cand) == []


def test_a_record_without_the_workload_is_unrecorded_not_a_changed_campaign(run_dir: Path) -> None:
    # a state from an earlier runner kept no workload: the campaign did not change
    state_path = run_dir / "r/campaign_state.json"
    state = json.loads(state_path.read_text())
    del state["candidates"]["mnt4096"]["measured"]["workload"]
    state_path.write_text(json.dumps(state))
    code, out = write(run_dir)
    assert code == 3 and codes(out) == ["measured_config_unrecorded"]


def test_a_measured_replica_without_its_record_is_never_pinned_from_the_others(
    run_dir: Path,
) -> None:
    # the state lists two measured replicas; only one record is left: the digest it shows is
    # never pinned for both
    state_path = run_dir / "r/campaign_state.json"
    state = json.loads(state_path.read_text())
    state["candidates"]["mnt4096"]["engagement"]["gone-1"] = {"state": "engaged"}
    state_path.write_text(json.dumps(state))
    code, out = write(run_dir)
    assert code == 3 and codes(out) == ["image_digest_incomplete"]
    assert "gone-1" in out["errors"][0]["message"] and "Restore" in out["errors"][0]["fix"]


def test_a_case_left_out_unrepaired_is_named_in_the_written_result(run_dir: Path) -> None:
    state_path = run_dir / "r/campaign_state.json"
    state = json.loads(state_path.read_text())
    state["candidates"]["mnt4096"]["unrepaired"] = ["c8-r2"]  # compare leaves it out
    state_path.write_text(json.dumps(state))
    code, out = write(run_dir)
    assert code == 0 and out["facts"]["winner"] == "before"
    assert any("mnt4096 left out" in w for w in out["facts"]["warnings"])


# ---------------------------------------------------------------- review round 6


def operator_run(run_dir: Path, image_id: str) -> None:
    """Turn the finished Docker campaign into a NIM Operator one whose replicas report IMAGE_ID:
    the same exports and state, a saved NIMService and a pod record per measured replica."""
    image = DOCKER_BLOCK["image"]
    env = [{"name": k, "value": v} for k, v in DOCKER_BLOCK["env"].items()]
    edit_campaign(run_dir, target="nim-operator", context="ctx", namespace="ns",
                  base_manifest="base.json", client_manifest="client.json",
                  docker=None)  # fmt: skip
    state_path = run_dir / "r/campaign_state.json"
    state = json.loads(state_path.read_text())
    for cid in ("before", "mnt4096"):
        repo, _, tag = image.rpartition(":")
        deployed = {"spec": {"image": {"repository": repo, "tag": tag}, "env": env}}
        (run_dir / "r" / cid / "deploy.json").write_text(json.dumps(deployed))
        pod = {"spec": {"containers": [{"name": "nim"}]},
               "status": {"containerStatuses": [{"name": "nim", "image": image,
                                                 "imageID": image_id}]}}  # fmt: skip
        for old in (run_dir / "r" / cid).glob("*.pod.json"):
            old.unlink()
        for name in state["candidates"][cid]["engagement"]:
            (run_dir / "r" / cid / f"{name}.pod.json").write_text(json.dumps(pod))
        state["candidates"][cid]["image_id"] = image_id.rsplit("@", 1)[-1]  # its digest
    state_path.write_text(json.dumps(state))


def test_a_campaign_digest_never_stands_in_for_measured_replicas(run_dir: Path) -> None:
    # every replica has its record, none shows a pullable digest of the repository: campaign
    # image_digest can only confirm what was measured, never replace it
    operator_run(run_dir, "sha256:" + "ef" * 32)  # an image id, not a pullable digest
    code, out = write(run_dir)
    assert code == 3 and codes(out) == ["image_digest_incomplete"], out
    assert DIGEST in out["errors"][0]["message"]
    assert not (run_dir / "r/final/recommended_config.json").exists()
    code, _ = write(run_dir, "--allow-unpinned-image")  # the user accepts an unpinned image
    assert code == 0 and written(run_dir)["image_digest"] is None
    # the replicas show the digest: pinned, and campaign.json confirms it
    operator_run(run_dir, f"nvcr.io/nim/nvidia/vllm-model-free-nim@{DIGEST}")
    code, out = write(run_dir)
    assert code == 0 and written(run_dir)["image_digest"] == DIGEST, out
    edit_campaign(run_dir, image_digest="sha256:" + "cd" * 32)  # one that contradicts them
    code, out = write(run_dir)
    assert code == 3 and codes(out) == ["image_digest_conflict"]


def test_cases_measured_on_different_images_are_never_written(run_dir: Path) -> None:
    state_path = run_dir / "r/campaign_state.json"
    state = json.loads(state_path.read_text())
    assert state["candidates"]["before"]["image_id"] == state["candidates"]["mnt4096"]["image_id"]
    state["candidates"]["before"]["image_id"] = "sha256:" + "12" * 32
    state_path.write_text(json.dumps(state))
    code, out = write(run_dir)
    assert code == 3 and codes(out) == ["image_changed"]
    # the winner's saved record shows another image than its runs were measured on
    state["candidates"]["before"]["image_id"] = state["candidates"]["mnt4096"]["image_id"]
    state_path.write_text(json.dumps(state))
    for path in (run_dir / "r/mnt4096").glob("*.pod.json"):
        path.write_text(json.dumps([{"Image": "sha256:" + "12" * 32}]))
    code, out = write(run_dir)
    assert code == 3 and codes(out) == ["image_changed"]
    # a state without the image (an earlier runner) is unrecorded
    del state["candidates"]["mnt4096"]["image_id"]
    state_path.write_text(json.dumps(state))
    code, out = write(run_dir)
    assert code == 3 and "measured_config_unrecorded" in codes(out)


def test_a_moved_tag_during_a_repair_never_ranks_two_images(tmp_path: Path) -> None:
    # mnt4096 measured its repeats on image A; one is lost, and the redeploy that repairs it
    # runs image B (the tag moved): every run of it is measured again on B, never A and B mixed
    run_dir = measure(tmp_path)
    env = base_env(
        {"NIM_TUNE_DOCKER": str(run_dir / "docker"), "FAKE_DB": str(run_dir / "db.json")}
    )
    env["HOME"] = str(run_dir / "home")
    (run_dir / "r/mnt4096/c8-r3/aiperf.log").unlink()
    fake = run_dir / "docker"
    fake.write_text(fake.read_text().replace('"Image": "sha256:" + "ef" * 32',
                                             '"Image": "sha256:" + "12" * 32'))  # fmt: skip
    db = json.loads((run_dir / "db.json").read_text())
    before = len(db["profiled"])
    code, _ = docker_tests.go(run_dir, run_dir / "campaign.json", env)
    assert code == 0
    profiled = json.loads((run_dir / "db.json").read_text())["profiled"][before:]
    assert profiled == ["http://nimtune-me-mnt4096:8000"] * 6  # 2 loads x 3 repeats, all on B
    state = json.loads((run_dir / "r/campaign_state.json").read_text())
    assert state["candidates"]["mnt4096"]["image_id"] == "sha256:" + "12" * 32
    # the untuned NIM ran A: the two are not compared as one image
    code, out = write(run_dir)
    assert code == 3 and codes(out) == ["image_changed"]


def test_malformed_annotations_are_a_malformed_pod_record(tmp_path: Path, capsys: Any) -> None:
    pod = {"metadata": {"name": "p", "annotations": ["junk"]},
           "spec": {"containers": [{"name": "nim", "image": "nvcr.io/nim/x:1"}]},
           "status": {"containerStatuses": [{"name": "nim", "image": "nvcr.io/nim/x:1",
                                             "imageID": f"nvcr.io/nim/x@{DIGEST}"}]}}  # fmt: skip
    (tmp_path / "c1").mkdir()
    (tmp_path / "c1" / "p.pod.json").write_text(json.dumps(pod))
    call = lambda: wr.pod_digests(tmp_path, "c1", "nvcr.io/nim/x:1")  # noqa: E731
    assert refused(capsys, call) == ["pod_record_malformed"]
