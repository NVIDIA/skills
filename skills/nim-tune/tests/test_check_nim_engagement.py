# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract test for scripts/check_nim_engagement.py: one JSON object and a documented exit."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import pytest
from testenv import base_env

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_nim_engagement.py"
# launch line as logged by vllm-model-free-nim 2.1.4 on the NIM Operator (2026-10-25)
LAUNCH = json.dumps(
    {
        "level": "INFO",
        "file_name": "lifecycle.py",
        "message": "Launching vLLM: vllm serve /opt/nim/.cache/tmp/nim_x   "
        "--tensor-parallel-size 2 --pipeline-parallel-size 1 --gpu-memory-utilization 0.92 "
        "--port 8001 --served-model-name qwen7b",
    }
)


def pod(args: list[str], env: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "metadata": {"uid": "u1"},
        "spec": {
            "containers": [
                {"image": "nvcr.io/nim/nvidia/vllm-model-free-nim:2.1.4", "args": args, "env": env}
            ]
        },
        "status": {
            "phase": "Running",
            "containerStatuses": [{"imageID": "sha256:abc", "restartCount": 0}],
        },
    }


def check(
    tmp_path: Path, spec: dict[str, Any], log: str, *flags: str
) -> tuple[int, dict[str, Any]]:
    spec_file, log_file = tmp_path / "pod.json", tmp_path / "startup.log"
    spec_file.write_text(json.dumps(spec))
    log_file.write_text(log)
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "k8s",
            "p1",
            "--spec-file",
            str(spec_file),
            "--log-file",
            str(log_file),
            *flags,
        ],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=60,
        check=False,
        env=base_env(),  # never the developer's: their credentials would join the scrubber
    )
    out = json.loads(proc.stdout)
    assert set(out) == {"ok", "state", "next", "facts", "errors"}
    return proc.returncode, out


def test_arg_in_launch_line_is_engaged(tmp_path: Path) -> None:
    code, out = check(
        tmp_path,
        pod(["--tensor-parallel-size", "2"], []),
        "boot\n" + LAUNCH + "\n",
        "--arg",
        "--tensor-parallel-size 2",
    )
    assert code == 0 and out["state"] == "engaged" and out["next"] == "smoke_test"


def test_a_value_that_embeds_a_short_secret_is_masked_as_a_whole(tmp_path: Path) -> None:
    # $(NAME) splices a short credential into a value and an argument where letters touch it
    env = [{"name": "HF_TOKEN", "value": "zq7"},
           {"name": "NIM_X", "value": "pre$(HF_TOKEN)x"}]  # fmt: skip
    args = ["--tensor-parallel-size", "2", "--extra", "a$(NIM_X)b", "--max-num-seqs=v$(HF_TOKEN)w"]
    # the check row reports the delivered value after "=" ("vzq7w"): masked too
    code, out = check(tmp_path, pod(args, env), LAUNCH + "\n", "--arg", "--max-num-seqs 64")
    text = json.dumps(out)
    assert "zq7" not in text, text  # neither the derived value nor the argument shows it


def test_a_redeclared_variable_keeps_the_secret_its_old_value_carried(tmp_path: Path) -> None:
    # NIM_X copies the short credential; declaring NIM_X again from $(NIM_X) glues a letter to
    # it, so only its taint (not the whole-token mask of "zq7") can keep it out of the output
    env = [{"name": "HF_TOKEN", "value": "zq7"},
           {"name": "NIM_X", "value": "$(HF_TOKEN)"},
           {"name": "NIM_X", "value": "$(NIM_X)y"}]  # fmt: skip
    args = ["--tensor-parallel-size", "2", "--extra", "a$(NIM_X)b"]
    code, out = check(tmp_path, pod(args, env), LAUNCH + "\n", "--arg", "--tensor-parallel-size 2")
    text = json.dumps(out)
    assert "zq7" not in text, text


def test_an_env_entry_with_no_value_and_no_source_is_empty(tmp_path: Path) -> None:
    # Kubernetes omits an empty value from the pod JSON: {"name": "EMPTY"} is the value ""
    spec = pod(["--served-model-name", "qwen7b$(EMPTY)"], [{"name": "EMPTY"}])
    code, out = check(
        tmp_path, spec, LAUNCH + "\n", "--arg", "--served-model-name qwen7b"
    )
    assert code == 0 and out["state"] == "engaged", out


def test_a_later_message_naming_vllm_serve_is_not_the_launch(tmp_path: Path) -> None:
    # NIM 2.1.6 prints this after startup; read as the launch line, every argument looked unused
    later = json.dumps(
        {
            "level": "INFO",
            "message": "Scale-out endpoints are disabled. Restart vllm serve with "
            "--enable-scale-out to enable them.",
        }
    )
    code, out = check(
        tmp_path,
        pod(["--tensor-parallel-size", "2"], []),
        "boot\n" + LAUNCH + "\n" + later + "\n",
        "--arg",
        "--tensor-parallel-size 2",
    )
    assert code == 0 and out["state"] == "engaged"
    assert out["facts"]["launch_line"].startswith("Launching vLLM")


SCALE_OUT = (
    "INFO 11-01 12:00:05 [api.py:10] Scale-out endpoints are disabled. Restart vllm serve with "
    "--enable-scale-out to enable them."
)


def test_a_prefix_before_the_launch_marker_is_still_the_launch(tmp_path: Path) -> None:
    # a text-format log: the marker is not at the start of the message
    log = (
        "[2026-11-01 12:00:00] INFO lifecycle.py: Launching vLLM: vllm serve /m "
        "--tensor-parallel-size 2\n" + SCALE_OUT + "\n"
    )
    code, out = check(
        tmp_path, pod(["--tensor-parallel-size", "2"], []), log, "--arg=--tensor-parallel-size 2"
    )
    assert code == 0 and out["state"] == "engaged"
    assert out["facts"]["launch_line"].startswith("Launching vLLM: vllm serve /m")


def test_a_sentence_naming_the_command_never_confirms_a_flag(tmp_path: Path) -> None:
    # no launch line at all: the words of the scale-out sentence must not be read as arguments
    code, out = check(
        tmp_path, pod(["--enable-scale-out"], []), "boot\n" + SCALE_OUT + "\n",
        "--arg=--enable-scale-out",
    )  # fmt: skip
    assert code == 3 and out["facts"]["launch_line"] is None
    assert "no_launch_line" in {e["code"] for e in out["errors"]}


def test_a_bare_command_after_a_restart_is_the_last_launch(tmp_path: Path) -> None:
    # docker logs keep every restart; the live engine runs the last command, marker or not
    first = json.dumps({"message": "Launching vLLM: vllm serve /m --max-num-seqs 64"})
    code, out = check(
        tmp_path, pod(["--max-num-seqs", "64"], []),
        first + "\nvllm serve /m --max-num-seqs 256\n", "--arg=--max-num-seqs 64",
    )  # fmt: skip
    assert code == 3 and out["state"] == "mismatch"


def test_any_entrypoint_after_the_launch_marker_is_the_launch(tmp_path: Path) -> None:
    # the marker says it is a launch, whatever command follows it
    for command in (
        "python3 -m vllm.entrypoints.openai.api_server --model /m",
        "sglang serve --model-path /m",
    ):
        line = json.dumps({"message": f"Launching vLLM: {command} --tensor-parallel-size 2"})
        code, out = check(
            tmp_path, pod(["--tensor-parallel-size", "2"], []), line + "\n" + SCALE_OUT,
            "--arg=--tensor-parallel-size 2",
        )  # fmt: skip
        assert code == 0 and out["state"] == "engaged", command


def test_require_launch_needs_a_real_launch_line(tmp_path: Path) -> None:
    # the check of a case with no settings, such as the untuned NIM
    code, out = check(tmp_path, pod([], []), LAUNCH + "\n" + SCALE_OUT, "--require-launch")
    assert code == 0 and out["state"] == "engaged"
    code, out = check(tmp_path, pod([], []), "boot\n" + SCALE_OUT, "--require-launch")
    assert code == 3 and out["state"] == "unconfirmed"
    assert "no_launch_line" in {e["code"] for e in out["errors"]}


def test_wrong_value_is_mismatch(tmp_path: Path) -> None:
    code, out = check(
        tmp_path,
        pod(["--tensor-parallel-size", "2"], []),
        LAUNCH,
        "--arg",
        "--tensor-parallel-size=4",
    )
    assert code == 3 and out["state"] == "mismatch"
    assert out["errors"][0]["code"] == "setting_mismatch"


def test_json_value_is_compared_by_content(tmp_path: Path) -> None:
    want = '{"method": "ngram", "num_speculative_tokens": 3}'
    for shown in (want, "'" + want + "'", "{'method': 'ngram', 'num_speculative_tokens': 3}"):
        line = json.dumps(
            {"message": f"Launching vLLM: vllm serve /m --speculative-config {shown}"}
        )
        code, out = check(
            tmp_path,
            pod(["--speculative-config", want], []),
            line,
            f"--arg=--speculative-config {want}",
        )
        assert code == 0 and out["state"] == "engaged", shown
    line = json.dumps(
        {
            "message": "Launching vLLM: vllm serve /m --speculative-config "
            '{"method": "ngram", "num_speculative_tokens": 5}'
        }
    )
    code, out = check(
        tmp_path,
        pod(["--speculative-config", want], []),
        line,
        f"--arg=--speculative-config {want}",
    )
    assert code == 3 and out["state"] == "mismatch"


def test_the_last_copy_of_a_repeated_flag_wins(tmp_path: Path) -> None:
    # the NIM appended its own value after the user's: the engine runs the last one
    line = json.dumps(
        {"message": "Launching vLLM: vllm serve /m --max-num-seqs 64 --max-num-seqs 256"}
    )
    code, out = check(tmp_path, pod(["--max-num-seqs", "64"], []), line, "--arg=--max-num-seqs 64")
    assert code == 3 and out["state"] == "mismatch"


def test_mapped_env_is_checked_in_launch_line(tmp_path: Path) -> None:
    env: list[dict[str, Any]] = [
        {"name": "NIM_KVCACHE_PERCENT", "value": "0.92"},
        {"name": "HF_TOKEN", "valueFrom": {"secretKeyRef": {}}},
    ]
    code, out = check(tmp_path, pod([], env), LAUNCH, "--env", "NIM_KVCACHE_PERCENT=0.92")
    assert code == 0, out
    assert out["facts"]["checks"][0]["engine"] == "--gpu-memory-utilization 0.92"


def test_clamped_env_is_mismatch(tmp_path: Path) -> None:
    env = [{"name": "NIM_KVCACHE_PERCENT", "value": "0.95"}]
    code, out = check(tmp_path, pod([], env), LAUNCH, "--env", "NIM_KVCACHE_PERCENT=0.95")
    assert code == 3 and out["state"] == "mismatch"


def test_unmapped_env_needs_engine_evidence(tmp_path: Path) -> None:
    env = [{"name": "NIM_ENABLE_KV_CACHE_REUSE", "value": "1"}]
    code, out = check(tmp_path, pod([], env), LAUNCH, "--env", "NIM_ENABLE_KV_CACHE_REUSE=1")
    assert code == 3 and out["state"] == "unconfirmed" and out["next"] == "find_engine_evidence"
    code, out = check(
        tmp_path,
        pod([], env),
        LAUNCH + "\nprefix caching enabled\n",
        "--env",
        "NIM_ENABLE_KV_CACHE_REUSE=1",
        "--log-pattern",
        "prefix caching enabled",
    )
    assert code == 0, out


def test_no_launch_line_is_unconfirmed(tmp_path: Path) -> None:
    code, out = check(
        tmp_path,
        pod(["--tensor-parallel-size", "2"], []),
        "still downloading\n",
        "--arg",
        "--tensor-parallel-size 2",
    )
    assert code == 3 and "no_launch_line" in {e["code"] for e in out["errors"]}


def test_secret_values_are_redacted(tmp_path: Path) -> None:
    fake = "x" + uuid.uuid4().hex  # built at run time: no credential-shaped literal in the repo
    env = [{"name": "NGC_API_KEY", "value": fake}]
    # the value also shows in an unlabeled engine message that reaches engine_fallbacks, a field
    # no name-based mask covers: only registering the value with the scrubber hides it there
    log = LAUNCH + f"\nkey {fake} not supported, falling back\n"
    code, out = check(tmp_path, pod([], env), log, "--env", f"NGC_API_KEY={fake}")
    assert code == 3 and out["facts"]["engine_fallbacks"]  # the message is reported
    assert fake not in json.dumps(out)


def test_a_credential_argument_value_is_redacted(tmp_path: Path) -> None:
    fake = "x" + uuid.uuid4().hex
    line = json.dumps({"message": f"Launching vLLM: vllm serve /m --api-key {fake}"})
    code, out = check(
        tmp_path, pod(["--api-key", fake], []), line + f"\nusing key {fake}\n",
        f"--arg=--api-key {fake}",
    )  # fmt: skip
    assert code == 0 and out["state"] == "engaged"
    assert fake not in json.dumps(out)
    assert out["facts"]["checks"][0]["delivered"] == "<redacted>"


def test_an_engine_default_is_not_proof_of_delivery(tmp_path: Path) -> None:
    # the engine shows --tensor-parallel-size 2, but the candidate never passed it
    code, out = check(tmp_path, pod([], []), LAUNCH, "--arg=--tensor-parallel-size 2")
    assert code == 3 and out["state"] == "mismatch"
    assert out["facts"]["checks"][0]["delivered"] is None


def test_kubernetes_env_references_in_args_are_expanded(tmp_path: Path) -> None:
    # Kubernetes delivers $(MAX_MODEL_LEN) as 4096: that is what the container received
    env = [{"name": "MAX_MODEL_LEN", "value": "4096"}]
    spec = pod(["--max-model-len", "$(MAX_MODEL_LEN)"], env)
    line = json.dumps({"message": "Launching vLLM: vllm serve /m --max-model-len 4096"})
    code, out = check(tmp_path, spec, line, "--arg=--max-model-len 4096")
    assert code == 0 and out["state"] == "engaged", out
    # a reference with no readable value (valueFrom) is never confirmed: Kubernetes may deliver
    # 4096, so it is unconfirmed, not a mismatch; an escaped one is literal text, a mismatch
    spec = pod(["--max-model-len", "$(FROM_SECRET)"], [{"name": "FROM_SECRET", "valueFrom": {}}])
    code, out = check(tmp_path, spec, line, "--arg=--max-model-len 4096")
    assert code == 3 and out["state"] == "unconfirmed", out
    spec = pod(["--max-model-len", "$$(MAX_MODEL_LEN)"], env)  # the literal text $(MAX_MODEL_LEN)
    code, out = check(tmp_path, spec, line, "--arg=--max-model-len 4096")
    assert code == 3 and out["state"] == "mismatch"
    # Kubernetes reads a reference up to ")", so names like -LIMIT expand too
    dashed = [{"name": "-LIMIT", "value": "4096"}]
    code, out = check(tmp_path, pod(["--max-model-len", "$(-LIMIT)"], dashed), line,
                      "--arg=--max-model-len 4096")  # fmt: skip
    assert code == 0 and out["state"] == "engaged", out
    # an env value can use an earlier one: Kubernetes resolves them in order
    chained = [{"name": "BASE", "value": "4096"}, {"name": "LIMIT", "value": "$(BASE)"}]
    code, out = check(tmp_path, pod(["--max-model-len", "$(LIMIT)"], chained), line,
                      "--arg=--max-model-len 4096")  # fmt: skip
    assert code == 0 and out["state"] == "engaged", out
    # a value that is not a string is a malformed pod, not a traceback
    code, out = check(tmp_path, pod(["--max-model-len", "$(N)"], [{"name": "N", "value": 4096}]),
                      line, "--arg=--max-model-len 4096")  # fmt: skip
    assert code == 2 and out["errors"][0]["code"] == "bad_spec"


def test_the_default_container_annotation_picks_the_spec_container(tmp_path: Path) -> None:
    # `kubectl logs` without -c reads the annotated container; the spec must come from it too
    spec = pod(["--max-num-seqs", "64"], [])
    spec["spec"]["containers"][0]["name"] = "nim"
    side = {"name": "sidecar", "image": "busybox", "args": ["--max-num-seqs", "256"], "env": []}
    spec["spec"]["containers"].insert(0, side)
    spec["metadata"]["annotations"] = {"kubectl.kubernetes.io/default-container": "nim"}
    line = json.dumps({"message": "Launching vLLM: vllm serve /m --max-num-seqs 64"})
    code, out = check(tmp_path, spec, line, "--arg=--max-num-seqs 64")
    assert code == 0 and out["facts"]["identity"]["image"].startswith("nvcr.io/nim")


FAKE_KUBECTL = """#!/usr/bin/env python3
import json, os, sys
args = sys.argv[1:]
with open(os.environ["FAKE_KUBECTL_CALLS"], "a") as fh:
    fh.write(json.dumps(args) + "\\n")
if "get" in args:
    with open(os.environ["FAKE_KUBECTL_CALLS"]) as fh:
        seen = sum(1 for line in fh if '"get"' in line)
    # with FAKE_KUBECTL_POD_AFTER, the pod changes between the first get and the next
    after = os.environ.get("FAKE_KUBECTL_POD_AFTER")
    print(after if after and seen > 1 else os.environ["FAKE_KUBECTL_POD"])
elif "logs" in args:
    print("Launching vLLM: vllm serve /m --max-num-seqs 64")
"""


def test_a_stale_default_container_annotation_falls_back_to_the_first(tmp_path: Path) -> None:
    # kubectl logs warns and reads the first container when the annotation names a missing one;
    # the spec and the log must both come from that first container
    spec = pod(["--max-num-seqs", "64"], [])
    spec["spec"]["containers"][0]["name"] = "nim"
    side = {"name": "sidecar", "image": "busybox", "args": ["--max-num-seqs", "256"], "env": []}
    spec["spec"]["containers"].append(side)
    spec["metadata"]["annotations"] = {"kubectl.kubernetes.io/default-container": "old-nim"}
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "kubectl"
    fake.write_text(FAKE_KUBECTL)
    fake.chmod(0o755)
    calls = tmp_path / "calls.jsonl"
    env = base_env({"PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
           "FAKE_KUBECTL_CALLS": str(calls), "FAKE_KUBECTL_POD": json.dumps(spec)})  # fmt: skip
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "k8s", "p1", "-n", "ns", "--context", "c",
         "--arg=--max-num-seqs 64"],
        capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60, check=False,
        env=env,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stdout
    logs = [c for c in map(json.loads, calls.read_text().splitlines()) if "logs" in c]
    assert len(logs) == 1 and logs[0][-2:] == ["-c", "nim"]


@pytest.mark.parametrize(
    "meta, body",
    [
        ({"annotations": ["x"]}, None),
        (["x"], None),
        ({}, ["x"]),
        ({"annotations": []}, None),  # empty lists are malformed too
        ([], None),
        ({}, []),
        ({}, {"containers": 1}),  # containers must be a list of objects
        ({}, {"containers": ["nim"]}),
    ],
)
def test_a_malformed_pod_object_is_bad_spec(tmp_path: Path, meta: Any, body: Any) -> None:
    spec = pod([], [])
    spec["metadata"] = meta
    if body is not None:
        spec["spec"] = body
    code, out = check(tmp_path, spec, LAUNCH, "--require-launch")
    assert code == 2 and out["errors"][0]["code"] == "bad_spec"


def test_unbalanced_quotes_in_an_arg_are_bad_arg(tmp_path: Path) -> None:
    code, out = check(tmp_path, pod([], []), LAUNCH, "--arg", "--served-model-name 'qwen")
    assert code == 2 and out["errors"][0]["code"] == "bad_arg"


def test_unreadable_saved_evidence_is_bad_input(tmp_path: Path) -> None:
    base = [sys.executable, str(SCRIPT), "k8s", "p1", "--require-launch"]
    log_file = tmp_path / "startup.log"
    log_file.write_text(LAUNCH)
    bad_spec = tmp_path / "pod.json"
    bad_spec.write_text("{not json")
    for extra, want in (
        (["--spec-file", str(tmp_path / "missing.json"), "--log-file", str(log_file)],
         "unreadable_input"),
        (["--spec-file", str(bad_spec), "--log-file", str(log_file)], "bad_spec"),
        (["--spec-file", str(bad_spec), "--log-file", str(tmp_path / "missing.log")],
         "bad_spec"),
    ):  # fmt: skip
        proc = subprocess.run(
            base + extra, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60,
            check=False, env=base_env(),
        )  # fmt: skip
        out = json.loads(proc.stdout)
        assert proc.returncode == 2 and out["errors"][0]["code"] == want, extra
    spec_file = tmp_path / "good.json"
    spec_file.write_text(json.dumps(pod([], [])))
    proc = subprocess.run(
        base + ["--spec-file", str(spec_file), "--log-file", str(tmp_path / "missing.log")],
        capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60, check=False,
        env=base_env(),
    )  # fmt: skip
    assert proc.returncode == 2
    assert json.loads(proc.stdout)["errors"][0]["code"] == "unreadable_input"


FAKE_DOCKER = """#!/usr/bin/env python3
import json, os, sys
args = sys.argv[1:]
with open(os.environ["FAKE_DOCKER_CALLS"], "a") as fh:
    fh.write(json.dumps(args) + "\\n")
if args[0] == "inspect":  # generic: an image of the same name answers it
    print(json.dumps([{"Id": "sha256:img", "RepoTags": [args[1]], "Args": None,
        "Config": {"Image": "", "Env": ["NIM_X=from-image"], "Cmd": ["--max-num-seqs", "64"]}}]))
elif args[:2] == ["container", "inspect"]:
    if os.environ.get("FAKE_DOCKER_NO_CONTAINER"):
        sys.stderr.write("Error: No such container: " + args[2] + "\\n"); sys.exit(1)
    with open(os.environ["FAKE_DOCKER_CALLS"]) as fh:
        seen = sum(1 for line in fh if line.startswith('["container", "inspect"'))
    # with FAKE_DOCKER_RESTART, the container restarts between the first inspect and the next
    start = "2026-11-01T12:05:00Z" if os.environ.get("FAKE_DOCKER_RESTART") and seen > 1 else (
        "2026-11-01T12:00:00.5Z")
    # with FAKE_DOCKER_RENAMED, the name resolves to another container after the first inspect
    cid = "beef" if os.environ.get("FAKE_DOCKER_RENAMED") and seen > 1 and args[2] != "c0ffee" else (
        "c0ffee")
    state = {"Running": True} if os.environ.get("FAKE_DOCKER_NO_START") else {
        "Running": True, "StartedAt": start}
    print(json.dumps([{"Id": cid, "Image": "sha256:img", "Args": ["--max-num-seqs", "64"],
        "State": state, "Config": {"Image": "nim:2", "Env": []}}]))
elif args[0] == "logs":
    old = "Launching vLLM: vllm serve /m --max-num-seqs 64"
    new = "Launching vLLM: vllm serve /m --max-num-seqs 256"
    # before the restart the engine ran 64; since the current start, only the new launch
    print(new if "--since" in args else old + "\\n" + new.replace("256", "64"))
"""


def test_live_docker_logs_come_from_the_inspected_container_since_its_start(
    tmp_path: Path,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "docker"
    fake.write_text(FAKE_DOCKER)
    fake.chmod(0o755)
    calls = tmp_path / "calls.jsonl"
    env = base_env({"PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
           "FAKE_DOCKER_CALLS": str(calls)})  # fmt: skip
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "docker", "nimtune-x", "--arg=--max-num-seqs 64"],
        capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60, check=False,
        env=env,
    )  # fmt: skip
    out = json.loads(proc.stdout)
    logs = [c for c in map(json.loads, calls.read_text().splitlines()) if c[0] == "logs"]
    assert logs == [["logs", "--since", "2026-11-01T12:00:00.5Z", "c0ffee"]]
    # the launch from the current start runs 256: the earlier 64 never counts as engaged
    assert proc.returncode == 3 and out["state"] == "mismatch"
    # a restart between the inspect and the log read: no evidence from either start counts
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "docker", "nimtune-x", "--arg=--max-num-seqs 64"],
        capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60, check=False,
        env={**env, "FAKE_DOCKER_RESTART": "1", "FAKE_DOCKER_CALLS": str(tmp_path / "c2.jsonl")},
    )  # fmt: skip
    out = json.loads(proc.stdout)
    assert proc.returncode == 3 and out["errors"][0]["code"] == "target_restarted"


def test_a_container_renamed_during_the_log_read_is_not_the_names_evidence(
    tmp_path: Path,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "docker"
    fake.write_text(FAKE_DOCKER)
    fake.chmod(0o755)

    def go(**extra: str) -> tuple[int, dict[str, Any], list[list[str]]]:
        calls = tmp_path / f"calls-{uuid.uuid4().hex}.jsonl"
        env = base_env({"PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
                        "FAKE_DOCKER_CALLS": str(calls), **extra})  # fmt: skip
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "docker", "nimtune-x", "--arg=--max-num-seqs 64"],
            capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60, check=False,
            env=env,
        )  # fmt: skip
        seen = [json.loads(line) for line in calls.read_text().splitlines()]
        return proc.returncode, json.loads(proc.stdout), seen

    # unchanged: the name still resolves to the read container, so its evidence is judged
    # (the container was given 64, the engine now runs 256: a mismatch, not a refusal)
    code, out, seen = go()
    assert code == 3 and out["state"] == "mismatch", out
    # renamed away (same id and start by id) while another container took the name
    code, out, _ = go(FAKE_DOCKER_RENAMED="1")
    assert code == 3 and out["errors"][0]["code"] == "target_restarted", out
    assert seen[-1] == ["container", "inspect", "nimtune-x"]  # the name is resolved last
    # no start time to bind the log to: refused before any log read
    code, out, seen = go(FAKE_DOCKER_NO_START="1")
    assert code == 3 and out["errors"][0]["code"] == "no_target", out
    assert not [c for c in seen if c[0] == "logs"]


def test_a_same_named_image_is_never_read_as_the_container(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "docker"
    fake.write_text(FAKE_DOCKER)
    fake.chmod(0o755)
    calls = tmp_path / "calls.jsonl"
    env = base_env({"PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
           "FAKE_DOCKER_CALLS": str(calls), "FAKE_DOCKER_NO_CONTAINER": "1"})  # fmt: skip
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "docker", "nimtune-x", "--arg=--max-num-seqs 64"],
        capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60, check=False,
        env=env,
    )  # fmt: skip
    out = json.loads(proc.stdout)  # the JSON contract, also on this failure
    assert proc.returncode == 4 and out["ok"] is False
    assert out["errors"][0]["code"] == "tool_failed"
    seen = [json.loads(line) for line in calls.read_text().splitlines()]
    assert seen == [["container", "inspect", "nimtune-x"]]  # no generic inspect, no log read


def test_k8s_without_context_is_bad_input() -> None:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "k8s", "p1", "--arg", "--tensor-parallel-size 2"],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=60,
        check=False,
        env=base_env(),
    )
    assert proc.returncode == 2
    assert json.loads(proc.stdout)["errors"][0]["code"] == "unpinned_kubectl"


def test_usage_error_is_contract_json() -> None:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "podman", "p1"],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=60,
        check=False,
        env=base_env(),
    )
    assert proc.returncode == 2
    assert json.loads(proc.stdout)["errors"][0]["code"] == "bad_usage"


def test_only_the_selected_container_is_read(tmp_path: Path) -> None:
    spec = pod(["--max-num-seqs", "64"], [])
    side = {"name": "sidecar", "image": "busybox", "args": ["--max-num-seqs", "256"], "env": []}
    spec["spec"]["containers"][0]["name"] = "nim"
    spec["spec"]["containers"].insert(0, side)
    line = json.dumps({"message": "Launching vLLM: vllm serve /m --max-num-seqs 64"})
    code, out = check(tmp_path, spec, line, "--container", "nim", "--arg=--max-num-seqs 64")
    assert code == 0 and out["facts"]["identity"]["image"].startswith("nvcr.io/nim")


def test_the_last_launch_wins(tmp_path: Path) -> None:
    first = json.dumps({"message": "Launching vLLM: vllm serve /m --max-num-seqs 64"})
    later = json.dumps({"message": "Launching vLLM: vllm serve /m --max-num-seqs 256"})
    code, out = check(
        tmp_path, pod(["--max-num-seqs", "64"], []), first + "\n" + later, "--arg=--max-num-seqs 64"
    )
    assert code == 3 and out["state"] == "mismatch"  # the engine now runs the later value


def test_engine_fallbacks_are_reported(tmp_path: Path) -> None:
    line = json.dumps({"message": "Launching vLLM: vllm serve /m --max-num-seqs 64"})
    warn = (
        "(APIServer pid=1) WARNING 10-26 19:22:30 [vllm.py:716] Model Runner V2 does not yet "
        "support speculative method 'ngram'; using the V1 model runner instead."
    )
    code, out = check(
        tmp_path, pod(["--max-num-seqs", "64"], []), line + "\n" + warn, "--arg=--max-num-seqs 64"
    )
    assert code == 0
    assert out["facts"]["engine_fallbacks"] == [warn.split("] ", 1)[1]]


def test_mask_uses_the_shared_secret_name_pattern() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("check_nim_engagement", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert not hasattr(mod, "SECRET_HINTS")  # one definition of "secret name", in targets.py
    for key in ("HF_TOKEN", "NGC_API_KEY", "DB_PASSWORD", "NIM_API_AUTH", "MY_PASSWD"):
        assert mod.mask(key, "value") == "<redacted>", key
    for key in ("NIM_MAX_NUM_BATCHED_TOKENS", "NIM_TOKENIZER", "NIM_AUTHOR"):
        assert mod.mask(key, "value") == "value", key
    assert mod.mask("HF_TOKEN", None) is None


def test_null_pod_fields_mean_none(tmp_path: Path) -> None:
    # an acceptance check, not a regression test: the new type checks must still read a null
    # annotations field as "none" (the rejections are covered by the malformed-pod cases above)
    spec = pod(["--max-num-seqs", "64"], [])
    spec["metadata"] = {"annotations": None}
    line = json.dumps({"message": "Launching vLLM: vllm serve /m --max-num-seqs 64"})
    code, out = check(tmp_path, spec, line, "--arg=--max-num-seqs 64")
    assert code == 0, out


def test_null_pod_metadata_is_read_as_none(tmp_path: Path) -> None:
    spec = pod(["--max-num-seqs", "64"], [])
    spec["metadata"] = None
    line = json.dumps({"message": "Launching vLLM: vllm serve /m --max-num-seqs 64"})
    code, out = check(tmp_path, spec, line, "--arg=--max-num-seqs 64")
    assert code == 0 and out["facts"]["identity"]["pod_uid"] is None, out


def test_non_list_containers_with_a_named_container_is_bad_spec(tmp_path: Path) -> None:
    spec = pod([], [])
    spec["spec"] = {"containers": 1}
    code, out = check(tmp_path, spec, LAUNCH, "--require-launch", "--container", "nim")
    assert code == 2 and out["errors"][0]["code"] == "bad_spec"


@pytest.mark.parametrize("name", [17, "", None, ["nim"]])
def test_a_container_name_that_is_not_a_nonempty_string_is_bad_spec(
    tmp_path: Path, name: Any
) -> None:
    # a saved pod spec and a live log: the default container's name would go to `kubectl logs
    # -c`, so a malformed one is refused before any kubectl call (never a traceback)
    spec = pod(["--max-num-seqs", "64"], [])
    spec["spec"]["containers"][0]["name"] = name
    spec_file = tmp_path / "pod.json"
    spec_file.write_text(json.dumps(spec))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "kubectl"
    fake.write_text(FAKE_KUBECTL.replace("/usr/bin/env python3", sys.executable))
    fake.chmod(0o755)
    calls = tmp_path / "calls.jsonl"
    env = base_env({"PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
           "FAKE_KUBECTL_CALLS": str(calls), "FAKE_KUBECTL_POD": json.dumps(spec)})  # fmt: skip
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "k8s", "p1", "-n", "ns", "--context", "c",
         "--spec-file", str(spec_file), "--arg=--max-num-seqs 64"],
        capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60, check=False,
        env=env,
    )  # fmt: skip
    out = json.loads(proc.stdout)
    assert proc.returncode == 2 and out["errors"][0]["code"] == "bad_spec", proc.stdout
    assert not calls.exists()  # refused before kubectl logs


def test_a_valid_container_name_still_reads_the_live_log_by_it(tmp_path: Path) -> None:
    spec = pod(["--max-num-seqs", "64"], [])
    spec["spec"]["containers"][0]["name"] = "nim"
    spec_file = tmp_path / "pod.json"
    spec_file.write_text(json.dumps(spec))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "kubectl"
    fake.write_text(FAKE_KUBECTL.replace("/usr/bin/env python3", sys.executable))
    fake.chmod(0o755)
    calls = tmp_path / "calls.jsonl"
    env = base_env({"PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
           "FAKE_KUBECTL_CALLS": str(calls), "FAKE_KUBECTL_POD": json.dumps(spec)})  # fmt: skip
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "k8s", "p1", "-n", "ns", "--context", "c",
         "--spec-file", str(spec_file), "--arg=--max-num-seqs 64"],
        capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60, check=False,
        env=env,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stdout
    logs = [c for c in map(json.loads, calls.read_text().splitlines()) if "logs" in c]
    assert len(logs) == 1 and logs[0][-2:] == ["-c", "nim"]


FAKE_SECRET = "v" + uuid.uuid4().hex  # built at run time, never a literal


@pytest.mark.parametrize(
    "arg",
    [
        f'--api-key="{FAKE_SECRET}',  # unclosed quote: shlex fails
        f"--api-key '{FAKE_SECRET}",
        f"--api-key {FAKE_SECRET} extra",  # extra tokens
        f"--api-key={FAKE_SECRET} extra",
        f'--hf-token "{FAKE_SECRET} x" y',
        f'--api-key"{FAKE_SECRET}',  # no separator before the quote
        f"{FAKE_SECRET}",  # a value passed as its own --arg
        f"'{FAKE_SECRET}",
    ],
)
def test_a_malformed_secret_arg_never_prints_its_value(tmp_path: Path, arg: str) -> None:
    spec_file, log_file = tmp_path / "pod.json", tmp_path / "startup.log"
    spec_file.write_text(json.dumps(pod([], [])))
    log_file.write_text(LAUNCH)
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "k8s", "p1", "--spec-file", str(spec_file),
         "--log-file", str(log_file), "--arg", arg],
        capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60, check=False,
        env=base_env(),
    )  # fmt: skip
    out = json.loads(proc.stdout)
    assert proc.returncode == 2 and out["errors"][0]["code"] == "bad_arg", proc.stdout
    assert FAKE_SECRET not in proc.stdout and FAKE_SECRET not in proc.stderr
    if arg.startswith("--"):  # the flag is still named
        assert out["errors"][0]["message"].startswith(("--arg '--api-key'", "--arg '--hf-token'"))


@pytest.mark.parametrize(
    "argv",
    [
        ["--env", f"={FAKE_SECRET}"],  # a value with no name
        ["--env", FAKE_SECRET],  # no "=" at all: the whole item may be the value
        ["--arg", "--api-key", FAKE_SECRET],  # the value as a stray argument
    ],
)
def test_other_malformed_input_never_prints_a_value(tmp_path: Path, argv: list[str]) -> None:
    spec_file, log_file = tmp_path / "pod.json", tmp_path / "startup.log"
    spec_file.write_text(json.dumps(pod([], [])))
    log_file.write_text(LAUNCH)
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "k8s", "p1", "--spec-file", str(spec_file),
         "--log-file", str(log_file), *argv],
        capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60, check=False,
        env=base_env(),
    )  # fmt: skip
    out = json.loads(proc.stdout)
    assert proc.returncode == 2 and not out["ok"], proc.stdout
    assert FAKE_SECRET not in proc.stdout and FAKE_SECRET not in proc.stderr


def test_malformed_metadata_is_bad_spec_with_a_named_container(tmp_path: Path) -> None:
    spec = pod([], [])
    spec["spec"]["containers"][0]["name"] = "nim"
    spec["metadata"] = ["x"]
    code, out = check(tmp_path, spec, LAUNCH, "--require-launch", "--container", "nim")
    assert code == 2 and out["errors"][0]["code"] == "bad_spec"


def test_a_credential_expanded_into_an_argument_is_masked(tmp_path: Path) -> None:
    fake = "x" + uuid.uuid4().hex
    env = [{"name": "NGC_API_KEY", "value": fake}]
    spec = pod(["--served-model-name", "$(NGC_API_KEY)"], env)  # a non-secret flag carries it
    line = json.dumps({"message": f"Launching vLLM: vllm serve /m --served-model-name {fake}"})
    code, out = check(tmp_path, spec, line, f"--arg=--served-model-name {fake}")
    # the expansion must succeed: a sanitized error would also leave the value out
    assert code == 0 and out["state"] == "engaged", out
    assert fake not in json.dumps(out)


def test_a_dependent_env_value_is_compared_as_delivered(tmp_path: Path) -> None:
    env = [{"name": "BASE", "value": "4096"}, {"name": "NIM_MAX_MODEL_LEN", "value": "$(BASE)"}]
    line = json.dumps({"message": "Launching vLLM: vllm serve /m --max-model-len 4096"})
    code, out = check(tmp_path, pod(["--max-model-len", "$(NIM_MAX_MODEL_LEN)"], env), line,
                      "--env", "NIM_MAX_MODEL_LEN=4096", "--arg=--max-model-len 4096")  # fmt: skip
    assert code == 0 and out["state"] == "engaged", out


def test_a_reference_kubernetes_fills_from_an_unreadable_source_is_unconfirmed(
    tmp_path: Path,
) -> None:
    line = json.dumps({"message": "Launching vLLM: vllm serve /m --max-model-len 4096"})
    hidden = {"name": "LIMIT", "valueFrom": {"configMapKeyRef": {"name": "c", "key": "k"}}}
    # through a variable whose own value uses a valueFrom one: delivered, but not readable here
    spec = pod(["--max-model-len", "$(LEN)"], [hidden, {"name": "LEN", "value": "$(LIMIT)"}])
    code, out = check(tmp_path, spec, line, "--arg=--max-model-len 4096")
    assert code == 3 and out["state"] == "unconfirmed", out
    # the joined form, and a variable checked with --env, read the same way
    spec = pod(["--max-model-len=$(LIMIT)"], [hidden, {"name": "NIM_MAX_MODEL_LEN",
                                                       "value": "$(LIMIT)"}])  # fmt: skip
    code, out = check(tmp_path, spec, line, "--arg=--max-model-len 4096",
                      "--env", "NIM_MAX_MODEL_LEN=4096")  # fmt: skip
    assert code == 3 and out["state"] == "unconfirmed", out
    assert {r["status"] for r in out["facts"]["checks"]} == {"unconfirmed"}
    # with envFrom, any name not set literally may come from it
    spec = pod(["--max-model-len", "$(FROM_MAP)"], [])
    spec["spec"]["containers"][0]["envFrom"] = [{"configMapRef": {"name": "c"}}]
    code, out = check(tmp_path, spec, line, "--arg=--max-model-len 4096")
    assert code == 3 and out["state"] == "unconfirmed", out
    # an escaped reference is literal text whatever the variable: still a mismatch
    code, out = check(tmp_path, pod(["--max-model-len", "$$(LIMIT)"], [hidden]), line,
                      "--arg=--max-model-len 4096")  # fmt: skip
    assert code == 3 and out["state"] == "mismatch", out
    # the engine running another value is a mismatch, whatever was delivered
    other = json.dumps({"message": "Launching vLLM: vllm serve /m --max-model-len 2048"})
    code, out = check(tmp_path, pod(["--max-model-len", "$(LIMIT)"], [hidden]), other,
                      "--arg=--max-model-len 4096")  # fmt: skip
    assert code == 3 and out["state"] == "mismatch", out


def live_k8s(tmp_path: Path, spec: dict[str, Any], after: Any) -> tuple[int, Any]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    fake = bin_dir / "kubectl"
    fake.write_text(FAKE_KUBECTL.replace("/usr/bin/env python3", sys.executable))
    fake.chmod(0o755)
    calls = tmp_path / f"calls-{uuid.uuid4().hex}.jsonl"
    extra = {"PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
             "FAKE_KUBECTL_CALLS": str(calls), "FAKE_KUBECTL_POD": json.dumps(spec)}  # fmt: skip
    if after is not None:
        extra["FAKE_KUBECTL_POD_AFTER"] = json.dumps(after)
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "k8s", "p1", "-n", "ns", "--context", "c",
         "--arg=--max-num-seqs 64"],
        capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60, check=False,
        env=base_env(extra),
    )  # fmt: skip
    return proc.returncode, json.loads(proc.stdout)


def test_a_pod_replaced_or_restarted_during_the_log_read_is_not_evidence(
    tmp_path: Path,
) -> None:
    spec = pod(["--max-num-seqs", "64"], [])
    spec["spec"]["containers"][0]["name"] = "nim"
    spec["status"]["containerStatuses"][0].update(name="nim", containerID="containerd://a")
    code, out = live_k8s(tmp_path, spec, None)
    assert code == 0 and out["state"] == "engaged", out
    replaced = json.loads(json.dumps(spec))
    replaced["metadata"]["uid"] = "u2"  # a new pod under the same name
    restarted = json.loads(json.dumps(spec))
    restarted["status"]["containerStatuses"][0].update(containerID="containerd://b",
                                                       restartCount=1)  # fmt: skip
    for after in (replaced, restarted, ["not a pod"]):
        code, out = live_k8s(tmp_path, spec, after)
        assert code == 3 and out["errors"][0]["code"] == "target_restarted", out


def test_saved_evidence_is_read_as_utf8_whatever_the_locale(tmp_path: Path) -> None:
    env = base_env({"LC_ALL": "en_US.ISO8859-1", "LANG": "en_US.ISO8859-1", "PYTHONUTF8": "0"})
    probe = subprocess.run(
        [sys.executable, "-c", "import locale; print(locale.getpreferredencoding(False))"],
        capture_output=True, text=True, env=env, timeout=60, check=False,
    )  # fmt: skip
    if "8859" not in probe.stdout:
        pytest.skip("no ISO-8859-1 locale on this machine")
    spec_file, log_file = tmp_path / "pod.json", tmp_path / "startup.log"
    spec_file.write_text(json.dumps(pod([], [])), encoding="utf-8")
    line = json.dumps({"message": "Launching vLLM: vllm serve /m --served-model-name café"},
                      ensure_ascii=False)  # fmt: skip
    log_file.write_text(line + "\n", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "k8s", "p1", "--spec-file", str(spec_file),
         "--log-file", str(log_file), "--require-launch"],
        capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60, check=False,
        env=env,
    )  # fmt: skip
    out = json.loads(proc.stdout)
    assert proc.returncode == 0 and out["facts"]["launch_line"].endswith("café"), out


def test_a_short_credential_expanded_into_an_argument_is_masked(tmp_path: Path) -> None:
    # under 6 characters: a value seen in passing is not masked at that length, so a secret-named
    # variable must register it as a named credential before $(NAME) carries it into an argument
    fake = "q" + uuid.uuid4().hex[:4]
    env = [{"name": "NGC_API_KEY", "value": fake}]
    spec = pod(["--served-model-name", "$(NGC_API_KEY)"], env)
    line = json.dumps({"message": f"Launching vLLM: vllm serve /m --served-model-name {fake}"})
    code, out = check(tmp_path, spec, line, f"--arg=--served-model-name {fake}")
    # the expansion must succeed: a sanitized error would also leave the value out
    assert code == 0 and out["state"] == "engaged", out
    assert fake not in json.dumps(out)
    # a short value expected with --env is masked too, also in an unlabeled engine message (a
    # pod without the variable: only the --env registration can hide it)
    log = LAUNCH + f"\nkey {fake} not supported, falling back\n"
    code, out = check(tmp_path, pod([], []), log, "--env", f"NGC_API_KEY={fake}")
    assert code == 3 and out["facts"]["engine_fallbacks"]
    assert fake not in json.dumps(out)


def test_a_default_container_annotation_can_name_a_sidecar(tmp_path: Path) -> None:
    # kubectl logs finds the annotated name among init (sidecar) containers too: the spec, the
    # status and the log must all come from that container, not the first regular one
    spec = pod(["--max-num-seqs", "256"], [])
    spec["spec"]["containers"][0]["name"] = "app"
    spec["spec"]["initContainers"] = [
        {"name": "nim", "image": "nvcr.io/nim/x:2", "restartPolicy": "Always",
         "args": ["--max-num-seqs", "64"]},
    ]  # fmt: skip
    spec["metadata"]["annotations"] = {"kubectl.kubernetes.io/default-container": "nim"}
    spec["status"]["containerStatuses"][0].update(name="app", containerID="containerd://a")
    spec["status"]["initContainerStatuses"] = [
        {"name": "nim", "imageID": "sha256:nim", "containerID": "containerd://n", "restartCount": 0}
    ]
    line = json.dumps({"message": "Launching vLLM: vllm serve /m --max-num-seqs 64"})
    code, out = check(tmp_path, spec, line, "--arg=--max-num-seqs 64")
    assert code == 0 and out["state"] == "engaged", out
    assert out["facts"]["identity"]["image_id"] == "sha256:nim"
    code, out = live_k8s(tmp_path, spec, None)
    assert code == 0 and out["state"] == "engaged", out
    # the sidecar restarting during the live read is caught from its own status
    restarted = json.loads(json.dumps(spec))
    restarted["status"]["initContainerStatuses"][0].update(containerID="containerd://m",
                                                           restartCount=1)  # fmt: skip
    code, out = live_k8s(tmp_path, spec, restarted)
    assert code == 3 and out["errors"][0]["code"] == "target_restarted", out


def test_a_name_declared_again_takes_its_last_state(tmp_path: Path) -> None:
    line = json.dumps({"message": "Launching vLLM: vllm serve /m --max-model-len 4096"})
    hidden = {"name": "LIMIT", "valueFrom": {"configMapKeyRef": {"name": "c", "key": "k"}}}
    # a literal, then a source not readable here: the earlier 4096 is no longer known to be it
    spec = pod(["--max-model-len", "$(LIMIT)"], [{"name": "LIMIT", "value": "4096"}, hidden])
    code, out = check(tmp_path, spec, line, "--arg=--max-model-len 4096")
    assert code == 3 and out["state"] == "unconfirmed", out
    # an optional source that may be missing keeps the uncertainty too, never the old literal
    optional = {"name": "LIMIT", "valueFrom": {"configMapKeyRef": {"name": "c", "key": "k",
                                                                   "optional": True}}}  # fmt: skip
    spec = pod(["--max-model-len", "$(LIMIT)"], [{"name": "LIMIT", "value": "4096"}, optional])
    code, out = check(tmp_path, spec, line, "--arg=--max-model-len 4096")
    assert code == 3 and out["state"] == "unconfirmed", out
    # a source, then a literal: the literal is what the container receives
    later = [{"name": "NIM_MAX_MODEL_LEN", "valueFrom": {"secretKeyRef": {}}},
             {"name": "NIM_MAX_MODEL_LEN", "value": "4096"}]  # fmt: skip
    code, out = check(tmp_path, pod([], later), line, "--env", "NIM_MAX_MODEL_LEN=4096")
    assert code == 0 and out["state"] == "engaged", out


def test_a_service_variable_reference_is_unconfirmed(tmp_path: Path) -> None:
    # the kubelet injects service variables that the pod spec does not declare
    line = json.dumps({"message": "Launching vLLM: vllm serve /m --host 10.0.0.1"})
    spec = pod(["--host", "$(KUBERNETES_SERVICE_HOST)"], [])
    code, out = check(tmp_path, spec, line, "--arg=--host 10.0.0.1")
    assert code == 3 and out["state"] == "unconfirmed", out
    spec = pod(["--host", "$(MY_DB_SERVICE_HOST)"], [])
    code, out = check(tmp_path, spec, line, "--arg=--host 10.0.0.1")
    assert code == 3 and out["state"] == "unconfirmed", out
    # without service links only the kubernetes service's variables are injected
    spec["spec"]["enableServiceLinks"] = False
    code, out = check(tmp_path, spec, line, "--arg=--host 10.0.0.1")
    assert code == 3 and out["state"] == "mismatch", out
    # an escaped reference stays literal text, and the engine running another value mismatches
    spec = pod(["--host", "$$(KUBERNETES_SERVICE_HOST)"], [])
    code, out = check(tmp_path, spec, line, "--arg=--host 10.0.0.1")
    assert code == 3 and out["state"] == "mismatch", out
    spec = pod(["--host", "$(KUBERNETES_SERVICE_HOST)"], [])
    code, out = check(tmp_path, spec, line, "--arg=--host 10.9.9.9")
    assert code == 3 and out["state"] == "mismatch", out


@pytest.mark.parametrize("short", ["7", "on", "true"])
def test_a_short_value_of_a_secret_named_variable_is_masked_without_changing_the_verdict(
    tmp_path: Path, short: str
) -> None:
    # however short or plain, a secret-named value is masked wherever it stands as a token;
    # inside a longer word it is left alone, and never the verdict or the contract changes
    env = [{"name": "NIM_TENSOR_PARALLEL_SIZE", "value": "2"},
           {"name": "NIM_FEATURE_KEY", "value": short}]  # fmt: skip
    launch = f"vllm serve /m --tensor-parallel-size 2 --served-model-name m{short} --x {short}"
    line = json.dumps({"message": f"Launching vLLM: {launch}"})
    code, out = check(tmp_path, pod([], env), line, "--env", "NIM_TENSOR_PARALLEL_SIZE=2")
    assert code == 0 and out["state"] == "engaged" and out["ok"] is True, out
    assert out["facts"]["launch_line"].endswith(f"--served-model-name m{short} --x ***"), out
    assert out["facts"]["effective_nim_env"]["NIM_FEATURE_KEY"] == "<redacted>"


def test_a_one_character_secret_named_value_never_garbles_the_evidence(tmp_path: Path) -> None:
    # HF_HUB_DISABLE_IMPLICIT_TOKEN=1 is secret-named (TOKEN): its "1" is masked where it stands
    # alone, never inside 128, an image tag, the pod uid or the image digest
    spec = pod(["--max-num-seqs", "128"], [{"name": "HF_HUB_DISABLE_IMPLICIT_TOKEN", "value": "1"}])
    spec["metadata"]["uid"] = "u1"
    spec["spec"]["containers"][0]["image"] = "nvcr.io/nim/x:1.2"
    spec["status"]["containerStatuses"][0]["imageID"] = "nvcr.io/nim/x@sha256:1111"
    line = ("INFO 01-01 00:00:00 [api_server.py:1] Launching vLLM: vllm serve m "
            "--max-num-seqs 128 --tensor-parallel-size 1")  # fmt: skip
    code, out = check(tmp_path, spec, line, "--arg=--max-num-seqs 128")
    assert code == 0 and out["state"] == "engaged", out
    facts = out["facts"]
    assert facts["launch_line"].endswith("--max-num-seqs 128 --tensor-parallel-size ***"), out
    assert facts["checks"][0]["delivered"] == facts["checks"][0]["expected"] == "128"
    assert facts["checks"][0]["engine"] == "--max-num-seqs 128"
    ident = facts["identity"]
    assert ident["image"] == "nvcr.io/nim/x:1.2" and ident["pod_uid"] == "u1", ident
    assert ident["image_id"] == "nvcr.io/nim/x@sha256:1111", ident


def test_a_short_credential_flag_value_is_masked_in_engine_fallbacks(tmp_path: Path) -> None:
    # under 6 characters, and only in an unlabeled engine message (no flag next to it): the
    # value of a secret-named flag is masked at any length there too
    fake = "q" + uuid.uuid4().hex[:4]
    launch = json.dumps({"message": f"Launching vLLM: vllm serve /m --api-key {fake}"})
    log = launch + f"\nkey {fake} not supported, falling back\n"
    code, out = check(tmp_path, pod(["--api-key", fake], []), log, f"--arg=--api-key {fake}")
    assert code == 0 and out["state"] == "engaged", out
    assert out["facts"]["engine_fallbacks"] == ["key *** not supported, falling back"], out
    assert out["facts"]["checks"][0]["delivered"] == "<redacted>"
    assert fake not in json.dumps(out["facts"]["checks"]) and fake not in json.dumps(out)


def test_a_four_character_value_of_a_secret_named_variable_is_still_masked(
    tmp_path: Path,
) -> None:
    short = "zq" + "7w"  # built at test time: a short opaque credential, not a toggle
    env = [{"name": "NIM_TENSOR_PARALLEL_SIZE", "value": "2"},
           {"name": "NIM_FEATURE_KEY", "value": short}]  # fmt: skip
    line = json.dumps({"message": "Launching vLLM: vllm serve /m --tensor-parallel-size 2 "
                                  f"--served-model-name {short}"})  # fmt: skip
    code, out = check(tmp_path, pod([], env), line, "--env", "NIM_TENSOR_PARALLEL_SIZE=2")
    assert code == 0 and out["state"] == "engaged", out
    assert short not in json.dumps(out)
