# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""targets.py on its own: manifest kinds, client cleanup, the client's name and AIPerf version,
saved container env and kubectl's NotFound. Fakes stand in for kubectl and docker."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
_spec = importlib.util.spec_from_file_location("targets_under_test", SCRIPTS / "targets.py")
assert _spec and _spec.loader
targets = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(targets)


def proc(code: int = 0, out: str = "", err: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], code, out, err)


# ------------------------------------------------------------------ Kubernetes


def k8s(tmp: Path, client: dict[str, Any], base: dict[str, Any] | None = None) -> Any:
    (tmp / "client.json").write_text(json.dumps(client))
    nim = {
        "apiVersion": "apps.nvidia.com/v1alpha1",
        "kind": "NIMService",
        "metadata": {"name": "x"},
    }
    (tmp / "base.json").write_text(json.dumps(base or nim))
    c = {
        "name": "t1",
        "context": "ctx",
        "namespace": "ns",
        "name_prefix": "p",
        "base_manifest": "base.json",
        "client_manifest": "client.json",
        "candidates": [],
    }
    return targets.K8sTarget(c, tmp, lambda m: None)


@pytest.mark.parametrize(
    "client",
    [
        {"apiVersion": "apps/v1", "kind": "Deployment", "metadata": {"name": "aiperf-client"}},
        {"apiVersion": "v1", "kind": "Service", "metadata": {"name": "aiperf-client"}},
        {"kind": "Pod", "metadata": {"name": "aiperf-client"}},  # no apiVersion
        {"apiVersion": "v1", "kind": "Pod", "metadata": {}},
        {"apiVersion": "v1", "kind": "Pod", "metadata": {"name": "c", "namespace": "other"}},
    ],
)
def test_a_client_manifest_that_is_not_one_pod_here_is_never_applied(
    tmp_path: Path, client: dict[str, Any]
) -> None:
    t = k8s(tmp_path, client)
    calls: list[tuple[str, ...]] = []
    t.kube.run = lambda *a, **k: calls.append(a) or proc(1, "", "Error from server (NotFound)")
    with pytest.raises(targets.KubeError):
        t.ensure_client()
    assert calls == []  # refused before any read or apply


@pytest.mark.parametrize("name", ["client.json", "base.json", "client.yaml"])
def test_a_manifest_that_is_not_json_is_an_error_not_a_traceback(
    tmp_path: Path, name: str
) -> None:
    t = k8s(tmp_path, {})
    (tmp_path / name).write_text('{"token": "s3cret",')  # truncated
    t.c["client_manifest" if name.startswith("client") else "base_manifest"] = name
    # a YAML file is converted by kubectl; its output is parsed the same way
    t.kube.run = lambda *a, **k: proc(0, '{"token": "s3cret",')
    with pytest.raises(targets.KubeError, match=f"cannot read .*{name}") as err:
        t.client_manifest() if name.startswith("client") else t.base()
    assert "s3cret" not in str(err.value)  # never the file's text


def test_a_v1_pod_client_manifest_is_applied(tmp_path: Path) -> None:
    pod = {"apiVersion": "v1", "kind": "Pod", "metadata": {"name": "c", "namespace": "ns"}}
    t = k8s(tmp_path, pod)
    applied: list[dict[str, Any]] = []
    uids = iter([None, {"metadata": {"uid": "u1"}}])
    t.kube.json = lambda *a: next(uids)
    t.kube.create = lambda obj: applied.append(obj) or {"metadata": {"uid": "u1"}}
    t.kube.run = lambda *a, **k: proc()
    assert t.ensure_client() is True and applied == [pod]


def test_a_base_manifest_that_is_not_a_nimservice_is_refused_in_the_dry_run(
    tmp_path: Path,
) -> None:
    pod = {"apiVersion": "v1", "kind": "Pod", "metadata": {"name": "c"}}
    t = k8s(tmp_path, pod, base={"apiVersion": "apps/v1", "kind": "Deployment",
                                 "metadata": {"name": "x"}})  # fmt: skip
    t.kube.apply = lambda obj, dry_run=False: pytest.fail("applied")
    cand = {"id": "a", "replicas": 1, "args": [], "env": {}, "gpus": 1}
    with pytest.raises(targets.KubeError, match="NIMService"):
        t.dry_run(cand)


def test_a_failed_client_pod_delete_is_raised(tmp_path: Path, monkeypatch: Any) -> None:
    t = k8s(tmp_path, {"apiVersion": "v1", "kind": "Pod", "metadata": {"name": "c"}})
    t.client = "c"
    seen: list[list[str]] = []
    denied = 'Error from server (Forbidden): pods "c" is forbidden'
    monkeypatch.setattr(
        targets, "run_cmd", lambda argv, *a, **k: seen.append(argv) or proc(1, "", denied)
    )
    with pytest.raises(targets.KubeError, match="Forbidden"):
        t.remove_client()
    assert "--ignore-not-found" in seen[0]  # an already-deleted pod is not an error


def test_the_client_pod_is_deleted_only_with_its_recorded_uid(
    tmp_path: Path, monkeypatch: Any
) -> None:
    t = k8s(tmp_path, {"apiVersion": "v1", "kind": "Pod", "metadata": {"name": "c"}})
    t.client = "c"
    seen: list[tuple[list[str], Any]] = []
    answer = [proc()]
    monkeypatch.setattr(
        targets, "run_cmd", lambda argv, stdin=None, *a, **k: seen.append((argv, stdin)) or answer[0]
    )
    t.remove_client("uid-1")
    argv, body = seen[-1]
    assert argv[-4:] == ["--raw", "/api/v1/namespaces/ns/pods/c", "-f", "-"]
    assert json.loads(body)["preconditions"] == {"uid": "uid-1"}  # checked by the API server
    answer[0] = proc(1, "", 'Error from server (Conflict): Precondition failed: UID in precondition')
    t.remove_client("uid-1")  # a same-named replacement: nothing deleted, not an error
    answer[0] = proc(1, "", 'Error from server (NotFound): pods "c" not found')
    t.remove_client("uid-1")  # already gone
    answer[0] = proc(1, "", 'Error from server (Forbidden): pods "c" is forbidden')
    with pytest.raises(targets.KubeError, match="Forbidden"):
        t.remove_client("uid-1")


def test_only_the_api_servers_notfound_means_gone(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(targets, "RETRY_WAITS", (0, 0))
    kube = targets.Kube("ctx", "ns", lambda m: None)
    err = "error: context was not found for specified context: ctx"
    monkeypatch.setattr(kube, "run", lambda *a, **k: proc(1, "", err))
    with pytest.raises(targets.KubeError, match="kept failing"):
        kube.json("get", "pod", "c")
    gone = 'Error from server (NotFound): pods "c" not found'
    monkeypatch.setattr(kube, "run", lambda *a, **k: proc(1, "", gone))
    assert kube.json("get", "pod", "c") is None


def test_a_nimservice_with_no_pods_still_exists(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(targets, "RETRY_WAITS", (0, 0))
    t = k8s(tmp_path, {"apiVersion": "v1", "kind": "Pod", "metadata": {"name": "c"}})
    seen: list[tuple[str, ...]] = []
    nim = json.dumps({"kind": "NIMService", "metadata": {"name": "old-x"}})
    pods = json.dumps({"items": []})

    def run(*a: str, **k: Any) -> subprocess.CompletedProcess[str]:
        seen.append(a)
        return proc(0, nim if a[1] == "nimservice" else pods)

    monkeypatch.setattr(t.kube, "run", run)
    assert t.replicas("old-x") == [] and t.exists("old-x") is True
    assert ("get", "nimservice", "old-x", "-o", "json") in seen  # the resource, not its pods
    gone = 'Error from server (NotFound): nimservices.apps.nvidia.com "old-x" not found'
    monkeypatch.setattr(t.kube, "run", lambda *a, **k: proc(1, "", gone))
    assert t.exists("old-x") is False
    denied = "Error from server (Forbidden): nimservices is forbidden"
    monkeypatch.setattr(t.kube, "run", lambda *a, **k: proc(1, "", denied))
    with pytest.raises(targets.KubeError):
        t.exists("old-x")  # an unreadable resource never reads as gone


# ------------------------------------------------------------------ Docker


def docker_target(**d: Any) -> Any:
    block = {"image": "i", "gpus": "0", **d}
    c = {"name": "d1", "name_prefix": "p", "docker": block}
    return targets.DockerTarget(c, Path("."), lambda m: None)


def recorder(t: Any, answers: dict[str, subprocess.CompletedProcess[str]]) -> list[list[str]]:
    calls: list[list[str]] = []

    def docker(*args: str, **k: Any) -> subprocess.CompletedProcess[str]:
        calls.append(list(args))
        return answers.get(args[0], proc())

    t.docker = docker
    return calls


def test_the_aiperf_version_is_validated_and_never_shell_source() -> None:
    t = docker_target(aiperf_version="0.13.0; touch /tmp/pwned")
    calls = recorder(t, {})
    with pytest.raises(targets.KubeError, match="aiperf_version"):
        t.ensure_client()
    assert calls == []  # refused before the network or the client is made
    with pytest.raises(targets.KubeError, match="aiperf_version"):
        t.dry_run({"id": "a"})

    t = docker_target(aiperf_version="0.14.0rc1")
    t.inspect = lambda name: None
    calls = recorder(t, {})
    assert t.ensure_client() is True
    run = next(c for c in calls if c[0] == "run")
    program = run[run.index("-c") + 1]
    assert program == targets.CLIENT_PROGRAM and "0.14" not in program
    assert run[-1] == "aiperf==0.14.0rc1"  # one argument to the fixed program

    t = docker_target()  # the default version is kept
    t.inspect = lambda name: None
    calls = recorder(t, {})
    t.ensure_client()
    assert next(c for c in calls if c[0] == "run")[-1] == "aiperf==0.13.0"


def test_a_candidate_named_like_the_client_is_refused_before_any_docker_change() -> None:
    t = docker_target()
    cand = {"id": "aiperf", "gpus": 1, "args": [], "env": {}, "replicas": 1}
    calls = recorder(t, {})
    t.inspect = lambda name: pytest.fail("inspected")
    with pytest.raises(targets.KubeError, match="benchmark client"):
        t.dry_run(cand)
    failure = t.deploy(cand)
    assert failure is not None and failure[0] == "config"
    assert t.remove("p-aiperf") is False  # the client is removed only by remove_client
    assert calls == []


def test_a_failed_client_container_removal_is_raised() -> None:
    t = docker_target()
    t.owner = lambda name: "d1"
    recorder(t, {"rm": proc(1, "", "Error response from daemon: permission denied")})
    with pytest.raises(targets.KubeError, match="permission denied"):
        t.remove_client()
    t = docker_target()
    t.owner = lambda name: "d1"
    recorder(t, {"rm": proc(1, "", "Error: No such container: p-aiperf"), "network": proc(1)})
    t.remove_client()  # already gone: removed


def test_saved_env_keeps_only_names_the_campaign_sets() -> None:
    t = docker_target(env={"NIM_MODEL_PATH": "hf://org/m"})
    box = {
        "Config": {
            "Env": [
                "NIM_MODEL_PATH=hf://org/m",
                "NIM_OPAQUE_FROM_ENV_FILE=opaque-value",  # from env_file: name not declared
                "VLLM_SOMETHING=1",
                "NIM_MAX_MODEL_LEN=4096",
            ]
        }
    }
    t.inspect = lambda name: box
    assert t.spec("c", allow=["NIM_MAX_MODEL_LEN"])[0]["Config"]["Env"] == [
        "NIM_MODEL_PATH=hf://org/m",
        "NIM_MAX_MODEL_LEN=4096",
    ]


def test_the_client_is_removed_by_the_id_the_campaign_created() -> None:
    t = docker_target()
    t.owner = lambda name: "d1"
    calls = recorder(t, {"network": proc(1)})
    t.remove_client("c0ffee")
    assert ["rm", "-f", "c0ffee"] in calls and ["rm", "-f", "p-aiperf"] not in calls


def test_the_saved_log_is_the_inspected_containers_since_its_latest_start() -> None:
    t = docker_target()
    t.inspect = lambda name: {"Id": "c0ffee", "State": {"StartedAt": "2026-11-23T10:00:00Z"}}
    calls = recorder(t, {})
    t.logs("p-a")
    assert calls[-1] == ["logs", "--since=2026-11-23T10:00:00Z", "c0ffee"]
    t.logs("p-a", tail=200)  # a startup tail keeps earlier crash output for classification
    assert calls[-1] == ["logs", "--tail=200", "p-a"]
    t.inspect = lambda name: {"Id": "c0ffee", "State": {"StartedAt": "0001-01-01T00:00:00Z"}}
    t.logs("p-a")
    assert calls[-1] == ["logs", "c0ffee"]


def test_a_restart_during_the_saved_log_read_is_read_again() -> None:
    t = docker_target()
    starts = iter(["T1", "T2", "T2", "T2"])  # restarted during the first read only
    t.inspect = lambda name: {"Id": "c0ffee", "State": {"StartedAt": next(starts)}}
    calls = recorder(t, {})
    t.logs("p-a")
    assert [c for c in calls if c[0] == "logs"] == [
        ["logs", "--since=T1", "c0ffee"],
        ["logs", "--since=T2", "c0ffee"],  # read again from the new start
    ]
    t.inspect = lambda name, n=iter(range(100)): {"Id": "c", "State": {"StartedAt": f"T{next(n)}"}}
    with pytest.raises(targets.KubeError, match="kept restarting"):
        t.logs("p-a")


def test_a_failed_saved_log_read_is_an_error_not_evidence() -> None:
    t = docker_target()
    t.inspect = lambda name: {"Id": "c0ffee", "State": {"StartedAt": "T1"}}
    recorder(t, {"logs": proc(1, "", "context deadline exceeded")})
    with pytest.raises(targets.KubeError, match="could not read the log of p-a"):
        t.logs("p-a")
    # a failed read during a restart is read again from the new start
    starts = iter(["T1", "T2", "T2", "T2"])
    t.inspect = lambda name: {"Id": "c0ffee", "State": {"StartedAt": next(starts)}}
    answers = iter([proc(1, "", "container restarting"), proc(0, "launch line", "")])
    t.docker = lambda *args, **k: next(answers)
    assert t.logs("p-a") == "launch line"


def test_a_stopped_container_still_exists() -> None:
    t = docker_target()
    t.inspect = lambda name: {"Id": "c0ffee", "State": {"Running": False}}
    assert t.exists("old-x") is True
    t.inspect = lambda name: None
    assert t.exists("old-x") is False


def test_a_failed_candidate_delete_is_raised_not_taken_for_gone(
    tmp_path: Path, monkeypatch: Any
) -> None:
    t = k8s(tmp_path, {"apiVersion": "v1", "kind": "Pod", "metadata": {"name": "c"}})
    ours = {"metadata": {"name": "p-a", "labels": {targets.LABEL: "t1"}}}  # no uid: by name
    t.kube.json = lambda *a: ours
    t.kube.pods = lambda name: []  # no pods right now: only the delete's result tells
    answer = [proc(1, "", 'Error from server (Forbidden): nimservices "p-a" is forbidden')]
    monkeypatch.setattr(targets, "run_cmd", lambda argv, *a, **k: answer[0])
    with pytest.raises(targets.KubeError, match="Forbidden"):
        t.remove("p-a")
    answer[0] = proc(1, "", 'Error from server (NotFound): nimservices "p-a" not found')
    assert t.remove("p-a") is True  # already gone

    d = docker_target()
    d.inspect = lambda name: {"Id": "c0ffee", "Config": {"Labels": {targets.LABEL: "d1"}}}
    recorder(d, {"rm": proc(1, "", "Error response from daemon: permission denied")})
    with pytest.raises(targets.KubeError, match="permission denied"):
        d.remove("p-a")


# ------------------------------------------------------------------ review fixes


def test_only_a_container_answers_the_existence_check() -> None:
    t = docker_target()
    volume = json.dumps([{"Name": "p-net", "Driver": "local", "Labels": {}}])
    calls: list[list[str]] = []

    def docker(*args: str, **k: Any) -> subprocess.CompletedProcess[str]:
        calls.append(list(args))
        if args[:2] == ("container", "inspect"):
            return proc(1, "", "Error: No such container: p-net")
        return proc(0, volume)  # a generic `docker inspect` finds the volume of that name

    t.docker = docker
    assert t.exists("p-net") is False and t.owner("p-net") is None and t.replicas("p-net") == []
    assert all(c[:2] == ["container", "inspect"] for c in calls)


def test_client_ownership_is_read_from_the_recorded_container_id() -> None:
    ours = {"Id": "c0ffee", "Config": {"Labels": {targets.LABEL: "d1"}}}
    theirs = {"Id": "beef", "Config": {"Labels": {targets.LABEL: "other"}}}
    t = docker_target()
    # ours was renamed; the client's name now belongs to another campaign's container
    t.inspect = lambda ref: {"c0ffee": ours, "p-aiperf": theirs}.get(ref)
    calls = recorder(t, {"network": proc(1)})
    t.remove_client("c0ffee")
    assert ["rm", "-f", "c0ffee"] in calls
    # and the reverse: the name is ours, the recorded id is someone else's: nothing removed
    t.inspect = lambda ref: {"beef": theirs, "p-aiperf": ours}.get(ref)
    calls = recorder(t, {"network": proc(1)})
    t.remove_client("beef")
    assert not [c for c in calls if c[0] == "rm"]


def test_the_saved_spec_is_the_containers_whose_log_was_read() -> None:
    t = docker_target(env={"NIM_X": "1"})
    old = {"Id": "old", "State": {"StartedAt": "T1"}, "Config": {"Env": ["NIM_X=1"]}}
    new = {"Id": "new", "State": {"StartedAt": "T9"}, "Config": {"Env": ["NIM_X=2"]}}
    boxes = {"p-a": old, "old": old}
    t.inspect = lambda ref: boxes.get(ref)
    recorder(t, {"logs": proc(0, "launch line")})
    assert t.logs("p-a") == "launch line"
    assert t.spec("p-a")[0]["Id"] == "old"  # unchanged: the same container
    t.logs("p-a")
    boxes.clear()
    boxes["p-a"] = new  # replaced under the same name after the log was read
    with pytest.raises(targets.KubeError, match="replaced or restarted"):
        t.spec("p-a")
    boxes.update({"p-a": old, "old": old})
    t.logs("p-a")
    boxes["old"] = {**old, "State": {"StartedAt": "T2"}}  # or restarted since the read
    with pytest.raises(targets.KubeError, match="replaced or restarted"):
        t.spec("p-a")



def test_a_saved_log_of_a_renamed_container_is_not_the_names_spec() -> None:
    t = docker_target(env={"NIM_X": "1"})
    old = {"Id": "old", "State": {"StartedAt": "T1"}, "Config": {"Env": ["NIM_X=1"]}}
    new = {"Id": "new", "State": {"StartedAt": "T9"}, "Config": {"Env": ["NIM_X=2"]}}
    boxes = {"p-a": old, "old": old}
    t.inspect = lambda ref: boxes.get(ref)
    recorder(t, {"logs": proc(0, "launch line")})
    t.logs("p-a")
    # the read container was renamed (still inspectable by its id, same start) and another
    # took its name: its settings are not the name's
    boxes.update({"p-a": new, "new": new})
    with pytest.raises(targets.KubeError, match="replaced or restarted"):
        t.spec("p-a")


def test_a_restart_between_the_spec_reads_is_not_the_logs_container() -> None:
    t = docker_target(env={"NIM_X": "1"})
    old = {"Id": "old", "State": {"StartedAt": "T1"}, "Config": {"Env": ["NIM_X=1"]}}
    boxes = {"p-a": old, "old": old}
    reads: list[str] = []

    def inspect(ref: str) -> Any:
        reads.append(ref)
        return boxes.get(ref)

    t.inspect = inspect
    recorder(t, {"logs": proc(0, "launch line")})
    t.logs("p-a")
    reads.clear()

    def restart_after_the_id_read(ref: str) -> Any:
        reads.append(ref)
        box = boxes.get(ref)
        if ref == "old":  # the container restarts right after its settings were read by id
            boxes["p-a"] = boxes["old"] = {**old, "State": {"StartedAt": "T2"}}
        return box

    t.inspect = restart_after_the_id_read
    # same id under the name, but another start: the settings read before it are stale
    with pytest.raises(targets.KubeError, match="replaced or restarted"):
        t.spec("p-a")
    assert reads == ["old", "p-a"]
    # an inspect with no Id (or none at all) never passes as the same container
    for gone in ({"State": {"StartedAt": "T1"}}, None, ["not", "a", "box"]):
        boxes.update({"p-a": old, "old": old})
        t.inspect = inspect
        t.logs("p-a")
        boxes["p-a"] = gone
        with pytest.raises(targets.KubeError, match="replaced or restarted"):
            t.spec("p-a")
    # unchanged: still passes
    boxes.update({"p-a": old, "old": old})
    t.logs("p-a")
    assert t.spec("p-a")[0]["Id"] == "old"


def test_a_short_named_value_is_masked_only_as_a_whole_token(monkeypatch: Any) -> None:
    # HF_HUB_DISABLE_IMPLICIT_TOKEN=1: a secret name with a one-character value; the value is
    # masked where it stands alone, never inside a number, image tag, uid or path
    monkeypatch.setenv("SHORT_PASS", "1")
    sc = targets.Scrubber()
    sc.add_env(["SHORT_PASS"])
    line = "vllm serve m --max-num-seqs 128 --tensor-parallel-size 1 nim/x:1.2 u1 /m/1b x=1"
    assert sc.text(line) == (
        "vllm serve m --max-num-seqs 128 --tensor-parallel-size *** nim/x:1.2 u1 /m/1b x=***"
    )
    assert sc.text("1") == "***" and sc.text('"1"') == '"***"'
    assert sc.text("key 1. (1) =1, a/1 1:x") == "key ***. (***) =***, a/*** ***:x"
    # only an ASCII letter or digit touching it, or "." joining it to a digit, keeps it
    assert sc.text("sha256:1abc 11 v1.2 1.x a_1 1-b") == "sha256:1abc 11 v1.2 ***.x a_*** ***-b"
    # an escape before it is a boundary; one that only looks like one, and a colour code, are not
    assert sc.text("\\u00311 %411 \x1b[1m1x \\u0031 %41 \\x1 \x1b[1m") == (
        "\\u0031*** %41*** \x1b[1m1x \\u0031 %41 \\x1 \x1b[1m"
    )
    # a value of 6+ characters is still masked inside other text
    sc.add("abcdef", named=True)
    assert sc.text("xabcdefx") == "x***x"


def test_a_short_named_value_in_a_urls_userinfo_is_masked(monkeypatch: Any) -> None:
    # glued to the URL by ":" and "@", it is still the credential
    monkeypatch.setenv("SHORT_PASS", "pw")
    sc = targets.Scrubber()
    sc.add_env(["SHORT_PASS"])
    assert sc.text("http://user:pw@host/x http://pw@h http://pw:x@h") == (
        "http://user:***@host/x http://***@h http://***:x@h"
    )
    # ":" and "/" are boundaries too: never a word joined to it
    assert sc.text("nim/pw:1 a:pw/b") == "nim/***:1 a:***/b"


@pytest.mark.parametrize(
    ("text", "masked"),
    [
        ("curl -u admin:ab1 http://x", "curl -u admin:*** http://x"),
        ("-H x-api-key:ab1", "-H x-api-key:***"),
        ("password:ab1", "password:***"),
        ("secret=ab1:9", "secret=***:9"),
        ("token=ab1/x", "token=***/x"),
        ("token=ab1.", "token=***."),
        ("auth=ab1-", "auth=***-"),
        ("pw ab1_x", "pw ***_x"),
        ('{"msg":"line\\nab1"}', '{"msg":"line\\n***"}'),  # a JSON escape before it
        ('"a\\tab1\\r"', '"a\\t***\\r"'),
        ("user:ab1@host", "user:***@host"),
        ('{"k": "\\u00e9ab1"}', '{"k": "\\u00e9***"}'),  # after any JSON escape
        ('"\\bab1"', '"\\b***"'),
        ('"\\fab1"', '"\\f***"'),
        ("?a=%20ab1", "?a=%20***"),  # after a URL escape
        ("\x1b[1mab1", "\x1b[1m***"),  # after an ANSI colour code
        ("\x1b[1;32mab1\x1b[0m", "\x1b[1;32m***\x1b[0m"),
        ("\x1b(Bab1 \x1bMab1", "\x1b(B*** \x1bM***"),  # charset and one-byte ESC sequences
        ("\x9b1mab1", "\x9b1m***"),  # the 8-bit CSI
        ("key=%2520ab1 k=%252520ab1", "key=%2520*** k=%252520***"),  # URL-encoded 2-3 times
        ("\x1b[38:5:196mab1 \x1b[2 qab1 \x1b[>4;1mab1", "\x1b[38:5:196m*** \x1b[2 q*** \x1b[>4;1m***"),
        ("\\e[0mab1 \\E[0mab1 \\U001b[1mab1", "\\e[0m*** \\E[0m*** \\U001b[1m***"),
        ("repr \\x1b[1mab1 \\033[0mab1", "repr \\x1b[1m*** \\033[0m***"),  # escaped ESC
        ('"\\u001b[1mab1"', '"\\u001b[1m***"'),
        ("ab1%20x ab1\\u00e9 ab1\\n", "***%20x ***\\u00e9 ***\\n"),  # an escape after it
        ("\x1b[1mab1x %20ab12 \\u00e9ab1c", "\x1b[1mab1x %20ab12 \\u00e9ab1c"),
        ("ab1.2 x:ab1.0", "ab1.2 x:ab1.0"),  # "." joins it to a digit: a version
        ("xab1 ab1x 9ab1", "xab1 ab1x 9ab1"),  # a letter or digit touches it
    ],
)
def test_a_short_named_value_is_masked_wherever_no_letter_or_digit_touches_it(
    monkeypatch: Any, text: str, masked: str
) -> None:
    monkeypatch.setenv("SHORT_PASS", "ab1")
    sc = targets.Scrubber()
    sc.add_env(["SHORT_PASS"])
    assert sc.text(text) == masked
    assert "ab1" not in sc.dumps({"a": text}) or masked == text


def test_a_short_named_value_equal_to_a_number_keeps_json_records_valid(monkeypatch: Any) -> None:
    monkeypatch.setenv("SHORT_PASS", "1")
    sc = targets.Scrubber()
    sc.add_env(["SHORT_PASS"])
    record = {"status": {"containerStatuses": [{"restartCount": 1}]}, "env": [{"value": "1"}]}
    assert json.loads(sc.dumps(record)) == {
        "status": {"containerStatuses": [{"restartCount": 1}]},
        "env": [{"value": "***"}],
    }


def test_a_manual_restart_changes_the_replicas_identity() -> None:
    t = docker_target()
    box = {"Id": "c0ffee", "RestartCount": 0, "State": {"Running": True, "StartedAt": "T1"}}
    t.inspect = lambda ref: box
    before = t.replicas("p-a")[0]["metadata"]["uid"]
    # `docker restart`: same id, RestartCount unchanged (only a restart policy counts),
    # running again by the next sample; only the start time shows it
    box["State"] = {"Running": True, "StartedAt": "T2"}
    after = t.replicas("p-a")[0]["metadata"]["uid"]
    assert before != after
    status = t.replicas("p-a")[0]["status"]["containerStatuses"][0]
    assert status["restartCount"] == 0  # the restart count alone would hide it
    box["Id"] = "beef"  # a same-named replacement at the same start is another replica too
    assert t.replicas("p-a")[0]["metadata"]["uid"] != after

@pytest.mark.parametrize(
    "bad",
    ["0.13.0banana", "0.13.0\n", " 0.13.0", "0.13.*", "0.13.0+", "0.13.0+\u017f", "1.0alpha1", "v"],
)
def test_an_aiperf_version_pip_would_refuse_is_refused_first(bad: str) -> None:
    t = docker_target(aiperf_version=bad)
    calls = recorder(t, {})
    with pytest.raises(targets.KubeError, match="aiperf_version"):
        t.ensure_client()
    assert calls == []  # before the network or the client exists


@pytest.mark.parametrize(
    "good",
    ["0.13.0", "1", "0.14.0rc1", "1.0a1", "0.13.0.post1", "1.0.dev2", "v0.13.0", "0.14.0-rc.1",
     "1.0RC1", "0.13.0-1", "1!0.13.0", "0.13.0+x"],
)  # fmt: skip
def test_pep_440_versions_pip_accepts_are_accepted(good: str) -> None:
    # passed to pip as one argument, as written: pip reads each spelling as that exact version
    assert docker_target(aiperf_version=good).aiperf_package() == f"aiperf=={good}"


def test_a_short_named_credential_never_changes_a_saved_control_value(monkeypatch: Any) -> None:
    fingerprint = "nim-t".encode().hex()  # 10 hex digits, fixed, built at test time
    uid = f"ff{fingerprint[2:8]}ff"
    # overlap a status, and the hex fingerprint and uid
    short, longer = "measured"[1:3], fingerprint[2:8]
    monkeypatch.setenv("SHORT_PASS", short)
    monkeypatch.setenv("LONGER_PASS", longer)
    sc = targets.Scrubber()
    sc.add_env(["SHORT_PASS", "LONGER_PASS"])
    state = {
        "client_uid": uid,
        "candidates": {
            "a": {"status": "measured", "fingerprint": fingerprint, "reason": f"x {short} y"},
            "b": {"status": short, "kind": "config", "cleanup_pending": "p-measured"},
            "c": {"cleanup_target": "kube:measured/ns", "flags_sha256": fingerprint},
            "d": {"cleanup_uid": uid},  # read back: a resume removes only that resource
        },
    }
    out = json.loads(sc.dumps(state, indent=1, sort_keys=True))
    a, b = out["candidates"]["a"], out["candidates"]["b"]
    assert a["status"] == "measured" and a["fingerprint"] == fingerprint
    assert out["client_uid"] == uid and b["cleanup_pending"] == "p-measured"
    c = out["candidates"]["c"]
    assert c == {"cleanup_target": "kube:measured/ns", "flags_sha256": fingerprint}
    assert out["candidates"]["d"] == {"cleanup_uid": uid}
    assert short not in a["reason"]  # free text is still masked
    assert b["status"] == "***"  # a control value that IS a credential is masked whole


def test_a_client_pod_replaced_while_it_started_is_not_owned(tmp_path: Path) -> None:
    pod = {"apiVersion": "v1", "kind": "Pod", "metadata": {"name": "c", "namespace": "ns"}}
    t = k8s(tmp_path, pod)
    uids = iter([None, {"metadata": {"uid": "u2"}}])
    t.kube.json = lambda *a: next(uids)  # absent, then a replacement after the wait
    t.kube.create = lambda obj: {"metadata": {"uid": "u1"}}  # the pod this call created
    t.kube.run = lambda *a, **k: proc()
    assert t.ensure_client() is False and t.created_uid == ""
    uids = iter([None, {"metadata": {"uid": "u1"}}])
    assert t.ensure_client() is True and t.created_uid == "u1"

    d = docker_target()
    d.inspect = lambda name: None
    recorder(d, {"run": proc(0, "c0ffee\n")})
    assert d.ensure_client() is True and d.created_uid == "c0ffee"  # the id docker run printed


def test_the_client_uid_is_the_one_create_returned_never_a_later_lookup(tmp_path: Path) -> None:
    pod = {"apiVersion": "v1", "kind": "Pod", "metadata": {"name": "c", "namespace": "ns"}}
    t = k8s(tmp_path, pod)
    # created as u1, then replaced by u2 before any lookup: a lookup would read u2 twice
    uids = iter([None, {"metadata": {"uid": "u2"}}, {"metadata": {"uid": "u2"}}])
    t.kube.json = lambda *a: next(uids)
    t.kube.create = lambda obj: {"metadata": {"uid": "u1"}}
    t.kube.apply = lambda obj, dry_run=False: None
    t.kube.run = lambda *a, **k: proc()
    assert t.ensure_client() is False and t.created_uid == ""  # u2 is never recorded as ours


def test_a_client_pod_made_by_someone_else_before_the_create_is_not_owned(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setattr(targets.time, "sleep", lambda s: None)
    pod = {"apiVersion": "v1", "kind": "Pod", "metadata": {"name": "c", "namespace": "ns"}}
    t = k8s(tmp_path, pod)
    theirs = {"metadata": {"name": "c", "uid": "u9"}, "status": {"phase": "Running"}}
    gets = [proc(1, "", 'Error from server (NotFound): pods "c" not found')]
    seen: list[tuple[str, ...]] = []

    def kubectl(*a: str, **k: Any) -> subprocess.CompletedProcess[str]:
        seen.append(a)
        if a[0] == "get":
            return gets.pop(0) if gets else proc(0, json.dumps(theirs))
        if a[0] == "create":
            return proc(1, "", 'Error from server (AlreadyExists): pods "c" already exists')
        return proc()

    t.kube.run = kubectl
    assert t.ensure_client() is False and t.created_uid == ""
    assert ("create", "-f", "-", "-o", "json") in seen
    assert not [a for a in seen if a[0] == "apply"]  # apply would take over their pod


def test_kube_create_returns_the_created_object_and_raises_other_errors(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setattr(targets.time, "sleep", lambda s: None)
    kube = targets.Kube("ctx", "ns", lambda m: None)
    made = {"metadata": {"name": "c", "uid": "u1"}}
    kube.run = lambda *a, **k: proc(0, json.dumps(made))
    assert kube.create({"kind": "Pod"}) == made
    gone = proc(1, "", 'Error from server (NotFound): pods "c" not found')

    def answer(created: subprocess.CompletedProcess[str]) -> Any:
        return lambda *a, **k: created if a[0] == "create" else gone

    kube.run = answer(proc(1, "", "Error from server (Forbidden): no"))
    with pytest.raises(targets.KubeError, match="Forbidden"):
        kube.create({"kind": "Pod"})
    kube.run = answer(proc(0, "pod/c created"))
    with pytest.raises(targets.KubeError, match="did not print"):
        kube.create({"kind": "Pod"})


def create_then_read_back(
    created: subprocess.CompletedProcess[str], nonce_matches: bool | None
) -> tuple[Any, list[tuple[str, ...]]]:
    """A Kube whose create answers CREATED, and whose read back finds the pod with this call's
    nonce (True), another nonce (False), or no pod (None)."""
    kube = targets.Kube("ctx", "ns", lambda m: None)
    seen: list[tuple[str, ...]] = []
    sent: dict[str, Any] = {}

    def kubectl(*a: str, **k: Any) -> subprocess.CompletedProcess[str]:
        seen.append(a)
        if a[0] == "create":
            sent.update(json.loads(k["stdin"]))
            return created
        if nonce_matches is None:
            return proc(1, "", 'Error from server (NotFound): pods "c" not found')
        nonce = sent["metadata"]["annotations"][targets.CREATE_NONCE]
        there = {"metadata": {"name": "c", "uid": "u7", "annotations": {
            targets.CREATE_NONCE: nonce if nonce_matches else "another-attempt"}}}  # fmt: skip
        return proc(0, json.dumps(there))

    kube.run = kubectl
    return kube, seen


@pytest.mark.parametrize(
    "created",
    [proc(124, "", "context deadline exceeded"), proc(0, "pod/c created")],
)
def test_a_pod_a_failed_create_still_made_is_known_by_its_nonce(
    created: subprocess.CompletedProcess[str], monkeypatch: Any
) -> None:
    # the API server made the pod, but kubectl timed out or printed no object: without the
    # nonce the pod would be reused as someone else's on the rerun and never removed
    monkeypatch.setattr(targets.time, "sleep", lambda s: None)
    pod = {"apiVersion": "v1", "kind": "Pod", "metadata": {"name": "c"}}
    kube, seen = create_then_read_back(created, True)
    made = kube.create(pod)
    assert made is not None and made["metadata"]["uid"] == "u7"
    assert ("get", "pod", "c", "-o", "json") in seen
    assert pod == {"apiVersion": "v1", "kind": "Pod", "metadata": {"name": "c"}}  # not changed
    # a same-named pod with another nonce, or none there: never this call's, raised as before
    for matches in (False, None):
        kube, seen = create_then_read_back(created, matches)
        with pytest.raises(targets.KubeError):
            kube.create(pod)


def test_a_nimservice_deploy_records_the_uid_its_create_returned(
    tmp_path: Path, monkeypatch: Any
) -> None:
    # the uid comes from the create's answer, never from a later read by name: a same-named
    # NIMService made after it is not the one deployed
    monkeypatch.setattr(targets.time, "sleep", lambda s: None)
    t = k8s(tmp_path, {"apiVersion": "v1", "kind": "Pod", "metadata": {"name": "c"}})
    cand = {"id": "a", "replicas": 1, "args": [], "env": {}, "gpus": 1}
    seen: list[tuple[str, ...]] = []
    sent: dict[str, Any] = {}

    def kubectl(*a: str, **k: Any) -> subprocess.CompletedProcess[str]:
        seen.append(a)
        if a[0] == "create":
            sent.update(json.loads(k["stdin"]))
            return answers["create"]
        if a[0] == "replace":
            sent["applied"] = json.loads(k["stdin"])
            return answers["apply"]
        if "get" in answers:
            return answers["get"]
        if len(seen) < 2 or seen[-2][0] != "create":  # the lookup before the create: absent
            return proc(1, "", 'Error from server (NotFound): nimservices "p-a" not found')
        nonce = sent["metadata"]["annotations"][targets.CREATE_NONCE]
        there = {"metadata": {"name": "p-a", "uid": "u7", "annotations": {
            targets.CREATE_NONCE: nonce}}}  # fmt: skip
        return proc(0, json.dumps(there))  # a read back: carries this create's nonce

    t.kube.run = kubectl
    made = {"metadata": {"name": "p-a", "uid": "u1"}}
    answers = {"create": proc(0, json.dumps(made))}
    assert t.deploy(cand) is None and t.deployed == "u1"
    # the uid comes from the create's answer: no read by name after it (only the lookup before)
    assert seen[-1] == ("create", "-f", "-", "-o", "json") and seen[0][0] == "get"
    # kubectl timed out, but the API server made it: known by this create's nonce
    seen.clear()
    answers = {"create": proc(124, "", "context deadline exceeded")}
    assert t.deploy(cand) is None and t.deployed == "u7"
    # this campaign's one is already there (a resume): applied bound to the version read
    seen.clear()
    ours = {"metadata": {"uid": "u9", "resourceVersion": "5",
                         "labels": {targets.LABEL: t.c["name"]}},
            "spec": {"nodeSelector": {"old": "x"}}, "status": {"x": 1}}  # fmt: skip
    answers = {"create": proc(1, "", "Error from server (AlreadyExists): exists"),
               "get": proc(0, json.dumps(ours)),
               "apply": proc(0, json.dumps({"metadata": {"uid": "u9"}}))}  # fmt: skip
    assert t.deploy(cand) is None and t.deployed == "u9"
    assert ("replace", "-f", "-", "-o", "json") in seen
    # bound to the object read: another uid is immutable, another version a Conflict
    assert sent["applied"]["metadata"]["resourceVersion"] == "5"
    assert sent["applied"]["metadata"]["uid"] == "u9"
    assert not [a for a in seen if a[0] == "create"]  # no create permission needed for it
    # the spec is exactly the approved one: a field it dropped (nodeSelector) is gone
    assert "nodeSelector" not in sent["applied"]["spec"] and "status" not in sent["applied"]
    # an answer naming another uid, or none: never recorded as this campaign's
    for answer in (json.dumps({"metadata": {"uid": "other"}}), "nimservice/p-a configured"):
        answers["apply"] = proc(0, answer)
        with pytest.raises(targets.KubeError):
            t.deploy(cand)
        assert t.deployed is None
    # someone else's NIMService holds the name: refused, never applied over
    seen.clear()
    theirs = {"metadata": {"uid": "u5", "resourceVersion": "1", "labels": {}}}
    answers["get"] = proc(0, json.dumps(theirs))
    with pytest.raises(targets.RefusedError):
        t.deploy(cand)
    assert not [a for a in seen if a[0] in ("apply", "patch", "replace")] and t.deployed is None


def test_the_dry_run_rehearses_the_operation_the_live_run_uses(
    tmp_path: Path, monkeypatch: Any
) -> None:
    # create when the NIMService is absent; the uid- and version-bound replace when this
    # campaign's is already there (another verb, so another permission), never an apply
    monkeypatch.setattr(targets.time, "sleep", lambda s: None)
    t = k8s(tmp_path, {"apiVersion": "v1", "kind": "Pod", "metadata": {"name": "c"}})
    cand = {"id": "a", "replicas": 1, "args": [], "env": {}, "gpus": 1}
    seen: list[tuple[str, ...]] = []
    sent: dict[str, Any] = {}
    there: dict[str, Any] = {}

    def kubectl(*a: str, **k: Any) -> subprocess.CompletedProcess[str]:
        seen.append(a)
        if a[0] == "get":
            return proc(0, json.dumps(there)) if there else proc(1, "", "(NotFound)")
        sent[a[0]] = json.loads(k["stdin"])
        return proc(0, json.dumps({"metadata": {"uid": "u9"}}))

    t.kube.run = kubectl
    t.dry_run(cand)
    assert ("create", "--dry-run=server", "-f", "-") in seen
    assert not [a for a in seen if a[0] in ("apply", "replace")]
    seen.clear()
    there.update({"metadata": {"uid": "u9", "resourceVersion": "5",
                               "labels": {targets.LABEL: t.c["name"]}},
                  "spec": {"nodeSelector": {"old": "x"}}})  # fmt: skip
    t.dry_run(cand)
    assert ("replace", "--dry-run=server", "-f", "-", "-o", "json") in seen
    assert sent["replace"]["metadata"]["resourceVersion"] == "5"
    assert "nodeSelector" not in sent["replace"]["spec"]
    assert not [a for a in seen if a[0] in ("apply", "create")]
    there["metadata"]["labels"] = {}  # someone else's: a dry-run failure, never rehearsed over
    with pytest.raises(targets.KubeError):
        t.dry_run(cand)


def test_a_client_pod_taken_by_a_terminating_one_is_created_after_it_goes(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setattr(targets.time, "sleep", lambda s: None)
    pod = {"apiVersion": "v1", "kind": "Pod", "metadata": {"name": "c", "namespace": "ns"}}
    t = k8s(tmp_path, pod)
    dying = {"metadata": {"uid": "u0", "deletionTimestamp": "2026-11-23T00:00:00Z"},
             "status": {"phase": "Running"}}  # fmt: skip
    # absent, then (after AlreadyExists) a pod still terminating, then gone, then the new one
    gets = iter([None, dying, None, {"metadata": {"uid": "u1"}}])
    t.kube.json = lambda *a: next(gets)
    answers = iter([None, {"metadata": {"uid": "u1"}}])  # AlreadyExists, then made
    creates: list[dict[str, Any]] = []
    t.kube.create = lambda obj: creates.append(obj) or next(answers)
    t.kube.run = lambda *a, **k: proc()
    assert t.ensure_client() is True and t.created_uid == "u1" and len(creates) == 2
    # still taken after the second create, and gone: never reused, never owned
    gets = iter([None, dying, None, None])
    t.kube.create = lambda obj: None
    with pytest.raises(targets.ClientLostError, match="gone"):
        t.ensure_client()
    # taken by a pod that is not terminating: reused as someone else's, one create only
    gets = iter([None, {"metadata": {"uid": "u9"}, "status": {"phase": "Running"}}])
    creates.clear()
    t.kube.create = lambda obj: creates.append(obj)
    assert t.ensure_client() is False and t.created_uid == "" and len(creates) == 1


def test_the_client_manifest_is_checked_by_a_server_side_dry_run(tmp_path: Path) -> None:
    pod = {"apiVersion": "v1", "kind": "Pod", "metadata": {"name": "c", "namespace": "ns"}}
    t = k8s(tmp_path, pod)
    seen: list[tuple[tuple[str, ...], Any]] = []
    there: list[Any] = [None]
    t.kube.json = lambda *a: there[0]
    t.kube.run = lambda *a, **k: seen.append((a, k.get("stdin"))) or proc()
    assert t.dry_run_client() is True
    assert seen == [(("create", "--dry-run=server", "-f", "-", "-o", "json"), json.dumps(pod))]
    # refused by the API server (RBAC, PodSecurity, quota): raised before the plan is approved
    t.kube.run = lambda *a, **k: proc(1, "", 'Error from server (Forbidden): pods "c" is forbidden')
    with pytest.raises(targets.KubeError, match="Forbidden"):
        t.dry_run_client()
    # a pod of that name is there: it would be reused, never created, so nothing to check
    there[0], seen[:] = {"metadata": {"uid": "u9"}}, []
    t.kube.run = lambda *a, **k: seen.append((a, k.get("stdin"))) or proc()
    assert t.dry_run_client() is False and seen == []
    assert docker_target().dry_run_client() is False


def sidecar_pod(note: str | None) -> dict[str, Any]:
    return {
        "metadata": {"annotations": {targets.DEFAULT_CONTAINER: note} if note else {}},
        "spec": {
            "containers": [{"name": "app"}],
            "initContainers": [{"name": "nim", "restartPolicy": "Always"}],
            "ephemeralContainers": [{"name": "debug"}],
        },
        "status": {
            "containerStatuses": [{"name": "app", "imageID": "a"}],
            "initContainerStatuses": [{"name": "nim", "imageID": "n"}],
        },
    }


def test_pod_container_picks_the_container_kubectl_logs_reads() -> None:
    nim = ("initContainerStatuses", {"name": "nim", "restartPolicy": "Always"},
           {"name": "nim", "imageID": "n"})  # fmt: skip
    assert targets.pod_container(sidecar_pod("nim")) == nim  # a sidecar the annotation names
    assert targets.pod_container(sidecar_pod(None), "nim") == nim  # -c NAME, any kind
    app = ("containerStatuses", {"name": "app"}, {"name": "app", "imageID": "a"})
    assert targets.pod_container(sidecar_pod(None)) == app  # no annotation: the first
    assert targets.pod_container(sidecar_pod("old")) == app  # a stale one: the first
    debug = ("ephemeralContainerStatuses", {"name": "debug"}, None)  # no status yet
    assert targets.pod_container(sidecar_pod("debug")) == debug
    assert targets.pod_container(sidecar_pod(None), "missing") is None
    assert targets.pod_container({"spec": {}}) is None
    bad = sidecar_pod(None)
    bad["spec"]["initContainers"] = {"name": "nim"}
    with pytest.raises(ValueError, match="spec.initContainers"):
        targets.pod_container(bad, "nim")
    with pytest.raises(ValueError, match="annotations"):
        targets.pod_container({"metadata": {"annotations": []}, "spec": {}})


@pytest.mark.parametrize("name", [None, "app", "nim"])
def test_a_malformed_later_container_list_is_reported_even_after_a_match(
    name: str | None,
) -> None:
    bad = sidecar_pod(None)
    bad["spec"]["ephemeralContainers"] = [{"name": ""}]  # found first: app or nim
    with pytest.raises(ValueError, match=r"spec.ephemeralContainers\[\].name"):
        targets.pod_container(bad, name)
    bad["spec"]["ephemeralContainers"] = "debug"
    with pytest.raises(ValueError, match="spec.ephemeralContainers"):
        targets.pod_container(bad, name)


def test_a_candidate_is_removed_only_as_the_instance_whose_owner_was_read(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setattr(targets.time, "sleep", lambda s: None)
    t = k8s(tmp_path, {"apiVersion": "v1", "kind": "Pod", "metadata": {"name": "c"}})
    nim = {
        "apiVersion": "apps.nvidia.com/v1alpha1",
        "metadata": {"name": "p-a", "uid": "u1", "labels": {targets.LABEL: "t1"}},
    }
    reads = [nim, None]  # ours, then gone after the delete
    t.kube.json = lambda *a: reads.pop(0) if reads else None
    t.kube.pods = lambda name: []
    seen: list[tuple[tuple[str, ...], Any]] = []
    answer = [proc()]
    t.kube.run = lambda *a, stdin=None, **k: seen.append((a, stdin)) or answer[0]
    assert t.remove("p-a") is True
    args, body = seen[-1]
    assert args == (
        "delete", "--raw", "/apis/apps.nvidia.com/v1alpha1/namespaces/ns/nimservices/p-a",
        "-f", "-",
    )  # fmt: skip
    assert json.loads(body)["preconditions"] == {"uid": "u1"}
    reads[:] = [nim]
    answer[0] = proc(
        1, "", "Error from server (Conflict): Precondition failed: UID in precondition"
    )
    assert t.remove("p-a") is False  # replaced since the read: the replacement is not touched
    reads[:] = [nim]
    answer[0] = proc(1, "", 'Error from server (NotFound): nimservices "p-a" not found')
    assert t.remove("p-a") is True  # the checked one is already gone: removed

    d = docker_target()
    d.inspect = lambda name: {"Id": "c0ffee", "Config": {"Labels": {targets.LABEL: "d1"}}}
    calls = recorder(d, {})
    assert d.remove("p-a") is True
    assert ["rm", "-f", "c0ffee"] in calls and ["rm", "-f", "p-a"] not in calls


def test_a_resumed_removal_is_bound_to_the_identity_recorded_with_its_mark(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setattr(targets.time, "sleep", lambda s: None)
    t = k8s(tmp_path, {"apiVersion": "v1", "kind": "Pod", "metadata": {"name": "c"}})
    # same name and campaign label, another uid: another run's (perhaps another user's)
    nim = {
        "apiVersion": "apps.nvidia.com/v1alpha1",
        "metadata": {"name": "p-a", "uid": "u2", "labels": {targets.LABEL: "t1"}},
    }
    t.kube.json = lambda *a: nim
    t.kube.pods = lambda name: []
    seen: list[tuple[str, ...]] = []
    t.kube.run = lambda *a, stdin=None, **k: seen.append(a) or proc()
    assert t.ident("p-a") == "u2" and t.kept("p-a", "u1") is False
    assert t.remove("p-a", "u1") is False and seen == []  # never deleted
    reads = [nim, None]
    t.kube.json = lambda *a: reads.pop(0) if reads else None
    assert t.remove("p-a", "u2") is True and seen[-1][:2] == ("delete", "--raw")

    d = docker_target()
    boxes = {"p-a": {"Id": "beef", "Config": {"Labels": {targets.LABEL: "d1"}}}}
    d.inspect = lambda ref: boxes.get(ref)
    calls = recorder(d, {})
    assert d.ident("p-a") == "beef"
    assert d.remove("p-a", "c0ffee") is False and calls == []  # the name is another container
    assert d.remove("p-a", "beef") is True and ["rm", "-f", "beef"] in calls
    # the recorded container renamed: still found by its Id, so it is not taken for gone
    boxes.clear()
    boxes["c0ffee"] = {"Id": "c0ffee", "Config": {"Labels": {targets.LABEL: "d1"}}}
    assert d.ident("p-a") is None and d.kept("p-a", "c0ffee") is True
    assert d.kept("p-a", "") is False and d.strays("p-a") is False


def test_a_changed_container_is_recreated_by_its_id_and_a_failed_rm_is_raised() -> None:
    cand = {"id": "a", "gpus": 1, "args": [], "env": {}, "replicas": 1}
    old = {"Id": "c0ffee", "Config": {"Labels": {targets.LABEL: "d1", "nim-tune/fingerprint": "x"}}}
    t = docker_target()
    t.inspect = lambda name: old
    calls = recorder(t, {})
    assert t.deploy(cand) is None
    assert ["rm", "-f", "c0ffee"] in calls and ["rm", "-f", "p-a"] not in calls
    assert calls[-1][0] == "run"
    t = docker_target()
    t.inspect = lambda name: old
    calls = recorder(t, {"rm": proc(1, "", "Error response from daemon: permission denied")})
    with pytest.raises(targets.KubeError, match="permission denied"):
        t.deploy(cand)
    assert not [c for c in calls if c[0] == "run"]  # never started over the one still there
    t = docker_target()
    t.inspect = lambda name: old
    calls = recorder(t, {"rm": proc(1, "", "Error: No such container: c0ffee")})
    assert t.deploy(cand) is None and calls[-1][0] == "run"  # already gone: removed
    # another campaign's container is still refused and never removed
    t = docker_target()
    t.inspect = lambda name: {"Id": "beef", "Config": {"Labels": {targets.LABEL: "other"}}}
    calls = recorder(t, {})
    failure = t.deploy(cand)
    assert failure is not None and failure[0] == "config" and calls == []


def test_a_container_deploy_records_the_id_its_docker_run_made() -> None:
    # the Id `docker run -d` printed, or (when it failed or printed none) the container under
    # the name only if it carries this run's nonce label: never a same-named one made since
    cand = {"id": "a", "gpus": 1, "args": [], "env": {}, "replicas": 1}
    cid = "ab" * 32
    t = docker_target()
    t.inspect = lambda name: None
    calls = recorder(t, {"run": proc(0, f"{cid}\n")})
    assert t.deploy(cand) is None and t.deployed == cid
    run = next(c for c in calls if c[0] == "run")
    nonce = next(a for a in run if a.startswith(targets.CREATE_NONCE + "="))
    assert run[:2] == ["run", "-d"] and run.index(nonce) < run.index("i")  # before the image

    def made_by(labels: dict[str, str]) -> Any:
        return {"Id": "beef", "Config": {"Labels": {targets.LABEL: "d1", **labels}}}

    for out, code in (("Unable to find image\nnot an id", 0), ("", 125)):
        t = docker_target()
        boxes: list[Any] = [None]  # nothing before the run
        t.inspect = lambda name: boxes.pop(0) if boxes else made
        calls = recorder(t, {"run": proc(code, out, "Error response from daemon: x")})
        made = None

        def started(*args: str, **k: Any) -> subprocess.CompletedProcess[str]:
            nonlocal made
            calls.append(list(args))
            tag = next(a for a in args if a.startswith(targets.CREATE_NONCE + "="))
            made = made_by({targets.CREATE_NONCE: tag.split("=", 1)[1]})  # this run's
            return proc(code, out, "Error response from daemon: x")

        t.docker = started
        t.deploy(cand)
        assert t.deployed == "beef"  # made by this run (created, perhaps never started)
        t = docker_target()
        replaced = [None, made_by({targets.CREATE_NONCE: "another-run"})]
        t.inspect = lambda name: replaced.pop(0)
        recorder(t, {"run": proc(code, out, "Error response from daemon: x")})
        t.deploy(cand)
        assert t.deployed == ""  # a same-named container of another run: never recorded
    t = docker_target()
    looks: list[Any] = [None]

    def unreadable(name: str) -> Any:
        if looks:
            return looks.pop(0)
        raise targets.KubeError("docker inspect p-a kept failing")

    t.inspect = unreadable
    recorder(t, {"run": proc(125, "", "Error response from daemon: x")})
    t.deploy(cand)
    assert t.deployed is None  # unknown: never removed by the runner
    # reused (same settings): the container inspected; refused before any run: none
    t = docker_target()
    spec = t.describe(cand)
    t.inspect = lambda name: made_by({"nim-tune/fingerprint": spec["fingerprint"]})
    recorder(t, {})
    assert t.deploy(cand) is None and t.deployed == "beef"
    t = docker_target()
    t.inspect = lambda name: {"Id": "f00d", "Config": {"Labels": {targets.LABEL: "other"}}}
    recorder(t, {})
    assert t.deploy(cand) is not None and t.deployed == ""


def test_a_saved_log_binding_never_outlives_its_container() -> None:
    t = docker_target()
    old = {"Id": "old", "State": {"StartedAt": "T1"}, "Config": {"Labels": {targets.LABEL: "d1"}}}
    new = {"Id": "new", "State": {"StartedAt": "T9"}, "Config": {"Labels": {targets.LABEL: "d1"}}}
    boxes: dict[str, Any] = {"p-a": old, "old": old}
    t.inspect = lambda ref: boxes.get(ref)
    recorder(t, {"logs": proc(0, "launch line")})
    t.logs("p-a")  # read, then the check failed before spec(): the binding is left behind
    assert t.remove("p-a") is True
    boxes.clear()
    boxes.update({"p-a": new, "new": new})  # the next deploy of that name
    assert t.spec("p-a")[0]["Id"] == "new"  # not "replaced": that binding was for the old one
    t.logs("p-a")
    cand = {"id": "a", "gpus": 1, "args": [], "env": {}, "replicas": 1}
    boxes.clear()
    t.deploy(cand)  # (re)deployed: a fresh instance under the name
    boxes.update({"p-a": old, "old": old})
    assert t.spec("p-a")[0]["Id"] == "old"


def test_a_short_named_credential_never_changes_read_back_run_paths_or_settings(
    monkeypatch: Any,
) -> None:
    # short overlaps run paths ("c8-r1") and a flag value ("c8x"); longer is built at test time
    short, longer = "c8", "tuning".encode().hex()
    monkeypatch.setenv("SHORT_PASS", short)
    monkeypatch.setenv("LONGER_PASS", longer)
    sc = targets.Scrubber()
    sc.add_env(["SHORT_PASS", "LONGER_PASS"])
    measured = {
        "gpus": 1,
        "args": ["--max-num-seqs", "c8x", "--tag=c8-r1"],
        "env": {"NIM_X": "c8-c8", "NIM_Y": "1"},
        "docker": {"image": "nim:c8", "env": {"NIM_Z": "abc"}, "env_from_host": ["NGC_API_KEY"]},
    }
    state = {
        "candidates": {
            "a": {
                "status": "measured",
                "failed_runs": ["c8-r1", "sla-search"],
                "overloaded_runs": ["c8-r1"],
                "measured": measured,
                "reason": f"died at {short}",
            },
            "b": {
                "failed_runs": [short],  # a value that IS a credential is masked whole
                "measured": {
                    "args": [f"--api-key={short}", f"x{longer}y", "nvapi-" + "A" * 20],
                    "env": {"NIM_P": longer},
                },
            },
        },
    }
    for text in (sc.dumps(state, indent=1, sort_keys=True), json.dumps(sc.data(state))):
        out = json.loads(text)
        a, b = out["candidates"]["a"], out["candidates"]["b"]
        assert a["failed_runs"] == ["c8-r1", "sla-search"] and a["overloaded_runs"] == ["c8-r1"]
        assert a["measured"] == measured  # unchanged: write_recommended compares it
        assert short not in a["reason"]  # free text is still masked
        assert b["failed_runs"] == ["***"]
        assert b["measured"]["args"] == ["--api-key=***", "x***y", "***"]
        assert b["measured"]["env"] == {"NIM_P": "***"}
        assert longer not in text


def test_a_short_named_credential_never_changes_the_saved_order_or_nodes(monkeypatch: Any) -> None:
    short = "gpu-node-01"[4:6]  # "no": inside a node name and a candidate id
    monkeypatch.setenv("SHORT_PASS", short)
    sc = targets.Scrubber()
    sc.add_env(["SHORT_PASS"])
    state = {"order": ["before", "nolimit"], "nodes": ["gpu-node-01"], "baseline_fallbacks": []}
    assert json.loads(sc.dumps(state)) == state
