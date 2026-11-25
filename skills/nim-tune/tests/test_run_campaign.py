# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for scripts/run_campaign.py; a fake kubectl stands in for the cluster."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import contextlib
import json
import os
import secrets
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from testenv import base_env

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_campaign.py"
_spec = importlib.util.spec_from_file_location("run_campaign", SCRIPT)
assert _spec and _spec.loader
rc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rc)

FAKE_KUBECTL = """#!/usr/bin/env python3
import os
import sys
import time
args = sys.argv[1:]
if "get" in args:
    sys.stderr.write("Error from server (NotFound): not found")
    sys.exit(1)  # nothing exists yet
if "apply" in args and "--dry-run=server" in args:
    sys.stdin.read()
    sys.exit(0)
if "create" in args and "--dry-run=server" in args:
    if "FAKE_REFUSE_CLIENT" in os.environ:
        sys.stderr.write('pods "aiperf-client" is forbidden: violates PodSecurity')
        sys.exit(1)
    sys.stdout.write(sys.stdin.read())  # create -o json prints the object; nothing is stored
    sys.exit(0)
sys.exit(9)  # anything else would change the cluster
"""


@pytest.fixture(autouse=True)
def _unfreeze_inputs() -> Iterator[None]:
    """A test that calls main() freezes input files; never let it serve stale bytes to the next."""
    yield
    rc.targets._INPUTS.clear()
    rc.targets._FROZEN = False


def fake_secret(prefix: str = "") -> str:
    """A credential-shaped value made at test time, so no key-like literal sits in the source."""
    return prefix + secrets.token_hex(12)


def campaign(tmp: Path, **over: Any) -> Path:
    (tmp / "base.json").write_text(
        json.dumps(
            {
                "apiVersion": "apps.nvidia.com/v1alpha1",
                "kind": "NIMService",
                "metadata": {"name": "x", "labels": {"owner": "me"}},
                "spec": {"env": [{"name": "NIM_SERVED_MODEL_NAME", "value": "m"}]},
            }
        )
    )
    c: dict[str, Any] = {
        "name": "t1",
        "context": "ctx",
        "namespace": "ns",
        "gpu_budget": 4,
        "base_manifest": "base.json",
        "client_manifest": "client.json",
        "name_prefix": "nimtune-me",
        "baseline": "before",
        "workload": {"model": "m", "tokenizer": "org/m"},
        "candidates": [
            {"id": "tp2x2", "gpus": 2, "replicas": 2, "args": ["--tensor-parallel-size", "2"]},
            {"id": "before", "gpus": 4},
        ],
    }
    c.update(over)
    if not (tmp / "client.json").exists():  # a test's own client manifest is kept
        (tmp / "client.json").write_text(
            json.dumps({"apiVersion": "v1", "kind": "Pod", "metadata": {"name": "aiperf-client"}})
        )
    path = tmp / "campaign.json"
    path.write_text(json.dumps(c))
    return path


def approve(path: Path) -> list[str]:
    """The user's approval of this exact campaign file (its dry-run plan_id)."""
    return ["--approved", rc.plan_id(path)]


def run(tmp: Path, *args: str, env: dict[str, str] | None = None) -> tuple[int, dict[str, Any]]:
    fake = tmp / "kubectl"
    fake.write_text(FAKE_KUBECTL.replace("/usr/bin/env python3", sys.executable))
    fake.chmod(0o755)
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=60,
        check=False,
        env=base_env({"NIM_TUNE_KUBECTL": str(fake), **(env or {})}),
    )
    out = json.loads(proc.stdout)
    assert set(out) == {"ok", "state", "next", "facts", "errors"}
    return proc.returncode, out


def test_dry_run_plans_baseline_first_and_changes_nothing(tmp_path: Path) -> None:
    code, out = run(tmp_path, str(campaign(tmp_path)), "--out", str(tmp_path / "r"), "--dry-run")
    assert code == 0 and out["state"] == "planned" and out["next"] == "ask_user"
    assert out["facts"]["order"] == ["before", "tp2x2"]
    assert out["facts"]["candidates"]["tp2x2"]["name"] == "nimtune-me-tp2x2"
    assert out["facts"]["aiperf_runs_fixed_loads"] == 2 * 3 * 3
    assert out["facts"]["client_checked"] is True


def test_dry_run_reports_a_client_pod_the_api_server_refuses(tmp_path: Path) -> None:
    path = campaign(tmp_path)
    code, out = run(tmp_path, str(path), "--out", str(tmp_path / "r"), "--dry-run",
                    env={"FAKE_REFUSE_CLIENT": "1"})  # fmt: skip
    assert code == rc.EXIT_EXTERNAL and [e["code"] for e in out["errors"]] == ["dry_run_failed"]
    assert "PodSecurity" in out["errors"][0]["message"]


def test_bad_campaign_is_rejected_before_any_cluster_call(tmp_path: Path) -> None:
    path = campaign(
        tmp_path,
        workload={"model": "m", "tokenizer": "t", "repeats": 1},
        candidates=[{"id": "big", "gpus": 8}],
    )
    code, out = run(tmp_path, str(path), "--out", str(tmp_path / "r"), "--dry-run")
    assert code == 2 and out["next"] == "fix_errors"
    codes = {e["code"] for e in out["errors"]}
    assert {"too_few_repeats", "over_budget", "baseline_missing"} <= codes


def test_render_sets_only_the_candidate_fields() -> None:
    c = {"name": "t1", "namespace": "ns", "name_prefix": "nimtune-me"}
    base: dict[str, Any] = {
        "metadata": {"name": "x", "uid": "u"},
        "spec": {"env": [{"name": "A", "value": "1"}, {"name": "B", "value": "2"}]},
    }
    cand = {"id": "kv", "gpus": 1, "replicas": 4, "args": ["--max-num-seqs", "64"], "env": {"B": 3}}
    m = rc.targets.render(c, base, cand)
    assert m["metadata"]["name"] == "nimtune-me-kv" and "uid" not in m["metadata"]
    assert m["metadata"]["labels"]["nim-tune/campaign"] == "t1"
    assert m["spec"]["replicas"] == 4 and m["spec"]["resources"]["limits"]["nvidia.com/gpu"] == 1
    assert {e["name"]: e["value"] for e in m["spec"]["env"]} == {"A": "1", "B": "3"}
    assert base["spec"]["env"][1]["value"] == "2"  # base untouched
    assert rc.arg_pairs(["--a", "1", "--flag", "--b", "x"]) == ["--a 1", "--flag", "--b x"]


def test_sla_search_finds_the_boundary_both_ways() -> None:
    up = rc.sla_search(lambda n: n <= 37, start=8, hi_cap=512, resolution_pct=10)
    assert up["complete"] and up["max_concurrency"] <= 37 < up["first_failing"]
    assert up["first_failing"] - up["max_concurrency"] <= max(1, up["max_concurrency"] // 10)
    down = rc.sla_search(lambda n: n <= 3, start=8, hi_cap=512, resolution_pct=10)
    assert down["max_concurrency"] == 3 and down["first_failing"] == 4
    none = rc.sla_search(lambda n: False, start=8, hi_cap=512, resolution_pct=10)
    assert none["max_concurrency"] is None
    cap = rc.sla_search(lambda n: True, start=8, hi_cap=100, resolution_pct=10)
    assert cap["max_concurrency"] == 100 and cap["first_failing"] is None


def test_classify_separates_config_from_intermittent() -> None:
    def pod(**cs: Any) -> list[dict[str, Any]]:
        return [{"status": {"containerStatuses": [cs]}}]

    oom = pod(lastState={"terminated": {"reason": "OOMKilled"}})
    assert rc.classify(oom, "") == ("config", "OOMKilled")
    pull = pod(state={"waiting": {"reason": "ImagePullBackOff"}})
    assert rc.classify(pull, "") == ("intermittent", "ImagePullBackOff")
    crash = pod(state={"waiting": {"reason": "CrashLoopBackOff"}}, restartCount=3)
    got = rc.classify(crash, "torch.OutOfMemoryError: CUDA out of memory")
    assert got is not None and got[0] == "config"
    assert rc.classify(pod(restartCount=0), "loading weights") is None


# A stateful stand-in for kubectl: NIMServices become pods, AIPerf prints a console table whose
# TTFT p99 grows with load (30 ms x C for the baseline, 10 ms x C otherwise), reading the CSV back
# fails (so results come from the console log), and a candidate with --bogus crash-loops.
FAKE_CLUSTER = r"""#!/usr/bin/env python3
import json, os, re, sys
db_path = os.environ["FAKE_DB"]
db = json.load(open(db_path)) if os.path.exists(db_path) else {"objs": {}, "calls": []}
a = sys.argv[1:]
a = a[a.index("--request-timeout=60s") + 1:]
db["calls"].append(" ".join(a[:3]))
def save(code=0, out="", err=""):
    json.dump(db, open(db_path, "w"))
    sys.stdout.write(out)
    sys.stderr.write(err)
    if code == 1 and not out and a[0] == "get":
        sys.stderr.write("Error from server (NotFound): not found\n")
    sys.exit(code)
def pods(name):
    o = db["objs"].get("NIMService/" + name)
    if not o:
        return []
    bad = "--bogus" in o["spec"]["args"]
    out = []
    for i in range(o["spec"]["replicas"]):
        cs = {"restartCount": 3, "state": {"waiting": {"reason": "CrashLoopBackOff"}}} if bad \
            else {"restartCount": 0, "state": {"running": {}}}
        # the image the NIM container runs, as the kubelet reports it once pulled
        cs.update(name="nim", image="nvcr.io/nim/x:1", imageID="nvcr.io/nim/x@sha256:" + "ab" * 32)
        node = os.environ.get("FAKE_NODE") or o["spec"].get("nodeSelector", {}).get(
            "kubernetes.io/hostname", "any-node")
        out.append({"metadata": {"name": f"{name}-{i}"},
                    "spec": {"nodeName": node,
                             "containers": [{"name": "nim", "args": o["spec"]["args"],
                                             "env": o["spec"].get("env", [])}]},
                    "status": {"podIP": f"10.0.0.{i}", "containerStatuses": [cs]}})
    return out
def all_pods():
    return [p for k in db["objs"] if k.startswith("NIMService/") for p in pods(k.split("/")[1])]
if a[0] in ("apply", "create"):
    obj = json.loads(sys.stdin.read())
    key = obj["kind"] + "/" + obj["metadata"]["name"]
    if a[0] == "create" and key in db["objs"]:
        save(1, "", f"Error from server (AlreadyExists): {key} already exists\n")
    if "--dry-run=server" not in a:
        db["objs"][key] = obj
        if obj["kind"] == "Pod":
            db.pop("client_phase", None)  # a new client pod runs
            obj["metadata"]["uid"] = f"uid-{len(db['calls'])}"  # each pod is a new identity
    save(0, json.dumps(obj) if a[0] == "create" else "")  # create -o json prints the object
if a[0] == "get":
    kind, name = a[1], (a[2] if len(a) > 2 and a[2] != "-o" else None)
    if kind == "pods":
        save(0, json.dumps({"items": all_pods()}))
    if kind == "pod":
        if name in [p["metadata"]["name"] for p in all_pods()]:
            save(0, json.dumps([p for p in all_pods() if p["metadata"]["name"] == name][0]))
        o = db["objs"].get("Pod/" + name)
        if o and o["metadata"].get("deletionTimestamp"):
            db["objs"].pop("Pod/" + name)  # it finishes terminating after this read
        if o:
            o = {**o, "status": {"phase": db.get("client_phase", "Running")}}
        save(0 if o else 1, json.dumps(o) if o else "")
    if kind == "nimservice":
        o = db["objs"].get("NIMService/" + name)
        save(0 if o else 1, json.dumps(o) if o else "")
    if kind == "deploy":
        o = db["objs"].get("NIMService/" + name)
        ok = o and "--bogus" not in o["spec"]["args"]
        save(0, json.dumps({"status": {"readyReplicas": o["spec"]["replicas"] if ok else 0}}))
if a[0] == "logs":
    o = db["objs"].get("NIMService/" + a[1].rsplit("-", 1)[0])
    if "--bogus" in o["spec"]["args"]:
        save(0, "vllm: error: argument --bogus: unrecognized\n")
    tp = o["spec"]["args"][1] if o["spec"]["args"][:1] == ["--tensor-parallel-size"] else str(
        o["spec"]["resources"]["limits"]["nvidia.com/gpu"])
    tp_arg = o["spec"]["args"][:1] == ["--tensor-parallel-size"]
    rest = "" if tp_arg else " ".join(o["spec"]["args"])
    line = {"level": "INFO", "message": f"Launching vLLM: vllm serve /m --tensor-parallel-size {tp}"
            f" --gpu-memory-utilization 0.92 --port 8001 {rest}".rstrip()}
    save(0, json.dumps(line) + "\n")
if a[0] == "delete" and a[1] == "--raw":  # DELETE with a uid precondition, as the API does it
    key = "Pod/" + a[2].rsplit("/", 1)[1]
    want = json.loads(sys.stdin.read())["preconditions"]["uid"]
    got = db["objs"].get(key)
    if got is None:
        save(1, "", 'Error from server (NotFound): pods not found')
    elif got.get("metadata", {}).get("uid") != want:
        save(1, "", "Error from server (Conflict): Precondition failed: UID in precondition")
    else:
        db["objs"].pop(key)
        save()
elif a[0] == "delete":
    db["objs"].pop(("NIMService/" if a[1] == "nimservice" else "Pod/") + a[2], None)
    save()
if a[0] in ("wait", "describe"):
    save(0, "  nvidia.com/gpu  4  4\n")
if a[0] == "exec" and a[1] == "-i":  # an upload: keep the bytes' hash under the target path
    import hashlib
    path = a[-1].split("cat > ")[1]
    db.setdefault("files", {})[path] = hashlib.sha256(sys.stdin.read().encode()).hexdigest()
    save()
if a[0] == "exec":
    cmd = a[a.index("--") + 1:]
    if cmd[0] == "python" and "hashlib" in cmd[2]:
        f = db.get("files", {}).get(cmd[3])
        save(0 if f else 1, (f or "") + "\n")
    if db.get("client_phase", "Running") != "Running" and a[1] == "aiperf-client":
        save(1, "error: cannot exec into a container in a completed pod")
    det = None
    if cmd[:2] == ["python", "-c"] and "killpg" in cmd[2]:  # the detached launcher
        db["stops"] = db.get("stops", 0) + 1
        if len(cmd) == 4:  # stop only
            save(0)
        det, cmd = (cmd[4], cmd[5]), cmd[6:]
    remote = db.setdefault("remote", {})
    dies = int(os.environ.get("FAKE_CLIENT_DIES_AFTER", "0"))
    if cmd[:2] == ["aiperf", "profile"] and dies:
        db["profiles"] = db.get("profiles", 0) + 1
        if db["profiles"] > dies:
            db["client_phase"] = "Succeeded"  # its sleep ran out mid-run
            save(0 if det else 137)
    if cmd[:2] == ["aiperf", "profile"] and cmd[cmd.index("--random-seed") + 1] == os.environ.get(
            "FAKE_FAIL_SEED"):
        remote[det[0]], remote[det[1]] = "error: server went away\n", "1\n"
        save(0)
    if cmd[:2] == ["aiperf", "--version"]:
        save(0, "aiperf 0.13.0\n")
    if cmd[:2] == ["test", "-e"]:
        save(0 if cmd[2] in remote else 1)
    if cmd[0] == "cat":
        got = remote.get(cmd[1])
        save(0 if got is not None else 1, got or "")
    if cmd[0] == "python":
        save(0, "200\n" if "chat/completions" in cmd[2] else "100 50 60 0 0\n")
    if cmd[:2] == ["vllm", "serve"]:
        save(0, "  --tensor-parallel-size TP\n  --max-num-seqs N\n  --bogus X\n")
    if cmd[:2] == ["aiperf", "profile"]:
        c = int(cmd[cmd.index("--concurrency") + 1])
        url = cmd[cmd.index("--url") + 1]
        db.setdefault("profiled", []).append(url)
        db.setdefault("profile_cmds", []).append(cmd)
        p99 = (30 if url.endswith("-before:8000") else 10) * c
        b = "│"
        rows = [("Time to First Token (ms)", p99 / 2, p99), ("Inter Token Latency (ms)", 5, 6),
                ("Output Token Throughput (tokens/sec)", 100 * c, None),
                ("Request Error Rate (%)", 0, None)]
        lines = ["NVIDIA AIPerf | LLM Metrics"] + [
            b + b.join([n, f"{v:,.2f}", "1", "2", "N/A" if p is None else f"{p:,.2f}", "1", "1",
                        "0"]) + b for n, v, p in rows] + ["└--"]
        out = "\n".join(lines) + "\n"
        if det:
            remote[det[0]], remote[det[1]] = out, "0\n"
            # the exec stream can drop after the run started; the run itself goes on
            save(1 if os.environ.get("FAKE_STREAM_DROPS") else 0, "connection reset by peer")
        save(0, out)
save(9)
"""


def test_full_campaign_on_a_fake_cluster_then_resume(tmp_path: Path) -> None:
    fake = tmp_path / "kubectl"
    fake.write_text(FAKE_CLUSTER.replace("/usr/bin/env python3", sys.executable))
    fake.chmod(0o755)
    (tmp_path / "client.json").write_text(
        json.dumps({"apiVersion": "v1", "kind": "Pod", "metadata": {"name": "aiperf-client"}})
    )
    path = campaign(
        tmp_path,
        workload={"model": "m", "tokenizer": "org/m", "concurrency": [8, 32], "repeats": 3},
        sla={
            "metric": "ttft_p99",
            "max_ms": 500,
            "search": {"start": 8, "max": 32, "resolution_pct": 25},
        },
        candidates=[
            {"id": "before", "gpus": 4},
            {"id": "tp2x2", "gpus": 2, "replicas": 2, "args": ["--tensor-parallel-size", "2"]},
            {"id": "bad", "gpus": 1, "args": ["--bogus", "1"]},
            {"id": "noflag", "gpus": 4, "args": ["--not-a-flag", "1"]},
            {"id": "spec", "gpus": 4, "args": ["--speculative-config", "{}"]},
        ],
    )
    env = base_env({"NIM_TUNE_KUBECTL": str(fake), "FAKE_DB": str(tmp_path / "db.json")})

    def go() -> tuple[int, dict[str, Any]]:
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), str(path), "--out", str(tmp_path / "r"), *approve(path)],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=120,
            check=False,
            env=env,
        )
        return proc.returncode, json.loads(proc.stdout)

    code, out = go()
    states = out["facts"]["candidates"]
    assert states["before"]["status"] == "measured" and states["tp2x2"]["status"] == "measured"
    assert states["bad"]["status"] == "failed" and states["bad"]["kind"] == "config"
    assert code == 3
    codes = sorted(e["code"] for e in out["errors"])
    assert codes == ["candidate_failed", "candidate_pruned", "candidate_rejected"]
    # never deployed: the engine's own flag list does not have it
    assert states["noflag"]["status"] == "rejected" and "not-a-flag" in states["noflag"]["reason"]
    assert states["noflag"].get("attempts", 0) == 0
    # speculative decoding is pruned on random synthetic prompts
    assert states["spec"]["status"] == "pruned" and "real text" in states["spec"]["reason"]
    assert (tmp_path / "r" / "engine_flags.txt").is_file()
    assert states["tp2x2"]["engagement"]["nimtune-me-tp2x2-0"]["state"] == "engaged"
    assert set(states["tp2x2"]["engagement"]) == {"nimtune-me-tp2x2-0", "nimtune-me-tp2x2-1"}
    # TTFT p99 = 30 x C for the baseline and 10 x C for tp2x2 against a 500 ms target
    # within 25%: the baseline passes 16 and fails 20; tp2x2 still passes at the 32 cap
    assert states["before"]["sla_search"]["max_concurrency"] == 16
    assert states["before"]["sla_search"]["first_failing"] == 20
    assert states["tp2x2"]["sla_search"]["max_concurrency"] == 32
    boundary = out["facts"]["compare"]["sla_boundary"]
    assert (
        boundary["before"]["max_concurrency"] == 16 and boundary["tp2x2"]["max_concurrency"] == 32
    )
    assert (tmp_path / "r" / "final" / "before_after.md").is_file()
    db = json.loads((tmp_path / "db.json").read_text())
    assert db["objs"] == {}  # every NIMService and the client pod were removed
    applied = [c for c in db["calls"] if c.startswith(("apply -f", "create -f"))]
    assert len(applied) == 4  # client, before, tp2x2, bad: never noflag or spec

    profiled_before = len(db["profiled"])
    code, out = go()  # resume: nothing is re-measured and the config failure is not retried
    db = json.loads((tmp_path / "db.json").read_text())
    # every AIPerf profile the fake ran, however it was launched (the runner uses python -c)
    assert len(db["profiled"]) == profiled_before
    assert out["facts"]["candidates"]["bad"]["attempts"] == 1


def test_lost_client_stops_without_blaming_the_candidate(tmp_path: Path) -> None:
    fake = tmp_path / "kubectl"
    fake.write_text(FAKE_CLUSTER.replace("/usr/bin/env python3", sys.executable))
    fake.chmod(0o755)
    (tmp_path / "client.json").write_text(
        json.dumps({"apiVersion": "v1", "kind": "Pod", "metadata": {"name": "aiperf-client"}})
    )
    path = campaign(
        tmp_path,
        workload={"model": "m", "tokenizer": "org/m", "concurrency": [8, 32], "repeats": 3},
        candidates=[
            {"id": "before", "gpus": 4},
            {"id": "seqs64", "gpus": 4, "args": ["--max-num-seqs", "64"]},
        ],
    )
    db_path = tmp_path / "db.json"
    env = base_env({"NIM_TUNE_KUBECTL": str(fake), "FAKE_DB": str(db_path)})

    def go(**extra: str) -> tuple[int, dict[str, Any]]:
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), str(path), "--out", str(tmp_path / "r"), *approve(path)],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=120,
            check=False,
            env={**env, **extra},
        )
        return proc.returncode, json.loads(proc.stdout)

    code, out = go(FAKE_CLIENT_DIES_AFTER="7")  # the baseline's 6 runs finish, then it dies
    assert code == 4 and out["state"] == "client_lost"
    assert [e["code"] for e in out["errors"]] == ["client_lost"]
    state = json.loads((tmp_path / "r" / "campaign_state.json").read_text())
    assert state["candidates"]["before"]["status"] == "measured"
    assert state["candidates"]["seqs64"]["status"] == "pending"  # not failed: rerun it
    db = json.loads(db_path.read_text())
    assert db["objs"] == {}  # the candidate's GPUs were freed and its own client removed

    # a client still terminating is never reused: the runner waits for it to go, then creates one
    db["objs"]["Pod/aiperf-client"] = {
        "kind": "Pod",
        "metadata": {"name": "aiperf-client", "deletionTimestamp": "2026-10-26T17:06:00Z"},
    }
    db_path.write_text(json.dumps(db))
    applied_before = sum(c.startswith(("apply -f", "create -f")) for c in db["calls"])
    code, out = go(FAKE_CLIENT_DIES_AFTER="1")  # dies again, at once, so the test stays short
    db = json.loads(db_path.read_text())
    assert sum(c.startswith(("apply -f", "create -f")) for c in db["calls"][len(db["calls"]) - 40 :]) >= 1
    assert sum(c.startswith(("apply -f", "create -f")) for c in db["calls"]) > applied_before
    assert out["state"] == "client_lost"
    db = json.loads(db_path.read_text())
    db.pop("client_phase", None)
    db["objs"].pop("Pod/aiperf-client", None)

    # a finished client the campaign finds at start is refused, not exec'd into
    db["objs"]["Pod/aiperf-client"] = {"kind": "Pod", "metadata": {"name": "aiperf-client"}}
    db["client_phase"] = "Succeeded"
    db_path.write_text(json.dumps(db))
    code, out = go()
    assert code == 4 and out["state"] == "client_lost"

    db = json.loads(db_path.read_text())
    db["objs"].pop("Pod/aiperf-client", None)  # the user deletes the finished client pod
    db.pop("client_phase", None)
    done_before = len(db["profiled"])
    db_path.write_text(json.dumps(db))
    code, out = go()
    assert out["facts"]["candidates"]["seqs64"]["status"] == "measured"
    # every finished run is kept, the baseline's 6 and seqs64's first; only the other 5 rerun
    redone = json.loads(db_path.read_text())["profiled"][done_before:]
    assert redone == ["http://nimtune-me-seqs64:8000"] * 5


def test_dataset_is_uploaded_once_and_runs_greedy(tmp_path: Path) -> None:
    fake = tmp_path / "kubectl"
    fake.write_text(FAKE_CLUSTER.replace("/usr/bin/env python3", sys.executable))
    fake.chmod(0o755)
    (tmp_path / "client.json").write_text(
        json.dumps({"apiVersion": "v1", "kind": "Pod", "metadata": {"name": "aiperf-client"}})
    )
    (tmp_path / "ds.jsonl").write_text(
        json.dumps({"session_id": "a", "turns": [{"text": "hi", "output_length": 9}]}) + "\n"
    )
    spec = '{"method": "ngram", "num_speculative_tokens": 3, "prompt_lookup_max": 4}'
    path = campaign(
        tmp_path,
        workload={
            "model": "m",
            "tokenizer": "org/m",
            "concurrency": [8],
            "repeats": 3,
            "dataset": {"file": "ds.jsonl"},
        },
        candidates=[
            {"id": "before", "gpus": 4},
            {"id": "ngram3", "gpus": 4, "args": ["--speculative-config", spec]},
        ],
    )
    db_path = tmp_path / "db.json"
    env = base_env({"NIM_TUNE_KUBECTL": str(fake), "FAKE_DB": str(db_path)})
    # the fake engine lists the flag, so the candidate is not rejected
    flags = tmp_path / "flags.txt"
    flags.write_text("  --max-num-seqs N\n  --speculative-config CONFIG\n")
    # the flags file decides what is skipped, so it is part of the approved plan
    flags_sha = hashlib.sha256(flags.read_bytes()).hexdigest()
    plan = rc.plan_id(path, {"engine_flags_sha256": flags_sha})

    def go(**extra: str) -> dict[str, Any]:
        proc = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                str(path),
                "--out",
                str(tmp_path / "r"),
                "--approved",
                plan,
                "--engine-flags",
                str(flags),
            ],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=120,
            check=False,
            env={**env, **extra},
        )
        out: dict[str, Any] = json.loads(proc.stdout)
        return out

    # every exec stream drops after starting AIPerf, and repeat 2 at load 8 (seed 802) fails twice
    out = go(FAKE_STREAM_DROPS="1", FAKE_FAIL_SEED="802")
    states = out["facts"]["candidates"]
    assert states["ngram3"]["failed_runs"] == ["c8-r2"]
    assert not states["ngram3"].get("overloaded_runs")  # AIPerf failed: a missing run, no overload
    log = (tmp_path / "r" / "campaign.log").read_text()
    # a dropped stream loses nothing: only the seed-802 runs needed a second attempt
    assert log.count("attempt 2") == 2
    out = go()  # resume: only the failed runs are measured again
    states = out["facts"]["candidates"]
    assert all(states[c]["status"] == "measured" and not states[c]["failed_runs"] for c in states)
    db = json.loads(db_path.read_text())
    assert db["profiled"][-2:] == [
        "http://nimtune-me-before:8000",
        "http://nimtune-me-ngram3:8000",
    ]
    # with a real dataset, speculative decoding is measured, not pruned, and flagged for accuracy
    assert states["ngram3"]["needs_accuracy_check"]
    db = json.loads(db_path.read_text())
    sha = hashlib.sha256((tmp_path / "ds.jsonl").read_bytes()).hexdigest()
    remote = f"/work/datasets/{sha[:12]}-ds.jsonl"
    assert db["files"] == {remote: sha}
    assert sum(c.startswith("exec -i") for c in db["calls"]) == 1  # uploaded once
    for cmd in db["profile_cmds"]:
        assert cmd[cmd.index("--custom-dataset-type") + 1] == "multi_turn"
        assert cmd[cmd.index("--input-file") + 1] == remote
        assert "temperature:0.0" in cmd and "ignore_eos:true" not in cmd and "--isl" not in cmd
    manifest = json.loads((tmp_path / "r" / "ngram3" / "manifest.json").read_text())
    assert manifest["workload"]["dataset"]["sha256"] == sha  # runs on other data never compare


def test_launcher_runs_detached_and_stops_the_previous_run(tmp_path: Path) -> None:
    # the real launcher, run locally: a fresh client pod has no run folder yet
    pidfile = str(tmp_path / "w" / "aiperf.pid")

    def launch(name: str, script: str) -> tuple[Path, Path]:
        out, ex = tmp_path / "a" / f"{name}.console.log", tmp_path / "a" / f"{name}.exit"
        subprocess.run(
            [sys.executable, "-c", rc.LAUNCH, pidfile, str(out), str(ex), "sh", "-c", script],
            check=True,
            timeout=30,
        )
        return out, ex

    def wait_for(f: Path) -> str:
        for _ in range(200):
            if f.is_file():
                return f.read_text().strip()
            time.sleep(0.05)
        return ""

    _, slow_ex = launch("slow", "sleep 30; echo late")  # an orphan, e.g. from a killed runner
    out, ex = launch("next", "echo hi; exit 3")  # the next launch stops the orphan first
    assert wait_for(ex) == "3" and out.read_text() == "hi\n"
    time.sleep(0.3)
    assert not slow_ex.exists()  # killed: it never reports, so it cannot pass for the new run
    subprocess.run([sys.executable, "-c", rc.LAUNCH, pidfile], check=True, timeout=30)


def test_kubectl_errors_are_retried_and_never_read_as_gone(
    tmp_path: Path, monkeypatch: Any
) -> None:
    fake = tmp_path / "kubectl"
    count = tmp_path / "n"
    fake.write_text(
        f"#!{sys.executable}\n"
        "import pathlib, sys\n"
        f"p = pathlib.Path({str(count)!r}); n = int(p.read_text()) if p.exists() else 0\n"
        "p.write_text(str(n + 1))\n"
        "if 'gone' in sys.argv: sys.stderr.write('Error from server (NotFound)'); sys.exit(1)\n"
        "if n < 2: sys.stderr.write('connection reset by peer'); sys.exit(1)\n"
        "print('{\"ok\": 1}')\n"
    )
    fake.chmod(0o755)
    monkeypatch.setenv("NIM_TUNE_KUBECTL", str(fake))
    monkeypatch.setattr(rc.targets, "RETRY_WAITS", (0, 0, 0, 0))
    kube = rc.targets.Kube("ctx", "ns", lambda m: None)
    assert kube.json("get", "pod", "x") == {"ok": 1}  # two resets, then it answers
    assert kube.json("get", "pod", "gone") is None  # only NotFound means gone
    count.write_text("-10")  # ten failures in a row
    try:
        kube.json("get", "pod", "x")
        raise AssertionError("expected KubeError")
    except rc.targets.KubeError as exc:
        assert "kept failing" in str(exc)


def test_log_match_alone_does_not_fail_a_healthy_pod() -> None:
    healthy = [{"status": {"containerStatuses": [{"restartCount": 0, "state": {"running": {}}}]}}]
    assert rc.classify(healthy, "WARNING ValueError: deprecated option ignored") is None
    crashed = [
        {
            "status": {
                "containerStatuses": [
                    {"restartCount": 1, "lastState": {"terminated": {"reason": "Error"}}}
                ]
            }
        }
    ]
    assert rc.classify(crashed, "torch.OutOfMemoryError: CUDA out of memory")[0] == "config"


def test_a_fatal_line_counts_only_for_the_replica_that_crashed(monkeypatch: Any) -> None:
    import argparse

    healthy = {
        "metadata": {"name": "n-0"},
        "status": {"containerStatuses": [{"restartCount": 0, "state": {"running": {}}}]},
    }
    looping = {"restartCount": 2, "state": {"waiting": {"reason": "CrashLoopBackOff"}}}
    crashed = {"metadata": {"name": "n-1"}, "status": {"containerStatuses": [looping]}}
    logs = {
        "n-0": "WARNING ValueError: max_model_len is larger than the cache; truncated\n",
        "n-1": "loading weights\n",
    }

    class T:
        def ready_count(self, name: str) -> int:
            return 0

        def replicas(self, name: str) -> list[dict[str, Any]]:
            return [healthy, crashed]

        def logs(self, replica: str, previous: bool = False, tail: int | None = None) -> str:
            return "" if previous else logs[replica]

    camp = object.__new__(rc.Campaign)
    camp.t, camp.log = T(), lambda m: None
    camp.args = argparse.Namespace(ready_timeout=60)
    monkeypatch.setattr(rc.time, "sleep", lambda s: None)
    # the healthy replica's warning never makes the other's crash a config failure (no retry)
    assert camp.wait_ready("n", 2) == ("intermittent", "CrashLoopBackOff")
    logs["n-1"] = "torch.OutOfMemoryError: CUDA out of memory\n"  # its own fatal evidence wins
    assert camp.wait_ready("n", 2) == ("config", "OutOfMemoryError")


def test_prune_and_flags_read_values_safely() -> None:
    c = {"workload": {"isl": 1024, "osl": 256, "dataset_args": []}}
    cand = {"gpus": 4, "env": {}, "args": ["--max-model-len", "32k"]}
    assert "cannot be read" in rc.prune(c, cand)
    spec = '{"method":"ngram","x":"a=b"}'
    assert rc.flag_values(["--speculative-config", spec]) == {"--speculative-config": spec}
    assert rc.flag_values(["--max-num-seqs=64"]) == {"--max-num-seqs": "64"}


def test_failed_baseline_stops_the_campaign(tmp_path: Path) -> None:
    fake = tmp_path / "kubectl"
    fake.write_text(FAKE_CLUSTER.replace("/usr/bin/env python3", sys.executable))
    fake.chmod(0o755)
    (tmp_path / "client.json").write_text(
        json.dumps({"apiVersion": "v1", "kind": "Pod", "metadata": {"name": "aiperf-client"}})
    )
    path = campaign(
        tmp_path,
        workload={"model": "m", "tokenizer": "org/m", "concurrency": [8], "repeats": 3},
        candidates=[
            {"id": "before", "gpus": 4, "args": ["--bogus", "1"]},  # crash-loops
            {"id": "seqs64", "gpus": 4, "args": ["--max-num-seqs", "64"]},
        ],
    )
    env = base_env({"NIM_TUNE_KUBECTL": str(fake), "FAKE_DB": str(tmp_path / "db.json")})
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), str(path), "--out", str(tmp_path / "r"), *approve(path)],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=120,
        check=False,
        env=env,
    )
    out = json.loads(proc.stdout)
    assert out["state"] == "baseline_failed"
    assert out["facts"]["stop_condition"] == "baseline not measured"
    assert "seqs64" not in out["facts"]["candidates"]  # never deployed: nothing to compare with
    db = json.loads((tmp_path / "db.json").read_text())
    assert not any("nimtune-me-seqs64" in json.dumps(o) for o in db["objs"].values())


def test_a_run_with_failed_requests_is_measured_again(tmp_path: Path) -> None:
    d = tmp_path / "c8-r1"
    d.mkdir()
    b = "│"
    rows = [("Time to First Token (ms)", 50, 60), ("Request Error Rate (%)", 2.5, None)]
    (d / "aiperf.log").write_text(
        "\n".join(
            ["NVIDIA AIPerf | LLM Metrics"]
            + [
                b
                + b.join(
                    [n, f"{v:,.2f}", "1", "2", "N/A" if p is None else f"{p:,.2f}", "1", "1", "0"]
                )
                + b
                for n, v, p in rows
            ]
            + ["└--"]
        )
        + "\n"
    )
    assert not rc.run_ok(d)  # not counted, so a rerun measures it again
    (d / "aiperf.log").write_text((d / "aiperf.log").read_text().replace("2.50", "0.00"))
    assert rc.run_ok(d)


def test_a_run_without_an_error_rate_row_is_not_counted(tmp_path: Path) -> None:
    def console(rows: list[tuple[str, float, float | None]]) -> str:
        b = "│"
        cells = [
            b + b.join([n, f"{v:,.2f}", "1", "2", "N/A" if p is None else f"{p:,.2f}", "1", "1",
                        "0"]) + b
            for n, v, p in rows
        ]  # fmt: skip
        return "\n".join(["NVIDIA AIPerf | LLM Metrics", *cells, "└--"]) + "\n"

    ttft = ("Time to First Token (ms)", 50.0, 60.0)
    otps = ("Output Token Throughput (tokens/sec)", 900.0, None)
    d = tmp_path / "x" / "c8-r1"
    d.mkdir(parents=True)
    (d / "aiperf.log").write_text(console([ttft, otps]))  # metrics, but no error-rate row
    assert not rc.run_ok(d)  # no evidence that every request succeeded: measured again
    (d / "aiperf.log").write_text(console([ttft, otps, ("Request Error Rate (%)", 0.0, None)]))
    assert rc.run_ok(d)
    # the SLA search never passes a load on a run without the row either
    camp = object.__new__(rc.Campaign)
    camp.out, camp.gate = tmp_path, lambda *a, **k: None
    camp.c = {"sla": {"metric": "ttft_p99", "max_ms": 1000}}
    camp.w = {"repeats": 1}
    camp.measure = lambda cid, name, conc, rep: True
    assert camp.point_passes("x", "n", 8) is True
    (d / "aiperf.log").write_text(console([ttft, otps]))
    assert camp.point_passes("x", "n", 8) is None  # no evidence either way: measured again
    # a zero error rate and throughput but no value for the target metric is unmeasured, not a
    # failure (throughput is kept, so only the missing target metric can decide this)
    (d / "aiperf.log").write_text(console([otps, ("Request Error Rate (%)", 0.0, None)]))
    camp.c = {"sla": {"metric": "ttft_p99", "max_ms": 1000}}
    assert camp.point_passes("x", "n", 8) is None
    # only a readable, positive rate is a measured overload; a missing one is measured again,
    # so an SLA search stays resumable instead of recording the load as failed
    assert not rc.requests_failed(d)
    (d / "aiperf.log").write_text(console([ttft, ("Request Error Rate (%)", 5.0, None)]))
    assert rc.requests_failed(d)


def test_cut_multibyte_output_does_not_crash(tmp_path: Path) -> None:
    fake = tmp_path / "kubectl"
    # a truncated UTF-8 box-drawing character, as when kubectl exec output is cut off
    fake.write_text(f"#!{sys.executable}\nimport sys\nsys.stdout.buffer.write(b'ok \\xe2\\x94')\n")
    fake.chmod(0o755)
    os.environ["NIM_TUNE_KUBECTL"] = str(fake)
    try:
        proc = rc.targets.Kube("ctx", "ns", lambda m: None).run("exec", "pod", "--", "x")
    finally:
        del os.environ["NIM_TUNE_KUBECTL"]
    assert proc.stdout.startswith("ok ")


def test_prune_and_flag_rules() -> None:
    c: dict[str, Any] = {"workload": {"isl": 1024, "osl": 256, "dataset_args": []}}

    def cand(*args: str, gpus: int = 4, **env: str) -> dict[str, Any]:
        return {"id": "x", "gpus": gpus, "args": list(args), "env": env}

    assert rc.prune(c, cand("--max-num-seqs", "64")) is None
    assert "boolean" in rc.prune(c, cand("--enable-chunked-prefill", "false"))
    assert "GPUs per replica" in rc.prune(c, cand("--tensor-parallel-size", "2"))
    assert "headroom" in rc.prune(c, cand("--gpu-memory-utilization", "0.98"))
    oom = rc.prune(c, cand("--max-num-batched-tokens", "2048", "--gpu-memory-utilization", "0.92"))
    assert "out-of-memory" in oom
    assert "ISL + OSL" in rc.prune(c, cand("--max-model-len", "1200"))
    assert "real text" in rc.prune(c, cand(NIM_SPECDEC_ENABLE="1"))
    real = {"workload": {**c["workload"], "dataset_args": ["--input-file", "d.jsonl"]}}
    assert rc.prune(real, cand("--speculative-config", "{}")) is None
    assert rc.needs_accuracy_check(cand("--kv-cache-dtype", "fp8"))
    assert not rc.needs_accuracy_check(cand("--max-num-seqs", "64"))
    help_text = "  --max-num-seqs N\n  --enable-chunked-prefill, --no-enable-chunked-prefill\n"
    assert (
        rc.unsupported_flags(["--max-num-seqs", "8", "--no-enable-chunked-prefill"], help_text)
        == []
    )
    assert rc.unsupported_flags(["--max-num-seq", "8"], help_text) == ["--max-num-seq"]


def test_candidates_are_pinned_to_the_approved_node(tmp_path: Path) -> None:
    fake = tmp_path / "kubectl"
    fake.write_text(FAKE_CLUSTER.replace("/usr/bin/env python3", sys.executable))
    fake.chmod(0o755)
    (tmp_path / "client.json").write_text(
        json.dumps({"apiVersion": "v1", "kind": "Pod", "metadata": {"name": "aiperf-client"}})
    )
    path = campaign(
        tmp_path,
        node="gpu-node-01",
        workload={"model": "m", "tokenizer": "org/m", "concurrency": [8], "repeats": 3},
        candidates=[{"id": "before", "gpus": 4}],
    )
    base = {"NIM_TUNE_KUBECTL": str(fake), "FAKE_DB": str(tmp_path / "db.json")}

    def go(**extra: str) -> dict[str, Any]:
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), str(path), "--out", str(tmp_path / "r"), *approve(path)],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=120,
            check=False,
            env=base_env({**base, **extra}),
        )
        out: dict[str, Any] = json.loads(proc.stdout)
        return out

    out = go(FAKE_NODE="someone-elses-node")  # the scheduler ignored the pin
    st = out["facts"]["candidates"]["before"]
    assert st["status"] == "failed" and "not gpu-node-01" in st["reason"]
    m = json.loads((tmp_path / "r" / "before" / "deploy.json").read_text())
    assert m["spec"]["nodeSelector"] == {"kubernetes.io/hostname": "gpu-node-01"}


def test_the_time_budget_also_stops_the_load_search() -> None:
    camp = object.__new__(rc.Campaign)
    camp.c = {"sla": {"metric": "ttft_p99", "max_ms": 1000}}
    camp.state = {}
    camp.gate = lambda *a, **k: "time budget of 1 h spent"
    assert camp.point_passes("x", "nimtune-x", 64) is None  # no run is started
    assert camp.state["stop_condition"] == "time budget of 1 h spent"


def test_a_candidate_that_will_not_go_stops_the_campaign(monkeypatch: Any) -> None:
    class Stuck:
        def remove(self, name: str, uid: str | None = None) -> bool:
            return True

        def replicas(self, name: str) -> list[dict[str, Any]]:
            return [{"metadata": {"name": name + "-0"}}]

    camp = object.__new__(rc.Campaign)
    camp.t = Stuck()
    camp.log = lambda m: None
    monkeypatch.setattr(rc.time, "sleep", lambda s: None)
    try:
        camp.remove("nimtune-x")
        raise AssertionError("expected KubeError")
    except rc.targets.KubeError as exc:
        assert "still running" in str(exc)


def test_an_engine_that_cannot_initialize_is_a_config_failure() -> None:
    # a speculative-decoding draft that crashes every worker: retrying cannot fix it
    crashed = [
        {
            "status": {
                "containerStatuses": [
                    {"restartCount": 2, "state": {"waiting": {"reason": "CrashLoopBackOff"}}}
                ]
            }
        }
    ]
    log = "RuntimeError: Engine core initialization failed. See root cause above."
    assert rc.classify(crashed, log)[0] == "config"


def test_saved_specs_never_hold_credential_values() -> None:
    ngc, hf = fake_secret("nvapi-"), fake_secret("hf_")
    pod = {
        "spec": {
            "initContainers": [{"name": "init", "env": [{"name": "NGC_API_KEY", "value": ngc}]}],
            "containers": [
                {
                    "name": "nim",
                    "env": [
                        {"name": "HF_TOKEN", "value": hf},
                        {"name": "NIM_MODEL_PATH", "value": "hf://org/m"},
                        {"name": "NGC_API_KEY", "valueFrom": {"secretKeyRef": {"name": "s"}}},
                    ],
                }
            ],
        }
    }
    out = json.dumps(rc.targets.redact(pod))
    assert ngc not in out and hf not in out
    assert "hf://org/m" in out and "secretKeyRef" in out  # non-secrets and references stay
    nimservice = {"spec": {"env": [{"name": "MY_PASSWORD", "value": fake_secret()}]}}
    assert rc.targets.redact(nimservice)["spec"]["env"][0]["value"] == "***"
    argv = ["run", "-e", f"NGC_API_KEY={ngc}", "-e", "NIM_MODEL_PATH=hf://org/m", "-e", "HF_TOKEN"]
    assert rc.targets.redact({"docker": argv})["docker"] == [
        "run",
        "-e",
        "NGC_API_KEY=***",
        "-e",
        "NIM_MODEL_PATH=hf://org/m",
        "-e",
        "HF_TOKEN",
    ]


def test_deploy_record_masks_a_literal_key_in_the_base_manifest(tmp_path: Path) -> None:
    fake = tmp_path / "kubectl"
    fake.write_text(FAKE_CLUSTER.replace("/usr/bin/env python3", sys.executable))
    fake.chmod(0o755)
    (tmp_path / "client.json").write_text(
        json.dumps({"apiVersion": "v1", "kind": "Pod", "metadata": {"name": "aiperf-client"}})
    )
    path = campaign(
        tmp_path,
        workload={"model": "m", "tokenizer": "org/m", "concurrency": [8], "repeats": 3},
        candidates=[{"id": "before", "gpus": 4}],
    )
    base = json.loads((tmp_path / "base.json").read_text())
    literal = fake_secret("nvapi-")
    base["spec"]["env"].append({"name": "NGC_API_KEY", "value": literal})
    (tmp_path / "base.json").write_text(json.dumps(base))
    env = base_env({"NIM_TUNE_KUBECTL": str(fake), "FAKE_DB": str(tmp_path / "db.json")})
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), str(path), "--out", str(tmp_path / "r"), *approve(path)],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=120,
        check=False,
        env=env,
    )
    out = json.loads(proc.stdout)
    assert out["facts"]["candidates"]["before"]["status"] == "measured", out
    # the deploy record exists, and its credential entry is there but masked
    deployed = json.loads((tmp_path / "r" / "before" / "deploy.json").read_text())
    keys = [e for e in deployed["spec"]["env"] if e["name"] == "NGC_API_KEY"]
    assert keys == [{"name": "NGC_API_KEY", "value": "***"}]
    assert literal not in proc.stdout + proc.stderr
    for f in (tmp_path / "r").rglob("*"):
        if f.is_file():
            assert literal not in f.read_text(errors="ignore"), f


def test_a_credential_is_masked_whatever_its_type() -> None:
    got = rc.targets.redact({"env": {"API_KEY": 12345, "MY_TOKEN": None, "NIM_X": 1}})
    assert got["env"] == {"API_KEY": "***", "MY_TOKEN": "***", "NIM_X": 1}
    assert rc.targets.redact({"name": "SECRET_PIN", "value": 4321})["value"] == "***"


def test_candidate_env_values_must_be_scalars(tmp_path: Path) -> None:
    path = campaign(tmp_path, candidates=[{"id": "before", "gpus": 4, "env": {"A": ["x"]}}])
    code, out = run(tmp_path, str(path), "--out", str(tmp_path / "r"), "--dry-run")
    assert code == 2 and "bad_candidate_env" in [e["code"] for e in out["errors"]]


def test_the_scrubber_masks_secrets_but_never_engine_settings() -> None:
    sc = rc.targets.Scrubber()
    api, hf, registered, ngc, bearer = (fake_secret() for _ in range(5))
    sc.add(registered)
    line = (
        f"vllm serve --max-num-batched-tokens 4096 --tokenizer org/m --api-key {api} "
        f"--hf-token={hf} {registered} nvapi-{ngc} Bearer {bearer}"
    )
    out = sc.text(line)
    assert "--max-num-batched-tokens 4096" in out and "--tokenizer org/m" in out
    for secret in (api, hf, registered, ngc, bearer):
        assert secret not in out, secret
    assert rc.targets.redact({"env": {"NIM_MAX_NUM_BATCHED_TOKENS": "8192"}}) == {
        "env": {"NIM_MAX_NUM_BATCHED_TOKENS": "8192"}
    }


def test_the_scrubber_masks_quoted_secrets_with_escaped_quotes() -> None:
    sc = rc.targets.Scrubber()
    out = sc.text('curl --password "ab\\"cdTAIL" --x')
    assert "cdTAIL" not in out and "ab" not in out and out.endswith("--x")
    out = sc.text('--password="\\"correct-horse-battery-staple" --x')
    assert "correct-horse" not in out and "staple" not in out
    out = sc.text("--api-key 'ab\\'cdTAIL' --x")
    assert "cdTAIL" not in out and out.endswith("--x")
    assert sc.text('--api-key "plain" --x') == "--api-key *** --x"


def write_campaign(tmp: Path, **over: Any) -> Path:
    path = campaign(tmp, **over)
    return path


def codes_for(tmp: Path, **over: Any) -> list[str]:
    _, errors = rc.load_campaign(write_campaign(tmp, **over))
    return [e["code"] for e in errors]


def test_malformed_campaign_fields_are_refused_up_front(tmp_path: Path) -> None:
    one = [{"id": "before", "gpus": 4}]
    assert "bad_candidate_args" in codes_for(
        tmp_path, candidates=[{"id": "before", "gpus": 4, "args": "--enforce-eager"}]
    )
    assert "bad_candidate_args" in codes_for(
        tmp_path, candidates=[{"id": "before", "gpus": 4, "args": ["--max-num-seqs", 64]}]
    )
    assert "bad_sla" in codes_for(tmp_path, sla=None, candidates=one)
    assert "bad_sla" in codes_for(
        tmp_path, sla={"metric": "ttft_p99", "max_ms": 1000, "search": [8]}, candidates=one
    )
    assert "bad_field_type" in codes_for(tmp_path, workload=[1], candidates=one)
    assert "bad_field_type" in codes_for(tmp_path, candidates=["before"])
    assert "bad_field_type" in codes_for(tmp_path, candidates={"id": "before"})
    wl: dict[str, Any] = {"model": "m", "tokenizer": "org/m", "concurrency": []}
    assert "campaign_field" in codes_for(tmp_path, workload=wl, candidates=one)
    wl = {"model": "m", "tokenizer": "org/m", "concurrency": 8}
    assert "campaign_field" in codes_for(tmp_path, workload=wl, candidates=one)
    assert "bad_name_prefix" in codes_for(tmp_path, name_prefix=5, candidates=one)
    assert "bad_name" in codes_for(tmp_path, name="../../etc", candidates=one)
    assert "bad_pattern" in codes_for(
        tmp_path, candidates=[{"id": "before", "gpus": 4, "engagement_pattern": "(unclosed"}]
    )


def test_a_dataset_is_uploaded_byte_for_byte(tmp_path: Path, monkeypatch: Any) -> None:
    raw = b'{"session_id": "a", "turns": [{"text": "x\xc3\xa9"}]}\r\n{"session_id": "b"}\r\n'
    (tmp_path / "ds.jsonl").write_bytes(raw)
    sha = hashlib.sha256(raw).hexdigest()
    sent = tmp_path / "sent.bin"
    # a kubectl that keeps the exact bytes it gets on stdin (binary: nothing normalizes CRLF
    # here) and answers the hash check with the hash of what it got
    fake = tmp_path / "kubectl"
    fake.write_text(
        f"#!{sys.executable}\n"
        "import hashlib, pathlib, sys\n"
        f"sent = pathlib.Path({str(sent)!r})\n"
        "if '-i' in sys.argv:\n"
        "    sent.write_bytes(sys.stdin.buffer.read())\n"
        "elif 'hashlib' in ' '.join(sys.argv):\n"
        "    print(hashlib.sha256(sent.read_bytes()).hexdigest() if sent.exists() else '')\n"
    )
    fake.chmod(0o755)
    monkeypatch.setenv("NIM_TUNE_KUBECTL", str(fake))
    camp = object.__new__(rc.Campaign)
    camp.args = type("A", (), {"campaign": str(tmp_path / "campaign.json")})()
    camp.w = {"dataset": {"file": "ds.jsonl", "sha256": sha}}
    camp.log = lambda m: None
    camp.t = rc.targets.K8sTarget({"context": "ctx", "namespace": "ns"}, tmp_path, camp.log)
    camp.t.client = "aiperf-client"
    camp.upload_dataset()  # raises unless the bytes the client got hash to the approved file's
    assert sent.read_bytes() == raw  # CRLF and UTF-8 arrive unchanged


def test_live_runs_need_the_approved_plan(tmp_path: Path) -> None:
    path = campaign(tmp_path)
    code, out = run(tmp_path, str(path), "--out", str(tmp_path / "r"), "--dry-run")
    plan = out["facts"]["plan_id"]
    assert plan == rc.plan_id(path)
    code, out = run(tmp_path, str(path), "--out", str(tmp_path / "r"))
    assert code == 5 and out["errors"][0]["code"] == "not_approved"
    path.write_text(path.read_text().replace('"t1"', '"t2"'))  # the plan changed after review
    code, out = run(tmp_path, str(path), "--out", str(tmp_path / "r"), "--approved", plan)
    assert code == 5 and out["errors"][0]["code"] == "not_approved"


def test_the_plan_id_covers_the_options_that_decide_what_runs(tmp_path: Path) -> None:
    path = campaign(tmp_path)
    out_dir = str(tmp_path / "r")
    _, dry = run(tmp_path, str(path), "--out", out_dir, "--dry-run", "--only", "tp2x2")
    plan = dry["facts"]["plan_id"]
    assert plan != rc.plan_id(path)  # not the id of the full campaign
    for option in (
        [],
        ["--only", "before"],
        ["--max-candidates", "1"],
        ["--max-hours", "1"],
        ["--ready-timeout", "60"],
        ["--max-consecutive-failures", "5"],
    ):
        code, out = run(tmp_path, str(path), "--out", out_dir, "--approved", plan, *option)
        assert code == 5 and out["errors"][0]["code"] == "not_approved", option
    for option in (["--max-hours", "1"], ["--ready-timeout", "60"], ["--max-candidates", "1"]):
        _, d = run(tmp_path, str(path), "--out", out_dir, "--dry-run", *option)
        assert d["facts"]["plan_id"] != rc.plan_id(path), option
    _, again = run(tmp_path, str(path), "--out", out_dir, "--dry-run", "--only", "tp2x2")
    assert again["facts"]["plan_id"] == plan  # same options, same id


def test_main_computes_the_plan_id_once() -> None:
    """The id is hashed from the campaign and its input files; main must do that one time and
    reuse the value (it used to hash the same files four times). It is not cached across runs:
    the id exists to notice a changed file between the dry run and the live run."""
    tree = ast.parse(SCRIPT.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    calls = [
        n
        for n in ast.walk(main)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "plan_id"
    ]
    assert len(calls) == 1


def test_overload_in_the_load_search_misses_the_target(tmp_path: Path) -> None:
    camp = object.__new__(rc.Campaign)
    camp.out = tmp_path
    camp.c = {"sla": {"metric": "ttft_p99", "max_ms": 1000}}
    camp.w = {"repeats": 3}
    camp.state = {}
    camp.gate = lambda *a, **k: None
    camp.search_crashed = False

    def overloaded(cid: str, name: str, conc: int, rep: int) -> bool:
        camp.requests_failed = True
        return False

    camp.measure = overloaded
    assert camp.point_passes("x", "n", 512) is False  # a miss, not "unmeasurable"
    camp.measure = lambda cid, name, conc, rep: None  # a replica restarted
    assert camp.point_passes("x", "n", 512) is None and camp.search_crashed


def test_a_stray_failed_repeat_at_a_fixed_load_is_no_search_miss(
    tmp_path: Path, monkeypatch: Any
) -> None:
    camp = object.__new__(rc.Campaign)
    camp.out = tmp_path
    camp.c = {"sla": {"metric": "ttft_p99", "max_ms": 1000}}
    camp.w = {"repeats": 3, "concurrency": [8, 32]}
    camp.state = {}
    camp.gate = lambda *a, **k: None
    failing: set[str] = set()
    monkeypatch.setattr(rc, "requests_failed", lambda d: d.name in failing)
    failing.update({"c32-r2"})  # two counted repeats and one failed: a gap, not a miss
    assert camp.point_passes("x", "n", 32) is None
    failing.update({"c32-r1", "c32-r3"})  # every repeat failed: a measured overload
    assert camp.point_passes("x", "n", 32) is False
    failing.update({"c48-r1"})  # a search-only load stops at its first failed repeat
    assert camp.point_passes("x", "n", 48) is False


def test_partly_measured_candidates_are_excluded_from_ranking(monkeypatch: Any) -> None:
    seen: list[list[str]] = []

    def fake_run(cmd: list[str], **kw: Any) -> Any:
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, '{"ok": true, "facts": {}}', "")

    monkeypatch.setattr(rc.subprocess, "run", fake_run)
    c: dict[str, Any] = {
        "baseline": "before",
        "candidates": [
            {"id": "before", "gpus": 4, "replicas": 1},
            {"id": "x", "gpus": 4, "replicas": 1},
        ],
        "workload": {"concurrency": [8], "repeats": 3},
    }
    states = {
        "before": {"status": "measured"},
        "x": {"status": "measured", "failed_runs": ["c8-r2"]},
    }
    rc.compare(c, Path("/tmp/out"), states)
    cmd = seen[0]
    assert cmd[cmd.index("--exclude") + 1] == "x" and cmd.count("--exclude") == 1


def test_load_search_and_workload_values_are_checked_before_any_gpu_time(tmp_path: Path) -> None:
    one = [{"id": "before", "gpus": 4}]
    sla: dict[str, Any] = {"metric": "ttft_p99", "max_ms": 1000}
    for search in (
        {"start": "8"},
        {"start": 64, "max": 8},
        {"start": 0},
        {"max": True},
        {"resolution_pct": 0},
        {"resolution_pct": "10"},
        {"resolution_pct": float("inf")},
    ):
        got = codes_for(tmp_path, sla={**sla, "search": search}, candidates=one)
        assert "bad_sla" in got, search
    assert codes_for(tmp_path, sla={**sla, "search": {}}, candidates=one) == []
    assert "bad_sla" in codes_for(tmp_path, sla={**sla, "max_ms": -5}, candidates=one)
    wl: dict[str, Any] = {"model": "m", "tokenizer": "org/m"}
    assert "bad_field_type" in codes_for(
        tmp_path, workload={**wl, "dataset": {"file": 5}}, candidates=one
    )
    assert "campaign_field" in codes_for(tmp_path, workload={**wl, "isl": "1024"}, candidates=one)
    assert "bad_candidate_gpus" in codes_for(tmp_path, candidates=[{"id": "before", "gpus": True}])
    assert "bad_candidate_env" in codes_for(
        tmp_path, candidates=[{"id": "before", "gpus": 4, "env": {"A=B": "1"}}]
    )


def test_host_variable_lists_hold_names_only(tmp_path: Path) -> None:
    def docker(**d: Any) -> list[str]:
        block = {"image": "nvcr.io/nim/x:1", "gpus": "0", **d}
        return codes_for(
            tmp_path, target="docker", docker=block, candidates=[{"id": "before", "gpus": 1}]
        )

    secret = "NGC_API_KEY=" + fake_secret()
    assert "literal_secret" in docker(env_from_host=[secret])
    assert "literal_secret" in docker(client_env_from_host=[secret])
    assert "bad_env_name" in docker(env_from_host=[""])
    assert "bad_env_name" in docker(env={"1X": "y"})
    assert docker(env_from_host=["NGC_API_KEY"]) == []


def test_the_plan_id_covers_the_manifests_and_the_dataset(tmp_path: Path) -> None:
    (tmp_path / "data.jsonl").write_text('{"session_id": "a", "turns": [{"text": "x"}]}\n')
    (tmp_path / "client.json").write_text('{"kind": "Pod", "metadata": {"name": "c"}}')
    wl = {"model": "m", "tokenizer": "org/m", "dataset": {"file": "data.jsonl"}}
    path = campaign(tmp_path, workload=wl)
    plan = rc.plan_id(path)
    for name, old, new in (
        ("base.json", '"m"', '"other"'),
        ("client.json", '"c"', '"d"'),
        ("data.jsonl", '"x"', '"y"'),
    ):
        f = tmp_path / name
        original = f.read_text()
        f.write_text(original.replace(old, new))
        assert rc.plan_id(path) != plan, name  # an approval of the old file is void
        code, out = run(tmp_path, str(path), "--out", str(tmp_path / "r"), "--approved", plan)
        assert code == 5 and out["errors"][0]["code"] == "not_approved", name
        f.write_text(original)
    assert rc.plan_id(path) == plan


def test_a_frozen_input_is_read_once(tmp_path: Path) -> None:
    f = tmp_path / "base.json"
    f.write_text("one")
    (tmp_path / "other.json").write_text("other")
    link = tmp_path / "link.json"
    link.symlink_to(f)
    rc.targets.freeze_inputs()
    try:
        assert rc.targets.read_input(f) == b"one"
        assert rc.targets.read_input(link) == b"one"
        f.write_text("two")  # changed after the plan was hashed
        assert rc.targets.read_input(f) == b"one"  # the run still uses the approved bytes
        link.unlink()
        link.symlink_to(tmp_path / "other.json")  # the symlink retargeted after approval
        assert rc.targets.read_input(link) == b"one"
    finally:
        rc.targets._INPUTS.clear()
        rc.targets._FROZEN = False


def test_escaped_secret_values_are_masked_in_json() -> None:
    sc = rc.targets.Scrubber()
    tail = fake_secret()
    secret = 'pa"ss\\wo\nrd-' + tail  # a quote, a backslash and a newline, escaped by JSON
    sc.add(secret)
    for text in (
        json.dumps({"v": secret}),
        json.dumps({"v": secret + "é"}, ensure_ascii=True),
        sc.dumps({"v": secret, secret: [secret]}),
    ):
        out = sc.text(text)
        assert "pa" + '\\"ss' not in out and tail not in out, out
    # a quoted flag value is redacted before encoding, and the result is still valid JSON
    value = fake_secret()
    flag = f'--api-key="{value}"'
    for obj in ({"cmd": f"vllm serve {flag}"}, {"cmd": [flag, secret]}, {"s": 'a\\"b'}):
        out = sc.dumps(obj, sort_keys=True)
        assert value not in out and tail not in out, out
        json.loads(out)


def test_a_lost_client_frees_only_the_resource_this_run_deployed() -> None:
    removed: list[tuple[str, str | None]] = []
    camp = object.__new__(rc.Campaign)
    camp.current, camp.current_cid = "nimtune-me-x", "x"
    camp.state = {"candidates": {"x": {"deployed_uid": "u1"}}}
    camp.log = lambda msg: None
    camp.remove = lambda name, uid=None: removed.append((name, uid))  # type: ignore[method-assign]
    camp.free_current()
    assert removed == [("nimtune-me-x", "u1")]  # bound to the deployed uid, never by name alone
    camp.state["candidates"]["x"].pop("deployed_uid")
    removed.clear()
    camp.free_current()
    assert removed == [("nimtune-me-x", "")]  # nothing recorded: a live resource is not removed


def test_a_candidate_that_cannot_be_removed_is_reported(capsys: Any) -> None:
    import argparse

    import pytest

    class Target:
        def owner(self, name: str) -> str | None:
            return None  # the target cannot be asked either

        def remove_client(self, uid: str | None = None) -> None:
            raise rc.KubeError("unreachable")

    def remove(name: str, uid: str | None = None) -> None:
        raise rc.KubeError("unreachable")

    camp = object.__new__(rc.Campaign)
    camp.c = {"name": "t1", "name_prefix": "nimtune-me", "candidates": [{"id": "x"}]}
    camp.t = Target()
    camp.args = argparse.Namespace(keep_client=False)
    camp.facts, camp.state = {}, {}
    camp.current = "nimtune-me-x"
    camp.created_client, camp.client_released, camp.client_removed = True, False, False
    camp.state["client_uid"] = "uid-1"
    camp.t.client = "aiperf-client"
    camp.t.client_uid = lambda: "uid-1"  # still the client this campaign created
    camp.log = lambda msg: None
    camp.save = lambda: None
    camp.remove = remove
    camp.free_current()
    with pytest.raises(SystemExit):
        camp.stop("client_lost", "fix_errors", [], 4)
    out = json.loads(capsys.readouterr().out)
    # both still hold resources, and the printed result says so
    assert out["facts"]["left_running"] == ["nimtune-me-x", "aiperf-client"]

    # the target cannot even be listed: the candidate being measured is still reported
    def unreadable(name: str) -> str | None:
        raise rc.KubeError("unreachable")

    camp.t.owner = unreadable
    camp.facts, camp.client_released = {}, False
    with pytest.raises(SystemExit):
        camp.stop("error", "fix_errors", [], 4)
    out = json.loads(capsys.readouterr().out)
    assert out["facts"]["left_running"] == ["aiperf-client", "nimtune-me-x"]


def test_a_load_where_requests_failed_does_not_exclude_the_baseline(monkeypatch: Any) -> None:
    seen: list[list[str]] = []

    def fake_run(cmd: list[str], **kw: Any) -> Any:
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, '{"ok": true, "facts": {}}', "")

    monkeypatch.setattr(rc.subprocess, "run", fake_run)
    ids = ("before", "over", "gone", "partial")
    c: dict[str, Any] = {
        "baseline": "before",
        "candidates": [{"id": i, "gpus": 4, "replicas": 1} for i in ids],
        "workload": {"concurrency": [8, 128], "repeats": 3},
    }
    states = {
        # requests failed at the top load: normal for an untuned NIM, still measured
        "before": {
            "status": "measured",
            "failed_runs": ["c128-r1"],
            "overloaded_runs": ["c128-r1"],
        },
        "over": {"status": "measured", "failed_runs": ["c128-r2"], "overloaded_runs": ["c128-r2"]},
        # a run that never produced a result is really missing
        "gone": {"status": "measured", "failed_runs": ["c8-r2"]},
        "partial": {
            "status": "measured",
            "failed_runs": ["c8-r1", "c128-r1"],
            "overloaded_runs": ["c128-r1"],
        },
    }
    rc.compare(c, Path("/tmp/out"), states)
    cmd = seen[0]
    excluded = [cmd[i + 1] for i, w in enumerate(cmd) if w == "--exclude"]
    assert excluded == ["gone", "partial"]
    states["before"] = {"status": "measured", "failed_runs": ["c8-r1"]}  # the baseline lacks a run
    rc.compare(c, Path("/tmp/out"), states)
    assert "before" in seen[1][seen[1].index("--exclude") + 1 :]


def test_requests_failed_is_reset_for_every_attempt(tmp_path: Path, monkeypatch: Any) -> None:
    camp = object.__new__(rc.Campaign)
    camp.out, camp.c = tmp_path, {"name": "me"}
    camp.w = {"model": "m", "tokenizer": "t", "endpoint_type": "chat", "extra_inputs": []}
    camp.salt_ok = False
    camp.requests_failed = False
    camp.workload_args = lambda: []
    camp.check_client = lambda: None
    camp.log = lambda msg: None
    camp.restarts = lambda name: 0
    camp.write = lambda path, text: None
    camp.t = type(
        "T", (), {"client_exec": lambda *a, **k: subprocess.CompletedProcess([], 1, "", "")}
    )()
    results = iter([(0, "ok"), (124, "no exit code")])  # attempt 1 ran, attempt 2 timed out
    camp.run_detached = lambda cmd, remote: next(results)
    monkeypatch.setattr(rc.compare_aiperf, "read_log", lambda p: {"error_rate": 0.1})
    monkeypatch.setattr(rc, "run_ok", lambda d: False)
    assert camp.measure("x", "n", 8, 1) is False
    assert camp.requests_failed is False  # attempt 2 timed out: nothing says "overload" now
    camp.measure = lambda cid, name, conc, rep: False
    camp.gate = lambda *a, **k: None
    camp.c = {"sla": {"metric": "ttft_p99", "max_ms": 1000}}
    camp.w = {"repeats": 1}
    assert camp.point_passes("x", "n", 8) is None  # unmeasurable, not a missed target


def test_waits_end_on_wall_time_not_only_on_sleeps(monkeypatch: Any) -> None:
    import argparse

    clock = [0.0]
    monkeypatch.setattr(rc.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(rc.time, "sleep", lambda s: None)
    calls: list[str] = []

    class Slow:
        """Every read takes 100 s (kubectl retries against a slow API server)."""

        def ready_count(self, name: str) -> int:
            calls.append("ready")
            clock[0] += 100
            return 0

        def replicas(self, name: str) -> list[dict[str, Any]]:
            return []

        def client_exec(self, *cmd: str, **kw: Any) -> subprocess.CompletedProcess[str]:
            if cmd[0] == "cat":
                calls.append("poll")
                clock[0] += 100
                return subprocess.CompletedProcess(list(cmd), 1, "", "")
            return subprocess.CompletedProcess(list(cmd), 0, "", "")

        def check_client(self) -> None:
            pass

    camp = object.__new__(rc.Campaign)
    camp.t, camp.log, camp.pidfile = Slow(), lambda m: None, "/work/t/aiperf.pid"
    camp.args = argparse.Namespace(ready_timeout=60)
    # counting only the 20 s sleeps, it would read three times over a 60 s budget
    assert camp.wait_ready("n", 1) == ("intermittent", "not ready after 100s")
    assert calls == ["ready"]
    calls.clear()
    # counting only the 15 s sleeps, it would poll ten times over a 150 s budget
    assert camp.run_detached(["aiperf"], "/work/t/x", timeout=150)[0] == 124
    assert calls == ["poll", "poll"]


def test_the_supervisor_outage_wait_ends_on_wall_time(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    path = campaign(tmp_path)
    clock, probes = [0.0], []

    def reachable(c: Any, d: Any) -> bool:
        probes.append(clock[0])
        if len(probes) == 1:
            return True  # answers at the start
        clock[0] += 3600  # then every probe hangs for an hour
        return False

    monkeypatch.setattr(rc.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(rc.time, "sleep", lambda s: None)
    monkeypatch.setattr(rc, "target_reachable", reachable)
    monkeypatch.setattr(rc, "run_child", lambda cmd, env: (0, "{}"))
    try:
        rc.main([str(path), "--out", str(tmp_path / "r"), *approve(path), "--supervise",
                 "--max-wait-hours", "1"])  # fmt: skip
    except SystemExit as exc:
        code = int(exc.code or 0)
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert code == rc.EXIT_EXTERNAL and out["errors"][0]["code"] == "target_unreachable"
    assert len(probes) == 2  # counting only the 60 s sleeps, it would probe 60 times


def test_the_engine_flags_file_is_part_of_the_plan(tmp_path: Path) -> None:
    path = campaign(tmp_path)
    out_dir = str(tmp_path / "r")
    flags = tmp_path / "flags.txt"
    flags.write_text("  --max-num-seqs N\n")  # no --tensor-parallel-size: tp2x2 is skipped
    _, dry = run(tmp_path, str(path), "--out", out_dir, "--dry-run", "--engine-flags", str(flags))
    plan = dry["facts"]["plan_id"]
    assert "skipped" in dry["facts"]["candidates"]["tp2x2"]
    assert plan != rc.plan_id(path)
    # without the file the baseline is probed, and tp2x2 could run: not the approved plan
    code, out = run(tmp_path, str(path), "--out", out_dir, "--approved", plan)
    assert code == 5 and out["errors"][0]["code"] == "not_approved"
    flags.write_text("  --max-num-seqs N\n  --tensor-parallel-size TP\n")  # changed after review
    code, out = run(
        tmp_path, str(path), "--out", out_dir, "--approved", plan, "--engine-flags", str(flags)
    )
    assert code == 5 and out["errors"][0]["code"] == "not_approved"
    missing = str(tmp_path / "nope.txt")
    code, out = run(tmp_path, str(path), "--out", out_dir, "--dry-run", "--engine-flags", missing)
    assert code == 2 and out["errors"][0]["code"] == "bad_usage"
    empty = tmp_path / "empty.txt"
    empty.write_text("  \n")  # a failed `vllm serve --help=all > file` redirect
    code, out = run(tmp_path, str(path), "--out", out_dir, "--dry-run", "--engine-flags", str(empty))
    assert code == 2 and out["errors"][0]["code"] == "bad_usage"


def test_a_pending_removal_saved_under_another_name_is_kept() -> None:
    import pytest

    # name_prefix changed since the failed removal: the old resource still runs, so the marker
    # stays, and the current prefix is never removed in its place
    camp = object.__new__(rc.Campaign)
    camp.c = {"name_prefix": "new"}
    camp.log = lambda m: None
    camp.save = lambda: None
    removed: list[str] = []
    camp.remove = removed.append
    running = {"old-x"}
    camp.t = type(
        "T",
        (),
        {
            "replicas": lambda self, n: [{"name": n}] if n in running else [],
            "exists": lambda self, n: n in running,
        },
    )()
    camp.state = {"candidates": {"x": {"status": "measured", "cleanup_pending": "old-x"}}}
    with pytest.raises(rc.PendingCleanupError, match="old-x"):
        camp.clear_pending()  # the run stops: old-x may still hold GPUs
    assert removed == [] and camp.state["candidates"]["x"]["cleanup_pending"] == "old-x"
    running.clear()  # the user removed old-x: the rerun clears the marker and goes on
    camp.clear_pending()
    assert removed == [] and "cleanup_pending" not in camp.state["candidates"]["x"]
    camp.state["candidates"]["x"]["cleanup_pending"] = True  # a marker from before this change
    camp.clear_pending()  # nothing of it is there: cleared, nothing removed
    assert removed == [] and "cleanup_pending" not in camp.state["candidates"]["x"]


def test_a_pending_removal_under_another_name_is_kept_while_the_resource_has_no_pods() -> None:
    import pytest

    # the old NIMService still exists but has no pods right now (interrupted before deletion,
    # or between pods): it can create them later, so the marker stays and the run stops
    camp = object.__new__(rc.Campaign)
    camp.c = {"name_prefix": "new"}
    camp.log = lambda m: None
    camp.save = lambda: None
    removed: list[str] = []
    camp.remove = removed.append
    present = {"old-x"}
    camp.t = type(
        "T",
        (),
        {"replicas": lambda self, n: [], "exists": lambda self, n: n in present},
    )()
    camp.state = {"candidates": {"x": {"status": "measured", "cleanup_pending": "old-x"}}}
    with pytest.raises(rc.PendingCleanupError, match="old-x"):
        camp.clear_pending()
    assert removed == [] and camp.state["candidates"]["x"]["cleanup_pending"] == "old-x"
    present.clear()  # the user deleted the NIMService: now the marker is cleared
    camp.clear_pending()
    assert removed == [] and "cleanup_pending" not in camp.state["candidates"]["x"]


def test_a_candidate_whose_removal_failed_is_removed_before_anything_else(tmp_path: Path) -> None:
    import pytest

    camp = object.__new__(rc.Campaign)
    camp.c = {"name_prefix": "p", "baseline": "x", "context": "ctx", "namespace": "ns"}
    camp.out, camp.w = tmp_path, {"concurrency": [8], "repeats": 1}
    (tmp_path / "x" / "c8-r1").mkdir(parents=True)
    (tmp_path / "x" / "c8-r1" / "aiperf.log").write_text(aiperf_console(CLEAN))  # it counts
    camp.log = lambda m: None
    saved: list[dict[str, Any]] = []
    camp.save = lambda: saved.append(json.loads(json.dumps(camp.state)))

    def stuck(name: str, uid: str | None = None) -> None:
        raise rc.KubeError(f"{name} is still running 5 minutes after removal")

    camp.t = type("T", (), {"ident": lambda self, n: "uid-1"})()
    # measured, then its removal failed: the result is saved, the removal marked pending
    camp.state = {"candidates": {"x": {"status": "measured", "deployed_uid": "uid-1"}}}
    camp.remove = stuck
    with pytest.raises(rc.KubeError):
        camp.retire("p-x", camp.state["candidates"]["x"])
    assert saved[-1]["candidates"]["x"] == {
        "status": "measured",
        "deployed_uid": "uid-1",
        "cleanup_uid": "uid-1",
        "cleanup_pending": "p-x",
        "cleanup_target": "kube:ctx/ns",
    }

    # the rerun removes it first, and keeps its results instead of measuring it again
    removed: list[str] = []
    camp.remove = lambda name, uid=None: removed.append(name)
    camp.fingerprint = lambda cand: None
    camp.run_candidate({"id": "x"})
    assert removed == ["p-x"]
    assert camp.state["candidates"]["x"] == {"status": "measured", "deployed_uid": "uid-1"}


def test_a_rerun_removes_a_pending_cleanup_whatever_it_selects_or_stops_on(
    tmp_path: Path,
) -> None:
    fake = tmp_path / "kubectl"
    fake.write_text(FAKE_CLUSTER.replace("/usr/bin/env python3", sys.executable))
    fake.chmod(0o755)
    (tmp_path / "client.json").write_text(
        json.dumps({"apiVersion": "v1", "kind": "Pod", "metadata": {"name": "aiperf-client"}})
    )
    path = campaign(
        tmp_path,
        workload={"model": "m", "tokenizer": "org/m", "concurrency": [8], "repeats": 3},
        candidates=[
            {"id": "before", "gpus": 4},
            {"id": "tp2x2", "gpus": 2, "replicas": 2, "args": ["--tensor-parallel-size", "2"]},
        ],
    )
    db_path, state_path = tmp_path / "db.json", tmp_path / "r" / "campaign_state.json"
    env = base_env({"NIM_TUNE_KUBECTL": str(fake), "FAKE_DB": str(db_path)})

    def go(*options: str, plan: dict[str, Any] | None = None) -> dict[str, Any]:
        approved = ["--approved", rc.plan_id(path, plan)]
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), str(path), "--out", str(tmp_path / "r"), *approved,
             *options],
            capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=120, check=False,
            env=env,
        )  # fmt: skip
        out: dict[str, Any] = json.loads(proc.stdout)
        return out

    first = go()
    assert {k: v["status"] for k, v in first["facts"]["candidates"].items()} == {
        "before": "measured", "tp2x2": "measured"}  # fmt: skip
    measured = json.loads(state_path.read_text())["candidates"]
    profiled = len(json.loads(db_path.read_text())["profiled"])

    def removal_failed() -> None:
        # tp2x2's removal failed in the last run: it still holds its GPUs
        state = json.loads(state_path.read_text())
        state["candidates"]["tp2x2"]["cleanup_pending"] = "nimtune-me-tp2x2"  # the name saved
        state["candidates"]["tp2x2"]["cleanup_uid"] = "uid-tp2x2"  # and which one it was
        state_path.write_text(json.dumps(state))
        db = json.loads(db_path.read_text())
        db["objs"]["NIMService/nimtune-me-tp2x2"] = {
            "kind": "NIMService",
            "metadata": {"name": "nimtune-me-tp2x2", "labels": {rc.LABEL: "t1"},
                         "uid": "uid-tp2x2"},
            "spec": {"args": ["--tensor-parallel-size", "2"], "replicas": 2},
        }
        db_path.write_text(json.dumps(db))

    # --only leaves tp2x2 out, and the candidate cap stops before any candidate: either way the
    # documented rerun removes it first, and keeps every measurement
    for options, plan in ((["--only", "before"], {"only": ["before"]}),
                          (["--max-candidates", "1"], {"max_candidates": 1})):  # fmt: skip
        removal_failed()
        go(*options, plan=plan)
        db = json.loads(db_path.read_text())
        assert "NIMService/nimtune-me-tp2x2" not in db["objs"], options
        assert len(db["profiled"]) == profiled, options  # nothing measured again
        state = json.loads(state_path.read_text())["candidates"]
        assert state == measured, options


def test_a_comparison_timeout_is_a_contract_error(tmp_path: Path, monkeypatch: Any) -> None:
    def slow(cmd: list[str], **kw: Any) -> Any:
        raise subprocess.TimeoutExpired(cmd, 300)

    monkeypatch.setattr(rc.subprocess, "run", slow)
    c: dict[str, Any] = {
        "baseline": "before",
        "candidates": [{"id": "before", "gpus": 4, "replicas": 1}],
        "workload": {"concurrency": [8], "repeats": 3},
    }
    res = rc.compare(c, tmp_path, {"before": {"status": "measured"}})
    assert res["ok"] is False and [e["code"] for e in res["errors"]] == ["compare_failed"]
    assert "kept" in res["errors"][0]["fix"]


def test_a_replaced_replica_voids_the_run(tmp_path: Path) -> None:
    camp = object.__new__(rc.Campaign)
    camp.out, camp.c = tmp_path, {"name": "me"}
    camp.w = {"model": "m", "tokenizer": "t", "endpoint_type": "chat", "extra_inputs": []}
    camp.salt_ok = False
    camp.workload_args = lambda: []
    camp.check_client = lambda: None
    camp.log = lambda msg: None
    camp.write = lambda path, text: None
    camp.run_detached = lambda cmd, remote: (0, "ok")
    # the old pod was replaced mid-run: both have restartCount 0, so the totals match
    uids = iter(["pod-a", "pod-b"])

    class T:
        def replicas(self, name: str) -> list[dict[str, Any]]:
            meta = {"name": "n-0", "uid": next(uids)}
            return [{"metadata": meta, "status": {"containerStatuses": [{"restartCount": 0}]}}]

    camp.t = T()
    assert camp.measure("x", "n", 8, 1) is None


def test_a_replaced_docker_container_voids_the_run(tmp_path: Path) -> None:
    camp = object.__new__(rc.Campaign)
    camp.out, camp.c = tmp_path, {"name": "me"}
    camp.w = {"model": "m", "tokenizer": "t", "endpoint_type": "chat", "extra_inputs": []}
    camp.salt_ok = False
    camp.workload_args = lambda: []
    camp.check_client = lambda: None
    camp.log = lambda msg: None
    camp.write = lambda path, text: path.write_text(text)
    camp.run_detached = lambda cmd, remote: (0, "ok")
    # `docker rm -f n; docker run --name n ...` mid-run: same name, running, RestartCount 0
    ids = (f"{n:06x}" for n in range(1 << 20))  # each inspect sees a new container
    t = rc.targets.DockerTarget(
        {"name": "me", "name_prefix": "n", "docker": {"image": "i", "gpus": "0"}},
        tmp_path,
        lambda m: None,
    )
    t.inspect = lambda name: {"Id": next(ids), "State": {"Running": True}, "RestartCount": 0}
    camp.t = t
    assert camp.measure("x", "n", 8, 1) is None
    t.inspect = lambda name: {"Id": "c0ffee", "State": {"Running": True}, "RestartCount": 0}
    assert t.replicas("n")[0]["metadata"] == {"name": "n", "uid": "c0ffee"}


def test_only_the_client_this_campaign_created_is_removed() -> None:
    import argparse

    removed: list[str] = []

    class T:
        client = "aiperf-client"
        uid = "uid-2"  # someone else's pod took the name during the run

        def client_uid(self) -> str:
            return self.uid

        def remove_client(self, uid: str | None = None) -> None:
            removed.append(f"{self.client}@{uid}")

    camp = object.__new__(rc.Campaign)
    camp.t, camp.log = T(), lambda m: None
    camp.args = argparse.Namespace(keep_client=False)
    camp.facts, camp.state = {}, {"client_uid": "uid-1"}
    camp.created_client, camp.client_released, camp.client_removed = True, False, False
    camp.release_client()
    assert removed == [] and camp.client_removed and "client_uid" not in camp.state
    camp.t.uid, camp.state["client_uid"], camp.client_released = "uid-1", "uid-1", False
    camp.release_client()
    assert removed == ["aiperf-client@uid-1"]  # the delete carries the recorded uid


def test_runner_owned_aiperf_options_in_dataset_args_are_refused(tmp_path: Path) -> None:
    # a custom export name would leave the fetch reading an earlier run's default-named CSV
    one = [{"id": "before", "gpus": 4}]
    for args in (
        ["--profile-export-prefix", "x"], ["--concurrency=4"], ["--artifact-dir", "/a"],
        ["-m", "other"], ["--num-requests", "5"], ["-u", "http://x"], ["--endpoint", "/v1/x"],
        ["--custom-endpoint=/v1/x"], ["--num-profile-runs", "3"],
    ):  # fmt: skip
        wl = {"model": "m", "tokenizer": "org/m", "dataset_args": ["--input-file", "d", *args]}
        assert "bad_dataset_args" in codes_for(tmp_path, workload=wl, candidates=one), args


def test_credentials_in_aiperf_dataset_args_are_refused(tmp_path: Path) -> None:
    one = [{"id": "before", "gpus": 4}]
    value = fake_secret()
    for args in (
        ["--api-key", value],
        [f"--api-key={value}"],
        ["-H", f"Authorization: Bearer {value}"],
        ["--header", f"X-Api-Key: {value}"],
        [f"--header=Cookie: session={value}"],
    ):
        wl = {"model": "m", "tokenizer": "org/m", "dataset_args": ["--input-file", "d", *args]}
        _, errors = rc.load_campaign(write_campaign(tmp_path, workload=wl, candidates=one))
        assert "literal_secret" in [e["code"] for e in errors], args
        assert value not in json.dumps(errors)  # the value never reaches the output
    wl = {"model": "m", "tokenizer": "org/m", "dataset_args": ["-H", "X-Request-Source: bench"]}
    assert codes_for(tmp_path, workload=wl, candidates=one) == []


def test_every_header_in_a_json_or_listed_aiperf_header_is_checked(tmp_path: Path) -> None:
    one = [{"id": "before", "gpus": 4}]
    value = fake_secret()
    harmless = {"X-Request-Source": "bench"}
    for args in (
        # a harmless header first used to hide the credential behind it
        ["-H", json.dumps({**harmless, "Authorization": f"Bearer {value}"})],
        ["--header", json.dumps({**harmless, "X-Api-Key": value})],
        [f"--header={json.dumps({**harmless, 'Cookie': f'session={value}'})}"],
        ["-H", '{"X-Request-Source": "bench",', f'"Authorization": "Bearer {value}"}}'],
        ["-H", f"X-Request-Source: bench,Authorization: Bearer {value}"],
        ["-H", '{"X-Request-Source": "bench", "Authorization": broken'],  # not JSON: all of it
        [f"-HAuthorization: Bearer {value}"],  # the attached short form
        ["-H", "X-Route:nim", f"X-Api-Key:{value}"],  # -H takes a list: the second one counts
        ["--header", "X-Route: nim", f"Authorization: Bearer {value}"],
    ):
        assert rc.secret_aiperf_args(args) in (["-H"], ["--header"]), args  # names, no value
        wl = {"model": "m", "tokenizer": "org/m", "dataset_args": ["--input-file", "d", *args]}
        _, errors = rc.load_campaign(write_campaign(tmp_path, workload=wl, candidates=one))
        assert "literal_secret" in [e["code"] for e in errors], args
        assert value not in json.dumps(errors)  # the value never reaches the output
    for args in (
        ["-H", json.dumps({**harmless, "Accept": "application/json"})],
        ["-H", "X-Request-Source: bench,X-Trace: 1"],
        ["--header=X-Request-Source: bench run"],
        ["-H", "X-Route:nim", "X-Trace:1"],  # a harmless list stays allowed
    ):
        assert rc.secret_aiperf_args(args) == [], args
        wl = {"model": "m", "tokenizer": "org/m", "dataset_args": args}
        assert codes_for(tmp_path, workload=wl, candidates=one) == [], args


def test_docker_aiperf_version_and_the_client_name_are_checked_up_front(tmp_path: Path) -> None:
    one = [{"id": "before", "gpus": 2}]

    def docker(**d: Any) -> dict[str, Any]:
        return {"image": "nvcr.io/nim/x:1", "gpus": "0,1", **d}

    ok = codes_for(tmp_path, target="docker", docker=docker(aiperf_version="0.14.0rc1"),
                   candidates=one)  # fmt: skip
    assert ok == []
    for version in ("0.13.0; touch /tmp/x", "$(id)", 13):
        got = codes_for(tmp_path, target="docker", docker=docker(aiperf_version=version),
                        candidates=one)  # fmt: skip
        assert "bad_field_type" in got, version
    named = [*one, {"id": "aiperf", "gpus": 2}]
    got = codes_for(tmp_path, target="docker", docker=docker(), candidates=named)
    assert "bad_candidate_id" in got
    assert "bad_candidate_id" not in codes_for(tmp_path, candidates=[*one, {"id": "aiperf"}])


def test_malformed_budget_and_metric_are_input_errors(tmp_path: Path) -> None:
    one = [{"id": "before", "gpus": 4}]
    got = codes_for(tmp_path, gpu_budget="4", candidates=one)
    assert "campaign_field" in got and "over_budget" not in got
    assert "bad_sla" in codes_for(tmp_path, sla={"metric": [], "max_ms": 1000}, candidates=one)
    code, out = run(tmp_path, str(campaign(tmp_path, gpu_budget="4")), "--out",
                    str(tmp_path / "r"), "--dry-run")  # fmt: skip
    assert code == 2 and out["next"] == "fix_errors"


def test_a_config_failure_on_any_replica_wins_over_an_image_pull() -> None:
    pull = {"status": {"containerStatuses": [{"state": {"waiting": {"reason": "ErrImagePull"}}}]}}
    killed = {"lastState": {"terminated": {"reason": "OOMKilled"}}}
    oom = {"status": {"containerStatuses": [killed]}}
    assert rc.classify([pull, oom], "") == ("config", "OOMKilled")
    assert rc.classify([oom, pull], "") == ("config", "OOMKilled")
    assert rc.classify([pull], "") == ("intermittent", "ErrImagePull")


def test_the_load_search_assumes_latency_grows_with_load() -> None:
    # passes up to 10, fails from 11, passes again from 30 (noise or a cache effect): the
    # search reports the boundary below the first failing load and never probes above it
    got = rc.sla_search(lambda n: n <= 10 or n >= 30, start=8, hi_cap=64, resolution_pct=10)
    assert got["complete"] and got["max_concurrency"] == 10 and got["first_failing"] == 11
    assert max(got["tried"]) == 16  # the first failure; 30 and above are never probed


# ---------------------------------------------------------------- one runner, supervised resume


def dead_pid() -> int:
    """The id of a process that has already exited."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def write_lock(out: Path, pid: int, host: str | None = None) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    lock = out / rc.LOCK_NAME
    lock.write_text(json.dumps({"pid": pid, "host": host or rc.socket.gethostname(), "plan_id": "p"}))
    return lock


def hold_lock(out: Path, pid: int, host: str | None = None) -> int:
    """Play a live runner: write its lock and hold the flock on it (close the fd to let go)."""
    lock = write_lock(out, pid, host)
    fd = os.open(lock, os.O_RDWR)
    rc.fcntl.flock(fd, rc.fcntl.LOCK_EX | rc.fcntl.LOCK_NB)
    return fd


def test_a_second_runner_in_the_same_folder_is_refused(tmp_path: Path) -> None:
    path = campaign(tmp_path)
    fd = hold_lock(tmp_path / "r", os.getpid())  # this test process plays the running copy
    try:
        code, out = run(tmp_path, str(path), "--out", str(tmp_path / "r"), *approve(path))
    finally:
        os.close(fd)
    assert code == rc.EXIT_REFUSED and out["errors"][0]["code"] == "already_running"
    assert not (tmp_path / "r" / "campaign_state.json").exists()  # refused before any work


def test_a_lock_nobody_holds_is_taken_over_even_if_its_pid_is_reused(tmp_path: Path) -> None:
    # after a reboot or crash the recorded pid can belong to an unrelated live process
    path = campaign(tmp_path)
    write_lock(tmp_path / "r", os.getpid())  # alive, but holds no flock
    _, out = run(tmp_path, str(path), "--out", str(tmp_path / "r"), *approve(path))
    assert "already_running" not in {e["code"] for e in out["errors"]}


def test_only_one_of_several_simultaneous_runners_gets_the_folder(tmp_path: Path) -> None:
    write_lock(tmp_path / "r", dead_pid())  # all of them race to take over the same stale lock
    code = (
        f"import sys, time; sys.path.insert(0, {str(SCRIPT.parent)!r}); import run_campaign as rc;"
        f"from pathlib import Path; rc.acquire_lock(Path({str(tmp_path / 'r')!r}), 'p');"
        "time.sleep(3)"
    )
    procs = [subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.DEVNULL)
             for _ in range(6)]  # fmt: skip
    codes = [p.wait(timeout=60) for p in procs]
    assert codes.count(0) == 1 and codes.count(rc.EXIT_REFUSED) == 5


def test_a_lock_from_another_host_is_never_taken_over(tmp_path: Path) -> None:
    path = campaign(tmp_path)
    write_lock(tmp_path / "r", dead_pid(), host="some-other-host")
    code, out = run(tmp_path, str(path), "--out", str(tmp_path / "r"), *approve(path))
    assert code == rc.EXIT_REFUSED and out["errors"][0]["code"] == "already_running"


def test_a_stale_lock_is_replaced_and_released_at_exit(tmp_path: Path) -> None:
    path = campaign(tmp_path)
    lock = write_lock(tmp_path / "r", dead_pid())
    _, out = run(tmp_path, str(path), "--out", str(tmp_path / "r"), *approve(path))
    assert "already_running" not in {e["code"] for e in out["errors"]}
    assert not lock.exists()  # the run's own lock is gone once it exits


def test_a_supervised_child_shares_its_supervisor_lock(tmp_path: Path, monkeypatch: Any) -> None:
    fd = hold_lock(tmp_path / "r", os.getppid())
    monkeypatch.setenv(rc.SUPERVISOR_ENV, str(os.getppid()))
    try:
        assert rc.acquire_lock(tmp_path / "r", "p") is None
    finally:
        os.close(fd)


def test_abbreviated_options_are_refused(tmp_path: Path) -> None:
    # "--sup" would supervise but not be stripped from the attempt's own command line
    path = campaign(tmp_path)
    code, out = run(tmp_path, str(path), "--out", str(tmp_path / "r"), *approve(path), "--sup")
    assert code == rc.EXIT_INPUT and out["errors"][0]["code"] == "bad_usage"


def test_keep_awake_watches_the_supervisor_not_each_attempt(monkeypatch: Any) -> None:
    monkeypatch.setattr(rc.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(rc.sys, "platform", "darwin")
    assert rc.awake_command(42) == ["caffeinate", "-is", "-w", "42"]
    monkeypatch.setattr(rc.sys, "platform", "linux")
    cmd = rc.awake_command(42)
    assert cmd is not None and cmd[0] == "systemd-inhibit" and cmd[-4:-2] == ["tail", "--pid=42"]


def test_supervise_is_not_part_of_the_plan(tmp_path: Path) -> None:
    path = campaign(tmp_path)
    _, plain = run(tmp_path, str(path), "--out", str(tmp_path / "r"), "--dry-run")
    _, sup = run(
        tmp_path, str(path), "--out", str(tmp_path / "r"), "--dry-run", "--supervise",
        "--max-resumes", "3", "--keep-awake",
    )
    assert plain["facts"]["plan_id"] == sup["facts"]["plan_id"]


def test_child_argv_drops_only_the_supervisor_options() -> None:
    argv = ["c.json", "--out", "r", "--supervise", "--max-resumes", "5", "--resume-wait=1",
            "--keep-awake", "--max-hours", "8", "--approved", "abc"]
    assert rc.child_argv(argv) == ["c.json", "--out", "r", "--max-hours", "8", "--approved", "abc"]


def supervised(tmp_path: Path, monkeypatch: Any, results: list[tuple[int, dict[str, Any]]],
               capsys: Any, *extra: str) -> tuple[int, dict[str, Any], list[list[str]]]:
    """Run main() with --supervise against scripted child results; nothing is deployed."""
    path = campaign(tmp_path)
    calls: list[list[str]] = []

    def fake_child(cmd: list[str], env: dict[str, str] | None) -> tuple[int, str]:
        calls.append(cmd)
        # the attempt inherits the supervisor's environment, with only this name set in it
        assert env is None and os.environ[rc.SUPERVISOR_ENV] == str(os.getpid())
        code, payload = results.pop(0)
        return code, json.dumps({"ok": False, "state": "error", "next": "fix_errors",
                                 "facts": {}, **payload})

    monkeypatch.setattr(rc, "run_child", fake_child)
    monkeypatch.setattr(rc, "target_reachable", lambda c, d: True)
    monkeypatch.setattr(rc.time, "sleep", lambda s: None)
    monkeypatch.setattr(rc, "start_awake", lambda log: calls.append(["keep-awake"]) or None)
    argv = [str(path), "--out", str(tmp_path / "r"), *approve(path), "--supervise", *extra]
    try:
        rc.main(argv)
    except SystemExit as exc:
        code = int(exc.code or 0)
    return code, json.loads(capsys.readouterr().out.strip().splitlines()[-1]), calls


def transient(code: str) -> tuple[int, dict[str, Any]]:
    return rc.EXIT_EXTERNAL, {"errors": [{"code": code, "message": "m", "fix": "f"}]}


def test_supervise_resumes_after_transient_stops_then_ends(tmp_path: Path, monkeypatch: Any,
                                                           capsys: Any) -> None:
    done = (0, {"ok": True, "state": "measured", "next": "decide", "errors": []})
    code, out, calls = supervised(
        tmp_path, monkeypatch, [transient("kubectl_failed"), transient("client_lost"), done], capsys
    )
    assert code == 0 and out["state"] == "measured" and len(calls) == 3
    assert out["facts"]["supervisor"]["resumes"] == 2
    assert all("--supervise" not in c and "--approved" in c for c in calls)
    assert not (tmp_path / "r" / rc.LOCK_NAME).exists()


def test_supervise_never_retries_a_real_failure(tmp_path: Path, monkeypatch: Any,
                                                capsys: Any) -> None:
    failed = (rc.EXIT_PRECONDITION, {"state": "baseline_failed",
                                     "errors": [{"code": "candidate_failed", "message": "OOM",
                                                 "fix": "f"}]})
    code, out, calls = supervised(tmp_path, monkeypatch, [failed], capsys)
    assert code == rc.EXIT_PRECONDITION and len(calls) == 1
    assert out["errors"][0]["code"] == "candidate_failed"


def test_supervise_gives_up_after_max_resumes(tmp_path: Path, monkeypatch: Any,
                                              capsys: Any) -> None:
    code, out, calls = supervised(
        tmp_path, monkeypatch, [transient("kubectl_failed")] * 3, capsys, "--max-resumes", "2"
    )
    assert code == rc.EXIT_EXTERNAL and len(calls) == 3
    assert out["facts"]["supervisor"]["resumes"] == 2


def test_keep_awake_starts_once_and_attempts_run_directly(tmp_path: Path, monkeypatch: Any,
                                                         capsys: Any) -> None:
    done = (0, {"ok": True, "state": "measured", "next": "decide", "errors": []})
    _, _, calls = supervised(
        tmp_path, monkeypatch, [transient("kubectl_failed"), done], capsys, "--keep-awake"
    )
    assert calls[0] == ["keep-awake"]  # started once, before the first attempt
    assert [c[0] for c in calls[1:]] == [sys.executable, sys.executable]  # no inhibitor prefix


def test_the_same_unexpected_error_twice_in_a_row_is_final(tmp_path: Path, monkeypatch: Any,
                                                          capsys: Any) -> None:
    code, out, calls = supervised(
        tmp_path, monkeypatch, [transient("unexpected_error")] * 2, capsys
    )
    assert code == rc.EXIT_EXTERNAL and len(calls) == 2
    assert out["facts"]["supervisor"]["resumes"] == 1
    assert out["errors"][0]["code"] == "unexpected_error"
    done = (0, {"ok": True, "state": "measured", "next": "decide", "errors": []})
    (tmp_path / "b").mkdir()
    _, out, calls = supervised(
        tmp_path / "b", monkeypatch,
        [transient("unexpected_error"), transient("kubectl_failed"), transient("unexpected_error"),
         done],
        capsys,
    )  # fmt: skip
    assert out["state"] == "measured" and len(calls) == 4  # not in a row: each retried


def test_an_attempt_killed_by_a_signal_is_not_restarted(tmp_path: Path, monkeypatch: Any,
                                                       capsys: Any) -> None:
    # kill of the attempt's pid, or the system's out-of-memory killer: no result, never resumed
    code, out, calls = supervised(tmp_path, monkeypatch, [(-9, {})], capsys)
    assert len(calls) == 1 and code == rc.EXIT_PRECONDITION
    assert out["state"] == "stopped" and out["errors"][0]["code"] == "stopped"
    assert not (tmp_path / "r" / rc.LOCK_NAME).exists()


def test_stopping_the_supervisor_stops_its_attempt_and_frees_the_folder(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    path = campaign(tmp_path)
    attempt: list[subprocess.Popen[bytes]] = []

    started, stopped = tmp_path / "grandchild.started", tmp_path / "grandchild.stopped"
    # the attempt's own call in flight acknowledges the stop itself: once its parent is gone it
    # is reaped by whatever PID 1 the test runs under (maybe never, in a container), so its pid
    # alone cannot tell a stopped process from a running one
    grandchild = (
        "import signal, sys, time\n"
        f"def stop(*_):\n    open({str(stopped)!r}, 'w').write('1'); sys.exit(0)\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        f"open({str(started)!r}, 'w').write('1')\n"
        "time.sleep(60)\n"
    )

    def fake_child(cmd: list[str], env: dict[str, str]) -> tuple[int, str]:
        # an attempt with a kubectl call of its own in flight, in its own group like run_child's
        code = (
            "import subprocess, sys, time; "
            f"subprocess.Popen([sys.executable, '-c', {grandchild!r}]); time.sleep(60)"
        )
        proc = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
        attempt.append(proc)
        deadline = time.monotonic() + 30
        while not started.exists():  # its handler is in place before it says it started
            assert time.monotonic() < deadline, "the attempt's own subprocess never started"
            time.sleep(0.05)
        rc.RUNNING["child"] = proc
        os.kill(os.getpid(), signal.SIGTERM)  # `kill SUPERVISOR_PID` while the attempt runs
        proc.wait(timeout=30)
        rc.RUNNING["child"] = None
        return proc.returncode, ""

    monkeypatch.setattr(rc, "run_child", fake_child)
    monkeypatch.setattr(rc, "target_reachable", lambda c, d: True)
    old = signal.getsignal(signal.SIGTERM)
    try:
        try:
            rc.main([str(path), "--out", str(tmp_path / "r"), *approve(path), "--supervise"])
        except SystemExit as exc:
            code = int(exc.code or 0)
        out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert code == rc.EXIT_PRECONDITION and out["state"] == "stopped"
        assert attempt[0].returncode == -signal.SIGTERM  # the attempt was stopped with it
        for _ in range(100):  # and so was the call it had in flight: it says so itself
            if stopped.exists():
                break
            time.sleep(0.1)
        else:
            raise AssertionError("the attempt's own subprocess never got the stop")
        assert not (tmp_path / "r" / rc.LOCK_NAME).exists()  # a rerun can start
        assert signal.getsignal(signal.SIGTERM) == old  # the handler is put back
    finally:
        # if the stop missed anything, never leave the attempt's group running after the test;
        # this runs after the assertions, so it cannot make a missed stop pass
        for proc in attempt:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(proc.pid, signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=5)


def test_supervise_gives_up_when_the_target_stays_away(tmp_path: Path, monkeypatch: Any,
                                                       capsys: Any) -> None:
    path = campaign(tmp_path)
    answers = iter([True])  # answers once at the start, then never again
    calls: list[Any] = []
    monkeypatch.setattr(rc, "target_reachable", lambda c, d: next(answers, False))
    monkeypatch.setattr(rc, "run_child", lambda cmd, env: calls.append(cmd) or (0, "{}"))
    monkeypatch.setattr(rc.time, "sleep", lambda s: None)
    try:
        rc.main([str(path), "--out", str(tmp_path / "r"), *approve(path), "--supervise",
                 "--max-wait-hours", "0.05"])  # fmt: skip
    except SystemExit as exc:
        code = int(exc.code or 0)
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert code == rc.EXIT_EXTERNAL and out["errors"][0]["code"] == "target_unreachable"
    assert calls == [] and not (tmp_path / "r" / rc.LOCK_NAME).exists()


def test_supervisor_log_lines_are_scrubbed(tmp_path: Path, monkeypatch: Any,
                                           capsys: Any) -> None:
    monkeypatch.setattr(rc.SCRUB, "text", lambda s, **k: s.replace("supervisor", "<scrubbed>"))
    done = (0, {"ok": True, "state": "measured", "next": "decide", "errors": []})
    supervised(tmp_path, monkeypatch, [done], capsys)
    log = (tmp_path / "r" / "campaign.log").read_text()
    assert "<scrubbed>" in log and "supervisor" not in log


def test_supervise_fails_fast_when_the_target_never_answered(tmp_path: Path, monkeypatch: Any,
                                                             capsys: Any) -> None:
    path = campaign(tmp_path)
    calls: list[Any] = []
    monkeypatch.setattr(rc, "target_reachable", lambda c, d: False)
    monkeypatch.setattr(rc, "run_child", lambda cmd, env: calls.append(cmd) or (0, "{}"))
    try:
        rc.main([str(path), "--out", str(tmp_path / "r"), *approve(path), "--supervise"])
    except SystemExit as exc:
        code = int(exc.code or 0)
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert code == rc.EXIT_EXTERNAL and out["errors"][0]["code"] == "target_unreachable"
    assert calls == [] and not (tmp_path / "r" / rc.LOCK_NAME).exists()


def test_a_resumed_removal_waits_for_pods_that_outlived_their_nimservice(monkeypatch: Any) -> None:
    import pytest

    camp = object.__new__(rc.Campaign)
    camp.log = lambda m: None
    left = [[{"metadata": {"name": "p-x-0"}}]] * 3 + [[]]  # pods still terminating, then gone

    class T:
        def remove(self, name: str, uid: str | None = None) -> bool:
            return False  # the NIMService is already gone, so nothing to delete

        def exists(self, name: str) -> bool:
            return False

        def replicas(self, name: str) -> list[dict[str, Any]]:
            return left.pop(0) if len(left) > 1 else left[0]

    camp.t = T()
    monkeypatch.setattr(rc.time, "sleep", lambda s: None)
    camp.remove("p-x")  # returns only once the pods are gone
    assert left == [[]]
    left[:] = [[{"metadata": {"name": "p-x-0"}}]]  # they never go: never taken for removed
    with pytest.raises(rc.KubeError, match="still running"):
        camp.remove("p-x")


def test_a_refused_removal_is_never_a_confirmed_one(tmp_path: Path) -> None:
    # a same-named resource of someone else replaced ours: never touched, and the marker stays
    # (this campaign's replicas may still hold GPUs)
    camp = object.__new__(rc.Campaign)
    camp.log = lambda m: None
    camp.c = {"name_prefix": "p", "target": "docker", "docker": {}}
    camp.save = lambda: None

    class T:
        what = "container"

        def remove(self, name: str, uid: str | None = None) -> bool:
            return False

        def exists(self, name: str) -> bool:
            return True

        def ident(self, name: str) -> str:
            return "id-1"  # read as the recorded one, then refused at removal

        def replicas(self, name: str) -> list[dict[str, Any]]:
            return []

        def docker(self, *a: Any, **k: Any) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess([], 0, "daemon-1\n", "")  # docker info: its ID

    camp.t = T()
    st = {"status": "measured", "cleanup_pending": "p-x", "cleanup_target": "docker:local#daemon-1",
          "cleanup_uid": "id-1"}  # fmt: skip
    with pytest.raises(rc.ForeignCleanupError):  # its fix never says to change name_prefix
        camp.clear_one("x", st)
    assert st["cleanup_pending"] == "p-x"


def test_an_earlier_runs_engine_flags_file_never_decides_a_new_plan(tmp_path: Path) -> None:
    camp = object.__new__(rc.Campaign)
    camp.flags_path, camp.state = tmp_path / "engine_flags.txt", {}
    camp.adopt_engine_flags("  --max-num-seqs N\n")  # a run approved with --engine-flags
    assert camp.flags_path.read_text() == "  --max-num-seqs N\n"
    assert camp.state["engine_flags_source"] == "file"
    assert not camp.missing_engine_help()
    camp.adopt_engine_flags(None)  # a new plan without it: refused in main, nothing changed
    assert camp.state["engine_flags_source"] == "file" and camp.missing_engine_help()
    camp.flags_path.write_text("probed help text")  # flags this campaign probed itself
    camp.state["engine_flags_source"] = "probed"
    camp.adopt_engine_flags(None)  # a resume keeps them
    assert camp.flags_path.read_text() == "probed help text" and not camp.missing_engine_help()
    camp.flags_path.unlink()  # the probed copy is gone: candidates it checked are not rechecked
    assert camp.missing_engine_help()
    camp.flags_path.write_text("probed help text")
    camp.state.pop("engine_flags_source")  # a file from before sources were recorded
    camp.adopt_engine_flags(None)
    assert not camp.flags_path.exists()  # its origin is unknown: probed again
    assert (tmp_path / "engine_flags.unrecorded.txt").read_text() == "probed help text"  # kept
    assert not camp.missing_engine_help()


def test_a_run_without_the_engine_help_earlier_runs_used_is_refused(tmp_path: Path) -> None:
    path = campaign(tmp_path)
    out = tmp_path / "r"
    out.mkdir()
    state = {"active_seconds": 0, "candidates": {}, "engine_flags_source": "file"}
    (out / "campaign_state.json").write_text(json.dumps(state))
    code, res = run(tmp_path, str(path), "--out", str(out), *approve(path))
    assert code == rc.EXIT_REFUSED and [e["code"] for e in res["errors"]] == ["engine_help_missing"]
    assert "--engine-flags" in res["errors"][0]["fix"]


def test_the_approved_engine_flags_file_in_out_is_never_rewritten(
    tmp_path: Path, monkeypatch: Any
) -> None:
    # --engine-flags OUT/engine_flags.txt: a scrubbed rewrite would change the bytes the plan
    # id hashed, and the same approved command would be refused on the next run
    monkeypatch.setattr(rc.SCRUB, "values", {"num-seqs"})
    camp = object.__new__(rc.Campaign)
    camp.flags_path, camp.state = tmp_path / "engine_flags.txt", {}
    camp.flags_path.write_text("  --max-num-seqs N\n")
    camp.adopt_engine_flags("  --max-num-seqs N\n", str(camp.flags_path))
    assert camp.flags_path.read_text() == "  --max-num-seqs N\n"
    assert camp.state["engine_flags_source"] == "file"


def test_a_saved_run_without_the_target_metric_is_measured_again(tmp_path: Path) -> None:
    def console(rows: list[tuple[str, float]]) -> str:
        # columns: name, avg, min, max, p99, ...; p99 is what ttft_p99 reads
        cells = ["│" + "│".join([n, f"{v:,.2f}", "1", "2", f"{v:,.2f}", "1", "1", "0"]) + "│"
                 for n, v in rows]  # fmt: skip
        return "\n".join(["NVIDIA AIPerf | LLM Metrics", *cells, "└--"]) + "\n"

    camp = object.__new__(rc.Campaign)
    camp.out, camp.gate = tmp_path, lambda *a, **k: None
    camp.c = {"sla": {"metric": "ttft_p99", "max_ms": 1000}}
    camp.w = {"repeats": 1}
    d = tmp_path / "x" / "c8-r1"
    d.mkdir(parents=True)
    # throughput and a clean error rate, but no TTFT p99: only the target metric is missing
    (d / "aiperf.log").write_text(console([("Output Token Throughput (tokens/sec)", 900.0),
                                           ("Request Error Rate (%)", 0.0)]))  # fmt: skip
    runs: list[int] = []

    def measure(cid: str, name: str, conc: int, rep: int) -> bool:
        if rc.run_ok(d):
            return True  # the real cache check: a saved run that counts is reused
        runs.append(conc)
        (d / "aiperf.log").write_text(console([("Time to First Token (ms)", 50.0),
                                               ("Output Token Throughput (tokens/sec)", 900.0),
                                               ("Request Error Rate (%)", 0.0)]))  # fmt: skip
        return True

    camp.measure = measure
    assert camp.point_passes("x", "n", 8) is True and runs == [8]  # measured again, now decides
    assert (d / "aiperf.log.incomplete").is_file()  # the incomplete one is kept as evidence


def test_a_supplied_engine_flags_file_is_saved_scrubbed(tmp_path: Path) -> None:
    secret = "nvapi-" + fake_secret()
    camp = object.__new__(rc.Campaign)
    camp.flags_path, camp.state = tmp_path / "engine_flags.txt", {}
    camp.adopt_engine_flags(f"  --max-num-seqs N\n  # captured with NGC_API_KEY={secret}\n")
    saved = camp.flags_path.read_text()
    assert "--max-num-seqs" in saved and secret not in saved


def test_a_non_finite_target_metric_is_unmeasured_in_the_search(
    tmp_path: Path, monkeypatch: Any
) -> None:
    # a truncated row reads as NaN: max([50, nan, 50]) is 50, so it could pass the target
    camp = object.__new__(rc.Campaign)
    camp.out, camp.gate = tmp_path, lambda *a, **k: None
    camp.c = {"sla": {"metric": "ttft_p99", "max_ms": 1000}}
    camp.w = {"repeats": 3}
    values = {1: 50.0, 2: float("nan"), 3: 50.0}
    measured: list[int] = []

    def measure(cid: str, name: str, conc: int, rep: int) -> bool:
        measured.append(rep)
        d = tmp_path / cid / f"c{conc}-r{rep}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "profile_export_aiperf.csv").write_text("x")
        return True

    camp.measure = measure
    monkeypatch.setattr(
        rc,
        "run_results",
        lambda d: {"error_rate": 0.0, "otps_avg": 900.0, "ttft_p99": values[int(d.name[-1])]},
    )
    assert camp.point_passes("x", "n", 8) is None  # never a pass on a NaN
    values[2] = float("inf")
    assert camp.point_passes("x", "n", 8) is None
    values[2] = 50.0
    assert camp.point_passes("x", "n", 8) is True


def test_a_fixed_load_run_without_the_target_metric_is_not_counted(
    tmp_path: Path, monkeypatch: Any
) -> None:
    camp = object.__new__(rc.Campaign)
    camp.out = tmp_path
    camp.c = {"name": "me", "sla": {"metric": "ttft_p99", "max_ms": 1000}}
    camp.w = {"model": "m", "tokenizer": "t", "endpoint_type": "chat", "extra_inputs": []}
    camp.salt_ok = False
    camp.workload_args = lambda: []
    camp.check_client = lambda: None
    camp.log = lambda msg: None
    camp.restarts = lambda name: 0
    camp.write = lambda path, text: None
    camp.t = type(
        "T", (), {"client_exec": lambda *a, **k: subprocess.CompletedProcess([], 1, "", "")}
    )()
    runs: list[int] = []
    camp.run_detached = lambda cmd, remote: runs.append(1) or (0, "ok")
    monkeypatch.setattr(rc.compare_aiperf, "read_log", lambda p: {"error_rate": 0.0})
    # every request succeeded, but no TTFT p99
    got: dict[str, float] = {"error_rate": 0.0, "otps_avg": 900.0}
    monkeypatch.setattr(rc, "run_results", lambda d: dict(got))
    d = tmp_path / "x" / "c8-r1"
    d.mkdir(parents=True)
    (d / "aiperf.log").write_text("saved")
    # the saved run is set aside and measured again; both attempts lack it: a failed run
    assert camp.measure("x", "n", 8, 1) is False and runs == [1, 1]
    assert (d / "aiperf.log.incomplete").is_file() and camp.requests_failed is False
    got["ttft_p99"] = 50.0
    assert camp.measure("x", "n", 8, 1) is True and runs == [1, 1]  # a complete run is reused
    del got["ttft_p99"]
    camp.c = {"name": "me"}  # without a target, a clean run counts as before
    assert camp.measure("x", "n", 8, 1) is True and runs == [1, 1]


def test_the_scrubber_reads_credentials_by_name_and_never_lists_the_environment(
    monkeypatch: Any,
) -> None:
    known, named, other = fake_secret(), fake_secret(), fake_secret()
    monkeypatch.setenv("NGC_API_KEY", known)  # a well-known credential variable
    monkeypatch.setenv("MY_PASS_FROM_HOST", named)  # one a campaign passes from the host
    monkeypatch.setenv("UNRELATED_TOKEN", other)  # neither: the runner never reads it
    sc = rc.targets.Scrubber()
    sc.add_env(["MY_PASS_FROM_HOST"])
    out = sc.text(f"{known} {named} {other}")
    assert known not in out and named not in out
    assert other in out  # by design: only named variables are read, the environment is not listed


def test_the_supervisor_leaves_the_environment_as_it_found_it(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    monkeypatch.delenv(rc.SUPERVISOR_ENV, raising=False)
    done = (0, {"ok": True, "state": "measured", "next": "decide", "errors": []})
    supervised(tmp_path, monkeypatch, [done], capsys)
    assert rc.SUPERVISOR_ENV not in os.environ  # set only while its attempts run


def test_a_named_credential_is_masked_at_any_length(monkeypatch: Any) -> None:
    seven, five = fake_secret()[:7], fake_secret()[:5]
    monkeypatch.setenv("SHORT_PASS_A", seven)
    monkeypatch.setenv("SHORT_PASS_B", five)
    monkeypatch.delenv("UNSET_PASS", raising=False)
    sc = rc.targets.Scrubber()
    before = set(sc.values)
    sc.add_env(["UNSET_PASS"])  # unset: nothing registered, so output is never corrupted
    assert sc.values == before
    sc.add_env(["SHORT_PASS_A", "SHORT_PASS_B"])
    out = sc.text(f"a={seven} b={five} plain text")
    assert seven not in out and five not in out and out.endswith("plain text")


def test_a_test_subprocess_never_sees_the_developers_home() -> None:
    from testenv import base_env

    proc = subprocess.run(
        [sys.executable, "-c", "import os; print(os.environ.get('HOME', ''))"],
        capture_output=True, text=True, env=base_env(), check=True, timeout=60,
    )  # fmt: skip
    assert proc.stdout.strip() and proc.stdout.strip() != os.path.expanduser("~")


def test_a_short_named_credential_never_breaks_a_flag_or_the_json_keys(monkeypatch: Any) -> None:
    opaque = fake_secret()
    # named credentials derived from the names they overlap, never written as literals
    part_of_flag, part_of_keys, a_field = "api-key".split("-")[1], "ok"[:1], "errors"
    monkeypatch.setenv("SHORT_PASS", part_of_flag)
    monkeypatch.setenv("TINY_PASS", part_of_keys)
    monkeypatch.setenv("FIELD_PASS", a_field)  # equals a documented field name
    sc = rc.targets.Scrubber()
    sc.add_env(["SHORT_PASS", "TINY_PASS", "FIELD_PASS"])
    out = sc.text(f"--api-key {opaque}")
    assert opaque not in out  # the flag is still recognized, so the value after it is masked
    message = f"{part_of_flag} {part_of_keys} {a_field}"
    data = json.loads(sc.dumps({"ok": False, "errors": [{"message": message}]}))
    assert set(data) == {"ok", "errors"} and set(data["errors"][0]) == {"message"}  # intact
    masked = data["errors"][0]["message"]
    assert part_of_flag not in masked and a_field not in masked and part_of_keys not in masked


def aiperf_console(rows: list[tuple[str, float]]) -> str:
    """An AIPerf console 'LLM Metrics' table with these rows (avg and p99 set to the value)."""
    cells = ["│" + "│".join([n, f"{v:,.2f}", "1", "2", f"{v:,.2f}", "1", "1", "0"]) + "│"
             for n, v in rows]  # fmt: skip
    return "\n".join(["NVIDIA AIPerf | LLM Metrics", *cells, "└--"]) + "\n"


def measuring_campaign(tmp_path: Path, csv: str = "") -> Any:
    """A Campaign with only what measure() needs; the client returns CSV for the export."""
    camp = object.__new__(rc.Campaign)
    camp.out, camp.c = tmp_path, {"name": "me"}
    camp.w = {"model": "m", "tokenizer": "t", "endpoint_type": "chat", "extra_inputs": []}
    camp.salt_ok = False
    camp.requests_failed = False
    camp.workload_args = lambda: []
    camp.check_client = lambda: None
    camp.log = lambda msg: None
    camp.restarts = lambda name: {"pod-a": 0}
    camp.write = lambda path, text: path.write_text(text)
    rc_ = 0 if csv else 1
    camp.t = type(
        "T", (), {"client_exec": lambda *a, **k: subprocess.CompletedProcess([], rc_, csv, "")}
    )()
    return camp


CLEAN = [
    ("Time to First Token (ms)", 50.0),
    ("Output Token Throughput (tokens/sec)", 900.0),
    ("Request Error Rate (%)", 0.0),
]


def test_a_run_without_throughput_does_not_count(tmp_path: Path) -> None:
    # compare_aiperf ranks every case by throughput and needs it from every counted repeat: a
    # repeat without it would leave a gap no rerun measures
    d = tmp_path / "x" / "c8-r1"
    d.mkdir(parents=True)
    (d / "aiperf.log").write_text(aiperf_console([CLEAN[0], CLEAN[2]]))
    assert rc.run_ok(d) and not rc.run_counts(d, None) and not rc.run_counts(d, "ttft_p99")
    (d / "aiperf.log").write_text(aiperf_console(CLEAN))
    assert rc.run_counts(d, None) and rc.run_counts(d, "ttft_p99")
    # measure() sets the saved run aside and measures it again
    (d / "aiperf.log").write_text(aiperf_console([CLEAN[0], CLEAN[2]]))
    camp = measuring_campaign(tmp_path)
    runs: list[int] = []
    camp.run_detached = lambda cmd, remote: runs.append(1) or (0, aiperf_console(CLEAN))
    assert camp.measure("x", "n", 8, 1) is True and runs == [1]
    assert (d / "aiperf.log.incomplete").is_file()


def test_only_the_validated_attempt_counts(tmp_path: Path) -> None:
    # an export from an earlier attempt (metrics, no error rate) must not decide a new attempt
    # whose own export could not be read back
    d = tmp_path / "x" / "c8-r1"
    d.mkdir(parents=True)
    stale = "Metric,avg,p99\nOutput Token Throughput (tokens/sec),900,900\n"
    (d / "profile_export_aiperf.csv").write_text(stale)
    camp = measuring_campaign(tmp_path)  # the CSV download fails
    runs: list[int] = []
    camp.run_detached = lambda cmd, remote: runs.append(1) or (0, aiperf_console(CLEAN))
    assert camp.measure("x", "n", 8, 1) is True and runs == [1]
    assert (d / "profile_export_aiperf.csv.incomplete").read_text() == stale
    # a replica restarted during the run: its console is never saved where a resume reuses it
    e = tmp_path / "x" / "c8-r2"
    snaps = iter([{"pod-a": 0}, {"pod-a": 1}])
    camp.restarts = lambda name: next(snaps)
    assert camp.measure("x", "n", 8, 2) is None
    assert not (e / "aiperf.log").exists() and not rc.run_counts(e, None)
    assert (e / "aiperf.restarted.log").is_file()


def test_runs_count_only_on_the_replicas_whose_engagement_was_checked(tmp_path: Path) -> None:
    camp = measuring_campaign(tmp_path)
    runs: list[int] = []
    camp.run_detached = lambda cmd, remote: runs.append(1) or (0, aiperf_console(CLEAN))
    camp.engaged = {"n": {"pod-old": 0}}  # replaced after engagement, before the first run
    assert camp.measure("x", "n", 8, 1) is None and runs == []
    camp.engaged = {"n": {"pod-a": 0}}  # the engaged replica serves the run
    assert camp.measure("x", "n", 8, 1) is True and runs == [1]


def test_an_overloaded_runs_export_is_kept_for_the_comparison(tmp_path: Path) -> None:
    # the console has no error row; only the export shows the failed requests
    over = (
        "Metric,avg,p99\n"
        "Output Token Throughput (tokens/sec),900,900\n"
        "Request Error Rate (%),5\n"
    )
    camp = measuring_campaign(tmp_path, csv=over)
    camp.run_detached = lambda cmd, remote: (0, aiperf_console(CLEAN[:2]))
    assert camp.measure("x", "n", 8, 1) is False and camp.requests_failed is True
    rejected: list[str] = []
    no_rate: list[str] = []
    rc.compare_aiperf.collect(tmp_path, rejected, no_rate)
    assert rejected == ["x/c8-r1"] and no_rate == []  # a measured overload, not missing evidence
    # attempt 1 overloaded, attempt 2's export could not be read: attempt 1's never stands in
    d = tmp_path / "x" / "c8-r2"
    exports = iter([over, ""])

    def fetch(*a: Any, **k: Any) -> subprocess.CompletedProcess[str]:
        text = next(exports)
        return subprocess.CompletedProcess([], 0 if text else 1, text, "")

    camp.t = type("T", (), {"client_exec": lambda self, *a, **k: fetch()})()
    assert camp.measure("x", "n", 8, 2) is False and camp.requests_failed is False
    assert not (d / "profile_export_aiperf.csv").exists()
    assert (d / "profile_export_aiperf.attempt1.csv").read_text() == over


def test_a_pending_removal_is_bound_to_the_target_it_was_left_on() -> None:
    camp = object.__new__(rc.Campaign)
    camp.c = {"name_prefix": "p", "context": "ctx", "namespace": "ns-b"}
    camp.log = lambda m: None
    camp.save = lambda: None
    removed: list[str] = []
    camp.remove = lambda name, uid=None: removed.append(name)
    camp.t = type("T", (), {"replicas": lambda self, n: [], "exists": lambda self, n: True,
                            "ident": lambda self, n: "u1"})()  # fmt: skip
    left = {"status": "measured", "cleanup_pending": "p-x", "cleanup_target": "kube:ctx/ns-a",
            "cleanup_uid": "u1"}  # fmt: skip
    camp.state = {"candidates": {"x": dict(left)}}
    with pytest.raises(rc.PendingCleanupError, match="ns-a"):
        camp.clear_pending()  # p-x may still hold GPUs in ns-a: never cleared from ns-b
    assert removed == [] and camp.state["candidates"]["x"] == left
    camp.c["namespace"] = "ns-a"  # back on the original target: it is removed there
    camp.clear_pending()
    assert removed == ["p-x"] and camp.state["candidates"]["x"] == {"status": "measured"}


class _Held:
    """A target whose name `p-x` resolves to `now` (None: nothing), whose recorded resource is
    `kept` under any name, and with `strays` pods under the name."""

    what = "NIMService"

    def __init__(self, now: str | None, kept: bool = False, strays: bool = False) -> None:
        self.now, self.is_kept, self.has_strays = now, kept, strays
        self.removed: list[tuple[str, str | None]] = []

    def ident(self, name: str) -> str | None:
        return self.now

    def exists(self, name: str) -> bool:
        return self.now is not None

    def kept(self, name: str, uid: str) -> bool:
        return self.is_kept or (bool(uid) and self.now == uid)

    def strays(self, name: str) -> bool:
        return self.has_strays

    def replicas(self, name: str) -> list[dict[str, Any]]:
        return [{"metadata": {"name": "p-x-0"}}] if self.has_strays else []

    def remove(self, name: str, uid: str | None = None) -> bool:
        if self.now is None or (uid is not None and uid != self.now):
            return False
        self.removed.append((name, uid))
        self.now = None
        return True


def held_campaign(t: _Held) -> Any:
    camp = object.__new__(rc.Campaign)
    camp.c = {"name": "me", "name_prefix": "p", "context": "ctx", "namespace": "ns"}
    camp.log = lambda m: None
    camp.save = lambda: None
    camp.t = t
    return camp


def test_a_resumed_removal_removes_only_the_resource_recorded_with_the_mark() -> None:
    left = {"status": "measured", "cleanup_pending": "p-x", "cleanup_target": "kube:ctx/ns",
            "cleanup_uid": "u1"}  # fmt: skip
    # the name still holds the recorded resource: removed, bound to that uid
    t = _Held("u1")
    st = dict(left)
    held_campaign(t).clear_one("x", st)
    assert t.removed == [("p-x", "u1")] and st == {"status": "measured"}
    # the recorded one is gone and another run (perhaps another user's, with the same campaign
    # name) made a same-named one: never removed; nothing of the old one runs, so cleared
    t = _Held("u2")
    st = dict(left)
    held_campaign(t).clear_one("x", st)
    assert t.removed == [] and st == {"status": "measured"}
    # pods under the name may still be the old one's: never removed, the run stops
    t = _Held("u2", strays=True)
    st = dict(left)
    with pytest.raises(rc.PendingCleanupError) as exc:
        held_campaign(t).clear_one("x", st)
    assert t.removed == [] and st == left and "cleanup_uid" in (exc.value.fix or "")
    # the recorded container lives on under another name (docker rename): the run stops
    t = _Held(None, kept=True)
    st = dict(left)
    with pytest.raises(rc.PendingCleanupError):
        held_campaign(t).clear_one("x", st)
    assert t.removed == [] and st == left
    # nothing was there when it was marked, and something is now: never removed
    t = _Held("u2")
    st = {**left, "cleanup_uid": ""}
    with pytest.raises(rc.PendingCleanupError):
        held_campaign(t).clear_one("x", st)
    assert t.removed == [] and st["cleanup_pending"] == "p-x"
    # gone, as recorded: nothing to remove, the mark is cleared
    t = _Held(None)
    st = dict(left)
    held_campaign(t).clear_one("x", st)
    assert t.removed == [] and st == {"status": "measured"}


def test_a_mark_without_a_recorded_identity_is_never_removed_by_the_runner() -> None:
    # an earlier runner's mark, or one whose resource could not be read: the resource under the
    # name may be another run's with the same campaign name, so the user decides
    for uid in ({}, {"cleanup_uid": None}):
        t = _Held("u2")
        st = {"status": "measured", "cleanup_pending": "p-x", "cleanup_target": "kube:ctx/ns",
              **uid}  # fmt: skip
        with pytest.raises(rc.PendingCleanupError) as exc:
            held_campaign(t).clear_one("x", st)
        assert t.removed == [] and st["cleanup_pending"] == "p-x"
        assert f"{rc.LABEL}=me" in (exc.value.fix or "") and "cleanup_uid" in (exc.value.fix or "")
        t = _Held(None)  # nothing under the name and no pods: cleared, nothing removed
        held_campaign(t).clear_one("x", st)
        assert t.removed == [] and st == {"status": "measured"}


def test_the_identity_is_recorded_before_the_mark_is_saved_and_kept_by_retire() -> None:
    t = _Held("u1")
    camp = held_campaign(t)
    saved: list[dict[str, Any]] = []
    camp.state = {"candidates": {"x": {"status": "measured", "deployed_uid": "u1"}}}
    camp.save = lambda: saved.append(json.loads(json.dumps(camp.state)))
    camp.retire("p-x", camp.state["candidates"]["x"])
    assert saved[0]["candidates"]["x"]["cleanup_uid"] == "u1" and t.removed == [("p-x", "u1")]
    # fail() records it before its diagnostics; replaced before retire, the replacement is
    # neither recorded nor removed (the mark stays for a resume to decide)
    t = _Held("u1")
    camp = held_campaign(t)
    camp.out = None
    camp.state = {"candidates": {"x": {"status": "measured", "deployed_uid": "u1"}}}
    camp.cand_state = lambda cid: camp.state["candidates"][cid]

    def replaced(name: str) -> list[dict[str, Any]]:
        t.now = "u2"  # another run's, made while the logs were read
        return []

    t.replicas = replaced  # type: ignore[method-assign]
    with pytest.raises(rc.ForeignCleanupError):
        camp.fail("x", "p-x", ("config", "bad"))
    st = camp.state["candidates"]["x"]
    assert t.removed == [] and st["cleanup_uid"] == "u1" and st["cleanup_pending"] == "p-x"


def test_retire_and_fail_never_remove_a_same_named_replacement_of_the_deployed_one() -> None:
    # this run deployed u1; it was removed outside the run and a same-named resource with the
    # same campaign label (u2) made before the cleanup: the replacement is never deleted, the
    # mark records u1, and a resume clears it without touching u2
    def deployed_then_replaced() -> tuple[_Held, Any, dict[str, Any]]:
        t = _Held("u2")
        camp = held_campaign(t)
        camp.out = None
        camp.state = {"candidates": {"x": {"status": "measured", "deployed_uid": "u1"}}}
        camp.cand_state = lambda cid: camp.state["candidates"][cid]
        return t, camp, camp.state["candidates"]["x"]

    t, camp, st = deployed_then_replaced()
    with pytest.raises(rc.ForeignCleanupError):
        camp.retire("p-x", st)
    assert t.removed == [] and st["cleanup_uid"] == "u1" and st["cleanup_pending"] == "p-x"
    camp.clear_one("x", st)  # u1 is gone, u2 is not it: cleared, never removed
    assert t.removed == [] and t.now == "u2" and "cleanup_pending" not in st
    t, camp, st = deployed_then_replaced()
    with pytest.raises(rc.ForeignCleanupError):
        camp.fail("x", "p-x", ("intermittent", "smoke test failed"))
    assert t.removed == [] and st["cleanup_uid"] == "u1" and st["cleanup_pending"] == "p-x"
    # no identity recorded by a deploy (a state from before the field): nothing is removed, and
    # the mark fails closed like any unrecorded one
    for done in (lambda c, s: c.retire("p-x", s), lambda c, s: c.fail("x", "p-x", ("c", "r"))):
        t, camp, st = deployed_then_replaced()
        del st["deployed_uid"]
        t.now = "u1"  # even this run's own: unrecorded, so a resume decides
        with pytest.raises(rc.ForeignCleanupError):
            done(camp, st)
        assert t.removed == [] and st["cleanup_uid"] is None and st["cleanup_pending"] == "p-x"
    # the deploy made nothing (""): nothing is removed, and nothing there is fine
    t, camp, st = deployed_then_replaced()
    st["deployed_uid"], t.now = "", None
    camp.retire("p-x", st)
    assert t.removed == [] and "cleanup_pending" not in st


def test_retire_keeps_the_mark_while_the_deployed_container_lives_on_renamed() -> None:
    # u1 was renamed away (Docker): nothing holds the name, but u1 still holds GPUs
    t = _Held(None, kept=True)
    camp = held_campaign(t)
    camp.state = {"candidates": {"x": {"status": "measured", "deployed_uid": "u1"}}}
    st = camp.state["candidates"]["x"]
    with pytest.raises(rc.PendingCleanupError):
        camp.retire("p-x", st)
    assert t.removed == [] and st["cleanup_pending"] == "p-x" and st["cleanup_uid"] == "u1"


def test_the_deployed_identity_is_saved_before_the_wait_and_bound_on_the_retry(
    tmp_path: Path, monkeypatch: Any
) -> None:
    # each attempt's own container is removed by the Id its deploy recorded, never by what the
    # name resolves to (ident() answers a replacement's Id here)
    camp = deployed_campaign(tmp_path, monkeypatch, uid="replacement")
    del camp.retire  # the real one
    saved: list[dict[str, Any]] = []
    camp.save = lambda: saved.append(json.loads(json.dumps(camp.state)))
    made = iter(["cid-1", "cid-2"])

    def deploy(cand: dict[str, Any]) -> None:
        camp.t.deployed = next(made)

    camp.t.deploy = deploy
    seen: list[Any] = []

    def wait_ready(name: str, n: int) -> tuple[str, str]:
        seen.append(saved[-1]["candidates"]["x"].get("deployed_uid"))  # already saved
        return ("intermittent", "not ready after 1s")

    camp.wait_ready = wait_ready
    removed: list[tuple[str, str | None]] = []
    camp.remove = lambda name, uid=None: removed.append((name, uid))
    camp.run_candidate({"id": "x", "gpus": 1, "replicas": 1, "args": [], "env": {}})
    assert seen == ["cid-1", "cid-2"]
    assert removed == [("p-x", "cid-1"), ("p-x", "cid-2")]
    st = camp.state["candidates"]["x"]
    assert st["status"] == "failed" and "cleanup_pending" not in st


def test_a_mark_that_is_not_the_recorded_resource_says_how_to_clear_it() -> None:
    left = {"status": "measured", "cleanup_pending": "p-x", "cleanup_target": "kube:ctx/ns",
            "cleanup_uid": "u1"}  # fmt: skip
    for t, st in ((_Held("u2"), {**left, "cleanup_uid": ""}), (_Held(None, kept=True), left)):
        st = dict(st)
        with pytest.raises(rc.PendingCleanupError) as exc:
            held_campaign(t).clear_one("x", st)
        fix = exc.value.fix or ""
        assert "someone else's" in fix and "Leave p-x alone" in fix
        assert "cleanup_uid of x in campaign_state.json" in fix and t.removed == []
    assert "docker ps -a --filter id=u1" in fix and f"{rc.LABEL}=me" in fix


class _Deploying(Exception):
    """Raised where run_candidate would start deploying."""


def deciding_campaign(tmp_path: Path, monkeypatch: Any) -> Any:
    """A Campaign that runs run_candidate up to the deploy, then raises _Deploying."""
    camp = object.__new__(rc.Campaign)
    camp.c = {"name_prefix": "p", "name": "t1", "baseline": "b"}
    camp.out, camp.state = tmp_path, {"candidates": {}}
    camp.flags_path = tmp_path / "engine_flags.txt"
    camp.log = lambda m: None
    camp.save = lambda: None
    camp.fingerprint = lambda cand: "fp"

    def owner(self: Any, name: str) -> None:
        raise _Deploying

    camp.t = type("T", (), {"owner": owner})()
    monkeypatch.setattr(rc, "prune", lambda c, cand: None)
    return camp


def test_a_rejection_is_decided_again_when_the_engine_help_changes(
    tmp_path: Path, monkeypatch: Any
) -> None:
    camp = deciding_campaign(tmp_path, monkeypatch)
    cand = {"id": "x", "args": ["--new-flag", "1"], "env": {}}
    camp.adopt_engine_flags("  --max-num-seqs N\n")
    camp.run_candidate(cand)  # the approved help does not list it
    st = camp.state["candidates"]["x"]
    assert st["status"] == "rejected" and st["flags_sha256"]
    camp.run_candidate(cand)  # the same help: still rejected, not deployed
    assert st["status"] == "rejected"
    camp.adopt_engine_flags("  --max-num-seqs N\n  --new-flag N\n")  # a new plan lists it
    with pytest.raises(_Deploying):
        camp.run_candidate(cand)
    assert st["status"] == "pending" and "reason" not in st


def test_a_rejection_is_never_reopened_without_engine_help(
    tmp_path: Path, monkeypatch: Any
) -> None:
    camp = deciding_campaign(tmp_path, monkeypatch)
    camp.flags_text = None
    camp.state["candidates"]["x"] = {"status": "rejected", "flags_sha256": "old"}
    camp.run_candidate({"id": "x", "args": ["--new-flag", "1"], "env": {}})  # not deployed
    assert camp.state["candidates"]["x"]["status"] == "rejected"


def test_flags_are_checked_against_the_approved_help_not_its_scrubbed_copy(
    tmp_path: Path, monkeypatch: Any
) -> None:
    # a registered credential that overlaps a flag name masks it in engine_flags.txt
    monkeypatch.setattr(rc.SCRUB, "values", {"num-seqs"})
    camp = deciding_campaign(tmp_path, monkeypatch)
    camp.adopt_engine_flags("  --max-num-seqs N\n")
    assert "--max-num-seqs" not in camp.flags_path.read_text()
    with pytest.raises(_Deploying):  # the dry run listed it: the live run deploys it too
        camp.run_candidate({"id": "x", "args": ["--max-num-seqs", "8"], "env": {}})


def test_the_client_uid_is_the_one_read_when_it_was_created() -> None:
    # a same-named client replaced it between creation and the lookup: never recorded as ours
    camp = object.__new__(rc.Campaign)
    camp.state, camp.w, camp.facts = {}, {}, {}
    camp.save = lambda: None

    class T:
        created_uid = "mine"

        def ensure_client(self) -> bool:
            return True

        def client_uid(self) -> str:
            return "theirs"

        def client_exec(self, *a: Any, **k: Any) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess([], 0, "0.13.0\n", "")

    camp.t = T()
    camp.ensure_client()
    assert camp.state["client_uid"] == "mine" and camp.created_client is True


def test_an_unreadable_export_does_not_count_and_does_not_crash(tmp_path: Path) -> None:
    (tmp_path / rc.compare_aiperf.CSV_NAME).write_bytes(b"Metric,avg\n\xff\xfe,1\n")
    assert rc.run_results(tmp_path) == {} and rc.run_ok(tmp_path) is False


def export_csv(
    ttft: float | None = 50.0, itl: float | None = 8.0, otps: float | None = 900.0, err: str = "0"
) -> str:
    """An AIPerf CSV export; a None metric leaves its row out (AIPerf writes no ITL row for
    outputs of one token)."""
    rows = ["Metric,avg,min,max,sum,p1,p5,p10,p25,p50,p75,p90,p95,p99,std"]
    if ttft is not None:
        rows.append(f"Time to First Token (ms),{ttft},,,,,,,,,,,,{ttft * 2},")
    if itl is not None:
        rows.append(f"Inter Token Latency (ms),{itl},,,,,,,,,,,,{itl * 1.1},")
    rows += ["", "Metric,Value"]
    if otps is not None:
        rows.append(f"Output Token Throughput (tokens/sec),{otps}")
    rows.append(f"Request Error Rate (%),{err}")
    return "\n".join(rows) + "\n"


def put_run(root: Path, run: str, text: str) -> Path:
    d = root / run
    d.mkdir(parents=True, exist_ok=True)
    (d / rc.compare_aiperf.CSV_NAME).write_text(text)
    return d


def comparing(out: Path, states: dict[str, Any], loads: list[int]) -> dict[str, Any]:
    """rc.compare over OUT for a baseline `before` and a candidate `cand`, with a TTFT target."""
    c = {
        "baseline": "before",
        "candidates": [
            {"id": "before", "gpus": 1, "replicas": 1},
            {"id": "cand", "gpus": 1, "replicas": 1},
        ],
        "workload": {"concurrency": loads, "repeats": 3},
        "sla": {"metric": "ttft_p99", "max_ms": 1000},
    }
    return rc.compare(c, out, states)


def test_a_run_the_runner_counts_is_never_too_few_repeats_in_the_comparison(
    tmp_path: Path,
) -> None:
    # one rule for both: a run counts with throughput and the target's metric, whatever else
    # it lacks (no ITL row at all, a NaN TTFT average in one repeat); the comparison never asks
    # for a repeat the runner would reuse instead of measuring again
    variants = {
        "before": [export_csv(otps=900.0, itl=None)] * 3,
        "cand": [
            export_csv(otps=1800.0, itl=None),
            export_csv(otps=1800.0, itl=8.0),
            export_csv(otps=1800.0, ttft=float("nan"), itl=8.0).replace(
                ",,,,,,,,,,,,nan,", ",,,,,,,,,,,,60,"
            ),
        ],
    }
    for case, texts in variants.items():
        for r, text in enumerate(texts, 1):
            d = put_run(tmp_path, f"{case}/c8-r{r}", text)
            assert rc.run_counts(d, "ttft_p99") and rc.run_counts(d, None)
    states = {"before": {"status": "measured"}, "cand": {"status": "measured"}}
    res = comparing(tmp_path, states, [8])
    assert not [e for e in res.get("errors") or [] if e["code"] == "too_few_repeats"], res
    assert res["state"] == "compared" and res["facts"]["recommendation"]["provisional"] is False
    # without throughput the runner does not count the run (it is measured again), and the
    # comparison calls the same repeat missing
    d = put_run(tmp_path, "cand/c8-r3", export_csv(otps=None))
    assert rc.run_ok(d) and not rc.run_counts(d, "ttft_p99")
    res = comparing(tmp_path, states, [8])
    assert any("otps_avg" in e["message"] for e in res["errors"] if e["code"] == "too_few_repeats")
    assert rc.compare_aiperf.required_metrics("ttft_p99") == ("otps_avg", "ttft_p99")


def test_a_search_point_stopped_by_failed_requests_is_final_for_both_sides(
    tmp_path: Path,
) -> None:
    # repeat 1 counts, repeat 2 fails requests: the search reads a miss and records no failed
    # run; the comparison must read the same miss, or the case stays provisional with nothing
    # a rerun would measure
    for r in (1, 2, 3):
        put_run(tmp_path, f"before/c8-r{r}", export_csv(otps=900.0))
        put_run(tmp_path, f"cand/c8-r{r}", export_csv(otps=1800.0))
    camp = object.__new__(rc.Campaign)
    camp.out, camp.gate = tmp_path, lambda *a, **k: None
    camp.c = {"sla": {"metric": "ttft_p99", "max_ms": 1000}}
    camp.w = {"repeats": 3}
    reps: list[int] = []

    def measure(cid: str, name: str, conc: int, rep: int) -> bool:
        reps.append(rep)
        camp.requests_failed = rep == 2
        put_run(tmp_path, f"{cid}/c{conc}-r{rep}", export_csv(err="3.5" if rep == 2 else "0"))
        return rep != 2

    camp.measure = measure
    assert camp.point_passes("cand", "n", 16) is False and reps == [1, 2]
    states = {"before": {"status": "measured"}, "cand": {"status": "measured"}}
    res = comparing(tmp_path, states, [8])
    assert res["state"] == "compared", res["errors"]
    assert res["facts"]["sla_boundary"]["cand"]["first_failing"] == 16
    assert res["facts"]["recommendation"]["provisional"] is False


IMAGE_A = "nvcr.io/nim/x@sha256:" + "aa" * 32
IMAGE_B = "nvcr.io/nim/x@sha256:" + "bb" * 32


def pod_record(image_id: str) -> dict[str, Any]:
    """A saved pod record whose NIM container runs IMAGE_ID."""
    return {"spec": {"containers": [{"name": "nim"}]},
            "status": {"containerStatuses": [{"name": "nim", "imageID": image_id}]}}  # fmt: skip


def deployed_campaign(tmp_path: Path, monkeypatch: Any, uid: str = "old") -> Any:
    """A Campaign whose run_candidate deploys (stubbed) one replica with this uid, then measures
    with the real measure() at one load and three repeats."""
    camp = measuring_campaign(tmp_path)
    camp.c = {"name": "me", "name_prefix": "p", "baseline": "before", "target": "docker"}
    camp.w.update(concurrency=[8], repeats=3)
    camp.state = {"candidates": {}, "order": []}
    camp.flags_path = tmp_path / "engine_flags.txt"
    camp.flags_path.write_text("  --max-num-seqs N\n")
    camp.save = lambda: None
    camp.fingerprint = lambda cand: "fp"
    camp.gate = lambda *a, **k: None
    camp.smoke = lambda name, salt=False: True
    camp.counters = lambda name: {}
    camp.engagement = lambda cand, pod, spec, log: {"exit": 0, "state": "engaged"}
    camp.wait_ready = lambda name, n: None
    camp.write_manifest = lambda cand, st: None
    camp.measured_config = lambda cand: {}
    # removed at once: the cleanup marker goes, as after a real removal
    camp.retire = lambda name, st: st.pop("cleanup_pending", None)
    camp.facts = {}
    monkeypatch.setattr(rc, "prune", lambda c, cand: None)
    deploys: list[str] = []
    replica = {"metadata": {"name": "p-x-0", "uid": uid},
               "status": {"containerStatuses": [{"restartCount": 0}]}}  # fmt: skip

    class T:
        what = "container"
        deployed: str | None = None

        def owner(self, name: str) -> None:
            return None

        def kept(self, name: str, uid: str) -> bool:
            return False  # removed, never renamed away

        def ident(self, name: str) -> str:
            return "cid-" + uid

        def describe(self, cand: dict[str, Any]) -> dict[str, Any]:
            return {}

        def deploy(self, cand: dict[str, Any]) -> None:
            deploys.append(cand["id"])
            self.deployed = "cid-" + uid  # the container this deploy started

        def replicas(self, name: str) -> list[dict[str, Any]]:
            return [replica]

        def logs(self, pod: str, tail: int | None = None) -> str:
            return ""

        def spec(self, pod: str, allow: list[str]) -> dict[str, Any]:
            return pod_record(IMAGE_A)

        def client_exec(self, *a: Any, **k: Any) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess([], 1, "", "")  # no export: the console counts

        def docker(self, *a: Any, **k: Any) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess([], 0, "daemon-1\n", "")  # docker info: its ID

    camp.t = T()
    camp.deploys = deploys
    return camp


def test_a_restart_between_runs_is_intermittent_and_resumes(
    tmp_path: Path, monkeypatch: Any
) -> None:
    camp = deployed_campaign(tmp_path, monkeypatch)
    runs: list[int] = []
    camp.run_detached = lambda cmd, remote: runs.append(1) or (0, aiperf_console(CLEAN))
    # repeat 1 runs on the checked replica; then it is replaced before repeat 2 starts
    seen = iter([{"old": 0}, {"old": 0}, {"new": 0}])
    camp.restarts = lambda name: next(seen)
    cand = {"id": "x", "gpus": 1, "replicas": 1, "args": [], "env": {}}
    camp.run_candidate(cand)
    st = camp.state["candidates"]["x"]
    assert runs == [1] and rc.run_counts(tmp_path / "x" / "c8-r1", None)
    assert st["status"] == "failed" and st["kind"] == "intermittent"
    assert st["reason"] == "a replica restarted or was replaced after its engagement check"
    # the resume redeploys, checks engagement again, and reuses the counted repeat
    camp.restarts = lambda name: {"old": 0}
    camp.run_candidate(cand)
    assert camp.deploys == ["x", "x"] and runs == [1, 1, 1]
    assert st["status"] == "measured" and st["failed_runs"] == []


def test_a_restart_during_a_run_is_still_a_config_failure(
    tmp_path: Path, monkeypatch: Any
) -> None:
    camp = deployed_campaign(tmp_path, monkeypatch)
    camp.run_detached = lambda cmd, remote: (0, aiperf_console(CLEAN))
    seen = iter([{"old": 0}, {"old": 1}])  # restarted under load
    camp.restarts = lambda name: next(seen)
    camp.run_candidate({"id": "x", "gpus": 1, "replicas": 1, "args": [], "env": {}})
    st = camp.state["candidates"]["x"]
    assert st["status"] == "failed" and st["kind"] == "config"
    assert st["reason"] == "a replica restarted under load"


def test_a_measured_overload_is_not_measured_again(tmp_path: Path, monkeypatch: Any) -> None:
    camp = deciding_campaign(tmp_path, monkeypatch)
    camp.w = {"repeats": 3, "concurrency": [32]}
    whole = [f"c32-r{r}" for r in (1, 2, 3)]
    for r in whole:  # their exports show the failed requests compare_aiperf reads
        put_run(tmp_path, f"x/{r}", export_csv(err="4.5"))
    # every repeat at 32 failed requests: compare_aiperf reads a measured overload, so the
    # measured candidate is kept, never redeployed on each resume
    camp.state["candidates"]["x"] = {
        "status": "measured", "failed_runs": list(whole), "overloaded_runs": list(whole)
    }  # fmt: skip
    cand = {"id": "x", "args": [], "env": {}}
    camp.run_candidate(cand)
    assert camp.state["candidates"]["x"]["status"] == "measured"
    # one failed repeat next to counted ones is a stray failure: measured again
    camp.state["candidates"]["x"] = {
        "status": "measured", "failed_runs": ["c32-r2"], "overloaded_runs": ["c32-r2"]
    }  # fmt: skip
    with pytest.raises(_Deploying):
        camp.run_candidate(cand)
    assert camp.state["candidates"]["x"]["status"] == "pending"
    # a run that failed for another reason, or a search that did not finish: measured again
    for failed in (["c8-r1", *whole], ["sla-search", *whole]):
        camp.state["candidates"]["x"] = {
            "status": "measured", "failed_runs": failed, "overloaded_runs": list(whole)
        }  # fmt: skip
        with pytest.raises(_Deploying):
            camp.run_candidate(cand)


def test_a_resume_checks_flags_against_the_scrubbed_probed_copy_as_scrubbed(
    tmp_path: Path, monkeypatch: Any
) -> None:
    # the first run probed the help and saved it scrubbed; a credential overlapping a flag name
    # masked that name. The resume reads only the saved copy: the flag the first run accepted
    # must still be found, and a flag the engine never listed still rejected
    monkeypatch.setattr(rc.SCRUB, "values", {"num-seqs"})
    camp = deciding_campaign(tmp_path, monkeypatch)
    camp.flags_path.write_text(rc.SCRUB.text("  --max-num-seqs N\n"))
    assert "--max-num-seqs" not in camp.flags_path.read_text() and camp.flags_text is None
    with pytest.raises(_Deploying):
        camp.run_candidate({"id": "x", "args": ["--max-num-seqs", "8"], "env": {}})
    camp.run_candidate({"id": "y", "args": ["--not-a-flag", "1"], "env": {}})
    assert camp.state["candidates"]["y"]["status"] == "rejected"


def test_a_repair_pass_keeps_a_measured_overload(tmp_path: Path, monkeypatch: Any) -> None:
    # 8 has one failed repeat to measure again; every repeat at 32 failed requests (a measured
    # overload): the repair pass measures only 8's repeat and keeps 32 as it is
    camp = deployed_campaign(tmp_path, monkeypatch)
    camp.w["concurrency"] = [8, 32]
    camp.restarts = lambda name: {"old": 0}
    whole = [f"c32-r{r}" for r in (1, 2, 3)]
    for r in whole:
        put_run(tmp_path, f"x/{r}", export_csv(err="4.5"))
    for r in ("c8-r1", "c8-r3"):
        put_run(tmp_path, f"x/{r}", export_csv())
    camp.state["candidates"]["x"] = {
        "status": "measured", "failed_runs": ["c8-r2", *whole], "overloaded_runs": list(whole),
        "image_id": "sha256:" + "aa" * 32,
    }  # fmt: skip
    runs: list[str] = []
    camp.run_detached = lambda cmd, remote: runs.append(remote) or (0, aiperf_console(CLEAN))
    camp.run_candidate({"id": "x", "gpus": 1, "replicas": 1, "args": [], "env": {}})
    st = camp.state["candidates"]["x"]
    assert camp.deploys == ["x"] and [r.rsplit("/", 1)[-1] for r in runs] == ["c8-r2"]
    assert st["status"] == "measured"
    assert st["failed_runs"] == whole and st["overloaded_runs"] == whole
    assert all(rc.requests_failed(tmp_path / "x" / r) for r in whole)  # the evidence stays
    # an overload whose exports are gone is no evidence: measured again
    (tmp_path / "x" / "c32-r3" / rc.compare_aiperf.CSV_NAME).unlink()
    assert rc.kept_overloads(tmp_path / "x", st, 3) == set()


def test_a_stop_before_a_kept_overload_keeps_it_listed(tmp_path: Path, monkeypatch: Any) -> None:
    # the gate stops the repair pass before it reaches 32: the kept overload must stay listed,
    # or the next rerun would benchmark it again
    camp = deployed_campaign(tmp_path, monkeypatch)
    camp.w["concurrency"] = [8, 32]
    camp.restarts = lambda name: {"old": 0}
    whole = [f"c32-r{r}" for r in (1, 2, 3)]
    for r in whole:
        put_run(tmp_path, f"x/{r}", export_csv(err="4.5"))
    camp.state["candidates"]["x"] = {
        "status": "measured", "failed_runs": ["c8-r2", *whole], "overloaded_runs": list(whole),
        "image_id": "sha256:" + "aa" * 32,
    }  # fmt: skip
    camp.gate = lambda *a, **k: "time budget of 1 h spent"
    camp.run_candidate({"id": "x", "gpus": 1, "replicas": 1, "args": [], "env": {}})
    st = camp.state["candidates"]["x"]
    assert st["status"] == "stopped" and st["overloaded_runs"] == whole
    assert rc.kept_overloads(tmp_path / "x", st, 3) == set(whole)


def test_the_search_never_benchmarks_a_measured_overload_again(tmp_path: Path) -> None:
    # every fixed repeat at 32 failed requests; the search probes 32: a miss read from those
    # exports, never set aside and measured again
    camp = object.__new__(rc.Campaign)
    camp.out, camp.gate = tmp_path, lambda *a, **k: None
    camp.c = {"sla": {"metric": "ttft_p99", "max_ms": 1000}}
    camp.w = {"repeats": 3}
    for r in (1, 2, 3):
        put_run(tmp_path, f"x/c32-r{r}", export_csv(err="4.5"))

    def measure(cid: str, name: str, conc: int, rep: int) -> bool:
        raise AssertionError("measured again")

    camp.measure = measure
    assert camp.point_passes("x", "n", 32) is False
    assert all(rc.requests_failed(tmp_path / "x" / f"c32-r{r}") for r in (1, 2, 3))


def test_a_run_that_never_counts_is_repaired_a_bounded_number_of_times(
    tmp_path: Path, monkeypatch: Any
) -> None:
    camp = deciding_campaign(tmp_path, monkeypatch)
    camp.w = {"repeats": 3, "concurrency": [8]}
    camp.facts = {}
    for r in (1, 3):
        put_run(tmp_path, f"x/c8-r{r}", export_csv())
    put_run(tmp_path, "x/c8-r2", export_csv(otps=None))  # never counts
    st = {"status": "measured", "failed_runs": [], "overloaded_runs": []}
    camp.state["candidates"]["x"] = st
    cand = {"id": "x", "args": [], "env": {}}
    for _ in range(rc.MAX_REPAIRS):
        with pytest.raises(_Deploying):
            camp.run_candidate(cand)
        st["status"] = "measured"  # the repair pass measured it, still without throughput
    camp.run_candidate(cand)  # the cap: left as a gap, not redeployed again
    assert st["status"] == "measured" and st["repairs"] == rc.MAX_REPAIRS
    assert camp.facts["unrepaired"] == {"x": ["c8-r2"]} and st["unrepaired"] == ["c8-r2"]


def test_a_measured_candidate_with_a_run_that_does_not_count_is_measured_again(
    tmp_path: Path, monkeypatch: Any
) -> None:
    camp = deployed_campaign(tmp_path, monkeypatch)
    camp.restarts = lambda name: {"old": 0}
    for r in (1, 3):
        put_run(tmp_path, f"x/c8-r{r}", export_csv())
    # repeat 2 was saved without an error-rate row: it does not count
    put_run(tmp_path, "x/c8-r2", export_csv().replace("Request Error Rate (%),0\n", ""))
    camp.state["candidates"]["x"] = {"status": "measured", "failed_runs": [], "overloaded_runs": [],
                                     "image_id": "sha256:" + "aa" * 32}  # fmt: skip
    runs: list[str] = []
    camp.run_detached = lambda cmd, remote: runs.append(remote) or (0, aiperf_console(CLEAN))
    cand = {"id": "x", "gpus": 1, "replicas": 1, "args": [], "env": {}}
    camp.run_candidate(cand)
    assert camp.deploys == ["x"] and [r.rsplit("/", 1)[-1] for r in runs] == ["c8-r2"]
    assert camp.state["candidates"]["x"]["status"] == "measured"
    assert (tmp_path / "x" / "c8-r2" / f"{rc.compare_aiperf.CSV_NAME}.incomplete").is_file()
    camp.run_candidate(cand)  # every saved run counts now: skipped, never redeployed
    assert camp.deploys == ["x"]


def test_a_sidecar_restart_is_a_replica_restart(tmp_path: Path, monkeypatch: Any) -> None:
    # the engagement check can read a NIM in a restartable init container (a sidecar): its
    # restart between repeats, or during one, must not leave a run counted on an unchecked engine
    pod = {"metadata": {"uid": "u"},
           "status": {"containerStatuses": [{"restartCount": 0}],
                      "initContainerStatuses": [{"name": "nim", "restartCount": 0}]}}  # fmt: skip
    before = rc.replica_restarts([pod])
    pod["status"]["initContainerStatuses"][0]["restartCount"] = 1
    assert rc.replica_restarts([pod]) != before
    for during in (False, True):
        (tmp_path / str(during)).mkdir()
        camp = deployed_campaign(tmp_path / str(during), monkeypatch)
        del camp.restarts  # the real snapshot, read from the replica record
        side = {"name": "nim", "restartCount": 0}
        camp.t.replicas("p-x")[0]["status"]["initContainerStatuses"] = [side]

        def run(cmd: list[str], remote: str, during: bool = during, side: Any = side) -> Any:
            side["restartCount"] += during  # restarted under load
            return 0, aiperf_console(CLEAN)

        def write(path: Path, text: str, side: Any = side) -> None:
            path.write_text(text)
            if "c8-r1" in path.parts and path.name == "aiperf.log":
                side["restartCount"] += 1  # after repeat 1 was checked, before repeat 2

        camp.run_detached = run
        if not during:
            camp.write = write
        camp.run_candidate({"id": "x", "gpus": 1, "replicas": 1, "args": [], "env": {}})
        st = camp.state["candidates"]["x"]
        assert st["status"] == "failed"
        assert rc.run_counts(tmp_path / str(during) / "x" / "c8-r1", None) is not during
        if during:
            assert st["kind"] == "config" and st["reason"] == "a replica restarted under load"
        else:
            assert st["kind"] == "intermittent" and st["reason"] == rc.RESTARTED_SINCE_CHECK


def test_a_complete_comparison_with_a_measured_overload_ends_ok(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    # compare is ok with only run_had_request_errors (whole-load overloads): a reported fact,
    # never a reason for the run to end in fix_errors
    path = campaign(tmp_path)

    class Quiet(rc.Campaign):
        def ensure_client(self) -> None:
            pass

        def node_gpus(self) -> str:
            return "n/a"

        def release_client(self) -> None:
            pass

        def run_candidate(self, cand: dict[str, Any]) -> None:
            self.cand_state(cand["id"])["status"] = "measured"

    overload = {"code": "run_had_request_errors", "message": "before/c128-r1", "fix": "f"}
    result: dict[str, Any] = {"ok": True, "state": "compared", "facts": {}, "errors": [overload]}
    monkeypatch.setattr(rc, "Campaign", Quiet)
    monkeypatch.setattr(rc, "compare", lambda c, out, states: result)
    argv = [str(path), "--out", str(tmp_path / "r"), *approve(path)]
    with pytest.raises(SystemExit) as exc:
        rc.main(argv)
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert exc.value.code == 0 and out["ok"] and out["next"] == "decide" and out["errors"] == []
    assert out["facts"]["compare"]["notes"] == [overload]
    # an incomplete comparison still fails the run, with its errors
    result.update(ok=False, state="incomplete")
    rc.release_lock(tmp_path / "r" / rc.LOCK_NAME, os.getpid())  # the first run's, in-process
    with pytest.raises(SystemExit) as exc:
        rc.main(argv)
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert exc.value.code == rc.EXIT_PRECONDITION and out["errors"] == [overload]
    rc.release_lock(tmp_path / "r" / rc.LOCK_NAME, os.getpid())


def test_a_failed_candidate_stays_marked_for_removal_when_its_logs_cannot_be_read(
    tmp_path: Path,
) -> None:
    camp = object.__new__(rc.Campaign)
    camp.out, camp.log = tmp_path, lambda m: None
    camp.c = {"target": "docker", "docker": {}}
    camp.state = {"candidates": {}}
    saved: list[dict[str, Any]] = []
    camp.save = lambda: saved.append(json.loads(json.dumps(camp.state)))

    class T:
        def replicas(self, name: str) -> list[dict[str, Any]]:
            raise rc.KubeError("unreachable")

        def ident(self, name: str) -> str:
            raise rc.KubeError("unreachable")

        def docker(self, *a: Any, **k: Any) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess([], 0, "daemon-1\n", "")  # docker info: its ID

    camp.t = T()
    with pytest.raises(rc.KubeError):
        camp.fail("x", "p-x", ("config", "settings not engaged"))
    st = saved[-1]["candidates"]["x"]
    assert st["status"] == "failed" and st["cleanup_pending"] == "p-x"
    assert st["cleanup_uid"] is None  # not read: a resume never removes it unchecked


def test_counters_read_after_a_replica_was_replaced_are_not_recorded(
    tmp_path: Path, monkeypatch: Any
) -> None:
    # every run passed its restart check on the measured replica; it was replaced only after the
    # last one: the results stand, but counters read from the new replica explain nothing
    camp = deployed_campaign(tmp_path, monkeypatch)
    camp.run_detached = lambda cmd, remote: (0, aiperf_console(CLEAN))
    snaps = iter([{"old": 0}] * 6 + [{"new": 0}])  # two reads per repeat, then the check
    camp.restarts = lambda name: next(snaps)
    zero = {"spec_draft_tokens": 0.0, "spec_accepted_tokens": 0.0}
    reads = iter([{"p-x-0": zero}, {"p-x-0": {"spec_draft_tokens": 9.0,
                                              "spec_accepted_tokens": 3.0}}])  # fmt: skip
    camp.counters = lambda name: next(reads)
    camp.run_candidate({"id": "x", "gpus": 1, "replicas": 1, "args": [], "env": {}})
    st = camp.state["candidates"]["x"]
    assert st["status"] == "measured" and st["failed_runs"] == []
    assert "spec_acceptance" not in st and st["counters"]["after"] == {}


def test_the_candidate_cap_never_blocks_repairing_a_counted_candidate() -> None:
    import argparse

    camp = object.__new__(rc.Campaign)
    camp.args = argparse.Namespace(max_candidates=1, max_hours=0, max_consecutive_failures=3)
    camp.state = {"candidates": {"before": {"status": "measured"}}, "order": ["before", "x"]}
    assert camp.gate("before") is None  # counted already: its repair deploys nothing new
    assert camp.gate("x") == "1 candidates run, the cap"  # a new candidate stops at the cap
    assert camp.gate() == "1 candidates run, the cap"


def test_a_rerun_at_the_cap_repairs_the_baseline_and_deploys_nothing_new(tmp_path: Path) -> None:
    fake = tmp_path / "kubectl"
    fake.write_text(FAKE_CLUSTER.replace("/usr/bin/env python3", sys.executable))
    fake.chmod(0o755)
    path = campaign(
        tmp_path,
        workload={"model": "m", "tokenizer": "org/m", "concurrency": [8], "repeats": 3},
        candidates=[
            {"id": "before", "gpus": 4},
            {"id": "tp2x2", "gpus": 2, "replicas": 2, "args": ["--tensor-parallel-size", "2"]},
        ],
    )
    db_path, out = tmp_path / "db.json", tmp_path / "r"
    env = base_env({"NIM_TUNE_KUBECTL": str(fake), "FAKE_DB": str(db_path)})

    def go() -> dict[str, Any]:
        approved = ["--approved", rc.plan_id(path, {"max_candidates": 1})]
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), str(path), "--out", str(out), *approved,
             "--max-candidates", "1"],
            capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=120, check=False,
            env=env,
        )  # fmt: skip
        res: dict[str, Any] = json.loads(proc.stdout)
        return res

    first = go()
    assert {k: v["status"] for k, v in first["facts"]["candidates"].items()} == {
        "before": "measured"}  # fmt: skip
    profiled = len(json.loads(db_path.read_text())["profiled"])
    for f in (out / "before" / "c8-r2").iterdir():
        f.unlink()  # a repeat that no longer counts
    second = go()
    assert len(json.loads(db_path.read_text())["profiled"]) == profiled + 1  # repaired
    assert set(second["facts"]["candidates"]) == {"before"}  # nothing new deployed
    assert rc.run_counts(out / "before" / "c8-r2", None)


def test_a_rejected_attempts_console_never_counts_on_the_next_run(tmp_path: Path) -> None:
    # the export lacks throughput (not counted) while the console has it: the next invocation
    # must measure again, not count that console
    d = tmp_path / "x" / "c8-r1"
    camp = measuring_campaign(tmp_path, csv=export_csv(otps=None))
    runs: list[int] = []
    camp.run_detached = lambda cmd, remote: runs.append(1) or (0, aiperf_console(CLEAN))
    assert camp.measure("x", "n", 8, 1) is False and runs == [1, 1]
    assert not rc.run_counts(d, None)
    assert (d / "aiperf.attempt1.log").is_file() and (d / "aiperf.log.incomplete").is_file()
    assert camp.measure("x", "n", 8, 1) is False and runs == [1, 1, 1, 1]  # measured again


def test_a_stray_failed_request_left_after_the_repairs_is_left_out_of_the_comparison(
    tmp_path: Path, monkeypatch: Any
) -> None:
    # near saturation one repeat lost a few requests on every repair pass: after the cap the
    # candidate is left out (candidate_unrepaired), so the rest can be compared
    camp = deciding_campaign(tmp_path, monkeypatch)
    camp.w = {"repeats": 3, "concurrency": [8]}
    camp.facts = {}
    for r in (1, 3):
        put_run(tmp_path, f"x/c8-r{r}", export_csv())
    put_run(tmp_path, "x/c8-r2", export_csv(err="0.4"))  # a few failed requests
    st = {"status": "measured", "failed_runs": ["c8-r2"], "overloaded_runs": ["c8-r2"]}
    camp.state["candidates"]["x"] = st
    cand = {"id": "x", "args": [], "env": {}}
    for _ in range(rc.MAX_REPAIRS):
        with pytest.raises(_Deploying):
            camp.run_candidate(cand)
        # the repair pass measured it again: requests failed again
        st.update(status="measured", failed_runs=["c8-r2"], overloaded_runs=["c8-r2"])
    camp.run_candidate(cand)
    assert st["unrepaired"] == ["c8-r2"]
    seen: list[list[str]] = []

    def fake_run(cmd: list[str], **kw: Any) -> Any:
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, '{"ok": true}', "")

    monkeypatch.setattr(rc.subprocess, "run", fake_run)
    c: dict[str, Any] = {
        "baseline": "b",
        "candidates": [{"id": k, "gpus": 1, "replicas": 1} for k in ("b", "x")],
        "workload": {"concurrency": [8], "repeats": 3},
    }
    rc.compare(c, tmp_path, {"b": {"status": "measured", "unrepaired": ["c8-r1"]}, "x": st})
    cmd = seen[0]
    # the baseline is never left out: without it nothing compares, so it stays provisional
    assert cmd.count("--exclude") == 1 and cmd[cmd.index("--exclude") + 1] == "x"


def test_engine_help_is_checked_in_the_dry_run_and_a_probed_rerun_passes(tmp_path: Path) -> None:
    path = campaign(tmp_path)
    out = tmp_path / "r"
    out.mkdir()
    state = {"active_seconds": 0, "candidates": {}, "engine_flags_source": "dropped"}
    (out / "campaign_state.json").write_text(json.dumps(state))
    code, res = run(tmp_path, str(path), "--out", str(out), "--dry-run")
    assert code == rc.EXIT_REFUSED and [e["code"] for e in res["errors"]] == ["engine_help_missing"]
    assert "restore" not in res["errors"][0]["fix"]  # no probed file can clear "dropped"
    # a probed copy kept in OUT: neither the dry run nor a supervised child is refused
    state["engine_flags_source"] = "probed"
    (out / "campaign_state.json").write_text(json.dumps(state))
    (out / "engine_flags.txt").write_text("  --max-num-seqs N\n")
    code, res = run(tmp_path, str(path), "--out", str(out), "--dry-run")
    assert "engine_help_missing" not in [e["code"] for e in res["errors"]]
    assert not rc.engine_help_missing(state, False, out / "engine_flags.txt")
    argv = [str(path), "--out", str(out), "--supervise", "--engine-flags", "f", "--approved", "p"]
    assert "--engine-flags" in rc.child_argv(argv)  # the child keeps the approved help


def test_probed_engine_help_is_recorded_before_any_run(tmp_path: Path, monkeypatch: Any) -> None:
    # a probe the state does not name would be set aside on the next run as of unknown origin
    camp = deployed_campaign(tmp_path, monkeypatch)
    camp.flags_path.unlink()
    camp.flags_text = None
    camp.restarts = lambda name: {"old": 0}
    runs: list[int] = []
    camp.run_detached = lambda cmd, remote: runs.append(1) or (0, aiperf_console(CLEAN))
    camp.t.exec = lambda *a, **k: subprocess.CompletedProcess([], 0, "  --max-num-seqs N\n", "")
    saves: list[tuple[Any, int]] = []
    camp.save = lambda: saves.append((camp.state.get("engine_flags_source"), len(runs)))
    camp.run_candidate({"id": "x", "gpus": 1, "replicas": 1, "args": [], "env": {}})
    assert ("probed", 0) in saves


# ---------------------------------------------------------------- review round 6


def test_a_lost_overload_repeat_is_a_gap_at_the_campaigns_repeat_count(tmp_path: Path) -> None:
    # a 5-repeat campaign: three repeats at 32 failed requests and two were lost. Only all five
    # failing is a measured overload; three of five is a gap, for the runner and the writer alike
    for r in range(1, 6):
        for conc in (8, 32):
            put_run(tmp_path, f"before/c{conc}-r{r}", export_csv(otps=900.0))
        put_run(tmp_path, f"cand/c8-r{r}", export_csv(otps=1800.0))
    for r in (1, 2, 3):
        put_run(tmp_path, f"cand/c32-r{r}", export_csv(otps=1800.0, err="4.5"))
    c = {
        "baseline": "before",
        "candidates": [
            {"id": "before", "gpus": 1, "replicas": 1},
            {"id": "cand", "gpus": 1, "replicas": 1},
        ],
        "workload": {"concurrency": [8, 32], "repeats": 5},
        "sla": {"metric": "ttft_p99", "max_ms": 1000},
    }
    states = {"before": {"status": "measured"}, "cand": {"status": "measured"}}
    res = rc.compare(c, tmp_path, states)
    assert res["state"] == "incomplete", res
    assert "cand has no counted runs at 32 concurrent" in [
        e["message"] for e in res["errors"] if e["code"] == "load_missing"
    ]
    for r in (4, 5):  # all five failed: now the measured overload
        put_run(tmp_path, f"cand/c32-r{r}", export_csv(otps=1800.0, err="4.5"))
    assert rc.compare(c, tmp_path, states)["state"] == "compared"


def test_a_header_only_export_next_to_the_console_counts_for_both(tmp_path: Path) -> None:
    # the runner counts a run whose CSV holds only headers from its console log; the comparison
    # must read the same run, or it calls it unfinished and nothing measures it again
    d = tmp_path / "before" / "c8-r1"
    d.mkdir(parents=True)
    (d / rc.compare_aiperf.CSV_NAME).write_text("Metric,avg,p99\n\nMetric,Value\n")
    (d / rc.compare_aiperf.LOG_NAME).write_text(aiperf_console(CLEAN))
    assert rc.run_counts(d, "ttft_p99")
    unfinished: list[str] = []
    runs = rc.compare_aiperf.collect(tmp_path, unfinished=unfinished)
    assert unfinished == [] and runs["before"][8][0]["otps_avg"] == 900.0
    # an export with data still wins over the console
    (d / rc.compare_aiperf.CSV_NAME).write_text(export_csv(otps=1234.0))
    assert rc.compare_aiperf.collect(tmp_path)["before"][8][0]["otps_avg"] == 1234.0
    assert rc.run_results(d)["otps_avg"] == 1234.0


def test_a_repair_under_a_new_candidate_cap_finishes_the_missing_repeat(
    tmp_path: Path, monkeypatch: Any
) -> None:
    # resumed with --max-candidates 1 and two measured cases: the measured one is admitted for
    # its repair, and the per-load gate must not count it again once it is set back to pending
    import argparse

    camp = deployed_campaign(tmp_path, monkeypatch)
    del camp.gate  # the real gates
    camp.args = argparse.Namespace(max_candidates=1, max_hours=0, max_consecutive_failures=3)
    camp.t0 = time.monotonic()
    camp.state.update(active_seconds=0.0, order=["before", "x"])
    camp.restarts = lambda name: {"old": 0}
    for r in (1, 3):
        put_run(tmp_path, f"x/c8-r{r}", export_csv())
    camp.state["candidates"] = {
        "before": {"status": "measured"},
        "x": {"status": "measured", "failed_runs": ["c8-r2"], "overloaded_runs": [],
              "image_id": "sha256:" + "aa" * 32},
    }  # fmt: skip
    runs: list[str] = []
    camp.run_detached = lambda cmd, remote: runs.append(remote) or (0, aiperf_console(CLEAN))
    assert camp.gate("x") is None  # main admits it: counted already
    camp.run_candidate({"id": "x", "gpus": 1, "replicas": 1, "args": [], "env": {}})
    st = camp.state["candidates"]["x"]
    assert [r.rsplit("/", 1)[-1] for r in runs] == ["c8-r2"], st
    assert st["status"] == "measured" and "stop_condition" not in camp.state
    assert camp.gate("y") == "2 candidates run, the cap"  # a new candidate is still stopped
    camp.args.max_hours = 1e-9  # the time gate still applies inside a candidate
    assert camp.gate(cap=False) == f"time budget of {1e-9} h spent"


def test_a_stop_before_validation_leaves_nothing_a_resume_reuses(tmp_path: Path) -> None:
    # the console is written, then the runner is stopped while the export is fetched: no file
    # under the run's names may then pass as a finished run
    class Stopped(Exception):
        pass

    def fetch(*a: Any, **k: Any) -> Any:
        raise Stopped

    camp = measuring_campaign(tmp_path)
    camp.t = type("T", (), {"client_exec": lambda self, *a, **k: fetch()})()
    camp.run_detached = lambda cmd, remote: (0, aiperf_console(CLEAN))
    d = tmp_path / "x" / "c8-r1"
    with pytest.raises(Stopped):
        camp.measure("x", "n", 8, 1)
    assert not rc.run_counts(d, None) and not (d / rc.compare_aiperf.LOG_NAME).exists()
    assert rc.compare_aiperf.read_run(d) is None  # nothing the comparison reads either
    # a run that passes is published under the run's names, and its staging is gone
    camp = measuring_campaign(tmp_path)
    camp.run_detached = lambda cmd, remote: (0, aiperf_console(CLEAN))
    assert camp.measure("x", "n", 8, 1) is True
    assert rc.run_counts(d, None) and not (d / ".attempt").exists()


def docker_cleanup_campaign(daemon: str) -> tuple[Any, list[str]]:
    """A Docker campaign whose daemon reports DAEMON as its ID ('' when docker info fails)."""
    camp = object.__new__(rc.Campaign)
    camp.c = {"name": "me", "name_prefix": "p", "target": "docker", "docker": {}}
    camp.log = lambda m: None
    camp.save = lambda: None
    removed: list[str] = []
    camp.remove = lambda name, uid=None: removed.append(name)
    info = subprocess.CompletedProcess([], 0 if daemon else 1, daemon + "\n", "")
    camp.t = type("T", (), {"docker": lambda self, *a, **k: info,
                            "replicas": lambda self, n: [],
                            "exists": lambda self, n: True,
                            "ident": lambda self, n: "cid-1",
                            "kept": lambda self, n, u: u == "cid-1"})()  # fmt: skip
    return camp, removed


def test_a_docker_cleanup_is_bound_to_the_daemon_it_was_left_on() -> None:
    # DOCKER_HOST or the docker context picks the daemon: a resume against another daemon on
    # the same host must never clear the mark while the container may hold GPUs on the first
    camp, removed = docker_cleanup_campaign("daemon-A")
    st: dict[str, Any] = {"status": "measured"}
    camp.retire("p-x", st)  # recorded with the daemon's ID, then removed
    assert removed == ["p-x"] and st == {"status": "measured"}
    left = {"status": "measured", "cleanup_pending": "p-x", "cleanup_target": camp.target_id(),
            "cleanup_uid": "cid-1"}  # fmt: skip
    assert left["cleanup_target"] == "docker:local#daemon-A"
    other, removed = docker_cleanup_campaign("daemon-B")
    other.state = {"candidates": {"x": dict(left)}}
    with pytest.raises(rc.PendingCleanupError, match="daemon-A"):
        other.clear_pending()
    assert removed == [] and other.state["candidates"]["x"] == left  # the mark stays
    # a mark without the daemon (an earlier runner, or docker info failed) fails closed
    for where in ("docker:local", "docker:local#?"):
        other.state = {"candidates": {"x": {**left, "cleanup_target": where}}}
        with pytest.raises(rc.PendingCleanupError) as exc:
            other.clear_pending()
        assert "campaign_state.json" in (exc.value.fix or "") and removed == []
    # the daemon does not answer now: transient, never a verdict on the mark
    silent, removed = docker_cleanup_campaign("")
    silent.state = {"candidates": {"x": dict(left)}}
    with pytest.raises(rc.KubeError):
        silent.clear_pending()
    assert silent.state["candidates"]["x"] == left
    same, removed = docker_cleanup_campaign("daemon-A")  # back on the original daemon
    same.state = {"candidates": {"x": dict(left)}}
    same.clear_pending()
    assert removed == ["p-x"] and same.state["candidates"]["x"] == {"status": "measured"}


def test_an_image_change_sets_aside_the_old_search_probes_so_the_comparison_completes(
    tmp_path: Path, monkeypatch: Any
) -> None:
    # on image A the search probed 16; on image B it never gets there: the old probe folders
    # must not stay where compare_aiperf counts every load with a run folder
    camp = deployed_campaign(tmp_path, monkeypatch)
    camp.c["sla"] = {"metric": "ttft_p99", "max_ms": 100,
                     "search": {"start": 8, "max": 32, "resolution_pct": 25}}  # fmt: skip
    camp.restarts = lambda name: {"old": 0}
    for r in (1, 2, 3):
        put_run(tmp_path, f"x/c8-r{r}", export_csv())
        put_run(tmp_path, f"x/c16-r{r}", export_csv())
    (tmp_path / "x" / "c16-r1.superseded").mkdir()  # an earlier set-aside: never overwritten
    st = {"status": "pending", "image_id": "sha256:" + "aa" * 32,
          "sla_search": {"tried": {"8": True, "16": True}}}  # fmt: skip
    camp.state["candidates"]["x"] = st
    camp.t.spec = lambda pod, allow: pod_record(IMAGE_B)  # the tag moved

    def slow_from_8(cmd: list[str], remote: str) -> tuple[int, str]:
        ttft = 60.0 if int(cmd[cmd.index("--concurrency") + 1]) < 8 else 500.0
        rows = [("Time to First Token (ms)", ttft), ("Output Token Throughput (tokens/sec)", 900.0),
                ("Request Error Rate (%)", 0.0)]  # fmt: skip
        return 0, aiperf_console(rows)

    camp.run_detached = slow_from_8
    camp.run_candidate({"id": "x", "gpus": 1, "replicas": 1, "args": [], "env": {}})
    assert st["status"] == "measured" and "16" not in st["sla_search"]["tried"]
    folder = tmp_path / "x"
    assert not [p for p in folder.glob("c16-r*") if rc.compare_aiperf.RUN_DIR.match(p.name)]
    assert (folder / "c16-r1.superseded-2" / "profile_export_aiperf.csv").is_file()
    assert (folder / "c16-r2.superseded" / "profile_export_aiperf.csv").is_file()
    proc = subprocess.run(
        [sys.executable, str(SCRIPT.parent / "compare_aiperf.py"), str(tmp_path), "--baseline",
         "x", "--loads", "8", "--sla", "ttft_p99<=100", "--min-repeats", "3"],
        capture_output=True, text=True, timeout=120, check=False,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stdout


def test_runs_saved_on_another_image_are_measured_again(tmp_path: Path, monkeypatch: Any) -> None:
    # two repeats ran on image A; the redeploy for the third runs B (the tag moved): runs on A
    # never rank next to runs on B. On the same image the counted runs are reused
    camp = deployed_campaign(tmp_path, monkeypatch)
    camp.restarts = lambda name: {"old": 0}
    runs: list[str] = []
    camp.run_detached = lambda cmd, remote: runs.append(remote) or (0, aiperf_console(CLEAN))
    for r in (1, 2):
        put_run(tmp_path, f"x/c8-r{r}", export_csv())
    (tmp_path / "x" / "gone-0.pod.json").write_text(json.dumps(pod_record(IMAGE_A)))
    st = {"status": "measured", "failed_runs": ["c8-r3"], "overloaded_runs": [],
          "image_id": "sha256:" + "aa" * 32}  # fmt: skip
    camp.state["candidates"]["x"] = st
    cand = {"id": "x", "gpus": 1, "replicas": 1, "args": [], "env": {}}
    camp.t.spec = lambda pod, allow: pod_record(IMAGE_B)
    camp.run_candidate(cand)
    assert sorted(r.rsplit("/", 1)[-1] for r in runs) == ["c8-r1", "c8-r2", "c8-r3"]
    assert st["status"] == "measured" and st["image_id"] == "sha256:" + "bb" * 32
    # the old run folders are set aside whole, kept as evidence under a name never read
    assert (tmp_path / "x" / "c8-r1.superseded" / "profile_export_aiperf.csv").is_file()
    # a replica no longer serving keeps its record, under a name the writer never reads
    assert (tmp_path / "x" / "gone-0.pod.json.superseded").is_file()
    # the same image again: nothing is measured again
    runs.clear()
    (tmp_path / "x" / "c8-r3" / rc.compare_aiperf.LOG_NAME).unlink()
    st.update(status="measured", failed_runs=["c8-r3"])
    camp.run_candidate(cand)
    assert [r.rsplit("/", 1)[-1] for r in runs] == ["c8-r3"]
    # the digest identifies the image, whatever repository a node names it by
    mirror = pod_record("mirror.example.com/x@sha256:" + "aa" * 32)
    assert rc.image_identity([pod_record(IMAGE_A), mirror]) == "sha256:" + "aa" * 32
    assert rc.image_identity([pod_record(IMAGE_A), [{"Image": "sha256:" + "1" * 64}]]) == (
        f"sha256:{'1' * 64},sha256:{'aa' * 32}"
    )
    assert rc.image_identity([pod_record(IMAGE_A), {}]) is None  # one replica unknown: none


def test_the_retry_removal_is_marked_pending_first(tmp_path: Path, monkeypatch: Any) -> None:
    # attempt 1 failed to start; removing it before attempt 2 finds a same-named container of
    # someone else: the mark the foreign_resource fix speaks of must exist
    camp = deployed_campaign(tmp_path, monkeypatch)
    del camp.retire  # the real one: mark, save, remove
    saved: list[dict[str, Any]] = []
    camp.save = lambda: saved.append(json.loads(json.dumps(camp.state)))
    camp.wait_ready = lambda name, n: ("intermittent", "not ready after 1s")

    def remove(name: str, uid: str | None = None) -> None:
        raise rc.ForeignCleanupError(f"container {name} exists and was not created by it")

    camp.remove = remove
    with pytest.raises(rc.ForeignCleanupError):
        camp.run_candidate({"id": "x", "gpus": 1, "replicas": 1, "args": [], "env": {}})
    st = saved[-1]["candidates"]["x"]
    assert st["cleanup_pending"] == "p-x" and st["cleanup_target"] == "docker:local#daemon-1"


def test_an_unrecorded_engine_flags_file_never_overwrites_an_earlier_one(tmp_path: Path) -> None:
    camp = object.__new__(rc.Campaign)
    camp.flags_path, camp.state = tmp_path / "engine_flags.txt", {}
    camp.flags_path.write_text("first\n")
    camp.adopt_engine_flags(None)
    camp.flags_path.write_text("second\n")
    camp.adopt_engine_flags(None)
    assert (tmp_path / "engine_flags.unrecorded.txt").read_text() == "first\n"
    assert (tmp_path / "engine_flags.unrecorded.1.txt").read_text() == "second\n"
    assert not camp.flags_path.exists()
