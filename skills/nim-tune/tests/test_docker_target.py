# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""run_campaign.py with "target": "docker"; a fake docker CLI stands in for the GPU host."""

from __future__ import annotations

import importlib.util
import json
import os
import secrets
import subprocess
import sys
from pathlib import Path
from typing import Any
from testenv import base_env

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SCRIPT = SCRIPTS / "run_campaign.py"
sys.path.insert(0, str(SCRIPTS))
_rc_spec = importlib.util.spec_from_file_location("run_campaign", SCRIPT)
assert _rc_spec and _rc_spec.loader
rc = importlib.util.module_from_spec(_rc_spec)
_rc_spec.loader.exec_module(rc)
_spec = importlib.util.spec_from_file_location("targets", SCRIPTS / "targets.py")
assert _spec and _spec.loader
targets = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(targets)

# A stateful stand-in for `docker`: containers live in a JSON file. A NIM container answers the
# health probe once it runs; one started with --bogus exits with a fatal log line. AIPerf prints a
# console table whose TTFT p99 is 30 ms x C for the baseline and 10 ms x C otherwise.
FAKE_DOCKER = r"""#!/usr/bin/env python3
import json, os, sys
db_path = os.environ["FAKE_DB"]
db = (json.load(open(db_path)) if os.path.exists(db_path)
      else {"boxes": {}, "nets": [], "calls": [], "remote": {}})
a = sys.argv[1:]
db["calls"].append(a[:2])
def save(code=0, out="", err=""):
    json.dump(db, open(db_path, "w"))
    sys.stdout.write(out); sys.stderr.write(err); sys.exit(code)
VALUED = {"--name", "--network", "--label", "--gpus", "--restart", "-u", "-e", "--env-file", "-v"}
if a[0] == "version":
    save(0, "27.0\n")
if a[:2] == ["network", "inspect"]:
    if a[2] not in db["nets"]:
        save(1, "", "No such network")
    on = {k: {} for k, b in db["boxes"].items() if b.get("network") == a[2]}
    save(0, json.dumps([{"Name": a[2], "Labels": {"nim-tune/campaign": "d1"}, "Containers": on}]))
if a[:2] == ["network", "create"]:
    db["nets"].append(a[-1]); save(0, "net-id\n")
if a[:2] == ["network", "rm"]:
    db["nets"].remove(a[2]); save(0)
if a[:2] == ["image", "inspect"]:
    db["calls"].append(a[1:])
    if os.environ.get("FAKE_IMAGE_INSPECT_FAIL"):
        save(1, "", "image unavailable")
    digests = [] if os.environ.get("FAKE_NO_DIGEST") else [
        "nvcr.io/nim/nvidia/vllm-model-free-nim@sha256:" + "ab" * 32]
    save(0, json.dumps([{"RepoDigests": digests}]))
def box_name(ref):  # docker accepts a container name or its id
    return next((n for n, b in db["boxes"].items() if b.get("Id") == ref), ref)
if a[0] == "inspect":
    b = db["boxes"].get(box_name(a[1]))
    save(0, json.dumps([b]), "") if b else save(1, "", "Error: No such object: " + a[1])
if a[:2] == ["container", "inspect"]:  # containers only, as the runner asks
    b = db["boxes"].get(box_name(a[2]))
    save(0, json.dumps([b]), "") if b else save(1, "", "Error: No such container: " + a[2])
if a[0] == "run":
    i, labels, env, opts = 2, {}, [], {}
    while a[i].startswith("-"):
        if a[i] in VALUED:
            if a[i] == "--label":
                k, _, v = a[i + 1].partition("="); labels[k] = v
            if a[i] == "-e":
                env.append(a[i + 1])
            opts[a[i]] = a[i + 1]; i += 2
        else:
            i += 1
    image, args = a[i], a[i + 1:]
    bad = "--bogus" in args
    db["boxes"][opts["--name"]] = {
        "Id": "id-" + opts["--name"] + str(len(db["calls"])), "Name": "/" + opts["--name"],
        "State": {"Running": not bad, "Status": "exited" if bad else "running",
                  "OOMKilled": False, "ExitCode": 1 if bad else 0},
        "RestartCount": 0, "Args": args, "Image": "sha256:" + "ef" * 32,
        "Config": {"Labels": labels, "Env": env, "Image": image, "Cmd": args},
        "gpus": opts.get("--gpus"), "network": opts.get("--network"),
    }
    save(0, db["boxes"][opts["--name"]]["Id"] + "\n")  # docker run -d prints the new id
if a[0] == "rm":
    db["boxes"].pop(box_name(a[-1]), None); save(0)
if a[0] == "logs":
    b = db["boxes"][box_name(a[-1])]
    if "--bogus" in b["Args"]:
        save(0, "vllm: error: argument --bogus: unrecognized arguments\n")
    msg = "Launching vLLM: vllm serve /m --tensor-parallel-size 2 " + " ".join(b["Args"])
    save(0, json.dumps({"level": "INFO", "message": msg.strip()}) + "\n")
if a[0] == "exec":
    i = 2 if a[1] == "-i" else 1
    box, cmd = a[i], a[i + 1:]
    if box.endswith("-aiperf") and not db["boxes"].get(box, {}).get("State", {}).get("Running"):
        save(1, "", "Error: container is not running")
    if cmd[:2] == ["aiperf", "--version"]:
        save(0, "0.13.0\n")
    if cmd[:2] == ["vllm", "serve"]:
        save(0, "  --max-num-seqs N\n  --max-num-batched-tokens N\n  --bogus X\n")
    if cmd[0] == "cat":
        r = db["remote"].get(cmd[1]); save(0 if r is not None else 1, r or "")
    if cmd[:2] == ["test", "-e"]:
        save(0 if cmd[2] in db["remote"] else 1)
    if cmd[:2] == ["python", "-c"] and "killpg" in cmd[2]:
        if len(cmd) == 4:
            save(0)
        out, ex, prof = cmd[4], cmd[5], cmd[6:]
        c = int(prof[prof.index("--concurrency") + 1]); url = prof[prof.index("--url") + 1]
        db.setdefault("profiled", []).append(url)
        p99 = (30 if url.endswith("-before:8000") else 10) * c
        b = "│"
        rows = [("Time to First Token (ms)", p99 / 2, p99), ("Inter Token Latency (ms)", 5, 6),
                ("Output Token Throughput (tokens/sec)", 100 * c, None),
                ("Request Error Rate (%)", 0, None)]
        cell = lambda v: "N/A" if v is None else f"{v:,.2f}"
        lines = ["NVIDIA AIPerf | LLM Metrics"] + [
            b + b.join([n, cell(v), "1", "2", cell(p), "1", "1", "0"]) + b
            for n, v, p in rows] + ["└--"]
        db["remote"][out], db["remote"][ex] = "\n".join(lines) + "\n", "0\n"
        save(0)
    if cmd[:2] == ["python", "-c"]:
        code = cmd[2]
        if "health/ready" in code:
            nim = db["boxes"].get(cmd[3], {})
            save(0, "200\n" if nim.get("State", {}).get("Running") else "0\n")
        if "chat/completions" in code:
            save(0, "200\n")
        save(0, "100 50 60 0 0\n")
save(9, "", "unexpected: " + " ".join(a))
"""


def setup(tmp: Path, **over: Any) -> tuple[Path, dict[str, str]]:
    fake = tmp / "docker"
    fake.write_text(FAKE_DOCKER.replace("/usr/bin/env python3", sys.executable))
    fake.chmod(0o755)
    c: dict[str, Any] = {
        "name": "d1",
        "target": "docker",
        "gpu_budget": 2,
        "name_prefix": "nimtune-me",
        "baseline": "before",
        "docker": {
            "image": "nvcr.io/nim/nvidia/vllm-model-free-nim:2.1.4",
            "gpus": "2,3",
            "env": {"NIM_MODEL_PATH": "hf://org/m", "NIM_SERVED_MODEL_NAME": "m"},
            "env_from_host": ["NGC_API_KEY"],
        },
        "workload": {"model": "m", "tokenizer": "org/m", "concurrency": [8, 32], "repeats": 3},
        "candidates": [
            {"id": "before", "gpus": 2},
            {"id": "mnt4096", "gpus": 2, "args": ["--max-num-batched-tokens", "4096"]},
            {"id": "bad", "gpus": 1, "args": ["--bogus", "1"]},
        ],
    }
    c.update(over)
    path = tmp / "campaign.json"
    path.write_text(json.dumps(c))
    home = tmp / "home"  # "~/" on a local Docker host resolves here, never to the user's home
    home.mkdir(exist_ok=True)
    env = base_env({"NIM_TUNE_DOCKER": str(fake), "FAKE_DB": str(tmp / "db.json")})
    env["HOME"] = str(home)
    return path, env


def fake_secret(prefix: str = "") -> str:
    """A synthetic credential made at run time, so no credential-like literal is committed."""
    return prefix + secrets.token_hex(12)


def go(tmp: Path, path: Path, env: dict[str, str], *extra: str) -> tuple[int, dict[str, Any]]:
    # a live run carries the user's approval of this exact file, as the dry run prints it
    approval = [] if "--dry-run" in extra else ["--approved", rc.plan_id(path)]
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), str(path), "--out", str(tmp / "r"), *extra, *approval],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=120,
        check=False,
        env=env,
    )
    return proc.returncode, json.loads(proc.stdout)


def test_docker_campaign_end_to_end(tmp_path: Path) -> None:
    path, env = setup(tmp_path)
    code, out = go(tmp_path, path, env)
    states = out["facts"]["candidates"]
    assert states["before"]["status"] == "measured" and states["mnt4096"]["status"] == "measured"
    assert states["bad"]["status"] == "failed" and states["bad"]["kind"] == "config"
    assert states["mnt4096"]["engagement"]["nimtune-me-mnt4096"]["state"] == "engaged"
    db = json.loads((tmp_path / "db.json").read_text())
    assert db["boxes"] == {}  # every candidate and the client container were removed
    assert db["nets"] == []  # and the network this campaign created
    runs = [c for c in db["calls"] if c[0] == "run"]
    assert len(runs) == 4  # client, before, mnt4096, bad (once: a config failure is not retried)
    spec = json.loads((tmp_path / "r" / "mnt4096" / "deploy.json").read_text())["docker"]
    assert spec[spec.index("--gpus") + 1] == '"device=2,3"'  # only the GPUs the user listed
    assert "-e" in spec and "NGC_API_KEY" in spec  # the secret by name, never by value
    image = "nvcr.io/nim/nvidia/vllm-model-free-nim:2.1.4"
    assert spec[-3:] == [image, "--max-num-batched-tokens", "4096"]  # engine args after the image
    bad = json.loads((tmp_path / "r" / "bad" / "deploy.json").read_text())["docker"]
    assert bad[bad.index("--gpus") + 1] == '"device=2"'
    assert set(db["profiled"]) == {
        "http://nimtune-me-before:8000",
        "http://nimtune-me-mnt4096:8000",
    }
    assert code == 3  # the failed candidate is reported
    assert out["facts"]["compare"]["picks"]


def test_docker_dry_run_changes_nothing(tmp_path: Path) -> None:
    path, env = setup(tmp_path)
    code, out = go(tmp_path, path, env, "--dry-run")
    assert code == 0 and out["facts"]["target"] == "docker"
    assert "client container nimtune-me-aiperf" in out["facts"]["creates"]
    db = json.loads((tmp_path / "db.json").read_text())
    assert not any(c[0] in ("run", "rm", "network") for c in db["calls"])


def test_docker_needs_one_replica_within_the_listed_gpus(tmp_path: Path) -> None:
    path, env = setup(
        tmp_path,
        candidates=[{"id": "before", "gpus": 2}, {"id": "x", "gpus": 2, "replicas": 2}],
    )
    code, out = go(tmp_path, path, env, "--dry-run")
    assert code == 2 and "bad_candidate_gpus" in [e["code"] for e in out["errors"]]


def test_ssh_runs_every_docker_command_as_one_quoted_line() -> None:
    t = targets.DockerTarget(
        {"name": "d1", "name_prefix": "p", "docker": {"image": "i", "gpus": "0", "ssh": "u@h"}},
        Path("."),
        lambda m: None,
    )
    argv = t.host_argv(["docker", "run", "--gpus", '"device=0,1"', "-e", "A=b c"])
    assert argv[:5] == ["ssh", "-o", "BatchMode=yes", "--", "u@h"]  # "--": the host is no option
    assert "'\"device=0,1\"'" in argv[5] and "'A=b c'" in argv[5]


def test_a_changed_candidate_never_resumes_into_old_results(tmp_path: Path) -> None:
    path, env = setup(tmp_path, candidates=[{"id": "before", "gpus": 2}])
    go(tmp_path, path, env)
    assert not (tmp_path / "r" / "campaign_state.json.tmp").exists()  # written atomically
    c = json.loads(path.read_text())
    c["candidates"][0]["args"] = ["--max-num-seqs", "64"]  # same id, different setting
    path.write_text(json.dumps(c))
    code, out = go(tmp_path, path, env)
    assert code == 5 and [e["code"] for e in out["errors"]] == ["resume_mismatch"]


def test_dry_run_checks_host_files_and_resolves_home(tmp_path: Path) -> None:
    path, env = setup(tmp_path)
    c = json.loads(path.read_text())
    c["docker"]["env_file"] = "~/no-such-nimtune-test.env"  # ~ is the Docker host's home
    path.write_text(json.dumps(c))
    code, out = go(tmp_path, path, env, "--dry-run")
    assert code == 4 and out["errors"][0]["code"] == "dry_run_failed"
    msg = out["errors"][0]["message"]
    assert "no-such-nimtune-test.env is missing" in msg and "~" not in msg
    assert f"{env['HOME']}/no-such-nimtune-test.env" in msg  # the test's home, not the user's


def test_saved_container_specs_hold_no_secret_values(tmp_path: Path) -> None:
    path, env = setup(tmp_path, candidates=[{"id": "before", "gpus": 2}])
    db = tmp_path / "db.json"
    # the fake records -e NAME as given; a real docker inspect resolves it to NAME=value
    key = fake_secret("nvapi-")
    env = {**env, "NGC_API_KEY": key}
    go(tmp_path, path, env)
    t = targets.DockerTarget(json.loads(path.read_text()), tmp_path, lambda m: None)
    box = {
        "Id": "x",
        "Config": {
            "Image": "i",
            "Env": [
                f"NGC_API_KEY={key}",
                f"HF_TOKEN={fake_secret('hf_')}",
                "NIM_MODEL_PATH=hf://org/m",
                f"MY_PASSWORD={fake_secret()}",
                "PATH=/usr/bin",
            ],
        },
    }
    t.inspect = lambda name: box
    env_saved = t.spec("c")[0]["Config"]["Env"]
    assert env_saved == ["NIM_MODEL_PATH=hf://org/m"]
    for f in (tmp_path / "r").rglob("*"):
        if f.is_file():
            assert key not in f.read_text(errors="ignore"), f
    assert db.is_file()


def test_a_literal_secret_in_a_campaign_is_refused_and_never_echoed(tmp_path: Path) -> None:
    secret = fake_secret("nvapi-")
    cand = {"id": "before", "gpus": 2, "env": {"NGC_API_KEY": secret}}
    path, env = setup(tmp_path, candidates=[cand])
    code, out = go(tmp_path, path, env, "--dry-run")
    assert code == 2 and "literal_secret" in [e["code"] for e in out["errors"]]
    assert secret not in json.dumps(out)
    c = json.loads(path.read_text())
    flag_value, hf = fake_secret(), fake_secret("hf_")
    c["candidates"] = [{"id": "before", "gpus": 2, "args": ["--api-key", flag_value]}]
    c["docker"]["env"]["HF_TOKEN"] = hf
    path.write_text(json.dumps(c))
    code, out = go(tmp_path, path, env, "--dry-run")
    assert [e["code"] for e in out["errors"]].count("literal_secret") == 2
    assert flag_value not in json.dumps(out) and hf not in json.dumps(out)


def test_secrets_in_the_runners_environment_never_reach_results(tmp_path: Path) -> None:
    path, env = setup(tmp_path, candidates=[{"id": "before", "gpus": 2}])
    # a secret reaches the NIM only by a name the campaign passes from the host: the runner
    # masks those values (read by name), wherever the NIM prints them
    c = json.loads(path.read_text())
    c["docker"]["env_from_host"] = ["NGC_API_KEY", "MY_API_KEY"]
    path.write_text(json.dumps(c))
    leaked = "env-secret-value-42"
    # a NIM that logs the key it was given: the startup log must still be clean
    fake = (
        (tmp_path / "docker")
        .read_text()
        .replace(
            'msg = "Launching vLLM',
            'msg = "key=" + os.environ.get("MY_API_KEY", "") + " Launching vLLM',
        )
    )
    (tmp_path / "docker").write_text(fake)
    go(tmp_path, path, {**env, "MY_API_KEY": leaked})
    files = [f for f in (tmp_path / "r").rglob("*") if f.is_file()]
    assert any("startup.log" in f.name for f in files)
    for f in files:
        assert leaked not in f.read_text(errors="ignore"), f


def test_sudo_cannot_pass_host_variables(tmp_path: Path) -> None:
    path, env = setup(tmp_path)
    c = json.loads(path.read_text())
    c["docker"]["sudo"] = True  # env_from_host is ["NGC_API_KEY"] in setup
    path.write_text(json.dumps(c))
    code, out = go(tmp_path, path, env, "--dry-run")
    assert code == 2 and "sudo_env" in [e["code"] for e in out["errors"]]


def test_unsafe_docker_options_are_refused(tmp_path: Path) -> None:
    path, env = setup(tmp_path)
    c = json.loads(path.read_text())
    c["docker"]["ssh"] = "-oProxyCommand=touch /tmp/x"
    c["docker"]["extra_run_args"] = ["--gpus", "all", "--rm"]
    path.write_text(json.dumps(c))
    code, out = go(tmp_path, path, env, "--dry-run")
    got = [e["code"] for e in out["errors"]]
    assert code == 2 and "bad_ssh" in got and "bad_run_args" in got


def test_a_destination_that_starts_with_a_dash_is_refused(tmp_path: Path) -> None:
    path, env = setup(tmp_path)
    for bad_host in ("-Fevil.conf", "-oProxyCommand=x", "u@-oProxyCommand=x", "-F@h"):
        c = json.loads(path.read_text())
        c["docker"]["ssh"] = bad_host
        path.write_text(json.dumps(c))
        code, out = go(tmp_path, path, env, "--dry-run")
        assert code == 2 and "bad_ssh" in [e["code"] for e in out["errors"]], bad_host
    c["docker"]["ssh"] = "alice@gpu-host-01"
    path.write_text(json.dumps(c))
    # a stand-in ssh on PATH runs the quoted remote line locally: no DNS lookup, no connection
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ssh = bin_dir / "ssh"
    ssh.write_text(
        f"#!{sys.executable}\n"
        "import os, subprocess, sys\n"
        "open(os.environ['FAKE_SSH_LOG'], 'a').write(sys.argv[-2] + '\\n')\n"
        "sys.exit(subprocess.run(['sh', '-c', sys.argv[-1]]).returncode)\n"
    )
    ssh.chmod(0o755)
    log = tmp_path / "ssh.log"
    env = {**env, "PATH": f"{bin_dir}{os.pathsep}{env['PATH']}", "FAKE_SSH_LOG": str(log)}
    code, out = go(tmp_path, path, env, "--dry-run")
    assert code == 0 and "bad_ssh" not in [e["code"] for e in out.get("errors", [])], out
    assert set(log.read_text().split()) == {"alice@gpu-host-01"}  # every call went to the fake


def test_attached_short_flags_cannot_pass_the_run_arg_checks(tmp_path: Path) -> None:
    path, env = setup(tmp_path)
    for args, code_expected in (
        ([f"-eNGC_API_KEY={fake_secret('nvapi-')}"], "bad_run_args"),
        (["-lnim-tune/fingerprint=x"], "bad_run_args"),
        (["-ie", "A=b"], "bad_run_args"),
        (["--device", "/dev/nvidia1"], "bad_run_args"),
        (["--device=/dev/nvidia1"], "bad_run_args"),
        (["--cap-add", "SYS_ADMIN"], "bad_run_args"),
        (["--privileged"], "bad_run_args"),
        (["--api-key", fake_secret()], "literal_secret"),
        ([f"--hf-token={fake_secret()}"], "literal_secret"),
    ):
        c = json.loads(path.read_text())
        c["docker"]["extra_run_args"] = args
        path.write_text(json.dumps(c))
        code, out = go(tmp_path, path, env, "--dry-run")
        assert code == 2 and code_expected in [e["code"] for e in out["errors"]], args
    for args in (["--shm-size=16g"], ["--ipc", "host"], ["-it"], ["-p", "8000:8000"]):
        c["docker"]["extra_run_args"] = args
        path.write_text(json.dumps(c))
        code, out = go(tmp_path, path, env, "--dry-run")
        assert code == 0, (args, out["errors"])


def test_a_container_with_other_settings_is_recreated_not_measured(tmp_path: Path) -> None:
    path, env = setup(tmp_path, candidates=[{"id": "before", "gpus": 2}])
    db = tmp_path / "db.json"
    stale = {
        "Id": "old", "Name": "/nimtune-me-before",
        "State": {"Running": True, "Status": "running", "OOMKilled": False, "ExitCode": 0},
        "RestartCount": 0, "Args": [],
        "Config": {"Labels": {"nim-tune/campaign": "d1", "nim-tune/fingerprint": "older"},
                   "Env": [], "Image": "old-image:1", "Cmd": []},
    }  # fmt: skip
    db.write_text(json.dumps({"boxes": {"nimtune-me-before": stale}, "nets": [], "calls": [],
                              "remote": {}}))  # fmt: skip
    _, out = go(tmp_path, path, env)
    calls = json.loads(db.read_text())["calls"]
    i_rm = calls.index(["rm", "-f"])
    assert any(c[0] == "run" for c in calls[i_rm:])  # removed first, then started fresh
    assert out["facts"]["candidates"]["before"]["status"] == "measured"
    assert "recommendation" in out["facts"]["compare"]


def test_a_lost_client_stops_the_readiness_wait() -> None:
    t = object.__new__(targets.DockerTarget)

    def lost() -> None:
        raise targets.ClientLostError("client container x is exited")

    t.check_client = lost
    try:
        t.ready_count("nimtune-me-before")
        raise AssertionError("expected ClientLostError")
    except targets.ClientLostError:
        pass
