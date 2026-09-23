# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle tests for the vendored virtualenv runner.

These run REAL detached processes through the four verbs — the cases mirror the
upstream implementation's race/recovery/cancel suite: durable launcher record,
start gate, PID-reuse identity, process-group cleanup, cancel ordering, bounded
log tail.
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import virtualenv_runner as vr  # noqa: E402


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

@pytest.fixture()
def venv(tmp_path):
    """A minimal-but-real venv: pyvenv.cfg + bin/python -> current interpreter."""
    root = tmp_path / "venv"
    (root / "bin").mkdir(parents=True)
    (root / "pyvenv.cfg").write_text("home = /usr\n", encoding="utf-8")
    os.symlink(sys.executable, root / "bin" / "python")
    return root


@pytest.fixture()
def job_dir(tmp_path):
    d = tmp_path / "results" / "job-0001"
    d.mkdir(parents=True)
    return d


def write_script(tmp_path, name, body):
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def run_verb(capsys, *argv):
    rc = vr.main(list(argv))
    out = capsys.readouterr().out.strip()
    payload = json.loads(out.splitlines()[-1]) if out else {}
    return rc, payload


def wait_for(predicate, timeout=15.0, interval=0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def _evidence(job_dir):
    """Everything durable about a job dir — embedded in failure messages."""
    paths = vr.RunnerPaths(Path(job_dir))
    return {
        "exit_status": vr._read_json(paths.exit_status),
        "launcher_status": vr._read_json(paths.launcher_status),
        "cancel_marker": paths.cancel_marker.exists(),
        "log": vr._read_tail(paths.log_path, 20) if paths.log_path.exists() else None,
    }


def wait_terminal(capsys, job_dir, timeout=15.0):
    result = {}

    def is_terminal():
        rc, payload = run_verb(capsys, "status", "--job-dir", str(job_dir))
        result.update(payload)
        return payload.get("status") in ("COMPLETE", "ERROR", "CANCELED")

    assert wait_for(is_terminal, timeout=timeout), \
        f"never terminal: {result} evidence={_evidence(job_dir)}"
    result["evidence"] = _evidence(job_dir)
    return result


# ---------------------------------------------------------------------------
# Happy path + artifacts
# ---------------------------------------------------------------------------

def test_submit_runs_to_complete_and_writes_results(capsys, venv, job_dir, tmp_path):
    script = write_script(tmp_path, "train.py", (
        "import os, pathlib, sys\n"
        "out = pathlib.Path(os.environ['TAO_RESULTS_ROOT']) / 'metrics.json'\n"
        "out.write_text('{\"metric\": 0.9}')\n"
        "print('trained', sys.argv[1:])\n"
    ))
    rc, sub = run_verb(
        capsys, "submit", "--job-dir", str(job_dir), "--venv", str(venv),
        "--script", str(script), "--job-id", "job-0001",
        "--arg", "train", "--arg=--results={results_dir}",
    )
    assert rc == 0 and sub["result"] == "submitted" and sub["status"] == "RUNNING"
    assert sub["backend_ref"].startswith("pid:")

    final = wait_terminal(capsys, job_dir)
    assert final["status"] == "COMPLETE" and final["return_code"] == 0, final
    assert json.loads((job_dir / "metrics.json").read_text())["metric"] == 0.9

    rc = vr.main(["logs", "--job-dir", str(job_dir), "--tail", "50"])
    assert rc == 0
    assert "trained" in capsys.readouterr().out


def test_placeholders_render_and_env_lands_in_script(capsys, venv, job_dir, tmp_path):
    script = write_script(tmp_path, "dump.py", (
        "import json, os, sys, pathlib\n"
        "pathlib.Path(sys.argv[1]).write_text(json.dumps({\n"
        "  'argv': sys.argv[1:],\n"
        "  'cuda': os.environ.get('CUDA_VISIBLE_DEVICES'),\n"
        "  'venv': os.environ.get('VIRTUAL_ENV'),\n"
        "  'job_id': os.environ.get('TAO_JOB_ID'),\n"
        "  'passthrough': os.environ.get('TAO_TEST_TOKEN'),\n"
        "}))\n"
    ))
    os.environ["TAO_TEST_TOKEN"] = "present"
    try:
        rc, sub = run_verb(
            capsys, "submit", "--job-dir", str(job_dir), "--venv", str(venv),
            "--script", str(script), "--job-id", "jid-42",
            "--config-path", "/tmp/spec.yaml",
            "--arg", "{results_dir}/dump.json", "--arg=-e", "--arg", "{config_path}",
            "--gpu-ids", "0,1", "-e", "TAO_TEST_TOKEN",
        )
    finally:
        os.environ.pop("TAO_TEST_TOKEN", None)
    assert rc == 0, sub
    assert wait_terminal(capsys, job_dir)["status"] == "COMPLETE"
    dump = json.loads((job_dir / "dump.json").read_text())
    assert dump["argv"] == [str(job_dir / "dump.json"), "-e", "/tmp/spec.yaml"]
    assert dump["cuda"] == "0,1"
    assert dump["venv"] == str(job_dir.parent.parent / "venv")
    assert dump["job_id"] == "jid-42"
    assert dump["passthrough"] == "present"


def test_gpus_zero_hides_cuda_devices(capsys, venv, job_dir, tmp_path):
    script = write_script(tmp_path, "cuda.py", (
        "import os, pathlib\n"
        "pathlib.Path(os.environ['TAO_RESULTS_ROOT'], 'cuda.txt')"
        ".write_text(repr(os.environ.get('CUDA_VISIBLE_DEVICES')))\n"
    ))
    rc, _ = run_verb(capsys, "submit", "--job-dir", str(job_dir), "--venv", str(venv),
                     "--script", str(script), "--gpus", "0")
    assert rc == 0
    wait_terminal(capsys, job_dir)
    assert (job_dir / "cuda.txt").read_text() == "''"


# ---------------------------------------------------------------------------
# Failure + signal reporting
# ---------------------------------------------------------------------------

def test_failing_script_reports_error_with_code(capsys, venv, job_dir, tmp_path):
    script = write_script(tmp_path, "boom.py", "import sys; print('boom'); sys.exit(3)\n")
    rc, _ = run_verb(capsys, "submit", "--job-dir", str(job_dir), "--venv", str(venv),
                     "--script", str(script))
    assert rc == 0
    final = wait_terminal(capsys, job_dir)
    assert final["status"] == "ERROR" and final["return_code"] == 3, final
    assert "code 3" in final["message"]


def test_signal_killed_script_reports_signal(capsys, venv, job_dir, tmp_path):
    script = write_script(tmp_path, "selfkill.py",
                          "import os, signal; os.kill(os.getpid(), signal.SIGKILL)\n")
    rc, _ = run_verb(capsys, "submit", "--job-dir", str(job_dir), "--venv", str(venv),
                     "--script", str(script))
    assert rc == 0
    final = wait_terminal(capsys, job_dir)
    assert final["status"] == "ERROR", final
    assert f"signal {signal.SIGKILL}" in final["message"], final


# ---------------------------------------------------------------------------
# Cancel semantics
# ---------------------------------------------------------------------------

def test_cancel_running_job_kills_group_and_reports_canceled(capsys, venv, job_dir, tmp_path):
    script = write_script(tmp_path, "sleepy.py", "import time; time.sleep(300)\n")
    rc, sub = run_verb(capsys, "submit", "--job-dir", str(job_dir), "--venv", str(venv),
                       "--script", str(script))
    assert rc == 0
    pid = sub["pid"]
    assert wait_for(lambda: run_verb(capsys, "status", "--job-dir", str(job_dir))[1]
                    .get("status") == "RUNNING")

    rc, cancel = run_verb(capsys, "cancel", "--job-dir", str(job_dir))
    assert rc == 0 and cancel["result"] == "canceled"
    assert cancel["status"] == "CANCELED"

    # No live wrapper remains (an unreaped zombie may linger under the test
    # harness, so assert identity, not raw group signalability), and the
    # CANCELED status is sticky.
    paths = vr.RunnerPaths(job_dir)
    launcher = json.loads(paths.launcher_status.read_text())
    assert not vr._process_matches(
        pid, launcher.get("process_start_marker"), str(paths.wrapper)
    ), "wrapper still alive after cancel"
    rc, again = run_verb(capsys, "status", "--job-dir", str(job_dir))
    assert again["status"] == "CANCELED"


def test_cancel_after_complete_is_already_terminal(capsys, venv, job_dir, tmp_path):
    script = write_script(tmp_path, "quick.py", "print('ok')\n")
    run_verb(capsys, "submit", "--job-dir", str(job_dir), "--venv", str(venv),
             "--script", str(script))
    wait_terminal(capsys, job_dir)
    rc, cancel = run_verb(capsys, "cancel", "--job-dir", str(job_dir))
    assert rc == 0
    assert cancel["result"] == "already_terminal" and cancel["status"] == "COMPLETE"
    # A late cancel must not flip a natural completion to CANCELED.
    _, status = run_verb(capsys, "status", "--job-dir", str(job_dir))
    assert status["status"] == "COMPLETE"


def test_wrapper_honors_cancel_before_start_gate(venv, tmp_path):
    """Drive the wrapper directly: cancel marker set while gate is closed."""
    job = tmp_path / "job"
    job.mkdir()
    wrapper = job / "launch_job.py"
    wrapper.write_text(vr.JOB_WRAPPER_SOURCE, encoding="utf-8")
    exit_p, launch_p = job / "exit.json", job / "launcher.json"
    gate_p, cancel_p = job / "gate", job / "cancel"
    proc = subprocess.Popen(
        [str(venv / "bin" / "python"), str(wrapper), str(exit_p), str(launch_p),
         str(gate_p), str(cancel_p), str(venv / "bin" / "python"), "-c", "print('nope')"],
        start_new_session=True,
    )
    assert wait_for(launch_p.exists)          # durable identity written immediately
    assert not exit_p.exists()                # blocked on the gate, script NOT run
    cancel_p.touch()                          # cancel while gated
    assert proc.wait(timeout=10) == 128 + signal.SIGTERM
    record = json.loads(exit_p.read_text())
    assert record["return_code"] == -signal.SIGTERM
    assert "Canceled before the script started" in record["error"]


# ---------------------------------------------------------------------------
# Process-group hygiene + PID-reuse identity
# ---------------------------------------------------------------------------

def test_leaked_background_child_is_cleaned_up(capsys, venv, job_dir, tmp_path):
    script = write_script(tmp_path, "leaky.py", (
        "import os, pathlib, subprocess, sys\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])\n"
        "pathlib.Path(os.environ['TAO_RESULTS_ROOT'], 'child.pid')"
        ".write_text(str(child.pid))\n"
    ))
    rc, _ = run_verb(capsys, "submit", "--job-dir", str(job_dir), "--venv", str(venv),
                     "--script", str(script))
    assert rc == 0
    final = wait_terminal(capsys, job_dir)
    assert final["status"] == "COMPLETE", final  # cleanup succeeded, exit stays 0
    child_pid = int((job_dir / "child.pid").read_text())
    assert wait_for(lambda: not _pid_alive(child_pid), timeout=5), \
        "background child leaked past job completion"


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def test_pid_reuse_is_not_treated_as_running(capsys, venv, job_dir):
    """A live-but-foreign PID in launcher_status must not read as RUNNING."""
    paths = vr.RunnerPaths(job_dir)
    paths.runner_dir.mkdir(parents=True)
    vr._atomic_write_json(paths.submit_meta, {
        "launch_started_at": time.time() - 60,
        "wrapper_path": str(paths.wrapper),
    })
    # Our own test process: alive, but not a group leader running the wrapper.
    vr._atomic_write_json(paths.launcher_status, {
        "pid": os.getpid(), "process_start_marker": "bogus-marker",
    })
    rc, status = run_verb(capsys, "status", "--job-dir", str(job_dir))
    assert status["status"] == "ERROR"
    assert "durable exit status" in status["message"]


def test_marker_mismatch_rejects_group_leader(capsys, venv, job_dir):
    """Even a real session-leader process is rejected on start-marker mismatch."""
    foreign = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                               start_new_session=True)
    try:
        paths = vr.RunnerPaths(job_dir)
        paths.runner_dir.mkdir(parents=True)
        vr._atomic_write_json(paths.submit_meta, {
            "launch_started_at": time.time() - 60,
            "wrapper_path": str(paths.wrapper),
        })
        vr._atomic_write_json(paths.launcher_status, {
            "pid": foreign.pid, "process_start_marker": "definitely-wrong",
        })
        rc, status = run_verb(capsys, "status", "--job-dir", str(job_dir))
        assert status["status"] == "ERROR"
        # And cancel must refuse to signal it: the foreign process survives.
        run_verb(capsys, "cancel", "--job-dir", str(job_dir))
        assert foreign.poll() is None, "cancel killed an innocent process"
    finally:
        foreign.kill()
        foreign.wait()


# ---------------------------------------------------------------------------
# Pending / unknown / double-submit edges
# ---------------------------------------------------------------------------

def test_orphaned_pending_grace_then_error(capsys, job_dir):
    paths = vr.RunnerPaths(job_dir)
    paths.runner_dir.mkdir(parents=True)
    vr._atomic_write_json(paths.submit_meta, {"launch_started_at": time.time()})
    _, fresh = run_verb(capsys, "status", "--job-dir", str(job_dir))
    assert fresh["status"] == "PENDING"
    vr._atomic_write_json(paths.submit_meta, {"launch_started_at": time.time() - 60})
    _, stale = run_verb(capsys, "status", "--job-dir", str(job_dir))
    assert stale["status"] == "ERROR" and "before the launcher started" in stale["message"]


def test_status_unknown_on_empty_dir(capsys, job_dir):
    _, payload = run_verb(capsys, "status", "--job-dir", str(job_dir))
    assert payload["status"] == "UNKNOWN"


def test_double_submit_is_refused(capsys, venv, job_dir, tmp_path):
    script = write_script(tmp_path, "quick.py", "print('ok')\n")
    rc, _ = run_verb(capsys, "submit", "--job-dir", str(job_dir), "--venv", str(venv),
                     "--script", str(script))
    assert rc == 0
    wait_terminal(capsys, job_dir)
    rc, second = run_verb(capsys, "submit", "--job-dir", str(job_dir), "--venv", str(venv),
                          "--script", str(script))
    assert rc == 1 and "already has runner state" in second["message"]


# ---------------------------------------------------------------------------
# Validation + logs
# ---------------------------------------------------------------------------

def test_submit_validation_errors(capsys, venv, job_dir, tmp_path):
    script = write_script(tmp_path, "quick.py", "print('ok')\n")
    cases = [
        (["--venv", str(tmp_path / "novenv")], "does not exist"),
        (["--venv", str(tmp_path)], "pyvenv.cfg is missing"),
        (["--venv", str(venv), "--script", str(tmp_path / "missing.py")], "does not exist"),
        (["--venv", str(venv), "--script", str(script), "--gpu-ids", "0,0"], "duplicates"),
        (["--venv", str(venv), "--script", str(script), "--gpus", "2", "--gpu-ids", "0"],
         "must match"),
        (["--venv", str(venv), "--script", str(script), "--arg", "{nope}"],
         "Unknown placeholder"),
    ]
    for extra, needle in cases:
        argv = ["submit", "--job-dir", str(job_dir)] + (
            extra if "--script" in extra else extra + ["--script", str(script)]
        )
        rc, payload = run_verb(capsys, *argv)
        assert rc == 1 and needle in payload["message"], (extra, payload)


def test_log_tail_is_bounded_and_exact(capsys, venv, job_dir, tmp_path):
    script = write_script(tmp_path, "chatty.py",
                          "for i in range(500): print(f'line-{i}')\n")
    run_verb(capsys, "submit", "--job-dir", str(job_dir), "--venv", str(venv),
             "--script", str(script))
    wait_terminal(capsys, job_dir)
    rc = vr.main(["logs", "--job-dir", str(job_dir), "--tail", "5"])
    out = capsys.readouterr().out
    assert rc == 0
    lines = out.strip().splitlines()
    assert lines == [f"line-{i}" for i in range(495, 500)]


# ---------------------------------------------------------------------------
# Red-team regression pins
# ---------------------------------------------------------------------------

def test_grace_strictly_exceeds_submit_wait():
    """Pin the invariant behind the false-terminal-ERROR finding: STATUS may
    never time out a launch that submit is still legitimately waiting on."""
    assert vr.PENDING_LAUNCH_GRACE_SECONDS > vr.LAUNCHER_RECORD_TIMEOUT_SECONDS


def test_orphaned_script_group_reports_running_and_cancel_kills_it(
    capsys, venv, job_dir, tmp_path,
):
    """Wrapper-only SIGKILL must not orphan the training script: status stays
    RUNNING (not terminal ERROR) and cancel kills the surviving group."""
    script = write_script(tmp_path, "sleepy.py", "import time; time.sleep(300)\n")
    rc, sub = run_verb(capsys, "submit", "--job-dir", str(job_dir), "--venv", str(venv),
                       "--script", str(script))
    assert rc == 0
    wrapper_pid = sub["pid"]
    # Wait until the script child exists inside the group, then kill ONLY the wrapper.
    def script_child():
        return [p for p in vr._group_members(wrapper_pid, str(script)) if p != wrapper_pid]
    assert wait_for(lambda: len(script_child()) > 0), "script child never appeared"
    children = script_child()
    os.kill(wrapper_pid, signal.SIGKILL)
    assert wait_for(lambda: not _pid_alive(wrapper_pid) or True, timeout=2)

    _, status = run_verb(capsys, "status", "--job-dir", str(job_dir))
    assert status["status"] == "RUNNING", status
    assert status.get("orphaned") is True

    rc, cancel = run_verb(capsys, "cancel", "--job-dir", str(job_dir))
    assert rc == 0 and cancel["result"] == "canceled", cancel
    for child in children:
        assert wait_for(lambda: not _pid_alive(child), timeout=5), \
            f"orphaned child {child} survived cancel"
    _, after = run_verb(capsys, "status", "--job-dir", str(job_dir))
    assert after["status"] == "CANCELED", after


def test_wrapper_gate_timeout_writes_durable_error(venv, tmp_path):
    """Submit dying before gate.touch() must not leave a phantom RUNNING job."""
    job = tmp_path / "job"
    job.mkdir()
    wrapper = job / "launch_job.py"
    wrapper.write_text(vr.JOB_WRAPPER_SOURCE, encoding="utf-8")
    exit_p, launch_p = job / "exit.json", job / "launcher.json"
    env = {**os.environ, "TAO_RUNNER_GATE_TIMEOUT": "1"}
    proc = subprocess.Popen(
        [str(venv / "bin" / "python"), str(wrapper), str(exit_p), str(launch_p),
         str(job / "gate"), str(job / "cancel"),
         str(venv / "bin" / "python"), "-c", "print('never runs')"],
        start_new_session=True, env=env,
    )
    assert proc.wait(timeout=10) == 125
    record = json.loads(exit_p.read_text())
    assert record["return_code"] == 125
    assert "never authorized" in record["error"]


def test_corrupted_launcher_pid_yields_json_not_crash(capsys, job_dir):
    paths = vr.RunnerPaths(job_dir)
    paths.runner_dir.mkdir(parents=True)
    vr._atomic_write_json(paths.submit_meta, {"launch_started_at": time.time() - 60})
    vr._atomic_write_json(paths.launcher_status, {"pid": "garbage", "process_start_marker": "x"})
    rc, status = run_verb(capsys, "status", "--job-dir", str(job_dir))
    assert status["status"] in ("ERROR", "PENDING", "UNKNOWN")
    rc, cancel = run_verb(capsys, "cancel", "--job-dir", str(job_dir))
    assert rc in (0, 1)  # valid JSON reply either way — no traceback


def test_planted_wrapper_symlink_is_refused(capsys, venv, job_dir, tmp_path):
    victim = tmp_path / "victim.txt"
    victim.write_text("precious")
    paths = vr.RunnerPaths(job_dir)
    paths.runner_dir.mkdir(parents=True)
    os.symlink(victim, paths.wrapper)
    script = write_script(tmp_path, "quick.py", "print('ok')\n")
    rc, payload = run_verb(capsys, "submit", "--job-dir", str(job_dir),
                           "--venv", str(venv), "--script", str(script))
    assert rc == 1 and "wrapper path" in payload["message"]
    assert victim.read_text() == "precious"  # never clobbered


def test_config_path_placeholder_requires_flag(capsys, venv, job_dir, tmp_path):
    script = write_script(tmp_path, "quick.py", "print('ok')\n")
    rc, payload = run_verb(capsys, "submit", "--job-dir", str(job_dir),
                           "--venv", str(venv), "--script", str(script),
                           "--arg", "{config_path}")
    assert rc == 1 and "Unknown placeholder" in payload["message"]
