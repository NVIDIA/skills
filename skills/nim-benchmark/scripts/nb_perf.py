#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""nb_perf: performance mode. Plans, runs and checks AIPerf runs against one endpoint.

It builds the plan, starts AIPerf in a client (this host, a Docker container or a Kubernetes pod
you name), and reads each run's export file. It trusts the file, never the exit code: a missing,
short, cancelled or failed export is refused. A run counts only with every request answered,
and a load counts only when every planned repeat (at least three) is counted. Seeds are 100 x load + repeat, and every run
gets a fresh cache salt so one run cannot reuse another run's cached prompts.
Run it to print the defaults: python3 scripts/nb_perf.py --defaults.
Exit codes: 0 ok, 2 bad input. Never prompts. Logs go to stderr. Stdlib only.
"""

from __future__ import annotations

import secrets
import shlex
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import nb_common
from nb_common import (
    AIPERF_VERSION,
    DEFAULT_CLIENT_IMAGE,
    MIN_REPEATS,
    ContractParser,
    emit,
    error,
)

DEFAULT_LOADS = (8, 32, 128)
DEFAULT_ISL, DEFAULT_OSL = 1024, 256
DEFAULT_IMAGE_SIDE = 512
DEFAULT_IMAGE_FORMAT = "jpeg"
IMAGE_FORMATS = ("jpeg", "png")
# AIPerf makes a new random-noise image for every request, so a server's image cache cannot
# make a repeated image look fast. This is its own default, written out so it is in the plan.
IMAGE_SOURCE = "noise"
EXPORT_NAME = "profile_export_aiperf.json"
LOG_NAME = "aiperf.log"
# AIPerf files that hold the raw prompts and answers; they are removed after each run
RAW_FILES = ("inputs.json", "profile_export.jsonl")
SUPPORTED_SCHEMA_MAJOR = "1"
RETRIES = 1  # one more try for a run that had failed requests
# AIPerf export key -> (our key, lower is better)
METRICS = {
    "time_to_first_token": ("ttft", True),
    "inter_token_latency": ("itl", True),
    "request_latency": ("e2e", True),
    "output_token_throughput": ("otps", False),
    "request_throughput": ("rps", False),
}
LATENCY_KEYS = ("time_to_first_token", "inter_token_latency", "request_latency")
# fields that must match for two runs to be compared
MATCH_FIELDS = (
    "aiperf_version",
    "client_image",
    "isl",
    "osl",
    "loads",
    "repeats",
    "seeds",
    "tokenizer",
    "requests_per_load",
    "extra_inputs",
    "images_per_request",
    "image_width",
    "image_height",
    "image_format",
    "image_source",
)


def seed_for(load: int, repeat: int) -> int:
    return 100 * load + repeat


def requests_for(load: int) -> int:
    """Ten requests per concurrent slot, and never fewer than 40."""
    return max(40, 10 * load)


def build_plan(
    *,
    endpoint: str,
    model: str,
    tokenizer: str,
    loads: list[int],
    repeats: int,
    isl: int,
    osl: int,
    key_env: str | None,
    target: str,
    image: str,
    namespace: str | None,
    pod: str | None,
    images: int = 0,
    image_width: int | None = None,
    image_height: int | None = None,
    image_format: str = DEFAULT_IMAGE_FORMAT,
) -> dict[str, Any]:
    """Everything a performance run will do. Its hash is the plan id the user approves.

    With images above 0 the model is a vision-language model: every request carries that many
    synthetic images of one size. Text models keep images at 0 and no image fields.
    """
    vision = images > 0
    return {
        "mode": "perf",
        "endpoint": endpoint,
        "model": model,
        "tokenizer": tokenizer,
        "loads": loads,
        "repeats": repeats,
        "seeds": [seed_for(load, r) for load in loads for r in range(1, repeats + 1)],
        "isl": isl,
        "osl": osl,
        "requests_per_load": {str(load): requests_for(load) for load in loads},
        "extra_inputs": {"ignore_eos": True},
        "aiperf_version": AIPERF_VERSION,
        "client_target": target,
        "client_image": image if target == "docker" else None,
        "namespace": namespace,
        "pod": pod,
        "key_env": key_env,
        "images_per_request": images,
        "image_width": (image_width or DEFAULT_IMAGE_SIDE) if vision else None,
        "image_height": (image_height or DEFAULT_IMAGE_SIDE) if vision else None,
        "image_format": image_format if vision else None,
        "image_source": IMAGE_SOURCE if vision else None,
    }


def aiperf_args(
    plan: dict[str, Any], load: int, repeat: int, artifact_dir: str, salt: str | None
) -> list[str]:
    """The AIPerf command line for one load and repeat. It holds no secret."""
    args = [
        "aiperf",
        "profile",
        "--model",
        plan["model"],
        "--url",
        plan["endpoint"],
        "--endpoint-type",
        "chat",
        "--streaming",
        "--tokenizer",
        plan["tokenizer"],
        "--concurrency",
        str(load),
        "--request-count",
        str(plan["requests_per_load"][str(load)]),
        "--random-seed",
        str(seed_for(load, repeat)),
        "--isl",
        str(plan["isl"]),
        "--osl",
        str(plan["osl"]),
        "--osl-stddev",
        "0",
        "--artifact-dir",
        artifact_dir,
        "--extra-inputs",
        "ignore_eos:true",
    ]
    if plan.get("images_per_request"):
        args += [
            "--image-batch-size",
            str(plan["images_per_request"]),
            "--image-width-mean",
            str(plan["image_width"]),
            "--image-width-stddev",
            "0",
            "--image-height-mean",
            str(plan["image_height"]),
            "--image-height-stddev",
            "0",
            "--image-format",
            plan["image_format"],
            "--image-source",
            plan["image_source"],
        ]
    if salt:
        args += ["--extra-inputs", f"cache_salt:{salt}"]
    return args


def shell_line(args: list[str], key_env: str | None) -> str:
    """One shell line. The key is a reference to a variable, expanded where AIPerf runs.

    The expanded key is in the `aiperf` command line, so other users of that host or pod can
    see it with `ps`. AIPerf 0.13.0 has no documented way to read it from the environment.
    """
    line = " ".join(shlex.quote(a) for a in args)
    return f'{line} --api-key "${key_env}"' if key_env else line


def client_argv(plan: dict[str, Any], line: str, out: Path, name: str) -> list[str]:
    """The program that runs one AIPerf line in the chosen client."""
    target = plan["client_target"]
    if target == "local":
        return ["sh", "-c", f"cd {shlex.quote(str(out.resolve()))} && {line}"]
    if target == "kubernetes":
        return ["kubectl", "exec", "-n", plan["namespace"], plan["pod"], "--", "sh", "-c", line]
    argv = ["docker", "run", "--rm", "--name", name, "--network", "host"]
    argv += ["-v", f"{out.resolve()}:/out", "-e", "HF_TOKEN"]
    if plan["key_env"]:
        argv += ["-e", plan["key_env"]]
    inner = f"pip install -q aiperf=={plan['aiperf_version']} && {line}"
    return [*argv, plan["client_image"], "sh", "-c", inner]


def artifact_dir_for(plan: dict[str, Any], out: Path, rel: str) -> str:
    """Where AIPerf writes, as the client sees it.

    In a pod the folder is named after the local run folder, so the candidate and the baseline
    endpoint never share a pod folder.
    """
    target = plan["client_target"]
    if target == "docker":
        return f"/out/{rel}"
    if target == "kubernetes":
        tag = nb_common.sha256_text(str(out.resolve()))[:12]
        # A path inside the client pod, not on this machine; the hash keeps runs apart.
        return f"/tmp/nim-benchmark/{tag}/{rel}"  # nosec B108
    return rel  # local: relative to the run folder, which is the working directory


def pod_exec(plan: dict[str, Any], *command: str) -> subprocess.CompletedProcess[str]:
    """Run one short command in the Kubernetes pod."""
    argv = ["kubectl", "exec", "-n", plan["namespace"], plan["pod"], "--", *command]
    return nb_common.run_command(argv, timeout=300)


def read_remote_export(plan: dict[str, Any], artifact_dir: str) -> str | None:
    """Fetch an export file from a Kubernetes pod. Other targets write to the run folder."""
    if plan["client_target"] != "kubernetes":
        return None
    path = f"{artifact_dir}/{EXPORT_NAME}"
    done = pod_exec(plan, "cat", path)
    if done.returncode:
        stderr = done.stderr or ""
        if done.returncode == 1 and path in stderr and "No such file or directory" in stderr:
            return None
        raise subprocess.CalledProcessError(done.returncode, done.args)
    return done.stdout


def drop_remote_raw_files(plan: dict[str, Any], artifact_dir: str) -> None:
    """Delete the raw prompt and answer files in the pod. Other targets keep none there."""
    if plan["client_target"] == "kubernetes":
        done = pod_exec(plan, "rm", "-f", *(f"{artifact_dir}/{name}" for name in RAW_FILES))
        if done.returncode:
            raise OSError("Kubernetes artifact cleanup failed")


def stop_client(plan: dict[str, Any], name: str, artifact_dir: str) -> bool:
    """Stop an AIPerf run that timed out. False when it may still be running.

    Local runs need nothing here: run_command kills the whole process group on a timeout.
    """
    target = plan["client_target"]
    if target == "docker":
        done = nb_common.run_command(["docker", "rm", "-f", name], timeout=60)
        return done.returncode == 0
    elif target == "kubernetes":
        # killing `kubectl exec` leaves aiperf running in the pod; match this run's folder only.
        # Python, not pkill: every pod that runs aiperf has Python, slim images lack procps.
        done = pod_exec(plan, "python3", "-c", POD_STOP_SCRIPT, f"artifact-dir {artifact_dir}")
        return done.returncode == 0
    return True


# Signal the selected run and its workers, then confirm their exit within five seconds.
# PID start times prevent a reused PID from being signalled or mistaken for the old process.
POD_STOP_SCRIPT = """
import os, signal, sys, time
from pathlib import Path
snapshot = {}
selected = set()
for pid in filter(str.isdigit, os.listdir("/proc")):
    pid = int(pid)
    if pid == os.getpid():
        continue
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        line = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\\0", b" ").decode(errors="replace")
        snapshot[pid] = (int(fields[1]), fields[19])
        if sys.argv[1] + " " in line + " ":
            selected.add(pid)
    except (FileNotFoundError, ProcessLookupError):
        pass
while True:
    children = {pid for pid, (parent, _) in snapshot.items() if parent in selected} - selected
    if not children:
        break
    selected.update(children)
for pid in tuple(selected):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[19] == snapshot[pid][1]:
            os.kill(pid, signal.SIGTERM)
    except (FileNotFoundError, ProcessLookupError):
        pass
deadline = time.monotonic() + 5
while selected:
    for pid in tuple(selected):
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            if fields[19] != snapshot[pid][1] or fields[0] in ("Z", "X"):
                selected.remove(pid)
        except (FileNotFoundError, ProcessLookupError):
            selected.remove(pid)
    if selected and time.monotonic() >= deadline:
        sys.exit(1)
    if selected:
        time.sleep(0.05)
"""


def drop_raw_files(folder: Path) -> None:
    """Delete raw prompts and answers; let callers report deletion failures."""
    failures = []
    for name in RAW_FILES:
        try:
            (folder / name).unlink(missing_ok=True)
        except OSError as exc:
            failures.append(exc)
    if failures:
        raise failures[0]


def metric(doc: dict[str, Any], key: str) -> dict[str, Any] | None:
    value = doc.get(key)
    return value if isinstance(value, dict) and "avg" in value else None


def check_export(doc: Any, requested: int) -> tuple[str, str]:
    """Judge one export file: ('ok' | 'request_errors' | 'tool_failed', why).

    It trusts the file's own numbers: the schema version, a finished run, the number of answered
    requests, the error rate and the metrics we need. The exit code of AIPerf is never used.
    """
    if not isinstance(doc, dict):
        return "tool_failed", "the export is not a JSON object"
    if doc.get("aiperf_version") != AIPERF_VERSION:
        return "tool_failed", f"aiperf_version must be {AIPERF_VERSION}"
    version = str(doc.get("schema_version", ""))
    if version.split(".")[0] != SUPPORTED_SCHEMA_MAJOR:
        return "tool_failed", f"unsupported schema_version {version or 'missing'}"
    if doc.get("was_cancelled") or doc.get("is_complete") is False:
        return "tool_failed", "the run was cancelled or did not finish"
    missing = [key for key in METRICS if metric(doc, key) is None]
    if missing:
        return "tool_failed", f"the export has no {', '.join(missing)}"
    if any("p99" not in doc[key] for key in LATENCY_KEYS):
        return "tool_failed", "the export has no p99 for a latency"
    answered = (metric(doc, "request_count") or {}).get("avg")
    if answered is None or int(answered) != requested:
        return "tool_failed", f"{answered} of {requested} requests are in the export"
    failed = (metric(doc, "request_error_rate") or {}).get("avg", 0)
    if failed or doc.get("error_summary"):
        return "request_errors", "some requests failed"
    return "ok", ""


def summarize_run(doc: dict[str, Any]) -> dict[str, float]:
    """The numbers one counted run contributes."""
    out: dict[str, float] = {}
    for key, (short, _lower) in METRICS.items():
        entry = doc[key]
        out[f"{short}_avg"] = float(entry["avg"])
        if key in LATENCY_KEYS:
            out[f"{short}_p99"] = float(entry["p99"])
    return out


def aggregate(runs: list[dict[str, float]]) -> dict[str, dict[str, Any]]:
    """Mean, min and max of every number across the counted repeats of one load."""
    keys = sorted(runs[0]) if runs else []
    return {
        k: {
            "mean": round(statistics.fmean(r[k] for r in runs), 4),
            "min": round(min(r[k] for r in runs), 4),
            "max": round(max(r[k] for r in runs), 4),
            "values": [r[k] for r in runs],
        }
        for k in keys
    }


def run_once(
    plan: dict[str, Any], out: Path, load: int, repeat: int, salt: str | None
) -> tuple[str, str, dict[str, Any] | None]:
    """Run AIPerf once; an internal client_not_stopped verdict prevents further load."""
    rel = f"perf/c{load}-r{repeat}"
    local = out / rel
    export = local / EXPORT_NAME
    try:
        local.mkdir(parents=True, exist_ok=True)
        export.unlink(missing_ok=True)  # an old export must never pass for this run's
    except OSError as exc:
        return "tool_failed", f"AIPerf output could not be prepared ({type(exc).__name__})", None
    adir = artifact_dir_for(plan, out, rel)
    line = shell_line(aiperf_args(plan, load, repeat, adir, salt), plan["key_env"])
    if plan["client_target"] == "kubernetes":
        line = f"rm -f {shlex.quote(f'{adir}/{EXPORT_NAME}')} && {line}"
    name = f"nim-benchmark-{plan.get('plan_id', 'run')}-c{load}-r{repeat}"
    argv = client_argv(plan, line, out, name)
    failures: list[str] = []
    stopped = True
    finished = False
    console = ""
    text = None
    try:
        done = nb_common.run_command(argv, timeout=3600)
        finished = True
        console = done.stdout + done.stderr
        nb_common.write_text(local / LOG_NAME, console)
        text = read_remote_export(plan, adir)
    except subprocess.TimeoutExpired:
        try:
            stopped = stop_client(plan, name, adir)
        except (OSError, subprocess.TimeoutExpired):
            stopped = False
        where = "pod" if plan["client_target"] == "kubernetes" else "container"
        failures.append(
            "AIPerf timed out" if stopped
            else f"AIPerf timed out and may still run in the {where}; stop it"
        )
    except subprocess.CalledProcessError as exc:
        failures.append(f"Pod export read failed (exit status {exc.returncode})")
    except OSError as exc:
        reason = "AIPerf output could not be saved or read" if finished else "AIPerf could not start"
        failures.append(f"{reason} ({type(exc).__name__})")
    finally:
        try:
            drop_raw_files(local)
        except OSError:
            failures.append("local artifact cleanup failed")
        try:
            drop_remote_raw_files(plan, adir)
        except (OSError, subprocess.TimeoutExpired):
            failures.append("Kubernetes artifact cleanup failed")
    if failures:
        verdict = "tool_failed" if stopped else "client_not_stopped"
        return verdict, "; ".join(failures), None
    try:
        if text is not None:
            nb_common.write_text(export, text)
        if not export.exists():
            if salt and "cache_salt" in console:
                return "salt_rejected", "the server rejected cache_salt", None
            return "tool_failed", "AIPerf wrote no export file", None
        try:
            doc = nb_common.read_json(export)
        except ValueError:
            return "tool_failed", "the export is not valid JSON", None
    except OSError as exc:
        return "tool_failed", f"AIPerf output could not be saved or read ({type(exc).__name__})", None
    verdict, why = check_export(doc, plan["requests_per_load"][str(load)])
    if verdict == "request_errors" and "cache_salt" in console:
        return "salt_rejected", "the server rejected cache_salt", doc
    return verdict, why, doc


def run_plan(plan: dict[str, Any], out: Path) -> dict[str, Any]:
    """Run every load and repeat. Returns the summary, the incomplete loads and the problems."""
    salt_ok = True
    client_not_stopped = False
    loads: dict[str, Any] = {}
    incomplete: list[dict[str, Any]] = []
    problems: list[dict[str, str]] = []
    error_runs: list[str] = []
    nb_common.SCRUBBER.add_env(plan["key_env"])
    # every side runs the same plan, so a complete load has the same count on both sides
    needed = max(plan["repeats"], MIN_REPEATS)
    for load in plan["loads"]:
        counted: list[dict[str, float]] = []
        for repeat in range(1, plan["repeats"] + 1):
            if client_not_stopped:
                break
            for _attempt in range(RETRIES + 1):
                salt = secrets.token_hex(6) if salt_ok else None
                verdict, why, doc = run_once(plan, out, load, repeat, salt)
                if verdict == "salt_rejected":
                    salt_ok = False  # the server refuses cache_salt: run without it from now on
                    verdict, why, doc = run_once(plan, out, load, repeat, None)
                if verdict == "client_not_stopped":
                    client_not_stopped = True
                if verdict == "ok" and doc is not None:
                    counted.append(summarize_run(doc))
                    break
                if verdict == "request_errors":
                    error_runs.append(f"c{load}-r{repeat}")
                    continue
                folder = out / f"perf/c{load}-r{repeat}"
                fixes: list[str] = []
                if client_not_stopped:
                    if plan["client_target"] == "kubernetes":
                        client = f"AIPerf in pod {plan['namespace']}/{plan['pod']}"
                    else:
                        client = (
                            f"benchmark container nim-benchmark-{plan.get('plan_id', 'run')}"
                            f"-c{load}-r{repeat}"
                        )
                    fixes.append(f"Stop {client} before any rerun.")
                if "local artifact cleanup failed" in why:
                    fixes.append(
                        f"Restore write/delete access to {folder} and remove remaining raw "
                        "prompt and answer files before rerunning."
                    )
                if "Kubernetes artifact cleanup failed" in why:
                    remote = artifact_dir_for(plan, out, f"perf/c{load}-r{repeat}")
                    fixes.append(
                        f"Restore exec and file-delete access in pod {plan['namespace']}/"
                        f"{plan['pod']}; remove remaining raw prompt and answer files from "
                        f"{remote} before rerunning."
                    )
                if "Pod export read failed" in why:
                    remote = artifact_dir_for(plan, out, f"perf/c{load}-r{repeat}")
                    fixes.append(
                        f"Restore pod exec and export read access in pod {plan['namespace']}/"
                        f"{plan['pod']} for {remote}/{EXPORT_NAME} before rerunning."
                    )
                if "AIPerf could not start" in why:
                    executable = {"local": "aiperf", "docker": "docker", "kubernetes": "kubectl"}[
                        plan["client_target"]
                    ]
                    fixes.append(
                        f"Check the {executable} executable on PATH, execution permissions, "
                        "and access to the selected client before rerunning."
                    )
                if "AIPerf output could not" in why:
                    fixes.append(f"Check output-folder permissions and client export access for {folder}.")
                if not fixes:
                    fixes.append(
                        "Investigate the client timeout and endpoint health before rerunning."
                        if "timed out" in why else f"Read {folder}/{LOG_NAME}"
                    )
                problems.append(
                    error(
                        "tool_failed",
                        f"load {load} repeat {repeat}: {why}",
                        " ".join(fixes),
                    )
                )
                break
        if len(counted) >= needed:
            loads[str(load)] = {"repeats": len(counted), "metrics": aggregate(counted)}
        else:
            incomplete.append({"load": load, "counted": len(counted), "needed": needed})
    return {
        "loads": loads,
        "incomplete": incomplete,
        "run_had_request_errors": error_runs,
        "cache_salt": "used" if salt_ok else "dropped",
        "problems": problems,
        "client_not_stopped": client_not_stopped,
    }


def main(argv: list[str] | None = None) -> None:
    parser = ContractParser(
        description="Performance mode helpers for nim-benchmark. Run it to print the defaults.",
        epilog="Example: python3 scripts/nb_perf.py --defaults",
    )
    parser.add_argument("--defaults", action="store_true", help="print the default workload")
    parser.parse_args(argv)
    emit(
        True,
        "defaults",
        "done",
        {
            "loads": list(DEFAULT_LOADS),
            "isl": DEFAULT_ISL,
            "osl": DEFAULT_OSL,
            "repeats": MIN_REPEATS,
            "aiperf": AIPERF_VERSION,
            "client_image": DEFAULT_CLIENT_IMAGE,
        },
    )


if __name__ == "__main__":
    main()
