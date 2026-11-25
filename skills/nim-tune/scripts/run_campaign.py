#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""run_campaign: deploy, verify and benchmark NIM candidates one at a time on the NIM Operator.

Reads a campaign file (JSON, see references/campaign.md). For each candidate, baseline first:
render a NIMService from the base manifest, apply it, wait for every replica, classify a failed
start (config or intermittent; intermittent is retried once), prove the settings reached the
engine with check_nim_engagement.py (next to this script), smoke-test the Service,
run AIPerf from a CPU-only client pod at every load and repeat, optionally search for the highest
concurrency that meets the latency target, record per-replica engine counters, then remove the
candidate. Finally it runs compare_aiperf.py and writes OUT/final/before_after.md.

Results land in OUT/CASE/cCONC-rREPEAT/ (aiperf.log always, profile_export_aiperf.csv when it
can be read back) with OUT/CASE/manifest.json. State in OUT/campaign_state.json makes a rerun
resume: finished runs are skipped and a candidate that failed on its configuration is not retried.
Stop gates: --max-hours (active time, kept across resumes) and --max-consecutive-failures.

It creates only NIMServices named PREFIX-ID labelled nim-tune/campaign=NAME, plus the client pod,
and deletes only those. It never creates or reads secrets. --dry-run renders every manifest and
checks it with a server-side dry run without changing the cluster.

Prints one JSON object: {"ok", "state", "next", "facts", "errors": [{"code", "message", "fix"}]}.
Exit codes: 0 ok, 2 bad input, 3 precondition failed, 4 external tool failed, 5 refused.
Never prompts. Progress goes to stderr and OUT/campaign.log. Stdlib only. Set NIM_TUNE_KUBECTL
to use another kubectl binary.
"""

from __future__ import annotations

import argparse
import contextlib
import atexit
import csv
import fcntl
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, NoReturn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_aiperf
import targets
from targets import ClientLostError, KubeError, RefusedError, make_target

EXIT_OK, EXIT_INPUT, EXIT_PRECONDITION, EXIT_EXTERNAL, EXIT_REFUSED = 0, 2, 3, 4, 5
HERE = Path(__file__).resolve().parent
ENGAGEMENT = HERE / "check_nim_engagement.py"
LABEL = targets.LABEL
ID_RE = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")
SLA_METRICS = {f"{m}_{s}" for m in ("ttft", "itl", "e2e") for s in ("avg", "p99")}
# Startup-log evidence that the configuration itself cannot run; anything else is intermittent.
FATAL = re.compile(
    r"CUDA out of memory|OutOfMemoryError|EngineDeadError|NCCL error|illegal memory access|"
    r"unrecognized arguments|error: argument|ValueError: .*(memory|cache|max_model_len)|"
    r"not enough memory|"
    r"No available memory for the cache blocks|"
    r"Engine core initialization failed|WorkerProc initialization failed",
    re.IGNORECASE,
)
CONFIG_REASONS = {"OOMKilled", "CreateContainerConfigError", "InvalidImageName"}
INTERMITTENT_REASONS = {"ImagePullBackOff", "ErrImagePull"}
# Runs in the client pod: stops the AIPerf run recorded in PIDFILE (an earlier attempt, a run left
# behind by a killed runner, or one that timed out), then, unless only stopping, starts the command
# in its own session so it outlives the exec stream, writing its console output to OUT and its exit
# code to EXIT when it ends. Usage: python -c LAUNCH PIDFILE [OUT EXIT CMD...]
LAUNCH = """
import os, signal, subprocess, sys
pidfile = sys.argv[1]
try:
    os.killpg(int(open(pidfile).read()), signal.SIGKILL)
except (OSError, ValueError):
    pass
if len(sys.argv) > 2:
    out, ex, cmd = sys.argv[2], sys.argv[3], sys.argv[4:]
    for f in (out, ex, pidfile):
        os.makedirs(os.path.dirname(f) or ".", exist_ok=True)
    if os.path.exists(ex):
        os.remove(ex)
    wrap = 'x=$1; shift; "$@" > "$0" 2>&1; echo $? > "$x.tmp"; mv "$x.tmp" "$x"'
    quiet = subprocess.DEVNULL
    p = subprocess.Popen(["sh", "-c", wrap, out, ex, *cmd], start_new_session=True,
                         stdin=quiet, stdout=quiet, stderr=quiet)
    open(pidfile, "w").write(str(p.pid))
"""
# per campaign: two campaigns sharing a client never stop each other's AIPerf
PIDFILE = "/work/{name}/aiperf.pid"
POLL_SECONDS = 15
UNKNOWN_DAEMON = "#?"  # a Docker cleanup target whose daemon did not report its ID
TARGETS = ("nim-operator", "docker")
# every byte the runner writes (results, logs, state, its JSON output) goes through this
SCRUB = targets.Scrubber()


class ContractParser(argparse.ArgumentParser):
    """Argument parser that reports a usage error as the contract JSON, not as text."""

    def error(self, message: str) -> NoReturn:
        """Print the usage error in the contract JSON format and exit."""
        emit(
            False,
            "error",
            "fix_errors",
            errors=[err("bad_usage", message, "Run with --help and fix the arguments.")],
            code=EXIT_INPUT,
        )


def err(code: str, message: str, fix: str) -> dict[str, str]:
    """One contract error: a code, what is wrong, and how to fix it."""
    return {"code": code, "message": message, "fix": fix}


def emit(
    ok: bool,
    state: str,
    next_: str,
    facts: dict[str, Any] | None = None,
    errors: list[dict[str, str]] | None = None,
    code: int | None = None,
) -> NoReturn:
    """Print the contract JSON (ok, state, next, facts, errors) with secrets masked, then exit."""
    payload = {
        "ok": ok,
        "state": state,
        "next": next_,
        "facts": facts or {},
        "errors": errors or [],
    }
    sys.stdout.write(SCRUB.dumps(payload, sort_keys=True, default=str) + "\n")
    raise SystemExit(code if code is not None else (EXIT_OK if ok else EXIT_INPUT))


# ---------------------------------------------------------------- campaign file


# Kubernetes label value (the campaign name labels everything it creates and names client paths)
NAME_RE = re.compile(r"^[A-Za-z0-9]([-A-Za-z0-9_.]{0,61}[A-Za-z0-9])?$")
ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
IMAGE_DIGEST = re.compile(r"^sha256:[a-f0-9]{64}$")
# never a leading "-": ssh would read the destination as an option (-Fevil.conf)
SSH_RE = re.compile(r"^(?!-)[A-Za-z0-9._-]+@(?!-)[A-Za-z0-9._:-]+$|^(?!-)[A-Za-z0-9._:-]+$")
# docker run options the runner owns: GPUs, network, names, labels, restarts, env and removal
RESERVED_RUN_ARGS = (
    "--gpus", "--rm", "--network", "--net", "--name", "--label", "-l", "--restart",
    "-e", "--env", "--env-file", "--runtime", "--privileged", "--device", "--cap-add",
)  # fmt: skip


def owned_run_arg(arg: str) -> bool:
    """True if a docker run argument sets an option the runner owns.

    Long options match by name (`--env=A=b`). A short option can carry its value attached
    (`-eNGC_API_KEY=x`) or sit in a bundle (`-ie X`), so every letter before the first
    value-taking one is checked.
    """
    if arg.startswith("--"):
        return arg.split("=", 1)[0] in RESERVED_RUN_ARGS
    if not arg.startswith("-"):
        return False
    for letter in arg[1:]:
        if f"-{letter}" in RESERVED_RUN_ARGS:
            return True
        if letter in "achmpuvw":  # takes a value: the rest of the word is that value
            return False
    return False


def load_campaign(path: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Read and validate the campaign file; return it with defaults filled in, and any errors.

    Every field is type-checked before it is used, so a malformed file is refused here with a
    contract error instead of crashing (or wasting GPU time) later.
    """
    errors: list[dict[str, str]] = []

    def bad(code: str, message: str, fix: str = "See campaign.md.") -> None:
        """Record one validation error for the campaign file."""
        errors.append(err(code, message, fix))

    if not path.is_file():
        return {}, [
            err(
                "campaign_missing",
                f"{path.resolve()} does not exist",
                "Pass the campaign file's full path, or cd to its folder first.",
            )
        ]
    try:
        c = json.loads(targets.read_input(path).decode("utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        return {}, [err("campaign_unreadable", f"{path}: {exc}", "Pass a valid JSON file.")]
    if not isinstance(c, dict):
        return {}, [err("campaign_shape", f"{path} is not an object", "See campaign.md.")]
    c.setdefault("target", "nim-operator")
    if c["target"] not in TARGETS:
        bad("bad_target", f"target {c['target']!r}", f"Use one of {TARGETS}.")
    keys = ["name", "baseline"]
    if c["target"] == "nim-operator":
        keys += ["context", "namespace", "base_manifest", "client_manifest"]
    for key in keys:
        if not isinstance(c.get(key), str) or not c[key]:
            bad("campaign_field", f"missing string field {key!r}")
    if isinstance(c.get("name"), str) and c["name"] and not NAME_RE.match(c["name"]):
        bad("bad_name", f"name {c['name']!r}", "Use letters, digits, - _ . (63 max).")
    c.setdefault("name_prefix", "nimtune")
    if not isinstance(c["name_prefix"], str) or not ID_RE.match(c["name_prefix"]):
        bad("bad_name_prefix", f"name_prefix {c['name_prefix']!r}", "Use [a-z0-9-].")
    elif len(c["name_prefix"]) + 21 > 63:
        bad("prefix_too_long", "name_prefix is too long", "Use 40 chars or fewer.")
    if not positive_int(c.get("gpu_budget")):
        bad("campaign_field", "gpu_budget must be a positive integer", "Set it.")
    if c.get("node") is not None and not isinstance(c["node"], str):
        bad("bad_field_type", "node must be a node name (string)")

    devices = 0
    if c["target"] == "docker":
        d = c.get("docker")
        if not isinstance(d, dict) or not d.get("image") or not str(d.get("gpus", "")).strip():
            bad("campaign_field", 'target docker needs a docker block with image and gpus "0,1"')
        else:
            devices = len([x for x in str(d["gpus"]).split(",") if x.strip()])
            sudo = bool(d.get("sudo"))
            d.setdefault("client_env_from_host", [] if sudo else ["HF_TOKEN"])
            d.setdefault("env", {})
            d.setdefault("env_from_host", [])
            d.setdefault("extra_run_args", [])
            version = d.get("aiperf_version", "0.13.0")
            if not isinstance(version, str) or not targets.AIPERF_VERSION.fullmatch(version):
                bad("bad_field_type", "docker.aiperf_version must be a version such as 0.13.0")
            if not isinstance(d["env"], dict):
                bad("bad_field_type", "docker.env must be an object {NAME: value}")
            elif not all(isinstance(k, str) and ENV_NAME.match(k) for k in d["env"]):
                bad("bad_env_name", "docker.env keys must be variable names", "Use NAME: value.")
            elif any(targets.SECRET_NAME.search(str(k)) for k in d["env"]):
                bad(
                    "literal_secret",
                    "docker.env sets a credential as a literal value",
                    "Use docker.env_from_host or docker.env_file for secrets.",
                )
            for key in ("env_from_host", "client_env_from_host", "extra_run_args"):
                if not isinstance(d[key], list) or not all(isinstance(x, str) for x in d[key]):
                    bad("bad_field_type", f"docker.{key} must be a list of strings")
                elif key != "extra_run_args" and not all(ENV_NAME.match(x) for x in d[key]):
                    # "NAME=value" would put the value on the docker command line
                    bad(
                        "literal_secret" if any("=" in x for x in d[key]) else "bad_env_name",
                        f"docker.{key} lists something other than variable names",
                        'List names only, e.g. ["NGC_API_KEY"]; the value stays in the '
                        "host environment.",
                    )
            if sudo and (d["env_from_host"] or d["client_env_from_host"]):
                # sudo resets the environment, so -e NAME would pass nothing: the NIM starts
                # without its key and fails, after GPU time is spent
                bad(
                    "sudo_env",
                    "docker.sudo drops variables named in env_from_host",
                    "With sudo, put the secrets in docker.env_file on the host instead.",
                )
            ssh = d.get("ssh")
            if ssh is not None and (not isinstance(ssh, str) or not SSH_RE.match(ssh)):
                bad("bad_ssh", f"docker.ssh {ssh!r}", "Use user@host (no options, no leading -).")
            if isinstance(d["extra_run_args"], list):
                owned = [a for a in d["extra_run_args"] if isinstance(a, str) and owned_run_arg(a)]
                if targets.SECRET_FLAG.search(" ".join(map(str, d["extra_run_args"]))):
                    bad(
                        "literal_secret",
                        "docker.extra_run_args carries a credential value",
                        "Use docker.env_from_host or docker.env_file for secrets.",
                    )
                if owned:
                    bad(
                        "bad_run_args",
                        f"docker.extra_run_args sets {owned}, which the runner controls",
                        "Remove them: GPUs, network, names, labels, restarts and env come from "
                        "the docker block.",
                    )

    w = c.setdefault("workload", {})
    if not isinstance(w, dict):
        bad("bad_field_type", "workload must be an object")
        return c, errors
    for key in ("model", "tokenizer"):
        if not isinstance(w.get(key), str):
            bad("campaign_field", f"workload.{key} missing")
    w.setdefault("isl", 1024)
    w.setdefault("osl", 256)
    w.setdefault("concurrency", [8, 32, 128])
    w.setdefault("repeats", 3)
    w.setdefault("endpoint_type", "chat")
    w.setdefault("dataset_args", [])
    ds = w.get("dataset")
    if ds is not None and (
        not isinstance(ds, dict)
        or not isinstance(ds.get("file"), str)
        or not isinstance(ds.get("type", ""), str)
    ):
        bad("bad_field_type", "workload.dataset must be {file: PATH, type: NAME}")
    elif ds is not None:
        f = Path(ds["file"])
        f = f if f.is_absolute() else path.resolve().parent / f
        if not ds["file"] or not f.is_file():
            bad(
                "dataset_missing",
                f"workload.dataset file {f} not found",
                "Write it with prep_dataset.py, or fix workload.dataset.file.",
            )
        else:
            raw = targets.read_input(f)
            try:
                raw.decode("utf-8")
            except UnicodeDecodeError:
                bad("dataset_not_utf8", f"{f} is not UTF-8", "Rewrite it with prep_dataset.py.")
            ds.setdefault("type", "multi_turn")
            # the hash goes into every manifest, so runs on different data are never compared
            ds["sha256"] = hashlib.sha256(raw).hexdigest()
    # synthetic prompts pin the output length; a real dataset keeps its own lengths, and runs
    # greedy so every candidate generates the same tokens (speculative decoding is then lossless)
    w.setdefault("extra_inputs", ["temperature:0.0"] if has_dataset(w) else ["ignore_eos:true"])
    w.setdefault("cache_salt", True)
    if not isinstance(w["dataset_args"], list) or not all(
        isinstance(a, str) for a in w["dataset_args"]
    ):
        bad("campaign_field", "workload.dataset_args must be a list of strings", "Fix.")
    elif secret_aiperf_args(w["dataset_args"]):
        # AIPerf would send it in clear text to the NIM, and it would sit on the client's
        # command line and in every saved command
        bad(
            "literal_secret",
            f"workload.dataset_args sets {secret_aiperf_args(w['dataset_args'])} with a value",
            "Remove --api-key and credential headers (-H/--header) from workload.dataset_args; "
            "the campaign's NIM takes no client credential.",
        )
    if isinstance(w["dataset_args"], list) and owned_aiperf_args(
        [a for a in w["dataset_args"] if isinstance(a, str)]
    ):
        bad(
            "bad_dataset_args",
            f"workload.dataset_args sets {owned_aiperf_args(w['dataset_args'])}, which the "
            "runner sets on every run",
            "Remove them from workload.dataset_args: the load, model, endpoint and export "
            "location come from the campaign.",
        )
    for key in ("isl", "osl"):
        if not positive_int(w[key]):
            bad("campaign_field", f"workload.{key} must be a positive integer")
    if not isinstance(w["endpoint_type"], str) or not isinstance(w["cache_salt"], bool):
        bad("bad_field_type", "workload.endpoint_type must be a string, cache_salt true/false")
    if not isinstance(w["extra_inputs"], list) or not all(
        isinstance(a, str) for a in w["extra_inputs"]
    ):
        bad("bad_field_type", "workload.extra_inputs must be a list of strings")
    conc = w["concurrency"]
    if (
        not isinstance(conc, list)
        or not conc
        or not all(isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in conc)
    ):
        bad("campaign_field", "workload.concurrency must be a non-empty list of positive ints")
    if not positive_int(w["repeats"]) or w["repeats"] < 3:
        bad("too_few_repeats", "workload.repeats is below 3", "Use 3 or more repeats.")

    if "sla" in c:
        sla = c["sla"]
        if not isinstance(sla, dict):
            bad("bad_sla", f"sla {sla!r}", "Use an object, or leave sla out.")
        else:
            metric = sla.get("metric")
            if (
                not isinstance(metric, str)  # a list or object is unhashable: no set lookup
                or metric not in SLA_METRICS
                or not positive_number(sla.get("max_ms"))
            ):
                bad(
                    "bad_sla",
                    f"sla {sla!r}",
                    "Use {'metric': 'ttft_p99', 'max_ms': 1000} with metric ttft, itl or e2e "
                    "plus _avg or _p99.",
                )
            search = sla.get("search")
            if search is not None:
                if not isinstance(search, dict):
                    bad("bad_sla", "sla.search must be an object {start, max, resolution_pct}")
                else:
                    search.setdefault("start", 8)
                    search.setdefault("max", 512)
                    search.setdefault("resolution_pct", 10)
                    # checked now: a bad value would otherwise fail mid-search, after GPU time
                    lo, hi, res = search["start"], search["max"], search["resolution_pct"]
                    if not (positive_int(lo) and positive_int(hi) and lo <= hi) or not (
                        positive_number(res) and res < 100
                    ):
                        bad(
                            "bad_sla",
                            f"sla.search {search!r}",
                            "Use whole numbers 1 <= start <= max, and 0 < resolution_pct < 100.",
                        )

    cands = c.get("candidates")
    if cands is not None and not isinstance(cands, list):
        bad("bad_field_type", "candidates must be a list")
        cands = []
    ids: list[str] = []
    for i, cand in enumerate(cands or []):
        if not isinstance(cand, dict):
            bad("bad_field_type", f"candidate {i} must be an object")
            continue
        cid = cand.get("id")
        if not isinstance(cid, str) or not ID_RE.match(cid) or len(cid) > 20:
            bad("bad_candidate_id", f"candidate {i}: id {cid!r}", "Use [a-z0-9-], 20 max.")
            continue
        if c["target"] == "docker" and cid == "aiperf":  # {name_prefix}-aiperf is the client
            bad("bad_candidate_id", "candidate id 'aiperf' is the client's name", "Rename it.")
            continue
        if cid in ids:
            bad("duplicate_candidate", f"id {cid!r} repeats", "Make ids unique.")
        ids.append(cid)
        cand.setdefault("replicas", 1)
        cand.setdefault("args", [])
        cand.setdefault("env", {})
        g, r = cand.get("gpus"), cand["replicas"]
        if not positive_int(g) or not positive_int(r):
            bad("bad_candidate_gpus", f"{cid}: gpus and replicas", "Positive ints.")
        elif positive_int(c.get("gpu_budget")) and g * r > c["gpu_budget"]:
            # a malformed budget is already reported above; comparing with it would crash
            bad(
                "over_budget",
                f"{cid} uses {g * r} GPUs, budget {c.get('gpu_budget')}",
                "Lower gpus or replicas, or ask the user for a larger budget.",
            )
        args_ok = isinstance(cand["args"], list) and all(isinstance(a, str) for a in cand["args"])
        if not args_ok:
            # a string would be deployed one character per argument
            bad("bad_candidate_args", f"{cid}: args", "Use a list of strings.")
        env_ok = isinstance(cand["env"], dict) and all(
            isinstance(k, str)
            and ENV_NAME.match(k)
            and isinstance(v, str | int | float)
            and not isinstance(v, bool)
            for k, v in cand["env"].items()
        )
        if not env_ok:
            bad(
                "bad_candidate_env",
                f"{cid}: env",
                "Use {NAME: value} with string or number values.",
            )
        literal = [k for k in (cand["env"] if env_ok else {}) if targets.SECRET_NAME.search(str(k))]
        literal += [
            a
            for a in (arg_pairs(cand["args"]) if args_ok else [])
            if targets.SECRET_FLAG.fullmatch(a)
        ]
        if literal:
            # a literal secret would sit in the NIMService spec or on a docker run command line
            bad(
                "literal_secret",
                f"{cid}: {', '.join(x.split()[0] for x in literal)} set as a literal value",
                "Pass secrets by reference: a valueFrom secret in the base manifest, or "
                "docker.env_from_host / docker.env_file.",
            )
        pattern = cand.get("engagement_pattern")
        if pattern is not None:
            try:
                re.compile(str(pattern))
            except re.error as exc:
                bad(
                    "bad_pattern", f"{cid}: engagement_pattern {pattern!r}: {exc}", "Fix the regex."
                )
        if c["target"] == "docker" and (r != 1 or (isinstance(g, int) and devices and g > devices)):
            bad(
                "bad_candidate_gpus",
                f"{cid}: Docker runs one replica on at most the {devices} listed GPUs",
                "Set replicas 1 and gpus within docker.gpus.",
            )
    # recorded for scripts/write_recommended.py; the runner itself does not use them
    for key in ("model", "model_profile", "gpu_name"):
        if key in c and (not isinstance(c[key], str) or not c[key].strip()):
            bad("bad_field_type", f"{key} must be a non-empty string")
    if "image_digest" in c and not (
        isinstance(c["image_digest"], str) and IMAGE_DIGEST.match(c["image_digest"])
    ):
        bad("bad_image_digest", "image_digest must look like sha256:<64 hex characters>")
    if not ids:
        bad("no_candidates", "candidates is empty", "Add the baseline at least.")
    elif c.get("baseline") not in ids:
        bad("baseline_missing", f"baseline {c.get('baseline')!r} is not a candidate", "Add it.")
    return c, errors


def positive_int(x: Any) -> bool:
    """True for an int above zero (a bool does not count)."""
    return isinstance(x, int) and not isinstance(x, bool) and x > 0


def positive_number(x: Any) -> bool:
    """True for a finite int or float above zero (a bool does not count)."""
    return isinstance(x, int | float) and not isinstance(x, bool) and 0 < x < math.inf


def input_files(c: dict[str, Any], campaign_dir: Path) -> list[Path]:
    """The local files a campaign deploys or sends, besides the campaign file itself."""
    names = [c.get("base_manifest"), c.get("client_manifest")]
    if c.get("target") != "nim-operator":
        names = []  # Docker deploys from the campaign file alone
    ds = c.get("workload", {}).get("dataset")
    if isinstance(ds, dict):
        names.append(ds.get("file"))
    files = [Path(n) for n in names if isinstance(n, str) and n]
    return [f if f.is_absolute() else campaign_dir / f for f in files]


def plan_id(path: Path, options: dict[str, Any] | None = None) -> str:
    """The id of a reviewed plan: a hash of the campaign file, of every local file it deploys
    or sends (manifests, dataset) and of the options that decide what runs (see plan_options).
    A live run must quote it; a change to any of them voids it.
    """
    h = hashlib.sha256(targets.read_input(path))
    if options:
        h.update(b"\0options\0" + json.dumps(options, sort_keys=True).encode())
    c, _ = load_campaign(path)
    for f in input_files(c, path.resolve().parent):
        digest = hashlib.sha256(targets.read_input(f)).hexdigest() if f.is_file() else "missing"
        h.update(f"\0{f.name}\0{digest}".encode())
    return h.hexdigest()[:12]


# the options that change which candidates run, for how long, or how long a deploy may take
# (plus the --engine-flags file's hash, added by plan_options)
PLAN_OPTIONS = ("only", "max_candidates", "max_hours", "max_consecutive_failures", "ready_timeout")


def plan_options(args: argparse.Namespace, defaults: argparse.ArgumentParser) -> dict[str, Any]:
    """The plan-changing options that differ from their default, in a form that hashes the same
    on every run. A dry run and the live run must pass the same ones, or the plan id differs.
    """
    out: dict[str, Any] = {}
    for name in PLAN_OPTIONS:
        value = sorted(set(args.only)) if name == "only" else getattr(args, name)
        if value != defaults.get_default(name) and value != []:
            out[name] = value
    if args.engine_flags:
        # the file decides which candidates are skipped before deploy; without it the baseline
        # is probed instead, which can let a candidate the approved plan skipped run
        raw = targets.read_input(Path(args.engine_flags))
        out["engine_flags_sha256"] = hashlib.sha256(raw).hexdigest()
    return out


def engine_flags_text(args: argparse.Namespace) -> str | None:
    """The --engine-flags file's text, from the same bytes the plan id hashed; None without it."""
    if not args.engine_flags:
        return None
    return targets.read_input(Path(args.engine_flags)).decode("utf-8", errors="replace")


def engine_help_missing(state: dict[str, Any], given: bool, flags_path: Path) -> bool:
    """Campaign.missing_engine_help on a saved STATE; GIVEN: this run has --engine-flags."""
    src = state.get("engine_flags_source")
    if given or src is None:
        return False
    return src != "probed" or not flags_path.is_file()


def engine_help_error(out: Path, source: Any) -> dict[str, str]:
    """The engine_help_missing refusal, with the fix that can clear SOURCE."""
    restore = ", or restore the probed OUT/engine_flags.txt; " if source == "probed" else "; "
    return err(
        "engine_help_missing",
        f"earlier runs in {out} checked candidates against engine help ({source}) that this run "
        "does not have",
        "Rerun the dry run and the run with the same --engine-flags FILE the earlier runs used "
        "(any saved `vllm serve --help=all` output of this image when they recorded none)"
        + restore
        + "or start a new --out folder.",
    )


def run_results(d: Path) -> dict[str, float]:
    """The metrics of one finished run, from its CSV export, else from its console log."""
    try:
        got = compare_aiperf.read_run(d) or {}  # the rule compare_aiperf reads runs by
    except (OSError, UnicodeDecodeError, csv.Error):
        # an export compare_aiperf cannot read is an unfinished run there: here it does not
        # count, so it is set aside and measured again instead of ending the campaign
        return {}
    return got


def run_ok(d: Path) -> bool:
    """A finished run that counts: results readable and an error-rate row that reads 0.

    AIPerf writes that row on every run; without it nothing shows that every request
    succeeded, so the run is not counted (like an unreadable row) and a rerun measures it again.
    """
    got = run_results(d)
    return bool(got) and got.get("error_rate") == 0


def help_digest(text: str | None) -> str | None:
    """The sha256 of engine help text as saved (scrubbed, so a probed engine_flags.txt read back
    on a resume hashes the same); None without help text."""
    if text is None:
        return None
    return hashlib.sha256(SCRUB.text(text).encode("utf-8")).hexdigest()


def finite(value: Any) -> bool:
    """A real measurement: a number that is not NaN or infinite."""
    return isinstance(value, (int, float)) and math.isfinite(value)


def run_counts(d: Path, metric: str | None) -> bool:
    """run_ok, with a finite value of every metric compare_aiperf needs from each counted repeat
    (throughput, and with a target set, its metric; the same rule, so the two never disagree):
    a run without one decides nothing and is measured again rather than reused. Any other
    metric (ITL is absent for 1-token outputs) never stops a run from counting."""
    got = run_results(d)
    return run_ok(d) and all(finite(got.get(k)) for k in compare_aiperf.required_metrics(metric))


def kept_overloads(d: Path, st: dict[str, Any], repeats: int) -> set[str]:
    """The runs of every load where each repeat failed requests (a measured overload, as
    compare_aiperf reads it) and whose exports in D still show it: a rerun keeps them as they
    are instead of benchmarking them again. A load whose evidence is gone is not kept."""
    over = set(st.get("overloaded_runs", []))
    out: set[str] = set()
    for conc in {m[1] for r in over if (m := compare_aiperf.RUN_DIR.match(r))}:
        runs = [f"c{conc}-r{k}" for k in range(1, repeats + 1)]
        if all(r in over and requests_failed(d / r) for r in runs):
            out.update(runs)
    return out


def unsettled_runs(st: dict[str, Any], repeats: int) -> list[str]:
    """The failed runs a rerun must measure again: every one except those at a load where every
    repeat failed requests, which compare_aiperf reads as a measured overload. A failed run next
    to counted runs at the same load is a stray failure to repeat, as is a failed search."""
    over = set(st.get("overloaded_runs", []))
    out = []
    for r in st.get("failed_runs", []):
        m = compare_aiperf.RUN_DIR.match(r)
        if not (m and all(f"c{m[1]}-r{k}" in over for k in range(1, repeats + 1))):
            out.append(r)
    return out


def replica_restarts(replicas: list[dict[str, Any]]) -> dict[str, int]:
    """{each replica's identity (pod uid or container id, else name): its restarts}, summed over
    every container of the pod: the engagement check can read a NIM in a (restartable) init or
    ephemeral container, so its restarts count too."""
    out: dict[str, int] = {}
    for p in replicas:
        meta, status = p.get("metadata", {}), p.get("status", {})
        out[str(meta.get("uid") or meta.get("name"))] = sum(
            cs.get("restartCount", 0)
            for kind in ("containerStatuses", "initContainerStatuses", "ephemeralContainerStatuses")
            for cs in status.get(kind, []) or []
        )
    return out


def image_identity(records: list[Any]) -> str | None:
    """The immutable image the measured replicas ran, from their saved records (OUT/ID/*.pod.json):
    Docker's image ID of each container, or on Kubernetes the digest in the imageID of each pod's
    NIM container (kubectl's default container; the repository a node names it by is left out);
    None when any replica does not show one."""
    ids: set[str] = set()
    for rec in records:
        if isinstance(rec, list):  # Docker: the inspected containers
            got = [b.get("Image") if isinstance(b, dict) else None for b in rec] or [None]
        else:
            try:
                found = targets.pod_container(rec)
            except ValueError:
                return None
            got = [((found[2] if found else None) or {}).get("imageID")]
        if not all(isinstance(x, str) and x for x in got):
            return None
        ids.update(x.rsplit("@", 1)[-1] for x in got)
    return ",".join(sorted(ids)) or None


def publish(stage: Path, d: Path) -> None:
    """Move a validated attempt's files from STAGE to the run's names in D, the console log last
    (a stop in between leaves at most the export, which the shared read rule judges alone)."""
    for f in (compare_aiperf.CSV_NAME, compare_aiperf.LOG_NAME):
        if (stage / f).is_file():
            (stage / f).replace(d / f)
        else:
            (d / f).unlink(missing_ok=True)
    shutil.rmtree(stage, ignore_errors=True)


def set_aside_incomplete(d: Path) -> None:
    """Rename a run's results to *.incomplete: kept as evidence, never read as a result."""
    for f in (compare_aiperf.CSV_NAME, compare_aiperf.LOG_NAME):
        if (d / f).is_file():
            (d / f).replace(d / f"{f}.incomplete")


def requests_failed(d: Path) -> bool:
    """True only when the run shows failed requests: a readable, positive error rate. A missing
    or unreadable rate is missing evidence, to measure again, never a measured overload."""
    rate = run_results(d).get("error_rate")
    return rate is not None and math.isfinite(rate) and 0 < rate <= 100


def acceptance(before: dict[str, Any], after: dict[str, Any]) -> float | None:
    """Accepted / proposed draft tokens over the campaign's runs, summed over replicas."""
    drafts = accepted = 0.0
    for pod, a in after.items():
        b = before.get(pod)
        if isinstance(a, dict) and isinstance(b, dict):
            drafts += a["spec_draft_tokens"] - b["spec_draft_tokens"]
            accepted += a["spec_accepted_tokens"] - b["spec_accepted_tokens"]
    return round(accepted / drafts, 3) if drafts > 0 else None


def has_dataset(w: dict[str, Any]) -> bool:
    """True when the workload sends a dataset file or dataset arguments."""
    return bool(w.get("dataset") or w.get("dataset_args"))


def dataset_remote(w: dict[str, Any]) -> str:
    """Where the dataset file lives in the client pod; the hash keeps versions apart."""
    ds = w["dataset"]
    return f"/work/datasets/{ds['sha256'][:12]}-{Path(ds['file']).name}"


def ordered(c: dict[str, Any], only: list[str]) -> list[dict[str, Any]]:
    """Baseline first and alone, then the rest in file order."""
    cands = [x for x in c["candidates"] if not only or x["id"] in only or x["id"] == c["baseline"]]
    return sorted(cands, key=lambda x: x["id"] != c["baseline"])


def arg_pairs(args: list[str]) -> list[str]:
    """['--tensor-parallel-size', '2', '--enforce-eager'] -> ['--tensor-parallel-size 2', ...]."""
    out: list[str] = []
    for a in args:
        if a.startswith("-") or not out:
            out.append(a)
        else:
            out[-1] += f" {a}"
    return out


# an HTTP header that carries a credential ("Authorization" is not matched by SECRET_NAME)
SECRET_HEADER = re.compile(r"authoriz|cookie", re.IGNORECASE)


def header_names(value: str) -> list[str]:
    """The header names in one AIPerf -H/--header value, which AIPerf reads as 'Name: v', as a
    comma-separated 'A:1,B:2' list or as a JSON object. A value that starts like JSON but is not
    an object is returned whole, so all of its text is checked."""
    names = [value.partition(":")[0]]  # the first name, however the rest is written
    text = value.strip()
    if text.startswith("{"):
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            obj = None
        names += [str(k) for k in obj] if isinstance(obj, dict) else [text]
    else:
        names += [item.partition(":")[0] for item in text.split(",")]
    return names


# AIPerf options the runner sets on every run (and where it reads the export from): one in
# workload.dataset_args would change what is measured or point the fetch at an earlier run's file
OWNED_AIPERF_ARGS = (
    "--model", "--tokenizer", "--url", "--endpoint-type", "--streaming", "--concurrency",
    "--request-count", "--warmup-request-count", "--random-seed", "--artifact-dir",
    "--output-artifact-dir", "--profile-export-prefix", "--profile-export-file",
    # and their AIPerf aliases
    "-m", "--model-names", "-u", "--num-requests", "--num-warmup-requests", "--no-streaming",
    # the request path: the smoke test checked the endpoint type's own
    "--endpoint", "--custom-endpoint",
    # the profiles per run: the plan counts one per repeat
    "--num-profile-runs",
)  # fmt: skip


def owned_aiperf_args(args: list[str]) -> list[str]:
    """The runner-owned AIPerf options set in `args` (as `--opt value` or `--opt=value`)."""
    return sorted({o for a in args for o in OWNED_AIPERF_ARGS if a == o or a.startswith(o + "=")})


def secret_aiperf_args(args: list[str]) -> list[str]:
    """The AIPerf options in `args` that carry a credential value: --api-key X, or a -H/--header
    with any header (also inside a JSON object or a comma-separated list) that is Authorization,
    a cookie or a credential-named one. Only the option names are returned, never a value."""
    found: list[str] = []
    i = 0
    while i < len(args):
        # one option and the values after it, kept as separate arguments: -H takes a list, so
        # `-H X-Route:nim X-Api-Key:SECRET` holds two headers
        rest = []
        j = i + 1
        while j < len(args) and not args[j].startswith("-"):
            rest.append(args[j])
            j += 1
        flag, values = args[i], rest
        if flag.startswith("-H") and not flag.startswith("--") and len(flag) > 2:
            flag, values = "-H", [flag[2:], *rest]  # the attached form -HName:value
        elif flag.startswith("--") and "=" in flag:  # --header=Authorization: Bearer x
            flag, inline = flag.split("=", 1)
            values = [inline, *rest]
        if flag in ("-H", "--header"):
            # each value alone (a list of headers) and all of them joined (one spaced value,
            # or a JSON object split across arguments)
            names = [h for v in (" ".join(values), *values) for h in header_names(v)]
            if any(SECRET_HEADER.search(h) or targets.SECRET_NAME.search(h) for h in names):
                found.append(flag)
        elif targets.SECRET_FLAG.match(" ".join([args[i], *rest])):  # --api-key X, --api-key=X
            found.append(flag)
        i = j
    return found


def flag_values(args: list[str]) -> dict[str, str]:
    """{'--max-num-seqs': '64', '--enforce-eager': ''} from an argument list (also --flag=value)."""
    out: dict[str, str] = {}
    for pair in arg_pairs(args):
        flag, _, value = pair.partition(" ")
        if "=" in flag:  # --flag=value; never split an '=' inside the value itself
            flag, _, inline = flag.partition("=")
            value = f"{inline} {value}".strip()
        out[flag] = value.strip()
    return out


SPEC_FLAGS = ("--speculative-config", "--speculative-model", "--num-speculative-tokens")
SPEC_ENV = ("NIM_SPECDEC_ENABLE", "NIM_SPECDEC_ARGS", "NIM_DRAFT_MODEL_PATH")


def is_spec(cand: dict[str, Any]) -> bool:
    """True when the candidate turns on speculative decoding by flag or environment."""
    return any(f in flag_values(cand["args"]) for f in SPEC_FLAGS) or any(
        k in cand["env"] for k in SPEC_ENV
    )


def needs_accuracy_check(cand: dict[str, Any]) -> bool:
    """True for candidates that can change output quality: speculative or low precision."""
    f = flag_values(cand["args"])
    lowp = any(
        f.get(k, "").lower().startswith(("fp8", "nvfp4", "int8", "int4"))
        for k in ("--kv-cache-dtype", "--quantization", "--dtype")
    )
    return is_spec(cand) or lowp


def prune(c: dict[str, Any], cand: dict[str, Any]) -> str | None:
    """Reason to skip a candidate before any GPU time, or None. Pruned candidates are reported."""
    try:
        return _prune(c, cand)
    except ValueError as exc:
        return f"a value cannot be read ({exc}); write numbers in full, e.g. 32768"


def _prune(c: dict[str, Any], cand: dict[str, Any]) -> str | None:
    """The reason to skip a candidate before any GPU time, or None; may raise ValueError."""
    f = flag_values(cand["args"])
    w = c["workload"]
    for flag, value in f.items():
        if value.lower() in ("false", "true") and not flag.startswith("--no-"):
            return f"{flag} {value}: write a boolean as --FLAG or --no-FLAG, never true/false"
    tp = int(f.get("--tensor-parallel-size", "0") or 0)
    pp = int(f.get("--pipeline-parallel-size", "1") or 1)
    if tp and tp * pp != cand["gpus"]:
        return f"TP {tp} x PP {pp} does not equal the {cand['gpus']} GPUs per replica"
    mem = float(f.get("--gpu-memory-utilization") or cand["env"].get("NIM_KVCACHE_PERCENT") or 0)
    if mem > 0.95:
        return f"memory fraction {mem} leaves no activation headroom (0.95 at most)"
    mnt = int(f.get("--max-num-batched-tokens", "0") or 0)
    if mnt and mnt <= 2048 and mem >= 0.90:
        return f"batched tokens {mnt} with memory fraction {mem}: out-of-memory risk"
    mml = int(f.get("--max-model-len", "0") or cand["env"].get("NIM_MAX_MODEL_LEN") or 0)
    if mml and not has_dataset(w) and mml < w["isl"] + w["osl"] + 64:
        return f"max model length {mml} is below ISL + OSL + chat-template headroom"
    if is_spec(cand) and not has_dataset(w):
        return "speculative decoding needs a real text dataset (workload.dataset)"
    return None


def unsupported_flags(args: list[str], help_text: str, scrub: bool = False) -> list[str]:
    """Flags in `args` that the engine's `vllm serve --help=all` does not list. With `scrub`,
    help_text is the scrubbed copy (engine_flags.txt), so each flag is looked up as scrubbed
    too: a credential that overlaps a flag name masks it the same way in both."""
    missing = []
    for flag in flag_values(args):
        base = "--" + flag[5:] if flag.startswith("--no-") else flag
        look = SCRUB.text(base) if scrub else base
        if not re.search(rf"(?<![\w-]){re.escape(look)}(?![\w-])", help_text):
            missing.append(flag)
    return missing


def spent(start: float, slept: float) -> float:
    """Seconds a wait loop has used since `start` (time.monotonic()): wall time, so the commands
    it runs count too, and at least the time it slept."""
    return max(slept, time.monotonic() - start)


def request_count(conc: int) -> int:
    """How many requests one AIPerf run sends at a concurrency."""
    return max(200, 10 * conc)


def seed(conc: int, rep: int) -> int:
    """The random seed of one run, so each load and repeat differs but a rerun repeats it."""
    return 100 * conc + rep


RESTARTED_SINCE_CHECK = "a replica restarted or was replaced after its engagement check"
# repair passes a measured candidate gets for runs that do not count, before the gap is left
MAX_REPAIRS = 2

# ---------------------------------------------------------------- SLA search


def sla_search(
    passes: Callable[[int], bool | None], start: int, hi_cap: int, resolution_pct: float
) -> dict[str, Any]:
    """Find the highest concurrency that meets the target.

    Double from `start` while points pass (probing `hi_cap` last), or halve toward 1 when `start`
    fails; then bisect until the gap is within `resolution_pct` of the passing load (at least 1).
    `passes` returns None when a point could not be measured, which ends the search.

    It assumes latency grows with load, so no load above a failing one is probed: `complete`
    means the boundary below the first failing load was found, not that every higher load fails.
    """
    tried: dict[int, bool] = {}

    def probe(n: int) -> bool | None:
        """Measure one load once; None when it could not be measured."""
        if n not in tried:
            got = passes(n)
            if got is None:
                return None
            tried[n] = got
        return tried[n]

    lo: int | None = None
    hi: int | None = None
    first = probe(start)
    if first is None:
        return {"max_concurrency": None, "first_failing": None, "tried": tried, "complete": False}
    if first:
        lo, n = start, start
        while hi is None:
            nxt = min(n * 2, hi_cap)
            if nxt == n:
                break
            got = probe(nxt)
            if got is None:
                return {
                    "max_concurrency": lo,
                    "first_failing": None,
                    "tried": tried,
                    "complete": False,
                }
            if got:
                lo, n = nxt, nxt
            else:
                hi = nxt
    else:
        hi, n = start, start
        while lo is None and n > 1:
            n = max(1, n // 2)
            got = probe(n)
            if got is None:
                return {
                    "max_concurrency": None,
                    "first_failing": hi,
                    "tried": tried,
                    "complete": False,
                }
            if got:
                lo = n
            else:
                hi = n
    while lo is not None and hi is not None and hi - lo > max(1, int(lo * resolution_pct / 100)):
        mid = (lo + hi) // 2
        got = probe(mid)
        if got is None:
            return {"max_concurrency": lo, "first_failing": hi, "tried": tried, "complete": False}
        if got:
            lo = mid
        else:
            hi = mid
    return {"max_concurrency": lo, "first_failing": hi, "tried": tried, "complete": True}


# ---------------------------------------------------------------- failure classification


def classify(pods: list[dict[str, Any]], logs: dict[str, str] | str) -> tuple[str, str] | None:
    """Return (kind, reason) for a failed start, kind 'config' or 'intermittent'; None if healthy.

    OOMKilled, bad arguments or fatal engine errors in the log mean the configuration cannot run.
    Image pulls, scheduling and timeouts without such evidence are intermittent. `logs` maps each
    pod's name to its own log (a plain string is the log of every listed pod), so a fatal line
    counts only for the replica that logged it and only when that replica crashed.
    """
    # every replica is checked before an intermittent cause is returned: one replica's image
    # pull must not hide another's OOMKilled, whatever order they are listed in
    intermittent: tuple[str, str] | None = None
    for pod in pods:
        own = logs if isinstance(logs, str) else logs.get(pod.get("metadata", {}).get("name"), "")
        m = FATAL.search(own or "")
        for cs in pod.get("status", {}).get("containerStatuses", []) or []:
            term = (cs.get("lastState", {}) or {}).get("terminated") or {}
            wait = (cs.get("state", {}) or {}).get("waiting") or {}
            if term.get("reason") in CONFIG_REASONS:
                return "config", term["reason"]
            if wait.get("reason") in CONFIG_REASONS:
                return "config", wait["reason"]
            if wait.get("reason") in INTERMITTENT_REASONS:
                intermittent = intermittent or ("intermittent", wait["reason"])
                continue
            if wait.get("reason") == "CrashLoopBackOff" or cs.get("restartCount", 0) >= 2:
                if m:
                    return "config", m.group(0)
                intermittent = intermittent or ("intermittent", "CrashLoopBackOff")
                continue
            crashed = term or cs.get("restartCount", 0) or "terminated" in (cs.get("state") or {})
            if m and crashed:
                return "config", m.group(0)
    # a log line alone, in a pod that has not crashed, is not a failure (a warning can match)
    return intermittent


# ---------------------------------------------------------------- cluster access


# ---------------------------------------------------------------- the campaign


class Campaign:
    """One campaign run: deploys candidates in turn, measures them, and keeps resumable state."""

    # {name: each replica's identity and restarts when its engagement was checked}; a run only
    # counts on those same replicas
    engaged: dict[str, dict[str, int]] | None = None
    # why measure() last returned None: (kind, reason) for fail()
    crash: tuple[str, str] | None = None
    # the approved engine help text, unscrubbed (engine_flags.txt is the scrubbed copy)
    flags_text: str | None = None

    def __init__(self, c: dict[str, Any], out: Path, args: argparse.Namespace) -> None:
        """Open the run folder and log, and load the saved state when there is one."""
        self.c, self.out, self.args = c, out, args
        self.w = c["workload"]
        out.mkdir(parents=True, exist_ok=True)
        self.logf = (out / "campaign.log").open("a", encoding="utf-8")
        self.t: targets.Target = make_target(c, Path(args.campaign).resolve().parent, self.log)
        self.state_path = out / "campaign_state.json"
        self.state: dict[str, Any] = (
            json.loads(self.state_path.read_text())
            if self.state_path.is_file()
            else {"active_seconds": 0.0, "candidates": {}}
        )
        self.t0 = time.monotonic()
        self.current = ""  # the candidate being measured, removed if the client is lost
        self.current_cid = ""
        self.pidfile = PIDFILE.format(name=c["name"])
        self.requests_failed = False  # the last measure() failed only because requests failed
        self.search_crashed = False
        self.created_client = False
        self.client_released = False
        self.client_removed = False
        self.facts: dict[str, Any] = {"run_dir": str(out)}
        self.flags_path = out / "engine_flags.txt"
        self.adopt_engine_flags(engine_flags_text(args), args.engine_flags)
        self.salt_ok = bool(self.w["cache_salt"])

    def adopt_engine_flags(self, help_text: str | None, src: str | None = None) -> None:
        """Make OUT/engine_flags.txt match the approved plan: the --engine-flags file's bytes
        when the plan has one (SRC, its path, is never rewritten: the next run hashes it again);
        otherwise only flags this campaign probed are kept. A run without the --engine-flags
        file an earlier run used is refused in main (missing_engine_help)."""
        self.flags_text = help_text  # checked as approved: the scrubbed copy can differ
        if help_text is not None:
            same = src is not None and self.flags_path.is_file() and self.flags_path.samefile(src)
            if not same:
                # scrubbed like every result file: flag names survive, a credential never does
                self.write(self.flags_path, help_text)
            self.state["engine_flags_source"] = "file"
        elif self.flags_path.is_file() and self.state.get("engine_flags_source") is None:
            # its origin is unknown: never read as engine help, kept (never deleted, nor one set
            # aside before), probed again
            kept = self.flags_path.with_name("engine_flags.unrecorded.txt")
            n = 0
            while kept.exists():
                n += 1
                kept = self.flags_path.with_name(f"engine_flags.unrecorded.{n}.txt")
            self.flags_path.replace(kept)

    def missing_engine_help(self) -> bool:
        """True when earlier runs checked candidates against engine help this run lacks: an
        --engine-flags file this run was not given, or a probed engine_flags.txt since deleted.
        Deploying the rest unchecked could run a flag the engine does not list."""
        return engine_help_missing(self.state, self.flags_text is not None, self.flags_path)

    def engine_help(self) -> str | None:
        """The engine help text candidates are checked against: the approved or probed text as
        read, else a probed engine_flags.txt kept from an earlier run; None when there is none."""
        if self.flags_text is not None:
            return self.flags_text
        if self.flags_path.is_file():
            return self.flags_path.read_text(encoding="utf-8")
        return None

    # -- bookkeeping
    def write(self, path: Path, text: str) -> None:
        """Every result file goes through the scrubber: no credential is ever written."""
        path.write_text(SCRUB.text(text), encoding="utf-8")

    def write_json(self, path: Path, obj: Any, **kwargs: Any) -> None:
        """A JSON record, masked before it is encoded: a short credential that equals a number
        in it (a value of 1) never turns the file into invalid JSON."""
        path.write_text(SCRUB.dumps(obj, **kwargs), encoding="utf-8")

    def log(self, msg: str) -> None:
        """Write one masked, timestamped line to stderr and to campaign.log."""
        line = SCRUB.text(f"[{time.strftime('%H:%M:%S')}] {msg}")
        print(line, file=sys.stderr, flush=True)
        self.logf.write(line + "\n")
        self.logf.flush()

    @property
    def client(self) -> str:
        """The name of the AIPerf client (pod or container)."""
        return str(self.t.client)

    def save(self) -> None:
        """Write the state atomically, so a runner killed mid-write still resumes."""
        self.state["active_seconds"] = self.active()
        self.t0 = time.monotonic()
        tmp = self.state_path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            f.write(SCRUB.dumps(self.state, indent=1, sort_keys=True))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.state_path)

    def active(self) -> float:
        """Seconds of active time, counting earlier runs of this campaign."""
        return float(self.state["active_seconds"]) + time.monotonic() - self.t0

    def cand_state(self, cid: str) -> dict[str, Any]:
        """The saved state of one candidate, created as pending on first use."""
        st: dict[str, Any] = self.state["candidates"].setdefault(
            cid, {"status": "pending", "attempts": 0}
        )
        return st

    def gate(self, cid: str | None = None, cap: bool = True) -> str | None:
        """Deterministic stop gates, checked before each candidate and each load. The candidate
        cap never stops CID when it is already measured: it was counted, and a repair of its
        runs deploys no new candidate. Without CAP (each load of a candidate already admitted
        before it deployed) only the time and failure gates apply: a repair sets its status
        back to pending, and the cap must not then count it as a new candidate."""
        done = [
            v
            for v in self.state["candidates"].values()
            if v.get("status") in ("measured", "failed")
        ]
        counted = cid is not None and self.state["candidates"].get(cid, {}).get("status") == (
            "measured"
        )
        capped = cap and self.args.max_candidates and len(done) >= self.args.max_candidates
        if capped and not counted:
            return f"{len(done)} candidates run, the cap"
        if self.args.max_hours and self.active() > self.args.max_hours * 3600:
            return f"time budget of {self.args.max_hours} h spent"
        streak = 0
        for cid in self.state.get("order", []):
            s = self.state["candidates"].get(cid, {}).get("status")
            streak = streak + 1 if s == "failed" else 0 if s == "measured" else streak
        if streak >= self.args.max_consecutive_failures:
            return f"{streak} candidates in a row failed"
        return None

    # -- client
    def ensure_client(self) -> None:
        """Start or find the client, check AIPerf in it, and upload the dataset."""
        created = self.t.ensure_client()
        # the identity of the client this call created, read at creation: a later lookup could
        # see a same-named client someone else started meanwhile and record it as ours
        uid = self.t.created_uid if created else self.t.client_uid()
        if created:
            self.state["client_uid"] = uid
        # ownership is the client's identity, never a remembered flag: a same-named client that
        # someone else started after this campaign removed its own is never deleted
        self.created_client = bool(uid) and self.state.get("client_uid") == uid
        self.save()
        for _ in range(60):
            proc = self.t.client_exec("aiperf", "--version")
            if proc.returncode == 0:
                self.facts["aiperf_version"] = proc.stdout.strip().splitlines()[-1]
                self.upload_dataset()
                return
            time.sleep(10)
        raise KubeError(f"aiperf did not become available in {self.client}")

    def upload_dataset(self) -> None:
        """Copy workload.dataset into the client pod unless the same bytes are already there."""
        ds = self.w.get("dataset")
        if not ds:
            return
        remote = dataset_remote(self.w)
        check = (
            "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())"
        )
        if self.t.client_exec("python", "-c", check, remote).stdout.strip() == ds["sha256"]:
            return
        local = self.path(ds["file"])
        self.t.client_exec(
            "sh",
            "-c",
            f"mkdir -p /work/datasets && cat > {shlex.quote(remote)}",
            # the approved bytes, byte-exact: CRLF stays CRLF
            stdin=targets.read_input(local).decode("utf-8"),
            timeout=600,
        )
        got = self.t.client_exec("python", "-c", check, remote).stdout.strip()
        if got != ds["sha256"]:
            raise KubeError(f"dataset upload to {self.client}:{remote} did not verify")
        self.log(f"dataset {local.name} uploaded to {remote}")

    def workload_args(self) -> list[str]:
        """AIPerf's input arguments: the dataset file, extra dataset flags, or synthetic lengths."""
        w = self.w
        if w.get("dataset"):
            return [
                "--custom-dataset-type",
                w["dataset"]["type"],
                "--input-file",
                dataset_remote(w),
                *w["dataset_args"],
            ]
        if w["dataset_args"]:
            return list(w["dataset_args"])
        return ["--isl", str(w["isl"]), "--osl", str(w["osl"]), "--osl-stddev", "0"]

    def check_client(self) -> None:
        """Raise when the client no longer runs, so its loss is never blamed on a candidate."""
        self.t.check_client()

    def path(self, p: str) -> Path:
        """A path from the campaign file; a relative one is read from the campaign's folder."""
        q = Path(p)
        return q if q.is_absolute() else Path(self.args.campaign).resolve().parent / q

    def node_gpus(self) -> str:
        """The target's GPU facts, for the plan and the result."""
        return str(self.t.gpus())

    # -- one candidate
    def wait_ready(self, name: str, replicas: int) -> tuple[str, str] | None:
        """Wait for every replica; return a (kind, reason) failure or None when ready."""
        start, slept, step = time.monotonic(), 0, 20
        # the timeout is wall time: slow target reads and their retries use it up too
        while (waited := spent(start, slept)) < self.args.ready_timeout:
            if self.t.ready_count(name) == replicas:
                self.log(f"{name}: {replicas} replica(s) ready after {waited:.0f}s")
                return None
            pods = self.t.replicas(name)
            # each replica's own log: a healthy replica's warning never explains another's crash
            logs = {
                p["metadata"]["name"]: "".join(
                    self.t.logs(p["metadata"]["name"], previous=prev, tail=200)
                    for prev in (False, True)  # the crashed container's log holds the cause
                )
                for p in pods
            }
            bad = classify(pods, logs)
            if bad:
                return bad
            if any(
                "terminated" in (cs.get("state") or {})
                for p in pods
                for cs in p.get("status", {}).get("containerStatuses", []) or []
            ):
                return "intermittent", "the server exited during startup"
            if spent(start, slept) >= self.args.ready_timeout:
                break  # the reads used up the budget: no sleep and no further read
            time.sleep(step)
            slept += step
        return "intermittent", f"not ready after {spent(start, slept):.0f}s"

    def remove(self, name: str, uid: str | None = None) -> None:
        """Remove this campaign's candidate and confirm it is gone; stop if it will not go.

        Its replicas must be gone too: a deleted NIMService's pods can outlive it (also on a
        resume, when only the pods are left), and they still hold GPUs. With `uid` (the identity
        recorded with its cleanup mark) only that resource is removed.
        """
        if not self.t.remove(name, uid) and self.t.exists(name):
            # not this campaign's (it may have replaced ours): never touched, and never taken
            # for a removal, since this campaign's replicas may still hold GPUs
            raise ForeignCleanupError(
                f"{self.t.what} {name} exists and was not created by this campaign; this "
                "campaign's own removal of it stays pending"
            )
        for _ in range(30):
            if not self.t.replicas(name):
                self.log(f"{name}: removed")
                return
            time.sleep(10)
        # never deploy the next candidate while this one may still hold GPUs
        raise KubeError(f"{name} is still running 5 minutes after removal")

    def counters(self, name: str) -> dict[str, Any]:
        """Per-replica vLLM request and prefix-cache counters, read from the client pod."""
        out: dict[str, Any] = {}
        code = (
            "import sys,urllib.request as u\n"
            "t=u.urlopen('http://'+sys.argv[1]+':8000/metrics',timeout=10).read().decode()\n"
            "g=lambda k: sum(float(l.split()[-1]) for l in t.splitlines() if l.startswith(k))\n"
            "print(g('vllm:request_success_total'), g('vllm:prefix_cache_hits_total'),"
            " g('vllm:prefix_cache_queries_total'), g('vllm:spec_decode_num_draft_tokens_total'),"
            " g('vllm:spec_decode_num_accepted_tokens_total'))\n"
        )
        for p in self.t.replicas(name):
            ip = p.get("status", {}).get("podIP")
            proc = self.t.client_exec("python", "-c", code, str(ip))
            try:
                req, hits, queries, drafts, accepted = (float(x) for x in proc.stdout.split())
                out[p["metadata"]["name"]] = {
                    "requests": req,
                    "prefix_hits": hits,
                    "prefix_queries": queries,
                    "spec_draft_tokens": drafts,
                    "spec_accepted_tokens": accepted,
                }
            except ValueError:
                out[p["metadata"]["name"]] = "unavailable"
        return out

    def engagement(self, cand: dict[str, Any], pod: str, spec: Path, log: Path) -> dict[str, Any]:
        """Run check_nim_engagement.py on the saved pod spec and startup log."""
        checks = [f"--arg={a}" for a in arg_pairs(cand["args"])]
        # a credential's value is never read back or compared: that would put it in results
        checks += [
            f"--env={k}={v}" for k, v in cand["env"].items() if not targets.SECRET_NAME.search(k)
        ]
        if cand.get("engagement_pattern"):
            # settings the launch line cannot show (e.g. NIM_* variables): a log line proves them
            checks.append(f"--log-pattern={cand['engagement_pattern']}")
        if not checks:
            checks = ["--require-launch"]
        proc = subprocess.run(
            [
                sys.executable,
                str(ENGAGEMENT),
                *self.t.engagement_args(pod),
                "--spec-file",
                str(spec),
                "--log-file",
                str(log),
                *checks,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
            check=False,
        )
        try:
            out: dict[str, Any] = json.loads(proc.stdout)
        except json.JSONDecodeError:
            out = {"ok": False, "state": "unreadable", "facts": {}, "errors": []}
        out["exit"] = proc.returncode
        return out

    def smoke(self, name: str, salt: bool = False) -> bool:
        """Send one small chat request to the NIM; True when it answers 200."""
        code = (
            "import json,sys,urllib.request as u\n"
            "b='http://'+sys.argv[1]+':8000'\n"
            "q={'model':sys.argv[2],'max_tokens':8,\n"
            "   'messages':[{'role':'user','content':'Say hi'}]}\n"
            "q.update({'cache_salt':'smoke'} if sys.argv[3]=='1' else {})\n"
            "d=json.dumps(q).encode()\n"
            "r=u.urlopen(u.Request(b+'/v1/chat/completions',d,"
            "{'Content-Type':'application/json'}),timeout=120)\n"
            "print(r.status)\n"
        )
        self.check_client()
        proc = self.t.client_exec("python", "-c", code, name, self.w["model"], "1" if salt else "0")
        return proc.stdout.strip().endswith("200")

    def measure(self, cid: str, name: str, conc: int, rep: int) -> bool | None:
        """One AIPerf run; True when results were saved, None when the server crashed."""
        d = self.out / cid / f"c{conc}-r{rep}"
        metric = (self.c.get("sla") or {}).get("metric")
        if run_counts(d, metric):
            return True
        # anything saved that does not count (no finite metric, no error rate, an attempt cut
        # short) is set aside: kept as evidence, never read as part of the new measurement
        set_aside_incomplete(d)
        d.mkdir(parents=True, exist_ok=True)
        remote = f"/work/{self.c['name']}/{cid}/c{conc}-r{rep}"
        cmd = [
            "aiperf",
            "profile",
            "--model",
            self.w["model"],
            "--tokenizer",
            self.w["tokenizer"],
            "--url",
            f"http://{name}:8000",
            "--endpoint-type",
            self.w["endpoint_type"],
            "--streaming",
            "--concurrency",
            str(conc),
            "--request-count",
            str(request_count(conc)),
            "--warmup-request-count",
            str(conc),
            *self.workload_args(),
            "--random-seed",
            str(seed(conc, rep)),
            "--artifact-dir",
            remote,
        ]
        for extra in self.w["extra_inputs"]:
            cmd += ["--extra-inputs", extra]
        if self.salt_ok:
            # a fresh salt per run keeps one repeat from warming the prefix cache for the next
            cmd += ["--extra-inputs", f"cache_salt:{name}-c{conc}-r{rep}"]
        restarts_before = self.restarts(name)
        engaged = (self.engaged or {}).get(name)
        if engaged is not None and restarts_before != engaged:
            # the replicas serving now are not the ones whose engagement was checked; no run
            # was under way, so it is no verdict on the setting: a resume redeploys and checks
            # engagement again, and reuses the runs already counted
            self.crash = ("intermittent", RESTARTED_SINCE_CHECK)
            self.log(f"{cid}: {RESTARTED_SINCE_CHECK}")
            return None
        log, csv_path = d / compare_aiperf.LOG_NAME, d / compare_aiperf.CSV_NAME
        # an attempt's files are written here first, and moved to the names a resume reuses only
        # once they count (or show the overload): a stop in between leaves nothing to reuse
        stage = d / ".attempt"
        for attempt in (1, 2):
            # per attempt: a stale True from attempt 1 must not describe a crash in attempt 2
            self.requests_failed = False
            self.check_client()
            self.log(f"{cid} c={conc} r={rep} attempt {attempt}")
            shutil.rmtree(stage, ignore_errors=True)
            code, console = self.run_detached(cmd, remote)
            if attempt == 2:
                # keep the failed attempt's evidence under names compare_aiperf never reads
                if log.is_file():
                    log.rename(d / "aiperf.attempt1.log")
                if csv_path.is_file():
                    csv_path.rename(d / "profile_export_aiperf.attempt1.csv")
            if self.restarts(name) != restarts_before:
                self.crash = ("config", "a replica restarted under load")
                self.log(f"{cid}: a replica restarted or was replaced during the run")
                # evidence only: a result file written before this check could be reused
                self.write(d / "aiperf.restarted.log", console)
                return None
            stage.mkdir(parents=True, exist_ok=True)
            self.write(stage / compare_aiperf.LOG_NAME, console)
            if code == 0 and compare_aiperf.read_log(stage / compare_aiperf.LOG_NAME):
                csv = self.t.client_exec("cat", f"{remote}/{compare_aiperf.CSV_NAME}")
                if csv.returncode == 0 and csv.stdout:
                    self.write(stage / compare_aiperf.CSV_NAME, csv.stdout)
                if run_counts(stage, metric):
                    publish(stage, d)
                    return True
                if run_ok(stage):
                    want = " or ".join(compare_aiperf.required_metrics(metric))
                    self.log(f"{cid} c={conc} r={rep}: no finite {want}; not counted")
                elif requests_failed(stage):
                    self.log(f"{cid} c={conc} r={rep}: requests failed; not counted")
                    self.requests_failed = True
                    publish(stage, d)  # compare_aiperf reads the overload from its export
                    continue
                else:
                    self.log(f"{cid} c={conc} r={rep}: no readable error rate; not counted")
            # evidence only, under a name nothing reads as a result (attempt 1's as it was)
            (stage / compare_aiperf.CSV_NAME).unlink(missing_ok=True)
            kept = "aiperf.attempt1.log" if attempt == 1 else "aiperf.log.incomplete"
            if (stage / compare_aiperf.LOG_NAME).is_file():
                (stage / compare_aiperf.LOG_NAME).replace(d / kept)
            shutil.rmtree(stage, ignore_errors=True)
        if not self.requests_failed:
            # a rejected attempt's console must not count on the next invocation without a new
            # measurement: only a run with failed requests keeps its export (the overload)
            set_aside_incomplete(d)
        return False

    def run_detached(self, cmd: list[str], remote: str, timeout: float = 7200) -> tuple[int, str]:
        """Run AIPerf in the client pod, detached, and poll for its exit code.

        A kubectl exec stream can drop (API server resets); the benchmark must outlive it, so the
        exec only starts it and later calls read its exit code and console output from files.
        """
        token = str(int(time.time()))  # per-attempt files: no attempt reads another's output
        out, rc = f"{remote}.{token}.console.log", f"{remote}.{token}.exit"
        start = self.t.client_exec("python", "-c", LAUNCH, self.pidfile, out, rc, *cmd)
        begin, slept = time.monotonic(), 0.0
        # wall time: each poll's exec (and its retries) counts toward the timeout, not only sleeps
        while (waited := spent(begin, slept)) < timeout:
            got = self.t.client_exec("cat", rc)
            if got.returncode == 0 and got.stdout.strip().lstrip("-").isdigit():
                console = self.t.client_exec("cat", out, timeout=300)
                return int(got.stdout.strip()), console.stdout
            self.check_client()  # a lost client stops the campaign; the run is redone on resume
            if waited >= 2 * POLL_SECONDS and self.t.client_exec("test", "-e", out).returncode:
                # no console file: it never started; never wait out the whole timeout for it
                self.stop_aiperf()
                return 127, f"AIPerf did not start: {(start.stdout + start.stderr).strip()[-400:]}"
            time.sleep(POLL_SECONDS)
            slept += POLL_SECONDS
        self.stop_aiperf()  # never leave a load generator running against the next candidate
        return 124, f"no exit code after {timeout:.0f}s"

    def stop_aiperf(self) -> None:
        """Stop any AIPerf run this campaign started in the client."""
        self.t.client_exec("python", "-c", LAUNCH, self.pidfile)

    def restarts(self, name: str) -> dict[str, int]:
        """Each replica's identity (pod uid or container id, else name) and its restarts.

        Compared before and after a run: a replacement pod or container starts at restartCount 0,
        so equal totals can hide a run that spanned two; a changed identity shows it.
        """
        return replica_restarts(self.t.replicas(name))

    def point_passes(self, cid: str, name: str, conc: int) -> bool | None:
        """Measure one load for the SLA search; True if it meets the target, None if it cannot."""
        stop = self.gate(cap=False)
        if stop:  # the time budget covers the search too
            self.state["stop_condition"] = stop
            return None
        sla = self.c["sla"]
        need = compare_aiperf.required_metrics(sla["metric"])
        reps = range(1, self.w["repeats"] + 1)
        failed = [requests_failed(self.out / cid / f"c{conc}-r{rep}") for rep in reps]
        if conc in self.w.get("concurrency", []):
            # a fixed load is a measured overload only when every repeat failed requests; a
            # stray failed repeat next to counted ones is a gap the repair measures again,
            # never a miss that ends the search early
            if all(failed):
                return False
            if any(failed):
                return None
        elif any(failed):
            # an earlier probe's miss at a search-only load: never benchmarked again
            return False
        vals: list[float] = []
        for rep in reps:
            d = self.out / cid / f"c{conc}-r{rep}"
            if run_ok(d) and not run_counts(d, sla["metric"]):
                # a saved run without a finite value of a required metric (missing, NaN or inf)
                # decides nothing: set it aside (kept as evidence) so this probe measures it
                # again instead of reusing it on every rerun
                set_aside_incomplete(d)
            ok = self.measure(cid, name, conc, rep)
            if ok is None:
                self.search_crashed = True  # a replica restarted under load
                return None
            if not ok:
                # failed requests mean this load is beyond the server: it misses the target
                return False if self.requests_failed else None
            got = run_results(d)
            if "error_rate" not in got or not all(finite(got.get(k)) for k in need):
                # not measured (NaN or inf too, which max() could skip): no evidence either
                # way, measured again on a rerun
                return None
            if got["error_rate"] != 0:
                return False
            vals.append(got[sla["metric"]])
        return max(vals) <= float(sla["max_ms"])

    def uncounted_runs(self, cid: str, st: dict[str, Any]) -> list[str]:
        """A measured candidate's fixed-load runs that do not count, apart from a measured
        overload whose exports still show it."""
        metric = (self.c.get("sla") or {}).get("metric")
        keep = kept_overloads(self.out / cid, st, self.w["repeats"])
        runs = [
            f"c{conc}-r{rep}"
            for conc in self.w["concurrency"]
            for rep in range(1, self.w["repeats"] + 1)
        ]
        return [r for r in runs if r not in keep and not run_counts(self.out / cid / r, metric)]

    def check_image(self, cid: str, st: dict[str, Any], pods: list[str]) -> None:
        """Record the image this deployment runs (image_identity of its saved records). Runs
        saved on another image, or on one that was not recorded, never count with runs on this
        one: their folders are set aside whole (kept as evidence, as cC-rR.superseded, never
        read as a run: a new search may not probe the same loads) and measured again, with the
        search and the run lists; so are the records of replicas no longer serving."""
        folder = self.out / cid
        image = image_identity([json.loads((folder / f"{p}.pod.json").read_text()) for p in pods])
        if image is not None and image == st.get("image_id"):
            return
        runs = [r for r in folder.glob("c*-r*") if compare_aiperf.RUN_DIR.match(r.name)]
        names = (compare_aiperf.CSV_NAME, compare_aiperf.LOG_NAME)
        if any((r / f).is_file() for r in runs for f in names):
            self.log(f"{cid}: its saved runs are not known to be on image {image}; measuring again")
        for r in runs:
            aside = r.with_name(f"{r.name}.superseded")
            for n in range(2, 1000):
                if not aside.exists():
                    break
                aside = r.with_name(f"{r.name}.superseded-{n}")  # never over an earlier one
            r.rename(aside)
        for old in folder.glob("*.pod.json"):
            if old.name.removesuffix(".pod.json") not in pods:
                old.replace(old.with_name(old.name + ".superseded"))
        for k in ("sla_search", "failed_runs", "overloaded_runs"):
            st.pop(k, None)
        st["image_id"] = image
        self.save()

    def fingerprint(self, cand: dict[str, Any]) -> str:
        """What a candidate's results depend on; a resume never mixes results across a change."""
        keys = ("gpus", "replicas", "args", "env", "engagement_pattern")
        setup = {
            "candidate": {k: cand.get(k) for k in keys},
            "workload": self.w,
            "target": self.c["target"],
            "deploy": targets.redact(self.t.describe(cand)),  # a rotated key is not a change
        }
        return hashlib.sha256(json.dumps(setup, sort_keys=True).encode()).hexdigest()[:16]

    def measured_config(self, cand: dict[str, Any]) -> dict[str, Any]:
        """The settings one candidate is deployed with, kept in the state so that
        write_recommended.py can tell a campaign file changed after the measurement."""
        config: dict[str, Any] = {k: cand.get(k) for k in ("gpus", "replicas", "args", "env")}
        config["workload"] = self.w  # what was sent: a later edit would rank stale exports
        if self.c["target"] == "docker":
            d = self.c["docker"]
            config["docker"] = {k: d.get(k) for k in ("image", "env", "env_from_host", "env_file")}
        return config

    def run_candidate(self, cand: dict[str, Any]) -> None:
        """Deploy one candidate, measure every load and repeat, and record the result."""
        cid, st = cand["id"], self.cand_state(cand["id"])
        name = f"{self.c['name_prefix']}-{cid}"
        fp = self.fingerprint(cand)
        if st.get("fingerprint") and st["fingerprint"] != fp:
            raise ResumeMismatchError(
                f"{cid} changed since its results in {self.out} were measured"
            )
        if st.get("cleanup_pending"):
            # its removal failed last time: free its GPUs before this or any later candidate
            # deploys (its results are kept; raises if it still will not go)
            self.clear_one(cid, st)
        redo = (
            unsettled_runs(st, self.w["repeats"])
            if st["status"] == "measured" and st.get("failed_runs")
            else []
        )
        if st["status"] == "measured" and not redo:
            # a saved run that does not count (no error rate, no finite required metric, its
            # export gone) leaves a gap compare_aiperf reports: measure it again now
            redo = self.uncounted_runs(cid, st)
        st.pop("unrepaired", None)
        if redo and st.get("repairs", 0) >= MAX_REPAIRS:
            # a run that never counts (no finite metric, no error-rate row) is not redeployed on
            # every rerun: the gap stays, and the comparison excludes the case or calls it
            # provisional
            self.log(f"{cid}: {redo} still do not count after {MAX_REPAIRS} repairs; left")
            self.facts.setdefault("unrepaired", {})[cid] = redo
            st["unrepaired"] = redo  # compare() leaves a candidate other than the baseline out
            redo = []
        if redo:
            # a load where every repeat failed requests is a measured overload: kept as it is
            # (the measuring pass below skips it), never benchmarked again
            st["repairs"] = st.get("repairs", 0) + 1
            self.log(f"{cid}: re-measuring {redo} that failed before")
            kept = sorted(kept_overloads(self.out / cid, st, self.w["repeats"]))
            st["status"], st["failed_runs"], st["overloaded_runs"] = "pending", kept, list(kept)
        if (
            st["status"] == "rejected"
            and (known := self.engine_help()) is not None
            and st.get("flags_sha256") != help_digest(known)
        ):
            # rejected against other engine help than this plan's: decide it again (a rejected
            # candidate has no measurements to lose). Without help it stays rejected: never
            # deployed unchecked
            self.log(f"{cid}: the engine help changed since it was rejected; checking it again")
            st["status"] = "pending"
            for k in ("reason", "flags_sha256"):
                st.pop(k, None)
        if st["status"] in ("measured", "pruned", "rejected") or (
            st["status"] == "failed" and st.get("kind") == "config"
        ):
            self.log(f"{cid}: already {st['status']}, skipped")
            return
        reason = prune(self.c, cand)
        if reason:
            st.update(status="pruned", reason=reason)
            self.log(f"{cid}: pruned before deploy: {reason}")
            self.save()
            return
        help_text = self.engine_help()
        if help_text is not None:
            # a probed engine_flags.txt read back on a resume is the scrubbed copy (flags_text
            # is set only when this run adopted or probed the help text)
            missing = unsupported_flags(cand["args"], help_text, scrub=self.flags_text is None)
            if missing:
                # rejected, not failed: it never used a GPU, so it does not trip the failure gate
                st.update(
                    status="rejected",
                    reason=f"engine does not list {missing}",
                    flags_sha256=help_digest(help_text),  # the help text that decided it
                )
                self.log(f"{cid}: not deployed, unsupported flags {missing}")
                self.save()
                return
        if needs_accuracy_check(cand):
            st["needs_accuracy_check"] = True
        owner = self.t.owner(name)
        if owner is not None and owner != self.c["name"]:
            raise RefusedError(f"{self.t.what} {name} exists and was not created by this campaign")
        manifest = self.t.describe(cand)
        self.current, self.current_cid = name, cid
        st["fingerprint"] = fp
        st["measured"] = self.measured_config(cand)
        (self.out / cid).mkdir(parents=True, exist_ok=True)
        # what was deployed, with any literal credential masked (the live apply keeps it)
        self.write_json(self.out / cid / "deploy.json", targets.redact(manifest), indent=1)
        self.log(f"===== {cid}: {cand['gpus']} GPU(s) x {cand['replicas']}, args {cand['args']}")
        failure: tuple[str, str] | None = None
        for attempt in (1, 2):
            st["attempts"] = st.get("attempts", 0) + 1
            st.pop("deployed_uid", None)  # an earlier deploy's: never this one's
            failure = self.t.deploy(cand)
            # which resource this deploy made or reused, from its own create (the uid the API
            # server returned, or the Id `docker run -d` printed): retire() and fail() remove
            # only that one, never whatever holds the name by then. Saved before anything else
            st["deployed_uid"] = self.t.deployed
            self.save()
            failure = failure or self.wait_ready(name, cand["replicas"])
            if failure is None or failure[0] == "config" or attempt == 2:
                break
            self.log(f"{cid}: intermittent start failure ({failure[1]}), retrying once")
            self.retire(name, st)  # marked pending first, like every other removal
        if failure:
            self.fail(cid, name, failure)
            return
        replicas = self.t.replicas(name)
        pods = [p["metadata"]["name"] for p in replicas]
        if len(pods) != cand["replicas"]:
            # an empty list must never pass the engagement check by default
            self.fail(cid, name, ("intermittent", f"listed {len(pods)} of {cand['replicas']} pods"))
            return
        engaged = {}
        st["nodes"] = sorted({p.get("spec", {}).get("nodeName") or "?" for p in replicas})
        want = self.c.get("node") if self.c["target"] == "nim-operator" else None
        if want and st["nodes"] != [want]:
            # measured anywhere else, it would use GPUs the user did not approve
            self.fail(cid, name, ("config", f"scheduled on {st['nodes']}, not {want}"))
            return
        for pod in pods:
            log = self.out / cid / f"{pod}.startup.log"
            spec = self.out / cid / f"{pod}.pod.json"
            self.write(log, self.t.logs(pod))
            self.write_json(spec, self.t.spec(pod, allow=list(cand["env"])))
            engaged[pod] = self.engagement(cand, pod, spec, log)
        self.check_image(cid, st, pods)
        st["engagement"] = {
            p: {"state": e.get("state"), "launch_line": e.get("facts", {}).get("launch_line")}
            for p, e in engaged.items()
        }
        fell = sorted(
            {f for e in engaged.values() for f in e.get("facts", {}).get("engine_fallbacks", [])}
        )
        if cid == self.c["baseline"]:
            self.state["baseline_fallbacks"] = fell
        else:
            # only what this setting changed: the untuned NIM's own fallbacks are not news
            fell = [f for f in fell if f not in self.state.get("baseline_fallbacks", [])]
        if fell and cid != self.c["baseline"]:
            # the result measures the setting plus these engine changes; say so next to it
            st["engine_fallbacks"] = fell
            self.log(f"{cid}: engine fallbacks: {fell}")
        if any(e["exit"] != 0 for e in engaged.values()):
            unconfirmed = all(e.get("state") == "unconfirmed" for e in engaged.values())
            self.fail(
                cid,
                name,
                ("unconfirmed", "no engine evidence; add engagement_pattern to the candidate")
                if unconfirmed
                else ("config", "settings not engaged"),
            )
            return
        # the replicas whose engagement was just checked: a run on any other does not count
        self.engaged = {name: replica_restarts(replicas)}
        self.save()  # early, so a restart keeps the engagement and client ownership
        if not self.smoke(name):
            self.fail(cid, name, ("intermittent", "smoke test failed"))
            return
        if self.salt_ok and not self.smoke(name, salt=True):
            self.salt_ok = False
            self.facts["cache_salt"] = "rejected by the server; runs use distinct seeds only"
            self.log("server rejected cache_salt; continuing without it")
        if not self.flags_path.is_file():
            helped = self.t.exec(pods[0], "vllm", "serve", "--help=all", timeout=300)
            if helped.returncode == 0 and "--max-num-seqs" in helped.stdout:
                self.write(self.flags_path, helped.stdout)
                self.flags_text = helped.stdout
                self.state["engine_flags_source"] = "probed"
                self.save()  # at once: a probed file the state does not name is set aside
                self.log(f"engine flags saved from {pods[0]}")
            else:
                self.facts["engine_flags"] = "could not run vllm serve --help=all; flags unverified"
        before = self.counters(name)
        crashed = False
        self.crash = None
        # this pass decides every run again (counted ones are reused): failures recorded by an
        # earlier, interrupted pass must not outlive it and exclude a now complete case. A
        # measured overload whose exports still show it is kept, never benchmarked again
        keep = sorted(kept_overloads(self.out / cid, st, self.w["repeats"]))
        # listed now, not when the loop reaches them: a stop before that never drops them
        st["failed_runs"], st["overloaded_runs"] = list(keep), list(keep)
        for conc in self.w["concurrency"]:
            stop = self.gate(cap=False)  # admitted before it deployed: never a new candidate
            if stop:
                self.state["stop_condition"] = stop
                break
            for rep in range(1, self.w["repeats"] + 1):
                if f"c{conc}-r{rep}" in keep:
                    continue
                got = self.measure(cid, name, conc, rep)
                if got is None:
                    crashed = True
                    break
                if not got:
                    st.setdefault("failed_runs", []).append(f"c{conc}-r{rep}")
                    if self.requests_failed:  # the NIM answered, but some requests failed
                        st.setdefault("overloaded_runs", []).append(f"c{conc}-r{rep}")
            if crashed:
                break
            self.save()
        if (
            not crashed
            and self.c.get("sla", {}).get("search")
            and not self.state.get("stop_condition")
        ):
            s = self.c["sla"]["search"]
            self.search_crashed = False
            result = sla_search(
                lambda n: self.point_passes(cid, name, n),
                s["start"],
                s["max"],
                s["resolution_pct"],
            )
            result["tried"] = {str(k): v for k, v in result["tried"].items()}
            st["sla_search"] = result
            self.log(f"{cid}: highest load within target {result['max_concurrency']}")
            crashed = self.search_crashed  # a replica restart during the search fails it
            if not crashed and not result.get("complete") and not self.state.get("stop_condition"):
                # a probe failed: keep the candidate resumable; its finished runs are reused
                st.setdefault("failed_runs", []).append("sla-search")
        after = self.counters(name)
        if not crashed and self.restarts(name) != (self.engaged or {}).get(name):
            # read from replicas other than the measured ones: no evidence about this candidate.
            # Every run was checked on the measured replicas, so the results stand
            after = {}
            self.log(f"{cid}: {RESTARTED_SINCE_CHECK}; its counters are not recorded")
        st["counters"] = {"before": before, "after": after}
        rate = acceptance(before, after)
        if rate is not None:
            st["spec_acceptance"] = rate  # explains a result; never ranks candidates
        self.write_manifest(cand, st)
        if crashed:
            self.fail(cid, name, self.crash or ("config", "a replica restarted under load"))
            return
        st["status"] = "stopped" if self.state.get("stop_condition") else "measured"
        self.retire(name, st)

    def clear_pending(self) -> None:
        """Remove every candidate whose removal failed in an earlier run, before anything else.

        Runs before --only and the stop gates are applied, so a rerun always frees their GPUs,
        even when that candidate is not selected or a gate stops the campaign at once. Their
        results are kept; raises like remove() if one still will not go.
        """
        for cid, st in self.state["candidates"].items():
            if st.get("cleanup_pending"):
                self.clear_one(cid, st)

    def clear_one(self, cid: str, st: dict[str, Any]) -> None:
        """Remove one candidate left by a failed removal, by the name it was saved under.

        A marker saved under another name (name_prefix changed since) is never removed by the
        new name: while that resource still exists (a NIMService can have no pods for a while,
        and create them later) or any replica of it runs, the run stops; once it is gone (the
        user removed it) the marker is cleared.
        """
        saved = st["cleanup_pending"]
        name = f"{self.c['name_prefix']}-{cid}"
        where = st.get("cleanup_target")
        left = saved if isinstance(saved, str) else name
        here = self.target_id() if where is not None else ""
        if here.endswith(UNKNOWN_DAEMON):
            # never a verdict on the marker: the daemon did not answer (retried on a resume)
            raise KubeError("cannot read the Docker daemon's identity (docker info)")
        if self.c.get("target") == "docker" and (
            where is None or "#" not in str(where) or str(where).endswith(UNKNOWN_DAEMON)
        ):
            # recorded without the daemon it ran on (an earlier runner, or docker info failed):
            # DOCKER_HOST or the docker context may have pointed anywhere, so nothing here
            # proves it gone
            raise PendingCleanupError(
                f"{left} was left from the last run on {where} (its removal failed), and the "
                "Docker daemon it ran on was not recorded",
                f"On every Docker daemon this campaign may have used (DOCKER_HOST, docker "
                f"context), run `docker ps -a --filter label={LABEL}={self.c.get('name')}` and "
                f"remove {left} if it is there, once the user agrees; then delete cleanup_pending, cleanup_target "
                f"and cleanup_uid of {cid} in campaign_state.json and rerun.",
            )
        if where is not None and where != here:
            # the resource is on the target it was deployed to; looking for it here would find
            # nothing and clear the marker while it still holds GPUs there
            raise PendingCleanupError(
                f"{left} was left from the last run on {where} (its removal failed), and this "
                f"run targets {here}"
            )
        if isinstance(saved, str) and saved != name:
            if not self.t.exists(saved) and not self.t.replicas(saved):
                self.log(f"{cid}: {saved}, left from the last run, is gone")
                self.unmark(st)
                return
            raise PendingCleanupError(
                f"{saved} was left from the last run (its removal failed) and name_prefix has "
                f"changed since, so this run would not remove it"
            )
        uid = st.get("cleanup_uid")
        owner = f"{LABEL}={self.c.get('name')}"
        if not isinstance(uid, str):
            # which resource was left is not recorded (an earlier runner, or it could not be
            # read): the one under its name now may be another run's with the same campaign
            # name, so it is never removed; only nothing there at all clears the mark
            if not self.t.exists(name) and not self.t.replicas(name):
                self.log(f"{cid}: {name}, left from the last run, is gone")
                self.unmark(st)
                return
            raise PendingCleanupError(
                f"{name} was left from the last run (its removal failed), and the state does "
                "not record which resource that was, so this run does not remove it",
                f"Check that {name} carries the label {owner} and was created by this "
                f"campaign's last run (its owner and creation time); if so, remove it once the user agrees. "
                f"Then delete cleanup_pending, cleanup_target and cleanup_uid of {cid} in "
                "campaign_state.json and rerun.",
            )
        now = self.t.ident(name)
        if now != uid and (uid == "" and now is not None or self.t.kept(name, uid)):
            # not there when marked, yet something is now; or the recorded one lives on under
            # another name (a renamed container): never removed by the runner
            raise PendingCleanupError(
                f"{name} was left from the last run (its removal failed), and what this run "
                "finds is not the resource recorded then, so it does not remove it",
                (
                    f"Leave {name} alone: nothing had that name when the removal was marked, so "
                    "it is not the resource recorded and may be someone else's. Confirm nothing "
                    "of this campaign's last run still runs (pods labelled "
                    f"app={name}, or containers labelled {owner})"
                    if uid == ""
                    else f"Leave {name} alone: it is not the {self.t.what} recorded and may be "
                    f"someone else's. The recorded one ({uid}) still exists under another name: "
                    f"if it carries the label {owner} (`docker ps -a --filter id={uid}`), remove "
                    "it once the user agrees, and confirm it is gone"
                )
                + f"; then delete cleanup_pending, cleanup_target and cleanup_uid of {cid} in "
                "campaign_state.json and rerun.",
            )
        if now is not None and now != uid:
            # the recorded one is gone and another resource holds the name (another run's,
            # perhaps another user's with the same campaign name): never touched
            if self.t.strays(name):
                raise PendingCleanupError(
                    f"{name}, left from the last run, was replaced by another resource of that "
                    "name; replicas under that name still run and may be the old one's",
                    f"Leave {name} alone unless it is yours. Once no replica of the old one "
                    f"runs (pods labelled app={name} created before the current {name}), "
                    f"delete cleanup_pending, cleanup_target and cleanup_uid of {cid} in "
                    "campaign_state.json and rerun.",
                )
            self.log(f"{cid}: {name}, left from the last run, is gone (replaced; not touched)")
            self.unmark(st)
            return
        self.log(f"{cid}: removing it, left from the last run")
        self.remove(name, uid)
        self.unmark(st)

    def unmark(self, st: dict[str, Any]) -> None:
        """Clear a candidate's cleanup mark and save the state."""
        for key in ("cleanup_pending", "cleanup_target", "cleanup_uid"):
            st.pop(key, None)
        self.save()

    @staticmethod
    def deployed(st: dict[str, Any]) -> str | None:
        """The uid or container Id this run's deploy recorded for the candidate ('' when it
        made nothing), None when none was recorded (a resume then never removes it). Never read
        by name: a same-named resource made since the deploy is not the one deployed."""
        uid = st.get("deployed_uid")
        return uid if isinstance(uid, str) else None

    def retire(self, name: str, st: dict[str, Any]) -> None:
        """Save a finished candidate's result, then remove it.

        The result is saved first, so a failed removal never costs a benchmark; the removal is
        marked pending until it succeeds, so a rerun frees the GPUs before deploying anything.
        """
        if st.get("cleanup_pending") != name or "cleanup_uid" not in st:
            # which resource, recorded by its deploy: a rerun removes only that one
            st["cleanup_uid"] = self.deployed(st)
        st["cleanup_pending"] = name  # the name to remove on a rerun, checked before removing
        st["cleanup_target"] = self.target_id()  # and where: a rerun elsewhere must not clear it
        self.save()
        # none recorded: nothing is removed here (never whatever holds the name); the mark stays
        uid = st["cleanup_uid"] or ""
        self.remove(name, uid)
        if uid and self.t.kept(name, uid):
            # renamed away since it was deployed (Docker): it still holds GPUs, the mark stays
            raise PendingCleanupError(
                f"{name} ({uid[:12]}) was renamed while it ran and was not removed",
                f"Find it with `docker ps -a --filter id={uid}`, remove it once the user agrees if it is this "
                "campaign's, then delete cleanup_pending, cleanup_target and cleanup_uid of "
                "that candidate in campaign_state.json and rerun.",
            )
        self.unmark(st)

    def target_id(self) -> str:
        """Where this campaign deploys: the kube context and namespace, or the Docker host and
        its daemon's ID (DOCKER_HOST or the docker context picks the daemon, not the host
        alone); `#?` when the daemon does not say."""
        if self.c.get("target") == "docker":
            host = self.c.get("docker", {}).get("ssh") or "local"
            got = self.t.docker("info", "--format", "{{.ID}}", timeout=30)
            daemon = got.stdout.strip() if got.returncode == 0 else ""
            return f"docker:{host}#{daemon}" if daemon else f"docker:{host}{UNKNOWN_DAEMON}"
        return f"kube:{self.c.get('context')}/{self.c.get('namespace')}"

    def running(self) -> list[str]:
        """This campaign's candidates that still exist, and the client."""
        names = [f"{self.c['name_prefix']}-{x['id']}" for x in self.c["candidates"]]
        alive = [n for n in names if self.t.owner(n) == self.c["name"]]
        return alive + ([self.client] if self.client and not self.client_removed else [])

    def note_left_running(self, *names: str) -> None:
        """Remember resources that could not be removed, to list them in the result."""
        left = self.facts.setdefault("left_running", [])
        left += [n for n in names if n and n not in left]

    def free_current(self) -> None:
        """Remove the candidate being measured, freeing its GPUs; list it if that fails."""
        if not self.current:
            return
        try:
            # only the resource this run deployed: never a same-named one made since
            st = self.state.get("candidates", {}).get(getattr(self, "current_cid", ""), {})
            self.remove(self.current, self.deployed(st) or "")
        except (KubeError, RefusedError) as exc:
            self.log(f"could not remove {self.current}: {exc}")
            self.note_left_running(self.current)

    def release_client(self) -> None:
        """Remove the client this campaign created (once), unless --keep-client.

        Called before the result is printed, so a client that could not be removed is listed
        in left_running of that result.
        """
        if self.client_released:
            return
        self.client_released = True
        if self.client and self.created_client and not self.args.keep_client:
            try:
                # only the client this campaign created: a same-named one that replaced it
                # during the run (another uid or container id) belongs to someone else
                uid = self.state.get("client_uid")
                if self.t.client_uid() == uid:
                    self.t.remove_client(uid or None)
                else:
                    self.log(f"the client {self.client} was replaced or is gone; not removing it")
                self.client_removed = True
                self.state.pop("client_uid", None)
            except KubeError as exc:  # the target is unreachable: say what may still run
                self.log(f"could not remove the client: {exc}")
                self.note_left_running(self.client)

    def stop(
        self,
        state: str,
        next_: str,
        errors: list[dict[str, str]],
        code: int,
    ) -> NoReturn:
        """End a stopped campaign: release the client, account for what still runs, then emit."""
        self.release_client()
        try:
            self.note_left_running(*self.running())
        except KubeError as exc:
            # the target cannot be read: list what may still hold GPUs rather than nothing
            self.log(f"could not list what is still running: {exc}")
            client = self.client if self.created_client and not self.client_removed else ""
            self.note_left_running(self.current, client)
        self.save()
        emit(False, state, next_, self.facts, errors, code)

    def fail(self, cid: str, name: str, failure: tuple[str, str]) -> None:
        """Mark a candidate failed, save its replica logs, and remove it."""
        st = self.cand_state(cid)
        st.update(status="failed", kind=failure[0], reason=failure[1])
        # pending before the diagnostics: a failed log read must not leave it running unmarked
        st.update(cleanup_uid=self.deployed(st))  # the one a rerun may remove (retire keeps it)
        st.update(cleanup_pending=name, cleanup_target=self.target_id())
        self.save()
        self.log(f"{cid}: FAILED ({failure[0]}): {failure[1]}")
        for p in self.t.replicas(name):
            pod = p["metadata"]["name"]
            (self.out / cid).mkdir(parents=True, exist_ok=True)
            self.write(self.out / cid / f"{pod}.failure.log", self.t.logs(pod, tail=400))
        self.retire(name, st)

    def write_manifest(self, cand: dict[str, Any], st: dict[str, Any]) -> None:
        """Write the candidate's manifest.json with credentials masked."""
        manifest = {
            "workload": self.w,
            "aiperf_version": self.facts.get("aiperf_version"),
            # where the pods actually ran, not where the campaign asked them to
            "node": ",".join(st.get("nodes", [])) or self.c.get("node"),
            "client": self.client,
            "cache_salt_used": self.salt_ok,
            "candidate": cand,
            "engagement": st.get("engagement"),
        }
        # the candidate's own env can hold a literal credential: never write its value
        self.write_json(
            self.out / cand["id"] / "manifest.json", targets.redact(manifest), indent=1
        )


class ForeignCleanupError(RefusedError):
    """A same-named resource this campaign did not create stands where its removal is pending."""


class ResumeMismatchError(Exception):
    """A candidate or the workload changed since the results in --out were measured."""


class PendingCleanupError(Exception):
    """A candidate whose removal failed cannot be removed from here: saved under a name this
    campaign no longer uses, or on another target. FIX, when given, replaces the usual fix."""

    def __init__(self, message: str, fix: str | None = None) -> None:
        super().__init__(message)
        self.fix = fix


# ---------------------------------------------------------------- main


def compare(c: dict[str, Any], out: Path, states: dict[str, Any]) -> dict[str, Any]:
    """Run compare_aiperf.py over the results; unfinished or partly measured cases are excluded."""
    cmd = [
        sys.executable,
        str(HERE / "compare_aiperf.py"),
        str(out),
        "--baseline",
        c["baseline"],
        "--report",
        str(out / "final" / "before_after.md"),
    ]
    for cand in c["candidates"]:
        cmd += ["--gpus", f"{cand['id']}={cand['gpus'] * cand['replicas']}"]
        st = states.get(cand["id"], {})
        missing = [r for r in st.get("failed_runs", []) if r not in st.get("overloaded_runs", [])]
        unrepaired = st.get("unrepaired") and cand["id"] != c["baseline"]
        if st.get("status") != "measured" or missing or unrepaired:
            # a failed, stopped or partly measured case must never be ranked. A load where
            # requests failed (overload, normal for an untuned baseline at the top load) is not
            # missing: compare lists it as run_had_request_errors and the case stays in
            cmd += ["--exclude", cand["id"]]
    cmd += ["--loads", ",".join(str(x) for x in c["workload"]["concurrency"])]
    # the repeats every load was measured with: a measured overload needs all of them to fail
    cmd += ["--min-repeats", str(c["workload"]["repeats"])]
    if c.get("sla"):
        cmd += ["--sla", f"{c['sla']['metric']}<={c['sla']['max_ms']}"]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
            check=False,
        )
    except subprocess.TimeoutExpired:
        # after the measurements: never end on a traceback; the results stay for a retry
        fix = f"The measurements in {out} are kept. Retry only the report with: {shlex.join(cmd)}"
        return {"ok": False, "errors": [err("compare_failed", "compare_aiperf.py timed out", fix)]}
    try:
        res: dict[str, Any] = json.loads(proc.stdout)
    except json.JSONDecodeError:
        fix = f"Run it alone to see the error: {shlex.join(cmd)}"
        res = {"ok": False, "errors": [err("compare_failed", proc.stderr[-300:], fix)]}
    return res


# ---------------------------------------------------------------- one runner, supervised resume

LOCK_NAME = ".run_campaign.lock"
SUPERVISOR_ENV = "NIM_TUNE_SUPERVISOR_PID"
# A run that stopped on one of these can resume with the same command once the target answers:
# a dropped VPN or API server, a lost client pod, an unexpected error (all keep finished runs).
TRANSIENT_CODES = frozenset({"kubectl_failed", "docker_failed", "client_lost", "unexpected_error"})
SUPERVISOR_ONLY = {
    "--supervise": 0,
    "--keep-awake": 0,
    "--max-resumes": 1,
    "--resume-wait": 1,
    "--max-wait-hours": 1,
}
LOCK_FDS: dict[str, int] = {}  # this process's lock files, kept open: the flock lives with the fd
RUNNING: dict[str, Any] = {"child": None, "signal": None}  # the supervisor's attempt and stop


def read_lock(path: Path) -> dict[str, Any]:
    """Who wrote the lock file (pid, host, plan id), or {} when it is empty or unreadable."""
    try:
        held = json.loads(path.read_text() or "{}")
    except (OSError, ValueError):
        return {}
    return held if isinstance(held, dict) else {}


def refuse_lock(path: Path, held: dict[str, Any]) -> NoReturn:
    """Stop: another runner works in this folder."""
    emit(
        False,
        "refused",
        "ask_user",
        {"lock": str(path), "held_by": held},
        [
            err(
                "already_running",
                f"another run_campaign.py (pid {held.get('pid')} on {held.get('host')}, plan "
                f"{held.get('plan_id')}) is working in {path.parent}",
                "Wait for it to finish, or stop it first (kill PID; a supervisor stops its run "
                "too) and rerun the same command (it resumes). Never start a second copy. If that "
                f"runner is gone on another host, delete {path}.",
            )
        ],
        EXIT_REFUSED,
    )


def acquire_lock(out: Path, plan: str) -> Path | None:
    """Claim the run folder for this process; refuse while another runner works in it.

    Two runners in one folder stop each other's AIPerf and write the same state files. The claim is
    an flock on OUT/.run_campaign.lock, which the OS drops when its process ends, so a crashed
    runner or a reused process id never blocks a rerun and two starters cannot both win. A lock
    held by this process's supervisor is shared (returns None). A lock file written on another
    host (a shared folder) is never taken over.
    """
    out.mkdir(parents=True, exist_ok=True)
    path = out / LOCK_NAME
    host = socket.gethostname()
    while True:
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            held = read_lock(path)
            supervisor = os.environ.get(SUPERVISOR_ENV)
            if held.get("host") == host and supervisor == str(held.get("pid")) == str(
                os.getppid()
            ):
                return None  # our own supervisor holds the folder for us
            refuse_lock(path, held)
        try:
            same_file = os.fstat(fd).st_ino == os.stat(path).st_ino
        except FileNotFoundError:
            same_file = False
        if not same_file:  # the holder removed the file between our open and our lock
            os.close(fd)
            continue
        held = read_lock(path)
        if held.get("host") not in (None, host):
            os.close(fd)
            refuse_lock(path, held)
        os.ftruncate(fd, 0)
        record = {"pid": os.getpid(), "host": host, "plan_id": plan, "started": time.time()}
        os.write(fd, json.dumps(record).encode())
        LOCK_FDS[str(path)] = fd
        atexit.register(release_lock, path, os.getpid())
        return path


def release_lock(path: Path, pid: int) -> None:
    """Remove the lock if this process still holds it, then let the claim go."""
    fd = LOCK_FDS.pop(str(path), None)
    if read_lock(path).get("pid") == pid:
        path.unlink(missing_ok=True)
    if fd is not None:
        os.close(fd)


def child_argv(argv: list[str]) -> list[str]:
    """The run's own arguments: the command line without the supervisor-only options."""
    out, skip = [], 0
    for tok in argv:
        if skip:
            skip -= 1
            continue
        name = tok.split("=", 1)[0]
        if name in SUPERVISOR_ONLY:
            skip = SUPERVISOR_ONLY[name] if "=" not in tok else 0
            continue
        out.append(tok)
    return out


def awake_command(pid: int) -> list[str] | None:
    """A command that keeps this machine from sleeping until process PID ends, if the host has one.
    It watches the supervisor, so the machine stays awake between attempts too, and each attempt
    stays a direct child of the supervisor (which the shared lock checks)."""
    if sys.platform == "darwin" and shutil.which("caffeinate"):
        return ["caffeinate", "-is", "-w", str(pid)]
    if shutil.which("systemd-inhibit") and shutil.which("tail"):
        return ["systemd-inhibit", "--what=sleep:idle", "--why=nim-tune campaign",
                "tail", f"--pid={pid}", "-f", "/dev/null"]  # fmt: skip
    return None


def start_awake(log: Callable[[str], None]) -> subprocess.Popen[bytes] | None:
    """Start the keep-awake process for this supervisor's whole life."""
    cmd = awake_command(os.getpid())
    if cmd is None:
        log("supervisor: --keep-awake has no effect here (no caffeinate or systemd-inhibit)")
        return None
    return subprocess.Popen(
        cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


class StopRequestedError(Exception):
    """The supervisor got SIGTERM or SIGINT while no attempt was running."""


def on_stop_signal(signum: int, _frame: Any) -> None:
    """Stop the supervised run: pass the signal to the attempt, or end the wait between attempts.
    The attempt stops at once; finished runs stay for a resume."""
    RUNNING["signal"] = signum
    child = RUNNING["child"]
    if child is None or child.poll() is not None:
        raise StopRequestedError
    try:
        os.killpg(child.pid, signum)  # the attempt and the kubectl or docker calls it started
    except (ProcessLookupError, PermissionError):
        child.send_signal(signum)


def target_reachable(c: dict[str, Any], campaign_dir: Path) -> bool:
    """True when the target answers: the Kubernetes API server, or the Docker daemon."""
    if c["target"] == "docker":
        t = make_target(c, campaign_dir, lambda m: None)
        return t.docker("version", "--format", "{{.Server.Version}}", timeout=30).returncode == 0
    kubectl = os.environ.get("NIM_TUNE_KUBECTL", "kubectl")
    argv = [kubectl, "--context", c["context"], "--request-timeout=15s", "get", "--raw", "/version"]
    return targets.run_cmd(argv, None, 30).returncode == 0


def run_child(cmd: list[str], env: dict[str, str] | None = None) -> tuple[int, str]:
    """Run one supervised attempt; its progress goes to stderr, its contract JSON to stdout.

    It inherits this process's environment (the supervisor sets SUPERVISOR_ENV in it first).
    """
    # its own process group, so a stop reaches the commands it started too
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, env=env, start_new_session=True)
    RUNNING["child"] = proc
    try:
        stdout, _ = proc.communicate()
    finally:
        RUNNING["child"] = None
    return proc.returncode, stdout


def stopped_result(why: str) -> dict[str, Any]:
    """The supervisor's result after it was stopped on purpose."""
    return {
        "ok": False,
        "state": "stopped",
        "next": "ask_user",
        "facts": {},
        "errors": [err("stopped", why, "Rerun the same command to resume; finished runs stay.")],
    }


def supervise(argv: list[str], args: argparse.Namespace, c: dict[str, Any], plan: str) -> NoReturn:
    """Run the approved campaign until it ends, resuming after transient stops.

    The supervisor holds the folder lock, so no second runner can start between attempts. Each
    attempt is the same approved command; a rerun resumes and keeps finished runs. Any other stop
    (a failed baseline, a refused plan, resume_mismatch, all candidates done) is final.
    """
    campaign_dir = args.campaign.resolve().parent
    if not target_reachable(c, campaign_dir):
        # a wrong context, a missing CLI or no network at the start is a setup problem: say so
        # now instead of waiting forever for a target that was never there
        emit(
            False,
            "error",
            "fix_errors",
            errors=[
                err(
                    "target_unreachable",
                    "the target does not answer (kubectl get --raw /version, or docker version)",
                    "Check the context, the kubectl or docker binary and the network or VPN, "
                    "then start again. --supervise waits out outages only after a good start.",
                )
            ],
            code=EXIT_EXTERNAL,
        )
    lock = acquire_lock(args.out, plan)
    log_path = args.out / "campaign.log"

    def log(msg: str) -> None:
        line = SCRUB.text(f"[{time.strftime('%H:%M:%S')}] {msg}")  # like every campaign.log line
        print(line, file=sys.stderr, flush=True)
        with log_path.open("a") as f:
            f.write(line + "\n")

    awake = start_awake(log) if args.keep_awake else None
    cmd = [sys.executable, str(Path(__file__).resolve()), *child_argv(argv)]
    # each attempt inherits this one name, nothing is copied: it tells the attempt that the
    # folder lock it meets is its own supervisor's
    previous_env = os.environ.get(SUPERVISOR_ENV)
    os.environ[SUPERVISOR_ENV] = str(os.getpid())
    env = None
    history: list[dict[str, Any]] = []
    resumes = 0
    RUNNING.update(child=None, signal=None)
    previous = {s: signal.signal(s, on_stop_signal) for s in (signal.SIGTERM, signal.SIGINT)}

    def finish(payload: dict[str, Any], code: int) -> NoReturn:
        for s, handler in previous.items():
            signal.signal(s, handler)
        if previous_env is None:
            os.environ.pop(SUPERVISOR_ENV, None)
        else:
            os.environ[SUPERVISOR_ENV] = previous_env
        if awake is not None:
            awake.terminate()
        facts = dict(payload.get("facts") or {})
        facts["supervisor"] = {"resumes": resumes, "attempts": history}
        if lock is not None:
            release_lock(lock, os.getpid())  # before the result, not only at interpreter exit
        emit(
            bool(payload.get("ok")),
            str(payload.get("state")),
            str(payload.get("next")),
            facts,
            payload.get("errors") or [],
            code,
        )

    log(f"supervisor {os.getpid()}: plan {plan}, up to {args.max_resumes} resumes")
    last_unexpected = False
    try:
        while True:
            start, slept = time.monotonic(), 0
            while not target_reachable(c, campaign_dir):
                # wall time: each reachability probe (up to 30 s) counts, not only the sleeps
                waited = int(spent(start, slept))
                if waited >= args.max_wait_hours * 3600:
                    log(f"supervisor: target not reachable for {waited // 60} min; giving up")
                    unreachable = err(
                        "target_unreachable",
                        f"the target did not answer for {waited // 60} min",
                        "Fix the context, credentials or network, then rerun the same command "
                        "(it resumes; finished runs stay).",
                    )
                    finish({"state": "error", "next": "fix_errors", "errors": [unreachable]},
                           EXIT_EXTERNAL)  # fmt: skip
                if slept % 600 == 0:  # every tenth probe
                    log(f"supervisor: target not reachable; waiting ({waited // 60} min so far)")
                time.sleep(60)
                slept += 60
            code, stdout = run_child(cmd, env)
            if RUNNING["signal"] or code < 0:
                # a stop on purpose (kill of either process), or the system killed the attempt
                # (out of memory): never resumed by itself
                sig = RUNNING["signal"] or -code
                history.append({"exit": code, "state": "stopped", "codes": ["stopped"]})
                log(f"supervisor: the attempt ended on signal {sig}; not resumed")
                finish(stopped_result(f"the run was stopped by signal {sig}"), EXIT_PRECONDITION)
            try:
                payload = json.loads(stdout.strip().splitlines()[-1])
            except (IndexError, ValueError):
                payload = {
                    "ok": False,
                    "state": "crashed",
                    "next": "fix_errors",
                    "facts": {},
                    "errors": [err("unexpected_error", f"no result (exit {code})", "Rerun.")],
                }
            codes = sorted({e.get("code", "?") for e in payload.get("errors", [])})
            history.append({"exit": code, "state": payload.get("state"), "codes": codes})
            unexpected = "unexpected_error" in codes
            # an unexpected error is retried once: twice in a row is a bug, not an outage
            transient = bool(codes) and set(codes) <= TRANSIENT_CODES
            repeated = unexpected and last_unexpected
            last_unexpected = unexpected
            if transient and not repeated and resumes < args.max_resumes:
                resumes += 1
                log(f"supervisor: stopped on {codes}; resume {resumes} after the target answers")
                time.sleep(args.resume_wait)
                continue
            if repeated:
                log("supervisor: an unexpected error twice in a row; not resumed again")
            elif transient:
                log(f"supervisor: stopped on {codes}; {args.max_resumes} resumes used, giving up")
            else:
                log(f"supervisor: done ({payload.get('state')}, {codes or 'no errors'})")
            finish(payload, code)
    except StopRequestedError:
        sig = RUNNING["signal"]
        history.append({"exit": None, "state": "stopped", "codes": ["stopped"]})
        log(f"supervisor: stopped by signal {sig} between attempts")
        finish(stopped_result(f"the supervisor was stopped by signal {sig}"), EXIT_PRECONDITION)


def main(argv: list[str] | None = None) -> None:
    """Plan with --dry-run, or run the approved campaign, and print the contract JSON."""
    parser = ContractParser(
        description="Deploy, verify and benchmark NIM candidates one at a time "
        "(NIM Operator or Docker).",
        epilog="Example: python3 scripts/run_campaign.py campaign.json --out runs/qwen7b --dry-run",
        allow_abbrev=False,  # --supervise must be spelled out, or it would reach the attempt too
    )
    parser.add_argument("campaign", type=Path, help="campaign file (JSON), see campaign.md")
    parser.add_argument("--out", type=Path, required=True, help="run folder for all results")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="render and server-dry-run every manifest; change nothing",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="ID",
        help="run only these candidates (the baseline always runs)",
    )
    parser.add_argument(
        "--max-hours",
        type=float,
        default=0.0,
        help="stop after this much active time, kept across resumes (0: no cap)",
    )
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=2,
        help="stop after this many failed candidates in a row (default 2)",
    )
    parser.add_argument(
        "--ready-timeout",
        type=int,
        default=2700,
        help="seconds to wait for every replica (default 2700)",
    )
    parser.add_argument("--keep-client", action="store_true", help="leave the client running")
    parser.add_argument(
        "--max-candidates", type=int, default=0, help="stop after this many candidates (0: no cap)"
    )
    parser.add_argument(
        "--approved",
        metavar="PLAN_ID",
        help="the plan_id from --dry-run, after the user approved that plan (required to run)",
    )
    parser.add_argument(
        "--engine-flags",
        help="saved `vllm serve --help=all` output; without it the baseline pod is probed",
    )
    parser.add_argument(
        "--supervise",
        action="store_true",
        help="keep the approved campaign going unattended: resume after a lost target, client or "
        "unexpected error once the target answers again (not part of the plan id)",
    )
    parser.add_argument(
        "--max-resumes",
        type=int,
        default=20,
        help="with --supervise: give up after this many resumes (default 20)",
    )
    parser.add_argument(
        "--resume-wait",
        type=int,
        default=30,
        help="with --supervise: seconds to wait before a resume (default 30)",
    )
    parser.add_argument(
        "--keep-awake",
        action="store_true",
        help="with --supervise: keep this machine awake (caffeinate on macOS, systemd-inhibit "
        "on Linux) while the campaign runs",
    )
    parser.add_argument(
        "--max-wait-hours",
        type=float,
        default=4.0,
        help="with --supervise: give up when the target does not answer for this long (default 4)",
    )
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv)

    targets.freeze_inputs()  # the plan's hash and the run use the same bytes of every file
    c, errors = load_campaign(args.campaign)
    if errors:
        emit(False, "error", "fix_errors", errors=errors, code=EXIT_INPUT)
    # the secrets this campaign passes from the host are masked too, read by name
    d = c.get("docker") or {}
    SCRUB.add_env([*d.get("env_from_host", []), *d.get("client_env_from_host", [])])
    unknown = [i for i in args.only if i not in {x["id"] for x in c["candidates"]}]
    if unknown:
        emit(
            False,
            "error",
            "fix_errors",
            errors=[
                err("unknown_candidate", f"--only {unknown}", "Use ids from the campaign file.")
            ],
            code=EXIT_INPUT,
        )
    flags_bytes = b""
    if args.engine_flags:
        with contextlib.suppress(OSError):
            flags_bytes = Path(args.engine_flags).read_bytes()
    # an empty file would make every flag "unsupported" in the live run, not in the dry run
    if args.engine_flags and not flags_bytes.strip():
        emit(
            False,
            "error",
            "fix_errors",
            errors=[
                err(
                    "bad_usage",
                    f"--engine-flags {args.engine_flags} is not a file or is empty",
                    "Pass the saved `vllm serve --help=all` output, or leave --engine-flags out.",
                )
            ],
            code=EXIT_INPUT,
        )
    cands = ordered(c, args.only)
    options = plan_options(args, parser)
    plan = plan_id(args.campaign, options)

    if args.dry_run:
        try:  # the run would refuse this plan: never shown for approval
            saved = json.loads((args.out / "campaign_state.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            saved = {}
        saved = saved if isinstance(saved, dict) else {}
        if engine_help_missing(saved, bool(args.engine_flags), args.out / "engine_flags.txt"):
            src = saved["engine_flags_source"]
            emit(False, "refused", "fix_errors", errors=[engine_help_error(args.out, src)],
                 code=EXIT_REFUSED)  # fmt: skip
        t = make_target(c, args.campaign.resolve().parent, lambda m: print(m, file=sys.stderr))
        try:
            rendered: dict[str, Any] = {}
            help_text = engine_flags_text(args)
            # the client pod's create, checked by the API server (RBAC, PodSecurity, quota)
            # before approval; a client that exists is reused, so there is nothing to check
            client_checked = t.dry_run_client()
            for cand in cands:
                reason = prune(c, cand)
                missing = unsupported_flags(cand["args"], help_text) if help_text else []
                if reason or missing:
                    rendered[cand["id"]] = {
                        "skipped": reason or f"engine does not list {missing}",
                    }
                    continue
                name = f"{c['name_prefix']}-{cand['id']}"
                owner = t.owner(name)
                if owner is not None and owner != c["name"]:
                    emit(
                        False,
                        "refused",
                        "ask_user",
                        errors=[
                            err(
                                "foreign_resource",
                                f"{t.what} {name} already exists",
                                "Pick another name_prefix or ask its owner.",
                            )
                        ],
                        code=EXIT_REFUSED,
                    )
                t.dry_run(cand)
                rendered[cand["id"]] = {
                    "name": name,
                    "gpus_total": cand["gpus"] * cand["replicas"],
                    "args": cand["args"],
                    "env": targets.redact({"env": cand["env"]})["env"],
                    "needs_accuracy_check": needs_accuracy_check(cand),
                }
        except KubeError as exc:
            emit(
                False,
                "error",
                "fix_errors",
                errors=[err("dry_run_failed", str(exc), "Fix the campaign file or the target.")],
                code=EXIT_EXTERNAL,
            )
        w = c["workload"]
        runnable = [x for x in rendered.values() if "skipped" not in x]
        runs = len(runnable) * len(w["concurrency"]) * w["repeats"]
        emit(
            True,
            "planned",
            "ask_user",
            {
                "target": c["target"],
                "candidates": rendered,
                "order": [x["id"] for x in cands],
                "aiperf_runs_fixed_loads": runs,
                "sla_search": bool(c.get("sla", {}).get("search")),
                "engine_flags": "checked"
                if help_text
                else "probed from the untuned NIM at run time",
                "creates": t.created_resources([x["name"] for x in runnable]),
                "client_checked": client_checked,
                "deletes": "only what it creates",
                "plan_id": plan,
                "plan_options": options,
                "run_with": f"--approved {plan}",
            },
        )

    if args.approved != plan:
        # nothing changes on a cluster or host until the user approved this exact plan
        emit(
            False,
            "refused",
            "ask_user",
            {"plan_id": plan, "plan_options": options},
            [
                err(
                    "not_approved",
                    "no --approved, or the campaign file or the run options (--only, "
                    "--max-candidates, --max-hours, --max-consecutive-failures, "
                    "--ready-timeout, --engine-flags and that file) changed since its plan was "
                    "shown",
                    "Run --dry-run with the options you will run with, show the plan to the "
                    "user, and after their yes rerun with the same options and "
                    "--approved PLAN_ID from the dry run.",
                )
            ],
            EXIT_REFUSED,
        )

    if args.supervise:
        supervise(argv, args, c, plan)
    acquire_lock(args.out, plan)
    camp = Campaign(c, args.out, args)
    camp.state["order"] = [x["id"] for x in cands]
    camp.state.pop("stop_condition", None)  # each invocation checks the gates afresh
    errors = []
    try:
        camp.clear_pending()  # whatever --only selects and whether or not a gate stops the run
        if camp.missing_engine_help():
            # refused before any deploy: no candidate runs unchecked, nothing is reopened
            source = camp.state["engine_flags_source"]
            camp.facts["engine_flags_source"] = source
            camp.stop(
                "refused", "fix_errors", [engine_help_error(args.out, source)], EXIT_REFUSED
            )
        camp.ensure_client()
        camp.facts["gpus_at_start"] = camp.node_gpus()
        for cand in cands:
            stop = camp.gate(cand["id"])
            if stop:
                camp.state["stop_condition"] = stop
                camp.log(f"stop: {stop}")
                break
            camp.run_candidate(cand)
            camp.save()
            if camp.state.get("stop_condition"):
                break
            if cand["id"] == c["baseline"] and camp.cand_state(cand["id"])["status"] != "measured":
                # nothing can be compared without it; never spend GPU time on the rest
                camp.state["stop_condition"] = "baseline not measured"
                camp.log("stop: baseline not measured")
                break
    except PendingCleanupError as exc:
        camp.stop(
            "refused",
            "ask_user",
            [
                err(
                    "cleanup_pending",
                    str(exc),
                    exc.fix
                    or "Restore the earlier name_prefix and target (for Docker also DOCKER_HOST "
                    "or the docker context) and rerun so the runner removes it, or remove it "
                    "yourself after checking it is this campaign's, then rerun.",
                )
            ],
            EXIT_REFUSED,
        )
    except ResumeMismatchError as exc:
        camp.stop(
            "refused",
            "ask_user",
            [
                err(
                    "resume_mismatch",
                    str(exc),
                    "Use a new --out for the changed campaign, or restore the original setting.",
                )
            ],
            EXIT_REFUSED,
        )
    except ForeignCleanupError as exc:
        fix = (
            "Ask its owner to remove it; never remove it yourself. Keep name_prefix (a cleanup "
            "mark under this name stops a rerun with another one), then rerun the same command."
        )
        camp.stop("refused", "ask_user", [err("foreign_resource", str(exc), fix)], EXIT_REFUSED)
    except RefusedError as exc:
        camp.stop(
            "refused",
            "ask_user",
            [err("foreign_resource", str(exc), "Pick another name_prefix or ask its owner.")],
            EXIT_REFUSED,
        )
    except ClientLostError as exc:
        camp.log(f"stop: {exc}")
        camp.free_current()  # the candidate stays pending for a rerun
        camp.stop(
            "client_lost",
            "fix_errors",
            [
                err(
                    "client_lost",
                    str(exc),
                    f"{camp.t.client_fix()}; finished runs are kept and the interrupted"
                    " candidate restarts.",
                )
            ],
            EXIT_EXTERNAL,
        )
    except KubeError as exc:
        camp.log(f"stop: {exc}")
        camp.stop(
            "error",
            "fix_errors",
            [
                err(
                    "kubectl_failed" if c["target"] == "nim-operator" else "docker_failed",
                    str(exc),
                    "Check access to the target (context and namespace, or the Docker host), then"
                    " rerun the same command: it reuses what is in left_running and keeps finished"
                    " runs.",
                )
            ],
            EXIT_EXTERNAL,
        )
    except Exception as exc:  # never end on a traceback: keep state, say what is still running
        camp.log(f"stopped on an unexpected error: {exc!r}")
        camp.stop(
            "crashed",
            "fix_errors",
            [
                err(
                    "unexpected_error",
                    f"{type(exc).__name__}: {exc}"[:400],
                    "Rerun the same command to resume; it reuses the running NIMService. Or "
                    "ask the user before removing what is listed in left_running.",
                )
            ],
            EXIT_EXTERNAL,
        )
    finally:
        camp.release_client()  # no-op when stop() already did; covers an interrupt
    camp.facts["gpus_at_end"] = camp.node_gpus()
    camp.state.setdefault("stop_condition", "all candidates done")
    camp.save()
    camp.facts["candidates"] = camp.state["candidates"]
    camp.facts["stop_condition"] = camp.state["stop_condition"]
    camp.facts["active_hours"] = round(camp.active() / 3600, 2)
    result = compare(c, args.out, camp.state["candidates"])
    camp.facts["compare"] = {
        "ok": result.get("ok"),
        "state": result.get("state"),
        # recommendation.provisional is true when the comparison is incomplete
        **{
            k: result.get("facts", {}).get(k)
            for k in ("picks", "sla_boundary", "recommendation", "report", "rejected_runs")
        },
    }
    failed = [k for k, v in camp.state["candidates"].items() if v["status"] == "failed"]
    errors = [
        err(
            "candidate_failed",
            f"{k}: {camp.state['candidates'][k].get('reason')}",
            "Read OUT/ID/*.failure.log; fix or drop the candidate, then rerun to resume.",
        )
        for k in failed
    ]
    errors += [
        err(
            f"candidate_{v['status']}",
            f"{k}: {v.get('reason')}",
            "Not deployed. Fix the setting (check engine_flags.txt) or drop the candidate.",
        )
        for k, v in camp.state["candidates"].items()
        if v["status"] in ("pruned", "rejected")
    ]
    if result.get("ok"):
        # a complete comparison lists only measured overloads (run_had_request_errors): facts,
        # kept in compare, never a reason to fail the run
        camp.facts["compare"]["notes"] = list(result.get("errors") or [])
    else:
        errors += list(result.get("errors", []))
    errors += [
        err(
            "candidate_unrepaired",
            f"{k}: {runs} still do not count after {MAX_REPAIRS} repair passes",
            "Read OUT/ID/cCONC-rREPEAT/aiperf.log for why. "
            + (
                "The untuned NIM is needed for every comparison: measure again in a new --out "
                "folder, without the loads it cannot complete in workload.concurrency."
                if k == c["baseline"]
                else "The comparison leaves this case out; measure it again in a new --out "
                "folder, or drop it."
            ),
        )
        for k, runs in camp.facts.get("unrepaired", {}).items()
    ]
    base_ok = camp.state["candidates"].get(c["baseline"], {}).get("status") == "measured"
    if not base_ok:
        emit(False, "baseline_failed", "fix_errors", camp.facts, errors, EXIT_PRECONDITION)
    emit(
        not errors,
        "measured",
        "decide" if not errors else "fix_errors",
        camp.facts,
        errors,
        EXIT_OK if not errors else EXIT_PRECONDITION,
    )


if __name__ == "__main__":
    main()
