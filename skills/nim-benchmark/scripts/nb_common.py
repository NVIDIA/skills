#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""nb_common: shared helpers for nim-benchmark: the output contract, plan ids and secrets.

Provides the one-JSON-object output contract, the plan id (a hash of everything a run will do),
a scrubber that masks credentials in every file the skill writes, endpoint helpers, and the
single place that starts external programs, so tests can replace it.
Run it directly to print its version: python3 scripts/nb_common.py --version.
Exit codes: 0 ok, 2 bad input, 3 precondition failed, 4 external tool failed, 5 refused.
Never prompts. Logs go to stderr. Stdlib only.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, NoReturn

EXIT_OK, EXIT_INPUT, EXIT_PRECONDITION, EXIT_EXTERNAL, EXIT_REFUSED = 0, 2, 3, 4, 5
VERSION = "0.1.0"
AIPERF_VERSION = "0.13.0"
DEFAULT_CLIENT_IMAGE = "python:3.12-slim"
MIN_REPEATS = 3
NOISE_FLOOR_PCT = 2.0
# a variable name, not a value: what the user may give for a key
VARIABLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SECRET_NAME = re.compile(r"KEY|TOKEN(?!S|IZER)|SECRET|PASSW|CREDENTIAL|AUTH(?!OR)", re.IGNORECASE)
# well-known credential shapes, masked wherever they appear in text
TOKEN_SHAPES = re.compile(
    r"\b(?:nvapi-[A-Za-z0-9_\-]{16,}|hf_[A-Za-z0-9]{16,}|glpat-[A-Za-z0-9_\-]{16,}|"
    r"gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_\-]{20,}|AKIA[0-9A-Z]{16})\b|"
    r"(?<=Bearer )[A-Za-z0-9._\-]{16,}"
)
# a command-line flag that carries a credential: --api-key X, --hf-token=X
SECRET_FLAG = re.compile(
    r"(--?[A-Za-z0-9_-]*(?:key|token(?!s|izer)|secret|passw|credential|auth(?!or))"
    r"[A-Za-z0-9_-]*)(=|[ \t]+)(\"(?:[^\"\\\n]|\\.)*\"|'(?:[^'\\\n]|\\.)*'|[^\s\"',\]}\\]+)",
    re.IGNORECASE,
)
MASK = "***"


class ContractParser(argparse.ArgumentParser):
    """An argparse parser whose errors print the contract object and exit 2."""

    def error(self, message: str) -> NoReturn:
        emit(
            False,
            "bad_arguments",
            "fix_errors",
            errors=[error("bad_input", message, "Run the command with --help and fix the flags")],
            code=EXIT_INPUT,
        )


def error(code: str, message: str, fix: str) -> dict[str, str]:
    return {"code": code, "message": message, "fix": fix}


def emit(
    ok: bool,
    state: str,
    next_: str,
    facts: dict[str, Any] | None = None,
    errors: list[dict[str, str]] | None = None,
    code: int | None = None,
) -> NoReturn:
    """Print the one JSON object every script prints, then exit."""
    payload = {
        "ok": ok,
        "state": state,
        "next": next_,
        "facts": facts or {},
        "errors": errors or [],
    }
    sys.stdout.write(SCRUBBER.scrub(json.dumps(payload, sort_keys=True)) + "\n")
    raise SystemExit(code if code is not None else (EXIT_OK if ok else EXIT_INPUT))


class Scrubber:
    """Masks credentials in any text the skill writes: results, logs, state and its output."""

    def __init__(self) -> None:
        """Start with the values of credential-named variables in the environment."""
        self.values: set[str] = set()
        for name, value in os.environ.items():
            if SECRET_NAME.search(name):
                self.add(value)

    def add(self, value: object) -> None:
        text = str(value)
        if len(text) >= 8:
            self.values.add(text)

    def add_env(self, name: str | None) -> None:
        """Mask the value of a variable the user named for a key, whatever its name is."""
        if name and os.environ.get(name):
            self.add(os.environ[name])

    def scrub(self, text: str) -> str:
        for value in sorted(self.values, key=len, reverse=True):
            text = text.replace(value, MASK)
        text = TOKEN_SHAPES.sub(MASK, text)
        return SECRET_FLAG.sub(lambda m: f"{m[1]}{m[2]}{MASK}", text)


SCRUBBER = Scrubber()


def canonical_json(value: Any) -> str:
    """The same value always gives the same text: sorted keys, no spaces."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def plan_id(plan: dict[str, Any]) -> str:
    """A short hash of everything the run will do. The user approves this value."""
    return hashlib.sha256(canonical_json(plan).encode()).hexdigest()[:12]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Write a scrubbed JSON file. The text is scrubbed before it reaches the disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(SCRUBBER.scrub(json.dumps(value, indent=2, sort_keys=True)) + "\n")
    tmp.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(SCRUBBER.scrub(text))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def strip_userinfo(url: str) -> str:
    """Drop user:password@ from a URL before it is saved or shown.

    Any scheme and letter case, no scheme at all (//user:pw@host, user:pw@host), and a userinfo that
    holds spaces or a raw @: it ends at the last @ of the authority, as urlsplit reads it."""
    return re.sub(r"^(\s*(?:(?:[A-Za-z][A-Za-z0-9+.-]*:)?//)?)[^/?#]*@", r"\1", url)


def split_endpoint(url: str) -> tuple[str, str]:
    """The base URL AIPerf takes and the /v1 URL Gym takes, from what the user typed."""
    base = strip_userinfo(url.strip()).rstrip("/")
    base = re.sub(r"/v1$", "", base)
    return base, base + "/v1"


def run_command(
    argv: list[str], timeout: float = 3600, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Start one external program. The only place the skill does so; tests replace it.

    The program gets its own process group. On a timeout (or any other stop) the whole group is
    killed, so nothing it started, such as an `aiperf` under `sh -c`, is left running.
    Invalid or truncated UTF-8 in console output is replaced instead of aborting the run.
    """
    print("+ " + SCRUBBER.scrub(" ".join(argv)), file=sys.stderr)
    with subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        start_new_session=True,
    ) as proc:
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except BaseException:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
            raise
    return subprocess.CompletedProcess(argv, proc.returncode, stdout, stderr)


POD_REQUEST_FAILED = 42  # exit code of the in-pod probe whose HTTP request failed


class PodProbeError(Exception):
    """The probe ran in the pod and its request failed; the text is the error type and reason."""


def _pod_names(pod: tuple[str, str]) -> None:
    if any(not name or name.startswith("-") for name in pod):
        raise ValueError("namespace and pod must be nonempty and must not start with '-'")


def get_json(
    url: str,
    key_env: str | None = None,
    timeout: float = 15,
    *,
    pod: tuple[str, str] | None = None,
) -> Any:
    """GET a URL and parse the JSON reply. The key is read from the named variable."""
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:
        # A constant message: urlsplit puts the whole netloc, password included, in its error.
        raise ValueError("endpoint URL could not be parsed") from None
    if parts.scheme not in ("http", "https"):
        # Only the scheme is named: a URL can carry credentials, and this text may be logged.
        raise ValueError(f"only http and https endpoints are read, got scheme {parts.scheme!r}")
    if parts.username is not None or parts.password is not None:
        # In a pod the URL is an argument of `kubectl exec`; the key goes through key_env instead.
        raise ValueError("an endpoint URL must not contain credentials; use the key variable")
    if pod:
        _pod_names(pod)
        # Read credentials where AIPerf runs. Only the variable name crosses the command line.
        # A failed request prints its type and a short reason and exits with POD_REQUEST_FAILED,
        # so it cannot be confused with `kubectl exec` or python3 failing in the pod.
        script = f"""
import os, sys, urllib.request
try:
    request = urllib.request.Request(sys.argv[1])
    key = os.environ.get(sys.argv[2]) if sys.argv[2] else None
    if key:
        # unredirected: urllib never copies it to a redirect target (another origin, or http)
        request.add_unredirected_header("Authorization", "Bearer " + key)
    with urllib.request.urlopen(request, timeout=float(sys.argv[3])) as reply:
        print(reply.read().decode())
except Exception as exc:
    # A ValueError can quote the header value, which holds the key: print its type only.
    reason = "" if isinstance(exc, ValueError) else " ".join(
        str(getattr(exc, "reason", None) or exc).split()
    )[:200]
    print(type(exc).__name__ + (": " + reason if reason else ""))
    sys.exit({POD_REQUEST_FAILED})
"""
        done = run_command(
            [
                "kubectl",
                "exec",
                "-n",
                pod[0],
                pod[1],
                "--",
                "python3",
                "-c",
                script,
                url,
                key_env or "",
                str(timeout),
            ],
            timeout=timeout + 15,
        )
        if done.returncode == POD_REQUEST_FAILED:
            lines = done.stdout.strip().splitlines()
            raise PodProbeError(SCRUBBER.scrub(lines[-1] if lines else "request failed"))
        if done.returncode:
            raise subprocess.CalledProcessError(done.returncode, done.args)
        return json.loads(done.stdout)
    request = urllib.request.Request(url)
    SCRUBBER.add_env(key_env)
    if key_env and os.environ.get(key_env):
        # unredirected: urllib never copies it to a redirect target (another origin, or http)
        request.add_unredirected_header("Authorization", f"Bearer {os.environ[key_env]}")
    # The scheme check at the top keeps file: and custom schemes away from urlopen.
    with urllib.request.urlopen(request, timeout=timeout) as reply:  # nosec B310
        return json.loads(reply.read().decode())


def served_models(
    base: str,
    key_env: str | None = None,
    *,
    pod: tuple[str, str] | None = None,
) -> list[str]:
    """The model names an endpoint serves, from its /v1/models list."""
    payload = get_json(base + "/v1/models", key_env, pod=pod)
    return [str(item["id"]) for item in payload.get("data", []) if "id" in item]


def check_endpoint(
    base: str,
    model: str,
    key_env: str | None = None,
    *,
    pod: tuple[str, str] | None = None,
) -> tuple[list[str], list[dict[str, str]]]:
    """The served names and any errors for one endpoint and model."""
    shown = strip_userinfo(base)  # a diagnostic must never carry credentials from the URL
    try:
        names = served_models(base, key_env, pod=pod)
    except PodProbeError as exc:
        return [], [
            error(
                "endpoint_unreachable",
                f"cannot read {shown}/v1/models from pod {pod[0]}/{pod[1]}: {exc}"
                if pod
                else f"cannot read {shown}/v1/models: {exc}",
                "Check the URL, the port and the network path from the pod",
            )
        ]
    except subprocess.CalledProcessError as exc:
        where = f"pod {pod[0]}/{pod[1]}" if pod else "the client"
        return [], [
            error(
                "client_missing",
                f"cannot run the probe in {where}: kubectl exec failed (exit code {exc.returncode})",
                "Check that the pod exists and is Running, and that it has python3",
            )
        ]
    except (
        urllib.error.URLError,
        TimeoutError,
        ValueError,
        OSError,
        subprocess.TimeoutExpired,
    ) as exc:
        return [], [
            error(
                "endpoint_unreachable",
                f"cannot read {shown}/v1/models: {type(exc).__name__}",
                "Check the URL, the port and the network path",
            )
        ]
    if model not in names:
        return names, [
            error(
                "model_not_served",
                f"{model} is not served; the endpoint serves {', '.join(names) or 'nothing'}",
                "Use one of the served names",
            )
        ]
    return names, []


_DENIED = re.compile(r"forbidden|unauthorized|provide credentials", re.I)
_NO_EXECUTABLE = re.compile(r"executable file not found|not found in \$PATH|no such file", re.I)


def aiperf_version_ok(output: str) -> bool:
    """Accept only the pinned release, including when the version command prints a label."""
    versions = re.findall(r"(?<![\w.])\d+\.\d+\.\d+(?:[\w.+-]*)", output)
    return versions == [AIPERF_VERSION]


def check_client_pod(namespace: str, pod: str) -> tuple[bool, list[dict[str, str]]]:
    """Look at the Kubernetes client pod itself: it exists, it is Running, and AIPerf runs in it.

    Returns (aiperf runs in the pod, errors). kubectl's own messages are matched, never passed
    on: they can name other resources, so the errors say only what was found."""
    _pod_names((namespace, pod))
    where = f"{namespace}/{pod}"
    try:
        found = run_command(
            ["kubectl", "get", "pod", "-n", namespace, pod, "-o", "jsonpath={.status.phase}"],
            timeout=30,
        )
        if found.returncode:
            missing = "not found" in found.stderr.lower()
            return False, [
                error(
                    "client_pod_not_found" if missing else "client_missing",
                    f"pod {where} was not found" if missing else f"pod {where} cannot be read",
                    "Check --namespace and --pod (kubectl get pod -n NAMESPACE)",
                )
            ]
        phase = found.stdout.strip() or "Unknown"
        if phase != "Running":
            return False, [
                error(
                    "client_pod_not_running",
                    f"pod {where} is {phase}, not Running",
                    "Wait for the pod to be Running, or name a pod that is",
                )
            ]
        ran = run_command(
            ["kubectl", "exec", "-n", namespace, pod, "--", "aiperf", "--version"], timeout=60
        )
    except subprocess.TimeoutExpired:
        return False, [
            error(
                "client_missing",
                f"timed out looking at pod {where}",
                "Check that the pod is Running and the cluster answers",
            )
        ]
    if not ran.returncode:
        if aiperf_version_ok(ran.stdout + ran.stderr):
            return True, []
        return False, [error(
            "client_version_mismatch", f"pod {where} must run AIPerf {AIPERF_VERSION}",
            "Install aiperf==" + AIPERF_VERSION + " in the selected pod",
        )]
    if _DENIED.search(ran.stderr):
        message, fix = (
            f"not allowed to run commands in pod {where} (kubectl exec was refused)",
            "Ask the cluster admin for pods/exec permission in that namespace",
        )
    elif _NO_EXECUTABLE.search(ran.stderr):
        message, fix = (
            f"aiperf is not installed in pod {where}",
            "Use a client pod image with aiperf==" + AIPERF_VERSION,
        )
    else:
        message, fix = (
            f"aiperf did not run in pod {where} (exit code {ran.returncode})",
            "Run `aiperf --version` in the pod yourself; it needs aiperf==" + AIPERF_VERSION,
        )
    return False, [error("client_missing", message, fix)]


def main(argv: list[str] | None = None) -> None:
    parser = ContractParser(
        description="Shared helpers for nim-benchmark. Run it to print the version.",
        epilog="Example: python3 scripts/nb_common.py --version",
    )
    parser.add_argument("--version", action="store_true", help="print the skill version")
    args = parser.parse_args(argv)
    emit(True, "version", "done", {"version": VERSION, "aiperf": AIPERF_VERSION, **vars(args)})


if __name__ == "__main__":
    main()
