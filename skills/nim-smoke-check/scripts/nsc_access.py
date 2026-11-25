#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How nim-smoke-check reaches a running NIM, read-only: `import nsc_access as nsa`.

The checks need one base URL. These helpers work it out and say how (Access):
  from_url(url, health_url)             a URL as given; place "hosted" for integrate.api.nvidia.com
  from_docker(name)                     the published port of a container on this machine
  from_k8s(target, context, namespace)  a kubectl port-forward the script owns, to one Running and
                                        Ready pod of nimservice/NAME, helm/RELEASE, svc/NAME or
                                        pod/NAME
  running_image(access)                 the ID or digest of the image the container or pod runs
  image_default_passthrough(image)      the NIM_PASSTHROUGH_ARGS the image itself sets
  PortForward                           start, check, restart once and stop that port-forward
  tool(name), run_tool(argv, timeout=)  the only way the scripts start a program

Programs: kubectl and docker only, found in the absolute folders of PATH; never a shell; stdin
closed; a timeout on every call. run_tool refuses (refused_command, exit 5) anything outside this
list before it starts it, and every kubectl command but `config get-contexts -o name` names
--context CTX -n NS:
  kubectl auth can-i create pods --subresource=portforward
  kubectl get nimservice NAME -o json        kubectl get service NAME -o json
  kubectl get services -l SELECTOR -o json   kubectl get pods -l SELECTOR -o json
  kubectl get pod NAME -o json               kubectl port-forward pod/POD [LOCAL]:PORT
  kubectl config get-contexts -o name
  docker container inspect NAME              docker port NAME
  docker image inspect --format '{{json .Config.Env}}' IMAGE   (an image ID or REPO@sha256:...)
  docker context inspect --format '{{.Endpoints.docker.Host}}'
Of a container's or a pod's environment, only NIM_SERVER_PORT, NIM_HEALTH_PORT, NIM_SSL_MODE and
NIM_PASSTHROUGH_ARGS are read; no other value is kept or shown, and no Secret is ever read.

Stdlib only. Run directly, it prints a notice: it has no command of its own.
Exit codes: 0 ok, 2 bad input, 3 precondition failed, 4 external tool or endpoint failed,
5 refused.
"""

from __future__ import annotations

import atexit
import contextlib
import ipaddress
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
import tempfile
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nsc_common as nsc

PROGRAMS = ("kubectl", "docker")
HOSTED_HOST = "integrate.api.nvidia.com"
UNKNOWN = "unknown"  # a pod setting taken from a ConfigMap or a Secret, which is never read
DEFAULT_PORT = 8000  # the NIM's API port when NIM_SERVER_PORT is not set
NIM_ENV = ("NIM_SERVER_PORT", "NIM_HEALTH_PORT", "NIM_SSL_MODE", "NIM_PASSTHROUGH_ARGS")
OPERATOR_PORT = "api"  # the NIM Operator's Service and container port name
HELM_PORT = "http-openai"  # the NIM Helm chart's Service and container port name
HELM_CHART = "nim-llm"  # the NIM Helm chart (NGC catalog): its app.kubernetes.io/name label
LEADER_LABEL = "nim-llm-role"  # multi-node: the pod labelled leader serves the API
KSERVE_LABEL = "serving.kserve.io/inferenceservice"
KUBECTL_TIMEOUT = 30.0
DOCKER_TIMEOUT = 20.0
FORWARD_START = 30.0  # seconds kubectl port-forward gets to listen
STOP_WAIT = 10.0  # seconds a stopped port-forward gets to exit before it is killed
SETTLE_WAIT = 2.0  # seconds a forward that broke gets to exit (it closes, then exits)
TIMED_OUT, NOT_RUN = 124, 127  # run_tool's exit codes: a timeout, a program that cannot start
MAX_SHOWN = 300  # characters of a tool's message shown in an error
READ_LIMIT = 65_536  # bytes read of a port-forward's output file

DNS_NAME = re.compile(r"[a-z0-9]([-a-z0-9.]{0,251}[a-z0-9])?")  # an object name
DNS_LABEL = re.compile(r"[a-z0-9]([-a-z0-9]{0,61}[a-z0-9])?")  # a namespace
RELEASE = re.compile(r"[a-z0-9]([-a-z0-9]{0,51}[a-z0-9])?")  # a Helm release name
CONTEXT = re.compile(r"[^\s\x00-\x1f\x7f-][^\s\x00-\x1f\x7f]{0,252}")  # a kube context
CONTAINER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,254}")  # a container or context name, an ID
IMAGE_REF = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/:@-]{0,511}")  # an image reference or ID
# an image named by content: an ID (sha256:HEX) or a repository digest (REPO@sha256:HEX)
DIGEST_REF = re.compile(r"(?:[A-Za-z0-9][A-Za-z0-9._/:-]{0,255}@)?sha256:[0-9a-f]{64}")
LABEL_KEY = re.compile(r"[A-Za-z0-9]([A-Za-z0-9._/-]{0,315}[A-Za-z0-9])?")
LABEL_VALUE = re.compile(r"([A-Za-z0-9]([A-Za-z0-9._-]{0,61}[A-Za-z0-9])?)?")
PORT_SPEC = re.compile(r"(\d{1,5})?:(\d{1,5})")  # a port-forward's [LOCAL]:REMOTE
PORT_LINE = re.compile(r"\s*(\d{1,5})/(tcp|udp|sctp)\s*->\s*(\S+):(\d{1,5})\s*")
FORWARDING = re.compile(r"^Forwarding from (127\.0\.0\.1|\[::1\]):(\d{1,5}) -> ", re.MULTILINE)
# kubectl's forms of a missing object (on lower-case text): "Error from server (NotFound): ..."
# and '<kind> "<name>" not found'. Any other "not found" (a credential plugin, a host) is not.
NOT_FOUND = re.compile(r"\(notfound\)|\"[^\"\n]*\" not found")
# kubectl could not get credentials (client-go: "getting credentials: exec: executable aws not
# found", a credential plugin that is missing or fails) or reach the API server: the cluster was
# never asked, so nothing can be called missing or forbidden there.
NO_CLUSTER = re.compile(
    r"getting credentials|exec: executable|credential plugin|unable to connect to the server"
)
CREDENTIALS = re.compile(r"getting credentials|exec: executable|credential plugin")

TOOL_FIX = {
    "kubectl": "Install kubectl in an absolute folder of PATH, or pass --url with an address that "
    "reaches the NIM, such as an Ingress or a LoadBalancer",
    "docker": "Run the check on the Docker host with the docker CLI on PATH, or pass --url with "
    "the address where the NIM answers",
}
FAILED = {"kubectl": "kubectl_failed", "docker": "docker_failed"}
RBAC_FIX = (
    "First confirm the context and the namespace with the user: Kubernetes also answers no (or "
    "forbidden) for a namespace that does not exist or that the role does not cover, so a typo "
    "looks the same. If both are right, ask the cluster admin for a role in this namespace that "
    "allows create on pods/portforward, with get and list on pods and services (the built-in "
    "edit role has them; view does not), or pass --url with an Ingress or LoadBalancer address "
    "that reaches the NIM. Never use kubectl run or exec instead"
)
NOT_READY_FIX = (
    "Wait until the NIM's pod is Running and Ready (its readiness probe passes once the model has "
    "loaded, which can take many minutes), or rerun with --wait-ready S to let the run wait up to "
    "S seconds (at most 3600). A pod that does not start is nim-deploy's to diagnose"
)
POD_POLL = 5.0  # seconds between pod reads while --wait-ready waits for a Ready pod
FORWARD_FIX = (
    "Check that kubectl reaches the cluster with this context (a login prompt cannot be answered "
    "here: log in first), that the pod is Ready, then rerun; or pass --url with an address that "
    "reaches the NIM"
)
LOST_FIX = (
    "The pod may have restarted, or the NIM refused a connection, which ends a kubectl "
    "port-forward. Check the pod's restarts and readiness with the user (a NIM that keeps "
    "restarting is nim-deploy's to diagnose), then rerun"
)
CREDENTIALS_FIX = (
    "kubectl cannot get credentials for this context: the user installs the credential plugin "
    "the message names (such as gke-gcloud-auth-plugin, aws or kubelogin) or logs in as they "
    "normally do (a login prompt cannot be answered here), then rerun; or pass --url with an "
    "address that reaches the NIM"
)
POD_FIX = "Pass --k8s pod/NAME with a pod of the NIM, or --url"
REFUSED_FIX = (
    "Do not run the command another way: a smoke check never changes a deployment. If a normal "
    "run hit this, report it as a bug in the skill"
)


# --- programs ------------------------------------------------------------------------------------


def tool(name: str) -> str:
    """The absolute path of `name` (kubectl or docker), found only in the absolute folders of
    PATH: an empty or relative entry (`::`, a trailing `:`, `.`) means the current folder, often
    a project someone else wrote, so it is skipped. No other variable picks the program.
    ScriptError tool_missing (exit 3) when it is not there."""
    if name not in PROGRAMS:
        raise ValueError(f"the scripts run only kubectl and docker, not {name!r}")
    entries = os.environ.get("PATH", os.defpath).split(os.pathsep)
    folders = os.pathsep.join(entry for entry in entries if os.path.isabs(entry))
    found = shutil.which(name, path=folders) if folders else None
    if not found or not os.path.isabs(found):
        raise nsc.ScriptError(
            "tool_missing",
            f"{name} was not found in an absolute folder of PATH (relative folders are skipped)",
            TOOL_FIX[name],
            nsc.EXIT_PRECONDITION,
        )
    return found


KUBECTL_VALUE_FLAGS = frozenset(
    {"--context", "-n", "--namespace", "-o", "--output", "-l", "--selector", "--subresource"}
)
DOCKER_VALUE_FLAGS = frozenset({"--format", "-f"})
FLAG_ALIASES = {
    "kubectl": {"--namespace": "-n", "--output": "-o", "--selector": "-l"},
    "docker": {"-f": "--format"},
}
READ_KINDS = frozenset(
    {
        "nimservice", "nimservices", "nimservice.apps.nvidia.com", "nimservices.apps.nvidia.com",
        "svc", "service", "services", "pod", "pods",
    }
)
# kubectl commands and the flags each takes besides --context and -n
KUBECTL_COMMANDS: dict[tuple[str, ...], frozenset[str]] = {
    ("get",): frozenset({"-o", "-l"}),
    ("auth", "can-i"): frozenset({"--subresource"}),
    ("port-forward",): frozenset(),
    ("config", "get-contexts"): frozenset({"-o"}),
}
# docker commands, how many names each takes, and whether it takes --format
DOCKER_COMMANDS: dict[tuple[str, ...], tuple[int, int, bool]] = {
    ("container", "inspect"): (1, 1, True),
    ("image", "inspect"): (1, 1, True),
    ("port",): (1, 2, False),
    ("context", "inspect"): (0, 1, True),
}


def why_refused(argv: list[str]) -> str | None:
    """Why run_tool refuses `argv`; None when it is on the read-only list of the module
    docstring: an absolute path to kubectl or docker, then one listed command with only the flags
    it needs. Every kubectl command but `config get-contexts -o name` names --context and -n. No
    word may be empty or hold a control character, and no flag value may start with -."""
    if not isinstance(argv, (list, tuple)) or not argv:
        return "no command"
    if not all(isinstance(word, str) and word for word in argv):
        return "an empty or non-text word"
    if any(ord(char) < 0x20 or ord(char) == 0x7F for word in argv for char in word):
        return "a control character"
    program = argv[0]
    name = os.path.basename(program)
    if name not in PROGRAMS:
        return f"{name} is neither kubectl nor docker"
    if not os.path.isabs(program):
        return f"{program} is not an absolute path"
    parsed = _split_words(list(argv[1:]), name)
    if isinstance(parsed, str):
        return parsed
    words, flags = parsed
    if name == "kubectl":
        return _kubectl_refusal(words, flags)
    return _docker_refusal(words, flags)


def _split_words(tokens: list[str], program: str) -> tuple[list[str], dict[str, str | None]] | str:
    """(the words that are not flags, {flag: value or None}); a text when the flags are bad."""
    value_flags = KUBECTL_VALUE_FLAGS if program == "kubectl" else DOCKER_VALUE_FLAGS
    aliases = FLAG_ALIASES[program]
    words: list[str] = []
    flags: dict[str, str | None] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        index += 1
        if not token.startswith("-") or token == "-":
            words.append(token)
            continue
        if token == "--":
            return "an end-of-options marker (--)"
        flag, equals, given = token.partition("=")
        value: str | None = given if equals else None
        if not equals and flag in value_flags:
            if index >= len(tokens):
                return f"{flag} has no value"
            value = tokens[index]
            index += 1
        if value is not None and (not value or value.startswith("-")):
            return f"{flag} has an empty value or one that starts with -"
        key = aliases.get(flag, flag)
        if key in flags:
            return f"{flag} is given twice"
        flags[key] = value
    return words, flags


def _kubectl_refusal(words: list[str], flags: dict[str, str | None]) -> str | None:
    if not words:
        return "kubectl with no command"
    command = tuple(words[:2]) if words[0] in ("auth", "config") else (words[0],)
    shown = "kubectl " + " ".join(command)
    if command not in KUBECTL_COMMANDS:
        return f"{shown} is not on the read-only list"
    rest = words[len(command):]
    scoped = command != ("config", "get-contexts")
    allowed = KUBECTL_COMMANDS[command] | ({"--context", "-n"} if scoped else set())
    extra = sorted(set(flags) - allowed)
    if extra:
        return f"{shown} with {extra[0]}"
    if scoped and not (flags.get("--context") and flags.get("-n")):
        return f"{shown} without --context and -n"
    if command == ("get",):
        if not 1 <= len(rest) <= 2 or rest[0].lower() not in READ_KINDS:
            return "kubectl get of something other than a NIMService, a Service or a pod"
        if len(rest) == 2 and not DNS_NAME.fullmatch(rest[1]):
            return "kubectl get of a name that is not a Kubernetes name"
        if flags.get("-o", "json") != "json":
            return "kubectl get with an output other than -o json"
        if "-l" in flags and not _selector_ok(flags["-l"] or ""):
            return "kubectl get with a label selector other than key=value pairs"
    elif command == ("auth", "can-i"):
        if rest != ["create", "pods"] or flags.get("--subresource") != "portforward":
            return "kubectl auth can-i other than create pods --subresource=portforward"
    elif command == ("port-forward",):
        if len(rest) != 2 or not rest[0].startswith("pod/") or not DNS_NAME.fullmatch(rest[0][4:]):
            return "kubectl port-forward other than to pod/NAME and one port"
        match = PORT_SPEC.fullmatch(rest[1])
        if not match or not 0 < int(match.group(2)) < 65536 or int(match.group(1) or 0) > 65535:
            return "kubectl port-forward with a port other than [LOCAL]:PORT"
    elif rest or flags.get("-o") != "name":
        return "kubectl config get-contexts other than -o name"
    return None


def _docker_refusal(words: list[str], flags: dict[str, str | None]) -> str | None:
    if not words:
        return "docker with no command"
    command = (words[0],) if words[0] == "port" else tuple(words[:2])
    shown = "docker " + " ".join(command)
    if command not in DOCKER_COMMANDS:
        return f"{shown} is not on the read-only list"
    rest = words[len(command):]
    low, high, takes_format = DOCKER_COMMANDS[command]
    if not low <= len(rest) <= high:
        return f"{shown} with {len(rest)} names"
    extra = sorted(set(flags) - ({"--format"} if takes_format else set()))
    if extra:
        return f"{shown} with {extra[0]}"
    pattern = IMAGE_REF if command == ("image", "inspect") else CONTAINER
    if rest and not pattern.fullmatch(rest[0]):
        return f"{shown} of a name that docker would not take"
    if len(rest) == 2 and not re.fullmatch(r"\d{1,5}(/(tcp|udp|sctp))?", rest[1]):
        return f"{shown} with a second word that is not a port"
    return None


def _display(argv: Any) -> str:
    """The command as one shell-quoted line, with the program's bare name (for messages)."""
    try:
        words = [os.path.basename(str(argv[0])), *map(str, argv[1:])]
        return shlex.join(words)[:MAX_SHOWN]
    except (IndexError, TypeError):
        return "an empty command"


def run_tool(
    argv: list[str],
    *,
    timeout: float,
    background: bool = False,
    stdout: IO[bytes] | None = None,
    stderr: IO[bytes] | None = None,
) -> Any:
    """Run one read-only command: the only place the scripts start a program.

    `argv` must pass why_refused and its program must be the one tool() finds; otherwise it is
    ScriptError refused_command (exit 5) and nothing runs. Never a shell; stdin is closed, so a
    login prompt cannot wait for an answer; the command line goes to stderr.
    Returns a subprocess.CompletedProcess with text output: exit code 124 when it did not finish
    within `timeout` seconds, 127 when it could not start, with the reason in stderr.
    background=True starts it and returns the Popen, its output going to the `stdout` and
    `stderr` files (never a pipe nobody reads); the caller stops it (PortForward does)."""
    argv = list(argv) if isinstance(argv, (list, tuple)) else []
    why = why_refused(argv)
    if why is None and os.path.normpath(argv[0]) != os.path.normpath(
        tool(os.path.basename(argv[0]))
    ):
        why = f"{argv[0]} is not the program found on PATH"
    if why is not None:
        raise nsc.ScriptError(
            "refused_command",
            f"refused to run {_display(argv)}: {why}. The smoke check runs only read-only "
            "kubectl and docker commands",
            REFUSED_FIX,
            nsc.EXIT_REFUSED,
        )
    timeout = float(timeout)
    if not (timeout > 0 and math.isfinite(timeout)):
        raise ValueError(f"the timeout must be a positive number of seconds, not {timeout}")
    name = os.path.basename(argv[0])
    nsc.log(f"nim-smoke-check runs: {_display(argv)}")
    if background:
        if stdout is None or stderr is None or subprocess.PIPE in (stdout, stderr):
            raise ValueError("a background command writes to files, never to a pipe")
        try:
            return subprocess.Popen(
                argv, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, shell=False
            )
        except OSError as exc:
            raise nsc.ScriptError(
                FAILED[name],
                f"cannot start {name}: {exc.strerror or exc}",
                TOOL_FIX[name],
                nsc.EXIT_EXTERNAL,
            ) from exc
    try:
        return subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
            shell=False,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            argv, TIMED_OUT, "", f"{name} did not finish within {timeout:g} s"
        )
    except OSError as exc:
        return subprocess.CompletedProcess(
            argv, NOT_RUN, "", f"cannot run {name}: {exc.strerror or exc}"
        )


# --- small readers --------------------------------------------------------------------------------


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _port_number(value: Any) -> int | None:
    """A TCP port from an int or a string of ASCII digits (1-65535); None otherwise. Not
    str.isdigit(): it is True for "²", which int() refuses."""
    if isinstance(value, bool):
        return None
    if isinstance(value, str) and re.fullmatch(r"[0-9]{1,5}", value.strip()):
        value = int(value.strip())
    return value if isinstance(value, int) and 0 < value < 65536 else None


def _first_line(*texts: str | None) -> str:
    """The first line that says something, masked, then cut to MAX_SHOWN characters
    (nsc.clip: a cut never keeps the first part of a key that the whole line shows)."""
    for text in texts:
        for line in (text or "").splitlines():
            if line.strip():
                return nsc.clip(line, MAX_SHOWN)
    return "no message"


def _env_values(entries: Any) -> dict[str, str]:
    """NIM_ENV values from a Docker `NAME=value` list; every other entry is skipped unread."""
    found: dict[str, str] = {}
    for entry in _list(entries):
        if isinstance(entry, str):
            key, equals, value = entry.partition("=")
            if equals and key in NIM_ENV:
                found[key] = value
    return found


def _tls(mode: str | None, host: str) -> tuple[str, str | None]:
    """(scheme, note) for NIM_SSL_MODE: https for TLS and MTLS (verification stays on)."""
    value = (mode or "").strip().upper()
    if value not in ("TLS", "MTLS"):
        return "http", None
    note = (
        f"NIM_SSL_MODE={value}: the URL is https and the certificate is verified. Pass --ca-file "
        f"with the CA that signed it; if it does not name {host}, pass --url with an address "
        "it names"
    )
    if value == "MTLS":
        note += ", and --client-cert and --client-key"
    return "https", note + "."


# --- Access ---------------------------------------------------------------------------------------


@dataclass
class Access:
    """How the checks reach the NIM.

    place: "url" (a URL as given), "hosted" (integrate.api.nvidia.com), "docker", "kubernetes".
    url: the base URL the checks append /v1/... to.
    health_url: the base URL of /v1/health/*: url, or the published NIM_HEALTH_PORT; None when
      health answers on a port this machine cannot reach (notes say why).
    how: "direct" (a URL as given, or Docker host networking), "published-port" (a Docker -p
      mapping) or "port-forward" (the script's own kubectl port-forward).
    image, image_id: the image reference the container or pod runs, and its ID or digest.
    effective_passthrough: NIM_PASSTHROUGH_ARGS of the deployment. Docker: the container's value,
      which is the image default unless `-e` replaced it. Kubernetes: the pod spec's literal
      value; None when it does not set it (the image default applies), UNKNOWN when it comes
      from a ConfigMap or a Secret (valueFrom), or may come from one (envFrom), which is not
      read.
    effective_args: the container arguments (Docker: the command after the image).
    config_known: True when the container or pod spec was read. False for a URL: then
      effective_passthrough and effective_args are unknown, not empty (pass args=None to
      nsc_features.tier_b).
    notes: plain sentences for the report.
    port_forward: PortForward.facts() on Kubernetes; a restart updates url and port_forward."""

    place: str
    url: str
    health_url: str | None
    how: str
    image: str | None = None
    effective_passthrough: str | None = None
    effective_args: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    port_forward: dict[str, Any] | None = None
    config_known: bool = False
    image_id: str | None = None

    def facts(self) -> dict[str, Any]:
        """A JSON-ready summary for facts.access, without the raw NIM_PASSTHROUGH_ARGS text and
        arguments (those stay attributes; nsc_features.tier_b reads them)."""
        if not self.config_known:
            passthrough = "unknown"
        elif self.effective_passthrough is None:
            passthrough = "not set"
        else:
            passthrough = "unknown" if self.effective_passthrough == UNKNOWN else "set"
        return {
            "place": self.place,
            "url": self.url,
            "health_url": self.health_url,
            "how": self.how,
            "image": self.image,
            "image_id": self.image_id,
            "passthrough": passthrough,
            "args": len(self.effective_args) if self.config_known else None,
            "port_forward": dict(self.port_forward) if self.port_forward else None,
            "notes": list(self.notes),
        }


def _lookup_name(url: str) -> str:
    """The host of a base URL as the name lookup reads it: socket encodes a name with the idna
    codec, so full-width letters or an ideographic full stop spell the same ASCII name; in lower
    case and without a final dot."""
    host = urllib.parse.urlsplit(url).hostname or ""
    try:
        host = host.encode("idna").decode("ascii")
    except UnicodeError:  # not a name the codec takes (an empty or a too long label)
        pass
    return host.rstrip(".").lower()


def from_url(url: str, health_url: str | None = None) -> Access:
    """A URL as given (nsc_common.normalize_base: http or https, no user:pass@, a trailing / and
    /v1 trimmed). place "hosted" when the host is integrate.api.nvidia.com, in any spelling the
    name lookup reads as it (_lookup_name), which has no /v1/health routes; else "url".
    ScriptError bad_input (exit 2) for a bad URL, and for the hosted host over http: its key
    would cross the network in clear text."""
    base = nsc.normalize_base(url)
    health = nsc.normalize_base(health_url) if health_url else base
    for given in (base, health):
        if _lookup_name(given) == HOSTED_HOST and urllib.parse.urlsplit(given).scheme != "https":
            raise nsc.ScriptError(
                "bad_input",
                f"{HOSTED_HOST} takes https only; an http:// URL would send its key in clear "
                "text",
                f"Pass --url https://{HOSTED_HOST}/v1",
                nsc.EXIT_INPUT,
            )
    if _lookup_name(base) == HOSTED_HOST:
        notes = [
            "The NVIDIA API catalog: it has no /v1/health routes, and /v1/models lists the "
            "whole catalog, not one served model."
        ]
        return Access("hosted", base, health, "direct", notes=notes)
    return Access("url", base, health, "direct")


# --- Docker ---------------------------------------------------------------------------------------


def parse_port_lines(text: str) -> dict[int, list[tuple[str, int]]]:
    """`docker port NAME` output as {container port: [(host IP, host port)]}, TCP only: a daemon
    that publishes on all addresses prints "8000/tcp -> 0.0.0.0:8005" and "8000/tcp -> [::]:8005"
    (brackets removed)."""
    found: dict[int, list[tuple[str, int]]] = {}
    for line in (text or "").splitlines():
        match = PORT_LINE.fullmatch(line)
        if not match or match.group(2) != "tcp":
            continue
        host = match.group(3)
        if host.startswith("[") and host.endswith("]"):
            host = host[1:-1]
        container, published = _port_number(match.group(1)), _port_number(match.group(4))
        if container and published:
            found.setdefault(container, []).append((host, published))
    return found


def _port_table(info: dict[str, Any]) -> dict[int, list[tuple[str, int]]] | None:
    """The TCP port mappings of a `docker container inspect` record; None when it has none."""
    ports = _dict(info.get("NetworkSettings")).get("Ports")
    if not isinstance(ports, dict):
        return None
    table: dict[int, list[tuple[str, int]]] = {}
    for key, bindings in ports.items():
        number, _, proto = str(key).partition("/")
        container = _port_number(number)
        if container is None or proto not in ("", "tcp"):
            continue
        for binding in _list(bindings):
            host_ip = _dict(binding).get("HostIp")
            published = _port_number(str(_dict(binding).get("HostPort") or ""))
            if isinstance(host_ip, str) and published:
                table.setdefault(container, []).append((host_ip, published))
    return table


def pick_binding(bindings: list[tuple[str, int]]) -> tuple[str, int, str | None]:
    """(URL host, port, note) for one published port, as seen from the Docker host (this
    machine): all addresses (0.0.0.0, "" or ::) -> 127.0.0.1 (or [::1] for :: alone);
    127.0.0.1 or ::1 -> itself, answering on the Docker host alone; another address -> that
    address, as localhost does not answer there. IPv4 first. ValueError when none is usable."""
    ranked: list[tuple[int, str, int, str | None]] = []
    for host_ip, port in bindings:
        bare = host_ip.strip("[]")
        # Docker's published address, read to pick a URL; nothing binds here.
        if bare in ("", "0.0.0.0"):  # nosec B104
            ranked.append((0, "127.0.0.1", port, None))
            continue
        if bare == "::":
            ranked.append((2, "[::1]", port, None))
            continue
        try:
            address = ipaddress.ip_address(bare)
        except ValueError:
            continue
        shown = f"[{bare}]" if address.version == 6 else bare
        if address.is_loopback:
            note = f"Port {port} is published on {bare} only, so it answers on the Docker host "
            note += "alone."
            ranked.append((1 if address.version == 4 else 3, shown, port, note))
        else:
            note = f"Port {port} is published on {bare} only: the URL uses that address, as "
            note += "localhost does not answer there."
            ranked.append((4 if address.version == 4 else 5, shown, port, note))
    if not ranked:
        raise ValueError("no usable published address")
    _, host, port, note = min(ranked, key=lambda item: item[0])
    return host, port, note


def docker_endpoint(docker: str) -> str | None:
    """The daemon endpoint of the active Docker context (DOCKER_CONTEXT and the Docker config
    pick it): `docker context inspect`; None when that fails (an old CLI without contexts)."""
    proc = run_tool(
        [docker, "context", "inspect", "--format", "{{.Endpoints.docker.Host}}"],
        timeout=DOCKER_TIMEOUT,
    )
    lines = (proc.stdout or "").strip().splitlines()
    return lines[0].strip() if proc.returncode == 0 and lines else None


def _local_endpoint(endpoint: str) -> bool:
    return endpoint.startswith(("unix://", "npipe://"))


def _docker_remote(endpoint: str, where: str, keyed: bool = False) -> nsc.ScriptError:
    """docker_remote. With a key (`keyed`), the fix also says that the http:// URL it names would
    send the key over the network, which key_over_plain_http refuses unless the user agrees. The
    flag ends the text: the masker reads the word after a flag whose name holds "key" as a key."""
    shown = nsc.clip(nsc.strip_userinfo(endpoint), MAX_SHOWN)
    try:
        host = urllib.parse.urlsplit(shown).hostname or "HOST"
    except ValueError:
        host = "HOST"
    fix = (f"Pass --url with the address where the NIM answers, for example http://{host}:8000 "
           "(ask the user for the published port), or run the check on the Docker host")
    if keyed:
        fix += ("; with --api-key-env, that http:// URL would send the key over the network "
                "unencrypted and stops with key_over_plain_http: prefer the Docker host, where "
                "the key stays on loopback, or leave the key out (a NIM checks none); only when "
                "the user agrees, add --allow-plain-http-key")
    return nsc.ScriptError(
        "docker_remote",
        f"{where} points to a remote Docker daemon ({shown}): the container's published port is "
        "on that host, not on this machine",
        fix,
        nsc.EXIT_PRECONDITION,
    )


def _refuse_remote_docker(docker: str, keyed: bool = False) -> None:
    host = os.environ.get("DOCKER_HOST", "").strip()
    if host and not _local_endpoint(host):
        raise _docker_remote(host, "DOCKER_HOST", keyed)
    endpoint = docker_endpoint(docker)
    if endpoint and not _local_endpoint(endpoint):
        raise _docker_remote(endpoint, "The active Docker context", keyed)


def _docker_error(proc: subprocess.CompletedProcess[str], what: str) -> nsc.ScriptError:
    detail = _first_line(proc.stderr)
    low = (proc.stderr or "").lower()
    if "permission denied" in low:
        fix = "The user needs access to the Docker daemon (the docker group); ask the user"
    elif "cannot connect to the docker daemon" in low or "failed to connect to the docker" in low:
        fix = "Start the Docker daemon on this machine, or pass --url with the NIM's address"
    else:
        fix = TOOL_FIX["docker"]
    return nsc.ScriptError("docker_failed", f"docker could not {what}: {detail}", fix,
                           nsc.EXIT_EXTERNAL)


def _inspect_container(docker: str, name: str) -> dict[str, Any]:
    proc = run_tool([docker, "container", "inspect", name], timeout=DOCKER_TIMEOUT)
    if proc.returncode != 0:
        # only "No such container": a missing Docker socket says "connect: no such file or
        # directory", which is a daemon problem (_docker_error), not a missing container
        if "no such container" in (proc.stderr or "").lower():
            raise nsc.ScriptError(
                "target_not_found",
                f"there is no Docker container {name} on this machine: {_first_line(proc.stderr)}",
                "Check the name with the user; docker ps lists the running containers",
                nsc.EXIT_PRECONDITION,
            )
        raise _docker_error(proc, f"inspect the container {name}")
    try:
        data = json.loads(proc.stdout)
    except (ValueError, RecursionError):  # not JSON, or nested too deeply to read
        data = None
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data[0]
    # the output is never shown: it holds the container's whole environment
    raise nsc.ScriptError(
        "docker_failed",
        f"docker container inspect {name} printed no container record",
        TOOL_FIX["docker"],
        nsc.EXIT_EXTERNAL,
    )


def _require_running(name: str, info: dict[str, Any]) -> None:
    """ScriptError container_not_running (exit 3) unless the container runs: State.Status
    "running" (Running true when there is no Status), and neither restarting nor paused. Docker
    keeps Running true while its restart policy waits to start an exited container again, and
    empties the published ports meanwhile; a paused container answers no request."""
    state = _dict(info.get("State"))
    status = (_text(state.get("Status")) or "").lower()
    if state.get("Restarting") is True or status == "restarting":
        count = info.get("RestartCount")
        times = f", {count} restarts so far" if isinstance(count, int) and count > 0 else ""
        raise nsc.ScriptError(
            "container_not_running",
            f"the container {name} is restarting: the NIM exits and Docker's restart policy "
            f"starts it again{times}",
            "Tell the user the NIM keeps exiting; finding why (its logs say) is nim-deploy's "
            "job. Rerun once the container stays running",
            nsc.EXIT_PRECONDITION,
        )
    if state.get("Paused") is True or status == "paused":
        raise nsc.ScriptError(
            "container_not_running",
            f"the container {name} is paused, so it answers no request",
            f"Ask the user to unpause it (docker unpause {name} is theirs to run; the smoke "
            "check changes nothing), then rerun",
            nsc.EXIT_PRECONDITION,
        )
    if state.get("Running") is False or (status and status != "running"):
        raise nsc.ScriptError(
            "container_not_running",
            f"the container {name} is {status or 'not running'}",
            "Ask the user which container serves the NIM; starting or restarting a NIM is "
            "nim-deploy's job",
            nsc.EXIT_PRECONDITION,
        )


def from_docker(name: str, *, keyed: bool = False) -> Access:
    """The URL of a NIM container on this machine, from one `docker container inspect NAME`.

    The API port is NIM_SERVER_PORT (default 8000); health moves to NIM_HEALTH_PORT when set;
    NIM_SSL_MODE TLS or MTLS makes it https. Host networking: 127.0.0.1 and that port (-p does
    not apply). Otherwise the published mapping of that port (pick_binding); when the record has
    no port table, `docker port NAME` is read instead. Only the four NIM_ENV variables are read
    from the container's environment, which may hold an NGC key; nothing else is kept.
    ScriptError: bad_input (exit 2) for a bad name; docker_remote (DOCKER_HOST or the active
    context names a remote daemon; its fix speaks of the key when `keyed`), tool_missing,
    target_not_found, container_not_running (stopped, restarting or paused) and
    port_not_published (exit 3); docker_failed (exit 4)."""
    if not isinstance(name, str) or not CONTAINER.fullmatch(name):
        raise nsc.ScriptError(
            "bad_input",
            "--docker takes the name or ID of a container (letters, digits, _, . and -)",
            "Pass the name docker ps shows, for example --docker llama",
            nsc.EXIT_INPUT,
        )
    docker = tool("docker")
    _refuse_remote_docker(docker, keyed)
    info = _inspect_container(docker, name)
    config = _dict(info.pop("Config", None))
    env = _env_values(config.pop("Env", None))  # the rest of the environment is dropped here
    image = _text(config.get("Image"))
    command = [arg for arg in _list(config.get("Cmd")) if isinstance(arg, str)]
    _require_running(name, info)
    server = _port_number(env.get("NIM_SERVER_PORT")) or DEFAULT_PORT
    health = _port_number(env.get("NIM_HEALTH_PORT"))
    network = _text(_dict(info.get("HostConfig")).get("NetworkMode")) or ""
    notes: list[str] = []
    if network == "host":
        host, port, how = "127.0.0.1", server, "direct"
        notes.append(
            f"Host networking: the NIM listens on port {server} of this host; -p does not apply."
        )
        table: dict[int, list[tuple[str, int]]] = {server: [("127.0.0.1", server)]}
        if health:
            table[health] = [("127.0.0.1", health)]
    else:
        found = _port_table(info)
        table = found if found is not None else parse_port_lines(_docker_port(docker, name))
        try:
            host, port, note = pick_binding(table.get(server, []))
        except ValueError:
            mode = f" (network mode {network})" if network and network != "bridge" else ""
            raise nsc.ScriptError(
                "port_not_published",
                f"the container {name} does not publish its NIM port {server}/tcp to this "
                f"host{mode}",
                "Pass --url with an address that reaches the NIM, or ask the user to publish "
                "the port (-p), which is nim-deploy's job",
                nsc.EXIT_PRECONDITION,
            ) from None
        how = "published-port"
        if note:
            notes.append(note)
    scheme, tls_note = _tls(env.get("NIM_SSL_MODE"), host.strip("[]"))
    if tls_note:
        notes.append(tls_note)
    url = f"{scheme}://{host}:{port}"
    health_url: str | None = url
    if health and health != server:
        try:
            health_host, health_port, _ = pick_binding(table.get(health, []))
        except ValueError:
            health_url = None
            notes.append(
                f"NIM_HEALTH_PORT={health}: the health routes moved to that port, which the "
                "container does not publish; the health check cannot reach them (pass "
                "--health-url if another address does)."
            )
        else:
            health_url = f"{scheme}://{health_host}:{health_port}"
    return Access(
        "docker",
        url,
        health_url,
        how,
        image=image,
        effective_passthrough=env.get("NIM_PASSTHROUGH_ARGS"),
        effective_args=command,
        notes=notes,
        config_known=True,
        image_id=_text(info.get("Image")),
    )


def _docker_port(docker: str, name: str) -> str:
    """`docker port NAME`, for a record with no port table. ScriptError docker_failed (exit 4)
    when docker fails or times out: only an answer that lists no port is port_not_published."""
    proc = run_tool([docker, "port", name], timeout=DOCKER_TIMEOUT)
    if proc.returncode != 0:
        raise _docker_error(proc, f"read the published ports of the container {name}")
    return proc.stdout or ""


def running_image(access: Access) -> str | None:
    """A reference that names exactly the image the container or pod runs, for
    image_default_passthrough. Docker: the container's image ID (sha256:...). Kubernetes: the
    pod's imageID digest (REPO@sha256:...; a runtime prefix such as docker-pullable:// is
    dropped), which this machine's daemon answers only when it holds that very image. Never the
    tag: a pull after the start moves a tag such as :latest to another image, with another
    default. None when there is no such reference (the image default is then unknown)."""
    text = access.image_id if isinstance(access.image_id, str) else ""
    if access.place == "kubernetes":
        text = text.split("://", 1)[-1]
    elif access.place != "docker":
        return None
    return text if DIGEST_REF.fullmatch(text) else None


def image_default_passthrough(image: str | None) -> str | None:
    """The NIM_PASSTHROUGH_ARGS the image itself sets, as the NIM docs check it:
    `docker image inspect --format '{{json .Config.Env}}' IMAGE` on this machine's daemon, where
    IMAGE is best an ID or a digest (running_image). "" when the image sets none; None when
    unknown (no docker, the image is not on the daemon, or no output that can be read). Never
    raises; no other variable of the image is kept."""
    if not isinstance(image, str) or not IMAGE_REF.fullmatch(image):
        return None
    try:
        docker = tool("docker")
        proc = run_tool(
            [docker, "image", "inspect", "--format", "{{json .Config.Env}}", image],
            timeout=DOCKER_TIMEOUT,
        )
    except nsc.ScriptError:
        return None
    if proc.returncode != 0:
        return None
    lines = (proc.stdout or "").strip().splitlines()
    if not lines:  # docker printed nothing: no evidence either way
        return None
    try:
        entries = json.loads(lines[0])
    except (ValueError, RecursionError):
        return None
    if entries is None:
        return ""  # the image sets no variable at all
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if isinstance(entry, str) and entry.startswith("NIM_PASSTHROUGH_ARGS="):
            return entry.partition("=")[2]
    return ""


# --- the port-forward ----------------------------------------------------------------------------


class PortForwardError(nsc.ScriptError):
    """A port-forward that did not start; `busy` when the local port was taken."""

    def __init__(self, *args: Any, busy: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.busy = busy


def _read(path: Path | None, tail: bool = False) -> str:
    """At most READ_LIMIT bytes of a port-forward output file, its start or its end."""
    if path is None:
        return ""
    try:
        with open(path, "rb") as handle:
            if tail:
                handle.seek(max(0, os.fstat(handle.fileno()).st_size - READ_LIMIT))
            return handle.read(READ_LIMIT).decode("utf-8", "replace")
    except OSError:
        return ""


def _last_error(text: str) -> str:
    """kubectl's last `error: ...` line, else its last line, masked, then cut to MAX_SHOWN
    characters (nsc.clip)."""
    lines = [" ".join(line.split()) for line in text.splitlines() if line.strip()]
    errors = [line for line in lines if line.lower().startswith("error")]
    return nsc.clip((errors or lines or ["no message"])[-1], MAX_SHOWN)


def _no_cluster(text: str, where: str) -> tuple[str, str] | None:
    """(message, fix) when kubectl could not get credentials or reach the API server (NO_CLUSTER):
    the line that says so, which names the credential plugin, rather than the last line, which
    may be only a docs URL. None for any other failure."""
    for line in text.splitlines():
        if NO_CLUSTER.search(line.lower()):
            said = nsc.clip(line, MAX_SHOWN)
            fix = CREDENTIALS_FIX if CREDENTIALS.search(line.lower()) else FORWARD_FIX
            return f"kubectl could not reach the cluster to {where}: {said}", fix
    return None


def _connects(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


class PortForward:
    """`kubectl port-forward pod/POD :PORT --context CTX -n NS`, owned by the script.

    start() runs it with its output going to files in a private temporary folder (kubectl writes
    a line per connection, so an unread pipe would block it), waits for "Forwarding from
    127.0.0.1:N" and a TCP connect, and returns N: kubectl picks the free local port. Call
    ensure() before each check: kubectl ends the whole forward when the pod restarts or the NIM
    refuses one connection, so a stopped forward is restarted once (on the same local port when
    it is free), and a second stop is ScriptError port_forward_lost (exit 4). After a request
    that got no answer through it, call settle(): kubectl exits a moment after it closes the
    connection, and ensure() must not see it still running. stop() sends SIGINT
    (terminate() off POSIX), waits 10 s, then kills; `with` and the end of the script stop it
    too, and while one runs SIGTERM and SIGHUP end the script through its finally blocks.
    A forward goes to one pod: the Service's load balancing does not apply."""

    def __init__(
        self,
        kubectl: str,
        context: str,
        namespace: str,
        pod: str,
        remote_port: int,
        *,
        start_timeout: float = FORWARD_START,
        max_restarts: int = 1,
        scheme: str = "http",
    ) -> None:
        if not isinstance(pod, str) or not DNS_NAME.fullmatch(pod):
            raise ValueError(f"not a pod name: {pod!r}")
        if _port_number(remote_port) is None or isinstance(remote_port, str):
            raise ValueError(f"not a port number: {remote_port!r}")
        self.kubectl = kubectl
        self.context = context
        self.namespace = namespace
        self.pod = pod
        self.remote_port = int(remote_port)
        self.start_timeout = float(start_timeout)
        self.max_restarts = max_restarts
        self.scheme = scheme
        self.restarts = 0
        self.local_port: int | None = None
        self.host = "127.0.0.1"
        self.access: Access | None = None  # from_k8s sets it; a restart updates its url
        self._local = ""  # the LOCAL part of the port argument: "" lets kubectl pick
        self._proc: subprocess.Popen[bytes] | None = None
        self._folder: Path | None = None
        self._runs = 0
        self._listened = False  # whether any run reached listening: ensure() restarts only then
        self._err: Path | None = None

    @property
    def argv(self) -> list[str]:
        """The command of the current (or next) run."""
        return [
            self.kubectl, "port-forward", f"pod/{self.pod}", f"{self._local}:{self.remote_port}",
            "--context", self.context, "-n", self.namespace,
        ]

    @property
    def url(self) -> str | None:
        """The base URL through the forward; None before it listens."""
        if self.local_port is None:
            return None
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"{self.scheme}://{host}:{self.local_port}"

    def facts(self) -> dict[str, Any]:
        shown = ["kubectl", *self.argv[1:]]
        return {
            "command": shlex.join(shown),
            "argv": shown,
            "pod": self.pod,
            "remote_port": self.remote_port,
            "local_port": self.local_port,
            "restarts": self.restarts,
            "context": self.context,
            "namespace": self.namespace,
        }

    def alive(self) -> bool:
        """Whether kubectl still runs (it exits when the forward is lost)."""
        return self._proc is not None and self._proc.poll() is None

    def start(self, timeout: float | None = None) -> int:
        """Start kubectl (unless it runs) and return the local port once it accepts a TCP
        connection. ScriptError: rbac_missing or pod_not_ready (exit 3), port_forward_failed
        (exit 4: kubectl stopped, or it did not listen within `timeout` seconds, default 30)."""
        if self.alive() and self.local_port is not None:
            return self.local_port
        self._halt()
        return self._launch("", self.start_timeout if timeout is None else float(timeout))

    def ensure(self) -> int:
        """The local port of a running forward. A forward that stopped is restarted once; the
        local port may then change, so read url (or Access.url) after this call. kubectl that
        still runs but no longer accepts a connection on its local port is ending: it gets
        SETTLE_WAIT seconds to exit (settle), then the forward is restarted."""
        if self.alive() and self.local_port is not None:
            if _connects(self.host, self.local_port):
                return self.local_port
            self.settle()
        if not self._listened:  # never listened (a first start that failed): not a restart
            return self.start()
        return self.restart()

    def settle(self, timeout: float | None = None) -> bool:
        """After a request through the forward got no answer: wait up to `timeout` seconds
        (default SETTLE_WAIT) for kubectl to exit, as it does when the forward breaks. It closes
        the client's connection first and exits a moment later, so a check right after the
        failure would still find it running and reuse its dead port. True when kubectl no
        longer runs (the next ensure() restarts it); False when it still runs, as it does when
        the NIM itself failed."""
        wait = SETTLE_WAIT if timeout is None else float(timeout)
        deadline = time.monotonic() + max(wait, 0.0)
        while self.alive():
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.02)
        return True

    def restart(self) -> int:
        """Start kubectl again, on the same local port when it is free, else on a new one.
        Only max_restarts times (1): after that ScriptError port_forward_lost (exit 4)."""
        detail = _last_error(_read(self._err, tail=True)) if self._err else ""
        said = f" (kubectl: {detail})" if detail and detail != "no message" else ""
        if self.restarts >= self.max_restarts:
            self.stop()
            raise nsc.ScriptError(
                "port_forward_lost",
                f"the port-forward to pod {self.pod} stopped again after {self.restarts} "
                f"restart(s){said}",
                LOST_FIX,
                nsc.EXIT_EXTERNAL,
            )
        previous = self.local_port
        self._halt()
        self.restarts += 1
        nsc.log(f"nim-smoke-check: the port-forward to pod {self.pod} stopped{said}; restarting")
        if previous is not None:
            try:
                return self._launch(str(previous), self.start_timeout)
            except PortForwardError as exc:
                if not exc.busy:
                    raise
        return self._launch("", self.start_timeout)

    def stop(self) -> None:
        """Stop kubectl and remove its output files. Safe to call more than once."""
        self._halt()
        _untrack(self)
        folder, self._folder = self._folder, None
        if folder is not None:
            shutil.rmtree(folder, ignore_errors=True)

    def __enter__(self) -> PortForward:
        try:
            self.start()
        except BaseException:
            self.stop()
            raise
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def _files(self) -> tuple[Path, Path]:
        if self._folder is None:
            self._folder = Path(tempfile.mkdtemp(prefix="nim-smoke-check-forward-"))
        self._runs += 1
        out = self._folder / f"run-{self._runs}.out"
        self._err = self._folder / f"run-{self._runs}.err"
        return out, self._err

    def _launch(self, local: str, budget: float) -> int:
        self._local = local
        self.local_port = None
        out, err = self._files()
        with open(out, "wb") as stdout, open(err, "wb") as stderr:
            self._proc = run_tool(
                self.argv, timeout=budget, background=True, stdout=stdout, stderr=stderr
            )
        _track(self)
        deadline = time.monotonic() + budget
        found: tuple[str, int] | None = None
        while True:
            if found is None:
                found = self._forwarding(out)
            if found is not None and _connects(*found):
                self.host, self.local_port = found
                self._listened = True
                self._sync()
                return self.local_port
            if self._proc is not None and self._proc.poll() is not None:
                problem = self._start_error(err)
                self._halt()
                raise problem
            if time.monotonic() >= deadline:
                self._halt()
                said = _last_error(_read(err, tail=True))
                what = "accepted no connection" if found else "printed no Forwarding line"
                raise PortForwardError(
                    "port_forward_failed",
                    f"kubectl port-forward to pod {self.pod} {what} within {budget:g} s "
                    f"(kubectl: {said})",
                    FORWARD_FIX,
                    nsc.EXIT_EXTERNAL,
                )
            time.sleep(0.05)

    @staticmethod
    def _forwarding(out: Path) -> tuple[str, int] | None:
        """(host, local port) from kubectl's "Forwarding from" lines, 127.0.0.1 first."""
        found = {match.group(1): int(match.group(2)) for match in FORWARDING.finditer(_read(out))}
        for host, shown in (("127.0.0.1", "127.0.0.1"), ("::1", "[::1]")):
            if shown in found and 0 < found[shown] < 65536:
                return host, found[shown]
        return None

    def _start_error(self, err: Path) -> nsc.ScriptError:
        text = _read(err, tail=True)
        low = text.lower()
        said = _last_error(text)
        where = f"pod {self.pod} (context {self.context}, namespace {self.namespace})"
        if "unable to listen" in low:
            return PortForwardError(
                "port_forward_failed",
                f"kubectl port-forward could not listen on local port {self._local or 'any'}: "
                f"{said}",
                FORWARD_FIX,
                nsc.EXIT_EXTERNAL,
                busy=True,
            )
        cluster = _no_cluster(text, f"port-forward to {where}")
        if cluster:
            return PortForwardError("kubectl_failed", *cluster, nsc.EXIT_EXTERNAL)
        if "forbidden" in low:
            return PortForwardError(
                "rbac_missing", f"kubectl may not port-forward to {where}: {said}", RBAC_FIX,
                nsc.EXIT_PRECONDITION,
            )
        if "pod is not running" in low or NOT_FOUND.search(low):
            return PortForwardError(
                "pod_not_ready", f"kubectl cannot port-forward to {where}: {said}",
                NOT_READY_FIX, nsc.EXIT_PRECONDITION,
            )
        return PortForwardError(
            "port_forward_failed", f"kubectl port-forward to {where} stopped: {said}",
            FORWARD_FIX, nsc.EXIT_EXTERNAL,
        )

    def _halt(self) -> None:
        """Stop the kubectl process: SIGINT (terminate() off POSIX), wait, then kill."""
        proc, self._proc = self._proc, None
        if proc is None:
            return
        if proc.poll() is None:
            with contextlib.suppress(OSError):
                if os.name == "posix":
                    proc.send_signal(signal.SIGINT)
                else:
                    proc.terminate()
            try:
                proc.wait(timeout=STOP_WAIT)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(OSError):
                    proc.kill()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=5)

    def _sync(self) -> None:
        """Keep a bound Access on the current URL after a (re)start."""
        access = self.access
        if access is None or self.url is None:
            return
        if access.health_url == access.url:
            access.health_url = self.url
        access.url = self.url
        access.port_forward = self.facts()


_ACTIVE: set[PortForward] = set()
_ACTIVE_LOCK = threading.Lock()
_SAVED_HANDLERS: dict[int, Any] = {}


def _exit_on_signal(signum: int, frame: Any) -> None:
    raise SystemExit(128 + signum)


def _track(forward: PortForward) -> None:
    """Remember a running forward. With the first one, SIGTERM and SIGHUP (left at their default,
    which kills Python at once) are turned into SystemExit, so the finally blocks stop kubectl.
    Only from the main thread."""
    with _ACTIVE_LOCK:
        first = not _ACTIVE
        _ACTIVE.add(forward)
    if not first or threading.current_thread() is not threading.main_thread():
        return
    for name in ("SIGTERM", "SIGHUP"):
        signum = getattr(signal, name, None)
        if signum is None:
            continue
        with contextlib.suppress(OSError, ValueError):
            if signal.getsignal(signum) == signal.SIG_DFL:
                signal.signal(signum, _exit_on_signal)
                _SAVED_HANDLERS[signum] = signal.SIG_DFL


def _untrack(forward: PortForward) -> None:
    with _ACTIVE_LOCK:
        _ACTIVE.discard(forward)
        last = not _ACTIVE
    if not last or threading.current_thread() is not threading.main_thread():
        return
    for signum, handler in list(_SAVED_HANDLERS.items()):
        with contextlib.suppress(OSError, ValueError):
            if signal.getsignal(signum) is _exit_on_signal:
                signal.signal(signum, handler)
        _SAVED_HANDLERS.pop(signum, None)


@atexit.register
def _stop_all() -> None:
    """At exit, stop any forward a caller left running."""
    with _ACTIVE_LOCK:
        active = list(_ACTIVE)
    for forward in active:
        with contextlib.suppress(Exception):
            forward.stop()


# --- Kubernetes -----------------------------------------------------------------------------------

TARGET_KINDS = {"nimservice": "nimservice", "helm": "helm", "svc": "svc", "service": "svc",
                "pod": "pod"}


def parse_target(target: str) -> tuple[str, str]:
    """("nimservice" | "helm" | "svc" | "pod", NAME) from KIND/NAME (`service/` means svc/).
    ScriptError bad_input (exit 2) for anything else."""
    kind, slash, name = (target if isinstance(target, str) else "").strip().partition("/")
    found = TARGET_KINDS.get(kind.lower()) if slash else None
    pattern = RELEASE if found == "helm" else DNS_NAME
    if found is None or not pattern.fullmatch(name):
        raise nsc.ScriptError(
            "bad_input",
            "--k8s takes nimservice/NAME, helm/RELEASE, svc/NAME or pod/NAME, with a Kubernetes "
            "name (lower-case letters, digits, - and .)",
            "Pass, for example, --k8s nimservice/llama-3-1-8b-instruct --context CTX "
            "--namespace NS",
            nsc.EXIT_INPUT,
        )
    return found, name


def _check_scope(context: str, namespace: str) -> None:
    if not isinstance(context, str) or not CONTEXT.fullmatch(context):
        raise nsc.ScriptError(
            "bad_input",
            "--context takes the name of a context in kubectl's config (no spaces, not starting "
            "with -)",
            "Ask the user which kube context to use; never pick the current one",
            nsc.EXIT_INPUT,
        )
    if not isinstance(namespace, str) or not DNS_LABEL.fullmatch(namespace):
        raise nsc.ScriptError(
            "bad_input",
            "--namespace takes a Kubernetes namespace name (lower-case letters, digits and -)",
            "Ask the user which namespace the NIM runs in",
            nsc.EXIT_INPUT,
        )


def _selector_ok(text: str) -> bool:
    pairs = [pair.partition("=") for pair in text.split(",")]
    return bool(text) and all(
        equals and LABEL_KEY.fullmatch(key) and LABEL_VALUE.fullmatch(value)
        for key, equals, value in pairs
    )


def _name(obj: Any) -> str:
    return str(_dict(_dict(obj).get("metadata")).get("name") or "")


def _labels(obj: Any) -> dict[str, Any]:
    return _dict(_dict(_dict(obj).get("metadata")).get("labels"))


def _ready(pod: Any) -> bool:
    """Running, Ready and not being deleted."""
    if _dict(_dict(pod).get("metadata")).get("deletionTimestamp"):
        return False
    status = _dict(_dict(pod).get("status"))
    return status.get("phase") == "Running" and any(
        _dict(item).get("type") == "Ready" and _dict(item).get("status") == "True"
        for item in _list(status.get("conditions"))
    )


def _pod_state(pod: Any) -> str:
    status = _dict(_dict(pod).get("status"))
    if _dict(_dict(pod).get("metadata")).get("deletionTimestamp"):
        state = "terminating"
    elif status.get("phase") == "Running":
        state = "Running, not Ready"
    else:
        state = str(status.get("phase") or "unknown")
    return f"{_name(pod) or '?'}: {state}"


@dataclass
class _Found:
    """What a target resolves to before a pod is picked."""

    what: str  # for messages: "the NIMService llama"
    selector: dict[str, str] | None = None
    pod: dict[str, Any] | None = None
    port: str | int | None = None  # the Service's targetPort: a name or a number
    image: str | None = None  # the image a NIMService asks for
    notes: list[str] = field(default_factory=list)


@dataclass
class _Kube:
    kubectl: str
    context: str
    namespace: str

    def run(self, *words: str, timeout: float = KUBECTL_TIMEOUT) -> subprocess.CompletedProcess:
        argv = [self.kubectl, *words, "--context", self.context, "-n", self.namespace]
        return run_tool(argv, timeout=timeout)

    def get(self, *words: str, what: str) -> dict[str, Any]:
        proc = self.run("get", *words, "-o", "json")
        if proc.returncode != 0:
            raise self.error(proc, what)
        try:
            data = json.loads(proc.stdout)
        except (ValueError, RecursionError):  # not JSON, or nested too deeply to read
            data = None
        if not isinstance(data, dict):
            raise nsc.ScriptError(
                "kubectl_failed",
                f"kubectl get {' '.join(words)} did not print a JSON object",
                FORWARD_FIX,
                nsc.EXIT_EXTERNAL,
            )
        return data

    def contexts(self) -> list[str]:
        """The context names of kubectl's config (`config get-contexts -o name`), for a message."""
        proc = run_tool([self.kubectl, "config", "get-contexts", "-o", "name"], timeout=10)
        if proc.returncode != 0:
            return []
        return [line.strip() for line in proc.stdout.splitlines() if line.strip()][:30]

    def error(self, proc: subprocess.CompletedProcess, what: str) -> nsc.ScriptError:
        text = f"{proc.stderr or ''}\n{proc.stdout or ''}"
        low = text.lower()
        said = _last_error(text)
        where = f"in namespace {self.namespace} of context {self.context}"
        if proc.returncode == TIMED_OUT:
            return nsc.ScriptError(
                "kubectl_failed", f"kubectl could not {what} {where}: {said}", FORWARD_FIX,
                nsc.EXIT_EXTERNAL,
            )
        cluster = _no_cluster(text, f"{what} {where}")
        if cluster:  # no credentials, or no API server: nothing was asked of the cluster
            return nsc.ScriptError("kubectl_failed", *cluster, nsc.EXIT_EXTERNAL)
        if "forbidden" in low:
            fix = RBAC_FIX
            if "nimservice" in low:
                fix = (
                    "Ask the cluster admin for a role that can get nimservices.apps.nvidia.com "
                    "in this namespace (the built-in view role cannot), or pass --k8s "
                    "svc/NAME: the NIM Operator names the Service after the NIMService"
                )
            return nsc.ScriptError(
                "rbac_missing", f"the role of context {self.context} cannot {what} {where}: "
                f"{said}", fix, nsc.EXIT_PRECONDITION,
            )
        if re.search(r"context \S+ (does not exist|not found)|no context exists", low):
            names = self.contexts()
            listed = f" Contexts in kubectl's config: {', '.join(names)}." if names else ""
            return nsc.ScriptError(
                "context_not_found",
                f"the kube context {self.context} is not in kubectl's config: {said}",
                "Ask the user which context to use; never pick one yourself." + listed,
                nsc.EXIT_PRECONDITION,
            )
        if "doesn't have a resource type" in low:
            return nsc.ScriptError(
                "target_not_found",
                f"cannot {what}: the cluster has no such resource type ({said})",
                "A NIMService needs the NIM Operator; otherwise pass --k8s helm/RELEASE, "
                "svc/NAME or pod/NAME",
                nsc.EXIT_PRECONDITION,
            )
        if NOT_FOUND.search(low):
            return nsc.ScriptError(
                "target_not_found", f"cannot {what} {where}: {said}",
                "Check the name and the namespace with the user", nsc.EXIT_PRECONDITION,
            )
        return nsc.ScriptError(
            "kubectl_failed", f"kubectl could not {what} {where}: {said}", FORWARD_FIX,
            nsc.EXIT_EXTERNAL,
        )

    def require_port_forward(self) -> None:
        """`kubectl auth can-i create pods --subresource=portforward`: no is rbac_missing."""
        proc = self.run("auth", "can-i", "create", "pods", "--subresource=portforward",
                        timeout=20)
        if proc.returncode == 0:
            return
        if (proc.stdout or "").strip().lower().startswith("no"):
            raise nsc.ScriptError(
                "rbac_missing",
                f"kubectl auth can-i create pods --subresource=portforward says no for context "
                f"{self.context} in namespace {self.namespace}: the role may not port-forward "
                "there, or the namespace is not the right one (can-i says no for a namespace "
                "that does not exist too)",
                RBAC_FIX,
                nsc.EXIT_PRECONDITION,
            )
        raise self.error(proc, "check the port-forward permission")

    def ready_pod(self, found: _Found) -> dict[str, Any]:
        """One Running and Ready pod of the selector; a pod labelled nim-llm-role=leader
        first, then by name."""
        selector = _selector_text(found.selector or {})
        data = self.get("pods", "-l", selector, what=f"list the pods of {found.what}")
        pods = [pod for pod in _list(data.get("items")) if isinstance(pod, dict)]
        if not pods:
            raise nsc.ScriptError(
                "pod_not_ready",
                f"no pod of {found.what} in namespace {self.namespace} (selector {selector})",
                NOT_READY_FIX,
                nsc.EXIT_PRECONDITION,
            )
        ready = [pod for pod in pods if _ready(pod)]
        if not ready:
            states = "; ".join(_pod_state(pod) for pod in pods[:5])
            raise nsc.ScriptError(
                "pod_not_ready",
                f"no pod of {found.what} is Running and Ready ({states})",
                NOT_READY_FIX,
                nsc.EXIT_PRECONDITION,
            )
        ready.sort(key=lambda pod: (_labels(pod).get(LEADER_LABEL) != "leader", _name(pod)))
        return ready[0]


def _selector_text(selector: dict[str, str]) -> str:
    return ",".join(f"{key}={value}" for key, value in sorted(selector.items()))


def _selector(service: dict[str, Any], what: str) -> dict[str, str]:
    raw = _dict(_dict(service.get("spec")).get("selector"))
    selector = {str(key): str(value) for key, value in raw.items()}
    if not selector or not _selector_ok(_selector_text(selector)):
        raise nsc.ScriptError(
            "target_not_found",
            f"{what} has no usable pod selector, so no pod can be found for it",
            POD_FIX,
            nsc.EXIT_PRECONDITION,
        )
    return selector


def _service_port(service: dict[str, Any], names: tuple[str, ...]) -> tuple[str | int, str | None]:
    """(targetPort of the first TCP port named in `names`, else of the first TCP port; a note
    then). kubectl port-forward carries TCP only, so a UDP or SCTP port is never picked; a port
    with no protocol is TCP."""
    spec = _dict(service.get("spec"))
    ports = [
        port for port in _list(spec.get("ports"))
        if isinstance(port, dict) and port.get("protocol", "TCP") in ("TCP", None)
    ]
    if not ports:
        raise nsc.ScriptError(
            "target_not_found",
            f"the Service {_name(service)} has no TCP port",
            POD_FIX,
            nsc.EXIT_PRECONDITION,
        )
    chosen, note = None, None
    for wanted in names:
        chosen = next((port for port in ports if port.get("name") == wanted), None)
        if chosen is not None:
            break
    if chosen is None:
        chosen = ports[0]
        shown = chosen.get("name") or chosen.get("port")
        note = (
            f"The Service {_name(service)} has no port named {' or '.join(names)}; the check uses "
            f"its first TCP port, {shown}."
        )
    target = chosen.get("targetPort")
    if isinstance(target, str) and target and not re.fullmatch(r"[0-9]+", target):
        return target, note
    number = _port_number(target) or _port_number(chosen.get("port"))
    if number is None:
        raise nsc.ScriptError(
            "target_not_found",
            f"the Service {_name(service)} has no usable port number",
            POD_FIX,
            nsc.EXIT_PRECONDITION,
        )
    return number, note


def _spec_image(image: Any) -> str | None:
    spec = _dict(image)
    repository, tag = _text(spec.get("repository")), _text(spec.get("tag"))
    if repository is None:
        return None
    if tag is None:
        return repository
    return f"{repository}@{tag}" if tag.startswith("sha256:") else f"{repository}:{tag}"


def _resolve_nimservice(kube: _Kube, name: str) -> _Found:
    service = kube.get("nimservice", name, what=f"read the NIMService {name}")
    spec = _dict(service.get("spec"))
    found = _Found(f"the NIMService {name}", image=_spec_image(spec.get("image")))
    state = _dict(service.get("status")).get("state")
    if isinstance(state, str) and state and state != "Ready":
        found.notes.append(f"The NIMService reports state {state}.")
    if str(spec.get("inferencePlatform") or "").lower() == "kserve":
        found.selector = {KSERVE_LABEL: name}
        found.port = OPERATOR_PORT
        found.notes.append(
            f"KServe platform: the pod was found by its label {KSERVE_LABEL}={name}."
        )
        return found
    svc = kube.get("service", name, what=f"read the Service {name} of the NIMService")
    found.selector = _selector(svc, f"the Service {name}")
    found.port, note = _service_port(svc, (OPERATOR_PORT,))
    if note:
        found.notes.append(note)
    return found


def _resolve_helm(kube: _Kube, release: str) -> _Found:
    labels = f"app.kubernetes.io/name={HELM_CHART},app.kubernetes.io/instance={release}"
    data = kube.get("services", "-l", labels,
                    what=f"list the Services of the Helm release {release}")
    items = [item for item in _list(data.get("items")) if isinstance(item, dict)]
    live = [item for item in items if _dict(item.get("spec")).get("clusterIP") != "None"]
    if not live:
        only = " (only a headless one)" if items else ""
        raise nsc.ScriptError(
            "target_not_found",
            f"no Service of the Helm release {release} in namespace {kube.namespace}{only}: "
            f"none has the labels {labels}",
            f"Check the release name and the namespace with the user: kubectl get services "
            f"--context {kube.context} -n {kube.namespace} -l app.kubernetes.io/instance={release} "
            "lists the release's Services",
            nsc.EXIT_PRECONDITION,
        )
    named = [
        item for item in live
        if any(_dict(port).get("name") == HELM_PORT
               for port in _list(_dict(item.get("spec")).get("ports")))
    ]
    pool = sorted(named or live, key=_name)
    service = pool[0]
    found = _Found(f"the Helm release {release}")
    if len(pool) > 1:
        found.notes.append(
            f"The release has {len(pool)} Services; the check uses {_name(service)}."
        )
    found.selector = _selector(service, f"the Service {_name(service)}")
    found.port, note = _service_port(service, (HELM_PORT, OPERATOR_PORT))
    if note:
        found.notes.append(note)
    return found


def _resolve_service(kube: _Kube, name: str) -> _Found:
    service = kube.get("service", name, what=f"read the Service {name}")
    found = _Found(f"the Service {name}")
    found.selector = _selector(service, f"the Service {name}")
    found.port, note = _service_port(service, (OPERATOR_PORT, HELM_PORT))
    if note:
        found.notes.append(note)
    return found


class _PodMissing(nsc.ScriptError):
    """target_not_found for pod/NAME that kubectl does not find: a StatefulSet pod that is
    being created again keeps its name, so --wait-ready reads it again (_ready_target)."""


def _resolve_pod(kube: _Kube, name: str) -> _Found:
    try:
        pod = kube.get("pod", name, what=f"read the pod {name}")
    except nsc.ScriptError as exc:
        # for a pod, target_not_found is kubectl's NotFound for this very name
        if exc.code != "target_not_found":
            raise
        raise _PodMissing(exc.code, exc.message, exc.fix, exc.exit_code) from None
    if not _ready(pod):
        raise nsc.ScriptError(
            "pod_not_ready",
            f"the pod {_pod_state(pod)}",
            NOT_READY_FIX,
            nsc.EXIT_PRECONDITION,
        )
    return _Found(f"the pod {name}", pod=pod)


RESOLVERS = {
    "nimservice": _resolve_nimservice,
    "helm": _resolve_helm,
    "svc": _resolve_service,
    "pod": _resolve_pod,
}


def _containers(pod: dict[str, Any]) -> list[dict[str, Any]]:
    spec = _dict(pod.get("spec"))
    return [item for item in _list(spec.get("containers")) if isinstance(item, dict)]


def _declared_ports(container: dict[str, Any]) -> list[tuple[Any, int]]:
    found = []
    for port in _list(container.get("ports")):
        number = _port_number(_dict(port).get("containerPort"))
        if number is not None and _dict(port).get("protocol", "TCP") in ("TCP", None):
            found.append((_dict(port).get("name"), number))
    return found


def _nim_container(pod: dict[str, Any], port: str | int | None) -> dict[str, Any]:
    """The container that serves the forwarded port: for a numeric targetPort, the one container
    that declares that TCP port; else the one that declares it by name (the Service's
    targetPort, then api and http-openai); else the first container."""
    containers = _containers(pod)
    if not containers:
        raise nsc.ScriptError(
            "target_not_found", f"the pod {_name(pod)} has no containers",
            "Check the target with the user", nsc.EXIT_PRECONDITION,
        )
    if isinstance(port, int):
        serving = [
            container for container in containers
            if any(number == port for _, number in _declared_ports(container))
        ]
        if len(serving) == 1:
            return serving[0]
    wanted = [port] if isinstance(port, str) else []
    for name in (*wanted, OPERATOR_PORT, HELM_PORT):
        for container in containers:
            if any(declared == name for declared, _ in _declared_ports(container)):
                return container
    return containers[0]


def _not_read(pod: str, name: str, what: str) -> nsc.ScriptError:
    """setting_not_read (exit 3): the NIM container of `pod` takes `name`, a setting the run
    needs to reach the NIM, from a ConfigMap or a Secret (valueFrom), which is never read."""
    return nsc.ScriptError(
        "setting_not_read",
        f"the NIM container of pod {pod} takes {name} from a ConfigMap or a Secret (valueFrom), "
        f"which the smoke check never reads, so it cannot tell {what}",
        # --client-key stays last: the scrubber masks the word after a flag that names a key
        "Ask the user for an address that reaches the NIM and pass it with --url (https:// with "
        "--ca-file when NIM_SSL_MODE is TLS or MTLS, and for MTLS also --client-cert and "
        "--client-key)",
        nsc.EXIT_PRECONDITION,
    )


def _container_port(
    container: dict[str, Any], port: str | int | None, env: dict[str, str], pod: str
) -> tuple[int, str | None]:
    """The numeric container port to forward to, from the Service's targetPort (a name is
    looked up in the container's ports); for a pod target, its port api or http-openai, else its
    first port, else NIM_SERVER_PORT or 8000. A note when it had to fall back. setting_not_read
    when it needs NIM_SERVER_PORT and the pod takes it from valueFrom."""
    if isinstance(port, int):
        return port, None
    declared = _declared_ports(container)
    miss = None
    if isinstance(port, str):
        number = next((number for name, number in declared if name == port), None)
        if number is not None:
            return number, None
        miss = f"The pod declares no port named {port}"
    for wanted in (OPERATOR_PORT, HELM_PORT):
        number = next((number for name, number in declared if name == wanted), None)
        if number is not None:
            note = f"{miss}; the check uses its port {wanted}, {number}." if miss else None
            return number, note
    if declared:
        number, how = declared[0][1], f"its first port, {declared[0][1]}"
    else:
        if env.get("NIM_SERVER_PORT") == UNKNOWN:
            raise _not_read(pod, "NIM_SERVER_PORT", "which port the NIM listens on (the "
                            "container declares no port)")
        number = _port_number(env.get("NIM_SERVER_PORT")) or DEFAULT_PORT
        how = f"port {number} (NIM_SERVER_PORT, or the NIM default)"
    lead = miss or "The pod names no port api or http-openai"
    return number, f"{lead}; the check uses {how}."


def _pod_env(container: dict[str, Any]) -> dict[str, str]:
    """The NIM_ENV values of a container spec: the literal value, or UNKNOWN for one taken from
    a ConfigMap or a Secret (valueFrom), which is never read, so that a value set there is not
    taken for one that is not set. Every other variable is skipped unread."""
    found: dict[str, str] = {}
    for item in _list(container.get("env")):
        entry = _dict(item)
        key = entry.get("name")
        if key not in NIM_ENV:
            continue
        if "valueFrom" in entry:
            found[key] = UNKNOWN
            continue
        value = entry.get("value")
        found[key] = value if isinstance(value, str) else ""
    return found


def _image_id(pod: dict[str, Any], container: dict[str, Any]) -> str | None:
    for status in _list(_dict(pod.get("status")).get("containerStatuses")):
        if _dict(status).get("name") == container.get("name"):
            return _text(_dict(status).get("imageID"))
    return None


def _ready_target(
    kube: _Kube, kind: str, name: str, wait: float, poll: float
) -> tuple[_Found, dict[str, Any]]:
    """The resolved target and one Running and Ready pod of it. With `wait` > 0 (--wait-ready),
    a target with no such pod yet (pod_not_ready: a NIM still loading its model, or a pod not
    yet created; or pod/NAME that kubectl does not find yet) is read again every `poll`
    seconds, with the same read-only gets, until a pod is Ready or `wait` seconds have passed;
    the error then says how long it waited. Any other error stops at once."""
    wait = max(float(wait), 0.0)
    deadline = time.monotonic() + wait
    while True:
        try:
            found = RESOLVERS[kind](kube, name)
            pod = found.pod if found.pod is not None else kube.ready_pod(found)
            return found, pod
        except nsc.ScriptError as exc:
            waits = exc.code == "pod_not_ready" or isinstance(exc, _PodMissing)
            if not waits or wait <= 0:
                raise
            left = deadline - time.monotonic()
            if left <= 0:
                raise nsc.ScriptError(
                    exc.code, f"{exc.message}, after waiting {wait:g} s (--wait-ready)",
                    exc.fix, exc.exit_code,
                ) from exc
            nsc.log(f"nim-smoke-check: waiting for a Running and Ready pod ({exc.message}); "
                    f"{left:.0f} s of --wait-ready left")
            time.sleep(max(min(float(poll), left), 0.01))


def from_k8s(
    target: str,
    context: str,
    namespace: str,
    *,
    start_timeout: float = FORWARD_START,
    wait: float = 0.0,
    poll: float | None = None,
) -> tuple[Access, PortForward]:
    """Reach a NIM on Kubernetes through a port-forward this script owns, and return it running.

    target: nimservice/NAME (the NIM Operator: Service NAME, port api; on the KServe platform the
    predictor pod label), helm/RELEASE (the Service labelled app.kubernetes.io/name=HELM_CHART
    and app.kubernetes.io/instance=RELEASE that is not headless, port http-openai), svc/NAME (its
    selector; port api or http-openai, else the first TCP port) or pod/NAME.
    First `kubectl auth can-i create pods --subresource=portforward`; then one Running and Ready
    pod of the selector (nim-llm-role=leader first), waited for up to `wait` seconds
    (--wait-ready; read every `poll` seconds, default POD_POLL) when none is Ready yet; its
    container's image, args and literal NIM_PASSTHROUGH_ARGS (UNKNOWN from valueFrom or
    envFrom); and the numeric container port of the named port. NIM_SSL_MODE from valueFrom
    (and NIM_SERVER_PORT, when no port is declared) is setting_not_read: http or https, or the
    port, cannot be told without reading a ConfigMap or a Secret. Every kubectl call names
    --context and -n. The caller stops the forward (with, or PortForward.stop in a finally
    block). ScriptError: bad_input (exit 2); tool_missing, rbac_missing, context_not_found,
    target_not_found, pod_not_ready, setting_not_read (exit 3); kubectl_failed,
    port_forward_failed (exit 4)."""
    kind, name = parse_target(target)
    _check_scope(context, namespace)
    kube = _Kube(tool("kubectl"), context, namespace)
    kube.require_port_forward()
    found, pod = _ready_target(kube, kind, name, wait, POD_POLL if poll is None else poll)
    pod_name = _name(pod)
    if not DNS_NAME.fullmatch(pod_name):
        raise nsc.ScriptError(
            "kubectl_failed", f"kubectl listed a pod of {found.what} with no usable name",
            FORWARD_FIX, nsc.EXIT_EXTERNAL,
        )
    container = _nim_container(pod, found.port)
    env = _pod_env(container)
    if env.get("NIM_SSL_MODE") == UNKNOWN:  # TLS or MTLS would need https: never guess http
        raise _not_read(pod_name, "NIM_SSL_MODE", "whether the NIM serves http or https")
    remote, port_note = _container_port(container, found.port, env, pod_name)
    notes = list(found.notes)
    if port_note:
        notes.append(port_note)
    image = _text(container.get("image"))
    if found.image and image and found.image != image:
        notes.append(
            f"The NIMService asks for {found.image}, and pod {pod_name} runs {image}: a rollout "
            "may be in progress."
        )
    passthrough = env.get("NIM_PASSTHROUGH_ARGS")
    if passthrough == UNKNOWN:
        notes.append(
            "NIM_PASSTHROUGH_ARGS comes from a ConfigMap or a Secret (valueFrom), which is not "
            "read; its value is unknown."
        )
    if env.get("NIM_HEALTH_PORT") == UNKNOWN:
        notes.append(
            "NIM_HEALTH_PORT comes from a ConfigMap or a Secret (valueFrom), which is not read: "
            "if it moved the health routes to another port, the health check does not reach "
            "them (pass --health-url if another address does)."
        )
    unseen = [key for key in NIM_ENV if key not in env]  # an env entry wins over envFrom
    if unseen and _list(container.get("envFrom")):
        if passthrough is None:
            passthrough = UNKNOWN  # envFrom may replace the image default
        names = f"{', '.join(unseen[:-1])} or {unseen[-1]}" if len(unseen) > 1 else unseen[0]
        notes.append(
            "The container also takes variables from envFrom (a ConfigMap or a Secret), which is "
            f"not read; it may set {names}, which this run cannot see."
        )
    scheme, tls_note = _tls(env.get("NIM_SSL_MODE"), "127.0.0.1")
    if tls_note:
        notes.append(tls_note)
    health = _port_number(env.get("NIM_HEALTH_PORT"))
    forward = PortForward(
        kube.kubectl, context, namespace, pod_name, remote, start_timeout=start_timeout,
        scheme=scheme,
    )
    try:
        forward.start()
        url = forward.url or ""
        health_url: str | None = url
        if health and health != remote:
            health_url = None
            notes.append(
                f"NIM_HEALTH_PORT={health}: the health routes moved to that pod port, which the "
                "port-forward does not reach; the health check cannot reach them (pass "
                "--health-url if another address does)."
            )
        access = Access(
            "kubernetes",
            url,
            health_url,
            "port-forward",
            image=image or found.image,
            effective_passthrough=passthrough,
            effective_args=[arg for arg in _list(container.get("args")) if isinstance(arg, str)],
            notes=notes,
            port_forward=forward.facts(),
            config_known=True,
            image_id=_image_id(pod, container),
        )
        forward.access = access
    except BaseException:
        forward.stop()
        raise
    return access, forward


def main(argv: list[str] | None = None) -> None:
    parser = nsc.JsonArgumentParser(
        description="How nim-smoke-check reaches a NIM; import it, do not run it.",
        epilog="Example: python3 scripts/nsc_access.py --help",
    )
    parser.parse_args(argv)
    nsc.emit(True, "library", "done", {"note": "import this module; it has no command of its own"})


if __name__ == "__main__":
    main()
