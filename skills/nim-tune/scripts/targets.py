#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""targets: where run_campaign.py deploys candidates and runs AIPerf.

K8sTarget deploys each candidate as a NIMService on the NIM Operator and runs AIPerf in a client
pod; DockerTarget runs each candidate as a container on one GPU host (locally or over ssh) and
AIPerf in a client container on the same Docker network. Both present a candidate's replicas as
pod-like dicts ({"metadata": {"name"}, "status": {"podIP", "containerStatuses"}, "spec":
{"nodeName"}}), so the campaign logic, failure classification and resume are shared.

Library module for run_campaign.py; it has no command line of its own. Stdlib only.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shlex
import subprocess
import time
import uuid
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

LABEL = "nim-tune/campaign"
# a nonce made for each create, so an object a failed `kubectl create` still made is known as ours
CREATE_NONCE = "nim-tune/create-nonce"
# a full container Id, as `docker run -d` prints it
CONTAINER_ID = re.compile(r"[0-9a-f]{64}")
# names whose values are credentials; never written to a result file
# ("tokens" and "tokenizer" are engine settings, e.g. --max-num-batched-tokens, never secrets)
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


# credential variables a NIM, the benchmark client or Hugging Face read: their values are
# masked in everything the runner writes, whether or not a campaign names them
KNOWN_SECRET_ENV = (
    "NGC_API_KEY",
    "NGC_CLI_API_KEY",
    "NVIDIA_API_KEY",
    "NIM_API_KEY",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "OPENAI_API_KEY",
)


# the field names of the scripts' JSON output (their documented contract): never masked
OUTPUT_FIELDS = frozenset(
    {"ok", "state", "next", "facts", "errors", "code", "message", "fix", "warnings"}
)
# fields whose string values are read back as data (a status, an id, a resource name): a
# registered value masks one only when it is the whole value, never a part of it, so a short
# named credential cannot turn "measured" into "mea***" and make a resume redo a candidate
CONTROL_FIELDS = frozenset(
    {
        "state",
        "next",
        "code",
        "status",
        "kind",
        "fingerprint",
        "client_uid",
        "cleanup_pending",
        "cleanup_uid",
        "deployed_uid",
        "cleanup_target",
        "engine_flags_source",
        "flags_sha256",
    }
)
# fields whose whole value (lists and objects too) is read back and compared: the run paths a
# resume re-measures and the settings write_recommended.py compares to campaign.json. A string in
# them keeps flag and token-shape masking, a registered value that is all of it is masked at any
# length, and one inside it only from 6 characters (the floor for values seen in passing and for
# keys): a short named credential cannot turn "--max-num-seqs" into "--max-num-s***s" and make
# the writer refuse an unchanged campaign. A literal value in `measured.env` or `measured.args`
# is as protected as before where it matters: the campaign refuses secret-named variables and
# secret flags as literals (literal_secret), and a value that IS a credential is still masked.
# `order`, `nodes` and `baseline_fallbacks` are read back by a resume the same way
READBACK_FIELDS = frozenset(
    {"failed_runs", "overloaded_runs", "measured", "order", "nodes", "baseline_fallbacks"}
)


def json_keys(obj: Any) -> list[str]:
    """Every dictionary key in a JSON value, in order."""
    if isinstance(obj, dict):
        return [k for key, v in obj.items() for k in (key, *json_keys(v))]
    if isinstance(obj, list):
        return [k for v in obj for k in json_keys(v)]
    return []


def json_controls(obj: Any) -> list[tuple[str, Any]]:
    """Every string value of a CONTROL_FIELDS key and every value of a READBACK_FIELDS key in a
    JSON value, with its key, in order."""
    if isinstance(obj, dict):
        return [
            c
            for key, v in obj.items()
            for c in (
                [(key, v)]
                if key in READBACK_FIELDS or (key in CONTROL_FIELDS and isinstance(v, str))
                else json_controls(v)
            )
        ]
    if isinstance(obj, list):
        return [c for v in obj for c in json_controls(v)]
    return []


# A registered value shorter than this is masked only by name (a named credential), and then
# wherever it stands as its own token. It is left only where an ASCII letter or digit touches it
# on either side ("128", "u1", "/m/1b", "sha256:1abc"), or a "." joins it to a digit (a version:
# "1.2", "x:1.2"). Anything else is a boundary: "-", "_", ":", "/", "=", "@", quotes, whitespace,
# and an escape before it, whose last character is a letter or digit but not part of the value:
# a JSON escape ("\nab1" is "ab1" on a new line; \b \f \n \r \t and \uXXXX), a URL escape
# ("%20ab1") or an ANSI colour code ("\x1b[1mab1", raw or as \u001b; SHORT_AFTER_CSI). So
# "--tp 1", "admin:ab1", "auth=ab1-", "token=ab1/x" and "//u:pw@host" mask it. After it, an escape
# starts with "\" or "%", never a letter or digit, so it is a boundary there too.
SHORT_FLOOR = 6
SHORT_TOKEN = (
    r"(?:(?<![A-Za-z0-9])|(?<=\\[bfnrt])|(?<=\\u[0-9A-Fa-f]{{4}})|(?<=%[0-9A-Fa-f]{{2}})"
    r"|(?<=%25[0-9A-Fa-f]{{2}})|(?<=%2525[0-9A-Fa-f]{{2}}))"
    r"(?<![0-9]\.){}(?![A-Za-z0-9])(?!\.[0-9])"
)
# an escape sequence before it: of any length, so it is matched and kept (group 1) rather than
# looked behind. ESC raw or escaped (\u001b, \x1b, \033), then a control sequence ("[",
# parameters, a final byte), a charset designation ("(B") or a one-byte one ("M"); or the 8-bit
# CSI (\x9b)
SHORT_AFTER_CSI = (
    r"((?:(?:\x1b|\\[uU]001[bB]|\\x1[bB]|\\033|\\[eE])"
    r"(?:\[[0-?]*[ -/]*[@-~]|[()*+][0-9A-Za-z]|[@-Z\\_-])"
    r"|\x9b[0-?]*[ -/]*[@-~])){}(?![A-Za-z0-9])(?!\.[0-9])"
)


class Scrubber:
    """Masks credentials in any text the runner writes: results, logs, state and its output.

    It knows the values of the well-known credential variables (KNOWN_SECRET_ENV) and of the
    variables a campaign names (add_env), any value registered with add(), and masks well-known
    token shapes and secret-looking flags. The environment is read by name only, never listed.
    """

    def __init__(self) -> None:
        """Start with the values of the well-known credential variables, read by name."""
        self.values: set[str] = set()  # masked wherever they appear
        self.tokens: set[str] = set()  # short named values: masked only as a whole token
        self.add_env(KNOWN_SECRET_ENV)

    def add_env(self, names: Iterable[str]) -> None:
        """Register the values of these environment variables (the ones that are set).

        A named credential is masked at any length; an unset or empty one registers nothing
        (an empty value would put the mask between every character).
        """
        for name in names:
            value = os.environ.get(name, "")
            if value:
                self.add(value, named=True)

    def add(self, value: Any, named: bool = False) -> None:
        """Register a secret value, also in its JSON-escaped forms, so it is masked everywhere.

        A value seen in passing needs 6 characters (a shorter one would mask ordinary words); a
        NAMED credential (add_env) is registered whatever its length, never when empty. A named
        value under 6 characters is masked only where it stands as a whole token (see
        SHORT_TOKEN), so "1" never turns "--max-num-seqs 128" or "x:1.2" into "***28", "x:***.2".
        """
        floor = 1 if named else SHORT_FLOOR
        if isinstance(value, str | int) and len(str(value)) >= floor:
            v = str(value)
            # also as JSON escapes it ("a\"b", "é"), for text that was serialized first
            forms = {v, json.dumps(v)[1:-1], json.dumps(v, ensure_ascii=False)[1:-1]}
            if len(v) < SHORT_FLOOR:
                self.tokens |= forms
            else:
                self.values |= forms

    def text(self, s: str, min_len: int = 1) -> str:
        """The text with secret-looking flags, token shapes and registered values masked.

        Flags and token shapes go first: a short registered value (a named credential) must not
        break a flag name (`--api-key` into `--api-***`) and leave the value after it unmasked.
        MIN_LEN skips shorter registered values, for text that is structure, not content.
        """
        s = SECRET_FLAG.sub(lambda m: f"{m.group(1)}{m.group(2)}***", s)
        s = TOKEN_SHAPES.sub("***", s)
        for v in sorted(self.values | self.tokens, key=len, reverse=True):
            if len(v) < min_len:
                continue
            if v in self.values:
                s = s.replace(v, "***")
            else:
                s = re.sub(SHORT_AFTER_CSI.format(re.escape(v)), r"\1***", s)
                s = re.sub(SHORT_TOKEN.format(re.escape(v)), "***", s)
        return s

    def data(self, obj: Any, readback: bool = False) -> Any:
        """A copy of a JSON-ready value with every string masked; scrub before serializing.

        READBACK is set inside a READBACK_FIELDS value: its strings are masked as described
        there (a registered value inside one only from 6 characters).
        """
        if isinstance(obj, dict):
            # the documented field names are structure, never masked; any other key (an env or
            # file name) is masked by registered values of 6+ characters, values at any length
            return {
                str(k) if str(k) in OUTPUT_FIELDS else self.text(str(k), min_len=6): (
                    # a control value: flags, token shapes and a registered value that is all
                    # of it are masked (no registered value shorter than it is applied)
                    self.text(v, min_len=len(v))
                    if str(k) in CONTROL_FIELDS and isinstance(v, str)
                    else self.data(v, readback or str(k) in READBACK_FIELDS)
                )
                for k, v in obj.items()
            }
        if isinstance(obj, list | tuple):
            return [self.data(v, readback) for v in obj]
        if not isinstance(obj, str):
            return obj
        # min(6, len): a value shorter than the string can match only inside it, so a short one
        # masks it only when it is the whole string
        return self.text(obj, min_len=min(6, len(obj))) if readback else self.text(obj)

    def dumps(self, obj: Any, **kwargs: Any) -> str:
        """JSON of the scrubbed value: every string is redacted before it is encoded.

        A text pass then masks a registered value in its escaped form; if that would break the
        JSON, the redacted-before-encoding text is kept instead (never invalid JSON).
        """
        clean = json.dumps(self.data(obj), **kwargs)
        masked = self.text(clean, min_len=6)  # values were masked at any length above
        try:
            # and never if it renamed a key (the field names are the scripts' contract) or
            # changed a control or read-back value (it is read back)
            after, before = json.loads(masked), json.loads(clean)
            if json_keys(after) != json_keys(before):
                return clean
            if json_controls(after) != json_controls(before):
                return clean
        except ValueError:
            return clean
        return masked


# Local input files (campaign, manifests, dataset), read once per run once frozen: the bytes
# the approved plan hashed are the bytes the run deploys, even if a file changes meanwhile.
_INPUTS: dict[str, bytes] = {}
_FROZEN = False


def freeze_inputs() -> None:
    """From now on each local input file is read once and its first bytes are kept."""
    global _FROZEN
    _INPUTS.clear()
    _FROZEN = True


def read_input(path: Path) -> bytes:
    """The bytes of a local input file; once inputs are frozen, every read gives the same bytes."""
    # keyed by the path as named (absolute, symlinks NOT resolved): retargeting a symlink after
    # the plan was hashed must not make a later read return other bytes
    key = os.path.abspath(path)
    if key in _INPUTS:
        return _INPUTS[key]
    data = path.read_bytes()
    if _FROZEN:
        _INPUTS[key] = data
    return data


RETRY_WAITS = (0, 5, 15, 30)  # seconds before each try of a failed read
NIM_PORT = 8000


class KubeError(Exception):
    """A target command failed (kubectl or docker); the campaign stops and can be resumed."""


class RefusedError(Exception):
    """A resource that belongs to someone else is in the way; the campaign refuses to touch it."""


class ClientLostError(KubeError):
    """The benchmark client stopped; the candidate is not at fault."""


def redact(obj: Any) -> Any:
    """A copy safe to write to a result file: every credential-like value masked, anywhere.

    Masks `{"name": N, "value": V}` entries (container, init container and NIMService env),
    `{"env": {N: V}}` maps (a candidate's env) and `NAME=value` arguments after `-e` in a docker
    run line, when N looks like a credential.
    """
    if isinstance(obj, dict):
        out = {k: redact(v) for k, v in obj.items()}
        if isinstance(out.get("name"), str) and "value" in out and SECRET_NAME.search(out["name"]):
            out["value"] = "***"
        if isinstance(out.get("env"), dict):  # a candidate's or docker block's {NAME: value}
            out["env"] = {
                k: "***" if SECRET_NAME.search(str(k)) else v  # any type: a number can be a key
                for k, v in out["env"].items()
            }
        return out
    if isinstance(obj, list):
        items = [redact(v) for v in obj]
        for i, v in enumerate(items):
            prev = items[i - 1] if i else None
            if not isinstance(v, str):
                continue
            if prev in ("-e", "--env") and "=" in v:
                name = v.partition("=")[0]
                if SECRET_NAME.search(name):
                    items[i] = f"{name}=***"
            elif isinstance(prev, str) and SECRET_FLAG.fullmatch(f"{prev} {v}"):
                items[i] = "***"  # ["--api-key", "value"]
            elif v.startswith("-") and "=" in v and SECRET_FLAG.fullmatch(v):
                items[i] = v.partition("=")[0] + "=***"  # "--api-key=value"
        return items
    return obj


# where each kind of container and its status live in a pod; `kubectl logs -c NAME` and the
# default-container annotation find a name in all three (a sidecar is a restartable init container)
CONTAINER_KINDS = (
    ("containers", "containerStatuses"),
    ("initContainers", "initContainerStatuses"),
    ("ephemeralContainers", "ephemeralContainerStatuses"),
)
DEFAULT_CONTAINER = "kubectl.kubernetes.io/default-container"


def pod_containers(pod: Any, field: str = "containers") -> list[dict[str, Any]]:
    """spec.<field> of a pod object (spec.containers by default), checked: absent or null is [];
    ValueError naming the part when it is not a list of objects, or a container name is present
    but not a nonempty string (it is passed to `kubectl logs -c`)."""
    if not isinstance(pod, dict):
        raise ValueError("JSON")
    body = {} if pod.get("spec") is None else pod["spec"]
    if not isinstance(body, dict):
        raise ValueError("spec")
    boxes = [] if body.get(field) is None else body[field]
    if not isinstance(boxes, list) or not all(isinstance(b, dict) for b in boxes):
        raise ValueError(f"spec.{field}")
    if any("name" in b and not (isinstance(b["name"], str) and b["name"]) for b in boxes):
        raise ValueError(f"spec.{field}[].name")
    return boxes


def pod_container(
    pod: Any, name: str | None = None
) -> tuple[str, dict[str, Any], dict[str, Any] | None] | None:
    """The container `kubectl logs POD [-c NAME]` reads, as (status field, container, status),
    by kubectl's rules; None when there is no such container.

    With NAME: the container of that name of any kind (CONTAINER_KINDS). Without: the one the
    default-container annotation names, of any kind, else (no annotation, or a stale one) the
    first of spec.containers. STATUS is the entry of status.<status field> with the container's
    name, None when there is none (or the status is malformed). ValueError naming the part when
    the pod, its metadata, annotations or a container list is malformed.
    """
    if not isinstance(pod, dict):
        raise ValueError("JSON")
    for field, _ in CONTAINER_KINDS:  # every list, before any search: one found early
        pod_containers(pod, field)  # never hides a malformed later kind
    if name is None:
        # the types are checked before any default: only an absent or null field means "none"
        meta = {} if pod.get("metadata") is None else pod["metadata"]
        notes = meta.get("annotations") if isinstance(meta, dict) else []
        if not isinstance(meta, dict) or not isinstance(notes, dict | None):
            raise ValueError("metadata or annotations")
        named = (notes or {}).get(DEFAULT_CONTAINER)
        found = pod_container(pod, named) if isinstance(named, str) and named else None
        if found:
            return found
        boxes = pod_containers(pod)
        if not boxes:
            return None
        name = boxes[0].get("name")
        if not isinstance(name, str):
            return "containerStatuses", boxes[0], None  # unnamed: no status can be its own
    for field, status_field in CONTAINER_KINDS:
        for box in pod_containers(pod, field):
            if box.get("name") == name:
                status = pod.get("status")
                entries = status.get(status_field) if isinstance(status, dict) else None
                mine = [
                    e
                    for e in (entries if isinstance(entries, list) else [])
                    if isinstance(e, dict) and e.get("name") == name
                ]
                return status_field, box, (mine[0] if mine else None)
    return None


def run_cmd(
    argv: list[str], stdin: str | None = None, timeout: float = 120
) -> subprocess.CompletedProcess[str]:
    """Run a command with a timeout and capture its text output; never raises on a bad exit code."""
    try:
        return subprocess.run(
            argv,
            input=stdin,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return subprocess.CompletedProcess(argv, 124, "", str(exc))


# ---------------------------------------------------------------- Kubernetes (NIM Operator)


class Kube:
    """Thin kubectl wrapper bound to one context and namespace."""

    def __init__(self, context: str, namespace: str, log: Callable[[str], None]) -> None:
        """Remember the context and namespace; NIM_TUNE_KUBECTL can name the kubectl binary."""
        self.bin = os.environ.get("NIM_TUNE_KUBECTL", "kubectl")
        self.base = [self.bin, "--context", context, "-n", namespace, "--request-timeout=60s"]
        self.context, self.namespace, self.log = context, namespace, log

    def run(
        self, *args: str, stdin: str | None = None, timeout: float = 120, check: bool = False
    ) -> subprocess.CompletedProcess[str]:
        """Run kubectl in the saved context and namespace; check=True raises KubeError on error."""
        proc = run_cmd([*self.base, *args], stdin, timeout)
        if check and proc.returncode != 0:
            raise KubeError(f"kubectl {' '.join(args[:3])}: {proc.stderr.strip()[-400:]}")
        return proc

    def json(self, *args: str) -> Any:
        """The object as JSON, None when it does not exist; retries other errors, then raises.

        A failed call must never read as "gone": that would remove a healthy NIMService or blame
        a healthy client.
        """
        last = ""
        for wait in RETRY_WAITS:
            time.sleep(wait)
            proc = self.run(*args, "-o", "json")
            if proc.returncode == 0:
                try:
                    return json.loads(proc.stdout)
                except json.JSONDecodeError:
                    last = "unreadable JSON"
                    continue
            # only the API server's NotFound means gone: a kubectl configuration error such as
            # "context was not found" is a failed read, retried and then raised
            if "(NotFound)" in proc.stderr:
                return None
            last = proc.stderr.strip()[-300:]
        raise KubeError(f"kubectl {' '.join(args[:3])} kept failing: {last}")

    def apply(self, obj: dict[str, Any], dry_run: bool = False) -> dict[str, Any] | None:
        """Apply one object from JSON on stdin, as a server-side dry run when dry_run is set;
        the object the API server returned (None for a dry run, or when kubectl printed none)."""
        if dry_run:
            self.run("apply", "--dry-run=server", "-f", "-", stdin=json.dumps(obj), check=True)
            return None
        proc = self.run("apply", "-f", "-", "-o", "json", stdin=json.dumps(obj), check=True)
        try:
            made = json.loads(proc.stdout)
        except json.JSONDecodeError:
            return None
        return made if isinstance(made, dict) else None

    def create(self, obj: dict[str, Any]) -> dict[str, Any] | None:
        """Create one object from JSON on stdin; the object the API server made, with its uid.

        None when an object of that name already exists: that one is someone else's, never
        this call's. Unlike apply, create never takes over an object it did not make.

        The object carries a nonce made for this call (annotation CREATE_NONCE): when kubectl
        fails another way (a request timeout) or prints no object, the API server may still
        have made it, so it is read back by name and counts as made by this call only when it
        carries that nonce; anything else raises as before.
        """
        nonce = uuid.uuid4().hex
        obj = copy.deepcopy(obj)
        meta = obj.setdefault("metadata", {})
        meta["annotations"] = {**(meta.get("annotations") or {}), CREATE_NONCE: nonce}
        proc = self.run("create", "-f", "-", "-o", "json", stdin=json.dumps(obj))
        if proc.returncode != 0:
            if "(AlreadyExists)" in proc.stderr:
                return None
            error = f"kubectl create: {proc.stderr.strip()[-400:]}"
        else:
            try:
                made = json.loads(proc.stdout)
            except json.JSONDecodeError:
                made = None
            if isinstance(made, dict):
                return made
            error = "kubectl create did not print the object it created"
        try:
            there = self.json("get", str(obj.get("kind", "")).lower(), str(meta.get("name", "")))
        except KubeError:
            there = None  # unreadable: never known to be this call's, so the create failed
        there_meta = there.get("metadata") if isinstance(there, dict) else None
        notes = there_meta.get("annotations") if isinstance(there_meta, dict) else None
        if isinstance(there, dict) and isinstance(notes, dict) and notes.get(CREATE_NONCE) == nonce:
            self.log(f"kubectl create failed ({error[-120:]}), but the object it made is there")
            return there
        raise KubeError(error)

    def pods(self, name: str) -> list[dict[str, Any]]:
        """The pods of one NIMService, found by its app label."""
        # the NIM Operator labels pods app=NAME; listing a busy namespace unfiltered is slow
        data = self.json("get", "pods", "-l", f"app={name}")
        items = data.get("items", []) if isinstance(data, dict) else []
        return [p for p in items if p["metadata"]["name"].startswith(f"{name}-")]

    def exec(self, pod: str, *cmd: str, timeout: float = 120) -> subprocess.CompletedProcess[str]:
        """Run a command inside a pod."""
        return self.run("exec", pod, "--", *cmd, timeout=timeout)


def to_json_manifest(kube: Kube, path: Path) -> dict[str, Any]:
    """Load a manifest; YAML is converted by kubectl (client-side, no cluster change).

    The bytes come from read_input, so they are the ones the approved plan hashed.
    """
    try:
        raw = read_input(path)
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise KubeError(f"cannot read {path}: {exc}") from None
    if path.suffix != ".json":
        proc = kube.run("create", "--dry-run=client", "-o", "json", "-f", "-", stdin=text)
        if proc.returncode != 0:
            raise KubeError(f"cannot read {path}: {proc.stderr.strip()[-300:]}")
        text = proc.stdout
    try:
        data = json.loads(text)
    except ValueError as exc:  # the position only: the file's text may hold a secret
        raise KubeError(f"cannot read {path}: not valid JSON ({exc})") from None
    if not isinstance(data, dict):
        raise KubeError(f"{path} is not one object")
    return data


def check_kind(data: dict[str, Any], path: Path, kind: str, api: str | None = None) -> None:
    """Raise KubeError unless the manifest is one named object of this kind (and apiVersion).

    `kubectl apply` acts on whatever the file defines: a client manifest that is not a Pod would
    change another resource, and one the runner then never finds or deletes.
    """
    meta = data.get("metadata")
    if (
        data.get("kind") != kind
        or (api is not None and data.get("apiVersion") != api)
        or not isinstance(meta, dict)
        or not isinstance(meta.get("name"), str)
        or not meta["name"]
    ):
        want = f"{api} {kind}" if api else kind
        raise KubeError(f"{path} must define one {want} with metadata.name")


def render(c: dict[str, Any], base: dict[str, Any], cand: dict[str, Any]) -> dict[str, Any]:
    """Return the NIMService for one candidate: name, label, replicas, GPUs, args and env."""
    m = copy.deepcopy(base)
    meta = m.setdefault("metadata", {})
    meta["name"] = f"{c['name_prefix']}-{cand['id']}"
    meta["namespace"] = c["namespace"]
    for k in ("resourceVersion", "uid", "creationTimestamp", "generation", "managedFields"):
        meta.pop(k, None)
    meta.setdefault("labels", {}).update({LABEL: c["name"], "nim-tune/candidate": cand["id"]})
    m.pop("status", None)
    spec = m.setdefault("spec", {})
    if c.get("node"):
        # pin to the approved node: anywhere else would use GPUs the user did not approve
        spec.setdefault("nodeSelector", {})["kubernetes.io/hostname"] = c["node"]
    spec["replicas"] = cand["replicas"]
    spec["args"] = list(cand["args"])
    limits = spec.setdefault("resources", {}).setdefault("limits", {})
    limits["nvidia.com/gpu"] = cand["gpus"]
    env = [e for e in spec.get("env", []) if e.get("name") not in cand["env"]]
    env += [{"name": k, "value": str(v)} for k, v in cand["env"].items()]
    spec["env"] = env
    return m


class K8sTarget:
    """Target that runs each candidate as a NIMService on a Kubernetes cluster."""

    kind = "nim-operator"
    what = "NIMService"

    def __init__(self, c: dict[str, Any], campaign_dir: Path, log: Callable[[str], None]) -> None:
        """Bind the target to the campaign settings and a kubectl wrapper."""
        self.c, self.dir, self.log = c, campaign_dir, log
        self.kube = Kube(c["context"], c["namespace"], log)
        self.client = ""
        self.created_uid = ""  # the uid of the client pod ensure_client created, else ""
        # the uid of the NIMService the last deploy() made or reused, from the API server's
        # answer to it ("" when it made none, None when unknown); never read back by name
        self.deployed: str | None = None
        self._base: dict[str, Any] | None = None

    def path(self, p: str) -> Path:
        """A path from the campaign file; a relative one is read from the campaign's folder."""
        q = Path(p)
        return q if q.is_absolute() else self.dir / q

    def base(self) -> dict[str, Any]:
        """The base NIMService manifest as JSON, loaded once."""
        if self._base is None:
            path = self.path(self.c["base_manifest"])
            base = to_json_manifest(self.kube, path)
            check_kind(base, path, "NIMService")  # render() keeps its kind; remove() deletes one
            self._base = base
        return self._base

    # -- client
    def client_manifest(self) -> dict[str, Any]:
        """The client pod manifest, checked to be one v1 Pod in the campaign's namespace."""
        path = self.path(self.c["client_manifest"])
        manifest = to_json_manifest(self.kube, path)
        # checked before any read or apply: the file must be the client pod and nothing else
        check_kind(manifest, path, "Pod", "v1")
        if manifest["metadata"].get("namespace") not in (None, self.c["namespace"]):
            raise KubeError(f"{path} is for another namespace than {self.c['namespace']}")
        self.client = manifest["metadata"]["name"]
        return manifest

    def settled_client(self) -> dict[str, Any] | None:
        """The client pod once it is not terminating, None when there is none; ClientLostError
        when it is still terminating after 5 minutes."""
        pod = self.kube.json("get", "pod", self.client)
        for _ in range(30):
            if pod is None or not pod.get("metadata", {}).get("deletionTimestamp"):
                return pod
            # still listed as Running while it terminates: never reuse it
            self.log(f"client pod {self.client} is terminating; waiting for it to go")
            time.sleep(10)
            pod = self.kube.json("get", "pod", self.client)
        if pod is not None and pod.get("metadata", {}).get("deletionTimestamp"):
            raise ClientLostError(f"client pod {self.client} is still terminating after 5 minutes")
        return pod

    def dry_run_client(self) -> bool:
        """Check the client pod manifest with a server-side dry run of its create (RBAC,
        admission such as PodSecurity, quota), when no pod of that name is there to reuse;
        whether it was checked. KubeError when the API server refuses it."""
        manifest = self.client_manifest()
        if self.kube.json("get", "pod", self.client) is not None:
            return False  # an existing pod is reused, never created
        proc = self.kube.run(
            "create", "--dry-run=server", "-f", "-", "-o", "json", stdin=json.dumps(manifest)
        )
        if proc.returncode != 0:
            raise KubeError(f"client pod {self.client}: {proc.stderr.strip()[-400:]}")
        return True

    def ensure_client(self) -> bool:
        """Start or reuse the client pod; True when this call created the pod that is there now."""
        manifest = self.client_manifest()
        pod = self.settled_client()
        created = False
        self.created_uid = ""
        # a second create only when the name was taken by a pod that then went away (it was
        # terminating): a pod that is there and not terminating is reused, never owned
        for _ in range(2):
            if pod is not None:
                break
            # the uid comes from the create response: a later lookup by name could read a
            # same-named pod someone else made meanwhile and record it as this campaign's
            made = self.kube.create(manifest)
            if made is not None:
                created = True
                self.created_uid = str(made.get("metadata", {}).get("uid", ""))
                break
            self.log(f"client pod {self.client} appeared before it was created; not owned")
            pod = self.settled_client()
        pod = pod or {}
        if not created and pod.get("status", {}).get("phase") not in ("Pending", "Running"):
            phase = pod.get("status", {}).get("phase") or "gone"
            raise ClientLostError(f"client pod {self.client} is {phase}; it cannot run AIPerf")
        self.kube.run(
            "wait", "--for=condition=Ready", f"pod/{self.client}", "--timeout=600s", timeout=660
        )
        if created and self.client_uid() != self.created_uid:
            # replaced while it started: the pod there now is someone else's, never this
            # campaign's to delete
            self.log(f"client pod {self.client} was replaced while it started; not owned")
            self.created_uid = ""
            return False
        return created

    def client_uid(self) -> str:
        """The uid of the AIPerf client pod, to notice when it is replaced."""
        return str(
            (self.kube.json("get", "pod", self.client) or {}).get("metadata", {}).get("uid", "")
        )

    def check_client(self) -> None:
        """Raise ClientLostError unless the client pod exists and is running."""
        pod = self.kube.json("get", "pod", self.client) or {}
        phase = pod.get("status", {}).get("phase")
        if pod.get("metadata", {}).get("deletionTimestamp"):
            phase = "terminating"
        if phase != "Running":
            raise ClientLostError(f"client pod {self.client} is {phase or 'gone'}")

    def client_exec(
        self, *cmd: str, stdin: str | None = None, timeout: float = 120
    ) -> subprocess.CompletedProcess[str]:
        """Run a command in the client pod, sending stdin when given."""
        if stdin is not None:
            return self.kube.run(
                "exec", "-i", self.client, "--", *cmd, stdin=stdin, timeout=timeout
            )
        return self.kube.exec(self.client, *cmd, timeout=timeout)

    def remove_client(self, uid: str | None = None) -> None:
        """Delete the client pod without waiting for it to end; KubeError if that fails.

        A pod that is already gone counts as removed. With `uid` the API server deletes only
        that pod (a DeleteOptions uid precondition): a same-named replacement is never deleted.
        """
        if not uid:
            self.kube.run(
                "delete", "pod", self.client, "--wait=false", "--ignore-not-found", check=True
            )
            return
        body = {"kind": "DeleteOptions", "apiVersion": "v1", "preconditions": {"uid": uid}}
        path = f"/api/v1/namespaces/{self.kube.namespace}/pods/{self.client}"
        proc = self.kube.run("delete", "--raw", path, "-f", "-", stdin=json.dumps(body))
        if proc.returncode == 0 or "(NotFound)" in proc.stderr:
            return  # removed, or already gone
        if "(Conflict)" in proc.stderr or "Precondition failed" in proc.stderr:
            self.log(f"the client {self.client} was replaced (another uid); not removed")
            return
        raise KubeError(f"kubectl delete {self.client}: {proc.stderr.strip()[-300:]}")

    def client_fix(self) -> str:
        """The manual step that clears a lost client, for the error message."""
        return f"Delete pod {self.client}, then rerun the same command"

    # -- candidates
    def owner(self, name: str) -> str | None:
        """The campaign that owns `name` ('' if someone else's), or None if it does not exist."""
        obj = self.kube.json("get", "nimservice", name)
        if not obj:
            return None
        return str(obj["metadata"].get("labels", {}).get(LABEL, ""))

    def exists(self, name: str) -> bool:
        """Whether the NIMService itself exists, with or without pods; raises if unreadable."""
        return self.kube.json("get", "nimservice", name) is not None

    def describe(self, cand: dict[str, Any]) -> dict[str, Any]:
        """The manifest for one candidate, rendered from the base manifest."""
        return render(self.c, self.base(), cand)

    def deploy(self, cand: dict[str, Any]) -> tuple[str, str] | None:
        """Create the candidate's NIMService; returns None like the Docker target on success.

        Its uid, from the create's answer (or the read back bound to the create's nonce), goes
        to `deployed`. One already there (run_candidate checked it is this campaign's: a resume
        reuses it) is replaced (replace_owned), its uid taken from that answer: the one configured."""
        self.deployed = None
        obj = self.describe(cand)
        # the operation the dry run rehearsed: the replace of one already there (no create, so
        # no create permission needed), else a create; one made in between is AlreadyExists
        there = self.kube.json("get", "nimservice", obj["metadata"]["name"]) is not None
        made = None if there else self.kube.create(obj)
        if made is None:  # there, or AlreadyExists: reused only when it is this campaign's
            made, uid = self.replace_owned(obj)
            got = (made or {}).get("metadata", {}).get("uid") if isinstance(made, dict) else None
            if got != uid:
                raise KubeError(f"nimservice {obj['metadata']['name']} changed while it was applied")
        meta = made.get("metadata") if isinstance(made, dict) else None
        uid = meta.get("uid") if isinstance(meta, dict) else None
        self.deployed = str(uid) if uid else None  # none in the answer: unknown, never removed
        return None

    def replace_owned(
        self, obj: dict[str, Any], dry_run: bool = False
    ) -> tuple[dict[str, Any] | None, str]:
        """Replace (PUT) this campaign's NIMService that holds obj's name with obj's spec; the
        answer and the uid it was bound to.

        The spec becomes exactly the approved one (a field the manifest dropped is gone too),
        and the API server refuses another uid (precondition) or version (Conflict), so one
        replaced since is never configured. One without this campaign's label is refused."""
        name = obj["metadata"]["name"]
        now = self.kube.json("get", "nimservice", name) or {}
        meta = now.get("metadata") or {}
        if (meta.get("labels") or {}).get(LABEL) != self.c["name"]:
            raise RefusedError(f"nimservice {name} exists and was not created by this campaign")
        if not meta.get("uid") or not meta.get("resourceVersion"):
            raise KubeError(f"nimservice {name} has no uid or resourceVersion to bind to")
        body = copy.deepcopy(now)
        body.pop("status", None)
        bmeta = body["metadata"]
        bmeta.pop("managedFields", None)
        for k in ("labels", "annotations"):
            bmeta[k] = {**(bmeta.get(k) or {}), **(obj["metadata"].get(k) or {})}
        body["spec"] = obj.get("spec") or {}
        flags = ["--dry-run=server"] if dry_run else []
        proc = self.kube.run(
            "replace", *flags, "-f", "-", "-o", "json", stdin=json.dumps(body), check=True
        )
        try:
            made = json.loads(proc.stdout)
        except json.JSONDecodeError:
            made = None
        return (made if isinstance(made, dict) else None), str(meta["uid"])

    def dry_run(self, cand: dict[str, Any]) -> None:
        """Check the candidate's manifest with a server-side dry run of the operation the live
        run uses: a create, or the uid- and version-bound replace of this campaign's NIMService
        already there (another verb, so another permission)."""
        obj = self.describe(cand)
        if self.kube.json("get", "nimservice", obj["metadata"]["name"]) is None:
            self.kube.run("create", "--dry-run=server", "-f", "-", stdin=json.dumps(obj), check=True)
            return
        try:
            self.replace_owned(obj, dry_run=True)
        except RefusedError as exc:  # run_candidate's owner check refuses it live
            raise KubeError(str(exc)) from exc

    def ready_count(self, name: str) -> int:
        """How many replicas of the deployment are ready."""
        dep = self.kube.json("get", "deploy", name) or {}
        return int(dep.get("status", {}).get("readyReplicas", 0) or 0)

    def replicas(self, name: str) -> list[dict[str, Any]]:
        """The pods that serve one NIMService."""
        return self.kube.pods(name)

    def logs(self, replica: str, previous: bool = False, tail: int | None = None) -> str:
        """The log of one pod, optionally the previous container or only the last lines."""
        extra = (["--previous"] if previous else []) + ([f"--tail={tail}"] if tail else [])
        return self.kube.run("logs", replica, *extra).stdout

    def spec(self, replica: str, allow: list[str] | None = None) -> Any:
        """The pod's spec with literal credentials masked."""
        # a literal credential in any container, init container included, stays out of results
        return redact(self.kube.json("get", "pod", replica) or {})

    def exec(
        self, replica: str, *cmd: str, timeout: float = 120
    ) -> subprocess.CompletedProcess[str]:
        """Run a command inside one replica pod."""
        return self.kube.exec(replica, *cmd, timeout=timeout)

    def engagement_args(self, replica: str) -> list[str]:
        """The arguments that tell check_nim_engagement.py how to reach a pod."""
        return ["k8s", replica, "-n", self.c["namespace"], "--context", self.c["context"]]

    def ident(self, name: str) -> str | None:
        """The NIMService's uid ('' if it has none), None if it does not exist; raises if
        unreadable."""
        obj = self.kube.json("get", "nimservice", name)
        return None if not obj else str(obj["metadata"].get("uid") or "")

    def kept(self, name: str, uid: str) -> bool:
        """Whether the NIMService recorded as `uid` still exists: a NIMService is never
        renamed, so one that `name` no longer resolves to is gone."""
        return bool(uid) and self.ident(name) == uid

    def strays(self, name: str) -> bool:
        """Whether pods that may be a gone NIMService's still run under its name: pods are
        found by the app label, which a same-named replacement's pods carry too, so any counts."""
        return bool(self.kube.pods(name))

    def remove(self, name: str, uid: str | None = None) -> bool:
        """Delete one NIMService if this campaign made it; return whether it did.

        KubeError when the delete fails, so a caller never takes it for gone (one that is
        already gone counts as removed). Only the NIMService whose owner was read is deleted (a
        uid precondition): a same-named replacement made since is never deleted. With `uid`
        (the one recorded when its removal was marked) only that NIMService is deleted."""
        obj = self.kube.json("get", "nimservice", name)
        if not obj or obj["metadata"].get("labels", {}).get(LABEL, "") != self.c["name"]:
            return False
        if uid is not None and str(obj["metadata"].get("uid") or "") != uid:
            self.log(f"{name} is not the NIMService recorded (another uid); not removed")
            return False
        uid, api = obj["metadata"].get("uid"), obj.get("apiVersion")
        if not uid or not api:  # nothing to bind the delete to (an API server sets both)
            gone = self.kube.run("delete", "nimservice", name, "--wait=true", timeout=600)
            if gone.returncode != 0 and "(NotFound)" not in gone.stderr:
                raise KubeError(f"kubectl delete nimservice {name}: {gone.stderr.strip()[-300:]}")
        else:
            body = {"kind": "DeleteOptions", "apiVersion": "v1", "preconditions": {"uid": uid}}
            path = f"/apis/{api}/namespaces/{self.kube.namespace}/nimservices/{name}"
            gone = self.kube.run("delete", "--raw", path, "-f", "-", stdin=json.dumps(body))
            if "(Conflict)" in gone.stderr or "Precondition failed" in gone.stderr:
                self.log(f"{name} was replaced (another uid); not removed")
                return False
            if gone.returncode != 0 and "(NotFound)" not in gone.stderr:
                raise KubeError(f"kubectl delete nimservice {name}: {gone.stderr.strip()[-300:]}")
            # like --wait=true: until that NIMService (its finalizers done) is gone
            for _ in range(60):
                now = self.kube.json("get", "nimservice", name)
                if not now or now["metadata"].get("uid") != uid:
                    break
                time.sleep(10)
            else:
                raise KubeError(f"nimservice {name} is still being deleted after 10 minutes")
        for _ in range(30):
            if not self.kube.pods(name):
                break
            time.sleep(10)
        return True

    def gpus(self) -> str:
        """The last two nvidia.com/gpu lines of the target node, for the plan."""
        node = self.c.get("node")
        if not node:
            return "node not set"
        proc = self.kube.run("describe", "node", node)
        lines = [ln.strip() for ln in proc.stdout.splitlines() if "nvidia.com/gpu" in ln]
        return "; ".join(lines[-2:]) or "unavailable"

    def created_resources(self, names: list[str]) -> list[str]:
        """What a live run creates, for the plan shown to the user."""
        return [*names, f"client pod from {self.c['client_manifest']}"]


# ---------------------------------------------------------------- Docker on one GPU host

# one exact PEP 440 version, in any spelling pip accepts in `aiperf==VERSION` (0.13.0, v0.13.0,
# 0.14.0rc1, 0.14.0-rc.1, 1!0.13.0, 0.13.0+local): pip refuses anything else (0.13.0banana,
# 0.13.*), and only after the network and the client were started. ASCII only, no whitespace.
AIPERF_VERSION = re.compile(
    r"""
    v?
    ([0-9]+!)?                                  # epoch
    [0-9]+(\.[0-9]+)*                           # release
    ([-_.]?(a|b|c|rc|pre)[-_.]?[0-9]*)?         # pre-release (older pip refuses alpha, beta)
    (-[0-9]+|[-_.]?(post|rev|r)[-_.]?[0-9]*)?   # post-release
    ([-_.]?dev[-_.]?[0-9]*)?                    # dev release
    (\+[a-z0-9]+([-_.][a-z0-9]+)*)?             # local version
    """,
    re.VERBOSE | re.IGNORECASE | re.ASCII,
)
# the client's fixed program: the package to install is its one argument ("$1"), never source
CLIENT_PROGRAM = 'pip install -q "$1" && sleep infinity'


def started_of(box: Any) -> str:
    """A `docker inspect` entry's State.StartedAt as text, "" when it has none."""
    state = box.get("State") if isinstance(box, dict) else None
    return str((state.get("StartedAt") if isinstance(state, dict) else None) or "")


class DockerTarget:
    """Candidates as containers on one Docker host; AIPerf in a client container beside them.

    The campaign's `docker` block names the image, the GPU devices, the model environment, the
    host's NIM cache folder and, for a remote host, `ssh` (and `sudo` when the host needs it).
    Secrets are passed by name (`env_from_host`) or as a file on the host (`env_file`), never by
    value. Containers are labelled with the campaign and only those are ever removed.
    """

    kind = "docker"
    what = "container"

    def __init__(self, c: dict[str, Any], campaign_dir: Path, log: Callable[[str], None]) -> None:
        """Bind the target to the campaign settings; names the network and client container."""
        self.c, self.dir, self.log = c, campaign_dir, log
        self.d = c["docker"]
        self.network = self.d.get("network") or f"{c['name_prefix']}-net"
        self.client = f"{c['name_prefix']}-aiperf"
        self._home: str | None = None
        self.created_uid = ""  # the id of the client container ensure_client started, else ""
        # the Id of the container the last deploy() started or reused ("" when it started none,
        # None when unknown): from `docker run -d`, never read back by name later
        self.deployed: str | None = None
        # per replica, the container (id, start) its saved log was read from: spec() must match
        self._log_from: dict[str, tuple[str, str]] = {}

    def host_path(self, p: str) -> str:
        """A path on the Docker host; `~/` is that host's home, read from the host itself."""
        if not p.startswith("~/"):
            return p
        if self._home is None:
            got = run_cmd(self.host_argv(["sh", "-c", 'printf %s "$HOME"']))
            if got.returncode != 0 or not got.stdout.startswith("/"):
                raise KubeError(
                    f"cannot read the home folder on the Docker host: {got.stderr[-200:]}"
                )
            self._home = got.stdout.strip()
        return f"{self._home}/{p[2:]}"

    # -- the docker CLI, local or over ssh
    def host_argv(self, argv: list[str]) -> list[str]:
        """The argv to run on the GPU host: add sudo, and wrap it in ssh when docker.ssh is set."""
        bin_ = os.environ.get("NIM_TUNE_DOCKER", "docker")
        if argv and argv[0] == "docker":
            argv = [*(["sudo", "-n"] if self.d.get("sudo") else []), bin_, *argv[1:]]
        if self.d.get("ssh"):
            # one remote shell line; every argument quoted, so values never split or expand
            return ["ssh", "-o", "BatchMode=yes", "--", self.d["ssh"], shlex.join(argv)]
        return argv

    def docker(
        self, *args: str, stdin: str | None = None, timeout: float = 120
    ) -> subprocess.CompletedProcess[str]:
        """Run one docker command on the GPU host (locally or over ssh)."""
        return run_cmd(self.host_argv(["docker", *args]), stdin, timeout)

    def inspect(self, name: str) -> dict[str, Any] | None:
        """`docker container inspect` of one container, None if it does not exist; retries
        other errors. Containers only: a volume, network or image with that name is not one."""
        last = ""
        for wait in RETRY_WAITS:
            time.sleep(wait)
            proc = self.docker("container", "inspect", name)
            if proc.returncode == 0:
                try:
                    data = json.loads(proc.stdout)
                except json.JSONDecodeError:
                    last = "unreadable JSON"
                    continue
                return data[0] if isinstance(data, list) and data else None
            if "No such" in proc.stderr or "no such" in proc.stderr:
                return None
            last = proc.stderr.strip()[-300:]
        raise KubeError(f"docker inspect {name} kept failing: {last}")

    def labels(self, cid: str) -> list[str]:
        """The labels that mark a container as this campaign's, for one candidate."""
        return ["--label", f"{LABEL}={self.c['name']}", "--label", f"nim-tune/candidate={cid}"]

    def ensure_network(self) -> None:
        """Create the campaign's docker network if it does not exist."""
        if self.docker("network", "inspect", self.network).returncode != 0:
            made = self.docker(
                "network", "create", "--label", f"{LABEL}={self.c['name']}", self.network
            )
            if made.returncode != 0 and "already exists" not in made.stderr:
                raise KubeError(
                    f"docker network create {self.network}: {made.stderr.strip()[-300:]}"
                )

    # -- client
    def aiperf_package(self) -> str:
        """The pip requirement for the client, `aiperf==VERSION`; KubeError for a bad version."""
        version = self.d.get("aiperf_version", "0.13.0")
        if not isinstance(version, str) or not AIPERF_VERSION.fullmatch(version):
            raise KubeError(f"docker.aiperf_version {version!r} is not a version such as 0.13.0")
        return f"aiperf=={version}"

    def ensure_client(self) -> bool:
        """Make sure the network and client container exist; True if it started the client."""
        package = self.aiperf_package()  # before any change on the host
        self.created_uid = ""
        self.ensure_network()
        box = self.inspect(self.client)
        created = False
        if box is None:
            env = [x for k in self.d.get("client_env_from_host", ["HF_TOKEN"]) for x in ("-e", k)]
            started = self.docker(
                "run",
                "-d",
                "--name",
                self.client,
                "--network",
                self.network,
                *self.labels("aiperf-client"),
                *env,
                "--restart",
                "no",
                self.d.get("client_image", "python:3.12-slim"),
                "bash",
                "-c",
                CLIENT_PROGRAM,
                "aiperf-client",  # $0
                package,  # $1
                timeout=300,
            )
            if started.returncode != 0:
                raise KubeError(f"cannot start {self.client}: {started.stderr.strip()[-300:]}")
            created = True
            # the id `docker run -d` printed: the container this call made, not a later lookup
            self.created_uid = (started.stdout.strip().splitlines() or [""])[-1]
        elif not box.get("State", {}).get("Running"):
            raise ClientLostError(
                f"client container {self.client} is {box.get('State', {}).get('Status')}"
            )
        return created

    def client_uid(self) -> str:
        """The id of the client container, to notice when it is replaced."""
        return str((self.inspect(self.client) or {}).get("Id", ""))

    def check_client(self) -> None:
        """Raise ClientLostError unless the client container is running."""
        box = self.inspect(self.client) or {}
        if not box.get("State", {}).get("Running"):
            status = box.get("State", {}).get("Status") or "gone"
            raise ClientLostError(f"client container {self.client} is {status}")

    def client_exec(
        self, *cmd: str, stdin: str | None = None, timeout: float = 120
    ) -> subprocess.CompletedProcess[str]:
        """Run a command in the client container, sending stdin when given."""
        extra = ["-i"] if stdin is not None else []
        return self.docker("exec", *extra, self.client, *cmd, stdin=stdin, timeout=timeout)

    def remove_client(self, uid: str | None = None) -> None:
        """Remove the client container and, if nothing else uses it, the network.

        With `uid`, docker removes that container id, never a same-named replacement.
        KubeError when docker cannot remove the client; one that is already gone counts as
        removed.
        """
        # ownership is read from the container that is removed: the recorded id when given
        if self.owner(uid or self.client) == self.c["name"]:
            gone = self.docker("rm", "-f", uid or self.client)
            if gone.returncode != 0 and "no such container" not in gone.stderr.lower():
                raise KubeError(f"docker rm {self.client}: {gone.stderr.strip()[-300:]}")
        # the network too, when this campaign made it and nothing else is still attached
        net = self.docker("network", "inspect", self.network)
        try:
            info = json.loads(net.stdout)[0] if net.returncode == 0 else {}
        except (json.JSONDecodeError, IndexError):
            info = {}
        if (info.get("Labels") or {}).get(LABEL) == self.c["name"] and not info.get("Containers"):
            self.docker("network", "rm", self.network)

    def client_fix(self) -> str:
        """The manual step that clears a lost client, for the error message."""
        return f"Remove container {self.client} (docker rm -f), then rerun the same command"

    # -- candidates
    def owner(self, name: str) -> str | None:
        """The campaign name on a container's label; None if it does not exist."""
        box = self.inspect(name)
        if box is None:
            return None
        return str((box.get("Config", {}).get("Labels") or {}).get(LABEL, ""))

    def exists(self, name: str) -> bool:
        """Whether the container exists, running or stopped; raises if it cannot be read."""
        return self.inspect(name) is not None

    def devices(self, n: int) -> str:
        """The first n GPU indexes from docker.gpus, as a comma list."""
        devs = [str(x) for x in str(self.d["gpus"]).split(",") if str(x).strip()]
        return ",".join(devs[:n])

    def describe(self, cand: dict[str, Any]) -> dict[str, Any]:
        """The `docker run` arguments for one candidate (no secret values)."""
        name = f"{self.c['name_prefix']}-{cand['id']}"
        env = {**self.d.get("env", {}), **cand["env"]}
        argv = [
            "run",
            "-d",
            "--name",
            name,
            "--network",
            self.network,
            *self.labels(cand["id"]),
            "--gpus",
            f'"device={self.devices(cand["gpus"])}"',
            f"--shm-size={self.d.get('shm_size', '16g')}",
            "--restart",
            "no",
        ]
        if self.d.get("user"):
            argv += ["-u", str(self.d["user"])]
        for k in self.d.get("env_from_host", []):
            argv += ["-e", k]  # the value comes from the host's environment, never from here
        if self.d.get("env_file"):
            argv += ["--env-file", self.host_path(self.d["env_file"])]
        for k, v in env.items():
            argv += ["-e", f"{k}={v}"]
        if self.d.get("cache_dir"):
            argv += ["-v", f"{self.host_path(self.d['cache_dir'])}:/opt/nim/.cache"]
        argv += [*self.d.get("extra_run_args", []), self.d["image"], *cand["args"]]
        # the exact settings, as a label: a container from an older run is never measured
        print_ = json.dumps(redact(argv), sort_keys=True).encode()
        stamp = ["--label", f"nim-tune/fingerprint={hashlib.sha256(print_).hexdigest()[:16]}"]
        argv[2:2] = stamp
        return {"name": name, "docker": argv, "fingerprint": stamp[1].split("=", 1)[1]}

    def name_clash(self, cand: dict[str, Any]) -> str | None:
        """Why the candidate's container name is the client's, or None when it is not."""
        name = f"{self.c['name_prefix']}-{cand['id']}"
        if name != self.client:
            return None
        return f"candidate id {cand['id']!r} names container {name}, the benchmark client"

    def deploy(self, cand: dict[str, Any]) -> tuple[str, str] | None:
        """Start the candidate's container; None if started or already running as configured."""
        self.deployed = ""
        clash = self.name_clash(cand)
        if clash:  # never recreate the client as a NIM: refused before any docker command
            return ("config", clash)
        spec = self.describe(cand)
        # a log read before this deploy binds no later spec(): it read another instance
        self._log_from.pop(spec["name"], None)
        box = self.inspect(spec["name"])
        if box is not None:
            labels = box.get("Config", {}).get("Labels") or {}
            if labels.get("nim-tune/fingerprint") == spec["fingerprint"]:
                # this campaign's container with the same settings: reuse it (the one inspected)
                self.deployed = str(box.get("Id") or "") or None
                return None
            if labels.get(LABEL) == self.c["name"]:
                # ours, but started with other settings (an older run): never measure it
                self.log(f"{spec['name']}: settings changed since it started; recreating it")
                # by the inspected id: a same-named replacement made since is never removed;
                # a failed rm is raised, never taken for gone (docker run would then clash)
                gone = self.docker("rm", "-f", str(box.get("Id") or spec["name"]), timeout=300)
                if gone.returncode != 0 and "no such container" not in gone.stderr.lower():
                    raise KubeError(f"docker rm {spec['name']}: {gone.stderr.strip()[-300:]}")
            else:
                return ("config", f"container {spec['name']} exists and is not this campaign's")
        # a nonce made for this run, as a label: a container a failed or unparsed `docker run`
        # still made is known as this call's only when it carries it (never a same-named one)
        nonce = uuid.uuid4().hex
        argv = [*spec["docker"][:2], "--label", f"{CREATE_NONCE}={nonce}", *spec["docker"][2:]]
        started = self.docker(*argv, timeout=900)
        printed = (started.stdout.strip().splitlines() or [""])[-1].strip()
        if started.returncode == 0 and CONTAINER_ID.fullmatch(printed):
            self.deployed = printed  # the Id `docker run -d` printed: the container it made
        else:
            try:
                box = self.inspect(spec["name"])
            except KubeError:
                self.deployed = None  # unknown: never removed by the runner
            else:
                labels = (box or {}).get("Config", {}).get("Labels") or {}
                if box is not None and labels.get(CREATE_NONCE) == nonce:
                    self.deployed = str(box.get("Id") or "") or None
        if started.returncode == 0:
            return None
        msg = started.stderr.strip()[-300:]
        bad = (
            "unknown flag",
            "invalid",
            "no such image",
            "manifest unknown",
            "could not select",
            "no such file or directory",  # a missing env file or mount: retrying cannot fix it
        )
        return ("config" if any(b in msg.lower() for b in bad) else "intermittent", msg)

    def dry_run_client(self) -> bool:
        """Docker has no server-side dry run; the client's AIPerf version is checked in
        dry_run. Never checked here (False), like K8sTarget when it would reuse a client."""
        return False

    def dry_run(self, cand: dict[str, Any]) -> None:
        """Check the daemon and the files docker run will read, without starting anything."""
        clash = self.name_clash(cand)
        if clash:
            raise KubeError(f"{clash}; rename the candidate")
        self.aiperf_package()  # the client's version, checked before the plan is approved
        if self.docker("version", "--format", "{{.Server.Version}}").returncode != 0:
            raise KubeError("cannot reach the Docker daemon on the target host")
        # the files docker run will read, checked on the host before any GPU is used
        checks = [("env_file", "-r"), ("cache_dir", "-d")]
        for key, test in checks:
            if self.d.get(key):
                where = self.host_path(self.d[key])
                if run_cmd(self.host_argv(["test", test, where])).returncode != 0:
                    raise KubeError(f"docker.{key} {where} is missing on the Docker host")

    def ready_count(self, name: str) -> int:
        """1 when the NIM's ready endpoint answers 200, else 0."""
        probe = (
            "import sys,urllib.request as u\n"
            "try:\n"
            "    print(u.urlopen('http://'+sys.argv[1]+':8000/v1/health/ready',timeout=5).status)\n"
            "except Exception:\n"
            "    print(0)\n"
        )
        # the probe runs in the client: a lost client must stop the campaign, not fail the NIM
        self.check_client()
        box = self.inspect(name) or {}
        if not box.get("State", {}).get("Running"):
            return 0
        return 1 if self.client_exec("python", "-c", probe, name).stdout.strip() == "200" else 0

    def replicas(self, name: str) -> list[dict[str, Any]]:
        """The container as a pod-like record; a stopped container counts as a restart."""
        box = self.inspect(name)
        if box is None:
            return []
        st = box.get("State", {})
        if st.get("Running"):
            state: dict[str, Any] = {"running": {}}
            last: dict[str, Any] = {}
        else:
            reason = "OOMKilled" if st.get("OOMKilled") else "Error"
            term = {"reason": reason, "exitCode": st.get("ExitCode")}
            state, last = {"terminated": term}, {"terminated": term}
        # a container that stopped counts as a restart: under load it means the server crashed
        restarts = int(box.get("RestartCount", 0) or 0) + (0 if st.get("Running") else 1)
        uid = box.get("Id")
        if uid and st.get("StartedAt"):
            uid = f"{uid}@{st['StartedAt']}"
        return [
            {
                # the container id and its latest start are its identity: a same-named
                # replacement has another id, and a manual `docker restart` (RestartCount
                # unchanged, same id) another start
                "metadata": {"name": name, "uid": uid},
                "spec": {"nodeName": self.d.get("ssh") or "localhost"},
                "status": {
                    "podIP": name,
                    "containerStatuses": [
                        {"restartCount": restarts, "state": state, "lastState": last}
                    ],
                },
            }
        ]

    def logs(self, replica: str, previous: bool = False, tail: int | None = None) -> str:
        """The container's log, optionally only the last lines."""
        if previous:
            return ""  # a stopped container keeps its own log
        if tail is not None:
            proc = self.docker("logs", f"--tail={tail}", replica, timeout=120)
            return proc.stdout + proc.stderr
        # the full log is engagement evidence: read the inspected container by id, only since
        # its latest start, and only while that start holds (--since filters by time, so a
        # restart during the read could still let the earlier launch line through)
        for _ in range(3):
            box = self.inspect(replica) or {}
            started = started_of(box)
            since = [f"--since={started}"] if started and not started.startswith("0001-") else []
            cid = str(box.get("Id") or replica)
            proc = self.docker("logs", *since, cid, timeout=120)
            after = self.inspect(cid) or {}
            if started_of(after) != started:
                continue
            if proc.returncode != 0:
                # a failed read is no evidence: its error text must not pass as the NIM's log
                raise KubeError(f"could not read the log of {replica}: {proc.stderr.strip()[-300:]}")
            self._log_from[replica] = (cid, started)  # spec() reads this same container
            return proc.stdout + proc.stderr
        raise KubeError(f"{replica} kept restarting while its log was read")

    def spec(self, replica: str, allow: list[str] | None = None) -> Any:
        """The container's settings for the engagement check, without secret values.

        `docker inspect` resolves every variable, including those from env_from_host and
        env_file (whose names the runner never sees), so only an allow-list is kept: the names
        the campaign itself sets (`docker.env` and the candidate's env), never one that looks
        like a credential.

        After the full log was read, the spec is that log's container (by id), at the same
        start, and still the one the name resolves to: one replaced (even if renamed and
        still there) or restarted since is a KubeError, never mismatched evidence.
        """
        if replica in self._log_from:
            cid, started = self._log_from.pop(replica)  # once: a later deploy reads anew
            box = self.inspect(cid)
            # the name is resolved last, and must give the same id AT THE SAME START: a restart
            # between the two reads keeps the id but makes the settings read above stale
            now = self.inspect(replica) or {}  # a renamed one stays inspectable by its id

            def held(b: Any) -> bool:
                return isinstance(b, dict) and b.get("Id") == cid and started_of(b) == started

            if not held(box) or not held(now):
                raise KubeError(f"{replica} was replaced or restarted after its log was read")
        else:
            box = self.inspect(replica)
        if not box:
            return []
        declared = set(self.d.get("env", {})) | set(allow or [])  # the candidate's own names
        hidden = set(self.d.get("env_from_host", []))
        cfg = box.get("Config", {})
        env = []
        for item in cfg.get("Env") or []:
            name = item.partition("=")[0]
            if name in declared and name not in hidden and not SECRET_NAME.search(name):
                env.append(item)
        keep = {k: box.get(k) for k in ("Id", "Image", "Args", "RestartCount", "State")}
        keep["Config"] = {"Image": cfg.get("Image"), "Cmd": cfg.get("Cmd"), "Env": env}
        return [keep]

    def exec(
        self, replica: str, *cmd: str, timeout: float = 120
    ) -> subprocess.CompletedProcess[str]:
        """Run a command inside the container."""
        return self.docker("exec", replica, *cmd, timeout=timeout)

    def engagement_args(self, replica: str) -> list[str]:
        """The arguments that tell check_nim_engagement.py how to reach the container."""
        return ["docker", replica]

    def ident(self, name: str) -> str | None:
        """The container's Id ('' if none), None if it does not exist; raises if unreadable."""
        box = self.inspect(name)
        return None if box is None else str(box.get("Id") or "")

    def kept(self, name: str, uid: str) -> bool:
        """Whether the container recorded as `uid` still exists, under any name (a renamed
        one is still found by its Id)."""
        return bool(uid) and self.inspect(uid) is not None

    def strays(self, name: str) -> bool:
        """Never: a container is its own replica, so kept() answers for it."""
        return False

    def remove(self, name: str, uid: str | None = None) -> bool:
        """Remove one container if this campaign made it; return whether it did. With `uid`
        (the Id recorded when its removal was marked) only if `name` resolves to that Id."""
        if name == self.client:  # the client: remove_client
            return False
        self._log_from.pop(name, None)  # a saved log's binding never outlives its container
        box = self.inspect(name)
        if box is None or (box.get("Config", {}).get("Labels") or {}).get(LABEL) != self.c["name"]:
            return False
        if uid is not None and str(box.get("Id") or "") != uid:
            self.log(f"{name} is not the container recorded (another id); not removed")
            return False
        # by the inspected id: a same-named replacement made since is never removed
        gone = self.docker("rm", "-f", str(box.get("Id") or name), timeout=300)
        if gone.returncode != 0 and "no such container" not in gone.stderr.lower():
            raise KubeError(f"docker rm {name}: {gone.stderr.strip()[-300:]}")  # never taken as gone
        return True

    def gpus(self) -> str:
        """The GPU memory and use on the host, for the plan."""
        q = ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader"]
        proc = run_cmd(self.host_argv(q))
        return "; ".join(ln.strip() for ln in proc.stdout.splitlines()) or "unavailable"

    def created_resources(self, names: list[str]) -> list[str]:
        """What a live run creates, for the plan shown to the user."""
        return [*names, f"client container {self.client}", f"network {self.network}"]


Target = K8sTarget | DockerTarget


def make_target(c: dict[str, Any], campaign_dir: Path, log: Callable[[str], None]) -> Target:
    """The target for the campaign: Docker when target is docker, else Kubernetes."""
    if c.get("target") == "docker":
        return DockerTarget(c, campaign_dir, log)
    return K8sTarget(c, campaign_dir, log)
