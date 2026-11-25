#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""nim-smoke-check: check that a running NIM serves the features its docs promise.

  run       reach the NIM (--url, --docker NAME or --k8s KIND/NAME), identify the model, match
            its record in scripts/model_features.json, plan the checks and run them; write
            results.json, report.md and evidence/ in a private --out folder
  features  offline: the record that matches a model, image or family, how it matched, and
            the plan of every check with its reason and sources
  report    print the summary of a finished run again, from its results.json

Read-only: GET on the health, model, version and metadata routes and POST on
/v1/chat/completions of the URL the user names, nothing else; no redirect is followed. On
Kubernetes and Docker only read-only kubectl and docker commands run (references/access.md),
and the script owns its kubectl port-forward and stops it. A key is read only from the variable
named with --api-key-env (NVIDIA_API_KEY for https://integrate.api.nvidia.com; plain http to it
is refused) and is masked in everything printed or saved, before any text is cut. Over plain
http it goes only to this machine (loopback): to another host the run stops before any request
(key_over_plain_http), unless --allow-plain-http-key, which keeps a warning in the results.

Prints one JSON object {"ok", "state", "next", "facts", "errors"} on stdout; logs go to stderr.
Never prompts. Stdlib only.
Exit codes: 0 every check passed or is not applicable; 2 bad input; 3 precondition failed
(model_ambiguous, no_models, rbac_missing, ...); 4 a check failed (checks_failed), the
endpoint, kubectl or docker failed, or the run stopped on an unexpected error (internal_error);
5 refused; 130 interrupted (Ctrl-C). A run that stops early still writes the checks that ran,
except a stop while it finds the model to test (model_ambiguous, no_models, no answer from
/v1/models): that one writes nothing, so the rerun it asks for can use the same --out.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import re
import sys
import threading
import time
import traceback
import unicodedata
import urllib.parse
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NoReturn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nsc_access as nsa
import nsc_checks as nc
import nsc_common as nsc
import nsc_features as nf

RESULTS_SCHEMA = "nim-smoke-check/results/v1"
RESULTS_FILE = "results.json"
REPORT_FILE = "report.md"
HOSTED_KEY_ENV = "NVIDIA_API_KEY"  # read only for integrate.api.nvidia.com, when no name is given
MAX_RESULTS_BYTES = 20_000_000
MAX_TIMEOUT = 3600.0
EXIT_INTERRUPTED = 130  # 128 + SIGINT, as a shell reports a run stopped with Ctrl-C
# no answer to the first POST stops the run, unless kubectl exited during it: the next check then
# restarts the port-forward, and its POST counts as the first
UNREACHABLE = ("endpoint_unreachable", "timeout")
# health reasons that end a run with --wait-ready: the NIM still says it is not ready (or live)
NOT_READY = ("not_ready", "not_live")
# the tier B signals of the parser flags a deployment runs with (nsc_features.tier_b)
IN_EFFECT = ("effective", "tools_in_deployment", "reasoning_parser_in_deployment")
SLOW_DOC = (
    "https://docs.nvidia.com/nim/large-language-models/2.1.5/troubleshooting/requests.html"
    "#requests-exceed-the-client-timeout"
)
# reasons of a check whose requests a lost port-forward can explain: kubectl is then waited for
FORWARD_LOSS = ("endpoint_unreachable", "connection_lost", "timeout", "health_timeout")
FORWARD_LOST = (
    "The port-forward stopped during this check (kubectl exited), so its requests got no "
    "answer through it."
)
STATE_BY_EXIT = {
    nsc.EXIT_INPUT: "input",
    nsc.EXIT_PRECONDITION: "precondition",
    nsc.EXIT_EXTERNAL: "unreachable",
    nsc.EXIT_REFUSED: "refused",
}
NEXT_BY_EXIT = {
    nsc.EXIT_INPUT: "ask_user",
    nsc.EXIT_PRECONDITION: "ask_user",
    nsc.EXIT_EXTERNAL: "fix_errors",
    nsc.EXIT_REFUSED: "fix_errors",
}
DEPLOY_WHY = (
    "A check failed because of a server setting; the documented switch is in each hint. "
    "Changing the deployment is nim-deploy's job."
)
# The fix of key_over_plain_http: the safe ways first, the flag only once the user agrees. The
# flag ends the text: the masker reads the word after a flag whose name holds "key" as a key.
PLAIN_HTTP_FIX = (
    "Tell the user the key would cross the network unencrypted, and offer the safer ways: an "
    "https:// URL for this endpoint; a run on the machine that serves it, with a loopback URL "
    "(Docker: on the Docker host; Kubernetes: --k8s, whose port-forward is loopback); or no "
    "--api-key-env when nothing in front of the NIM needs a key (a NIM checks none). Only when "
    "the user agrees to send it over plain http (a private gateway they trust), add "
    "--allow-plain-http-key"
)


# --- small helpers ------------------------------------------------------------------------------


def fail(exc: nsc.ScriptError, facts: Mapping[str, Any] | None = None) -> NoReturn:
    """Emit a ScriptError as the one JSON object, with its exit code."""
    code = exc.exit_code
    merged = {**dict(facts or {}), **exc.facts}
    nsc.emit(False, STATE_BY_EXIT.get(code, "input"), NEXT_BY_EXIT.get(code, "ask_user"),
             merged, [exc.as_error()], code)


def bad_input(message: str, fix: str) -> nsc.ScriptError:
    return nsc.ScriptError("bad_input", message, fix, nsc.EXIT_INPUT)


def split_list(values: list[str] | None) -> list[str] | None:
    """--checks a,b --checks c -> [a, b, c]; None when the flag was not given."""
    if values is None:
        return None
    out = [item.strip() for value in values for item in value.split(",") if item.strip()]
    if not out:
        raise bad_input("an empty list of checks", f"Use check ids from: {', '.join(nf.CHECKS)}")
    return out


def seconds(value: float, flag: str, low: float, high: float = MAX_TIMEOUT) -> float:
    if not (math.isfinite(value) and low <= value <= high):
        raise bad_input(f"{flag} {value:g} is out of range",
                        f"Pass {flag} a number of seconds from {low:g} to {high:g}")
    return float(value)


def utc_text(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def counts_of(results: list[Mapping[str, Any]]) -> dict[str, int]:
    counts = {status: 0 for status in nc.STATUSES}
    for result in results:
        counts[str(result.get("status"))] = counts.get(str(result.get("status")), 0) + 1
    counts["total"] = len(results)
    return counts


def summary_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """The stdout form of one result: never the prompt or the answer."""
    keys = ("id", "status", "reason", "message", "evidence", "sources", "hint")
    return {key: result.get(key) for key in keys}


def handoff_of(results: list[Mapping[str, Any]]) -> dict[str, Any] | None:
    """nim-deploy, with the documented switches, when a check failed on a server setting."""
    changes = [
        {"check": r.get("id"), "reason": r.get("reason"), "hint": r.get("hint")}
        for r in results if r.get("status") == nc.FAIL and r.get("hint")
    ]
    if not changes:
        return None
    return {"skill": "nim-deploy", "why": DEPLOY_WHY, "changes": changes}


def nearest_private_folder(folder: Path) -> Path:
    """The nearest existing folder of `folder` (itself when it exists), checked to be a folder
    (bad_input) that no other user can change (out_not_private); nothing is created."""
    probe = folder
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    if not probe.is_dir():
        raise bad_input(f"{probe} is not a folder", "Pass --out a folder path")
    nsc.check_private(probe)
    return probe


def require_empty(folder: Path) -> None:
    """bad_input unless the existing folder `folder` is empty: old files never mix with a run."""
    try:
        empty = not any(folder.iterdir())
    except OSError as exc:
        raise bad_input(f"cannot read the folder {folder}: {exc.strerror}",
                        "Pass --out a folder you can read and write") from exc
    if not empty:
        raise bad_input(
            f"{folder} is not empty",
            "Pass --out a new or empty folder, or leave it out for a new folder under "
            f"{nsc.CACHE_FOLDER}/",
        )


def check_out(path: Path, create: bool) -> Path:
    """The --out folder: private and new or empty. Without `create` nothing is made: the
    nearest existing folder must be a private folder, and `path`, when it exists, an empty one,
    so a bad --out is refused before any request. With `create` it is made (0700,
    nsc.ensure_private_dir) and checked again, in case it changed while the checks ran."""
    folder = Path(path).expanduser().absolute()
    if not create:
        if nearest_private_folder(folder) == folder:
            require_empty(folder)
        return folder
    folder = nsc.ensure_private_dir(folder)
    require_empty(folder)
    return folder


def check_default_out() -> None:
    """Before any request, for a run with no --out: the default folder is named after the model,
    known only later, so its parent (~/.cache/nim-smoke-check, or the nearest existing folder
    above it) is checked now (out_not_private, bad_input)."""
    try:
        cache = Path(nsc.CACHE_FOLDER).expanduser()
    except RuntimeError as exc:  # no home folder to expand ~ with
        raise bad_input("the home folder is not known, so the default --out folder has no place",
                        "Pass --out a new or empty private folder") from exc
    nearest_private_folder(cache.absolute())


def claim_default_out(model: str | None, now: datetime) -> Path:
    """Make and return the default --out folder: nsc.default_out_dir, or that name with -2, -3,
    ... when another run of the same model that started in the same second took it first (for
    example several candidates checked side by side). One mkdir claims a name, so two runs never
    share a folder."""
    base = nsc.default_out_dir(model, now)
    parent = nsc.ensure_private_dir(base.parent)
    for number in range(1, 100):
        folder = parent / (base.name if number == 1 else f"{base.name}-{number}")
        try:
            folder.mkdir(mode=0o700)
        except FileExistsError:
            continue
        except OSError as exc:
            raise bad_input(f"cannot create the folder {folder}: {exc.strerror}",
                            "Pass --out a folder you can write") from exc
        return folder
    raise bad_input(f"{base} and 98 numbered folders next to it exist",
                    "Pass --out a new or empty folder")


class BufferedEvidence(nc.Evidence):
    """Evidence kept in memory until the run knows its --out folder (the default folder name
    holds the model id), then written with flush(); a dry run writes nothing."""

    def __init__(self) -> None:
        super().__init__(Path("."))
        self.pending: list[tuple[str, dict[str, Any]]] = []
        self.lock = threading.Lock()

    def save(
        self,
        check: str,
        attempt: int,
        request: Mapping[str, Any] | None,
        http_result: nsc.HttpResult | None,
        parsed: Mapping[str, Any] | None = None,
    ) -> str:
        part = re.sub(r"[^a-z0-9_]+", "_", str(check).lower()).strip("_") or "check"
        name = f"{part}-{int(attempt)}.json"
        record = nc.evidence_record(check, attempt, request, http_result, parsed)
        with self.lock:
            self.pending = [item for item in self.pending if item[0] != name]
            self.pending.append((name, record))
        return f"evidence/{name}"

    def flush(self, out: Path) -> None:
        """Write the pending evidence in OUT/evidence/. A record leaves `pending` only once its
        file is written, so after a failed write the next flush (abort's) writes the rest."""
        with self.lock:
            pending = list(self.pending)
        for item in pending:
            name, record = item
            nsc.write_json_file(out / "evidence" / name, record)
            with self.lock:
                self.pending = [other for other in self.pending if other is not item]


# --- run: inputs --------------------------------------------------------------------------------


def validate(args: argparse.Namespace) -> dict[str, Any]:
    """Every input check that needs no network, in this order: the key variable's name, the
    target, the URLs, TLS files, the numbers, the check ids, the documented sources and switches,
    the family, --out."""
    if args.api_key_env is not None:
        nsc.check_env_name(args.api_key_env)
    if args.k8s and not (args.context and args.namespace):
        raise bad_input("--k8s needs --context and --namespace",
                        "Pass --context CTX --namespace NS with --k8s KIND/NAME")
    if not args.k8s and (args.context or args.namespace):
        raise bad_input("--context and --namespace go with --k8s",
                        "Pass --k8s KIND/NAME, or leave out --context and --namespace")
    hosted = False
    if args.url is not None:  # bad_input for a bad URL, or for http to the hosted API
        hosted = nsa.from_url(args.url, args.health_url).place == "hosted"
    elif args.health_url is not None:
        nsa.from_url(args.health_url)
    tls = nsc.Tls(args.ca_file, args.client_cert, args.client_key)
    tls.context()  # a file that cannot be loaded is bad_input now, not a failed request later
    if args.concurrency is not None:
        limit = nc.HOSTED_CONCURRENCY if hosted else nc.MAX_CONCURRENCY
        if args.concurrency > limit:  # the hosted API's rate limits: it stays at 4 there
            raise nc.concurrency_too_high(args.concurrency, hosted)
        if args.concurrency < 1:
            raise bad_input(f"--concurrency {args.concurrency}: send at least 1 request",
                            f"Use --concurrency 1 to {limit}")
    # --timeout given: every chat request gets that budget, up or down from the formula
    timeout = None if args.timeout is None else seconds(args.timeout, "--timeout", 1.0)
    wait = seconds(args.wait_ready, "--wait-ready", 0.0)
    checks = nf.select_checks(split_list(args.checks), split_list(args.skip), args.extended)
    if wait > 0 and "health" not in checks:
        raise bad_input("--wait-ready polls the health check, which --checks or --skip leave out",
                        "Keep health in the checks, or leave out --wait-ready")
    documented: dict[str, str] = {}
    for text in args.documented or []:
        feature, url = nf.parse_documented(text)
        documented[feature] = url
    switches = {}
    if args.reasoning_on is not None:
        switches["on"] = nf.parse_switch(args.reasoning_on)
    if args.reasoning_off is not None:
        switches["off"] = nf.parse_switch(args.reasoning_off)
    table = nf.load_table()
    if args.family is not None:
        nf.match(table, family=args.family)  # unknown_family (exit 2) before any request
    if args.image is not None and not nsa.IMAGE_REF.fullmatch(args.image):
        raise bad_input("--image is not an image reference",
                        "Pass the image as nvcr.io/nim/TEAM/REPO:TAG")
    out = None
    if args.out:  # a private folder that is new or empty, or no request is sent
        out = check_out(Path(args.out), create=False)
    else:
        check_default_out()
    return {"tls": tls, "timeout": timeout, "wait": wait, "checks": checks,
            "documented": documented, "switches": switches, "table": table, "out": out,
            "hosted": hosted}


def open_access(args: argparse.Namespace) -> tuple[nsa.Access, nsa.PortForward | None]:
    """The Access of the target; on Kubernetes the running port-forward too (the caller stops
    it), to a pod that --wait-ready waits up to its seconds for, when none is Ready yet."""
    if args.url is not None:
        return nsa.from_url(args.url, args.health_url), None
    if args.docker is not None:
        access = nsa.from_docker(args.docker, keyed=args.api_key_env is not None)
    else:
        access, forward = nsa.from_k8s(args.k8s, args.context, args.namespace,
                                       wait=float(args.wait_ready or 0.0))
        if args.health_url is not None:
            access.health_url = nsc.normalize_base(args.health_url)
        return access, forward
    if args.health_url is not None:
        access.health_url = nsc.normalize_base(args.health_url)
    return access, None


def read_key(
    args: argparse.Namespace, hosted: bool, checks: tuple[str, ...] | list[str] = ()
) -> tuple[str | None, str]:
    """(key, how it is shown in facts.access.auth). A named variable that is unset is bad input,
    and its name is not repeated (a key given in its place may pass the name check);
    NVIDIA_API_KEY is read only for the hosted API, which is https only, when no name is
    given. The hosted API answers every chat request without a key with 401, so a hosted run
    of a chat check (`checks`, not a dry run) with no key is bad input before any request."""
    keyless = "an endpoint without a key" if not hosted else None
    if args.api_key_env is not None:
        key = nsc.read_key(args.api_key_env)
        if key is None:
            raise bad_input(
                "the environment variable that --api-key-env names is not set or is empty "
                "(the name is not repeated here, in case it is a key)",
                "Ask the user to export the key in the variable passed to --api-key-env, in "
                "their own shell (never pass or paste the key itself)"
                + (f", or leave out --api-key-env for {keyless}" if keyless else ""),
            )
        return key, f"bearer from ${args.api_key_env}"
    if hosted:
        key = nsc.read_key(HOSTED_KEY_ENV)
        if key is not None:
            return key, f"bearer from ${HOSTED_KEY_ENV}"
        if not args.dry_run and any(check not in ("health", "models") for check in checks):
            raise bad_input(
                f"the hosted API needs a key for its chat requests, and {HOSTED_KEY_ENV} is "
                "not set or is empty",
                f"Ask the user to export the key in {HOSTED_KEY_ENV} (or in another variable "
                "passed with --api-key-env NAME) in their own shell; never pass or paste the key "
                "itself. A --dry-run needs no key",
            )
    return None, "none"


def key_in_clear(url: str) -> bool:
    """Whether a key sent to `url` crosses the network unencrypted: plain http to a host that is
    not this machine. https encrypts it. This machine (nsc.is_loopback) is the name localhost,
    127.0.0.0/8 and ::1, also in the forms the connection reads as one (127.1,
    ::ffff:127.0.0.1), so the script's kubectl port-forward too, and 0.0.0.0 and ::, which
    connect to it. A *.localhost name and a host with a final dot (127.0.0.1., localhost.) are
    other hosts here, though they skip the proxy: the resolver may look them up in DNS."""
    host = nsc.connect_host(url)
    local = nsc.is_loopback(url) and not host.endswith((".", ".localhost"))
    return urllib.parse.urlsplit(url).scheme.lower() != "https" and not local


# --- run: identify the target -------------------------------------------------------------------


def probe_get(ep: nc.Endpoint, ev: nc.Evidence, path: str, label: str) -> Any:
    """One GET that is recorded, never judged; the JSON body of a 200, else None."""
    url = ep.url(path)
    headers = {**ep.headers_for(url), "X-Request-Id": f"nsc-{ep.run_id}-probe_{label}-1"}
    res = nsc.request_json("GET", url, headers=headers, timeout=nc.request_timeout(ep),
                           tls=ep.tls, loopback=ep.loopback_for(url))
    sent = {"User-Agent": nsc.USER_AGENT, "Accept": "application/json", **headers}
    ev.save(f"probe_{label}", 1, {"method": "GET", "url": url, "headers": sent, "body": None},
            res, {"role": label})
    data = res.json_body() if res.status == 200 else None
    return data if isinstance(data, dict) else None


def text_of(value: Any, limit: int = 200) -> str | None:
    """A server's text on one line, masked, then cut to `limit` characters; None when empty."""
    return nsc.clip(value, limit) if isinstance(value, str) and value.strip() else None


def profile_of(metadata: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not metadata:
        return None
    profile = {
        "id": text_of(metadata.get("selectedModelProfileId") or metadata.get("profile_id")),
        "name": text_of(metadata.get("profile_name") or metadata.get("profileName")),
    }
    return profile if any(profile.values()) else None


def backend_of(card: Mapping[str, Any] | None, profile: Mapping[str, Any] | None) -> str | None:
    """vllm or sglang: from owned_by in /v1/models, else the profile name prefix."""
    owner = str((card or {}).get("owned_by") or "").lower()
    if owner in ("vllm", "sglang"):
        return owner
    name = str((profile or {}).get("name") or "").lower()
    for backend in ("vllm", "sglang"):
        if name.startswith(backend + "-"):
            return backend
    return None


def version_of(image: str | None, version: Mapping[str, Any] | None) -> str | None:
    """The NIM version: the image tag when it is a version, else /v1/version's release."""
    tag = nf.image_tag(image)
    if tag and nf.generation_of(tag):
        return tag
    if version:
        return text_of(version.get("release") or version.get("version"), 40)
    return tag


def tier_b_signals(access: nsa.Access, warnings: list[str]) -> dict[str, Any]:
    """The image default against the deployment's effective NIM_PASSTHROUGH_ARGS (tier B). The
    default is read from the image the container or pod runs, named by its ID or digest
    (nsa.running_image), never by its tag: a pull after the start can move a tag to an image
    with another default. No such name: the default is unknown."""
    if not access.config_known:
        return nf.tier_b(None, None, None)
    running = nsa.running_image(access)
    default = nsa.image_default_passthrough(running) if running else None
    effective = access.effective_passthrough
    if effective == nsa.UNKNOWN:
        warnings.append(
            "NIM_PASSTHROUGH_ARGS comes from a ConfigMap or a Secret (valueFrom), or may come from "
            "one (envFrom), which is not read; whether it replaced the image default is unknown."
        )
        return nf.tier_b(None, None, access.effective_args)
    if access.place == "docker" and effective is not None:
        # docker merges the image environment into the container's: its value is the image
        # default unless -e replaced it, and the NIM runs with it either way.
        if default is None:
            # Unknown default: the parser flags the value turns on still count, but nothing is
            # said about the image default, or whether -e set or replaced the value.
            seen = nf.tier_b(None, effective, access.effective_args)
            signals = nf.tier_b(None, None, access.effective_args)
            signals.update({key: seen[key] for key in IN_EFFECT})
            return signals
        if effective == default:
            effective = None
    return nf.tier_b(default, effective, access.effective_args)


# --- run: report --------------------------------------------------------------------------------


# Text in report.md that comes from the endpoint or the command line (a model id, a message, a
# warning) is put on one line and escaped, so it cannot add a line, a heading, a table cell, a
# code span, a link or HTML to the report that an agent reads.
MD_SPECIAL = re.compile(r"([\\`|\[\]<>])")
# What starts a heading, a quote, a list, a fence or a rule at the start of a line
MD_LINE_START = re.compile(
    r"^(?:#{1,6}(?=[ \t]|$)|>|[-+*](?=[ \t]|$)|\d{1,9}(?=[.)](?:[ \t]|$))|~~~|"
    r"(?:[-*_][ \t]*){3,}$)"
)
MD_CODE = re.compile(r"[A-Za-z0-9_.:/-]+")  # shown as `code` as it is
MD_EVIDENCE = re.compile(r"evidence/[A-Za-z0-9_.-]+\.json")  # linked as it is


def md_text(value: object) -> str:
    """`value` as one line of Markdown text: control, format and line-separator characters
    become spaces, runs of white space one space; then it is masked (nsc.SCRUBBER) and \\ ` | [ ]
    < > are escaped, as is a start that could begin a heading, a quote, a list or a rule."""
    text = "".join(
        " " if unicodedata.category(char) in ("Cc", "Cf", "Zl", "Zp") else char
        for char in str(value)
    )
    text = MD_SPECIAL.sub(r"\\\1", nsc.SCRUBBER.scrub(" ".join(text.split())))
    return MD_LINE_START.sub(
        lambda found: found.group(0) + "\\" if found.group(0)[0].isdigit()
        else "\\" + found.group(0),
        text,
    )


def md_code(value: object) -> str:
    """`value` as a code span when it is a plain token (a check id, a reason code), else as
    escaped text (md_text)."""
    text = str(value)
    return f"`{text}`" if MD_CODE.fullmatch(text) else md_text(text)


def md_link(path: object) -> str:
    """A link to an evidence file, or the path as escaped text when it is not one."""
    text = str(path)
    return f"[{text}]({text})" if MD_EVIDENCE.fullmatch(text) else md_text(text)


def report_markdown(doc: Mapping[str, Any]) -> str:
    """report.md: the run line, the results table, why checks are not applicable, what the docs
    say to change, sources and warnings. No prompt or answer text; every value is escaped
    (md_text, md_code, md_link)."""
    target = doc.get("target") or {}
    access = doc.get("access") or {}
    results = doc.get("results") or []
    lines = [
        f"# nim-smoke-check: {md_text(doc.get('model') or 'unknown model')} at "
        f"{md_text(doc.get('url'))}",
        "",
        " · ".join([
            f"Run {md_text(doc.get('created'))}",
            f"NIM {md_text(target.get('version') or 'unknown')}",
            f"backend {md_text(target.get('backend') or 'unknown')}",
            f"access {md_text(access.get('place'))} ({md_text(access.get('how'))})",
            f"record {md_text(target.get('record') or 'none')} ({md_text(target.get('how'))})",
            f"table reviewed {md_text(target.get('reviewed'))}",
        ]),
        "",
        f"Result: **{md_text(doc.get('state'))}**. " + ", ".join(
            f"{md_text(count)} {md_text(status)}"
            for status, count in (doc.get("counts") or {}).items() if status != "total"
        ) + ".",
        "",
        "| check | result | reason | evidence |",
        "|---|---|---|---|",
    ]
    for result in results:
        evidence = ", ".join(md_link(path) for path in result.get("evidence") or []) or "-"
        lines.append(f"| {md_code(result.get('id'))} | {md_text(result.get('status'))} | "
                     f"{md_code(result.get('reason'))} | {evidence} |")
    skipped = [r for r in results if r.get("status") == nc.NOT_APPLICABLE]
    if skipped:
        lines += ["", "## Not applicable: why", ""]
        for result in skipped:
            reason = str(result.get("reason"))
            # the result's own message: one reason code can stand for several cases (opt-in, or
            # a default the docs do not state), and the message says which one this is
            why = (str(result.get("message") or "") or nc.NA_MESSAGES.get(reason)
                   or nc.REASONS.get(reason) or reason)
            why = why if why.endswith(".") else why + "."
            sources = ", ".join(md_text(url) for url in result.get("sources") or [])
            line = (f"- {md_code(result.get('id'))} ({md_code(reason)}): {md_text(why)} "
                    f"Sources: {sources or 'no source'}")
            if result.get("hint"):
                line += f". Hint: {md_text(result.get('hint'))}"
            lines.append(line)
    failed = [r for r in results if r.get("status") == nc.FAIL]
    if failed:
        lines += ["", "## Failed: what the docs say to change", ""]
        for result in failed:
            reason = str(result.get("reason"))
            text = result.get("hint") or nc.REASONS.get(reason, reason)
            lines.append(f"- {md_code(result.get('id'))} ({md_code(reason)}): {md_text(text)}")
            issues = result.get("known_issues") or []
            if nc.points_to_known_issues(nc.FAIL, reason, issues):  # not for a 401 or a timeout
                lines += [f"  - Known issue: {md_text(url)}" for url in issues
                          if isinstance(url, str)]
        if doc.get("handoff"):
            lines += ["", "Hand the server changes above to nim-deploy; this skill changes "
                      "nothing."]
    sources = list(dict.fromkeys(
        url for result in results for url in result.get("sources") or []
    ))
    if sources:
        lines += ["", "## Sources", ""] + [f"- {md_text(url)}" for url in sources]
    warnings = [md_text(text) for text in doc.get("warnings") or []] + [
        f"{md_code(r.get('id'))}: {md_text(text)}" for r in results
        for text in r.get("warnings") or []
    ]
    if warnings:
        lines += ["", "## Warnings", ""] + [f"- {text}" for text in dict.fromkeys(warnings)]
    return "\n".join(lines) + "\n"


def warnings_of(doc: Mapping[str, Any]) -> list[str]:
    """The run's warnings, then each check's as "<check>: <text>", then the known issues of each
    check that failed on how the model answered (its message points here), once each."""
    results = doc.get("results") or []
    found = [str(text) for text in doc.get("warnings") or []]
    found += [f"{r.get('id')}: {text}" for r in results for text in r.get("warnings") or []]
    found += [
        f"{r.get('id')}: a known issue of this model may explain the fail: {url}"
        for r in results
        if nc.points_to_known_issues(str(r.get("status")), str(r.get("reason")),
                                     r.get("known_issues") or [])
        for url in r.get("known_issues") or [] if isinstance(url, str)
    ]
    return list(dict.fromkeys(found))


def facts_of(doc: Mapping[str, Any]) -> dict[str, Any]:
    """The facts of a finished run, for run and report alike. facts.warnings holds the run's
    warnings and every check's (stdout results keep only the summary keys)."""
    results = doc.get("results") or []
    return {
        "url": doc.get("url"),
        "model": doc.get("model"),
        "out": doc.get("out"),
        "report": doc.get("report"),
        "results": [summary_result(result) for result in results],
        "counts": doc.get("counts") or counts_of(results),
        "target": doc.get("target"),
        "access": doc.get("access"),
        "warnings": warnings_of(doc),
        "handoff": doc.get("handoff"),
    }


def plan_view(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    keys = ("check", "decision", "reason", "sources", "conflicts", "hint", "known_issues",
            "request", "warnings")
    return [{key: entry.get(key) for key in keys} for entry in plan.get("checks", [])]


# --- run ----------------------------------------------------------------------------------------


class Run:
    """One `run`: the state the steps share."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.now = datetime.now(timezone.utc)
        self.run_id = self.now.strftime("%Y%m%dT%H%M%SZ")
        self.inputs: dict[str, Any] = {}
        self.access: nsa.Access | None = None
        self.forward: nsa.PortForward | None = None
        self.ep: nc.Endpoint | None = None
        self.ev = BufferedEvidence()
        self.auth = "none"
        self.model: str | None = None
        self.target: dict[str, Any] = {}
        self.warnings: list[str] = []
        self.results: list[dict[str, Any]] = []
        self.out: Path | None = None
        self.plan: Mapping[str, Any] | None = None
        self.ready_by: float | None = None  # monotonic end of the --wait-ready budget

    # the facts shown when the run stops early
    def partial_facts(self) -> dict[str, Any]:
        facts: dict[str, Any] = {}
        if self.access is not None:
            facts["url"] = self.access.url
            facts["access"] = self.access_facts()
        if self.model:
            facts["model"] = self.model
        if self.target:
            facts["target"] = self.target
        if self.results:
            facts["results"] = [summary_result(r) for r in self.results]
        # the run's warnings so far, such as a key sent over plain http (check_cleartext)
        warnings = warnings_of({"warnings": self.warnings, "results": self.results})
        if warnings:
            facts["warnings"] = warnings
        return facts

    def access_facts(self) -> dict[str, Any]:
        assert self.access is not None
        if self.forward is not None:
            self.access.port_forward = self.forward.facts()
        facts = self.access.facts()
        facts["auth"] = self.auth
        return facts

    def endpoint(self, key: str | None) -> nc.Endpoint:
        access = self.access
        assert access is not None
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        health = access.health_url if access.health_url != access.url else None
        return nc.Endpoint(
            base=access.url, health_base=health, headers=headers, tls=self.inputs["tls"],
            loopback=nsc.is_loopback(access.url), timeout=self.inputs["timeout"],
            hosted=access.place == "hosted", run_id=self.run_id,
            port_forward=self.forward is not None,
        )

    def follow_forward(self) -> bool:
        """Before each check: a stopped port-forward is restarted once (port_forward_lost the
        second time), and the endpoint follows its local port. True when this call restarted
        it: the next request through it then counts as the first."""
        if self.forward is None or self.ep is None or self.access is None:
            return False
        before = getattr(self.forward, "restarts", 0)
        self.forward.ensure()
        self.ep.base = self.access.url
        health = self.access.health_url
        self.ep.health_base = health if health and health != self.access.url else None
        return getattr(self.forward, "restarts", 0) != before

    def settle_forward(self, result: dict[str, Any] | None) -> bool:
        """After a check whose requests got no answer through the port-forward: kubectl closes
        the connection first and exits a moment later, so wait for it (PortForward.settle);
        the next check's follow_forward then restarts it instead of using its dead port. True
        when kubectl exited: the forward was lost during this check."""
        if self.forward is None or result is None:
            return False
        if result.get("status") != nc.FAIL or result.get("reason") not in FORWARD_LOSS:
            return False
        if not self.forward.settle():
            return False
        result.setdefault("warnings", []).append(FORWARD_LOST)
        return True

    def execute(self) -> NoReturn:
        args = self.args
        self.inputs = validate(args)
        key, self.auth = read_key(args, self.inputs["hosted"], self.inputs["checks"])
        try:
            # --wait-ready is one budget: a Kubernetes run first waits for a Ready pod, and the
            # health check gets what is left
            self.ready_by = time.monotonic() + self.inputs["wait"]
            self.access, self.forward = open_access(args)
            self.ep = self.endpoint(key)
            if key:
                self.check_cleartext()
            self.probe_and_plan()
        finally:
            if self.forward is not None:
                self.forward.stop()
        raise AssertionError("unreachable")  # probe_and_plan always emits

    def check_cleartext(self) -> None:
        """Before any request, when a key is set: a URL that would send it over plain http to a
        host that is not this machine (key_in_clear) is refused, key_over_plain_http (exit 2),
        unless --allow-plain-http-key; then the run keeps a warning, in facts.warnings however
        it ends (partial_facts too) and in results.json and report.md when it writes them.
        The key goes only to the scheme, host and port of the URL (nc.Endpoint.headers_for), so
        a health URL elsewhere gets no key and is not checked."""
        assert self.access is not None
        url = self.access.url
        if not key_in_clear(url):
            return
        host = urllib.parse.urlsplit(url).hostname or "?"
        if not self.args.allow_plain_http_key:
            raise nsc.ScriptError(
                "key_over_plain_http",
                f"the key would go to {host} over plain http, so anyone on the network path "
                "could read it; the run stopped before any request",
                PLAIN_HTTP_FIX,
                nsc.EXIT_INPUT,
            )
        text = (f"The key goes to {host} over plain http (--allow-plain-http-key), so anyone on "
                "the network path can read it; use an https URL.")
        self.warnings.append(text)
        nsc.log(f"nim-smoke-check: warning: {text}")

    def health_result(self, selected: tuple[str, ...]) -> dict[str, Any] | None:
        if "health" not in selected:
            return None
        assert self.access is not None and self.ep is not None
        wait = self.inputs["wait"]
        if wait > 0 and self.ready_by is not None:  # what the wait for a Ready pod left
            wait = max(self.ready_by - time.monotonic(), 0.0)
        if self.access.health_url is None:  # NIM_HEALTH_PORT is not published or forwarded
            note = next((text for text in self.access.notes if "NIM_HEALTH_PORT" in text),
                        "the health port cannot be reached from here")
            return nc.check_health(self.ep, self.ev, None, wait=wait,
                                   unreachable=note).as_dict()
        self.follow_forward()
        return nc.check_health(self.ep, self.ev, None, wait=wait).as_dict()

    def probe_and_plan(self) -> NoReturn:
        args, ep, access = self.args, self.ep, self.access
        assert ep is not None and access is not None
        selected = self.inputs["checks"]
        nsc.log(f"nim-smoke-check: {access.place} {access.url} ({access.how})")
        health = self.health_result(selected)
        self.settle_forward(health)
        if (self.inputs["wait"] > 0 and not args.dry_run and health is not None
                and health.get("status") == nc.FAIL and health.get("reason") in NOT_READY):
            self.stop_not_ready(health)
        self.follow_forward()
        models, chosen, cards = nc.check_models(ep, args.model, self.ev)
        self.model = chosen
        self.warnings += models.warnings
        card = next((c for c in cards if c.get("id") == chosen), None)
        version = metadata = None
        if not ep.hosted:
            version = probe_get(ep, self.ev, "/v1/version", "version")
            metadata = probe_get(ep, self.ev, "/v1/metadata", "metadata")
        image = args.image or access.image
        profile = profile_of(metadata)
        backend = backend_of(card, profile)
        ep.backend = backend
        nim_version = version_of(image, version)
        table = self.inputs["table"]
        root = (card or {}).get("root") if card else None
        root_id = root if isinstance(root, str) else None
        record, how = nf.match(table, image=image, served_id=chosen, root=root_id,
                               family=args.family)
        if args.family is not None:  # another family's verdicts on an unrelated model
            mismatch = nf.family_warning(table, args.family, image=image, served_id=chosen,
                                         root=root_id)
            if mismatch:
                self.warnings.append(mismatch)
                nsc.log(f"nim-smoke-check: warning: {mismatch}")
        signals = tier_b_signals(access, self.warnings)
        plan = nf.decide(
            record, checks=selected, hosted=ep.hosted, documented=self.inputs["documented"],
            switches=self.inputs["switches"], generation=nim_version, backend=backend,
            profile=profile, signals=signals, table=table, image=image,
        )
        self.plan = plan
        self.warnings += plan["warnings"]
        self.target = {
            "image": image,
            "resource_id": nf.resource_id(image),
            "version": nim_version,
            "generation": plan["generation"],
            "backend": plan["backend"] or backend,
            "profile": profile,
            "served_id": chosen,
            "root": root if isinstance(root, str) else None,
            "record": plan["record"],
            "name": plan["name"],
            "status": plan["status"],
            "how": how,
            "reviewed": plan["reviewed"],
            "tier_b": {key: signals.get(key) for key in (
                "image_default_known", "set_in_deployment", "overridden",
                "tools_on_by_default", "tools_dropped_by_override", "tools_in_deployment",
            )},
        }
        if args.dry_run:
            self.emit_plan(plan)
        if chosen is None:
            self.results = [r for r in (health, models.as_dict()) if r is not None]
            self.warnings.append(f"No POST check ran: {models.message}")
            self.finish(stopped=None)
        self.run_checks(plan, health, models.as_dict())

    def stop_not_ready(self, health: dict[str, Any]) -> NoReturn:
        """--wait-ready ended and the NIM still answers that it is not ready (or not live): no
        other check runs, as each would fail on that, not on a feature. The run ends as
        checks_failed with health, its evidence and the checks not run."""
        self.results = [health]
        not_run = self.not_run()
        self.warnings.append(
            f"--wait-ready {self.inputs['wait']:g} ended and health still fails "
            f"({health.get('reason')}), so these checks did not run: "
            f"{', '.join(not_run) or 'none'}."
        )
        self.finish(None, not_run=not_run)

    def emit_plan(self, plan: Mapping[str, Any]) -> NoReturn:
        assert self.access is not None
        out = self.inputs["out"] or nsc.default_out_dir(self.model, self.now)
        facts = {
            "url": self.access.url,
            "model": self.model,
            "out": str(out),
            "dry_run": True,
            "plan": plan_view(plan),
            "target": self.target,
            "access": self.access_facts(),
            "warnings": list(dict.fromkeys(self.warnings)),
        }
        nsc.emit(True, "planned", "run", facts)

    def run_checks(
        self, plan: Mapping[str, Any], health: dict[str, Any] | None, models: dict[str, Any]
    ) -> NoReturn:
        """Run the planned checks in order. A port-forward that is gone and cannot be restarted
        (port_forward_lost, pod_not_ready, ...) stops the requests, not the run: the checks that
        still need one are listed as not run, and finish() writes what ran, with its evidence,
        before it reports that error."""
        assert self.ep is not None and self.model is not None
        posted = False
        stopped: str | None = None
        halted: nsc.ScriptError | None = None
        not_run: list[str] = []
        for entry in plan["checks"]:
            check = entry["check"]
            if check == "health":
                if health is not None:
                    self.results.append(health)
                continue
            if check == "models":
                self.results.append(models)
                continue
            options = {}
            if check == "concurrency" and self.args.concurrency is not None:
                options["n"] = self.args.concurrency
            testing = entry.get("decision") == "test"
            if testing and halted is None:
                try:
                    if self.follow_forward():
                        posted = False  # a restarted forward: its first POST counts as the first
                except nsc.ScriptError as exc:
                    halted = exc
            if testing and halted is not None:
                not_run.append(check)
                continue
            result = nc.run_check(check, self.ep, self.ev, self.model, entry, **options)
            nsc.log(f"nim-smoke-check: {check}: {result.status} ({result.reason})")
            self.results.append(result.as_dict())
            if not testing or result.attempts == 0:
                continue
            if self.settle_forward(self.results[-1]):
                # kubectl exited: the forward was lost, not the NIM's answer. The next check
                # restarts it, and its POST counts as the first
                posted = False
                continue
            if not posted and result.status == nc.FAIL and result.reason in UNREACHABLE:
                stopped = result.reason
                break
            posted = True
        self.finish(stopped, halted, not_run)

    def write_run(self, state: str, not_run: list[str] | None = None) -> dict[str, Any]:
        """Write the evidence, results.json and report.md of the run in its --out folder (made
        and checked the first time; a second write replaces the run's own files) and return
        the results document."""
        assert self.access is not None
        if self.out is None:
            out = self.inputs["out"] or claim_default_out(self.model, self.now)
            self.out = check_out(out, create=True)
        self.ev.flush(self.out)
        results = self.results
        doc = {
            "schema": RESULTS_SCHEMA,
            "state": state,
            "created": utc_text(self.now),
            "run_id": self.run_id,
            "url": self.access.url,
            "model": self.model,
            "out": str(self.out),
            "report": str(self.out / REPORT_FILE),
            "target": self.target,
            "access": self.access_facts(),
            "counts": counts_of(results),
            "results": results,
            "not_run": list(not_run or []),
            "warnings": list(dict.fromkeys(self.warnings)),
            "handoff": handoff_of(results),
            "table": {"reviewed": self.target.get("reviewed"),
                      "docs": self.inputs["table"].get("docs", {})},
        }
        nsc.write_json_file(self.out / RESULTS_FILE, doc)
        nsc.write_text_file(self.out / REPORT_FILE, report_markdown(doc))
        return doc

    def finish(
        self,
        stopped: str | None,
        halted: nsc.ScriptError | None = None,
        not_run: list[str] | None = None,
    ) -> NoReturn:
        """Write results.json, report.md and the evidence, then emit the run's result: the stop
        on the first chat request (endpoint_unreachable, or first_request_timeout when the NIM
        answered the GETs but not that request in time), the error that `halted` the checks
        (with the checks `not_run`), checks_failed, or passed."""
        failed = [r for r in self.results if r.get("status") == nc.FAIL]
        if halted is not None:
            state = STATE_BY_EXIT.get(halted.exit_code, "unreachable")
            self.warnings.append(
                "The port-forward stopped and could not be restarted, so these checks did not "
                f"run: {', '.join(not_run or []) or 'none'}."
            )
        elif stopped == "timeout":
            state = "timeout"
        elif stopped:
            state = "unreachable"
        else:
            state = "failed" if failed else "passed"
        doc = self.write_run(state, not_run)
        facts = facts_of(doc)
        if halted is not None:
            fail(nsc.ScriptError(
                halted.code,
                f"{halted.message}; the checks that ran, with their evidence, are in {self.out}",
                halted.fix,
                halted.exit_code,
            ), facts)
        if stopped == "timeout":
            nsc.emit(False, "timeout", "fix_errors", facts, [self.slow_error()],
                     nsc.EXIT_EXTERNAL)
        if stopped:
            message = (
                f"the first chat request got no answer ({stopped}), so the run stopped; "
                f"evidence in {self.out}"
            )
            fix = ("Check the URL, the port and that the NIM is ready with the user (for "
                   "Kubernetes, the port-forward); then rerun")
            nsc.emit(False, "unreachable", "fix_errors", facts,
                     [nsc.error("endpoint_unreachable", message, fix)], nsc.EXIT_EXTERNAL)
        if failed:
            shown = ", ".join(f"{r.get('id')} ({r.get('reason')})" for r in failed[:8])
            total = doc["counts"]["total"]
            fix = "Show the user report.md"
            if doc["handoff"]:
                fix += "; the documented server switches are in the hints, a change for nim-deploy"
            nsc.emit(False, "failed", "report", facts,
                     [nsc.error("checks_failed",
                                f"{len(failed)} of {total} checks failed: {shown}", fix)],
                     nsc.EXIT_EXTERNAL)
        nsc.emit(True, "passed", "report", facts)

    def slow_error(self) -> dict[str, str]:
        """first_request_timeout: the NIM answered /v1/models but not the first chat request in
        time. It is reachable, so the fix is time (--timeout), not the URL; a larger value is
        suggested while one is left."""
        result = next((r for r in reversed(self.results) if r.get("reason") == "timeout"), {})
        took = float(result.get("latency_ms") or 0.0) / 1000
        given = self.inputs.get("timeout")
        message = (
            f"the NIM answered /v1/models, but the first chat request ({result.get('id')}) got "
            f"no complete answer within {took:.0f} s, so the run stopped; evidence in {self.out}"
        )
        if given is not None and given >= MAX_TIMEOUT:
            fix = (f"It got no answer even with --timeout {MAX_TIMEOUT:g}, the most allowed: the "
                   "NIM is stuck or overloaded. Show the user the evidence; diagnosing the NIM is "
                   f"nim-deploy's job ({SLOW_DOC})")
        else:
            suggest = min(MAX_TIMEOUT, max(300.0, math.ceil(2 * max(took, given or 0) / 100) * 100))
            fix = (
                "The NIM is reachable but slow: ask the user whether it is still warming up or "
                "busy, then rerun once with more time for each chat request, for example "
                f"--timeout {suggest:g} (at most {MAX_TIMEOUT:g}). Slow generation is documented: "
                f"{SLOW_DOC}. If it still times out, diagnosing the NIM is nim-deploy's job"
            )
        return nsc.error("first_request_timeout", message, fix)

    def not_run(self) -> list[str]:
        """The planned checks (all selected ones, before the plan) that have no result yet."""
        done = {r.get("id") for r in self.results}
        planned = ([entry["check"] for entry in self.plan["checks"]] if self.plan is not None
                   else list(self.inputs.get("checks") or []))
        return [check for check in planned if check not in done]

    def abort(self, code: str, message: str, fix: str, exit_code: int, state: str,
              why: str) -> NoReturn:
        """Stop a run that an interrupt or an unexpected error ended, still with one JSON object:
        the checks that ran keep results.json, report.md and their evidence (with the state and
        the checks not run), unless nothing ran, it is a dry run, or the folder cannot be
        written."""
        facts = self.partial_facts()
        if self.access is not None and self.inputs and not self.args.dry_run and (
            self.results or self.ev.pending
        ):
            not_run = self.not_run()
            if not_run:
                self.warnings.append(f"{why}, so these checks did not run: {', '.join(not_run)}.")
            try:
                doc = self.write_run(state, not_run)
            except (nsc.ScriptError, OSError) as exc:
                problem = exc.message if isinstance(exc, nsc.ScriptError) else nsc.clip(exc, 200)
                message += f"; the results could not be written ({problem})"
            else:
                facts = facts_of(doc)
                message += f"; the checks that ran, with their evidence, are in {self.out}"
        nsc.emit(False, state, "ask_user", facts, [nsc.error(code, message, fix)], exit_code)


def cmd_run(args: argparse.Namespace) -> None:
    run = Run(args)
    try:
        run.execute()
    except nsc.ScriptError as exc:
        fail(exc, run.partial_facts())
    except KeyboardInterrupt:  # Ctrl-C: the port-forward is already stopped (execute's finally)
        run.abort(
            "interrupted",
            f"the run was interrupted (Ctrl-C) after {len(run.results)} check(s)",
            "Ask the user whether to rerun the smoke check; nothing about the NIM changed",
            EXIT_INTERRUPTED, "interrupted", "The run was interrupted",
        )
    except BrokenPipeError:
        raise  # stdout is gone: nothing can be printed (main suppresses it)
    except Exception as exc:  # a bug, or an answer no check expected: never a bare traceback
        nsc.log("nim-smoke-check: unexpected error:\n" + traceback.format_exc())
        run.abort(
            "internal_error",
            f"the run stopped on an unexpected error ({type(exc).__name__}: "
            f"{nsc.clip(exc, 200)}): a bug in the skill, or an answer of the endpoint that no "
            "check expected",
            "Show the user this error and the report; a rerun will likely stop the same way. "
            "Report it as a bug in the skill, with the evidence folder and the Python traceback "
            "from stderr (both are masked)",
            nsc.EXIT_EXTERNAL, "internal_error", "The run stopped on an unexpected error",
        )


# --- features -----------------------------------------------------------------------------------


def urls_of(items: Any) -> list[str]:
    found = [item.get("url") if isinstance(item, Mapping) else item for item in items or []]
    return list(dict.fromkeys(url for url in found if isinstance(url, str)))


def feature_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in nf.FEATURE_KEYS:
        obj = record.get(key)
        if not isinstance(obj, Mapping):
            continue
        item = {k: v for k, v in obj.items() if k not in ("sources", "conflicts", "mode")}
        item["sources"] = urls_of(obj.get("sources"))
        if obj.get("conflicts"):
            item["conflicts"] = urls_of(obj.get("conflicts"))
        out[key] = item
    return out


def cmd_features(args: argparse.Namespace) -> None:
    try:
        if not (args.model or args.image or args.family):
            raise bad_input("features needs --model, --image or --family",
                            "Pass the served model id, the image or a family id")
        if args.image is not None and not nsa.IMAGE_REF.fullmatch(args.image):
            raise bad_input("--image is not an image reference",
                            "Pass the image as nvcr.io/nim/TEAM/REPO:TAG")
        table = nf.load_table()
        record, how = nf.match(table, image=args.image, served_id=args.model,
                               family=args.family)
        version = args.version or nf.image_tag(args.image)
        plan = nf.decide(record, checks=nf.CHECKS, generation=version, backend=args.backend,
                         table=table, image=args.image)
    except nsc.ScriptError as exc:
        fail(exc)
    warnings = list(plan["warnings"])
    if args.family is not None:
        mismatch = nf.family_warning(table, args.family, image=args.image, served_id=args.model)
        if mismatch:
            warnings.insert(0, mismatch)
    facts = {
        "record": plan["record"],
        "name": plan["name"],
        "status": plan["status"],
        "kind": plan["kind"],
        "how": how,
        "input": {"model": args.model, "image": args.image, "family": args.family,
                  "resource_id": nf.resource_id(args.image), "version": version},
        "generation": plan["generation"],
        "reviewed": plan["reviewed"],
        "docs": table.get("docs", {}),
        "features": feature_summary(record) if record else {},
        "server_switches": (record or {}).get("server_switches") or {},
        "plan": plan_view(plan),
        "warnings": warnings,
    }
    nsc.emit(True, "matched" if record else "no_record", "done", facts)


# --- report -------------------------------------------------------------------------------------


def load_results(folder: str) -> dict[str, Any]:
    """results.json of a finished run: a regular file of at most 20 MB, not a symlink, with
    this skill's schema, as facts_of reads it: each result an object, and its warnings and
    known_issues, and the run's warnings, lists when they are set."""
    path = Path(folder).expanduser() / RESULTS_FILE
    fix = "Pass the --out folder of a finished run (it holds results.json)"
    if path.is_symlink() or not path.is_file():
        raise bad_input(f"{path} is not a results file", fix)
    if path.stat().st_size > MAX_RESULTS_BYTES:
        raise bad_input(f"{path} is larger than {MAX_RESULTS_BYTES} bytes", fix)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except RecursionError as exc:
        raise bad_input(f"{path} is nested too deeply to read", fix) from exc
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise bad_input(f"{path} cannot be read as JSON: {exc}", fix) from exc
    if not isinstance(doc, dict) or doc.get("schema") != RESULTS_SCHEMA:
        raise bad_input(f"{path} is not a nim-smoke-check results file", fix)
    if not isinstance(doc.get("results"), list):
        raise bad_input(f"{path} has no results list", fix)
    if not isinstance(doc.get("warnings") or [], list):
        raise bad_input(f"{path} has warnings that are not a list", fix)
    for number, result in enumerate(doc["results"], 1):
        if not (isinstance(result, dict) and all(
            isinstance(result.get(key) or [], list) for key in ("warnings", "known_issues")
        )):
            raise bad_input(f"{path}: result {number} is not a check result", fix)
    return doc


def cmd_report(args: argparse.Namespace) -> None:
    try:
        doc = load_results(args.dir)
    except nsc.ScriptError as exc:
        fail(exc)
    facts = facts_of(doc)
    facts["out"] = str(Path(args.dir).expanduser().resolve())
    facts["report"] = str(Path(facts["out"]) / REPORT_FILE)
    facts["run_state"] = doc.get("state")
    facts["created"] = doc.get("created")
    nsc.emit(True, str(doc.get("state") or "unknown"), "done", facts)


# --- the parser ---------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = nsc.JsonArgumentParser(
        prog="nimsmoke.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: python3 scripts/nimsmoke.py run --url http://127.0.0.1:8000",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser(
        "run",
        help="run the smoke checks against a running NIM",
        description="Reach the NIM, identify it, match its record, plan and run the checks. "
        "Writes results.json, report.md and evidence/ in --out (default "
        f"{nsc.CACHE_FOLDER}/<UTC time>-<model>/, private). --dry-run prints the plan after "
        "the GET probes and sends no POST.",
        epilog="Example: python3 scripts/nimsmoke.py run --k8s nimservice/llama --context lab "
        "--namespace nim --api-key-env NGC_GATEWAY_TOKEN",
    )
    where = run.add_mutually_exclusive_group(required=True)
    where.add_argument("--url", metavar="URL",
                       help="the base URL, http(s)://HOST:PORT[/prefix] (a trailing /v1 is fine)")
    where.add_argument("--docker", metavar="NAME", help="a NIM container on this machine")
    where.add_argument("--k8s", metavar="KIND/NAME",
                       help="nimservice/NAME, helm/RELEASE, svc/NAME or pod/NAME")
    run.add_argument("--context", metavar="CTX", help="the kube context (with --k8s)")
    run.add_argument("--namespace", metavar="NS", help="the namespace (with --k8s)")
    run.add_argument("--model", metavar="ID", help="the served model id to test")
    run.add_argument("--image", metavar="IMAGE",
                     help="the image the NIM runs, when the target does not show it")
    run.add_argument("--family", metavar="ID", help="a family id of scripts/model_features.json")
    run.add_argument("--api-key-env", metavar="NAME",
                     help="the NAME of the environment variable that holds the key, never the "
                     "key itself")
    run.add_argument("--allow-plain-http-key", action="store_true",
                     help="send the key over plain http to a host that is not this machine (a "
                     "private gateway), only once the user agrees; without it such a run stops "
                     "before any request (key_over_plain_http). The run keeps a warning")
    run.add_argument("--health-url", metavar="URL", help="the base URL of the health routes")
    run.add_argument("--ca-file", metavar="FILE", help="a PEM file of the endpoint's CA")
    run.add_argument("--client-cert", metavar="FILE", help="a PEM client certificate (mTLS)")
    run.add_argument("--client-key", metavar="FILE", help="its PEM key, with no passphrase")
    run.add_argument("--checks", action="append", metavar="LIST",
                     help=f"only these checks, comma-separated: {', '.join(nf.CHECKS)}")
    run.add_argument("--skip", action="append", metavar="LIST", help="leave out these checks")
    run.add_argument("--extended", action="store_true",
                     help="also tools_stream, tools_parallel and tools_none")
    run.add_argument("--concurrency", type=int, metavar="N",
                     help=f"requests at once in the concurrency check, 1 to {nc.MAX_CONCURRENCY}"
                     f" (default {nc.DEFAULT_CONCURRENCY})")
    run.add_argument("--documented", action="append", metavar="FEATURE=URL",
                     help=f"a public page that documents FEATURE ({', '.join(nf.FEATURES)}) for "
                     "a model with no record")
    run.add_argument("--reasoning-on", metavar="SWITCH",
                     help="system=TEXT, user_suffix=TEXT, kwargs.KEY=VALUE, effort=LEVEL or "
                     "default")
    run.add_argument("--reasoning-off", metavar="SWITCH", help="the same syntax")
    run.add_argument("--timeout", type=float, metavar="S",
                     help="the seconds each chat request may take, up to "
                     f"{MAX_TIMEOUT:g}: more for a slow NIM, or less (default 60 + 0.05 per "
                     f"max_tokens token, at most {nc.TIMEOUT_CAP:g}); a GET takes at most "
                     f"{nc.GET_TIMEOUT:g}")
    run.add_argument("--wait-ready", type=float, default=0.0, metavar="S",
                     help="wait up to S seconds for the NIM to be ready before the checks: on "
                     "Kubernetes for a Running and Ready pod, then /v1/health/ready is polled; "
                     "a NIM still not ready then gets no other check")
    run.add_argument("--out", metavar="DIR", help="a new or empty private folder")
    run.add_argument("--dry-run", action="store_true",
                     help="print the plan and stop: no POST, no file")
    run.set_defaults(func=cmd_run)

    features = sub.add_parser(
        "features",
        help="offline: the matched record and the plan of every check",
        description="Offline: match a model id, an image or a family against "
        "scripts/model_features.json and print how it matched, each feature with its sources, "
        "and the plan of every check with its reason. Sends no request.",
        epilog="Example: python3 scripts/nimsmoke.py features --image "
        "nvcr.io/nim/openai/gpt-oss-20b:2.1.5",
    )
    features.add_argument("--model", metavar="ID", help="a served model id")
    features.add_argument("--image", metavar="IMAGE", help="an image reference")
    features.add_argument("--family", metavar="ID", help="a family id")
    features.add_argument("--version", metavar="VERSION",
                          help="the NIM version (default: the image tag)")
    features.add_argument("--backend", choices=("vllm", "sglang"), help="the backend")
    features.set_defaults(func=cmd_features)

    report = sub.add_parser(
        "report",
        help="print the summary of a finished run again",
        description="Read results.json of a finished run and print its summary, in the same "
        "facts shape as run. Sends no request.",
        epilog="Example: python3 scripts/nimsmoke.py report "
        f"{nsc.CACHE_FOLDER}/20261111T120000Z-meta-llama-3.1-8b-instruct",
    )
    report.add_argument("dir", metavar="DIR", help="the --out folder of the run")
    report.set_defaults(func=cmd_report)
    return parser


def crashed(exc: Exception) -> NoReturn:
    """features or report hit an unexpected error: still one JSON object (run has its own
    handler, which also keeps the evidence)."""
    nsc.log("nim-smoke-check: unexpected error:\n" + traceback.format_exc())
    nsc.emit(False, "internal_error", "ask_user", {}, [nsc.error(
        "internal_error",
        f"an unexpected error ({type(exc).__name__}: {nsc.clip(exc, 200)}): a bug in the skill",
        "Show the user this error; report it as a bug in the skill, with the Python traceback "
        "from stderr",
    )], nsc.EXIT_EXTERNAL)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    with contextlib.suppress(BrokenPipeError):
        try:
            args.func(args)
        except BrokenPipeError:
            raise
        except Exception as exc:
            crashed(exc)


if __name__ == "__main__":
    main()
