#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check a running NIM endpoint (stdlib urllib only).

  ready   poll /v1/health/ready (NIM; 503 while loading, 200 when ready) or /health (when the
          first path is 404), then GET /v1/models; a run that pins a profile also reads
          GET /v1/metadata to confirm the active profile (see profile_check)
  smoke   one chat completion with max_tokens 1
  triage  match container or pod logs against log-triage.json and print the fixes

With --run RUN the result is stored in the run and the run advances (ready: wait_ready > verify or
diagnose; a failed smoke > diagnose). Prints one JSON object.
Exit codes: 0 ok, 2 bad input, 3 precondition failed, 4 external tool or endpoint failed.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, NoReturn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nd_common as nd
import nextstep
import nimdeploy

TRIAGE_RULES = Path(__file__).resolve().parent / "log-triage.json"
NIM_HEALTH, FALLBACK_HEALTH = "/v1/health/ready", "/health"


def advance(state: dict[str, Any] | None, *path: str) -> None:
    """Follow the given states while each edge is legal; stop silently otherwise."""
    for to in path if state else ():
        assert state is not None
        if to not in nd.EDGES.get(state["state"], ()):
            return
        try:
            nd.transition(state, to)
        except ValueError as exc:
            nd.log(f"verify: {exc}")
            return


def remember(state: dict[str, Any] | None, key: str, facts: dict[str, Any]) -> None:
    if state is not None:
        state.setdefault("verify", {})[key] = facts
        nd.save_state(state)


def finish(
    state: dict[str, Any] | None,
    ok: bool,
    next_: str,
    facts: dict[str, Any],
    errors: list[dict[str, str]] | None = None,
    code: int | None = None,
) -> NoReturn:
    if state is not None:
        facts = {"next_command": nextstep.next_step(state)[1], **facts}
    default = nd.EXIT_OK if ok else nd.EXIT_EXTERNAL
    nd.emit(
        ok,
        state["state"] if state else "verify",
        next_,
        facts,
        errors,
        code if code is not None else default,
    )


def get_json(
    url: str, body: dict[str, Any] | None = None, timeout: float = 10
) -> tuple[int, Any, str]:
    """(status, parsed JSON or None, error text). Status 0 means the endpoint was unreachable."""
    try:
        scheme = urllib.parse.urlsplit(url).scheme
    except ValueError:  # an unclosed IPv6 bracket, or a host that NFKC normalization rejects
        # A constant message: urlsplit puts the whole netloc, password included, in its error.
        return 0, None, "endpoint URL could not be parsed"
    if scheme not in ("http", "https"):
        # Only the scheme is named: a URL can carry credentials, and this text may be logged.
        return 0, None, f"only http and https endpoints are checked, got scheme {scheme!r}"
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"} if data else {}
    )
    try:
        # The scheme check above keeps file: and custom schemes away from urlopen.
        with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310
            raw = response.read().decode(errors="replace")
            return response.status, json.loads(raw) if raw.strip() else None, ""
    except urllib.error.HTTPError as exc:
        return exc.code, None, exc.read().decode(errors="replace")[:300]
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return 0, None, str(exc)[:300]


def served_ids(base: str) -> tuple[int, list[str], str]:
    status, models, detail = get_json(f"{base}/v1/models")
    items = (models or {}).get("data", []) if isinstance(models, dict) else []
    return (
        status,
        [item["id"] for item in items if isinstance(item, dict) and item.get("id")],
        detail,
    )


# Keys that name the one profile the server runs. The public NIM API reference says /v1/metadata
# returns the active profile, model info and license; its exact JSON shape is not verified against
# a live NIM, so any of these keys at any depth counts. A plural key ("profiles") lists what the
# image has, not what runs: it and everything under it is ignored.
ACTIVE_PROFILE_KEY = re.compile(
    r"^(?:(?:active|selected|current|running|chosen)[_ -]?(?:model[_ -]?)?profile"
    r"(?:[_ -]?(?:id|name))?|(?:model[_ -]?)?profile(?:[_ -]?(?:id|name))?)$",
    re.IGNORECASE,
)
# Lists of what the image has ("profiles", "available_profiles", "supported_profile_ids"): their
# children are not the running profile, so they are not searched.
AVAILABLE_PROFILES_KEY = re.compile(
    r"^(?!(?:active|selected|current|running|chosen))"
    r".*(?:profiles|profile[_ -]?(?:ids|names|list))$",
    re.IGNORECASE,
)
PROFILE_ID = re.compile(r"^[a-f0-9]{64}$", re.IGNORECASE)


def active_profiles(data: Any) -> list[str]:
    """Every string under an active-profile key (a nested {id, name} counts too)."""
    found: list[str] = []
    if isinstance(data, list):
        for item in data:
            found += active_profiles(item)
    elif isinstance(data, dict):
        for key, value in data.items():
            if ACTIVE_PROFILE_KEY.match(str(key)):
                if isinstance(value, str) and value:
                    found.append(value)
                elif isinstance(value, dict):
                    names = ("id", "profile_id", "profileId", "name", "profile_name")
                    found += [str(value[n]) for n in names if isinstance(value.get(n), str)]
            if isinstance(value, (dict, list)) and not AVAILABLE_PROFILES_KEY.match(str(key)):
                found += active_profiles(value)
    return found


def profile_check(state: dict[str, Any] | None, base: str) -> dict[str, Any] | None:
    """Compare the run's pinned profile with the one the server reports (None: nothing pinned).

    status: confirmed, mismatch (the server names another profile), or not_confirmed (no
    /v1/metadata, or no active-profile field in it): a visible warning, never a silent pass."""
    expected = nd.input_value(state, "model_profile") if state else None
    if not expected:
        return None
    name = ((state or {}).get("handoff") or {}).get("profile", {}).get("name")
    check: dict[str, Any] = {"expected": expected, "expected_name": name}
    status, data, detail = get_json(f"{base}/v1/metadata")
    if status != 200 or not isinstance(data, (dict, list)):
        reason = f"GET /v1/metadata -> {status or detail or 'no JSON'}"
        return {**check, "status": "not_confirmed", "detail": reason}
    found = active_profiles(data)
    check["found"] = found
    if expected in found:
        return {**check, "status": "confirmed", "detail": "/v1/metadata names the pinned profile"}
    # A coarse name (vllm-bf16-tp1-pp1) is shared by several profiles, so it only counts when the
    # server gave no id: a found id that is not the pinned one is a mismatch.
    ids = [value for value in found if PROFILE_ID.match(value)]
    if found and not ids:
        if name and name in found:
            return {
                **check,
                "status": "confirmed",
                "detail": "/v1/metadata names the pinned profile by name only (no id given)",
            }
        # No id to compare, and no matching name: the run may still be correct (set by hand, or
        # the server formats names differently), so warn instead of sending it to diagnose.
        return {
            **check,
            "status": "not_confirmed",
            "detail": f"/v1/metadata names {found} by name only; cannot match the pinned id",
        }
    if found:
        return {**check, "status": "mismatch", "detail": f"/v1/metadata names {found}"}
    return {**check, "status": "not_confirmed", "detail": "/v1/metadata has no profile field"}


def profile_warning(check: dict[str, Any] | None) -> list[str]:
    if not check or check["status"] != "not_confirmed":
        return []
    return [
        f"PROFILE NOT CONFIRMED: {check['detail']}; the run pins {check['expected']}, but the "
        "server did not say which profile it runs. Tell the user."
    ]


def parse_mapping(text: str) -> dict[str, str]:
    """`docker port NAME` lines (`8000/tcp -> 0.0.0.0:8100`) as {container port: host port}."""
    found: dict[str, str] = {}
    for line in text.splitlines():
        match = re.match(r"\s*(\d+)/tcp\s*->\s*\S*?:(\d+)\s*$", line)
        if match:
            found[match[1]] = match[2]
    return found


def port_problem(state: dict[str, Any] | None) -> dict[str, str] | None:
    """For a docker run: the real `docker port` mapping must match what the run asked for."""
    if state is None or state["target"] != "docker" or not shutil.which("docker"):
        return None
    name = nextstep.container_name(state)
    container, host = nd.input_value(state, "container_port"), nd.input_value(state, "port")
    if not container or not host:
        return None
    proc = subprocess.run(
        ["docker", "port", name],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    mapping = parse_mapping(proc.stdout)
    if proc.returncode != 0 or str(mapping.get(str(container))) == str(host):
        return None
    image_ports = nd.input_value(state, "image_ports") or "unknown"
    shown = ", ".join(f"{c}->{h}" for c, h in mapping.items()) or "nothing"
    message = (
        f"image exposes {image_ports}, you mapped {container}->{host}; docker port shows {shown}"
    )
    fix = (
        "Ask the user, then nimdeploy.py set RUN container_port=N (the port the container log says "
        "it serves on), go diagnose > render and approve again"
    )
    return nd.error("port_mismatch", message, fix)


def cmd_ready(args: argparse.Namespace) -> None:
    state = nimdeploy.load_or_fail(args.run) if args.run else None
    advance(state, "wait_ready")
    mismatch = port_problem(state)
    if mismatch:
        remember(state, "ready", {"ready": False, "port_mismatch": mismatch["message"]})
        advance(state, "diagnose")
        finish(state, False, "diagnose", {"ready": False, "url": args.url}, [mismatch])
    base = args.url.rstrip("/")
    auto = args.health_path == "auto"
    health = NIM_HEALTH if auto else "/" + args.health_path.lstrip("/")
    deadline, polls, last = time.monotonic() + args.timeout, 0, ""
    while True:
        polls += 1
        status, _, detail = get_json(f"{base}{health}")
        if status == 404 and auto and health == NIM_HEALTH:  # some servers only serve GET /health
            health = FALLBACK_HEALTH
            status, _, detail = get_json(f"{base}{health}")
        last = f"{health} -> {status or detail}"
        if status == 200:
            status, ids, detail = served_ids(base)
            if status == 200 and ids:
                facts = {
                    "ready": True,
                    "models": ids,
                    "polls": polls,
                    "url": base,
                    "health_path": health,
                }
                check = profile_check(state, base)
                if check:
                    facts.update(profile_check=check, warnings=profile_warning(check))
                if check and check["status"] == "mismatch":
                    profile_mismatch(state, facts, check)
                remember(state, "ready", facts)
                advance(state, "verify")
                finish(state, True, "verify", facts)
            last = f"/v1/models -> {status or detail}"
        if time.monotonic() + args.interval > deadline:
            break
        time.sleep(args.interval)
    facts = {"ready": False, "polls": polls, "url": base, "last": last, "health_path": health}
    remember(state, "ready", facts)
    advance(state, "diagnose")
    fix = (
        "Run verify.py triage (--docker NAME or --k8s NS/POD --context CTX) "
        "and show the matches to the user"
    )
    if state is not None and state["target"] == "docker":
        fix += (
            f". The run maps container port {nd.input_value(state, 'container_port')} to host port "
            f"{nd.input_value(state, 'port')}; if the container log shows another serving port, "
            "ask the user and nimdeploy.py set RUN container_port=N"
        )
    finish(
        state,
        False,
        "diagnose",
        facts,
        [nd.error("not_ready", f"{base} not ready after {args.timeout:g}s ({last})", fix)],
    )


def profile_mismatch(
    state: dict[str, Any] | None, facts: dict[str, Any], check: dict[str, Any]
) -> NoReturn:
    """The server runs another profile than the pinned one: not done; diagnose and ask."""
    facts["ready"] = False
    remember(state, "ready", facts)
    advance(state, "diagnose")
    message = (
        f"the server runs profile {', '.join(check['found'])}, not the pinned {check['expected']} "
        "(chosen in nim-select)"
    )
    fix = (
        "Stop and ask the user. Never drop model_profile to let NIM pick: run verify.py triage, "
        "check the image's list-model-profiles for the pinned id, and roll back if it is missing"
    )
    finish(state, False, "diagnose", facts, [nd.error("profile_mismatch", message, fix)])


def cmd_smoke(args: argparse.Namespace) -> None:
    state = nimdeploy.load_or_fail(args.run) if args.run else None
    base = args.url.rstrip("/")
    requested = args.model or (nd.input_value(state, "model") if state else None)
    _, ids, _ = served_ids(base)
    if requested and (requested in ids or not ids):
        model = requested
    elif len(ids) == 1:
        model = ids[0]
    else:
        message = f"{base}/v1/models serves {ids or 'no models'} and none is {requested!r}"
        finish(
            state,
            False,
            "ask_user",
            {"requested": requested, "served_models": ids},
            [
                nd.error(
                    "model_ambiguous",
                    message,
                    "Ask the user which id to test, then pass --model ID",
                )
            ],
            nd.EXIT_PRECONDITION,
        )
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "max_tokens": 1,
    }
    start = time.perf_counter()
    status, payload, detail = get_json(f"{base}/v1/chat/completions", body, args.timeout)
    latency = round((time.perf_counter() - start) * 1000, 1)
    choices = (payload or {}).get("choices") or [] if isinstance(payload, dict) else []
    if status == 200 and choices:
        facts = {
            "model": model,
            "latency_ms": latency,
            "finish_reason": choices[0].get("finish_reason"),
        }
        check = ((state or {}).get("verify") or {}).get("ready", {}).get("profile_check")
        if check:  # repeated here so the final report shows it
            facts.update(profile_check=check, warnings=profile_warning(check))
        remember(state, "smoke", facts)
        if state is not None:  # smoke passed: the next step is to record, not to smoke again
            facts["next_command"] = nextstep.script(
                f"nimdeploy.py record {state['run_id']} --result ok"
            )
        finish(state, True, "record", facts)
    facts = {"model": model, "latency_ms": latency, "status": status}
    remember(state, "smoke", facts)
    advance(state, "diagnose")
    message = f"chat completion returned {status}: {detail or payload}"[:300]
    finish(
        state,
        False,
        "diagnose",
        facts,
        [
            nd.error(
                "smoke_failed",
                message,
                "Check the model id with GET /v1/models, then run verify.py triage",
            )
        ],
    )


def collect_logs(args: argparse.Namespace) -> str:
    if args.log:
        return Path(args.log).read_text(errors="replace")
    if args.docker:
        commands = [["docker", "logs", "--tail", str(args.tail), args.docker]]
    else:
        namespace, _, pod = args.k8s.partition("/")
        scope = ["--context", args.context, "-n", namespace]
        commands = [
            ["kubectl", "logs", pod, *scope, "--all-containers", "--tail", str(args.tail)],
            ["kubectl", "describe", "pod", pod, *scope],
        ]
    text: list[str] = []
    failures: list[str] = []
    for argv in commands:
        if not shutil.which(argv[0]):
            raise RuntimeError(f"{argv[0]} is not on PATH")
        proc = subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=60,
            check=False,
        )
        # a failed command printed an error, not the workload's logs: never scan it for patterns
        (text if proc.returncode == 0 else failures).append(proc.stdout)
    if not text:
        detail = " | ".join(failure.strip() for failure in failures)
        raise RuntimeError(f"no log command succeeded: {detail}"[:300])
    return "\n".join(text)


def cmd_triage(args: argparse.Namespace) -> None:
    state = nimdeploy.load_or_fail(args.run) if args.run else None
    if args.k8s and (not args.context or "/" not in args.k8s):
        fix = "Pass --k8s NS/POD and --context CTX (never rely on the current context)"
        nd.emit(
            False,
            "triage",
            "ask_user",
            errors=[nd.error("bad_input", "--k8s needs NS/POD and --context", fix)],
        )
    try:
        lines = collect_logs(args).splitlines()[-args.tail :]
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        fix = "Check the container or pod name and that the tool is installed"
        nd.emit(
            False,
            "triage",
            "ask_user",
            errors=[nd.error("logs_unavailable", str(exc)[:300], fix)],
            code=nd.EXIT_EXTERNAL,
        )
    matches = []
    for rule in json.loads(TRIAGE_RULES.read_text()):
        pattern = re.compile(rule["pattern"], re.I)
        line = next((line for line in lines if pattern.search(line)), None)
        if line is not None:
            matches.append(
                {
                    "id": rule["id"],
                    "meaning": rule["meaning"],
                    "fix": nd.absolute_scripts(rule["fix"]),
                    "line": line.strip()[:300],
                }
            )
    facts: dict[str, Any] = {"matches": matches, "lines_scanned": len(lines)}
    if not matches:
        facts["tail"] = [line[:300] for line in lines[-20:]]
    remember(state, "triage", {"matches": [match["id"] for match in matches]})
    advance(state, "diagnose")
    finish(state, True, "ask_user", facts)


def build_parser() -> argparse.ArgumentParser:
    parser = nd.JsonArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: python3 scripts/verify.py ready --url http://localhost:8000 --run RUN",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    ready = sub.add_parser("ready", help="poll until the endpoint is ready")
    ready.add_argument("--url", required=True)
    ready.add_argument("--timeout", type=float, default=900)
    ready.add_argument("--interval", type=float, default=5)
    ready.add_argument(
        "--health-path",
        default="auto",
        help="auto (default): /v1/health/ready, falling back to /health on 404",
    )
    ready.add_argument("--run")
    ready.set_defaults(func=cmd_ready)
    smoke = sub.add_parser("smoke", help="one chat completion with max_tokens 1")
    smoke.add_argument("--url", required=True)
    smoke.add_argument(
        "--model", help="model id (default: the run's model; else the single served id)"
    )
    smoke.add_argument("--timeout", type=float, default=120)
    smoke.add_argument("--run")
    smoke.set_defaults(func=cmd_smoke)
    triage = sub.add_parser("triage", help="match logs against known NIM failures")
    source = triage.add_mutually_exclusive_group(required=True)
    source.add_argument("--log", help="log file")
    source.add_argument("--docker", metavar="NAME", help="container name")
    source.add_argument("--k8s", metavar="NS/POD", help="pod; also reads kubectl describe events")
    triage.add_argument("--context", help="kube context (required with --k8s)")
    triage.add_argument("--tail", type=int, default=500)
    triage.add_argument("--run")
    triage.set_defaults(func=cmd_triage)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
