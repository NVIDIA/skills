#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""check_nim_engagement: decide whether a NIM candidate's settings reached the live engine.

Read-only. Takes the container (`docker container inspect`) or pod (`kubectl get pod -o json`) spec and the
startup log, then checks three things for each expected setting: it was delivered (the variable or
argument is on the container), and the engine launch line (`Launching vLLM: vllm serve ...`) shows
it. Variables with a known engine flag are checked in the launch line; others need --log-pattern.
An argument counts only when both the container and the launch line carry the expected value.
Without --container, the pod's default container (as `kubectl logs` picks it) gives both the spec
and the log; a live Docker log is read by the inspected container's id since its current start,
and a live pod log counts only if the pod and its container did not change while it was read.
--require-launch only asks for a launch line (a case with no settings, such as an untuned NIM).

Status: engaged (exit 0), unconfirmed (delivered but not visible in the engine, exit 3) or
mismatch (absent or a different value, exit 3).

Prints one JSON object: {"ok", "state", "next", "facts", "errors": [{"code", "message", "fix"}]}.
Exit codes: 0 ok, 2 bad input, 3 precondition failed, 4 external tool failed, 5 refused.
Never prompts. Logs go to stderr. Stdlib only.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, NoReturn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import targets
from targets import SECRET_NAME, Scrubber

SCRUB = Scrubber()

EXIT_OK, EXIT_INPUT, EXIT_PRECONDITION, EXIT_EXTERNAL, EXIT_REFUSED = 0, 2, 3, 4, 5
# An engine launch: any command after "Launching vLLM:" / "Launching SGLang:" (the marker says it
# is one, whatever the entrypoint), or a known engine command at the start of a log message. A
# sentence that only names the command is not a launch: NIM 2.1.6 prints "Restart vllm serve with
# --enable-scale-out ..." after startup.
LAUNCH_COMMAND = re.compile(
    r"Launching (?:vLLM|SGLang):\s*\S+\s+\S"
    r"|^(?:vllm serve|sglang serve|(?:\S*/)?python3?(?:\.\d+)? -m "
    r"(?:sglang\.launch_server|vllm\.entrypoints\.openai\.api_server))\s+\S"
)
# NIM variables whose effect shows as an engine flag in the launch line (vLLM backend, NIM 2.1.x)
ENV_TO_FLAG = {
    "NIM_TENSOR_PARALLEL_SIZE": "--tensor-parallel-size",
    "NIM_PIPELINE_PARALLEL_SIZE": "--pipeline-parallel-size",
    "NIM_KVCACHE_PERCENT": "--gpu-memory-utilization",
    "NIM_MAX_MODEL_LEN": "--max-model-len",
    "NIM_SERVED_MODEL_NAME": "--served-model-name",
}


def emit(
    ok: bool,
    state: str,
    next_: str,
    facts: dict[str, Any] | None = None,
    errors: list[dict[str, str]] | None = None,
    code: int | None = None,
) -> NoReturn:
    """Print the contract JSON (ok, state, next, facts, errors), then exit."""
    payload = {
        "ok": ok,
        "state": state,
        "next": next_,
        "facts": facts or {},
        "errors": errors or [],
    }
    # a launch line can carry a secret flag (--api-key X): never print it
    sys.stdout.write(SCRUB.dumps(payload, sort_keys=True) + "\n")
    raise SystemExit(code if code is not None else (EXIT_OK if ok else EXIT_INPUT))


def fail(code: str, message: str, fix: str, exit_code: int) -> NoReturn:
    """Print one contract error and exit with the given code."""
    emit(
        False,
        "error",
        "fix_errors",
        errors=[{"code": code, "message": message, "fix": fix}],
        code=exit_code,
    )


def add_named(value: str | None) -> None:
    """Register the value of a secret-named variable or flag, masked at any length: a short
    value can still be a credential. Under 6 characters it is masked wherever no letter or digit
    touches it (targets.SHORT_TOKEN), so "1" masks "--tensor-parallel-size 1" but never 128, a
    version or a uid; display fields only, never the verdict or the JSON contract."""
    if value:
        SCRUB.add(value, named=True)


def started_at(box: Any) -> str | None:
    """A `docker inspect` entry's State.StartedAt, or None when it has none (or no State)."""
    state = box.get("State") if isinstance(box, dict) else None
    started = state.get("StartedAt") if isinstance(state, dict) else None
    return started if isinstance(started, str) and started else None


def mask(key: str, value: str | None) -> str | None:
    """The value, or <redacted> when the name looks like a credential."""
    if value is not None and SECRET_NAME.search(key):
        return "<redacted>"
    return value


def run(cmd: list[str], with_stderr: bool = False) -> str:
    """Run a command and return its stdout (and stderr if asked); a failure ends the check."""
    print("running: " + " ".join(cmd[:6]), file=sys.stderr)
    try:
        done = subprocess.run(
            cmd, capture_output=True, text=True, timeout=90, check=False, stdin=subprocess.DEVNULL
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        fail(
            "tool_failed",
            f"cannot run {cmd[0]}: {exc}",
            f"Install {cmd[0]} or check access, then rerun.",
            EXIT_EXTERNAL,
        )
    if done.returncode != 0:
        fail(
            "tool_failed",
            f"{' '.join(cmd[:4])} failed: {done.stderr.strip()[:300]}",
            "Check the target name, namespace and context, then rerun.",
            EXIT_EXTERNAL,
        )
    return done.stdout + done.stderr if with_stderr else done.stdout


def read_file(path: Path, what: str) -> str:
    """The text of a saved spec or log; an unreadable file ends the check as bad input."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")  # as the runner saves it
    except OSError as exc:
        fail(
            "unreadable_input",
            f"{what} {path}: {exc.strerror or exc}",
            f"Pass a readable {what}, or drop it for a live lookup.",
            EXIT_INPUT,
        )


def load_json(text: str, source: str, code: str, exit_code: int) -> Any:
    """Parse a spec; JSON that does not parse ends the check with `code`."""
    try:
        return json.loads(text)
    except ValueError as exc:
        fail(code, f"{source} is not JSON: {exc}", f"Check {source}, then rerun.", exit_code)


# as Kubernetes parses it: $$ is a literal $, and $(NAME) reads NAME up to the closing ")"
# (env names may hold "-" and "." anywhere, e.g. "-LIMIT")
K8S_REF = re.compile(r"\$\$|\$\(([^)]+)\)")


def refs(text: str) -> set[str]:
    """The names a text refers to as $(NAME) ($$ is a literal $, no reference)."""
    return {m[1] for m in K8S_REF.finditer(text) if m[0] != "$$" and m[1]}


def k8s_expand(text: str, known: dict[str, str]) -> str:
    """One pass of Kubernetes $(NAME) expansion: $$ is a literal $, a name with no known value
    stays as written."""
    return K8S_REF.sub(lambda m: "$" if m[0] == "$$" else known.get(m[1], m[0]), text)


# the variables the kubelet injects for a service ({NAME}_SERVICE_HOST, {NAME}_PORT_80_TCP, ...):
# always for the `kubernetes` service, for every service in the namespace unless the pod sets
# enableServiceLinks: false. Kubernetes resolves $(NAME) from them too; they are not in the spec
SERVICE_VAR = re.compile(
    r"[A-Z0-9_]+_(?:SERVICE_HOST|SERVICE_PORT(?:_[A-Z0-9_]+)?"
    r"|PORT(?:_\d+_(?:TCP|UDP|SCTP)(?:_(?:PROTO|PORT|ADDR))?)?)"
)


def service_var(name: str, service_links: bool) -> bool:
    """Whether the kubelet may inject NAME for a service (its value is not readable here)."""
    return bool(SERVICE_VAR.fullmatch(name)) and (service_links or name.startswith("KUBERNETES_"))


def k8s_unread(
    text: str, known: dict[str, str], hidden: set[str], env_from: bool, service_links: bool
) -> bool:
    """Whether Kubernetes resolves a $(NAME) in TEXT from a value not readable here: a
    valueFrom variable (`hidden`), any name not set literally when the container has envFrom,
    or a service variable the kubelet may inject. An escaped $$(NAME), or a name defined
    nowhere, stays literal text."""
    return any(
        m[0] != "$$"
        and m[1] not in known
        and (m[1] in hidden or env_from or service_var(m[1], service_links))
        for m in K8S_REF.finditer(text)
    )


def bad_spec(what: str) -> NoReturn:
    fail(
        "bad_spec",
        f"the pod's {what} is not well formed",
        "Pass the output of `kubectl get pod NAME -o json`.",
        EXIT_INPUT,
    )


def pod_containers(spec: Any, field: str = "containers") -> list[dict[str, Any]]:
    """spec.<field> of a pod (spec.containers by default), checked by targets.pod_containers:
    absent or null is [], anything malformed is bad_spec (never a traceback)."""
    try:
        return targets.pod_containers(spec, field)
    except ValueError as exc:
        bad_spec(str(exc))


def find_container(spec: Any, name: str) -> tuple[dict[str, Any], str] | None:
    """The named container of any kind and the status field that holds its status, else None."""
    try:
        found = targets.pod_container(spec, name)
    except ValueError as exc:
        bad_spec(str(exc))
    return (found[1], found[0]) if found else None


def status_field_of(spec: Any, name: str | None) -> str:
    """The status field for the named container (spec.containers' when it is not found)."""
    found = find_container(spec, name) if name else None
    return found[1] if found else "containerStatuses"


def default_container(spec: Any) -> str | None:
    """The container `kubectl logs` reads without -c (targets.pod_container, kubectl's rules:
    the default-container annotation naming a container of any kind, else the first
    container). Both the spec and the log are then read from this one container."""
    if not isinstance(spec, dict):
        return None
    try:
        found = targets.pod_container(spec)
    except ValueError as exc:
        bad_spec(str(exc))
    return found[1].get("name") if found else None


def parse_pairs(items: list[str]) -> dict[str, str]:
    """Turn KEY=VALUE items into a dict; an item without = ends the check."""
    out: dict[str, str] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not key or not sep:
            fail(
                "bad_env",
                # never the item: without "=" all of it may be a credential's value
                "an --env is not KEY=VALUE",
                "Pass each variable as --env KEY=VALUE.",
                EXIT_INPUT,
            )
        out[key] = value
    return out


# the name of a flag at the start of an --arg: letters, digits, "-" and "_" only, so a value
# joined to it by "=", a space or a quote is never part of it
FLAG_NAME = re.compile(r"--[A-Za-z0-9][A-Za-z0-9_-]*")


def parse_arg(text: str) -> tuple[str, str | None]:
    """'--tp 2' or '--tp=2' -> ('--tp', '2'); a switch has no value; a JSON value stays whole."""
    flag, _, rest = text.replace("=", " ", 1).partition(" ")
    if flag.startswith("--") and rest.strip().startswith(("{", "[")):
        return flag, rest.strip()  # --speculative-config {"method": "ngram", ...}
    try:
        parts = shlex.split(text.replace("=", " ", 1)) if text.startswith("--") else []
    except ValueError:  # unbalanced quotes
        parts = []
    if not parts or not parts[0].startswith("--") or len(parts) > 2:
        # name only the flag, never the text: the value of a secret flag (--api-key "...) is
        # not yet registered for masking and an unclosed quote hides it from the flag scrubber
        name = FLAG_NAME.match(text)
        fail(
            "bad_arg",
            f"--arg {name.group(0)!r} is not a flag with at most one value"
            if name
            else "an --arg does not start with --FLAG",
            "Pass --arg '--flag VALUE' or --arg '--flag' for a switch.",
            EXIT_INPUT,
        )
    return parts[0], parts[1] if len(parts) == 2 else None


def flag_value(tokens: list[str], flag: str) -> tuple[bool, str | None]:
    """Whether `flag` is present and its value; the last copy wins, as in the engine's argparse."""
    found, value = False, None
    for i, tok in enumerate(tokens):
        if tok == flag:
            nxt = tokens[i + 1] if i + 1 < len(tokens) else None
            found, value = True, None if nxt is None or nxt.startswith("--") else nxt
        elif tok.startswith(flag + "="):
            found, value = True, tok.split("=", 1)[1]
    return found, value


def structured(text: str | None) -> Any:
    """A JSON or Python-literal object or list, else None."""
    if not text or not text.lstrip().startswith(("{", "[")):
        return None
    try:
        return json.loads(text)
    except ValueError:
        pass
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None


def raw_object(line: str, flag: str) -> str | None:
    """The balanced {...} or [...] right after `flag` in a log line, however it was quoted."""
    i = line.rfind(flag)  # the last copy wins
    if i < 0:
        return None
    j = i + len(flag)
    while j < len(line) and line[j] in " ='\"":
        j += 1
    if j >= len(line) or line[j] not in "{[":
        return None
    close = {"{": "}", "[": "]"}
    depth, quote = [], ""
    for k in range(j, len(line)):
        ch = line[k]
        if quote:
            quote = "" if ch == quote and line[k - 1] != "\\" else quote
        elif ch in "\"'":
            quote = ch
        elif ch in close:
            depth.append(close[ch])
        elif depth and ch == depth[-1]:
            depth.pop()
            if not depth:
                return line[j : k + 1]
    return None


def same(want: str | None, have: str | None) -> bool:
    """True when the wanted setting equals the live one: structured, numeric or plain text match."""
    if want is None:
        return True
    if have is None:
        return False
    if structured(want) is not None:
        return bool(structured(want) == structured(have))
    try:
        return float(want) == float(have)
    except ValueError:
        return want == have


def read_spec(
    mode: str, spec: Any, container: str | None = None
) -> tuple[dict[str, str], list[str], list[str], list[bool], dict[str, Any]]:
    """Return (env, value_from names, args, unread, identity) from docker inspect or a pod
    object; unread[i] is True when args[i] holds a reference whose value is not readable here.

    In a pod, only the selected container (`container`, else the first) is read, so evidence
    from different containers is never mixed.
    """
    if mode == "docker":
        info = spec[0] if isinstance(spec, list) else spec
        if not info:
            fail(
                "no_target",
                "docker inspect returned nothing",
                "Check the container name with `docker ps -a`.",
                EXIT_PRECONDITION,
            )
        cfg = info.get("Config", {})
        docker_env = dict(e.partition("=")[::2] for e in cfg.get("Env") or [])
        docker_args = list(info.get("Args") or cfg.get("Cmd") or [])
        ident = {
            "image": cfg.get("Image"),
            "image_id": info.get("Image"),
            "running": info.get("State", {}).get("Running"),
            "restart_count": info.get("RestartCount"),
        }
        return docker_env, [], docker_args, [False] * len(docker_args), ident
    if not isinstance(spec, dict):
        bad_spec("JSON")
    containers = pod_containers(spec)
    if not containers:
        fail(
            "no_target",
            "pod has no containers",
            "Check the pod name with `kubectl get pods`.",
            EXIT_PRECONDITION,
        )
    found = find_container(spec, container) if container else (containers[0], "containerStatuses")
    if not found:
        fail(
            "no_target",
            f"pod has no container {container!r}",
            "Pass a container name from the pod's spec (containers, initContainers or "
            "ephemeralContainers).",
            EXIT_PRECONDITION,
        )
    c, status_field = found
    env: dict[str, str] = {}
    resolved: dict[str, str] = {}  # env as Kubernetes delivers it, $(NAME) resolved in order
    hidden: set[str] = set()  # names whose delivered value is not readable here
    tainted: set[str] = set()  # names whose value is, or embeds, a secret-named variable's
    env_from = bool(c.get("envFrom"))
    # unless the pod turns them off, the kubelet injects a variable for each service
    service_links = (spec.get("spec") or {}).get("enableServiceLinks") is not False
    for e in c.get("env", []):
        # a name declared again replaces its earlier state (the last declaration is delivered)
        if "value" in e or "valueFrom" not in e:
            # Kubernetes omits an empty value from the JSON: no value and no source is ""
            value = e.get("value", "")
            if not isinstance(value, str):
                bad_spec(f"env {e.get('name')!r} value")
            if SECRET_NAME.search(str(e["name"])):
                # $(NAME) can carry it into any argument: mask it wherever it shows up, at any
                # length (the 6-character floor is for values seen in passing)
                add_named(value)
            if k8s_unread(value, resolved, hidden, env_from, service_links):
                # Kubernetes fills it in from a value not readable here: unknown, like valueFrom
                hidden.add(e["name"])
                resolved.pop(e["name"], None)
                continue
            # read the taint of $(NAME) before replacing this name's own: $(SELF) refers to the
            # earlier value, as k8s_expand does
            secret = bool(SECRET_NAME.search(str(e["name"])) or refs(value) & tainted)
            resolved[e["name"]] = k8s_expand(value, resolved)
            hidden.discard(e["name"])
            tainted.discard(e["name"])
            if secret:
                # a secret, or a value that embeds one through $(NAME): masked as a whole, since
                # a short secret glued to other text is no token of its own
                tainted.add(e["name"])
                add_named(resolved[e["name"]])
        else:
            # unknown, also when an optional source is missing and the earlier value stays
            hidden.add(e["name"])
            resolved.pop(e["name"], None)
    value_from = list(hidden)
    env.update(resolved)  # the values the container receives, $(NAME) resolved
    # Kubernetes expands $(NAME) from the container's env in command and args ($$ is a
    # literal $): compare what the container receives. A name with no readable value
    # (valueFrom, envFrom, a service variable) is unread: never confirmed, never a mismatch on
    # its own. An unset name stays as written, as Kubernetes leaves it
    raw = [str(a) for a in list(c.get("command") or []) + list(c.get("args") or [])]
    args = [k8s_expand(a, resolved) for a in raw]
    for a, got in zip(raw, args, strict=True):
        if refs(a) & tainted:
            # an argument that embeds a secret: masked as a whole, and so is its value after
            # "=", which is what the checks report as delivered (flag_value)
            add_named(got)
            if "=" in got:
                add_named(got.split("=", 1)[1])
    unread = [k8s_unread(a, resolved, hidden, env_from, service_links) for a in raw]
    statuses = [
        st
        for st in spec.get("status", {}).get(status_field) or []
        if st.get("name") in (None, c.get("name"))
    ][:1]
    ident = {
        "image": c.get("image"),
        "image_id": statuses[0].get("imageID") if statuses else None,
        "pod_uid": (spec.get("metadata") or {}).get("uid"),
        "phase": spec.get("status", {}).get("phase"),
        "restart_count": sum(s.get("restartCount", 0) for s in statuses),
    }
    return env, value_from, args, unread, ident


def value_unread(tokens: list[str], unread: list[bool], flag: str) -> bool:
    """Whether the last copy of `flag` (as flag_value reads it) takes its value from a token
    holding a reference whose value is not readable here."""
    hit = False
    for i, tok in enumerate(tokens):
        if tok == flag:
            hit = i + 1 < len(tokens) and not tokens[i + 1].startswith("--") and unread[i + 1]
        elif tok.startswith(flag + "="):
            hit = unread[i]
    return hit


def pod_instance(
    spec: Any, name: str | None, status_field: str = "containerStatuses"
) -> tuple[Any, ...]:
    """Which pod and container start a live read saw: the pod's uid, and the named
    container's id and restart count, from STATUS_FIELD (malformed parts read as None, never a
    traceback)."""
    if not isinstance(spec, dict):
        return (None, None, None)
    meta = spec.get("metadata") if isinstance(spec.get("metadata"), dict) else {}
    status = spec.get("status") if isinstance(spec.get("status"), dict) else {}
    boxes = status.get(status_field)
    boxes = boxes if isinstance(boxes, list) else []
    st = next((b for b in boxes if isinstance(b, dict) and b.get("name") in (None, name)), {})
    return (meta.get("uid"), st.get("containerID"), st.get("restartCount"))


def log_messages(log: str) -> list[str]:
    """Each log line's message: JSON lines (also after a "(Worker pid=N)" prefix) give their
    `message`; vLLM's own "LEVEL MM-DD hh:mm:ss [file.py:N]" prefix is dropped."""
    out = []
    for raw in log.splitlines():
        line = raw
        k = raw.find("{")
        if k >= 0:
            with contextlib.suppress(json.JSONDecodeError, AttributeError):
                line = str(json.loads(raw[k:]).get("message", raw))
        line = re.sub(r"^\(\w+ pid=\d+\)\s*", "", line)
        line = re.sub(r"^[A-Z]+ \d\d-\d\d [\d:]+ \[[^\]]+\]\s*", "", line)
        out.append(line.strip())
    return out


def launch_line(log: str) -> str | None:
    """The last engine launch: a log can hold several (restarts, saved logs); the last one runs.
    Only a command-shaped message counts (LAUNCH_COMMAND), from the "Launching ..." marker or the
    command on; text before it on the line (a timestamp or logger prefix) is dropped."""
    found = None
    for line in log_messages(log):
        match = LAUNCH_COMMAND.search(line)
        if match:
            found = line[match.start() :].strip()
    return found


# The engine quietly switching to another code path because of a setting (a confounder: the
# result then measures the setting plus the fallback). Reported, never a failure on its own.
FALLBACK = re.compile(
    r"(?:not (?:yet )?support\w*.*(?:instead|disabled)|will be disabled|falling back|"
    r"using the V1 model runner)",
    re.IGNORECASE,
)


def fallbacks(log: str) -> list[str]:
    """The log messages that say the engine fell back from a setting, without repeats."""
    seen: list[str] = []
    for m in log_messages(log):
        if FALLBACK.search(m) and m[:300] not in seen:
            seen.append(m[:300])
    return seen


class ContractParser(argparse.ArgumentParser):
    """Argument parser that reports a usage error as the contract JSON."""

    def error(self, message: str) -> NoReturn:
        """Report a usage error as a bad_usage contract error and exit.

        Unrecognized arguments are not echoed: a stray token can be a credential's value
        (`--arg --api-key VALUE` passes VALUE on its own).
        """
        if message.startswith("unrecognized arguments:"):
            message = "unrecognized arguments (not shown: one could be a credential's value)"
        fail("bad_usage", message, "Run with --help and fix the arguments.", EXIT_INPUT)


def join_flag_values(argv: list[str]) -> list[str]:
    """Turn `--arg --flag=1` into `--arg=--flag=1`; argparse rejects values that start with `--`."""
    out: list[str] = []
    it = iter(argv)
    for tok in it:
        if tok in ("--arg", "--log-pattern"):
            out.append(f"{tok}={next(it, '')}")
        else:
            out.append(tok)
    return out


def main(argv: list[str] | None = None) -> None:
    """Read the live engine's settings and report whether the candidate's settings reached it."""
    parser = ContractParser(
        description="Check that a NIM candidate's settings reached the live engine (read-only).",
        epilog=(
            "Example: python3 scripts/check_nim_engagement.py k8s nimtune-mnt4096-abc12 "
            "-n nim-tuning --context my-cluster --arg '--max-num-batched-tokens 4096'\n"
            "Example: python3 scripts/check_nim_engagement.py docker nimbench-kv085 --sudo "
            "--env NIM_KVCACHE_PERCENT=0.85"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "mode",
        choices=["docker", "k8s"],
        help="docker container or Kubernetes pod (NIM Operator or Helm)",
    )
    parser.add_argument("target", help="container name or pod name")
    parser.add_argument(
        "--env", action="append", default=[], metavar="KEY=VALUE", help="variable that must be set"
    )
    parser.add_argument(
        "--arg",
        action="append",
        default=[],
        metavar="'--flag VALUE'",
        help="engine argument that must be launched",
    )
    parser.add_argument(
        "--log-pattern",
        action="append",
        default=[],
        metavar="REGEX",
        help="regex the startup log must match; engine evidence for a variable with no launch flag",
    )
    parser.add_argument(
        "--require-launch",
        action="store_true",
        help="the log must hold an engine launch line; the check for a case with no settings",
    )
    parser.add_argument("-n", "--namespace", help="Kubernetes namespace (required for k8s)")
    parser.add_argument("--context", help="kubectl context (required for k8s)")
    parser.add_argument("--container", help="container in the pod, when there are several")
    parser.add_argument("--sudo", action="store_true", help="run docker as `sudo -n docker`")
    parser.add_argument(
        "--spec-file", type=Path, help="saved docker inspect or pod JSON instead of a live lookup"
    )
    parser.add_argument("--log-file", type=Path, help="saved startup log instead of a live lookup")
    args = parser.parse_args(join_flag_values(sys.argv[1:] if argv is None else argv))

    if not (args.env or args.arg or args.log_pattern or args.require_launch):
        fail(
            "nothing_to_check",
            "no --env, --arg, --log-pattern or --require-launch given",
            "Pass the candidate's changed setting, e.g. --arg '--tensor-parallel-size 2'.",
            EXIT_INPUT,
        )
    if (
        args.mode == "k8s"
        and not (args.spec_file and args.log_file)
        and not (args.namespace and args.context)
    ):
        fail(
            "unpinned_kubectl",
            "k8s mode needs -n and --context",
            "Pin the namespace and kube context: -n NAMESPACE --context CONTEXT.",
            EXIT_INPUT,
        )

    expected_env = parse_pairs(args.env)
    expected_args = [parse_arg(a) for a in args.arg]
    patterns = []
    for p in args.log_pattern:
        try:
            patterns.append((p, re.compile(p)))
        except re.error as exc:
            fail(
                "bad_pattern",
                f"--log-pattern {p!r}: {exc}",
                "Pass a valid Python regex.",
                EXIT_INPUT,
            )

    docker = ["sudo", "-n", "docker"] if args.sudo else ["docker"]
    kubectl = ["kubectl", "--context", str(args.context), "-n", str(args.namespace)]
    if args.spec_file:
        spec = load_json(read_file(args.spec_file, "--spec-file"), str(args.spec_file),
                         "bad_spec", EXIT_INPUT)  # fmt: skip
    elif args.mode == "docker":
        # containers only: an image, volume or network of that name is never read as the spec
        spec = load_json(run([*docker, "container", "inspect", args.target]),
                         "docker container inspect", "tool_failed", EXIT_EXTERNAL)  # fmt: skip
    else:
        spec = load_json(run([*kubectl, "get", "pod", args.target, "-o", "json"]),
                         "kubectl get pod", "tool_failed", EXIT_EXTERNAL)  # fmt: skip
    # one container for both the spec and the log, as `kubectl logs` would pick it
    if args.mode == "k8s":
        # checked whatever picks the container: default_container() is skipped with --container
        if not isinstance(spec, dict):
            bad_spec("JSON")
        meta = spec.get("metadata")
        if meta is not None and not isinstance(meta, dict):
            bad_spec("metadata")
    container = args.container or (default_container(spec) if args.mode == "k8s" else None)
    env, value_from, container_args, args_unread, ident = read_spec(args.mode, spec, container)
    if args.log_file:
        log = read_file(args.log_file, "--log-file")
    elif args.mode == "docker":
        # by the inspected container's id, not its name, and only since its current start: a
        # replaced container or a launch line from before a restart is never read as evidence
        info = (spec[0] if isinstance(spec, list) and spec else spec) or {}
        cid = info.get("Id") if isinstance(info, dict) else None
        if not isinstance(cid, str) or not cid:
            fail(
                "no_target",
                "the container spec has no Id to read its log by",
                "Pass a full `docker inspect` output, or both --spec-file and --log-file.",
                EXIT_PRECONDITION,
            )
        started = started_at(info)
        if started is None:
            # without its start time a restart during the read could not be told apart
            fail(
                "no_target",
                "the container spec has no State.StartedAt to bind its log to",
                "Pass a full `docker inspect` output, or both --spec-file and --log-file.",
                EXIT_PRECONDITION,
            )
        since = ["--since", started] if not started.startswith("0001-") else []
        log = run([*docker, "logs", *since, cid], with_stderr=True)
        # --since filters by time only: a restart between the inspect and the read can still
        # let an earlier start's launch line through, so the start must be unchanged after it
        # and the name must still resolve to it: one renamed away (still inspectable by its
        # id) and another given the name would report the old container's settings for it
        for ref in (cid,) if args.spec_file else (cid, args.target):
            again = load_json(run([*docker, "container", "inspect", ref]),
                              "docker container inspect", "tool_failed", EXIT_EXTERNAL)  # fmt: skip
            again = (again[0] if isinstance(again, list) and again else again) or {}
            if not isinstance(again, dict) or again.get("Id") != cid or started_at(again) != started:
                fail(
                    "target_restarted",
                    f"container {cid[:12]} was replaced, renamed or restarted while its log "
                    "was read",
                    "Wait until it is ready, then run the check again.",
                    EXIT_PRECONDITION,
                )
    else:
        log = run(
            kubectl + ["logs", args.target] + (["-c", container] if container else []),
            with_stderr=True,
        )
        if not args.spec_file:
            # a pod replaced under the same name, or a container restart, between the spec read
            # and the log read would mix the old spec with the new launch line
            name = container or pod_containers(spec)[0].get("name")
            again = load_json(run([*kubectl, "get", "pod", args.target, "-o", "json"]),
                              "kubectl get pod", "tool_failed", EXIT_EXTERNAL)  # fmt: skip
            kind = status_field_of(spec, name)
            if pod_instance(again, name, kind) != pod_instance(spec, name, kind):
                fail(
                    "target_restarted",
                    f"pod {args.target} was replaced or restarted while its log was read",
                    "Wait until it is ready, then run the check again.",
                    EXIT_PRECONDITION,
                )
    launch = launch_line(log)
    launch_tokens = (
        shlex.split(launch.split("vllm serve", 1)[-1])
        if launch and "vllm serve" in launch
        else (shlex.split(launch) if launch else [])
    )

    checks: list[dict[str, Any]] = []
    for key, want in expected_env.items():
        have = env.get(key)
        if SECRET_NAME.search(key):  # a named credential: masked at any length
            add_named(want)
            add_named(have)
        row: dict[str, Any] = {
            "setting": key,
            "expected": mask(key, want),
            "delivered": mask(key, have),
        }
        if key in value_from:
            row["status"], row["note"] = (
                "unconfirmed",
                "set from a secret or reference; value not readable here",
            )
        elif not same(want, have):
            row["status"] = "mismatch"
        elif key in ENV_TO_FLAG:
            found, val = flag_value(launch_tokens, ENV_TO_FLAG[key])
            row["engine"] = f"{ENV_TO_FLAG[key]} {val}" if found else None
            row["status"] = (
                "engaged" if found and same(want, val) else ("mismatch" if found else "unconfirmed")
            )
            if found and not same(want, val):
                row["note"] = (
                    "engine got another value; NIM may have clamped it (see the dry-run provenance)"
                )
        else:
            row["status"] = "unconfirmed" if not patterns else "engaged_by_pattern"
        checks.append(row)
    for flag, want_arg in expected_args:
        delivered, dval = flag_value(container_args, flag)
        found, val = flag_value(launch_tokens, flag)
        if structured(want_arg) is not None and launch:
            # a JSON value has spaces and quotes that shlex splits apart; read it from the line
            raw = raw_object(launch, flag)
            found, val = (True, raw) if raw is not None else (found, val)
        if SECRET_NAME.search(flag):  # --api-key X: mask X wherever it appears, the log included
            for v in (want_arg, dval, val):
                add_named(v)  # a named credential: masked at any length, like an env value
        shown = mask(flag, val)
        row = {
            "setting": flag,
            "expected": mask(flag, want_arg),
            "delivered": mask(flag, dval) if delivered else None,
            "engine": (f"{flag} {shown}" if shown is not None else flag) if found else None,
        }
        engine_ok = found and same(want_arg, val)
        delivered_ok = delivered and same(want_arg, dval)
        if delivered and value_unread(container_args, args_unread, flag):
            # Kubernetes fills the value in from a source not readable here: delivery is
            # unknown, so never engaged; the engine running another value is still a mismatch
            row["status"] = "mismatch" if found and not engine_ok else "unconfirmed"
            row["note"] = "delivered through a reference whose value is not readable here"
        elif engine_ok and delivered_ok:
            row["status"] = "engaged"
        elif (found and not engine_ok) or (delivered and not delivered_ok):
            row["status"] = "mismatch"
        elif engine_ok:
            # the engine runs this value, but the candidate never passed it: an engine default
            # is not proof that the candidate's setting reached the engine
            row["status"] = "mismatch"
            row["note"] = "not delivered: the container's arguments do not set it"
        else:
            row["status"] = "unconfirmed"
        checks.append(row)
    pattern_hits = {p: bool(rx.search(log)) for p, rx in patterns}
    for row in checks:
        if row["status"] == "engaged_by_pattern":
            row["status"] = "engaged" if all(pattern_hits.values()) else "unconfirmed"

    statuses = {r["status"] for r in checks} | (
        {"unconfirmed"}
        if not all(pattern_hits.values()) or (args.require_launch and launch is None)
        else set()
    )
    status = (
        "mismatch"
        if "mismatch" in statuses
        else "unconfirmed"
        if "unconfirmed" in statuses
        else "engaged"
    )
    facts = {
        "mode": args.mode,
        "target": args.target,
        "status": status,
        "identity": ident,
        "checks": checks,
        "log_patterns": pattern_hits,
        "launch_line": launch[:600] if launch else None,
        "engine_fallbacks": fallbacks(log),
        "effective_nim_env": {
            k: mask(k, v) for k, v in sorted(env.items()) if k.startswith("NIM_")
        },
    }
    if status == "engaged":
        emit(True, "engaged", "smoke_test", facts)
    errors = []
    if launch is None:
        errors.append(
            {
                "code": "no_launch_line",
                "message": "no engine launch line in the startup log",
                "fix": "Wait for the NIM to be ready and rerun, or pass --log-file.",
            }
        )
    for r in checks:
        if r["status"] == "mismatch":
            errors.append(
                {
                    "code": "setting_mismatch",
                    "message": f"{r['setting']}: expected {r['expected']}, "
                    f"delivered {r.get('delivered')}, engine {r.get('engine')}",
                    "fix": "Fix the manifest or command so this value is delivered; redeploy.",
                }
            )
        elif r["status"] == "unconfirmed":
            errors.append(
                {
                    "code": "setting_unconfirmed",
                    "message": f"{r['setting']} is not visible in the engine",
                    "fix": "Pass --log-pattern for the engine line showing it; no benchmark yet.",
                }
            )
    for p, hit in pattern_hits.items():
        if not hit:
            errors.append(
                {
                    "code": "pattern_missing",
                    "message": f"log does not match {p!r}",
                    "fix": "Check the pattern against the log; the setting may not have acted.",
                }
            )
    emit(
        False,
        status,
        "fix_candidate" if status == "mismatch" else "find_engine_evidence",
        facts,
        errors,
        EXIT_PRECONDITION,
    )


if __name__ == "__main__":
    main()
