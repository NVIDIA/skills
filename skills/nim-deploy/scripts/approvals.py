#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Approved command forms for guard.py: find the approved runs and check one command against them.

A command is allowed only when it is the one plain command in the Bash call (`single_command`) and
matches the latest unexpired approval of a run of the right target: the exact kubectl apply/delete
forms, the approved helm command, or a form the run's runner declares (runners/__init__.py).
`python3 scripts/approvals.py` lists the runs that currently hold an approval.
Prints one JSON object. Exit code: 0.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nd_common as nd
import runners
import shellparse

ACTIVE_STATES = ("apply", "wait_ready", "verify", "diagnose", "rollback")
SAFE_ENV = {"NO_COLOR", "TERM"}
KUBECTL_EXTRAS = {
    "--server-side",
    "--field-manager=nim-deploy",
    "--dry-run=server",
    "--force-conflicts",
}
HELM_DURATION = re.compile(r"(?:[0-9]+(?:ms|s|m|h))+")
TARGET_KINDS = {
    "kubectl": tuple(nd.MANIFESTS),
    "helm": ("helm",),
    "docker": ("docker",),
    "script": ("docker",),
    "runner": ("docker",),
}
CHANGED = "{path} changed after approval; re-render, re-validate and get a new approval"


@dataclass
class Approval:
    state: dict[str, Any]
    record: dict[str, Any]
    folder: Path

    @property
    def target(self) -> str | None:
        return self.record.get("target")

    def artifacts_problem(self) -> str | None:
        """None when every approved artifact still hashes as approved."""
        for path, wanted in (self.record.get("artifacts") or {}).items():
            if not Path(path).is_file() or nd.sha256_file(path) != wanted:
                return CHANGED.format(path=path)
        return None

    def artifact_problem(self, value: str, cwd: Path) -> str | None:
        """None when VALUE (relative to CWD) is an approved artifact whose bytes still hash."""
        if value in ("", "-") or "://" in value:
            return f"file argument {value!r} cannot be verified; pass the approved file path"
        path = (cwd / Path(value).expanduser()).resolve()
        wanted = (self.record.get("artifacts") or {}).get(str(path))
        if wanted is None:
            return f"{path} is not an approved artifact of run {self.state.get('run_id')}"
        if not path.is_file() or nd.sha256_file(path) != wanted:
            return CHANGED.format(path=path)
        return None


# --- finding approvals ----------------------------------------------------------------------


def runs_roots(cwd: Path) -> list[Path]:
    """Existing .nim-deploy roots; never creates one."""
    if os.environ.get("NIM_DEPLOY_HOME"):
        return [Path(os.environ["NIM_DEPLOY_HOME"])]
    nearest = [f / ".nim-deploy" for f in (cwd, *cwd.parents) if (f / ".nim-deploy").is_dir()]
    project = os.environ.get("CLAUDE_PROJECT_DIR")
    candidates = [nd.default_runs_root(cwd), *nearest[:1]]
    candidates += [Path(project) / ".nim-deploy"] if project else []
    roots: list[Path] = []
    for root in candidates:
        if root.is_dir() and root.resolve() not in [r.resolve() for r in roots]:
            roots.append(root)
    return roots


def approval_expiry(record: Any) -> datetime | None:
    try:
        expires = datetime.fromisoformat(str(record["expires_at"]).replace("Z", "+00:00"))
    except (KeyError, TypeError, ValueError):
        return None
    return expires if expires.tzinfo else expires.replace(tzinfo=timezone.utc)  # noqa: UP017


def active_approvals(cwd: Path, now: datetime) -> list[Approval]:
    approvals = []
    for root in runs_roots(cwd):
        for path in sorted((root / "runs").glob("*/state.json")):
            try:
                state = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if state.get("state") not in ACTIVE_STATES or not state.get("approvals"):
                continue
            record = state["approvals"][-1]
            expires = approval_expiry(record)
            if expires is not None and expires > now:
                approvals.append(Approval(state, record, path.parent))
    return approvals


def runner_programs(approvals: list[Approval]) -> dict[str, list[ModuleType]]:
    """Programs the approved runs' runners start with (other than shells), with those runners.

    Such a program is gated while the run is approved, except for the commands its runner calls
    read-only (the optional `read_only(argv)` function of the runner)."""
    found: dict[str, list[ModuleType]] = {}
    for approval in approvals:
        if approval.target == "docker":
            runner = runners.load(approval.record.get("runner"))
            for form in runner.guard_forms(approval):
                if form and Path(form[0]).name not in shellparse.SHELLS:
                    found.setdefault(Path(form[0]).name, []).append(runner)
    return found


# --- the single-command allowlist -----------------------------------------------------------


def single_command(text: str) -> tuple[list[str], str | None]:
    """(argv, None) when TEXT is one plain simple command, else ([], why not)."""
    text = text.replace("\\\n", "").strip()
    quote, index = None, 0
    while index < len(text):
        char = text[index]
        if quote == "'":
            quote = None if char == "'" else quote
        elif char == "\\":
            index += 1
        elif char == '"':
            quote = None if quote == '"' else '"'
        elif char == "'" and quote is None:
            quote = "'"
        elif char == "`":
            return [], "it has a backtick substitution"
        elif char == "$":
            return [], "it has a $-expansion the guard cannot resolve"
        elif quote is None and char in ";&|<>()\n#{}*?[]":
            what = {"\n": "a newline", "#": "a comment"}.get(char, repr(char))
            return [], f"it has {what} outside quotes (several commands, a redirection or a glob)"
        index += 1
    if quote:
        return [], "it has an unterminated quote"
    try:
        argv = shlex.split(text)
    except ValueError as exc:
        return [], f"it does not parse ({exc})"
    while argv and shellparse.ASSIGNMENT.match(argv[0]):
        name = argv[0].split("=", 1)[0]
        if name not in SAFE_ENV:
            return [], f"it sets {name} (only NO_COLOR and TERM may be set)"
        argv = argv[1:]
    return (argv, None) if argv else ([], "it is empty")


def take_options(
    args: list[str], value_names: dict[str, str], extras: set[str] | frozenset[str] = frozenset()
) -> tuple[dict[str, list[str]], list[str], str | None]:
    """Split ARGS into {key: [values]} for VALUE_NAMES (flag -> key) and positionals. Any other
    flag is a problem, except the exact EXTRAS tokens."""
    found: dict[str, list[str]] = {key: [] for key in set(value_names.values())}
    words: list[str] = []
    index = 0
    while index < len(args):
        token = args[index]
        index += 1
        if token in extras:
            continue
        if not token.startswith("-"):
            words.append(token)
            continue
        name, has_value, value = token.partition("=")
        if name not in value_names:
            return found, words, f"{token!r} is not part of the approved form"
        if not has_value:
            if index >= len(args):
                return found, words, f"{name} needs a value"
            value, index = args[index], index + 1
        found[value_names[name]].append(value)
    return found, words, None


def scope_problem(found: dict[str, list[str]], approval: Approval, context_flag: str) -> str | None:
    for key, flag in (("namespace", "-n"), ("context", context_flag)):
        wanted = approval.record.get(key)
        if found[key] != [wanted]:
            return f"needs exactly '{flag} {wanted}' once (got {found[key] or 'none'})"
    return None


def check_kubectl(argv: list[str], cwd: Path, approval: Approval) -> str | None:
    if argv[0] != "kubectl" or argv[1:2] not in (["apply"], ["delete"]):
        return (
            "the only approved kubectl forms are 'kubectl apply -f FILE --context CTX -n NS' and "
            "'kubectl delete -f FILE --context CTX -n NS' (verb right after kubectl)"
        )
    names = {
        "-f": "files",
        "--filename": "files",
        "-n": "namespace",
        "--namespace": "namespace",
        "--context": "context",
    }
    found, words, problem = take_options(argv[2:], names, KUBECTL_EXTRAS)
    if problem or words:
        return problem or f"unexpected arguments {words}"
    if not found["files"]:
        return "needs -f with the approved file"
    problems = (approval.artifact_problem(value, cwd) for value in found["files"])
    return scope_problem(found, approval, "--context") or next((p for p in problems if p), None)


def helm_extras(argv: list[str]) -> tuple[list[str], str | None]:
    """Drop the optional --dry-run[=client|server], --wait and --timeout DUR from a helm argv."""
    kept: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        index += 1
        if token in ("--wait", "--dry-run", "--dry-run=client", "--dry-run=server"):
            continue
        if token == "--timeout" or token.startswith("--timeout="):
            if "=" in token:
                value = token.split("=", 1)[1]
            else:
                value, index = (argv[index] if index < len(argv) else ""), index + 1
            if not HELM_DURATION.fullmatch(value):
                return kept, f"--timeout {value!r} is not a duration like 10m"
            continue
        kept.append(token)
    return kept, None


def check_helm(argv: list[str], cwd: Path, approval: Approval) -> str | None:
    verb = argv[1] if len(argv) > 1 else ""
    if argv[0] != "helm" or verb not in ("upgrade", "install", "uninstall"):
        return (
            "the only approved helm forms are the command in the run's helm-cmd.txt and "
            "'helm uninstall RELEASE -n NS --kube-context CTX'"
        )
    argv, problem = helm_extras(argv)
    if problem:
        return problem
    if verb == "uninstall":
        names = {"-n": "namespace", "--namespace": "namespace", "--kube-context": "context"}
        found, words, problem = take_options(argv[2:], names)
        if problem:
            return problem
        if words != [approval.record.get("release")]:
            return f"must uninstall exactly the approved release {approval.record.get('release')!r}"
        return scope_problem(found, approval, "--kube-context")
    command_file = approval.folder / "helm-cmd.txt"
    problem = approval.artifact_problem(str(command_file), cwd)
    if problem:
        return problem
    expected = shlex.split(command_file.read_text())
    if len(argv) != len(expected):
        return "differs from the approved helm-cmd.txt (extra or missing arguments)"
    for index, (given, wanted) in enumerate(zip(argv, expected, strict=True)):
        if index and expected[index - 1] in ("-f", "--values"):
            approved = approval.folder / wanted
            if (cwd / Path(given).expanduser()).resolve() != approved.resolve():
                return f"-f {given} is not the approved {approved}"
            problem = approval.artifact_problem(str(approved), cwd)
            if problem:
                return problem
        elif given != wanted:
            return f"{given!r} differs from the approved helm-cmd.txt ({wanted!r})"
    return None


def resolved(argv: list[str], cwd: Path) -> list[str]:
    """ARGV with path-like words resolved, so symlinked folders compare equal."""
    return [str((cwd / w).resolve()) if "/" in w and not w.startswith("-") else w for w in argv]


def check_runner(argv: list[str], cwd: Path, approval: Approval) -> str | None:
    """The command must be exactly one of the forms the approved run's runner declares."""
    forms = runners.load(approval.record.get("runner")).guard_forms(approval)
    if resolved(argv, cwd) not in [resolved(list(form), cwd) for form in forms]:
        allowed = "; ".join(shlex.join(form) for form in forms)
        return f"is not one of the forms approved for this run ({allowed})"
    return approval.artifacts_problem()


CHECKS = {
    "kubectl": check_kubectl,
    "helm": check_helm,
    "docker": check_runner,
    "script": check_runner,
    "runner": check_runner,
}


def main(argv: list[str] | None = None) -> None:
    parser = nd.JsonArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: python3 scripts/approvals.py",
    )
    parser.parse_args(argv)
    now = nd.utc_now()
    held = [
        {"run_id": a.state["run_id"], "target": a.target, "expires_at": a.record["expires_at"]}
        for a in active_approvals(Path.cwd(), now)
    ]
    nd.emit(True, "approvals", "done" if held else "ask_user", {"approved_runs": held})


if __name__ == "__main__":
    main()
