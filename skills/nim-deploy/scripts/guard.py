#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PreToolUse hook: allowlist guard for kubectl, helm, docker and runner commands.

Reads the Claude Code hook JSON on stdin (or `--check COMMAND`). For Bash it splits the command
(shellparse.py) and checks every simple command against guard-rules.json:

- deny rules always block (docker.sock chmod, curl | sh, kubeconfig changes, helm --set, secrets
  on a command line, prune, ...), and so do hidden forms (a program built from $VAR, a shell or
  interpreter fed tool text on stdin, xargs or find -exec running a tool);
- gated rules (kubectl/helm/docker mutations, run scripts) allow one exact form each, and only
  when it is the ONLY command in the Bash call and matches the latest unexpired approval of a run
  (approvals.py): the exact kubectl apply/delete forms, the approved helm command, or a form the
  run's runner declares in guard_forms() (runners/__init__.py). A program that runner starts with
  is gated while the run is approved.
  Dry runs (kubectl --dry-run=server, helm --dry-run) and read-only commands pass.

Allow: exit 0. Deny: exit 2 with the reason on stderr (Claude Code blocks the call). Any internal
error also denies. Both print one JSON object (the write-skill script-output contract).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import approvals as ap
import nd_common as nd
import runners
import shellparse
from shellparse import Command, GuardError

RULES_FILE = Path(__file__).resolve().parent / "guard-rules.json"
FLAGS = r"(?: -[^ ]+(?: [^- ][^ ]*)?)*"
INTERPRETER = re.compile(r"python[0-9.]*|perl|node|nodejs|ruby")
TOOL_HINT = re.compile(
    r"(?<![\w.-])(?:kubectl|helm|docker-compose|docker)(?![\w.-])|docker-run\.sh|helm-cmd\.txt"
)
TOOL_PROGRAMS = {"kubectl", "helm", "docker", "docker-compose"}
MUTATING_WORDS = set(
    [
        "apply",
        "create",
        "replace",
        "patch",
        "delete",
        "scale",
        "run",
        "rm",
        "stop",
        "kill",
        "exec",
        "install",
        "upgrade",
        "uninstall",
        "rollback",
        "update",
        "up",
        "down",
    ]
)
SPAWN = re.compile(
    r"system|subprocess|popen|exec|spawn|child_process|`|Process|pty|run\(|call\(", re.I
)
DYNAMIC_PROGRAM = re.compile(r"[$`*?\[\]{}]|__SUBST_")
NIM_IMAGE = re.compile(r"nvcr\.io/nim/[a-z0-9][a-z0-9._/-]*(?::[A-Za-z0-9_][A-Za-z0-9_.-]{0,127})?")
CONTAINER_CACHE = "/opt/nim/.cache"
SYSTEM_DIRS = [
    "/etc",
    "/bin",
    "/sbin",
    "/usr",
    "/boot",
    "/proc",
    "/sys",
    "/dev",
    "/var/run",
    "/run",
    "/root",
    "/lib",
    "/lib64",
]
APPROVE_FIX = (
    "Fix: run nimdeploy.py approval-summary RUN, show it to the user, and after the user replies "
    "with the token run nimdeploy.py approve RUN --token T. Then run the apply command from "
    "'nimdeploy.py status RUN'."
)
SINGLE_FIX = (
    "Fix: run the apply as one single command, exactly as 'nimdeploy.py status RUN' prints it (no "
    "&&, |, ;, redirection, cd, $-expansion, sudo/env/timeout wrapper or VAR=value prefix). Run "
    "anything else in a separate Bash call."
)
Decision = tuple[str, str, str]  # (allow | deny, rule id, reason)


def load_rules() -> dict[str, Any]:
    rules: dict[str, Any] = json.loads(RULES_FILE.read_text())
    rules["deny"] += runners.deny_rules()  # runner-specific denies live with each runner
    for group in ("deny", "gated"):
        for rule in rules[group]:
            rule["regex"] = re.compile(rule["pattern"].replace("{F}", FLAGS))
    return rules


# --- read-only exemptions -------------------------------------------------------------------


def is_dry_run(tokens: list[str], kind: str) -> bool:
    """A kubectl/helm argv whose last --dry-run asks for a dry run, nothing built at run time."""
    if any(DYNAMIC_PROGRAM.search(token) for token in tokens):
        return False
    options = tokens[: tokens.index("--")] if "--" in tokens else tokens  # after `--`: pod args
    flags = [t for t in options if t == "--dry-run" or t.startswith("--dry-run=")]
    values = [token.partition("=")[2] for token in flags]
    wanted = ("server", "client") if kind == "kubectl" else ("", "server", "client", "true")
    return bool(values) and values[-1] in wanted


def cache_dir_is_safe(host: str) -> bool:
    for prefix in ("~/", "$HOME/", "${HOME}/"):
        if host.startswith(prefix):
            host = str(Path.home() / host[len(prefix) :])
            break
    if not host.startswith("/") or DYNAMIC_PROGRAM.search(host) or "~" in host:
        return False
    path, home = Path(host).resolve(), Path.home().resolve()
    if path in (Path("/"), home) or (path.exists() and not path.is_dir()):
        return False
    # compare the folder as written and as resolved with each system folder in both forms:
    # on macOS /etc resolves to /private/etc, which would otherwise slip past "/etc"
    written = Path(os.path.normpath(host))
    roots = {Path(d) for d in SYSTEM_DIRS} | {Path(d).resolve() for d in SYSTEM_DIRS}
    system = any(p == r or p.is_relative_to(r) for p in (written, path) for r in roots)
    return path.is_relative_to(home) or not system


def is_profile_listing(tokens: list[str]) -> bool:
    """Read-only: docker run --rm [--gpus G] [-e NGC_API_KEY|HF_TOKEN] [-v DIR:/opt/nim/.cache]
    nvcr.io/nim/IMAGE list-model-profiles, with DIR an absolute non-system folder."""
    if tokens[1:2] != ["run"] or len(tokens) < 5 or tokens[-1] != "list-model-profiles":
        return False
    options, image, index, removes = tokens[2:-2], tokens[-2], 0, False
    while index < len(options):
        name, has_value, value = options[index].partition("=")
        index += 1
        if name == "--rm":
            removes = True
            continue
        if name not in ("--gpus", "-e", "--env", "-v", "--volume"):
            return False
        if not has_value:
            if index >= len(options):
                return False
            value, index = options[index], index + 1
        if name in ("-e", "--env") and value not in ("NGC_API_KEY", "HF_TOKEN"):
            return False
        if name in ("-v", "--volume"):
            host, _, rest = value.partition(":")
            mounts = (CONTAINER_CACHE, f"{CONTAINER_CACHE}:ro", f"{CONTAINER_CACHE}:rw")
            if rest not in mounts or not cache_dir_is_safe(host):
                return False
    return removes and bool(NIM_IMAGE.fullmatch(image))


# --- hidden forms ---------------------------------------------------------------------------


def reads_stdin(command: Command) -> bool:
    """A shell or interpreter that runs whatever arrives on stdin (a pipe or a here-string)."""
    reader = command.prog in shellparse.SHELLS or INTERPRETER.fullmatch(command.prog)
    operands = [t for t in command.tokens[1:] if not t.startswith("-")]
    bare = shellparse.command_payload(command.tokens) is None
    bare = bare and interpreter_code(command) is None and not operands
    return bool(reader and (bare or "-s" in command.tokens[1:]))


def stdin_script_problem(pipeline: list[Command], rules: dict[str, Any]) -> str | None:
    """The id of a deny or gated rule that text fed to a stdin shell would trigger."""
    readers = [command for command in pipeline if reads_stdin(command)]
    if not readers:
        return None
    texts = [text for reader in readers for text in reader.stdin_text]
    for command in pipeline:
        if command not in readers:
            texts += [" ".join(command.tokens[1:]), *command.tokens[1:]]
    for line in "\n".join(texts).replace("\\n", "\n").splitlines():
        for start in re.finditer(r"\b\w", line):
            suffix = line[start.start() :]
            for candidate in (suffix, re.sub(r"['\");)}`\\\s]+$", "", suffix)):
                for rule in rules["deny"] + rules["gated"]:
                    if rule["regex"].search(candidate.strip()):
                        return str(rule["id"])
    return None


def interpreter_code(command: Command) -> str | None:
    if not INTERPRETER.fullmatch(command.prog):
        return None
    for index, token in enumerate(command.tokens[1:], 1):
        if token in ("--eval", "--print") or re.fullmatch(r"-[a-zA-Z]*[ceEp]", token):
            return command.tokens[index + 1] if index + 1 < len(command.tokens) else ""
        if not token.startswith("-"):
            return None  # a script path: its arguments are not code
    return None


def hidden_problem(command: Command, raw: str) -> str | None:
    """A deploy tool run in a way the guard cannot see."""
    word, rest = command.tokens[0], command.tokens[1:]
    operands = [t for t in rest if not t.startswith("-")]
    dynamic = word not in ("[", "[[") and DYNAMIC_PROGRAM.search(word)
    if dynamic and (TOOL_HINT.search(raw) or (operands[:1] and operands[0] in MUTATING_WORDS)):
        return (
            f"the program word {word!r} is built at run time, so the guard cannot see which tool "
            "runs. Fix: name the tool literally (kubectl, helm, docker)."
        )
    code = interpreter_code(command)
    if code is not None and TOOL_HINT.search(code) and SPAWN.search(code):
        return (
            f"{command.prog} code runs a deploy tool the guard cannot check. "
            "Fix: run the kubectl/helm/docker command itself in Bash."
        )
    reader = command.prog in shellparse.SHELLS or INTERPRETER.fullmatch(command.prog)
    no_script = shellparse.command_payload(command.tokens) is None and code is None and not operands
    if reader and TOOL_HINT.search(raw) and (no_script or "-s" in rest):
        return (
            f"{command.prog} reads its commands from a pipe, here-doc or file, so the guard "
            "cannot check them. Fix: run the command itself in Bash."
        )
    if (
        "xargs" in command.wrappers
        and command.prog in TOOL_PROGRAMS | shellparse.SHELLS
        and not operands
    ):
        return (
            f"xargs runs {command.prog} with arguments from stdin, so the guard cannot see the "
            f"verb. Fix: run {command.prog} directly with all of its arguments."
        )
    exec_flags = ("-exec", "-execdir", "-ok", "-okdir")
    if command.prog == "find" and any(t in exec_flags for t in rest) and TOOL_HINT.search(raw):
        return "find -exec runs a deploy tool the guard cannot check. Fix: run the tool in Bash."
    return None


# --- gating ---------------------------------------------------------------------------------


def script_in_runs(command: Command, cwd: Path) -> bool:
    """True when the shell script this command runs lives under a .nim-deploy/runs folder."""
    if command.prog in shellparse.SHELLS or command.tokens[0] in ("source", "."):
        if shellparse.command_payload(command.tokens) is not None:
            return False
        args = [t for t in command.tokens[1:] if not t.startswith("-")]
        candidate = args[0] if args else None
    else:
        candidate = command.tokens[0] if "/" in command.tokens[0] else None
    if not candidate:
        return False
    path = (cwd / Path(candidate).expanduser()).resolve()
    return any(path.is_relative_to((root / "runs").resolve()) for root in ap.runs_roots(cwd))


def read_only_runner_command(command: Command, extra: dict[str, list[Any]]) -> bool:
    """True when a runner declares this command of its program read-only (nim list, nim info)."""
    return any(
        getattr(runner, "read_only", lambda argv: False)(command.tokens)
        for runner in extra.get(command.prog, [])
    )


def gated_rule(
    command: Command, rules: dict[str, Any], cwd: Path, extra: dict[str, list[Any]]
) -> dict[str, Any] | None:
    rule: dict[str, Any] | None = next(
        (r for r in rules["gated"] if r["regex"].search(command.norm)), None
    )
    if rule is None:
        if command.prog in extra and not read_only_runner_command(command, extra):
            return {"id": "runner-command", "kind": "runner"}
        return None
    kind = rule["kind"]
    if (
        kind in ("kubectl", "helm")
        and "xargs" not in command.wrappers
        and is_dry_run(command.tokens, kind)
    ):
        return None
    if kind == "script" and not script_in_runs(command, cwd):
        return None
    if kind == "docker" and is_profile_listing(command.tokens):
        return None
    return rule


def evaluate(command_line: str, cwd: Path, now: datetime) -> Decision:
    """Decide for a Bash command: (decision, rule id, reason)."""
    rules = load_rules()
    try:
        pipelines = shellparse.parse(command_line)
    except GuardError as exc:
        reason = (
            f"[unparseable] {exc}. Fix: write the command as plain shell without unbalanced quotes."
        )
        return "deny", "unparseable", reason
    commands = [command for pipeline in pipelines for command in pipeline]
    for pipeline in pipelines:
        texts = [c.norm for c in pipeline]
        texts += [" | ".join(texts)] if len(pipeline) > 1 else []
        for rule in rules["deny"]:
            if any(rule["regex"].search(text) for text in texts):
                return "deny", rule["id"], f"[{rule['id']}] {rule['reason']}"
    for pipeline in pipelines:
        rule_id = stdin_script_problem(pipeline, rules)
        if rule_id:
            reason = (
                f"[{rule_id}] a shell reads this command from stdin, so the guard would not see "
                f"it as a command, and the text matches the rule {rule_id}. Fix: run the command "
                "itself in Bash (and only if the rules allow it)."
            )
            return "deny", rule_id, reason
    for command in commands:
        problem = hidden_problem(command, command_line)
        if problem:
            return "deny", "hidden-command", f"[hidden-command] {problem}"
    approvals = ap.active_approvals(cwd, now)
    extra = ap.runner_programs(approvals)
    gated = [(c, rule) for c in commands if (rule := gated_rule(c, rules, cwd, extra))]
    if not gated:
        return "allow", "", ""
    command, rule = gated[0]
    rule_id, kind = rule["id"], rule["kind"]
    head = f"[{rule_id}] '{command.norm[:80]}' changes a deployment"
    argv, problem = ap.single_command(command_line)
    if problem is None and (
        len(commands) != 1 or command.wrappers or command.env_names - ap.SAFE_ENV
    ):
        problem = "it runs through a wrapper or alongside other commands"
    if problem:
        return (
            "deny",
            rule_id,
            f"{head}, so it must be the only command in the Bash call, but {problem}. {SINGLE_FIX}",
        )
    if not approvals:
        return "deny", rule_id, f"{head} and no run has an unexpired approval. {APPROVE_FIX}"
    targets = ap.TARGET_KINDS[kind]
    matching = [a for a in approvals if a.target in targets]
    if not matching:
        have = ", ".join(sorted({str(a.target) for a in approvals}))
        return (
            "deny",
            rule_id,
            f"{head}, but no approved run has target {' or '.join(targets)} "
            f"(approved: {have}). {APPROVE_FIX}",
        )
    problems = [ap.CHECKS[kind](argv, cwd, approval) for approval in matching]
    if None in problems:
        return "allow", rule_id, ""
    run_id = matching[0].state["run_id"]
    return (
        "deny",
        rule_id,
        (
            f"{head} but does not match the approved run {run_id}: {problems[0]}. Use the exact "
            "command from 'nimdeploy.py status RUN'; if the plan changed, re-render and ask the "
            "user again."
        ),
    )


def decide(raw: str) -> Decision:
    """Decide for one hook payload; anything unexpected denies."""
    try:
        payload = json.loads(raw)
        tool, tool_input = payload["tool_name"], payload["tool_input"]
        cwd = Path(payload.get("cwd") or Path.cwd())
    except (ValueError, KeyError, TypeError) as exc:
        return (
            "deny",
            "malformed-input",
            f"[malformed-input] the guard could not read the hook input ({exc}).",
        )
    if tool != "Bash":
        return "allow", "", ""
    command = tool_input.get("command") if isinstance(tool_input, dict) else None
    if not isinstance(command, str):
        return "deny", "malformed-input", "[malformed-input] Bash hook input has no command string."
    return evaluate(command, cwd, nd.utc_now())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: python3 scripts/guard.py --check 'kubectl get pods --context c -n ns'",
    )
    parser.add_argument(
        "--check",
        metavar="COMMAND",
        help="decide for COMMAND instead of reading the hook JSON on stdin",
    )
    parser.add_argument(
        "--cwd", default=None, help="folder the command runs in (default: the current folder)"
    )
    args = parser.parse_args(argv)
    try:
        if args.check is not None:
            decision, rule_id, reason = evaluate(
                args.check, Path(args.cwd or Path.cwd()), nd.utc_now()
            )
        else:
            decision, rule_id, reason = decide(sys.stdin.read())
    except Exception as exc:  # fail closed on any bug
        reason = (
            f"[guard-error] the guard failed ({type(exc).__name__}: {exc}); "
            "ask the user to run the command themselves."
        )
        decision, rule_id = "deny", "guard-error"
    allowed = decision != "deny"
    reason = nd.absolute_scripts(reason)  # the hook text works from any folder and install
    errors = (
        []
        if allowed
        else [
            nd.error(
                rule_id.replace("-", "_"),
                reason,
                "Follow the Fix in the message, then run the command again",
            )
        ]
    )
    payload = {
        "ok": allowed,
        "state": decision,
        "next": "run_command" if allowed else "fix_errors",
        "facts": {"rule": rule_id},
        "errors": errors,
    }
    if not allowed:
        print(reason, file=sys.stderr)
    sys.stdout.write(json.dumps(payload) + "\n")
    if not allowed:
        sys.exit(2)  # Claude Code blocks a PreToolUse hook that exits 2


if __name__ == "__main__":
    main()
