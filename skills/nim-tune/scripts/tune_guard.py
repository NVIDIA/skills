#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""guard: PreToolUse hook that enforces nim-tune's must-never rules on Bash commands.

Denied (exit 2, reason on stderr):
  - a kubectl, helm or docker command that changes a cluster or host (apply, create, delete,
    patch, run, rm, ...): only run_campaign.py changes the target, after an approved plan;
  - a live run_campaign.py without --approved PLAN_ID (the dry run prints the id);
  - a credential value on a command line (--api-key X, nvapi-..., hf_..., NAME_KEY=value).
Commands behind a wrapper (sudo -u x, timeout 60, env, xargs, ...), inside sh -c '...', ssh HOST
'...', eval or $(...) are checked as well. Only a real dry run passes: kubectl --dry-run=server
or client, helm --dry-run; --dry-run=none or false is a live change.
Read-only commands, dry runs and everything else pass (exit 0).

Claude Code passes the tool call as JSON on stdin ({"tool_input": {"command": ...}}).
`--check COMMAND` prints the decision as one JSON object instead: {"ok", "state", "next",
"facts", "errors"}. Exit codes: 0 allowed, 2 denied. Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path
from typing import NoReturn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from targets import SECRET_FLAG, SECRET_NAME, TOKEN_SHAPES

MUTATING = {
    "kubectl": {
        "apply", "create", "delete", "patch", "replace", "scale", "edit", "label", "annotate",
        "set", "drain", "cordon", "uncordon", "taint", "run", "expose", "autoscale",
        "debug", "certificate",
    },
    "helm": {"install", "upgrade", "uninstall", "delete", "rollback", "test"},
    "docker": {
        "run", "rm", "rmi", "stop", "kill", "restart", "create", "start", "update", "pull",
        "push", "build", "tag", "load", "import", "commit", "cp", "pause", "unpause", "rename",
        "prune",
    },
}  # fmt: skip
# docker management groups (docker container rm x, docker image prune): a change unless the
# group's subcommand only reads
DOCKER_GROUPS = {
    "container", "image", "network", "volume", "system", "builder", "buildx", "compose",
    "plugin", "swarm", "service", "stack", "node", "secret", "config", "context", "manifest",
    "trust",
}  # fmt: skip
DOCKER_READS = {
    "ls", "list", "inspect", "ps", "logs", "top", "stats", "port", "diff", "history", "df",
    "events", "info", "version", "images", "show",
}  # fmt: skip
# kubectl rollout: only these subcommands change anything (status and history only read)
ROLLOUT_CHANGES = {"restart", "undo", "pause", "resume"}
# how many leading positionals may hold the verb: a global flag we do not know ("--as admin",
# "-v 6") that takes a value leaves its value in front of the verb
VERB_WINDOW = 3
# read verbs whose next positionals are resource names, never a subcommand: after one of these a
# word like "delete" is a name (`kubectl get pod delete`). Any other first verb, including a
# group such as `helm plugin install`, keeps the full window.
NAMED_READS = {
    "kubectl": {"get", "describe", "logs", "explain", "top", "events"},
    "helm": {"status", "get", "history", "show", "template", "list", "ls"},
    "docker": {"logs", "inspect", "ps", "port", "diff", "stats", "top", "history", "images"},
}
# flags that take a value: their value is never the command's verb, nor a dry-run option
VALUE_FLAGS = {
    "-n", "--namespace", "--context", "--kubeconfig", "--cluster", "--user", "-s", "--server",
    "--kube-context", "-H", "--host", "--config", "--log-level", "-c", "--request-timeout",
    "-f", "--filename", "-o", "--output", "-l", "--selector", "-p", "--patch", "--values",
    "--as", "--as-group", "--as-uid", "--token", "--certificate-authority", "--client-certificate",
    "--client-key", "--tls-server-name", "--username", "--cache-dir", "--tlscacert", "--tlscert",
    "--tlskey", "--repository-config", "--repository-cache", "--registry-config",
    "--kube-apiserver", "--kube-token", "--kube-as-user", "--kube-as-group", "--kube-ca-file",
}  # fmt: skip
# the dry-run values that change nothing; anything else (none, false, ...) is a live change
DRY_RUN = {
    "kubectl": {"--dry-run", "--dry-run=server", "--dry-run=client"},
    "helm": {"--dry-run", "--dry-run=true", "--dry-run=client", "--dry-run=server"},
    "docker": set(),
}
# commands that run another command: options that take a value, and positionals before it
WRAPPERS: dict[str, tuple[set[str], int]] = {
    "sudo": ({"-u", "--user", "-g", "--group", "-h", "--host", "-p", "--prompt", "-C",
              "--close-from", "-D", "--chdir", "-r", "--role", "-t", "--type", "-T",
              "--command-timeout", "-U", "--other-user"}, 0),
    "env": ({"-u", "--unset", "-C", "--chdir"}, 0),
    "timeout": ({"-s", "--signal", "-k", "--kill-after"}, 1),  # timeout [opts] DURATION CMD
    "nice": ({"-n", "--adjustment"}, 0),
    "ionice": ({"-c", "--class", "-n", "--classdata", "-p", "--pid"}, 0),
    "stdbuf": ({"-i", "-o", "-e"}, 0),
    "watch": ({"-n", "--interval"}, 0),
    "xargs": ({"-a", "-d", "-E", "-I", "-L", "-n", "-P", "-s", "--arg-file", "--delimiter",
               "--max-args", "--max-procs", "--replace"}, 0),
    "nohup": (set(), 0), "command": (set(), 0), "exec": (set(), 0), "time": (set(), 0),
    "builtin": (set(), 0), "doas": ({"-u", "-C"}, 0), "runuser": ({"-u", "-g"}, 0),
}  # fmt: skip
SHELLS = {"sh", "bash", "zsh", "dash", "ksh", "fish"}
# ssh options that take a value; the first other word is the host, the rest the remote command
SSH_VALUE = {
    "-b", "-c", "-D", "-E", "-e", "-F", "-I", "-i", "-J", "-L", "-l", "-m", "-O", "-o", "-p",
    "-Q", "-R", "-S", "-W", "-w", "-B", "-P",
}  # fmt: skip
ASSIGN = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", re.DOTALL)
SEPARATORS = {";", ";;", "&", "&&", "|", "||", "|&", "(", ")", "`", "\n"}
MAX_DEPTH = 6


def tokens(command: str) -> list[str]:
    """Shell words and operators of a command line; quotes respected, comments dropped."""
    lx = shlex.shlex(command, posix=True, punctuation_chars=";&|()<>`\n")
    lx.whitespace = " \t\r"
    lx.whitespace_split = True
    return list(lx)


def segments(command: str) -> list[list[str]]:
    """Simple commands of a shell line, split on ; & && || | ( ) ` and newlines.

    A line shlex cannot read (an unbalanced quote) is split on whitespace instead, which is
    stricter: quoted text is then checked as if it were a command.
    """
    try:
        words = tokens(command)
    except ValueError:
        words = re.split(r"\s+|(?=[;&|()`])|(?<=[;&|()`])", command)
        words = [w.strip("'\"") for w in words if w]
    out: list[list[str]] = [[]]
    for w in words:
        if w in SEPARATORS or set(w) <= set(";&|()<>`\n"):
            out.append([])
        else:
            out[-1].append(w)
    return [s for s in out if s]


def unwrap(words: list[str]) -> list[str]:
    """The command a wrapper runs: `sudo -u root kubectl x` -> `kubectl x`."""
    while words:
        if ASSIGN.match(words[0]):
            words = words[1:]
            continue
        prog = Path(words[0]).name
        if prog not in WRAPPERS:
            return words
        values, positionals = WRAPPERS[prog]
        i = 1
        # options and env A=1 assignments, in any order, up to the command or "--"
        while i < len(words) and (words[i].startswith("-") or ASSIGN.match(words[i])):
            if words[i] == "--":
                i += 1
                break
            i += 2 if words[i] in values else 1
        words = words[i + positionals :]
    return words


def inner_commands(words: list[str]) -> list[str]:
    """Command strings a segment runs elsewhere: sh -c, ssh HOST, eval, env -S, $(...)."""
    prog = Path(words[0]).name
    inner = [w[w.index("$(") + 2 :] for w in words if "$(" in w]  # $(...) inside a quoted word
    if re.search(r"\s", words[0]):  # watch 'CMD': a whole command line as one word
        inner.append(" ".join(words))
    elif prog in SHELLS:
        for i, w in enumerate(words[1:], 1):
            if re.fullmatch(r"-[a-zA-Z]*c[a-zA-Z]*", w) and i + 1 < len(words):
                inner.append(words[i + 1])  # sh -c 'CMD', bash -lc 'CMD'
                break
    elif prog in ("ssh", "eval", "su"):
        if prog == "su":
            rest = [words[i + 1] for i, w in enumerate(words[:-1]) if w in ("-c", "--command")]
        elif prog == "eval":
            rest = [" ".join(words[1:])]
        else:
            i = 1
            while i < len(words) and words[i].startswith("-"):
                i += 2 if words[i] in SSH_VALUE else 1
            rest = [" ".join(words[i + 1 :])] if i + 1 < len(words) else []
        inner += rest
    elif prog == "env":
        inner += [words[i + 1] for i, w in enumerate(words[:-1]) if w in ("-S", "--split-string")]
    return inner


def changing_action(prog: str, verbs: list[str], known_verb: bool = False) -> list[str]:
    """The words of a changing kubectl, helm or docker verb among the first positionals, or [].

    The window is wider than one word on purpose: a value flag that is not in VALUE_FLAGS
    (`kubectl --as admin delete po x`) puts its value before the verb, and the guard fails closed.
    With `known_verb` (no unknown option before the first positional) the first positional is the
    verb; when it is a read verb in NAMED_READS, a resource named like a verb
    (`kubectl get pod delete`) is not read as one.
    """
    if known_verb and verbs and verbs[0] in NAMED_READS.get(prog, ()):
        return []
    for i, word in enumerate(verbs[:VERB_WINDOW]):
        sub = verbs[i + 1] if i + 1 < len(verbs) else ""
        if prog == "docker" and word in DOCKER_GROUPS:
            if sub and sub not in DOCKER_READS and (word, sub) != ("compose", "config"):
                return [word, sub]
        elif prog == "kubectl" and (word, sub) == ("auth", "reconcile"):  # writes RBAC objects
            return [word, sub]
        elif prog == "kubectl" and word == "rollout":
            # a flag we do not know can put its value between "rollout" and the subcommand
            # (`rollout --as admin restart`), so a changing subcommand anywhere after it counts
            later = next((w for w in verbs[i + 1 :] if w in ROLLOUT_CHANGES), None)
            if later:
                return [word, later]
        elif word in MUTATING.get(prog, ()):
            return [word]
    return []


def check_segment(words: list[str], depth: int) -> str | None:
    """The reason to deny one simple command, or None."""
    run = unwrap(words)
    # the segment and the command its wrappers run: sudo sh -c 'CMD', timeout 60 ssh HOST 'CMD'
    for form in (words, run) if run != words else (words,):
        for nested in inner_commands(form) if form else []:
            reason = decide(nested, depth + 1)
            if reason:
                return reason
    words = run
    if not words:
        return None
    prog = Path(words[0]).name
    if SECRET_FLAG.search(" ".join(words)):
        return "a secret flag carries a value. Fix: pass the secret by reference."
    verbs: list[str] = []
    options: list[str] = []
    skip = False
    known_verb = True  # until an option we cannot size comes before the first positional
    for w in words[1:]:
        if skip:
            skip = False
        elif w.startswith("-"):
            options.append(w)
            skip = w in VALUE_FLAGS  # "-n x": x is the flag's value
            if not verbs and not skip and "=" not in w:
                known_verb = False  # "-v 6": its value may be the first positional
        else:
            verbs.append(w)
    # any word, not only the parsed options: a value-taking flag before it (`--output-dir -n
    # --post-renderer x`, where helm reads -n as the folder) must not hide it
    if prog == "helm" and any(w.split("=", 1)[0] == "--post-renderer" for w in words[1:]):
        # helm runs that program on the rendered manifests, even for template or --dry-run
        return (
            "`helm --post-renderer` runs a program that can change the target. Fix: change it "
            "through run_campaign.py with an approved plan, or ask the user to run it."
        )
    action = changing_action(prog, verbs, known_verb)
    if action:
        if any(o in DRY_RUN[prog] for o in options):
            return None
        return (
            f"`{prog} {' '.join(action)}` changes the target directly. Fix: change it through "
            "run_campaign.py with an approved plan, or ask the user to run it."
        )
    if any(w.endswith("run_campaign.py") for w in words):
        approved = "--approved" in words or any(w.startswith("--approved=") for w in words)
        if "--dry-run" not in words and not approved:
            return (
                "a live campaign needs an approved plan. Fix: run --dry-run, show the plan "
                "to the user, and after their yes add --approved PLAN_ID."
            )
    return None


def decide(command: str, depth: int = 0) -> str | None:
    """The reason to deny `command`, or None to allow it."""
    if depth > MAX_DEPTH:
        return "the command nests shells too deeply to check. Fix: run the inner command."
    if TOKEN_SHAPES.search(command):
        return "a credential is on the command line. Fix: reference it by name (env var, secret)."
    for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)=(\S+)", command):
        if SECRET_NAME.search(m[1]) and not m[2].startswith("$"):
            return f"{m[1]} is set to a literal value. Fix: export it outside the command."
    for words in segments(command):
        reason = check_segment(words, depth)
        if reason:
            return reason
    return None


def emit(reason: str | None) -> NoReturn:
    """Print the decision as JSON and exit with 0 (allowed) or 2 (denied)."""
    payload: dict[str, object] = {
        "ok": reason is None,
        "state": "allowed" if reason is None else "denied",
        "next": "run" if reason is None else "fix_errors",
        "facts": {},
        "errors": [] if reason is None else [{"code": "denied", "message": reason, "fix": reason}],
    }
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
    raise SystemExit(0 if reason is None else 2)


def main() -> None:
    """Decide one command: from --check, or from the hook JSON on stdin; exit 2 to deny.

    Fails closed: unreadable input or any unexpected error exits 2, because Claude Code lets
    the command run after any other failing exit code (1 is only a non-blocking error). A Bash
    event without a command string is denied; a valid event of another tool is allowed.
    """
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: python3 scripts/tune_guard.py --check 'kubectl delete nimservice x'",
    )
    p.add_argument("--check", metavar="COMMAND", help="decide one command and print JSON")
    a = p.parse_args()
    try:
        if a.check is not None:
            reason = decide(a.check)
        else:
            call = json.load(sys.stdin)
            tool_input = call.get("tool_input", {})
            command = tool_input.get("command")
            if call.get("tool_name") == "Bash" or "command" in tool_input:
                # a Bash event must carry a command to check: never coerce one into text
                if not isinstance(command, str) or not command.strip():
                    raise ValueError("the Bash event has no command string")
                reason = decide(command)
            else:
                reason = None  # another tool
    except Exception as exc:  # fail closed: a crash must never let a command through
        sys.stderr.write(
            f"nim-tune guard: cannot decide ({type(exc).__name__}: {exc}); the command is "
            "blocked. Fix: send the hook JSON on stdin, or run the command yourself.\n"
        )
        raise SystemExit(2) from None
    if a.check is not None:
        emit(reason)
    if reason:
        sys.stderr.write(f"nim-tune guard: {reason}\n")
        raise SystemExit(2)
    raise SystemExit(0)


if __name__ == "__main__":
    main()
