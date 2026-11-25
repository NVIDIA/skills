#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""bench_guard: a PreToolUse hook that keeps benchmark load behind the plan id.

It denies a shell command that starts AIPerf, GenAI-Perf or NeMo Gym directly: `aiperf`,
`gym`, `python -m aiperf`, and a `docker run`, `kubectl exec` or `ssh` that starts them, also
behind `sudo`, `env`, `xargs`, `uvx`, `uv run` or `pipx run`, inside `sh -c`, `eval`, `$(...)` or
backticks, and a shell that reads a line naming a tool on stdin (`echo aiperf | sh`,
`bash <<<'aiperf'`). `command -v aiperf` passes. It also denies a credential on a command line:
a token shape, or a key flag with a value instead of a variable reference. The one way to
send load is `scripts/nimbench.py`, which needs the approved plan id. Everything else passes.

Claude Code passes the tool call as JSON on stdin ({"tool_input": {"command": ...}}).
`--check COMMAND` prints the decision as one JSON object instead: {"ok", "state", "next",
"facts", "errors"}. Exit codes: 0 allowed, 2 denied. Never prompts. Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nb_common import SECRET_FLAG, SECRET_NAME, TOKEN_SHAPES, ContractParser, emit, error

TOOLS = {"aiperf", "genai-perf", "gym", "ng", "nemo-gym", "nemo_gym"}
TOOL_MODULES = {"aiperf", "genai_perf", "nemo_gym"}
TOOL_WORD = re.compile(r"(?:^|[/:_-])(?:aiperf|genai-perf|nemo-gym|nemo_gym)(?:[:@/_-]|$)", re.I)
WRAPPERS = {
    "sudo",
    "env",
    "nohup",
    "timeout",
    "time",
    "nice",
    "command",
    "exec",
    "stdbuf",
    "setsid",
    "xargs",
    "uvx",
}
RUN_WRAPPERS = {"uv", "pipx", "poetry"}  # `uv run CMD`, `uv tool run CMD`, `pipx run CMD`
SSH_VALUE_FLAGS = set("bcDEeFIiJLlmOopQRSWw")
RUNNERS = {"docker", "podman", "nerdctl", "kubectl"}
SHELLS = {"sh", "bash", "zsh", "dash"}
SHELL_C = re.compile(r"-[a-zA-Z]*c[a-zA-Z]*")
PYTHONS = re.compile(r"^python[0-9.]*$")
ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
NAME_VALUE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)=(\S+)")
MAX_DEPTH = 4


def segments(command: str) -> list[list[str]]:
    """Simple commands of a shell line, split on ; & && || | ( ) and newlines."""
    out: list[list[str]] = []
    for line in command.replace("`", "\n").splitlines():  # `cmd` runs like $(cmd)
        try:
            lexer = shlex.shlex(line, posix=True, punctuation_chars=True)
            lexer.whitespace_split = True
            words = list(lexer)
        except ValueError:  # an unbalanced quote: split on spaces, which is stricter
            words = line.split()
        current: list[str] = []
        for word in words:
            if set(word) <= set(";&|()<>"):
                out.append(current)
                current = []
            else:
                current.append(word)
        out.append(current)
    return [s for s in out if s]


def wrapped_forms(words: list[str]) -> list[list[str]]:
    """The segment, plus every command a wrapper might run: `sudo -u root aiperf x`.

    A wrapper's own options may take a value (`sudo -u root`, `timeout 60`), so we cannot tell
    which word starts the real command. We check every suffix after a wrapper: it fails closed.
    """
    forms = [words]
    start = 0
    while start < len(words) and ASSIGN.match(words[start]):
        start += 1
        forms.append(words[start:])
    if start + 1 >= len(words):
        return [f for f in forms if f]
    prog = Path(words[start]).name
    if prog == "command" and words[start + 1] in ("-v", "-V"):
        return [f for f in forms if f]  # a read-only lookup: `command -v aiperf`
    if prog in WRAPPERS:
        forms += [words[i:] for i in range(start + 1, len(words))]
    elif prog in RUN_WRAPPERS and "run" in words[start + 1 :]:
        run = words.index("run", start + 1)
        forms += [words[i:] for i in range(run + 1, len(words))]
    return [f for f in forms if f]


def inner_commands(words: list[str]) -> list[str]:
    """Command strings a segment runs elsewhere: sh -c, eval, ssh HOST CMD, $(...)."""
    prog = Path(words[0]).name
    inner = [w[w.index("$(") + 2 :] for w in words if "$(" in w]
    if prog in SHELLS:
        for i, word in enumerate(words[1:], 1):
            if SHELL_C.fullmatch(word) and i + 1 < len(words):
                inner.append(words[i + 1])
                break
    elif prog in RUNNERS:
        inner += [w for w in words[1:] if re.search(r"\s", w)]  # a whole command as one word
    elif prog == "eval":
        inner.append(" ".join(words[1:]))
    elif prog == "ssh":
        inner.append(" ".join(ssh_remote_command(words)))
    return inner


def skip_ssh_options(words: list[str], i: int) -> int:
    """The index of the first word at or after i that is not an ssh option or its value."""
    while i < len(words) and words[i].startswith("-") and len(words[i]) > 1:
        if words[i] == "--":
            return i + 1
        letters = words[i][1:]
        for j, letter in enumerate(letters):
            if letter in SSH_VALUE_FLAGS:
                i += j == len(letters) - 1  # `-p 22` takes the next word, `-p22` does not
                break
        i += 1
    return i


def ssh_remote_command(words: list[str]) -> list[str]:
    """The command `ssh [OPTIONS] HOST [OPTIONS] CMD...` runs on the host."""
    host = skip_ssh_options(words, 1)
    return words[skip_ssh_options(words, host + 1) :]


def mentions_tool(command: str) -> bool:
    """Whether any word of the text names a benchmark tool."""
    words = re.split(r"[\s;&|()<>'\"`$]+", command)
    return any(w in TOOLS or TOOL_WORD.search(w) for w in words if w)


def reads_stdin(words: list[str]) -> bool:
    """Whether a shell runs its stdin: `sh`, `bash -s`, but not `sh -c CMD` or `sh FILE`."""
    if Path(words[0]).name not in SHELLS:
        return False
    return all(w.startswith("-") and not SHELL_C.fullmatch(w) for w in words[1:])


def deny_reason_for_tool(words: list[str]) -> str | None:
    """Why one simple command starts a benchmark tool directly, or None."""
    prog = Path(words[0]).name
    if prog.split("@")[0] in TOOLS:  # `uvx aiperf@0.13` too
        return f"{prog} must run through scripts/nimbench.py"
    if PYTHONS.match(prog) and "-m" in words:
        module = words[words.index("-m") + 1] if words.index("-m") + 1 < len(words) else ""
        if module.split(".")[0] in TOOL_MODULES:
            return f"python -m {module} must run through scripts/nimbench.py"
    if prog in RUNNERS and any(w in TOOLS or TOOL_WORD.search(w) for w in words[1:]):
        return f"{prog} must not start a benchmark tool; use scripts/nimbench.py"
    return None


def secret_reason(command: str) -> str | None:
    """Why a command line carries a credential, or None."""
    if TOKEN_SHAPES.search(command):
        return "a credential is on the command line; pass the name of a variable"
    for match in SECRET_FLAG.finditer(command):
        flag, value = match[1], match[3].strip("'\"")
        if flag.endswith("-env") or value.startswith("$") or not value:
            continue
        return f"{flag} has a value; pass the name of a variable, never the key"
    for name, value in NAME_VALUE.findall(command):
        if (
            SECRET_NAME.search(name)
            and value.strip("'\"")
            and not value.lstrip("'\"").startswith("$")
        ):
            return f"{name} is set to a value; pass the name of a variable, never the key"
    return None


def decide(command: str, depth: int = 0) -> str | None:
    """The reason to deny a shell line, or None when it may run."""
    if depth > MAX_DEPTH:
        return "the command is nested too deeply to check"
    reason = secret_reason(command)
    if reason:
        return reason
    for words in segments(command):
        for form in wrapped_forms(words):
            reason = deny_reason_for_tool(form)
            if reason:
                return reason
            if reads_stdin(form) and mentions_tool(command):
                return f"{form[0]} reads a benchmark command on stdin; use scripts/nimbench.py"
            for nested in inner_commands(form):
                reason = decide(nested, depth + 1)
                if reason:
                    return reason
    return None


def main(argv: list[str] | None = None) -> None:
    parser: argparse.ArgumentParser = ContractParser(
        description="PreToolUse hook for nim-benchmark. Reads the tool call as JSON on stdin.",
        epilog='Example: python3 scripts/bench_guard.py --check "aiperf profile --model m"',
    )
    parser.add_argument("--check", metavar="COMMAND", help="print the decision for COMMAND")
    args = parser.parse_args(argv)
    if args.check is not None:
        reason = decide(args.check)
        if reason:
            emit(
                False,
                "denied",
                "use_nimbench",
                {"command": args.check},
                [
                    error(
                        "denied",
                        reason,
                        "Run it through scripts/nimbench.py with the approved plan id",
                    )
                ],
                2,
            )
        emit(True, "allowed", "run", {"command": args.check})
    try:
        payload = json.loads(sys.stdin.read())
        command = payload.get("tool_input", {}).get("command", "")
    except (ValueError, AttributeError):
        print("bench_guard: cannot read the tool call", file=sys.stderr)
        raise SystemExit(2) from None
    reason = decide(command) if isinstance(command, str) else None
    if reason:
        print(f"bench_guard: {reason}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
