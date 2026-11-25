#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Split a Bash command line into simple commands for guard.py.

`parse` returns pipelines of `Command`. It looks inside $(...), <(...), backticks, `bash -c`,
`su -c`, `eval`, and strips wrapper programs (sudo, env, timeout, xargs, ...) and VAR=value
prefixes, so the guard sees the real program and arguments. `python3 scripts/shellparse.py --check
CMD` prints the parse as JSON. Raises GuardError when a command cannot be understood; the guard
then denies it. Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path

MAX_DEPTH = 6
SHELLS = {"bash", "sh", "zsh", "dash", "ksh"}
PAYLOAD_PROGRAMS = SHELLS | {"su"}
KEYWORDS = {
    "!",
    "{",
    "}",
    "if",
    "then",
    "else",
    "elif",
    "fi",
    "do",
    "done",
    "while",
    "until",
    "time",
}
ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")
# wrapper -> (flags that take a value, positional arguments to skip before the real command)
WRAPPERS: dict[str, tuple[set[str], int]] = {
    "sudo": ({"-u", "-g", "-h", "-p", "-C", "-D", "-r", "-t", "-U", "-T"}, 0),
    "doas": ({"-u", "-C"}, 0),
    "env": ({"-u", "--unset", "-C", "--chdir"}, 0),
    "timeout": ({"-s", "--signal", "-k", "--kill-after"}, 1),
    "nohup": (set(), 0),
    "time": ({"-f", "--format", "-o", "--output"}, 0),
    "nice": ({"-n"}, 0),
    "stdbuf": ({"-i", "-o", "-e"}, 0),
    "command": (set(), 0),
    "exec": ({"-a"}, 0),
    "setsid": (set(), 0),
    "watch": ({"-n", "--interval"}, 0),
    "xargs": ({"-I", "-L", "-n", "-P", "-d", "-E", "-s", "-a"}, 0),
}


class GuardError(ValueError):
    """The command could not be understood; the guard denies it."""


@dataclass
class Command:
    tokens: list[str]
    wrappers: list[str] = field(default_factory=list)
    env_names: set[str] = field(default_factory=set)
    stdin_text: list[str] = field(default_factory=list)  # here-strings: `bash <<< 'text'`

    @property
    def prog(self) -> str:
        return Path(self.tokens[0]).name

    @property
    def norm(self) -> str:
        """Program basename plus arguments joined by single spaces: what the rules match."""
        return " ".join([self.prog, *self.tokens[1:]])


def _first_word(body: str) -> str:
    words = body.strip().split()
    return re.sub(r"[^A-Za-z0-9_.-]", "", Path(words[0]).name) if words else "empty"


def extract_substitutions(text: str) -> tuple[str, list[str]]:
    """Replace $(..), <(..), >(..) and `..` with __SUBST_<prog>__; return (outer, bodies)."""
    out: list[str] = []
    bodies: list[str] = []
    i, quote = 0, None
    while i < len(text):
        char = text[i]
        if quote == "'":
            quote = None if char == "'" else quote
            out.append(char)
            i += 1
            continue
        if char == "\\":
            out.append(text[i : i + 2])
            i += 2
            continue
        if char in "\"'":
            if char == '"' or quote is None:
                quote = None if quote == char else char
            out.append(char)
            i += 1
            continue
        if char == "`":
            end = text.find("`", i + 1)
            if end < 0:
                raise GuardError("unbalanced backtick")
            body, i = text[i + 1 : end], end + 1
        elif text.startswith(("$(", "<(", ">("), i) and not (quote and text[i] != "$"):
            depth, j, inner = 1, i + 2, None
            while j < len(text) and depth:
                ch = text[j]
                if inner:
                    if ch == inner:
                        inner = None
                    elif ch == "\\" and inner == '"':
                        j += 1
                elif ch in "'\"":
                    inner = ch
                elif ch == "\\":
                    j += 1
                elif ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                j += 1
            if depth:
                raise GuardError("unbalanced $(")
            body, i = text[i + 2 : j - 1], j
        else:
            out.append(char)
            i += 1
            continue
        bodies.append(body)
        out.append(f" __SUBST_{_first_word(body)}__ ")
    if quote:
        raise GuardError("unterminated quote")
    return "".join(out), bodies


def strip_comments(text: str) -> str:
    """Drop bash comments: an unquoted # at the start of a word runs to the end of the line."""
    out: list[str] = []
    quote, index = None, 0
    while index < len(text):
        char = text[index]
        if quote == "'":
            quote = None if char == "'" else quote
        elif char == "\\":
            out.append(text[index : index + 2])
            index += 2
            continue
        elif char in "\"'" and (quote is None or quote == char):
            quote = None if quote else char
        elif char == "#" and quote is None and (index == 0 or text[index - 1] in " \t\n;&|()<>"):
            end = text.find("\n", index)
            index = len(text) if end < 0 else end
            continue
        out.append(char)
        index += 1
    return "".join(out)


def tokenize(text: str) -> list[str]:
    text = text.replace("\\\n", "").replace("\n", " ; ")  # bash joins `kube\<newline>ctl`
    lexer = shlex.shlex(text, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        return list(lexer)
    except ValueError as exc:
        raise GuardError(f"cannot parse command: {exc}") from exc


def is_operator(token: str) -> bool:
    return bool(token) and all(char in "();<>|&" for char in token)


def command_payload(tokens: list[str]) -> str | None:
    """The command line given with -c to a shell or su, else None. A shell's -c counts only before
    its first operand (later words are script arguments)."""
    name = Path(tokens[0]).name
    for index, token in enumerate(tokens[1:], 1):
        if token in ("-c", "--command") or (
            name in SHELLS and re.fullmatch(r"-[a-zA-Z]*c[a-zA-Z]*", token)
        ):
            following = [word for word in tokens[index + 1 :] if word != "--"][:1]
            return following[0] if following else ""
        if name in SHELLS and not token.startswith(("-", "+")):
            return None
    return None


def env_split_string(tokens: list[str], i: int) -> tuple[list[str] | None, int]:
    """GNU `env -S 'cmd args'`, `-S'cmd'`, `-iS 'cmd'`, `--split-string[=]'cmd'` at TOKENS[i].

    Returns (the words of the string, index after the option), or (None, i + 1) for any other
    option. A string that cannot be split exactly like env does (unbalanced quotes, `$` or `\\`
    expansions) raises GuardError, so the guard never lets a hidden command through."""
    token = tokens[i]
    if token.startswith("--"):
        if token != "--split-string" and not token.startswith("--split-string="):
            return None, i + 1
        text = token.partition("=")[2] if "=" in token else None
        end, key = i + 1, "--split-string"
    else:
        position = token.find("S")
        if position < 1:
            return None, i + 1
        text, end, key = token[position + 1 :] or None, i + 1, "-S"
    if text is None:
        if end >= len(tokens):
            raise GuardError(f"env {key} has no command string")
        text, end = tokens[end], end + 1
    if "$" in text or "\\" in text:
        raise GuardError(f"env {key} with an expansion or escape cannot be checked")
    try:
        return shlex.split(text), end
    except ValueError as exc:
        raise GuardError(f"env {key} string cannot be parsed: {exc}") from exc


def strip_prefixes(tokens: list[str], env_names: set[str], wrappers: list[str]) -> list[str]:
    """Drop shell keywords, VAR=value assignments and wrapper programs from the front."""
    while tokens:
        head, name = tokens[0], Path(tokens[0]).name
        if head in KEYWORDS:
            tokens = tokens[1:]
        elif ASSIGNMENT.match(head):
            env_names.add(head.split("=", 1)[0])
            tokens = tokens[1:]
        elif name in WRAPPERS:
            wrappers.append(name)
            value_flags, skip = WRAPPERS[name]
            i = 1
            split: list[str] | None = None
            while i < len(tokens) and tokens[i].startswith("-"):
                if tokens[i] == "--":
                    i += 1
                    break
                if name == "env" and tokens[i] not in value_flags:
                    split, i = env_split_string(tokens, i)
                    if split is not None:
                        break
                    continue
                i += 2 if tokens[i] in value_flags else 1
            tokens = split + tokens[i:] if split is not None else tokens[i + skip :]
        else:
            break
    return tokens


def nested(command: Command, depth: int) -> list[list[Command]]:
    """Commands hidden inside `bash -c '...'`, `su -c '...'` and `eval ...`."""
    if command.prog in PAYLOAD_PROGRAMS:
        payload = command_payload(command.tokens)
        if payload is not None:
            return parse(payload, depth + 1)
    if command.prog == "eval" and len(command.tokens) > 1:
        return parse(" ".join(command.tokens[1:]), depth + 1)
    return []


def parse(text: str, depth: int = 0) -> list[list[Command]]:
    """Split TEXT into pipelines of simple commands, recursing into nested command strings."""
    if depth > MAX_DEPTH:
        raise GuardError("command nests too deeply")
    outer, bodies = extract_substitutions(strip_comments(text))
    pipelines: list[list[Command]] = []
    for body in bodies:
        pipelines.extend(parse(body, depth + 1))
    current: list[Command] = []
    words: list[str] = []
    herestrings: list[str] = []

    def finish() -> None:
        nonlocal words
        env_names: set[str] = set()
        wrappers: list[str] = []
        cleaned = strip_prefixes(words, env_names, wrappers)
        words = []
        if not cleaned:
            return
        command = Command(cleaned, wrappers, env_names, list(herestrings))
        herestrings.clear()
        pipelines.extend(nested(command, depth))
        current.append(command)

    tokens = tokenize(outer)
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if not is_operator(token):
            words.append(token)
        elif "<" in token or ">" in token:
            if words and words[-1].isdigit():  # a file descriptor: 2>&1
                words.pop()
            if token == "<<<" and i + 1 < len(tokens):
                herestrings.append(tokens[i + 1])
            i += 1  # skip the redirection target
        elif token in ("|", "|&"):
            finish()
        else:
            finish()
            if current:
                pipelines.append(current)
            current = []
        i += 1
    finish()
    if current:
        pipelines.append(current)
    return pipelines


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Example: python3 scripts/shellparse.py --check 'sudo kubectl get pods | head'",
    )
    parser.add_argument(
        "--check", required=True, metavar="COMMAND", help="print how COMMAND splits"
    )
    args = parser.parse_args(argv)
    try:
        result = [[command.norm for command in pipeline] for pipeline in parse(args.check)]
    except GuardError as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "state": "parse",
                    "next": "fix_errors",
                    "facts": {},
                    "errors": [
                        {
                            "code": "unparseable",
                            "message": str(exc),
                            "fix": "Write the command as plain shell",
                        }
                    ],
                }
            )
        )
        sys.exit(2)
    print(
        json.dumps(
            {
                "ok": True,
                "state": "parse",
                "next": "done",
                "facts": {"pipelines": result},
                "errors": [],
            }
        )
    )


if __name__ == "__main__":
    main()
