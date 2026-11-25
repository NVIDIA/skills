#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic secret lint / redaction for agent-assembled launch artifacts.

Two modes over shell commands, sbatch scripts, k8s manifests (YAML or JSON), or
job-record text:

  lint    — exit 1 if any *literal* credential material is embedded; findings
            name the variable/flag and line only (the secret value is never
            printed).
  redact  — emit the text with literal credential material replaced by
            placeholders, for safe persistence (e.g. into job records). The
            secret value never survives the output.

This is a deterministic PATTERN lint, not a general/entropy secret scanner. In
scope:
  1. NAME=literal env assignments where NAME matches a credential word
     (SECRET/PASSWORD/TOKEN/API_KEY/ACCESS_KEY/NGC_KEY/CREDENTIAL) — incl.
     ``export`` / ``docker -e`` / ``sbatch --export=ALL,NAME=...`` forms.
     ``$VAR`` / ``${VAR}`` / ``$(VAR)`` references are sanctioned and pass;
     ``${VAR:-literal}`` default-expansions are caught via their fallback.
  2. ``--password`` / ``--token`` / ``--api-key`` values (space, ``=``, or
     ``["--token", "value"]`` array form), and ``-p<val>`` / ``-p val`` on
     lines containing ``login`` (``docker run -p 8080:80`` port maps pass).
  3. k8s env entries — a literal ``value:`` under a credential-matching
     ``name:``, in block, flow (``{name: X, value: Y}``), or JSON form, any key
     order. ``valueFrom.secretKeyRef`` and ``value: $(VAR)`` references pass.
  4. Presigned-URL query credentials (``X-Amz-Signature/Credential/Security-Token``).

Known boundaries (out of scope, by design — use $VAR/stdin/secretKeyRef):
  base64/other obfuscation; ``curl -H "Authorization: Bearer ..."`` headers;
  ``https://user:token@host`` URL userinfo; Secret ``stringData:``/``data:``
  values; ``$(echo secret)`` command substitution; secrets in files referenced
  by path; a literal piped to ``--password-stdin`` (indistinguishable from
  ``printf "$VAR"``). No tao_sdk imports; stdlib only.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# Credential-looking identifier, with letter boundaries so TOKENIZERS_PARALLELISM,
# max_new_tokens, skip_special_tokens etc. do NOT match while HF_TOKEN, NGC_KEY,
# NGC_API_KEY, AWS_SECRET_ACCESS_KEY do.
SECRET_NAME_RE = re.compile(
    r"(?i)(?<![A-Za-z])(SECRET|PASSWORD|PASSWD|API_?KEY|ACCESS_KEY|NGC_KEY|TOKEN|CREDENTIAL)(?![A-Za-z])"
)

# A shell "word": adjacent quoted/unquoted segments with no intervening space.
# Unquoted runs stop at whitespace, quotes, and shell separators & ? , so that
# `--export=ALL,NGC_KEY=x` splits on the comma and URL params don't over-run.
_WORD = r"(?:\"[^\"]*\"|'[^']*'|[^\s\"'&?,])+"

_ENV_ASSIGN_RE = re.compile(rf"(?P<name>[A-Za-z_][A-Za-z0-9_]*)=(?P<val>{_WORD})")
# flag value: `--token x`, `--token=x`, or array `"--token", "x"`
_FLAG_RE = re.compile(
    rf"(?P<flag>--password|--token|--api-key)(?:[=\s]+|\"?\s*,\s*\"?)(?P<val>{_WORD})"
)
# -p on login lines, attached (-pVAL) or separated (-p VAL / -p=VAL)
_LOGIN_P_RE = re.compile(rf"(?<!\S)-p[=\s]*(?P<val>{_WORD})")
_LOGIN_LINE_RE = re.compile(r"\blogin\b")

_YAML_NAME_RE = re.compile(r"\s*-?\s*name:\s*[\"']?(?P<name>[A-Za-z_][A-Za-z0-9_]*)[\"']?\s*$")
_YAML_VALUE_RE = re.compile(rf"^(?P<indent>\s*)value:\s*(?P<val>{_WORD}|[|>][+-]?)\s*$")
_YAML_VALUEFROM_RE = re.compile(r"^\s*valueFrom:")

# single-line flow/JSON env entry: find every name:/value: on one line
_KV_NAME_RE = re.compile(r"[\"']?name[\"']?\s*:\s*[\"']?(?P<name>[A-Za-z_][A-Za-z0-9_]*)")
_KV_VALUE_RE = re.compile(rf"[\"']?value[\"']?\s*:\s*(?P<val>{_WORD})")

_PRESIGNED_RE = re.compile(
    r"(?P<param>X-Amz-(?:Signature|Credential|Security-Token))=(?P<val>[^&\s\"']+)"
)

_BLOCK_INDICATOR_RE = re.compile(r"^[|>][+-]?$")
_PARAM_DEFAULT_RE = re.compile(r"^\$\{[A-Za-z_][A-Za-z0-9_]*:[-=+](?P<default>.*)\}$")

# FIELD mode: a record free-text value is data, not a shell command line, so the
# value is consumed WHOLE (no split on & ? ,). Used only by redact_field().
_WORD_FIELD = r"(?:\"[^\"]*\"|'[^']*'|[^\s\"'])+"
_ENV_ASSIGN_FIELD_RE = re.compile(rf"(?P<name>[A-Za-z_][A-Za-z0-9_]*)=(?P<val>{_WORD_FIELD})")
_FLAG_FIELD_RE = re.compile(rf"(?P<flag>--password|--token|--api-key)(?:[=\s]+)(?P<val>{_WORD_FIELD})")

# Format-based credential shapes — deterministic and safe to redact ANYWHERE
# (shell text, free-text field, or path/URI field). Applied in addition to the
# shell rules, and flagged by lint.
_TOKEN_PREFIX_RE = re.compile(
    r"(?:nvapi-[A-Za-z0-9_-]{16,}|hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}"
    r"|github_pat_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16}|xox[baprs]-[A-Za-z0-9-]{10,}"
    r"|sk-[A-Za-z0-9]{20,})"
)
_URL_USERINFO_RE = re.compile(
    r"(?P<pre>[a-zA-Z][a-zA-Z0-9+.\-]*://[^:/@\s]+:)(?P<pw>[^@/\s]+)(?P<at>@)"
)
_JSON_CRED_RE = re.compile(
    r'(?P<pre>"(?P<key>password|passwd|pass|token|api_?key|secret|access_key|secret_key)"\s*:\s*")'
    r'(?P<val>[^"]*)(?P<post>")',
    re.IGNORECASE,
)


def _apply_format_rules(text: str) -> str:
    """Redact format-based secrets (known token prefixes, URL userinfo, JSON
    credential keys). Safe on shell text, free-text, and path/URI fields."""
    text = _TOKEN_PREFIX_RE.sub("<redacted>", text)
    text = _URL_USERINFO_RE.sub(lambda m: f"{m.group('pre')}<redacted>{m.group('at')}", text)
    text = _JSON_CRED_RE.sub(lambda m: f"{m.group('pre')}<redacted>{m.group('post')}", text)
    return text


def redact_field(value: str) -> str:
    """Redact a structured record FREE-TEXT field (e.g. a job-record message).

    A credential value is consumed WHOLE (no split on & ? ,) because the field
    is data, not a command line — biasing toward over-redaction so a secret can
    never survive a durable record."""
    def env_sub(m: re.Match) -> str:
        if SECRET_NAME_RE.search(m.group("name")) and _is_literal(m.group("val")):
            return f'{m.group("name")}="${{{m.group("name")}}}"'
        return m.group(0)

    def flag_sub(m: re.Match) -> str:
        val = m.group("val")
        if _strip_quotes(val) == "stdin" or not _is_literal(val) or _strip_quotes(val).startswith("-"):
            return m.group(0)
        return f"{m.group('flag')} <redacted>"

    value = _ENV_ASSIGN_FIELD_RE.sub(env_sub, value)
    value = _FLAG_FIELD_RE.sub(flag_sub, value)
    value = _PRESIGNED_RE.sub(lambda m: f"{m.group('param')}=<redacted>", value)
    return _apply_format_rules(value)


def redact_uri(value: str) -> str:
    """Redact a PATH/URI-typed record field (e.g. results_dir, image).

    Applies ONLY format-based rules (URL userinfo, presigned query, token
    prefixes) — never the NAME=value shell rule, so a legitimate path segment
    like /lustre/exp/token=v1/run is left intact."""
    value = _PRESIGNED_RE.sub(lambda m: f"{m.group('param')}=<redacted>", value)
    return _apply_format_rules(value)


def _strip_quotes(val: str) -> str:
    if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
        return val[1:-1]
    return val


def _is_literal(val: str) -> bool:
    """True if val embeds actual material rather than referencing an env var."""
    val = val.strip()
    if not val:
        return False
    # A wholly single-quoted value suppresses shell expansion — always literal.
    if len(val) >= 2 and val[0] == "'" and val[-1] == "'":
        return len(val) > 2
    inner = _strip_quotes(val)
    if not inner or inner == "<redacted>":
        return False
    # ${NAME:-default} / :=default / :+alt — the baked default is a literal
    m = _PARAM_DEFAULT_RE.match(inner)
    if m:
        return _is_literal(m.group("default"))
    # $VAR / ${VAR} / $(VAR) reference (incl. k8s dependent-env $(VAR))
    if inner.startswith("$"):
        return False
    return True


@dataclass
class Finding:
    line_no: int
    kind: str
    name: str  # variable / flag / param name — never the value

    def render(self, src: str) -> str:
        return f"{src}:{self.line_no}: {self.kind}: {self.name}"


def _join_continuations(text: str) -> list[tuple[int, str, list[str]]]:
    """Group physical lines into logical lines across trailing-backslash splices.

    Returns (start_line_no, logical_string, [original_physical_lines]).
    """
    raw = text.split("\n")
    if raw and raw[-1] == "":
        raw = raw[:-1]  # split artifact for trailing newline
    out: list[tuple[int, str, list[str]]] = []
    i = 0
    while i < len(raw):
        start = i
        segs = [raw[i]]
        while (segs[-1].endswith("\\") and not segs[-1].endswith("\\\\")
               and i + 1 < len(raw)):
            i += 1
            segs.append(raw[i])
        parts = [s[:-1] if idx < len(segs) - 1 else s for idx, s in enumerate(segs)]
        out.append((start + 1, "".join(parts), segs))
        i += 1
    return out


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def scan(text: str) -> list[Finding]:
    findings: list[Finding] = []
    current_env_name: str | None = None
    block_indent: int | None = None

    for line_no, line, _segs in _join_continuations(text):
        # inside a k8s block scalar under a credential name: continuation carries
        # the secret; skip until dedent (the value: line itself was flagged)
        if block_indent is not None:
            if line.strip() == "" or _indent(line) > block_indent:
                continue
            block_indent = None

        claimed: list[tuple[int, int]] = []

        for m in _PRESIGNED_RE.finditer(line):
            findings.append(Finding(line_no, "presigned-url credential in query string", m.group("param")))
            claimed.append(m.span())

        # k8s block env-entry state machine
        nm = _YAML_NAME_RE.match(line)
        vfrom = _YAML_VALUEFROM_RE.match(line)
        vm = _YAML_VALUE_RE.match(line)
        if nm:
            current_env_name = nm.group("name")
        elif vfrom:
            current_env_name = None
        elif vm and current_env_name and SECRET_NAME_RE.search(current_env_name):
            val = vm.group("val")
            if _BLOCK_INDICATOR_RE.match(val):
                findings.append(Finding(line_no, "inline manifest credential (use valueFrom.secretKeyRef)", current_env_name))
                block_indent = len(vm.group("indent"))
                current_env_name = None
                continue
            if _is_literal(val):
                findings.append(Finding(line_no, "inline manifest credential (use valueFrom.secretKeyRef)", current_env_name))
            current_env_name = None
        else:
            # single-line flow/JSON env entry: name and value on one line
            names = [m.group("name") for m in _KV_NAME_RE.finditer(line)]
            if any(SECRET_NAME_RE.search(n) for n in names):
                secret_name = next(n for n in names if SECRET_NAME_RE.search(n))
                for m in _KV_VALUE_RE.finditer(line):
                    if _is_literal(m.group("val")):
                        findings.append(Finding(line_no, "inline manifest credential (use valueFrom.secretKeyRef)", secret_name))
                        break

        for m in _FLAG_RE.finditer(line):
            claimed.append(m.span())
            val = m.group("val")
            if _strip_quotes(val) == "stdin":
                continue
            if _is_literal(val) and not _strip_quotes(val).startswith("-"):
                findings.append(Finding(line_no, "credential passed as CLI argument", m.group("flag")))

        if _LOGIN_LINE_RE.search(line):
            for m in _LOGIN_P_RE.finditer(line):
                claimed.append(m.span())
                if _is_literal(m.group("val")):
                    findings.append(Finding(line_no, "credential passed as CLI argument", "-p"))

        for m in _ENV_ASSIGN_RE.finditer(line):
            if any(m.start() >= s and m.end() <= e for s, e in claimed):
                continue
            if SECRET_NAME_RE.search(m.group("name")) and _is_literal(m.group("val")):
                findings.append(Finding(line_no, "literal credential assignment", m.group("name")))

        # format-based shapes (never emit the secret value in the finding name)
        if _TOKEN_PREFIX_RE.search(line):
            findings.append(Finding(line_no, "credential token (known prefix)", "token"))
        if _URL_USERINFO_RE.search(line):
            findings.append(Finding(line_no, "credential in URL userinfo", "url-userinfo"))
        for m in _JSON_CRED_RE.finditer(line):
            if m.group("val"):
                findings.append(Finding(line_no, "credential in JSON value", m.group("key").lower()))

    return findings


def _redact_line(line: str) -> str:
    line = _PRESIGNED_RE.sub(lambda m: f"{m.group('param')}=<redacted>", line)

    def _flag_sub(m: re.Match) -> str:
        val = m.group("val")
        if _strip_quotes(val) == "stdin" or not _is_literal(val) or _strip_quotes(val).startswith("-"):
            return m.group(0)
        return f"{m.group('flag')} <redacted>"

    line = _FLAG_RE.sub(_flag_sub, line)

    if _LOGIN_LINE_RE.search(line):
        line = _LOGIN_P_RE.sub(
            lambda m: "-p <redacted>" if _is_literal(m.group("val")) else m.group(0), line
        )

    def _env_sub(m: re.Match) -> str:
        if SECRET_NAME_RE.search(m.group("name")) and _is_literal(m.group("val")):
            return f'{m.group("name")}="${{{m.group("name")}}}"'
        return m.group(0)

    line = _ENV_ASSIGN_RE.sub(_env_sub, line)
    return line


def redact(text: str) -> str:
    trailing = "\n" if text.endswith("\n") else ""
    out: list[str] = []
    current_env_name: str | None = None
    block_indent: int | None = None

    for _line_no, line, segs in _join_continuations(text):
        if block_indent is not None:
            if line.strip() == "" or _indent(line) > block_indent:
                continue  # drop block-scalar secret continuation
            block_indent = None

        nm = _YAML_NAME_RE.match(line)
        vfrom = _YAML_VALUEFROM_RE.match(line)
        vm = _YAML_VALUE_RE.match(line)
        if nm:
            current_env_name = nm.group("name")
            out.extend(segs)
            continue
        if vfrom:
            current_env_name = None
            out.extend(segs)
            continue
        if vm and current_env_name and SECRET_NAME_RE.search(current_env_name):
            val = vm.group("val")
            indent = vm.group("indent")
            if _BLOCK_INDICATOR_RE.match(val):
                out.append(f"{indent}value: <redacted>  # use valueFrom.secretKeyRef")
                block_indent = len(indent)
                current_env_name = None
                continue
            if _is_literal(val):
                out.append(f"{indent}value: <redacted>  # use valueFrom.secretKeyRef")
                current_env_name = None
                continue
            current_env_name = None
            out.extend(segs)
            continue

        # single-line flow/JSON env entry
        names = [m.group("name") for m in _KV_NAME_RE.finditer(line)]
        if any(SECRET_NAME_RE.search(n) for n in names):
            new_line = _KV_VALUE_RE.sub(
                lambda m: (m.group(0)[:m.start("val") - m.start()] + "<redacted>")
                if _is_literal(m.group("val")) else m.group(0),
                line,
            )
            out.append(new_line)
            continue

        new_line = _redact_line(line)
        if new_line != line:
            out.append(new_line)  # collapse a modified continuation to one line
        else:
            out.extend(segs)  # unchanged: preserve original physical lines
    # final sweep for format-based shapes (token prefixes, URL userinfo, JSON creds)
    return _apply_format_rules("\n".join(out) + trailing)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("lint", "redact"))
    parser.add_argument("files", nargs="*", help="files to scan; stdin if omitted or '-'")
    args = parser.parse_args(argv)

    sources: list[tuple[str, str]] = []
    if not args.files or args.files == ["-"]:
        sources.append(("<stdin>", sys.stdin.read()))
    else:
        for f in args.files:
            sources.append((f, Path(f).read_text()))

    if args.mode == "redact":
        for _, text in sources:
            sys.stdout.write(redact(text))
        return 0

    total = 0
    for src, text in sources:
        for finding in scan(text):
            print(finding.render(src), file=sys.stderr)
            total += 1
    if total:
        print(f"FAIL: {total} embedded credential finding(s). Use $VAR references, "
              f"--password-stdin, or valueFrom.secretKeyRef.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
