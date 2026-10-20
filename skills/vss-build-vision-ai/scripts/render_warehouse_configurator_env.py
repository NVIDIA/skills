#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Materialize a fully-interpolated env file for the warehouse blueprint configurator.

`bp-configurator-<mode>` does not read its environment through Compose
interpolation. It declares:

    env_file:
      - ${BP_CONFIGURATOR_BASE_ENV_FILE:-$VSS_APPS_DIR/.../.env}
      - ${BP_CONFIGURATOR_ENV_FILE:-$VSS_APPS_DIR/.../overrides.env}

so with those knobs unset it loads the *checked-in* files directly, bypassing the
--env-file layering entirely. A build's `override.env` therefore cannot reach it,
and the pristine `HOST_IP='<HOST_IP>'` sentinel is baked into the container that
renders every stream and hardware config.

This script renders the effective env -- all four ordered layers, with ${VAR} and
$VAR references expanded to a fixpoint -- into a build-local file. Point
BP_CONFIGURATOR_ENV_FILE at it so the configurator sees the same values the rest
of the build resolved. Values are concrete, so no expansion is required at
env_file load time.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ASSIGN = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$")

# Never leak these into a file that is mounted into a container's environment
# wholesale; the configurator does not need them.
SECRET_KEYS = {"NGC_API_KEY", "NGC_CLI_API_KEY", "NVIDIA_API_KEY", "OPENAI_API_KEY", "HF_TOKEN"}

# No legitimate warehouse env value is anywhere near this; a hit means the
# expander ran away, not that a real value is large.
MAX_VALUE_BYTES = 8192

# A `${...}` or `$NAME` that survived expansion. `$$` is an escaped literal
# dollar and is deliberately not matched.
UNRESOLVED_REF = re.compile(r"(?<!\$)\$(?:\{[A-Za-z_]|[A-Za-z_])")


def strip_value(value: str) -> str:
    """Unquote and drop an inline comment, matching Compose's env_file parser.

    Compose ends an *unquoted* value at the first whitespace-preceded `#`, and
    for a quoted value takes the quoted span and ignores the remainder. Keeping
    the comment instead corrupts real values: warehouse-operations/.env ships
    `NVSTREAMER_IP=vss-vios-nvstreamer # Compose service DNS name; ...`, and a
    hostname with prose appended fails DNS inside the container.
    """
    if value[:1] in ("'", '"'):
        quote = value[0]
        end = value.find(quote, 1)
        if end != -1:
            return value[1:end]
        return value[1:]
    head, hash_sep, _ = value.partition("#")
    if hash_sep and (not head or head[-1].isspace()):
        value = head
    return value.strip()


def parse_env(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    if not path.is_file():
        return pairs
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = ASSIGN.match(line)
        if not match:
            continue
        key, value = match.group(1), match.group(2).strip()
        value = strip_value(value)
        pairs.append((key, value))
    return pairs


NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _match_brace(text: str, open_index: int) -> int | None:
    """Index of the '}' closing the '{' at open_index, honouring nesting."""
    depth = 0
    for i in range(open_index, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return None


def _split_ref(inner: str) -> tuple[str, str | None, str | None]:
    """Split '${...}' body into (name, operator, default) at brace depth 0."""
    depth = 0
    for i, ch in enumerate(inner):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif depth == 0 and ch in ":-":
            if ch == ":" and inner[i + 1 : i + 2] in ("-", "?", "="):
                return inner[:i], inner[i : i + 2], inner[i + 2 :]
            if ch == "-":
                return inner[:i], "-", inner[i + 1 :]
    return inner, None, None


def _lookup(name: str, values: dict[str, str], stack: frozenset[str]) -> str | None:
    """Expanded value of `name`, or None when unset or self-referential.

    A self-reference (`X="${X:-default}"`, the shell idiom used throughout
    containers.env) means X was unset when the layer was written, so it must
    fall through to the default rather than substitute X's own raw text.
    """
    if name in stack or name not in values:
        return None
    return _expand_str(values[name], values, stack | {name})


def _expand_str(text: str, values: dict[str, str], stack: frozenset[str]) -> str:
    out: list[str] = []
    i = 0
    while i < len(text):
        if text[i] == "$" and i + 1 < len(text):
            if text[i + 1] == "{":
                close = _match_brace(text, i + 1)
                if close is None:
                    out.append(text[i])
                    i += 1
                    continue
                inner = text[i + 2 : close]
                name, operator, default = _split_ref(inner)
                if not NAME_RE.fullmatch(name):
                    out.append(text[i : close + 1])
                    i = close + 1
                    continue
                value = _lookup(name, values, stack)
                if operator is None:
                    if value is None:
                        out.append("" if name in values else text[i : close + 1])
                    else:
                        out.append(value)
                elif operator in (":-", ":?", ":="):
                    out.append(
                        value if value else _expand_str(default or "", values, stack)
                    )
                else:  # bare '-': only an unset name takes the default
                    out.append(
                        value
                        if value is not None
                        else _expand_str(default or "", values, stack)
                    )
                i = close + 1
                continue
            match = NAME_RE.match(text, i + 1)
            if match:
                name = match.group(0)
                value = _lookup(name, values, stack)
                if value is None:
                    out.append("" if name in values else f"${name}")
                else:
                    out.append(value)
                i = match.end()
                continue
        out.append(text[i])
        i += 1
    return "".join(out)


def expand(values: dict[str, str]) -> dict[str, str]:
    """Expand ${VAR}, ${VAR:-default} and $VAR to a fixpoint.

    Defaults may nest (`${A:-${B}/x:${C}}`), so references are parsed with
    balanced-brace matching rather than a flat regex; a flat `[^}]*` default
    ends at the first '}' and leaves the remainder as literal text, which on a
    self-referential key grows the value on every pass.
    """
    return {key: _expand_str(value, values, frozenset()) for key, value in values.items()}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_dir", type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output-name", default="configurator.env")
    args = parser.parse_args()

    repo, build_dir = args.repo_root, args.build_dir
    foundation_dir = repo / "deploy/docker/industry-profiles/warehouse-operations"

    ordered: dict[str, str] = {}
    for path in (
        repo / "deploy/docker/containers.env",
        foundation_dir / ".env",
        foundation_dir / "overrides.env",
        build_dir / "override.env",
    ):
        for key, value in parse_env(path):
            ordered[key] = value

    resolved = expand(ordered)

    leftover = sorted(k for k, v in resolved.items() if "<HOST_IP>" in v or "/path/to" in v)
    if leftover:
        print(
            "ERROR: unresolved stock sentinels remain after expansion: "
            + ", ".join(leftover)
            + "\n       set concrete values in override.env and re-render",
            file=sys.stderr,
        )
        raise SystemExit(1)

    dangling = sorted(k for k, v in resolved.items() if UNRESOLVED_REF.search(v))
    if dangling:
        print(
            "ERROR: unresolved variable references remain after expansion: "
            + ", ".join(f"{k}={resolved[k]!r}" for k in dangling)
            + "\n       env_file values are literal, so these would reach the"
            " container verbatim;\n       set concrete values in override.env and"
            " re-render",
            file=sys.stderr,
        )
        raise SystemExit(1)

    # A runaway expansion is not visible in the value count -- only in the
    # size. Fail here, where the cause is, rather than at bring-up where the
    # symptom is a bare 'argument list too long' from the configurator.
    oversized = sorted(
        (len(v), k) for k, v in resolved.items() if len(v) > MAX_VALUE_BYTES
    )
    if oversized:
        print(
            "ERROR: runaway variable expansion in "
            + ", ".join(f"{k} ({n} bytes)" for n, k in reversed(oversized))
            + f"\n       each value must stay under {MAX_VALUE_BYTES} bytes; a"
            " self-referential\n       value with a nested default is the usual"
            " cause",
            file=sys.stderr,
        )
        raise SystemExit(1)

    out = build_dir / args.output_name
    lines = [
        "# Generated by render_warehouse_configurator_env.py -- do not edit.",
        "# Fully-interpolated effective env for bp-configurator-<mode>, which",
        "# loads env_file directly and cannot see Compose --env-file layering.",
    ]
    for key in sorted(resolved):
        if key in SECRET_KEYS:
            continue
        value = resolved[key]
        if "\n" in value:
            print(
                f"WARNING: {key} contains a newline and cannot be represented in "
                "an env_file; omitted from the rendered output",
                file=sys.stderr,
            )
            continue
        lines.append(f"{key}={value}")
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out} ({len(lines) - 3} values)")


if __name__ == "__main__":
    main()
