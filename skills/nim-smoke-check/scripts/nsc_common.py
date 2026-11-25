#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the nim-smoke-check scripts: `import nsc_common as nsc`.

Holds:
  - the JSON output contract (one object on stdout), the exit codes, the error that library code
    raises for a script to report, and an argument parser that reports usage errors as JSON;
  - key handling: a key is read only from the environment variable the user names, and it is
    masked, with well-known token shapes, in everything the scripts print or save (SCRUBBER);
  - the base URL rules (normalize_base) and the one place a URL is opened (open_url): no redirect
    is followed, a loopback URL never goes through a proxy, TLS verification is always on, and
    one deadline ends the whole request;
  - request_json (one JSON answer) and request_stream (an event stream, read as it arrives);
    parse_json reads a server's JSON and refuses one nested deeper than MAX_JSON_DEPTH levels
    or holding NaN or Infinity;
  - private output folders and files (0700 folders, 0600 files, atomic writes).

Stdlib only. Run directly, it prints a notice: it has no command of its own.
Exit codes: 0 ok, 2 bad input, 3 precondition failed, 4 external tool or endpoint failed,
5 refused.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import http.client
import io
import ipaddress
import json
import math
import os
import re
import socket
import ssl
import stat
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NoReturn

EXIT_OK, EXIT_INPUT, EXIT_PRECONDITION, EXIT_EXTERNAL, EXIT_REFUSED = 0, 2, 3, 4, 5
USER_AGENT = "nim-smoke-check/0.1"
CACHE_FOLDER = "~/.cache/nim-smoke-check"  # the parent of the default --out folder
MAX_JSON_BYTES = 8_000_000  # bytes read of one JSON answer
MAX_STREAM_BYTES = 2_000_000  # bytes read of one event stream
READ_CHUNK = 65_536
# Lists and objects nested deeper than this in a server's JSON make it "not JSON" (parse_json):
# a chat answer nests about 6 levels, and the masking and cutting walkers recurse once a level.
MAX_JSON_DEPTH = 64
# The walkers (Scrubber.scrub_obj, and the evidence's cut and image redaction) replace anything
# nested deeper than this with NESTED, so no value can exhaust Python's recursion limit.
MAX_WALK_DEPTH = 2 * MAX_JSON_DEPTH
NESTED = f"[nested deeper than {MAX_WALK_DEPTH} levels: not kept]"

# HttpResult.error: one of these codes, some followed by ": detail"
ERR_REDIRECT = "redirect_refused"  # a 3xx answer; status holds the code; it is never followed
ERR_TIMEOUT = "timeout"  # no answer in time (status 0), or the body did not end in time
ERR_TOO_LARGE = "body_too_large"  # the body was cut at max_bytes
ERR_CONNECT = "connection_failed"  # no answer: refused, unknown host, reset before an answer
ERR_LOST = "connection_lost"  # the connection broke after the answer started
ERR_TLS = "tls_error"  # the TLS handshake failed, for example the certificate check
ERR_REQUEST = "bad_request"  # the request could not be sent: a bad URL or header


# --- the output contract ------------------------------------------------------------------------


def error(code: str, message: str, fix: str) -> dict[str, str]:
    """One entry of the `errors` list."""
    return {"code": code, "message": message, "fix": fix}


class ScriptError(Exception):
    """An error a script reports as one `errors` entry, with the exit code to use."""

    def __init__(
        self,
        code: str,
        message: str,
        fix: str,
        exit_code: int = EXIT_INPUT,
        facts: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.fix = fix
        self.exit_code = exit_code
        self.facts: dict[str, Any] = facts or {}

    def as_error(self) -> dict[str, str]:
        return error(self.code, self.message, self.fix)


def log(message: str) -> None:
    """Progress and diagnostics go to stderr, masked; stdout carries only the JSON object."""
    print(SCRUBBER.scrub(message), file=sys.stderr)


def emit(
    ok: bool,
    state: str,
    next_: str,
    facts: dict[str, Any] | None = None,
    errors: list[dict[str, str]] | None = None,
    code: int | None = None,
) -> NoReturn:
    """Print the one JSON object of the script-output contract, masked by SCRUBBER, and exit
    (default: 0 when ok, else 2)."""
    payload = {
        "ok": ok,
        "state": state,
        "next": next_,
        "facts": facts or {},
        "errors": errors or [],
    }
    sys.stdout.write(SCRUBBER.dumps(payload, sort_keys=True) + "\n")
    sys.stdout.flush()
    raise SystemExit(code if code is not None else (EXIT_OK if ok else EXIT_INPUT))


class JsonArgumentParser(argparse.ArgumentParser):
    """Report usage errors as the standard JSON result (bad_input, exit 2), not argparse text.
    A flag counts only when written in full (allow_abbrev is always off): --api-key=VALUE is an
    unknown flag, never a short form of --api-key-env. Subparsers get the same class."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["allow_abbrev"] = False
        super().__init__(*args, **kwargs)

    def error(self, message: str) -> NoReturn:
        fix = f"See: {self.prog} --help"
        errors = [error("bad_input", _without_values(message), fix)]
        emit(False, "input", "ask_user", errors=errors, code=EXIT_INPUT)


UNRECOGNIZED = "unrecognized arguments: "
INVALID_CHOICE = re.compile(r"(invalid choice: )('[^'\n]*'|\"[^\"\n]*\")")


def _without_values(message: str) -> str:
    """argparse's "unrecognized arguments: ..." with the unknown flags only: a value given to an
    unknown flag (a key typed after --api, say) is never echoed. In "invalid choice: 'X'", an X
    that looks like a key (a stray key taken as the command) is not shown either."""
    message = INVALID_CHOICE.sub(
        lambda found: found.group(1) + (
            "(a value that looks like a key, not shown)"
            if looks_like_key(found.group(2)[1:-1]) else found.group(2)
        ),
        message,
    )
    if not message.startswith(UNRECOGNIZED):
        return message
    words = message[len(UNRECOGNIZED):].split()
    flags = [word.split("=", 1)[0] for word in words if FLAG.fullmatch(word.split("=", 1)[0])]
    if len(words) > len(flags) or any("=" in word for word in words):
        flags.append("(values not shown)")
    return UNRECOGNIZED + " ".join(flags)


# --- keys and masking -----------------------------------------------------------------------

MASK = "***"
# Shorter registered values are not masked (they would mask common words), so read_key refuses a
# key that short: every key the scripts send is one they can mask.
MIN_SECRET = 6
# What --api-key-env takes: the name of a variable, never its value.
ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}\Z")
# A flag or a name that carries a credential ("tokens" and "tokenizer" are engine settings).
SECRET_WORD = re.compile(r"key|token(?!s|izer)|secret|passw|credential|auth(?!or)", re.IGNORECASE)
# Well-known credential shapes, masked wherever they appear.
TOKEN_SHAPES = re.compile(
    r"\b(?:nvapi-[A-Za-z0-9_\-]{16,}|hf_[A-Za-z0-9]{16,}|glpat-[A-Za-z0-9_\-]{16,}|"
    r"gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_\-]{20,}|AKIA[0-9A-Z]{16})\b"
)
BEARER = re.compile(r"\b((?i:bearer)[ \t]+)([A-Za-z0-9._~+/\-]{16,}=*)")
# A flag with its value (--api-key X, --hf-token=X); _mask_flag masks the value when the flag
# names a credential. The look-behind keeps a word such as "re-key" from counting as a flag.
FLAG_VALUE = re.compile(
    r"(?<![A-Za-z0-9_-])(--?[A-Za-z0-9][A-Za-z0-9_-]*)(=|[ \t]+)"
    r"(\"(?:[^\"\\\n]|\\.)*\"|'(?:[^'\\\n]|\\.)*'|[^\s\"',\]}\\]+)"
)
FLAG = re.compile(r"--?[A-Za-z0-9][A-Za-z0-9_-]*")
# The same flag and value as two items of a list written as JSON or Python: ["--api-key", "X"].
FLAG_ITEMS = re.compile(
    r"([\"'])(--?[A-Za-z0-9][A-Za-z0-9_-]*)\1([ \t]*,[ \t]*)([\"'])((?:\\.|(?!\4)[^\\\n])*)\4"
)
# A name that holds a credential, in a header, an assignment or a JSON field: x-api-key,
# NGC_API_KEY, client_secret, "password", Authorization, HF_TOKEN, x-auth-token. Narrower than
# SECRET_WORD, as a bare name is common in prose and in answers: "key" alone (a reasoning switch's
# field), "auth" (how the key was sent), "token" (a log-probability's field), "eos_token" and
# "credentials" (kubectl's "getting credentials: exec: ...") are not.
CREDENTIAL_NAME = re.compile(
    r"api[-_]?key|access[-_]?key|private[-_]?key|secret|passw|authorization"
    r"|(?<![A-Za-z])(?:hf|hub|ngc|gh|github|gitlab|api|auth|access|refresh|session|bearer|id)"
    r"[-_]?token(?![A-Za-z])",
    re.IGNORECASE,
)
# A credential name, then ":" or "=" (after the name's closing quote in JSON), an optional scheme
# word that stays readable ("Authorization: Basic ***"), and the value: quoted, or up to a space
# or a delimiter. _mask_named masks the value (not a variable name given to an ..._env name).
# A name starts only where no name character comes before it, and holds a credential word (the
# look-ahead), so a plain name ("https:") consumes nothing and the time stays linear. A name
# given to an -env flag ("--api-key-env NVIDIA_API_KEY: never ...") is a variable, not a field.
NAMED_VALUE = re.compile(
    r"(?<![A-Za-z0-9_.-])(?<![-_]env[ \t=])(?<![-_]env[ \t=][\"'])"
    rf"(?=[A-Za-z0-9_.-]*?(?i:{CREDENTIAL_NAME.pattern}))"
    r"([A-Za-z0-9_.-]+)([\"']?[ \t]*[:=][ \t]*)"
    r"((?i:bearer|basic|digest|token|apikey)[ \t]+)?"
    r"(\"(?:[^\"\\\n]|\\.)*\"|'(?:[^'\\\n]|\\.)*'|[^\s\"',;&\]}\\]+)"
)
SCHEME = re.compile(r"((?i:bearer|basic|digest|token|apikey)[ \t]+)?(.*)", re.DOTALL)


MIN_KEY_TAIL = 16  # letters and digits after the last "_" of a prefixed key (gsk_..., r8_...)


def _mixed(text: str) -> bool:
    return any(char.isdigit() for char in text) and any(char.isalpha() for char in text)


def looks_like_key(text: str) -> bool:
    """Whether `text` looks like a credential rather than a name: a well-known token shape, a
    bearer token, 20 characters or more that mix letters and digits with no underscore, as
    random keys do, or such a random run of 16 characters or more after the last underscore,
    as keys with a short prefix have (gsk_..., r8_..., sk_live_..., a gateway's prod_...)."""
    if TOKEN_SHAPES.search(text) or BEARER.search(text):
        return True
    if "_" not in text:
        return len(text) >= 20 and _mixed(text)
    tail = text.rsplit("_", 1)[1]
    return len(tail) >= MIN_KEY_TAIL and tail.isascii() and tail.isalnum() and _mixed(tail)


def _names_variable(name: str, value: str) -> bool:
    """Whether `value` is the name of a variable, given to a name or flag that ends in "env"
    (--api-key-env NAME, api_key_env=NAME), and not a key."""
    # a sentence may end right after the name ("--api-key-env NAME; never ...", "(... NAME)"):
    # the punctuation is not part of it
    bare = value.strip("\"'").rstrip(".,;:!?)")
    return (
        name.lower().endswith(("-env", "_env"))
        and bool(ENV_NAME.match(bare))
        and not looks_like_key(bare)
    )


def _secret_flag(flag: str, value: str) -> bool:
    """Whether `value`, given to `flag`, is to be masked: the flag names a credential, and the
    value is not a variable name given to an -env flag."""
    return bool(SECRET_WORD.search(flag)) and not _names_variable(flag, value)


def _masked(value: str) -> str:
    """MASK in place of `value`, after its scheme word when it has one ("Basic ***")."""
    scheme = SCHEME.match(value)
    return (scheme.group(1) if scheme and scheme.group(1) else "") + MASK


def _mask_flag(match: re.Match[str]) -> str:
    flag, separator, value = match.group(1), match.group(2), match.group(3)
    if not _secret_flag(flag, value):
        return match.group(0)
    return f"{flag}{separator}{MASK}"


def _mask_flag_items(match: re.Match[str]) -> str:
    quote, flag, comma, open_, value = match.groups()
    if not _secret_flag(flag, value) or value == MASK or FLAG.fullmatch(value):
        return match.group(0)
    return f"{quote}{flag}{quote}{comma}{open_}{MASK}{open_}"


def _secret_field(name: str, value: str) -> bool:
    """Whether `value`, held by the header, variable or field `name`, is to be masked."""
    return bool(CREDENTIAL_NAME.search(name)) and not _names_variable(name, value)


def _mask_named(match: re.Match[str]) -> str:
    name, separator, scheme, value = match.groups()
    quote = value[0] if value[:1] in ("'", '"') else ""
    inner = value[1:-1] if quote else value
    if not _secret_field(name, inner) or inner in ("", MASK, "null", "true", "false"):
        return match.group(0)
    return f"{name}{separator}{scheme or ''}{quote}{_masked(inner)}{quote}"


class Scrubber:
    """Masks credentials in any text the scripts print or save: the values registered with add()
    (the key read_key read), well-known token shapes (nvapi-, hf_, glpat-, gh?_, sk-, AKIA),
    bearer tokens, the values of credential flags (--api-key X, --api-key=X, and the item after
    "--api-key" in a list such as ["--api-key", "X"]), and the values of credential names
    (CREDENTIAL_NAME) in a header, an assignment or a JSON field (x-api-key: X, NGC_API_KEY=X,
    {"password": "X"}, Authorization: Basic X). It reads no environment variable: a key is
    registered when read_key reads it."""

    def __init__(self) -> None:
        self.values: frozenset[str] = frozenset()

    def add(self, value: object) -> None:
        """Register a secret value, also in its JSON-escaped forms, for text that was serialized
        first. Values shorter than MIN_SECRET characters are ignored."""
        text = str(value)
        if len(text) < MIN_SECRET:
            return
        forms = {text, json.dumps(text)[1:-1], json.dumps(text, ensure_ascii=False)[1:-1]}
        self.values = self.values | forms  # a new set: another thread may be reading the old one

    def scrub(self, text: str) -> str:
        """The text with registered values, token shapes, credential flag values and the values
        of credential names masked."""
        text = str(text)
        for value in sorted(self.values, key=len, reverse=True):
            text = text.replace(value, MASK)
        text = TOKEN_SHAPES.sub(MASK, text)
        text = BEARER.sub(lambda match: match.group(1) + MASK, text)
        text = NAMED_VALUE.sub(_mask_named, text)
        text = FLAG_ITEMS.sub(_mask_flag_items, text)
        return FLAG_VALUE.sub(_mask_flag, text)

    def scrub_obj(self, obj: Any, depth: int = 0) -> Any:
        """A copy of a JSON-ready value with every string masked, dictionary keys included; a
        list or object nested deeper than MAX_WALK_DEPTH levels becomes the text NESTED, and a
        number JSON has no form for (NaN, an infinity) its text ("nan", "inf", "-inf"). The
        text or number of a field whose name holds a credential ({"x-api-key": X}) and the item
        after a credential flag in a list (["--api-key", X]) become MASK."""
        if isinstance(obj, (dict, list, tuple, set, frozenset)) and depth >= MAX_WALK_DEPTH:
            return NESTED
        if isinstance(obj, dict):
            return {
                (self.scrub(key) if isinstance(key, str) else key): self._field(key, value, depth)
                for key, value in obj.items()
            }
        if isinstance(obj, (list, tuple, set, frozenset)):
            items = list(obj)
            return [
                MASK if index and self._after_flag(items[index - 1], value)
                else self.scrub_obj(value, depth + 1)
                for index, value in enumerate(items)
            ]
        if isinstance(obj, str):
            return self.scrub(obj)
        if isinstance(obj, Path):
            return self.scrub(str(obj))
        if isinstance(obj, float) and not math.isfinite(obj):
            return str(obj)  # json.dumps would write NaN or Infinity, which JSON readers refuse
        return obj

    def _field(self, name: Any, value: Any, depth: int) -> Any:
        """The masked value of the field `name` of an object."""
        plain = isinstance(value, (str, int, float)) and not isinstance(value, bool)
        if isinstance(name, str) and plain and _secret_field(name, str(value)):
            if isinstance(value, str):
                return value if value in ("", MASK) else _masked(value)
            return MASK
        return self.scrub_obj(value, depth + 1)

    @staticmethod
    def _after_flag(before: Any, value: Any) -> bool:
        """Whether the list item `value` is the value of the credential flag `before`."""
        return (
            isinstance(before, str) and isinstance(value, str) and bool(FLAG.fullmatch(before))
            and not FLAG.fullmatch(value) and _secret_flag(before, value)
        )

    def dumps(self, obj: Any, **kwargs: Any) -> str:
        """JSON of the masked value (a value JSON cannot hold, NaN and the infinities too, is
        written as its str()). A text pass then masks what only the serialized form shows; when
        that would break the JSON, the text of the first pass is kept, so the result is always
        valid JSON."""
        kwargs.setdefault("default", str)
        clean = json.dumps(self.scrub_obj(obj), **kwargs)
        masked = self.scrub(clean)
        if masked == clean:
            return clean
        try:
            json.loads(masked)
        except ValueError:
            return clean
        return masked


SCRUBBER = Scrubber()


def check_env_name(value: str, flag: str = "--api-key-env") -> str:
    """`value` when it is the name of an environment variable. Else ScriptError literal_secret
    (exit 2): it is not a variable name, or it looks like a key. The value is never echoed."""
    if not isinstance(value, str) or not ENV_NAME.match(value) or looks_like_key(value):
        raise ScriptError(
            "literal_secret",
            f"{flag} takes the name of an environment variable, and the value given is not a "
            "variable name or looks like a key (it is not shown here)",
            f"Put the key in an environment variable and pass its name, for example {flag} "
            "NVIDIA_API_KEY; never pass the key itself",
            EXIT_INPUT,
        )
    return value


def read_key(name: str) -> str | None:
    """The value of the environment variable `name` (checked with check_env_name), without the
    surrounding white space, registered with SCRUBBER; None when it is unset or empty. A value
    with a line break or another control character, which an HTTP header cannot carry, or one
    shorter than MIN_SECRET characters, which SCRUBBER cannot mask, is ScriptError bad_input
    (exit 2), and the value is not shown."""
    check_env_name(name)
    raw = os.environ.get(name)
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return None
    SCRUBBER.add(raw)
    SCRUBBER.add(value)
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in value):
        raise ScriptError(
            "bad_input",
            f"${name} holds a line break or another control character, which an HTTP header "
            "cannot carry (the value is not shown)",
            f"Set {name} to the key alone, on one line",
            EXIT_INPUT,
        )
    if len(value) < MIN_SECRET:
        raise ScriptError(
            "bad_input",
            f"${name} holds a key shorter than {MIN_SECRET} characters, too short to be masked "
            "in what the scripts print and save (the value is not shown)",
            f"Set {name} to the whole key; an endpoint that needs no key needs no --api-key-env",
            EXIT_INPUT,
        )
    return value


# --- URLs ----------------------------------------------------------------------------------------

BASE_URL_FIX = (
    "Pass --url the base URL of the endpoint, http(s)://HOST:PORT with an optional path prefix, "
    "for example http://127.0.0.1:8000 or https://integrate.api.nvidia.com/v1"
)


def _bad_url(message: str, fix: str = BASE_URL_FIX) -> ScriptError:
    return ScriptError("bad_input", message, fix, EXIT_INPUT)


def _ipv6_zone(host: str) -> bool:
    """Whether `host` is an IPv6 address with a zone (fe80::1%eth0)."""
    try:
        return ipaddress.IPv6Address(host).scope_id is not None
    except ValueError:
        return False


def normalize_base(url: str) -> str:
    """The base URL that the scripts append /v1/... to: http or https only, no user:pass@, no
    query or fragment, no percent-encoded host (only an IPv6 zone, [fe80::1%25eth0]); the
    surrounding white space, the trailing "/" and one trailing "/v1" are removed
    (https://integrate.api.nvidia.com/v1 -> https://integrate.api.nvidia.com).
    ScriptError bad_input (exit 2) otherwise. An error never shows the URL as given, as it may
    hold a password; at most its scheme and host."""
    text = url.strip() if isinstance(url, str) else ""
    if not text:
        raise _bad_url("the URL is empty")
    if any(ord(char) <= 0x20 or ord(char) == 0x7F for char in text):
        raise _bad_url("the URL holds a space or a control character")
    try:
        parts = urllib.parse.urlsplit(text)
    except ValueError as exc:
        raise _bad_url(f"the URL cannot be read: {exc}") from exc
    if "@" in parts.netloc:
        raise _bad_url(
            "the URL holds a user name or password (user:pass@HOST); it is not shown here",
            "Pass --url without user:pass@; for a gateway token, put it in an environment "
            "variable and pass its name with --api-key-env",
        )
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https"):
        shown = f"'{scheme}:'" if scheme else "no scheme"
        raise _bad_url(f"the URL must start with http:// or https:// (it has {shown})")
    try:
        host, port = parts.hostname, parts.port
    except ValueError as exc:
        raise _bad_url(f"the URL has a bad port or host: {exc}") from exc
    if not host:
        raise _bad_url("the URL names no host")
    # urllib.request decodes the host before it connects, so with a % the host checked here
    # is not the one reached: 127.0.0.1%2Egw.example.com is the name 127.0.0.1.gw.example.com.
    # Only an IPv6 address with a zone keeps its % once decoded ([fe80::1%25eth0]).
    if "%" in host and not _ipv6_zone(urllib.parse.unquote(host)):
        raise _bad_url(
            "the host of the URL holds a % (percent-encoding), which the connection decodes "
            "into another host name",
            "Pass --url with the host written out, without percent-encoding, for example "
            "http://127.0.0.1:8000 or https://gw.example.com",
        )
    if parts.query or parts.fragment:
        where = f"{scheme}://{host}" + (f":{port}" if port else "")
        raise _bad_url(f"the URL of {where} has a query or a fragment (? or #)")
    path = parts.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[: -len("/v1")].rstrip("/")
    return f"{scheme}://{parts.netloc}{path}"


# An IPv4 address in a form the system resolver also reads (127.1, 2130706433, 0x7f000001,
# 0177.0.0.1, 127.000.000.001): up to four parts of decimal, octal (leading 0) or hex (0x) digits.
LEGACY_IPV4 = re.compile(r"(?:0x[0-9a-f]+|[0-9]+)(?:\.(?:0x[0-9a-f]+|[0-9]+)){0,3}")


def _legacy_ipv4(host: str) -> ipaddress.IPv4Address | None:
    """The address of a numeric host in a legacy IPv4 form, read as the connection reads it
    (inet_aton: no DNS lookup); None when `host` is not one."""
    if not LEGACY_IPV4.fullmatch(host):
        return None
    try:
        return ipaddress.IPv4Address(socket.inet_aton(host))
    except (OSError, ValueError):
        return None


def connect_host(url: str) -> str:
    """The host a request to `url` connects to, as urllib reads it: without brackets,
    percent-decoded (urllib.request decodes the host, so 127.0.0.1%2Egw.example.com is the name
    127.0.0.1.gw.example.com, not an address) and in lower case; "" when the URL has no host or
    cannot be read."""
    try:
        host = urllib.parse.urlsplit(url).hostname
    except ValueError:
        return ""
    return urllib.parse.unquote(host or "").lower()


def is_loopback(url: str) -> bool:
    """Whether the URL's host is this machine, by name or address, without a DNS lookup:
    localhost (and *.localhost), 127.0.0.0/8, ::1 (also as ::ffff:127.x.x.x), and the unspecified
    0.0.0.0 and ::, which reach this host too; an IPv4 address also in the short, decimal, octal
    or hex forms the connection accepts (127.1, 2130706433, 0x7f000001). The host is read as the
    connection reads it (connect_host), and only an IPv6 literal has a zone (::1%lo). A
    port-forward URL is loopback."""
    host = connect_host(url).rstrip(".")
    if not host:
        return False
    if host == "localhost" or host.endswith(".localhost"):
        return True
    address: ipaddress.IPv4Address | ipaddress.IPv6Address | None
    try:
        address = ipaddress.ip_address(host.split("%", 1)[0] if ":" in host else host)
    except ValueError:
        address = _legacy_ipv4(host)
        if address is None:
            return False
    mapped = getattr(address, "ipv4_mapped", None)
    if mapped is not None:
        address = mapped
    return address.is_loopback or address.is_unspecified


def origin(url: str) -> tuple[str, str, int] | None:
    """The scheme, host and port of an http(s) URL: the scheme and host in lower case, and the
    scheme's default port when none is written. None when the URL cannot be read or is not
    http(s). The host is kept as written, final dot included: gw.example.com. is looked up as
    it is, while gw.example.com may get the resolver's search domains first, so the two can
    reach different servers and are two origins."""
    try:
        parts = urllib.parse.urlsplit(url)
        port = parts.port
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    host = parts.hostname or ""
    if scheme not in ("http", "https") or not host:
        return None
    return scheme, host, port if port is not None else (443 if scheme == "https" else 80)


def same_origin(url: str, other: str) -> bool:
    """Whether two URLs have the same scheme, host and port (origin); the path does not count."""
    found = origin(url)
    return found is not None and found == origin(other)


def strip_userinfo(url: str) -> str:
    """The URL without user:pass@, for a URL a server sent (a redirect's Location)."""
    return re.sub(r"^([A-Za-z][A-Za-z0-9+.-]*://)[^/@\s]*@", r"\1", url)


# --- TLS ----------------------------------------------------------------------------------------


def _expand(path: str | None) -> str | None:
    return os.path.expanduser(path) if path else None


def _no_password() -> bytes:
    """The passphrase callback of load_cert_chain: an empty one, so that OpenSSL never prompts
    on the terminal for the passphrase of an encrypted key; loading that key fails instead."""
    return b""


@dataclass
class Tls:
    """TLS settings of the requests (--ca-file, --client-cert, --client-key). Verification is
    always on: the certificate chain and the host name are checked, against the system CA store,
    or against the CA file alone when one is given. No setting turns it off. The context is built
    once; a file that cannot be loaded is ScriptError bad_input (exit 2)."""

    ca_file: str | None = None
    client_cert: str | None = None
    client_key: str | None = None
    _context: ssl.SSLContext | None = field(default=None, init=False, repr=False, compare=False)

    def context(self) -> ssl.SSLContext:
        if self._context is None:
            self._context = self._build()
        return self._context

    def _build(self) -> ssl.SSLContext:
        if self.client_key and not self.client_cert:
            raise ScriptError(
                "bad_input",
                "--client-key needs --client-cert",
                "Pass the client certificate with --client-cert as well, or leave out --client-key",
            )
        ca_file = _expand(self.ca_file)
        try:
            context = ssl.create_default_context(cafile=ca_file)
        except OSError as exc:  # ssl.SSLError is an OSError
            raise ScriptError(
                "bad_input",
                f"cannot load the CA file {ca_file}: {_short(exc)}",
                "Pass --ca-file a readable PEM file with the CA certificates of the endpoint",
            ) from exc
        if self.client_cert:
            cert, key = _expand(self.client_cert), _expand(self.client_key)
            try:
                context.load_cert_chain(cert, key, password=_no_password)
            except OSError as exc:
                raise ScriptError(
                    "bad_input",
                    f"cannot load the client certificate {cert}: {_short(exc)}",
                    "Pass --client-cert (and --client-key) readable PEM files; a key protected "
                    "by a passphrase is not supported",
                ) from exc
        context.check_hostname = True
        context.verify_mode = ssl.CERT_REQUIRED
        return context


DEFAULT_TLS = Tls()  # the system CA store, verification on


# --- HTTP ---------------------------------------------------------------------------------------


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """Follows no redirect: urllib then raises HTTPError with the 3xx status, which the request
    helpers report as redirect_refused. Following one would send the request, and its key, to a
    host the user never named (references/security.md)."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None

    def http_error_302(self, *args: Any, **kwargs: Any) -> None:
        return None

    http_error_301 = http_error_303 = http_error_307 = http_error_308 = http_error_302


def _time_left(deadline: float) -> float:
    """The seconds left until `deadline` (a time.monotonic() value); TimeoutError when none."""
    left = deadline - time.monotonic()
    if left <= 0:
        raise TimeoutError("the request did not end in time")
    return left


class _InTime(io.RawIOBase):
    """The socket reads of one answer, each given only the time left until `deadline`. A
    socket's timeout counts for one read, so an answer that keeps arriving, a byte at a time,
    never reaches it: this ends such an answer (its status line, headers or body) at the
    deadline, with TimeoutError."""

    def __init__(self, raw: Any, sock: Any, deadline: float) -> None:
        super().__init__()
        self._raw = raw  # the socket's own reader (socket.makefile)
        self._sock = sock
        self._deadline = deadline

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Any) -> int | None:
        self._sock.settimeout(_time_left(self._deadline))
        return self._raw.readinto(buffer)

    def close(self) -> None:
        if not self.closed:
            self._raw.close()
        super().close()


class _Answer(http.client.HTTPResponse):
    """An answer that reads its status line, headers and body through _InTime."""

    def __init__(self, sock: Any, *args: Any, deadline: float, **kwargs: Any) -> None:
        super().__init__(sock, *args, **kwargs)
        self.fp = io.BufferedReader(_InTime(self.fp.detach(), sock, deadline))


def _connect_in_time(
    deadline: float,
    address: tuple[str, int],
    timeout: object = None,
    source_address: tuple[str, int] | None = None,
) -> socket.socket:
    """socket.create_connection, but each address of the host gets only the time left until
    `deadline` (`timeout` is not used). create_connection gives each address the whole timeout,
    so a host name with several addresses that do not answer would take that long for each. As
    in create_connection, an address that fails is skipped and the last failure is raised;
    TimeoutError once no time is left."""
    host, port = address
    failure: OSError | None = None
    for family, kind, proto, _, where in socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM):
        left = _time_left(deadline)
        sock = None
        try:
            sock = socket.socket(family, kind, proto)
            sock.settimeout(left)
            if source_address:
                sock.bind(source_address)
            sock.connect(where)
            return sock
        except OSError as exc:  # TimeoutError too
            if sock is not None:
                sock.close()
            failure = exc
    raise failure or OSError(f"no address found for {host}")


class _Connection(http.client.HTTPConnection):
    """The connection of one request, which ends by its `deadline` (_connector sets it): each
    address of the host gets only the time left (_connect_in_time), the socket then gets what
    is left after the connection (for the TLS handshake of _TlsConnection and for sending the
    request), and the answer is an _Answer."""

    deadline: float

    def connect(self) -> None:
        self.timeout = _time_left(self.deadline)
        # HTTPConnection.connect makes the TCP connection with self._create_connection
        self._create_connection = functools.partial(_connect_in_time, self.deadline)
        super().connect()
        self.sock.settimeout(_time_left(self.deadline))


class _TlsConnection(http.client.HTTPSConnection, _Connection):
    """_Connection over TLS. HTTPSConnection.connect calls _Connection.connect (the next class
    in this order) for the TCP connection, then makes the handshake, which Python bounds as a
    whole by the socket's timeout."""


def _connector(
    kind: type[_Connection], deadline: float, **fixed: Any
) -> Callable[..., _Connection]:
    """What urllib's do_open calls to make the connection of one request: a `kind` connection
    with the request's deadline, and `fixed` arguments (the TLS context)."""

    def connect(host: str, **kwargs: Any) -> _Connection:
        connection = kind(host, **fixed, **kwargs)
        connection.deadline = deadline
        connection.response_class = functools.partial(_Answer, deadline=deadline)
        return connection

    return connect


class _HttpHandler(urllib.request.HTTPHandler):
    """Opens an http URL with a _Connection that ends by `deadline`."""

    def __init__(self, deadline: float) -> None:
        super().__init__()
        self.deadline = deadline

    def http_open(self, req: urllib.request.Request) -> Any:
        return self.do_open(_connector(_Connection, self.deadline), req)


class _HttpsHandler(urllib.request.HTTPSHandler):
    """Opens an https URL with a _TlsConnection that ends by `deadline` and verifies with
    `context`."""

    def __init__(self, context: ssl.SSLContext, deadline: float) -> None:
        super().__init__(context=context)
        self.tls_context = context
        self.deadline = deadline

    def https_open(self, req: urllib.request.Request) -> Any:
        connect = _connector(_TlsConnection, self.deadline, context=self.tls_context)
        return self.do_open(connect, req)


def open_url(
    request: urllib.request.Request,
    *,
    timeout: float,
    tls: Tls | None = None,
    loopback: bool = False,
    deadline: float | None = None,
) -> Any:
    """Open `request`: the only place the scripts open a URL. http and https only (ValueError
    otherwise); no redirect is followed (a 3xx raises HTTPError); TLS is verified with `tls`
    (default: the system CA store). For a loopback URL (`loopback`, or a loopback host) the proxy
    settings of the environment are ignored, as a proxy cannot reach this machine's loopback;
    other URLs use them, as urllib does. `deadline` (a time.monotonic() value; default: `timeout`
    seconds from now) ends the whole request: the TCP connection, the TLS handshake, the sending
    and every read of the answer get only the time left, so an answer whose status line, headers
    or body keep arriving slowly still ends then (TimeoutError). Only the name lookup has no
    limit of its own: the system resolver's."""
    scheme = urllib.parse.urlsplit(request.full_url).scheme.lower()
    if scheme not in ("http", "https"):
        shown = scheme or "a URL with no scheme"
        raise ValueError(f"only http and https URLs are opened, not {shown}")
    deadline = time.monotonic() + timeout if deadline is None else deadline
    handlers: list[urllib.request.BaseHandler] = [
        NoRedirect(),
        _HttpHandler(deadline),
        _HttpsHandler((tls or DEFAULT_TLS).context(), deadline),
    ]
    if loopback or is_loopback(request.full_url):
        handlers.append(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener(*handlers).open(request, timeout=timeout)


@dataclass
class HttpResult:
    """One request and its answer.

    status: the HTTP status; 0 means no answer, and `error` says why.
    headers: the answer's headers, with lower-case names.
    body: the raw body bytes, at most max_bytes (an event stream: the raw SSE bytes).
    elapsed_ms: from the start of the request to the end of the body; ttfb_ms: to the first body
    byte (None when no body byte came).
    error: None, or a code from the ERR_* constants, some followed by ": detail": a 3xx is
    "redirect_refused" (status holds the code); a 4xx or 5xx is no error (judge the status).
    request_id: the X-Request-Id the server sent back, else the one sent.
    """

    status: int
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""
    elapsed_ms: float = 0.0
    ttfb_ms: float | None = None
    error: str | None = None
    request_id: str | None = None

    @property
    def error_code(self) -> str | None:
        """The code part of `error` ("timeout", "connection_failed", ...)."""
        return self.error.split(":", 1)[0] if self.error else None

    @property
    def content_type(self) -> str:
        """The media type of the body, lower case, without parameters ("" when not sent)."""
        return self.headers.get("content-type", "").split(";", 1)[0].strip().lower()

    def json_body(self) -> Any:
        """The body parsed as JSON (parse_json); None when it is empty, not JSON, or nested
        deeper than MAX_JSON_DEPTH levels."""
        try:
            return parse_json(self.body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            return None


def _no_constant(name: str) -> NoReturn:
    raise ValueError(f"{name} is not a JSON number")


def _finite(text: str) -> float:
    number = float(text)
    if not math.isfinite(number):
        raise ValueError("a number in the JSON is too large for a float")
    return number


def parse_json(text: str | bytes, limit: int = MAX_JSON_DEPTH) -> Any:
    """`text` parsed as JSON, for text a server sent: ValueError when it is not JSON, or when its
    lists and objects nest deeper than `limit` levels. json.loads raises RecursionError on deep
    nesting, and the walkers that mask and cut a value recurse once a level: a broken or hostile
    answer is then reported as not JSON, and never ends the run. NaN, Infinity and -Infinity,
    which json.loads takes but JSON does not have, and a number too large for a float (1e999)
    are not JSON either, so no such value reaches the output, where JSON readers would refuse
    it."""
    try:
        value = json.loads(text, parse_constant=_no_constant, parse_float=_finite)
    except RecursionError as exc:
        raise ValueError("the JSON is nested too deeply to read") from exc
    if not _within_depth(value, limit):
        raise ValueError(f"the JSON is nested deeper than {limit} levels")
    return value


def _within_depth(value: Any, limit: int) -> bool:
    """Whether `value` nests lists and objects at most `limit` levels deep (walked with a list,
    not by recursion)."""
    stack = [(value, 1)] if isinstance(value, (dict, list)) else []
    while stack:
        item, depth = stack.pop()
        if depth > limit:
            return False
        children = item.values() if isinstance(item, dict) else item
        stack.extend((child, depth + 1) for child in children
                     if isinstance(child, (dict, list)))
    return True


def request_json(
    method: str,
    url: str,
    body: Any = None,
    *,
    headers: Mapping[str, str | None] | None = None,
    timeout: float,
    tls: Tls | None = None,
    loopback: bool = False,
    max_bytes: int = MAX_JSON_BYTES,
) -> HttpResult:
    """GET (no body) or POST (`body`: a JSON value, or bytes sent as they are) and read the whole
    answer, within `timeout` seconds in all. Headers: User-Agent, Accept: application/json,
    Content-Type: application/json on a POST, then `headers` (an Authorization header is added
    with add_unredirected_header; a None value is skipped). Never raises for a network problem:
    see HttpResult."""
    if method not in ("GET", "POST"):
        raise ValueError(f"the scripts send only GET and POST, not {method}")
    if method == "GET" and body is not None:
        raise ValueError("a GET sends no body")
    data = _encode(body)
    return _exchange(
        method,
        url,
        data,
        headers,
        "application/json",
        timeout=timeout,
        tls=tls,
        loopback=loopback,
        max_bytes=max_bytes,
        until_done=False,
    )


def request_stream(
    url: str,
    body: Any,
    *,
    headers: Mapping[str, str | None] | None = None,
    timeout: float,
    tls: Tls | None = None,
    loopback: bool = False,
    max_bytes: int = MAX_STREAM_BYTES,
) -> HttpResult:
    """POST `body` with Accept: text/event-stream and read the answer as it arrives: ttfb_ms is
    the time to the first body byte, and body holds the raw bytes. Reading stops when the stream
    ends, after a `data: [DONE]` line, after `timeout` seconds in all (error "timeout"), or past
    `max_bytes` (error "body_too_large", body cut to max_bytes). An answer that ignores
    `stream: true` is read the same way. Never raises for a network problem."""
    return _exchange(
        "POST",
        url,
        _encode(body),
        headers,
        "text/event-stream",
        timeout=timeout,
        tls=tls,
        loopback=loopback,
        max_bytes=max_bytes,
        until_done=True,
    )


def _encode(body: Any) -> bytes | None:
    """The bytes of a request body: bytes as they are, None as no body, anything else as JSON."""
    if body is None or isinstance(body, bytes):
        return body
    return json.dumps(body).encode("utf-8")


def _exchange(
    method: str,
    url: str,
    data: bytes | None,
    headers: Mapping[str, str | None] | None,
    accept: str,
    *,
    timeout: float,
    tls: Tls | None,
    loopback: bool,
    max_bytes: int,
    until_done: bool,
) -> HttpResult:
    timeout = float(timeout)
    if not (timeout > 0 and math.isfinite(timeout)):
        raise ValueError(f"the timeout must be a positive number of seconds, not {timeout}")
    started = time.monotonic()
    deadline = started + timeout  # one deadline for connecting, the headers and the body
    sent_id: str | None = None
    try:
        request = urllib.request.Request(url, data=data, method=method)
        request.add_header("User-Agent", USER_AGENT)
        request.add_header("Accept", accept)
        if data is not None:
            request.add_header("Content-Type", "application/json")
        for name, value in (headers or {}).items():
            if value is None:
                continue
            if name.lower() == "authorization":
                request.add_unredirected_header("Authorization", value)
            else:
                request.add_header(name, value)
        sent_id = request.get_header("X-request-id")
        response = open_url(
            request, timeout=timeout, tls=tls, loopback=loopback, deadline=deadline
        )
    except urllib.error.HTTPError as exc:  # an answer: 3xx, 4xx or 5xx
        response = exc
    except (OSError, ValueError, http.client.HTTPException) as exc:  # URLError is an OSError
        return HttpResult(0, {}, b"", _ms_since(started), None, _failure(exc), sent_id)
    if isinstance(response, urllib.error.HTTPError):
        status, stream = response.code, response.fp
    else:
        status, stream = response.status, response
    try:
        received = _lower_headers(response.headers)
        body, ttfb, failure = b"", None, None
        if stream is not None:
            body, ttfb, failure = _read_body(stream, started, deadline, max_bytes, until_done)
    finally:
        with contextlib.suppress(Exception):
            response.close()
    if 300 <= status < 400:
        failure = ERR_REDIRECT
    request_id = received.get("x-request-id") or sent_id
    return HttpResult(status, received, body, _ms_since(started), ttfb, failure, request_id)


DONE_LINES = (b"data: [DONE]", b"data:[DONE]")


def _read_body(
    stream: Any, started: float, deadline: float, max_bytes: int, until_done: bool
) -> tuple[bytes, float | None, str | None]:
    """Read `stream` as it arrives (read1: what one read returns) until it ends, the deadline
    passes, more than `max_bytes` arrive, or (`until_done`) a `data: [DONE]` line arrives.
    Returns (the body cut to max_bytes, ms to the first body byte, the error or None). An end
    before the Content-Length the answer declared is connection_lost (_cut_short)."""
    read = getattr(stream, "read1", None) or stream.read
    chunks: list[bytes] = []
    size = 0
    ttfb: float | None = None
    failure: str | None = None
    pending = b""  # the unfinished last line, for until_done
    while True:
        if time.monotonic() >= deadline:
            failure = ERR_TIMEOUT
            break
        try:
            chunk = read(min(READ_CHUNK, max_bytes + 1 - size))
        except TimeoutError:  # socket.timeout too; each socket read gets only the time left
            failure = ERR_TIMEOUT
            break
        except http.client.IncompleteRead as exc:
            chunk = exc.partial
            failure = f"{ERR_LOST}: the connection closed in the middle of the body"
        except (OSError, ValueError, http.client.HTTPException) as exc:
            failure = f"{ERR_LOST}: {_short(exc)}"
            break
        if chunk:
            if ttfb is None:
                ttfb = _ms_since(started)
            chunks.append(chunk)
            size += len(chunk)
        elif not failure:
            failure = _cut_short(stream)
        if failure or not chunk:
            break
        if size > max_bytes:
            failure = ERR_TOO_LARGE
            break
        if until_done:
            done, pending = _done_seen(pending, chunk)
            if done:
                break
    return b"".join(chunks)[:max_bytes], ttfb, failure


def _done_seen(pending: bytes, chunk: bytes) -> tuple[bool, bytes]:
    """Whether a complete `data: [DONE]` line is in `pending` + `chunk`, and the unfinished last
    line to keep (a line too long to be that line is kept as one byte that never matches)."""
    lines = (pending + chunk).split(b"\n")
    rest = lines.pop()
    if any(line.rstrip() in DONE_LINES for line in lines):
        return True, b""
    return False, (b"\x00" if len(rest) > 64 else rest)


def _cut_short(stream: Any) -> str | None:
    """connection_lost when `stream` ended before the Content-Length it declared (http.client's
    read1 then returns b"" with no error, and `length` keeps the bytes still owed); None when it
    ended where it should, or declared no length (the close ends such a body)."""
    left = getattr(stream, "length", None)
    if isinstance(left, int) and left > 0:
        return f"{ERR_LOST}: the connection closed {left} bytes before the end of the body"
    return None


def _lower_headers(message: Any) -> dict[str, str]:
    found: dict[str, str] = {}
    for name, value in message.items() if message is not None else []:
        key = name.lower()
        found[key] = f"{found[key]}, {value}" if key in found else value
    return found


def _ms_since(started: float) -> float:
    return round((time.monotonic() - started) * 1000, 1)


def clip(text: object, limit: int) -> str:
    """`text` on one line, masked by SCRUBBER, then cut to `limit` characters. Masking comes
    first: a cut can then never keep the first part of a key that the whole value would show."""
    return SCRUBBER.scrub(" ".join(str(text).split()))[: max(int(limit), 0)]


def _short(exc: BaseException) -> str:
    """A one-line reason of at most 300 characters, masked."""
    if isinstance(exc, OSError) and not isinstance(exc, ssl.SSLError) and exc.strerror:
        text = exc.strerror
    else:
        text = str(exc) or type(exc).__name__
    return clip(text, 300)


def _failure(exc: BaseException) -> str:
    """The HttpResult.error of a request that got no answer."""
    reason: Any = exc.reason if isinstance(exc, urllib.error.URLError) else exc
    if isinstance(reason, TimeoutError):
        return ERR_TIMEOUT
    if isinstance(reason, ssl.SSLError):  # a failed certificate check too
        return f"{ERR_TLS}: {_short(reason)}"
    if isinstance(reason, ValueError):  # http.client.InvalidURL too
        return f"{ERR_REQUEST}: {_short(reason)}"
    if isinstance(reason, BaseException):
        return f"{ERR_CONNECT}: {_short(reason)}"
    return f"{ERR_CONNECT}: {clip(reason, 300)}"


# --- private files ------------------------------------------------------------------------------

PRIVATE_FIX = f"Pass --out a folder of your own, such as one under {CACHE_FOLDER}/ (the default)"


def private_group(gid: int) -> bool:
    """Whether the group `gid` is the user's private group, so that its write permission lets no
    one else in: the user's primary group, named after the user, with no other member and no
    other account that has it as its primary group (of the accounts the system lists). A group
    such as `users`, which many accounts share, is not; nor is a group the system cannot name."""
    if not hasattr(os, "getuid") or gid != os.getgid():
        return False
    try:
        import grp
        import pwd

        user = pwd.getpwuid(os.getuid())
        group = grp.getgrgid(gid)
        accounts = pwd.getpwall()
    except (ImportError, KeyError, OSError):
        return False
    if group.gr_name != user.pw_name or any(name != user.pw_name for name in group.gr_mem):
        return False
    return not any(
        account.pw_gid == gid and account.pw_uid != user.pw_uid for account in accounts
    )


def check_private(path: Path) -> Path:
    """The existing folder `path` with its symbolic links followed: the folder to write in,
    which this checks. Raise ScriptError out_not_private (exit 2) when another user could
    change the files written in it, or choose where they go:
      - an entry of `path` as written (read with lstat, not followed) is a symbolic link of
        another user (root aside);
      - a folder on the way belongs to another user (root aside), or other users can write in
        it (others, or a group that is not the user's private group: private_group) and it has
        no sticky bit. The folders on the way are the folder found, each folder that holds a
        link of `path`, and the folders above them. A sticky folder such as /tmp lets no one
        replace another user's entry.
    No check on a system without POSIX owners."""
    asked = Path(path).expanduser().absolute()
    resolved = asked.resolve()
    if not hasattr(os, "getuid"):
        return resolved
    uid = os.getuid()
    folders = dict.fromkeys((resolved, *resolved.parents))
    for part in (asked, *asked.parents):
        info = os.lstat(part)
        if not stat.S_ISLNK(info.st_mode):
            continue
        if info.st_uid not in (uid, 0):
            raise ScriptError(
                "out_not_private",
                f"{part} is a symbolic link of another user, who could choose where the files "
                f"of {path} go",
                PRIVATE_FIX,
                EXIT_INPUT,
            )
        holder = part.parent.resolve()  # who can write here can replace the link
        folders.update(dict.fromkeys((holder, *holder.parents)))
    for part in folders:
        info = part.stat()
        mode = info.st_mode
        shared = mode & stat.S_IWOTH or (mode & stat.S_IWGRP and not private_group(info.st_gid))
        if info.st_uid not in (uid, 0):
            problem = "belongs to another user"
        elif shared and not mode & stat.S_ISVTX:
            problem = "can be written by other users"
        else:
            continue
        raise ScriptError(
            "out_not_private",
            f"{part} {problem}, so another user could change the files written in {path}",
            PRIVATE_FIX,
            EXIT_INPUT,
        )
    return resolved


def ensure_private_dir(path: Path) -> Path:
    """The folder `path` (a leading ~ is the home folder), created when missing, and returned
    resolved: the folder check_private checked last, so a link changed after that check cannot
    move the files. The nearest existing folder is checked before anything is created in it, and
    every folder created gets mode 0700. ScriptError out_not_private (check_private), or
    bad_input when a part is not a folder or cannot be created (both exit 2)."""
    folder = Path(path).expanduser().absolute()
    missing: list[Path] = []
    probe = folder
    while not probe.exists() and probe != probe.parent:
        missing.append(probe)
        probe = probe.parent
    if not probe.is_dir():
        raise ScriptError(
            "bad_input", f"{probe} is not a folder", "Pass --out a folder path", EXIT_INPUT
        )
    check_private(probe)
    for part in reversed(missing):
        try:
            os.mkdir(part, 0o700)
        except FileExistsError:
            pass  # made meanwhile, or a dangling symlink: checked below
        except OSError as exc:
            raise ScriptError(
                "bad_input",
                f"cannot create the folder {part}: {exc.strerror}",
                "Pass --out a folder you can write",
                EXIT_INPUT,
            ) from exc
        else:
            os.chmod(part, 0o700)
    if not folder.is_dir():
        raise ScriptError(
            "bad_input", f"{folder} is not a folder", "Pass --out a folder path", EXIT_INPUT
        )
    return check_private(folder)


def _write_private(path: Path, text: str) -> None:
    path = Path(path).expanduser()
    folder = ensure_private_dir(path.parent)
    handle, temporary = tempfile.mkstemp(dir=folder, prefix=f".{path.name}.", suffix=".tmp")
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(handle, 0o600)
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(temporary, folder / path.name)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise


def write_json_file(path: Path, obj: Any) -> None:
    """Write `obj` as indented JSON, masked by SCRUBBER: to a new 0600 temporary file in the same
    private folder (ensure_private_dir), which then replaces `path`, so a symlink planted at
    `path` is replaced, never followed. ScriptError for the folder; OSError when a write fails."""
    _write_private(Path(path), SCRUBBER.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def write_text_file(path: Path, text: str) -> None:
    """Write `text`, masked by SCRUBBER, the same way as write_json_file."""
    _write_private(Path(path), SCRUBBER.scrub(text))


def model_slug(model: str | None, limit: int = 64) -> str:
    """A folder-name form of a model id: meta/llama-3.1-8b-instruct -> meta-llama-3.1-8b-instruct
    (lower case; [a-z0-9._-] only; "nim" when nothing is left)."""
    text = re.sub(r"[^a-z0-9._-]+", "-", (model or "").lower()).strip("-._")
    return text[:limit].strip("-._") or "nim"


def default_out_dir(model: str | None, now: datetime) -> Path:
    """~/.cache/nim-smoke-check/<UTC yyyymmddThhmmssZ>-<model slug> (a naive `now` is taken as
    UTC). Not created here: ensure_private_dir does that."""
    moment = now if now.tzinfo is None else now.astimezone(timezone.utc)
    stamp = moment.strftime("%Y%m%dT%H%M%SZ")
    return Path(CACHE_FOLDER).expanduser() / f"{stamp}-{model_slug(model)}"


def main(argv: list[str] | None = None) -> None:
    parser = JsonArgumentParser(
        description="Shared helpers for the nim-smoke-check scripts; import it, do not run it.",
        epilog="Example: python3 scripts/nsc_common.py --help",
    )
    parser.parse_args(argv)
    emit(True, "library", "done", {"note": "import this module; it has no command of its own"})


if __name__ == "__main__":
    main()
