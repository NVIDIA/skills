#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the nim-select scripts: `import ns_common as ns`.

Holds the JSON output contract (one object on stdout), the exit codes, the error that library code
raises for a script to report, the JSON files of --out and of the list's cache (written only in a
folder no other user can change, and read back only when no other user could have changed them),
the one way the scripts open a URL (open_url: no redirect is followed, so a request goes only to
the host it names), one piece of work run in a thread with a time budget (within), the word
match of a catalog search or a --query (every_word_in) and small parsers. Stdlib only. Run
directly, it prints a notice: it has no command of its own.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shlex
import stat
import sys
import tempfile
import threading
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, NoReturn, TypeVar

EXIT_OK, EXIT_INPUT, EXIT_PRECONDITION, EXIT_EXTERNAL, EXIT_REFUSED = 0, 2, 3, 4, 5
T = TypeVar("T")


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
    """Progress and diagnostics go to stderr; stdout carries only the JSON object."""
    print(message, file=sys.stderr)


def emit(
    ok: bool,
    state: str,
    next_: str,
    facts: dict[str, Any] | None = None,
    errors: list[dict[str, str]] | None = None,
    code: int | None = None,
) -> NoReturn:
    """Print the one JSON object of the script-output contract and exit."""
    payload = {
        "ok": ok,
        "state": state,
        "next": next_,
        "facts": facts or {},
        "errors": errors or [],
    }
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
    sys.stdout.flush()
    raise SystemExit(code if code is not None else (EXIT_OK if ok else EXIT_INPUT))


NOT_UTF8_FIX = (
    "Pass the file a nim-select command wrote with --out, or save the file as UTF-8 (Windows "
    "PowerShell's > and Out-File write UTF-16: use Set-Content -Encoding utf8)"
)


def load_json_file(path: Path, code: str = "bad_input") -> Any:
    """Read a JSON file (a leading ~ is the home folder; UTF-8, with or without a byte order
    mark), or raise a ScriptError (exit 2) that names the file: one that cannot be read, is not
    UTF-8 text (UTF-16, as Windows PowerShell's > writes it) or is not JSON."""
    try:
        return json.loads(path.expanduser().read_text(encoding="utf-8-sig"))
    except OSError as exc:
        raise ScriptError(
            code, f"cannot read {path}: {exc.strerror}", f"Pass a readable file: {path}"
        ) from exc
    except UnicodeDecodeError as exc:
        raise ScriptError(
            code,
            f"{path} is not UTF-8 text (byte {exc.start}: {exc.reason}); a UTF-16 file cannot be "
            "read",
            NOT_UTF8_FIX,
        ) from exc
    except json.JSONDecodeError as exc:
        raise ScriptError(
            code,
            f"{path}:{exc.lineno}: not valid JSON ({exc.msg})",
            "Pass the file a nim-select command wrote with --out",
        ) from exc
    except (ValueError, RecursionError) as exc:  # a number too long, or nesting too deep
        raise ScriptError(
            code,
            f"{path}: not valid JSON ({type(exc).__name__}: {str(exc)[:200]})",
            "Pass the file a nim-select command wrote with --out",
        ) from exc


def check_private(folder: Path) -> Path:
    """Raise ScriptError out_not_private when another user could replace a file written in
    `folder`: the folder, or one above it, belongs to another user (root aside), or a group or
    other users can write in it, and it has no sticky bit. A sticky folder such as /tmp lets no
    one replace a file of another user. A folder a group can write is refused even when the group
    is the user's own: whether another account is in a group cannot be verified everywhere
    (directory services, initgroups), and with a POSIX ACL the group bits show the ACL's mask,
    which bounds every user and group the ACL names. For a folder of the user's, the fix names it
    and the chmod that leaves it writable by the user alone; root, which owns the system folders,
    is sent to a folder of its own instead. Each folder is read as it is (lstat), never through a
    symlink: one that is no folder now changed after `resolve` (another user can swap a folder of
    theirs for a symlink to a folder of the user's) and is refused. So once the check passes, no
    other user can rename or replace a folder on the way, as far as the owners and the mode bits
    show (an ACL of another kind, as on macOS, is not read). Return the folder checked, its
    symlinks resolved, to write in: a symlink on the way could change after the check. No check
    on a system without POSIX owners."""
    resolved = folder.resolve()
    if not hasattr(os, "getuid"):
        return resolved
    uid = os.getuid()
    for part in (resolved, *resolved.parents):
        info = part.lstat()
        mode = info.st_mode
        shared = mode & (stat.S_IWGRP | stat.S_IWOTH)  # with a POSIX ACL, the group bits: its mask
        fix = "Pass --out a file in a folder of your own, such as ~/.cache/nim-select/"
        if not stat.S_ISDIR(mode):
            problem = "is no longer a folder (it changed during the check)"
        elif info.st_uid not in (uid, 0):
            problem = "belongs to another user"
        elif shared and not mode & stat.S_ISVTX:
            problem = "can be written by other users"
            # The user's own folder: the user can take that access away. Not as root: root owns
            # the system folders, and a group shares some of them on purpose (/var/local is
            # root:staff 2775 on Debian), so root keeps the fix above.
            if info.st_uid == uid != 0:
                who = "go" if mode & stat.S_IWOTH else "g"
                fix = (
                    f"Ask the user to run chmod {who}-w {shlex.quote(str(part))}, so that only "
                    "they can write in it, or pass --out a file in a folder only the user can "
                    "write"
                )
        else:
            continue
        raise ScriptError(
            "out_not_private",
            f"{part} {problem}, so another user could replace the file written in {folder}",
            fix,
        )
    return resolved


def make_folders(folder: Path) -> None:
    """Create `folder` and each missing folder above it, one level at a time from the top, each
    with mode 0700. Path.mkdir(parents=True) gives the folders above the last one 0777 less the
    umask: with a umask of 002 a new ~/.cache would be group-writable, which check_private
    refuses unless the group is the user's private group. A folder that exists is left as it
    is. OSError when one cannot be created."""
    missing: list[Path] = []
    while not folder.is_dir() and folder.parent != folder:
        missing.append(folder)
        folder = folder.parent
    for part in reversed(missing):
        try:
            part.mkdir(mode=0o700)
        except OSError:  # one made meanwhile is fine; EEXIST can come as EACCES or EROFS
            if not part.is_dir():
                raise


def write_json_file(path: Path, data: Any, compact: bool = False) -> None:
    """Write `data` as indented JSON (`compact`: on one line; a leading ~ in `path` is the home
    folder). Each missing folder on the way is created private (0700, make_folders); a folder
    that another user could change is refused (check_private), and the file is written in the
    folder checked. The JSON goes to a new temporary file (mode 0600) that then replaces `path`,
    so a symlink at `path` is replaced, never followed. OSError when the folder or the file cannot
    be written."""
    path = path.expanduser()
    folder = path.parent
    make_folders(folder)
    folder = check_private(folder)  # resolved; refused when another user can change a folder
    path = folder / path.name
    if compact:
        text = json.dumps(data, separators=(",", ":"), sort_keys=True) + "\n"
    else:
        text = json.dumps(data, indent=1, sort_keys=True) + "\n"
    handle, temporary = tempfile.mkstemp(dir=folder, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise


PRIVATE_FOLDER = "~/.cache/nim-select"  # the skill's own folder for its files
MAX_PRIVATE_FILE = 64_000_000  # bytes read of one private file


def full_path(path: str) -> str:
    """The absolute path of a file named on the command line (a leading ~ is the home folder)."""
    return str(Path(path).expanduser().resolve())


def private_folder() -> Path:
    """~/.cache/nim-select, the folder SKILL.md keeps the files in; the list keeps its cache here."""
    return Path(PRIVATE_FOLDER).expanduser()


def read_private_json(path: Path) -> tuple[Any, str | None]:
    """(JSON data, None) of a file write_json_file wrote; (None, None) when there is no file; and
    (None, why) when it is refused or cannot be used: a symlink, not a regular file, another user's,
    one that the group or others can write, in a folder another user could change (check_private),
    larger than MAX_PRIVATE_FILE, or not JSON. The file is opened without following a symlink, and
    the opened file must be the one checked."""
    path = path.expanduser()
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return None, None
    except OSError as exc:
        return None, f"cannot read {path}: {exc.strerror}"
    if stat.S_ISLNK(info.st_mode):
        return None, f"{path} is a symlink"
    if not stat.S_ISREG(info.st_mode):
        return None, f"{path} is not a regular file"
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        return None, f"{path} belongs to another user"
    if info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        return None, f"{path} can be written by other users"
    if info.st_size > MAX_PRIVATE_FILE:
        return None, f"{path} is larger than {MAX_PRIVATE_FILE:,} bytes"
    try:
        check_private(path.parent)
        handle = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except ScriptError as exc:
        return None, exc.message
    except OSError as exc:
        return None, f"cannot read {path}: {exc.strerror}"
    with os.fdopen(handle, "rb") as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
            return None, f"{path} was replaced while it was read"
        raw = stream.read(MAX_PRIVATE_FILE + 1)
    try:
        return json.loads(raw.decode("utf-8")), None
    except (ValueError, RecursionError):  # not UTF-8, not JSON, a number too long, too deep
        return None, f"{path} is not valid JSON"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """Follows no redirect. urllib then raises HTTPError with the 3xx status, which the caller
    reports: a redirect is a proxy or a portal answering, and following it would send the request
    to a host the scripts never name (references/security.md)."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


def open_url(request: urllib.request.Request, timeout: float) -> Any:
    """Open `request` without following a redirect (a 3xx answer raises HTTPError). The proxy
    settings of the environment apply, as with urllib.request.urlopen."""
    return urllib.request.build_opener(NoRedirect).open(request, timeout=timeout)


def within(
    budget: float, work: Callable[[], T], late: Callable[[str], T], *, reraise: bool = False
) -> T:
    """work() run in a daemon thread: its result, or late(why) when it takes longer than `budget`
    seconds (a thread still waiting is left behind and ends with the process), with why
    `did not answer within BUDGET s`. When work() fails, with `reraise` (cnd.deadline) its
    exception, whatever it is, is raised again here; without (registry_labels.within) an
    Exception goes to late(why) at once, with why `failed (TYPE: MESSAGE)`: a bug or an odd
    answer must not fail the command."""
    box: list[T] = []
    raised: list[BaseException] = []
    failed: list[str] = []
    caught = BaseException if reraise else Exception

    def run() -> None:
        try:
            box.append(work())
        except caught as exc:
            if reraise:
                raised.append(exc)
            else:
                failed.append(f"failed ({type(exc).__name__}: {exc})")

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(budget)
    if box:
        return box[0]
    if raised:
        raise raised[0]
    return late(failed[0] if failed else f"did not answer within {budget:g} s")


def in_text(word: str, text: str) -> bool:
    """The word is part of the text, ignoring case (the word is lower case): a plain substring
    test, so `8b` is in `128b`."""
    return word in text.lower()


def every_word_in(
    words: Sequence[str], texts: Sequence[str], word_in: Callable[[str, str], bool] = in_text
) -> bool:
    """One of the texts holds every word, by word_in(word, text). With no words, any text holds
    them all: each caller first decides what no words mean."""
    return any(all(word_in(word, text) for word in words) for text in texts)


_VERSION = re.compile(r"^\s*(\d+(?:\.\d+)*)")


def parse_version(text: object) -> tuple[int, ...] | None:
    """Leading dotted integers: '580.126.09' -> (580, 126, 9); None when there are none."""
    if not isinstance(text, str):
        return None
    match = _VERSION.match(text)
    if match is None:
        return None
    return tuple(int(part) for part in match.group(1).split("."))


def as_list(value: object) -> list[str]:
    """A field that may be absent, a string or a list of strings (the API may turn a string into a
    list later), as a list of non-empty strings."""
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    return [item.strip() for item in items if isinstance(item, str) and item.strip()]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Shared helpers for the nim-select scripts; import it, do not run it.",
        epilog='Example: python3 -I "<SKILL_DIR>/scripts/ns_common.py" --help',
    )
    parser.parse_args(argv)
    emit(True, "library", "done", {"note": "import this module; it has no command of its own"})


if __name__ == "__main__":
    main()
