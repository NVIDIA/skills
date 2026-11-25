#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Add the TAO agent identity to a project's AGENTS.md as a marked block.

Codex and Cursor read AGENTS.md at the root of the project a session works in.
Neither loads the AGENTS.md of an installed plugin, so the identity this bank
ships reaches them only as part of the project's own file. This script puts it
there between two marker lines and leaves every other byte of the file alone:

    tao_agent_identity.py add       # create the file, append the block, or update the block
    tao_agent_identity.py status    # is the block there, and is it the current identity?
    tao_agent_identity.py remove    # delete the block again

Claude Code needs none of this: the plugin's session-start hook supplies the
identity there, and Claude Code does not read AGENTS.md.

The target is AGENTS.md at the top of the git work tree that contains the
current directory (the nearest folder with a .git entry, which is where Codex
looks for the project root), or in the current directory outside a work tree.
--project DIR names another project folder and --file PATH any other file.

What each command does:
  add     Target missing or empty: it is created with only the block. Block
          present: the block is replaced in place and nothing outside the
          markers changes. Otherwise the block is appended after the existing
          content and one blank line; when the file did not end with a newline,
          one is added first, and that is the only change outside the block.
  remove  Deletes the block and the blank line before it. What stands before
          and after stays, and a file that is left with nothing in it is deleted.
  status  Reports whether the block is there and equals the identity of the bank.

Nothing is ever copied over a whole file and no backup file is made. --dry-run
(add, remove) prints what would change and writes nothing. The identity is
AGENTS.md of the bank: $TAO_SKILL_BANK_PATH, or the folder above this script.

A line is a marker only when it starts in the first column with the BEGIN or
END text of this plugin and ends with "-->". The same text indented, quoted or
in the middle of a line is the user's own text and is left alone.

Refused, with nothing written:
  - a target that is a directory or not a regular file, a target in a folder
    that does not exist, and an existing file that is read-only;
  - a file with a BEGIN marker and no END marker, with an END marker and no
    BEGIN marker, or with more than one block;
  - a file whose BEGIN marker stands inside a fenced code block: what stands
    between such markers may be an example the user wrote, not a block of this
    script. For the same reason "add" does not append to a file that ends
    inside a code block that is not closed;
  - for "add", a target that is the identity file of a bank, a whole copy of
    the identity, or a file without a block that starts with the heading of
    the identity (the copy an earlier installer made of this or another
    release), because a block there would give the agent the identity twice;
  - for "add", a file in UTF-16 or with NUL bytes: the block is UTF-8;
  - a bank without an AGENTS.md.

A changed file is written to a temporary file that then takes its place. It
keeps its mode, and a symbolic link to it stays a link; a second name of the
file (a hard link) keeps the old content.

Exit codes: 0 done, or nothing to do; 1 ``status`` found no block or one that
differs from the identity; 2 refused, the message says why.
"""

from __future__ import annotations

import argparse
import os
import re
import stat
import sys
import tempfile
from pathlib import Path

PLUGIN_NAME = "nvidia-tao-toolkit"
TARGET_NAME = "AGENTS.md"
BLOCK = f"{PLUGIN_NAME} agent identity block"

# The two marker lines. A line is a marker when it starts with the prefix and
# ends with "-->", so a later version may reword the rest of the BEGIN line and
# still find the blocks that this version wrote. The prefixes are a contract
# with files in users' projects: do not change them.
BEGIN_PREFIX = f"<!-- BEGIN {PLUGIN_NAME} agent identity".encode()
END_PREFIX = f"<!-- END {PLUGIN_NAME} agent identity".encode()
BEGIN_LINE = BEGIN_PREFIX + (
    b': generated block. "tao_agent_identity.py add" of the plugin updates it and "remove" '
    b"deletes it; keep your own text outside the markers. -->"
)
END_LINE = END_PREFIX + b" -->"
_MARKER = rb"\b[^\r\n]*-->[ \t]*(?=\r?$)"
BEGIN_MARKER = re.compile(rb"^" + re.escape(BEGIN_PREFIX) + _MARKER, re.M)
END_MARKER = re.compile(rb"^" + re.escape(END_PREFIX) + _MARKER, re.M)
BYTE_ORDER_MARK = b"\xef\xbb\xbf"
UTF16_MARKS = (b"\xff\xfe", b"\xfe\xff")
# A line that opens or closes a fenced code block in Markdown: at most three
# spaces, then three or more backticks or tildes (CommonMark, "Fenced code blocks").
FENCE = re.compile(rb"^ {0,3}(`{3,}|~{3,})(.*)$")

# Codex reads the AGENTS.md files of a project from the project root down to the
# working directory and stops at project_doc_max_bytes; what follows is dropped.
# The documented default is 32 KiB ("stops adding files once the combined size
# reaches the limit defined by project_doc_max_bytes (32 KiB by default)",
# https://developers.openai.com/codex/guides/agents-md). Codex CLI 0.154.0
# carries project_doc_max_bytes = 32768 and cuts a longer file at that byte.
CODEX_PROJECT_DOC_MAX_BYTES = 32 * 1024
# In one folder Codex reads this file instead of AGENTS.md (same guide).
CODEX_OVERRIDE_NAME = "AGENTS.override.md"


class Refusal(Exception):
    """The request cannot be carried out safely; nothing was written."""


def bank_root() -> Path:
    """The bank whose identity is used: $TAO_SKILL_BANK_PATH, or the one this script ships in."""
    return Path(
        os.environ.get("TAO_SKILL_BANK_PATH") or Path(__file__).resolve().parents[1]
    ).expanduser()


def unix_newlines(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n")


def read_identity(bank: Path) -> bytes:
    """The identity text of ``bank`` with LF line ends and no blank lines around it."""
    source = bank / TARGET_NAME
    if not source.is_file():
        raise Refusal(
            f"the bank has no {TARGET_NAME}: {source} is missing. The bank is the folder that "
            f"TAO_SKILL_BANK_PATH names or, when that is not set, the folder above this script."
        )
    identity = unix_newlines(source.read_bytes()).strip(b"\n")
    if not identity.strip():
        raise Refusal(f"{source} is empty; there is no identity to add")
    if BEGIN_MARKER.search(identity) or END_MARKER.search(identity):
        raise Refusal(f"{source} contains a marker line of the {BLOCK} itself")
    return identity


def render_block(identity: bytes, newline: bytes) -> bytes:
    """The block from its BEGIN line to its END line, without a final line end."""
    return newline.join([BEGIN_LINE, *identity.split(b"\n"), END_LINE])


def newline_of(data: bytes) -> bytes:
    """The line end most lines of ``data`` have; LF when there is no majority."""
    windows = data.count(b"\r\n")
    return b"\r\n" if windows > data.count(b"\n") - windows else b"\n"


def line_number(data: bytes, offset: int) -> int:
    return data.count(b"\n", 0, offset) + 1


def open_fence(data: bytes) -> int | None:
    """The line on which the fenced code block opens that ``data`` ends inside, or None.

    A line after ``data`` would be code in that block. A fence in a block quote
    or deeper in a list is not seen; no line that starts in the first column
    can be inside one.
    """
    if data.startswith(BYTE_ORDER_MARK):
        data = data[len(BYTE_ORDER_MARK):]
    fence, opened = b"", 0
    for number, line in enumerate(data.split(b"\n"), 1):
        match = FENCE.match(line)           # the CR of a Windows line end counts as blank
        if not match:
            continue
        run, rest = match.groups()
        if not fence:
            if not (run.startswith(b"`") and b"`" in rest):   # no backtick after a backtick fence
                fence, opened = run, number
        elif run[:1] == fence[:1] and len(run) >= len(fence) and not rest.strip():
            fence = b""
    return opened if fence else None


def find_block(data: bytes, name: str) -> tuple[int, int] | None:
    """Where the block is: (first byte of the BEGIN line, byte after the END marker).

    None when ``data`` has no marker line. Marker lines that do not make
    exactly one block, or that stand inside a fenced code block, are refused:
    this script cannot know which lines are the user's own. A byte order mark
    at the start of the file stays outside.
    """
    skip = len(BYTE_ORDER_MARK) if data.startswith(BYTE_ORDER_MARK) else 0
    begins = [match.start() + skip for match in BEGIN_MARKER.finditer(data[skip:])]
    ends = [match.end() + skip for match in END_MARKER.finditer(data[skip:])]
    if not begins and not ends:
        return None

    begin_lines = ", ".join(str(line_number(data, offset)) for offset in begins)
    end_lines = ", ".join(str(line_number(data, offset - 1)) for offset in ends)
    repair = (
        " Nothing was written. Repair the file by hand: keep one BEGIN line and one END line "
        "around the block, or delete the marker lines and what stands between them."
    )
    if not ends:
        raise Refusal(f"{name} has a BEGIN marker (line {begin_lines}) and no END marker."
                      f"{repair}")
    if not begins:
        raise Refusal(f"{name} has an END marker (line {end_lines}) and no BEGIN marker."
                      f"{repair}")
    if len(begins) > 1 or len(ends) > 1:
        raise Refusal(f"{name} has more than one {BLOCK}: BEGIN markers on line {begin_lines}, "
                      f"END markers on line {end_lines}.{repair}")
    if ends[0] < begins[0]:
        raise Refusal(f"{name} has its END marker (line {end_lines}) before its BEGIN marker "
                      f"(line {begin_lines}).{repair}")
    fence = open_fence(data[:begins[0]])
    if fence:
        raise Refusal(
            f"{name} has its BEGIN marker (line {begin_lines}) inside the fenced code block that "
            f"opens on line {fence}, so what stands between the markers may be an example of "
            f"yours and not a block this script wrote. Nothing was written. Close that code "
            f"block above the BEGIN line if it was left open; if the markers are an example, "
            f"indent them or word them differently so that they do not start a line."
        )
    return begins[0], ends[0]


def first_line(data: bytes) -> bytes:
    """The first line of ``data`` that is not blank, without a byte order mark or line end."""
    if data.startswith(BYTE_ORDER_MARK):
        data = data[len(BYTE_ORDER_MARK):]
    for line in data.split(b"\n"):
        if line.strip():
            return line.rstrip()
    return b""


def plan_add(existing: bytes | None, identity: bytes, name: str) -> tuple[str, bytes]:
    """(what ``add`` does, the content afterwards) for a file that holds ``existing``."""
    if not existing:
        return ("create" if existing is None else "fill"), render_block(identity, b"\n") + b"\n"
    if existing.startswith(UTF16_MARKS) or b"\x00" in existing:
        raise Refusal(
            f"{name} is not UTF-8 text: it starts with a UTF-16 byte order mark or holds NUL "
            f'bytes, as a file does that Windows PowerShell 5 wrote with ">". The block is '
            f"UTF-8 and would leave a file that no program reads as one text. Convert the file "
            f"to UTF-8 first."
        )
    twice = (" A block added to it would give the agent the identity twice. Name your project "
             "with --project, or delete that copy or restore your own file first.")
    if unix_newlines(existing).strip() == identity.strip():
        raise Refusal(
            f"{name} is itself a copy of the identity: the bank's own {TARGET_NAME}, or a file "
            f"that an earlier installer copied whole.{twice}"
        )
    newline = newline_of(existing)
    block = render_block(identity, newline)
    found = find_block(existing, name)
    if found:
        start, end = found
        if existing[start:end] == block:
            return "current", existing
        return "update", existing[:start] + block + existing[end:]
    heading = first_line(identity)
    if first_line(existing) == heading:
        raise Refusal(
            f"{name} has no {BLOCK} and starts with the heading of the identity itself "
            f"({heading.decode('utf-8', 'backslashreplace')}): it looks like a whole copy of the "
            f"identity of another release, as earlier installers made.{twice}"
        )
    fence = open_fence(existing)
    if fence:
        raise Refusal(
            f"{name} ends inside a fenced code block that opens on line {fence} and is not "
            f"closed, so a block appended to it would become part of that code. Close the code "
            f"block first."
        )
    if existing.endswith(b"\n"):
        return "append", existing + newline + block + newline
    return "append-newline", existing + newline + newline + block + newline


def plan_remove(existing: bytes, name: str) -> tuple[str, bytes]:
    """(what ``remove`` does, the content afterwards); empty content means: delete the file."""
    found = find_block(existing, name)
    if not found:
        return "absent", existing
    start, end = found
    before, after = existing[:start], existing[end:]
    for line_end in (b"\r\n", b"\n"):          # the line end of the END line goes with the block
        if after.startswith(line_end):
            after = after[len(line_end):]
            break
    action = "remove"
    for line_end in (b"\r\n", b"\n"):          # and so does the blank line "add" put before it
        if before.endswith(b"\n" + line_end):
            before = before[:-len(line_end)]
            action = "remove-with-blank-line"
            break
    rest = before + after
    if not rest.replace(BYTE_ORDER_MARK, b"").strip():
        return "delete", b""
    return action, rest


def project_root(start: Path) -> Path:
    """The nearest folder at or above ``start`` with a .git entry, or ``start`` without one."""
    for folder in (start, *start.parents):
        if (folder / ".git").exists():
            return folder
    return start


def resolve_target(project: str | None, file: str | None) -> Path:
    """The file to work on, as an absolute path; links in it are not resolved here."""
    if file == "" or project == "":
        raise Refusal("--file and --project need a path; an empty one was given")
    if file is not None:
        return Path(os.path.abspath(os.path.expanduser(file)))
    if project is not None:
        folder = Path(os.path.abspath(os.path.expanduser(project)))
        if not folder.is_dir():
            raise Refusal(f"--project {project}: {folder} is not an existing folder")
        return folder / TARGET_NAME
    try:
        here = os.getcwd()
    except OSError as exc:
        raise Refusal(f"the current directory cannot be read ({exc}); it may have been deleted. "
                      f"Name the project with --project.") from exc
    return project_root(Path(os.path.abspath(here))) / TARGET_NAME


def real_file(target: Path, *, create: bool = False) -> tuple[Path, bool]:
    """(the file that is read and written, whether ``target`` is a link to it).

    A link is followed and stays a link: the file it points at gets the block.
    ``create`` says that a missing file is about to be written, which needs its folder.
    """
    linked = target.is_symlink()
    path = Path(os.path.realpath(target)) if linked else target
    if linked and not path.exists():
        raise Refusal(f"{target} is a symbolic link to {path}, which does not exist")
    if path.is_dir():
        raise Refusal(f"{target} is a directory, not a file. Name the file with --file, or the "
                      f"project folder with --project.")
    if path.exists() and not path.is_file():
        raise Refusal(f"{target} is not a regular file")
    if create and not path.parent.is_dir():
        raise Refusal(f"the folder {path.parent} does not exist; {target} cannot be created "
                      f"there")
    return path, linked


def refuse_identity_file(path: Path, name: str, bank: Path) -> None:
    """Refuse a target that is the identity file of ``bank`` or of the bank this script is in."""
    for source in (bank / TARGET_NAME, Path(__file__).resolve().parents[1] / TARGET_NAME):
        if path.exists() and source.exists() and os.path.samefile(path, source):
            raise Refusal(
                f"{name} is the identity file of the bank in {source.parent}, not the "
                f"{TARGET_NAME} of a project of yours. Name your project with --project."
            )


def refuse_read_only(path: Path, name: str) -> None:
    """Refuse to change a file that was made read-only, as some do to keep agents from editing it.

    The write would succeed all the same: it replaces the file, which needs
    write access to the folder and not to the file.
    """
    if path.exists() and not os.access(path, os.W_OK):
        raise Refusal(f"{name} is read-only, so it was left as it is. Make it writable first "
                      f"(chmod u+w) if this script is to change it.")


def read_file(path: Path) -> bytes | None:
    """The content of ``path``, or None when it does not exist."""
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise Refusal(f"cannot read {path}: {exc}") from exc


def write_file(path: Path, data: bytes) -> None:
    """Put ``data`` into ``path`` in one step; an existing file keeps its mode.

    The content goes to a temporary file in the same folder that then takes the
    place of ``path``, so a failed write leaves the file as it was. The file is
    a new one afterwards: another name of the old file (a hard link) keeps the
    old content, and owner and extended attributes are those of a new file.
    """
    mode = None
    if path.exists():
        mode = stat.S_IMODE(path.stat().st_mode)
    try:
        handle, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
        )
    except OSError as exc:
        raise Refusal(f"cannot write in {path.parent}: {exc}") from exc
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
        if mode is None:
            umask = os.umask(0)
            os.umask(umask)
            mode = 0o666 & ~umask
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    except BaseException as exc:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        if isinstance(exc, OSError):
            raise Refusal(f"cannot write {path}: {exc}") from exc
        raise


def warn(message: str) -> None:
    print(f"tao_agent_identity: WARNING: {message}", file=sys.stderr)


def warn_about_codex(path: Path, size: int) -> None:
    """Say when Codex would not read the file, or not all of it."""
    override = path.parent / CODEX_OVERRIDE_NAME
    if path.name == TARGET_NAME and override.is_file():
        warn(f"{override} exists. Codex reads that file instead of {path.name} in this folder, "
             f"so it will not see the block.")
    if size > CODEX_PROJECT_DOC_MAX_BYTES:
        warn(f"With the block, {path} has {size} bytes. Codex reads at most "
             f"{CODEX_PROJECT_DOC_MAX_BYTES} bytes of a project's {TARGET_NAME} files (the "
             f"default of project_doc_max_bytes) and drops what follows, so the end of this "
             f"file may not reach the agent.")


def command_add(target: Path, dry_run: bool) -> int:
    path, linked = real_file(target, create=True)
    name = f"{target} (a link to {path})" if linked else str(target)
    bank = bank_root()
    identity = read_identity(bank)
    refuse_identity_file(path, name, bank)
    existing = read_file(path)
    action, content = plan_add(existing, identity, name)
    if action != "current":
        refuse_read_only(path, name)
    kept = len(existing or b"")
    did, would = {
        "create": (f"Created {name} with the {BLOCK}.",
                   f"Would create {name} with the {BLOCK}."),
        "fill": (f"Wrote the {BLOCK} into {name}, which was empty.",
                 f"Would write the {BLOCK} into {name}, which is empty."),
        "current": (f"The {BLOCK} in {name} is already the current identity. Nothing to do.",) * 2,
        "update": (f"Updated the {BLOCK} in {name}. Every byte outside the markers is unchanged.",
                   f"Would update the {BLOCK} in {name} and leave every byte outside the markers "
                   f"unchanged."),
        "append": (f"Appended the {BLOCK} to {name} after one blank line. The {kept} bytes that "
                   f"were there are unchanged.",
                   f"Would append the {BLOCK} to {name} after one blank line and leave the {kept} "
                   f"bytes that are there unchanged."),
        "append-newline": (
            f"Appended the {BLOCK} to {name} after one blank line. The {kept} bytes that were "
            f"there are unchanged; they did not end with a newline, so one was added.",
            f"Would append the {BLOCK} to {name} after one blank line. The {kept} bytes that are "
            f"there do not end with a newline, so one would be added; they stay unchanged "
            f"otherwise."),
    }[action]
    if dry_run:
        print(would)
        if action != "current":
            print(f"The file would have {len(content)} bytes. Dry run: nothing was written.")
    else:
        if action != "current":
            write_file(path, content)
        print(did)
        if action != "current":
            print('Codex and Cursor read this file when a session starts in the project. Run '
                  '"add" again to update the block and "remove" to delete it.')
    warn_about_codex(path, len(content))
    return 0


def command_remove(target: Path, dry_run: bool) -> int:
    path, linked = real_file(target)
    name = f"{target} (a link to {path})" if linked else str(target)
    existing = read_file(path)
    if existing is None:
        print(f"{name} does not exist. Nothing to do.")
        return 0
    action, content = plan_remove(existing, name)
    if action == "absent":
        print(f"{name} has no {BLOCK}. Nothing to do.")
        return 0
    refuse_read_only(path, name)
    if action != "delete":
        what = BLOCK + (" and the blank line before it" if action != "remove" else "")
        did = f"Removed the {what} from {name}. Every other byte is unchanged."
        would = f"Would remove the {what} from {name} and leave every other byte unchanged."
    elif linked:
        # Deleting the file would leave the link pointing at nothing.
        did = (f"Removed the {BLOCK} from {name}. The file held nothing else and is now empty; "
               f"it was kept because a link points at it.")
        would = (f"Would remove the {BLOCK} from {name}. The file holds nothing else and would "
                 f"be left empty, because a link points at it.")
    else:
        did = f"Removed the {BLOCK}. {name} held nothing else and was deleted."
        would = f"Would remove the {BLOCK} and delete {name}, which holds nothing else."
    if dry_run:
        print(would)
        print("Dry run: nothing was written.")
        return 0
    if action == "delete" and not linked:
        try:
            path.unlink()
        except OSError as exc:
            raise Refusal(f"cannot delete {path}: {exc}") from exc
    else:
        write_file(path, content)
    print(did)
    return 0


def command_status(target: Path) -> int:
    path, linked = real_file(target)
    name = f"{target} (a link to {path})" if linked else str(target)
    bank = bank_root()
    identity = read_identity(bank)
    existing = read_file(path)
    if existing is None:
        print(f"{name} does not exist, so it has no {BLOCK}.")
        return 1
    found = find_block(existing, name)
    if not found:
        print(f"{name} has no {BLOCK}.")
        return 1
    warn_about_codex(path, len(existing))
    start, end = found
    if unix_newlines(existing[start:end]) == render_block(identity, b"\n"):
        print(f"{name} has the {BLOCK}, and it is the current identity of {bank}.")
        return 0
    print(f'{name} has the {BLOCK}, but it differs from the current identity of {bank}. Run '
          f'"add" to update it.')
    return 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    target = argparse.ArgumentParser(add_help=False)
    where = target.add_mutually_exclusive_group()
    where.add_argument(
        "--project", metavar="DIR",
        help=f"work on {TARGET_NAME} in this folder instead of the top of the git work tree "
             f"around the current directory",
    )
    where.add_argument(
        "--file", metavar="PATH", help=f"work on this file instead of a project's {TARGET_NAME}"
    )
    writes = argparse.ArgumentParser(add_help=False)
    writes.add_argument(
        "--dry-run", action="store_true", help="print what would change and write nothing"
    )
    commands = parser.add_subparsers(dest="command", metavar="{add,remove,status}")
    commands.required = True
    commands.add_parser(
        "add", parents=[target, writes],
        help="create the file with the block, append the block, or update the block in place",
    )
    commands.add_parser(
        "remove", parents=[target, writes],
        help="delete the block; a file with nothing else in it is deleted",
    )
    commands.add_parser(
        "status", parents=[target],
        help="say whether the block is there and equals the current identity (exit 1 if not)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    # A path that the encoding of the terminal cannot show is printed with escapes.
    # Without this, the message about a file that was just written raises an error.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    args = parse_args(argv)
    try:
        target = resolve_target(args.project, args.file)
        if args.command == "add":
            return command_add(target, args.dry_run)
        if args.command == "remove":
            return command_remove(target, args.dry_run)
        return command_status(target)
    except Refusal as exc:
        print(f"tao_agent_identity: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
