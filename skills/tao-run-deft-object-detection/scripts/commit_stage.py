#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Commit one finished DEFT OD stage: the only supported writer of run state.

Everything else in this skill reads ``deft_state.json`` and ``loop_log.jsonl``;
this is the single place they are written. That matters because the two files
are the loop's entire memory across context compaction, and they are only useful
while they agree with each other and with the artifacts on disk. Two writers —
or one writer plus a helpful hand-edit — and a resumed run silently re-does a
stage, or skips one, or evaluates a checkpoint that was never trained.

So every rule the loop depends on is enforced here, before anything is written:

  exclusivity   One writer at a time per run directory (``.deft_commit.lock``).
                ``deft_stages`` renames a fixed tmp path, so two overlapping
                commits race: the loser's rename fails after the winner consumed
                that tmp file, and its rollback then restores a snapshot taken
                before the winner wrote — erasing a stage the winner had already
                reported as committed.
  ordering      The stage must be the legal next stage of its phase
                (``deft_stages.validate_transition``), and the phase before it
                must have finished. An out-of-order commit is rejected, not
                repaired. A ``--status error`` commit is exempt: a stage can fail
                at any point, and refusing to record that would lose the only
                evidence of the failure.
  artifacts     Every path in ``STAGE_ARTIFACTS[stage]`` must be passed and must
                exist on disk with the right kind. "The stage printed success"
                is not proof; the file is.
  atomicity     A journal of both files is written first, then state, then the
                log. If the audit rejects the result — or the process is killed
                between the two writes — the journal restores them byte-for-byte.
                State and log are never left disagreeing, so the next resume has
                one story, and an interrupted commit is undone by the next
                invocation instead of wedging the run forever.

The post-commit audit is the acceptance test, but it judges *this commit*, not
the whole history. A run can already be inconsistent — an operator deleted a
staged-image directory to free disk — and refusing every later commit for a fault
that predates it would make ``--status error`` and ``loop_stop`` unrecordable, so
the run could never be finalized at all. The audit therefore runs before and
after the write, and only errors this commit *introduced* roll it back.

Disk is the source of truth: state and log are re-read here every time, and
nothing passed on the command line is trusted to describe them.

The artifact flags are derived from ``STAGE_ARTIFACTS`` at import time rather
than typed out, so the CLI cannot drift from the contract. Flags this script
does not know about are recorded on the phase entry as-is instead of being
rejected — an overlay that records one more path should not need a code change —
but they may not overwrite a field a declared flag owns and checks.

CLI:

    python3 scripts/commit_stage.py \\
        --results-dir /abs/results/run_X --iter-label iter1 --stage mine \\
        --mining-output /abs/.../final_unique_files.parquet \\
        --mining-summary /abs/.../summary.json \\
        --summary "mined 500 unique images" --duration-sec 612

Exit codes:
  0  committed; this commit introduced no new audit error
  1  rejected before any write — state and log are untouched
  2  written, then rolled back because this commit made the run inconsistent
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import errno
import fcntl
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from deft_stages import (  # noqa: E402
    ALL_STAGES,
    STAGE_ARTIFACTS,
    TERMINAL_STAGE,
    check_artifact,
    derive_rare_classes,
    iter_number,
    log_path,
    phase_entry,
    phase_order,
    previous_phase,
    read_log,
    read_state,
    state_path,
    validate_transition,
    write_log_atomic,
    write_state_atomic,
)
from audit_deft_run import EXTRA_ARTIFACT_FIELDS  # noqa: E402

AUDIT_SCRIPT = Path(__file__).resolve().parent / "audit_deft_run.py"

# One writer at a time, plus a crash-recovery record for the pair of writes.
LOCK_NAME = ".deft_commit.lock"
JOURNAL_NAME = ".deft_commit.journal"
LOCK_TIMEOUT_SEC = 120

# Fields a declared, validated flag owns. An undeclared --flag may record anything
# else on the phase entry, but not one of these: that would put an unchecked path,
# an unmeasured early-stop claim, or hand-written bookkeeping where the gates
# promise a verified value.
RESERVED_RECORD_FIELDS = {
    "map_value": "--map-value",
    "weak_image_count": "--weak-image-count",
    "zero_weak_images": "--zero-weak-images",
    "pool_remaining": "--pool-remaining",
    "pool_exhausted": "--pool-exhausted",
}
STATE_BOOKKEEPING_FIELDS = {"stage_completed", "status", "failed_stage"}


def _dest(flag: str) -> str:
    """argparse's dest for a long flag, which is also the state field name."""
    return flag.lstrip("-").replace("-", "_")


def _artifact_flags() -> dict[str, tuple[str, str, str]]:
    """flag -> (state field, kind, owning stage), derived from STAGE_ARTIFACTS."""
    flags: dict[str, tuple[str, str, str]] = {}
    for stage, fields in STAGE_ARTIFACTS.items():
        for field, (flag, kind) in fields.items():
            previous = flags.get(flag)
            if previous is not None:
                raise ValueError(
                    f"STAGE_ARTIFACTS reuses {flag} for {previous[2]}/{previous[0]} "
                    f"and {stage}/{field}; a flag must name one artifact"
                )
            flags[flag] = (field, kind, stage)
    return flags


def _extra_artifact_flags() -> dict[str, str]:
    """flag -> kind for the optional paths, from the table the audit re-checks.

    Defined once, in ``audit_deft_run.py``, so a path this script verifies at
    write time is exactly the one the audit re-verifies on every later run. While
    the two tables were separate, deleting a recorded ``--kpi-log`` — the only
    place mAP appears — left the run auditing VALID and complete.
    """
    flags: dict[str, str] = {}
    for field, (flag, kind) in EXTRA_ARTIFACT_FIELDS.items():
        if _dest(flag) != field:
            raise ValueError(
                f"EXTRA_ARTIFACT_FIELDS maps {field!r} to {flag}, whose dest is "
                f"{_dest(flag)!r}; the flag must spell its own state field"
            )
        flags[flag] = kind
    return flags


ARTIFACT_FLAGS = _artifact_flags()
EXTRA_ARTIFACT_FLAGS = _extra_artifact_flags()


def _coerce(value: str) -> Any:
    """Best-effort JSON scalar for an undeclared flag's value."""
    lowered = value.strip().lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        number = float(value)
    except ValueError:
        return value
    # NaN/Infinity round-trip through Python's json but are not valid JSON.
    return number if math.isfinite(number) else value


def _parse_extras(tokens: list[str]) -> dict[str, Any]:
    """Turn leftover ``--foo bar`` / ``--foo=bar`` / ``--foo`` into a dict."""
    extras: dict[str, Any] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            raise ValueError(
                f"unexpected argument {token!r}; every extra must be a --flag with a value"
            )
        if "=" in token:
            name, value = token[2:].split("=", 1)
            index += 1
        else:
            name = token[2:]
            following = tokens[index + 1] if index + 1 < len(tokens) else None
            if following is not None and not following.startswith("--"):
                value = following
                index += 2
            else:
                value = "true"  # a bare flag records a boolean
                index += 1
        key = _dest(f"--{name}")
        if not key:
            raise ValueError(f"unexpected argument {token!r}: empty flag name")
        extras[key] = _coerce(value)
    return extras


def _resolve_artifact(raw: str, flag: str, kind: str) -> str:
    """Absolute, resolved, and present on disk — or raise naming the flag."""
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise ValueError(
            f"{flag} must be an absolute host path, got {raw!r}; the loop mounts "
            "host paths as themselves, so a relative path is unusable in a container"
        )
    resolved = path.resolve()
    problem = check_artifact(str(resolved), kind)
    if problem:
        raise ValueError(f"{flag}: {problem}")
    return str(resolved)


# ── one writer per run directory ─────────────────────────────────────────────

@contextlib.contextmanager
def _run_lock(results_dir: Path):
    """Hold an exclusive lock on the run for the whole read-modify-write cycle."""
    path = results_dir / LOCK_NAME
    handle = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o644)
    deadline = time.monotonic() + LOCK_TIMEOUT_SEC
    try:
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                if time.monotonic() >= deadline:
                    raise ValueError(
                        f"another commit_stage.py has held {path} for more than "
                        f"{LOCK_TIMEOUT_SEC}s; commit one stage at a time"
                    ) from exc
                time.sleep(0.05)
        yield
    finally:
        try:
            fcntl.flock(handle, fcntl.LOCK_UN)
        finally:
            os.close(handle)


# ── durability ───────────────────────────────────────────────────────────────

def _fsync_path(path: Path) -> None:
    """Flush one file — or a directory entry — to stable storage. Best effort."""
    try:
        handle = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(handle)
    except OSError:
        pass
    finally:
        os.close(handle)


def _write_text_atomic(path: Path, text: str) -> None:
    """Replace ``path`` with ``text`` through a uniquely named tmp file.

    A fixed tmp name is squattable: with a directory sitting at
    ``loop_log.jsonl.rollback.tmp``, a rollback restored one file and abandoned
    the other — the exact divergence the rollback exists to prevent.
    """
    handle, tmp = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _snapshot(results_dir: Path) -> list[tuple[Path, str | None]]:
    """Exact bytes of both files, so a rollback restores them unchanged."""
    snapshot: list[tuple[Path, str | None]] = []
    for path in (state_path(results_dir), log_path(results_dir)):
        snapshot.append(
            (path, path.read_text(encoding="utf-8") if path.is_file() else None)
        )
    return snapshot


def _restore(snapshot: list[tuple[Path, str | None]]) -> None:
    """Put both files back. Atomic per file, same as every write here."""
    for path, text in snapshot:
        if text is None:
            path.unlink(missing_ok=True)
            continue
        _write_text_atomic(path, text)


# ── journal: two renames, made recoverable ───────────────────────────────────

def _journal_path(results_dir: Path) -> Path:
    return results_dir / JOURNAL_NAME


def _write_journal(
    results_dir: Path,
    snapshot: list[tuple[Path, str | None]],
    phase: str,
    stage: str,
) -> None:
    """Record how to undo this commit, before either file is touched."""
    payload = {
        "written_at": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ"
        ),
        "phase": phase,
        "stage": stage,
        "files": {path.name: text for path, text in snapshot},
    }
    _write_text_atomic(_journal_path(results_dir), json.dumps(payload, indent=2))
    _fsync_path(results_dir)


def _clear_journal(results_dir: Path) -> None:
    _journal_path(results_dir).unlink(missing_ok=True)
    _fsync_path(results_dir)


def _recover_journal(results_dir: Path) -> None:
    """Undo a commit that died mid-write, so the run is committable again.

    ``write_state_atomic`` and ``write_log_atomic`` are two independent renames.
    A kill — OOM, Ctrl-C, container stop — between them leaves state ahead of the
    log, and from then on every commit refuses with "state and log already
    disagree": the same stage, the next stage, and ``loop_stop`` alike, with
    ``init`` refusing too without ``--force``. The journal turns that permanent
    wedge into a rollback: whatever the interrupted commit wrote is undone and the
    stage is simply run again.
    """
    path = _journal_path(results_dir)
    if not path.is_file():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        files = payload["files"]
        if not isinstance(files, dict):
            raise ValueError("journal 'files' must be an object")
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(
            f"{path} is present but unreadable ({exc}); it records how to undo an "
            "interrupted commit, so move it aside only after comparing deft_state.json "
            "and loop_log.jsonl by hand"
        ) from exc
    # The journal is written before the two renames and cleared after the audit
    # accepts, so a crash in between leaves it describing a commit that already
    # landed and is already durable. Rolling that back discards an accepted stage.
    # Whether it landed is answerable from the log: the last event is this commit's.
    phase, stage = payload.get("phase"), payload.get("stage")
    landed = False
    try:
        events = [json.loads(line) for line in
                  log_path(results_dir).read_text(encoding="utf-8").splitlines() if line.strip()]
        if events:
            last = events[-1]
            landed = last.get("iter") == phase and last.get("stage") == stage
    except (OSError, ValueError):
        landed = False

    if landed:
        path.unlink(missing_ok=True)
        _fsync_path(results_dir)
        print(
            f"an interrupted commit of {phase}/{stage} had already been written and is "
            "recorded in loop_log.jsonl; cleared its journal rather than undoing it",
            file=sys.stderr,
        )
        return

    _restore([(results_dir / name, text) for name, text in files.items()])
    path.unlink(missing_ok=True)
    _fsync_path(results_dir)
    print(
        f"recovered an interrupted commit of {phase}/{stage}: "
        "deft_state.json and loop_log.jsonl were restored to the state before it. That "
        "stage was NOT recorded — run it again",
        file=sys.stderr,
    )


# ── the audit, as this script's acceptance test ──────────────────────────────

def _run_audit(results_dir: Path) -> tuple[int, dict[str, Any], str]:
    """Return (exit code, report, stderr). An unparseable report is an empty dict."""
    proc = subprocess.run(
        [sys.executable, str(AUDIT_SCRIPT), "--results-dir", str(results_dir), "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        report = json.loads(proc.stdout)
    except json.JSONDecodeError:
        report = {}
    return proc.returncode, report if isinstance(report, dict) else {}, proc.stderr


def _new_errors(before: list[str], after: list[str]) -> list[str]:
    """The errors this commit introduced, ignoring the ones it inherited."""
    remaining = list(before)
    fresh: list[str] = []
    for error in after:
        if error in remaining:
            remaining.remove(error)
        else:
            fresh.append(error)
    return fresh


def _commit_is_visible(
    report: dict[str, Any], phase: str, stage: str, status: str, expected_entries: int
) -> str | None:
    """Confirm the audit is describing the write this process just made.

    An exit code alone is not proof. A run whose newest event was erased by a
    concurrent writer still audits VALID, so exit-code-only acceptance once
    printed "committed" for a stage that was no longer on disk.
    """
    if report.get("log_entries") != expected_entries:
        return (
            f"the audit sees {report.get('log_entries')} log events, expected "
            f"{expected_entries} — this commit is not on disk"
        )
    if stage == TERMINAL_STAGE:
        if not report.get("loop_stop_committed"):
            return "the audit does not see a committed loop_stop"
        return None
    if status == "error":
        if not report.get("run_failed"):
            return "the audit does not see the recorded failure"
        return None
    recorded = report.get("stage_completed_by_phase")
    recorded = recorded if isinstance(recorded, dict) else {}
    if recorded.get(phase) != stage:
        return f"the audit reports {phase} completed {recorded.get(phase)!r}, not {stage!r}"
    return None


def _completed_by_phase(events: list[dict[str, Any]]) -> dict[str, str]:
    """Last ok stage per phase, straight from the log."""
    completed: dict[str, str] = {}
    for event in events:
        if event.get("status") != "ok":
            continue
        label = str(event.get("iter", ""))
        stage = event.get("stage")
        try:
            order = phase_order(label)
        except ValueError:
            continue
        if stage in order:
            completed[label] = str(stage)
    return completed


def _build_parser() -> argparse.ArgumentParser:
    # allow_abbrev=False: an unrecognized flag must reach the extras parser
    # verbatim instead of being silently expanded into a declared one.
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], allow_abbrev=False
    )
    parser.add_argument("--results-dir", required=True,
                        help="Run directory holding deft_state.json and loop_log.jsonl.")
    parser.add_argument("--iter-label", required=True,
                        help="Phase label: prep, baseline, or iterN — never zero-padded.")
    parser.add_argument("--stage", required=True, choices=ALL_STAGES,
                        help="Stage that just finished.")
    parser.add_argument("--summary", required=True,
                        help="One-line outcome recorded in loop_log.jsonl.")
    parser.add_argument("--status", choices=("ok", "error"), default="ok",
                        help="error records a hard stop and fails the run. Never auto-retry after one.")
    parser.add_argument("--duration-sec", type=int, default=0,
                        help="Stage wall-clock seconds. Omit when no start time was captured; "
                             "it records 0. Do not invent a duration.")

    for flag, (field, kind, stage) in ARTIFACT_FLAGS.items():
        parser.add_argument(flag, metavar="PATH",
                            help=f"[{stage}] absolute path to an existing {kind}; "
                                 f"recorded as {field}.")
    for flag, kind in EXTRA_ARTIFACT_FLAGS.items():
        parser.add_argument(flag, metavar="PATH",
                            help=f"Optional; absolute path to an existing {kind}, "
                                 f"recorded as {_dest(flag)}.")

    parser.add_argument("--map-value", type=float, default=None,
                        help="Aggregate mAP parsed from the kpi_analyze log; the trend cannot "
                             "be reported without it. Must be finite — when the log prints "
                             "'mAP: nan', omit this flag and say so in --summary.")
    parser.add_argument("--weak-image-count", type=int, default=None,
                        help="[gap_analysis only] rows in weak_images.parquet. Required on an "
                             "ok gap_analysis commit; 0 is the documented early stop.")
    parser.add_argument("--zero-weak-images", action="store_true",
                        help="[gap_analysis only] record the zero-weak-image early stop "
                             "explicitly, when the row count itself is not to hand.")
    parser.add_argument("--pool-remaining", type=int, default=None,
                        help="[loop_stop only] Source-pool images not already in the "
                             "cumulative exclude set, counted before the mine that could not "
                             "run. Record the real count, whatever it is.")
    parser.add_argument("--pool-exhausted", action="store_true",
                        help="[loop_stop only] The pool cannot supply another iteration. "
                             "This is a legitimate terminal state and the audit accepts it "
                             "as a documented early stop; without it a stop before the "
                             "configured iterations reads as an abandoned run. The pool need "
                             "not be empty: too few images left to meet the budget is "
                             "exhaustion, since the miner raises rather than returning a "
                             "short result.")
    return parser


def _resolve_rare_classes(state: dict, pool_report: str | None) -> str | None:
    """Fill a null ``config.rare_class_list`` from the prepared pool's class counts.

    Only applies to ``class_stratified`` allocation, which is the only policy that
    reads the list. Returns a message describing what was written, or None when
    there was nothing to do. A pool report that cannot be read is not fatal here:
    the value stays null and `mine` refuses on a list it needs but does not have.
    """
    config = state.get("config")
    if not isinstance(config, dict):
        return None
    if config.get("allocation_policy") != "class_stratified":
        return None
    if config.get("rare_class_list"):
        return None
    if not pool_report:
        return ("rare_class_list is unset and no --pool-report was committed; "
                "`mine` requires it under class_stratified allocation")
    raw_targets = config.get("target_classes") or []
    if isinstance(raw_targets, str):
        raw_targets = raw_targets.split(",")
    targets = [str(c).strip() for c in raw_targets if str(c).strip()]
    try:
        data = json.loads(Path(pool_report).read_text(encoding="utf-8"))
        counts = {str(c): int(n) for c, n in (data.get("annotations_by_class") or {}).items()}
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return f"rare_class_list left unset: --pool-report is unreadable ({exc})"
    rare, detail = derive_rare_classes(counts, targets)
    if not rare:
        return (f"rare_class_list left unset: no target class holds a below-mean share "
                f"of the pool ({detail})")
    config["rare_class_list"] = ",".join(rare)
    return (f"rare_class_list resolved to {rare} from the pool's class counts ({detail})")


def main() -> int:
    parser = _build_parser()
    args, unknown = parser.parse_known_args()

    snapshot: list[tuple[Path, str | None]] | None = None
    results_dir: Path | None = None
    dirty = False
    rolled_back = False
    try:
        stage = args.stage
        phase = args.iter_label
        try:
            order = phase_order(phase)
        except ValueError as exc:
            raise ValueError(f"--iter-label: {exc}") from exc
        number = iter_number(phase)
        if number is not None and number < 1:
            # previous_phase("iter0") answers "iter-1", which nothing can resolve.
            raise ValueError(
                f"--iter-label {phase!r} is out of range; iterations are numbered from 1 "
                "and the zeroth phase is called 'baseline'"
            )
        if number is not None and phase != f"iter{number}":
            # iter01 and iter1 are distinct phases carrying one iteration number:
            # the max_iterations bound compares the number, so a padded alias buys
            # an extra full iteration, while previous_phase() still points the next
            # iteration at "iter1" — the mining target and the checkpoint that
            # trained last drift apart.
            raise ValueError(
                f"--iter-label {phase!r} is not a canonical label; use 'iter{number}'. "
                "Zero-padded aliases are distinct phases sharing one iteration number"
            )
        if not args.summary.strip():
            raise ValueError("--summary must be a non-empty one-line outcome")
        if args.duration_sec < 0:
            raise ValueError(f"--duration-sec must be >= 0, got {args.duration_sec}")
        if args.map_value is not None and not math.isfinite(args.map_value):
            # json.dump writes a bare NaN/Infinity token: it round-trips through
            # Python and is rejected by every other JSON parser that reads the run.
            raise ValueError(
                f"--map-value must be a finite number, got {args.map_value}. kpi_analyze "
                "prints 'mAP: nan' when a class has no ground truth in the KPI set — omit "
                "--map-value and record that in --summary rather than writing a non-JSON "
                "literal into deft_state.json"
            )

        results_dir = Path(args.results_dir).expanduser().resolve()
        if not results_dir.is_dir():
            raise FileNotFoundError(f"--results-dir is not a directory: {results_dir}")

        with _run_lock(results_dir):
            # An interrupted commit is undone before anything is read, so the
            # state/log agreement check below sees a coherent pair.
            _recover_journal(results_dir)

            # ── 1. disk is the truth: re-read both files, trust nothing passed in ──
            state = read_state(results_dir)
            if not isinstance(state, dict):
                raise ValueError(f"{state_path(results_dir)}: root must be a JSON object")
            recorded_dir = state.get("results_dir")
            if recorded_dir and Path(str(recorded_dir)).expanduser().resolve() != results_dir:
                raise ValueError(
                    f"state.results_dir={recorded_dir} does not match --results-dir "
                    f"{results_dir}; this commit belongs to a different run"
                )
            events = read_log(results_dir)

            # A finalized or hard-stopped run accepts nothing more. The audit would
            # reject these too, but rejecting here keeps the message readable and
            # leaves the files untouched.
            for event in events:
                if event.get("stage") == TERMINAL_STAGE and event.get("status") == "ok":
                    raise ValueError(
                        f"loop_stop is already committed (seq={event.get('seq')}); the run is "
                        "finalized and nothing may follow it"
                    )
            failure = next((e for e in events if e.get("status") == "error"), None)
            if failure is not None and stage != TERMINAL_STAGE:
                raise ValueError(
                    f"a hard stop is already recorded at seq={failure.get('seq')} "
                    f"({failure.get('iter')}/{failure.get('stage')}); only {TERMINAL_STAGE} may "
                    "be committed after it — surface the disk evidence, do not auto-retry"
                )

            completed_by_phase = _completed_by_phase(events)
            last_ok = completed_by_phase.get(phase)
            existing = state.get("iterations", {})
            existing = existing.get(phase) if isinstance(existing, dict) else None
            existing = existing if isinstance(existing, dict) else {}
            completed = existing.get("stage_completed")
            if completed != last_ok:
                raise ValueError(
                    f"state and log already disagree about {phase}: "
                    f"state.stage_completed={completed!r} but the log's last ok stage is "
                    f"{last_ok!r}. An interrupted commit is undone automatically, so this pair "
                    "was written by something other than commit_stage.py; run "
                    "audit_deft_run.py and repair before committing"
                )

            # init_deft_state.py freezes max_iterations under config; audit_deft_run.py
            # accepts a top-level copy. Resolve it the same way they do.
            config = state.get("config")
            config = config if isinstance(config, dict) else {}
            max_iterations = config.get("max_iterations", state.get("max_iterations"))
            if (
                number is not None
                and isinstance(max_iterations, int)
                and not isinstance(max_iterations, bool)
                and number > max_iterations
            ):
                raise ValueError(
                    f"--iter-label {phase} exceeds state.max_iterations={max_iterations}"
                )

            # ── 2. ordering: within the phase, and across phases ──────────────
            if args.status == "ok":
                validate_transition(phase, completed, stage)
                previous = previous_phase(phase) if stage != TERMINAL_STAGE else None
                if previous is not None:
                    final = phase_order(previous)[-1]
                    if completed_by_phase.get(previous) != final:
                        raise ValueError(
                            f"{phase}/{stage} cannot be committed until {previous} finishes "
                            f"{final}; {previous}'s last ok stage is "
                            f"{completed_by_phase.get(previous)!r}. {previous} is the phase "
                            f"{phase} is measured against — abandoning it mid-way leaves the "
                            "run with no mAP trend, which is the only thing the loop produces"
                        )

            # ── 3. artifacts: passed, absolute, and actually on disk ──────────
            misfiled = [
                flag
                for flag, (_field, _kind, owner) in ARTIFACT_FLAGS.items()
                if owner != stage and getattr(args, _dest(flag)) is not None
            ]
            if misfiled:
                owners = ", ".join(f"{f} -> {ARTIFACT_FLAGS[f][2]}" for f in sorted(misfiled))
                raise ValueError(
                    f"flag(s) do not belong to stage {stage!r}: {owners}. "
                    "Commit each stage's artifacts with that stage"
                )

            artifacts: dict[str, str] = {}
            unavailable: list[str] = []
            for field, (flag, kind) in STAGE_ARTIFACTS[stage].items():
                raw = getattr(args, _dest(flag))
                if raw is None:
                    if args.status == "error":
                        continue  # a stage that failed may have produced nothing
                    raise ValueError(
                        f"stage {stage!r} requires {flag} (recorded as {field}); "
                        f"pass an absolute path to the {kind} the stage produced"
                    )
                try:
                    artifacts[field] = _resolve_artifact(raw, flag, kind)
                except ValueError:
                    if args.status != "error":
                        raise
                    unavailable.append(flag)

            extras: dict[str, Any] = {}
            for flag, kind in EXTRA_ARTIFACT_FLAGS.items():
                raw = getattr(args, _dest(flag))
                if raw is None:
                    continue
                try:
                    extras[_dest(flag)] = _resolve_artifact(raw, flag, kind)
                except ValueError:
                    if args.status != "error":
                        raise
                    unavailable.append(flag)
            if args.map_value is not None:
                extras["map_value"] = args.map_value

            # The weak-image count is gap_analysis's measurement and no other
            # stage's. `entry.update(extras)` writes it straight onto the phase
            # entry, so accepting it elsewhere lets a later stage overwrite the real
            # count with its own — and the count is what decides whether an early
            # stop is a completed run.
            if stage != "gap_analysis":
                offered = [
                    flag
                    for flag, given in (
                        ("--weak-image-count", args.weak_image_count is not None),
                        ("--zero-weak-images", args.zero_weak_images),
                    )
                    if given
                ]
                if offered:
                    raise ValueError(
                        f"{' and '.join(offered)} "
                        f"{'belong' if len(offered) > 1 else 'belongs'} to gap_analysis, not "
                        f"{stage!r}; only gap_analysis measures weak images"
                    )
            elif args.status == "ok":
                if args.weak_image_count is None and not args.zero_weak_images:
                    raise ValueError(
                        "stage 'gap_analysis' requires --weak-image-count N, the row count of "
                        "weak_images.parquet. 0 is the documented early stop and the only proof "
                        "of it the audit can read — a --summary line is free text nothing parses"
                    )
                if (
                    args.zero_weak_images
                    and args.weak_image_count is not None
                    and args.weak_image_count > 0
                ):
                    raise ValueError(
                        f"--zero-weak-images contradicts --weak-image-count "
                        f"{args.weak_image_count}; pass one or the other"
                    )
            # Pool exhaustion is measured before the mine that could not run, so it
            # is recorded on the stop itself rather than on a stage that never
            # committed. Restricted to loop_stop for the same reason the weak-image
            # count is restricted to gap_analysis: a completion claim must come from
            # the one place that measured it.
            # Both belong to loop_stop for the same reason the weak-image count
            # belongs to gap_analysis: a completion claim must come from the one
            # place that measured it.
            for flag, given in (("--pool-remaining", args.pool_remaining is not None),
                                ("--pool-exhausted", args.pool_exhausted)):
                if given and stage != "loop_stop":
                    raise ValueError(
                        f"{flag} belongs to loop_stop, not {stage!r}; it records why the "
                        f"loop stopped, not the outcome of a stage"
                    )
                if given and phase in ("prep", "baseline"):
                    raise ValueError(
                        f"{flag} on {phase!r}: only an iteration mines, so neither "
                        f"phase can exhaust the pool"
                    )
            if args.pool_remaining is not None:
                if args.pool_remaining < 0:
                    raise ValueError(
                        f"--pool-remaining must be >= 0, got {args.pool_remaining}"
                    )
                extras["pool_remaining"] = args.pool_remaining
            if args.pool_exhausted:
                extras["pool_exhausted"] = True

            if args.weak_image_count is not None:
                if args.weak_image_count < 0:
                    raise ValueError(
                        f"--weak-image-count must be >= 0, got {args.weak_image_count}"
                    )
                extras["weak_image_count"] = args.weak_image_count
            if args.zero_weak_images:
                extras["zero_weak_images"] = True

            for key, value in _parse_extras(unknown).items():
                # An undeclared flag must not be able to write a field a gate owns:
                # that would record a path nothing checked, or an early stop nothing
                # measured.
                owner = next(
                    (f for f, (field, _k, _s) in ARTIFACT_FLAGS.items() if field == key), None
                )
                if owner is not None:
                    raise ValueError(
                        f"--{key.replace('_', '-')} would overwrite the checked artifact field "
                        f"{key}; use {owner} so the path is verified"
                    )
                if key in EXTRA_ARTIFACT_FIELDS:
                    raise ValueError(
                        f"--{key.replace('_', '-')} would overwrite the checked path field "
                        f"{key}; use {EXTRA_ARTIFACT_FIELDS[key][0]} so the path is verified"
                    )
                if key in RESERVED_RECORD_FIELDS:
                    raise ValueError(
                        f"--{key.replace('_', '-')} would overwrite the validated field {key}; "
                        f"use {RESERVED_RECORD_FIELDS[key]}"
                    )
                if key in STATE_BOOKKEEPING_FIELDS:
                    raise ValueError(
                        f"--{key.replace('_', '-')} would overwrite {key}, which this script "
                        "derives from the stage being committed"
                    )
                extras[key] = value

            # ── 4. what the run already looked like, before this commit ───────
            before_rc, before_report, _before_stderr = _run_audit(results_dir)
            inherited: list[str] = (
                before_report.get("errors", []) if before_rc != 2 else []
            )

            # ── 5. journal, then state, then log — every write atomic ─────────
            snapshot = _snapshot(results_dir)
            now = datetime.datetime.now(datetime.timezone.utc)

            entry = phase_entry(state, phase)
            if not isinstance(entry, dict):
                raise ValueError(f"state.iterations.{phase} must be an object")
            entry.update(artifacts)
            entry.update(extras)

            # Which target classes are rare is a property of the pool, so it can
            # only be settled once prep has built one. init leaves it null on a run
            # that still has to prep; this fills it from the pool's own class counts
            # before `mine`, the first stage that reads it, can run.
            if stage == "prep" and args.status == "ok":
                resolved = _resolve_rare_classes(state, extras.get("pool_report"))
                if resolved:
                    print(resolved, file=sys.stderr)

            # The deferral above is only safe if the value is guaranteed present by
            # the time it matters. class_stratified apportions the budget by rare
            # class; with no list it silently degrades to global allocation and the
            # run stops protecting the classes it exists to improve.
            if stage == "mine" and args.status == "ok":
                config = state.get("config")
                config = config if isinstance(config, dict) else {}
                if (config.get("allocation_policy") == "class_stratified"
                        and not config.get("rare_class_list")):
                    raise ValueError(
                        "config.rare_class_list is unset but allocation_policy is "
                        "class_stratified. It is derived at the prep commit from "
                        "pool_report.json; commit prep with --pool-report, or re-init "
                        "with --rare-class-list"
                    )

            run_failed = failure is not None or state.get("status") == "failed"
            if stage == TERMINAL_STAGE:
                state["stopped_at"] = now.strftime("%Y-%m-%dT%H:%M:%SZ")
            if args.status == "error":
                entry["status"] = "failed"
                entry["failed_stage"] = stage
                state["status"] = "failed"
            elif stage == TERMINAL_STAGE:
                # loop_stop finalizes; it never sets stage_completed, because the
                # phase's last ok stage is whatever ran before it. Whether the run
                # is *complete* is not this writer's call — a loop_stop at
                # iteration 1 of 5 is a stop, not a completion — so it records
                # "stopped" and is upgraded below only once the audit agrees.
                state["status"] = "failed" if run_failed else "stopped"
            else:
                # The run-level status belongs to init (running) and to the terminal
                # events above; a mid-run stage only advances its own phase.
                entry["stage_completed"] = stage
                entry["status"] = "complete" if stage == order[-1] else "in_progress"

            if number is not None:
                current = state.get("current_iteration")
                current = (
                    current
                    if isinstance(current, int) and not isinstance(current, bool)
                    else 0
                )
                state["current_iteration"] = max(number, current)

            # seq comes from disk, never from memory: compaction is invisible here.
            seq = max(
                (
                    e["seq"]
                    for e in events
                    if isinstance(e.get("seq"), int) and not isinstance(e["seq"], bool)
                ),
                default=0,
            ) + 1
            event = {
                "seq": seq,
                "ts": now.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "iter": phase,
                "stage": stage,
                "status": args.status,
                "summary": args.summary.strip(),
                "duration_sec": args.duration_sec,
                "context_tokens": 0,  # reserved; no reliable per-stage source exists
            }

            _write_journal(results_dir, snapshot, phase, stage)
            dirty = True
            write_state_atomic(results_dir, state)
            write_log_atomic(results_dir, events + [event])
            _fsync_path(state_path(results_dir))
            _fsync_path(log_path(results_dir))
            _fsync_path(results_dir)

            # ── 6. the audit is the acceptance test — for THIS commit ─────────
            audit_rc, report, audit_stderr = _run_audit(results_dir)
            if not report:
                rejection = f"audit_deft_run.py produced no report (exit {audit_rc})"
            elif report.get("load_failed"):
                rejection = "the audit can no longer read deft_state.json"
            else:
                rejection = _commit_is_visible(
                    report, phase, stage, args.status, len(events) + 1
                )
                if rejection is None:
                    introduced = _new_errors(inherited, report.get("errors", []))
                    if introduced:
                        rejection = "it introduced:\n  - " + "\n  - ".join(introduced)
            if rejection is not None:
                sys.stderr.write(audit_stderr)
                print(
                    f"ERROR: {phase}/{stage} was rejected by the post-commit audit — "
                    f"{rejection}",
                    file=sys.stderr,
                )
                _restore(snapshot)
                rolled_back = True
                _clear_journal(results_dir)
                print(
                    "rolled back deft_state.json and loop_log.jsonl; both are unchanged",
                    file=sys.stderr,
                )
                return 2

            # loop_stop's run-level verdict comes from the audit, not from a
            # completion rule this writer would have to re-derive and could get
            # wrong on its own.
            if stage == TERMINAL_STAGE and args.status == "ok" and not run_failed:
                state["status"] = "complete" if report.get("complete") else "stopped"
                write_state_atomic(results_dir, state)
                _fsync_path(state_path(results_dir))

            _clear_journal(results_dir)
    except Exception as exc:  # noqa: BLE001
        if dirty and snapshot is not None and not rolled_back:
            try:
                _restore(snapshot)
                print(
                    "rolled back deft_state.json and loop_log.jsonl; both are unchanged",
                    file=sys.stderr,
                )
                if results_dir is not None:
                    _clear_journal(results_dir)
            except OSError as restore_exc:
                print(
                    f"CRITICAL: rollback failed ({restore_exc}); {JOURNAL_NAME} still records "
                    "how to undo this commit and the next commit_stage.py run will apply it",
                    file=sys.stderr,
                )
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2 if dirty else 1

    for flag in unavailable:
        print(f"WARNING: {flag} was not on disk; not recorded (status=error)", file=sys.stderr)
    if inherited:
        print(
            "WARNING: this run was already inconsistent before this commit, and still is:\n  - "
            + "\n  - ".join(inherited),
            file=sys.stderr,
        )
    print(
        f"committed seq={seq} {phase}/{stage} status={args.status} · "
        f"run={report.get('status', 'unknown')} · "
        f"next={report.get('next_action', 'unknown')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
