# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SQLite connection helpers for the appointment-making demo database.

The database path comes from NVA_APPOINTMENT_DB_PATH. Schema is applied
idempotently on startup, and seed runs only when the schedule is empty,
missing the required schema, or too stale for current relative-date demos.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

DEFAULT_DB_PATH = "./data/appointment-making/appointment_schedule.sqlite"
_SCHEMA_PATH = Path(__file__).parent / "schema.sql"
_BUSY_TIMEOUT_SECONDS = 5.0
_REQUIRED_COLUMNS = {
    "slot_id",
    "legacy_index",
    "datetime",
    "doctor",
    "appointment_type",
    "patient",
    "patient_name",
    "date_of_birth",
    "visit_reason",
    "booked_at",
    "booking_id",
}


def db_path() -> Path:
    path = Path(os.environ.get("NVA_APPOINTMENT_DB_PATH", DEFAULT_DB_PATH))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def database_start_date() -> date:
    configured = os.environ.get("NVA_APPOINTMENT_DB_START_DATE", "")
    if configured:
        try:
            return date.fromisoformat(configured[:10])
        except ValueError:
            pass
    return date.today()


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(
        str(db_path()),
        check_same_thread=False,
        timeout=_BUSY_TIMEOUT_SECONDS,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def apply_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))


def init_db() -> Path:
    """Apply schema and seed if needed, preserving populated current databases."""
    path = db_path()
    conn = connect()
    try:
        apply_schema(conn)
        reason = _reseed_reason(conn)
    finally:
        conn.close()

    if reason in {"schema_mismatch", "stale"}:
        _backup_and_remove(path)
        conn = connect()
        try:
            apply_schema(conn)
            _seed(conn)
        finally:
            conn.close()
        return path

    if reason == "empty":
        conn = connect()
        try:
            _seed(conn)
        finally:
            conn.close()
    return path


def _reseed_reason(conn: sqlite3.Connection) -> str | None:
    columns = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(appointment_schedule)").fetchall()
    }
    if not _REQUIRED_COLUMNS.issubset(columns):
        return "schema_mismatch"

    row = conn.execute(
        "SELECT COUNT(*) AS count, MIN(date(datetime)) AS min_date, MAX(date(datetime)) AS max_date "
        "FROM appointment_schedule"
    ).fetchone()
    if not row or row["count"] == 0:
        return "empty"

    start_date = database_start_date()
    minimum_end_date = start_date + timedelta(days=300)
    if not row["min_date"] or not row["max_date"]:
        return "empty"
    if row["min_date"] > start_date.isoformat() or row["max_date"] < minimum_end_date.isoformat():
        return "stale"
    return None


def _seed(conn: sqlite3.Connection) -> None:
    from examples.generic.ambient_healthcare_appointment_database.seed import seed_all

    seed_all(conn, start_date=database_start_date())


def _backup_and_remove(path: Path) -> None:
    if path.exists():
        backup_path = path.with_suffix(path.suffix + f".bak-{datetime.now(timezone.utc):%Y%m%d%H%M%S}")
        path.replace(backup_path)
    for candidate in (path, Path(str(path) + "-wal"), Path(str(path) + "-shm")):
        candidate.unlink(missing_ok=True)
