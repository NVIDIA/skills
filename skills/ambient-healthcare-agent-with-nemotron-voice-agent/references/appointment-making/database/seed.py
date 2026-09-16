# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Seed data for a fresh appointment-making demo database.

The fixture rows live in JSONL files under seed_data/. This module owns
the load and insert logic and materializes a rolling 365-day schedule
from the appointment type and doctor fixtures.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import date, timedelta
from pathlib import Path
from typing import Any

_SEED_DATA_DIR = Path(__file__).parent / "seed_data"
_APPOINTMENT_TYPES_PATH = _SEED_DATA_DIR / "appointment_types.jsonl"
_DOCTORS_PATH = _SEED_DATA_DIR / "doctors.jsonl"
_SLOT_ID_HEX_LENGTH = 8


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def seed_all(conn: sqlite3.Connection, *, start_date: date) -> None:
    """Insert fixture appointment slots into an empty database."""
    appointment_types = _load_jsonl(_APPOINTMENT_TYPES_PATH)
    doctors = _load_jsonl(_DOCTORS_PATH)
    if not appointment_types:
        raise RuntimeError(f"Missing appointment type seed rows: {_APPOINTMENT_TYPES_PATH}")
    if not doctors:
        raise RuntimeError(f"Missing doctor seed rows: {_DOCTORS_PATH}")

    legacy_index = 0
    with conn:
        for day_offset in range(365):
            appointment_date = start_date + timedelta(days=day_offset)
            for type_index, type_row in enumerate(appointment_types):
                appointment_type = str(type_row["appointment_type"])
                slot_times = list(type_row.get("slot_times") or [])
                for slot_index, slot_time in enumerate(slot_times):
                    appointment_datetime = f"{appointment_date.isoformat()} {slot_time}"
                    doctor = str(doctors[(day_offset + type_index + slot_index + 2) % len(doctors)]["doctor"])
                    digest_input = "|".join(
                        (
                            appointment_type.lower(),
                            appointment_datetime.lower(),
                            doctor.lower(),
                            str(slot_index),
                        )
                    )
                    digest = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()
                    slot_id = "SLOT-" + digest[:_SLOT_ID_HEX_LENGTH].upper()
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO appointment_schedule (
                            slot_id, legacy_index, datetime, doctor, appointment_type
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (slot_id, legacy_index, appointment_datetime, doctor, appointment_type),
                    )
                    legacy_index += 1
