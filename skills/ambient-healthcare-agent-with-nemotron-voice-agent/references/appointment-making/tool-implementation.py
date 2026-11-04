# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Demo-only appointment lookup and booking tools for the NVA generic pipeline."""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from typing import Any, Mapping

from examples.generic.ambient_healthcare_appointment_database import db as appointment_db

REQUIRED_BOOKING_FIELDS = (
    "appointment_type",
    "appointment_datetime",
    "patient_name",
    "date_of_birth",
    "visit_reason",
)
DEMO_WARNING = "demo only; verify in a clinical scheduling system before relying on it"
VALID_TIME_WINDOWS = ("morning", "afternoon")
MONTH_ABBREVIATIONS = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)


def find_available_appointments_result(arguments: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return available demo appointment slots from the SQLite schedule."""
    payload = _clean_payload(arguments or {})
    missing_fields = [
        field
        for field in ("appointment_type", "start_date", "end_date", "time_window")
        if not payload.get(field)
    ]
    if missing_fields:
        return {
            "status": "needs_more_information",
            "missing_fields": missing_fields,
            "response_text": _missing_fields_response(missing_fields),
            "note": DEMO_WARNING,
        }

    time_window = payload["time_window"].lower()
    if time_window not in VALID_TIME_WINDOWS:
        return {
            "status": "invalid_time_window",
            "valid_time_windows": list(VALID_TIME_WINDOWS),
            "response_text": "Please ask whether the user prefers morning or afternoon before checking availability.",
            "note": DEMO_WARNING,
        }

    appointment_db.init_db()
    valid_types = _valid_appointment_types()
    appointment_type = payload["appointment_type"]
    if appointment_type not in valid_types:
        return {
            "status": "invalid_appointment_type",
            "valid_appointment_types": valid_types,
            "response_text": "Please choose a supported appointment type before checking availability.",
            "note": DEMO_WARNING,
        }

    start_date = payload["start_date"][:10]
    end_date = payload["end_date"][:10]
    if not _is_iso_date(start_date) or not _is_iso_date(end_date):
        return {
            "status": "invalid_date",
            "response_text": "Search dates must be in year-month-date format.",
            "note": DEMO_WARNING,
        }
    if end_date < start_date:
        return {
            "status": "invalid_date_range",
            "response_text": "The appointment search end date must be on or after the start date.",
            "note": DEMO_WARNING,
        }

    slots = _find_slots(
        appointment_type=appointment_type,
        start_date=start_date,
        end_date=end_date,
        time_window=time_window,
        limit=3,
    )
    if not slots:
        return {
            "status": "none_found",
            "appointment_type": appointment_type,
            "start_date": start_date,
            "end_date": end_date,
            "available_appointments": [],
            "data_path": str(appointment_db.db_path()),
            "response_text": "No available appointments were found for that appointment type and date range.",
            "note": DEMO_WARNING,
        }

    return {
        "status": "found",
        "appointment_type": appointment_type,
        "start_date": start_date,
        "end_date": end_date,
        "available_appointments": slots,
        "data_path": str(appointment_db.db_path()),
        "response_text": "Available appointments include " + _speech_list(_slot_summary(slot) for slot in slots) + ".",
        "note": DEMO_WARNING,
    }


def book_appointment_result(arguments: Mapping[str, Any] | None) -> dict[str, Any]:
    """Write the selected demo booking into the SQLite schedule."""
    payload = _clean_payload(arguments or {})
    missing_fields = [field for field in REQUIRED_BOOKING_FIELDS if not payload.get(field)]
    if missing_fields:
        return {
            "status": "needs_more_information",
            "missing_fields": missing_fields,
            "response_text": _missing_fields_response(missing_fields),
            "note": DEMO_WARNING,
        }

    appointment_db.init_db()
    valid_types = _valid_appointment_types()
    appointment_type = payload["appointment_type"]
    if appointment_type not in valid_types:
        return {
            "status": "invalid_appointment_type",
            "valid_appointment_types": valid_types,
            "response_text": "Please choose a supported appointment type before booking.",
            "note": DEMO_WARNING,
        }

    try:
        appointment_datetime = _normalize_datetime(payload["appointment_datetime"])
    except ValueError as exc:
        return {
            "status": "invalid_datetime",
            "response_text": str(exc),
            "note": DEMO_WARNING,
        }

    booking_id = _booking_identifier(
        "BOOK",
        {
            "appointment_type": appointment_type,
            "appointment_datetime": appointment_datetime,
            "patient_name": payload["patient_name"],
            "date_of_birth": payload["date_of_birth"],
        },
        ("appointment_type", "appointment_datetime", "patient_name", "date_of_birth"),
    )
    booked_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    with appointment_db.connect() as connection:
        cursor = connection.cursor()
        if payload.get("slot_id"):
            cursor.execute(
                """
                UPDATE appointment_schedule
                   SET patient = ?,
                       patient_name = ?,
                       date_of_birth = ?,
                       visit_reason = ?,
                       booked_at = ?,
                       booking_id = ?
                 WHERE slot_id = ?
                   AND appointment_type = ?
                   AND patient IS NULL
                """,
                (
                    "current_patient",
                    payload["patient_name"],
                    payload["date_of_birth"],
                    payload["visit_reason"],
                    booked_at,
                    booking_id,
                    payload["slot_id"],
                    appointment_type,
                ),
            )
        else:
            cursor.execute(
                """
                UPDATE appointment_schedule
                   SET patient = ?,
                       patient_name = ?,
                       date_of_birth = ?,
                       visit_reason = ?,
                       booked_at = ?,
                       booking_id = ?
                 WHERE datetime = ?
                   AND appointment_type = ?
                   AND patient IS NULL
                """,
                (
                    "current_patient",
                    payload["patient_name"],
                    payload["date_of_birth"],
                    payload["visit_reason"],
                    booked_at,
                    booking_id,
                    appointment_datetime,
                    appointment_type,
                ),
            )
        connection.commit()

        if cursor.rowcount != 1:
            return {
                "status": "unavailable",
                "booking_id": booking_id,
                "appointment_type": appointment_type,
                "appointment_datetime": appointment_datetime,
                "data_path": str(appointment_db.db_path()),
                "response_text": (
                    "That appointment slot is no longer available. "
                    "Check availability again before booking."
                ),
                "note": DEMO_WARNING,
            }

        row = connection.execute(
            """
            SELECT slot_id, datetime, doctor, appointment_type, patient_name, date_of_birth,
                   visit_reason, booked_at, booking_id
              FROM appointment_schedule
             WHERE booking_id = ?
            """,
            (booking_id,),
        ).fetchone()

    return {
        "status": "booked",
        "booking_id": booking_id,
        "appointment": _row_to_booking(row),
        "data_path": str(appointment_db.db_path()),
        "response_text": f"Appointment booking {booking_id} was saved.",
        "note": DEMO_WARNING,
    }


def _clean_payload(arguments: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(field): str(value or "").strip()
        for field, value in arguments.items()
        if str(value or "").strip()
    }


def _valid_appointment_types() -> list[str]:
    with appointment_db.connect() as connection:
        rows = connection.execute(
            "SELECT DISTINCT appointment_type FROM appointment_schedule ORDER BY appointment_type"
        ).fetchall()
    return [str(row["appointment_type"]) for row in rows]


def _missing_fields_response(missing_fields: list[str]) -> str:
    labels = [field.replace("_", " ") for field in missing_fields]
    if len(labels) == 1:
        return f"Please share the {labels[0]}."
    return "Please share the " + ", ".join(labels[:-1]) + f", and {labels[-1]}."


def _is_iso_date(value: str) -> bool:
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return False
    return True


def _normalize_datetime(value: str) -> str:
    raw_value = value.strip().replace("T", " ")
    try:
        parsed = datetime.fromisoformat(raw_value.replace("Z", "+00:00"))
    except ValueError:
        parsed = None
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %I:%M %p", "%Y-%m-%d %I %p"):
            try:
                parsed = datetime.strptime(raw_value, fmt)
                break
            except ValueError:
                continue
        if parsed is None:
            raise ValueError("appointment_datetime must include a date and time, such as 2026-08-24 09:00:00.")
    return parsed.strftime("%Y-%m-%d %H:%M:%S")


def _find_slots(
    *,
    appointment_type: str,
    start_date: str,
    end_date: str,
    time_window: str,
    limit: int,
) -> list[dict[str, str]]:
    lower_hour, upper_hour = _time_window_hours(time_window)
    query = """
        SELECT slot_id, datetime, doctor, appointment_type
          FROM appointment_schedule
         WHERE appointment_type = ?
           AND patient IS NULL
           AND date(datetime) BETWEEN ? AND ?
    """
    params: list[Any] = [appointment_type, start_date, end_date]
    if lower_hour is not None and upper_hour is not None:
        query += " AND CAST(strftime('%H', datetime) AS INTEGER) BETWEEN ? AND ?"
        params.extend([lower_hour, upper_hour])
    query += " ORDER BY datetime LIMIT ?"
    params.append(limit)

    with appointment_db.connect() as connection:
        rows = connection.execute(query, params).fetchall()
    return [_row_to_slot(row) for row in rows]


def _time_window_hours(time_window: str) -> tuple[int | None, int | None]:
    return {
        "morning": (0, 11),
        "afternoon": (12, 16),
    }[time_window.lower()]


def _row_to_slot(row: sqlite3.Row) -> dict[str, str]:
    appointment_datetime = str(row["datetime"])
    doctor = str(row["doctor"])
    return {
        "slot_id": str(row["slot_id"]),
        "appointment_datetime": appointment_datetime,
        "spoken_time_slot": f"{_speech_datetime(appointment_datetime)} with {doctor}",
        "doctor": doctor,
        "appointment_type": str(row["appointment_type"]),
    }


def _row_to_booking(row: sqlite3.Row) -> dict[str, str]:
    return {key: str(row[key] or "") for key in row.keys()}


def _slot_summary(slot: Mapping[str, str]) -> str:
    if slot.get("spoken_time_slot"):
        return str(slot["spoken_time_slot"])
    return f"{_speech_datetime(slot['appointment_datetime'])} with {slot['doctor']}"


def _speech_list(items: Any) -> str:
    values = [str(item) for item in items if str(item)]
    if len(values) <= 1:
        return "".join(values)
    if len(values) == 2:
        return f"{values[0]}, and {values[1]}"
    return ", ".join(values[:-1]) + f", and {values[-1]}"


def _speech_datetime(value: str) -> str:
    try:
        parsed = datetime.strptime(str(value), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return str(value)
    time_text = parsed.strftime("%I:%M %p").lstrip("0")
    month = MONTH_ABBREVIATIONS[parsed.month - 1]
    return f"{month} {_ordinal_day(parsed.day)} {parsed.year} at {time_text}"


def _ordinal_day(day: int) -> str:
    if 10 <= day % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return f"{day}{suffix}"


def _booking_identifier(prefix: str, payload: Mapping[str, str], fields: tuple[str, ...]) -> str:
    digest_input = "|".join(payload[field].lower() for field in fields)
    return prefix + "-" + hashlib.sha256(digest_input.encode("utf-8")).hexdigest()[:8].upper()


TOOL_RESULT_FUNCTIONS = {
    "find_available_appointments": find_available_appointments_result,
    "book_appointment": book_appointment_result,
}
