# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Demo-only patient intake tool logic for the NVA generic pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATA_DIR = Path(os.getenv("NVA_HEALTHCARE_DEMO_DATA_DIR", str(Path.cwd() / "data/patient-intake")))
RECORDS_PATH = DATA_DIR / "patient_intake_records.jsonl"

REQUIRED_INTAKE_FIELDS = (
    "patient_name",
    "date_of_birth",
    "symptoms",
    "current_medications",
    "current_pharmacy",
)

FIELD_LABELS = {
    "patient_name": "patient name",
    "date_of_birth": "date of birth",
    "symptoms": "symptoms or visit reason",
    "current_medications": "current medications",
    "current_pharmacy": "current pharmacy",
}


def review_patient_intake_result(arguments: Mapping[str, Any] | None) -> dict[str, Any]:
    """Build a speech-friendly review without persisting the intake."""
    payload = _normalized_intake_payload(arguments or {}, REQUIRED_INTAKE_FIELDS)
    missing_fields = [field for field in REQUIRED_INTAKE_FIELDS if not payload.get(field)]
    if missing_fields:
        return {
            "status": "needs_more_information",
            "missing_fields": missing_fields,
            "provided_fields": sorted(payload),
            "response_text": _missing_fields_response(missing_fields),
            "note": "demo only; no patient record was persisted",
        }

    return {
        "status": "review_required",
        "response_text": _review_response(payload),
        "note": "demo only; no patient record was persisted",
    }


def record_patient_intake_result(arguments: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate and persist a demo patient intake record."""
    raw_arguments = arguments or {}
    payload = _normalized_intake_payload(raw_arguments, REQUIRED_INTAKE_FIELDS)
    missing_fields = [field for field in REQUIRED_INTAKE_FIELDS if not payload.get(field)]
    if missing_fields:
        return {
            "status": "needs_more_information",
            "missing_fields": missing_fields,
            "provided_fields": sorted(payload),
            "response_text": _missing_fields_response(missing_fields),
            "note": "demo only; no patient record was persisted",
        }
    if raw_arguments.get("patient_confirmed") is not True:
        return {
            "status": "confirmation_required",
            "response_text": "Please confirm that the reviewed information is correct before I save it.",
            "note": "demo only; no patient record was persisted",
        }

    record_id = _intake_identifier("INTAKE", payload, REQUIRED_INTAKE_FIELDS)
    record = {
        "record_id": record_id,
        "created_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017 - validator supports Python 3.10
        "type": "patient_intake",
        **payload,
    }
    _append_jsonl(RECORDS_PATH, record)
    return {
        "status": "saved",
        "record_id": record_id,
        "data_path": str(RECORDS_PATH),
        "response_text": (
            f"Thank you, your information has been saved. Your confirmation number is {record_id}."
        ),
        "note": "demo only; replace JSONL storage before handling production PHI",
    }


def _normalized_intake_payload(arguments: Mapping[str, Any], fields: tuple[str, ...]) -> dict[str, str]:
    return {
        field: str(arguments.get(field, "") or "").strip()
        for field in fields
        if str(arguments.get(field, "") or "").strip()
    }


def _missing_fields_response(missing_fields: list[str]) -> str:
    labels = [FIELD_LABELS.get(field, field.replace("_", " ")) for field in missing_fields]
    if len(labels) == 1:
        return f"Please share your {labels[0]}."
    return "Please share your " + ", ".join(labels[:-1]) + f", and {labels[-1]}."


def _review_response(payload: Mapping[str, str]) -> str:
    """Narrate the intake fields naturally and ask for confirmation."""
    patient_name = payload["patient_name"].strip().rstrip(".")
    spoken_date = _format_date_for_speech(payload["date_of_birth"])
    visit_clause = _format_visit_for_speech(payload["symptoms"])
    medications = payload["current_medications"].strip().rstrip(".")
    pharmacy = payload["current_pharmacy"].strip().rstrip(".")
    return (
        f"Thank you. I have all the information I need. Your name is {patient_name}, "
        f"your date of birth is {spoken_date}, {visit_clause}, your current medications are "
        f"{medications}, and your current pharmacy is {pharmacy}. "
        "Is this all correct?"
    )


def _format_date_for_speech(value: str) -> str:
    """Convert common numeric dates into natural English speech."""
    raw_value = value.strip()
    spoken_date = re.fullmatch(
        r"(?P<month>[A-Za-z]+)\s+(?P<day>\d{1,2})(?P<suffix>st|nd|rd|th),?\s+(?P<year>\d{4})",
        raw_value,
        flags=re.IGNORECASE,
    )
    if spoken_date:
        return (
            f"{spoken_date.group('month')} {spoken_date.group('day')}"
            f"{spoken_date.group('suffix')} {spoken_date.group('year')}"
        )
    for date_format in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%m-%d-%Y"):
        try:
            parsed = datetime.strptime(raw_value, date_format)
        except ValueError:
            continue
        day = parsed.day
        if 10 < day % 100 < 14:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
        return f"{parsed.strftime('%B')} {day}{suffix} {parsed.year}"
    return raw_value


def _format_visit_for_speech(value: str) -> str:
    """Return a natural clause for symptoms or a planned service."""
    phrase = value.strip().rstrip(".")
    normalized = phrase.lower()
    visit_reason_markers = (
        "appointment",
        "checkup",
        "check-up",
        "consultation",
        "exam",
        "follow-up",
        "follow up",
        "physical",
        "refill",
        "screening",
        "shot",
        "test",
        "vaccination",
        "vaccine",
    )
    if any(marker in normalized for marker in visit_reason_markers):
        return f"your reason for the visit is {_format_visit_reason(phrase)}"
    return f"you're currently experiencing {_with_symptom_article(phrase)}"


def _format_visit_reason(value: str) -> str:
    """Turn common first-person appointment reasons into natural recap language."""
    phrase = re.sub(r"\bmy\b", "your", value.strip().rstrip("."), flags=re.IGNORECASE)
    needed = re.match(r"^(?:i\s+)?need(?:\s+to)?\s+(?P<reason>.+)$", phrase, flags=re.IGNORECASE)
    if needed:
        reason = needed.group("reason").strip()
        if any(marker in reason.lower() for marker in ("dose", "shot", "vaccination", "vaccine")):
            return f"to receive {reason}"
        return f"because you need {reason}"
    phrase = re.sub(r"^(?:i(?:'m| am)\s+)?here\s+for\s+", "", phrase, flags=re.IGNORECASE)
    if phrase.lower().startswith("to "):
        return phrase
    return _with_indefinite_article(phrase)


def _with_indefinite_article(value: str) -> str:
    """Add an article to a singular noun phrase when the user did not provide one."""
    phrase = value.strip().rstrip(".")
    if phrase.lower().startswith(("a ", "an ", "the ", "my ", "our ", "your ")):
        return phrase
    article = "an" if phrase[:1].lower() in "aeiou" else "a"
    return f"{article} {phrase}"


def _with_symptom_article(value: str) -> str:
    """Add an article to common singular symptom phrases."""
    phrase = value.strip().rstrip(".")
    normalized = phrase.lower()
    if normalized == "flu":
        return "the flu"
    if normalized.startswith(("a ", "an ", "the ", "some ", "my ", "our ", "your ")):
        return phrase
    uncountable = ("fatigue", "nausea", "pain", "shortness of breath", "vomiting")
    countable_symptom_starts = ("cold", "cough", "headache", "rash", "sore throat")
    if normalized.startswith(countable_symptom_starts):
        return _with_indefinite_article(phrase)
    if normalized in uncountable or "," in phrase or " and " in normalized or normalized.endswith("s"):
        return phrase
    return _with_indefinite_article(phrase)


def _intake_identifier(prefix: str, payload: Mapping[str, str], fields: tuple[str, ...]) -> str:
    digest_input = "|".join(payload[field].lower() for field in fields)
    return prefix + "-" + hashlib.sha256(digest_input.encode("utf-8")).hexdigest()[:8].upper()


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


TOOL_RESULT_FUNCTIONS = {
    "review_patient_intake": review_patient_intake_result,
    "record_patient_intake": record_patient_intake_result,
}
