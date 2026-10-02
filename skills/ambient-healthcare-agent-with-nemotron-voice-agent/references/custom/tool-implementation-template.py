# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Custom ambient healthcare tool template for the NVA generic pipeline."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

DATA_DIR = Path(os.getenv("NVA_CUSTOM_HEALTHCARE_DEMO_DATA_DIR", str(Path.cwd() / "data/custom-healthcare")))
CUSTOM_RECORDS_PATH = DATA_DIR / "custom_healthcare_records.jsonl"

REQUIRED_CUSTOM_FIELDS = (
    "required_field_one",
    "required_field_two",
    "required_field_three",
)

FIELD_LABELS = {
    "required_field_one": "required field one",
    "required_field_two": "required field two",
    "required_field_three": "required field three",
}


def custom_healthcare_tool_result(arguments: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate and persist a developer-defined ambient healthcare record."""
    payload = _normalized_custom_payload(arguments or {}, REQUIRED_CUSTOM_FIELDS)
    missing_fields = [field for field in REQUIRED_CUSTOM_FIELDS if not payload.get(field)]
    if missing_fields:
        return {
            "status": "needs_more_information",
            "missing_fields": missing_fields,
            "provided_fields": sorted(payload),
            "response_text": _missing_fields_response(missing_fields),
            "note": "demo only; no custom record was persisted",
        }

    custom_healthcare_request_id = _record_id("CUSTOM", payload)
    record = {
        "custom_healthcare_request_id": custom_healthcare_request_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "type": "custom_healthcare_request",
        **payload,
    }
    _append_jsonl(CUSTOM_RECORDS_PATH, record)
    return {
        "status": "saved",
        "custom_healthcare_request_id": custom_healthcare_request_id,
        "data_path": str(CUSTOM_RECORDS_PATH),
        "provided_fields": sorted(payload),
        "response_text": f"Custom healthcare request {custom_healthcare_request_id} was saved.",
        "note": "demo only; replace JSONL storage before handling production healthcare data",
    }


def _normalized_custom_payload(arguments: Mapping[str, Any], fields: tuple[str, ...]) -> dict[str, str]:
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


def _record_id(prefix: str, payload: Mapping[str, str]) -> str:
    digest_input = "|".join(payload[field].lower() for field in REQUIRED_CUSTOM_FIELDS)
    return prefix + "-" + hashlib.sha256(digest_input.encode("utf-8")).hexdigest()[:8].upper()


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


TOOL_RESULT_FUNCTIONS = {
    "custom_healthcare_tool": custom_healthcare_tool_result,
}
