# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for speech-friendly appointment dates."""

import sys
from pathlib import Path

NVA_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NVA_ROOT / "src"))

from examples.generic import ambient_agent_tools  # noqa: E402


def _appointment_instructions() -> str:
    catalog_text = (NVA_ROOT / "src/examples/generic/prompts.yaml").read_text()
    return catalog_text.split("appointment_making_healthcare:", 1)[1]


def test_appointment_prompt_preserves_conversation_state():
    """The prompt must prefer current state and prevent duplicate lookups."""
    prompt = _appointment_instructions()

    assert "CRITICAL RECENCY" in prompt
    assert "CRITICAL COMPLETED-LOOKUP STATE" in prompt
    assert "Never call find_available_appointments again" in prompt


def test_appointment_prompt_uses_safe_transition_wording():
    """Fragile short turns and demo writes have explicit safe wording."""
    prompt = _appointment_instructions()

    assert "Would you prefer a morning or afternoon appointment?" in prompt
    assert "Hello! Which previously offered option works for you?" in prompt
    assert "The demo booking was saved with booking ID <booking_id>" in prompt


def test_appointment_dates_use_abbreviated_month_year_and_no_comma():
    """User-facing slot dates must use the exact voice-friendly date contract."""
    expected_dates = {
        "1999-01-08 09:00:00": "Jan 8th 1999 at 9:00 AM",
        "1994-10-06 09:00:00": "Oct 6th 1994 at 9:00 AM",
        "2026-01-01 14:00:00": "Jan 1st 2026 at 2:00 PM",
        "2026-02-02 08:30:00": "Feb 2nd 2026 at 8:30 AM",
        "2026-03-03 16:00:00": "Mar 3rd 2026 at 4:00 PM",
        "2026-11-11 11:00:00": "Nov 11th 2026 at 11:00 AM",
        "2026-12-21 12:00:00": "Dec 21st 2026 at 12:00 PM",
    }

    for raw_datetime, spoken_datetime in expected_dates.items():
        actual = ambient_agent_tools._speech_datetime(raw_datetime)
        assert actual == spoken_datetime
        assert "," not in actual

    prompt = _appointment_instructions()
    assert "Jan 8th 1999" in prompt
    assert "never Jan 8th, 1999" in prompt


def test_appointment_slot_summary_uses_spoken_date_contract():
    """Fallback slot summaries must not leak a raw timestamp or date comma."""
    summary = ambient_agent_tools._slot_summary(
        {
            "appointment_datetime": "2026-09-01 14:00:00",
            "doctor": "Dr. Priya Shah",
        }
    )

    assert summary == "Sep 1st 2026 at 2:00 PM with Dr. Priya Shah"
    assert "2026-09-01" not in summary
    assert "," not in summary
