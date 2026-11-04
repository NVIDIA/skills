-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: Apache-2.0

-- Appointment-making demo SQLite schema.
-- Applied on startup via db.apply_schema. Safe to re-run.

CREATE TABLE IF NOT EXISTS appointment_schedule (
    slot_id           TEXT PRIMARY KEY,
    legacy_index      INTEGER NOT NULL,
    datetime          TEXT    NOT NULL,
    doctor            TEXT    NOT NULL,
    appointment_type  TEXT    NOT NULL,
    patient           TEXT,
    patient_name      TEXT,
    date_of_birth     TEXT,
    visit_reason      TEXT,
    booked_at         TEXT,
    booking_id        TEXT
);

CREATE INDEX IF NOT EXISTS idx_appointment_schedule_type_datetime
    ON appointment_schedule (appointment_type, datetime);
CREATE INDEX IF NOT EXISTS idx_appointment_schedule_booking_id
    ON appointment_schedule (booking_id);
