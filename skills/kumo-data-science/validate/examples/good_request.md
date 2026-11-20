# Example: Good Request (passes intake, then two confirmations)

## Input

> "Create a notebook that trains a Kumo model for our fraud use case. The data is in
> `USER_DATA.E_COMMERCE`. Entity: each customer. Outcome: at least one fraud event. Window: next 30 days."

Note what is **not** in the request: no column names, no types, no join keys, no output table.
That is a complete training request.

---

## Node 1a — Required fields: PASS
- operation: `train` ✓
- `source_scope`: `USER_DATA.E_COMMERCE` ✓
- `prediction_entity`: each customer ✓
- `prediction_target`: at least one fraud event (positive-label condition stated) ✓
- `time_window`: `specified`, 30 days ✓

## Node 1b — Atomic question: PASS
- Single target, operationally defined, horizon explicit ✓

## Node 1c — Preflight & graph gate

Agent runs read-only preflight (`accept/preflight.md`) and replies — then stops:

> Source scope: `USER_DATA.E_COMMERCE` [supplied] — 7 tables found
> Table mode: all-connected [default] — 7 of 7 tables reachable from the seeds
>
> **Metadata**
> - customers: primary key CUSTOMER_ID [inferred]
> - fraud_events: primary key EVENT_ID [inferred]
> - fraud_events: time column EVENT_AT [inferred]
> - bookings: primary key BOOKING_ID [inferred]; time column BOOKED_AT [inferred]
>
> **Links**
> - fraud_events.CUSTOMER_ID -> customers.CUSTOMER_ID [inferred]
> - bookings.CUSTOMER_ID -> customers.CUSTOMER_ID [inferred]
>
> **Excluded columns**
> - bookings.CANCELLED_AT [inferred] — not knowable at the row's BOOKED_AT
> - bookings.NIGHTS_STAYED [inferred] — realized duration of a stay that has not ended
>
> **Not found**
> - payment_lines: no primary key inferred [inferred] — treated as keyless; edge source only
>
> Confirm, or correct any line — your values take precedence.

User: "Confirmed."

## Node 1d — Task interpretation gate

Agent replies — then stops:

> - Entity: one prediction per customers.CUSTOMER_ID
> - Outcome: at least one fraud_events.EVENT_ID occurs
> - Window: 0 to 30 days after the prediction anchor
> - Task type: temporal binary classification
> - Output: notebook only, no predictions written
> - Threshold: 0.5 [default]
> - Embeddings: not requested
>
> Shall I generate the notebook?

User: "Yes."

## ACCEPT → Step 2

Structured handoff payload to `kumoai-reference/SKILL.md`:
```
operation: train
source_scope: USER_DATA.E_COMMERCE
table_mode: all-connected
tables:
  - name: CUSTOMERS       primary_key: CUSTOMER_ID
  - name: FRAUD_EVENTS    primary_key: EVENT_ID   time_column: EVENT_AT
  - name: BOOKINGS        primary_key: BOOKING_ID time_column: BOOKED_AT
      excluded_columns: [CANCELLED_AT, NIGHTS_STAYED]   # carried into column_names
excluded_columns_confirmed: true
links:
  - FRAUD_EVENTS.CUSTOMER_ID -> CUSTOMERS.CUSTOMER_ID
  - BOOKINGS.CUSTOMER_ID -> CUSTOMERS.CUSTOMER_ID
target:
  entity: CUSTOMERS.CUSTOMER_ID
  definition: at least one FRAUD_EVENTS row in window
  horizon_days: 30
  task_type: temporal binary classification
metric: AUC-ROC (default)
artifact: notebook — fit, then predict locally (no destination); one blocking fit call
```

If the user had written "…and write predictions to `SCORES_DB.PUBLIC.FRAUD_SCORES`", the Output
line at Node 1d would read "predictions written to `SCORES_DB.PUBLIC.FRAUD_SCORES` in the same
notebook", the payload would carry `output_location: SCORES_DB.PUBLIC.FRAUD_SCORES`, and the
notebook's predict call would target it. It is never asked for.

---

## Variant: the same request in two turns

Turn 1: "We want a 30-day fraud model on `USER_DATA.E_COMMERCE` — flag customers who will have any
fraud event." → all four required fields present; agent goes straight to Node 1c.

Turn 1: "Build me a fraud model for customers." → `source_scope` and `time_window` missing;
agent asks for exactly those two, nothing about schema or output.

---
