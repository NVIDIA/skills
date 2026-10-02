# Example Design Choices

This reference documents the bundled `appointment-making` and `patient-intake` overlays for the NVA `src/examples/generic/` example.

## Shared Generic NVA Strategy

Both overlays customize the existing NVA generic Pipecat pipeline in place. They add prompt entries, OpenAI-style tool schemas, Python tool result functions, and generic tool handler wiring under `src/examples/generic/`.

Both overlays use the LLM, ASR, and TTS selected for the target NVA `generic-assistant` example. Resolve the defaults and available catalog entries from the target checkout with `scripts/inspect_nva_generic_defaults.py`; do not hard-code model or service names from a specific NVA release. The default runtime uses public NVIDIA AI Endpoints from the generic cloud catalog for all three services. The user may instead choose NVA-managed local NIMs, existing NIM endpoints, or a mixed layout; `references/deployment-modes.md` defines those branches.

Both overlays keep the tool contract local to the generic assistant process: the LLM calls a Pipecat function tool and the generated handler invokes the Python implementation in `examples.generic.ambient_agent_tools`. Patient-intake review and save results provide deterministic speech that the handler sends directly through the normal text and TTS path; other results remain available for the assistant to summarize.

## Appointment-Making

Appointment-making is a two-tool workflow:

- `find_available_appointments`: read available slots from the demo SQLite schedule.
- `book_appointment`: write the selected intended booking back to the same SQLite schedule.

At session start, the appointment overlay queues `Hello and welcome to the appointment making agent. Let's get started. First, could you please tell me what type of appointment you're looking for?` directly through the text and TTS path. The shared startup handler's generated LLM intro is disabled for this prompt, so the opening is fixed and emitted once. When welcome messages are disabled, the agent waits for the user to speak.

### Appointment conversation stages

The prompt defines an explicit state machine:

1. Welcome the user and ask only for the appointment type.
2. Once any appointment type is known, ask for a user-selected date or date range and an explicit morning-or-afternoon preference. Appointment type alone never authorizes lookup.
3. When appointment type, user-provided date range, and time window are known, call `find_available_appointments` in that same turn without a spoken preamble.
4. Present no more than three matching slots in one speech-friendly sentence and ask the user to choose one.
5. After a slot is selected, ask for patient name, date of birth, and visit reason in three separate turns, in that order.
6. Call `book_appointment` only after the selected slot and all three explicitly supplied booking fields are present.

### Appointment safeguards and speech

Availability lookup deliberately does not require patient identity, date of birth, insurance, contact information, or visit reason. The required `time_window` field is restricted to `morning` or `afternoon`; the handler rejects a missing window and values such as `any time`, so prompt failure cannot silently broaden the search. The handler also caps results at three. These tool-side checks backstop the conversational state machine instead of relying on prompt compliance alone.

The booking stage treats appointment type and visit reason as separate fields. A selected slot is not a booking, and selecting a uniquely identifiable time, date, doctor, option position, or slot ID must advance to the next missing booking field rather than repeat the slot question. The assistant must not say an appointment was booked or provide a booking ID until `book_appointment` returns `status: booked`.

Relative dates such as tomorrow, next week, and next Monday are resolved from current datetime context injected into the generic pipeline. Current time is only a reference for a date phrase the user actually supplied; it must never become an invented default date range. Named weekdays require a final weekday/date consistency check before lookup, and a correctly resolved relative date should not trigger an extra confirmation turn. Expected conversations can set `current_datetime` and `current_timezone` to make those tests deterministic.

Availability results keep `slot_id` as machine-facing data for booking, but user-facing slot summaries must be TTS-friendly. The tool provides `spoken_time_slot` values and response text formatted like `Sep 1st 2026 at 9:00 AM with Dr. Maya Johnson`. Every user-facing appointment date uses an abbreviated English month, ordinal day, and four-digit year without a comma between the day and year. The assistant should not read raw timestamps, markdown, bullets, parentheses, full month names, or slot IDs when presenting options. ISO dates remain machine-facing tool arguments.

Conversation state survives short greetings and continuation utterances. If the user says `hello` after slots were offered, the assistant should acknowledge the greeting and ask which prior option works; it should not restart intake or replay the full list unless the user asks to hear it again.

If the assistant needs to describe a date format to the user, it should say `year-month-date` or `month-date-year`. Do not use letter-code format strings in user-facing speech.

### Appointment database and container paths

The database support follows the NVA airline database fixture pattern:

- `references/appointment-making/database/schema.sql` defines the SQLite schema.
- `references/appointment-making/database/seed_data/*.jsonl` contains committed fixture inputs.
- `references/appointment-making/database/seed.py` materializes a rolling schedule from those fixtures.
- `references/appointment-making/database/db.py` applies schema and seeds only when the database is empty, incompatible, or stale.

The applier copies that database package into `src/examples/generic/ambient_healthcare_appointment_database/` and initializes the host-visible SQLite file at:

```text
$NVA_ROOT/data/appointment-making/appointment_schedule.sqlite
```

The NVA base `docker-compose.yml` is left untouched. The applier creates or merges `docker-compose.override.yml` so Docker Compose mounts:

```text
./data/appointment-making:/app/data/appointment-making
```

and sets:

```text
NVA_APPOINTMENT_DB_PATH=/app/data/appointment-making/appointment_schedule.sqlite
```

This keeps the runtime SQLite file readable and writable from both the host and the running generic assistant container.

## Patient Intake

Patient intake is a two-tool confirmation workflow:

- `review_patient_intake`: generate a natural spoken review without persisting anything.
- `record_patient_intake`: write the confirmed lightweight intake record to JSONL.

### Intake review and confirmation

The assistant opens with the deterministic phrase `Hello and welcome to the patient intake agent. I'm here to help you get checked in for your appointment and will be asking you a few questions in order to get ready for your visit with the doctor. First, could you please tell me your full name?` It then collects full name, date of birth, current symptoms or visit reason, current medications, and current or preferred pharmacy. Once all five are present, it calls `review_patient_intake` silently and waits for an explicit confirmation or correction. A correction triggers another complete review; only an unambiguous confirmation of the latest review permits `record_patient_intake` with `patient_confirmed: true`.

Reviews are tool-authored natural prose rather than field dumps. They contain no markdown, headings, colon-labeled fields, raw variable notation, or narration about tool calls. Spoken dates use the full month, ordinal day, and year without a comma, such as `October 1st 1964`. Symptoms and planned services are narrated as natural clauses. The saved response is intentionally short: it confirms the information was saved and gives the confirmation number without repeating the intake.

### Intake runtime reliability guards

The patient-intake-only pipeline support adds coordinated runtime backstops:

- one shared conversation state, derived from assistant-question/user-answer history, for the reminder, speech guard, and guarded review/save handlers
- canonical next-field questions that replace acknowledgements, partial summaries, and repeated known values while collection is incomplete
- handler gates that reject review/save calls when conversation history is incomplete, even if the model invents complete-looking tool arguments
- a one-time deterministic welcome latch that bypasses model paraphrasing without allowing the welcome to restart later
- a tagged tool-response frame for review/save speech, so the guard can distinguish deterministic tool output from model-authored text without relying on asynchronous frame order
- silent handling of empty LLM response boundaries, which are normal while Pipecat transitions into a function call and must never produce synthetic retry speech
- interruption handling that clears buffered text and forwards `InterruptionFrame`, plus a patient-intake-only opt-out from muting the user during the opening welcome so barge-in remains available

Pipecat can invoke the tool handler and push its direct response before `FunctionCallInProgressFrame` reaches the downstream guard. The guard therefore must not infer whether speech is tool-authored from that frame's arrival order. Direct review/save text carries an explicit marker and is emitted with `run_llm=False`; empty response boundaries remain silent. This prevents duplicate fallback speech, model preambles, and rewritten reviews.

The patient-intake LLM temperature is set to zero. The save tool also rejects calls unless `patient_confirmed` is exactly true. The assistant must not say the intake was saved or provide a confirmation number until `record_patient_intake` returns `status: saved`.

### Intake record storage

The implementation writes to:

```text
${NVA_HEALTHCARE_DEMO_DATA_DIR:-./data/patient-intake}/patient_intake_records.jsonl
```

The patient-intake overlay does not add database fixtures, SQLite support, or Compose override settings. It is intentionally a minimal append-only JSONL demo tool for proving field collection, tool timing, and record ID behavior.

If a deployment needs host-visible patient intake records, set `NVA_HEALTHCARE_DEMO_DATA_DIR` to a mounted container path and add the corresponding mount through the deployment configuration used for that environment.

## Validation Expectations

Each default overlay includes `expected-conversation.yaml` with `test_conversations`. Every test history must end with the latest user message. The expected next assistant message is semantic intent, not an exact string.

The live validation runner seeds each test conversation history, runs one assistant turn, judges message content semantically through the skill-running agent, and deterministically checks:

- whether a tool call happened
- which tool was called when multiple tools are available
- expected tool arguments
- expected tool result status
- expected result ID field when required

The appointment-making expected conversations cover welcome behavior; the universal appointment-type gate; partial scheduling preferences; immediate availability lookup; weekday integrity; concise results with dates like `Oct 6th 1994` and no comma; greeting continuity; contextual slot selection; each separate booking-field question; and final booking. The patient-intake expected conversations cover the exact welcome; the immediate date-of-birth question without replaying the welcome or acknowledging the known name; symptom collection without summarizing name or date of birth; collection of current medications before pharmacy; natural five-field review without persistence; explicit confirmation before saving; correction and re-review; question-shaped ASR pharmacy input; comma-free spoken dates; and the brief saved confirmation.

Live expected conversations validate model and tool semantics, but they do not exercise the Pipecat processor sequence. The copied patient-intake unit suite is a separate required gate for partial-summary replacement, early hallucinated tool-call rejection, one-time welcome behavior, empty tool-transition silence, exactly-once tagged tool output, buffered-speech cancellation, and interruptible welcome configuration.

The runner's deterministic pass is necessary but not sufficient. Review every actual assistant message against `expected_next_message_content`. A turn fails semantically if it narrates a future tool call without making it, restarts or replays state after a greeting, asks the user to reconfirm a correctly resolved date, repeats the slot-selection question after an unambiguous choice, claims booking without a successful tool result, invents an ID, combines booking-field questions, or substitutes appointment type for visit reason. Correct the prompt, tool contract, handler, or expected artifact as appropriate and rerun the full suite before deployment.
