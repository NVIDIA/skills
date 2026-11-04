# Testing Protocol

Run these checks after applying a scenario to `$NVA_ROOT/src/examples/generic/`.

## Generic Compatibility Preflight

Run this after verifying the user-confirmed NVA repo and before applying a
scenario:

```bash
python3 "$SKILL_DIR/scripts/inspect_nva_generic_defaults.py" \
  --nva-root "$NVA_ROOT" \
  --check-compatibility
```

Use this output as the source of truth for the current generic default LLM,
ASR, TTS, prompt, service catalog sources, and available keys for all three
service kinds. If the preflight fails, stop and report the incompatible file or
registry shape instead of applying overlays.

## Static Checks

```bash
python3 - <<'PY'
import os
from pathlib import Path
import yaml
for path in [
    Path(os.environ["NVA_ROOT"]) / "src/examples/generic/prompts.yaml",
    Path(os.environ["NVA_ROOT"]) / "src/examples/generic/tools.yaml",
]:
    yaml.safe_load(path.read_text(encoding="utf-8"))
    print(f"{path}: ok")
PY

python3 -m py_compile \
  "$NVA_ROOT/src/examples/generic/ambient_agent_tools.py" \
  "$NVA_ROOT/src/examples/generic/tool_handlers.py" \
  "$NVA_ROOT/src/examples/shared/pipeline_utils.py" \
  "$SKILL_DIR/scripts/nva_generic_defaults.py" \
  "$SKILL_DIR/scripts/inspect_nva_generic_defaults.py" \
  "$SKILL_DIR/scripts/apply_generic_agent_template.py" \
  "$SKILL_DIR/scripts/run_expected_conversation.py" \
  "$SKILL_DIR/scripts/verify_docker_compose_access.py"

cd "$NVA_ROOT"
docker compose config --quiet
```

For patient intake, run its copied unit suite from the NVA root:

```bash
python3 -m py_compile src/examples/generic/patient_intake_speech_guard.py
uv run --frozen pytest -q tests/unit/test_ambient_agent_tools.py
```

These tests verify the review-before-save boundary, explicit confirmation,
the required current-medications field, natural symptom and visit-reason
narration, comma-free spoken dates, markdown and field-dump sanitization,
conversation-history completeness gates, canonical next-field speech,
one-time welcome behavior, silent empty tool transitions, exactly-once tagged
tool responses, interruption forwarding, and an interruptible welcome.

Do not substitute the live expected-conversation runner for this unit suite.
The runner validates model/tool semantics but does not reproduce Pipecat's
asynchronous response-boundary and function-call frame order.

For appointment-making, run its copied unit suite from the NVA root as well:

```bash
uv run --frozen pytest -q tests/unit/test_ambient_agent_tools.py
```

These tests verify that appointment slots use an abbreviated English month,
ordinal day, four-digit year, and no comma, such as `Oct 6th 1994`, while raw
ISO dates remain internal.

For appointment-making, inspect the applied prompt, schema, and handler to
confirm that availability lookup requires appointment type, a user-provided
date or date range, and an explicit `morning` or `afternoon` time window.
Appointment type alone must not trigger lookup. Patient identity, date of
birth, and visit reason must remain post-selection booking fields.

## Appointment Tool-Boundary Checks

Exercise the handler independently of the LLM. Use a temporary SQLite path so
the check does not alter the project demo database:

```bash
task_test_dir=$(mktemp -d)
NVA_APPOINTMENT_DB_PATH="$task_test_dir/appointment_schedule.sqlite" \
NVA_APPOINTMENT_DB_START_DATE=2026-08-17 \
PYTHONPATH="$NVA_ROOT/src" \
python3 - <<'PY'
from examples.generic.ambient_agent_tools import find_available_appointments_result

base = {
    "appointment_type": "Sick visits",
    "start_date": "2026-09-01",
    "end_date": "2026-09-03",
}

missing = find_available_appointments_result(base)
assert missing["status"] == "needs_more_information", missing
assert missing["missing_fields"] == ["time_window"], missing

invalid = find_available_appointments_result({**base, "time_window": "any time"})
assert invalid["status"] == "invalid_time_window", invalid

valid = find_available_appointments_result({**base, "time_window": "afternoon"})
assert valid["status"] in {"found", "none_found"}, valid
assert len(valid["available_appointments"]) <= 3, valid
assert all(
    12 <= int(slot["appointment_datetime"][11:13]) <= 16
    for slot in valid["available_appointments"]
), valid
for slot in valid["available_appointments"]:
    spoken = slot["spoken_time_slot"]
    assert spoken.startswith("Sep "), spoken
    assert " 2026 at " in spoken, spoken
    assert ", 2026" not in spoken, spoken
print("appointment tool boundary checks: PASS")
PY
```

This proves that prompt mistakes cannot turn a missing preference into an
unbounded search, that `any time` is rejected, and that spoken result sets stay
small.

## Post-Overlay Runtime Credential and Endpoint Gate

Use `references/deployment-modes.md` after applying the overlay and passing
static checks. Check the selected mode's credentials before starting the
customized app, then verify it and its services:

- Public NVIDIA AI Endpoints: require authenticated health checks from the
  customized Generic deployment and each selected service.
- NVA-managed local NIMs: use the public NVA deployment workflow's NGC, image-access,
  Hugging Face, and hardware preflights; do not make an unrelated public
  inference request.
- Existing NIM endpoints: verify the configured LLM, ASR, and TTS endpoints and
  credentials without printing secrets.

Missing credentials or unavailable endpoints do not prevent overlay and static
validation. Do not start the customized app without the selected credentials,
or run live histories or claim a ready app when a credential or endpoint gate fails.

## Setup Docker Gate

Run this before choosing or implementing a scenario:

```bash
python3 "$SKILL_DIR/scripts/verify_docker_compose_access.py"
```

The script verifies Docker daemon access, Docker Compose availability, and that
Docker Compose can bring up a disposable container. If it fails because of
Docker access, Docker daemon access, Docker network access, image pull access,
or filesystem permission, stop and report that the coding agent needs those
permissions before customization. Do not switch to an unrelated run mode or try
to bypass Docker access restrictions.

## Live LLM Expected Conversation Test

```bash
python3 "$SKILL_DIR/scripts/run_expected_conversation.py" \
  --nva-root "$NVA_ROOT" \
  --live-test-approved
```

Use `--expected-file <path>` to validate an expected-conversation YAML that was
not copied into `src/examples/generic/ambient_agent_expected_conversation.yaml`.

For the appointment-making and patient-intake examples, the user's selection
after the preset live-test notice authorizes this command only when the notice
names the selected LLM destination and discloses the bundled fictional names,
dates of birth, and healthcare-style details. Run it without a second approval
question unless the destination or payload changes. For a custom use case,
approval of the expected-conversation artifact authorizes this command only
when the user was told that the approved histories will be sent to the
configured LLM endpoint. If a custom-workflow user declines that use, run
static checks and local/mocked validation instead.

The script uses the selected LLM configured by the target NVA
`generic-assistant` example. It resolves that default from the same registry
and service catalogs reported by `inspect_nva_generic_defaults.py`; do not
assume a model name from a particular NVA release. By default, use a public
NVIDIA AI Endpoint entry from the generic cloud service catalog. Pass
`--llm-key <catalog-key>` to use another LLM catalog entry, or `--model` and
`--base-url` for an explicit OpenAI-compatible endpoint override. For an
NVA-managed local LLM, deploy it first through the public NVA deployment workflow
and use the documented host-reachable URL rather than its Compose DNS name.

The runner appends the same current datetime context used by the applied NVA
generic pipeline patch. Expected files may define `current_datetime` as an ISO
datetime and `current_timezone` as an IANA timezone to make relative-date tests
deterministic. Without those fields, the runner uses the actual current time
and `NVA_HEALTHCARE_AGENT_TIMEZONE`, defaulting to `UTC`.

Expected conversations use `test_conversations`. Each test conversation seeds
the model with a `history` list whose last message is the latest user message,
then runs the assistant's next turn. The script deterministically checks whether
a tool call happened when `expected_tool_call` says it should.

The script validates every field listed in `expected_tool_arguments` against the
tool call's arguments. For free-text tool arguments that the model may
paraphrase, use `expected_tool_argument_substrings` with one string or a list of
required substrings.

The script prints `EXPECTED_NEXT_MESSAGE_CONTENT` and the actual `ASSISTANT`
message. The skill-running agent must semantically judge whether the assistant
message aligns with the expected content. Do not use exact string matching or
substring checks for assistant message content.

A final deterministic `PASS` does not certify semantic behavior. Review every
printed assistant message before accepting the run. In appointment-making,
explicitly fail the run if any response:

- treats a known appointment type as permission to search before the user
  supplies both date/date range and morning/afternoon
- says it will check availability without making the lookup tool call in that
  same turn
- invents a date range or uses `any time` when the user did not supply it
- produces an ISO date that does not match a named weekday, or adds an
  unnecessary confirmation turn after resolving it correctly
- presents more than three options, uses bullets, or repeats the full list
  after a greeting
- formats a user-facing date with a full month name, omits its four-digit year,
  uses an ISO date in speech, or places a comma between its ordinal day and year;
  the required form is `Oct 6th 1994`
- restarts intake after a greeting instead of preserving the current state
- fails to recognize a uniquely identified offered slot, or asks which option
  the user wants again
- asks for multiple booking fields in one turn or changes their order
- treats appointment type as visit reason
- claims a booking or invents an ID without `book_appointment` returning
  `status: booked`

In patient intake, explicitly fail the run if any response:

- paraphrases or adds text to the configured welcome
- repeats or restarts the welcome after the patient has begun answering
- acknowledges, summarizes, or repeats known patient values instead of asking only the earliest missing-field question
- begins a review after only a subset of the five fields is present
- saves before the user explicitly confirms the latest complete review
- fails to review all five fields again after a correction
- writes a review as markdown, a heading, bullets, colon-labeled fields, or raw key-value notation
- places a comma between the ordinal day and year in a spoken date
- narrates a tool call or adds a model-authored preamble to a review or saved confirmation
- repeats the intake fields after saving instead of giving only the short confirmation and confirmation number

Separately fail the local patient-intake runtime checks if an empty LLM
response boundary produces speech, a direct review/save response is spoken
more than once, unmarked model text reaches TTS after collection is complete,
an early hallucinated tool call bypasses conversation-history completeness,
or an `InterruptionFrame` is swallowed. The opening welcome must leave the
microphone unmuted for barge-in only in the patient-intake preset.

For a custom workflow, explicitly fail any response or tool-authored
`response_text` that renders a calendar date in a different user-facing format.
The required form is an abbreviated English month, ordinal day, and four-digit
year without a comma, such as `Oct 6th 1994`. ISO dates remain valid in tool
arguments and internal data.

After correcting any deterministic or semantic failure, rerun the entire
expected-conversation suite. A passing failed case alone is not sufficient
because prompt changes can regress an earlier state transition.

For appointment availability, treat any request for identity, date of birth,
insurance, contact information, or visit reason before lookup as a semantic and
tool-timing failure. Also fail any lookup made before the user supplies the
date/date range and morning-or-afternoon preference. Tighten the prompt, tool
schema and description, and handler boundary together when needed, reapply the
overlay, and rerun the full suite. Do not deploy with only the no-tool or tool
timing cases passing.

For custom expected-conversation files, set `expected_result_id_field` when the
tool returns a scenario-specific ID key. The test script also falls back to the
first non-empty `*_id` field.

## Service Check

Use `references/deployment-modes.md` and the target NVA repo's `deploy` and
public NVA deployment/configuration documentation for exact commands. Select exactly one complete
recipe for the chosen public, NVA-managed local, existing-endpoint, or mixed
layout. Do not hard-code the cloud recipe when the user selected another mode.

If Docker Compose fails because of Docker daemon access, Docker network access,
image pull access, or filesystem permission, stop and report that the coding
agent needs permission to stand up Docker Compose services. Do not switch to an
unrelated run mode or try to bypass Docker access restrictions.

Resolve the UI scheme, host, and port from the selected NVA recipe. Its common
local default is:

```text
https://localhost:7860/
```
