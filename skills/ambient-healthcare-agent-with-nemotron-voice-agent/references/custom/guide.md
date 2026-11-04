# Custom Ambient Healthcare Agent Guide

Use this guide when the user chooses `customize your own use case` (the
`custom` workflow path).

## Design Questions

Ask these questions before editing files:

1. Which runtime services should this use? First run `scripts/inspect_nva_generic_defaults.py --nva-root "$NVA_ROOT"` from this skill and report the resolved LLM, ASR, and TTS keys, display names, models or servers, endpoints, and catalog sources. Default to public NVIDIA AI Endpoints for all three. Ask whether to keep that default, use NVA-managed local NIMs, point to existing NIM endpoints, or use a mixed layout. Do not hard-code service names from a specific NVA release; follow `references/deployment-modes.md` and the target repo's deploy/configure skills.
2. What ambient healthcare workflow should the agent accomplish for the user?
3. What information must be collected before the first tool call is allowed?
4. What tool calls are needed?
5. For each tool, what are the required input fields, optional input fields, and returned fields?
6. What should the tool write or read: JSONL, CSV, SQLite, an HTTP API, or no persistence?
7. What healthcare information is sensitive, regulated, or operationally risky, and what demo-only storage warnings should be included?
8. What representative conversation context should prove that the assistant asks for missing information, then calls the tool only after enough information is known?

After the user answers, propose concrete expected-conversation histories for review and disclose that approving them authorizes sending those histories to the configured LLM endpoint for live validation. Do not implement custom artifacts until the user approves or revises those histories. Once approved, do not ask for separate live-test approval later.

Do not make a public inference call until the user has seen the runtime choice. If the user does not request an override, use public NVIDIA AI Endpoints for the LLM, ASR, and TTS through the generic cloud catalog. Preserve any explicit local, existing-endpoint, or mixed choice throughout validation and deployment.

## Required Custom Artifacts

Create these files, then pass them to `scripts/apply_generic_agent_template.py`:

- custom prompt YAML with one prompt entry and `tools_available`
- custom tools YAML with OpenAI-style function tool schemas
- custom Python tool implementation defining `TOOL_RESULT_FUNCTIONS`
- custom expected conversation YAML

Use the generic templates in this directory as starting points:

- `prompt-template.yaml`
- `tools-template.yaml`
- `tool-implementation-template.py`
- `expected-conversation-template.yaml`

Replace every placeholder tool name, field name, response phrase, and expected
turn with values from the user's custom ambient healthcare scenario. Do not
leave the template's generic field names in a finished customization.

Preserve the custom prompt template's spoken-date rule in every finished
customization. Any calendar date shown or spoken to the user must use an
abbreviated English month, ordinal day, and four-digit year with no comma, such
as `Oct 6th 1994`. Keep ISO dates such as `1994-10-06` only in tool arguments or
internal data. Tool-authored `response_text` must follow the same user-facing
rule. If the custom workflow includes a date, add at least one expected
conversation that exercises this formatting requirement.

## Tool Implementation Contract

The copied Python file must define:

```python
TOOL_RESULT_FUNCTIONS = {
    "tool_name_from_tools_yaml": callable_result_function,
}
```

Each callable must accept `Mapping[str, Any] | None` and return a JSON-serializable dict.

Use defensive validation. If required fields are missing, return:

```python
{
    "status": "needs_more_information",
    "missing_fields": ["field_name"],
    "response_text": "Please share the missing field.",
}
```

When enough information is present, write or call the requested data IO and return:

```python
{
    "status": "saved",
    "record_id": "CUSTOM-123",
    "response_text": "Record CUSTOM-123 was saved.",
}
```

## Expected Conversation Format

```yaml
scenario: custom
prompt_key: custom_healthcare_agent
tool_name: custom_healthcare_tool
expected_final_status: saved
expected_result_id_field: custom_healthcare_request_id
test_conversations:
  - name: ask_for_remaining_required_detail
    history:
      - role: user
        content: "Initial request for the custom workflow."
      - role: assistant
        content: "Assistant asks for required details."
      - role: user
        content: "User provides some, but not all, required details."
    expected_next_message_content: >-
      Ask for the remaining required detail or details. Do not say the request
      was saved, submitted, recorded, or assigned an ID.
    expected_tool_call: false

  - name: save_completed_custom_request
    history:
      - role: user
        content: "Initial request for the custom workflow."
      - role: assistant
        content: "Assistant asks for required details."
      - role: user
        content: "User provides the first required details."
      - role: assistant
        content: "Assistant asks for the remaining required detail."
      - role: user
        content: "User provides the final missing required detail."
    expected_next_message_content: >-
      After the tool returns saved, tell the user the request was saved and
      include the returned scenario-specific ID.
    expected_tool_call: true
    expected_tool_arguments:
      field_name: "expected value"
    expected_tool_argument_substrings:
      free_text_field:
        - "important phrase"
handoff_note: "State what this test proves."
```

Use `expected_result_id_field` when the custom healthcare tool returns an ID
under a scenario-specific key such as `custom_healthcare_request_id`, `case_id`,
or `message_request_id`.

Each `history` must end with the latest user message. `expected_next_message_content`
is a semantic description, not exact wording. The Python validation script prints
the actual assistant message and this expected content for agent review; do not
turn it into exact string or substring matching.

When a custom workflow includes dates, the semantic expectation must explicitly
reject full month names, missing years, ISO dates in speech, and commas between
the ordinal day and year. For example, accept `Oct 6th 1994` and reject
`Oct 6th, 1994`.
