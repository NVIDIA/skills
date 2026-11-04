# Generic NVA Healthcare Customization Reference

Use this file with the public deployment and configuration documentation for the checked-out NVA release.

## Integration Points

The Generic Cascaded example lives at `nemotron-voice-agent/src/examples/generic/`.

- `prompts.yaml`: prompt presets. A prompt entry can declare `tools_available`.
- `tools.yaml`: OpenAI-style function tool schemas.
- `tools.py`: builds a Pipecat `ToolsSchema` from `tools.yaml`.
- `tool_handlers.py`: maps tool names to async Pipecat function handlers.
- `pipeline.py`: creates `NvidiaLLMService`, registers function handlers, and builds `LLMContext(..., tool_choice="auto")`.
- For applied ambient healthcare overlays, `pipeline.py` is patched to append current datetime context to the prompt at session start. Export `NVA_HEALTHCARE_AGENT_TIMEZONE` when relative date phrases should resolve in a specific timezone; it defaults to `UTC`.

The multi-turn memory is the live Pipecat `LLMContext` for the browser session. Refreshing the UI starts a new session and loses that memory.

## Modification Pattern

1. Add one prompt entry to `prompts.yaml`.
2. Add one or more tool entries to `tools.yaml`.
3. Copy the scenario implementation into `src/examples/generic/ambient_agent_tools.py`.
4. Add a marked ambient-agent handler block to `tool_handlers.py` that imports `TOOL_RESULT_FUNCTIONS` and registers every function in it.
5. Add a marked current-datetime context block to `pipeline.py`.
6. Restart or rebuild through Docker Compose according to the public NVA documentation.

Use `scripts/apply_generic_agent_template.py` to perform those edits deterministically.

## Tool Timing Rule

For multi-turn custom healthcare flows, the intended behavior is not "call a tool on every turn." The model should ask normal follow-up questions across multiple turns and call the tool only after all required fields are known from the conversation.

Use `tool_choice="auto"` and make the prompt and schema explicit:

- "Do not call the tool while any required field is missing."
- "Once all required fields are known, call the tool exactly once."
- Put required fields in the tool schema's `required` list.

## User-Facing Date Rule

For appointment-making and custom healthcare workflows, render every
user-facing calendar date with an abbreviated English month, ordinal day, and
four-digit year, with no comma between the day and year: `Oct 6th 1994`.
Machine-facing tool arguments may continue to use ISO values such as
`1994-10-06`. Apply the spoken format to assistant messages and any
tool-authored `response_text` that reaches the user.

## Data IO Rule

Bundled demo tools write JSONL files under:

```text
./data/custom-healthcare/
```

Override this with a scenario-specific environment variable when you need another writable location.

These files are local demo output only. Do not treat them as production storage for PHI, sensitive, or regulated healthcare data.
