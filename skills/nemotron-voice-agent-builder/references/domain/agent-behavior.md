# Agent Behavior

Read for every build. Convert the approved use case into a voice-safe system instruction
before generating framework code.

## Derive the Behavior

Use the user's description to define:

- identity and role
- who it serves and what it helps with
- locked response language and speaking style
- tasks it may complete
- boundaries, escalation, and when to ask for clarification
- tools it may call, if any

Do not invent business policies, private data, tool access, or unsupported capabilities.
Ask one focused follow-up only when a missing boundary would make the agent unsafe or
materially change its behavior.

## Voice-Safe Output

The system instruction must require:

- brief, conversational answers suitable for speech
- plain text with no markdown, tables, code blocks, or emojis
- no raw URLs, UUIDs, database keys, JSON, stack traces, or internal tool names
- natural pronunciation of numbers, dates, abbreviations, and units
- a short clarification instead of guessing when speech is ambiguous
- the locked response language unless the approved route allows switching

Keep spoken responses focused on the next useful action. Put machine-readable metadata in
application state or logs, never in TTS text.

Use the framework's current documented text filter when one exists. Do not invent a
filter API. Prompt constraints remain required because filtering is only a fallback.

## Persona and Boundaries

Give the agent a clear tone without role-playing beyond the approved use case. It must not
claim actions succeeded until the corresponding tool or service confirms them.

For regulated or sensitive domains, state the approved limitation and escalation path.
Do not add legal, medical, financial, or safety claims that the user did not provide.

## Tools

Add tools only when the user explicitly requests and approves the tool-backed action.
Query the selected framework documentation MCP for current tool and function-calling
APIs.

For every tool:

1. define its purpose and required inputs
2. validate inputs before calling
3. handle success, empty results, and failure
4. speak a user-friendly result, not the raw payload
5. keep identifiers and session state out of spoken output

Do not generate fake business data or placeholder tool results unless the user explicitly
asks for a demo. Label demo data clearly.

## Domain Vocabulary

This file owns meaning and response behavior. `domain/speech-customization.md` separately
owns how specialized terms are recognized and pronounced.

Add approved domain terms to the system instruction when they affect meaning, but do not
treat prompt vocabulary as ASR word boosting.

## Generated Project

Store the system instruction in agent code or a checked-in prompt file. Do not put it in
`.env`. Document where to edit behavior in `README.md`.

## Verify

Test at least:

- a normal in-scope request
- an ambiguous request
- an out-of-scope request
- one tool success and failure when tools exist
- a response containing an identifier, URL, or structured payload upstream

The spoken result must remain concise, plain, and free of internal data.
