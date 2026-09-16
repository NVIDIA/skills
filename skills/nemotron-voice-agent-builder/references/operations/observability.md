# Observability

Apply to every generated project. Logs must show where one spoken turn failed and where
latency was spent without exposing secrets or private audio.

## Use Documented Hooks

Query the selected framework documentation MCP for its current logging, observer, metrics,
and error hooks. Use those APIs rather than intercepting internal frames or inventing
framework methods.

## Structured Events

Emit machine-readable key-value or JSON logs to stdout. Every event should include:

- timestamp, level, and stable event name
- pipeline (`cascaded` or `omni`)
- transport label
- stage (`asr`, `omni`, `llm`, `tts`, or transport)
- per-turn correlation id when a turn exists
- endpoint label and model id for model requests
- duration in milliseconds for completed timed events

An endpoint label identifies `cloud`, `llm-local`, `asr-local`, `tts-local`, or
`omni-local`. Do not log credentials, signed URLs, authorization headers, raw audio, or
endpoint query strings. Transcripts and reply text are off by default. Enable them only
with explicit user approval.

## Startup

Log:

1. application and framework version
2. approved pipeline, transport, models, and local/cloud placement
3. each service readiness check and selected model id
4. transport initialization and client connection
5. agent-ready only after every required dependency is ready

Keep Compose service logs on stdout/stderr. Put
`docker compose logs --timestamps <service>` and the generated service names in the
README.

## Turn Timings

Capture monotonic timestamps and emit durations for:

| Pipeline point | Event |
| --- | --- |
| user speech | start and completed turn |
| streaming ASR | first partial and final transcript |
| LLM / Omni | request start, first token, response complete |
| TTS | request start, first audio, audio complete |
| output transport | first audio sent and turn complete |

At minimum report end-of-user-turn to first assistant audio and full turn duration.
Cascaded runs must separate ASR, LLM, and TTS. Omni runs replace ASR + LLM with one Omni
stage but still report request-to-first-token and request completion.

Use monotonic time for durations and wall-clock time only for log timestamps. Do not infer
server compute time when only client-side timing is available. Label it as observed
latency.

## Exceptions

Log the full traceback for unexpected exceptions, including background tasks and startup
failures. Add stage, endpoint label, model id, and turn id as context, then preserve the
original exception.

Expected transient failures may retry only when the framework or provider documents that
behavior. Log each retry and the terminal failure. Never catch an exception only to print
its message and continue with a silent broken stage.

## Handover

The README must state:

- how to enable the project's documented debug mode
- how to view agent and Compose service logs
- which latency events prove each pipeline stage ran
- that transcript logging can contain sensitive data
- when to open `operations/troubleshoot.md`

Before handover, trace one spoken exchange under one turn id and confirm every expected
stage event appears in order.
