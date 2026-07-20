---
name: nemo-relay-plugin-observability
description: Use this skill when choosing or configuring NeMo Relay observability through the built-in plugin, subscribers, or exporters, including raw ATOF events, ATIF trajectories, OpenTelemetry traces, OpenInference export, or custom event handling.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation and Affiliates
---

# Configure Observability Plugins

Start with one exporter managed by the built-in Observability plugin. This is
the default for reusable process configuration and the best first plugin for
most users because it makes Relay's captured activity visible.
Choose one proof output before layering additional telemetry destinations.

Use manual subscriber or exporter APIs only when a test, script, or application
needs direct control over registration names, collection windows, or flush
timing. Both paths consume the same canonical event stream.

## Choose The Output

Select the output that best matches the user's immediate inspection target:

- **Console or custom event handling**
  Use a manual subscriber for short-lived in-process inspection.
- **Raw canonical lifecycle events**
  Use ATOF JSONL; read `references/atof.md`.
- **Portable execution trajectories**
  Use ATIF; read `references/atif.md`.
- **General OTLP tracing**
  Use OpenTelemetry; read `references/opentelemetry.md`.
- **OpenInference-aware backends**
  Use OpenInference; read `references/openinference.md`.

Choose one output first and verify it before adding another. ATOF is the
default local proof because it preserves the raw event stream with the least
translation. Use synthetic, non-sensitive payloads for the first proof. Add and
verify sanitization before exporters receive production payloads, and never
display complete event records while validating an exporter.

## Embedded Event And Subscriber Model

Use this model when explaining how capture and export relate:

- NeMo Relay emits one canonical event stream from scopes, marks, managed tool
  calls, managed LLM calls, middleware, and manual lifecycle APIs.
- Subscribers consume events without defining the event model. Multiple
  subscribers can observe the same stream for logging, export, analytics, or
  diagnostics.
- Global subscribers remain active process-wide until removed.
- Scope-local subscribers are owned by one active scope and disappear when that
  scope closes.
- Plugin-installed subscribers are reusable, configuration-driven runtime
  components.
- Exporter-oriented subscribers preserve raw ATOF or translate the event stream
  into ATIF, OpenTelemetry, or OpenInference output.
- Event payloads reflect sanitized post-guardrail input and output when calls
  use managed helpers or manual lifecycle params provide those fields.
- LLM annotations follow the freshness rules:
  - Each owning agent scope starts fresh, and a `compaction` mark refreshes it.
  - The first subsequent LLM start retains complete annotation history. Later
    starts retain system instructions, the latest user message, and every
    following assistant or tool message.
  - When a request codec supplies an annotation, Relay applies the same
    event-only projection to provider-shaped event input without changing
    provider execution.
- Event fields include semantic input/output through the ATOF `data` field,
  typed profile data such as `model_name` and `tool_call_id`, and codec-provided
  annotated LLM request/response data for in-process subscribers and exporters.
- First-class skill tools and the requests to read a complete `SKILL.md`
  automatically emit `skill.load` marks under the tool span. The payload
  contains only `skill_name`; metadata records the load source and tool name.
  Partial reads do not count, and ambiguous slash-command expansions use the
  separate `skill.load.inferred` name. The eager mark remains present if tool
  execution later fails.

## Shared Lifecycle

1. Create the exporter or subscriber.
2. Register it with a unique name before the relevant scoped work.
3. Run NeMo Relay-instrumented work inside scopes.
4. Flush if deterministic delivery is needed and the binding supports it.
5. Deregister it, then shut it down when the process or subsystem is done.

## Binding Names

Use the names exported by the selected language binding:

- Python: `nemo_relay.subscribers.register(...)`,
  `AtofExporter`, `AtifExporter`, `OpenTelemetrySubscriber`, and
  `OpenInferenceSubscriber`
- Node.js: root exports `registerSubscriber(...)`, `AtofExporter`,
  `AtifExporter`, `OpenTelemetrySubscriber`, and `OpenInferenceSubscriber`
- Rust: `nemo_relay::api::subscriber` and `nemo_relay::observability::*`
- Go: source-first wrappers expose equivalent register, exporter, and subscriber
  lifecycle methods

## Load A Reference When

Load only the reference required by the selected output:

- Load `references/atof.md` for raw JSONL events used in local debugging or
  offline inspection.
- Load `references/atif.md` for ATIF trajectories.
- Load `references/opentelemetry.md` for OTLP/OpenTelemetry traces.
- Load `references/openinference.md` for OpenInference semantic traces.

## Use Another Skill When

Choose another skill when the task belongs to an adjacent workflow:

- Use `nemo-relay-plugin-build` to package subscriber-based export behavior as
  a reusable plugin.
- Use `nemo-relay-get-started` or `nemo-relay-instrument-calls` when no scope,
  tool call, or LLM call has been instrumented.
- Use `nemo-relay-debug-runtime-integration` to diagnose missing telemetry.

## Related Skills

Use these skills for adjacent workflows:

- Instrument application calls with `nemo-relay-instrument-calls`.
- Add typed wrappers with `nemo-relay-instrument-typed-wrappers`.
- Package reusable behavior with `nemo-relay-plugin-build`.
- Diagnose missing events with `nemo-relay-debug-runtime-integration`.
