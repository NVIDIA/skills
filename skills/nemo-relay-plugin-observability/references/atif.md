<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Export ATIF Trajectories

Use this reference when the user wants execution traces as ATIF documents rather than
live OTLP spans.

## Default Path

- Create an `AtifExporter` with session and agent metadata
- Register it before the instrumented work
- Run scoped tool and LLM activity
- Call `export()` or `export_json()`
- Clear between runs and deregister when done

## Embedded ATIF Semantics

- ATIF export translates NeMo Relay events into ATIF v1.7 trajectory data.
- LLM start events become `user` steps as follows:
  1. ATIF extracts the latest user message from the request annotation when
     possible. It does not export the annotation's complete history as the step
     message.
  2. Each owning agent scope starts fresh, and a `compaction` mark refreshes it.
  3. The first subsequent LLM start annotation retains complete history. Later
     starts retain system instructions, the latest user message, and every
     following assistant or tool message.
  4. When a request codec supplies an annotation, the event input uses the same
     projection. Provider execution remains unchanged.
- LLM end events become `agent` steps with response content, model metadata,
  token metrics, reasoning fields, and promoted `tool_calls` when the response
  uses a supported tool-call shape.
- Tool start events are skipped in the trajectory because tool calls are
  promoted from the preceding LLM end response.
- Tool end events become `system` observations. Observations are correlated to
  promoted tool calls by function name and source call ID when available.
- Point-in-time mark events and scope start/end events are structural and are
  not emitted as trajectory steps.
- Scope nesting becomes ancestry metadata on exported steps.
- Nested agent scopes become embedded `subagent_trajectories` with
  `subagent_trajectory_ref` observations in the parent trajectory.
- Event payloads become step input, step output, tool-call content, or
  observation content.
- The exporter preserves collected event order and uses lifecycle pairing to
  reconstruct the trajectory.
- Exporting does not clear the buffer. Use one exporter per run or call
  `clear()` between runs when concurrent agents share a process.
- Before using a trajectory in evaluation, confirm `schema_version` is
  `ATIF-v1.7`, agent metadata is correct, expected LLM/tool steps are present,
  tool observations follow tool calls, and sensitive fields are absent.

## Important Semantics

- ATIF exports the full event buffer collected so far.
- Consecutive tool observations can be merged into one system observation step.
- Trajectories reflect sanitized event payloads, not raw secrets that tool,
  LLM, mark, or scope event sanitizers removed before event emission.
- Response codecs can improve LLM end annotations, but they do not change the
  caller-visible LLM response.

## Checklist

- [ ] Session and agent metadata chosen
- [ ] Exporter registered before the relevant run
- [ ] Scope boundaries are correct so ancestry is meaningful
- [ ] Export timing is clear: whole buffer vs clear-between-runs
- [ ] LLM responses include `tool_calls` if ATIF tool-call entries are expected

## Related Skills

- `nemo-relay-plugin-observability`
- `nemo-relay-instrument-calls`
- `nemo-relay-debug-runtime-integration`
