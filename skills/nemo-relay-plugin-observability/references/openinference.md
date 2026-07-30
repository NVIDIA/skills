<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Export OpenInference Traces

Use this reference when the destination expects OpenInference semantic conventions,
for example Arize Phoenix or another OpenInference-aware OTLP backend.

## Default Path

- Confirm the installed NeMo Relay version before selecting configuration.
- For 0.6, build `OpenInferenceConfig` and construct
  `OpenInferenceSubscriber`.
- For 0.7, build `OpenTelemetryConfig` with type `openinference` and construct
  `OpenTelemetrySubscriber`.
- Set transport, service metadata, and headers
- Construct and register the version-appropriate subscriber
- Run instrumented scoped work
- Deregister, flush, and shut down when done

## Embedded OpenInference Semantics

- OpenInference export is for OTLP backends that understand model-centric
  OpenInference semantic conventions.
- Set `transport`, `endpoint`, and `service_name`, then add a namespace, version,
  instrumentation scope, headers, resource attributes, or timeout when needed.
- NeMo Relay projects lifecycle payload fields to typed OTLP attributes with
  dotted names. Non-LLM start metadata and all end metadata use
  `openinference.metadata`, while mark metadata uses
  `nemo_relay.mark.metadata`.
- NeMo Relay emits a top-level object or array field as a JSON string, omits a
  top-level `null` field, and no longer emits the old aggregate `*_json` payload
  attributes.
- In 0.6, OpenInference is a standalone exporter and `attribute_mappings` can
  alias projected attributes without changing their OTLP type.
- In 0.7, OpenInference is a typed endpoint in the unified OpenTelemetry
  exporter rather than a standalone section or class. It supports
  `mark_projection`, `mark_exclude_names`, and `attribute_mappings`;
  `semantic_selector` and `capture_content` are unsupported.
- For the 0.7 Observability plugin path, inject secret-bearing headers through
  `opentelemetry.endpoints[].header_env`; `header_env` is not a direct
  `OpenTelemetrySubscriber` setting. Never place resolved credential values in
  source code, committed configuration, command-line arguments, prompts,
  examples, or diagnostics.
- Start with `http_binary` transport and an OTLP/HTTP traces endpoint. In 0.6,
  use `grpc` only with an active Tokio runtime. In 0.7, the subscriber owns the
  runtime needed by `grpc`, including for synchronous direct construction.
- Scope, tool, and LLM start inputs become `input.value`.
- Scope, tool, and LLM end outputs become `output.value`.
- LLM annotations follow the freshness rules:
  - Each owning agent scope starts fresh, and a `compaction` mark refreshes it.
  - The annotated input for the first subsequent LLM start retains complete
    history. Later starts retain system instructions, the latest user message,
    and every following assistant or tool message.
  - When a request codec supplies an annotation, the event's provider-shaped
    input uses the same projection. Provider execution remains unchanged.
- LLM usage metadata maps token counters when provider responses include usage.
- Use explicit config fields for endpoint, headers, resource attributes, and
  service identity in application code.
- Validate 0.6 export by checking construction logs, collector traffic, and
  spans from the same `root_uuid`. For 0.7, check matching
  `nemo_relay.uuid` / `nemo_relay.parent_uuid` lineage; coding-agent sessions
  can use `nemo_relay.session.instance_id`, and endpoints receiving the same
  lifecycle events derive the same native trace and span IDs.

## Important Semantics

- Spans include OpenInference semantic attributes
- LLM spans derive `input.value` from request content, not request headers
- Scope types map to OpenInference span kinds
- Orphan mark events still export as zero-duration spans

## Troubleshooting Focus

- No spans in the OpenInference-aware backend
- Expected semantic attributes missing
- Wrong scope types or no active scope
- Wrong OTLP transport for the chosen binding or target

## Related Skills

- `nemo-relay-plugin-observability`
- `nemo-relay-instrument-typed-wrappers`
