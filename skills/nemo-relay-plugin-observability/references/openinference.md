<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Export OpenInference Traces

Use this reference when the destination expects OpenInference semantic conventions,
for example Arize Phoenix or another OpenInference-aware OTLP backend.

## Default Path

- Build the binding-specific `OpenInferenceConfig`
- Set endpoint, transport, service metadata, and headers
- Construct and register the subscriber
- Run instrumented scoped work
- Deregister, flush, and shut down when done

## Embedded OpenInference Semantics

- OpenInference export is for OTLP backends that understand model-centric
  OpenInference semantic conventions.
- Set `transport`, `endpoint`, and `service_name`, then add a namespace, version,
  instrumentation scope, headers, resource attributes, timeout, or
  `attribute_mappings` when needed.
- NeMo Relay projects lifecycle payload fields to typed OTLP attributes with
  dotted names. Non-LLM start metadata and all end metadata use
  `openinference.metadata`, while mark metadata uses
  `nemo_relay.mark.metadata`.
- NeMo Relay emits a top-level object or array field as a JSON string, omits a
  top-level `null` field, and no longer emits the old aggregate `*_json` payload
  attributes.
- Use `attribute_mappings` to copy a fully qualified projected attribute to a
  backend-specific alias without changing its OTLP type.
- Start with `http_binary` transport and an OTLP/HTTP traces endpoint. Use
  `grpc` only when a Tokio runtime is active.
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
- Validate export by checking construction logs, collector traffic, and spans
  from the same `root_uuid` in the tracing backend.

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
