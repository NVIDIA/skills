<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Export OpenTelemetry Traces

Use this reference when the destination is an OTLP/OpenTelemetry backend such as an
OpenTelemetry Collector, Jaeger, Tempo, or Honeycomb.

## Default Path

- Confirm the installed NeMo Relay version before selecting configuration.
- For 0.6, build `OpenTelemetryConfig` with an endpoint. OpenInference uses its
  separate exporter described in `openinference.md`.
- For 0.7, build `OpenTelemetryConfig` with a required projection type (`full`,
  `gen_ai`, or `openinference`) and endpoint.
- Set service identity; add authentication only when the collector requires it
- Construct the subscriber
- Register it before running scoped work
- Deregister, flush, and shut down when the process or subsystem is done

## Embedded OpenTelemetry Semantics

- In 0.6, the Observability plugin uses configuration version 2,
  `OpenTelemetrySubscriber` is separate from `OpenInferenceSubscriber`,
  `attribute_mappings` can alias projected attributes, and native gRPC requires
  an active Tokio runtime.
- In 0.7, the plugin uses configuration version 3. Each configured endpoint
  creates an `OpenTelemetrySubscriber` for exactly one typed projection:
  `full`, `gen_ai`, or `openinference`.
- OpenTelemetry export maps NeMo Relay runtime events into OTLP traces for
  tracing backends and collectors.
- Set `transport`, `endpoint`, and `service_name`, then add a namespace, version,
  instrumentation scope, headers, resource attributes, or timeout when needed.
- NeMo Relay projects start, end, handle, and mark payload fields to typed OTLP
  attributes with dotted names. Start and end metadata use
  `nemo_relay.start.metadata` and `nemo_relay.end.metadata`.
- NeMo Relay emits a top-level object or array field as a JSON string, omits a
  top-level `null` field, and no longer emits the old aggregate `*_json` payload
  attributes.
- In 0.7, `mark_projection`, `mark_exclude_names`, and `attribute_mappings`
  apply to `full` and `openinference` endpoints. The `gen_ai` endpoint ignores
  those controls. `semantic_selector` and `capture_content` are unsupported.
- In 0.6, `attribute_mappings` remains available on the separate exporters.
- The `gen_ai` projection emits no `nemo_relay.*` fields and omits messages,
  tool payloads, retrieval content, marks, rerankers, and unsupported scopes.
- Although `gen_ai` does not emit guardrail or reranker spans, it retains their
  parent context so supported descendant spans preserve their trace parentage.
- The `gen_ai` projection follows the pinned OpenTelemetry GenAI
  semantic-conventions v1.42-era snapshot at commit
  `43633a68ef8f8ed87a1d5eb205990311ca708bf1`.
- NeMo Relay deterministically derives compliant trace and span IDs from Relay
  lifecycle UUIDs. Endpoints that receive the same events therefore share
  native IDs and parentage.
- Start with `http_binary` transport and an OTLP traces endpoint such as a local
  collector on port `4318` unless deployment requirements differ.
- In 0.7, the subscriber owns the Tokio runtime needed by `grpc`, including for
  synchronous direct construction. In 0.6, use `grpc` only with an active Tokio
  runtime.
- Use explicit config objects for non-secret application behavior. In 0.7
  plugin configuration, map authentication header names to secret-bearing
  environment variables with `opentelemetry.endpoints[].header_env`; Relay
  snapshots those values when the plugin activates. In 0.6, construct headers
  in memory from the deployment's secret injection and pass them to the
  exporter. Never place resolved credential values in source code, committed
  configuration, command-line arguments, prompts, examples, or diagnostics.
- Prefer an unauthenticated loopback collector for the first local proof. For a
  remote collector, require TLS certificate verification and reject endpoints
  that embed credentials in URL user information or query parameters.
- Register before the first instrumented request, use stable service identity,
  flush during graceful shutdown, and redact sensitive payloads before
  production export.
- Validate export by checking subscriber construction, collector requests,
  backend spans for synthetic scopes/tools/LLMs, and span grouping by root
  scope. Report header names and response status only; never print header values.

## Things To Confirm

- Transport: `http_binary` vs `grpc`
- Endpoint, TLS verification, and required authentication header names
- Service naming and resource attributes
- Whether deterministic flush-before-exit is required
- Whether the chosen binding and target support the desired transport

## Troubleshooting Focus

- No spans visible
- Wrong endpoint or authentication: inspect response status and redacted header
  names without logging credential values
- Events emitted outside active scopes
- For 0.6, `grpc` selected without an active Tokio runtime; for 0.7, gRPC
  endpoint, TLS, or authentication mismatch
- Forgetting register/deregister or flush/shutdown steps

## Related Skills

- `nemo-relay-plugin-observability`
- `nemo-relay-debug-runtime-integration`
