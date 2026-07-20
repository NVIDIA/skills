<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Export OpenTelemetry Traces

Use this reference when the destination is an OTLP/OpenTelemetry backend such as an
OpenTelemetry Collector, Jaeger, Tempo, or Honeycomb.

## Default Path

- Build the binding-specific `OpenTelemetryConfig`
- Set endpoint and service identity; add authentication only when the collector
  requires it
- Construct the subscriber
- Register it before running scoped work
- Deregister, flush, and shut down when the process or subsystem is done

## Embedded OpenTelemetry Semantics

- OpenTelemetry export maps NeMo Relay runtime events into OTLP traces for
  tracing backends and collectors.
- Set `transport`, `endpoint`, and `service_name`, then add a namespace, version,
  instrumentation scope, headers, resource attributes, timeout, or
  `attribute_mappings` when needed.
- NeMo Relay projects start, end, handle, and mark payload fields to typed OTLP
  attributes with dotted names. Start and end metadata use
  `nemo_relay.start.metadata` and `nemo_relay.end.metadata`.
- NeMo Relay emits a top-level object or array field as a JSON string, omits a
  top-level `null` field, and no longer emits the old aggregate `*_json` payload
  attributes.
- Use `attribute_mappings` to copy a fully qualified projected attribute to a
  backend-specific alias without changing its OTLP type.
- Start with `http_binary` transport and an OTLP traces endpoint such as a local
  collector on port `4318` unless deployment requirements differ.
- `grpc` transport is available when a Tokio runtime is active.
- Use explicit config objects for non-secret application behavior. Load
  credentials at runtime through the deployment's secret-injection mechanism,
  construct authentication headers in memory, and pass them directly to the
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
- `grpc` selected without a Tokio runtime
- Forgetting register/deregister or flush/shutdown steps

## Related Skills

- `nemo-relay-plugin-observability`
- `nemo-relay-debug-runtime-integration`
