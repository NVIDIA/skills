<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Results, Evidence, And Errors

## RunResult Fields

Every invocation that reaches the adapter boundary returns a normalized
`RunResult`. Inspect `status` before reading output. Generated reference: the
[types reference](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/reference/api/python-library-reference/nemo_fabric.types.md).

| Field | Meaning |
| --- | --- |
| `status` | Terminal invocation status: `succeeded`, `failed`, or `cancelled`. Branch on this. |
| `error` | Structured failure metadata when available. Branch on `status`, not on this field. |
| `output` | Harness output normalized to the configured output schema. |
| `usage` | Optional `RunUsage` with invocation-local input, cached-input, output, and total token counts, reported cost in USD, and provider metadata. |
| `system_stats` | Optional `InvocationSystemStats`: adapter exchange duration, before/after process-tree observations, and an optional CPU counter difference estimate. |
| `resource_snapshot` | Optional `RuntimeResourceSnapshot` of the runtime, read once after the adapter returned the result. `None` when the read failed. |
| `artifacts` | Output files, logs, patches, and other materialized references. |
| `telemetry` | References to NVIDIA NeMo Relay or other telemetry streams from the run. |
| `events` | Ordered normalized lifecycle and invocation events. |
| `metadata` | Result-specific structured metadata. |
| `runtime_id`, `invocation_id`, `request_id` | Correlation IDs across runtimes, logs, telemetry, and artifacts. |

Handle a result with this status-first pattern:

```python
if result.status == "succeeded":
    use_output(result.output, result.artifacts, result.telemetry)
else:
    handle_failure(result.status, result.error, result.events)  # failed, cancelled, ...
```

Read usage only when the target reports it:

```python
if result.usage is not None:
    record_usage(
        input_tokens=result.usage.get("input_tokens"),
        output_tokens=result.usage.get("output_tokens"),
        total_tokens=result.usage.get("total_tokens"),
        cost_usd=result.usage.get("cost_usd"),
    )
```

Treat missing counters and cost as unknown, not zero. `cached_input_tokens` reports cached prompt tokens; `input_tokens_include_cache` declares whether `input_tokens` includes them (`True`), excludes them (`False`), or has unknown semantics (absent). Do not add cache to an inclusive input count. Adapters can put cache-write, reasoning, peak request, and per-model detail in the descriptor-declared `usage.extensions["nemo_fabric.detailed_token_usage.v1"]` object. NeMo Fabric preserves that object at `result.usage["metadata"]["nemo_fabric.detailed_token_usage.v1"]`. Treat detailed input and output independently: any detailed accounting field for a dimension makes that dimension authoritative, with no fallback to its top-level counters. The views can diverge. Calculate an inclusive total only when the selected detailed counters provide enough information; otherwise leave the total unknown. Keep estimated costs separate in provider metadata; `cost_usd` is reported cost, not an estimate.

## Runtime Resource Snapshots

`await runtime.resource_snapshot()` returns a `RuntimeResourceSnapshot` of the local adapter host and its descendants. By default, `RunResult.system_stats` brackets each invocation under its host lock: `before` and `after` have `status="available"` plus a typed `snapshot`, or `status="unavailable"` / `"unsupported"` plus `reason`. A failed read preserves the adapter outcome. `elapsed_seconds` uses a monotonic clock around the adapter exchange and excludes startup, endpoint collection, artifact preparation, and teardown. Older results and backends without collection have `system_stats=None`.

`cpu_time_seconds_estimate` subtracts the baseline CPU counter from the final counter, excluding previous invocations on a warm runtime. It is absent if an endpoint is missing, identities differ, or the counter decreases. Reaping, reparenting, and background work in the same tree can distort a positive estimate too. Endpoint memory and process counts are instantaneous, not peaks or averages. The compatible `resource_snapshot` field repeats the final available reading. Collection runs where the adapter executes, including inside a Harbor task runner, and requires no Relay collector or additional identity hierarchy. Default endpoint collection adds no artifact file; opt-in sampling retains the correlated files described below. Protocol failures without a normalized result still raise without a partial `RunResult`. Enforce limits in the operating system or execution environment.

## Native Sampled System Stats

Omitting `FabricConfig.system_stats` selects all supported groups at the endpoints. Set it explicitly to `None` to disable invocation stats and their artifacts; null survives planning and runner transport. The existing final `resource_snapshot` remains independent. `SystemStatsConfig()` enables endpoints without sampling. To sample, use `SystemStatsConfig(sampling=SystemStatsSamplingConfig(interval_ms=1000, max_samples=10000))`. These are NeMo Fabric-owned settings and are not projected into adapter runtime extensions. Sampling needs neither Relay nor recording.

`collect=["process", "cgroup.cpu"]` selects groups for both endpoint and periodic collection. Supported groups are `process`, `cgroup.cpu`, `cgroup.memory`, and `cgroup.pressure`; omission selects all, while empty lists, duplicates, and unknown names are rejected. The invocation collector skips excluded groups; the independent final `resource_snapshot` still reads the process tree. `collected_groups` records selection; excluded observations/measurements have `status="not_collected"`, distinct from selected-but-unavailable evidence. Unselected process reductions and an entirely excluded cgroup block are omitted. Setting `config.system_stats.sampling=None` disables only sampling and keeps the selected endpoints.

`RunResult.system_stats.sampled` carries gauge final/min/sampled peak/time-weighted mean, coverage, collection time, dropped samples, truncation, and a JSONL artifact reference. Both stats artifacts are referenced by `RunResult.artifacts`; each sample retains all three existing execution IDs. Late samples do not cross invocation boundaries. Retained JSONL offsets allow consumer-owned benchmark tooling to inspect trajectory windows. Gauge means use hold-last-value only across valid intervals no larger than twice cadence; extrema remain sampled.

`system_stats.cgroup` describes the adapter host's Linux cgroup v2, not exclusive invocation usage. Visible ancestor limits, accounted memory, swap, provider-lifetime memory peak, CPU throttling, OOM kills, and pressure are distinct from process-tree RSS/CPU. Scopes overlap; do not sum them. Hidden ancestors and missing sources remain unknown. A provider high-water mark is never relabeled an invocation peak. Collection does not modify limits or reset counters, and optional collection/persistence errors preserve agent status.

## Correlation IDs

`runtime_id` identifies the runtime lifecycle, `invocation_id` identifies one
invocation within it, and `request_id` correlates the caller's request.
Values generated by NVIDIA NeMo Fabric use type-specific prefixes such as
`runtime-`, `invocation-`, and `request-`; a caller can supply its own
`request_id`. Store
and log each field separately and treat every value as opaque — do not parse or
reuse the encoding.

## Error Hierarchy

All public SDK errors inherit from `FabricError`. NeMo Fabric raises these when it
cannot return a normalized result; it does not return a partial `RunResult`.
Refer to the [errors reference](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/reference/api/python-library-reference/nemo_fabric.errors.md).

| Error | Meaning |
| --- | --- |
| `FabricConfigError` | Invalid config, request, or override. |
| `FabricCapabilityError` | Selected adapter does not support the requested operation. |
| `FabricRuntimeError` | Startup, invocation, or shutdown failed before a normalized result. |
| `FabricStateError` | Invalid runtime state transition (invoking after stop, overlapping invocations). |
| `FabricNativeUnavailableError` | Native extension is not installed or importable. |
| `ResourceObservationUnavailableError` | A resource snapshot cannot be read because the runtime stopped, is stopping, its adapter host exited, a live process in its tree or the process table cannot be read, or the process tree kept changing during the read. Subclass of `FabricRuntimeError`. |
| `ResourceObservationUnsupportedError` | The runtime's adapter kind, environment provider, or operating system has no resource reader. Subclass of `FabricRuntimeError`. |

## Cleanup And Resilience

Core host-operation deadlines (`host_timeout`) become Python `TimeoutError`, then
`FabricRuntimeError` with `code="timeout"` and the failed lifecycle stage.
If invocation succeeds in `Fabric.run()` but stop reaches this deadline,
it returns a failed `RunResult` with `error.stage == "stop"` and
`error.code == "timeout"`. Check the returned result as well as raised errors.
Adapters can report execution deadlines in normalized results with
`result.error.code == "timeout"`. Other adapter-specific timeout codes are not
automatically reclassified. Do not classify deadlines from error text.

- Prefer `run(...)` and `async with` runtimes: both attempt cleanup
  automatically. Shutdown is attempted, not guaranteed — `stop()`, including the
  automatic call at `async with` exit, can raise `FabricRuntimeError`. On a
  normal block exit that error propagates; when an invocation already failed, the
  cleanup failure is attached to the original exception rather than replacing it.
- The consumer owns job-level retries and rollout failure policy. NeMo Fabric marks a
  runtime or invocation failed and returns structured error metadata when
  possible, but does not retry by default.
- Transient failures can carry retryable error metadata. Capacity pressure
  surfaces as a structured error or event (busy, rate limited, backpressure).
  The consumer decides whether to wait, retry, start a replacement runtime, or
  escalate.
