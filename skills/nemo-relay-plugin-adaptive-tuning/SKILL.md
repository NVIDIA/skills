---
name: nemo-relay-plugin-adaptive-tuning
description: Use this skill when baseline NeMo Relay instrumentation exists and the user wants to configure or evaluate adaptive plugin behavior, including telemetry, state, adaptive_hints, tool_parallelism, ACG, hint consumption, or measured rollout.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation and Affiliates
---

# Tune Adaptive Plugin Behavior

## Use This When

Use this skill when a user has baseline NeMo Relay instrumentation and wants to
improve latency, parallelism, prompt-cache behavior, or model-request behavior
from runtime signals.
Keep adaptive behavior measured against a known baseline.

## Do Not Use This When

Do not use this skill when the application is not instrumented yet. Start with
`nemo-relay-instrument-calls` or `nemo-relay-get-started` first.

## Default Guidance

- Observe first, compare against a baseline, then enable one behavior change at
  a time.
- Use the adaptive plugin component rather than inventing separate tuning logic
  or hand-registering adaptive behavior at every call site.
- Start with in-memory state and telemetry-only behavior for local development.
- Move to persistent state only when learned signals must survive restarts or be
  shared across workers.
- Add active behavior only after representative runtime events show what should
  change.

## Embedded Adaptive Model

- Adaptive behavior is configured through the first-party plugin component with
  kind `adaptive`.
- Adaptive requires existing NeMo Relay scopes and at least one relevant
  managed tool or LLM lifecycle event stream because it learns from runtime
  signals.
- Main configuration areas are state, telemetry, adaptive hints, tool
  parallelism, Adaptive Cache Governor (ACG), and rollout policy.
- State backends are `in_memory` and `redis`.
- Tool-parallelism modes are `observe_only`, `inject_hints`, and `schedule`.
- Adaptive Cache Governor providers are `passthrough`, `anthropic`, and
  `openai`; omit ACG until prompt-cache planning is needed.
- Helper APIs exist in Rust `nemo_relay_adaptive`, Python `nemo_relay.adaptive`,
  and Node.js `nemo-relay-node/adaptive`. Go and raw FFI are
  source-first or advanced surfaces.

## Default Path

Use this rollout sequence:

1. Confirm the app emits scope events and the managed tool or LLM events needed
   for the behavior being evaluated. Do not require both call types when the
   workflow uses only one.
2. Capture a baseline for the workflow you want to improve.
3. Enable adaptive telemetry with in-memory state.
4. Read `references/config.md` when exact plugin configuration fields are needed.
5. Run representative traffic and inspect reports or runtime events.
6. If configuration validation fails or expected events are absent, return the
   diagnostics and stop. Keep the last known working configuration active.
7. Before enabling scheduling, verify tool idempotency and race behavior. Before
   enabling ACG, verify that provider request payloads are stable.
8. Enable the smallest behavior change in config.
9. Read `references/hints.md` when application logic consumes adaptive hints,
   tool-parallelism guidance, or ACG diagnostics.
10. Compare results against the baseline. If latency, correctness, or failure
    rate regresses, restore the last known working configuration and retain the
    sanitized diagnostics for review.

## Failure Modes To Avoid

- Do not enable scheduling before tool idempotency and race behavior are known.
- Do not enable prompt-cache planning before provider payloads are stable.
- Do not treat adaptive hints as mandatory instructions unless the consuming
  path explicitly defines that contract.
- Do not use environment variables as the primary adaptive configuration model.
- Do not tune from a single run or unrepresentative traffic.
- Do not suppress or replace original tool and model errors.
- Do not add retries until the call owner defines their safety.
- Revert adaptive behavior when it increases the failure rate.

## Load A Reference When

- You need the exact adaptive config shape -> `references/config.md`
- You need to consume adaptive hints or scheduling guidance in app logic ->
  `references/hints.md`

## Use Another Skill When

- You need to build reusable plugin behavior instead of configuring the built-in
  adaptive component -> `nemo-relay-plugin-build`

## Related Skills

- `nemo-relay-get-started`
- `nemo-relay-instrument-calls`
- `nemo-relay-plugin-observability`
- `nemo-relay-plugin-build`
