<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Use Adaptive Hints And Guidance

## Use This When

Use this reference when the adaptive plugin is already configured and the
application wants to consume adaptive outputs such as model-request hints,
parallelism guidance, or ACG diagnostics.

## Do Not Use This When

Do not use this reference to design the first adaptive rollout or to configure
the plugin from scratch. Start with `nemo-relay-plugin-adaptive-tuning`; read
`references/config.md` when exact configuration fields are needed.

## Focus Areas

- `adaptive_hints` metadata injected into outgoing model requests.
- Latency sensitivity and request-local guidance.
- Parallel groups or tool-parallelism guidance.
- Adaptive Cache Governor reports and diagnostics during rollout.

## Embedded Hint Semantics

- Adaptive hints are request-intercept behavior. They can inject metadata into a
  configured header or body path; the default body path is
  `nvext.agent_hints`.
- Hint configuration includes `priority`, `break_chain`, `inject_header`, and
  `inject_body_path`. Lower priority values run earlier.
- Tool parallelism can run in `observe_only`, `inject_hints`, or `schedule`.
  Use `observe_only` until tool idempotency and race behavior are understood.
- ACG output is provider-specific prompt-cache planning guidance. Use more
  samples, raise stability thresholds, or switch to `passthrough` when cache
  planning is unstable.
- `set_latency_sensitivity(...)` is a request-local execution hint, not
  persistent adaptive configuration.
- Normal adaptive runtime behavior should come from explicit config objects, not
  environment variables. `NEMO_RELAY_ACG_DEBUG` is for cache-governor diagnostics
  and `NEMO_RELAY_RUN_REDIS_TESTS` is for Redis-backed tests.

## Default Path

1. Confirm adaptive telemetry and config validation are working.
2. Identify where the hint is injected or surfaced in the chosen binding.
3. Keep hints advisory unless the consuming API defines stronger semantics.
4. Test behavior when no prediction or hint is available.
5. Compare application output against the baseline after enabling hints.
6. Escalate from hints to scheduling only after idempotency and race behavior
   are understood.

## Failure Modes To Avoid

- Do not make application correctness depend on a prediction being present.
- Do not hide behavior changes behind adaptive hints without runtime evidence.
- Do not enable scheduling for non-idempotent or order-sensitive tools.
- Do not adjust intercept priority without checking other request intercepts.

## Related Skills

- `nemo-relay-plugin-adaptive-tuning`
- `nemo-relay-debug-runtime-integration`
