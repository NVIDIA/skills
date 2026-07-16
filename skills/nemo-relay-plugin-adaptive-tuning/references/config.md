<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Configure Adaptive Tuning

## Use This When

Use this reference when an application already intends to use NeMo Relay adaptive
features and needs the correct plugin configuration shape.

## Do Not Use This When

Do not use this reference when the user only needs first-time instrumentation,
request-specific middleware, or production trace debugging.

## Embedded Configuration Model

- Adaptive tuning is a top-level plugin component with kind `adaptive`.
- The adaptive object contains `version`, `agent_id`, `state`, `telemetry`,
  `adaptive_hints`, `tool_parallelism`, `acg`, and `policy`.
- Wrap the adaptive object in an adaptive `ComponentSpec`, insert it into the
  shared plugin config `components` list, validate the plugin config, then
  initialize the plugin system.
- Python uses `nemo_relay.adaptive.AdaptiveConfig(...)`,
  `nemo_relay.adaptive.ComponentSpec(...)`, and
  `nemo_relay.plugin.PluginConfig(...)`.
- Node.js uses `require("nemo-relay-node/adaptive")` helpers such as
  `defaultConfig()`, `inMemoryBackend()`, `toolParallelismConfig(...)`, and
  `ComponentSpec(...)`, then activates through `nemo-relay-node/plugin`.
- Rust uses `nemo_relay_adaptive::{AdaptiveConfig, ComponentSpec, ...}` and
  `nemo_relay::plugin::{validate_plugin_config, initialize_plugins}`.
- Go and raw FFI are source-first or advanced surfaces.
- Plugins install runtime behavior such as subscribers, guardrails, intercepts,
  and related helpers. Adaptive is a built-in plugin component, not a separate
  runtime model.

## Default Path

1. Build the shared plugin config document or binding-native helper config.
2. Add one top-level adaptive `ComponentSpec`.
3. Start with `state.backend = in_memory`.
4. Enable telemetry first.
5. Add only one active section at a time: `adaptive_hints`,
   `tool_parallelism`, or `acg`.
6. Validate the config before initialization.
7. Initialize through the shared plugin system.
8. Clear or replace the plugin configuration cleanly when the app lifecycle
   changes.

## Defaults To Remember

- `adaptive_hints.priority` defaults to `100`, `break_chain` to `false`,
  `inject_header` to `true`, and `inject_body_path` to `nvext.agent_hints`.
- `tool_parallelism.mode` defaults to `observe_only`.
- `acg.provider` defaults to `passthrough`, with priority `50` and observation
  window `100`.
- Redis-backed state is for persistence or cross-worker sharing, not the first
  local rollout.

## Failure Modes To Avoid

- Do not initialize before validation succeeds.
- Do not enable multiple active tuning sections in the first rollout.
- Do not put callables, clients, credentials, or framework objects in plugin
  config.
- Do not enable active scheduling or request rewriting without a baseline and a
  rollback path.

## Checklist

- [ ] Adaptive is modeled as a top-level plugin component.
- [ ] Backend chosen, with `in_memory` first unless persistence is required.
- [ ] Adaptive sections chosen explicitly.
- [ ] Config validated before initialization.
- [ ] Rollout starts from telemetry or observe-only behavior.
- [ ] Plugin lifecycle matches the app lifecycle.

## Related Skills

- `nemo-relay-plugin-adaptive-tuning`
- `nemo-relay-debug-runtime-integration`
- `nemo-relay-plugin-build`
