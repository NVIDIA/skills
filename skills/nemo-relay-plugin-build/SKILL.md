---
name: nemo-relay-plugin-build
description: Use this skill when building or packaging reusable NeMo Relay runtime behavior as an embedded configuration component or a manifest-backed `rust_dynamic` native or `worker` gRPC plugin, with deterministic validation and rollback-safe registration.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation and Affiliates
---

# Build a Plugin

Use this skill when a user wants to package reusable NeMo Relay runtime behavior
behind plugin configuration.
Keep reusable plugin behavior separate from one-off application startup code.

## Use This When

Use this skill when the behavior should be activated by shared config and reused
across applications, teams, or process startup paths.

Common cases:

- Register subscribers, guardrails, intercepts, or a small bundle of related
  runtime behavior.
- Validate operator-supplied config before changing runtime behavior.
- Give reusable behavior a stable plugin `kind` and activation lifecycle.
- Package behavior that should be enabled, disabled, or rolled out through
  plugin config rather than repeated application startup code.

## Do Not Use This When

Do not build a plugin when a narrower NeMo Relay surface is enough:

- One request or tenant needs temporary behavior -> use scope-local middleware.
- The user only needs first-time scopes, tool calls, or LLM calls ->
  `nemo-relay-instrument-calls`.
- The user only needs to choose an exporter path ->
  `nemo-relay-plugin-observability`.
- The behavior depends on live callables, provider clients, file handles,
  credentials, or framework objects inside config.

## Choose A Delivery Model

Choose the delivery model before designing config or registration:

- **Embedded component:** Use the shared plugin document when behavior ships
  with the Relay host application. Follow the embedded component model below.
- **Discoverable dynamic package:** Use a `relay-plugin.toml` manifest when the
  plugin ships independently of the host. Use `rust_dynamic` for a trusted
  in-process native Rust library, or `worker` for a local `grpc-v1` worker.
  Read the public native dynamic-plugin or gRPC worker guide before designing
  the package boundary. Do not reproduce native ABI or worker-protocol details
  from memory.

Native plugins are trusted C-ABI extensions that run in the Relay process and
are not sandboxed. Worker plugins provide process isolation, not a security
sandbox.

## Embedded Component Model

- Plugins package reusable process-level behavior.
- A plugin exposes a stable `kind` string and receives component-local config
  from a shared plugin document.
- Plugin config must be JSON-compatible across Rust, Python, Node.js, files,
  tests, and deployment systems.
- Validation is deterministic and side-effect free. It inspects config and
  returns structured diagnostics before runtime behavior changes.
- Registration runs after validation and installs real behavior through
  `PluginContext`, such as subscribers, guardrails, request intercepts,
  execution intercepts, or stream execution intercepts.
- `PluginContext` gives the plugin system enough ownership to qualify runtime
  names and roll back partial setup when activation fails.
- Disabled components should still validate when possible so operators can find
  config problems before rollout.

## Default Path

1. Decide whether a plugin is actually needed. Prefer direct instrumentation or
   scope-local behavior when the use case is not reusable process-level
   behavior.
2. Choose an embedded component or a discoverable dynamic package. For a
   dynamic package, establish the manifest-backed native or worker boundary
   before implementing behavior.
3. For native dynamic plugins, determine the target Relay version from the
   package constraint, lockfile, or deployment target before choosing callback
   APIs.
4. Pick one first runtime surface: subscriber-oriented export, sanitize
   guardrail, conditional guardrail, request intercept, execution intercept, or
   stream execution intercept.
5. Choose a stable plugin `kind` and the smallest JSON-compatible config shape.
6. Define diagnostics for missing fields, unsupported values, unknown fields,
   unsafe config, and invalid field combinations.
7. Validate config before initialization. Validation must not open network
   connections, create clients, register middleware, or mutate process state.
8. If validation returns error diagnostics, return them and stop without
   initialization or registration.
9. Register runtime behavior through `PluginContext`, not by hand-registering
   global behavior inside application startup.
10. Test activation, disabled components, validation failures, and registration
   failure rollback.
11. Document how to enable the plugin, what config fields are supported, and how
   to roll back the component.
12. For a dynamic plugin that should provide structured fields in
   `nemo-relay plugins edit`, declare the `config_schema` capability and
   reference a local Draft 7 or Draft 2020-12 JSON Schema file from
   `[config_schema].path` in `relay-plugin.toml`. Schema-less plugins remain
   editable as raw JSON objects.

## Config Shape

The top-level plugin document contains `version`, `components`, and `policy`.
Each component supplies the plugin `kind`, `enabled`, and component-local
`config`:

```json
{
  "version": 1,
  "components": [
    {
      "kind": "redaction-policy",
      "enabled": true,
      "config": {
        "preset": "strict"
      }
    }
  ],
  "policy": {
    "unknown_component": "warn",
    "unknown_field": "warn",
    "unsupported_value": "error"
  }
}
```

Keep business logic in plugin code, not in config. Use references to secrets or
endpoints rather than embedding sensitive values.

## Binding Pointers

- Python: `nemo_relay.plugin`
- Node.js: `nemo-relay-node/plugin`
- Rust: `nemo_relay::plugin`
- Go and raw FFI are source-first or advanced surfaces.

Use the same canonical `snake_case` config keys across bindings and files. Node
helper functions can be `camelCase`, but plugin config objects remain
`snake_case`.

## Dynamic Package Essentials

Keep the manifest package contract separate from the operator's plugin document.
The manifest must declare the lane-specific kind and load contract, a normal
SemVer Relay compatibility range, artifact integrity, and only the capabilities
the plugin needs. Use a local `config_schema` only when structured CLI editing
is needed.

## Native Version Compatibility

Choose the native callback model from the target Relay version; do not present
the 0.8 SDK as source-compatible with 0.7:

- **Relay 0.7:** Keep typed Rust middleware callbacks synchronous. Use the raw
  native ABI v3 completion-based registration path only when asynchronous work
  is required, and constrain the manifest to `compat.relay = ">=0.7,<0.8"`.
- **Relay 0.8:** Return futures through the typed Rust SDK and constrain
  the manifest to `compat.relay = ">=0.8.0,<1.0"`. The SDK runs typed middleware
  on an SDK-owned Tokio executor; subscribers and raw synchronous ABI
  registrations remain synchronous. Do not block executor workers. Scope context
  does not automatically propagate to tasks created with `tokio::spawn`, and
  teardown must stop new callbacks and drain accepted work before unload.

The 0.8 typed SDK lets native components configure its executor with a positive
`executor.worker_threads` value. When their manifest exposes a
`config_schema`, include that SDK-owned object and field in the schema. For
native ABI details, callback settlement or cancellation, and the full worker
protocol, use the relevant public dynamic-plugin guide rather than expanding
this skill.

## Failure Modes To Avoid

- Do not put callables, clients, credentials, framework objects, file handles,
  or caches in plugin config.
- Do not perform runtime registration during validation.
- Do not skip validation for disabled components.
- Do not register directly through global startup code when `PluginContext`
  should own the runtime behavior.
- Do not combine unrelated subscribers, request transforms, and policy checks
  in the first plugin unless one config document clearly owns the bundle.
- Do not export raw production payloads or secrets. Add telemetry sanitization
  before data leaves the process.
- Do not ignore partial activation failures. Roll back or surface a clear
  diagnostic.
- Do not treat native process isolation as a security boundary, or claim that a
  worker is sandboxed.
- Do not apply the 0.8 typed-async callback contract to Relay 0.7, or the 0.7
  raw completion contract to typed 0.8 middleware.
- Do not block the 0.8 SDK-owned executor.

## Validation Checklist

- [ ] Stable plugin `kind` chosen.
- [ ] Delivery model selected: embedded component, `rust_dynamic`, or `worker`.
- [ ] Config shape is JSON-compatible and uses `snake_case`.
- [ ] Required fields and unsupported values produce stable diagnostics.
- [ ] Unknown fields follow the configured policy.
- [ ] Disabled components still report config problems where possible.
- [ ] Initialization installs behavior through `PluginContext`.
- [ ] A forced registration failure does not leave partial runtime behavior
      active.
- [ ] Docs or examples show how to enable and roll back the plugin.
- [ ] Dynamic plugins that need structured CLI editing package a valid local
      JSON Schema and declare `config_schema` in `relay-plugin.toml`.
- [ ] Dynamic manifests validate their lane-specific kind, load contract,
      SemVer compatibility, integrity, and disabled-record behavior.
- [ ] Native callback APIs and compatibility constraints match the target Relay
      version: synchronous typed callbacks and raw ABI v3 completion work for
      0.7; typed async SDK middleware and executor configuration for 0.8.
- [ ] Async native paths cover cancellation, settlement, and drain-before-
      unload behavior.

## Use Another Skill When

- You only need to wrap direct tool or LLM calls ->
  `nemo-relay-instrument-calls`
- You need to set up traces or exporters without packaging a plugin ->
  `nemo-relay-plugin-observability`
- You need to debug plugin activation, missing events, or load failures ->
  `nemo-relay-debug-runtime-integration`
