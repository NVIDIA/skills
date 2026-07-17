---
license: Apache-2.0
name: doca-mgmt
description: >
  Use this skill when the user is doing hands-on DOCA Management
  programming against BlueField / ConnectX devices — standing up a
  management or representor context (doca_mgmt_dev_ctx /
  doca_mgmt_dev_rep_ctx), querying device caps (data-direct,
  caps-general), toggling congestion-control global status, modifying
  diagnostics-data, setting ICM quotas, or issuing a raw firmware
  command via doca_mgmt_raw_cmd with the right scope (CONFIGURATION /
  DEBUG_READ_ONLY / DEBUG_WRITE / DEBUG_WRITE_FULL). Trigger even when
  the user does not say "DOCA Management" — typical implicit phrasings
  include "fleet tool that walks every BlueField and reads device
  state", "toggle data-direct on a VF", "set an ICM quota per
  representor", "send a raw firmware command from C",
  "DOCA_ERROR_IO_FAILED from raw_cmd", or "fwctl ioctl is failing".
  Refuse and route elsewhere for mlxconfig direct operation, BFB /
  firmware reflash, streaming telemetry, doca_caps CLI snapshots, or
  DOCA install itself — those belong to other skills.
metadata:
  kind: library
compatibility: >
  Requires DOCA SDK installed at /opt/mellanox/doca on Linux (Ubuntu
  22.04/24.04 or RHEL/SLES) with a BlueField DPU or ConnectX NIC
  attached. Reads the user's local install via `pkg-config doca-mgmt`
  + `doca-common` and inspects
  /opt/mellanox/doca/{lib,include,samples,applications}; runtime
  operations require root (or device-admin
  cap_*) and an /dev/fwctl* character device exposed by the host
  kernel.
---

# DOCA Management

**Where to start:** This skill assumes DOCA is already installed
and the user is doing **hands-on management-plane work** against
a BlueField / ConnectX device — typically a fleet-management or
orchestration tool that needs to query or modify device-level
state programmatically. Open [`TASKS.md`](TASKS.md) if the user
wants to *do* something (install / configure / build / modify /
run / test / debug / use); open [`CAPABILITIES.md`](CAPABILITIES.md)
when the question is *what can doca-mgmt express on this version* —
the management context model, the raw-command scope ladder, the
sub-domain surfaces (caps-general, cc-global-status, diagnostics-
data, icm-quota), version compatibility, and the safety overlay.
If the user has not installed DOCA yet, route to
[`doca-setup`](../../doca-setup/SKILL.md) first.

## Example questions this skill answers well

The CLASSES of management-plane questions this skill is built to
answer, each with one worked example. The agent should treat the
*class* as the load-bearing piece — the worked example is a
single instance.

- **"Is `doca-mgmt` even the right surface, or do I want
  telemetry / bench / caps?"** — worked example: *"I'm building a
  fleet inventory tool — do I use doca-mgmt to query each
  BlueField's data-direct capability, or doca-telemetry, or
  doca_caps?"*. Answered by the *management-plane vs
  observability-plane vs read-only-CLI* selection rule in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  surface-selection table.
- **"How do I stand up a management context on a device (and a
  representor)?"** — worked example: *"open a `doca_mgmt_dev_ctx`
  on the device, then a `doca_mgmt_dev_rep_ctx` on a specific
  VF representor for caps-general programming"*. Answered by
  the management-context lifecycle in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the configure walk in
  [`TASKS.md ## configure`](TASKS.md#configure).
- **"How do I query a device capability — say, whether it
  supports data-direct?"** — worked example: *"create a
  caps-general handle, call `doca_mgmt_device_caps_general_get`
  on the representor context, read the data-direct flag"*.
  Answered by the capability-query pattern in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the test step in [`TASKS.md ## test`](TASKS.md#test).
- **"How do I modify a device-level feature flag safely?"** —
  worked example: *"toggle `data_direct` on a representor;
  capture pre-state, write, verify, prepare rollback"*.
  Answered by the apply-with-rollback workflow in
  [`TASKS.md ## modify`](TASKS.md#modify) layered on the
  bundle-wide hardware-safety meta-policy referenced from
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy).
- **"What does the `doca_mgmt_raw_cmd` scope mean and which
  scope should I use?"** — worked example: *"I have a vendor-
  documented opcode for a `DEBUG_READ_ONLY` query — what scope
  does that need and what is the blast radius?"*. Answered by
  the command-scope ladder in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the raw-command verb in [`TASKS.md ## use`](TASKS.md#use).
- **"What does this `DOCA_ERROR_*` from a `doca_mgmt_*` call
  mean and which layer caused it?"** — worked example:
  *"`DOCA_ERROR_IO_FAILED` from `doca_mgmt_raw_cmd`"*.
  Answered by the mgmt overlay on the cross-library taxonomy
  in [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
  + the layered ladder in
  [`TASKS.md ## debug`](TASKS.md#debug) that escalates to
  [`doca-debug`](../../doca-debug/SKILL.md) and to
  [`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md)
  when the cause is a device-state change.

## Audience

This skill serves **external developers building fleet-management,
orchestration, or device-administration tools that programmatically
query and modify BlueField / ConnectX device-level state** — i.e.,
users whose code calls `doca_mgmt_*` (directly in C/C++, or
through FFI/bindings from another language) to inspect device
capabilities, toggle device feature flags, query diagnostics
counters, set ICM quotas, or issue raw firmware-control commands.
The canonical caller is a fleet-management agent that walks every
BlueField in a data center and applies a desired-state diff. This
skill is *not* for NVIDIA developers contributing to DOCA
Management itself, and it is not the right surface for live
performance benchmarking or stream-based observability — those
belong to [`doca-bench`](../../tools/doca-bench/SKILL.md),
[`doca-telemetry`](../doca-telemetry/SKILL.md), and
[`doca-telemetry-exporter`](../doca-telemetry-exporter/SKILL.md).

## Language scope

DOCA Management ships as a C library with the `pkg-config` module
name `doca-mgmt`. The library's surface is C; the shipped samples
on a real install (where present) are C. C and C++ consumers are
the canonical case and the workflows in `TASKS.md` assume that
path. Other-language consumers (Rust, Go, Python, …) consume the
same `*.so` library through FFI or language-specific bindings;
the skill's contribution in that case is to keep the management-
context lifecycle, the command-scope ladder, the capability-query
pattern, the version-handling rule, and the safety overlay
language-neutral, and to route the agent to the public C ABI as
the authoritative surface that any wrapper will eventually call.
The skill does not author wrappers in any language.

## When to load this skill

Load this skill when the user is doing **hands-on DOCA Management
work** on a host with one or more BlueField / ConnectX devices.
Concretely:

- Standing up a `doca_mgmt_dev_ctx` on a `doca_dev` to inspect
  or modify device-level state.
- Standing up a `doca_mgmt_dev_rep_ctx` on a `doca_dev_rep` (or
  via `doca_mgmt_dev_rep_ctx_create_by_pci_addr` when the
  representor is not available) to inspect or modify a VF /
  function-level configuration.
- Querying a device's general capabilities via
  `doca_mgmt_device_caps_general_*` (data-direct support and
  similar device-wide attributes).
- Programming a representor's general capabilities via the same
  `set` path.
- Querying or modifying congestion-control global status via
  `doca_mgmt_cc_global_status_*` (RP / NP protocol type,
  priority, enable / disable).
- Modifying or querying multi-domain diagnostics data via
  `doca_mgmt_diagnostics_data_*` against a specific device.
- Setting ICM quota limits per device or per representor via
  `doca_mgmt_icm_quota_*` and reading the current allocation /
  max-reached.
- Issuing a `doca_mgmt_raw_cmd` against the device's
  firmware-control endpoint, choosing the right
  `enum doca_mgmt_cmd_scope` for the operation's blast radius.
- Debugging a `DOCA_ERROR_*` returned by a `doca_mgmt_*` call
  and deciding whether the cause is a configuration mistake, a
  lifecycle ordering bug, a missing capability on the device,
  an `fwctl` ioctl failure, or a firmware-side rejection.

Do **not** load this skill for general DOCA orientation, install
of DOCA itself, live performance benchmarking, or stream-based
telemetry. For those, use
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md),
[`doca-setup`](../../doca-setup/SKILL.md),
[`doca-bench`](../../tools/doca-bench/SKILL.md), and
[`doca-telemetry`](../doca-telemetry/SKILL.md) respectively.

## What this skill provides

This is a **thin loader**. The body keeps only the orientation
needed to pick the right next file. The substantive
management-specific material lives in two companion files:

- `CAPABILITIES.md` — what doca-mgmt can express on this
  version: the management-context object model
  (`doca_mgmt_dev_ctx` + `doca_mgmt_dev_rep_ctx`), the raw-
  command scope ladder (`DOCA_MGMT_CMD_SCOPE_CONFIGURATION`,
  `DOCA_MGMT_CMD_SCOPE_DEBUG_READ_ONLY`,
  `DOCA_MGMT_CMD_SCOPE_DEBUG_WRITE`,
  `DOCA_MGMT_CMD_SCOPE_DEBUG_WRITE_FULL`), the sub-domain
  surfaces (caps-general, cc-global-status, diagnostics-data,
  icm-quota) with their capability-query gates, the EXPERIMENTAL
  symbol-set policy (every public `doca_mgmt_*` symbol is
  tagged EXPERIMENTAL), the mgmt overlay on the cross-library
  `DOCA_ERROR_*` taxonomy, the observability surface (pre-state
  capture + post-write re-query), and the per-artifact safety
  overlay on the bundle-wide hardware-safety meta-policy.
- `TASKS.md` — step-by-step workflows for the eight in-scope
  verbs: `install`, `configure`, `build`, `modify`, `run`,
  `test`, `debug`, `use`. Plus a `Deferred task verbs` block
  that points out-of-scope questions at the right next skill.

The skill assumes a host or BlueField where DOCA is already
installed at the standard location and the user has the
privileges their public install profile expects (management-
plane operations typically require root or an equivalent
device-administration capability). It does not cover installing
DOCA — that path goes through
[`doca-setup`](../../doca-setup/SKILL.md).

## What this skill deliberately does not ship

This skill is **agent guidance**, not a samples or templates
bundle. To keep the boundary clean, it deliberately does not
contain — and pull requests should not add:

- **Pre-written DOCA Management application source code, in any
  language.** Management-plane code touches device state and a
  wrong write can take the device offline; authoring it from
  documentation prose, especially against an EXPERIMENTAL
  symbol set whose shape can change between releases, is
  forbidden by this skill. The agent's job is to route the user
  to verified reference code (the shipped DOCA samples on the
  installed package set are the canonical worked examples) and
  to prescribe a minimum-diff modification via the universal
  modify-a-sample workflow in
  [`doca-programming-guide`](../../doca-programming-guide/SKILL.md).
- **Standalone build manifests** parked inside the skill. The
  agent constructs the build manifest *in the user's project
  directory* against the user's installed DOCA, where
  `pkg-config --modversion doca-mgmt` is the source of truth.
- **Raw-command opcode catalogs.** The `doca_mgmt_raw_cmd`
  in-payload and out-payload are device / firmware specific;
  the skill names the *scope* of a command class and the
  *discipline* the agent applies (pre-state capture, scope
  ladder, rollback path) but it does NOT enumerate opcodes —
  those are vendor / device documentation and looked up via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).
- **A `samples/`, `bindings/`, or `reference/` subtree** of any
  kind. A mock or incomplete artifact in this skill's tree,
  even one labeled "reference", is misleading: users will read
  it as buildable.

## Loading order

1. Read this `SKILL.md` first to confirm the user's question is
   in scope.
2. **For the management-context object model, the raw-command
   scope ladder, the sub-domain surfaces, version
   compatibility, the error taxonomy, observability, and the
   per-artifact safety overlay, see
   [CAPABILITIES.md](CAPABILITIES.md).**
3. **For step-by-step workflows — install, configure, build,
   modify, run, test, debug, use — see [TASKS.md](TASKS.md).**

Both companion files cross-link to each other and to
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
whenever the right answer is "look it up in the public docs or
the installed package layout" rather than "mgmt-specific
guidance".

## Related skills

- [`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md) —
  the bundle-wide hardware-safety meta-policy. Because
  doca-mgmt is the surface through which agents most commonly
  *modify* device state programmatically, the safety overlay
  in `CAPABILITIES.md` cross-links the meta-policy
  load-bearingly and the agent walks the hardware-safety
  ladder whenever a write is recommended.
- [`doca-telemetry`](../doca-telemetry/SKILL.md) and
  [`doca-telemetry-exporter`](../doca-telemetry-exporter/SKILL.md) —
  the observability surfaces for *streaming* telemetry from the
  device. Use those when the user wants live metrics; use
  doca-mgmt when the user wants point-in-time programmatic
  query or modification.
- [`doca-caps`](../../tools/doca-caps/SKILL.md) — the
  read-only CLI for inspecting device capabilities. Use the
  tool when the user wants an interactive snapshot; use
  doca-mgmt when the user wants the same information through
  the C API in a fleet tool.
- [`doca-bench`](../../tools/doca-bench/SKILL.md) — the
  live-performance benchmarking tool. Out of scope for
  doca-mgmt; the skill names the boundary.
- [`doca-pcc`](../doca-pcc/SKILL.md) and
  [`doca-pcc-counters`](../../tools/doca-pcc-counters/SKILL.md) —
  programmable congestion control. doca-mgmt's
  `doca_mgmt_cc_global_status_*` surface is the *global
  enable/disable + protocol selection* control plane that
  layers on the deeper programmable-CC surface owned by
  doca-pcc.
- [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) — the
  routing table for every public DOCA documentation source and
  the on-disk layout of an installed DOCA package.
- [`doca-setup`](../../doca-setup/SKILL.md) — env preparation,
  install verification, and the *I have no install yet* path
  with the public NGC DOCA container.
- [`doca-programming-guide`](../../doca-programming-guide/SKILL.md) —
  general DOCA programming patterns shared by every library:
  the canonical `pkg-config` + meson build pattern, the
  universal modify-a-shipped-sample first-app workflow, the
  cross-library `DOCA_ERROR_*` taxonomy. This skill layers
  management-plane specifics on top.
- [`doca-debug`](../../doca-debug/SKILL.md) — the cross-cutting
  debug ladder. Management-specific debug overlays on top of
  it.
- [`doca-version`](../../doca-version/SKILL.md) — the version
  detection / four-way match rule every per-artifact `##
  Version compatibility` anchor builds on. This skill quotes
  the management-specific overlay only.
- [`doca-structured-tools-contract`](../../doca-structured-tools-contract/SKILL.md) —
  the JSON-schema contracts for the agent-preferred structured
  helpers; the `## Command appendix` in `TASKS.md` defers to
  them before falling back to the manual chain.

