# DOCA Management capabilities, version compatibility, errors, observability, safety

**Where to start:** The pattern overview below names the recurring
management-class patterns. Pick the pattern first, then drill into
the H2 that owns the substance. For the *how* of executing each
pattern, jump to [TASKS.md](TASKS.md).

Read this file when the loader sent you here from
[SKILL.md](SKILL.md). For step-by-step workflows that *use* these
capabilities (install, configure, build, modify, run, test, debug,
use) see [TASKS.md](TASKS.md). For where the underlying public
documentation and installed package paths live, defer to
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) — do
not duplicate URLs or install paths in this file.

## Pattern overview

Every management-class question this skill teaches resolves into
one of SIX patterns. The patterns are CLASSES — they apply across
every fleet-management or device-administration use case, not
just the worked example shown.

| Mgmt pattern | Class shape | Where the substance lives |
| --- | --- | --- |
| 1. Pick `doca-mgmt` vs telemetry vs caps vs bench | Decide *before* writing any code whether the operation is a point-in-time programmatic query / modification (mgmt), a continuous stream of metrics (telemetry), an interactive capability snapshot (caps CLI), or a live performance probe (bench) | [`## Capabilities and modes`](#capabilities-and-modes) surface-selection table |
| 2. Stand up the management context | Open `doca_mgmt_dev_ctx` on a `doca_dev`; optionally open a `doca_mgmt_dev_rep_ctx` on the device's representor (or by PCI address when the representor is unavailable) | [TASKS.md ## configure](TASKS.md#configure) |
| 3. Query a capability before acting | Each sub-domain has a *cap-supported-first* discipline: `doca_mgmt_cap_*_is_supported` for icm-quota and diagnostics-data; `doca_mgmt_device_caps_general_get` for the general device cap surface | [`## Capabilities and modes`](#capabilities-and-modes) cap-supported-first rule + [TASKS.md ## test](TASKS.md#test) |
| 4. Modify device state — with rollback wired before the write | The agent never recommends a `doca_mgmt_*_set` / `_modify` / `_set_limit` / `raw_cmd` write without (a) capturing the pre-state, (b) naming the rollback path, (c) cross-linking the bundle-wide hardware-safety meta-policy | [`## Safety policy`](#safety-policy) overlay + [TASKS.md ## modify](TASKS.md#modify) |
| 5. Issue a raw command with the correct scope | `doca_mgmt_raw_cmd` takes an `enum doca_mgmt_cmd_scope` that classifies blast radius: `CONFIGURATION` (the broadest), `DEBUG_READ_ONLY`, `DEBUG_WRITE`, `DEBUG_WRITE_FULL`. The agent picks the *narrowest* scope that satisfies the request | [`## Capabilities and modes`](#capabilities-and-modes) raw-command scope ladder + [TASKS.md ## use](TASKS.md#use) |
| 6. Interpret a `DOCA_ERROR_*` from a mgmt call | Map the error to a layer (configuration / capability / lifecycle / fwctl-ioctl / firmware-rejection / driver) and route | [`## Error taxonomy`](#error-taxonomy) mgmt overlay + [TASKS.md ## debug](TASKS.md#debug) |

Two cross-cutting rules that apply to *every* pattern above:

- **Capture pre-state before any write.** Every `_set` / `_modify`
  / `_set_limit` / `raw_cmd` with a write scope is preceded by
  the matching `_get` / `_query` against the current state. The
  pre-state is what the rollback path quotes; without it,
  rollback is impossible. This is the
  [`doca-hardware-safety` pre-flight inventory](../../doca-hardware-safety/CAPABILITIES.md#capabilities-and-modes)
  rule applied at the API surface.
- **Capability-supported-first.** Each sub-domain has a
  capability gate: for icm-quota, `doca_mgmt_cap_icm_quota_is_supported`;
  for multi-domain diagnostics data,
  `doca_mgmt_cap_diagnostics_data_multi_domain_is_supported`;
  for the general device caps, `doca_mgmt_device_caps_general_get`
  to read the current cap set. The agent does not recommend a
  `set` against a sub-domain whose `is_supported` returned a
  failure.

## Capabilities and modes

DOCA Management is the **management-plane API surface** for
BlueField / ConnectX device-level operations. The host-side
library exposes a management context per device, an optional
representor context per VF / function-level surface, four
sub-domains (general caps, congestion-control global status,
diagnostics data, ICM quota), and a raw-command pipe with an
explicit scope classifier.

### `doca-mgmt` vs telemetry vs caps vs bench

Four DOCA surfaces appear in fleet-management conversations.
They are NOT interchangeable; pick one before writing any code:

| Surface | Shape | When to pick |
| --- | --- | --- |
| `doca-mgmt` (this skill) | Programmatic point-in-time C API for *querying and modifying* device-level state | The fleet tool needs to read or write device-level state through code; the operation is request/response, not streaming |
| [`doca-telemetry`](../doca-telemetry/SKILL.md) + [`doca-telemetry-exporter`](../doca-telemetry-exporter/SKILL.md) | Streaming telemetry producer + exporter for continuous device metrics | The fleet tool needs continuous time-series metrics (counters, gauges, histograms) rather than a one-off query |
| [`doca-caps`](../../tools/doca-caps/SKILL.md) | Read-only CLI snapshot of device capabilities | The operator wants an interactive look at what a device supports, not a programmatic interface in a fleet agent |
| [`doca-bench`](../../tools/doca-bench/SKILL.md) | Live performance benchmark for DOCA workloads | The user wants to measure throughput / latency of a DOCA workload, not query or modify device state |

**Decision rule for the agent.** If the user's intent is *"my
fleet agent needs to programmatically read or write device-level
state"*, doca-mgmt is the right surface. If the intent contains
the words *continuous*, *streaming*, *Prometheus*, *time-series*,
or *real-time monitoring*, telemetry is the right answer. If the
operator is at an interactive prompt rather than writing code,
`doca-caps` is the answer. If the question is about throughput
or latency, `doca-bench` is the answer.

### The management context object model

Two objects expose the management plane. Verified surface
(`doca_mgmt.h`):

| Object | Calls | Note |
| --- | --- | --- |
| `doca_mgmt_dev_ctx` | `doca_mgmt_dev_ctx_create(dev, &ctx)`, `doca_mgmt_dev_ctx_destroy(ctx)`, `doca_mgmt_dev_ctx_get_doca_dev(ctx)` | Created on a `doca_dev`; the device-wide management context. Required for `doca_mgmt_raw_cmd`, `doca_mgmt_cap_*_is_supported`, and the device-wide icm-quota and diagnostics-data operations |
| `doca_mgmt_dev_rep_ctx` | `doca_mgmt_dev_rep_ctx_create(dev_ctx, rep, &ctx)` (with a `doca_dev_rep`), `doca_mgmt_dev_rep_ctx_create_by_pci_addr(dev_ctx, "0000:3a:00.2", &ctx)` (when a representor is not available for the VF), `doca_mgmt_dev_rep_ctx_destroy`, `doca_mgmt_dev_rep_ctx_get_doca_dev_rep`, `doca_mgmt_dev_rep_ctx_get_mgmt_dev_ctx` | Required for representor-targeted sub-domain calls (general caps `_set` / `_get`, icm-quota for representors). The `_by_pci_addr` form is the documented alternative path when the kernel-side representor is not exposed |

Every public `doca_mgmt_*` symbol is currently tagged
`DOCA_EXPERIMENTAL` in the version map under the `DOCA_3` version
bucket. The agent must surface this whenever it recommends a
symbol — see [`## Version compatibility`](#version-compatibility).

### The raw-command scope ladder

`doca_mgmt_raw_cmd(ctx, command_id, scope, in, in_size, out,
out_size)` is the **firmware-control passthrough** for
operations that don't have a dedicated sub-domain surface. The
`enum doca_mgmt_cmd_scope` field is the load-bearing piece:

| Scope | Class of operation | What the agent does before recommending it |
| --- | --- | --- |
| `DOCA_MGMT_CMD_SCOPE_CONFIGURATION` | Configuration writes that change device-visible state and persist | Capture pre-state; name rollback; cross-link [`doca-hardware-safety` ## Safety policy](../../doca-hardware-safety/CAPABILITIES.md#safety-policy) for the change-application discipline |
| `DOCA_MGMT_CMD_SCOPE_DEBUG_WRITE_FULL` | The widest debug-write surface; can mutate firmware-side state with broad blast radius | All of the above PLUS the maintenance-window + OOB-precondition rule from the meta-policy; refuse if either is missing |
| `DOCA_MGMT_CMD_SCOPE_DEBUG_WRITE` | A debug-write that mutates state with narrower scope than `WRITE_FULL` | Pre-state capture + rollback discipline; the meta-policy applies but `WRITE_FULL`'s harder gates may be relaxed for a single-device read-write |
| `DOCA_MGMT_CMD_SCOPE_DEBUG_READ_ONLY` | Read-only debug query; no state mutation | The agent may recommend it without the rollback discipline, but still inside a maintenance window when the device serves traffic |

**Scope selection rule for the agent.** Pick the *narrowest*
scope that satisfies the user's intent. A read-only query goes
through `DEBUG_READ_ONLY`; a write that has a dedicated
sub-domain surface (caps-general, cc-global-status,
diagnostics-data, icm-quota) should go through that surface, not
through `raw_cmd`. `raw_cmd` is the escape valve for
vendor-documented commands without a dedicated wrapper — not the
primary management surface.

### Sub-domain surfaces

Four sub-domains expose typed wrappers on top of the management
context. Each has its own capability gate and lifecycle:

| Sub-domain | Header | Capability gate | Object model |
| --- | --- | --- | --- |
| General device caps | `doca_mgmt_device_caps_general.h` | Implicit: the `_get` call returns `DOCA_ERROR_NOT_SUPPORTED` on devices without the surface | `doca_mgmt_device_caps_general` handle (`_create` / `_destroy`); apply via `_set(rep_ctx, handle)`; read via `_get(rep_ctx, handle)`; field accessors: `_set_data_direct` / `_get_data_direct`; `_clear` to reset before re-use |
| Congestion control global status | `doca_mgmt_cc_global_status.h` | Implicit: `_get` returns `DOCA_ERROR_NOT_SUPPORTED` on devices without the surface | `doca_mgmt_cc_global_status` handle; protocol enum (`RP` / `NP`); priority and enable accessors; `_set` / `_get` against a representor or device context |
| Diagnostics data | `doca_mgmt_diagnostics_data.h` | `doca_mgmt_cap_diagnostics_data_multi_domain_is_supported(dev_ctx)` for the multi-domain capability; per-field cap probes for narrower features | `doca_mgmt_diagnostics_data` handle; `_modify_for_dev` / `_query_for_dev`; multi-domain accessor |
| ICM quota | `doca_mgmt_icm_quota.h` | `doca_mgmt_cap_icm_quota_is_supported(dev_ctx)` for the surface; `doca_mgmt_cap_icm_quota_get_max_limit(dev_ctx, &max)` for the upper bound | `doca_mgmt_icm_quota` handle; per-device or per-representor; `_set_limit` (in 4 KB granularity; `DOCA_MGMT_ICM_QUOTA_LIMIT_UNLIMITED` is `UINT32_MAX`); `_get_current_allocation`, `_get_max_reached`, `_clear`, `_modify`, `_query` |

### Configuration shape

*Mandatory* before any sub-domain operation: a `doca_mgmt_dev_ctx`
on the target device, and (for representor-targeted operations) a
`doca_mgmt_dev_rep_ctx` on the target VF / function. The
sub-domain handle is a *transient* configuration carrier: create
it, populate fields, apply via `_set`, then destroy. Persistent
state lives on the device, not on the handle. The handle's
`_clear` call is for re-use of the same handle across multiple
fields; the handle's `_destroy` is the final teardown.

## Version compatibility

For the canonical DOCA version-detection chain, the four-way
match rule, NGC container semantics, and the headers-win-over-
docs rule, see
[`doca-version`](../../doca-version/SKILL.md). The body lives
there; this skill does not duplicate it.

**The doca-mgmt-specific overlay** is:

- **Every public `doca_mgmt_*` symbol is currently tagged
  `EXPERIMENTAL`.** The library's public version map (the
  authoritative source for what the shipped `.so` actually
  exports) places every `doca_mgmt_*` symbol under the
  `EXPERIMENTAL` set under the `DOCA_3` version bucket. The
  agent must surface this whenever it recommends a management
  symbol: the API can change shape between DOCA releases, and
  building a long-lived fleet-management tool against this
  surface requires explicit per-release re-verification (see
  [`TASKS.md ## test`](TASKS.md#test)).
- **The runtime authority is the installed headers, not the
  public docs.** Per the cross-cutting rule in
  [`doca-version CAPABILITIES.md ## Observability`](../../doca-version/CAPABILITIES.md#observability),
  the agent confirms a symbol exists by checking the installed
  header set (`doca_mgmt.h`, `doca_mgmt_device_caps_general.h`,
  `doca_mgmt_cc_global_status.h`,
  `doca_mgmt_diagnostics_data.h`,
  `doca_mgmt_icm_quota.h`) before recommending it.
- **`doca-mgmt.pc` plus `doca-common.pc` must match
  `doca_caps --version`** at the four-way-match check (per
  [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility)).
  Management-plane operations depend on the firmware version
  on the device matching the host-side library's expectations;
  a four-way-match failure is the *first* hypothesis when
  `doca_mgmt_raw_cmd` returns `DOCA_ERROR_IO_FAILED` for an
  opcode that the vendor docs say should work.
- **The closest public docs surface is the DOCA SDK index.**
  The agent does not quote a specific docs URL for doca-mgmt
  from agent memory; instead it routes to the
  [DOCA SDK index](https://docs.nvidia.com/doca/sdk/) and
  through
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
  for the up-to-date URL pattern. Where a *DOCA Management*
  page is published, it is the authoritative source for
  current opcodes, scopes, and supported sub-domains; until
  the agent has verified it on `docs.nvidia.com/doca/sdk/`,
  the agent frames version-specific claims as *"check against
  the installed headers and the release notes for your DOCA
  version"*.

## Error taxonomy

The cross-library `DOCA_ERROR_*` taxonomy (what each family
means and which debug layer it routes to) lives in
[`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../../doca-programming-guide/CAPABILITIES.md#error-taxonomy).
The mgmt-specific overlay names the families the agent will see
most often from `doca_mgmt_*` calls and what they specifically
indicate:

| Family | Most common mgmt cause | First action |
| --- | --- | --- |
| `DOCA_ERROR_INVALID_VALUE` | A handle / context pointer is NULL, a PCI address string is malformed for `_create_by_pci_addr`, or a `_set_*` field accessor got an out-of-range value | Re-walk the create/configure step in [`## Capabilities and modes`](#capabilities-and-modes); PCI addresses use the `"Domain:Bus:Device.Function"` HEX format (e.g. `"0000:3a:00.2"`) |
| `DOCA_ERROR_NOT_SUPPORTED` | The sub-domain's `_is_supported` / device-cap check returned a failure, OR the installed DOCA version does not export the requested symbol, OR the device's firmware does not implement the operation | Run the sub-domain's `_is_supported` against the active context; confirm the symbol is in the installed headers per [`## Version compatibility`](#version-compatibility); if both pass, the firmware is the gate |
| `DOCA_ERROR_BAD_CONFIG` | A sub-domain `_set` was called on a handle with required fields unset (e.g. `doca_mgmt_device_caps_general_set` returns this when `data_direct` was not set before `_set`) | Re-walk the sub-domain handle's field accessors; every field the `_set` requires must be populated via the matching `_set_<field>` before the apply call |
| `DOCA_ERROR_IN_USE` | A sub-domain `_set` was called against a representor context that is already initialized for the same surface | Re-read the surface's documented re-init rule; some sub-domains require a destroy-then-recreate cycle on the representor context to change fields |
| `DOCA_ERROR_EMPTY` | A field accessor `_get_*` was called on a handle whose field was never set or whose value the device has not yet returned | The agent surfaces this as a *not-yet-queried* condition, not as an error — call the matching `_get` against the context first |
| `DOCA_ERROR_OPERATING_SYSTEM` | The underlying `fwctl` RPC `ioctl` failed on the host side | The host kernel's `fwctl` interface is not available, or the user does not have permission to issue it. Confirm root privileges; route to [`doca-setup TASKS.md ## debug`](../../doca-setup/TASKS.md#debug) layer 5 (driver) for `fwctl`-side issues |
| `DOCA_ERROR_IO_FAILED` | The `fwctl` RPC reached the firmware but the firmware rejected the command | The firmware version, the device state, or the command opcode is wrong. Re-confirm the four-way version match per [`## Version compatibility`](#version-compatibility); confirm the opcode against the vendor docs; if both check out, route through [`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md) for the device-state investigation |
| `DOCA_ERROR_NO_MEMORY` | The library could not allocate internal state during `_create` | Inspect host-side resource limits; this is rarely an application bug |
| `DOCA_ERROR_DRIVER` | The layer below DOCA (mlx5 driver, firmware path the management plane traverses) reported a failure | Stop. This is not an API-spec problem. Capture `dmesg | tail` and `mlxconfig -d <pcie> q`; route to [`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug) layer 7 |

Quote `doca_error_get_descr()` verbatim — do not paraphrase. The
cross-cutting debug ladder
([`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug))
is the canonical layered diagnosis path that the agent escalates
to once the mgmt-specific cause has been narrowed.

## Observability

doca-mgmt's observability surface is **point-in-time and
request/response**, not streaming. Three primary signals the
agent should reach for:

1. **Pre-state capture before any write.** Every sub-domain
   exposes a `_get` / `_query` symmetric to its `_set` /
   `_modify`; the pre-state is the baseline every rollback
   depends on. Without it, the agent cannot tell what was
   changed if the user wants to back out. This is the bundle-
   wide [pre-flight inventory](../../doca-hardware-safety/CAPABILITIES.md#capabilities-and-modes)
   rule applied at the API surface.
2. **Post-write re-query.** After every `_set` / `_modify` /
   `_set_limit` / `raw_cmd` write, the agent re-runs the
   matching `_get` / `_query` to confirm the device adopted
   the new state. *"The call returned success"* is not
   sufficient evidence; the device's reported state is.
3. **Capability snapshots at start of session.** Every
   sub-domain's `_is_supported` and `_get_max_*` / `_get`
   result is a snapshot of *what the device said was possible*
   at the start. Save it as a baseline; if a later call fails
   with `DOCA_ERROR_NOT_SUPPORTED` the diff against this
   snapshot is the regression — either the firmware was reset
   between sessions or the device's representor was
   reconfigured.

For *continuous* observability (counters streamed over time,
historical metrics), doca-mgmt is the wrong surface — that is
[`doca-telemetry`](../doca-telemetry/SKILL.md). For cross-cutting
observability primitives (`--sdk-log-level`, `DOCA_LOG_LEVEL`)
see
[`doca-debug CAPABILITIES.md ## Observability`](../../doca-debug/CAPABILITIES.md#observability).
For the install-tree observability (logger names, package
layout) defer to
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

## Safety policy

> **Overlay on the bundle-wide hardware-safety meta-policy.** The rules below are this skill's per-artifact overlay on the cross-cutting rules in [`doca-hardware-safety` CAPABILITIES.md ## Safety policy](../../doca-hardware-safety/CAPABILITIES.md#safety-policy) (specifically [### Per-artifact overlay pattern](../../doca-hardware-safety/CAPABILITIES.md#per-artifact-overlay-pattern)). When the two layers disagree, the stricter wins; when either layer says STOP, the agent stops.

doca-mgmt is — alongside `mlxconfig` and BFB reflash — one of
the **primary surfaces through which an agent can modify
device-level state programmatically**. A wrong `raw_cmd`
opcode, a wrong scope, or an unrollbackable write can leave a
device in a state the operator cannot restore. Three policies
follow from that:

1. **Pre-state-capture is a hard precondition for any write.**
   The agent does NOT recommend any `_set` / `_modify` /
   `_set_limit` / `raw_cmd`-with-write-scope call without first
   walking the matching `_get` / `_query` to capture the
   current state. The captured state is what the rollback path
   quotes; the agent does not invent rollback values from
   memory. This is the
   [`doca-hardware-safety` pre-flight inventory](../../doca-hardware-safety/CAPABILITIES.md#capabilities-and-modes)
   rule applied at the API surface.
2. **Narrowest scope wins.** When the user describes an
   operation, the agent picks the *narrowest*
   `doca_mgmt_cmd_scope` (or sub-domain wrapper) that
   satisfies it. Read-only operations go through
   `DEBUG_READ_ONLY`. Write operations that have a dedicated
   wrapper (caps-general, cc-global-status, diagnostics-data,
   icm-quota) go through that wrapper, NOT through `raw_cmd`.
   `raw_cmd` with `DEBUG_WRITE_FULL` is the widest scope and
   requires the most precondition gates from the meta-policy —
   maintenance window, OOB access, replica-first validation,
   documented rollback.
3. **No documented rollback → refuse and escalate.** Per
   [`doca-hardware-safety ## Safety policy`](../../doca-hardware-safety/CAPABILITIES.md#safety-policy),
   any change for which the operator cannot describe a
   rollback path is refused and routed to the operator's
   change-control process. doca-mgmt is the surface where this
   refusal is enforced at the API call: the agent does not
   issue a write whose `_get` counterpart cannot reproduce the
   pre-write state.

For changes that touch hardware state below the doca-mgmt
library itself — `mlxconfig`-class writes that bypass DOCA,
NIC firmware burns, BlueField BFB reflash, host kernel boot
parameter changes — the cross-cutting meta-policy in
[`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md)
applies without modification. The agent walks the
hardware-safety ladder first whenever the recommended change
falls into any of those classes, then the doca-mgmt overlay
above for the API-surface specifics.

## Deferred topic boundaries

This skill scopes itself to the DOCA Management library.
Adjacent topics the agent will get asked but should route
elsewhere:

- **`mlxconfig`-class firmware-stored configuration outside
  the DOCA API** — out of scope. The doca-mgmt surface and
  `mlxconfig` partially overlap (both can affect device
  configuration); for `mlxconfig` proper, route to the
  [`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md)
  meta-policy and to the public Mellanox / NVIDIA firmware
  tools documentation reachable through
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).
- **Live telemetry streaming** — owned by
  [`doca-telemetry`](../doca-telemetry/SKILL.md) and
  [`doca-telemetry-exporter`](../doca-telemetry-exporter/SKILL.md).
- **Read-only capability snapshots from a CLI** — owned by
  [`doca-caps`](../../tools/doca-caps/SKILL.md).
- **Programmable congestion control** — owned by
  [`doca-pcc`](../doca-pcc/SKILL.md) and
  [`doca-pcc-counters`](../../tools/doca-pcc-counters/SKILL.md).
  doca-mgmt's `doca_mgmt_cc_global_status_*` surface is the
  *global enable/disable + protocol selection* control plane
  that *layers on* the deeper programmable-CC surface owned by
  doca-pcc.
- **Cross-library `DOCA_ERROR_*` taxonomy** — owned by
  [`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../../doca-programming-guide/CAPABILITIES.md#error-taxonomy).
  This skill adds the mgmt overlay, not the taxonomy itself.
- **Cross-cutting hardware-state change discipline** — owned
  by [`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md).
  This skill's safety overlay cross-links the meta-policy;
  it does not redefine the rules.
