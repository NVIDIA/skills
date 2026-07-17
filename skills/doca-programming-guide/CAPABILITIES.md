# DOCA programming guide — capabilities, version compatibility, errors, observability, safety

**Where to start:** Read [`## Pattern overview`](#pattern-overview)
first to pick the family; then drill into the matching H2. Every
prescriptive section below stays library-agnostic — overlays for a
specific library live in that library's `CAPABILITIES.md`.

Read this file when the loader sent you here from [SKILL.md](SKILL.md). For the step-by-step workflows that *use* the surface described here, see [TASKS.md](TASKS.md). For env-class equivalents (install profiles, env-class errors, env observability), see [`doca-setup`](../doca-setup/SKILL.md). For where to find official documentation, the on-disk install layout, or the Installation Guide, route through [`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md).

This file describes the **shape every DOCA program shares** — the architecture every library plugs into, the lifecycle every library follows, the error pattern every library returns, the observability surface every library emits into, and the safety rules every program must respect. Library-specific overlays live in the matching library skill.

> Lint note: this file references `/opt/mellanox/doca` and `docs.nvidia.com` paths several times. The lint warns about that for most library skills and asks for cross-links to `doca-public-knowledge-map` instead. For `doca-programming-guide` specifically the references are *intrinsic*: the skill's whole job is to describe how a DOCA program is shaped on disk and on the wire, and the canonical install tree (`/opt/mellanox/doca`) plus the canonical docs tree (`docs.nvidia.com/doca/sdk/`) are the two coordinate systems every prescription is anchored against. The repeated paths are intentional; do not refactor them out.

## Pattern overview

Every program-class question this skill teaches reduces to one of FIVE
patterns. Reach for the pattern first; the H2 below it carries the
substance.

| Pattern | Class shape | Where it lives |
| --- | --- | --- |
| 1. Plug into the right slot | Host / DPU / Switch; library / app / service / tool — orient the program | [`## Capabilities and modes`](#capabilities-and-modes) |
| 2. Follow the universal lifecycle | cfg → create → init → start → use → stop → destroy for every DOCA object | [`## Capabilities and modes`](#capabilities-and-modes) lifecycle subsection |
| 3. Build / link the program | `pkg-config doca-<library>` (C/C++ direct) OR FFI/bindings against the public C ABI | [TASKS.md ## build](TASKS.md#build) |
| 4. Handle `DOCA_ERROR_*` correctly | `doca_error_get_descr()` first; never paper over | [`## Error taxonomy`](#error-taxonomy) |
| 5. Validate before commit, observe before "fixing" | Pre-commit validation, log levels, sample-counter discipline | [`## Safety policy`](#safety-policy) + [`## Observability`](#observability) |

Two principles cut across all five:

- **Library-agnostic shape, library-specific overlay.** This skill
  prescribes the shape; the matching library skill (e.g.
  [`doca-flow`](../libs/doca-flow/SKILL.md)) overlays the
  library-specific specifics (Flow pipe topology, RDMA QP semantics,
  …). Both must be loaded for a complete answer.
- **Programs run *with* DOCA, not *of* DOCA.** Everything below
  assumes the user is calling `doca_*` from their own code. Patches
  to DOCA itself are out of scope and belong in the DOCA contributor
  repo, not this bundle.

## Capabilities and modes

DOCA is structured into a small set of orthogonal pieces. The agent must clarify which piece the user is touching before recommending any code or command.

**Where the code runs.** The same DOCA library can be linked from a program running in any of three locations. Which modes are *available* on the user's host is an env property and lives in [`doca-setup CAPABILITIES.md ## Capabilities and modes`](../doca-setup/CAPABILITIES.md#capabilities-and-modes); which mode the program *selects at init time* is a programming property and is described here.

| Location | What it is | When the user's program is here |
| --- | --- | --- |
| Host | x86 / Arm host with a BlueField as a SmartNIC, or a host with a ConnectX NIC | Most C / C++ first-app development; building, linking, and running on a developer workstation. The program sees physical ports of the BlueField from the host's view, via PF/VF representors. |
| DPU | Inside the BlueField OS itself (Arm). | Workloads that have to run *on* the DPU rather than across PCIe. The program sees local network interfaces and (in switch mode) representors of the host's PF/VF. |
| Switch | DPU-side, with the BlueField in switch (DPU) mode. | Multi-tenant steering, full virtual switch on the DPU. The program takes ownership of representors; library-imposed constraints apply (e.g. Flow forbids `rte_eth_dev_*` configure/start on representors in switch mode — Flow takes them over). |

The library guides describe which mode is meaningful for that library; the agent must clarify which of the three the user's program is in before recommending any port-id, representor, or queue naming.

**What kind of artifact you're building.** DOCA ships four categories of code; the build / run / debug pattern depends on which one the user's project is.

| Category | Where it lives in an installed tree | What the user does with it |
| --- | --- | --- |
| **Library** | `/opt/mellanox/doca/lib/<arch>-linux-gnu/libdoca_<library>.so` | The user's program links against this via `pkg-config doca-<library>`. The first-app workflow ([`TASKS.md ## modify`](TASKS.md#modify)) almost always derives from a sample that consumes one library. |
| **Sample** | `/opt/mellanox/doca/samples/doca_<library>/<sample_name>/` | One library, minimal scaffolding. The canonical *modify-a-shipped-sample* starting point. |
| **Reference application** | `/opt/mellanox/doca/applications/<app>/` | Larger, integrated example combining multiple libraries (e.g. Flow + Telemetry). Useful as a *read-only* worked example; rarely the right starting point for a first app. |
| **Service / tool** | `/opt/mellanox/doca/services/`, `/opt/mellanox/doca/tools/` | Long-running processes (services) or CLIs (tools like `doca_caps`). Not the user's program; the user's program *consumes* their output. |

The agent's rule: when the user says *"first app"*, the answer always starts from a **sample**, not from an application or a service. Applications are for reading; samples are for modifying.

**Build flavor selection.** The `pkg-config` module name selects which library variant your program links against — same library, different runtime properties. The env-side mechanics (where the trace `*.so` lives on disk, how to set `LD_LIBRARY_PATH`) are in [`doca-setup CAPABILITIES.md ## Capabilities and modes`](../doca-setup/CAPABILITIES.md#capabilities-and-modes); the program-side rationale for *which* flavor to pick is here.

| `pkg-config` module | Variant | When to pick it |
| --- | --- | --- |
| `doca-<library>` (e.g. `doca-flow`) | Release. Optimized, no extra sanitation. | Production and benchmarking. |
| `doca-<library>-trace` (e.g. `doca-flow-trace`) | Trace build. Enables additional input sanitation, slower, much louder logs. | **First-app development.** First-time-user errors are caught earlier and surfaced with explicit reasons; switch to release only after the staged run succeeds and you're moving to a performance reading. |

**The universal lifecycle.** Every DOCA library object the user creates follows the same six-phase lifecycle. Library skills name which specific API calls correspond to each phase.

| Phase | What it means | What goes wrong if you skip it |
| --- | --- | --- |
| `cfg-create` | Allocate a configuration object (`doca_<library>_cfg_*`). | NULL deref on the cfg pointer. |
| `cfg-set-*` | Populate the configuration with the user's settings (mode, capabilities, callbacks). | The library defaults what you didn't set — usually not what you meant. |
| `init` / `create` | Materialize the actual library object from the populated cfg. | Calls that need the object return `DOCA_ERROR_BAD_STATE`. |
| `start` | Move the object into a state where it can be used. | `use`-phase calls return `DOCA_ERROR_BAD_STATE`. |
| `use` | The library-specific work — pipe entries, queue ops, channel sends, etc. | Library-specific symptoms (no traffic, dropped packets, hung sends). |
| `stop` + `destroy` | Reverse-order tear-down; release in the inverse order of acquisition. | Resource leaks; on next run, capability discovery may report exhausted budgets. |

The agent's rule: never recommend a `use`-phase call before confirming the object is in `started` state, and never recommend re-creating an object that hasn't been `destroyed`.

## Version compatibility

For the canonical DOCA version-detection chain, the four-way match rule, NGC container semantics, the headers-win-over-docs rule, and the routing to the DOCA Compatibility Policy, see [`doca-version`](../doca-version/SKILL.md). The body lives there; this skill does not duplicate it.

**The program-side overlay** layers on top of the verified version:

- **Quote the version observed, never "latest" and never agent memory.** API names, sample filenames, and capability availability all depend on the user's installed version. Per the safety policy in [`doca-version CAPABILITIES.md ## Safety policy`](../doca-version/CAPABILITIES.md#safety-policy), a recommendation that quotes a version-pinned URL or invents a feature based on agent memory is the bundle's primary hallucination failure mode.
- **For non-C consumers (FFI / bindings).** The binding tooling generates declarations from the headers at build time. Re-generate after any DOCA upgrade — bindings pinned to a stale header will compile but call into the new `*.so` with mismatched signatures, returning `DOCA_ERROR_INVALID_VALUE` or, worse, silent corruption.
- **Use the per-library `doca_<library>_cap_*` query family at runtime** as the authoritative answer for *"is this capability supported on this device + this install"*. The version-matrix is the *promise*; the cap query is the *reality*. When they disagree, the cap query wins — see [`doca-version TASKS.md ## test`](../doca-version/TASKS.md#test) step 3.

## Error taxonomy

Every DOCA library returns a `doca_error_t` from any call that can fail. The error family is **library-agnostic at the type level**; library-specific overlays (e.g. *which* `DOCA_ERROR_*` Flow returns from which call, with what recovery) live in the matching library skill.

The general taxonomy the agent must recognize:

| `doca_error_t` value | Meaning | Programming-side response |
| --- | --- | --- |
| `DOCA_SUCCESS` | The call succeeded. | Continue. |
| `DOCA_ERROR_INVALID_VALUE` | One of the arguments is malformed (NULL where required, out-of-range, mismatched type). | The bug is in the *program*, not the library. Check the call's documentation in the library guide; correct the argument; re-validate. |
| `DOCA_ERROR_BAD_STATE` | The object is not in a state where this call is meaningful (e.g. `start` before `init`, `use` before `start`). | Lifecycle order is wrong. Walk the call sequence against the universal lifecycle in [`## Capabilities and modes`](#capabilities-and-modes); fix the order. |
| `DOCA_ERROR_NOT_SUPPORTED` | The hardware / firmware / mode does not implement this feature on this host. | Re-run capability discovery (`doca_caps`, the library's own capability-query API). The agent must *not* attempt code workarounds; either change the program's intent, switch hardware/mode, or escalate. |
| `DOCA_ERROR_NO_MEMORY` | Allocation failed. | Out-of-memory at the system level (rare during first-app development) or out of a library-managed pool. The library skill names the relevant pool / budget. |
| `DOCA_ERROR_TIME_OUT` | An async operation did not complete in time. | Increase the timeout if intended; investigate why the underlying operation is slow if not. |
| `DOCA_ERROR_INITIALIZATION` | Library / object initialization itself failed. | Almost always a missing prerequisite (driver, capability, peer). Walk through the library's `## configure` workflow; do not retry blindly. |
| `DOCA_ERROR_DRIVER` | The underlying driver / firmware reported failure. | Capture state and route to env-class debug ([`doca-setup ## debug`](../doca-setup/TASKS.md#debug)) — the layer below DOCA is the suspect, not the program. |
| `DOCA_ERROR_UNKNOWN` | Internal error not mappable to the taxonomy. | Capture state and escalate; do not paper over with a retry loop. |

For *any* returned `doca_error_t` other than `DOCA_SUCCESS`, the agent's first step is to call `doca_error_get_descr(err)` and quote the actual descriptor string — never paraphrase from memory. The descriptor is the library's own statement of what went wrong; library-specific guides can refine it but cannot contradict it.

The env-class error surface (missing `*.pc`, hugepages not mounted, representor not visible) is **separate** and lives in [`doca-setup CAPABILITIES.md ## Error taxonomy`](../doca-setup/CAPABILITIES.md#error-taxonomy). The agent's debug order is always env first ([`doca-setup ## debug`](../doca-setup/TASKS.md#debug) layers 1–4), then program ([`TASKS.md ## debug`](TASKS.md#debug) layers 2–5).

## Observability

Every DOCA program has access to the same shared observability surface; library-specific surfaces (Flow counters, RDMA queue stats, …) extend it.

**Logging.** DOCA libraries log via the `doca_log_register_*` API and emit to stderr by default. The CLI flag `--sdk-log-level <level>` (or the equivalent programmatic call `doca_log_backend_set_sdk_level()`) controls verbosity:

- `30` (`DOCA_LOG_LEVEL_INFO`) — quiet, production default.
- `50` (`DOCA_LOG_LEVEL_DEBUG`) — useful debug detail.
- `70` (`DOCA_LOG_LEVEL_TRACE`) — every call traced; **the right setting for any first run of any new code path**.

The trace build flavor enables additional sanitation that surfaces as log lines at `DEBUG` / `TRACE` level the release flavor would not emit.

**Capability snapshots.** `doca_caps` (CLI) and the library's own capability-query API (programmatic) are the source of truth for *what this program can attempt on this host*. Capture both at the start of any debugging session; a stale capability picture is a leading cause of *"it worked yesterday"*.

**Library-specific surfaces.** Pipe counters, queue statistics, channel send / recv counters, and tracing dumps are owned by each library. The library skill names the call (e.g. `doca_flow_pipe_query`) and the right cadence to query it; this skill does not duplicate that.

**Where logs go in long-running services.** DOCA services (DTS, telemetry exporter, etc.) configure their own log paths. Check `/var/log/doca*` or the service's own README before assuming output is missing; do not assume stderr.

## Safety policy

DOCA programs share state with the system, the device, and (in production) other tenants on the same fabric. The cross-library safety rules every library skill inherits:

1. **Validate before commit.** Every library that programs hardware exposes a *validate* call separate from the *commit / start / program* call. Use validate first; never enter a commit path with an un-validated spec. Library skills extend this with library-specific *what to validate* (Flow validates pipe specs; RDMA validates queue-pair attributes; Comch validates channel cfg).
2. **Stage first, widen later.** Run any new program against the smallest possible scope first — one representor, one queue, one channel. Read the output. Only after the staged run succeeds may the user widen the scope. The library's *unit of damage* (a representor for Flow, a QP for RDMA, a channel for Comch) defines what "smallest scope" means; the library skill names it.
3. **Trace flavor for development; release flavor for performance.** The `-trace` build variant adds runtime sanitation that is invaluable while learning the API surface but introduces measurable overhead. Make the switch to release a deliberate, documented step before any performance reading is taken seriously.
4. **Quote the version on every API claim.** Capability availability changes between releases. Statements of the form *"this works in DOCA 3.3"* must come from the user's *installed* `pkg-config --modversion doca-<library>`, not from agent memory or a public docs page that may describe a different release.
5. **Respect the env-side safety policy.** Hugepage changes, eswitch mode changes, `mlxconfig` resets, BFB updates — all are global system state. The env-side rules in [`doca-setup CAPABILITIES.md ## Safety policy`](../doca-setup/CAPABILITIES.md#safety-policy) bind every program; no library override changes them.
6. **Never auto-modify the install tree.** `/opt/mellanox/doca/lib*/` is the release. Programs link against it; programs do not rewrite it. Build flavor changes go via `LD_LIBRARY_PATH` or the `-trace` `pkg-config` module, never by editing `lib/`.
7. **Trust the C ABI as the source of truth, even from non-C languages.** Wrappers, FFI bindings, and codegen toolchains all eventually call the same `libdoca_<library>.so` the C samples link against. A binding-side decision that contradicts the public C ABI's lifecycle, error handling, or capability gate is wrong even if the binding compiles cleanly; the symptoms surface at runtime as `DOCA_ERROR_BAD_STATE` or `DOCA_ERROR_INVALID_VALUE` from the underlying C call.
