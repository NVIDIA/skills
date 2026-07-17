# DOCA Common capabilities, subsystems, version overlay, errors, observability, safety

**Where to start:** This file is the **substance of the doca-common
foundation**. Five subsystems live here as their own H2 anchors —
`## log`, `## buf`, `## ctx`, `## dev`, `## progress engine` — each
corresponds to a public-header family under
$(pkg-config --variable=includedir doca-common) and to a verb-side
workflow in [TASKS.md](TASKS.md). The opening `## Capabilities and
modes` is the routing layer that names which subsystem owns which
class of question; the closing `## Version compatibility`,
`## Error taxonomy`, `## Observability`, and `## Safety policy` are
the cross-subsystem overlays.

Read this file when the loader sent you here from
[SKILL.md](SKILL.md). For step-by-step workflows that *use* these
capabilities (configure / build / modify / run / test / debug / use /
log) see [TASKS.md](TASKS.md). For where the underlying public
documentation and installed package paths live, defer to
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) —
do not duplicate URLs or install paths in this file.

## Pattern overview

Every doca-common question this skill teaches resolves into one of
SIX patterns. The patterns are CLASSES — they apply across every
DOCA app and every higher-level library, not just the worked
example shown.

| Common pattern | Class shape | Where the substance lives |
| --- | --- | --- |
| 1. Establish the universal foundation | Before opening *any* per-library context, walk `doca_devinfo_create_list` → `doca_dev_open` → `doca_pe_create` → per-library `doca_*_create` → `doca_pe_connect_ctx` → `doca_ctx_start` → driving `doca_pe_progress` | [`## dev`](#dev), [`## ctx`](#ctx), [`## progress engine`](#progress-engine) + [TASKS.md ## configure](TASKS.md#configure) |
| 2. Gate on device capability before quoting any feature | Run `doca_*_cap_*` against the active `doca_devinfo` BEFORE assuming a feature is available; the runtime cap-query is the authority, the docs are the promise | [`## dev`](#dev) cap-query rule + [TASKS.md ## use](TASKS.md#use) |
| 3. Wire zero-copy buffers across libraries | `doca_mmap` registers user memory with one or more `doca_dev`s; `doca_buf_inventory` carves the registered range into `doca_buf` handles that flow across DOCA libraries (eth → dma → rdma → …) on the same memory | [`## buf`](#buf) + [TASKS.md ## configure](TASKS.md#configure), [TASKS.md ## use](TASKS.md#use) |
| 4. Drive the progress engine on every loop iteration | `doca_pe_progress` is the universal task-completion drain. Submitting tasks without driving the PE is the canonical "my program does nothing" failure mode for every DOCA Core context | [`## progress engine`](#progress-engine) + [TASKS.md ## run](TASKS.md#run) |
| 5. Get the two-tier log model right | SDK tier controls DOCA library internals (`--sdk-log-level` / `DOCA_LOG_LEVEL_SDK`); app tier controls user emissions (app-side registry). Independent setters, independent defaults; confusing them is the number-one first-app debug failure | [`## log`](#log) + [TASKS.md ## log](TASKS.md#log) |
| 6. Diagnose a `DOCA_ERROR_*` from any Common call | Map family (`BAD_STATE` / `INVALID_VALUE` / `NOT_SUPPORTED` / `NOT_PERMITTED` / `NO_MEMORY` / `DRIVER`) to layer (lifecycle / configuration / capability / permission / resource / driver) | [`## Error taxonomy`](#error-taxonomy) + [TASKS.md ## debug](TASKS.md#debug) |

Two cross-cutting rules that apply to *every* pattern above:

- **Discover the version-installed surface, do not assume.** Every
  pattern above gates on `pkg-config --modversion doca-common` and
  on the runtime cap-query against the active `doca_devinfo`. Quoting
  a Common API symbol or a device capability without checking is the
  most common hallucination failure mode at the foundation layer.
- **Lifecycle is universal across libraries.** `cfg-create →
  cfg-set-* → init → start → use → stop → destroy` is the same shape
  for every DOCA Core context regardless of which library it belongs
  to. Skipping a stage or calling an operation outside its allowed
  window returns `DOCA_ERROR_BAD_STATE`; the right response is to
  re-walk the lifecycle, not to retry.

## Capabilities and modes

DOCA Common is **the foundation library every other DOCA library
depends on**. Per the [DOCA Core Programming Guide](https://docs.nvidia.com/doca/sdk/DOCA+Programming+Guide.md),
the Core surface ships the universal primitives every DOCA
application touches before specializing into any higher-level
library. The `pkg-config` module name is `doca-common`; it is
always present on any healthy DOCA install.

| Subsystem | Header family | Owning H2 in this file | Owning verb in TASKS.md |
| --- | --- | --- | --- |
| Logging (two-tier SDK / app level model, source registration, `DOCA_LOG_*` emission, backends) | `doca_log.h` (under the installed infrastructure include tree) | [`## log`](#log) | [`## log`](TASKS.md#log) |
| Zero-copy data buffers (memory-mapping, inventory, per-buffer handles, buf pools, buf arrays) | `doca_mmap.h`, `doca_buf.h`, `doca_buf_inventory.h`, `doca_buf_pool.h`, `doca_buf_array.h` | [`## buf`](#buf) | [`## configure`](TASKS.md#configure), [`## use`](TASKS.md#use) |
| Context lifecycle (the universal `cfg-create → start → use → stop → destroy` shape every `doca_*_create` returns) | `doca_ctx.h` | [`## ctx`](#ctx) | [`## configure`](TASKS.md#configure), [`## run`](TASKS.md#run) |
| Device discovery (devinfo enumeration, opening, representor discovery, capability queries) | `doca_dev.h` | [`## dev`](#dev) | [`## configure`](TASKS.md#configure), [`## use`](TASKS.md#use) |
| Progress engine (universal task-completion drain) | `doca_pe.h` | [`## progress engine`](#progress-engine) | [`## run`](TASKS.md#run) |

Adjacent Common surfaces (sync events, graphs, clocks, UAR / umem,
buf arrays, mmap advise) are mentioned where they touch one of the
five primary subsystems. The agent's rule: when the user's question
lands on one of those adjacent surfaces, route to the *header on the
user's install* and to the public [DOCA Core Programming Guide](https://docs.nvidia.com/doca/sdk/DOCA+Programming+Guide.md)
for the per-surface detail rather than reciting it here.

## log

DOCA Log is **DOCA's standardized logging primitive**. Every DOCA
library (and every shipped DOCA sample and reference application)
emits its own log lines through DOCA Log, so log output from a DOCA
app interoperates with DOCA's centralized level controls and
downstream consumers (DOCA Log Service, DOCA Telemetry Service).

**The two-tier log-level model — the single most load-bearing
rule.** DOCA Log carries two *independent* verbosity tiers, each
with its own setter and its own default. The agent MUST walk this
table BEFORE writing any code:

| Tier | What it controls | How to set at runtime | Typical default | Symptom when confused |
| --- | --- | --- | --- | --- |
| **SDK log level** | Verbosity of DOCA *library internals* — log lines emitted from inside the DOCA libraries themselves as they call into hardware, manage state, validate inputs, … | `--sdk-log-level <level>` CLI flag on any DOCA sample / reference app; `DOCA_LOG_LEVEL_SDK` env var; programmatic SDK setter equivalent | `WARNING` | User sets `--sdk-log-level DEBUG` to see their own DEBUG lines and is flooded with internal DOCA spam instead, or *"my own DEBUG lines do not appear"* despite the flag |
| **App log level** | Verbosity of *user code* — log lines the user emits via `DOCA_LOG_*` macros from their own source files | App-side global lower-limit setter (`doca_log_level_set_global_lower_limit`); per-source via `doca_logger_set_level` once the source ID is in hand | `INFO` (per release; quote the observed default) | User sets app level to `INFO`, then wonders why their `DOCA_LOG_DBG`-shaped lines never print; or user expects `--sdk-log-level` to reach their own lines and it does not |

The agent's diagnostic rule: when the user reports *"my log levels
do nothing"*, the FIRST hypothesis is *tier confusion* (setting one
tier when they meant the other) — NOT a DOCA Log bug, NOT a hardware
bug, NOT a sample issue. Walk the table above before any code-side
investigation.

**The objects DOCA Log exposes.** The public surface is small and
stable. The agent must not invent additional objects; the
load-bearing call shapes are the rows below.

| Object | What it is | Lifecycle call | Used by |
| --- | --- | --- | --- |
| Log source | Per-source identity (typically one per `.c` file or one per component) under which `DOCA_LOG_*` lines are emitted | `doca_log_register_source(<name>, &source_id)` — called once at component init time, BEFORE any `DOCA_LOG_*` from that source. Mirrored by `doca_log_unregister_source` at teardown | The user's app, once per logical source |
| `DOCA_LOG_*` macros | User-facing emission macros — INFO, DEBUG, ERR (ERROR), CRIT (CRITICAL), WARN (WARNING) — each takes a source ID + a printf-style format and varargs | n/a (compile-time macro expansion) | The user's app, every emission |
| `doca_log_level_*` functions | Runtime-adjustment surface for the app-tier level (global lower / upper limit, per-source level) | n/a (call any time after the source is registered) | The user's app, on a level-change event (signal, RPC, config reload) |
| Backends / sinks | Pluggable output destinations — `doca_log_backend_create_standard` (the default `stderr` sink) plus `_with_file`, `_with_fd`, `_with_buf`, `_with_syslog`, each with an `_sdk` variant for the SDK tier | Created at app init time before the first `DOCA_LOG_*` emission; each backend has its own per-tier upper / lower level setter | The user's app, on log routing setup |

**Custom-sink writes inherit the sink's own permission envelope.**
When the user installs a custom backend (file, fd, buf, syslog)
instead of relying on the default `stderr`, the writes happen under
that sink's own permission rules — file permissions on the target
path, fd permissions inherited from the caller, syslog ACLs on the
destination. The DOCA Log API itself adds no permission elevation;
the user must ensure the process has whatever rights the sink needs.
A sink whose creation succeeds but whose writes fail is the worst
kind of *"my log lines disappear"* symptom — see
[`## Safety policy`](#safety-policy).

**Path-selection — DOCA Log vs language-native logging.** DOCA Log
is not the only logging framework available; the agent must walk
this rule before recommending DOCA Log setup. In a DOCA-app context
(any codebase that calls `doca_*`), DOCA Log is the right primitive
because its output interoperates with `--sdk-log-level` and with the
downstream DOCA Log Service / DOCA Telemetry Service consumers. In a
non-DOCA codebase with no DOCA-side context, language-native logging
(`printf`, `fprintf(stderr, …)`, `syslog`, `spdlog`, Python
`logging`, Rust `tracing`, …) is fine. The anti-pattern alert:
silently swapping a sample's `DOCA_LOG_*` calls for `printf` *for
simplicity* is a wrong answer — it severs interop with
`--sdk-log-level` and the downstream consumers, and makes the
modified sample look unlike every other DOCA artifact on the host.

For the verb-side workflow (configure / build / modify / run / test /
debug a DOCA Log integration), see [`TASKS.md ## log`](TASKS.md#log).

## buf

DOCA's zero-copy data plane rests on three objects layered together:

| Object | What it is | Lifecycle call |
| --- | --- | --- |
| `doca_mmap` | A user-space memory range registered with one or more `doca_dev`s for device-side access. Holds memory permissions and exports for cross-process / cross-device sharing | `doca_mmap_create` → `doca_mmap_set_memrange` (or `_set_dmabuf_memrange`) → `doca_mmap_add_dev` (one or more devices) → `doca_mmap_set_permissions` → `doca_mmap_start` → use → `doca_mmap_stop` → `doca_mmap_destroy` |
| `doca_buf_inventory` | A pool of `doca_buf` handles over one or more started `doca_mmap` ranges. Hands out `doca_buf` handles via `doca_buf_inventory_buf_get_by_args` | `doca_buf_inventory_create` → `doca_buf_inventory_start` → use → `doca_buf_inventory_stop` → `doca_buf_inventory_destroy` |
| `doca_buf` | A per-operation handle into a registered memory range. Carries head pointer, length, refcount, chain-list linkage. The unit that flows between DOCA libraries on the same memory | Allocated by the inventory; reference-counted via `doca_buf_inc_refcount` / `doca_buf_dec_refcount`; never created standalone |

The `doca_buf_pool` surface (allocate-by-pool variant) and the
`doca_buf_array` surface (DPA / GPU-targetable buffer arrays) are
adjacent flavors of the same model. The buf-chain primitives
(`doca_buf_chain_list`, `doca_buf_get_next_in_list`,
`doca_buf_unchain_list`) let the user assemble scatter-gather lists
without copying.

**The same buf flows between libraries.** A `doca_buf` allocated by
inventory A can be referenced by `doca_dma` on the source side and
`doca_rdma` on the destination side — both contexts see the same
underlying registered memory through the same handle. That is the
load-bearing property zero-copy buys: a single `doca_mmap` registers
once, the inventory carves it into N `doca_buf`s, and each `doca_buf`
crosses library boundaries without a second registration or a
copy.

**Permissions and exports.** `doca_mmap_set_permissions` gates
`PCI_READ` vs `PCI_READ_WRITE` vs `LOCAL_READ_WRITE`; the matching
cap-query family (e.g. `doca_mmap_cap_is_export_pci_supported`) tells
the agent whether the device supports the export class the user
wants. Exports flow through `doca_mmap_export_pci` / `_export_rdma`,
and the importing side uses `doca_mmap_create_from_export` to rebuild
the handle.

**Lifecycle order is load-bearing.** Destroying an `mmap` while a
`buf` from its inventory is still in flight returns
`DOCA_ERROR_BAD_STATE`; destroying a `doca_dev` while an `mmap` is
still registered on it returns the same. The teardown order is the
reverse of bring-up: outstanding tasks drain → contexts stop →
inventories stop → mmaps stop → devices close → mmaps / inventories /
devices destroy. See [TASKS.md ## use](TASKS.md#use) for the worked
buffer-lifecycle walk.

For the verb-side workflow (configuring buffers, modifying a sample's
buffer wiring), see [TASKS.md ## configure](TASKS.md#configure) and
[TASKS.md ## use](TASKS.md#use).

## ctx

`doca_ctx` is **the universal context handle every per-library
`doca_*_create` returns**. A `doca_flow_port`, a `doca_rdma`, a
`doca_eth_txq`, a `doca_dma`, a `doca_comch_server` — every one
carries an underlying `doca_ctx` that the user manipulates through
the universal lifecycle calls.

The lifecycle every DOCA Core context follows:

1. **Create.** Library-specific `doca_<library>_create(...)` returns
   a handle from which `doca_<library>_as_ctx()` (where exposed) or
   the library's own ctx accessor surfaces the underlying
   `doca_ctx *`. Some libraries expose the `doca_ctx *` directly; the
   shape is the same.
2. **Configure.** Library-specific `*_set_*` setters apply BEFORE
   `doca_ctx_start`. Once started, most setters return
   `DOCA_ERROR_BAD_STATE`.
3. **Connect to a progress engine.** `doca_pe_connect_ctx(pe, ctx)`
   registers the context with a PE so its task completions surface
   through `doca_pe_progress`. Skipping this step is the canonical
   "tasks submit but nothing completes" failure mode (see
   [`## progress engine`](#progress-engine)).
4. **Start.** `doca_ctx_start(ctx)` transitions to RUNNING. The
   library-specific *use* calls (`doca_<library>_task_*_submit`,
   …) only make sense in the RUNNING state.
5. **Use.** Submit tasks, drain completions through the PE.
6. **Stop.** `doca_ctx_stop(ctx)` transitions toward IDLE. Outstanding
   tasks must be flushed first via `doca_ctx_flush_tasks` or by
   draining the PE until `doca_ctx_get_num_inflight_tasks` reads 0.
7. **Destroy.** The library-specific `doca_<library>_destroy(...)`
   releases the context. Destroying before stop returns
   `DOCA_ERROR_BAD_STATE`.

**State-change callbacks.** `doca_ctx_set_state_changed_cb` lets the
user observe lifecycle transitions; `doca_ctx_get_state` queries the
current state synchronously. Both are advisory — the lifecycle order
is the contract.

**User data.** `doca_ctx_set_user_data` / `_get_user_data` attaches an
opaque cookie to the context so completion callbacks can find their
owning state without a global map.

**Datapath placement.** `doca_ctx_set_datapath_on_gpu` and
`_set_datapath_on_dpa` are experimental setters that move a context's
datapath to a GPU or DPA target instead of the host CPU. The cap
side (`doca_ctx_cap_get_num_completion_vectors`,
`doca_ctx_set_completion_vector`) lets the user pick a completion
vector for multi-queue workloads.

For the verb-side walk of the lifecycle, see
[TASKS.md ## configure](TASKS.md#configure) and
[TASKS.md ## run](TASKS.md#run).

## dev

`doca_dev` is the **device handle every per-library context
consumes**. Discovery is two-step: enumerate available devices
through `doca_devinfo_create_list`, then open the one the user wants
with `doca_dev_open`.

The discovery model:

| Object | What it is | Lifecycle call |
| --- | --- | --- |
| `doca_devinfo` | A read-only descriptor for a candidate device. Carries PCIe address, IB device name, interface name, IPv4 / IPv6 / MAC address, LID, and the matching cap-query family | `doca_devinfo_create_list(&devinfos, &num)` → iterate → `doca_devinfo_destroy_list(devinfos)` |
| `doca_dev` | An opened device handle. The unit per-library contexts consume | `doca_dev_open(devinfo, &dev)` → use → `doca_dev_close(dev)` |
| `doca_devinfo_rep` | A read-only descriptor for a *representor* — a host-side view of a virtual function or sub-function exposed by an embedded function (typically the DPU's eswitch) | `doca_devinfo_rep_create_list(dev, filter, &reps, &num)` → iterate → `doca_devinfo_rep_destroy_list(reps)` |
| `doca_dev_rep` | An opened representor handle | `doca_dev_rep_open(rep_info, &rep)` → use → `doca_dev_rep_close(rep)` |

**Capability discovery — the runtime-authority rule.** The
`doca_devinfo` carries a family of capability queries — both the
generic ones (`doca_devinfo_cap_is_hotplug_manager_supported`,
`doca_devinfo_cap_is_accelerate_resource_reclaim_supported`,
`doca_devinfo_cap_is_notification_moderation_supported`, …) and the
library-specific ones every higher-level skill cross-links to here
(`doca_flow_cap_*`, `doca_rdma_cap_*`, `doca_eth_cap_*`,
`doca_dma_cap_*`, …). The agent's rule: **the cap-query against the
active `doca_devinfo` is the runtime authority for what the device
+ firmware + DOCA version actually supports.** The public docs are
the *promise*; the cap-query is the *reality*. Quoting a feature
without running the cap-query is the most common foundation-layer
hallucination.

**Representors are how the host sees DPU-side functions.** On a
BlueField host with the DPU in switch mode, the host program opens
the host-side PF (`doca_dev_open(devinfo)`) and then enumerates
representors of the DPU-side VFs / SFs through
`doca_devinfo_rep_create_list`. The representor handle is what the
host program passes to `doca_flow` actions, `doca_eth` queues, etc.,
when steering traffic to a specific DPU-side function. The
representor-side capability queries
(`doca_devinfo_rep_cap_is_filter_*_supported`,
`doca_devinfo_rep_get_is_uplink_representor`,
`doca_devinfo_rep_get_is_host_representor`) tell the agent what kind
of representor it is and what filters can apply.

**Device-class properties.** `doca_devinfo_get_pci_addr_str`,
`_get_ibdev_name`, `_get_iface_name`, `_get_mac_addr`, `_get_ipv4_addr`,
`_get_ipv6_addr`, `_get_lid`, `_get_active_mtu` (experimental),
`_get_active_speed` (experimental), `_get_pci_func_type`
(experimental) are the load-bearing identifiers when the user is
disambiguating among multiple devices on the host. The agent should
quote these from the user's running `doca_devinfo_create_list`,
never from memory.

For the verb-side device-discovery walk, see
[TASKS.md ## configure](TASKS.md#configure) and
[TASKS.md ## use](TASKS.md#use).

## progress engine

`doca_pe` is the **universal task-completion drain** every DOCA Core
context relies on. Without it, tasks submitted into a context look
like they vanish — they completed at the hardware, but the
completion event has nowhere to surface. Driving the PE on every loop
iteration is the canonical run-loop shape for every DOCA library.

The PE model:

| Object | What it is | Lifecycle call |
| --- | --- | --- |
| `doca_pe` | A progress engine, owned by exactly one thread that calls `doca_pe_progress` on it. Tasks submitted into any connected `doca_ctx` surface their completions on the engine that owns the context | `doca_pe_create(&pe)` → connect contexts with `doca_pe_connect_ctx(pe, ctx)` → drive with `doca_pe_progress(pe)` in the run loop → `doca_pe_destroy(pe)` |
| Per-context connection | `doca_pe_connect_ctx` registers a `doca_ctx` with a PE so its task callbacks fire on `doca_pe_progress` | Called BEFORE `doca_ctx_start`; reversed by destroying the context |
| Notification handle | The PE exposes an OS-level fd (`doca_pe_get_notification_handle`) so the run loop can `epoll` / `select` it instead of polling. `doca_pe_request_notification` arms the fd for the next event; `doca_pe_clear_notification` re-arms after handling | Used by the user's run loop on a per-iteration cadence |

**The single rule.** Every iteration of the user's main loop calls
`doca_pe_progress(pe)`. The function returns the number of completions
serviced this iteration; a return of zero is fine and just means
nothing was ready. **Skipping the PE drive is the universal "my
program does nothing" symptom.** It applies regardless of which
library the user's primary work is in (Flow / RDMA / Eth / DMA /
Comch / …).

**One PE per thread is the default shape.** Sharing a PE across
threads requires the user's own synchronization; the simpler shape
is one PE per worker thread, with all contexts that need to
complete on that worker connected to that worker's PE. Multi-PE
designs are valid (e.g., one PE per CPU core in a many-core data
plane) but the user picks the shape; the agent does not invent
multi-PE topologies.

**Inflight bookkeeping.** `doca_pe_get_num_inflight_tasks(pe)` and
the per-ctx `doca_ctx_get_num_inflight_tasks(ctx)` are the way to
drain on shutdown — call `doca_ctx_flush_tasks` (or stop the ctx)
and then loop `doca_pe_progress(pe)` until the count is zero before
destroying.

**Notification mode.** `doca_pe_set_event_mode` (experimental) +
`doca_pe_get_notification_handle` + `doca_pe_request_notification`
let the run loop sleep on an fd instead of busy-polling. The classic
poll loop is `while (running) doca_pe_progress(pe);` — fine for a
data-plane thread, wasteful for a control-plane thread, which should
prefer the notification handle.

For the verb-side walk (configuring the PE, running the drive loop,
debugging missing completions), see [TASKS.md ## run](TASKS.md#run)
and [TASKS.md ## debug](TASKS.md#debug).

## Version compatibility

For the canonical DOCA version-detection chain, the four-way match
rule, NGC container semantics, and the headers-win-over-docs rule,
see [`doca-version`](../../doca-version/SKILL.md). The body lives
there; this skill does not duplicate it.

**The doca-common-specific overlay** is:

- **`pkg-config --modversion doca-common` is the universal DOCA
  install anchor.** It is published by every healthy DOCA install
  regardless of which higher-level libraries are also installed.
  Per [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility),
  it joins the four-way match against `doca_caps --version`,
  `cat /opt/mellanox/doca/applications/VERSION`, and the BFB anchor
  (where applicable). When the `pkg-config` query for doca-common
  fails, the install is broken — that is a
  [`doca-setup`](../../doca-setup/SKILL.md) concern, not a Common
  concern.
- **The set of `doca_*` symbols available is observable from the
  Common header set.** Per the headers-win-over-docs rule in
  [`doca-version`](../../doca-version/SKILL.md), the headers under
  $(pkg-config --variable=includedir doca-common) (`doca_buf.h`,
  `doca_ctx.h`, `doca_dev.h`, `doca_pe.h`, `doca_log.h`,
  `doca_mmap.h`, …) are the authoritative truth for what the built
  library exposes. When the user reports an `undefined reference`
  for a Common symbol, the first hypothesis is wrong-version
  documentation — confirm the installed version, then verify the
  symbol exists in the installed headers, then read the matching
  release notes.
- **Common-side experimental markers.** The Common ABI splits into
  a stable set (the `DOCA_2` global block in the library's
  version.map — `doca_buf_*`, `doca_ctx_*`, `doca_dev_*`,
  `doca_pe_*`, `doca_mmap_*`, `doca_devinfo_*`) and an experimental
  set (`doca_log_*`, `doca_graph_*`, `doca_sync_event_*`,
  `doca_clock_*`, `doca_uar_*`, `doca_umem_*`, `doca_buf_arr_*`,
  `doca_mmap_advise_*`, several newer `doca_ctx_*` and
  `doca_devinfo_*` extensions). The experimental markers are
  release-specific; the agent must read the installed
  `doca_common.h` family on the user's install before claiming
  experimental-tier surface is available.
- **No standalone `doca-log.pc` in this contract.** Per the source
  tree on this DOCA version, the log subsystem is exposed through
  the `doca-common` `pkg-config` module (its public headers live
  under the same include root as the rest of the Common surface,
  resolved via `pkg-config --variable=includedir doca-common`).
  The agent's probe rule is to `pkg-config
  --exists doca-common` first; if a release exposes a standalone
  `doca-log.pc`, probe that too and report which is present. The
  build line is whichever the probe found.

Version-specific tables of symbol availability are deliberately not
maintained in this file — they would drift out of date silently. The
discipline is "read the headers and the matching release notes", not
"trust this file's table".

## Error taxonomy

The cross-library `DOCA_ERROR_*` taxonomy (what each family means
and which debug layer it routes to) lives in
[`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../../doca-programming-guide/CAPABILITIES.md#error-taxonomy).
The Common-specific overlay names the families the agent will see
most often from `doca_buf_*` / `doca_ctx_*` / `doca_dev_*` /
`doca_pe_*` / `doca_mmap_*` / `doca_log_*` calls and what they
specifically indicate at the foundation layer:

| Error | Common subsystem where it shows up | Common-specific cause | First action |
| --- | --- | --- | --- |
| `DOCA_ERROR_BAD_STATE` | Any of `doca_ctx_*`, `doca_mmap_*`, `doca_buf_inventory_*`, `doca_pe_*`, `doca_log_*` | Lifecycle violation. The object was operated on outside its allowed window: `ctx_start` before `connect_ctx`; `mmap_add_dev` after `mmap_start`; `buf_inventory_buf_get_by_args` before `buf_inventory_start`; `pe_progress` after `pe_destroy`; `DOCA_LOG_*` emission from a source not yet registered. | Walk the lifecycle in [`## ctx`](#ctx) / [`## buf`](#buf) / [`## progress engine`](#progress-engine) / [`## log`](#log); confirm each call's preconditions BEFORE retrying. Static initializers that emit log lines BEFORE `main()`-time init are a frequent log-subsystem cause. |
| `DOCA_ERROR_INVALID_VALUE` | `doca_dev_open` (invalid devinfo), `doca_mmap_set_memrange` (bad range), `doca_buf_*` (out-of-range handle), `doca_log_*` (level enum / unregistered source) | The call's arguments did not pass validation. For `doca_log`, the most common cause is an unregistered source ID passed to a `DOCA_LOG_*` macro; for `doca_mmap`, a memrange that overlaps an existing range; for `doca_dev_open`, a devinfo that was destroyed (the destroy_list call invalidates every devinfo pointer it owned). | Re-check the call's arguments against the headers. The cap-query family is also `INVALID_VALUE`-prone when the devinfo passed in was destroyed — re-fetch from a live `doca_devinfo_create_list`. |
| `DOCA_ERROR_NOT_SUPPORTED` | Any `doca_*_cap_*` query that returns false at runtime, then the actual call that depends on it. Common examples: `doca_mmap_cap_is_export_pci_supported` returning false followed by `doca_mmap_export_pci`; a `doca_devinfo` that does not advertise hotplug-manager support followed by the corresponding `doca_dev_*` call; an experimental log level (`TRACE`) on a release that does not expose it | The device, firmware, or DOCA version does not support what the user requested. Climb back up: drop the optional capability, choose a different device, or upgrade the install. | Re-run the cap-query against the *active* `doca_devinfo`; if false, that is the answer. Do not retry the same call on the same device. |
| `DOCA_ERROR_NOT_PERMITTED` | `doca_dev_open` (insufficient permissions to open the device), `doca_mmap_*` (insufficient permissions on the registered memory or the device-side mapping) | Host-side env issue. Common causes: RDMA stack module loads, user not in the right group, ulimits, IOMMU mode mismatch, missing capabilities (`CAP_SYS_RAWIO` / `CAP_NET_ADMIN` depending on the device class). | Route to [`doca-setup TASKS.md ## debug`](../../doca-setup/TASKS.md#debug) — the layer below the Common API is the suspect. This is *not* a Common-spec error. |
| `DOCA_ERROR_NO_MEMORY` | `doca_buf_inventory_buf_get_by_args` (inventory pool exhausted), `doca_mmap_create` / `_add_dev` (cannot register memory), custom log sink with backpressure | Resource exhaustion. The buffer pool is sized for a maximum number of in-flight handles; exhausting it is normal back-pressure and the right response is `doca_buf_dec_refcount` on completed work before requesting new buffers. | Re-read [`## buf`](#buf) for the inventory sizing rule; do not retry blindly. For the log subsystem this is rare on the default `stderr` sink (which does not queue) and only fires with a custom queueing backend. |
| `DOCA_ERROR_DRIVER` | Any `doca_dev_*`, `doca_mmap_*`, or `doca_ctx_*` call that returns a kernel-driver-level failure | The kernel driver, firmware, or PCIe path is in a state the Common library cannot recover from. **Stop.** This is not a Common-spec problem. | Capture device state via the platform's diagnostic CLIs (`lspci`, `dmesg`, `ibv_devinfo`, `mlxconfig -d <pcie> q`) and escalate per [`doca-setup TASKS.md ## debug`](../../doca-setup/TASKS.md#debug) driver layer. |

The agent's rule: **never recommend a retry loop on `DOCA_ERROR_*`
from a Common call without first identifying which row above is the
cause.** Retrying masks the bug; lifecycle and capability errors do
not become success on retry.

Quote `doca_error_get_descr()` verbatim — do not paraphrase. The
cross-cutting debug ladder
([`doca-debug ## debug`](../../doca-debug/TASKS.md#debug)) is the
canonical layered diagnosis path that the agent escalates to once
the Common-specific cause has been narrowed.

## Observability

DOCA Common is the **observability foundation every higher-level
library builds on**. The agent's job in any DOCA session is to know
which observable surfaces Common exposes and which downstream
consumers can read them.

| Surface | What it shows | Owning subsystem |
| --- | --- | --- |
| DOCA Log lines (default `stderr` sink; per-line timestamp / level / source / message) | Every DOCA library's own log lines plus the user's app's emissions. The single most reached-for observable | [`## log`](#log) |
| Two-tier log levels | `--sdk-log-level` reaches DOCA library internals; the app-side global / per-source setter reaches the user's code. Change ONE tier at a time | [`## log`](#log) |
| PE notification handle | `doca_pe_get_notification_handle` exposes an fd the run loop can `epoll` to know when completions are ready | [`## progress engine`](#progress-engine) |
| Inflight-task counts | `doca_pe_get_num_inflight_tasks` and `doca_ctx_get_num_inflight_tasks` make the drain state explicit on shutdown | [`## ctx`](#ctx), [`## progress engine`](#progress-engine) |
| Capability snapshot | The output of every `doca_*_cap_*` query at configure time is the *baseline* the agent compares against when a later call returns `DOCA_ERROR_NOT_SUPPORTED` | [`## dev`](#dev) |
| Context state | `doca_ctx_get_state` (and the `_set_state_changed_cb` callback) tells the user where in the lifecycle the context currently is — the load-bearing input when a call returns `DOCA_ERROR_BAD_STATE` | [`## ctx`](#ctx) |
| Device-info fields | `doca_devinfo_get_pci_addr_str`, `_get_ibdev_name`, `_get_iface_name`, `_get_mac_addr`, `_get_active_mtu`, `_get_active_speed` give the operator-readable identity of every candidate device | [`## dev`](#dev) |

**Downstream consumers.** DOCA Log output is the input to the DOCA
Log Service and the DOCA Telemetry Service (per
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)):
when the user pipes / forwards their app's `stderr` (or routes
through a custom backend), those services can consume the lines
without any in-app wiring. This is part of *why* DOCA Log is
preferable to `printf` in a DOCA-app context — the downstream shape
is the same as every other DOCA artifact on the host.

**Cross-link to the cross-cutting debug ladder.** The
verbosity-escalation surface is owned by [`## log`](#log) (the
two-tier model). The *layered debug ladder* the Common
observability surface feeds into is owned by
[`doca-debug CAPABILITIES.md ## Observability`](../../doca-debug/CAPABILITIES.md#observability) —
which calls back here for the two-tier mechanics, the PE drive
loop, and the inflight-task drain on shutdown.

## Safety policy

> **Overlay on the bundle-wide hardware-safety meta-policy.** The rules below are this skill's per-artifact overlay on the cross-cutting rules in [`doca-hardware-safety` CAPABILITIES.md ## Safety policy](../../doca-hardware-safety/CAPABILITIES.md#safety-policy) (specifically [### Per-artifact overlay pattern](../../doca-hardware-safety/CAPABILITIES.md#per-artifact-overlay-pattern)). When the two layers disagree, the stricter wins; when either layer says STOP, the agent stops.

DOCA Common's per-artifact safety surface is the **disciplined
lifecycle ordering** every higher-level library inherits, plus the
log-side guardrails the rest of the bundle relies on.

- **Lifecycle order is the contract for every context.** Skipping
  `doca_pe_connect_ctx` before `doca_ctx_start`, destroying a
  `doca_mmap` while a `doca_buf` from its inventory is still in
  flight, destroying a `doca_dev` while a context is still using it,
  or calling `doca_ctx_destroy` before `doca_ctx_stop` is the
  canonical "the program crashes far from the offending line"
  failure mode. The agent's response is *always* to walk the
  lifecycle in [`## ctx`](#ctx) / [`## buf`](#buf) /
  [`## progress engine`](#progress-engine), not to retry. The
  ordering is a property of the library; ignoring it is a
  user-visible regression dressed up as a stack trace.
- **Capability-query before action.** Per [`## dev`](#dev), the
  cap-query against the active `doca_devinfo` is the runtime
  authority. Quoting a feature from documentation prose without
  running the cap-query is the most common foundation-layer
  hallucination. The cost is real: a `NOT_SUPPORTED` at runtime
  surfaces far from the place the assumption was made and wastes
  debug budget.
- **Custom-sink writes inherit the sink's own permission envelope.**
  When the user installs a custom log backend (file / fd / buf /
  syslog) instead of relying on the default `stderr`, the writes
  happen under that sink's own permission rules — file permissions
  on the target path, fd permissions inherited from the caller,
  syslog ACLs on the destination. The DOCA Log API itself adds no
  permission elevation; the user must ensure the process has
  whatever rights the sink needs. Validate the sink with a small
  write at registration time, before any production traffic.
- **Do not log secrets through `DOCA_LOG_*`.** Format strings
  passed to `DOCA_LOG_*` macros take printf-style varargs; there is
  no automatic redaction. Anything the user passes becomes part of
  the line and may be forwarded to the DOCA Log Service or DOCA
  Telemetry Service downstream. Treat `DOCA_LOG_*` as a
  public-equivalent emission surface and apply the user's normal
  redaction discipline at the call site, not inside the library.
- **One PE per thread is the default; cross-thread PE sharing
  requires the user's synchronization.** Calling
  `doca_pe_progress(pe)` from two threads on the same PE without
  synchronization produces interleaved completion delivery and
  occasional double-drain. The agent's default is to recommend one
  PE per worker thread; multi-PE topologies are valid but the user
  must own the synchronization model.
- **No mixing of `doca_buf` handles across inventories of different
  `doca_mmap`s.** A `doca_buf` belongs to exactly one
  `doca_buf_inventory`, which in turn was created over exactly one
  set of started `doca_mmap` ranges. Passing a `doca_buf` from
  inventory A to a context that only knows inventory B is
  unsupported; the visible symptom is `DOCA_ERROR_INVALID_VALUE`
  at submit time or silent data corruption. The agent's first
  response to *"can I share a buf across inventories?"* is **no** —
  re-register the memory in the second inventory or use the
  cross-context buf flow described in [`## buf`](#buf).

The agent's job is to **enforce these orderings and discipline in
the workflow**, not just describe them. When the user says
*"skip the lifecycle, just call the API"*, the right answer is to
refuse and explain the cost, not to comply.
