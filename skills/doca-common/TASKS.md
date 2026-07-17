# DOCA Common workflows

**Where to start:** The verbs run `configure → build → modify → run
→ test → debug`, with `## use` and `## log` as supplementary
foundation verbs (the `use` verb walks the universal foundation
skeleton; the `log` verb is the verb-side of the two-tier log model
folded in from the historical doca-log content). The `## test` verb
is an iterative loop (cap-query cross-check → lifecycle smoke →
PE-drive verification → loop back if any of the three findings
mutate the configure step), not a one-shot pass.

Read this file when the loader sent you here from
[SKILL.md](SKILL.md). For the universal primitives (`log` / `buf` /
`ctx` / `dev` / `progress engine`), the doca-common version overlay,
the Common error taxonomy, the observability surface, and the safety
policy that these workflows assume, see
[CAPABILITIES.md](CAPABILITIES.md). For the universal
modify-a-shipped-sample workflow and the cross-library
`DOCA_ERROR_*` taxonomy these workflows rest on, see
[`doca-programming-guide`](../../doca-programming-guide/SKILL.md).
For the cross-cutting debug ladder that the Common debug verb feeds
into, see [`doca-debug`](../../doca-debug/SKILL.md).

Each verb below describes the **shape of the workflow**, not a
copy-paste recipe. The agent's job is to walk the user through the
steps in order, verifying preconditions before recommending the
next call.

## install

DOCA Common is part of every DOCA install — installing doca-common
in isolation is not a thing. The install verb routes to
[`doca-setup`](../../doca-setup/SKILL.md) for the env / install
chain, and to
[`doca-public-knowledge-map ## Layout of an installed DOCA package`](../../doca-public-knowledge-map/SKILL.md#layout-of-an-installed-doca-package)
for the on-disk layout the rest of this file assumes.

The agent's checks before declaring the install ready for any
foundation work:

1. **`pkg-config --modversion doca-common` returns a semver
   string** — this is the universal anchor every doca-common
   workflow assumes. If the query fails, the install is broken;
   route to [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug)
   layer 1 (install).
2. **`doca_caps --version` agrees** with the `pkg-config` result
   per the four-way match rule in
   [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility).
3. **The Common headers are resolvable** under
   $(pkg-config --variable=includedir doca-common) (`doca_buf.h`,
   `doca_ctx.h`, `doca_dev.h`, `doca_pe.h`, `doca_log.h`,
   `doca_mmap.h`, …) per the headers-win-over-docs rule in
   [`doca-version`](../../doca-version/SKILL.md).

If any of the three checks fails, **stop** — this skill's
workflows assume the install is healthy, and a partial install is a
[`doca-setup`](../../doca-setup/SKILL.md) concern.

## configure

Goal: bring up the universal doca-common foundation that every
higher-level library context (Flow, RDMA, Eth, DMA, Comch,
Rmax, …) is going to consume.

Steps the agent should walk the user through:

1. **Confirm the installed DOCA version.** Use the procedure in
   [`doca-version TASKS.md ## configure`](../../doca-version/TASKS.md#configure).
   Quote the version observed (`pkg-config --modversion
   doca-common`, then `doca_caps --version`); do not assume
   "latest". The four-way match rule lives in
   [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility).
2. **Enumerate devices and pick one.** Call
   `doca_devinfo_create_list(&devinfos, &num)`; iterate; pick the
   `doca_devinfo` whose `doca_devinfo_get_pci_addr_str` /
   `_get_ibdev_name` / `_get_iface_name` matches the device the
   user named. Per [`CAPABILITIES.md ## dev`](CAPABILITIES.md#dev),
   the devinfo is read-only and the open step is separate.
3. **Run the capability queries for whatever the user wants to
   use.** Before opening the device, run the matching
   `doca_*_cap_*` family against the chosen `doca_devinfo`. This
   is the cap-query-is-runtime-authority rule from
   [`CAPABILITIES.md ## dev`](CAPABILITIES.md#dev). If a required
   capability is false, **stop** — climb back up to the user's
   intent (different device, different feature set), do not retry.
4. **Open the device.** Call `doca_dev_open(devinfo, &dev)`. From
   this point the `doca_devinfo` pointer is still valid until the
   matching `doca_devinfo_destroy_list` call; the opened `doca_dev`
   is what per-library contexts consume.
5. **Enumerate representors if the user's case needs them.** On a
   BlueField host with the DPU in switch mode, the host program
   discovers DPU-side function representors through
   `doca_devinfo_rep_create_list(dev, filter, &reps, &num)` and
   opens the ones it needs with `doca_dev_rep_open`. The
   representor handle is what the user passes to `doca_flow`
   actions, `doca_eth` queues, etc., when steering traffic to a
   specific DPU-side function.
6. **Create the progress engine.** `doca_pe_create(&pe)` produces
   the universal task-completion drain. One PE per worker thread
   is the default shape per
   [`CAPABILITIES.md ## progress engine`](CAPABILITIES.md#progress-engine).
7. **Wire DOCA Log if the user wants their own emissions.** Register
   each log source via `doca_log_register_source(<name>,
   &source_id)` at component init time; pick the level enum from
   the always-present set (`CRITICAL` / `ERROR` / `WARNING` /
   `INFO` / `DEBUG`) per
   [`CAPABILITIES.md ## log`](CAPABILITIES.md#log). The agent walks
   the two-tier model with the user BEFORE any code change — see
   [`## log`](#log) for the verb-side workflow.
8. **Hand off to the per-library skill.** Open the user's primary
   library context (`doca_rdma_create`, `doca_eth_txq_create`,
   `doca_comch_client_create`, …) per the matching library's
   `## configure` verb; `doca_pe_connect_ctx(pe, ctx)` BEFORE
   `doca_ctx_start(ctx)` so completions surface.

If any step fails with a `DOCA_ERROR_*`, route through the error
taxonomy in
[`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
before retrying.

## build

Goal: compile a doca-common consumer (or any DOCA app, since they
all link doca-common transitively) against the user's installed
DOCA, with `pkg-config` as the source of truth for include + link
flags.

The build pattern for any DOCA C/C++ consumer is fully documented
in
[`doca-programming-guide TASKS.md ## build`](../../doca-programming-guide/TASKS.md#build).
This skill carries only the doca-common-specific overlay:

| Slot | Value for doca-common | Why it matters |
| --- | --- | --- |
| `pkg-config` module name | `doca-common` (always present on any healthy install; the universal anchor) | The Common surface is the foundation every higher-level library depends on. The agent's probe rule is `pkg-config --exists doca-common` first; if it fails, the install is broken (route to [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug) layer 1) |
| Required runtime libs | `libdoca_common` (returned by `pkg-config --libs doca-common`) | Any DOCA app links this transitively through its primary library's `.pc`, but a Common-only consumer (e.g. a device-discovery tool) links it directly |
| Include flags | `pkg-config --cflags doca-common` resolves to includes under $(pkg-config --variable=includedir doca-common) (`doca_buf.h`, `doca_ctx.h`, `doca_dev.h`, `doca_pe.h`, `doca_log.h`, `doca_mmap.h`, `doca_error.h`, `doca_types.h`, …) | Hand-typed `-I` lines drift between releases; only `pkg-config` survives a DOCA upgrade |
| Link flags | `pkg-config --libs doca-common` returns the canonical `-l` list | Hand-typed `-l` lines are the failure mode; the order matters for static-linking and `pkg-config` knows the right shape |
| Minimum required DOCA version | Query with `pkg-config --modversion doca-common`; never hardcode in build files | The four-way match rule in [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility) requires the build-time version to be reachable; hardcoding it breaks the chain |
| Companion library probes (optional) | If the release exposes a standalone `doca-log.pc` (release-dependent), probe it with `pkg-config --exists doca-log` and use it for the log subsystem; otherwise the log surface ships through `doca-common` | Per [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility), the agent must verify on the user's install — never quote one shape from memory |

For non-C consumers (Rust, Go, Python), the wrapper consumes the
same `*.so` set through FFI; the build-time version visibility goes
through the language's own FFI generator (e.g. `bindgen` against
`doca_ctx.h` / `doca_dev.h` / `doca_buf.h` / `doca_pe.h` /
`doca_log.h`). The lifecycle, the cap-query rule, the PE drive
loop, and the two-tier log model still apply — the wrapper consumes
a `*.so` that has its own runtime version per
[`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility).

## modify

Goal: take the closest-fitting shipped DOCA sample (any of them —
doca-common primitives show up in every shipped sample's `*_main.c`)
and apply a **minimum diff** to add or change the foundation wiring
(device pick, buffer registration, PE drive loop, log lines),
without rewriting from scratch.

The universal modify-a-shipped-sample workflow is in
[`doca-programming-guide TASKS.md ## modify`](../../doca-programming-guide/TASKS.md#modify);
this skill provides the doca-common-specific slot fill.

| Slot | Value | Source |
| --- | --- | --- |
| Sample tree | Any shipped sample's `*_main.c` under `/opt/mellanox/doca/samples/<library>/<sample>/`. doca-common has no dedicated sample directory — the Common foundation is wired into every shipped sample's `*_main.c` (open device → create PE → create per-library ctx → connect_ctx → start → run loop → drain → stop → destroy) | Confirmed by `ls /opt/mellanox/doca/samples/` and reading any sample's `*_main.c` |
| Pick the closest sample | Whichever sample the user is *already* modifying for their primary library. The Common foundation is identical across them; modifying the foundation of an unrelated sample is a strictly larger diff than the user needs | Per the path-selection table in [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes) |
| Identify the foundation modify surface | Device pick (the `doca_devinfo_create_list` loop's filter / pick condition); buffer wiring (`doca_mmap_set_memrange`, `doca_buf_inventory_buf_get_by_args` callers); PE drive loop (`doca_pe_progress` location and cadence); log wiring (the `doca_log_register_source` call at init and the `DOCA_LOG_*` emissions across the file) | The sample's existing `*_main.c` is the carrier; the user's foundation edits are the diff |
| Keep the lifecycle intact | Do NOT delete `doca_pe_connect_ctx` (the sample wired it; removing it is the universal "tasks submit but nothing completes" failure mode); do NOT reorder `ctx_start` before `pe_connect_ctx`; do NOT remove the `doca_ctx_get_num_inflight_tasks` drain on shutdown | Per the lifecycle order in [`CAPABILITIES.md ## ctx`](CAPABILITIES.md#ctx) and [`CAPABILITIES.md ## progress engine`](CAPABILITIES.md#progress-engine) |
| Keep the build manifest unchanged | The sample's existing `meson.build` already wires the right `pkg-config` modules (`doca-common` at minimum); do not switch to a hand-rolled Makefile for *"simplicity"* — it removes the version-check rail | Per the build slot table in [`## build`](#build) |

The agent's anti-pattern alert: a *"clean rewrite"* that swaps the
sample's foundation wiring for hand-rolled code is almost always
the wrong shape. The sample is the verified source of truth for the
exact release that's installed; keeping the minimum diff is what
keeps the user's code on the supported path.

## run

Goal: actually execute the built program, observe completions
arriving, and demonstrate the PE drive loop is healthy before the
user starts adding real protocol logic.

Steps the agent should walk the user through:

1. **Confirm the active DOCA install.** Re-quote `pkg-config
   --modversion doca-common` and `doca_caps --version`; they must
   agree. A mismatch means the binary is loading a different
   `libdoca_common.so` than the one its build-time `pkg-config`
   saw — debug via
   [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug)
   before assuming a Common bug.
2. **Set the SDK log tier explicitly on first run.** Pass
   `--sdk-log-level WARNING` (most shipped samples / reference apps
   accept it) or `DOCA_LOG_LEVEL_SDK=WARNING` in the environment.
   The default is WARNING per
   [`CAPABILITIES.md ## log`](CAPABILITIES.md#log); for first runs
   the agent should default the SDK tier to WARNING (do not crank
   to DEBUG) so the user's own lines are not buried in DOCA-library
   internal trace.
3. **Set the app log tier explicitly on first run.** Use
   `doca_log_level_set_global_lower_limit` on the app-tier registry
   (or the per-source setter for fine-grained control). For first
   runs the agent should default the app tier to DEBUG so the user
   can see their own DEBUG-shaped lines; drop to INFO for
   steady-state operation.
4. **Drive the PE on every loop iteration.** The minimal run loop
   is `while (running) doca_pe_progress(pe);` — for a control-plane
   thread, prefer the notification handle (`doca_pe_get_notification_handle`
   + `doca_pe_request_notification` + `epoll_wait` + clear, then
   `progress`) so the thread is not busy-polling. Per
   [`CAPABILITIES.md ## progress engine`](CAPABILITIES.md#progress-engine),
   skipping the PE drive is the universal "my program does nothing"
   symptom.
5. **Submit one task and confirm it completes.** Use the
   per-library `task_*_submit` for the user's primary library;
   confirm the completion callback fires through `doca_pe_progress`.
   This is the cheapest smoke that the foundation is alive — if
   the callback never fires, the failure is at the foundation
   layer, not the per-library layer.
6. **Capture the structured log.** Redirect `stderr` to a file
   (`2> doca.log`) so the subsequent `## test` iterations have a
   stable artifact to diff against.

For the runtime version + `LD_LIBRARY_PATH` cross-checks that
underlie *"the program built but does nothing"*, see
[`doca-version TASKS.md ## run`](../../doca-version/TASKS.md#run).

## test

Goal: prove the foundation is behaving as expected — the device was
picked correctly, the cap-queries the configure step ran still
hold, the lifecycle is in order, and the PE drive loop is wired —
before claiming the *"wire doca-common into the app"* journey is
done.

**`## test` is an iterative loop, not a one-shot pass.** The agent's
job is to run the four steps below in order, and *loop back to
step 1 whenever the spec is mutated by a cross-check finding*.
Treating validate-once as good-enough is the failure mode this loop
replaces; every spec mutation re-opens validate.

The eval-loop overlay (rows apply to every Common skeleton, not
just one):

| Step | Why this is a loop, not a step | Where the substance lives |
| --- | --- | --- |
| 1 → 2 → 1 | Cap-query cross-check (step 2) may reveal the device does not support a feature the user assumed; loop back to step 1 with a different device or a different feature set | [`## configure`](#configure) step 3 + [`CAPABILITIES.md ## dev`](CAPABILITIES.md#dev) |
| 1 → 3 → 1 | Lifecycle smoke (step 3) may reveal a missing `doca_pe_connect_ctx` or a misordered `ctx_start`; loop back to step 1 with the order fixed | [`CAPABILITIES.md ## ctx`](CAPABILITIES.md#ctx) lifecycle |
| 1 → 4 → 1 | PE-drive verification (step 4) may reveal `doca_pe_progress` is never called or is called on the wrong PE; loop back to step 1 with the loop fixed | [`CAPABILITIES.md ## progress engine`](CAPABILITIES.md#progress-engine) |
| 4 → ## debug | Tasks submit, PE is driven, completions still do not arrive — escalate to the Common debug overlay | [`## debug`](#debug) |

The agent's rule: every mutation between steps re-opens the four
steps. Skipping the re-validate after a mutation is the universal
foundation-layer failure mode.

Steps:

1. **Cap-query baseline.** Save the output of every
   `doca_*_cap_*` query the user ran in
   [`## configure`](#configure) step 3 as a baseline. A later
   `DOCA_ERROR_NOT_SUPPORTED` is the diff against this snapshot —
   without it, the agent has no way to disambiguate "the device
   never supported this" from "something changed".
2. **Cap-query re-check.** Re-run the same `doca_*_cap_*` queries
   on the *running* program (or via `doca_caps --list-devs` if the
   library exposes that cap publicly). The result must match the
   configure-time baseline; if it does not, the device or firmware
   state changed (mode flip, firmware burn — see
   [`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md))
   and the user is operating on a different surface than they
   thought.
3. **Lifecycle smoke.** With the user's smallest possible
   reproducer (one device, one PE, one ctx, one task), confirm:
   `doca_pe_connect_ctx` happens BEFORE `doca_ctx_start`; the
   per-library task submit happens AFTER `doca_ctx_start`; the
   shutdown path drains via `doca_ctx_get_num_inflight_tasks` /
   `doca_pe_get_num_inflight_tasks` before destroy. Catches the
   most common lifecycle-order bugs.
4. **PE-drive verification.** Confirm `doca_pe_progress(pe)` is
   reachable in the user's main loop on the same thread that
   created the PE. Catches the case where the run loop drives a
   different PE than the one the contexts are connected to.

Loop termination: stop iterating once the four steps each pass
without mutating the prior step. Beyond that, escalate suspicious
behavior to
[`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug)
with the captured log + baseline + version state as evidence.

## debug

Goal: when a `doca_*` call returns a `DOCA_ERROR_*` or the program
does not make forward progress, narrow the cause to a single layer
before recommending any code change.

> **Routing summary.** This anchor is the **doca-common-specific
> debug overlay**: lifecycle errors, cap-query mismatches, PE drive
> loop gaps, log-tier confusion. For the **cross-cutting debug
> ladder** (install / version / build / link / runtime / program /
> driver) plus the cross-cutting tooling surface (`gdb`, `valgrind`,
> `--sdk-log-level`, `DOCA_LOG_LEVEL`, the `doca-<lib>-trace` build
> flavor, container-vs-native debug, core dumps, Developer Forum
> escalation), see
> [`doca-debug ## debug`](../../doca-debug/TASKS.md#debug). The
> agent should walk the cross-cutting ladder first whenever the
> symptom layer is not yet known; this Common overlay layers on top
> once the symptom is confirmed to be inside the foundation surface.

Walk in this order — do not skip steps:

1. **Tier-confusion sanity (cheapest).** If the symptom is *"my log
   lines do not appear"* or *"my console is flooded"*, the first
   hypothesis is two-tier confusion per
   [`CAPABILITIES.md ## log`](CAPABILITIES.md#log). Walk the table
   before any code change.
2. **Lifecycle order.** If the error is `DOCA_ERROR_BAD_STATE`, walk
   the lifecycle for whichever subsystem the call belongs to:
   `doca_ctx_*` → [`CAPABILITIES.md ## ctx`](CAPABILITIES.md#ctx);
   `doca_mmap_*` / `doca_buf_*` →
   [`CAPABILITIES.md ## buf`](CAPABILITIES.md#buf); `doca_pe_*` →
   [`CAPABILITIES.md ## progress engine`](CAPABILITIES.md#progress-engine);
   `doca_log_*` → [`CAPABILITIES.md ## log`](CAPABILITIES.md#log).
3. **Cap-query mismatch.** If the error is
   `DOCA_ERROR_NOT_SUPPORTED`, the call assumed a capability the
   active `doca_devinfo` does not advertise. Re-run the matching
   `doca_*_cap_*` query and compare against the cap baseline from
   [`## test`](#test) step 1.
4. **PE drive loop.** If tasks submit and never complete, the PE
   is not being driven on the thread that owns the connected
   contexts. Confirm `doca_pe_progress(pe)` is in the main loop
   and is the same `pe` that `doca_pe_connect_ctx` was called
   against.
5. **Permission envelope.** If the error is
   `DOCA_ERROR_NOT_PERMITTED`, this is a host-side env issue
   (kernel module loads, user group, ulimits, IOMMU mode, missing
   capabilities) — route to
   [`doca-setup TASKS.md ## debug`](../../doca-setup/TASKS.md#debug).
6. **Version sanity.** If a previously working spec now fails or
   behaves differently, confirm the installed DOCA version did not
   change via the four-source coherence check in
   [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug)
   layer 2. A library upgrade between sessions is a common and
   easy-to-miss cause.
7. **Escalation criteria.** If the lifecycle is correct, the
   cap-queries hold, the PE is driven, permissions are right, and
   the version is unchanged AND the symptom persists — the bug is
   below the Common API surface (driver or firmware). Stop
   attempting Common-spec changes; capture state per
   [`doca-debug ## test`](../../doca-debug/TASKS.md#test) (the
   read-only triple) and escalate via
   [`doca-debug ## debug` *Where to ask for help*](../../doca-debug/TASKS.md#debug)
   to the public DOCA Developer Forum.

## use

Goal: walk the **universal foundation skeleton** — the exact
sequence every DOCA app of every shape follows before the
per-library specialization begins.

This verb is the *use-the-foundation* counterpart to
[`## configure`](#configure): it names the sequence so the agent
can quote it the same way every time, regardless of which higher-
level library the user's primary work is in.

The universal skeleton, in order:

1. **`doca_devinfo_create_list(&devinfos, &num)`** — read-only
   enumeration of every candidate device DOCA sees on the host or
   BlueField.
2. **Pick the `doca_devinfo` the user wants** — by PCIe address
   (`doca_devinfo_get_pci_addr_str`), by IB device name
   (`_get_ibdev_name`), by interface name (`_get_iface_name`), or
   by any other identifier the user provided.
3. **Run the cap-queries** — for every feature the user assumed,
   run the matching `doca_*_cap_*` against the chosen
   `doca_devinfo`. The output is the capability snapshot
   (per [`CAPABILITIES.md ## Observability`](CAPABILITIES.md#observability))
   that the rest of the session compares against.
4. **`doca_dev_open(devinfo, &dev)`** — open the chosen device.
   From here the `doca_dev` is the handle every per-library
   `doca_*_set_dev` call consumes.
5. **(Optional) Enumerate representors.** On a BlueField with the
   DPU in switch mode, the host program calls
   `doca_devinfo_rep_create_list(dev, filter, &reps, &num)` and
   opens the representors it needs with `doca_dev_rep_open`.
6. **`doca_pe_create(&pe)`** — create the universal task-completion
   drain. One PE per worker thread is the default shape.
7. **Per-library `doca_<library>_create(...)`** — create the
   per-library context the user's primary work needs (a
   `doca_flow_port`, a `doca_rdma`, a `doca_eth_txq`, a `doca_dma`,
   …). The handle this returns wraps a `doca_ctx` under the hood.
8. **Configure the per-library context** — call the library's
   `*_set_*` setters BEFORE `doca_ctx_start`. Setters called after
   start return `DOCA_ERROR_BAD_STATE`.
9. **(For zero-copy I/O) wire the buffer stack.** `doca_mmap_create`
   → `doca_mmap_set_memrange` → `doca_mmap_add_dev(mmap, dev)` →
   `doca_mmap_set_permissions` → `doca_mmap_start`. Then
   `doca_buf_inventory_create` over the mmap → `doca_buf_inventory_start`.
   Buffers are handed out by
   `doca_buf_inventory_buf_get_by_args` on demand.
10. **`doca_pe_connect_ctx(pe, ctx)`** — register the context with
    the PE so its task completions surface through
    `doca_pe_progress`. Skipping this is the canonical "tasks
    submit but nothing completes" failure mode.
11. **`doca_ctx_start(ctx)`** — transition the context to RUNNING.
12. **Drive the run loop** — `while (running) doca_pe_progress(pe);`
    (or the notification-handle variant for control-plane threads).
13. **Submit tasks via the per-library task API.** Completions
    surface through the PE on the thread that drives it.
14. **Shutdown.** Drain inflight tasks
    (`doca_ctx_flush_tasks(ctx)` or
    `doca_pe_progress(pe)` until
    `doca_pe_get_num_inflight_tasks(pe) == 0`) → `doca_ctx_stop(ctx)`
    → per-library `doca_<library>_destroy(...)` → buffer-stack
    teardown (`doca_buf_inventory_stop` → `_destroy` →
    `doca_mmap_stop` → `_destroy`) → `doca_pe_destroy(pe)` →
    `doca_dev_close(dev)` → `doca_devinfo_destroy_list(devinfos)`.

The agent's rule: **this sequence is the same regardless of which
higher-level library is layered on top.** When the user asks
*"what's the doca-common skeleton I need before I open my Flow port
/ my RDMA context / my Eth queue / my DMA context?"*, the answer is
this list, with the per-library `_create` call slotted into step 7
and the library's `_set_*` setters into step 8.

## log

Goal: wire DOCA Log into a DOCA app (or into a freshly modified DOCA
sample) so the user's own emission lines run alongside the DOCA
library's own log lines, with the two tiers under independent
control.

This verb is the verb-side of the two-tier model documented in
[`CAPABILITIES.md ## log`](CAPABILITIES.md#log). It is folded from
the historical doca-log skill into this skill because the log
subsystem ships through the `doca-common` `pkg-config` module per
[`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility).

Steps the agent should walk the user through:

1. **Confirm DOCA Log is the right primitive for this app.** Walk
   the path-selection rule in
   [`CAPABILITIES.md ## log`](CAPABILITIES.md#log). If the codebase
   has no DOCA-side context (no `doca_*` calls, no DOCA libraries
   linked), DOCA Log is not the right answer — language-native
   logging is. Recommending DOCA Log *for* the user when the
   path-selection rule rules it out is a wrong answer regardless of
   how cleanly the rest of the configure step goes.
2. **Walk the two-tier model with the user, BEFORE any code.** Per
   the two-tier table in
   [`CAPABILITIES.md ## log`](CAPABILITIES.md#log), the agent must
   surface that SDK level controls *DOCA library internals*
   (default `WARNING`; setter `--sdk-log-level` /
   `DOCA_LOG_LEVEL_SDK`) and app level controls *user code
   emissions* (default `INFO`; setter is the app-side
   `doca_log_level_set_global_lower_limit`). Confusing the two is
   the number-one first-app debug failure. The agent should
   confirm the user's intent (do they want to see *DOCA library
   internal* lines, *their own* lines, or both) and pick the tiers
   accordingly.
3. **Register each log source ONCE before any emission.** For the
   user's own source files (typically one per `.c` file or per
   component), call `doca_log_register_source(<name>, &source_id)`
   at component init time. Per
   [`CAPABILITIES.md ## log`](CAPABILITIES.md#log) objects table,
   an unregistered source ID passed to a `DOCA_LOG_*` macro
   returns `DOCA_ERROR_INVALID_VALUE`. The register call sits
   BEFORE any `DOCA_LOG_*` from that source; do not invert the
   order.
4. **Optionally install a custom backend / sink.** The default sink
   is `stderr` (created implicitly via `doca_log_backend_create_standard`
   on most app init paths). If the user wants log lines to land in
   a file, an fd, a buffer, or syslog, use the matching
   `doca_log_backend_create_with_*` (and the `_sdk` variant for the
   SDK tier). Per
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
   custom-sink rule, the writes inherit the sink's own permission
   envelope; validate the sink with a small write at registration
   time before any production traffic.
5. **Smoke one emission per level the user cares about.** Emit one
   `DOCA_LOG_INFO`, one `DOCA_LOG_DBG`, and one `DOCA_LOG_ERR` from
   the user's own source. Confirm the line shape (timestamp / level
   / source / message) matches the per-line format the shipped
   DOCA samples produce.
6. **Iterate the tier flip.** Run with `SDK=WARNING, App=DEBUG`;
   confirm the user's own DEBUG lines appear and DOCA library
   internals are quiet. Flip to `SDK=DEBUG, App=WARNING`; confirm
   DOCA library internals flood and the user's own DEBUG lines
   disappear. If either iteration does not behave as expected,
   route to [`## debug`](#debug) step 1.

For the verb-side run / test iteration on the log surface, the
universal [`## run`](#run) steps 2 and 3 already cover the
first-run tier defaults. The `## test` step that exercises the
two-tier flip is the iteration described above.

## Deferred task verbs

The following verbs are out of scope for this skill but are
commonly asked in the same conversations. Route them as follows so
the agent does not invent guidance:

- **deploy.** Deploying DOCA apps at scale, routing log output to
  centralized aggregation systems, K8s sidecars — out of scope for
  this skill; partial guidance in
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
  for DOCA Log Service and DOCA Telemetry Service routing, and
  [`doca-container-deployment`](../../doca-container-deployment/SKILL.md)
  for the container path.
- **rollback.** Coordinated rollback across multiple hosts /
  DPUs — out of scope for this skill. For single-host config
  rollback within a session, the right verb is destroying contexts
  / mmaps / inventories in reverse-order and re-running
  [`## configure`](#configure) with corrected parameters.
- **performance tuning.** Per-context completion vectors,
  NUMA-aware PE placement, multi-thread PE topologies, high-rate
  log emission with async sinks — touched by this skill but owned
  by the per-library skill (for completion vectors and queue
  placement) and by the user's own application design (for PE
  topology and sink batching). DOCA Log itself is a synchronous
  primitive; performance-shaped concerns are owned by the
  custom-sink implementation, not by the DOCA Log API surface.
- **service-side logging.** Configuring log destinations for DOCA
  *services* (DMS, DTS, Firefly, …) is owned by the matching
  service skill, not by this library skill. This skill is the
  primitive the user wires into *their own* DOCA app.

## Command appendix

Every command below is **cross-cutting on doca-common** — it
answers a recurring class of question that comes up in the verbs
above. The agent should treat the *class* as load-bearing; the
worked example is a single instance. Run-as user is the
unprivileged user unless noted.

**Infra-aware preamble (every row below).** Per the bundle's
detect → prefer → fall back → report contract documented in
[`doca-structured-tools-contract ## The agent behavior contract`](../../doca-structured-tools-contract/SKILL.md#the-agent-behavior-contract),
the agent should:

1. Probe for the matching structured helper FIRST (`doca-env
   --json` for version + devices + libraries in one shot;
   `doca-capability-snapshot` for per-device capability flags;
   `version-matrix.json` for *"available since"* lookups).
2. If the probe succeeds, the structured tool's output is the
   authoritative answer and the agent SHOULD NOT also run the
   manual command in the row below. Report *"using structured
   `<tool>`"*.
3. If the probe fails, fall back to the manual command in the row.
   Report *"falling back to manual chain"*.
4. The schemas the structured tools emit are defined in
   [`doca-structured-tools-contract ## Schemas`](../../doca-structured-tools-contract/SKILL.md#schemas);
   the version-handling semantics (four-way match, NGC,
   headers-win) are owned by
   [`doca-version`](../../doca-version/SKILL.md).

| Command (worked example) | Owning step | Class of question it answers | What healthy output looks like |
| --- | --- | --- | --- |
| `pkg-config --modversion doca-common` | [`## install`](#install) check 1; [`## configure`](#configure) step 1; [`## build`](#build) | What is the build-time doca-common version on this install? | A semver string matching `doca_caps --version` |
| `pkg-config --cflags --libs doca-common` | [`## build`](#build) | What include + link flags does the linker need for any DOCA app? | `-I` paths under $(pkg-config --variable=includedir doca-common); `-l` line including `-ldoca_common` |
| `pkg-config --exists doca-log; echo $?` | [`## build`](#build); [`## log`](#log) | Does this install publish a standalone `doca-log.pc`, or is DOCA Log folded into `doca-common.pc`? | Exit 0 means standalone; exit nonzero means use `doca-common` for the log surface too |
| `doca_caps --version` | [`## install`](#install) check 2; [`## debug`](#debug) step 6 | What is the *runtime* DOCA version on this host? | A semver string matching `pkg-config --modversion doca-common` |
| `doca_caps --list-devs` | [`## configure`](#configure) step 2; [`## use`](#use) step 1 | Which devices on this host can be used as a `doca_dev`? | One row per visible device with PCIe address and capability flags |
| `ls "$(pkg-config --variable=includedir doca-common)/"` | [`## configure`](#configure); [`## modify`](#modify) | Which Common headers are installed (the headers-win-over-docs anchor)? | A list including `doca_buf.h`, `doca_ctx.h`, `doca_dev.h`, `doca_pe.h`, `doca_log.h`, `doca_mmap.h`, `doca_error.h` |
| `<binary> --sdk-log-level WARNING 2> doca.log` | [`## run`](#run) step 2 | What does the SDK tier emit at WARNING (production default)? | Mostly silent on a healthy run; warnings are rare and load-bearing when they fire |
| `<binary> --sdk-log-level DEBUG 2> doca.log` | [`## test`](#test) | What does the SDK tier emit at DEBUG (DOCA library internal trace)? | A flood of per-call DOCA library internal log lines; the user's own DEBUG lines remain controlled by the app tier and are *not* affected by this flag |
| `DOCA_LOG_LEVEL_SDK=DEBUG <binary> 2> doca.log` | [`## run`](#run) step 2 | Same as above via env var (for samples that don't accept `--sdk-log-level`) | Same shape |
| `grep -RH 'doca_log_register_source\|DOCA_LOG_INFO' /opt/mellanox/doca/samples/ \| head` | [`## modify`](#modify); [`## log`](#log) | Which shipped samples have canonical DOCA Log usage to read as a reference? | Multiple hits across `samples/<library>/<sample>/*_main.c` — DOCA Log is in every sample |
| `cat /opt/mellanox/doca/applications/VERSION` | [`## install`](#install) check 2; [`## debug`](#debug) step 6 | What does the install tree itself claim its version is? | A semver string matching the other version sources |

For commands shared across libraries (`pkg-config --modversion`,
`doca_caps`, `cat /opt/mellanox/doca/applications/VERSION`,
`DOCA_LOG_LEVEL` / `DOCA_LOG_LEVEL_SDK`) the cross-library overlay
is in
[`doca-debug TASKS.md ## Command appendix`](../../doca-debug/TASKS.md#command-appendix);
this table adds the doca-common-specific rows on top.

Three cross-cutting rules for this appendix:

- **Never invent Common symbols or paths.** The Common ABI is
  large and version-gated; `ls
  $(pkg-config --variable=includedir doca-common) ` on the user's install
  and the installed `doca_*.h` headers are the only safe sources.
- **Never paraphrase a Common `DOCA_ERROR_*`.** Quote
  `doca_error_get_descr()` verbatim — the layer-classifier in
  [`doca-debug ## debug`](../../doca-debug/TASKS.md#debug) layer 5
  needs the exact text.
- **Cross-link instead of duplicate.** Cross-cutting commands (the
  read-only triple, `dmesg`, `lspci`, `mlxconfig -d <pcie> q`) live
  in
  [`doca-debug TASKS.md ## Command appendix`](../../doca-debug/TASKS.md#command-appendix);
  this appendix names only the doca-common-specific ones.
