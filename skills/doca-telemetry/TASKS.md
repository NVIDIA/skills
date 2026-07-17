# DOCA Telemetry per-domain hardware-counter-reader workflows

This library is the **per-domain hardware-counter READER** half
of DOCA telemetry. It exposes six independent per-domain reader
sub-libraries — `doca_telemetry_pcc` / `_dpa` / `_diag` /
`_adp_retx` / `_phy` / `_pci` — each with its own header, its own
opaque context type, and its own `doca_telemetry_<domain>_*` C
API. There is **no** NetFlow / IPFIX / local-socket collector
surface, **no** schema-registration, and **no** socket / port /
publisher to configure. Each domain is read directly off an
already-open `doca_dev` via cap-query → `_create` → per-domain
setters → `_start` → per-domain read → `_stop` → `_destroy`. The
publishing / export side is the sibling
[`doca-telemetry-exporter`](../doca-telemetry-exporter/SKILL.md)
library — a separate skill.

**Where to start:** The verbs run `configure → build → modify →
run → test → debug`. Skip ahead only when the user is already
past a verb. The `## test` verb is an iterative loop (cap-query
sanity → single per-domain read smoke → multi-read cadence →
under-load sample-window behavior → loop back if the device or
domain set changes), not a one-shot pass — see the eval-loop
overlay in `## test` below.

Read this file when the loader sent you here from
[SKILL.md](SKILL.md). For the per-domain sub-libraries, the
reader-vs-exporter role split, the per-domain lifecycle on a
`doca_dev`, the per-domain capability-query rule, the error
taxonomy (including the `NOT_SUPPORTED`-means-domain-not-exposed
rule and the `AGAIN`-means-snapshot-not-ready rule),
observability, and safety policy, see
[CAPABILITIES.md](CAPABILITIES.md). For where to find docs, the
installed DOCA layout, or release notes, route through
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

Each verb below describes the **shape of the workflow**, not a
copy-paste recipe. The agent's job is to walk the user through
the steps in order, verifying preconditions before recommending
the next call.

## configure

Goal: pick the right per-domain reader sub-library, confirm the
device exposes it, and stand up the per-domain context on an
already-open `doca_dev` — before any counter read.

Steps the agent should walk the user through:

1. **Confirm the role: this is the READER (consume-from-device)
   side.** Before any code change, surface the
   reader-vs-exporter distinction per the role-split table in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes).
   `doca-telemetry` is what the user's application links to
   **read** hardware counters off a `doca_dev`; the publishing /
   export side is
   [`doca-telemetry-exporter`](../doca-telemetry-exporter/SKILL.md),
   a separate sibling library. An agent that walks the user
   toward the exporter skill when they wanted to read counters
   is wrong; an agent that recommends linking this reader when
   the user actually wanted to publish is wrong. State the role
   first, before any `pkg-config` mention or any code sketch.
2. **Pick the per-domain sub-library.** Per the sub-library
   table in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes),
   map the counters the user cares about to exactly one domain:
   PCC (`doca_telemetry_pcc.h`), DPA (`_dpa.h`), device
   diagnostics (`_diag.h`), adaptive-retransmit histogram
   (`_adp_retx.h`), physical layer (`_phy.h`), or PCI / PCIe
   (`_pci.h`). Each is a separate header and a separate context
   type — there is no single "telemetry" context that reads all
   of them. If the user needs counters from more than one
   domain, that is more than one per-domain context in the same
   application.
3. **Confirm the installed DOCA version and cap-query the
   domain.** Use the procedure in
   [`doca-version TASKS.md ## configure`](../../doca-version/TASKS.md#configure).
   Quote the version observed (`pkg-config --modversion
   doca-telemetry`, then `doca_caps --version`); do not assume
   "latest". Then call the matching per-domain capability query
   against the active `doca_devinfo`:
   `doca_telemetry_<domain>_cap_is_supported(devinfo)` for
   `pcc` / `dpa` / `diag` / `adp_retx` / `phy`. **For `phy`,
   also query the per-sub-area caps** (e.g.
   `doca_telemetry_phy_cap_counter_and_ber_info_is_supported`)
   for the specific sub-area you intend to read. **For `pci`
   there is NO single domain-level cap** — query the per-feature
   caps (`doca_telemetry_pci_cap_management_info_is_supported`,
   `_cap_perf_counters_1_is_supported`,
   `_cap_latency_histogram_is_supported`) for the PCI counter
   family you intend to read. Per the capability-query rule in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes),
   the queried value is the runtime authority, not the agent's
   memory. A `DOCA_ERROR_NOT_SUPPORTED` here is the answer (the
   device does not expose this domain), not a bug. Quote the
   values back to the user.
4. **Create the per-domain context on the `doca_dev`.** Call
   `doca_telemetry_<domain>_create(dev, &ctx)` against the
   already-open `doca_dev` (for the PCC representor path, use
   `doca_telemetry_pcc_rep_create(dev_rep, &ctx)`). The
   `doca_dev` must already be open — opening it is the
   `doca-common` / device-discovery concern, not this skill's.
5. **Apply the per-domain configuration knobs (domain-specific,
   optional for some domains).** Set only what the domain
   exposes — there is no transport / socket / schema to
   configure:
   - `diag`: set the sample mode
     (`doca_telemetry_diag_set_sample_mode`), sample period
     (`_set_sample_period`), and max num samples
     (`_set_log_max_num_samples`); then
     `doca_telemetry_diag_apply_config` and select the counter
     IDs with `doca_telemetry_diag_apply_counters_list_by_id`
     BEFORE start.
   - `adp_retx`: set the histogram shape (`_set_hist_num_bins`,
     `_set_hist_bin0_width`, `_set_hist_time_unit`,
     `_set_hist_clear_on_read`, optionally `_set_hist_vhca_id`),
     bounded by the `_cap_get_hist_max_bins` /
     `_cap_get_hist_time_units` caps from step 3.
   - `dpa`: optionally `_set_max_perf_event_samples` before
     reading perf-event lists.
   - `pcc` / `phy` / `pci`: typically no pre-start setters — read
     directly after start.
6. **Start the per-domain context.** Call
   `doca_telemetry_<domain>_start(ctx)`. Reads before start
   return `DOCA_ERROR_BAD_STATE`. For `diag`, start is only
   valid after `_apply_config`. If start fails, route through
   the error taxonomy in
   [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
   before retrying.

If any step fails with a `DOCA_ERROR_*`, route through the error
taxonomy in
[`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
before retrying. In particular, `DOCA_ERROR_NOT_SUPPORTED` on
the cap-query or first read is *not* a configure-time bug — it is
the canonical *"this device does not expose this counter
domain"* signal, and the correct response is to surface it, not
to retry.

## build

Goal: produce a reader binary that links DOCA Telemetry against
the user's installed DOCA, using the canonical cross-library
build pattern.

The build pattern for any DOCA C/C++ consumer is **identical**
across libraries — `pkg-config` for include + link flags, meson
or CMake as the build system — and is fully documented in
[`doca-programming-guide TASKS.md ## build`](../../doca-programming-guide/TASKS.md#build).
This skill carries only the reader-specific overlay:

| Slot | Value for the reader | Why it matters |
| --- | --- | --- |
| `pkg-config` module name | `doca-telemetry` | The reader's `.pc` file installed by the DOCA host packages. **Wrong module name = wrong direction** — `doca-telemetry-exporter` is the SIBLING publisher library, with its own `.pc` and its own skill at [`doca-telemetry-exporter`](../doca-telemetry-exporter/SKILL.md). Picking `doca-telemetry-exporter` when the user wanted to read counters (or `doca-telemetry` when the user wanted to publish) is the load-bearing first-app failure, NOT a typo — re-check the role per [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes) |
| Header to `#include` | The per-domain header for the chosen domain: `doca_telemetry_pcc.h` / `_dpa.h` / `_diag.h` / `_adp_retx.h` / `_phy.h` / `_pci.h` | Each domain is a separate header. Including the wrong domain's header gets the wrong symbol family. Resolve under `$(pkg-config --variable=includedir doca-common)` |
| Include flags | `pkg-config --cflags doca-telemetry` | Resolves to the telemetry headers on this install |
| Link flags | `pkg-config --libs doca-telemetry` | Pulls in whatever `pkg-config --libs` resolves on this install (do not predict the `-l<name>` form by hand — `.so` basenames use underscores, `.pc` names use hyphens, and `pkg-config` is the only correct translator) plus the transitive set the resolver computes |
| Minimum required DOCA version | Query with `pkg-config --modversion doca-telemetry`; never hardcode in build files | The per-domain counter set grows across releases; cross-version build / runtime mixing breaks per [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility) |

For non-C consumers (Rust, Go, Python), the link surface is the
same `*.so` files; the FFI wrapper layer is the language-specific
binding and is out of scope for this skill — but the slots above
are still the load-bearing inputs the wrapper needs.

## modify

Goal: take a shipped DOCA Telemetry reader sample as the
verified starting point and apply a **minimum-diff
modification** to express the user's intent.

The universal modify-a-shipped-sample workflow lives in
[`doca-programming-guide TASKS.md ## modify`](../../doca-programming-guide/TASKS.md#modify).
Use it as-is. The reader-specific overlay is the *modify-from-
sample contract fill* — the slots the agent must elicit from the
user before recommending any code-level edit:

| Slot | What the agent asks the user | Reader-specific consideration |
| --- | --- | --- |
| 1. Starting sample | Which sample under `/opt/mellanox/doca/samples/doca_telemetry/`? | Pick the sample for the SAME domain the user picked in [`## configure`](#configure) step 2 (e.g. the PHY sample for PHY counters). A smaller diff is always safer than a re-architecture across domains |
| 2. Domain + counters | Which per-domain sub-library, and which specific counters / sub-areas within it? | Re-validate against the per-domain cap-query per [`## configure`](#configure) step 3; a counter family present on one install / device may return `NOT_SUPPORTED` on another |
| 3. Sample shape (sampled domains) | For `diag` / `adp_retx`: what sample mode, sample period, and num-samples / histogram-bin shape? | Bound every setter by the `_cap_*` sizing caps from [`## configure`](#configure) step 3; a value past the cap returns `DOCA_ERROR_INVALID_VALUE`. For `diag`, the `_apply_config` + `_apply_counters_list_by_id` calls MUST stay before `_start` |
| 4. Read cadence + `AGAIN` behavior | How often does the modified reader read, and what does it do when a read returns `DOCA_ERROR_AGAIN`? | Per [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy) sample-window rule, the retry must respect the sample window (for diag, wait for the previous sampling cycle); a tight `AGAIN` spin is a sample gap that needs editing out before the modify lands |
| 5. Clear-on-read intent | Does the modified reader enable clear-on-read (`adp_retx` `_set_hist_clear_on_read`, `diag` data-clear)? | Per [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy), clear-on-read resets the underlying counters — a second reader sees the counters as they stand after the reset. Decide explicitly; default to NOT clearing unless the user wants destructive reads |
| 6. Build manifest | Keep the sample's existing `meson.build` (which already wires `pkg-config doca-telemetry`)? | Yes. Do not switch to a hand-rolled Makefile for *"simplicity"* — it removes the version-check rail. And do not silently swap the `pkg-config` module to `doca-telemetry-exporter` — that flips the role and is the load-bearing first-app failure |

The agent emits an *intent description + the filled slots*; the
*actual* unified diff against the sample source is produced by
the modify-from-sample renderer (deferred to a future round, per
[`doca-programming-guide TASKS.md ## modify`](../../doca-programming-guide/TASKS.md#modify)).
Until the renderer ships, the agent must walk the user through
the diff line-by-line against the sample source they read on
disk, and have the user paste back the result for validation.

## run

Goal: actually execute the built reader against the user's
installed DOCA, against an open `doca_dev`, reading the chosen
domain's counters.

Steps the agent should walk the user through:

1. **Open the `doca_dev` and cap-query the domain on it.** The
   reader's first job is to open the target device and confirm
   `doca_telemetry_<domain>_cap_is_supported(devinfo)` (or the
   `pci` per-feature caps) returns `DOCA_SUCCESS`. A
   `DOCA_ERROR_NOT_SUPPORTED` here means the device does not
   expose this domain — fix the device selection or the domain
   choice, not the read call.
2. **Run as a user with privilege for the counter domain
   (typically NOT blanket sudo).** Per
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy),
   some diagnostic / PHY / PCI counters require elevated device
   access. A `DOCA_ERROR_NOT_PERMITTED` on create / start / read
   means the running user lacks the specific privilege — grant
   it on the env side rather than reflexively running the whole
   app as root.
3. **Create → (configure) → start → read once.** Walk the
   per-domain lifecycle from [`## configure`](#configure):
   create on the `doca_dev`, apply any domain setters, start,
   then issue ONE read (e.g. `doca_telemetry_phy_get_counter_and_ber_info`,
   `doca_telemetry_pcc_get_counters`,
   `doca_telemetry_diag_query_counters`,
   `doca_telemetry_pci_read_perf_counters_1`,
   `doca_telemetry_dpa_read_cumul_info_list`). Confirm the read
   returns `DOCA_SUCCESS` and the output struct is populated
   before any loop.
4. **Capture the structured log.** Set `DOCA_LOG_LEVEL=trace`
   for the first run (see
   [`doca-debug CAPABILITIES.md ## Observability`](../../doca-debug/CAPABILITIES.md#observability)).
   This is the cheapest way to make the per-domain lifecycle
   transitions and the first failing read visible.
5. **Handle `DOCA_ERROR_AGAIN` with sample-window-aware
   retries, not a spin.** `AGAIN` on a read means the snapshot /
   sample cycle is not ready yet. Retry after the documented
   sample window (for `diag`, after the previous sampling cycle
   completes — see the note on `doca_telemetry_diag_query_counters`),
   per [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy).
6. **Stop and destroy on teardown.** Call
   `doca_telemetry_<domain>_stop` then
   `doca_telemetry_<domain>_destroy` in reverse-create order.

## test

Goal: prove the per-domain reader can actually read the chosen
counters off the device, end-to-end, before claiming the
*"build a first counter-reading app"* journey is done.

This is **a loop, not a one-shot pass.** Each iteration narrows
either the domain / sub-area selection, the cap-query result,
the sample-window behavior, or the per-read return. The loop
terminates when either (a) the reader reads the intended
counters off the device with the expected values and the
sample-window behavior matches the policy decided in
[`## modify`](#modify) slot 4, or (b) the agent has narrowed the
failure to a layer outside the reader itself (device / driver /
privilege) and escalated to the matching skill.

Iteration shape:

1. **Lifecycle smoke.** Create the per-domain context on the
   `doca_dev`, `_start`, and confirm both succeed before any
   read. A `DOCA_ERROR_BAD_STATE` here means a lifecycle
   ordering bug (read before start, or — for diag — start before
   `_apply_config`); a `DOCA_ERROR_NOT_PERMITTED` means a
   privilege gap. Validates the lifecycle BEFORE chasing counter
   values.
2. **Capability re-check.** Re-run the per-domain
   `doca_telemetry_<domain>_cap_is_supported(devinfo)` (and, for
   `phy` / `pci`, the per-sub-area / per-feature caps). If the
   sub-area / counter family the user wants returns
   `NOT_SUPPORTED`, that *is* the answer for this device + install;
   update the domain / sub-area selection (or the device) before
   continuing.
3. **Single-read smoke.** Issue ONE per-domain read and confirm
   it returns `DOCA_SUCCESS` with a populated output struct. If
   the read returns `DOCA_ERROR_AGAIN`, the snapshot is not ready
   — retry after the sample window, not in a tight loop. If it
   returns `NOT_SUPPORTED`, the sub-area is not exposed (back to
   step 2).
4. **Value-sanity pass.** Confirm the values read are plausible
   for the device's state (e.g. PHY BER non-negative, PCC
   per-algo counters consistent with the active algo slots,
   diag samples within the configured num-samples). Implausible
   values point at the wrong sub-area / wrong output-format
   interpretation, not a library bug.
5. **Multi-read cadence.** Loop a small N reads at the intended
   cadence with the sample-window discipline; confirm each read
   returns `DOCA_SUCCESS` (or a sample-window `AGAIN` that
   resolves on the next windowed retry) and the values evolve as
   expected. For clear-on-read configs, confirm the post-clear
   semantics match intent (per [`## modify`](#modify) slot 5).
6. **Under-load behavior.** If the user reads under device load,
   confirm the read cadence still converges and that `AGAIN`
   frequency tracks the sample window rather than a stuck cycle.

Eval-loop overlay — why this is a loop, not a one-shot pass:

| Iteration trigger | What it looks like | What changes next iteration |
| --- | --- | --- |
| `DOCA_ERROR_NOT_SUPPORTED` on cap-query or read | The device does not expose this domain / sub-area on this install | Re-pick the domain / sub-area per [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes), or confirm the right device is selected; this is the cap answer, not a retry case |
| `DOCA_ERROR_BAD_STATE` on read | Read before `_start`, or (diag) start before `_apply_config` | Fix the per-domain lifecycle ordering per [`## configure`](#configure); re-run the lifecycle smoke |
| `DOCA_ERROR_AGAIN` on read | Snapshot / sample cycle not ready yet | Retry after the sample window (diag: after the previous cycle); widen the read interval if a tight loop keeps hitting `AGAIN` |
| `DOCA_ERROR_INVALID_VALUE` on a setter or read | A sample / histogram value past the install's cap, or an undersized output buffer | Re-read the `_cap_get_*` sizing query and size the value / buffer to it |
| Same code reads on device A, returns NOT_SUPPORTED on device B | Different device family / firmware feature bits, or different DOCA version | Re-narrow to per-device cap-query; the reader behavior is the same, the variance is at the device / version layer |

Loop termination: stop iterating once two consecutive iterations
of the same kind don't change anything — that means the cause is
below the reader (device / driver / firmware feature gating).
Escalate to
[`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug)
with the captured cap-query + per-read evidence and the device /
version state.

## debug

Goal: when a DOCA Telemetry per-domain read returns a
`DOCA_ERROR_*` (or returns implausible values), narrow the cause
to a specific layer and act on it.

The cross-library debug ladder lives in
[`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug).
Walk through it in order — install → version → build → link →
runtime → program → driver — *before* recommending reader-
specific fixes. This skill's overlay names the reader-specific
manifestation at layers 5 (runtime) and 6 (program):

**Layer 5 (runtime) — reader overlay.**

- Walk the role rule: did the user actually want the reader and
  not the exporter? If the user is reading guides about emitting
  / publishing values, route them to
  [`doca-telemetry-exporter`](../doca-telemetry-exporter/SKILL.md)
  via the role-split table in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes),
  not to debug the reader further.
- Walk the cap-query rule: a `DOCA_ERROR_NOT_SUPPORTED` on
  cap-query or read means the device does not expose this domain
  / sub-area on this install. This is the answer, not a bug —
  confirm the device + domain selection per
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes).
- Walk the privilege state: a `DOCA_ERROR_NOT_PERMITTED` on
  create / start / read means the running user lacks privilege
  for this specific counter domain — grant it on the env side,
  resist reflexive global `sudo`.

**Layer 6 (program) — reader overlay.**

- Lifecycle order: cap-query → create → (configure / for diag
  `apply_config` + `apply_counters_list_by_id`) → start → read →
  stop → destroy. Out-of-order returns `DOCA_ERROR_BAD_STATE`.
  The most common case is reading before `_start`, or reading
  diag before `_apply_config`.
- Sample-window discipline: a `DOCA_ERROR_AGAIN` on read is the
  snapshot / sample cycle not being ready — the fix is a
  sample-window-aware retry, NOT a tight spin. For diag, the
  next `_query_counters` must wait for the previous sampling
  cycle to finish.
- Output-format interpretation: implausible counter values are
  usually the wrong sub-area read or the wrong output-format
  struct interpreted (e.g. diag format_0 vs format_1 vs
  format_2). Re-read the per-domain header's output-struct
  definition; the values are the device's, the interpretation is
  the program's.
- Buffer / range sizing: a `DOCA_ERROR_INVALID_VALUE` on a
  setter or read is a value past the install's cap or an
  undersized output buffer. Re-read the matching `_cap_get_*`
  sizing query; the fix is at the call site, not by widening a
  cap (it is device-bound).

Once the layer is identified, route to the matching debug verb
on the matching skill: install / build / link / driver to
[`doca-setup ## debug`](../../doca-setup/TASKS.md#debug);
version to [`doca-version ## debug`](../../doca-version/TASKS.md#debug);
cross-cutting runtime to
[`doca-debug ## debug`](../../doca-debug/TASKS.md#debug);
program-layer Core patterns to
[`doca-programming-guide TASKS.md ## debug`](../../doca-programming-guide/TASKS.md#debug);
publishing-side concerns to
[`doca-telemetry-exporter ## debug`](../doca-telemetry-exporter/TASKS.md#debug).

## Deferred task verbs

The following verbs are out of scope for this skill but are
commonly asked in the same conversations. Route them as follows
so the agent does not invent guidance:

- **install.** Installing DOCA, choosing packages, post-install
  verification, `pkg-config` wiring — defer to
  [`doca-setup`](../../doca-setup/SKILL.md) and to the
  install-tree layout in
  [doca-public-knowledge-map ## Layout of an installed DOCA package](../../doca-public-knowledge-map/SKILL.md#layout-of-an-installed-doca-package).
  This skill assumes DOCA is already installed and a `doca_dev`
  is open.
- **publish / export telemetry.** Wiring up the application-side
  publishing of labeled metrics / OTLP logs — out of scope for
  this skill. Route to
  [`doca-telemetry-exporter`](../doca-telemetry-exporter/SKILL.md)
  for the sibling publisher library. This skill is reader-side
  only.
- **operate the DOCA Telemetry Service (DTS) itself.** DTS is a
  separate, externally-productized DOCA service with its own
  public guide and is out of scope for this bundle; reach it via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
  non-goals.
- **stand up a NetFlow / IPFIX / socket collector.** The
  per-domain reader libraries do not expose a collector /
  schema-transport surface. Route to a generic collector outside
  the DOCA family, to
  [`doca-telemetry-exporter`](../doca-telemetry-exporter/SKILL.md)
  for publishing, or to DTS (out of scope) for productized
  aggregation.
- **deploy.** Deploying counter-reading applications at scale
  across many hosts, Kubernetes operator workflows — out of
  scope for Phase 1 and reserved for a future platform skill.
- **firmware burn / reset.** The reader does not depend on a
  firmware-burn step directly; if the debug ladder lands on a
  driver / firmware-feature-gating issue
  (`DOCA_ERROR_IO_FAILED` or persistent `NOT_SUPPORTED`), the
  fix is via the env-side skill:
  [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug) layer
  5, then upstream documentation reachable through
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

## Command appendix

Every command below is **cross-cutting on DOCA Telemetry
(per-domain reader)** — it answers a recurring class of question
that comes up in the verbs above. The agent should treat the
*class* as load-bearing; the worked example is a single
instance. Run-as user is the reader application's normal user
unless noted. Rows that need elevated privileges call that out explicitly. (and is needed only for
counter domains that require elevated device access, per
[`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)).

**Infra-aware preamble (every row below).** Per the bundle's
detect → prefer → fall back → report contract documented in
[`doca-structured-tools-contract ## The agent behavior contract`](../../doca-structured-tools-contract/SKILL.md#the-agent-behavior-contract),
the agent should:

1. Probe for the matching structured helper FIRST
   (`doca-env --json` for version + devices + libraries +
   drivers + hugepages in one shot; `doca-capability-snapshot`
   for per-device capability flags; `version-matrix.json` for
   *"available since"* lookups).
2. If the probe succeeds, the structured tool's output is the
   authoritative answer and the agent SHOULD NOT also run the
   manual command in the row below. Report *"using structured
   `<tool>`"*.
3. If the probe fails, fall back to the manual command in the
   row. Report *"falling back to manual chain"*.
4. The schemas the structured tools emit are defined in
   [`doca-structured-tools-contract ## Schemas`](../../doca-structured-tools-contract/SKILL.md#schemas);
   the version-handling semantics (four-way match, NGC,
   headers-win) are owned by
   [`doca-version`](../../doca-version/SKILL.md).

| Command (worked example) | Owning step | Class of question it answers | What healthy output looks like |
| --- | --- | --- | --- |
| `pkg-config --modversion doca-telemetry` | `## configure` step 3; `## build` slot | What is the build-time DOCA Telemetry (reader) version? | A semver string matching `doca_caps --version`. Disagreement = partial install (route to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug) layer 2). If the command returns *"Package 'doca-telemetry' was not found"* and the user actually wanted the publisher, route to [`doca-telemetry-exporter`](../doca-telemetry-exporter/SKILL.md) — wrong direction is the load-bearing first-app failure, not a typo |
| `pkg-config --cflags --libs doca-telemetry` | `## build` | What include + link flags does the linker need? | Trust whatever `pkg-config --cflags --libs` produces on this install. Do not hardcode either the `-I` include path or the `-l<name>` flag form — both can drift between DOCA install profiles and DOCA majors; the on-disk `.so` basenames use underscores while the `.pc` package names use hyphens, and `pkg-config` is the only thing that resolves both correctly |
| `ls /opt/mellanox/doca/samples/doca_telemetry/` | `## modify` slot 1 | Which reader samples ship in this install, and which is the closest starting point? | A list of sample directories named after the per-domain reader they demonstrate |
| `doca_caps --version` | `## configure` step 3; `## test` step 2 | What is the *runtime* DOCA version? | A semver string matching `pkg-config --modversion doca-telemetry` |
| `id` | `## run` step 2 | Does the running user have the privilege the counter domain requires? | The user's id has the device-access privilege the domain needs. Mismatch = `DOCA_ERROR_NOT_PERMITTED` on create / start / read — grant the specific privilege on the env side, not blanket sudo |
| `cat /opt/mellanox/doca/applications/VERSION` | `## configure` step 3; `## debug` layer 1 | What does the install tree itself claim its version is? | A semver string matching the other two version sources |
| `dmesg \| tail -n 40` (sudo) | `## debug` layer 7 | What did the kernel / driver log around the last reader call? | Empty or recent benign messages. Repeated mlx5 / firmware / device errors → driver / env-layer bug; route to [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug) |
| `DOCA_LOG_LEVEL=trace ./<binary>` | `## run` step 4 | What did the structured DOCA logger emit for the first failing read? | A trace-level line on every per-domain lifecycle transition and every read. Per-read `AGAIN` traces = snapshot not ready — apply a sample-window-aware retry, not a blind spin |

For commands shared across libraries (`pkg-config
--modversion`, `doca_caps`, `cat /opt/mellanox/doca/applications/VERSION`,
`DOCA_LOG_LEVEL`) the cross-library overlay is in
[`doca-debug TASKS.md ## Command appendix`](../../doca-debug/TASKS.md#command-appendix);
this table adds the per-domain reader rows on top.
