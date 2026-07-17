# DOCA Telemetry capabilities, version overlay, errors, observability, safety

**Where to start:** Pick the H2 anchor that matches your
question (reader-vs-exporter split / per-domain sub-libraries /
per-domain lifecycle / capability discovery / path selection /
version / errors / observability / safety) and read that section
end-to-end. The tables in each section are the load-bearing
content; the prose around them is interpretation.

This library is the **per-domain hardware-counter READER** half
of DOCA telemetry. It is **not** a NetFlow / IPFIX / local-socket
*collector* framework, and there is no schema-registration /
event-transport / publisher-consumer surface in the public
header. Each domain ships its own small `doca_telemetry_<domain>_*`
C API: a capability query on a `doca_devinfo`, a context created
on a `doca_dev`, a per-domain `_start`, per-domain read / sample
calls, then `_stop` / `_destroy`. The publishing / export side is
the sibling [`doca-telemetry-exporter`](../doca-telemetry-exporter/SKILL.md)
library — a separate skill.

Read this file when the loader sent you here from
[SKILL.md](SKILL.md). For the *how* of executing each pattern
(the verbs `configure / build / modify / run / test / debug`),
jump to [TASKS.md](TASKS.md). For the canonical DOCA
version-handling rules that this skill layers a per-domain
overlay on top of, see [`doca-version`](../../doca-version/SKILL.md).

## Pattern overview

Every hardware-counter-reader question this skill teaches
resolves into one of SIX patterns. The patterns are CLASSES —
they apply across every DOCA release and every per-domain reader
sub-library, not just the worked examples shown.

| Pattern | When it applies (class shape) | Where the substance lives |
| --- | --- | --- |
| 1. Pick the reader, not the exporter | The application wants to **read** hardware counters off a `doca_dev`; the publishing / export side is a separate library out of this skill's scope | [`## Capabilities and modes`](#capabilities-and-modes) role-split table |
| 2. Pick the right per-domain sub-library | The counters the user cares about live in exactly one of the six domains (`pcc` / `dpa` / `diag` / `adp_retx` / `phy` / `pci`); each is a separate header + context type | [`## Capabilities and modes`](#capabilities-and-modes) sub-library table |
| 3. Cap-query the domain BEFORE creating the context | `doca_telemetry_<domain>_cap_is_supported(devinfo)` (or, for `pci`, the per-feature `_cap_*_is_supported`) is the runtime authority for whether this device exposes this domain on this install | [`## Capabilities and modes`](#capabilities-and-modes) capability-query rule + [TASKS.md ## configure](TASKS.md#configure) step 3 |
| 4. Stand up the per-domain context + read loop | Per-domain lifecycle: cap-query → `_create(doca_dev)` → per-domain configure setters → `_start` → per-domain read / sample → `_stop` → `_destroy` | [`## Capabilities and modes`](#capabilities-and-modes) lifecycle table + [TASKS.md ## configure](TASKS.md#configure) |
| 5. Respect the per-domain sample window | A read may return `DOCA_ERROR_AGAIN` (snapshot / sample cycle not ready yet); the diag domain explicitly requires the previous sampling cycle to finish before the next read | [`## Safety policy`](#safety-policy) sample-window rule + [`## Error taxonomy`](#error-taxonomy) `AGAIN` row |
| 6. Diagnose a per-domain reader error | Map symptom (`BAD_STATE`, `INVALID_VALUE`, `AGAIN`, `NOT_PERMITTED`, `NOT_SUPPORTED`, `IO_FAILED`) to root cause; in particular recognise `NOT_SUPPORTED` as "this device does not expose this counter domain" rather than a library bug | [`## Error taxonomy`](#error-taxonomy) + [TASKS.md ## debug](TASKS.md#debug) |

Two cross-cutting rules that apply to *every* pattern above:

- **The reader reads; the exporter publishes.** `doca-telemetry`
  is the library the user's application links to *read* hardware
  counters off a `doca_dev`. The application-side publishing of
  labeled metrics / OTLP logs is
  [`doca-telemetry-exporter`](../doca-telemetry-exporter/SKILL.md),
  a sibling library with its own skill. Wiring the exporter's
  API into a reader application (or this library's API into a
  publishing application) is the load-bearing first-app failure
  and the agent must surface the distinction BEFORE any
  code-level guidance.
- **Cap-query is the only authority for "is this domain
  supported here".** Each domain's support is bound to BOTH the
  DOCA version AND the active `doca_devinfo`. Never assert a
  domain or counter family is available from memory; the
  `doca_telemetry_<domain>_cap_is_supported(devinfo)` query (or
  the `pci` per-feature caps) is the runtime authority.

## Capabilities and modes

DOCA Telemetry exposes **six independent per-domain reader
sub-libraries**. Each is its own header, its own opaque context
type, and its own `doca_telemetry_<domain>_*` symbol family.
There is no single shared "telemetry context" object and no DOCA
Core progress-engine / task surface here — each domain owns its
own `_create` / `_start` / read / `_stop` / `_destroy` lifecycle
directly on a `doca_dev`.

**Role split — reader (this library) vs exporter (publisher).**
The two halves of DOCA telemetry are separate libraries, and the
confusion between them is the #1 first-app failure.

| Side | What it does | What it does NOT do | Where it lives |
| --- | --- | --- | --- |
| Reader (this library, `doca-telemetry`) | Reads **hardware counters** off a `doca_dev` through one of the six per-domain contexts: PCC / DPA / DIAG / ADP-RETX / PHY / PCI counter snapshots | Publish / export counter values to a monitoring pipeline; define a schema; bind a socket / port; consume events from another process | Linked INTO the user's application; reads the local device through the per-domain C API |
| Exporter (sibling library, `doca-telemetry-exporter`) | Application-side **publishing** of labeled metrics / structured telemetry / OTLP logs | Read hardware counters off a device | A separate library with its own skill at [`doca-telemetry-exporter`](../doca-telemetry-exporter/SKILL.md). Wiring its API into a reader application is the load-bearing first-app failure this skill exists to prevent |

**Per-domain sub-libraries.** The public surface is exactly
these six. The agent must not invent a seventh, and must not
invent a generic "telemetry collector" context. Exact symbol
names are install-bound — confirm against the user's installed
headers under `$(pkg-config --variable=includedir doca-common)`
per the headers-win-over-docs rule in
[`doca-version`](../../doca-version/SKILL.md).

| Header | Domain | Counter family it reads | Per-domain cap query |
| --- | --- | --- | --- |
| `doca_telemetry_pcc.h` | Programmable Congestion Control | Per-algo-slot info + the PCC counter set (`doca_telemetry_pcc_get_counters` / `_get_and_clear_counters`, `_get_num_counters`, `_get_counter_info`, `_get_algo_*`) | `doca_telemetry_pcc_cap_is_supported(devinfo)` (plus `_cap_rep_is_supported` for the representor path and `_cap_get_max_*` sizing caps) |
| `doca_telemetry_dpa.h` | DPA | DPA process / thread lists, cumulative info, perf-event samples, per-counter start/restart/stop (`doca_telemetry_dpa_read_processes_list`, `_read_thread_list`, `_read_cumul_info_list`, `_read_perf_event_list`, `_counter_start`/`_counter_restart`/`_counter_stop`) | `doca_telemetry_dpa_cap_is_supported(devinfo)` |
| `doca_telemetry_diag.h` | Device diagnostic counters | Sampled diagnostic-counter data in one of three formats (`doca_telemetry_diag_query_counters` after `_apply_config` + `_apply_counters_list_by_id` + `_start`), with sample-mode / sample-period / num-samples setters | `doca_telemetry_diag_cap_is_supported(devinfo)` (plus `_cap_get_max_num_data_ids`, `_cap_get_log_max_num_samples`, `_cap_is_data_clear_supported`, `_cap_is_sample_mode_supported`) |
| `doca_telemetry_adp_retx.h` | Adaptive-retransmit histogram | Configurable retransmit-latency histogram (bin widths, time unit, vHCA id, clear-on-read) read off the device | `doca_telemetry_adp_retx_cap_is_supported(devinfo)` (plus `_cap_histogram_is_supported`, `_cap_get_hist_max_bins`, `_cap_get_hist_time_units`) |
| `doca_telemetry_phy.h` | Physical layer | Operation info, supported info, troubleshooting info, module info, counter-and-BER info, FEC histogram, management-cable info (`doca_telemetry_phy_get_*_info`) | `doca_telemetry_phy_cap_is_supported(devinfo)` **plus per-sub-area caps** `doca_telemetry_phy_cap_operation_info_is_supported`, `_cap_counter_and_ber_info_is_supported`, `_cap_fec_histogram_info_is_supported`, `_cap_module_info_is_supported`, … (query the specific sub-area you intend to read) |
| `doca_telemetry_pci.h` | PCI / PCIe | Management info, perf-counters set 1, latency histogram (`doca_telemetry_pci_read_management_info`, `_read_perf_counters_1`, `_read_latency_histogram`, plus `_by_pci_addr` variants) | **No single domain-level cap.** Query the per-feature caps: `doca_telemetry_pci_cap_management_info_is_supported`, `_cap_perf_counters_1_is_supported`, `_cap_latency_histogram_is_supported` (plus finer per-counter variants like `_cap_perf_counters_1_fec_error_is_supported`) |

**Per-domain lifecycle.** Every domain follows the same shape;
the verbs are domain-specific (`doca_telemetry_<domain>_*`), not
the generic DOCA Core `doca_ctx_*` surface:

| Step | Call (phy shown; substitute the domain) | Notes |
| --- | --- | --- |
| 0. Cap-query | `doca_telemetry_phy_cap_is_supported(devinfo)` | Runtime authority for "this device exposes this domain". For `pci`, query the per-feature `_cap_*_is_supported` instead. Returns `DOCA_ERROR_NOT_SUPPORTED` when the device does not expose the domain — that is the answer, not a bug |
| 1. Create | `doca_telemetry_phy_create(struct doca_dev *dev, struct doca_telemetry_phy **phy)` | Creates the per-domain context bound to an already-open `doca_dev`. `pcc` additionally offers `doca_telemetry_pcc_rep_create(struct doca_dev_rep *, …)` for the representor path |
| 2. Configure | domain setters (e.g. `doca_telemetry_diag_set_sample_mode` / `_set_sample_period`; `doca_telemetry_adp_retx_set_hist_num_bins`; `doca_telemetry_dpa_set_max_perf_event_samples`) | Domain-specific knobs only — there is no transport / socket / schema to configure. `diag` requires `doca_telemetry_diag_apply_config` (and `_apply_counters_list_by_id`) before start |
| 3. Start | `doca_telemetry_phy_start(phy)` | Reads before start return `DOCA_ERROR_BAD_STATE` |
| 4. Read | `doca_telemetry_phy_get_counter_and_ber_info(...)` etc. | Per-domain read / sample calls. `diag` reads via `doca_telemetry_diag_query_counters`; `pci` via the `_read_*` calls; `dpa` via the `_read_*_list` calls |
| 5. Stop / destroy | `doca_telemetry_phy_stop(phy)` then `doca_telemetry_phy_destroy(phy)` | Out-of-order calls return `DOCA_ERROR_BAD_STATE` |

**Capability discovery — the only rule.** Before assuming a
domain (or, for `phy` / `pci`, a sub-area / feature) is on this
install + this device, call the matching
`doca_telemetry_<domain>_cap_is_supported(devinfo)` query (per
the cross-cutting cap-query rule in
[`doca-version CAPABILITIES.md ## Observability`](../../doca-version/CAPABILITIES.md#observability)).
The query is the runtime authority. Asserting *"this device
supports PHY FEC histograms"* from memory without the cap query
is the silent-fail case — when the device does not expose it,
the read returns `DOCA_ERROR_NOT_SUPPORTED` and the user has no
idea why. The agent MUST quote the queried values back to the
user. **`doca_telemetry_pci` has no single domain-level cap**;
query its per-feature `_cap_*_is_supported` functions for the
specific PCI counter family you intend to read.

**Path selection — reader vs the adjacent options.** The reader
is for reading DOCA hardware counters off a `doca_dev`. It is not
the answer for every "telemetry" question; walk this rule before
recommending reader setup.

| Use DOCA Telemetry (this skill) when … | Use a different primitive when … |
| --- | --- |
| The user is reading hardware counters (PCC / DPA / DIAG / ADP-RETX / PHY / PCI) off a `doca_dev` from inside their own application | The user wants to **PUBLISH / EXPORT** counter values to a monitoring pipeline (OTLP / Prometheus / labeled metrics) — that is [`doca-telemetry-exporter`](../doca-telemetry-exporter/SKILL.md), the sibling publisher library |
| The user needs the structured per-domain counter snapshot the device exposes through the public per-domain C API | The user wants plain structured stdout / per-line logging from their own program — use `doca-log`; the per-domain cap-query discipline is overhead they do not need |
| The user is consuming the device's own hardware counters | The user wants a turnkey, deployed aggregation service across a fleet — that is the externally-productized DOCA Telemetry Service (DTS), **out of scope** for this bundle (see [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) non-goals); or a generic NetFlow / IPFIX / Prometheus collector outside the DOCA family |

## Version compatibility

For the canonical DOCA version-detection chain, the four-way
match rule, NGC container semantics, and the headers-win-over-docs
rule, see [`doca-version`](../../doca-version/SKILL.md). The body
lives there; this skill does not duplicate it.

**The per-domain reader overlay** is:

- **Use `pkg-config --modversion doca-telemetry` as the
  build-time anchor.** Per
  [`doca-version TASKS.md ## configure`](../../doca-version/TASKS.md#configure),
  this MUST match the other version sources in the four-way
  match. The set of domains — and the per-counter / per-sub-area
  surface each domain exposes — is bound to BOTH the DOCA
  version AND the active `doca_devinfo`; agent-memory limits are
  not authoritative and MUST be replaced with a
  `doca_telemetry_<domain>_cap_is_supported` query at runtime.
- **The reader is distinct from the exporter across every
  release.** When the user reports *"I'm reading guides about
  doca-telemetry-exporter — is that this library?"*, the answer
  is no: that is the publishing side, with its own skill at
  [`doca-telemetry-exporter`](../doca-telemetry-exporter/SKILL.md).
  The two `.pc` files coexist on a normal install; using the
  wrong one for the user's intent is the load-bearing first-app
  failure and the agent must walk the role split (per
  [`## Capabilities and modes`](#capabilities-and-modes)) BEFORE
  picking a `pkg-config` module.
- **`doca-telemetry.pc` plus `doca-common.pc` must both match
  `doca_caps --version`** at the four-way-match check (per
  [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility)).
  A common partial-install pattern after a DOCA upgrade is that
  `doca-telemetry.pc` lingers from the previous release while
  `doca-common.pc` was refreshed; route to
  [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug)
  ladder step 2 before any reader-layer diagnosis.
- **New per-domain counters / sub-areas appear across
  releases.** A counter family or PHY sub-area present on a newer
  DOCA may be absent on the user's install; the per-domain
  cap-query is the only safe way to discover what *this* install
  exposes. Never hardcode the available counter set in a build
  file or assume it from a docs page.

## Error taxonomy

Per-domain-reader overlays on the cross-library `DOCA_ERROR_*`
taxonomy. The cross-library taxonomy itself lives in
[`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../../doca-programming-guide/CAPABILITIES.md#error-taxonomy);
the rows below are the *reader surface* meaning the agent must
disambiguate before falling back to the cross-library response.

| Error | Reader context where it shows up | Reader-specific cause |
| --- | --- | --- |
| `DOCA_ERROR_NOT_SUPPORTED` | `doca_telemetry_<domain>_cap_is_supported(devinfo)`; first read on a domain whose cap was not checked | **This device does not expose this counter domain** (or, for `phy` / `pci`, this sub-area / feature). Typical for `dpa` on non-DPA-shipped devices, or `phy` / `pcc` sub-areas gated on firmware feature bits. This is the cap-query answer, NOT a library bug — do not retry, surface it to the user |
| `DOCA_ERROR_BAD_STATE` | Read call before `doca_telemetry_<domain>_start`; setter after start where the domain does not allow late reconfig; (diag) read before `_apply_config` | Lifecycle violation. Walk the per-domain lifecycle in [`## Capabilities and modes`](#capabilities-and-modes); the most common case is reading before `_start`, or (diag) reading before `_apply_config` + `_start` |
| `DOCA_ERROR_AGAIN` | A per-domain read / sample call (notably `diag` and the sampled domains) | The hardware-counter snapshot / sample cycle is **not ready yet**. The correct response is to retry after the documented sample window (for diag, after the previous sampling cycle completes — see the explicit note on `doca_telemetry_diag_query_counters`), NOT to spin in a tight loop. See [`## Safety policy`](#safety-policy) sample-window rule |
| `DOCA_ERROR_INVALID_VALUE` | A read into an undersized output buffer; a setter passed an out-of-range value (e.g. `diag` log-num-samples beyond the cap, `adp_retx` bin count beyond `_cap_get_hist_max_bins`) | A size / range mismatch against the install's caps. Re-read the relevant `_cap_get_*` sizing query and size the buffer / value to it; do not widen blindly |
| `DOCA_ERROR_NOT_PERMITTED` | Context create / start; a privileged counter read | The running user lacks the privilege the device requires for this counter domain (some diagnostic / PHY counters require elevated access). Walk the [`## Safety policy`](#safety-policy) permission row; this is endpoint/privilege-specific, not a blanket "add sudo" |
| `DOCA_ERROR_IO_FAILED` | Context create; read | The device / driver layer below DOCA reported failure (firmware query failed, device went away). Capture state and route to env-class debug ([`doca-setup TASKS.md ## debug`](../../doca-setup/TASKS.md#debug)) — the layer below DOCA is the suspect, not the reader program |

The agent's rule: **never recommend a retry loop on
`DOCA_ERROR_*` without first identifying which row above is the
cause**. `NOT_SUPPORTED` is the cap answer (surface it, don't
retry); `AGAIN` wants a sample-window-aware retry, not a spin;
the others want investigation.

## Observability

The reader's observability surface is the per-read return value
plus the per-domain cap snapshot. There is no progress-engine
completion stream and no external event transport — visibility
comes from inspecting each read's `doca_error_t` and the
configure-time cap-query results.

Two primary signals the agent should reach for:

1. **Per-read return.** Every `doca_telemetry_<domain>_*` read /
   query call returns a `doca_error_t`. Inspect it on every
   call: success means the snapshot was read into the output
   struct; `DOCA_ERROR_AGAIN` means the snapshot / sample cycle
   is not ready (retry after the sample window per
   [`## Safety policy`](#safety-policy)); `DOCA_ERROR_NOT_SUPPORTED`
   means the device does not expose this domain / sub-area; any
   other `DOCA_ERROR_*` maps to a row in
   [`## Error taxonomy`](#error-taxonomy).
2. **Capability snapshot at configure time.** The output of
   every `doca_telemetry_<domain>_cap_*` query is a snapshot of
   *what this device exposes on this install* before any read.
   Save it as the baseline; if a later read returns
   `DOCA_ERROR_NOT_SUPPORTED` or `DOCA_ERROR_INVALID_VALUE`, the
   diff against this snapshot is the bug, not the read call.
   Cap-query at configure time is the cheapest way to make a
   later capability-related error self-explanatory.

For the cross-cutting observability primitives
(`--sdk-log-level`, the `doca-<lib>-trace` build flavor, the
`DOCA_LOG_LEVEL` env var) see
[`doca-debug CAPABILITIES.md ## Observability`](../../doca-debug/CAPABILITIES.md#observability).
For the install-tree observability (logger names, package
layout) defer to
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

## Safety policy

> **Overlay on the bundle-wide hardware-safety meta-policy.** The rules below are this skill's per-artifact overlay on the cross-cutting rules in [`doca-hardware-safety` CAPABILITIES.md ## Safety policy](../../doca-hardware-safety/CAPABILITIES.md#safety-policy) (specifically [### Per-artifact overlay pattern](../../doca-hardware-safety/CAPABILITIES.md#per-artifact-overlay-pattern)). When the two layers disagree, the stricter wins; when either layer says STOP, the agent stops.

The reader's safety surface is **cap-query-before-read +
sample-window discipline + privilege for the specific counter
domain**. Each is the source of a specific first-app failure the
agent must prevent.

| Prerequisite | Required state | How the agent verifies | Where to fix |
| --- | --- | --- | --- |
| Domain (and sub-area) cap-queried before read | `doca_telemetry_<domain>_cap_is_supported(devinfo)` (or the `pci` per-feature caps) returned `DOCA_SUCCESS` for the domain / sub-area the app will read | Code review at modify time; the read does not return `DOCA_ERROR_NOT_SUPPORTED` | Add the cap-query before the read. A read that assumes a domain the device does not expose is the canonical *"NOT_SUPPORTED out of nowhere"* failure |
| Sample-window discipline on reads | The read loop retries on `DOCA_ERROR_AGAIN` only after the documented sample window; for `diag`, the next `_query_counters` waits for the previous sampling cycle to finish | Structured log shows spaced retries, not a tight spin; the diag read is not called mid-cycle | Space the retry to the sample period (`doca_telemetry_diag_set_sample_period` for diag); a tight `AGAIN` spin wastes CPU and never converges faster than the hardware sample window |
| Running user has privilege for the counter domain | Some diagnostic / PHY / PCI counters require elevated device access; the running user has it | `id`; the create / start / read does not return `DOCA_ERROR_NOT_PERMITTED` | This is privilege for the *specific* counter domain, not a blanket sudo. Fix via the env-side privilege grant (route to [`doca-setup TASKS.md ## configure`](../../doca-setup/TASKS.md#configure)) rather than running the whole app as root reflexively |

- **The reader is read-only.** Do not invent a `_publish()` /
  `_emit()` / `_register_schema()` shape on any per-domain
  context; the per-domain API reads the device's counters and
  nothing more. If the user wants their app to ALSO publish the
  values it read to a monitoring pipeline, that requires a
  SEPARATE
  [`doca-telemetry-exporter`](../doca-telemetry-exporter/SKILL.md)
  context inside the same application — not a method on the
  reader context.
- **Cap-query before every domain you read.** Each of the six
  domains is independently gated on the device + install. A
  device that exposes PHY counters may not expose DPA counters;
  query each domain you intend to read.
- **`adp_retx` / `diag` clear-on-read is destructive to the
  counter state.** When `set_hist_clear_on_read` /
  `data_clear` is enabled, reading resets the underlying
  counters. Surface this to the user before enabling it — a
  second reader (or a later read) sees the counters as they
  stand after the reset, not the cumulative total.

## Deferred topic boundaries

This skill scopes itself to the DOCA Telemetry per-domain
hardware-counter reader libraries. Adjacent topics the agent
will get asked but should route elsewhere:

- **The publishing / export side** —
  [`doca-telemetry-exporter`](../doca-telemetry-exporter/SKILL.md)
  is the sibling skill for the application-side publisher
  library. This skill is reader-side only; any "how do I
  emit / publish a counter value" question routes there.
- **The DOCA Telemetry Service (DTS) — operating / deploying
  the service itself** — DTS is an externally-productized NVIDIA
  service that is NOT in `doca/services/` at the bundle-aligned
  DOCA release; it is **out of scope** for this bundle. Reach its
  docs via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
  non-goals.
- **NetFlow / IPFIX / local-socket collector frameworks** — the
  per-domain reader libraries do NOT expose a collector /
  schema-transport surface. Route any such question to a generic
  collector outside the DOCA family, to
  [`doca-telemetry-exporter`](../doca-telemetry-exporter/SKILL.md)
  for publishing, or to DTS (out of scope) for productized
  aggregation.
- **DOCA Core context internals** — owned by
  [`doca-programming-guide`](../../doca-programming-guide/SKILL.md).
  Note the per-domain reader does NOT use the generic
  `doca_ctx_*` / progress-engine surface; it uses per-domain
  `_create` / `_start` / `_stop` / `_destroy` directly.
- **Plain structured logging from the reader's own program** —
  use `doca-log`; the per-domain cap-query discipline is
  overhead the user does not need for plain logs.
- **Cross-cutting `DOCA_ERROR_*` taxonomy** — owned by
  [`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../../doca-programming-guide/CAPABILITIES.md#error-taxonomy).
  This skill adds the reader overlay (the
  `NOT_SUPPORTED`-means-domain-not-exposed rule and the
  `AGAIN`-means-snapshot-not-ready rule), not the taxonomy
  itself.
- **Cross-cutting debug ladder** (install / version / build /
  link / runtime / program / driver) — owned by
  [`doca-debug ## debug`](../../doca-debug/TASKS.md#debug).
  This skill's `## debug` overlays the runtime + program layers.
