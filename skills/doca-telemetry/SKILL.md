---
license: Apache-2.0
name: doca-telemetry
description: >
  Use this skill to read DOCA hardware-counter events from a
  `doca_dev` through the per-domain Telemetry reader libraries:
  `doca_telemetry_pcc`, `_dpa`, `_diag`, `_adp_retx`, `_phy`,
  and `_pci`. It covers capability checks, context creation,
  startup, and per-domain reads or samples. Trigger for implicit
  requests such as "read PCC counters from my BlueField app",
  "sample DPA counter exports", or "expose PHY, PCI, or DIAG
  counters from this doca_dev". This is the counter-reader
  surface, not a NetFlow, IPFIX, or local-socket collector.
  Route publishing and export to `doca-telemetry-exporter`;
  route deployed DOCA Telemetry Service (DTS), collectors, and
  plain stdout logging elsewhere.
metadata:
  kind: library
compatibility: >
  Requires DOCA SDK installed at /opt/mellanox/doca on Linux (Ubuntu
  22.04/24.04 or RHEL/SLES) with a BlueField DPU or ConnectX NIC
  attached. Reads the user's local install via `pkg-config
  doca-telemetry` and inspects
  /opt/mellanox/doca/{lib,include,samples,applications}.
---

# DOCA Telemetry

**Where to start:** This skill assumes DOCA is already installed
and the user is doing **hands-on hardware-counter-reader work** —
opening a per-domain `doca_telemetry_<domain>` context against a
`doca_dev` and reading the latest hardware-counter snapshot for
that domain. The library is the **counter-READER** half of DOCA
telemetry; it is NOT a NetFlow / IPFIX collector and NOT a
generic schema-event consumer (the bundle previously framed it
that way and that framing was wrong — there is no NetFlow /
IPFIX / local-socket transport surface in the public header).
Open [`TASKS.md`](TASKS.md) if the user wants to *do* something
(configure / build / modify / run / test / debug); open
[`CAPABILITIES.md`](CAPABILITIES.md) when the question is *which
hardware-counter domains can this device read* on this install
(PCC, DPA, DIAG, ADP_RETX, PHY, PCI). If the user has not
installed DOCA yet, route to
[`doca-setup`](../../doca-setup/SKILL.md) first. If the user is
confused about whether they want this library (HW-counter
reader on a `doca_dev`) or
[`doca-telemetry-exporter`](../doca-telemetry-exporter/SKILL.md)
(the publisher / export side, which is a separate library and
publishes structured telemetry / labeled metrics / OTLP logs),
read the reader-vs-exporter role split in
[`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
BEFORE configuring anything; mixing the two is the load-bearing
first-app failure for this skill. If the user is asking about
**DOCA Telemetry Service (DTS) as deployed**, route to
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
non-goals — DTS is out of scope for this bundle.

## Audience

This skill serves **external developers building applications
that READ DOCA hardware counters from a `doca_dev` through one
or more of the six per-domain DOCA Telemetry reader libraries**
(`doca_telemetry_pcc` / `_dpa` / `_diag` / `_adp_retx` / `_phy` /
`_pci`) — i.e., users whose application code calls
`doca_telemetry_<domain>_*` (directly in C/C++, or through FFI /
bindings from another language) to open a per-domain context on
a `doca_dev`, configure the per-domain sample window, and read
the hardware-counter snapshot for that domain. It is *not* for
NVIDIA developers contributing to DOCA Telemetry itself, and it
is *not* for users writing the **publishing / export** side —
that is
[`doca-telemetry-exporter`](../doca-telemetry-exporter/SKILL.md),
a separate library and a separate skill.

**Language scope.** DOCA Telemetry's per-domain reader libraries
ship as a C surface with `pkg-config` module name
`doca-telemetry`. The shipped samples are written in C. C and
C++ readers are the canonical case; the worked examples in
`TASKS.md` assume that path. Other-language readers (Rust, Go,
Python, …) consume the same `*.so` through FFI or
language-specific bindings; the skill's contribution in that
case is to keep the reader-vs-exporter distinction, the per-
domain cap-query-first discipline, the per-domain DOCA Core
lifecycle on the `doca_dev`, the sample-window discipline, and
the error-taxonomy guidance language-neutral, and to route the
agent to the public per-domain C ABI as the authoritative
surface that any wrapper will eventually call.

## When to load this skill

Load this skill when the user is doing hands-on DOCA Telemetry
**hardware-counter-reader** work, in any language. Concretely:

- Picking the right per-domain header for the counters the user
  wants (`doca_telemetry_pcc.h` for Programmable Congestion
  Control counters, `_dpa.h` for DPA counters, `_diag.h` for
  generic device diagnostic counters, `_adp_retx.h` for ADP
  retransmit counters, `_phy.h` for physical-layer counters,
  `_pci.h` for PCI / PCIe counters) and confirming the device
  supports it via the per-domain `_cap_is_supported(devinfo)`
  query — **except `_pci`, which has no single `_cap_is_supported`
  and instead exposes per-feature caps like
  `doca_telemetry_pci_cap_management_info_is_supported` /
  `_cap_perf_counters_1_is_supported`.**
- Opening a per-domain `doca_telemetry_<domain>` context on a
  `doca_dev`, walking the per-domain lifecycle
  (`doca_telemetry_<domain>_create(dev)` → per-domain setters →
  `doca_telemetry_<domain>_start`), configuring the per-domain
  sample window, and reading the hardware-counter snapshot for
  that domain. Note this is a per-domain `_create`/`_start`
  surface, not the generic `doca_ctx_*` progress-engine
  lifecycle.
- Reading the device + library capability surface before assuming
  a counter family is available: use
  `doca_telemetry_<domain>_cap_is_supported` only for `pcc`,
  `dpa`, `diag`, `adp_retx`, and `phy`; use the matching
  per-feature `doca_telemetry_pci_cap_*_is_supported` query for
  PCI.
- Handling per-domain `DOCA_ERROR_*` returns from a counter
  read (lifecycle vs. device-doesn't-support-this-domain vs.
  per-domain `AGAIN`-means-snapshot-not-ready vs. permission /
  driver) and the per-read status reported back to the
  application.
- Choosing between DOCA Telemetry (hardware-counter READER) and
  an adjacent option:
  [`doca-telemetry-exporter`](../doca-telemetry-exporter/SKILL.md)
  when the user actually wants to PUBLISH / EXPORT the counter
  values (OTLP / Prometheus / labeled metrics);
  `doca-log` when plain structured stdout logging is enough; a
  generic Prometheus / OpenTelemetry client library when the
  counter source is a non-DOCA program; the externally-
  productized DOCA Telemetry Service (DTS, out of scope) when
  the user wants a turnkey aggregator.
- Designing or extending non-C bindings (Rust, Go, Python, …)
  that wrap the per-domain reader C ABI — for the reader-vs-
  exporter distinction, the per-domain cap-query-first rule,
  the per-domain `doca_dev` lifecycle, the sample-window
  discipline, and the error rules the wrapper must honor.

Do **not** load this skill for general DOCA orientation, install
of DOCA itself, the **publishing / export** side
([`doca-telemetry-exporter`](../doca-telemetry-exporter/SKILL.md)
has its own skill), the externally-productized DOCA Telemetry
Service (DTS — out of scope), or non-reader library questions.
For those, use
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

## What this skill provides

This is a **thin loader**. The body keeps only the orientation
needed to pick the right next file. The substantive
hardware-counter-reader material lives in two companion files:

- `CAPABILITIES.md` — what the per-domain readers can express on
  this install: the reader-vs-exporter role-split rule, the
  six shipped sub-libraries (`doca_telemetry_pcc` / `_dpa` /
  `_diag` / `_adp_retx` / `_phy` / `_pci`) and which counter
  family each one exposes, the per-domain DOCA Core lifecycle
  on a `doca_dev`, the domain-level capability query for PCC /
  DPA / DIAG / ADP_RETX / PHY and the per-feature capability
  queries for PCI, the
  reader error taxonomy (mapped onto the cross-library
  `DOCA_ERROR_*` set, with the `NOT_SUPPORTED`-means-domain-
  not-exposed-on-this-device rule and the `AGAIN`-means-
  snapshot-not-ready rule called out explicitly), the
  observability surface (per-read status + per-domain cap-
  query snapshot at configure time), the safety policy that
  gates per-domain reads behind the cap-query result, and the
  path-selection rule against
  [`doca-telemetry-exporter`](../doca-telemetry-exporter/SKILL.md),
  `doca-log`, and standalone Prometheus / OpenTelemetry / DTS.
- `TASKS.md` — step-by-step workflows for the six in-scope
  reader verbs: `configure`, `build`, `modify`, `run`, `test`,
  `debug`. Plus a `Deferred task verbs` block that points
  out-of-scope questions at the right next skill.

The skill assumes a host where DOCA is already installed at
the standard location and a target BlueField DPU or ConnectX NIC
is available. The [`TASKS.md ## run`](TASKS.md#run) workflow
opens the corresponding `doca_dev` and requires the per-domain
cap-query to return `DOCA_SUCCESS`. It does not cover installing
DOCA — that path goes through
[`doca-setup`](../../doca-setup/SKILL.md) — and it does not
cover writing the publishing / export side, which is
[`doca-telemetry-exporter`](../doca-telemetry-exporter/SKILL.md).

## Loading order

1. Read this `SKILL.md` first to confirm the user's question
   is in scope (specifically, that the user wants to READ a
   per-domain hardware counter via the per-domain reader API
   on a `doca_dev` — not PUBLISH counters, which is
   [`doca-telemetry-exporter`](../doca-telemetry-exporter/SKILL.md);
   not deploy DTS, which is out of scope; and not stand up a
   NetFlow / IPFIX collector, which this library does not
   expose a surface for).
2. **For the reader-vs-exporter rule, the six per-domain
   sub-libraries, the per-domain DOCA Core lifecycle on a
   `doca_dev`, the per-domain capability query, the error
   taxonomy (including the `NOT_SUPPORTED`-means-domain-
   not-exposed-on-this-device rule and the `AGAIN`-means-
   snapshot-not-ready rule), observability, the safety
   policy, and the path-selection rule, see
   [CAPABILITIES.md](CAPABILITIES.md).**
3. **For step-by-step workflows — configure, build, modify,
   run, test, debug — see [TASKS.md](TASKS.md).**

Both companion files cross-link to each other,
[`doca-version`](../../doca-version/SKILL.md) for the canonical
version-handling rules, and
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
whenever the right answer is "look it up in the public docs or
the installed package layout" rather than "reader-specific
guidance".

## Example questions this skill answers well

See [`references/details.md`](references/details.md#example-questions-this-skill-answers-well).
## What this skill deliberately does not ship

See [`references/details.md`](references/details.md#what-this-skill-deliberately-does-not-ship).
## Related skills

See [`references/details.md`](references/details.md#related-skills).
