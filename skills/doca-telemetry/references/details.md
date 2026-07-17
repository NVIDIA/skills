# doca-telemetry — reference detail

Moved out of `SKILL.md` to keep the loader under the per-file size budget. This is supporting detail, not routing logic.

## Example questions this skill answers well

The CLASSES of hardware-counter-reader questions this skill is
built to answer, each with one worked example. The agent should
treat the *class* as the load-bearing piece — the worked example
is a single instance.

- **"Which DOCA Telemetry sub-library do I want for the counters I
  care about?"** — worked example: *"I want PCC (Programmable
  Congestion Control) per-flow counters off my BlueField — which
  header do I `#include`?"*. Answered by the per-domain sub-lib
  inventory in
  [`CAPABILITIES.md ## Capabilities and modes`](../CAPABILITIES.md#capabilities-and-modes),
  which lists the six shipped per-domain headers
  (`doca_telemetry_pcc.h` / `_dpa.h` / `_diag.h` / `_adp_retx.h` /
  `_phy.h` / `_pci.h`) and the counter family each one exposes.
- **"How do I open a `doca_telemetry_<domain>` context on a
  `doca_dev` and read the latest counter snapshot?"** — worked
  example: *"open a doca_telemetry_phy context on my BF-3 and
  print the per-lane symbol-error counters once"*. Answered by
  the per-domain DOCA Core lifecycle in
  [`CAPABILITIES.md ## Capabilities and modes`](../CAPABILITIES.md#capabilities-and-modes)
  object table + the configure/read workflow in
  [`TASKS.md ## configure`](../TASKS.md#configure) +
  [`TASKS.md ## run`](../TASKS.md#run).
- **"How do I publish / export the counters I just read to
  OpenTelemetry, Prometheus, or my own pipeline?"** — worked
  example: *"I want my doca_telemetry_pcc-reading app to also
  publish the values as labeled metrics so Prometheus can scrape
  them"*. Answered by the reader-vs-exporter role split in
  [`CAPABILITIES.md ## Capabilities and modes`](../CAPABILITIES.md#capabilities-and-modes)
  — this library is the *reader*; the publishing / export
  surface is the sibling
  [`doca-telemetry-exporter`](../../doca-telemetry-exporter/SKILL.md)
  library, which is what the agent must route the user to. There
  is no NetFlow / IPFIX / local-socket *collector* surface in
  this library to point at.
- **"My `doca_telemetry_<domain>_cap_is_supported(devinfo)`
  returns `DOCA_ERROR_NOT_SUPPORTED` — what is wrong?"** —
  worked example: *"doca_telemetry_dpa cap query says
  NOT_SUPPORTED on my ConnectX-7"*. Answered by the
  `NOT_SUPPORTED` row in
  [`CAPABILITIES.md ## Error taxonomy`](../CAPABILITIES.md#error-taxonomy)
  (the device does not expose this counter domain — typical for
  DPA on non-DPA-shipped devices, or for `_phy` / `_pcc` on
  device families that gate them on firmware feature bits) + the
  cap-query-first rule in
  [`CAPABILITIES.md ## Safety policy`](../CAPABILITIES.md#safety-policy).
- **"My `doca_telemetry_<domain>_*_read` returns
  `DOCA_ERROR_AGAIN` — should I retry?"** — worked example: *"my
  PHY counter read returns AGAIN under load"*. Answered by the
  `AGAIN` row in
  [`CAPABILITIES.md ## Error taxonomy`](../CAPABILITIES.md#error-taxonomy):
  the per-domain hardware-counter snapshot is not ready yet — the
  correct response is to retry after the documented sample-window
  delay, not to spin.
- **"Is `doca_telemetry_<domain>` on my installed DOCA, and is the
  per-domain counter set I need supported on my device?"** —
  worked example: *"is `doca_telemetry_adp_retx` exposed on my
  install, and does the active devinfo support per-priority retx
  counters?"*. Answered by the version-compatibility overlay in
  [`CAPABILITIES.md ## Version compatibility`](../CAPABILITIES.md#version-compatibility),
  which cross-links the canonical detection chain in
  [`doca-version`](../../../doca-version/SKILL.md), plus the per-
  domain `_cap_is_supported` rule in
  [`CAPABILITIES.md ## Capabilities and modes`](../CAPABILITIES.md#capabilities-and-modes).
- **"How do I export DOCA telemetry to NetFlow / IPFIX, or stand
  up a NetFlow collector that consumes from this library?"** —
  worked example: *"I want a NetFlow collector backed by DOCA
  telemetry"*. The honest answer is that this library does NOT
  expose a NetFlow / IPFIX / local-socket *collector* transport
  surface — the previous bundle framing of that surface was wrong.
  The agent must refuse the collector framing and route the user
  to: (a) the sibling
  [`doca-telemetry-exporter`](../../doca-telemetry-exporter/SKILL.md)
  library if they actually want to PUBLISH counters from their
  own app over OTLP / Prometheus / labeled-metrics; or (b) the
  externally-productized DOCA Telemetry Service (DTS, out of
  scope — see
  [`doca-public-knowledge-map`](../../../doca-public-knowledge-map/SKILL.md)
  non-goals) if they wanted DTS-shaped aggregation; or (c) a
  generic NetFlow / IPFIX collector outside the DOCA family if
  that is what they actually need.

## What this skill deliberately does not ship

This skill is **agent guidance**, not a samples or templates
bundle. To keep the boundary clean, it deliberately does not
contain — and pull requests should not add:

- **Pre-written DOCA Telemetry per-domain reader source code,
  in any language.** The verified reader source code is the
  shipped C samples at
  `/opt/mellanox/doca/samples/doca_telemetry/`. The agent's
  job is to route the user to those files and prescribe a
  minimum-diff modification on them via the universal
  modify-a-sample workflow in
  [`doca-programming-guide`](../../../doca-programming-guide/SKILL.md),
  layered with the reader-specific overrides in
  [`TASKS.md ## modify`](../TASKS.md#modify).
- **A publishing / export telemetry application.** The
  publisher / export side lives in
  [`doca-telemetry-exporter`](../../doca-telemetry-exporter/SKILL.md);
  cross-link there for any "how do I emit / publish a counter
  value from my app" question.
- **A DOCA Telemetry Service (DTS) deployment / configuration
  guide.** DTS is an externally-productized NVIDIA service that
  is NOT in `doca/services/` at the bundle-aligned DOCA
  release; it is **out of scope** for this bundle (see
  [`doca-public-knowledge-map`](../../../doca-public-knowledge-map/SKILL.md)
  non-goals). The agent must refuse to write DTS deployment
  guidance and route the user to the public DTS guide via
  [`doca-public-knowledge-map`](../../../doca-public-knowledge-map/SKILL.md).
- **A NetFlow / IPFIX / local-socket collector framework.**
  The per-domain reader libraries do NOT expose a NetFlow /
  IPFIX / local-socket *collector* transport surface. The
  previous bundle framing of this library as a collector
  framework was wrong and is now disavowed. Pull requests that
  reintroduce collector / schema-query / NetFlow / IPFIX
  framing into this skill must be rejected — the agent must
  route those questions to the correct surface
  ([`doca-telemetry-exporter`](../../doca-telemetry-exporter/SKILL.md)
  for publish, DTS for productized aggregation, or a generic
  collector outside the DOCA family).
- **Standalone build manifests** (`meson.build`,
  `CMakeLists.txt`, `Cargo.toml`, …) parked inside the skill.
  The agent constructs the build manifest *in the user's
  project directory* against the user's installed DOCA, where
  `pkg-config --modversion doca-telemetry` is the source of
  truth.
- **A `samples/`, `bindings/`, or `reference/` subtree** of
  any kind. A mock or incomplete artifact in this skill's
  tree, even one labeled "reference", is misleading: users
  will read it as buildable.

## Related skills

- [`doca-telemetry-exporter`](../../doca-telemetry-exporter/SKILL.md) —
  the **publishing / export** side. This skill (the per-domain
  hardware-counter reader) and `doca-telemetry-exporter` (the
  application-side labeled-metrics / OTLP-logs publisher) are
  two separate libraries that solve two different problems —
  they do NOT form a publisher/collector pair (there is no
  collector transport surface here). An application that
  wants to EMIT counters to its monitoring pipeline links the
  exporter; an application that wants to READ hardware
  counters off a `doca_dev` links this skill. Wiring the
  wrong library into the user's app is the #1 first-app
  failure for both skills, and the load-bearing rule the
  agent must surface before any code-level guidance.
- [`doca-public-knowledge-map`](../../../doca-public-knowledge-map/SKILL.md) —
  the routing table for every public DOCA documentation source
  and the on-disk layout of an installed DOCA package. The
  per-domain reader's public guide URL is
  `https://docs.nvidia.com/doca/sdk/DOCA-Telemetry/index.html`
  (distinct from the publisher's guide at
  `https://docs.nvidia.com/doca/sdk/DOCA-Telemetry-Exporter/index.html`);
  the on-disk samples live under
  `/opt/mellanox/doca/samples/doca_telemetry/`. The DOCA
  Telemetry Service (DTS) is **externally productized** and
  out of scope for this bundle — reach its docs through that
  same routing table (`doca-public-knowledge-map` non-goals).
- [`doca-setup`](../../../doca-setup/SKILL.md) — env preparation,
  install verification, and the *I have no install yet* path
  with the public NGC DOCA container. This skill assumes its
  preconditions are satisfied (in particular, a `doca_dev` is
  open and `doca_telemetry_<domain>_cap_is_supported(devinfo)`
  returns `DOCA_SUCCESS` for the per-domain library the user
  wants to read).
- [`doca-version`](../../../doca-version/SKILL.md) — canonical
  DOCA version-handling rules. This skill's `## Version
  compatibility` cross-links the four-way match rule +
  detection chain and adds the per-domain reader overlay
  rules (each per-domain library's counter set can grow
  across DOCA releases; cap-query first, then read).
- [`doca-structured-tools-contract`](../../../doca-structured-tools-contract/SKILL.md) —
  the bundle's structured-tools precedence rule (detect /
  prefer / fall back / report). The Command appendix in
  [TASKS.md](../TASKS.md) honors this contract.
- [`doca-programming-guide`](../../../doca-programming-guide/SKILL.md) —
  general DOCA programming patterns shared by every library:
  the canonical `pkg-config` + meson build pattern, the
  universal modify-a-shipped-sample first-app workflow, the
  universal lifecycle, the cross-library `DOCA_ERROR_*`
  taxonomy, and the program-side debug order. This skill
  layers per-domain reader specifics on top.
- `doca-log` — the right primitive
  when the user actually wanted plain structured stdout
  logging from their own program rather than a structured
  hardware-counter snapshot. The reader's per-domain cap-
  query discipline is overhead the user does not need for
  plain logs; this skill's path-selection rule routes there
  when log shipping is the actual requirement.
- [`doca-debug`](../../../doca-debug/SKILL.md) — the cross-
  cutting debug ladder (install / version / build / link /
  runtime / program / driver). Per-domain reader-specific
  debug (domain not supported on this device, snapshot not
  yet ready, sample window too short, cap-query forgotten
  before read) overlays on top of that ladder.

