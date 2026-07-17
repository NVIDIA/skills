# DOCA Device Emulation capabilities, version overlay, errors, observability, safety

**Where to start:** Pick the H2 anchor that matches your
question (umbrella architecture / sub-library selection /
per-sub-library Core context / capability discovery / version
/ errors / safety) and read that section end-to-end. The
tables in each section are the load-bearing content; the prose
around them is interpretation.

Read this file when the loader sent you here from
[SKILL.md](SKILL.md). For the *how* of executing each pattern
(the verbs `configure / build / modify / run / test / debug`),
jump to [TASKS.md](TASKS.md). For the canonical DOCA
version-handling rules that this skill layers a Device
Emulation overlay on top of, see
[`doca-version`](../../doca-version/SKILL.md). For the
host-side kernel driver layer that consumes the emulated PCIe
device, route via
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
to the upstream Linux kernel virtio / PCIe documentation — the
host kernel driver layer is *not* part of DOCA and is not
re-explained here.

## Pattern overview

Every Device Emulation question this skill teaches resolves
into one of SIX patterns. The patterns are CLASSES — they apply
across every Device Emulation release and every BlueField
generation + firmware combination, not just the worked examples
shown.

| Pattern | When it applies (class shape) | Where the substance lives |
| --- | --- | --- |
| 1. Pick the right sub-library FIRST | Decide which Device Emulation sub-library matches the emulated PCIe class the user is building. The full public sub-library set is **five**: (a) PCI Generic (`doca_devemu_pci`), (b) virtio-net (`doca_devemu_vnet`), (c) virtio-fs (`doca_devemu_vfs`), (d) **virtio-blk (`doca_devemu_vblk`)** and (e) **NVMe (`doca_devemu_nvme`)`** — different sub-libraries, different APIs, different `pkg-config` modules. This skill covers the first three end-to-end; the bundle previously omitted the virtio-blk and NVMe sub-libraries entirely, so for those two: acknowledge the libraries exist as real shipped `doca_devemu_vblk` / `doca_devemu_nvme` C surfaces (see `libs/doca_devemu/virtio/vblk/` and `libs/doca_devemu/nvme/` in the DOCA source tree), but route the user to the **packaged DOCA SNAP Service** for the productized end-to-end path (per the bundle's non-goal scoping in [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)) rather than treating their absence here as "those backends are not supported." | [`## Capabilities and modes`](#capabilities-and-modes) sub-library selection table + [TASKS.md ## configure](TASKS.md#configure) step 2 |
| 2. Confirm the umbrella architecture is the right shape | The host must be able to load a kernel driver for the emulated device class; the DPU must be able to run the backend as a user-space DOCA program — Device Emulation is *not* the answer when standard NIC behavior is enough or when the user wants a packaged service | [`## Capabilities and modes`](#capabilities-and-modes) umbrella architecture + path-selection rules |
| 3. Stand up the per-sub-library Core context | DOCA Core lifecycle for the chosen sub-library: one context per emulated device per sub-library; configure → start → host driver attaches → use → stop → destroy | [`## Capabilities and modes`](#capabilities-and-modes) per-sub-library context shape + [TASKS.md ## configure](TASKS.md#configure) step 4 |
| 4. Discover per-sub-library capabilities | Query the matching `doca_devemu_<sub>_cap_*` family against the active `doca_devinfo` BEFORE assuming a feature bit or device characteristic is available; sub-libraries do not share capability surfaces | [`## Capabilities and modes`](#capabilities-and-modes) capability-query rule + [TASKS.md ## configure](TASKS.md#configure) step 3 |
| 5. Honor dual-axis env preconditions | DPU-side privileges to perform PCIe-level emulation (typically sudo) AND BlueField firmware-level enablement of the specific emulation type the user wants; both must be true before any `doca_devemu_*` create call | [`## Safety policy`](#safety-policy) env-precondition matrix + [TASKS.md ## configure](TASKS.md#configure) step 1 |
| 6. Diagnose a Device Emulation error | Map `DOCA_ERROR_NOT_SUPPORTED` / `_NOT_PERMITTED` / `_BAD_STATE` / `_INVALID_VALUE` / `_IO_FAILED` to a root cause without leaving the Device Emulation layer prematurely | [`## Error taxonomy`](#error-taxonomy) + [TASKS.md ## debug](TASKS.md#debug) |

Two cross-cutting rules that apply to *every* pattern above:

- **Sub-library selection precedes everything.** *"Use DOCA
  Device Emulation"* is not actionable on its own — the API
  surface, the `pkg-config` module, the capability-query
  family, the sample tree under
  `/opt/mellanox/doca/samples/doca_devemu/`, and
  the firmware-level enablement axis ALL depend on whether
  the user is building a PCI Generic device, a virtio-net
  device, or a virtio-fs device. An agent that begins writing
  code (or quoting symbols) before pinning the sub-library is
  wrong for every Device Emulation release.
- **The library is the building block; the packaged services
  are different artifacts.** DOCA SNAP Service and DOCA
  Virtio-net Service are packaged daemons built *on top of*
  this library. The library is the right answer when the user
  is writing a custom backend; the service is the right
  answer when the user wants a packaged solution and is
  willing to let the service own the backend implementation.
  Conflating the two is the single most common Device
  Emulation first-app design error and the cleanest place to
  fail fast.

## Capabilities and modes

DOCA Device Emulation is an **umbrella library**: a shared
architecture across three sub-libraries that each expose their
own API surface, `pkg-config` module, sample tree subdirectory,
and capability-query family. The umbrella architecture is
constant; the per-sub-library specifics differ.

**Umbrella architecture — what every Device Emulation
deployment looks like.**

| Side | What runs there | What the user provides | What DOCA provides |
| --- | --- | --- | --- |
| Host | The host operating system, the host's standard PCIe enumeration, and the host kernel driver matching the emulated device class (e.g. the upstream `virtio_net` kernel driver for a virtio-net emulation, the upstream `virtio_fs` kernel driver for a virtio-fs emulation, the host's generic PCIe driver model for a PCI Generic emulation) | Nothing DOCA-specific — the host sees a *real-looking* PCIe device and binds the standard kernel driver to it | The PCIe surface the host's kernel driver sees |
| DPU (BlueField Arm) | User-space DPU code that links against the chosen Device Emulation sub-library, runs the backend logic for the emulated device, and uses the doorbell / DMA primitives the sub-library exposes to react to host driver activity | The backend implementation (e.g. packet forwarding for a virtio-net device, the storage backend for a SNAP-style device, the filesystem implementation for a virtio-fs device) | The `doca_devemu_<sub>_*` C API surface, the DOCA Core context for the sub-library, the doorbell / DMA primitives, and the per-sub-library capability-query family |

The agent's rule: when the user asks *"what backend should I
write"*, that is a domain question (storage / networking /
filesystem design), NOT an API question — route via
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
to the public DOCA Device Emulation umbrella guide and the
per-sub-library guides linked from it, plus the user's own
domain expertise. When the user asks *"how do I expose an
emulated PCIe device from the DPU and react to host driver
calls"*, that is this skill's scope.

**Sub-library selection — the load-bearing first move.**
Before any code is written, the agent must pin the sub-library
the user needs. Different sub-libraries have different APIs,
different `pkg-config` modules (per the per-sub-library guides
reachable via
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)),
and different firmware-level enablement axes.

| User intent | Sub-library | Why this sub-library is the answer |
| --- | --- | --- |
| Build a custom emulated PCIe device whose interface does not match any standard virtio class (e.g. a custom PCIe peripheral for prototyping, a security-isolated device with a custom PCIe surface, a research device with bespoke registers) | PCI Generic — raw PCIe device emulation | Gives the user direct control over the PCIe surface the host sees; the host binds its generic PCIe driver model rather than a class-specific virtio driver. The user owns the entire register / interrupt / BAR / configuration-space contract |
| Expose a virtio network device to the host backed by custom DPU-side packet processing (e.g. a custom packet-forwarding logic the user wants the host to see as a standard virtio NIC) | virtio-net — emulated virtio network device | The host sees a standard virtio-net device and binds its upstream `virtio_net` kernel driver; the user only writes the DPU-side backend. The virtio framing + feature-negotiation is handled by the sub-library |
| Expose a virtio filesystem to the host backed by custom DPU-side filesystem logic (e.g. a network-attached or specialized filesystem the host should see as a standard virtio-fs device) | virtio-fs — emulated virtio filesystem device | The host sees a standard virtio-fs device and binds its upstream `virtio_fs` kernel driver; the user only writes the DPU-side backend. The virtio-fs request shape is handled by the sub-library |
| Want a packaged NVMe / virtio-blk storage backend without writing the backend code yourself | None of this skill — DOCA SNAP Service; route via [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) | SNAP is a packaged storage service built on top of Device Emulation; the user does not implement the backend. This skill is the *building block* the service uses; the service is a *separate artifact* |
| Want a packaged virtio-net daemon without writing the backend code yourself | None of this skill — DOCA Virtio-net Service; route via [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) | Virtio-net Service is a packaged virtio-net daemon built on top of this library's virtio-net sub-library; the user does not implement the backend. The library is the building block; the service is the packaged solution |
| Want standard NIC behavior on the BlueField (no new emulated device class; just shape the data plane on the existing NIC) | None of this skill — use [`doca-flow`](../doca-flow/SKILL.md) + [`doca-eth`](../doca-eth/SKILL.md) | Device Emulation is for *new* emulated PCIe devices the host did not previously see. The BlueField's built-in NIC personality is shaped by Flow + Eth, not by this library |

If the user does not yet know which sub-library they need,
the agent's job is to walk the rows above with the user
before writing any code. Picking a sub-library *for* the user
when the rows do not pin one is a wrong answer regardless of
how clean the rest of the setup goes.

**The per-sub-library DOCA Core context — one per emulated
device per sub-library.**

| Object class | Per | Lifetime | What it owns | Key calls |
| --- | --- | --- | --- | --- |
| The chosen sub-library's emulated-device context (one of: PCI Generic emulated-device context, virtio-net emulated-device context, virtio-fs emulated-device context) | One per emulated device the DPU is exposing in that sub-library | Created against a `doca_dev` that maps to a BlueField with the matching emulation type enabled in firmware; lives until the host driver is detached and the context is destroyed | The DOCA-side bookkeeping for that emulated device, the registration of the doorbell / DMA primitives the host driver will interact with, and the host-side observability for the emulated device's state | DOCA Core create / configure / start / stop / destroy on the per-sub-library context object; the per-sub-library `doca_devemu_<sub>_cap_*` family for capability queries; the per-sub-library doorbell / DMA primitives for host ↔ DPU interaction |

A DPU exposing more than one emulated device — or emulated
devices in more than one sub-library — needs one per-sub-library
context per emulated device. There is no *"global Device
Emulation context"* that spans sub-libraries; the sub-library
contexts are independent and follow their own lifecycles.

**Doorbell / DMA primitives — the bridge between host driver
and DPU backend.** Each sub-library exposes its own doorbell
and DMA primitives that the host's kernel driver and the
DPU-side backend use to coordinate without each side having to
poll the other unconditionally. The agent's rule for these
primitives is the same as for the Core context: per-sub-library,
not global.

| Surface | What it does | Why the agent must surface it explicitly |
| --- | --- | --- |
| Doorbell registration on the per-sub-library context | Lets the DPU-side backend react to host-side kernel driver activity (e.g. the host driver kicking a virtio queue) without busy-polling the PCIe surface | A first-app that omits the doorbell wiring will appear to load successfully — the host kernel driver may even bind — but no host-driver activity will produce any DPU-side reaction. The agent must explicitly verify the doorbell wiring before recommending a smoke run |
| DMA primitives for moving payload bytes between host memory and DPU memory under the emulated device's contract | Implements the actual data transfer the emulated device class demands (e.g. virtio descriptor payloads, raw PCIe BAR-mediated DMA for PCI Generic) | The DMA primitives the sub-library exposes are *its own* — they are not a substitute for the general-purpose [`doca-dma`](../doca-dma/SKILL.md) library and they enforce the emulated device contract. The agent must not recommend hand-rolling the data transfer with `doca-dma` directly when the sub-library already gives the contract-correct primitives |

**Per-sub-library capability discovery — the only rule.** Each
sub-library exposes its own `doca_devemu_<sub>_cap_*` family.
Before assuming a feature bit, a maximum, or a device
characteristic is available, call the matching family against
the active `doca_devinfo` for the BlueField device the user is
emulating from.

| Sub-library | Capability-query family | What to ask before sizing or feature-bit selection |
| --- | --- | --- |
| PCI Generic | `doca_devemu_pci_cap_*` against the active `doca_devinfo` | Whether the BlueField + DOCA + firmware combo supports raw PCIe device emulation, and what bounds the framework places on the PCIe surface the user is allowed to expose |
| virtio-net | `doca_devemu_virtio_cap_*` against the active `doca_devinfo` for the virtio side, narrowed to the virtio-net surface | Which virtio feature bits the framework supports on this combo, queue sizing bounds, and the supported virtio specification revisions |
| virtio-fs | `doca_devemu_vfs_cap_*` against the active `doca_devinfo` | Which virtio-fs feature bits the framework supports on this combo, and any sub-library-specific bounds on filesystem object sizes / queue sizing |

Each family is *per-sub-library*. An agent that quotes
capability values from one sub-library against a different
sub-library is wrong; the families do not interchange.

**Configuration shape.** *Mandatory* configurations before
any `doca_ctx_start()` on the per-sub-library context: the
chosen sub-library's emulated-device context must be created
against a `doca_dev` whose BlueField firmware has the
matching emulation type enabled; the per-sub-library doorbell
/ DMA primitives must be wired; the per-sub-library
capability queries (`doca_devemu_<sub>_cap_*`) must agree
that the deployment is supported on this combo; the per-
sub-library feature negotiation (virtio-net / virtio-fs only,
where the host driver negotiates features with the emulated
device) must be set up before the host kernel driver attempts
to bind. *Optional* configurations (per-device queue tuning,
per-device user-data) ride on top of the same cap-query rule.

## Version compatibility

For the canonical DOCA version-detection chain, the four-way
match rule, NGC container semantics, and the
headers-win-over-docs rule, see
[`doca-version`](../../doca-version/SKILL.md). The body lives
there; this skill does not duplicate it.

**The Device Emulation-specific overlay** is:

- **Version handling is per-sub-library, not per-umbrella.**
  Each sub-library has its own `pkg-config` module installed
  by the DOCA host packages on the DPU side; the user's
  build manifest must reference the module for the sub-library
  they picked (per the sub-library selection table in [`##
  Capabilities and modes`](#capabilities-and-modes)). The
  exact per-sub-library module name lives in the public
  Device Emulation umbrella guide and the per-sub-library
  guides linked from it, reachable via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md);
  the agent should *route the user there to read it off the
  guide and off their own `pkg-config` install* rather than
  hardcoding a module name from memory.
- **Cap-query is the runtime authority per sub-library.** Per
  the cross-cutting rule in
  [`doca-version CAPABILITIES.md ## Observability`](../../doca-version/CAPABILITIES.md#observability),
  the per-sub-library `doca_devemu_<sub>_cap_*` query against
  the active `doca_devinfo` is the runtime authority for *"is
  the emulation type I want supported on this BlueField + this
  DOCA install + this firmware"*. The public guides are the
  *promise*; the cap query is the *reality*. The four-way-match
  check (the chosen sub-library's `.pc` plus `doca-common.pc`
  plus `doca_caps --version`) per
  [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility)
  catches the partial-install case where the umbrella is
  installed but a sub-library is not.
- **Headers in $(pkg-config --variable=includedir doca-common)
  win over docs.** Per the headers-win-over-docs rule in
  [`doca-version`](../../doca-version/SKILL.md), if a public
  Device Emulation doc page mentions a `doca_devemu_*` symbol
  that is not in the installed headers, the headers describe
  what *this* install can call. The agent must quote the
  headers (and the cap query), not the docs URL, when the two
  disagree.

## Error taxonomy

Device Emulation-specific overlays on the cross-library
`DOCA_ERROR_*` taxonomy. The cross-library taxonomy itself
lives in
[`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../../doca-programming-guide/CAPABILITIES.md#error-taxonomy);
the rows below are the *Device Emulation surface* meaning that
the agent must disambiguate before falling back to the
cross-library response.

| Error | Device Emulation context where it shows up | Device Emulation-specific cause |
| --- | --- | --- |
| `DOCA_ERROR_BAD_STATE` | Any `doca_devemu_<sub>_*` call before `doca_ctx_start()` on the per-sub-library context, or after `doca_ctx_stop()`; tearing down the per-sub-library context while the host kernel driver is still attached | Lifecycle violation. Walk the call sequence against the lifecycle in [`doca-programming-guide CAPABILITIES.md ## Capabilities and modes`](../../doca-programming-guide/CAPABILITIES.md#capabilities-and-modes); the most common case is destroying the per-sub-library context before the host kernel driver has been unbound or the emulated device removed |
| `DOCA_ERROR_NOT_SUPPORTED` | Per-sub-library context create / start; the matching `doca_devemu_<sub>_cap_*` returning *not supported*; first feature-negotiation call (virtio-net / virtio-fs) | The chosen emulation type is not enabled at all on this combo — typically because the BlueField generation does not carry support for it, or because the user's DOCA install does not ship the sub-library at all (the umbrella ships, but a specific sub-library may not on every install). Re-run the matching `doca_devemu_<sub>_cap_*` against the active `doca_devinfo`; if false, that is the answer — do not paper over with a retry |
| `DOCA_ERROR_NOT_PERMITTED` | Per-sub-library context create / start; first doorbell registration | Either the DPU-side process lacks the privilege required for PCIe-level emulation (PCIe-level emulation is privileged; the DPU-side process typically requires sudo) OR — and this is the case the agent MUST surface — the BlueField firmware does not have the matching emulation type enabled. The two look identical at the `doca_devemu_<sub>_*` API surface and the fix is different: process privilege is a DPU-side OS / group-membership fix per [`doca-setup ## Safety policy`](../../doca-setup/CAPABILITIES.md#safety-policy); the firmware-level emulation type enable is a firmware-side flip per [`doca-setup`](../../doca-setup/SKILL.md). The agent must check the firmware-side enable BEFORE concluding *"this is a permission problem"* |
| `DOCA_ERROR_INVALID_VALUE` | Per-sub-library configuration calls — device descriptor / device identity values, virtio feature-bit negotiation mismatches, queue sizing past the cap | The per-device descriptor the user supplied is malformed for the chosen sub-library, OR (virtio-net / virtio-fs) the host kernel driver and the emulated device disagreed at feature negotiation, OR a sizing parameter exceeds what the per-sub-library cap-query advertised. Re-read the cap-query baseline; for the virtio cases, re-check the feature set the user opted in to against what the cap-query reports as supported on this combo |
| `DOCA_ERROR_IO_FAILED` | Host ↔ DPU interaction after start — host kernel driver reads, writes, kicks; DMA primitives moving payload | The host-side driver interaction failed below the DOCA layer. Most common cause is the host's kernel driver did not bind cleanly to the emulated device (e.g. the host kernel does not ship the matching virtio driver, or the host's `lspci` does see the device but the driver bind failed); second most common is a host-side mmap / DMA address that the DPU-side DMA primitive could not resolve. Inspect the host-side `dmesg` and `lspci` for the emulated device; the fix is typically host-side rather than DPU-side |

The agent's rule: **never recommend a retry loop on
`DOCA_ERROR_*` without first identifying which of the rows
above is the cause**. None of the Device Emulation errors
above want a retry — every row points at investigation (env /
firmware / sub-library selection / feature negotiation /
host-side driver), not at retry.

## Observability

Device Emulation observability surface is **two-sided**: there
is a DPU-side observability surface (per-sub-library
context state, capability-query snapshots, DOCA logger events
on the DPU side) AND a host-side observability surface (the
host kernel's view of the emulated PCIe device — `lspci`, the
matching kernel driver's sysfs / debugfs entries, `dmesg`).
The agent must reach for both, not just one — an emulated
device that *appears to load* on the DPU side but is invisible
to the host's `lspci`, or visible but unbound to its driver,
is a half-deployment that the DPU-side surface alone cannot
fully diagnose.

Three primary signals the agent should reach for:

1. **DPU-side per-sub-library context state and logger.** The
   DPU-side context's state transitions (configure → start →
   running → stop → destroy), the per-sub-library doorbell /
   DMA primitives' observability hooks, and the DOCA logger
   on the DPU side (set `DOCA_LOG_LEVEL=trace` for first runs)
   are the primary DPU-side signals. Absence of a doorbell
   reaction when the host driver is clearly active is
   *always* either a missing `doca_pe_progress()` in the
   DPU-side main loop OR a doorbell registration that was
   wired against the wrong sub-library context.
2. **Per-sub-library capability snapshot at configure time.**
   The output of every `doca_devemu_<sub>_cap_*` query against
   the active `doca_devinfo` is a snapshot of *what the
   library + the hardware + the firmware said was possible*
   before any emulated device was created. Save it; if a
   later call returns `DOCA_ERROR_NOT_SUPPORTED` or
   `_INVALID_VALUE` the diff against this baseline is the
   bug.
3. **Host-side enumeration and kernel-driver state.** On the
   host side, `lspci` shows whether the emulated PCIe device
   is enumerated at all; the matching kernel driver's sysfs
   / debugfs entries show whether the driver bound to it; the
   host's `dmesg` shows what the kernel saw at bind time
   (virtio feature negotiation issues, DMA failures, BAR
   mapping issues). A DPU-side context that started cleanly
   but produced no host-side enumeration is most often a
   firmware-side enable that has not taken effect yet (a
   BlueField reset after the firmware-side flip is typically
   required); a host-side enumeration without driver bind is
   most often a host kernel that ships no driver for the
   emulated class (the host kernel must have the matching
   driver, the DPU does not provide it).

For cross-cutting observability primitives
(`--sdk-log-level`, the `DOCA_LOG_LEVEL` env var, the trace
build flavor) see
[`doca-debug CAPABILITIES.md ## Observability`](../../doca-debug/CAPABILITIES.md#observability).
For the install-tree observability (logger names, package
layout, sample tree) defer to
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

## Safety policy

> **Overlay on the bundle-wide hardware-safety meta-policy.** The rules below are this skill's per-artifact overlay on the cross-cutting rules in [`doca-hardware-safety` CAPABILITIES.md ## Safety policy](../../doca-hardware-safety/CAPABILITIES.md#safety-policy) (specifically [### Per-artifact overlay pattern](../../doca-hardware-safety/CAPABILITIES.md#per-artifact-overlay-pattern)). When the two layers disagree, the stricter wins; when either layer says STOP, the agent stops.

Device Emulation's safety surface is **dual-axis-driven AND
host-driver-driven**. The three most common Device Emulation
first-app failures are (1) the BlueField firmware does not
have the chosen emulation type enabled; (2) the DPU-side
process lacks the privilege required for PCIe-level emulation;
and (3) the host's kernel does not ship the standard driver
for the emulated device class the user picked, or the host's
kernel ships it but it failed to bind for a feature-
negotiation reason. The agent's job is to verify all three
before any `doca_devemu_<sub>_*` create call, not after the
first `DOCA_ERROR_NOT_PERMITTED` or first silent host-side
non-enumeration.

The **env-precondition matrix** the agent must walk for any
new Device Emulation setup:

| Precondition | What must be true | How the agent verifies | Where to fix |
| --- | --- | --- | --- |
| BlueField generation supports the chosen sub-library | The BlueField generation actually carries support for the emulation type the user picked (PCI Generic / virtio-net / virtio-fs); not every BlueField generation supports every sub-library | The matching `doca_devemu_<sub>_cap_*` against the active `doca_devinfo` returns supported; the BlueField hardware generation is on the public umbrella's *supported BlueField generations* list (route to [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) for the per-sub-library guide) | [`doca-setup`](../../doca-setup/SKILL.md) for the env-side BlueField identification; this is **not** a code fix — if the hardware does not support the sub-library, the answer is *use a different sub-library or different hardware*, not a code change |
| BlueField firmware has the matching emulation type enabled | Device emulation requires a firmware-level enablement of the specific emulation type (PCI Generic / virtio-net / virtio-fs); this is **not a casual setting** and not flipped during normal BlueField operation. The firmware-side knob is documented in the public BlueField firmware configuration tooling and lives behind the env-side skill, not behind a host-side library call | A `doca_dev` open that succeeds but produces no support in `doca_devemu_<sub>_cap_*` for the chosen sub-library strongly suggests the firmware slot is disabled. The firmware-side configuration tools live in [`doca-setup`](../../doca-setup/SKILL.md); confirmation typically requires a BlueField reset after the slot is flipped | [`doca-setup`](../../doca-setup/SKILL.md) for the firmware-side enable; the BlueField typically needs a reset after the slot is flipped before the new state takes effect. The agent must surface this as a *firmware-level precondition*, not as a setting that is casually changed |
| DPU-side process privileges | The DPU-side process running the user's `doca_devemu_<sub>_*` code can perform PCIe-level emulation; this is privileged and typically requires sudo on the DPU side (the DPU-side baseline is stricter than host-side `doca_dev` access because exposing a PCIe device is a privileged operation) | The DPU-side process can open the target `doca_dev`; the user is either running with sudo on the DPU or is a member of the DPU-side group the public install profile grants the privilege to | [`doca-setup`](../../doca-setup/SKILL.md) for the DPU-side privilege; do **not** modify the program to silence the failure |
| Host kernel ships the standard driver for the emulated device class | The host kernel must include the standard kernel driver for the emulated device class the user picked: the upstream virtio-net driver for a virtio-net emulation, the upstream virtio-fs driver for a virtio-fs emulation, the host's generic PCIe driver model for a PCI Generic emulation | After the DPU-side context is started, `lspci` on the host shows the emulated device; the matching kernel driver's sysfs entry shows the driver bound. Host-side `dmesg` around the moment the DPU started the context is the cheapest place to see the bind succeed or fail | Host-side fix: the user's host kernel must ship (or load via a module) the matching standard driver. This is **not a DOCA-side fix** — the host kernel ships the driver, the DPU does not provide it. If the user is on a stripped host kernel without the matching driver, the answer is to load the matching kernel module, not to modify the DPU-side program |
| Single-emulated-device smoke succeeded before scaling | A trivial emulated-device setup (one device, one queue if applicable, one DPU-side context, one host-side driver bind, basic operation working) before any sophisticated backend or multi-device deployment is attempted | Walk the smoke step in [TASKS.md ## test](TASKS.md#test) step 1; a smoke that fails identifies env-side / firmware-side / sub-library-selection / host-driver gaps cheaply, before any backend design effort is wasted | Diagnose the smoke failure first; do NOT scale a broken smoke into a complex multi-device deployment |

**Choosing the library does not preclude later choosing a
service.** A user who starts with `doca-devemu`
and discovers they would rather adopt the packaged DOCA SNAP
Service or DOCA Virtio-net Service can do so — the services
are built on top of the same library and the firmware-side
preconditions are the same. The agent should surface this
escape hatch when the user's intent reveals a packaged
solution would suit them better; conversely, the agent should
surface this library path when a user trying to adopt a
service discovers the service does not implement what they
need and they will have to write a custom backend.

**Lifecycle ordering is per-sub-library.** The per-sub-library
context follows the universal DOCA Core teardown order on top
of the `doca_dev`; out-of-order teardown surfaces as
`DOCA_ERROR_BAD_STATE`. The Device Emulation-specific
addition is the host-driver-attached state: tearing down the
per-sub-library context while the host kernel driver is still
attached to the emulated device may leave the host side with
a half-removed PCIe surface that produces noisy `dmesg`
output on the host until the next BlueField reset. The agent
must surface this ordering explicitly — instruct the user to
detach the host kernel driver (or unload the host kernel
module) before tearing down the DPU-side context, and to
destroy emulated devices before destroying their parent
per-sub-library context.

**This skill does not define a backend.** `doca-devemu`
*provides the framework* for emulated PCIe devices; it does
not implement a specific storage backend, packet processor, or
filesystem. When the user asks *"what backend should I
write"*, the agent must refuse to invent backend bodies and
must route the user to the public DOCA Device Emulation
umbrella guide via
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
and to the user's own domain expertise.

## Deferred topic boundaries

This skill scopes itself to the **library** umbrella across
its three public sub-libraries. Adjacent topics the agent will
get asked but should route elsewhere:

- **DOCA SNAP Service** — packaged NVMe / virtio-blk storage
  daemon built on top of this library. Out of scope; route via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
  to the DOCA SNAP Services guide. The library is the
  building block; the service is the packaged solution and is
  a different artifact than what this skill covers.
- **DOCA Virtio-net Service** — packaged virtio-net daemon
  built on top of this library's virtio-net sub-library. Out
  of scope; route via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
  to the DOCA Virtio-net Service guide. Same library-vs-service
  distinction as SNAP above.
- **Host kernel drivers for the emulated device class**
  (upstream virtio-net, virtio-fs, virtio-blk drivers, the
  host's generic PCIe driver model) — outside DOCA entirely.
  Route to the upstream Linux kernel documentation; this skill
  assumes the host kernel ships the matching driver.
- **Designing the backend itself** (the storage backend body,
  the packet-forwarding logic, the filesystem implementation)
  — outside this skill. Route via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
  to the public DOCA Device Emulation umbrella guide and the
  per-sub-library guides for sub-library-specific guidance;
  the backend body itself is the user's domain expertise.
- **Standard NIC behavior on the BlueField data path**
  ([`doca-flow`](../doca-flow/SKILL.md) +
  [`doca-eth`](../doca-eth/SKILL.md) territory) — outside this
  skill. Device Emulation is for *custom* emulated PCIe
  devices; Flow + Eth are for the BlueField's existing NIC
  personality.
- **General-purpose host ↔ DPU memory copies**
  ([`doca-dma`](../doca-dma/SKILL.md)) — outside this skill.
  The per-sub-library DMA primitives in Device Emulation are
  contract-correct for the emulated device class; for raw
  general-purpose host ↔ DPU memcpy outside the emulation
  contract, DMA is the right library.
- **Host ↔ DPU control-plane messaging that does NOT involve
  the host seeing a new PCIe device class**
  ([`doca-comch`](../doca-comch/SKILL.md)) — outside this
  skill. Comch is an explicit host ↔ DPU IPC over PCIe;
  Device Emulation hides the DPU behind a real-looking PCIe
  device that uses the host's standard driver path.
- **DOCA Core context and progress engine internals** —
  owned by
  [`doca-programming-guide`](../../doca-programming-guide/SKILL.md).
  This skill *uses* the Core lifecycle; it does not redefine
  it.
- **Cross-cutting `DOCA_ERROR_*` taxonomy** — owned by
  [`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../../doca-programming-guide/CAPABILITIES.md#error-taxonomy).
  This skill adds the Device Emulation overlay, not the
  taxonomy itself.
- **Cross-cutting debug ladder** (install / version / build /
  link / runtime / program / driver) — owned by
  [`doca-debug ## debug`](../../doca-debug/TASKS.md#debug).
  This skill's `## debug` redirects there for layer 1-4;
  layers 5-7 carry the Device Emulation-specific overlay
  (including the firmware-side emulation-type enable route,
  the sub-library mis-selection diagnosis, and the host-kernel-
  driver bind state).
