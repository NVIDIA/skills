---
license: Apache-2.0
name: doca-devemu
description: >
  Use this skill when the user is doing hands-on DOCA Device Emulation
  on a BlueField DPU — exposing a custom emulated PCIe device the host
  sees as a real peripheral while DPU-side code runs the backend,
  picking the sub-library (PCI Generic, virtio-net, virtio-fs), wiring
  the per-sub-library Core context plus doorbell / DMA primitives,
  querying `doca_devemu_*_cap_*`, or debugging DOCA_ERROR_* from a
  `doca_devemu_*` call. Trigger even when the user does not say "devemu"
  — typical implicit phrasings include "expose a custom PCIe device from
  BlueField to the host", "host should see a virtio NIC backed by my DPU
  code", "lspci does not show my emulated device", "device enumerated
  but no driver binds", "DPU sees nothing when host kicks the queue", or
  "virtio feature negotiation failed at bind". Refuse and route
  elsewhere for the packaged DOCA SNAP / Virtio-net Services, host-side
  virtio kernel drivers, backend body design, or standard BlueField NIC
  behavior — those belong to other skills.
metadata:
  author: NVIDIA <doca@nvidia.com>
  kind: library
  doca_version: "3.5"
  tags:
    - doca
    - devemu
    - virtio
compatibility: >
  Requires DOCA SDK installed at /opt/mellanox/doca on BOTH the host AND
  the BlueField DPU (Ubuntu 22.04/24.04 or RHEL/SLES), with firmware
  emulation (PCI Generic / virtio-net / virtio-fs) enabled and the host
  kernel shipping the matching driver (virtio_net / virtio_fs / generic
  PCIe). Reads pkg-config for the chosen sub-library and inspects
  /opt/mellanox/doca/{lib,include,samples/doca_devemu}. The validation
  steps require a live DOCA install.
---

# DOCA Device Emulation

## Examples

- **"Expose a custom PCIe device from BlueField to the host."**
  — this is in scope. Pick PCI Generic / virtio-net /
  virtio-fs; do not stand up SNAP or Virtio-net Service here.
- **"lspci does not show my emulated device."** — confirm the
  firmware emulation type and host driver before changing
  DPU-side code.

## Instructions

Follow Loading order below. Open the tasks companion for configure /
emulate / debug. Open the capabilities companion for sub-libraries
and cap queries. Do not open every companion file.

**Where to start:** This skill assumes DOCA is already installed
on the host AND on the BlueField, the user is doing **hands-on
emulated-PCIe-device work** from the DPU side (writing the
backend that the host's kernel driver will talk to over the
emulated PCIe surface), and the user knows which CLASS of
emulated device they want to build. Open
the tasks companion if the user wants to *do* something
(configure / build / modify / run / test / debug); open
the capabilities companion when the question is
*what can Device Emulation express* on this DOCA version + this
BlueField generation + this firmware. If the user has not
installed DOCA yet, route to
`doca-setup/SKILL.md` first. **Before
anything else, the agent must route the user to the right
sub-library** — DOCA Device Emulation is an *umbrella* that
covers PCI Generic (raw PCIe device emulation), virtio-net
(emulated virtio network device), and virtio-fs (emulated
virtio filesystem device); each sub-library has its own
context, its own `pkg-config` module, and its own capability
surface. The sub-library selection rule lives in
the capabilities companion.
If the user wants a packaged solution rather than a library
(e.g. *"I want NVMe SNAP on my host without writing the
backend myself"*, or *"I want a managed virtio-net daemon"*),
route via
`doca-public-knowledge-map/SKILL.md`
to the DOCA SNAP Service / DOCA Virtio-net Service guides
— those services are *built on top of* this library and are a
different artifact than what this skill covers.

## Audience

This skill serves **external developers building applications
that consume the DOCA Device Emulation library** — i.e., users
whose DPU-side code calls `doca_devemu_pci_*`,
`doca_devemu_virtio_*`, or `doca_devemu_vfs_*` (directly
in C / C++, or through FFI / bindings from another language)
to expose an emulated PCIe device to the host that the host's
existing kernel drivers can drive as if it were a real PCIe
peripheral. It is *not* for NVIDIA developers contributing to
DOCA Device Emulation itself, and it is *not* the right
artifact for users who want a packaged emulated-device daemon
they do not have to write the backend for (the DOCA SNAP
Service and the DOCA Virtio-net Service are the packaged
options that build on top of this library).

**Language scope.** DOCA Device Emulation ships as a C library;
this skill covers three sub-libraries end-to-end. Select the exact
installed `pkg-config` module for the user's emulation class (see
the sub-library selection table in
the capabilities companion).
The shipped samples under
`/opt/mellanox/doca/samples/doca_devemu/` are
written in C. C and C++ consumers are the canonical case and
the worked examples in the tasks companion assume that path.
Other-language consumers (Rust, Go, Python, …) consume the
same `*.so` files through FFI or language-specific bindings;
the skill's contribution in that case is to keep the
sub-library selection, umbrella lifecycle, capability-discovery,
permission, and error-taxonomy guidance language-neutral, and
to route the agent to the public C ABI as the authoritative
surface that any wrapper will eventually call.

## When to load this skill

Load this skill when the user is doing hands-on DOCA Device
Emulation work from the DPU side, in any language. Concretely:

- Deciding which Device Emulation sub-library (PCI Generic,
  virtio-net, virtio-fs) the user needs — the umbrella
  selection question is *this skill's load-bearing first move*.
- Initializing the per-sub-library DOCA Core context on the
  DPU (one context per emulated device per sub-library) and
  configuring the doorbell / DMA primitives the host's PCIe
  driver will interact with.
- Reading per-sub-library capability surface via the
  `doca_devemu_pci_cap_*`, `doca_devemu_virtio_cap_*`, or
  `doca_devemu_vfs_cap_*` query families against the
  active `doca_devinfo` BEFORE assuming a particular feature
  bit or device characteristic is available.
- Choosing between writing the backend with `doca-devemu`
  yourself and adopting a packaged service (DOCA SNAP Service
  / DOCA Virtio-net Service) that already wraps this library.
- Debugging a `DOCA_ERROR_*` returned from a `doca_devemu_*`
  call — in particular disambiguating *firmware-level
  emulation type not enabled* from *BlueField generation does
  not support this sub-library at all* from *DPU-side process
  lacks privilege* from *host-side kernel driver did not bind*.
- Designing or extending non-C bindings (Rust, Go, Python, …)
  that wrap one of the device-emulation sub-libraries — for
  the sub-library selection, umbrella lifecycle, capability-
  discovery, permission, and error-taxonomy rules the wrapper
  must honor.

Do **not** load this skill for general DOCA orientation,
install of DOCA itself, the host-side kernel driver for the
emulated device class (virtio-net / virtio-blk / virtio-fs
kernel drivers ship with the host kernel and are not part of
DOCA), the packaged SNAP / Virtio-net services (they are
separate artifacts with their own service guides), or for
standard NIC behavior on the BlueField data path (use
`doca-flow/SKILL.md` +
`doca-eth/SKILL.md` instead — Device Emulation
is for *custom* emulated devices, not for shaping the
BlueField's built-in NIC personality).

## What this skill provides

This is a **thin loader**. The body keeps only the orientation
needed to pick the right next file. The substantive Device
Emulation-specific material lives in two companion files:

- the capabilities companion — what Device Emulation can express on
  this version + this BlueField generation + this firmware:
  the umbrella architecture (host sees an emulated PCIe
  device; DPU runs the backend), the sub-library selection
  rule (PCI Generic vs virtio-net vs virtio-fs), the per-
  sub-library Core context shape, the doorbell / DMA
  primitives that bridge host ↔ DPU, the per-sub-library
  capability-query family (`doca_devemu_*_cap_*`), the
  per-sub-library `pkg-config` module name, the Device
  Emulation error taxonomy mapped onto the cross-library
  `DOCA_ERROR_*` set, the observability surface, the
  library-vs-packaged-service path-selection rule, and the
  safety policy that gates env preconditions (DPU-side
  privileges, BlueField firmware-level emulation type
  enablement, BlueField generation actually supporting the
  emulation class).
- the tasks companion — step-by-step workflows for the six in-scope
  Device Emulation verbs: `configure`, `build`, `modify`,
  `run`, `test`, `debug`. Plus a `Deferred task verbs` block
  that points out-of-scope questions at the right next skill.

The skill assumes a host + BlueField pair where DOCA is
already installed at the standard location on both sides, the
BlueField firmware has the emulation type the user wants to
build enabled, the user has the privileges their public
install profile expects (in particular, sudo on the DPU side
to perform PCIe-level emulation), and the host kernel ships
the standard driver for the emulated device class the user is
building. It does not cover installing DOCA, flipping
firmware-level configuration, or installing host-side kernel
drivers — those paths go through
`doca-setup/SKILL.md`.

## Loading order

1. Read this skill first to confirm the user's question
   is in scope (custom emulated PCIe device built on the
   `doca-devemu` library, not the packaged SNAP /
   Virtio-net services, not the host-side kernel driver, not
   standard NIC behavior).
2. **For the umbrella architecture, the sub-library selection
   rule (PCI Generic vs virtio-net vs virtio-fs), the per-
   sub-library Core context shape, the doorbell / DMA
   primitives, the per-sub-library capability-query rule,
   the per-sub-library `pkg-config` modules, the library-vs-
   packaged-service path-selection rule, the error taxonomy,
   the observability surface, and the safety policy, see
   the capabilities companion.**
3. **For step-by-step workflows — configure, build, modify,
   run, test, debug — see the tasks companion.**

Both companion files cross-link to each other,
`doca-version/SKILL.md` for the
canonical DOCA version-handling rules (with the Device
Emulation overlay that the chosen sub-library's `pkg-config`
module plus the firmware-level emulation slot plus the
`doca_devemu_*_cap_*` query are all part of *"is this
emulation supported here"*), and
`doca-public-knowledge-map/SKILL.md`
whenever the right answer is "look it up in the public DOCA
Device Emulation umbrella guide, the per-sub-library guide
linked from it, the DOCA SNAP / Virtio-net Service guide, or
in the on-disk install layout" rather than "Device Emulation-
specific guidance".

## Related skills

- `doca-public-knowledge-map/SKILL.md` —
  the routing table for every public DOCA documentation
  source and the on-disk layout of an installed DOCA
  package. The Device Emulation umbrella URL is
  <https://docs.nvidia.com/doca/sdk/DOCA-Device-Emulation/index.html>;
  per-sub-library guides are linked from the umbrella, and
  the packaged DOCA SNAP / Virtio-net services are listed
  under the DOCA Services umbrella as separate artifacts.
- `doca-setup/SKILL.md` — env preparation,
  install verification, DPU-side privilege checks, BlueField
  firmware configuration (including the per-sub-library
  emulation type enable), and the *I have no install yet*
  path with the public NGC DOCA container. This skill assumes
  its preconditions are satisfied AND that the firmware-level
  emulation type for the user's chosen sub-library is enabled.
- `doca-version/SKILL.md` — canonical
  DOCA version-handling rules. This skill's Version
  compatibility overlay cross-links the four-way match rule
  and adds the Device Emulation-specific per-sub-library
  `pkg-config` + `doca_devemu_*_cap_*` overlay.
- `doca-structured-tools-contract/SKILL.md` —
  the bundle's structured-tools precedence rule (detect /
  prefer / fall back / report). The Command appendix in
  the tasks companion honors this contract.
- `doca-programming-guide/SKILL.md` —
  general DOCA programming patterns shared by every library:
  the canonical `pkg-config` + meson build pattern, the
  universal modify-a-shipped-sample first-app workflow, the
  universal Core-context lifecycle, the cross-library
  `DOCA_ERROR_*` taxonomy, and the program-side debug order.
  This skill layers Device Emulation specifics on top.
- `doca-comch/SKILL.md` — the right library
  when the user wants a *control-plane channel* between a
  host process and a DPU process over PCIe but does NOT want
  the host to see the DPU as an emulated PCIe device with a
  standard host kernel driver. Device Emulation hides the
  DPU behind a real-looking PCIe device class; Comch is an
  explicit host ↔ DPU IPC.
- `doca-flow/SKILL.md` and
  `doca-eth/SKILL.md` — the right libraries
  when the user wants to shape traffic on the BlueField's
  built-in NIC personality, not expose a *custom* emulated
  PCIe device. Device Emulation is for *new* emulated PCIe
  devices the host did not previously see; Flow / Eth are
  for the BlueField's existing NIC surface.
- `doca-debug/SKILL.md` — the cross-cutting
  debug ladder (install / version / build / link / runtime /
  program / driver). Device Emulation-specific debug
  (firmware-level emulation type not enabled, sub-library
  mis-selected, host kernel driver not binding to the
  emulated device, virtio feature-negotiation failures)
  overlays on top of that ladder.
