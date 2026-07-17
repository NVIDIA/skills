# doca-devemu — reference detail

Moved out of `SKILL.md` to keep the loader under the per-file size budget. This is supporting detail, not routing logic.

## Example questions this skill answers well

The CLASSES of Device Emulation questions this skill is built
to answer, each with one worked example. The agent should
treat the *class* as the load-bearing piece — the worked
example is a single instance.

- **"How do I expose a custom emulated PCIe device from the
  BlueField DPU to the host?"** — worked example: *"expose a
  virtio-net device from the BlueField so the host sees a
  virtio NIC backed by my own DPU-side code"*. Answered by
  the sub-library selection rule + the umbrella lifecycle in
  [`CAPABILITIES.md ## Capabilities and modes`](../CAPABILITIES.md#capabilities-and-modes)
  + the bring-up steps in
  [`TASKS.md ## configure`](../TASKS.md#configure).
- **"Which sub-library do I need — PCI Generic, virtio-net,
  or virtio-fs?"** — worked example: *"the host needs to see
  a custom block-like device that does not match any standard
  virtio class"*. Answered by the sub-library selection table
  in
  [`CAPABILITIES.md ## Capabilities and modes`](../CAPABILITIES.md#capabilities-and-modes)
  + the routing decision in
  [`TASKS.md ## configure`](../TASKS.md#configure) step 2.
- **"Does this BlueField + firmware actually let me emulate
  the device class I want, and what capabilities does my DOCA
  install expose for it?"** — worked example: *"can I emulate
  a virtio-fs device on this BlueField, and which of its
  feature bits is supported?"*. Answered by the dual-axis
  precondition rule (firmware-level emulation type must be
  enabled AND `doca_devemu_<sub>_cap_*` against the active
  `doca_devinfo` must agree) in
  [`CAPABILITIES.md ## Capabilities and modes`](../CAPABILITIES.md#capabilities-and-modes)
  + the env-precondition checklist in
  [`TASKS.md ## configure`](../TASKS.md#configure) step 1.
- **"Is this `doca-devemu` library the right tool,
  or should I use the DOCA SNAP Service / DOCA Virtio-net
  Service?"** — worked example: *"I want NVMe storage to the
  host without writing any DPU-side backend code"*. Answered
  by the library-vs-service path-selection rule in
  [`CAPABILITIES.md ## Capabilities and modes`](../CAPABILITIES.md#capabilities-and-modes)
  + the deferred-topic boundaries in
  [`CAPABILITIES.md ## Deferred topic boundaries`](../CAPABILITIES.md#deferred-topic-boundaries)
  which route to
  [`doca-public-knowledge-map`](../../../doca-public-knowledge-map/SKILL.md)
  for the packaged-service guides.
- **"Why does my virtio-net emulation create call return
  `DOCA_ERROR_NOT_PERMITTED` even though my DPU-side process
  has `doca_dev` access?"** — worked example: *"sudo on the
  DPU is fine but the virtio-net emulation slot is disabled
  in BlueField firmware"*. Answered by the dual-axis
  permission matrix in
  [`CAPABILITIES.md ## Safety policy`](../CAPABILITIES.md#safety-policy)
  + the firmware-side env fix routed via
  [`TASKS.md ## configure`](../TASKS.md#configure) step 1.
- **"Is this Device Emulation sub-library / capability on my
  installed DOCA version?"** — worked example: *"is the
  virtio-fs emulation surface available on the DOCA install I
  have, against this BlueField generation?"*. Answered by the
  version-compatibility overlay in
  [`CAPABILITIES.md ## Version compatibility`](../CAPABILITIES.md#version-compatibility),
  which cross-links the canonical detection chain in
  [`doca-version`](../../../doca-version/SKILL.md) and adds the
  per-sub-library `pkg-config` + `doca_devemu_*_cap_*` overlay.
- **"What does this `DOCA_ERROR_*` from a `doca_devemu_*` call
  mean and which layer caused it?"** — worked example:
  *"`DOCA_ERROR_NOT_SUPPORTED` from a virtio-net emulation
  create call — is it the BlueField generation, the firmware
  slot, or the DOCA version?"*. Answered by the Device
  Emulation overlay on the cross-library taxonomy in
  [`CAPABILITIES.md ## Error taxonomy`](../CAPABILITIES.md#error-taxonomy)
  + the layered ladder in
  [`TASKS.md ## debug`](../TASKS.md#debug) that escalates to
  [`doca-debug`](../../../doca-debug/SKILL.md).

## What this skill deliberately does not ship

This skill is **agent guidance**, not a samples or templates
bundle. To keep the boundary clean, it deliberately does not
contain — and pull requests should not add:

- **Pre-written DOCA Device Emulation application source
  code, in any language.** The verified source is the shipped
  C samples at
  `/opt/mellanox/doca/samples/doca_devemu/` (with
  sub-directories per sub-library). The agent's job is to
  route the user to those files and prescribe a minimum-diff
  modification on them via the universal modify-a-sample
  workflow in
  [`doca-programming-guide`](../../../doca-programming-guide/SKILL.md),
  layered with the Device Emulation-specific overrides in
  [`TASKS.md ## modify`](../TASKS.md#modify).
- **A specific emulated-device backend.** This library
  *provides the framework* for emulated PCIe devices; it does
  *not* implement a specific storage backend, a specific
  packet processor, or a specific filesystem. The agent must
  refuse to invent backend bodies and must route any *"what
  should my backend do"* question to the user's domain
  expertise and to the public sub-library guides reachable
  via
  [`doca-public-knowledge-map`](../../../doca-public-knowledge-map/SKILL.md).
- **Standalone build manifests** (`meson.build`,
  `CMakeLists.txt`, `Cargo.toml`, …) parked inside the skill.
  The agent constructs the build manifest *in the user's
  project directory* against the user's installed DOCA, where
  `pkg-config --modversion <the chosen sub-library's module>`
  is the source of truth.
- **A `samples/`, `bindings/`, or `reference/` subtree** of
  any kind. A mock or incomplete artifact in this skill's
  tree, even one labeled "reference", is misleading: users
  will read it as buildable.
- **DOCA SNAP Service / DOCA Virtio-net Service surface.**
  Those services are *separate artifacts* built on top of
  this library, with their own public service guides. Routing
  for them lives in
  [`doca-public-knowledge-map`](../../../doca-public-knowledge-map/SKILL.md);
  conflating either service with the `doca-devemu`
  library is the single most common Device Emulation first-app
  design error.

## Related skills

- [`doca-public-knowledge-map`](../../../doca-public-knowledge-map/SKILL.md) —
  the routing table for every public DOCA documentation
  source and the on-disk layout of an installed DOCA
  package. The Device Emulation umbrella URL is
  <https://docs.nvidia.com/doca/sdk/DOCA-Device-Emulation/index.html>;
  per-sub-library guides are linked from the umbrella, and
  the packaged DOCA SNAP / Virtio-net services are listed
  under the DOCA Services umbrella as separate artifacts.
- [`doca-setup`](../../../doca-setup/SKILL.md) — env preparation,
  install verification, DPU-side privilege checks, BlueField
  firmware configuration (including the per-sub-library
  emulation type enable), and the *I have no install yet*
  path with the public NGC DOCA container. This skill assumes
  its preconditions are satisfied AND that the firmware-level
  emulation type for the user's chosen sub-library is enabled.
- [`doca-version`](../../../doca-version/SKILL.md) — canonical
  DOCA version-handling rules. This skill's `## Version
  compatibility` cross-links the four-way match rule and adds
  the Device Emulation-specific per-sub-library `pkg-config`
  + `doca_devemu_*_cap_*` overlay.
- [`doca-structured-tools-contract`](../../../doca-structured-tools-contract/SKILL.md) —
  the bundle's structured-tools precedence rule (detect /
  prefer / fall back / report). The Command appendix in
  [TASKS.md](../TASKS.md) honors this contract.
- [`doca-programming-guide`](../../../doca-programming-guide/SKILL.md) —
  general DOCA programming patterns shared by every library:
  the canonical `pkg-config` + meson build pattern, the
  universal modify-a-shipped-sample first-app workflow, the
  universal Core-context lifecycle, the cross-library
  `DOCA_ERROR_*` taxonomy, and the program-side debug order.
  This skill layers Device Emulation specifics on top.
- [`doca-comch`](../../doca-comch/SKILL.md) — the right library
  when the user wants a *control-plane channel* between a
  host process and a DPU process over PCIe but does NOT want
  the host to see the DPU as an emulated PCIe device with a
  standard host kernel driver. Device Emulation hides the
  DPU behind a real-looking PCIe device class; Comch is an
  explicit host ↔ DPU IPC.
- [`doca-flow`](../../doca-flow/SKILL.md) and
  [`doca-eth`](../../doca-eth/SKILL.md) — the right libraries
  when the user wants to shape traffic on the BlueField's
  built-in NIC personality, not expose a *custom* emulated
  PCIe device. Device Emulation is for *new* emulated PCIe
  devices the host did not previously see; Flow / Eth are
  for the BlueField's existing NIC surface.
- [`doca-debug`](../../../doca-debug/SKILL.md) — the cross-cutting
  debug ladder (install / version / build / link / runtime /
  program / driver). Device Emulation-specific debug
  (firmware-level emulation type not enabled, sub-library
  mis-selected, host kernel driver not binding to the
  emulated device, virtio feature-negotiation failures)
  overlays on top of that ladder.

The DOCA SNAP Service and the DOCA Virtio-net Service are
**not in scope** for this skill — those are packaged daemons
built on top of `doca-devemu` for users who do not
want to write the backend themselves. They have their own
public service guides reachable via
[`doca-public-knowledge-map`](../../../doca-public-knowledge-map/SKILL.md).
Conflating either service with this library is the single
most common Device Emulation first-app design error.

