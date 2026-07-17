# DOCA Device Emulation workflows

**Where to start:** The verbs run `configure → build → modify →
run → test → debug`. Skip ahead only when the user is already
past a verb. The `## test` verb is an iterative loop
(single-emulated-device smoke → host-driver bind → basic
operation → loop back if a precondition changes), not a
one-shot pass — see the eval-loop overlay in `## test` below.

Read this file when the loader sent you here from
[SKILL.md](SKILL.md). For the umbrella architecture, the
sub-library selection rule, the per-sub-library Core context,
the doorbell / DMA primitives, the per-sub-library
capability-discovery rule, the library-vs-packaged-service
path-selection rule, the error taxonomy, the observability
surface, and the safety policy, see
[CAPABILITIES.md](CAPABILITIES.md). For the cross-library DOCA
patterns layered under everything below (the universal Core
lifecycle, the cross-library `DOCA_ERROR_*` taxonomy, the
modify-a-shipped-sample workflow), see
[`doca-programming-guide`](../../doca-programming-guide/SKILL.md).

Each verb below describes the **shape of the workflow**, not a
copy-paste recipe. The agent's job is to walk the user through
the steps in order, verifying preconditions before recommending
the next call.

## configure

Goal: stand up a per-sub-library Device Emulation context on
the DPU side against a `doca_dev` that maps to a BlueField
whose firmware has the matching emulation type enabled, wire
the per-sub-library doorbell / DMA primitives, confirm both
the DPU side and the host side are in a state where starting
the context will produce a visible emulated PCIe device on the
host that the host's kernel driver can bind, and confirm the
user has picked the right sub-library before any code is
written.

Steps the agent should walk the user through:

1. **Confirm the env preconditions on BOTH sides.** Per the
   env-precondition matrix in
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy):
   on the DPU side, the user / process can perform PCIe-level
   emulation (typically sudo on the DPU); on the BlueField,
   the firmware has the matching emulation type enabled (this
   is a firmware-level slot the user must verify is on, NOT
   a casual setting — it is the precondition baseline agents
   most often miss, and a disabled slot will look like a
   permission bug at the `doca_devemu_<sub>_*` API surface);
   on the host side, the host kernel ships the standard
   driver for the emulated device class the user picked. If
   ANY of these fails, this is an env / firmware / host-
   kernel problem to fix via
   [`doca-setup TASKS.md ## configure`](../../doca-setup/TASKS.md#configure),
   NOT a code change in the DPU-side program.
2. **Pin the sub-library BEFORE writing any code.** Per the
   sub-library selection table in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes),
   walk the rows with the user: PCI Generic (raw PCIe device
   the host binds via its generic PCIe driver model),
   virtio-net (emulated NIC the host binds via the upstream
   `virtio_net` driver), or virtio-fs (emulated filesystem
   the host binds via the upstream `virtio_fs` driver). If
   the user's intent does not pin one of the rows, stop and
   ask. Picking a sub-library *for* the user when the rows do
   not pin one is a wrong answer regardless of how clean the
   rest of the setup goes. If the user's intent is actually a
   packaged service (NVMe / virtio-blk SNAP-style storage, or
   a packaged virtio-net daemon), route via
   [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
   to the DOCA SNAP Service / DOCA Virtio-net Service guide —
   the user does not need this library at all in that case.
3. **Run per-sub-library capability discovery against the
   active `doca_devinfo`.** Per the capability-query rule in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes),
   run the matching `doca_devemu_<sub>_cap_*` family for the
   sub-library the user pinned in step 2 (e.g. for virtio-net,
   the `doca_devemu_virtio_cap_*` family narrowed to the
   virtio-net surface). Quote the queried values back to the
   user; do not assume from prior installs and do not quote
   numbers from memory. If the cap-query says *not
   supported*, that axis is the answer — do not proceed.
4. **Create the per-sub-library Core context against the
   chosen `doca_dev`.** Pick the `doca_dev` deliberately: it
   must map to the BlueField whose firmware has the matching
   emulation type enabled. A DPU exposing more than one
   emulated device in the same sub-library needs one
   per-sub-library context per device; there is no *"global
   Device Emulation context"*. This is a standard DOCA Core
   context create — the universal lifecycle from
   [`doca-programming-guide TASKS.md ## configure`](../../doca-programming-guide/TASKS.md#configure)
   applies; this skill adds the per-sub-library overlay.
5. **Wire the per-sub-library doorbell / DMA primitives.**
   Per the doorbell / DMA table in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes),
   register the doorbell that lets the DPU-side backend react
   to host-driver activity (e.g. virtio queue kicks for
   virtio-net / virtio-fs; raw PCIe register / BAR accesses
   for PCI Generic) and the DMA primitives that move payload
   bytes between host and DPU memory under the emulated
   device's contract. A configure pass that omits the doorbell
   wiring will appear to load successfully — the host driver
   may even bind — but no host-driver activity will produce
   any DPU-side reaction; the agent must explicitly verify the
   wiring before recommending a smoke run.
6. **Start the per-sub-library context** via
   `doca_ctx_start()`. This is the moment the emulated PCIe
   device becomes visible to the host. Per the lifecycle
   ordering in
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy),
   record the order so teardown happens in reverse: detach
   the host kernel driver / unload the host kernel module →
   stop the per-sub-library context → destroy emulated devices
   → destroy the per-sub-library context → close the
   `doca_dev`.
7. **Confirm the emulated device is visible on the host AND
   bound to its kernel driver.** Per the host-side
   observability rule in
   [`CAPABILITIES.md ## Observability`](CAPABILITIES.md#observability),
   `lspci` on the host shows the emulated device after
   `doca_ctx_start()` returns; the matching kernel driver's
   sysfs entry shows the driver is bound; the host's `dmesg`
   around the moment the DPU started the context shows a
   clean bind. If `lspci` does not show the device, that is
   most often a firmware-side enable that has not taken
   effect yet (a BlueField reset after the firmware-side flip
   is typically required); if `lspci` shows the device but no
   driver is bound, that is most often a host kernel that
   ships no driver for the emulated class. Both fixes are
   host- or firmware-side, not DPU-side code changes.
8. **Sanity check before any production use.** Confirm with
   the user: which sub-library was picked, which `doca_dev`
   the per-sub-library context was created against, which
   BlueField generation that maps to and whether its firmware
   has the matching emulation type enabled, what backend
   logic the user is running on the DPU side, what kernel
   driver the host has bound to the emulated device, and
   what they expect the first observable host-side operation
   to do. If any of those are unclear, stop and ask — do not
   invent.

For the canonical DOCA universal lifecycle that underlies
steps 4-6, see
[`doca-programming-guide TASKS.md ## configure`](../../doca-programming-guide/TASKS.md#configure).
This skill adds the Device Emulation overlay; do not
re-explain the lifecycle here.

## build

Goal: compile a Device Emulation-using DPU-side consumer
against the user's installed DOCA, using the canonical
cross-library build pattern, with `pkg-config` against the
chosen sub-library's module as the source of truth for include
+ link flags.

The build pattern for any DOCA C / C++ consumer is fully
documented in
[`doca-programming-guide TASKS.md ## build`](../../doca-programming-guide/TASKS.md#build).
This skill carries only the Device Emulation-specific overlay:

| Slot | Value | Why it matters |
| --- | --- | --- |
| `pkg-config` module name | Per-sub-library: a different module for PCI Generic, virtio-net, and virtio-fs. The exact module strings live in the public Device Emulation umbrella guide and the per-sub-library guides linked from it; read them off the install via `ls /opt/mellanox/doca/infrastructure/lib/pkgconfig/` and the per-sub-library guide reachable through [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) | Wrong module name = `pkg-config: Package '<module>' was not found`. Choosing the umbrella's module when the user picked a specific sub-library — or vice versa — is a common first-build error; the agent should quote the module name from the user's own `pkg-config` install, not from memory |
| Include flags | `pkg-config --cflags <the chosen sub-library's module>` | Resolves the per-sub-library headers under $(pkg-config --variable=includedir doca-common); the included headers are sub-library-specific (PCI Generic headers do not declare virtio symbols, and vice versa) |
| Link flags | `pkg-config --libs <the chosen sub-library's module>` | Pulls in the per-sub-library shared object plus `doca-common`. The companion libraries the link line needs (in particular `doca-common`) are resolved transitively |
| Companion DOCA libs | `doca-argp` for the standard DOCA argument-parsing surface (if the consumer uses the standard arg style); other DOCA libraries only if the user's DPU-side backend uses them independently | Adding unnecessary companion libraries bloats the link line and obscures real partial-install issues |
| Header check | The chosen sub-library's primary header resolvable under $(pkg-config --variable=includedir doca-common) after `pkg-config --cflags <module>` resolves | If `pkg-config --cflags <module>` resolves but the include is missing, the install is partial — route to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug) layer 2 |
| Minimum required DOCA version | Query with `pkg-config --modversion <the chosen sub-library's module>`; never hardcode in build files | Cross-version build / runtime mixing breaks per [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility) |

For non-C consumers (Rust, Go, Python), the link surface is
the same per-sub-library `*.so` files; the FFI wrapper layer
is the language-specific binding and is out of scope for this
skill — but the per-sub-library `pkg-config` module the
wrapper is built against is still the load-bearing input.

The agent's anti-pattern alert: building against the wrong
sub-library's `pkg-config` module is the most common Device
Emulation first-build error, because the umbrella name
*sounds* like it would be a single module and is not. The
agent should re-quote the user's chosen sub-library back from
step 2 of [`## configure`](#configure) before drafting any
build manifest.

## modify

Goal: take the closest-fitting shipped Device Emulation sample
(under the matching sub-library subdirectory) as the verified
starting point and apply a **minimum diff** to make it match
the user's intent, without rewriting from scratch and without
crossing the sub-library boundary in-place.

The universal modify-a-shipped-sample workflow is in
[`doca-programming-guide TASKS.md ## modify`](../../doca-programming-guide/TASKS.md#modify);
this skill provides the Device Emulation-specific slot fill.

| Slot | What the agent asks the user | Device Emulation-specific consideration |
| --- | --- | --- |
| 1. Starting sample | Which sample subdirectory under `/opt/mellanox/doca/samples/doca_devemu/` — the PCI Generic samples (`devemu_pci_*`) or the virtio-fs samples (`devemu_vfs_*`); virtio-net / virtio-blk / NVMe sub-libraries ship no in-tree sample (route those via the packaged SNAP / Virtio-net services or build from the public headers) — and which sample inside it? | Pick the sample whose sub-library matches the user's pinned sub-library from [`## configure`](#configure) step 2. Cross-sub-library re-use is a sample swap, not an in-place tweak — the API surfaces do not interchange and the build manifest must change. Pick the sample whose *shape* (single-device vs multi-device; one-queue vs multi-queue if applicable; the operation set it demonstrates) is closest to the user's intent |
| 2. Backend body | What is the backend actually doing (packet forwarding logic for virtio-net, storage backend for a SNAP-style PCI Generic device, filesystem implementation for virtio-fs)? | The agent's anti-pattern alerts: (a) do NOT invent a backend body in this skill — backend design is a domain question (storage / networking / filesystem), not an API question, route via [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md); (b) do NOT recommend re-implementing what the packaged DOCA SNAP / Virtio-net services already provide unless the user has confirmed those services do not meet their needs |
| 3. Device descriptor / identity values | What device identity does the host see (vendor / device IDs for PCI Generic; virtio device class identifiers and feature-bit subset for virtio-net / virtio-fs)? | Per the `INVALID_VALUE` row in [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy), a malformed or non-cap-supported descriptor is the most common first-submit failure. Re-validate against the per-sub-library cap-query from [`## configure`](#configure) step 3 before accepting the modify |
| 4. Doorbell / DMA wiring | Does the sample's doorbell / DMA wiring still match what the modified backend needs? | A modify that changes the backend without updating the doorbell / DMA wiring is a common way to introduce *"host driver is active but DPU sees nothing"* — surface this explicitly and re-check the wiring against [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes) doorbell / DMA table |
| 5. Re-validate against per-sub-library capabilities | Re-run the `doca_devemu_<sub>_cap_*` queries from [`## configure`](#configure) step 3 against the modified configuration — descriptor changes, feature-bit changes, or queue-sizing changes can flip a per-sub-library capability boundary | Per the cross-cutting rule in [`doca-version CAPABILITIES.md ## Observability`](../../doca-version/CAPABILITIES.md#observability), the cap query is the runtime authority |
| 6. Keep the build manifest pointing at the same sub-library | The sample's existing `meson.build` already wires the per-sub-library `pkg-config` module; do not switch to a hand-rolled Makefile or — worse — silently swap to a different sub-library's module | Per the build slot table in [`## build`](#build) |

The agent emits an *intent description + the filled slots*;
the *actual* unified diff against the sample source is
produced the way every other library skill in this bundle
handles modify — the agent walks the user through the diff
line-by-line against the sample source they read on disk, and
has the user paste back the result for validation. The agent's
anti-pattern alert: a *"clean rewrite"* from scratch is
almost always slower to first green than a minimum-diff
modify on a shipped Device Emulation sample, and removes the
user's ability to bisect against a known-good baseline.

## run

Goal: actually start the per-sub-library Device Emulation
context, observe the emulated PCIe device appear on the host,
see the host's standard kernel driver bind to it, and confirm
basic backend operation before any production use.

Steps the agent should walk the user through:

1. **Confirm the loaded sub-library and the `doca_dev`
   match the user's intent.** A DPU-side program that links
   cleanly but produces no host-side enumeration is most
   often opening the wrong `doca_dev` or running the wrong
   sub-library's context against a BlueField whose firmware
   has a *different* sub-library's emulation type enabled.
   Re-quote the per-sub-library cap-query output from
   [`## configure`](#configure) step 3 and confirm the
   sub-library and the device agree.
2. **Start the DPU-side context first.** The DPU-side context
   must be in `RUNNING` before the host can enumerate the
   emulated PCIe device. After `doca_ctx_start()` returns,
   verify with `lspci` on the host that the emulated device
   appears, and verify with the matching kernel driver's
   sysfs entry that the driver bound to it. Per
   [`CAPABILITIES.md ## Observability`](CAPABILITIES.md#observability),
   absence of host-side enumeration is most often a
   firmware-side enable that has not taken effect yet (a
   BlueField reset is typically required); absence of driver
   bind with the device enumerated is most often a host
   kernel without the matching driver module loaded.
3. **Capture the structured log on first failure.** Set
   `DOCA_LOG_LEVEL=trace` on the DPU side for the first run
   (see
   [`doca-debug CAPABILITIES.md ## Observability`](../../doca-debug/CAPABILITIES.md#observability));
   capture the host's `dmesg` around the moment the DPU
   started the context. The two log streams are the cheapest
   way to make the *DPU-side context start* / *host-side PCI
   enumeration* / *host-side driver bind* sequence visible.
4. **Drive the DPU-side `doca_pe_progress` loop.** The
   doorbell reactions and the DMA completions for the
   emulated device flow through the progress engine on the
   DPU side; a DPU-side program that starts the context and
   then blocks without progressing the PE will see no
   doorbell reactions and conclude *"the host driver is
   broken"* incorrectly. This is the cross-library *"PE not
   progressed"* failure mode applied to Device Emulation.
5. **Confirm basic operation on the host side.** For
   virtio-net, send a single packet (or `ip link set` the
   device up) and confirm the DPU-side backend reacts and
   the host sees the result. For virtio-fs, mount the
   exposed filesystem and read/write a small file. For PCI
   Generic, read / write a known register the user's backend
   implements. *Basic operation working* is the only
   evidence that the umbrella is wired end-to-end; absence
   of an error log is not.

For the runtime version + `LD_LIBRARY_PATH` cross-checks that
underlie *"the program built but does nothing"*, see
[`doca-version TASKS.md ## run`](../../doca-version/TASKS.md#run).

## test

Goal: prove the configured Device Emulation deployment
actually exposes the chosen emulated PCIe device to the host,
that the host's standard kernel driver binds to it, that the
DPU-side backend reacts to host driver activity, and that the
basic operation pattern for the chosen sub-library works
end-to-end, on the user's exact hardware + firmware + DOCA
combo.

This is **a loop, not a one-shot pass.** Each iteration
narrows either the sub-library selection, the env-precondition
set, the per-sub-library capability set, the doorbell / DMA
wiring, the host-side driver bind state, or the backend
behavior. The loop terminates when either (a) the user
observes the host-side standard kernel driver binding to one
emulated device and basic operation working end-to-end, or (b)
the agent has narrowed the failure cause to a layer outside
Device Emulation itself (BlueField firmware configuration,
host kernel driver missing, DOCA install) and escalated to
the matching skill.

Iteration shape:

1. **Single-emulated-device smoke.** Stand up ONE emulated
   device in the chosen sub-library — one virtio-net device,
   one virtio-fs device, or one PCI Generic device with a
   minimal register set — and confirm: the DPU-side
   `doca_ctx_start()` succeeds; the host's `lspci` shows the
   device; the matching kernel driver binds to it cleanly
   (host `dmesg` shows no bind error); a basic operation
   (one packet through virtio-net, one file open through
   virtio-fs, one register read through PCI Generic) works.
   If yes, advance. If no — per the safety-policy rule in
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy),
   DO NOT scale a broken smoke into a multi-device or
   sophisticated-backend deployment.
2. **Host-driver bind cross-check.** With the smoke device
   up, confirm host-side `lspci`, the matching kernel
   driver's sysfs / debugfs entries, and host `dmesg` agree:
   the device is enumerated; the driver is bound; no
   feature-negotiation error is logged. A device that is
   enumerated but not bound is *almost always* a host-kernel
   driver gap (the driver module is not loaded, or not
   shipped), not a DPU-side bug. Mismatches surface here, not
   at DPU-side configure time.
3. **DPU-side reaction cross-check.** Trigger a host-side
   operation (a packet, a file op, a register access
   appropriate to the sub-library) and confirm the DPU-side
   backend reacts via the wired doorbell. *No DPU-side
   reaction with the host clearly active* is *always* a
   missing `doca_pe_progress()` on the DPU side OR a
   doorbell wired against the wrong sub-library context — not
   a hardware bug.
4. **Sustained / multi-operation run.** Run the basic
   operation pattern repeatedly (many packets for virtio-net,
   many file ops for virtio-fs, repeated register access for
   PCI Generic) and confirm both sides remain healthy. This
   catches DPU-side completion-queue sizing bugs and host-
   side driver issues that only emerge under load.
5. **Multi-device or feature-bit-expansion follow-up (only
   if needed).** Once a single-device smoke is fully green,
   scale to multiple emulated devices in the same sub-library
   or opt in to additional feature bits (virtio-net /
   virtio-fs). Re-run the per-sub-library cap-query before
   each expansion to confirm the new shape is within the
   per-sub-library capability surface.

Eval-loop overlay — why this is a loop, not a one-shot pass:

| Iteration trigger | What it looks like | What changes next iteration |
| --- | --- | --- |
| `DOCA_ERROR_NOT_PERMITTED` on a BlueField the agent expected to work | DPU-side `doca_dev` access is fine on other libraries but `doca_devemu_<sub>_*` create / start fails | Firmware-side emulation-type slot is the most likely culprit; re-run the firmware-side check; the BlueField may need a reset after the slot is flipped before the new state takes effect. Do NOT diagnose this as a DPU-OS permission problem first |
| `DOCA_ERROR_NOT_SUPPORTED` on a BlueField the agent expected to work | The per-sub-library cap-query failed against the active `doca_devinfo` | The BlueField generation axis was missed (older BlueField may not support the chosen sub-library at all); re-run the BlueField identification check; if the BlueField genuinely does not support this sub-library, the answer is the hardware or a different sub-library, not a code change |
| DPU-side context starts cleanly; host `lspci` shows no device | The firmware-side slot enable has not taken effect; OR the DPU-side started a context for the wrong `doca_dev` | Confirm the firmware-side slot is on AND the BlueField has been reset since the slot was flipped; confirm the DPU-side context is created against the `doca_dev` for the BlueField the host actually sees |
| Host `lspci` shows the device but no kernel driver binds | The host kernel ships no driver for the emulated class, OR the driver module is not loaded, OR (virtio-net / virtio-fs) feature negotiation between the host driver and the emulated device failed | Load the matching kernel module on the host; if the driver still does not bind, check host `dmesg` for the feature-negotiation error and re-check the per-sub-library cap-query opt-in vs the host driver's expected feature set. The fix is host-side or sub-library-config-side, not DPU-backend-body-side |
| Host driver bound; basic operation produces no DPU-side reaction | Doorbell wiring wrong, OR `doca_pe_progress()` not being driven on the DPU side, OR the operation is going to the wrong queue / register | Re-walk the doorbell / DMA wiring per [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes) doorbell / DMA table; confirm the DPU-side PE is being progressed |
| Smoke works on one BlueField, fails on another | Different firmware-side emulation-type slot state, different BlueField generation, or different DOCA install | Re-run [`## configure`](#configure) step 1 (env preconditions) + [`doca-version TASKS.md ## test`](../../doca-version/TASKS.md#test) four-way match on the failing host |

Loop termination: stop iterating once two consecutive
iterations of the same kind don't change anything — that means
the cause is below Device Emulation (BlueField firmware slot,
host kernel driver, DOCA install). Escalate to
[`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug)
with the captured layer-1-through-5 evidence including BOTH
the DPU-side DOCA log and the host-side `dmesg` + `lspci`
output.

## debug

Goal: when a `doca_devemu_<sub>_*` call (or an emulated device
running on the BlueField) returns a `DOCA_ERROR_*` or does
not produce the expected host-side enumeration / driver bind /
operation, narrow the cause to a specific layer and act on it.

The cross-library debug ladder lives in
[`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug).
Walk through it in order — install → version → build → link →
runtime → program → driver — *before* recommending Device
Emulation-specific fixes. This skill's overlay names the
Device Emulation-specific manifestation at layers 5 (runtime),
6 (program), and 7 (driver):

**Layer 5 (runtime) — Device Emulation overlay.**

- An emulated device that does not appear in host `lspci`
  after `doca_ctx_start()` is *almost always* one of three
  things: the firmware-side emulation-type slot is disabled
  (or was enabled but the BlueField has not been reset
  since); the DPU-side context was created against a
  `doca_dev` for a different BlueField than the one the host
  sees; or the per-sub-library cap-query was missed at
  configure time and the sub-library is not supported on
  this combo. Confirm the env-side preconditions per
  [`## configure`](#configure) step 1 before assuming the
  DPU-side program itself is broken.
- An emulated device that appears in host `lspci` but to
  which no kernel driver binds is *almost always* a host-
  kernel gap: the driver module is not loaded, or the host
  kernel ships no driver for the emulated class, or
  (virtio-net / virtio-fs) feature negotiation between the
  host driver and the emulated device failed. Inspect host
  `dmesg` around the bind attempt; the fix is host-kernel
  side or sub-library-config side, not DPU-backend-body
  side.
- A bound driver that produces no DPU-side reaction is
  *almost always* a missing `doca_pe_progress()` on the
  DPU side, OR a doorbell registration wired against the
  wrong per-sub-library context. Confirm both before
  assuming the backend body itself is broken.

**Layer 6 (program) — Device Emulation overlay.**

- Sub-library mis-selection: a program that builds against
  the wrong sub-library's `pkg-config` module, or that creates
  a context in one sub-library while the user's intent was a
  different sub-library, will fail in non-obvious ways (the
  context may even start, but the host will see a device of
  the wrong class). Re-confirm the sub-library selection from
  [`## configure`](#configure) step 2 against the user's
  intent before any other diagnosis.
- Lifecycle ordering: per the host-driver-attached rule in
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy),
  tearing down the per-sub-library context while the host
  kernel driver is still attached returns `BAD_STATE` and
  may produce noisy host-side `dmesg` until the next
  BlueField reset. Reverse the configure order on teardown.
- Descriptor / feature-bit mismatch: `DOCA_ERROR_INVALID_VALUE`
  on a configuration call is a per-sub-library cap-vs-payload
  mismatch (virtio feature-bit not supported on this combo,
  device descriptor malformed for the chosen sub-library,
  sizing past the per-sub-library cap). Re-read the matching
  `doca_devemu_<sub>_cap_*` against the active `doca_devinfo`
  and align the configuration; do NOT silently adjust the
  value without re-reading the cap.

**Layer 7 (driver) — Device Emulation overlay.**

- `DOCA_ERROR_NOT_PERMITTED` from `doca_devemu_<sub>_*`
  create / start with DPU-side `doca_dev` access otherwise
  fine points at the BlueField firmware not having the
  matching emulation type enabled — this is a firmware-side
  fix, NOT a DPU-OS permission fix. Route to
  [`doca-setup TASKS.md ## debug`](../../doca-setup/TASKS.md#debug)
  layer 5 (driver) for the firmware-side enable; the
  BlueField typically needs a reset after the slot is
  flipped before the new state takes effect.
- `DOCA_ERROR_IO_FAILED` on host ↔ DPU interaction after
  start (host kernel driver reads / writes / kicks, DMA
  primitives moving payload) points at a host-side driver
  interaction failure below the DOCA layer. Inspect host
  `lspci` and `dmesg`; the fix is typically host-side
  (driver module load, feature-set re-alignment, kernel
  upgrade for missing driver) rather than DPU-side.
- A BlueField in the wrong mode that does NOT expose the
  emulation type at all surfaces as `DOCA_ERROR_NOT_SUPPORTED`
  at per-sub-library context create. Route to
  [`doca-setup TASKS.md ## debug`](../../doca-setup/TASKS.md#debug)
  layer 5 (driver) for the env-side BlueField mode fix; this
  is NOT a code change in the DPU-side program.

Once the layer is identified, route to the matching debug
verb on the matching skill: install / build / link / driver
to [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug);
version to
[`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug);
cross-cutting runtime to
[`doca-debug ## debug`](../../doca-debug/TASKS.md#debug);
program-layer Core-context patterns to
[`doca-programming-guide TASKS.md ## debug`](../../doca-programming-guide/TASKS.md#debug).

## Deferred task verbs

The following verbs are out of scope for this skill but are
commonly asked in the same conversations. Route them as
follows so the agent does not invent guidance:

- **install.** Installing DOCA, enabling BlueField firmware
  emulation-type slots, installing host-side kernel drivers
  (or loading kernel modules), choosing matched DOCA
  versions, post-install verification, `pkg-config` wiring —
  defer to [`doca-setup`](../../doca-setup/SKILL.md) and to
  the install-tree layout in
  [`doca-public-knowledge-map ## Layout of an installed DOCA package`](../../doca-public-knowledge-map/SKILL.md#layout-of-an-installed-doca-package).
  This skill assumes DOCA is installed, the firmware-side
  emulation type for the chosen sub-library is enabled, and
  the host kernel ships the matching driver.
- **deploy.** Deploying Device Emulation-using applications at
  scale (multi-BlueField clusters, Kubernetes operator
  workflows for emulated-device daemons, multi-tenant
  isolation between emulated devices) — out of scope for
  Phase 1 and reserved for a future platform skill. For
  single-host first-run testing the right verb in this skill
  is `## run`; do not invent a *"deploy"* workflow.
- **packaged-service adoption.** Adopting the DOCA SNAP
  Service or the DOCA Virtio-net Service as the user's
  *backend* rather than writing one with this library — out
  of scope. Route via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
  to the DOCA SNAP Services guide and the DOCA Virtio-net
  Service Guide; the services are separate artifacts with
  their own service guides.
- **backend design.** Designing the storage backend, the
  packet-forwarding logic, or the filesystem implementation
  the user runs on the DPU — out of scope. Route via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
  to the public DOCA Device Emulation umbrella guide and to
  the user's own domain expertise; this skill prescribes how
  to *expose* an emulated PCIe device, not what the backend
  should compute.
- **host-side kernel driver development.** Writing or
  modifying the host kernel driver for an emulated device
  class is upstream Linux kernel territory, not DOCA. Route
  to upstream Linux kernel documentation; this skill
  assumes the host kernel ships the standard driver for the
  chosen sub-library's emulated device class.

## Command appendix

Every command below is **cross-cutting on DOCA Device
Emulation** — it answers a recurring class of question that
comes up in the verbs above. The agent should treat the
*class* as load-bearing; the worked example is a single
instance. Run-as user is the unprivileged user unless noted. Rows that need elevated privileges call that out explicitly.

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
| `ls /opt/mellanox/doca/infrastructure/lib/pkgconfig/ \| grep -i devemu` | `## configure` step 2; `## build` `pkg-config` slot | Which Device Emulation sub-library `pkg-config` modules are installed on this DPU, and which sub-libraries are therefore available to build against? | One `.pc` entry per installed sub-library (PCI Generic, virtio-net, virtio-fs). A missing sub-library entry means the umbrella shipped but a sub-library did not — route to [`doca-setup`](../../doca-setup/SKILL.md) |
| `pkg-config --modversion <the chosen sub-library's module>` | `## configure` step 1; `## build` minimum-version slot | What is the build-time version of the sub-library the user picked? | A semver string matching `doca_caps --version`. Disagreement = partial install (route to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug) layer 2) |
| `pkg-config --cflags --libs <the chosen sub-library's module>` | `## build` | What include + link flags does the DPU-side linker need for the chosen sub-library? | Includes resolve under $(pkg-config --variable=includedir doca-common); libs include whatever `pkg-config --libs <module>` resolves on this install (the per-sub-library shared object plus its transitive closure; do not predict the `-l<name>` form by hand) |
| `doca_caps --list-devs` (DPU side) | `## configure` step 1; `## configure` step 3 | Which DOCA devices does the DPU see, and which advertise Device Emulation capability for the chosen sub-library? | One entry per `doca_dev` with the BlueField identity and the per-library capability flags. No entry advertising the chosen sub-library's emulation surface = firmware-side slot is disabled or the BlueField generation does not support this sub-library |
| `ls /opt/mellanox/doca/samples/doca_devemu/` | `## modify` slot 1 | Which Device Emulation sub-library sample subdirectories ship in this install, and which is the closest starting point? | A list of sub-library subdirectories (one per shipped sub-library), each containing sample directories named after the operation / shape they demonstrate |
| `lspci` (host side) | `## configure` step 7; `## run` step 2; `## debug` layer 5 | Has the emulated PCIe device appeared in the host's PCIe enumeration after the DPU-side context started? | A new entry that matches the device class the user emulated (a virtio NIC entry for virtio-net, a virtio filesystem entry for virtio-fs, a raw PCIe entry for PCI Generic). Empty = firmware-side enable did not take effect or the BlueField needs a reset |
| `dmesg \| tail -n 80` (host side, sudo) | `## configure` step 7; `## run` step 3; `## debug` layer 5 | What did the host kernel see at PCIe enumeration / driver bind / first operation time? | A clean bind line for the matching kernel driver; no virtio feature-negotiation errors; no DMA / BAR mapping failures. Failure messages here narrow the diagnosis to a host-kernel or feature-negotiation cause, not a DPU-side bug |
| `dmesg \| tail -n 40` (DPU side, sudo) | `## debug` layer 7 | What did the DPU kernel / driver log around the last Device Emulation call? | Empty or recent benign messages. Repeated mlx5 / device-emulation-driver errors → driver-layer bug; route to [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug) |
| `DOCA_LOG_LEVEL=trace ./<dpu-binary>` (DPU side) | `## run` step 3 | What did the structured DOCA logger emit for the first failing DPU-side Device Emulation call? | A trace-level line on every per-sub-library lifecycle transition and every doorbell registration. Silence after `doca_ctx_start()` = either DPU-side PE not progressed OR the doorbell was wired against the wrong sub-library context |

For commands shared across libraries (`pkg-config --modversion`,
`doca_caps`, `cat /opt/mellanox/doca/applications/VERSION`,
`DOCA_LOG_LEVEL`) the cross-library overlay is in
[`doca-debug TASKS.md ## Command appendix`](../../doca-debug/TASKS.md#command-appendix);
this table adds the Device Emulation-specific rows on top.
