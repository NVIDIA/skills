---
license: Apache-2.0
name: doca-dpa
description: >
  Use this skill when the user is doing hands-on DOCA DPA host-side
  work on a BlueField — creating the `doca_dpa` Core context, loading
  a DPACC-compiled DPA app image (`doca_dpa_app`), creating DPA
  threads, launching kernels via `doca_dpa_kernel_launch_update_*`,
  draining `doca_dpa_completion`, running `doca_dpa_cap_*` discovery,
  choosing between the DPA comm component (inter-DPA messaging) and the
  DPA verbs component (in-kernel RDMA), or debugging `DOCA_ERROR_*`
  from `doca_dpa_*`. Trigger even without "DOCA DPA" or "Data-Path
  Accelerator": "run compute on the DPA from my host", "DPA kernel
  hangs, no completion", "DOCA_ERROR_DRIVER on launch", "DOCA/DPACC
  version skew", or "does this BlueField expose a DPA". Route elsewhere
  for DPA-side kernel programming itself, DPACC compiler internals,
  host↔DPU messaging (doca-comch), host-side RDMA (doca-rdma), and
  GPU-initiated networking (doca-gpunetio).
metadata:
  kind: library
compatibility: >
  Requires DOCA SDK installed at /opt/mellanox/doca on Linux
  (Ubuntu 22.04/24.04 or RHEL/SLES) with a BlueField DPU whose
  generation exposes the DPA processor to the host, plus the
  matching DPACC compiler at a version listed compatible by
  the DOCA Compatibility Policy. Reads `pkg-config --modversion
  doca-dpa` and the installed `dpacc` version, and inspects
  /opt/mellanox/doca/{lib,include,samples/doca_dpa,applications}.
---

# DOCA DPA

**Where to start:** This skill assumes DOCA is already installed,
the user's BlueField has a DPA processor and the host can see it
through DOCA, and the user is doing **hands-on DPA work from the
host side** — i.e. using `doca-dpa` to load a DPA application
image, launch DPA kernels, and exchange data with the DPA
processor. Open [`TASKS.md`](TASKS.md) if the user wants to *do*
something (configure / build / modify / run / test / debug); open
[`CAPABILITIES.md`](CAPABILITIES.md) when the question is *what
can the host-side DPA API express* on this version + this
BlueField generation. If the user has not installed DOCA yet,
route to [`doca-setup`](../../doca-setup/SKILL.md) first; if the
user is asking how to *write* the DPA-side kernel itself (the
code that runs on the DPA processor, compiled by `dpacc`), that
is a different scope — route via
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
to the public DOCA DPA / DPACC / DPA-Comms / DPA-Verbs guides
(this skill does not redefine those DPA-side surfaces).

## Example questions this skill answers well

The CLASSES of DPA questions this skill is built to answer, each
with one worked example. The agent should treat the *class* as
the load-bearing piece — the worked example is a single
instance.

- **"How do I run a piece of compute on the DPA processor from
  my host program?"** — worked example: *"load a DPA kernel that
  counts events in a loop and reports the count back to the
  host"*. Answered by the two-side-program model + the
  host-side launch workflow in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the bring-up steps in
  [`TASKS.md ## configure`](TASKS.md#configure).
- **"Does this BlueField even have a DPA, and which DPA features
  does my DOCA install expose?"** — worked example: *"my host
  has a BlueField; can I use the DPA on it for a programmable
  control workload?"*. Answered by the dual-axis capability rule
  (BlueField-generation axis via `doca_dpa_cap_*` against the
  active `doca_devinfo` plus the DOCA-install axis via
  `pkg-config --modversion doca-dpa`) in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the device-enumeration step in
  [`TASKS.md ## configure`](TASKS.md#configure).
- **"Why does my host-side DPA setup fail with
  `DOCA_ERROR_NOT_SUPPORTED` even though DOCA Core looks
  healthy?"** — worked example: *"the BlueField generation in
  this host predates the DPA feature my code uses"*. Answered by
  the env-precondition matrix in
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
  + the env checklist in
  [`TASKS.md ## configure`](TASKS.md#configure) step 1.
- **"How do I get arguments and results between my host program
  and my DPA kernel?"** — worked example: *"pass a buffer pointer
  and a length into the DPA kernel as launch arguments; read a
  completion back when the kernel finishes"*. Answered by the
  launch-argument + completion overlay in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the launch + drain steps in
  [`TASKS.md ## run`](TASKS.md#run).
- **"Is the DPA host-side API I'm reading about on my installed
  DOCA?"** — worked example: *"is the host-side launch helper I
  see in the docs available against the DOCA + DPACC versions on
  this host?"*. Answered by the version-compatibility overlay in
  [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility)
  which cross-links the canonical detection chain in
  [`doca-version`](../../doca-version/SKILL.md) and adds the
  DPA-specific *DOCA must match DPACC* overlay.
- **"What does this `DOCA_ERROR_*` from a `doca_dpa_*` call mean
  and which layer caused it?"** — worked example: *"`DOCA_ERROR_DRIVER`
  on a host-side launch call — is it DOCA, the DPACC-produced
  image, or the DPA processor itself?"*. Answered by the DPA
  overlay on the cross-library taxonomy in
  [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
  + the layered ladder in
  [`TASKS.md ## debug`](TASKS.md#debug) that escalates to
  [`doca-debug`](../../doca-debug/SKILL.md).
- **"How does my DPA kernel send a small message to another
  DPA thread on the same DPA processor?"** — worked example:
  *"two DPA threads in the same loaded `doca_dpa_app`; thread A
  sends a counter value to thread B over a DPA-side comms
  endpoint and thread B signals the host through
  `doca_dpa_completion`"*. Answered by the DPA-Comms routing
  rule, primitive families, host-side capability-budget rule,
  and DPA-Comms error overlay in
  [`CAPABILITIES.md ## comms`](CAPABILITIES.md#comms) plus the
  configure / build / modify / run / test / debug overlay in
  [`TASKS.md ## comms`](TASKS.md#comms). Disambiguates the DPA
  device-side comm component from host-side `doca-comch` and
  host-side `doca-rdma`.
- **"How do I do RDMA directly from inside my DPA kernel to a
  remote peer, without round-tripping to the host?"** — worked
  example: *"my DPA kernel needs to fetch the next 4 KiB input
  buffer from a remote node via RDMA read before continuing
  compute; the host round-trip is the measured bottleneck"*.
  Answered by the 4-way RDMA matrix, the host-configures-QP /
  DPA-uses-QP coupling rule, the host-side cap-query rule for
  the specific verb, and the DPA-Verbs error overlay in
  [`CAPABILITIES.md ## verbs`](CAPABILITIES.md#verbs) plus the
  workflow overlay in [`TASKS.md ## verbs`](TASKS.md#verbs).
  Includes the climb-back rule for when the latency-tuning
  premise stops holding.

## Audience

This skill serves **external developers building applications
that consume the DOCA DPA library from the host side** — i.e.,
users whose code calls `doca_dpa_*` from host C / C++ to stand
up the per-DPA-instance context, load a DPA application image
that `dpacc` produced from their DPA-side source, create one or
more DPA threads, launch DPA kernels with arguments, and drain
completions. It is *not* for NVIDIA developers contributing to
DOCA DPA itself, nor is it the place to learn how to *write*
the DPA-side kernel code (that path goes through the public
DOCA DPA, DPACC, DPA-Comms, and DPA-Verbs guides via
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)).

**Language scope.** DOCA DPA ships as a host-side C library
with `pkg-config` module name `doca-dpa`. The host-side API is
C; the DPA-side kernel is a separate translation unit written
in the language the DPACC compiler accepts and compiled by
`dpacc` into a binary that the host packages into the
executable as the DPA application image. The shipped samples
under `/opt/mellanox/doca/samples/doca_dpa/` are written in C
plus DPA-side source (NVIDIA's choice). Other-language
consumers are limited in practice — the DPA-side kernel has no
FFI escape hatch because it must be a translation unit `dpacc`
accepts — but a Rust / Go / Python host-side wrapper that drives
`doca_dpa_*` setup and launches a DPA kernel image built
separately is still useful, and the skill keeps the lifecycle,
capability-discovery, env-precondition, and error-taxonomy
guidance language-neutral.

## When to load this skill

Load this skill when the user is doing hands-on DOCA DPA work
**from the host side**, in any host language plus a DPA-side
translation unit built by `dpacc`. Concretely:

- Initializing a `doca_dpa` against a `doca_dev` that maps to a
  BlueField with a DPA processor visible to the host.
- Loading a DPA application image (`doca_dpa_app`) that
  `dpacc` produced from the user's DPA-side source, into the
  `doca_dpa` context.
- Creating one or more DPA execution contexts (`doca_dpa_thread`)
  so DPA kernels have somewhere to run on the DPA processor.
- Launching a DPA kernel function with arguments from the host
  via the `doca_dpa_kernel_launch_update_*` family, and
  reasoning about which argument shape is supported on this
  install.
- Attaching a `doca_dpa_completion` to observe when async DPA
  work finishes, and draining it from the host side.
- Checking which DPA features are supported on the active
  `doca_devinfo` via the `doca_dpa_cap_*` family — BlueField
  generations differ in DPA hardware support.
- Debugging a `DOCA_ERROR_*` returned from a `doca_dpa_*` call
  — in particular disambiguating *DPA not present on this
  BlueField* from *DPA feature too new for this hardware
  generation* from *DPACC-produced image mismatched against
  the host-side DOCA install* from *DPA driver layer
  reporting failure*.
- Designing host-side bindings in a non-C language that drive
  a DPA application image they built separately with `dpacc` —
  the env-precondition and capability-discovery rules in this
  skill still apply.
- Writing DPA-side kernel code that calls the DPA device-side
  comm component **`libdoca_dpa_dev_comm.a`** (header
  `doca_dpa_dev_comch_msgq.h`) — for inter-DPA-thread
  messaging or coordination signals between DPA threads on the
  same `doca_dpa_app`. The DPA-Comms routing rule, primitive
  families, capability rule (there is no per-primitive host
  cap-query family — host-side DPA discovery is only
  `doca_dpa_cap_is_supported` /
  `doca_dpa_cap_get_max_kernel_time_alive_supported`), error
  overlay (`_AGAIN` → kernel must yield; `_BAD_STATE`
  disambiguation from the parent's host-side `_BAD_STATE`), and
  the configure / build / modify / run / test / debug overlay
  live in [`CAPABILITIES.md ## comms`](CAPABILITIES.md#comms) and
  [`TASKS.md ## comms`](TASKS.md#comms) under this same skill.
- Writing DPA-side kernel code that calls the DPA device-side
  verbs component **`libdoca_dpa_dev_verbs.a`** (header
  `doca_dpa_dev_verbs.h`) — for RDMA from inside
  the DPA kernel to a remote peer when the host round-trip is
  the measured latency bottleneck. The 4-way RDMA matrix, the
  host-configures-QP / DPA-uses-QP coupling rule, the
  capability rule (no per-verb host cap-query family exists;
  verb availability follows the BlueField generation + matched
  DOCA/DPACC install, read from `doca_dpa_dev_verbs.h` and the
  shipped sample), the IO_FAILED → CQE-inspection
  overlay, and the climb-back rule live in
  [`CAPABILITIES.md ## verbs`](CAPABILITIES.md#verbs) and
  [`TASKS.md ## verbs`](TASKS.md#verbs) under this same skill.

Do **not** load this skill for general DOCA orientation,
install of DOCA or the DPACC compiler, the DPA-side
  programming model itself (how to write a DPA kernel; the
  DPA device-side comm and verbs components
  (`libdoca_dpa_dev_comm.a` / `libdoca_dpa_dev_verbs.a`) that
  run *inside* the DPA kernel), or non-DPA library questions.
For those, route through
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
to the matching upstream guide.

## What this skill provides

This is a **thin loader**. The body keeps only the orientation
needed to pick the right next file. The substantive
DPA-specific material lives in two companion files:

- `CAPABILITIES.md` — what the host-side DPA API can express on
  this version + this BlueField generation: the per-DPA-instance
  `doca_dpa` context, the loaded `doca_dpa_app` image produced
  by `dpacc`, the `doca_dpa_thread` execution context, the
  host-initiated kernel launch surface
  (`doca_dpa_kernel_launch_update_*`), the `doca_dpa_completion`
  mechanism, the capability-query surface (`doca_dpa_cap_*`),
  the DPA error taxonomy mapped onto the cross-library
  `DOCA_ERROR_*` set, the observability surface (host-side
  completions plus the public DPA developer tools surface
  reachable via `doca-public-knowledge-map`), and the safety
  policy that gates env preconditions (DPA-capable BlueField,
  matched DOCA + DPACC versions, DPA-side image and host-side
  expected entry points agree).
- `TASKS.md` — step-by-step workflows for the six in-scope DPA
  verbs: `configure`, `build`, `modify`, `run`, `test`,
  `debug`. Plus a `Deferred task verbs` block that points
  out-of-scope questions at the right next skill.

The skill assumes a host where DOCA is already installed at
the standard location, a BlueField with a DPA processor is
physically present and visible to the host, the DPACC compiler
is installed at a version matched to the DOCA install per the
DOCA Compatibility Policy, and the user already knows how (at
least at a sketch level) to write the DPA-side kernel that
`dpacc` will compile. It does not cover installing DOCA or the
DPACC compiler — that path goes through
[`doca-setup`](../../doca-setup/SKILL.md).

## What this skill deliberately does not ship

This skill is **agent guidance**, not a samples or templates
bundle. To keep the boundary clean, it deliberately does not
contain — and pull requests should not add:

- **Pre-written DOCA DPA application source code or DPA-side
  kernel source, in any language.** The verified DPA source is
  the shipped C + DPA-side samples at
  `/opt/mellanox/doca/samples/doca_dpa/`. The agent's job is to
  route the user to those files and prescribe a minimum-diff
  modification on them via the universal modify-a-sample
  workflow in
  [`doca-programming-guide`](../../doca-programming-guide/SKILL.md),
  layered with the DPA-specific overrides in
  [`TASKS.md ## modify`](TASKS.md#modify).
- **Standalone build manifests** (`meson.build`,
  `CMakeLists.txt`, …) parked inside the skill. The agent
  constructs the build manifest *in the user's project
  directory* against the user's installed DOCA + DPACC
  compiler, where `pkg-config --modversion doca-dpa` and the
  installed `dpacc` are the two sources of truth.
- **A `samples/`, `bindings/`, or `reference/` subtree** of any
  kind. A mock or incomplete artifact in this skill's tree,
  even one labeled "reference", is misleading: users will read
  it as buildable.
- **DPA device-side content for the comm / verbs components
  (`libdoca_dpa_dev_comm.a` / `libdoca_dpa_dev_verbs.a`).**
  These are *DPA-side* archives shipped as part of `doca-dpa`
  (NOT separate pkg-config modules): their symbols are called
  from inside the DPA kernel and linked into the DPA image by
  `dpacc`, not from the host. Their public guides are reachable
  via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).
  This skill names them and routes; it does not redefine them.

## Loading order

1. Read this `SKILL.md` first to confirm the user's question is
   in scope (host-side DPA work, not DPA-side kernel-writing).
2. **For the DPA capability matrix, the `doca_dpa` per-instance
   context, the loaded `doca_dpa_app` image, the
   `doca_dpa_thread` execution context, the kernel-launch +
   completion model, the dual capability query, the
   env-precondition policy, the error taxonomy, the
   observability surface, and the safety policy, see
   [CAPABILITIES.md](CAPABILITIES.md).**
3. **For step-by-step workflows — configure, build, modify,
   run, test, debug — see [TASKS.md](TASKS.md).**

Both companion files cross-link to each other,
[`doca-version`](../../doca-version/SKILL.md) for the canonical
DOCA version-handling rules (with the DPA overlay that DOCA
must match the DPACC compiler), and
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
whenever the right answer is "look it up in the public DOCA
DPA, DPACC, DPA-Comms, or DPA-Verbs guide, or in the on-disk
install layout" rather than "DPA host-side-specific guidance".

## Related skills

- [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) —
  the routing table for every public DOCA documentation
  source and the on-disk layout of an installed DOCA package.
  The DPA public guide is at
  <https://docs.nvidia.com/doca/sdk/DOCA-DPA/index.html>; the
  DPACC compiler guide, the DPA-Comms guide (DPA-side
  communications), the DPA-Verbs guide (DPA-side verbs), and
  the DPA Tools umbrella (developer / admin CLIs for DPA)
  live in the same routing table and are *companion surfaces*
  to this skill rather than redefined by it.
- [`doca-setup`](../../doca-setup/SKILL.md) — env preparation,
  install verification, DPACC compiler install / verification,
  and the *I have no install yet* path with the public NGC
  DOCA container. This skill assumes its preconditions are
  satisfied AND that DPACC is installed at a version that
  matches DOCA.
- [`doca-version`](../../doca-version/SKILL.md) — canonical
  DOCA version-handling rules. This skill's `## Version
  compatibility` cross-links the four-way match rule and adds
  the DPA-specific *DOCA-and-DPACC must match* overlay per the
  DOCA Compatibility Policy.
- [`doca-structured-tools-contract`](../../doca-structured-tools-contract/SKILL.md) —
  the bundle's structured-tools precedence rule (detect /
  prefer / fall back / report). The Command appendix in
  [TASKS.md](TASKS.md) honors this contract.
- [`doca-programming-guide`](../../doca-programming-guide/SKILL.md) —
  general DOCA programming patterns shared by every library:
  the canonical `pkg-config` + meson build pattern, the
  universal modify-a-shipped-sample first-app workflow, the
  universal Core-context lifecycle, the cross-library
  `DOCA_ERROR_*` taxonomy, and the program-side debug order.
  This skill layers DPA specifics on top.
- [`doca-debug`](../../doca-debug/SKILL.md) — the cross-cutting
  debug ladder (install / version / build / link / runtime /
  program / driver). DPA-specific debug (DPACC + DOCA version
  skew, DPA not present on this BlueField generation, DPA
  kernel hangs that show no host-side completion,
  launch-argument shape mismatches between the host launch
  call and the DPA-side function signature) overlays on top
  of that ladder.

DOCA DPA's DPA device-side components — the comm archive
`libdoca_dpa_dev_comm.a` (communication primitives the DPA
kernel itself calls, header `doca_dpa_dev_comch_msgq.h`) and
the verbs archive `libdoca_dpa_dev_verbs.a` (ibverbs-like RDMA
verbs the DPA kernel itself calls, header
`doca_dpa_dev_verbs.h`) — are **DPA-side archives shipped
within `doca-dpa`**, not separate pkg-config modules, and each
has its own public guide. No library skill ships for them in
this bundle yet; for any DPA-side question, route via
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
to the public *DOCA DPA Comms* and *DOCA DPA Verbs* guides
and to the shipped `/opt/mellanox/doca/samples/doca_dpa/`
samples (which include both host-side and DPA-side
translation units). Conflating them with `doca-dpa` is the
single most common DPA first-app design error.
