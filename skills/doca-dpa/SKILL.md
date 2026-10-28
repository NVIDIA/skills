---
license: Apache-2.0
name: doca-dpa
description: >
  Use this skill when the user is doing hands-on DOCA DPA host-side work
  on a BlueField — creating the `doca_dpa` Core context, loading a
  DPACC-compiled DPA app image (`doca_dpa_app`), creating DPA threads,
  launching kernels via `doca_dpa_kernel_launch_update_*`, draining
  `doca_dpa_completion`, running `doca_dpa_cap_*` discovery, choosing
  between the DPA comm component (inter-DPA messaging) and the DPA verbs
  component (in-kernel RDMA), or debugging `DOCA_ERROR_*` from
  `doca_dpa_*`. Trigger even without "DOCA DPA" or "Data-Path
  Accelerator": "run compute on the DPA from my host", "DPA kernel
  hangs, no completion", "DOCA_ERROR_DRIVER on launch", "DOCA/DPACC
  version skew", or "does this BlueField expose a DPA". Route elsewhere
  for DPA-side kernel programming itself, DPACC compiler internals,
  host↔DPU messaging (doca-comch), host-side RDMA (doca-rdma), and
  GPU-initiated networking (doca-gpunetio).
metadata:
  author: NVIDIA <doca-feedback@nvidia.com>
  kind: library
  doca_version: "3.5"
  tags:
    - doca
    - dpa
    - accelerator
compatibility: >
  Requires DOCA SDK installed at /opt/mellanox/doca on Linux (Ubuntu
  22.04/24.04 or RHEL/SLES) with a BlueField DPU whose generation
  exposes the DPA processor to the host, plus the matching DPACC
  compiler at a version listed compatible by the DOCA Compatibility
  Policy. Reads `pkg-config --modversion doca-dpa` and the installed
  `dpacc` version, and inspects
  /opt/mellanox/doca/{lib,include,samples/doca_dpa,applications}. The
  validation steps require a live DOCA install.
---

# DOCA DPA

## Examples

- **"Run compute on the DPA from my host."** — this is in scope.
  Stay on `doca_dpa` + a DPACC-built image; do not rewrite it as
  host-CPU or GPUNetIO.
- **"DPA kernel hangs, no completion."** — drain
  `doca_dpa_completion` and check DOCA/DPACC version skew before
  changing the kernel.

## Instructions

Follow Loading order below. Open the tasks companion for configure /
launch / debug. Open the capabilities companion for threads vs
kernels and the DPA cap query. Do not open every companion file.

**Where to start:** This skill assumes DOCA is already installed,
the user's BlueField has a DPA processor and the host can see it
through DOCA, and the user is doing **hands-on DPA work from the
host side** — i.e. using `doca-dpa` to load a DPA application
image, launch DPA kernels, and exchange data with the DPA
processor. Open the tasks companion if the user wants to *do*
something (configure / build / modify / run / test / debug); open
the capabilities companion when the question is *what
can the host-side DPA API express* on this version + this
BlueField generation. If the user has not installed DOCA yet,
route to `doca-setup/SKILL.md` first; if the
user is asking how to *write* the DPA-side kernel itself (the
code that runs on the DPA processor, compiled by `dpacc`), that
is a different scope — route via
`doca-public-knowledge-map/SKILL.md`
to the public DOCA DPA / DPACC / DPA-Comms / DPA-Verbs guides
(this skill does not redefine those DPA-side surfaces).

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
`doca-public-knowledge-map/SKILL.md`).

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

> **"My DPA kernel hangs and I never get a completion" is a
> host-side question first.** Launch calls are asynchronous, so a
> successful return means *submitted*, not *finished*. Before
> reading a line of kernel code, establish that the host is
> actually in a position to observe completion: a
> `doca_dpa_completion` is attached, and the host is driving
> `doca_pe_progress` in parallel with the outstanding launches. A
> host that submits and then blocks without progressing the PE sees
> no completions and concludes *"the DPA is hung"* incorrectly —
> the kernel may have finished already. Only once that is excluded
> is a non-terminating kernel body the suspect. Both steps and the
> ordering are in ## run steps 2-3.
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
  live in ## comms and
  ## comms under this same skill.
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
  ## verbs and
  ## verbs under this same skill.

Do **not** load this skill for general DOCA orientation,
install of DOCA or the DPACC compiler, the DPA-side
  programming model itself (how to write a DPA kernel; the
  DPA device-side comm and verbs components
  (`libdoca_dpa_dev_comm.a` / `libdoca_dpa_dev_verbs.a`) that
  run *inside* the DPA kernel), or non-DPA library questions.
For those, route through
`doca-public-knowledge-map/SKILL.md`
to the matching upstream guide.

## What this skill provides

This is a **thin loader**. The body keeps only the orientation
needed to pick the right next file. The substantive
DPA-specific material lives in two companion files:

- the capabilities companion — what the host-side DPA API can express on
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
- the tasks companion — step-by-step workflows for the six in-scope DPA
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
`doca-setup/SKILL.md`.

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
  `doca-programming-guide/SKILL.md`,
  layered with the DPA-specific overrides in
  ## modify.
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
  `doca-public-knowledge-map/SKILL.md`.
  This skill names them and routes; it does not redefine them.

## Loading order

1. Read this skill first to confirm the user's question is
   in scope (host-side DPA work, not DPA-side kernel-writing).
2. **For the DPA capability matrix, the `doca_dpa` per-instance
   context, the loaded `doca_dpa_app` image, the
   `doca_dpa_thread` execution context, the kernel-launch +
   completion model, the dual capability query, the
   env-precondition policy, the error taxonomy, the
   observability surface, and the safety policy, see
   the capabilities companion.**
3. **For step-by-step workflows — configure, build, modify,
   run, test, debug — see the tasks companion.**

Both companion files cross-link to each other,
`doca-version/SKILL.md` for the canonical
DOCA version-handling rules (with the DPA overlay that DOCA
must match the DPACC compiler), and
`doca-public-knowledge-map/SKILL.md`
whenever the right answer is "look it up in the public DOCA
DPA, DPACC, DPA-Comms, or DPA-Verbs guide, or in the on-disk
install layout" rather than "DPA host-side-specific guidance".

## Related skills

- `doca-public-knowledge-map/SKILL.md` —
  the routing table for every public DOCA documentation
  source and the on-disk layout of an installed DOCA package.
  The DPA public guide is at
  <https://docs.nvidia.com/doca/sdk/DOCA-DPA/index.html>; the
  DPACC compiler guide, the DPA-Comms guide (DPA-side
  communications), the DPA-Verbs guide (DPA-side verbs), and
  the DPA Tools umbrella (developer / admin CLIs for DPA)
  live in the same routing table and are *companion surfaces*
  to this skill rather than redefined by it.
- `doca-setup/SKILL.md` — env preparation,
  install verification, DPACC compiler install / verification,
  and the *I have no install yet* path with the public NGC
  DOCA container. This skill assumes its preconditions are
  satisfied AND that DPACC is installed at a version that
  matches DOCA.
- `doca-version/SKILL.md` — canonical
  DOCA version-handling rules. This skill's `## Version
  compatibility` cross-links the four-way match rule and adds
  the DPA-specific *DOCA-and-DPACC must match* overlay per the
  DOCA Compatibility Policy.
- `doca-structured-tools-contract/SKILL.md` —
  the bundle's structured-tools precedence rule (detect /
  prefer / fall back / report). The Command appendix in
  the tasks companion honors this contract.
- `doca-programming-guide/SKILL.md` — shared build, first-app,
  Core-lifecycle and cross-library `DOCA_ERROR_*` patterns; this
  skill layers DPA specifics on top.
- `doca-debug/SKILL.md` — the cross-cutting debug ladder; this
  skill's `## debug` adds the DPA overlay (DOCA + DPACC skew, no
  DPA on this BlueField, kernel hangs with no host-side completion,
  launch-argument / signature mismatch).

### DPA device-side components

DOCA DPA's DPA device-side components — the comm archive
`libdoca_dpa_dev_comm.a` (communication primitives the DPA
kernel itself calls, header `doca_dpa_dev_comch_msgq.h`) and
the verbs archive `libdoca_dpa_dev_verbs.a` (ibverbs-like RDMA
verbs the DPA kernel itself calls, header
`doca_dpa_dev_verbs.h`) — are **DPA-side archives shipped
within `doca-dpa`**, not separate pkg-config modules, and each
has its own public guide. This skill covers them in its
`## comms` / `## verbs` sections; for per-symbol detail, route via
`doca-public-knowledge-map/SKILL.md`
to the public *DOCA DPA Comms* and *DOCA DPA Verbs* guides
and to the shipped `/opt/mellanox/doca/samples/doca_dpa/`
samples (which include both host-side and DPA-side
translation units). Conflating them with `doca-dpa` is the
single most common DPA first-app design error.
