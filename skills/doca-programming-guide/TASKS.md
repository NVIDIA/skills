# DOCA programming guide workflows

**Where to start:** Read [`## configure`](#configure) → [`## modify`](#modify)
for the first-app path, or [`## debug`](#debug) for an in-flight
program-class problem. The two cross-cutting reads every prescriptive
section assumes are the lifecycle table in
[`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
and the modify-schema in [`## modify`](#modify) below.

Read this file when the loader sent you here from [SKILL.md](SKILL.md). For the underlying surface — the shape of DOCA, the universal lifecycle, the cross-library version-compat rule, the `DOCA_ERROR_*` taxonomy, the program-side observability surface, and the program-side safety policy — see [CAPABILITIES.md](CAPABILITIES.md). For env-class workflows (install verification, `pkg-config` wiring, hugepages, the *I have no install yet* procedure with the NGC container fallback), see [`doca-setup`](../doca-setup/SKILL.md). For routing, docs URLs, the on-disk install layout, and how to check the installed DOCA version, see [`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md).

Each verb describes the **shape of the workflow**, not a copy-paste recipe. Library-specific overrides (which sample to start from, which fields to swap, which capabilities to check, which errors to expect) live in the matching library skill — never here.

## configure

Goal: get the program itself into the right starting state — the right version, the right build flavor, the right capability picture — *after* the env-class precondition (a clean `pkg-config doca-<library>` on the user's host) is already satisfied.

1. **Confirm the env precondition.** This skill assumes [`doca-setup ## configure`](../doca-setup/TASKS.md#configure) has already run and `pkg-config --modversion doca-<library>` returns a version. If it doesn't, **stop** and route the user to [`doca-setup ## no-install`](../doca-setup/TASKS.md#no-install) (the NGC container fallback is the universal first option there) or [`doca-setup ## debug`](../doca-setup/TASKS.md#debug) (if the env is broken in a recoverable way). Do not work around an env failure with a code change.

2. **Quote the installed DOCA version.** Use the procedure in [`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md). API names, sample filenames, and capability-matrix answers all depend on this — the version the user has on disk is the only honest source.

3. **Pick the build flavor.** Trace flavor (`doca-<library>-trace`) for first-app development; release flavor (`doca-<library>`) for performance work. Rationale and selection criteria are in [CAPABILITIES.md ## Capabilities and modes](CAPABILITIES.md#capabilities-and-modes); env-side mechanics (where the trace `*.so` lives, `LD_LIBRARY_PATH`) are in [`doca-setup ## configure`](../doca-setup/TASKS.md#configure).

4. **Discover the library's capabilities on this host.** Run `doca_caps`, plus the library's own capability-query API (Flow has one, RDMA has one, Comch has one) — the library skill names the call. Record the active mode, the supported features, and the budgets *before* designing the program; designing for features-not-on-this-hardware is the most common cause of *"the spec validates but won't program"*.

5. **Restate the user's intent in library-neutral terms.** Confirm *what runs where* (host or DPU), *which devices / queues / pipes / channels are involved*, *what the success criterion is*. If any of these is unclear, stop and ask. The library skill takes over from here for the library-specific configuration.

## build

Goal: produce a buildable artifact from DOCA application source — yours or a copy of a shipped sample — using the canonical `pkg-config` + meson pattern that any DOCA program follows.

This verb has two tracks because DOCA is a C ABI consumed from many languages. Pick the right one before issuing any command.

### Track 1 — C / C++ consumers (canonical)

The shipped DOCA samples and reference applications are C, built with meson. Your custom application uses the same shape.

1. **Stage the build out-of-tree.** Never build into `/opt/mellanox/doca/`. Standard pattern:

   ```bash
   meson /tmp/build-<project>
   ninja -C /tmp/build-<project>
   ```

2. **Declare the DOCA dependency in `meson.build` via `pkg-config`.** Use the library's own `pkg-config` module name (`doca-flow`, `doca-rdma`, `doca-comch`, …) — not a hand-typed `-l...` flag list. The `pkg-config` module is the source of truth for include paths and link flags; hard-coded paths drift across releases.

3. **Honor the build flavor decision from [`## configure`](#configure) step 3.** Trace flavor links against the `doca-<library>-trace` module; release flavor links against `doca-<library>`.

4. **Map any build-time error to the env taxonomy first.** If meson reports a missing dependency or ninja reports an undefined symbol, the symptom is almost always env-class (wrong install profile, wrong `PKG_CONFIG_PATH`, wrong build flavor) — route to [`doca-setup ## debug`](../doca-setup/TASKS.md#debug) before touching the source.

### Track 2 — Other languages (Rust, Go, Python, …)

DOCA does not ship official bindings in non-C languages inside this repository. The consumption path is FFI / language-specific bindings against the same `*.so` libraries the C samples link against.

1. **Confirm the install host gives you the C ABI surface.** `pkg-config --cflags --libs doca-<library>` returns the include path and link flags; the headers under $(pkg-config --variable=includedir doca-common) are the authoritative symbol declarations; the `*.so` files under `/opt/mellanox/doca/lib/<arch>-linux-gnu/` are what your binding loads at runtime. Verify all three are present before any binding-side work. If any of those checks fails, route to [`doca-setup ## debug`](../doca-setup/TASKS.md#debug).

2. **Pick the binding strategy honestly.** If a community or user-built binding for the user's language exists, point at its repository (the agent **must** verify it exists by fetching its repo or package registry — never invent a binding name). Otherwise the user is doing direct FFI: `bindgen` for Rust, `cgo` for Go, `cffi` / `ctypes` for Python, equivalents for other languages.

3. **Generate or write the binding in the user's own toolchain.** This skill does not author wrappers. The agent describes the C-side surface (header path, `*.so` filename, lifecycle order, error pattern) and lets the user's binding tooling do its job. The library skill (e.g. [`doca-flow`](../libs/doca-flow/SKILL.md)) supplies the API-surface guidance the wrapper has to honor.

4. **Read the C samples even if you're not writing C.** The order of API calls in `/opt/mellanox/doca/samples/doca_<library>/<sample>/` is the same regardless of the calling language. The wrapper translates the shape; it does not invent a different shape.

## modify

Goal: take a working shipped sample and **derive a custom first application** for the user, by editing a verbatim copy of the sample on the user's own DOCA-installed Linux host (bare-metal, VM, *or NGC container*). The substance of the modified file is NVIDIA's BSD-3 sample (verified, compiled, shipped) with a small, named set of user-domain values swapped. The agent does **not** author DOCA library source code from scratch.

**Language scope of this verb.** The shipped DOCA samples are written in C; this is the only application source code NVIDIA ships in this repository. The modify-a-sample workflow below is therefore the C / C++ first-app track. For consumers writing their first DOCA application in another language (Rust, Go, Python, …), this verb is *not* the right path — those users should still build a shipped sample (Track 1 of [`## build`](#build)) to verify their install is healthy, and then use Track 2 of [`## build`](#build) plus the matching library skill's bindings / FFI guidance for the wrapper-side work the user does in their own toolchain. The agent must not pretend the modify-a-sample workflow produces a Rust crate, a Go module, or a Python package; the workflow's output is a modified copy of NVIDIA's C sample.

The generic pattern below is library-agnostic *across DOCA libraries* (Flow, RDMA, Comch, …) but C-specific *within a library*. Library-specific values (which sample, which fields the user must change, which actions to keep) live in the matching library skill — for Flow, see [`doca-flow ## build`](../libs/doca-flow/TASKS.md#build).

### Precondition

This verb requires *all* of:

| Precondition | Check |
| --- | --- |
| Linux environment the agent can reach (the user's machine, an SSH session, a Cursor remote, *or a running NGC DOCA container* — any environment where the agent can `ls` the install tree) | shell available |
| DOCA installed | `ls /opt/mellanox/doca` returns a populated tree |
| The source sample is present | `ls /opt/mellanox/doca/samples/<library>/<sample_name>/` lists `meson.build` and the source files |
| `pkg-config` knows DOCA | `pkg-config --modversion doca-<library>` returns a version string |

If any precondition fails, **do not proceed and do not invent a substitute**. Route to [`doca-setup ## no-install`](../doca-setup/TASKS.md#no-install); that section is what the agent does instead, and its Path 0 (NGC DOCA container, `nvcr.io/nvidia/doca/doca`) is the universal way to satisfy these preconditions on macOS, Windows, or Linux without DOCA. Authoring application source code in *any* language (C, C++, Rust, Go, Python, …) from documentation prose to "fill the gap" is the failure mode this verb is here to prevent — it violates [`AGENTS.md`](../../AGENTS.md) ground rule #3 and would ship code that has never been compiled / linked / FFI-loaded against the live DOCA library.

### Modify-from-sample schema (the mental model)

Before any of the steps below, the agent fills in a **modify-schema**
for the user — a small table that names *what is being changed and why*.
The schema is the load-bearing artifact of this verb; the file edits are
the mechanical consequence of the schema. Without the schema, the agent
ends up swapping arbitrary fields and the result is a broken
hybrid of "what the user wanted" and "what the sample happened to be";
with the schema, every edit is traceable.

The schema has FIVE slots. Every modify-from-sample run fills all five
— in this order — before issuing any `sed`/edit command. Library skills
overlay library-specific values for each slot; this skill prescribes
the slots themselves.

| Slot | What goes here | Class of value | Where it comes from |
| --- | --- | --- | --- |
| 1. Source sample | Identifier of the shipped sample being copied from (path under `/opt/mellanox/doca/samples/<library>/`) | filesystem path | The library skill's `## build` overlay (it names the smallest viable sample for each shape). |
| 2. Sample contract (what it already does) | One-sentence summary of the sample's behavior verbatim from its source / README | textual | Reading the sample's `meson.build` and `*.c` files in the user's install. |
| 3. Deltas from user intent | The minimum set of user-domain values that differ between the sample and what the user asked for. Each row: field name, what the sample has, what the user wants, where the user-side value comes from. | enumerated | The library skill's modify overlay (it lists which fields are typically the deltas for each sample shape); the user supplies the new values. |
| 4. Boundaries (what stays verbatim) | The init / teardown / validate / error-handling calls and any DOCA-API call sequences kept untouched from the sample | enumerated | Always: every DOCA library call sequence. The library skill names which additional library-specific helpers are also "boundaries". |
| 5. Verification plan | The validate-before-commit calls + the smallest meaningful runtime probe that proves the modification is correct on this hardware | procedure | [`## test`](#test) below + library-specific overlay. |

The agent fills slots 1, 2, 4, 5 from the skill content (this skill +
the library skill); slot 3 comes from the conversation with the user.
If slot 3 is empty after the conversation, the user has not actually
asked for a custom app — they want to *run* the shipped sample, which
is [`## run`](#run) below, not [`## modify`](#modify).

The schema is also the artifact the agent writes into the README at
step 8 below (the "Modified fields:" line is literally slot 3
serialized). Library skills reference this schema by slot number when
their modify overlays prescribe which values are typically in slot 3
for a given sample shape.

### Steps (preconditions met)

1. **Identify the source sample.** Use the smallest shipped sample that already does something close to what the user asked for. Confirm it builds clean ([`## build`](#build) above) and runs clean ([`## run`](#run) below) *before* any modification. The library skill names the right starting sample for common shapes; do not pick from memory.

2. **Read the actual contents of the sample.** Before describing the file, list it (`ls`) and read the `meson.build` plus each `.c`/`.h`. The shape of the sample on the user's installed version is the truth, not what an older release looked like or what a docs page describes.

3. **Copy the sample out of `/opt/mellanox/doca/` into a writable location.**

   ```bash
   cp -r /opt/mellanox/doca/samples/<library>/<sample_name>/ ~/dev/my-first-<library>-app/
   cd ~/dev/my-first-<library>-app/
   ```

   Never edit the install tree itself ([`doca-setup CAPABILITIES.md ## Safety policy`](../doca-setup/CAPABILITIES.md#safety-policy) item 1). The copy is the user's source code from this point forward. *Inside an NGC container*, mount a host directory at `cp` time (e.g. `docker run -v $HOME/dev:/work …`) so the modified copy persists when the container exits.

4. **Identify the *minimum* set of values to change in the copy.** Library-specific list lives in the library skill. For *every* library the recipe is the same shape: keep all init / teardown boilerplate; keep the validation calls; keep the error handling; only swap the small set of user-domain values. Every byte not changed is a byte not debugged.

5. **Apply the swap as a minimum-diff edit on the copied file.** Where the user has given you a real value, substitute the literal. Where the user has not yet given you a value but the build would otherwise fail to compile, leave a `/* TODO: replace with your <thing>; see <how-to-find-it> */` comment around a syntactically-valid placeholder constant. The placeholder rule is **only for values that block compilation if absent** (constants, `#define`s the rest of the file references); it is not a license to leave function bodies or DOCA API call sequences unfilled. The init / teardown / validation / error-handling calls all stay verbatim from the upstream sample.

   Example placeholders (note: the *surrounding code* is the upstream sample's, untouched):

   ```c
   /* TODO: replace with your destination MAC. Use `ip link show <iface>` on
    * the originating host to obtain the real value. */
   uint8_t target_mac[6] = { 0x00, 0x11, 0x22, 0x33, 0x44, 0x55 };

   /* TODO: replace REPRESENTOR_PORT_ID with the port_id reported by the
    * sample's enumeration printout for your representor. */
   #define REPRESENTOR_PORT_ID 1
   ```

6. **Update the build manifest minimally.** The simplest correct change to the sample's `meson.build` is to rename the executable; do not refactor build options. If the original sample required `-D enable_<flag>=true`, keep that option in your build invocation. If the user's project must build *standalone* (outside the DOCA samples meson tree), the standalone `meson.build` depends on `doca-<library>` via `pkg-config` ([`## build`](#build) Track 1 step 2). The agent constructs that manifest *in the user's project directory*, not from a template pinned in this skill.

7. **Build and run staged.** Build with [`## build`](#build). Run with [`## run`](#run) against the smallest possible scope (one representor, one queue, one channel — whatever the library's unit of damage is per [CAPABILITIES.md ## Safety policy](CAPABILITIES.md#safety-policy) item 2). Read the output. Only after the staged run succeeds may the user widen the scope. *NGC-container caveat:* the build half of this step works inside the container; the actual hardware-runtime half does not — the container has no access to a real NIC / DPU. For runtime, the user has to graduate from Path 0 (container) to Path A or Path C in [`doca-setup ## no-install`](../doca-setup/TASKS.md#no-install).

8. **Document what was changed.** A two-line `README` next to the modified sample saying *"Derived from `<source-sample>` on `<date>` against DOCA `<version>`. Modified fields: `<list>`."* lets the user re-derive against the next DOCA release without having to re-read the agent's chat history.

## run

Goal: launch a built DOCA program (shipped sample or derived custom app) and read its output meaningfully.

1. **Pre-run checklist.** Re-verify the env preconditions via [`doca-setup ## test`](../doca-setup/TASKS.md#test) (hugepages mounted, devices visible, representors enumerated). A failed pre-run check is faster to diagnose than a runtime error. *Inside an NGC container*, several of these checks are vacuously satisfied (no real NIC) — that's expected; runs that need real hardware will fail at start, and the right move is to graduate the user to a hardware path per [`doca-setup ## no-install`](../doca-setup/TASKS.md#no-install).

2. **Use the program's own CLI flags.** Each shipped sample documents its `-h` flags. The agent must use those, not invent new ones — the most common runtime error is a flag rename across DOCA versions.

3. **Run with verbose logging the first time.** Add `--sdk-log-level 70` (DOCA `LOG_TRACE`) on the first run of any new code path. Re-running with reduced verbosity once the path is known to work is a deliberate later step; see [CAPABILITIES.md ## Observability](CAPABILITIES.md#observability).

4. **Where logs go.** DOCA libraries log to stderr by default. Some long-running services log via `journald` or a configured log directory; check `/var/log/doca*` or the service's own README before assuming output is missing.

5. **Map runtime failures to the right layer.** Startup failures of the form *"cannot find any working PCI driver"*, *"no free 2048 kB hugepages"*, *"representor X not found"* are **env-class** and route to [`doca-setup ## debug`](../doca-setup/TASKS.md#debug). Failures of the form `DOCA_ERROR_*` returned from a library API call are **program-class** and route to [`## debug`](#debug) below (with the library-specific overlay in the matching library skill).

## test

Goal: validate the program — and the system context around it — **before** committing to hardware / runtime side-effects.

**Program-class `## test` is an iterative loop, not a one-shot check.** The
agent runs the smallest meaningful validation, reads its output, picks the
next narrowest probe based on what's revealed, and only declares the program
"ready to commit" when every loop iteration is observed clean against the
*current* spec. A snapshot proves nothing about a spec that changes between
runs; the loop is what makes the validate-before-commit guarantee real.

The loop, library-agnostic:

```
   .--> 1. Spec validate (library's validate call against current spec)
   |
   |    2. Capability cross-check (does the hardware/version accept it?)
   |
   |    3. Known-good smoke (one shipped sample builds + runs clean)
   |
   |    4. Negative test (one deliberately wrong input rejected with the right DOCA_ERROR_*)
   |
   |    5. Read symptoms; classify:
   '----- spec changed since (1)?         -> back to (1) with the new spec
          env regression detected?        -> route to doca-setup ## test loop
          program-class fault remains?    -> route to ## debug below
          all four iterations clean?      -> declare program ready to commit
```

The four-step variant of the loop, in detail:

1. **Validate the spec / configuration before commit.** Every DOCA library that programs hardware (Flow in particular) exposes a *validate* call separate from the *commit / start / program* call. Use validate first; never enter a commit path with an un-validated spec. This is the cross-library validate-before-commit rule from [CAPABILITIES.md ## Safety policy](CAPABILITIES.md#safety-policy). Library skills extend it with library-specific *what to validate*. **Loop:** if validate rejects, fix the spec and re-run *this step*; do not move to step 2 with a failing validate.

2. **Capability cross-check.** Re-confirm that every feature your program intends to use is supported by the active mode and version on this host. Validation answers *"is the spec internally consistent"*; capability cross-check answers *"will this hardware / library actually accept it"*. **Loop:** if a capability is missing, do *not* code around it — back up to [`## configure`](#configure) step 4 (or to [`doca-setup`](../doca-setup/SKILL.md) if firmware / version is the gap) and re-enter the loop only after the capability is present or the program intent has changed to accommodate.

3. **Smoke-test by building and running one shipped sample first.** A *known-good* sample built and run cleanly is the cheapest end-to-end check that the install + your build env + your runtime preconditions are all healthy. Pick the smallest sample that exists for the library family the user cares about — the library skill names it. Inside an NGC container the build half is the meaningful check; the runtime half is reserved for a hardware path. **Loop:** if the shipped sample fails, the failure is almost certainly env-class — route to [`doca-setup ## test`](../doca-setup/TASKS.md#test); do not assume the library is broken until the canonical sample for that library runs cleanly.

4. **Negative test.** Construct one deliberately failing input and confirm the library rejects it with the expected `DOCA_ERROR_*`. This is the cheapest way to detect a stale or wrong-version library before going live. **Loop:** if the negative test is *accepted* (library returns success on a known-bad input), that is a library-version / install bug — re-run version detection ([CAPABILITIES.md ## Version compatibility](CAPABILITIES.md#version-compatibility)) before believing any successful positive test from this session.

The loop terminates when one of:

- All four iterations are clean in the same session against the same
  spec ⇒ declare the program ready to commit; record the
  validate/capability/sample/negative outputs as evidence.
- An env regression is detected at any iteration ⇒ hand off to
  [`doca-setup ## test`](../doca-setup/TASKS.md#test); restart this
  loop from the iteration that broke after the env is clean again.
- A program-class fault remains after the env is ruled clean ⇒ route
  to [`## debug`](#debug) with the captured iteration outputs.

## debug

Goal: when a DOCA program fails *after* the env is known healthy, walk the layered diagnosis tree top-down so the agent does not jump to library-internal code-fix recommendations against a symptom it can explain at a higher layer. This anchor is the **program-class half** of the layered debug ladder. The complementary **env-class half** lives in [`doca-setup ## debug`](../doca-setup/TASKS.md#debug). The **cross-cutting reference both halves escalate to** — the canonical layered ladder spanning install / version / build / link / runtime / program / driver, plus cross-cutting tooling (`gdb`, `valgrind`, `ldd`, `--sdk-log-level`, the `doca-<lib>-trace` build flavor, container introspection, core dumps) and the *Where to ask for help* escalation to the public Developer Forum — lives in [`doca-debug ## debug`](../doca-debug/TASKS.md#debug). Use this anchor for program-side specifics (lifecycle, `DOCA_ERROR_*`, `doca_error_get_descr()`); use `doca-debug ## debug` for the cross-cutting ladder shape and tooling.

Investigation order — **always**:

1. **Env layer (sanity check, then move on).** Re-run the env-class checks via [`doca-setup ## debug`](../doca-setup/TASKS.md#debug) layers 1–4 (install / version / build / runtime). Only continue here if those are clean; an env-class failure is not a programming bug.

2. **Version layer (program-side).** `pkg-config --modversion doca-<library>` against the runtime `doca_caps --version`. Mismatch ⇒ partial upgrade or stale build env; reinstall consistently before code changes ([CAPABILITIES.md ## Version compatibility](CAPABILITIES.md#version-compatibility)).

3. **Lifecycle order layer.** Did the program call the library's APIs in the universal order *cfg-create → init → start → use → stop → destroy*? Out-of-order calls produce `DOCA_ERROR_BAD_STATE`. The library skill names which specific calls map to which lifecycle phase.

4. **Capability layer.** Did the program ask for a feature the active mode does not support? Re-run capability discovery from [`## configure`](#configure) step 4 and cross-check; an unsupported feature returns `DOCA_ERROR_NOT_SUPPORTED`. Code-side workarounds for `NOT_SUPPORTED` are almost always wrong — change intent, switch hardware/mode, or escalate.

5. **Error-description layer.** For any returned `doca_error_t`, call `doca_error_get_descr()` and quote what it actually says — do not paraphrase from memory. The cross-library error meanings are in [CAPABILITIES.md ## Error taxonomy](CAPABILITIES.md#error-taxonomy); the library-specific overlay is in the matching library skill (Flow's `DOCA_ERROR_*` decision tree, for example, lives in [`doca-flow ## debug`](../libs/doca-flow/TASKS.md#debug)).

6. **Library layer.** Only after (1)–(5) are clean: route the conversation to the library skill for library-internal API semantics.

If the agent finds itself recommending a library-internal code change before completing (1)–(5), it is jumping layers — back up.

## sample-and-app-categorization

The cross-library **build-sweep model** the agent uses when the user
asks "build all DOCA samples" / "build all DOCA apps" / "which DOCA
samples work on my hardware". The honest answer is *"that depends on
which category you mean, which dependency stack you have, and which
target shape you are on"* — and treating *all DOCA samples* or *all
DOCA applications* as a single bucket is the load-bearing failure
mode the external feedback flagged (a build sweep that returns
"139/159 built" hides the fact that the 20 misses are mostly *env
not present* (GPUNetIO needs `nvcc`, RMAX needs `doca-rmax`), not
*the DOCA SDK is broken*).

This anchor is the **agent's reasoning model**, not a script: the
bundle does NOT ship a `build-sweep.sh`, a categorization JSON
manifest, or a dependency-resolver. It teaches the agent the
*classification*, the *evidence each tag is derived from*, and the
*report shape* a build sweep should produce so two operators reading
the same sweep output reach the same triage step.

### Samples vs applications

The two artifact shapes ship from different trees, are written for
different audiences, and answer different questions. The agent
makes the distinction *before* it picks a sweep strategy.

- **Samples** live under `/opt/mellanox/doca/samples/<library>/` on
  an installed host (and under `samples/` in the public
  [`NVIDIA-DOCA/doca-samples`](https://github.com/NVIDIA-DOCA/doca-samples)
  repo). A sample is a *library-level
  build probe* — it answers the question *"can this installed SDK
  compile code against this library's headers and link against its
  shipped `.so`?"*. A sample build pass is evidence for the
  install + the build env + that one library; a sample build pass
  is **not** evidence for application-level integration. Samples
  are the right target for the first `## test` step of every
  library skill.
- **Applications** live under `/opt/mellanox/doca/applications/` on
  an installed host (and under `applications/` in the public
  [`NVIDIA-DOCA/doca-samples`](https://github.com/NVIDIA-DOCA/doca-samples)
  repo — the public samples repo carries both `samples/` and
  `applications/` trees). An application is an
  *integrated reference workload* — it typically pulls in multiple
  DOCA libraries (Flow + Eth + Common; RDMA + DMA; etc.), often
  requires external packages (DPDK, SPDK, CUDA toolkit, MPI, the
  RMAX SDK, OpenMPI, UCX), often needs runtime topology to
  exercise (a peer host, a specific BlueField mode, a specific
  firmware capability, GPU + GPUDirect RDMA wiring), and sometimes
  needs a host ↔ DPU pair to run end-to-end. An application build
  pass is evidence for the *full integration stack* underneath
  it. An application build *fail* therefore has many more
  legitimate causes than a sample fail; classify before declaring.

The agent never tells a user *"build all apps"* without first
clarifying which subset: *all samples* (cheap, library-level),
*all top-level applications* (heavier, integration-level), *all
optional-stack apps* (per-category), or *build-only vs runnable on
this target* (whether the sweep is just a compile check or an
end-to-end run). The clarification is one sentence to the user,
not a guess.

### Category taxonomy

A canonical category labels every sample / app by the *DOCA stack
shape* it exercises. The agent applies categories from the
artifact's `meson.build`, its `README`, and the parent directory it
ships from — NOT from inferring the name. Categories the bundle
treats as first-class:

| Category | Stack shape | Representative artifacts |
| --- | --- | --- |
| `security` | DOCA App Shield, DOCA IPsec, packet inspection, file integrity, secure-channel patterns | `app_shield_agent`, `ipsec_security_gw`, `psp_gateway`, `secure_channel`, `yara_inspection`, APSH samples, file-integrity samples |
| `storage` | NVMe emulation, virtio-fs, storage path samples, NVMe-oF integration | `nvme_emulation`, `storage/*`, `virtiofs`, STA/NVMe-oF related samples |
| `networking` | DOCA Flow, DOCA Eth, DOCA RDMA, DOCA Comch — line-rate or control-plane packet/RDMA path | `eth_l2_fwd`, `simple_fwd_vnf`, `switch`, `upf_accel`, `ip_frag`, Flow samples, RDMA samples, Comch samples |
| `acceleration` | DOCA DMA, AES-GCM, SHA, Compress, Erasure Coding — offloading specific compute kernels | DMA samples, AES-GCM samples, SHA samples, Compress samples, Erasure Coding samples |
| `dpa-hpc` | DOCA DPA programs (DPACC build, FlexIO DEV) plus the PCC / UROM / RDMO HPC pieces | `dpa_all_to_all`, `pcc`, `urom_rdmo`, DPA samples, UROM samples |
| `gpu-media` | DOCA GPUNetIO, DOCA GPI, NVIDIA Rivermax integration, RMAX-class media samples | `gpu_packet_processing`, GPUNetIO samples, RMAX / Rivermax-shaped samples |
| `telemetry` | DOCA Telemetry library + DOCA Telemetry Exporter — counter readers and per-process counter publishers (the on-host **DOCA Telemetry Service** as-deployed binary is externally productized and out of bundle scope per [`AGENTS.md ## Non-goals`](../../AGENTS.md)) | Telemetry samples, telemetry-exporter samples, in-bundle telemetry-shaped diagnostics |

The agent does NOT add a sample / app to a category without
evidence (its meson target, its README's "what this does"
paragraph, the libraries it depends on per `pkg-config`). When
evidence is ambiguous, the agent reports the artifact uncategorized
rather than guessing — uncategorized is honest, mis-categorized is
the kind of drift that breaks operator trust.

### Dependency tags

Categories say *which DOCA stack shape*; dependency tags say *which
non-trivial dependencies the artifact needs at build / run time*.
A dependency tag is derived from the artifact's `meson.build`
(`dependency('libdpdk')`, `dependency('cuda')`, etc.), its build
log, its README, or — where authoritative information is missing —
from a known-good documented quirk (see below). The agent applies
the union of all that evidence; tags do not stand on
"probably-needs-this" guesses.

| Tag | Evidence | Meaning |
| --- | --- | --- |
| `needs_cuda` | `nvcc`, `dependency('cuda')` in meson, `<cuda.h>` include | Build requires CUDA Toolkit; runtime requires NVIDIA GPU driver loaded. |
| `needs_rmax` | `dependency('doca-rmax')`, `rmax/` include path, `doca-rmax` `pkg-config` | Build requires the NVIDIA Rivermax SDK (separate license + install). |
| `needs_mpi` | `mpicc` / `mpic++` / `mpirun` references, `dependency('mpi')` | Build requires an MPI implementation (OpenMPI, MPICH, …). On BlueField the MPI install can live at non-standard prefixes (e.g. `/usr/mpi/gcc/openmpi-<ver>/bin/`) — see [`doca-setup TASKS.md ## configure`](../doca-setup/TASKS.md#configure). |
| `needs_dpdk` | `dependency('libdpdk')` in meson, `rte_*` symbols | Build requires a matched DPDK install pulled in via `pkg-config libdpdk`. |
| `needs_spdk` | `dependency('spdk')` in meson, `spdk_*` symbols | Build requires the SPDK install (typically `/opt/mellanox/spdk/`). |
| `needs_dpa` | DPACC + FlexIO DEV build path, `dependency('doca-dpa')` | Build requires DOCA DPA + the DPACC compiler; runs only on BlueField. |
| `needs_host_peer` | README requires a peer-host process, *_host vs *_dpu split | Runtime requires a paired peer (the sample is incomplete without the other side). |
| `needs_lz4` | `dependency('liblz4')` in meson, `<lz4.h>` include | Build requires `liblz4-dev` (Ubuntu) / `lz4-devel` (RHEL). One known quirk lives at [`### known-sample-build-quirks`](#known-sample-build-quirks). |

The agent records the tag set for every artifact it sweeps. Missing
evidence is reported as "tag unknown" — never as "tag absent",
because the agent has no way to *prove* absence without reading the
artifact end-to-end.

### Skip vs fail (build-sweep classification)

The single most expensive classification error a build sweep can
make is conflating "skipped because dependency absent" with
"failed because the SDK is broken". The agent NEVER reports a
single `passed / failed` ratio without separating the two:

| Result class | Meaning | Operator action |
| --- | --- | --- |
| `built` | Build succeeded against this target's full required stack. | Move to runtime check (if the operator wanted runtime evidence). |
| `skipped-env-absent` | Build was attempted but a tagged dependency is not installed on this target (e.g. `needs_cuda` and `nvcc` is missing; `needs_rmax` and `doca-rmax` is not installed; `needs_dpa` and the operator is on host x86, not BlueField). | Report which dependency is missing, route the operator to install it (or accept that this artifact is not in scope for this target shape). |
| `built-with-known-quirk` | Build succeeded but only after applying a documented workaround listed in [`### known-sample-build-quirks`](#known-sample-build-quirks). | Report the quirk verbatim from the sweep output; do NOT silently suppress it. |
| `fail-sdk-class` | Build attempt reached the DOCA SDK headers / libs and a documented DOCA symbol / API surface was missing or wrong — this points at a partial install, a version-skew, or a real DOCA SDK bug. | Route to [`doca-debug ## debug`](../doca-debug/TASKS.md#debug) layer 3 (link) or layer 4 (runtime); do NOT close the sweep as "DOCA broken" without the layer 1-2 evidence in hand. |
| `fail-unknown` | Build failed and the failure does not match any of the patterns above. | Capture the build log, surface it verbatim to the operator, do NOT classify as SDK fail until evidence narrows it. |

A sweep report MUST present these as separate counters. The
canonical report shape:

```
sweep summary (target = <host shape>, DOCA = <observed pkg-config doca-common>):

  category          built  skipped-env-absent  built-with-known-quirk  fail-sdk-class  fail-unknown  total
  --------          -----  ------------------  ----------------------  --------------  ------------  -----
  security          N      0                   0                       0               0             N
  storage           N      0                   1 (lz4 quirk on X)      0               0             N+1
  networking        N      0                   0                       0               0             N
  acceleration      N      0                   0                       0               0             N
  dpa-hpc           0      M (needs_dpa)       0                       0               0             M
  gpu-media         0      P (needs_cuda)      0                       0               0             P
  telemetry         N      0                   0                       0               0             N
  uncategorized     -      -                   -                       -               -             -
  --------          -----  ------------------  ----------------------  --------------  ------------  -----
  total             ...    ...                 ...                     ...             ...           ...

dependency-tag inventory on this target (from probe, not from memory):
  needs_cuda: present? <yes/no/unknown>
  needs_rmax: present? <yes/no/unknown>
  needs_mpi:  present? <yes/no/unknown> (binary path: <mpicc path or "not on PATH">)
  needs_dpdk: present? <yes/no/unknown>
  needs_spdk: present? <yes/no/unknown>
  needs_dpa:  applicable? <yes if BlueField Arm, no if host x86, unknown if not detected>
  needs_lz4:  present? <yes/no/unknown>

per-fail evidence (only for fail-sdk-class + fail-unknown — never for skipped-env-absent):
  <artifact>: <one-line classification>  <log path on host or BF>
```

The point of this shape is that an external developer reading it
can: (a) tell at a glance which categories are usable on this
target *right now*; (b) tell which categories need a missing
dependency (and what the dependency is); (c) see the per-fail
evidence ONLY for real failures, not for missing-env skips that
are out of scope for this target.

### known-sample-build-quirks

A short, evidence-tagged list of build quirks the agent has seen
in shipped DOCA samples / applications. Each row is reported AS A
QUIRK in any sweep that hits it — the row exists so the agent does
not classify a documented quirk as `fail-sdk-class` and lose
operator trust by mis-reporting a workaround as an SDK fault.

| Quirk | Evidence | Documented workaround |
| --- | --- | --- |
| `doca_storage_gga_offload_sbc_generator` needs `-llz4` at link time | The application's link step fails on `lz4_*` symbols (`undefined reference to LZ4_compress_*`) even though its meson target does not explicitly declare a `liblz4` dependency in some shipped versions; the linker needs `-llz4` added (or the meson dependency added back). | Either pass `LDFLAGS="$(pkg-config --libs liblz4)"` for the build invocation, OR (if the operator is modifying the sample, per [`## modify`](#modify)) add `dependency('liblz4')` to the meson target. Mark the result as `built-with-known-quirk` (rationale = "DOCA storage sample expected `lz4` dependency was not declared by the meson target in some shipped versions; explicit `-llz4` closes the gap"). Do NOT classify the original failure as `fail-sdk-class`; this is a sample-build-script quirk, not a DOCA SDK bug. The agent surfaces the workaround verbatim, does NOT bake the workaround into a script, and does not silently suppress the original error. |

The list is intentionally small: the agent only adds a row when
the quirk is *documented evidence in a real build log* AND the
workaround has been *applied successfully*. The bundle does not
fabricate quirks from memory; an unfamiliar build failure is
reported as `fail-unknown` and surfaced verbatim, not silently
mapped to a fabricated "quirk".

## Command appendix

The commands the verbs above expect the agent to issue, grouped by
class so the agent reaches for the right family without searching prose.
Library-specific commands (Flow's per-pipe APIs, RDMA's QP APIs, …)
overlay in the matching library skill; this appendix lists only the
library-agnostic ones.

| Class | Command | Owning verb / anchor | Reads as healthy when … |
| --- | --- | --- | --- |
| Detect program version | `pkg-config --modversion doca-<library>` | [`## configure`](#configure) step 2 | Returns a version string identical to `doca_caps --version`. |
| Detect program version | `pkg-config --modversion doca-common` | [`## configure`](#configure) step 2 | Returns the unified DOCA version; quote *this* for API-availability answers. |
| Wire build (C/C++ Track 1) | `pkg-config --cflags doca-<library>` | [`## build`](#build) Track 1 step 2 | Returns `-I` flags rooted at `/opt/mellanox/doca/infrastructure/include`. |
| Wire build (C/C++ Track 1) | `pkg-config --libs doca-<library>` | [`## build`](#build) Track 1 step 2 | Returns `-L /opt/mellanox/doca/lib/<arch>-linux-gnu -ldoca_<library>` (plus deps). |
| Wire build (Track 2 FFI) | `pkg-config --cflags --libs doca-<library>` | [`## build`](#build) Track 2 step 1 | Same as above; binding tooling consumes both. |
| Find samples on disk | `ls /opt/mellanox/doca/samples/<library>/` | [`## modify`](#modify) precondition table | Lists `meson.build` plus the sample subdirectories. |
| Copy a sample | `cp -r /opt/mellanox/doca/samples/<library>/<sample>/ <writable>/` | [`## modify`](#modify) step 3 | Yields a writable copy outside the install tree. |
| Build the copy | `meson /tmp/build-<project> && ninja -C /tmp/build-<project>` | [`## build`](#build) Track 1 step 1 | Build succeeds; binary lives in `/tmp/build-<project>/`. |
| Run the program | `./<built-binary> --sdk-log-level 70 -h` | [`## run`](#run) steps 2-3 | `-h` lists the program's own CLI flags; trace logging in stderr for the first run. |
| Inspect `DOCA_ERROR_*` | `doca_error_get_descr(<rc>)` (called from program code) | [`## debug`](#debug) step 5 | Returns a string description; quote it verbatim. |

Two cross-cutting rules:

- **Never invent a `pkg-config` module name or a CLI flag.** Every
  command above comes from a public source (the DOCA SDK index in
  [`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md),
  the library's own `--help`, or the sample's `README`). Agent-invented
  flags are the failure mode the
  [AGENTS.md `## Ground rules`](../../AGENTS.md#ground-rules-every-agent-must-follow)
  anti-hallucination clause forbids; they break trust in every other
  command in the same answer.
- **Cite the version the user is on, not "latest".** Every command
  whose output is version-dependent (the API surface, the available
  pipe types, the supported capability bits) is answered against the
  observed `pkg-config --modversion doca-common`, not against an
  assumed release.

## Deferred task verbs

- **`install`.** Installing DOCA on a fresh host, or reaching an install from a no-install host (NGC container fallback at `nvcr.io/nvidia/doca/doca` for macOS / Windows / Linux without DOCA, lab box, cloud Linux without NIC, hardware path) — env scope. Defer to [`doca-setup ## no-install`](../doca-setup/TASKS.md#no-install) and the Installation Guide via [`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md).
- **Env preparation.** `PKG_CONFIG_PATH`, hugepages, devlink, representor enumeration — env scope. Defer to [`doca-setup ## configure`](../doca-setup/TASKS.md#configure).
- **Library API specifics.** Constructing a Flow pipe, RDMA queue setup, Comch channel construction, etc. — library scope. After this skill's verbs have produced a buildable, runnable program shape, hand off to the matching library skill (e.g. [`doca-flow`](../libs/doca-flow/SKILL.md)) for API-level guidance.
- **`deploy` / `rollback` at fleet scale.** Provisioning multiple BlueFields, coordinated firmware / BFB updates across a fleet, or staged rollback is **fleet-orchestration scope**, externally owned and out of this bundle's SDK scope. Do **not** hand-roll per-host deployment logic at that scale. Route to the orchestration entry-point in [`doca-public-knowledge-map ## Deploying DOCA services at scale`](../doca-public-knowledge-map/references/map.md#deploying-doca-services-at-scale--orchestration-entry-point-personascale-routing) (DOCA Platform Framework / NVIDIA Network Operator / Kubernetes Launch Kit, with authoritative links). Single-host / handful-of-DPUs PoC deployment stays in-bundle via [`doca-setup ## recognize`](../doca-setup/TASKS.md#recognize).
