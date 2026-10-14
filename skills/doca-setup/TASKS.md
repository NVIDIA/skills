# DOCA setup workflows

**Where to start:** For a deployment-shaped question whose deployment
path is not already known, walk [`## recognize`](#recognize) first.
It detects
the user's system shape (host x86 / BlueField Arm bare-metal /
DPU-only / fresh laptop with no hardware), asks the developer the
minimal set of questions needed to disambiguate, and routes to the
correct downstream skill: the container deployment path
([`doca-container-deployment`](../doca-container-deployment/SKILL.md)),
the bare-metal hardware deployment path
([`doca-bare-metal-deployment`](../doca-bare-metal-deployment/SKILL.md)),
or the no-hardware fallback ([`## no-install`](#no-install)). If you
already know the deployment shape and only need env prep, the verb
order from there is `configure` → `test` → `debug`; each section's
first paragraph names the precondition the previous one must have
satisfied.

Read this file when the loader sent you here from [SKILL.md](SKILL.md). For the underlying env surface (install profiles, build-flavor disk locations, version-detection commands, error taxonomy, observability cues, env-side safety constraints), see [CAPABILITIES.md](CAPABILITIES.md). For the *programming-class* counterparts (the canonical build pattern, the universal modify-a-shipped-sample first-app workflow, the universal lifecycle, the cross-library `DOCA_ERROR_*` debug order), see [`doca-programming-guide`](../doca-programming-guide/SKILL.md). For where to find official documentation, the on-disk install layout, or release notes, route through [`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md).

Each verb below describes the **shape of the workflow**, not a copy-paste recipe. The agent's job is to walk the user through the steps in order, verifying preconditions before recommending the next call.

This skill scopes itself to **env work and the front-door routing decision**. Three of the six lint-required task verbs (`## build`, `## modify`, `## run`) describe their own substance in [`doca-programming-guide`](../doca-programming-guide/SKILL.md) after the env / program split — the anchors here exist for lint compliance and route there. The verbs this skill *owns* are `## recognize` (the front door), `## configure`, `## test`, `## debug`, and the critical `## no-install` (the NGC container path included).

## recognize

The front-door system-shape walk lives in [references/recognize.md](references/recognize.md). Load that file next; do not improvise the leaf table.

## configure

Goal: prepare the user's host environment so that builds can find DOCA and runs can find the resources they need. **Precondition for this verb is that the host has DOCA installed.** If it doesn't, route to [`## no-install`](#no-install) first.

**Stop conditions BEFORE configure can proceed.** If any of the
following is true on the host, configure cannot start and the agent
must surface the gap to the user rather than push forward:

- **Apt repo / package vocabulary mismatch.** A package the agent is
  about to recommend (`doca-tools`, `doca-applications`, …) returns
  `Candidate: (none)` or `Unable to locate package` under
  `apt-cache policy <pkg>` on the user's host. See
  [`doca-version CAPABILITIES.md ## Apt-repo and OS-matrix preconditions`](../doca-version/CAPABILITIES.md#apt-repo-and-os-matrix-preconditions)
  for the precheck. Do NOT quote an install line whose packages do
  not exist in the user's configured repos.
- **Off-matrix host OS point release.** The host OS family is in the
  target DOCA's *Supported Operating Systems* table but the point
  release is outside the documented sub-range (e.g. Ubuntu 24.04.4
  on a DOCA 3.3 host whose matrix lists 24.04.x for `x ≤ 3`). Surface
  the gap verbatim, quote the supported sub-range from the release
  notes, and ask the user to either downgrade or accept the off-matrix
  risk explicitly before proceeding. Same source: [`doca-version
  CAPABILITIES.md ## Apt-repo and OS-matrix preconditions`](../doca-version/CAPABILITIES.md#apt-repo-and-os-matrix-preconditions).

Steps the agent should walk the user through:

1. **Confirm install presence and version.** Use the procedure in [`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md) (do not duplicate). Quote the observed `pkg-config --modversion doca-common`; do not assume *"latest"*. The version-detection rules in [CAPABILITIES.md ## Version compatibility](CAPABILITIES.md#version-compatibility) determine whether what the user has on disk is a coherent install or a partial upgrade that needs reinstalling first.

2. **Set `PKG_CONFIG_PATH` for the build environment.** The DOCA `*.pc` files live at `/opt/mellanox/doca/infrastructure/lib/pkgconfig/`. Add this to `PKG_CONFIG_PATH` in the user's shell profile (or in the build invocation itself) so that `pkg-config --modversion doca-flow` (or any other DOCA module) succeeds. Verify with `pkg-config --list-all | grep -i doca`. If the list is empty, the path or the install profile is wrong — see [CAPABILITIES.md ## Error taxonomy](CAPABILITIES.md#error-taxonomy).

   - **Non-DOCA `pkg-config` paths the user often also needs.** Many DOCA applications and samples integrate with DPDK, SPDK, FlexIO, UCX, or Rivermax — each ships its own `.pc` files in a separate prefix the kernel's default `pkg-config` discovery does NOT see. The agent surfaces the documented locations so the operator can decide whether to add them; it does not fabricate paths. The canonical locations are:

     - DPDK: `/opt/mellanox/dpdk/lib/<arch>-linux-gnu/pkgconfig/` (per the DOCA installer's documented layout).
     - SPDK: `/opt/mellanox/spdk/lib/pkgconfig/` (per the SPDK install on DOCA-shipped hosts).
     - FlexIO: `/opt/mellanox/flexio/lib/pkgconfig/` (present on hosts with the FlexIO toolchain installed).
     - UCX / OpenMPI: per the user's OS package manager (paths vary; verify with `pkg-config --list-all`).

     The verification step is the same as for DOCA: after adding any extra path, `pkg-config --list-all | grep -E '<module>'` and confirm the expected modules appear. Empty result = path wrong or component not installed; do NOT push forward with a build that depends on an invisible module.

3. **Confirm the build-tool baseline is installed.** A clean DOCA build environment expects the following host-OS packages to be present BEFORE the user attempts any sample or application build. On Ubuntu / Debian: `build-essential`, `meson`, `ninja-build`, `cmake`, `pkg-config`, `libjson-c-dev` (DOCA Argp + several services consume `json-c`), `liblz4-dev` (some shipped storage samples link `lz4` — see [`doca-programming-guide TASKS.md ## sample-and-app-categorization` → known sample-build quirks](../doca-programming-guide/TASKS.md#known-sample-build-quirks) for the `doca_storage_gga_offload_sbc_generator` quirk this is the documented fix for). On RHEL / OEL: the equivalent `@"Development Tools"` group plus `meson`, `ninja-build`, `cmake`, `pkgconfig`, `json-c-devel`, `lz4-devel`. The agent surfaces the package set as a precondition checklist (`dpkg -s <pkg>` / `rpm -q <pkg>` per package) BEFORE recommending an install line; on missing packages, ask the user to install them OR surface that this target shape can only build a subset of the DOCA catalog (per [`doca-programming-guide ## sample-and-app-categorization`](../doca-programming-guide/TASKS.md#sample-and-app-categorization) skip vs fail rules).

4. **Detect non-standard MPI layouts before MPI-dependent builds.** Several DOCA applications (UROM / HPC-class) require an MPI compiler wrapper (`mpicc`, `mpic++`, `mpirun`). On BlueField OS images the documented MPI install can live at a NON-standard prefix — for example `/usr/mpi/gcc/openmpi-<ver>/bin/mpicc` instead of `/usr/bin/mpicc`. Before the agent tells the user "the build will pick up `mpicc` automatically", the agent verifies: (a) `command -v mpicc` returns a non-empty path; (b) if (a) is empty, the agent runs the documented discovery (`ls /usr/mpi/gcc/openmpi-*/bin/mpicc 2>/dev/null`, `find /opt -name mpicc 2>/dev/null`, `dpkg -L openmpi-bin | grep bin/mpicc 2>/dev/null`, `rpm -ql openmpi 2>/dev/null`) to surface where MPI actually landed. If MPI is found at a non-standard prefix, the agent recommends `export PATH=<found-prefix>:$PATH` BEFORE the build (and surfaces that the build's `PATH` must be inherited correctly — `sudo` typically resets it). If MPI is genuinely not installed, the agent surfaces that the MPI-dependent subset of DOCA apps will skip-env-absent per the [`doca-programming-guide ## sample-and-app-categorization`](../doca-programming-guide/TASKS.md#sample-and-app-categorization) classification.

5. **Apt-source consistency precheck (load-bearing for any subsequent install).** Before recommending ANY `apt install doca-*` line on a host that already has DOCA installed, the agent walks the apt-source precheck in [`doca-version TASKS.md ## apt-source consistency`](../doca-version/TASKS.md#apt-source-consistency) to surface the most common partial-install root cause: `/etc/apt/sources.list.d/doca.list` pointed at one release channel (e.g. `latest` or `3.5`) while the host's installed packages are pinned to a different release (e.g. `3.1.0105`). Installing on top of a mismatched source is the most common cause of *"my BlueField was rolled back to 3.1, but my host packages came back as 3.5 and now nothing works"* — the agent NEVER skips this precheck.

6. **Set `LD_LIBRARY_PATH` for the runtime, if using the trace build flavor.** The program-side rationale for picking trace vs release lives in [`doca-programming-guide CAPABILITIES.md ## Capabilities and modes`](../doca-programming-guide/CAPABILITIES.md#capabilities-and-modes); the env mechanics here are: either link with the `doca-<lib>-trace` `pkg-config` module at build time, or set `LD_LIBRARY_PATH=/opt/mellanox/doca/lib/<arch>-linux-gnu/trace:$LD_LIBRARY_PATH` at runtime.

7. **Mount and reserve hugepages.** Required by DPDK-based DOCA
   libraries (Flow in particular). Hugepages are global state: first
   record the current `nr_hugepages`, derive the requested count from
   the selected application's documented memory requirement and the
   host's available memory. For 2 MiB pages, calculate
   `ceil(required_bytes / 2 MiB)` and use the application's installed
   sample/configuration guide as the only source for
   `required_bytes`; if it provides no requirement, stop and ask the
   application owner rather than guessing. Confirm the resulting
   reservation fits the operator-approved memory budget, show its
   impact, and obtain explicit approval. Never default to a fixed page
   count. This skill does not run a privileged write and does not
   pipe a value into `tee`. After approval, the operator applies the
   count through the site privilege process. On BlueField Ubuntu 22.04,
   prefer `doca-hugepages config` then `doca-hugepages reload` (the
   helper owns allocation; `config` alone does not apply). Do not mix
   `doca-hugepages` with raw sysfs writes. If the workload requires a
   hugetlbfs mount, the operator creates it through the same site
   process; this skill does not mount filesystems.

   Verify with `grep -i huge /proc/meminfo` and, if a mount was
   requested, `findmnt -t hugetlbfs`. Rollback restores the recorded
   previous count through the same site process that applied it, and
   unmounts `/mnt/huge` only if this workflow caused that mount and
   no workload is using it.

8. **Confirm device and representor visibility.** `devlink dev show` lists the network devices the kernel sees; `cat /sys/class/net/*/phys_port_name` shows the names of any active representors. If representors are expected but absent, first verify the target BDF, driver binding, SR-IOV/VF state, and current eswitch mode. Recommend `switchdev` only when that evidence and the version-matched deployment guide identify legacy eswitch mode as the cause; the change still requires explicit consent because it disrupts existing flows.

9. **Sanity-check before any program work.** Confirm with the user: which BlueField (or which host PCIe slot), which install version, which mode (host / DPU / switch). If any of these is unclear, stop and ask. Once these are confirmed, hand off to [`doca-programming-guide ## configure`](../doca-programming-guide/TASKS.md#configure) for the program-side configuration.

## build

> **Anchor exists for lint compliance.** The substance of this verb — the canonical `pkg-config doca-<library>` + meson build pattern, in two language tracks (C/C++ direct, non-C via FFI) — moved to [`doca-programming-guide ## build`](../doca-programming-guide/TASKS.md#build) when the env / program split happened. *Building a DOCA application is a programming verb, not an env verb.*

The env / program distinction is a CLASS distinction the agent must
make before it picks an answer. Route by which side of the line the
question sits on:

| The user asks ... | Route to ... | Reason |
| --- | --- | --- |
| "How do I build a DOCA application?" | [`doca-programming-guide ## build`](../doca-programming-guide/TASKS.md#build) | Programming verb. The canonical pattern + language tracks live there. |
| "How do I build a DOCA Flow app?" | Same. Then layer [`doca-flow ## build`](../libs/doca-flow/TASKS.md#build). | Programming verb with library-specific overlay. |
| "Why does my build fail with `pkg-config` not finding `doca-flow`?" | [`## debug`](#debug) layer 3 + [CAPABILITIES.md ## Error taxonomy](CAPABILITIES.md#error-taxonomy) | Env-class symptom (PKG_CONFIG_PATH / install profile). |
| "My build worked but `ld` says `cannot find -ldoca_flow`." | [`## debug`](#debug) layer 3 first; if env is clean, [`doca-debug ## debug`](../doca-debug/TASKS.md#debug) layer 4 (Link). | Could be either layer; rule env out first. |

The agent should never run an env-class build diagnosis against a
programming question, and vice versa. The four rows above are the
CLASSES; specific symptoms are instances.

## modify

> **Anchor exists for lint compliance.** The substance of this verb — the universal *derive a custom first application from a shipped sample* workflow, with C/C++ + non-C language tracks and an explicit precondition gate — moved to [`doca-programming-guide ## modify`](../doca-programming-guide/TASKS.md#modify) when the env / program split happened. *Deriving a first app from a sample is a programming verb, not an env verb.*

Route here only when the modify-class question is actually an env
question in disguise:

| The user asks ... | Route to ... | Reason |
| --- | --- | --- |
| "How do I derive a custom first app from a sample?" | [`doca-programming-guide ## modify`](../doca-programming-guide/TASKS.md#modify) | Programming verb. Owns the precondition table, the C/C++ + non-C tracks, the language-agnostic schema. |
| "I have no DOCA install — how do I even reach a sample to modify?" | [`## no-install`](#no-install) below | Env-class prerequisite to the modify workflow. |
| "I can't find any samples on disk." | [`doca-public-knowledge-map ## Layout of an installed DOCA package`](../doca-public-knowledge-map/SKILL.md#layout-of-an-installed-doca-package) | Routing-class question — where does the install put samples on disk. |
| "Samples are there but the build fails." | [`## debug`](#debug) layers 1–3 first | Likely env-class; rule out before assuming sample-content bug. |

The modify-from-sample workflow lives in programming-guide because it
generalizes across every DOCA library; this skill only owns the env
preconditions that workflow assumes are already true.

## run

> **Anchor exists for lint compliance.** The substance of this verb — running a built DOCA program, picking the right `--sdk-log-level`, mapping startup failures to env-class vs program-class — moved to [`doca-programming-guide ## run`](../doca-programming-guide/TASKS.md#run) when the env / program split happened. *Running a DOCA program is a programming verb; the env-class pre-run checklist (hugepages, devices, representors) is what this skill owns and what `## test` below verifies.*

The env-class pre-run checklist, listed here so an agent answering a
"my program won't start" question has the canonical sequence before
escalating to programming-guide:

| Pre-run check | Where it lives | What clean output looks like |
| --- | --- | --- |
| Hugepages mounted and reserved | [`## configure`](#configure) step 7 | `mount \| grep huge` shows hugetlbfs; `/proc/meminfo` shows non-zero free hugepages. |
| Network devices visible | [`## configure`](#configure) step 8 | `devlink dev show` lists the BlueField PCIe device. |
| Representors visible (Flow / Switching) | [`## configure`](#configure) step 8 | `/sys/class/net/*/phys_port_name` lists the expected representor indices. |
| Kernel modules loaded | [CAPABILITIES.md ## Error taxonomy](CAPABILITIES.md#error-taxonomy) | `lsmod \| grep mlx5` shows `mlx5_core`, `mlx5_ib`. |
| Mode set (host / DPU / switch) | [CAPABILITIES.md ## Capabilities and modes](CAPABILITIES.md#capabilities-and-modes) | `mlxconfig -d <pcie> q INTERNAL_CPU_MODEL` matches the mode the program expects. |

All five must be green before the program's own startup can be
debugged. A failure in any row is env-class; route to [`## debug`](#debug).
A clean checklist with a still-failing program is programming-class;
route to [`doca-programming-guide ## run`](../doca-programming-guide/TASKS.md#run).

## no-install

The Stage-1 / Stage-2 NGC path lives in [references/no-install.md](references/no-install.md). Load that file next. Do not invent an NGC tag or author application source on a host with no install.

### Stage 1 vs Stage 2 — open every "I am new to DOCA" answer with the staged roadmap

Lead with the two-row Stage 1 (container learning, no real packets) vs
Stage 2 (hardware runtime) table in
[references/no-install.md](references/no-install.md#stage-1-vs-stage-2--open-every-i-am-new-to-doca-answer-with-the-staged-roadmap)
before any command. **Resume point inside the container:** once the user
is inside Stage 1, they paste back `pkg-config --modversion doca-<library>`,
`pkg-config --cflags --libs doca-<library>`, and
`ls /opt/mellanox/doca/samples/<library>/<sample_name>/`; this skill then
resumes from [`doca-programming-guide ## modify`](../doca-programming-guide/TASKS.md#modify).

### How to pick an NGC tag without guessing

For first-app and CI-pinning questions, name and apply **How to pick an
NGC tag without guessing**, not just a generic Docker pull recommendation:

1. Open the [NGC DOCA Tags page](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/doca/containers/doca/tags).
   Match the requested release and platform using the full rule in
   [references/no-install.md](references/no-install.md#how-to-pick-an-ngc-tag-without-guessing).
2. Copy the exact visible tag for `nvcr.io/nvidia/doca/doca`; never assemble
   an example tag from version/architecture fragments. `:latest` is not a
   reproducible CI pin. If the catalog is unreachable, ask for a pasted
   candidate tag and stop short of claiming it was verified.
3. After pulling that image, run `pkg-config --modversion doca-common`
   **inside the selected container**, not on the outer host. Compare the
   output with the intended DOCA release using
   [doca-version](../doca-version/TASKS.md#configure). Record the exact image
   reference and observed version. A successful pull alone is not version
   verification; missing or mismatched output blocks the build handoff.

## test

Goal: verify the install is healthy enough that *any* program-level work is meaningful. Catch problems at the lowest layer before they propagate up.

**Env-class `## test` is an iterative loop, not a one-shot check.** The agent
runs the smallest meaningful probe at each layer, reads its output, picks the
next narrowest probe based on what's revealed, and only declares the env
"healthy" when every layer is observed clean in the same session. A snapshot
proves nothing about the next moment; the loop is what gives the install
durability claims.

The loop:

```
   .--> 1. Install layer probes
   |       (pkg-config / ls /opt/mellanox/doca / version)
   |
   |    2. Capability layer probes
   |       (doca_caps / library-specific cap APIs)
   |
   |    3. End-to-end smoke probe
   |       (build + run one known-good sample)
   |
   |    4. Read symptoms; classify:
   '----- env regression?   -> back to layer that broke; re-run that layer's probe
          program-class?    -> route to doca-programming-guide ## test
          clean across all? -> declare env healthy; record observed versions
```

The four-step variant of the loop, in detail:

1. **Install health check.** Run the [CAPABILITIES.md ## Observability](CAPABILITIES.md#observability) install-layer commands. Empty output where output is expected = install is wrong; do not proceed. If no install is reachable at all, this is a [`## no-install`](#no-install) situation. **Loop:** if any probe is unexpectedly empty, re-run after fixing (PKG_CONFIG_PATH set, profile reinstalled, etc.); do not skip ahead with one layer broken.

2. **Capability snapshot.** `doca_caps` and (for Flow) the Flow capability-query API. Save the output. Library-internal capability cross-checks live in the library skill — for Flow, [`doca-flow CAPABILITIES.md ## Capabilities and modes`](../libs/doca-flow/CAPABILITIES.md#capabilities-and-modes). **Loop:** if a capability the user's program will need is missing, do *not* recommend code changes — back up to layer 1 and check that the version / firmware actually exposes it; the env may need adjusting first.

3. **Smoke-test by building and running one shipped sample.** A *known-good* sample built and run cleanly is the cheapest possible end-to-end install validation. Use the canonical build pattern in [`doca-programming-guide ## build`](../doca-programming-guide/TASKS.md#build) (Track 1 for C/C++) and the run pattern in [`doca-programming-guide ## run`](../doca-programming-guide/TASKS.md#run). Inside an NGC container, the build half is the meaningful smoke test; the run half against real traffic requires a hardware path. **Loop:** if the build fails, the layer-1 / layer-2 probes either missed something or the env has drifted between probes — re-run them in this session, do not assume yesterday's output.

4. **Loopback / no-traffic run first, then introduce traffic.** Run the sample once with no real traffic offered to it. Successful start, clean shutdown, and zero counter increments is the expected baseline. **Loop:** if the no-traffic run is clean but the with-traffic run is not, the failure is almost certainly *program-class* now (the env passed every probe) — route to [`doca-programming-guide ## test`](../doca-programming-guide/TASKS.md#test) and stop drilling at the env layer.

The loop terminates when one of:

- Every layer is observed clean in the same session ⇒ env healthy; record the observed `pkg-config --modversion doca-common` and `doca_caps --version` strings so any later regression has a baseline.
- An env-class failure is found and fixed ⇒ restart at the layer that
  broke and re-run that layer plus every downstream layer, because the
  fix may invalidate downstream evidence. Earlier independent layers
  need not be repeated.
- The failure is reproducibly *not* env-class ⇒ hand off to [`doca-programming-guide ## test`](../doca-programming-guide/TASKS.md#test) with the captured probe outputs as evidence.

## debug

Goal: when something does not work, walk the layered diagnosis tree top-down so the agent does not jump to library-internal code-fix recommendations against an env-class symptom. This anchor is the **env-class half** of the layered debug ladder. The complementary **program-class half** (lifecycle, capability, error-description, library) lives in [`doca-programming-guide ## debug`](../doca-programming-guide/TASKS.md#debug). The **cross-cutting reference both halves escalate to** — tooling that spans env and program (`gdb`, `valgrind`, `ldd`, `strace`, `dmesg`, container introspection, core dumps, the `--sdk-log-level` and `doca-<lib>-trace` surfaces, the *Where to ask for help* escalation to the public Developer Forum) — lives in [`doca-debug ## debug`](../doca-debug/TASKS.md#debug). After completing layers (1)–(4) below, the natural next stop is `doca-debug ## debug` Layer 5 (Runtime) or `doca-programming-guide ## debug` for program-side errors.

Investigation order — **always**:

1. **Install layer.** Re-run the [CAPABILITIES.md ## Observability](CAPABILITIES.md#observability) install-layer commands. If `pkg-config --modversion doca-common` itself fails, no library-internal advice is meaningful. If no install is reachable, this is a [`## no-install`](#no-install) situation.
2. **Version layer.** `pkg-config --modversion doca-<lib>` against the user's runtime `doca_caps --version`. Mismatch ⇒ partial upgrade or stale build env; reinstall consistently before code changes.
3. **Build layer.** Did the user build into `/tmp/build*` (correct) or in-tree (wrong; permission errors will mislead)? Did they pick the right build flavor (release vs trace; the program-side rationale lives in [`doca-programming-guide`](../doca-programming-guide/SKILL.md))? Build-time symptoms map to [CAPABILITIES.md ## Error taxonomy](CAPABILITIES.md#error-taxonomy).
4. **Runtime layer.** Hugepages mounted? Modules loaded? Representors visible? The matrix in [CAPABILITIES.md ## Error taxonomy](CAPABILITIES.md#error-taxonomy) maps each symptom to its likeliest cause.
5. **Program / library layer.** Only after (1)–(4) are clean: route the conversation to [`doca-programming-guide ## debug`](../doca-programming-guide/TASKS.md#debug) for the universal lifecycle / capability / error-description / library order, and from there to the matching library skill (e.g. [`doca-flow ## debug`](../libs/doca-flow/TASKS.md#debug)) for library-specific overlays.

If the agent finds itself recommending a code change before completing (1)–(4), it is jumping layers — back up.

## Deferred task verbs

- **`install`.** Installing DOCA itself is a knowledge-map question. Defer to [`doca-public-knowledge-map ## Layout of an installed DOCA package`](../doca-public-knowledge-map/SKILL.md#layout-of-an-installed-doca-package) and [`## no-install`](#no-install) Path 0 for a host without DOCA.
- **Build / first-app derivation / running a program.** These are programming verbs. After `## configure` and `## test` are green, hand off to [`doca-programming-guide`](../doca-programming-guide/SKILL.md).
- **Library API specifics.** After the env is known-good, hand off to the matching library skill (for example [`doca-flow`](../libs/doca-flow/SKILL.md)).

## Command appendix

See [references/command-appendix.md](references/command-appendix.md).
