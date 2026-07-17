# DOCA setup workflows

**Where to start:** Before anything else, walk
[`## recognize`](#recognize) — that is the **front door**. It detects
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

Goal: detect the user's **system shape and deployment target** before any other env work begins, so the agent can route the user to the correct downstream skill instead of guessing. This is the **front door**. The agent MUST walk this before answering any deployment-shaped question (*"how do I deploy"*, *"how do I run my DOCA app"*, *"how do I get my DOCA service up"*, *"I just got a BlueField, what now"*, *"my code is built, what next"*).

**Why this exists.** The bundle today supports four distinct system shapes — host x86 + remote BlueField NIC, BlueField Arm bare-metal, DPU-only / converged-accelerator, and fresh-laptop-with-no-hardware — and **two parallel deployment paths** on top: containers (kubelet-standalone with a YAML pod-spec drop, owned by [`doca-container-deployment`](../doca-container-deployment/SKILL.md)) and bare-metal binaries (direct/tmux/systemd launch, owned by [`doca-bare-metal-deployment`](../doca-bare-metal-deployment/SKILL.md)). The wrong failure mode is to silently push every developer onto the container path because the agent loaded that skill first. The right behavior is to recognize the system, confirm with the developer, and route. This is what `## recognize` enforces.

**Step 0 — scale gate (run this BEFORE the four-leaf walk).** First decide *persona and scale*, because the four leaves below are the **developer / PoC / small-dev** path (a single host or a handful of DPUs). If the user's intent is **production / fleet-scale** — provisioning racks of DPUs, declarative lifecycle, coordinated BFB/firmware rollouts across many BlueFields (signals: *"across N DPUs"*, *"in production"*, *"at fleet/cluster scale"*, *"Kubernetes operator"*, *"declaratively manage"*) — do **not** walk the single-host leaves and do **not** hand-roll per-host `bfb-install` loops or static-pod drops. Route to the orchestration entry-point in [`doca-public-knowledge-map ## Deploying DOCA services at scale`](../doca-public-knowledge-map/references/map.md#deploying-doca-services-at-scale--orchestration-entry-point-personascale-routing), which points at DOCA Platform Framework (DPF), NVIDIA Network Operator, and the Kubernetes Launch Kit (externally owned, routed-not-owned). Only if the user is clearly in the developer/PoC persona (or unsure and small-scale) continue with the four-leaf walk below.

**The decision tree the agent walks.** The agent asks the **minimum** number of questions needed to land on one of the four leaves below. Do not ask all of them; stop as soon as the leaf is unambiguous. The system-shape × deployment-shape matrix the bundle covers lives in [CAPABILITIES.md ## Capabilities and modes](CAPABILITIES.md#capabilities-and-modes); this anchor is the *interactive* walk of that matrix.

1. **Detect first; ask only the residual.** Before any question, the agent attempts a non-destructive auto-detect against what is already on the wire:
   - `uname -m` (x86_64 vs aarch64 — the BlueField Arm side is aarch64).
   - `lspci -d 15b3:` (BlueField NIC PCIe visibility — non-empty output proves a BlueField is reachable from this host).
   - `pkg-config --modversion doca-common` (DOCA install presence + version; honor [`doca-structured-tools-contract`](../doca-structured-tools-contract/SKILL.md) — prefer the structured output and fall back to the manual command). If the install is absent, this leg lands on `## no-install`.
   - `cat /etc/nvidia/bf-release` *if it exists* (BlueField OS image marker — present only on BlueField Arm bare-metal).
   - `cat /proc/cpuinfo | grep -m1 'model name'` (host CPU model — distinguishes a developer laptop / x86 workstation from a server with a BlueField installed).
   The agent **quotes the observed output back to the user** and uses it as the prior; it does NOT infer the system shape from the user's phrasing alone.

2. **Ask up to three residual questions, in this priority order.** Each question is *closed-form* — multiple choice or yes/no — so the developer can answer in one word and the agent can route without ambiguity.
   - **Q1 (target):** *"Where will the DOCA workload run — on the host CPU talking to a BlueField NIC over PCIe, on the BlueField Arm cores directly, on a DPU-only / converged-accelerator card (e.g. SuperNIC-class), or you are not sure yet?"* The four answers map 1:1 to leaves 2-5 below. Skip if `uname -m` + `lspci -d 15b3:` already pinned this.
   - **Q2 (packaging):** *"Will the workload be deployed as a kubelet-standalone container (YAML pod-spec drop on the BlueField), as a bare-metal binary you launch directly (CLI / tmux / systemd), or you do not know yet?"* Two answers route to the two deployment paths; the *"do not know"* answer triggers the brief explainer in step 3 below. Skip if the developer has already produced a binary or a pod-spec YAML.
   - **Q3 (workload kind):** *"Is the workload a DOCA-linked **application** you wrote (or are about to write), or a packaged DOCA **service** from NVIDIA (Argus / DMS / Firefly / Flow-Inspector / OS-Inspector / UROM-svc)?"* Application + bare-metal is the most common mismatch the question catches; service + container is the canonical NVIDIA-packaged path. Skip if the matching artifact already exists on disk.

3. **If the developer answers "do not know" to Q2, give them the one-paragraph decision rule, then re-ask Q2.** The rule: *"Containers are the canonical way to deploy a packaged NVIDIA DOCA service (the BlueField OS image already ships kubelet-standalone + the runtime); the operator drops a YAML pod-spec into a documented directory and the pod starts. Bare-metal is the canonical way to run a DOCA-linked application you wrote yourself; you launch the binary directly and you own the lifecycle (CPU pinning, hugepages, systemd unit). The two paths are equally supported; the right answer depends on the workload, not on which path is 'better'."* Then re-ask Q2.

4. **Route to the matching leaf** (the agent states the routing explicitly: *"Based on what you said and what I detected, the right next skill is …"*). The four leaves:
   - **Leaf A — fresh laptop / no DOCA install / no BlueField reachable from this host:** route to [`## no-install`](#no-install). The NGC DOCA container is the universal Stage-1.
   - **Leaf B — host x86 with a reachable BlueField, application workload, bare-metal launch:** route to [`doca-bare-metal-deployment`](../doca-bare-metal-deployment/SKILL.md) for the launch contract, after completing [`## configure`](#configure) for the env prep on this host.
   - **Leaf C — host x86 (or BlueField Arm bare-metal) with a reachable BlueField, service workload, container launch:** route to [`doca-container-deployment`](../doca-container-deployment/SKILL.md) for the pod-spec drop, after completing [`## configure`](#configure) for the env prep.
   - **Leaf D — BlueField Arm bare-metal (DPU-side), application workload, bare-metal launch:** route to [`doca-bare-metal-deployment`](../doca-bare-metal-deployment/SKILL.md). Cross-link the BlueField-Arm-specific overlay there.
   - **Leaf E — DPU-only / converged-accelerator card (e.g. SuperNIC-class) with no host CPU available for DOCA:** today this is a constrained subset of Leaf D — route to [`doca-bare-metal-deployment`](../doca-bare-metal-deployment/SKILL.md) with the explicit caveat that some host-side env steps in [`## configure`](#configure) do not apply (e.g. the host-side `LD_LIBRARY_PATH`). The agent should ALSO load [`doca-hardware-safety`](../doca-hardware-safety/SKILL.md) early on this leaf because device-only systems are higher-stakes (no escape hatch on a host CPU).

5. **Stop and confirm before proceeding.** Before handing off to the routed skill, the agent restates the routing decision in one line — *"You are on a `<system shape>`, deploying a `<workload kind>` via the `<deployment path>` path; I'll continue with `<routed skill>`. Speak up now if any of that is wrong."* — and waits for the developer's acknowledgment. Routing failures are far cheaper to fix here than after the deployment is half-built.

**What this verb deliberately does NOT do.** It does not install DOCA (that's `## configure` after `## no-install` for the install-absent leaf). It does not bind hardware resources (that's [`doca-bare-metal-deployment ## run`](../doca-bare-metal-deployment/TASKS.md#run) or, for containers, the pod-spec mount in [`doca-container-deployment ## modify`](../doca-container-deployment/TASKS.md#modify)). It does not pick a DOCA version (that's [`doca-version`](../doca-version/SKILL.md)). It does not author the workload (that's [`doca-programming-guide`](../doca-programming-guide/SKILL.md)). It does not change device state (that's a [`doca-hardware-safety`](../doca-hardware-safety/SKILL.md) meta-policy concern). The verb is **exactly** *"detect, ask the minimum, route, confirm"* — nothing more.

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

7. **Mount and reserve hugepages.** Required by all DPDK-based DOCA libraries (Flow in particular). The agent must read [CAPABILITIES.md ## Safety policy](CAPABILITIES.md#safety-policy) item 2 before recommending the change; hugepages are global state. The minimum-viable sequence is:

   ```bash
   echo '1024' | sudo tee -a /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
   sudo mkdir -p /mnt/huge
   sudo mount -t hugetlbfs -o pagesize=2M nodev /mnt/huge
   ```

   Verify with `mount | grep huge` and `cat /proc/meminfo | grep -i huge`.

8. **Confirm device and representor visibility.** `devlink dev show` lists the network devices the kernel sees; `cat /sys/class/net/*/phys_port_name` shows the names of any active representors. If the user expects representors and the listing is empty, switching the eswitch to `switchdev` mode is required (`devlink dev eswitch set <pcie> mode switchdev`) — but only with the user's explicit consent, since the change disrupts existing flows.

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
| Hugepages mounted and reserved | [`## configure`](#configure) step 4 | `mount \| grep huge` shows hugetlbfs; `/proc/meminfo` shows non-zero free hugepages. |
| Network devices visible | [`## configure`](#configure) step 5 | `devlink dev show` lists the BlueField PCIe device. |
| Representors visible (Flow / Switching) | [`## configure`](#configure) step 5 | `/sys/class/net/*/phys_port_name` lists the expected representor indices. |
| Kernel modules loaded | [CAPABILITIES.md ## Error taxonomy](CAPABILITIES.md#error-taxonomy) | `lsmod \| grep mlx5` shows `mlx5_core`, `mlx5_ib`. |
| Mode set (host / DPU / switch) | [CAPABILITIES.md ## Capabilities and modes](CAPABILITIES.md#capabilities-and-modes) | `mlxconfig -d <pcie> q INTERNAL_CPU_MODEL` matches the mode the program expects. |

All five must be green before the program's own startup can be
debugged. A failure in any row is env-class; route to [`## debug`](#debug).
A clean checklist with a still-failing program is programming-class;
route to [`doca-programming-guide ## run`](../doca-programming-guide/TASKS.md#run).

## no-install

Goal: behave correctly when the user wants to do *anything* program-side (build a sample, derive a first app, run a program) but the env-class preconditions aren't met — i.e., the user is on a fresh laptop, a CI runner, a remote machine, or any host without DOCA installed. This applies regardless of the user's chosen language: Python, Rust, Go, and Node consumers all need the DOCA `*.so` to bind against, and a fresh host with no install can't provide that any more than it can provide the C samples.

The wrong behavior — and the failure mode this section exists to prevent — is for the agent to author DOCA application source code from documentation prose (in *any* language: C, C++, Rust, Go, Python wrapper, etc.), mark unknowns with placeholder comments, and present it as a *first app*. That output looks complete to the user, won't compile or link against any real DOCA install, and breaks the user's trust in every other answer the skill produces. Don't.

### Stage 1 vs Stage 2 — open every "I am new to DOCA" answer with the staged roadmap

When the user opens with *"I am new to DOCA, guide me to my first app"* (or any equivalent beginner orientation prompt), the agent's first move is **not** a command. The agent leads with the staged roadmap below and only *then* drops into the per-path details under [What the agent does instead](#what-the-agent-does-instead). This prevents the *command-first overwhelm* failure mode that a reviewer flagged ("the response was correct but too command-heavy for a beginner"). The table is also surfaced verbatim in [`README.md`](../../README.md) so a human reading the bundle without an agent sees the same roadmap; runtime agents MUST quote (or paraphrase faithfully) the same two stages here.

| Stage | What the user is doing | Where they are | Bundle path |
| --- | --- | --- | --- |
| **Stage 1 — container learning** | Read the API surface, build / modify a shipped C sample, smoke the build, learn the `pkg-config doca-<library>` + meson pattern. **No real packets cross hardware.** This is the universal entry point for any user on macOS, Windows, or Linux without DOCA. | The public **NGC DOCA container** `nvcr.io/nvidia/doca/doca:<tag>`, pulled with Docker and run with `-it --rm`. `/opt/mellanox/doca` is populated inside the container by construction. | [`## no-install`](#no-install) **Path 0** below, then [`doca-programming-guide ## modify`](../doca-programming-guide/TASKS.md#modify), then the matching library skill (e.g. [`doca-flow`](../libs/doca-flow/SKILL.md)). |
| **Stage 2 — hardware runtime** | Run the app you built in Stage 1 against real traffic on a real NIC / DPU. Programmed flows, real packets, counters move. | Either a **Linux host with a ConnectX / BlueField NIC** (Path C), or remote-into a **lab box that already has DOCA + hardware** (Path A). | [`## no-install`](#no-install) **Paths A / C** below, then [`doca-bare-metal-deployment`](../doca-bare-metal-deployment/SKILL.md) or [`doca-container-deployment`](../doca-container-deployment/SKILL.md) depending on how the user wants to ship the app. |

**The resume point inside the container is part of the answer.** Tell the user: *"Once you're inside the Stage-1 container, paste back the output of `pkg-config --modversion doca-<library>`, `pkg-config --cflags --libs doca-<library>`, and `ls /opt/mellanox/doca/samples/<library>/<sample_name>/`. I resume from [`doca-programming-guide ## modify`](../doca-programming-guide/TASKS.md#modify) step 1 with the real install in hand."* The resumption promise is what makes Stage 1 actionable rather than a dead end.

### What the agent does instead

1. **State the limitation explicitly, once.** Tailor the wording to the user's language but keep the substance the same: *"DOCA needs a Linux environment with the DOCA install tree, a `pkg-config doca-<library>` that resolves, and (for any language other than C/C++) the `*.so` libraries available for FFI / bindings to load. I can't write your first app from this environment because the verified pieces — the shipped C samples in the C/C++ track, or the `*.so` and headers your bindings will link against in any other language — live in an environment I can't reach yet. The good news is there's a public Stage-1 fallback that works on any OS with Docker — the NGC DOCA container — and three other paths depending on what you actually need. Here's the menu, and what I can do once you reach any of them."*

2. **Hand the user the procedure they will execute on the install host.** Library- and language-aware, with the substance owned by [`doca-programming-guide`](../doca-programming-guide/SKILL.md):
   - **C / C++ track** — the procedure is [`doca-programming-guide ## modify`](../doca-programming-guide/TASKS.md#modify) (the *modify-a-shipped-sample* workflow), with library-specific overrides from the matching library skill (e.g. [`doca-flow ## build`](../libs/doca-flow/TASKS.md#build) for Flow).
   - **Other-language track** — the procedure is [`doca-programming-guide ## build`](../doca-programming-guide/TASKS.md#build) Track 2 (FFI / bindings against the public C ABI), then library-specific guidance from the matching library skill.

   In both cases, *no* application source code is written in this conversation — the procedure is what the user runs against the real install once they reach it.

3. **Walk the user through reaching an install environment.** Four honest paths, named in order of *cost to start trying* (Path 0 is the universal default for any non-Linux user; the others are situational).

   | Path | When it fits | What the agent walks the user through |
   | --- | --- | --- |
   | **0. NGC DOCA container** (`nvcr.io/nvidia/doca/doca`) — **canonical first option for any user on macOS, Windows, or Linux without DOCA.** | The user wants to build samples, modify a sample, read the API surface, generate FFI bindings, learn — *anything except real-traffic runtime against a real NIC*. Works on any OS that runs Docker. Free; no NVIDIA hardware required. | (1) Install Docker (Docker Desktop on macOS / Windows; native Docker on Linux). (2) **Pick a tag using the deterministic rule below — never invent one.** (3) `docker pull nvcr.io/nvidia/doca/doca:<tag-copied-verbatim-from-the-catalog>`. The public DOCA images are anonymously pullable at the time of writing; if a particular tag asks for auth, sign up for a free NGC account at <https://ngc.nvidia.com>, generate an API key, and `docker login nvcr.io -u '$oauthtoken' -p <api-key>` once. (4) `docker run -it --rm -v $HOME/dev:/work nvcr.io/nvidia/doca/doca:<tag> bash`; inside the container the user has a real `/opt/mellanox/doca` install — `pkg-config --modversion doca-<library>` works, `ls /opt/mellanox/doca/samples/` works, and the workflows in [`## configure`](#configure), [`doca-programming-guide ## build`](../doca-programming-guide/TASKS.md#build), and [`doca-programming-guide ## modify`](../doca-programming-guide/TASKS.md#modify) all work. **Limitations to surface upfront:** no real NIC inside the container, so DPDK / DOCA calls that need real hardware will fail at runtime — that is *expected* and the right move at that point is to graduate the user to Path A or Path C below. |
   | **A. Existing Linux + DOCA host** (lab box, dev server, BlueField over `rshim`) | The user already has a DOCA-installed host — most common case at NVIDIA, less common for external users. | SSH or Cursor-remote into it; rerun the [`## configure`](#configure) workflow there, then hand off to [`doca-programming-guide ## modify`](../doca-programming-guide/TASKS.md#modify). |
   | **B. Fresh Linux instance, no NIC** (laptop running Ubuntu, cloud VM, etc.) | The user wants a *persistent, native* install (not container-scoped) but doesn't have NVIDIA hardware. | Pick any Linux distro listed under the [DOCA Host Supported OS table](https://docs.nvidia.com/doca/sdk/NVIDIA+DOCA+Installation+Guide+for+Linux). Install via the Installation Guide; the **build-only** parts of [`doca-programming-guide ## build`](../doca-programming-guide/TASKS.md#build) and [`doca-programming-guide ## modify`](../doca-programming-guide/TASKS.md#modify) work; the actual runtime needs hardware (Path C). For most users, **Path 0 is faster and lighter** unless the user explicitly wants a non-container install. |
   | **C. Linux + ConnectX or BlueField hardware** | The user wants the real end-to-end runtime, including programmed flows and real packet behavior. | Either user-owned hardware, an internal lab allocation, or the [DOCA Downloads page](https://developer.nvidia.com/doca-downloads) for the BFB image to bring up a BlueField. The agent does *not* recommend specific cloud SKUs by name unless they are listed in the public Supported OS table; cloud GPU/ARM SKUs do not generically include DOCA-eligible NICs and the agent must not pretend otherwise. |

### How to pick an NGC tag without guessing

Image tags are version-dated and platform-shaped; **never guess one** — that violates [AGENTS.md ground rule 3](../../AGENTS.md#ground-rules-every-agent-must-follow) (never invent symbols, URLs, paths, or package names). The deterministic rule:

1. Open the catalog **Tags** page directly: <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/doca/containers/doca/tags>. That is the *only* authoritative list of tags that actually exist for `nvcr.io/nvidia/doca/doca`. Do not fall back to memory.
2. Detect the user's host axes from a brief question or from output the user has already pasted:
   - **Architecture** — `uname -m`: `x86_64` ⇒ pick a tag containing `linux-amd64`; `aarch64` / Apple Silicon ⇒ pick one containing `linux-arm64`.
   - **OS family** — match the OS family fragment in the tag string (`ubuntu22.04`, `rhel9.4`, etc.) to what the user runs on the *outer* host. For a Mac on Docker Desktop, any Linux flavor is fine because the container is its own world; default to Ubuntu LTS.
   - **CUDA?** — default **no** for first-app work; CUDA-enabled variants are larger and only relevant if the user is also using CUDA (e.g. GPUNetIO / GPI / GPUDirect work).
   - **Host vs DPU flavor** — default **host flavor**. Pick a DPU flavor *only* when the user is targeting the BlueField Arm OS (a less common first-app axis).
3. From the *visible* tags list, pick the **highest-numbered** tag that matches all four axes. The tag string is treated as opaque text from the catalog — the agent does NOT assemble a tag from version + arch fragments out of memory.
4. `docker pull nvcr.io/nvidia/doca/doca:<tag-copied-verbatim-from-the-catalog>`.
5. **If the agent cannot reach the catalog page from this session, say so explicitly and ask the user to paste the candidate tag from the catalog.** The agent does NOT fabricate a tag string. This is the same *never invent symbols, URLs, paths, or package names* discipline as [AGENTS.md ground rule 3](../../AGENTS.md#ground-rules-every-agent-must-follow); fabricating a tag here is a release-blocker.

The same rule is documented for human readers in [`README.md ## Beginner roadmap`](../../README.md#beginner-roadmap--stage-1-container-learning--stage-2-hardware-runtime); the two surfaces stay in sync.

4. **Do not scaffold a project on the un-installed host.** Do not produce `meson.build`, `CMakeLists.txt`, `Cargo.toml`, `setup.py`, `go.mod`, an application source file in any language, project directories, or any artifact that would mislead the user into thinking a build is one command away. The agent's *only* artifacts in this state are: (a) the install / path procedure for the user to run in the chosen environment, and (b) the menu above. The skill's claim is *"I'll be useful the moment you reach a real install — including the NGC container, which is one `docker pull` away"*; making artifacts now would dilute that claim with files that are not buildable in this environment.

5. **Promise the resumption.** Tell the user: *"When you're inside the NGC container, the lab host, the cloud Linux VM, or the hardware host, paste me the output of `pkg-config --modversion doca-<library>` and `pkg-config --cflags --libs doca-<library>`, plus (C/C++ track) `ls /opt/mellanox/doca/samples/<library>/<sample_name>/` or (other-language track) the `*.so` filename your bindings will load. I'll resume from [`doca-programming-guide ## modify`](../doca-programming-guide/TASKS.md#modify) step 1 (C/C++) or from the matching library skill's bindings guidance via [`doca-programming-guide ## build`](../doca-programming-guide/TASKS.md#build) Track 2 (other languages) with the real install in hand."*

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
- An env-class failure is found and fixed ⇒ restart the loop from the layer that broke (NOT from the top; you've already validated everything above).
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

## Command appendix

The commands the verbs above expect the agent to issue or have the user
issue, grouped by class so the agent reaches for the right family
without searching prose. Each command appears with the verb it belongs
to and the H2 anchor in this file or in
[CAPABILITIES.md](CAPABILITIES.md) that explains the expected output.

| Class | Command | Owning verb / anchor | Reads as healthy when … |
| --- | --- | --- | --- |
| Detect install | `pkg-config --modversion doca-common` | [`## test`](#test) step 1 / [CAPABILITIES.md ## Version compatibility](CAPABILITIES.md#version-compatibility) | Returns a single version string. Empty / "not found" ⇒ install or `PKG_CONFIG_PATH` is wrong. |
| Detect install | `pkg-config --list-all \| grep -i doca` | [`## configure`](#configure) step 2 | Lists at least one `doca-*` module. Empty ⇒ install profile probably wrong. |
| Detect install | `doca_caps --version` | [`## test`](#test) step 2 | Matches `pkg-config --modversion doca-common`. Mismatch ⇒ partial upgrade. |
| Detect install | `cat /opt/mellanox/doca/applications/VERSION` | [CAPABILITIES.md ## Version compatibility](CAPABILITIES.md#version-compatibility) | Fallback when `pkg-config` is broken. Returns the same string. |
| Wire build | `pkg-config --cflags doca-<library>` | [`## configure`](#configure) step 2 | Returns `-I/opt/mellanox/doca/infrastructure/include` (or trace-flavor equivalent). |
| Wire build | `pkg-config --libs doca-<library>` | [`## configure`](#configure) step 2 / [`## run`](#run) checklist | Returns `-L/opt/mellanox/doca/lib/<arch>-linux-gnu -ldoca_<library>` plus deps. |
| Wire runtime | `mount \| grep huge` | [`## configure`](#configure) step 4 / [CAPABILITIES.md ## Observability](CAPABILITIES.md#observability) | Shows a `hugetlbfs` mount. Empty ⇒ no hugepages reserved. |
| Wire runtime | `cat /proc/meminfo \| grep -i huge` | [`## configure`](#configure) step 4 | Shows `HugePages_Total` > 0 and `HugePages_Free` > 0. |
| Wire runtime | `devlink dev show` | [`## configure`](#configure) step 5 | Lists the expected BlueField / ConnectX PCIe BDF. |
| Wire runtime | `cat /sys/class/net/*/phys_port_name` | [`## configure`](#configure) step 5 | Shows the representor port names for the device. |
| Wire runtime | `lsmod \| grep mlx5` | [CAPABILITIES.md ## Error taxonomy](CAPABILITIES.md#error-taxonomy) | Shows `mlx5_core`, `mlx5_ib` loaded. |
| Detect mode | `mlxconfig -d <pcie> q INTERNAL_CPU_MODEL` | [CAPABILITIES.md ## Capabilities and modes](CAPABILITIES.md#capabilities-and-modes) | Matches the mode the program expects (`EMBEDDED_CPU(1)` for switch mode). |
| Reach an install | `docker pull nvcr.io/nvidia/doca/doca:<tag>` | [`## no-install`](#no-install) Path 0 | Pull succeeds; tag exists on NGC. |
| Reach an install | `docker run -it --rm -v $HOME/dev:/work nvcr.io/nvidia/doca/doca:<tag> bash` | [`## no-install`](#no-install) Path 0 | Shell starts; `/opt/mellanox/doca` is populated inside. |

The appendix lists the *families* of commands; the verbs above describe
the *when* and *why*. New commands must come from a public NVIDIA source
(documented in [CAPABILITIES.md ## Capabilities and modes](CAPABILITIES.md#capabilities-and-modes)
or [`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md));
agent-invented flags fail the bundle's anti-hallucination contract
declared in [AGENTS.md `## Ground rules`](../../AGENTS.md#ground-rules-every-agent-must-follow).

## Deferred task verbs

- **`install`.** Installing DOCA itself on a fresh host or BlueField is a knowledge-map question — the canonical Installation Guide and the package profile choice are documented there. Defer to [`doca-public-knowledge-map ## Layout of an installed DOCA package`](../doca-public-knowledge-map/SKILL.md#layout-of-an-installed-doca-package) and the Installation Guide URL it routes to. The fastest way to reach an install on a non-Linux host (or any host without DOCA) is [`## no-install`](#no-install) Path 0 (NGC container).
- **Build / first-app derivation / running a program.** These are *programming verbs*, not env verbs, and live in [`doca-programming-guide`](../doca-programming-guide/SKILL.md). After this skill's `## configure` and `## test` have produced a known-good environment, hand off there.
- **Library API specifics.** Constructing a Flow pipe, RDMA queue setup, etc. — outside the scope of this skill. After this skill's verbs have produced a known-good environment, hand off to [`doca-programming-guide`](../doca-programming-guide/SKILL.md) for the cross-library patterns and then to the matching library skill (e.g. [`doca-flow`](../libs/doca-flow/SKILL.md)) for API-level guidance.
- **`deploy` / `rollback` at fleet scale.** Provisioning multiple BlueFields, coordinated firmware/BFB updates across a fleet, or staged rollback is **fleet-orchestration scope**, externally owned and out of this bundle's SDK scope. Do **not** hand-roll per-host `bfb-install` loops or static-pod rollouts at that scale. Route to the orchestration entry-point in [`doca-public-knowledge-map ## Deploying DOCA services at scale`](../doca-public-knowledge-map/references/map.md#deploying-doca-services-at-scale--orchestration-entry-point-personascale-routing), which points at DOCA Platform Framework (DPF), NVIDIA Network Operator, and the Kubernetes Launch Kit with authoritative links. Single-host / handful-of-DPUs PoC deployment stays in-bundle ([`doca-container-deployment`](../doca-container-deployment/SKILL.md) / [`doca-bare-metal-deployment`](../doca-bare-metal-deployment/SKILL.md)).
