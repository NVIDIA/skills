# DOCA GPI workflows

**Where to start:** The verbs run `install → configure → build →
modify → run → test → debug → use`. Skip ahead only when the user
is already past a verb. The `## test` verb is an iterative loop
(cap-query check → lifecycle order → descriptor exchange → single-
work-request smoke → loop back if any check fails), not a
one-shot pass — see the eval-loop overlay in `## test` below.

Read this file when the loader sent you here from
[SKILL.md](SKILL.md). For the underlying object model, version
compatibility, error taxonomy, observability surface, and safety
policy that these workflows assume, see
[CAPABILITIES.md](CAPABILITIES.md). For where to find docs, the
installed DOCA layout, or release notes, route through
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

Each verb below describes the **shape of the workflow**, not a
copy-paste recipe. The agent's job is to walk the user through
the steps in order, verifying preconditions before recommending
the next call.

## install

Goal: confirm the user's installed DOCA actually ships `doca-gpi`
plus a compatible CUDA Toolkit before any GPI-specific work
begins.

This skill does **not** own DOCA installation; that path lives in
[`doca-setup`](../../doca-setup/SKILL.md). The GPI-specific
preconditions the agent verifies after a DOCA install:

1. **`doca-gpi` `.pc` file is present.** `pkg-config
   --modversion doca-gpi` resolves and reports a semver matching
   `doca_caps --version`. If it does not resolve, the installed
   DOCA package set does not include GPI; the user needs to
   install the matching package (the exact package name is
   platform-specific and looked up via
   [`doca-public-knowledge-map ## Layout of an installed DOCA package`](../../doca-public-knowledge-map/SKILL.md#layout-of-an-installed-doca-package)).
2. **Supporting `.pc` files are present.** GPI's DOCA
   dependencies (`dependencies/meson.build`) are `doca-dpa`,
   `doca-gpunetio`, and `doca-verbs`. All of their `pkg-config
   --modversion` results must agree with `doca-gpi` on the same
   DOCA semver per the four-way match in
   [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility).
3. **Installed header exposes the symbols.** Check
   `doca_gpi.h` resolves under the installed DOCA infrastructure
   include tree. If the `.pc` resolves but the header is missing,
   the install is partial; do not attempt to build until the
   install is repaired.
4. **CUDA Toolkit is installed and compatible.** GPI's GPU-side
   handle is consumed by `nvcc`-compiled kernels; the user must
   have a CUDA Toolkit version paired with the installed DOCA
   release per the release notes (looked up via
   [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)).
   The agent does NOT quote a specific CUDA version from agent
   memory; the user reads the pairing from the DOCA release
   notes that match `doca_caps --version`.
5. **The host actually has an NVIDIA GPU reachable through
   GPUDirect-style memory mapping.** A host without a GPU on the
   PCIe topology cannot use GPI regardless of how cleanly the
   DOCA side installs. Verify via `nvidia-smi` (or the
   user's equivalent GPU-side probe). The agent does not
   prescribe specific GPU drivers; routing for those goes
   through [`doca-setup TASKS.md ## debug`](../../doca-setup/TASKS.md#debug)
   layer 5 (driver).

If any precondition fails, **stop and route to
[`doca-setup`](../../doca-setup/SKILL.md)**; a GPI-layer
diagnosis against a half-installed DOCA or a missing CUDA Toolkit
wastes the user's time.

## configure

Goal: bring up a `doca_gpi` context, size its channels and
queues against the device's reported capabilities, and reach the
state where the GPU-side handle is valid.

Steps the agent should walk the user through:

1. **Confirm the installed DOCA version + CUDA version.** Use
   the procedure in
   [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility).
   Quote the DOCA version (`pkg-config --modversion doca-gpi`,
   `doca_caps --version`) and the CUDA version (`nvidia-smi`
   for the driver-reported toolkit, `nvcc --version` for the
   build-time toolkit); do not assume "latest".
2. **Identify GPU-datapath-capable devices.** Run `doca_caps
   --list-devs` ([`doca-caps`](../../tools/doca-caps/SKILL.md))
   to see which devices have the GPU-datapath capability. Note
   that `doca_gpi.h` exposes **no** `doca_gpi_cap_*` devinfo
   query — GPI has no runtime capability API — so the agent does
   not invent per-device maxima; supported sizing ranges come
   from the device and the DOCA release notes.
3. **Pick the sizing.** Choose the domain channel count
   (`doca_gpi_domain_attr_set_num_channels`), endpoint / bind
   counts (`_set_num_ep`, `_set_num_binds`, `_set_bind_size`),
   and per-channel work-queue depths
   (`doca_gpi_channel_attr_set_sq_wqe_num`, `_set_srq_wqe_num`,
   `_set_gpu_wqe_num`). The agent does not invent values; if a
   value is out of range the create call returns a
   `DOCA_ERROR_*` rather than a cap-query rejection.
4. **Create and configure the GPI instance.** Call
   `doca_gpi_create(dev, &gpi)`, then set the instance attributes
   (`doca_gpi_set_num_domains`, `doca_gpi_set_gid_index`,
   `doca_gpi_set_port_num`, `doca_gpi_set_enable_err_monitor`)
   **before** `doca_gpi_start` — the header states start "must be
   called after setting all the GPI attributes". Where the
   application drives the GPU datapath, do the
   [`doca-gpunetio`](../doca-gpunetio/SKILL.md) setup here too.
5. **Start, then create the domain and attach memory.** Call
   `doca_gpi_start(gpi)`. Build a domain with
   `doca_gpi_domain_attr_create` + the sizing setters above +
   `doca_gpi_domain_create(gpi, attr, &domain)`, then attach each
   GPU-reachable region with
   `doca_gpi_domain_attach_local_mmap(domain, mmap, &bind_id)`
   (and `doca_gpi_domain_attach_remote_mmap` for a peer's
   `doca_mmap` exchanged out of band). The application is
   responsible for creating each `doca_mmap`.
6. **Create the channel.** Build a channel with
   `doca_gpi_channel_attr_create` + the WQE-depth setters +
   `doca_gpi_channel_create(gpu_dev, domain, attr, &channel)`.
   Note the GPU device is the first argument.
7. **Retrieve the GPU handle.** Call
   `doca_gpi_gpu_channel_get(channel, &gpu_channel)` to obtain
   the `struct doca_gpu_gpi_channel*` the CUDA kernel will use.
8. **Connect channel endpoints.** For each endpoint, call
   `doca_gpi_channel_ep_conn_info_create(channel, ep_idx,
   &conn_info_size, &conn_info)` to build the local connection
   info; transport it over the application's out-of-band channel
   to the remote peer; receive the peer's blob back; call
   `doca_gpi_channel_ep_connect(channel, ep_idx, peer_conn_info,
   peer_conn_info_size)`. After this call the endpoint is
   connected and the GPU side can issue work; free each local
   blob with `doca_gpi_channel_ep_conn_info_destroy`.

If any step fails with a `DOCA_ERROR_*`, route through the error
taxonomy in
[CAPABILITIES.md ## Error taxonomy](CAPABILITIES.md#error-taxonomy)
before retrying.

## build

Goal: produce a host-side binary plus a CUDA-side binary that
link DOCA GPI against the user's installed DOCA, using the
canonical cross-library build pattern.

The build pattern for any DOCA C/C++ consumer is **identical**
across libraries — `pkg-config` for include + link flags, meson
or CMake as the build system — and is fully documented in
[`doca-programming-guide TASKS.md ## build`](../../doca-programming-guide/TASKS.md#build).
This skill carries only the GPI-specific overlay:

| Slot | Value for GPI | Why it matters |
| --- | --- | --- |
| `pkg-config` module name (host side) | `doca-gpi` | The library's `.pc` file installed by the DOCA host packages |
| Co-required modules | `doca-gpunetio`, `doca-dpa`, `doca-verbs` | GPI's DOCA dependencies per `dependencies/meson.build`; the GPU-side handle type lives in `doca-gpunetio` and the transport layer is `doca-verbs`, so all must be reachable through `pkg-config` |
| Header check | `doca_gpi.h` resolvable under the installed DOCA infrastructure include tree (path via [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)) | If `pkg-config --cflags doca-gpi` resolves but the include is missing, the install is partial |
| CUDA-side compilation | The CUDA kernel that consumes `struct doca_gpu_gpi_channel*` is compiled with `nvcc`, against the DOCA GPU NetIO device-side header set — routed to [`doca-gpunetio`](../doca-gpunetio/SKILL.md) for the device-side surface | Mixing host and CUDA toolchains on the same translation unit is the canonical reason a CUDA-side build "fails for no reason"; routed to GPU NetIO for the device-side details |
| Minimum required DOCA version | Query with `pkg-config --modversion doca-gpi`; never hardcode in build files | Every `doca_gpi_*` symbol is `DOCA_EXPERIMENTAL`, so a version pin from agent memory is wrong by construction — the whole surface can shift between releases |
| CUDA Toolkit pairing | The DOCA release notes for the installed version name the compatible CUDA Toolkit range | The pairing is a release-notes lookup, not an agent-memory recall |

For non-C consumers (Rust, Go, Python), the host-side link
surface is the same `*.so` files; FFI wrappers are out of scope
for this skill. The CUDA-side surface is not wrappable — the
CUDA kernel is compiled and linked into the application's GPU
binary itself.

## modify

Goal: take an existing GPI-using component (the user's own code,
or a verified DOCA sample if one ships in the installed package
set) as the starting point and apply a minimum-diff modification
to express the new intent.

The universal modify-a-shipped-sample workflow lives in
[`doca-programming-guide TASKS.md ## modify`](../../doca-programming-guide/TASKS.md#modify).
Use it as-is. The GPI-specific overlay is the *five-slot fill*
the agent must elicit from the user before recommending any
code-level edit:

| Slot | What the agent asks the user | GPI-specific consideration |
| --- | --- | --- |
| 1. Starting code | Which GPI-using file or sample is the baseline? | If the user has no working baseline, *stop* — the EXPERIMENTAL periphery of the API surface means authoring GPI source from documentation prose is forbidden by this skill (per [`SKILL.md ## What this skill deliberately does not ship`](SKILL.md#what-this-skill-deliberately-does-not-ship)) |
| 2. Sizing change | Change domain channel count or per-channel work-queue depths? | Channel count is set on `doca_gpi_domain_attr` (`_set_num_channels`); WQE depths on `doca_gpi_channel_attr` (`_set_sq_wqe_num` / `_set_srq_wqe_num` / `_set_gpu_wqe_num`). There is no `doca_gpi_cap_*` query, so an out-of-range value fails at create time, not at a cap check; do not carry a number over from a different device |
| 3. Memory binding | Add or remove a `doca_gpi_domain_attach_local_mmap` / `doca_gpi_domain_attach_remote_mmap` region? | Each attach needs an application-created `doca_mmap`; a remote attach requires the peer's `doca_mmap` exchanged out of band and is not safe to re-export without re-doing that exchange |
| 4. Endpoint connection | Change how endpoint connection info is exchanged with the remote peer? | This is a re-architecture, not a tweak. The conn-info blob from `doca_gpi_channel_ep_conn_info_create` crosses an application-owned out-of-band channel (TCP, file, MPI, …); do not invent a built-in exchange |
| 5. GID / transport selection | Change `doca_gpi_set_gid_index` or `doca_gpi_set_port_num`? | GID / port selection determines which IB / RoCE path the channels use; mismatched GIDs between local and remote are silent — the endpoints connect but the channel does not carry traffic |

The agent emits an *intent description + the five filled slots*;
the actual unified diff against the user's baseline is produced
line-by-line and validated by the user pasting back the result.
Do not author GPI source code that the user did not start from.

## run

Goal: actually execute the built binary (host + CUDA) against
the user's installed DOCA on a host with both BlueField /
ConnectX and an NVIDIA GPU, with a remote responder reachable
on the wire.

Steps the agent should walk the user through:

1. **Confirm the remote peer is reachable.** GPI initiates RDMA
   work; running the binary on one side alone produces a
   misleading hang. The peer must be running an RDMA stack
   (commonly an application using
   [`doca-rdma`](../doca-rdma/SKILL.md) or the upstream verbs
   surface) reachable over IB / RoCE on the chosen GID.
2. **Launch the CUDA kernel.** Load the CUDA binary, allocate
   GPU memory, and launch the kernel that consumes the
   GPU-side channel handle. The kernel launch incantation is
   not GPI's concern; the CUDA programming model is owned by
   [`doca-gpunetio`](../doca-gpunetio/SKILL.md) and the upstream
   CUDA documentation.
3. **Capture the structured log on the host side.** Set
   `DOCA_LOG_LEVEL=trace` for the first run (see
   [`doca-debug CAPABILITIES.md ## Observability`](../../doca-debug/CAPABILITIES.md#observability)).
   This is the cheapest way to make the host-side lifecycle
   visible on first failure.
4. **Observe completions on the CUDA side.** Per
   [`CAPABILITIES.md ## Observability`](CAPABILITIES.md#observability),
   the CUDA kernel is the only thing that observes per-work-
   request completions. The host side will be quiet between
   `doca_gpi_start()` and the application's eventual stop /
   destroy. A run that produces no GPU-side completions but
   doesn't error on the host is almost always (a) the CUDA
   kernel did not launch, (b) the descriptor exchange landed
   wrong, or (c) the remote peer silently disconnected — walk
   all three before diving into GPI internals.

## test

Goal: prove the configured GPI instance can actually have a
CUDA kernel initiate RDMA work correctly on the user's
hardware, and that the sizing / descriptor exchange survived
end-to-end.

This is **a loop, not a one-shot pass.** Each iteration narrows
either the cap-gated sizing, the lifecycle order, the descriptor
exchange, or the GPU handoff. The loop terminates when either
(a) the user's intended GPU-initiated RDMA operation completes
end-to-end with the expected effect on the peer side, or (b) the
agent has narrowed the failure cause to a layer outside GPI
itself (CUDA, RDMA transport, driver, firmware, network) and
escalated to the matching skill.

Iteration shape:

1. **Sizing check.** Confirm every domain / channel attribute
   value the application set is accepted: an out-of-range value
   returns a `DOCA_ERROR_*` from `doca_gpi_domain_create` /
   `doca_gpi_channel_create` at configure time. GPI has no
   `doca_gpi_cap_*` query, so there is no cap to compare against;
   if the user has logged-and-ignored a create error, the GPI
   state is undefined.
2. **Lifecycle-order check.** Walk the configure sequence in
   [`## configure`](#configure) against the user's code: every
   `doca_gpi_set_*` instance attribute must precede
   `doca_gpi_start()`; domain / channel creation and the GPU
   handle retrieval (`doca_gpi_gpu_channel_get`) follow it.
3. **Out-of-band round-trip.** Confirm the remote `doca_mmap`
   (attached via `doca_gpi_domain_attach_remote_mmap`) and the
   endpoint connection-info blob (from
   `doca_gpi_channel_ep_conn_info_create`) were received intact
   by the remote peer (and vice versa). Diff the bytes if the
   peer reports an error; transport bugs are common in
   application-owned out-of-band channels.
4. **Single-work-request smoke.** Before driving any volume,
   have the CUDA kernel post ONE RDMA work request and confirm
   the expected effect on the peer side (the peer's counter
   moves, or the peer's mmap is observably modified). If the
   peer side does not see the effect, stop and walk the
   observability surface in
   [`CAPABILITIES.md ## Observability`](CAPABILITIES.md#observability);
   do not raise traffic into an unobserved path.
5. **Negative test.** Construct one deliberately oversized
   domain / channel attribute value and confirm the create call
   (`doca_gpi_domain_create` / `doca_gpi_channel_create`) returns
   a `DOCA_ERROR_*`. This validates that the agent's sizing
   understanding is itself correct on this DOCA version + this
   device — without relying on a cap query that GPI does not
   expose.

Eval-loop overlay — why this is a loop, not a one-shot pass:

| Iteration trigger | What it looks like | What changes next iteration |
| --- | --- | --- |
| `DOCA_ERROR_*` from `doca_gpi_gpu_channel_get` | The channel was not yet created on a configured, started GPI instance | Re-walk steps 4-7 of [`## configure`](#configure); confirm `doca_gpi_start` and `doca_gpi_channel_create` ran before the handle was requested |
| `DOCA_ERROR_*` from a create call | A domain / channel attribute value is out of range | GPI exposes no cap query, so re-derive the value from the device and release notes against the actual active device; clamp it |
| Endpoint connect silently fails | Both sides ran `doca_gpi_channel_ep_connect` without error; no traffic flows | The conn-info blob or remote `doca_mmap` transported wrong, or the GID indexes don't agree; diff the bytes and re-confirm the GID selection |
| CUDA kernel never observes a completion | The host side is fine; the kernel polls forever | Either the kernel was not launched, or the endpoint is not actually connected end-to-end, or the remote peer rejected the work; walk all three |
| `DOCA_ERROR_IN_USE` from `doca_gpi_destroy` | Domains or channels are still alive | Destroy every channel and domain and detach mmaps before destroy; the destroy does not auto-clean |

Loop termination: stop iterating once two consecutive iterations
of the same kind don't change anything — that means the cause is
below GPI. Escalate to
[`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug)
with the captured layer-1-through-5 evidence.

## debug

Goal: when a DOCA GPI call returns a `DOCA_ERROR_*` (or the
program doesn't make forward progress), narrow the cause to a
specific layer and act on it.

The cross-library debug ladder lives in
[`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug).
Walk through it in order — install → version → build → link →
runtime → program → driver — *before* recommending GPI-specific
fixes. This skill's overlay names the GPI-specific manifestation
at layers 5 (runtime) and 6 (program):

**Layer 5 (runtime) — GPI overlay.**

- Confirm `doca_gpi_start` was called *after* all
  `doca_gpi_set_*` instance attributes. The header states start
  "must be called after setting all the GPI attributes", and
  `doca_gpi_get_dpa` "can be called only if gpi not started";
  the `doca_gpi_set_*` calls cannot run after start.
- Confirm the CUDA kernel actually launched and is consuming
  the GPU-side handle. *No observable completions* is almost
  always a launch / handoff bug, not a GPI-spec bug; route to
  [`doca-gpunetio`](../doca-gpunetio/SKILL.md) for the
  CUDA-side launch verification.
- Confirm the descriptor exchange landed intact on both sides.
  This is application-owned and a common source of silent
  failures; diff the bytes.

**Layer 6 (program) — GPI overlay.**

- Lifecycle order: create → set instance attributes → start →
  create domain → attach mmaps → create channel → get GPU
  handle → connect endpoints → use → destroy channels / domains
  → stop → destroy. Out-of-order returns `DOCA_ERROR_BAD_STATE`
  or `DOCA_ERROR_IN_USE`. Re-check against
  [`## configure`](#configure).
- Single-handle discipline: the GPU handle belongs to exactly
  one channel and one consuming CUDA kernel. Reusing a handle
  across a `doca_gpi_stop` / `doca_gpi_start` restart is
  undefined behavior per
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy).
- Sizing discipline: sizing is set on `doca_gpi_domain_attr` /
  `doca_gpi_channel_attr` objects and GPI exposes no
  `doca_gpi_cap_*` query, so an out-of-range value fails at
  create time. Hand-coded values that survived a previous
  device are not portable to a new device.

Once the layer is identified, route to the matching debug verb
on the matching skill: install / build / link / driver to
[`doca-setup TASKS.md ## debug`](../../doca-setup/TASKS.md#debug);
cross-cutting runtime to
[`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug);
CUDA-side debug to
[`doca-gpunetio TASKS.md ## debug`](../doca-gpunetio/TASKS.md#debug);
program-layer Core-context patterns to
[`doca-programming-guide TASKS.md ## debug`](../../doca-programming-guide/TASKS.md#debug).

## use

Goal: integrate a working GPI component into a larger
application — typically a GPU-resident agent that consumes a
remote peer's memory at line rate without host CPU involvement.

The integration shape this skill teaches:

1. **Per-application init order.** The host-side init order is
   `doca_gpi_create` → set instance attributes
   (`doca_gpi_set_*`) → `doca_gpi_start` → create domain (with
   `doca_gpi_domain_attr` sizing) → attach mmaps → create
   channel (with `doca_gpi_channel_attr` sizing) →
   `doca_gpi_gpu_channel_get` → connect endpoints → CUDA kernel
   launch. The CUDA kernel does not run until the GPU handle and
   the endpoint connections are in place; the host does not stop
   the instance until the CUDA kernel has drained its
   outstanding work.
2. **Per-application teardown order.** Stop the CUDA kernel
   first; destroy every channel (`doca_gpi_channel_destroy`) and
   domain (`doca_gpi_domain_destroy`) and detach mmaps
   (`doca_gpi_domain_detach_mmap`); call `doca_gpi_stop()`; call
   `doca_gpi_destroy()` (which returns `DOCA_ERROR_IN_USE` if a
   domain or channel is still alive). The destroy-then-stop
   ordering is the load-bearing piece.
3. **Multi-channel discipline.** A GPI domain may expose
   multiple channels; each channel may expose multiple
   endpoints; each GPU-side handle is per-channel, not
   per-endpoint. The agent walks the user through which kernel
   consumes which channel so handles don't get crossed.
4. **Operational handoff.** Production deployment uses the
   bundle's hardware-safety meta-policy
   ([`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md))
   for any change that touches the BlueField BFB, firmware,
   IOMMU mode, or host kernel parameters that affect
   GPUDirect-style memory mapping. GPI itself does not modify
   hardware state, but every GPI-using component lives
   downstream of those changes.
5. **Per-release re-verification.** Because the entire API
   surface is `DOCA_EXPERIMENTAL` (no `DOCA_STABLE` subset),
   every DOCA upgrade — and every CUDA Toolkit upgrade —
   requires re-running
   [`## test`](#test) end-to-end against the new install. The
   agent does not assume a known-working integration survives
   a DOCA-version bump or a CUDA-Toolkit bump without
   re-testing.

## Deferred task verbs

The following verbs are out of scope for this skill but are
commonly asked in the same conversations. Route them as follows
so the agent does not invent guidance:

- **install (of DOCA itself).** Installing DOCA, choosing
  packages, post-install verification, `pkg-config` wiring —
  defer to [`doca-setup`](../../doca-setup/SKILL.md) and to the
  install-tree layout in
  [`doca-public-knowledge-map ## Layout of an installed DOCA package`](../../doca-public-knowledge-map/SKILL.md#layout-of-an-installed-doca-package).
  This skill's `## install` verb assumes DOCA is already
  installed and only checks the GPI-specific preconditions.
- **deploy.** Deploying GPI-using applications at scale across
  many hosts, Kubernetes operator workflows, multi-tenant
  RDMA isolation — out of scope and reserved for a future
  platform skill. For single-host first-run testing, the right
  verb is `## run`.
- **rollback.** Coordinated rollback of GPI-using applications
  across multiple hosts — out of scope and reserved for a
  future platform skill. For single in-session rollback, the
  right verb is destroying the GPI context and re-running
  `## configure` with the corrected parameters; do not invent
  a "rollback" workflow.
- **CUDA programming.** The CUDA toolchain, kernel launch,
  stream ordering, and GPU memory model belong to the upstream
  CUDA documentation; the DOCA-side wiring for the CUDA-
  consumed surface belongs to
  [`doca-gpunetio`](../doca-gpunetio/SKILL.md). This skill
  describes the GPI-side handoff (the GPU-side handle, the
  channel-connect call) but does not author CUDA kernels.
- **Kernel-level driver install / firmware burn / IOMMU
  reconfiguration.** Installing the `mlx5_core` driver,
  burning new ConnectX firmware, modifying `mlxconfig`
  parameters, switching IOMMU mode at host boot — out of
  scope. Route to
  [`doca-setup TASKS.md ## debug`](../../doca-setup/TASKS.md#debug)
  layer 5 and to
  [`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md)
  for the change-application discipline.

## Command appendix

Every command below is **cross-cutting on DOCA GPI** — it
answers a recurring class of question that comes up in the verbs
above. The agent should treat the *class* as load-bearing; the
worked example is a single instance.

**Infra-aware preamble (every row below).** Per the bundle's
detect → prefer → fall back → report contract documented in
[`doca-structured-tools-contract ## The agent behavior contract`](../../doca-structured-tools-contract/SKILL.md#the-agent-behavior-contract),
the agent probes for the matching structured helper FIRST
(`doca-env --json` for version + devices + libraries + drivers
in one shot; `doca-capability-snapshot` for per-device
capability flags; `version-matrix.json` for *"available since"*
lookups). If the probe succeeds, the structured tool's output is
the authoritative answer. If the probe fails, fall back to the
manual command in the row.

| Command (worked example) | Owning step | Class of question it answers | What healthy output looks like |
| --- | --- | --- | --- |
| `pkg-config --modversion doca-gpi` | [`## install`](#install) step 1; [`## configure`](#configure) step 1 | What is the build-time DOCA GPI version? | A semver matching `doca_caps --version`. Disagreement = partial install; route to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug) layer 2 |
| `pkg-config --modversion doca-gpi doca-gpunetio doca-dpa doca-verbs` | [`## install`](#install) step 2 | Do `doca-gpi` and its dependencies agree on the same DOCA semver? | A single semver repeated across all four. Any disagreement is the partial-install pattern |
| `pkg-config --cflags --libs doca-gpi` | [`## build`](#build) | What include + link flags does the linker need? | Includes resolve under whichever include directory `pkg-config --cflags` reports on this install (do not hardcode the path); libs include `-ldoca_gpi` alongside its dependencies (`-ldoca_gpunetio`, `-ldoca_dpa`, `-ldoca_verbs`) |
| `doca_caps --list-devs` | [`## install`](#install) step 5; [`## configure`](#configure) step 2 | Which devices on this host can be used as a `doca_dev` with GPU-datapath capability? | One row per visible device with PCIe address and capability flags; the GPU-datapath capability flag is the gate for GPI |
| `nvidia-smi` | [`## install`](#install) step 5 | Does the host have an NVIDIA GPU reachable on the PCIe topology? | One row per visible GPU with driver version, CUDA version, and PCIe address |
| `nvcc --version` | [`## install`](#install) step 4; [`## build`](#build) | What CUDA Toolkit version will compile the GPU-side code? | A version that pairs with the installed DOCA per the release notes |
| `cat /opt/mellanox/doca/applications/VERSION` | [`## install`](#install) step 1; [`## debug`](#debug) layer 1 | What does the install tree itself claim its version is? | A semver matching the other version sources |
| `DOCA_LOG_LEVEL=trace ./<binary>` | [`## run`](#run) step 3 | What did the structured DOCA logger emit for the first failing call? | A trace-level line on every lifecycle transition. Silence after `doca_gpi_start()` = the CUDA kernel was never launched or the host is no longer the right side to observe |
| `dmesg | tail -n 40` (sudo) | [`## debug`](#debug) layer 7 | What did the kernel / driver log around the last GPI call? | Empty or recent benign messages. Repeated mlx5 / GPU-driver errors → driver-layer bug; route to [`doca-setup TASKS.md ## debug`](../../doca-setup/TASKS.md#debug) |

For commands shared across libraries (`pkg-config --modversion`,
`doca_caps`, `cat /opt/mellanox/doca/applications/VERSION`,
`DOCA_LOG_LEVEL`) the cross-library overlay is in
[`doca-debug TASKS.md ## Command appendix`](../../doca-debug/TASKS.md#command-appendix);
this table adds the GPI-specific rows on top.
