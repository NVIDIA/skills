# DOCA GPUNetIO ib_write_bw — Tasks

**Where to start:** The verbs that carry real workflow
content are `## configure`, `## build`, `## run`, `## test`,
and `## debug`. The other verbs (`install`, `modify`, `use`)
carry preconditions, refuses-to-patch routing, and the
use-side decision shape respectively.

This file is loaded by [`SKILL.md`](SKILL.md) after
[`CAPABILITIES.md`](CAPABILITIES.md). It walks the agent
through the task verbs every artifact in this bundle exposes
(`install / configure / build / modify / run / test / debug /
use`), then defers task verbs that do not belong here.

## install

Goal: confirm the user's hosts (client AND server) have every
precondition the build + run sequence needs **before** any
GPUNetIO-specific work begins.

This skill does **not** own DOCA installation; that path
lives in [`doca-setup`](../../doca-setup/SKILL.md). The
`gpunetio_ib_write_bw`-specific preconditions:

1. **`doca-gpunetio.pc` is present.** Run
   `pkg-config --modversion doca-gpunetio` on each build
   host. If the `.pc` does not resolve, the installed DOCA
   does not include GPUNetIO; route to
   [`../../libs/doca-gpunetio/TASKS.md`](../../libs/doca-gpunetio/TASKS.md)
   for the package-name lookup and re-install.
2. **`doca-common.pc` and `doca-rdma.pc` agree on the same
   DOCA semver.** Per the four-way match in
   [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility).
3. **CUDA Toolkit + `nvcc` are on the build host.** The
   `client/kernel.cu` is compiled by `nvcc`; confirm
   `nvcc --version` resolves and the toolkit version is
   paired with the installed DOCA per the DOCA release
   notes (looked up via
   [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)).
4. **`nvidia_peermem` kernel module is loaded.** GPUDirect-
   style memory registration is the precondition for
   binding GPU memory to DOCA. Verify per
   [`../../libs/doca-gpunetio/CAPABILITIES.md#safety-policy`](../../libs/doca-gpunetio/CAPABILITIES.md#safety-policy)
   and
   [`doca-setup TASKS.md ## debug`](../../doca-setup/TASKS.md#debug).
5. **GPU visible to the host.** `nvidia-smi` lists the GPU
   and reports its PCIe bus ID.
6. **IB device visible to DOCA.** `doca_caps --list-devs`
   (per [`../doca-caps/TASKS.md#run`](../doca-caps/TASKS.md#run))
   reports the IB device for `-d <ibdev>`.
7. **OOB connectivity exists between client and server**
   for the TCP exchange the source's `oob_connection_*`
   helpers drive.

If any precondition fails, stop and route; tool-level
diagnosis against a half-installed environment wastes time.

## configure

Goal: pick the right GPU + IB device pair on each host,
commit to the runtime-surface choice, and prepare the build
tree.

Steps the agent walks the user through, in order:

1. **Commit to the runtime surface.** Walk
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
   surface-selection table. If the application has not
   committed to GPUNetIO, surface the GPI alternative and
   the CPU-initiated alternative; do not silently default
   to GPUNetIO.
2. **Identify the GPU + IB device pair and check the
   pairing.** On each host: `nvidia-smi
   --query-gpu=pci.bus_id` for the GPU,
   `ibdev2netdev -v` (or `doca_caps --list-devs`) for the
   IB device. Confirm the pairing per
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
   GPU-NIC pairing rule. The pairing is a precondition, not
   a tunable parameter.
3. **Confirm `nvidia_peermem`.** Without it, the doca-gpu
   context cannot bind GPU memory; the symptom shows up
   as a layer-4 lifecycle failure per
   [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy).
4. **Pick the GID index.** `--gid-index` is optional but
   the right value is GID-routing-rule-specific per
   [`../../libs/doca-rdma/CAPABILITIES.md`](../../libs/doca-rdma/CAPABILITIES.md).
   The same index must be used on both client and server.
5. **Pick the client / server roles and the OOB IP.** The
   server is started first (binds and waits on the TCP
   socket); the client is started with `-c <server-ip>`.
6. **Confirm the build inputs.** `PKG_CONFIG_PATH` includes
   the install's `pkgconfig` directory; the agent does not
   invent the literal path.

For the canonical DOCA universal lifecycle that underlies
program-side configuration (which the binary runs internally
per the GPUNetIO + RDMA libraries), see
[`doca-programming-guide TASKS.md ## configure`](../../doca-programming-guide/TASKS.md#configure).

## build

The tool is **not pre-built** under `/opt/mellanox/doca/`;
the user builds it from source under
`doca/tools/gpunetio_ib_write_bw/` against the installed
DOCA. The build pattern is the canonical `meson` flow:

1. **Set the right `PKG_CONFIG_PATH`** so `pkg-config` can
   find `doca-gpunetio.pc`, `doca-rdma.pc`, and
   `doca-common.pc`. On a stock install the path lives
   under the DOCA `pkgconfig` directory documented in
   [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).
2. **Set up the build directory.** `meson setup <build-dir>
   doca/tools/gpunetio_ib_write_bw/` from the workspace
   root. The top-level `meson.build` wires together the
   `client/` and `server/` subtrees; each carries its own
   `meson.build` and source files.
3. **Compile.** `meson compile -C <build-dir>` produces
   the client and server binaries (the target names are
   declared in the per-subtree `meson.build` files; the
   agent re-reads them rather than quoting from memory).
4. **Smoke the build artifacts.** Run `<client-binary>
   --help` and `<server-binary> --help` on each host before
   deploying. If `--help` does not resolve, the build did
   not produce the expected artifacts — re-route to
   [`## debug`](#debug) layer 2.

Routing for nearby "build" questions:

- *"Can I build the tool against a different DOCA than the
  one I have installed?"* → no, not through this skill;
  install a different DOCA first.
- *"I want to build my own GPUNetIO-based BW benchmark
  from scratch."* → route to
  [`../../libs/doca-gpunetio/TASKS.md`](../../libs/doca-gpunetio/TASKS.md)
  and
  [`doca-programming-guide TASKS.md ## build`](../../doca-programming-guide/TASKS.md#build).

The `## What this skill deliberately does not ship` block in
[`SKILL.md`](SKILL.md) explicitly forbids adding a verbatim
build recipe; revisit that policy before changing this
section.

## modify

**Do not patch the shipped tool source tree.** The shipped
`client/{main.c,common.c/h,kernel.cu,perftest.c}` and
`server/{main.c,common.c/h,perftest.c}` files are the
verified worked example for this benchmark class; modifying
them puts the user in contributor-to-DOCA territory.

What the agent *does* modify is the **build + invocation
environment** — the `PKG_CONFIG_PATH`, the chosen GPU + IB
device pair, the GID index, the client / server roles, the
OOB IP, the run-time environment variables
(`DOCA_LOG_LEVEL`, `CUDA_VISIBLE_DEVICES`, NUMA pinning
via `numactl`). Treat *"modify the environment, not the
source"* as the operating mode.

Routing for nearby "modify" questions:

- *"The reported columns are inconvenient — can I change
  them?"* → source-level change; out of scope for this
  skill.
- *"I want to add a new message-size sweep."* → out of
  scope; this would be a contribution to the shipped tool.
- *"I need a different metric than this tool reports."* →
  re-examine the runtime-surface choice in
  [`## configure`](#configure) step 1; if the user genuinely
  needs a bespoke benchmark, route to
  [`doca-programming-guide`](../../doca-programming-guide/SKILL.md)
  and the matching library skill.

## run

The smoke-before-bulk flow — every session goes through it.
The detailed flag surface lives in the binary's `--help` on
the installed build.

> **Do-not-invent guard (paths + binary layout).** Real
> downstream agents have hallucinated a
> `/opt/mellanox/doca/samples/gpunetio/` subtree for this
> tool — it does not exist. The bundle's verbatim source path
> is `/opt/mellanox/doca/tools/gpunetio_ib_write_bw/{client/,server/}`
> (per [`SKILL.md`](SKILL.md) compatibility block); discover
> with `ls /opt/mellanox/doca/tools/ | grep gpunetio_ib_write_bw`,
> NOT under `/opt/mellanox/doca/samples/`. CUDA pairing rules
> live in
> [`../../libs/doca-gpunetio/CAPABILITIES.md ## Version compatibility`](../../libs/doca-gpunetio/CAPABILITIES.md#version-compatibility);
> there is no `cuda-toolkit` skill in this bundle.

1. **Confirm the build artifacts and the environment.** Per
   [`## install`](#install) and [`## configure`](#configure).
2. **Bring up the server first.** On the host that hosts
   the remote buffer, run the `server` binary without `-c`.
   The server listens on the OOB TCP socket.
3. **Confirm the server's pre-run echo.** The tool logs
   the IB device name, the GPU PCIe address, the GID
   index, and the role. If the echo does not match intent,
   stop now; the client's connect will pin the wrong
   pairing.
4. **Bring up the client.** On the second host, run the
   `client` binary with `-c <server-ip>`, the chosen `-d`
   and `--gpu` on the client side, and the same
   `--gid-index` if a non-default was used on the server.
5. **Read the single smoke output.** The result line
   reports sustained throughput at the configured message
   size and iteration count per
   [`CAPABILITIES.md ## Observability`](CAPABILITIES.md#observability).
   Verify the number is in a defensible order of magnitude
   for the GPU-NIC pair and the link's documented
   capacity. If anything looks off, loop back to
   [`## debug`](#debug).
6. **Decompose the throughput** per
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
   throughput-decomposition table. Name the binding
   constraint (GPU compute occupancy / NIC issue rate /
   link saturation / wrong pairing) BEFORE quoting the
   number.
7. **Plan the bulk / swept run** only after the smoke is
   green. The tool's message-size and iteration surface
   lives in the binary's `--help`; the agent does not
   invent sweep flags.

When recording the run for downstream consumers, write
down: `pkg-config --modversion doca-gpunetio`,
`nvcc --version`, the host OS / kernel / NUMA topology /
firmware on each side, the GPU model and PCIe address on
each side, the IB device model and PCIe address on each
side, the GID index, the exact client and server command
lines, the binding constraint identified in the
decomposition, and the full unredacted stdout for both
halves. The downstream
[`## test`](#test) and [`## debug`](#debug) workflows
depend on those fields.

## test

`gpunetio_ib_write_bw` is a **measurement tool**, so its
`## test` verb is about *testing the measurement* —
confirming the numbers are sound and reproducible — not
unit-testing the tool.

**`## test` is an iterative loop.** A run that completes is
not the same as a run that produced a defensible number;
each iteration tightens one axis of measurement soundness.

The eval-loop overlay:

| Iteration trigger | What it looks like | What changes next iteration |
| --- | --- | --- |
| Smoke completes; BW is well below link capacity | One of three candidates (or pairing) — name which before iterating | Re-walk the throughput decomposition in [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes); confirm `nvidia-smi dmon` shows the SM at occupancy; confirm `ibstat` shows the link at expected rate. |
| BW swings between runs by > X% | Steady-state not reached; system not at idle | Lengthen the run (more iterations); confirm no background traffic on the link; confirm no concurrent CUDA workloads. |
| BW grows with message size but does not move with kernel-side concurrency | NIC issue rate is the binding constraint | Quote the NIC-issue-rate hypothesis; confirm against the device's documented per-transport submission rate. |
| BW grows with kernel-side concurrency but does not move with message size | GPU compute occupancy is the binding constraint | Re-walk the persistent-kernel pattern in [`../../libs/doca-gpunetio/CAPABILITIES.md#capabilities-and-modes`](../../libs/doca-gpunetio/CAPABILITIES.md#capabilities-and-modes). |
| BW does not move with either knob and sits well below link capacity | Likely PCIe crossover (wrong pairing) | Re-walk the GPU-NIC pairing precondition; the answer is platform-side, not benchmark-side. |
| Same invocation produces different numbers across hosts at the same DOCA version | NUMA / firmware / driver delta below DOCA | Capture the tuple on both hosts; route through [`doca-version TASKS.md ## test`](../../doca-version/TASKS.md#test) and [`doca-setup TASKS.md ## debug`](../../doca-setup/TASKS.md#debug). |
| Same invocation produces different numbers on the same host across DOCA versions | This is a regression signal — provided both tuples are captured | Cross-link both baselines, name the changed fields, route to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug). |

The agent's rule: every change to the environment re-opens
the loop. Re-running with a different GPU, NUMA pinning, or
firmware level without re-walking the decomposition is
exactly the failure mode this loop replaces.

**Baseline-capture rule.** The captured artifact includes
the multi-axis tuple per
[`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
alongside the stdout from both halves AND the named binding
constraint. Without all of them, the baseline cannot be
regression-tested later.

## debug

When `gpunetio_ib_write_bw` fails to build, bring up, or
produce defensible numbers, walk the
[`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
layers in order:

1. **Config-syntax.** Confirm flags exist in `--help` on
   the installed binaries; do not infer.
2. **Build-time.** Re-route through
   [`## build`](#build) and the GPUNetIO library skill's
   install verification. Common cause: missing
   `doca-gpunetio.pc`, missing `nvcc`, GCC / GLIBC
   mismatch.
3. **GPU-NIC pairing.** Re-walk
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
   GPU-NIC pairing rule; if the pairing is wrong, the fix
   is platform-side.
4. **GPUNetIO-lifecycle.** Most common: `nvidia_peermem`
   not loaded; CUDA buffer registration with DOCA happened
   *after* `doca_ctx_start()` instead of before. Route to
   [`../../libs/doca-gpunetio/CAPABILITIES.md#error-taxonomy`](../../libs/doca-gpunetio/CAPABILITIES.md#error-taxonomy).
5. **RDMA-connection.** Confirm GID index matches on
   client and server; confirm RDMA permissions include
   WRITE; confirm the mmap export was accepted. Route to
   [`../../libs/doca-rdma/CAPABILITIES.md`](../../libs/doca-rdma/CAPABILITIES.md).
6. **Measurement-soundness.** Walk the
   [`## test`](#test) eval loop; confirm warm-up applied;
   name the binding constraint.
7. **Version.** Cross-cutting partial-install / mixed-
   version. Walk
   [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug);
   confirm same DOCA + CUDA pair on both build hosts.
8. **Cross-cutting.** Cause is below DOCA. Hand off to
   [`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug)
   and
   [`doca-setup TASKS.md ## debug`](../../doca-setup/TASKS.md#debug).

In every case: **quote what the binaries reported.** Do
not paraphrase, do not reorder columns, do not summarize a
sweep into a single number.

## use

Goal: turn a captured throughput from this benchmark into a
class-of-workload decision — *"is the GPUNetIO path the
right runtime surface for my sustained-throughput
workload?"*.

The decision shape this skill teaches:

1. **Quote the BW alongside the binding constraint.** *"X
   Gbit/s, NIC-issue-rate-bound"* is a defensible quote;
   *"X Gbit/s"* alone is not.
2. **Compare against the right alternative.** If the
   workload class is *"GPU-initiated WRITE for sustained
   throughput"*, the alternative this benchmark answers is
   *"the same pattern on the CPU-initiated `perftest`
   path"* (data from the upstream tool, captured
   separately); the agent does not synthesize the
   CPU-side number.
3. **Apply the GPU-NIC pairing precondition to the
   downstream design.** A throughput that wins on this
   benchmark only carries over to production if the
   production GPU-NIC pair sits on the same PCIe / NVLink
   topology as the test bed.
4. **Per-release re-verification.** Every DOCA upgrade —
   and every CUDA Toolkit upgrade — requires re-running
   this benchmark before re-quoting the number. The agent
   does not assume a known-good number survives a
   DOCA-version bump or a CUDA-Toolkit bump.
5. **Hand off to the application's own benchmark for the
   final answer.** This tool measures the throughput of a
   single RDMA WRITE stream through GPUNetIO; the user's
   application sits *above* one WRITE stream and has its
   own bottlenecks. The right final answer for *"will my
   application sustain X Gbit/s in production"* is an
   application-level benchmark.

## Deferred task verbs

The verbs below are not `gpunetio_ib_write_bw` work and
should be routed out:

- **install DOCA** ⇒ [`doca-setup TASKS.md`](../../doca-setup/TASKS.md)
  (and `## no-install` for the NGC container path).
- **author a bespoke GPUNetIO-based BW benchmark** ⇒
  [`../../libs/doca-gpunetio/TASKS.md`](../../libs/doca-gpunetio/TASKS.md)
  and
  [`doca-programming-guide TASKS.md ## build`](../../doca-programming-guide/TASKS.md#build).
- **CPU-initiated WRITE BW** ⇒ upstream `perftest`
  `ib_write_bw` (not in this bundle); route via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).
- **GPU-initiated WRITE latency** ⇒
  [`../doca-gpunetio-ib-write-lat/SKILL.md`](../doca-gpunetio-ib-write-lat/SKILL.md)
  for the GPUNetIO latency analog; for the GPI programming
  surface (no shipped GPI benchmark binary) ⇒
  [`doca-gpi`](../../libs/doca-gpi/SKILL.md).
- **hardware-touching changes the benchmark surfaced a
  need for** (NIC firmware burn, BFB reflash, kernel
  command-line changes for IOMMU mode, hugepage
  reservation changes) ⇒
  [`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md).
