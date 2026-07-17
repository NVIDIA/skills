# DOCA GPUNetIO ib_write_lat — Tasks

**Where to start:** The verbs that carry real workflow
content are `## configure`, `## build`, `## run`, `## test`,
and `## debug`. The other verbs (`install`, `modify`, `use`)
carry preconditions, refuses-to-patch routing, and the
use-side decision shape for a real-time / control-loop
workload class.

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
`gpunetio_ib_write_lat`-specific preconditions:

1. **`doca-gpunetio.pc` is present.** Run
   `pkg-config --modversion doca-gpunetio`. If the `.pc`
   does not resolve, route to
   [`../../libs/doca-gpunetio/TASKS.md`](../../libs/doca-gpunetio/TASKS.md).
2. **`doca-common.pc` and `doca-rdma.pc` agree on the same
   DOCA semver** per the four-way match in
   [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility).
3. **CUDA Toolkit + `nvcc` are on the build host.** The
   shipped `common/kernel.cu` is compiled by `nvcc`;
   confirm the toolkit version is paired with the
   installed DOCA per the DOCA release notes (looked up
   via [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)).
4. **`nvidia_peermem` kernel module is loaded.** Verify
   per
   [`../../libs/doca-gpunetio/CAPABILITIES.md#safety-policy`](../../libs/doca-gpunetio/CAPABILITIES.md#safety-policy)
   and
   [`doca-setup TASKS.md ## debug`](../../doca-setup/TASKS.md#debug).
5. **GPU visible to the host.** `nvidia-smi` lists the
   GPU and reports its PCIe bus ID.
6. **IB device visible to DOCA.** `doca_caps --list-devs`
   reports the IB device for `-d <ibdev>`.
7. **OOB connectivity exists between client and server**
   for the TCP exchange the source's connection helpers
   drive.

If any precondition fails, stop and route; tool-level
diagnosis against a half-installed environment wastes time.

## configure

Goal: pick the right GPU + IB device pair on each host,
commit to the runtime-surface choice, and prepare the build
tree.

Steps the agent walks the user through, in order:

1. **Commit to the runtime surface.** Walk
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
   surface-selection table. The GPUNetIO vs GPI vs
   CPU-initiated choice is the load-bearing decision; do
   not silently default to GPUNetIO.
2. **Identify the GPU + IB device pair and check the
   pairing** on each host. `nvidia-smi --query-gpu=pci.bus_id`
   for the GPU; `ibdev2netdev -v` (or
   `doca_caps --list-devs`) for the IB device. The
   pairing is a hardware-topology precondition.
3. **Confirm `nvidia_peermem`** per
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes).
4. **Pick the GID index.** `--gid-index` is optional; the
   same index must be used on both client and server.
   GID-routing rules live in
   [`../../libs/doca-rdma/CAPABILITIES.md`](../../libs/doca-rdma/CAPABILITIES.md).
5. **Pick the client / server roles and the OOB IP.** The
   server is started first; the client is started with
   `-c <server-ip>`.
6. **Pick the workload's latency statistic UP FRONT.**
   Median, p99, p99.9, or jitter — per
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
   median-vs-p99-vs-jitter rule. The agent does NOT pick
   the statistic at quote time; the user picks it
   alongside the workload class at configure time.
7. **Confirm the build inputs.** `PKG_CONFIG_PATH` includes
   the install's `pkgconfig` directory per
   [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

For the canonical DOCA universal lifecycle that underlies
program-side configuration, see
[`doca-programming-guide TASKS.md ## configure`](../../doca-programming-guide/TASKS.md#configure).

## build

The tool is **not pre-built** under `/opt/mellanox/doca/`;
the user builds it from source under
`doca/tools/gpunetio_ib_write_lat/` against the installed
DOCA. The build pattern is the canonical `meson` flow:

1. **Set the right `PKG_CONFIG_PATH`** so `pkg-config`
   can find `doca-gpunetio.pc`, `doca-rdma.pc`, and
   `doca-common.pc`. The path lives under the DOCA
   `pkgconfig` directory documented in
   [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).
2. **Set up the build directory.** `meson setup
   <build-dir> doca/tools/gpunetio_ib_write_lat/` from
   the workspace root. The top-level `meson.build` wires
   together the `client/`, `server/`, and `common/`
   subtrees; the shared `common/` subtree carries the
   CUDA kernel (`kernel.cu`) and the OOB / lifecycle
   helpers shared by both halves.
3. **Compile.** `meson compile -C <build-dir>` produces
   the client and server binaries (target names declared
   in the per-subtree `meson.build`; re-read on the
   user's install).
4. **Smoke the build artifacts.** Run `<client-binary>
   --help` and `<server-binary> --help` before deploying.

Routing for nearby "build" questions:

- *"Can I build against a different DOCA than the one I
  have installed?"* → no; install the target DOCA first.
- *"I want to build my own GPUNetIO-based latency
  benchmark."* → route to
  [`../../libs/doca-gpunetio/TASKS.md`](../../libs/doca-gpunetio/TASKS.md)
  and
  [`doca-programming-guide TASKS.md ## build`](../../doca-programming-guide/TASKS.md#build).

The `## What this skill deliberately does not ship` block in
[`SKILL.md`](SKILL.md) forbids adding a verbatim build
recipe.

## modify

**Do not patch the shipped tool source tree.** The shipped
`client/`, `server/`, and `common/` subtrees are the
verified worked example. Modifying them puts the user in
contributor-to-DOCA territory.

What the agent *does* modify is the **build + invocation
environment** — `PKG_CONFIG_PATH`, the GPU + IB device
pair, the GID index, the client / server roles, the OOB
IP, the run-time environment variables (`DOCA_LOG_LEVEL`,
`CUDA_VISIBLE_DEVICES`), and NUMA pinning (`numactl`).
Treat *"modify the environment, not the source"* as the
operating mode.

Routing for nearby "modify" questions:

- *"The reported columns are inconvenient — can I change
  them?"* → source-level change; out of scope.
- *"I want to change the per-iteration timeout."* → the
  timeout is a kernel-side parameter the shipped source
  controls per `common/common.h`; changing it is a
  source-level change. If the user genuinely needs a
  different timeout, the right answer is *"author a
  bespoke benchmark against the GPUNetIO library"* per
  [`../../libs/doca-gpunetio/TASKS.md`](../../libs/doca-gpunetio/TASKS.md).
- *"I need a different metric than this tool reports."* →
  re-examine the runtime-surface choice in
  [`## configure`](#configure) step 1.

## run

> **Do-not-invent guard (paths).** Real downstream agents
> have hallucinated a `/opt/mellanox/doca/samples/gpunetio/`
> subtree for this tool — it does not exist. The bundle's
> verbatim source path is
> `/opt/mellanox/doca/tools/gpunetio_ib_write_lat/{client/,server/}`
> (per [`SKILL.md`](SKILL.md) compatibility block); discover
> with `ls /opt/mellanox/doca/tools/ | grep gpunetio_ib_write_lat`,
> NOT under `/opt/mellanox/doca/samples/`. CUDA pairing rules
> live in
> [`../../libs/doca-gpunetio/CAPABILITIES.md ## Version compatibility`](../../libs/doca-gpunetio/CAPABILITIES.md#version-compatibility);
> there is no `cuda-toolkit` skill in this bundle.

The smoke-before-bulk flow:

1. **Confirm the build artifacts and the environment**
   per [`## install`](#install) and
   [`## configure`](#configure).
2. **Bring up the server first.** On the chosen host,
   run the `server` binary without `-c`. The server
   listens on the OOB TCP socket.
3. **Confirm the server's pre-run echo.** The tool logs
   the IB device name, the GPU PCIe address, the GID
   index, and the role.
4. **Bring up the client.** On the second host, run the
   `client` binary with `-c <server-ip>`, the chosen
   `-d` and `--gpu` on the client side, and the same
   `--gid-index` if a non-default was used on the
   server.
5. **Read the single-iteration smoke output.** The result
   line carries `#bytes`, `#iterations`, `t_half_iter`,
   `t_full_iter`, `t_cuda` per
   [`CAPABILITIES.md ## Observability`](CAPABILITIES.md#observability).
   Verify the numbers are in a defensible order of
   magnitude for the GPU-NIC pair. If anything looks off,
   loop back to [`## debug`](#debug).
6. **For real-time / control-loop characterization,
   plan a long-iteration run.** A meaningful p99 / p99.9
   needs enough iterations to populate the tail (typical
   rule of thumb: at least 10^5 iterations for p99, 10^6
   for p99.9; the exact threshold is workload-class-
   specific). The agent surfaces this requirement to
   the user before they ask for a *"quick latency
   test"* that would not be statistically meaningful.
7. **Capture all iterations.** The verified `common.h`
   defines `NUM_ITER` constants and exposes a
   `t_cuda`-side measurement per iteration; the agent's
   rule for a real-time decision is *"capture the full
   distribution, not just the summary line"*.

When recording the run for downstream consumers, write
down: `pkg-config --modversion doca-gpunetio`,
`nvcc --version`, host OS / kernel / NUMA topology /
firmware on each side, GPU model + PCIe address on each
side, IB device model + PCIe address on each side, GID
index, exact client and server command lines, the chosen
latency statistic (median / p99 / p99.9 / jitter), and
the full unredacted stdout for both halves.

## test

`gpunetio_ib_write_lat` is a **measurement tool**, so its
`## test` verb is about *testing the measurement* —
confirming the numbers are sound and reproducible — not
unit-testing the tool.

**`## test` is an iterative loop.** A run that completes
is not the same as a run that produced a defensible
distribution; each iteration tightens one axis of
measurement soundness.

The eval-loop overlay:

| Iteration trigger | What it looks like | What changes next iteration |
| --- | --- | --- |
| Smoke completes; median is far below the NIC's documented one-way latency floor | Likely a measurement artifact — possibly the CUDA-side timer is reporting wall time that does not include the full WR-completion observation | Cross-check the CUDA-side number against the host-side `t_full_iter`; re-walk the warm-up rule. |
| Median fine, but p99 is many multiples of the median | Tail latency is real — this IS the answer the operator is looking for if the workload is real-time | Quote the p99 as the answer; capture the tail-event count alongside; do NOT average it away. |
| Iteration count reported is lower than configured | The per-iteration timeout was set too short and iterations were dropped per `common.h`'s `timeout_ns` argument | Lengthen the timeout in the run config; re-run; the dropped iterations are themselves a signal, not noise. |
| Same invocation produces different distributions across two hosts at the same DOCA version | NUMA / firmware / driver delta below DOCA | Capture the tuple on both hosts; route through [`doca-version TASKS.md ## test`](../../doca-version/TASKS.md#test) and [`doca-setup TASKS.md ## debug`](../../doca-setup/TASKS.md#debug). |
| Same invocation produces different distributions on the same host across DOCA versions | Regression signal — provided both tuples are captured | Route to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug). |
| Distribution shifts when the operator runs anything else on the GPU concurrently | The benchmark's measurement is corrupted by concurrent SM activity | Re-run with the GPU at idle (`CUDA_VISIBLE_DEVICES` isolation, `nvidia-smi` to confirm no other processes). |

The agent's rule: every change to the environment re-opens
the loop. Re-running with a different GID index, NUMA
pinning, or firmware level without re-walking the
distribution is exactly the failure mode this loop
replaces.

**Baseline-capture rule.** The captured artifact includes
the multi-axis tuple per
[`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
alongside the stdout from both halves AND the chosen
latency statistic (median / p99 / p99.9 / jitter). A
distribution captured without the statistic is half an
artifact.

## debug

When `gpunetio_ib_write_lat` fails to build, bring up, or
produce a defensible distribution, walk the
[`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
layers in order:

1. **Config-syntax.** Confirm flags exist in `--help` on
   the installed binaries.
2. **Build-time.** Re-route through
   [`## build`](#build) and the GPUNetIO library skill's
   install verification.
3. **GPU-NIC pairing.** Re-walk
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
   GPU-NIC pairing rule.
4. **GPUNetIO-lifecycle.** Most common cause:
   `nvidia_peermem` not loaded. Route to
   [`../../libs/doca-gpunetio/CAPABILITIES.md#error-taxonomy`](../../libs/doca-gpunetio/CAPABILITIES.md#error-taxonomy).
5. **RDMA-connection.** Confirm GID index matches;
   confirm RDMA permissions include WRITE; route to
   [`../../libs/doca-rdma/CAPABILITIES.md`](../../libs/doca-rdma/CAPABILITIES.md).
6. **Measurement-soundness.** Walk the
   [`## test`](#test) eval loop; confirm warm-up applied;
   confirm the per-iteration timeout was generous enough;
   confirm the right statistic was quoted.
7. **Version.** Cross-cutting partial-install /
   mixed-version. Walk
   [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug).
8. **Cross-cutting.** Hand off to
   [`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug)
   and
   [`doca-setup TASKS.md ## debug`](../../doca-setup/TASKS.md#debug).

In every case: **quote what the binaries reported.** Do
not paraphrase, do not collapse the distribution into a
single number.

## use

Goal: turn a captured latency distribution into a
class-of-workload decision — *"is the GPUNetIO path the
right runtime surface for my real-time / control-loop
workload?"*.

The decision shape this skill teaches:

1. **Quote the right statistic.** Median for typical-case,
   p99 / p99.9 for deadline-bound real-time, jitter for
   predictability-bound. Quoting a single number without
   naming the statistic is the cross-tool comparison
   failure mode.
2. **Compare against the right alternative.** If the
   workload class is *"GPU-initiated WRITE for a real-time
   control loop"*, the alternative this benchmark
   answers is *"the same pattern on the GPI surface"*
   per the [`doca-gpi`](../../libs/doca-gpi/SKILL.md)
   library (there is no shipped GPI `ib_write_lat`
   benchmark binary in `doca/tools/`);
   if the alternative is *"the same pattern on the host
   CPU"*, the comparison data has to come from the
   upstream CPU-initiated `perftest` `ib_write_lat`
   separately.
3. **Apply the GPU-NIC pairing precondition to the
   downstream design.** A latency that wins on this
   benchmark only carries over to production if the
   production GPU-NIC pair sits on the same PCIe /
   NVLink topology as the test bed.
4. **Account for the latency-vs-batching trade-off.**
   Per
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
   latency-vs-batching trade-off, the per-WR latency this
   benchmark reports reflects the shipped batching
   configuration. The user's production code may batch
   differently; the right *final* answer for a real-time
   deadline comes from an application-level test that
   matches the production batching shape.
5. **Per-release re-verification.** Every DOCA upgrade
   and every CUDA Toolkit upgrade requires re-running
   the benchmark before re-quoting the distribution.
6. **Hand off to the application's own benchmark for the
   real-time deadline.** This tool measures the WR
   latency through GPUNetIO; the user's full control
   loop sits *above* one WR and has its own latency
   contributors (compute on the GPU between WRs,
   host-side handoffs, scheduling jitter). The right
   *final* answer for a real-time deadline is an
   application-level benchmark.

## Deferred task verbs

The verbs below are not `gpunetio_ib_write_lat` work and
should be routed out:

- **install DOCA** ⇒ [`doca-setup TASKS.md`](../../doca-setup/TASKS.md).
- **author a bespoke GPUNetIO-based latency benchmark** ⇒
  [`../../libs/doca-gpunetio/TASKS.md`](../../libs/doca-gpunetio/TASKS.md).
- **CPU-initiated WRITE latency** ⇒ upstream `perftest`
  `ib_write_lat` (not in this bundle); route via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).
- **GPU-initiated WRITE bandwidth** ⇒
  [`../doca-gpunetio-ib-write-bw/SKILL.md`](../doca-gpunetio-ib-write-bw/SKILL.md).
- **GPI programming surface (same physical operation,
  different runtime framework; no shipped GPI benchmark
  binary)** ⇒ [`doca-gpi`](../../libs/doca-gpi/SKILL.md).
- **hardware-touching changes the benchmark surfaced a
  need for** ⇒
  [`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md).
