---
name: nvshmem-select-remote-transport
description: Select an NVSHMEM remote transport from target system and kernel evidence. Use for inter-node selection, compatibility checks, or configuration.
license: Apache-2.0
metadata:
  version: "1.0.0"
  author: NVIDIA NVSHMEM Team <nvshmem@nvidia.com>
  tags:
    - nvshmem
    - transport
    - tuning
---

# Select an NVSHMEM Remote Transport

## Purpose

Recommend one NVSHMEM transport configuration from static evidence. Separate hard eligibility from expected kernel fit, distinguish documented facts from inference.

Focus on read-only system probes and source inspection. Do not run NVSHMEM applications, correctness tests, or performance benchmarks.

## Prerequisites

- Require an inventory of installed transport plugins before recommending a transport that must be available in the selected installation.
- Record the installed NVSHMEM version when it is readily available, but do not require patch-level version verification for ordinary transport selection or stable settings such as `NVSHMEM_REMOTE_TRANSPORT`.
- Require exact-version evidence only when the decision depends on a release gate, known issue, changed setting, or release-specific operation or memory-mode limitation.
- Use the sibling `$nvshmem-docs` workflow, or its readable `../nvshmem-docs/SKILL.md`, when the user requests current/exact official verification or an exact-version-sensitive decision cannot be resolved from the bundled matrix. Do not invoke it merely because the target patch differs from a source version reviewed by the matrix.
- Require no API key, privileged command, network access for inspection, or benchmark execution.

## Inputs

### Core Recommendation Inputs

- Installed transport-plugin inventory, or explicit user confirmation of the available transport.
- Target fabric/provider: InfiniBand, RoCE, EFA, Slingshot/CXI, UCX-managed, or NVLink-only.
- For kernel/workload analysis only, selected CUDA kernel/application source, or a profile covering:
  - host, on-stream, or device API use;
  - RMA, AMO, signaling, synchronization, or collective operations;
  - typical and range of message sizes;
  - issuing threads, warps, and CTAs;
  - peer fan-out and expected simultaneous message rate.
- Job scale: total PEs and local PEs per node.
- Intended outcome: general default, latency-sensitive, throughput-oriented, or compatibility-first.

### Optional

- Exact NVSHMEM patch version when the decision is not release-sensitive.
- `NVSHMEM_PREFIX`, CMake cache, module/container details, launcher command, environment, and NVSHMEM configuration files.
- GPU, NIC, driver, kernel, OFED/rdma-core, UCX, libfabric, GPUNetIO, or DOCA details not available through inspection.
- GPU-to-NIC topology and local PE binding.
- Host/device traffic mix, CPU availability, persistent-kernel behavior, batching opportunities, and completion frequency.
- Required atomics, registered user buffers, VMM, DCI/DCT, multi-NIC, Spectrum-X, or custom-QP features.
- Preferred or prohibited transports.

Do not re-ask facts the user already supplied. Prefer one best-evidence recommendation over a list of conditional branches. State material assumptions and unknowns, but withhold the selection configuration only when no candidate passes a hard gate or the setting itself cannot be resolved safely.

## Instructions

### 1. Establish the Evidence Source

First establish whether the agent runs on the target system; ask unless the user already answered or supplied a complete target-system probe. Never infer this from the shell, scheduler, hostname, hardware, or repository.

If yes, inspect the complete bundled [collector script](scripts/collect-transport-facts.sh) before its first invocation in the current checkout. Run only that inspected file; if it is unavailable or incomplete, do not execute it. Then announce the read-only probe and invoke the collector without a prefix:

```text
run_script("scripts/collect-transport-facts.sh")
```

If it resolves an NVSHMEM prefix, tell the user which installation will be analyzed. Otherwise ask for an absolute prefix and rerun:

```text
run_script("scripts/collect-transport-facts.sh", "--prefix", "/opt/nvshmem")
```

Replace the example prefix with the resolved absolute prefix. Preserve partial output when it exits with status 2.

If the collector emits `manual_follow_up_command` entries, stop automatic probing and ask the user to run those commands manually in the target host or exact allocation/container used to launch NVSHMEM. Resume after the user returns their complete output. Do not repeat a command when equivalent successful output was already supplied.

If no, request complete output from `bash scripts/collect-transport-facts.sh` on the target, or its GPU/driver, NIC/fabric/provider, RDMA/provider-stack, and job-scale facts. Do not request an NVSHMEM prefix in this branch. Use installation evidence already supplied; otherwise eligibility remains conditional.

### 2. Choose the Analysis Scope

Ask one question with exactly two choices: **kernel/workload analysis** or **system eligibility only**. Skip this when already answered.

- For kernel/workload analysis, inspect only the selected kernel, its reachable device call graph, and directly corresponding host/on-stream calls when source is supplied. Otherwise use the communication-profile fields in [kernel-fit.md](references/kernel-fit.md). Do not search for other kernels.
- For system eligibility only, request no source/profile details. Select the compatibility-first default among candidates that pass the static system gates, emit its minimal selection configuration, and state that workload-specific performance was not assessed.

Treat assessment as read-only. Identify the job scale and optimization goal, and determine whether relevant communication can cross nodes; local P2P/NVLink traffic does not use the remote transport.

### 3. Classify the Communication

When kernel/workload analysis was selected, read [kernel-fit.md](references/kernel-fit.md) and record:

- API surface and exact operations;
- remote versus local paths;
- message-size distribution and whether operations are packed;
- concurrent issuers, peers, and expected physical message rate;
- ordering/completion frequency and QP pressure;
- host/device mix, CPU availability, and persistent-kernel behavior;
- scale-sensitive resource needs such as RC versus DCI/DCT.

Do not infer a high message rate merely from many logical elements. Account for application aggregation, one-thread issuance, warp coalescing, and batching.

### 4. Apply Hard Eligibility Gates

Read [transport-matrix.md](references/transport-matrix.md) completely. Exclude a candidate when any required condition is disproven:

- plugin unavailable in the selected NVSHMEM installation;
- target release or required operation is known to be unsupported;
- fabric/provider mismatch;
- missing driver, peer-memory/DMA-BUF, GDRCopy, DevX, UCX, libfabric, GPUNetIO, or DOCA prerequisite;
- incompatible memory mode, registered-buffer use, scale, or required feature.

Treat a missing fact as conditional only when it can invalidate the proposed configuration. Otherwise recommend the documented or compatibility-first default and list the missing fact under confidence and unknowns. Keep build support, runtime loadability, and hardware suitability separate.

For this skill, a loaded `gdrdrv` kernel module or visible `/dev/gdrdrv` device is sufficient positive evidence that the GDRCopy prerequisite is present. Do not require `pkg-config`, `ldconfig`, or a discovered `libgdrapi` path in addition, and do not treat missing userspace-library hints as evidence that GDRCopy is absent.

For IBGDA, evaluate both the GPU data path and a separate host remote transport. For GPUNetIO GDAKI, require `gpunetio` as the remote transport. Never enable IBGDA and GPUNetIO GDAKI together.

### 5. Rank Eligible Candidates

For system eligibility only, do not apply workload-fit rules. After hard eligibility, select a compatibility-first system default: IBRC for Mellanox InfiniBand or verified RoCE, native libfabric for EFA or Slingshot/CXI, a site-required UCX path, or `none` for a proven local-only deployment. Emit the selection block and do not claim it is the fastest. For kernel/workload analysis, apply these rules in order without assigning artificial scores:

1. **Mellanox/IB family gate:** On Mellanox InfiniBand or verified RoCE, evaluate the native IB paths first. Use IBRC as the conservative baseline, then consider IBDevX, IBGDA, or GPUNetIO only when their prerequisites and workload fit justify them.
2. **Other native-fabric gate:** Prefer libfabric with `efa` on EFA and `cxi` on Slingshot/CXI when no known release-specific operation or memory-mode limitation conflicts with the application.
3. **No remote network:** Select `none` only when all relevant PEs communicate within a peer-reachable NVLink domain and no remote path is required.
4. **Fine-grained parallel device traffic:** Favor an eligible GDAKI path when many GPU issuers submit small independent remote operations and a CPU proxy would serialize them.
5. **GDAKI choice:**
   - Favor IBGDA when DCI/DCT connection scaling is required, GPUNetIO is absent, or the exact release has stronger required operation coverage.
   - Favor GPUNetIO GDAKI when its RC connection scale is acceptable and a unified GPUNetIO CPU/GPU path or verified DOCA feature is useful.
   - Do not treat GPUNetIO design goals as proof that it outperforms IBGDA for this workload.
6. **Proxy-friendly traffic:** Favor a proxy path for host/on-stream-dominant work, packed larger transfers, low message concurrency, or latency-sensitive isolated operations.
7. **Mellanox proxy choice:** Use IBRC as the conservative verbs default. Recommend IBDevX only when mlx5 DevX is available, its required operation coverage is sufficient, and a concrete requirement such as avoiding a GDRCopy-dependent path justifies it. Do not claim general IBDevX superiority.
8. **UCX choice:** Favor UCX for a UCX-controlled deployment or a verified release-specific capability unavailable from the native alternatives. Preserve the bundled experimental/status caveats; consult exact documentation only when the selection depends on a release-specific detail.

If two candidates remain close, choose the compatibility-first option as the recommendation, report the other as runner-up, and state that static analysis cannot establish the performance winner.

Order the recommendation and candidate discussion by the target fabric. On Mellanox InfiniBand or verified RoCE, lead with IBRC, IBDevX, and IBGDA/GPUNetIO as applicable; do not lead with libfabric. On EFA or Slingshot/CXI, lead with the matching native libfabric provider.

### 6. Emit Minimal Selection Configuration

Emit only the variables required to select the resolved configuration:

| Configuration | Minimal exports |
| --- | --- |
| IBRC | `NVSHMEM_REMOTE_TRANSPORT=ibrc` |
| IBDevX | `NVSHMEM_REMOTE_TRANSPORT=ibdevx` |
| IBGDA | `NVSHMEM_REMOTE_TRANSPORT=<resolved-host-transport>` and `NVSHMEM_IB_ENABLE_IBGDA=1` |
| GPUNetIO CPU | `NVSHMEM_REMOTE_TRANSPORT=gpunetio` and `NVSHMEM_GPUNETIO_ENABLE_GDAKI=0` |
| GPUNetIO GDAKI | `NVSHMEM_REMOTE_TRANSPORT=gpunetio` and `NVSHMEM_GPUNETIO_ENABLE_GDAKI=1` |
| UCX | `NVSHMEM_REMOTE_TRANSPORT=ucx` |
| libfabric | `NVSHMEM_REMOTE_TRANSPORT=libfabric` and `NVSHMEM_LIBFABRIC_PROVIDER=<cxi|efa|verbs>` |
| No remote | `NVSHMEM_REMOTE_TRANSPORT=none` |

Do not add HCA mapping, QP counts, batching, NIC handlers, memory-mode changes, or provider tuning to this block. Put mandatory compatibility preconditions outside the block and route mapping/tuning to the appropriate sibling skill.

Default to emitting the minimal selection block for the recommended best-evidence configuration. Withhold it only when no viable candidate passes the hard gates or a required value cannot be resolved, such as a missing plugin, fabric/provider mismatch, or mandatory incompatible memory mode. Do not withhold merely because the patch version, workload profile, optional capability, or performance winner is unknown.

### 7. Return the Recommendation

Use this exact section order:

```text
Recommendation: <transport or combined host/GPU configuration; conditional if necessary>

Selection configuration
<minimal export block, or why it is withheld>

System eligibility evidence
<version, plugins, fabric/provider, driver/module/dependency evidence>

Kernel-fit evidence
<API surface, operation shape, sizes, concurrency, fan-out, completion, scale>

Runner-up
<candidate and the fact or tradeoff that kept it second>

Excluded candidates
<candidate: hard gate or unresolved prerequisite>

Confidence and unknowns
<high|medium|low; documented facts versus inference; missing evidence>

Static follow-up checks
<read-only checks or exact facts still needed; never benchmarks>
```

State the documentation version consulted when live verification was used. Describe performance as expected fit, not an assured result.

## Routing

- Route exact HCA selection and PE-to-NIC mapping to `$nvshmem-configure-nic-pe-mapping` when available.
- Route missing packages, plugins, or builds to `$nvshmem-install` when available.
- Route runtime initialization failures to `$nvshmem-troubleshoot-and-report-bugs` when available.
- If a sibling skill is unavailable, provide the boundary and required next evidence without inventing its behavior.

## Available Script

| Script | Purpose | Arguments |
| --- | --- | --- |
| [`scripts/collect-transport-facts.sh`](scripts/collect-transport-facts.sh) | Collect read-only NVSHMEM, GPU, kernel-module, RDMA, and provider evidence from a target compute node. | Optional `--prefix PATH` |


## Examples

- `Use $nvshmem-select-remote-transport. Inspect this target compute node and src/exchange.cu, then recommend a throughput-oriented transport for 256 PEs with 8 local PEs per node.`
- `Use $nvshmem-select-remote-transport with the attached collector output and communication profile. Recommend a compatibility-first transport for 32 PEs with 8 local PEs per node, emit the minimal selection exports, and state any assumptions.`


## Guardrails

- Do not run NVSHMEM binaries other than the read-only `nvshmem-info -n` query.
- Do not launch distributed jobs, allocate nodes, benchmark transports, or mutate system/application state.
- Do not recommend a plugin merely because its source exists; require evidence that the selected installation contains it.
- Do not convert absent evidence into compatibility.
- Do not claim that Ethernet link-layer evidence alone proves working RoCE.
- Do not let kernel heuristics override an incompatible fabric/provider or missing prerequisite.
- Do not emit both IBGDA and GPUNetIO GDAKI enables.

## Troubleshooting

- If the collector exits with status 2, preserve its output and resolve only the missing facts listed under `diagnostics`.
- When `nvidia-smi`, `ibstatus`, `ibv_devinfo`, or `rdma link` fails or cannot see devices while static sysfs/module evidence suggests hardware exists, treat device access as unresolved rather than treating the hardware as absent. Ask the user to run the failed commands manually in the target host or exact launch environment; do not retry with privileges or infer eligibility from host-visible sysfs alone.
- If a plugin is missing or fails to initialize, verify the selected prefix and dependencies, then route runtime failures to `$nvshmem-troubleshoot-and-report-bugs`.
- If a specialized capability remains conditional, recommend the best supported fallback and emit its configuration. Ask for exact version, provider, or operation evidence only when it could change the setting or no fallback passes the hard gates.

## Limitations

- Static analysis estimates architectural fit; it does not validate runtime correctness, measure performance, or guarantee the fastest transport.
- Some compatibility details are release-sensitive; verify the exact target release only when the recommendation depends on one of them.
- The collector describes one node and one selected installation. It does not prove cluster-wide uniformity, working RoCE, or complete operation coverage.
- Incomplete evidence lowers confidence and must be disclosed, but does not by itself prevent a best-evidence recommendation or minimal exports.

<!-- ci trigger -->
