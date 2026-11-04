---
name: nvshmem-configure-nic-pe-mapping
description: Recommend NVSHMEM NIC-to-PE mappings and environment exports. Use for HCA selection, multi-NIC configuration, or topology-based mapping diagnostics.
license: Apache-2.0
metadata:
  version: "1.0.0"
  author: NVIDIA NVSHMEM Team <nvshmem@nvidia.com>
  tags:
    - nvshmem
    - nic
    - mapping
---

# Tune NVSHMEM NIC-to-PE Mapping

## Purpose

Produce a reproducible recommendation from the target node's actual GPU/NIC topology. Do not promise a performance improvement; present the mapping as a topology-informed starting point that must be validated.

## Requirements

- Use an allocated target compute node with visible GPUs and RDMA devices; do not use login-node topology as evidence.
- Make `nvidia-smi` and `/sys/class/infiniband` readable on that node. If `/sys` is unavailable, make `ibv_devinfo` available or provide equivalent per-port evidence.
- Provide the NVSHMEM installation prefix, make `nvshmem-info` available on the target node, or supply the installed NVSHMEM version explicitly.
- Use the bundled read-only collector when possible. It requires no API keys, credentials, package installation, or writable filesystem access.

## Inputs

Resolve every required input before emitting exact exports. If values from different sources conflict, identify the conflict and request clarification rather than selecting a value silently.

### Required inputs

- **NVSHMEM version:** Prefer the user-confirmed installed version; otherwise use the collector's `[nvshmem-version]` output, then `nvshmem-info -n` on the target node. Retain the complete reported version after normalizing a leading `v`.
- **Remote transport:** Prefer the explicitly selected transport for the target run; otherwise use the target node's NVSHMEM initialization log. Do not infer a transport from topology; request a log when automatic selection is ambiguous.
- **Local PE count:** Prefer observed per-rank logs, then launcher configuration, then an explicit user-provided count. Confirm that all evidence describes the same node allocation.
- **Ordered local `PE -> GPU index or PCI BDF` binding:** Prefer observed per-rank binding logs, then launcher binding configuration, then an explicit user mapping. Use `CUDA_VISIBLE_DEVICES` only when its rank order is explicitly established.
- **Target-node GPU/NIC and HCA-port topology:** Prefer the newest complete output from `scripts/collect-nic-topology.sh` on the target node. Otherwise use the listed `nvidia-smi` commands plus per-port `/sys/class/infiniband` or `ibv_devinfo` evidence. Include GPU and NIC PCI BDFs, port state, and link layer.

### Optional inputs

- **Requested link layer:** Use the user's explicit InfiniBand or Ethernet/RoCE preference. If absent, choose from observed active ports as described in step 5.
- **Excess-port policy:** Use the user's explicit request to select a closest subset or use multiple NICs per PE. If absent, present supported alternatives conditionally and make no selection.
- **Current official documentation or citations:** Obtain only when the user requests them or the version/transport matrix is insufficient. Use `$nvshmem-docs` for that check.

## Scripts

| Script | Use | Invocation |
| --- | --- | --- |
| `scripts/collect-nic-topology.sh` | Collect read-only NVSHMEM version, GPU PCI, GPU/NIC topology, and RDMA-port evidence. It emits all evidence to stdout and may exit nonzero after producing useful partial output. | Resolve the script from this skill directory and invoke `run_script("scripts/collect-nic-topology.sh", "--prefix", "/absolute/nvshmem/prefix")` when the prefix is known, or omit `--prefix` to use environment/PATH discovery. |

## Read the Required References

- Read [references/version-transport-matrix.md](references/version-transport-matrix.md) before interpreting or emitting any mapping variable.
- Read [references/topology-and-output.md](references/topology-and-output.md) before ranking NICs or formatting a recommendation.
- Treat the bundled **NVSHMEM 3.8 or later BLOCK contract** as authoritative within its recorded transport scope. Do not invoke `$nvshmem-docs` merely to revalidate behavior that this contract resolves, including for releases later than 3.8.

## Instructions

### 1. Resolve the NVSHMEM Version and Transport

Obtain the version from the user first. Otherwise use the collector's `[nvshmem-version]` evidence or run `nvshmem-info -n` when it is already available on the target node. Normalize a leading `v` and retain the complete reported version.

Classify the release as:

- 3.7 or earlier;
- 3.8 or later.

Do not assume `latest` or silently treat an unknown version as 3.8. If the version remains unknown, request the installed version before continuing.

Determine the selected remote transport: IBRC, IBDevX, IBGDA, GPUNetIO, UCX, or libfabric. If automatic selection makes the transport ambiguous, ask for the relevant initialization log or the intended transport.

### 2. Resolve Documentation Uncertainty

Use [references/version-transport-matrix.md](references/version-transport-matrix.md) as the default authority for mapping behavior. Do not invoke `$nvshmem-docs` when the matrix or the bundled **NVSHMEM 3.8 or later BLOCK contract** resolves the version, transport, feature gate, and required mapping controls, including for a release later than 3.8.

Invoke the sibling `$nvshmem-docs` skill at `../nvshmem-docs/SKILL.md` only when the matrix and bundled contract are insufficient or conflicting, exact feature-version applicability remains unclear, or the user explicitly requests live official documentation or citations. Follow that skill's version and citation workflow.

If a documentation check is needed but the sibling skill, an HTTPS retrieval tool, or the live official pages are unavailable:

- continue from the matrix only when it resolves the needed behavior;
- otherwise withhold final exports and identify the unresolved behavior;
- label `Documentation freshness: unverified` only when a documentation check was attempted;
- state which official routes could not be checked; and
- never invent a behavior that the matrix does not establish.

When a documentation check is performed, report the exact documentation edition consulted and any remaining uncertainty. Otherwise identify the applicable matrix section as the basis for the recommendation.

### 3. Collect Topology Evidence

When already running on the target compute node, tell the user that the skill is running a read-only collector. Before its first invocation in the current checkout, inspect the complete bundled [collector script](scripts/collect-nic-topology.sh). Run only that inspected file; if it is unavailable or incomplete, do not execute it. Then run:

```text
run_script("scripts/collect-nic-topology.sh")
```

When the NVSHMEM installation prefix is known, pass its absolute path explicitly:

```text
run_script("scripts/collect-nic-topology.sh", "--prefix", "/opt/nvshmem")
```

Replace the example path with the absolute prefix. Without `--prefix`, discovery uses `NVSHMEM_PREFIX`, then `nvshmem-info` on `PATH`. Prefix libraries are scoped to that query. Exit 2 indicates incomplete evidence.

When not on the target node, ask the user for the collector's complete output from the target node. If the script is unavailable remotely, request all of:

```bash
nvshmem-info -n
nvidia-smi --query-gpu=index,pci.bus_id,name --format=csv,noheader
nvidia-smi topo -m
```

Also request, for every candidate HCA port, its HCA name, port number, PCI BDF, state, and link layer from `/sys/class/infiniband` or `ibv_devinfo`.

Do not run a collector from a login node and treat that topology as compute-node evidence.

### 4. Establish PE Placement

Obtain the number of local PEs per node. Assume nodes are homogeneous only after stating that assumption.

Obtain the ordered local `PE -> GPU index or PCI BDF` binding from explicit user input, launcher configuration, `CUDA_VISIBLE_DEVICES`, or per-rank logs. Do not infer a local PE's GPU solely from PE count or the order displayed by `nvidia-smi topo -m`.

If the binding is ambiguous, request it before producing an exact closest-NIC mapping. A conditional table is acceptable, but final exports are not. Do not bypass the missing binding by duplicating every usable NIC for every PE; that adds non-closest paths and is a different multi-port policy. Consider such an all-ports assignment only after the user explicitly requests it and accepts the distance tradeoff.

### 5. Filter and Rank Candidate Ports

Treat an HCA port, not merely an HCA device, as the selectable NIC unit.

- Exclude inactive or physically down ports and record why each was excluded.
- Partition active ports by link layer. Do not mix InfiniBand with Ethernet/RoCE by default.
- Honor an explicitly requested link layer. Otherwise choose the larger active partition; prefer InfiniBand on an exact tie. Ask the user if that default conflicts with their fabric or provider intent.
- Rank each bound GPU-to-port pair by `PIX < PXB < PHB < NODE < SYS` from `nvidia-smi topo -m`.
- Break equal-distance choices by lower assigned-port load, then canonical `hca:port` order.
- Share a port when fewer usable ports than PEs makes sharing necessary, while preserving distance first and balance second.

### 6. Resolve Excess NICs Explicitly

When usable NIC ports outnumber local PEs, stop before final exports and ask whether to:

1. select the closest balanced subset; or
2. use multiple NICs per PE.

Offer option 2 only when the resolved version and transport support it according to the matrix. Explain whether multi-NIC requires automatic distance assignment or supports an explicit BLOCK mapping. If the user does not answer, show both supported alternatives as conditional recommendations and do not choose silently.

If the user asked to use all NICs but the resolved version/transport does not support multi-NIC, explain the restriction and request approval to use the closest subset. Do not treat the unsupported request as implicit approval for a subset, and do not emit final exports until the user confirms it. When only the subset is supported, present that single supported choice for confirmation instead of offering a disabled multi-NIC option.

### 7. Synthesize Exactly One Selection Mechanism

Prefer `NVSHMEM_HCA_PE_MAPPING` for exact assignments. Use `NVSHMEM_HCA_LIST` only when selecting an eligible set and accepting transport-canonical ordering. Never set both variables.

Apply the version/transport rules in the matrix exactly:

- For 3.7 and earlier, preserve one expanded mapping slot per PE for explicit IB mappings. Use automatic distance assignment for supported multi-NIC paths.
- For 3.8 or later, apply the bundled BLOCK contract: set `NVSHMEM_ENABLE_NIC_PE_MAPPING=1` for explicit assignments and `NVSHMEM_ENABLE_MULTI_PORT=1` whenever recommending more than one NIC per PE.
- For UCX and libfabric, explain the provider-controlled boundary and do not fabricate common-HCA exact mappings.

Build `NVSHMEM_HCA_PE_MAPPING` with explicit `hca:port:count` entries. Expand every count before explaining which slots each PE receives. Preserve duplicates when intentional sharing is required.

Keep a mapping-only export block limited to NIC-assignment controls. Do not add `NVSHMEM_REMOTE_TRANSPORT` or provider/launcher settings merely because the transport is known; include them only when the user separately asks to select the transport and the matrix, or documentation when needed, verifies the value.

### 8. Return the Recommendation

Execute the [Output Contract](references/topology-and-output.md#output-contract) in order; that reference is the single authority for output fields, formatting, withholding messages, and validation guidance.

Treat its `Mapping` and `Exact Exports` sections as mandatory gates. Do not return until each gate is either completed or explicitly withheld for the reason allowed by the contract. Do not duplicate or paraphrase the contract in this file.

## Examples

- “My 4 local PEs are bound to GPUs 0–3. Given this target-node collector output and NVSHMEM 3.8.0 IBRC, recommend the closest balanced InfiniBand port mapping and exact exports.”
- “Interpret this `nvidia-smi topo -m` and per-port state/link-layer output. NVSHMEM is 3.7 with IBGDA; identify whether multiple NICs per PE are supported and what extra evidence is needed before emitting exports.”
- “We have 2 PEs and 5 active Ethernet/RoCE ports. Use this PE-to-GPU binding and initialization log to show the supported subset and multi-NIC alternatives without selecting one silently.”

## Troubleshooting

| Symptom | Response |
| --- | --- |
| The collector reports `status=partial`, an unknown NVSHMEM version, missing GPUs, or no HCA ports. | Re-run it on an allocated target compute node with the intended NVSHMEM installation and NVIDIA/RDMA device visibility. Paste the complete labeled output; a nonzero exit may still contain usable evidence. |
| No exact mapping can be produced. | Obtain the NVSHMEM version, resolved remote transport, local PE-to-GPU binding, and active-port evidence. Give only conditional guidance until all are available. |
| Ports have mixed link layers or a requested port is inactive. | Exclude inactive ports and select one link-layer partition unless the user explicitly requests another supported fabric. |
| NVSHMEM behavior is unclear for the selected version or transport. | Check the version/transport matrix first, then invoke `$nvshmem-docs` only if it remains insufficient or the user requires current official citations. |
| Logs select unexpected ports after applying the mapping. | Verify the export block, PE binding, selected transport, and provider constraints; compare against the unforced automatic configuration before changing other settings. |

## Limitations

- Limit recommendations to the observed target-node topology and port state; repeat evidence collection after hardware, fabric, launcher, or binding changes.
- Treat topology proximity as a selection heuristic, not a performance guarantee; validate with a representative workload.
- Do not produce an exact mapping until the NVSHMEM version, transport, local PE-to-GPU binding, and usable-port evidence are resolved.
- For UCX and libfabric, remain within the provider-controlled boundary and do not fabricate common-HCA mapping settings.

## Guardrails

- Do not select inactive ports or silently mix link layers.
- Do not assume that a numerically matching GPU and NIC index implies proximity.
- Do not turn an unknown PE-to-GPU binding into an all-ports-per-PE mapping.
- Do not emit 3.8-or-later BLOCK behavior for 3.7 or earlier.
- Do not emit legacy IBGDA multi-port controls as the primary 3.8-or-later control.
- Do not generate provider-native UCX or libfabric settings; explain them without inventing values.
- Do not overwrite user files or change the system while collecting evidence.

<!-- ci trigger -->
