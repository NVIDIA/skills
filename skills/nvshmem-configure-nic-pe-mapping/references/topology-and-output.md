# Topology Selection and Recommendation Output

## Contents

1. [Required evidence](#required-evidence)
2. [Interpret the collector](#interpret-the-collector)
3. [Filter ports](#filter-ports)
4. [Rank and assign](#rank-and-assign)
5. [Build the expanded mapping](#build-the-expanded-mapping)
6. [Output contract](#output-contract)
7. [Validation](#validation)

## Required Evidence

Do not produce an exact topology-aware export until all required values are known:

| Input | Acceptable evidence |
| --- | --- |
| NVSHMEM version | User statement, `nvshmem-info -n`, or an installed-library record that identifies a complete version |
| Remote transport | User/launcher selection or NVSHMEM initialization log |
| Local PE count | Launcher layout, scheduler layout, or explicit user value |
| Local PE-to-GPU binding | Explicit ordered binding, per-rank log, `CUDA_VISIBLE_DEVICES` interpretation with launcher rank order, or equivalent runtime evidence |
| GPU-to-NIC topology | `nvidia-smi topo -m` from the target compute node |
| Port eligibility | HCA, port, PCI BDF, active state, and link layer from sysfs or `ibv_devinfo` |

Assume every node has the same configuration only when the user has not supplied heterogeneous per-node evidence. State this assumption in the answer.

Do not equate global PE with local PE. Mapping formulas use the zero-based local PE index on each node.

## Interpret the Collector

The collector emits labeled sections:

- `[nvshmem-version]`: discovered version command and output;
- `[gpu-pci]`: GPU index, PCI bus ID, and model;
- `[gpu-nic-topology]`: raw `nvidia-smi topo -m` matrix and legend;
- `[rdma-ports]`: one normalized line for every sysfs HCA port;
- `[ibv-devinfo-fallback]`: raw fallback evidence when sysfs yielded no ports;
- `[diagnostics]`: missing tools, failed commands, and final completeness status.

Retain the raw evidence. Do not treat `status=partial` as an empty failure: identify exactly which required input is missing and ask for only that input.

Correlate the topology matrix's NIC labels with HCA names or PCI BDFs. If a topology NIC label cannot be tied unambiguously to an `hca:port`, mark its GPU distance unknown and do not invent a mapping from numerical suffixes.

The PCI BDF formats `00000000:3b:00.0` and `0000:3b:00.0` refer to the same domain/bus/device/function after normalizing the domain width. Preserve the original value in evidence and use a normalized value only for matching.

## Filter Ports

Create an exclusion ledger before ranking:

1. Exclude a port unless its state explicitly includes `ACTIVE` or equivalent active evidence.
2. Group remaining ports by normalized link layer:
   - `InfiniBand`;
   - `Ethernet`, which may represent RoCE but does not prove that RoCE is configured;
   - unknown/other.
3. Exclude unknown/other link layers unless the user supplies provider evidence that makes them usable.
4. Do not mix the InfiniBand and Ethernet groups in one recommendation by default.
5. If the user explicitly selects a fabric, use only its group.
6. Otherwise use the larger active group and prefer InfiniBand on an exact tie. State this default and allow the user to override it before finalizing if fabric intent is unclear.

Treat different active ports on one HCA as distinct selectable units, but disclose that they may share PCI or upstream bandwidth.

## Rank and Assign

Convert recognized GPU-to-NIC topology tokens to ordinal costs:

| Token | Cost | Meaning |
| --- | ---: | --- |
| `PIX` | 0 | At most one PCIe bridge |
| `PXB` | 1 | Multiple PCIe bridges without the host bridge |
| `PHB` | 2 | Traverses a PCIe host bridge |
| `NODE` | 3 | Traverses host bridges within one NUMA node |
| `SYS` | 4 | Traverses the inter-NUMA/system path |

Treat `X`, a missing cell, an uncorrelated NIC label, and an unknown token as unavailable evidence rather than a fabricated cost. Ask for clarification if it changes the selected port.

### One Port Per PE with Enough Ports

When `E >= N` and the user chooses a closest subset:

1. Select exactly N of the E usable ports.
2. Minimize the worst assigned distance, then total distance across PEs.
3. Break equal-cost assignments by lower reuse/load, then canonical `hca:port` order.
4. Report usable ports that were not selected.

This is a minimum-cost matching problem, not a rule that GPU 0 uses NIC 0.

Treat the closest subset as chosen only when the user requested it or confirmed it after learning that multi-NIC is unsupported. A request to use all ports is not subset approval; withhold final exports and ask before substituting the subset.

### Fewer Physical Ports Than PEs

Allow sharing. Assign every PE its closest port; for equal distances choose the currently less-loaded port, then canonical order.

For an exact mapping, emit N expanded slots even when fewer than N physical ports exist. Repeat the selected `hca:port:1` for each PE as needed. This preserves one exact slot per PE and avoids depending on an accidental shorter-list modulo pattern.

Apply this rule only after the local PE-to-GPU binding is known. Do not repeat all physical ports for all PEs as a workaround for unknown binding; doing so converts a one-port closest assignment into a multi-port assignment with avoidable distant paths.

### Multiple Ports Per PE

First verify version and transport support. If E usable ports will all be used with N PEs, calculate the BLOCK quota:

```text
quota(P) = floor(E / N) + (1 if P < E mod N else 0)
```

Assign distinct ports to PEs subject to those exact quotas. Minimize the worst distance, then total distance; resolve equal solutions by balanced upstream-HCA use and canonical port order. Group each PE's assigned ports contiguously in local-PE order when constructing the mapping.

Do not sacrifice a clearly better distance merely to make HCA names look symmetrical. If several ports share the same PCI device or upstream link, disclose that balanced port counts may not equal balanced physical bandwidth.

### Automatic Assignment Cases

For <=3.7 IBGDA or GPUNetIO multi-NIC, the skill can predict the eligible closest set but cannot encode an exact multi-NIC `HCA_PE_MAPPING`. Clearly distinguish:

- the predicted topology-aware selection;
- the exact variables that merely enable automatic selection;
- the runtime logs needed to confirm what the transport chose.

For UCX or libfabric, do not translate a predicted table into unsupported common-HCA controls.

## Build the Expanded Mapping

Use explicit port and count fields:

```text
mlx5_0:1:1,mlx5_1:1:1
```

Before emitting the export, write the expansion as an indexed sequence:

```text
slot 0 = mlx5_0:1
slot 1 = mlx5_1:1
```

Then write the version-specific interpretation:

- <=3.7 or `E <= N` under the **NVSHMEM 3.8 or later BLOCK contract**: `PE P -> slot P mod E`;
- the **NVSHMEM 3.8 or later BLOCK contract** with `E > N`: show `start`, `count`, and slot indices for every PE.

Compressed counts are valid only when their expanded order is identical to the intended sequence. Prefer `:1` entries for readability and auditable sharing.

For an exact multi-port mapping under the **NVSHMEM 3.8 or later BLOCK contract**, order slots as:

```text
[all PE 0 ports], [all PE 1 ports], ..., [all PE N-1 ports]
```

For a one-port-per-PE mapping, set E to N even when that requires duplicate physical ports.

## Output Contract

Use this compact structure. Treat the Mapping and Exact Exports sections as mandatory output gates, not optional supporting detail.

### Resolution

Report:

- requested and discovered NVSHMEM version;
- version bucket and transport;
- applicable matrix section; include documentation edition, official sources, and freshness only when a documentation check was needed;
- local PE count and homogeneity assumption;
- confirmed local PE-to-GPU order;
- selected link layer and its selection reason.

### Mapping

For every final topology recommendation, include a table with these columns:

| Local PE | GPU index/BDF | HCA:port | NIC BDF | Distance | Notes |
| ---: | --- | --- | --- | --- | --- |

Use one row per PE-port assignment. Multiple rows for a PE make multi-port allocation explicit.

Do not replace the table with prose. If required evidence is missing, state `Mapping table withheld: <missing evidence>` and do not present final exports.

### Expanded-Slot Interpretation

Show the complete indexed expansion and the applicable modulo or BLOCK calculation. For `HCA_LIST`, show the canonical eligible order, `usable`, and any removed trailing ports instead.

### Exact Exports

Never return an NVSHMEM environment-variable assignment, export block, or conditional example unless the complete installed NVSHMEM version is identified.

For every final explicit common-HCA recommendation, use a single fenced `bash` block. Include only variables supported by the resolved release/transport and exactly one of `NVSHMEM_HCA_PE_MAPPING` and `NVSHMEM_HCA_LIST`.

For automatic/provider-controlled cases, emit only verified supported controls and state explicitly that neither common-HCA selection variable is valid. If required evidence or a user decision is unresolved, state `Exact export block withheld: <reason>` rather than silently omitting or fabricating the block.

Do not include debug variables in the primary mapping block. Put validation variables in a separate block.

Do not include `NVSHMEM_REMOTE_TRANSPORT` or other launcher/provider controls in a mapping-only block unless the user also asks to select the transport and its exact value has been verified for the resolved release.

### Exclusions and Caveats

List:

- inactive ports;
- ports from the non-selected link layer;
- unmatched or unknown topology entries;
- unused eligible ports;
- shared PCI/upstream-path concerns;
- unknown provider or RoCE readiness;
- every assumption and any documentation-check uncertainty.

### Validation

Recommend:

```bash
export NVSHMEM_DEBUG=INFO
export NVSHMEM_DEBUG_SUBSYS=INIT,TRANSPORT
```

Run a small initialization or representative workload. Confirm version, transport, selected HCA ports, per-PE device count, and absence of fallback warnings in logs. Then compare correctness and performance with the same workload under automatic selection. Change one mapping decision at a time.

## Validation

Reject or revise a recommendation when any check fails:

- every local PE has a confirmed GPU binding;
- an ambiguous binding has not been hidden by assigning all ports to all PEs;
- every selected HCA port is active;
- all selected ports use one link layer unless the user explicitly overrides;
- the expanded slot count and per-PE interpretation match the version gate;
- multi-port is supported by the transport and version;
- a 3.8-or-later explicit multi-port recommendation includes both enable variables;
- only one HCA selection mechanism is exported;
- UCX/libfabric boundaries are not disguised as exact common-HCA control;
- the answer identifies the applicable matrix section and, when documentation was checked, its edition or an explicit freshness caveat.
- a final topology recommendation contains the mapping table, and a final explicit common-HCA recommendation contains the exact export block.
