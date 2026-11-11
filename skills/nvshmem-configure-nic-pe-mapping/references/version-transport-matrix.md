# NVSHMEM NIC Mapping Version and Transport Matrix

## Contents

1. [Precedence and terms](#precedence-and-terms)
2. [Environment-variable roles](#environment-variable-roles)
3. [NVSHMEM 3.7 and earlier](#nvshmem-37-and-earlier)
4. [NVSHMEM 3.8 or later BLOCK contract](#nvshmem-38-or-later-block-contract)
5. [Export patterns](#export-patterns)
6. [Evidence matrix](#evidence-matrix)

## Precedence and Terms

This reference bundles the **NVSHMEM 3.8 or later BLOCK contract** for common IB paths. The contract's transport scope is defined in its section below.
Never return an NVSHMEM environment-variable assignment, export block, or conditional example unless the complete installed NVSHMEM version is identified.

Apply evidence in this order:

1. Apply the bundled **NVSHMEM 3.8 or later BLOCK contract** within its recorded transport scope.
2. Use this bundled matrix for every version- and transport-specific behavior it resolves.
3. Do not invoke `$nvshmem-docs` merely to revalidate behavior resolved by the bundled contract, including for a release later than 3.8. Invoke it only when the contract and matrix are insufficient or conflicting, or the user explicitly requests live official evidence.
4. Do not invent behavior that neither the contract, matrix, nor a requested documentation check establishes.

Use these terms consistently:

- **Port**: one selectable HCA port, identified as `hca:port`.
- **Expanded slot**: one logical entry after expanding every `hca:port:count` item in `NVSHMEM_HCA_PE_MAPPING`.
- **N**: number of local processing elements (PEs) on a node.
- **E**: number of eligible HCA-list ports or expanded mapping slots, depending on the variable.
- **P**: zero-based local PE index, `0 <= P < N`.
- **Automatic assignment**: `NVSHMEM_ENABLE_NIC_PE_MAPPING=0`; the transport's topology-distance path chooses devices.
- **Explicit assignment**: `NVSHMEM_ENABLE_NIC_PE_MAPPING=1`; list or mapping semantics apply.

## Environment-Variable Roles

| Variable | Role | Guardrail |
| --- | --- | --- |
| `NVSHMEM_HCA_PE_MAPPING` | Ordered `hca:port:count` mapping; counts expand into logical slots | Prefer for exact assignments |
| `NVSHMEM_HCA_LIST` | Eligible HCA-port set | Use only when canonical transport ordering is acceptable |
| `NVSHMEM_ENABLE_NIC_PE_MAPPING` | Select explicit (`1`) versus automatic distance (`0`) assignment | Required as `1` by the 3.8-or-later contract for explicit recommendations |
| `NVSHMEM_ENABLE_MULTI_PORT` | Common 3.8-or-later multi-port control | Set explicitly to `1` for every multi-NIC recommendation under the contract |
| `NVSHMEM_IBGDA_ENABLE_MULTI_PORT` | Legacy IBGDA-specific multi-port control | Use only for 3.7-or-earlier IBGDA; treat as a deprecated alias under the 3.8-or-later contract |
| `NVSHMEM_LIBFABRIC_MAX_NIC_PER_PE` | Limits libfabric NICs per PE | Does not create a common-HCA exact mapping |

Never set both `NVSHMEM_HCA_LIST` and `NVSHMEM_HCA_PE_MAPPING`. In the 3.7 implementation, `HCA_LIST` wins their conflict with a warning, which makes a combined recommendation misleading.

The common HCA parser has historically imposed bounded list sizes. Avoid compressed counts that obscure the intended slots, keep recommendations small, and require runtime-log verification for unusually large mappings.

## NVSHMEM 3.7 and Earlier

### Common Explicit Selection

For common IB explicit mapping, expand the mapping and select exactly one slot per local PE:

```text
selected_slot(P) = P mod E
```

An exact one-NIC-per-PE recommendation therefore needs an expanded sequence in local-PE order. Example for four local PEs:

```bash
export NVSHMEM_ENABLE_NIC_PE_MAPPING=1
export NVSHMEM_HCA_PE_MAPPING='mlx5_0:1:1,mlx5_1:1:1,mlx5_2:1:1,mlx5_3:1:1'
```

Counts are expanded before the modulo. For example, `mlx5_0:1:2,mlx5_1:1:2` expands to `[mlx5_0:1, mlx5_0:1, mlx5_1:1, mlx5_1:1]`.

Automatic distance assignment can return multiple closest devices, but each transport decides how many it consumes.

Apply feature-introduction gates before the table: official release notes introduce IBGDA multi-port in the 2.11 family, libfabric multi-NIC in the 3.6 family (published in 3.6.5), and GPUNetIO in 3.7. Do not offer one of those paths for an older release merely because it appears in the `<=3.7` table. Consult `$nvshmem-docs` only when the exact release applicability remains unclear.

### Transport Rules

| Transport | Exact explicit mapping in <=3.7 | Multiple NICs per PE in <=3.7 | Recommendation rule |
| --- | --- | --- | --- |
| IBRC | One expanded slot per PE | No | Emit a one-slot explicit mapping or use automatic single-device selection |
| IBDevX | One expanded slot per PE | No | Emit a one-slot explicit mapping or use automatic single-device selection |
| IBGDA | One expanded slot per PE | Yes, only through automatic distance assignment with `NVSHMEM_IBGDA_ENABLE_MULTI_PORT=1` | Do not claim explicit multi-NIC assignment; do not set `HCA_PE_MAPPING` for the automatic multi-port case |
| GPUNetIO | One expanded slot per PE | Yes, through automatic selection | Explicit mapping remains single-slot; use automatic assignment for multiple selected NICs |
| libfabric | Common HCA mapping does not provide an exact provider-domain assignment | Yes, provider/automatic selection limited by `NVSHMEM_LIBFABRIC_MAX_NIC_PER_PE` | Explain provider control; optionally recommend only the documented max-NIC count |
| UCX | Common HCA mapping is not used | UCX manages device selection internally; 3.7 source does not establish common multi-NIC mapping | Explain the UCX boundary and do not generate HCA or provider-native selection variables |

For 3.7 IBGDA automatic multi-port:

```bash
export NVSHMEM_ENABLE_NIC_PE_MAPPING=0
export NVSHMEM_IBGDA_ENABLE_MULTI_PORT=1
```

Do not add `NVSHMEM_HCA_PE_MAPPING` or `NVSHMEM_HCA_LIST` to that block. Report that automatic topology selection, not an exact user slot list, chooses the NICs.

For 3.7 GPUNetIO automatic multi-NIC selection:

```bash
export NVSHMEM_ENABLE_NIC_PE_MAPPING=0
```

Do not invent a GPUNetIO multi-port environment variable.

IBGDA implementations cap the number of devices per PE; the 3.7 source cap is 15. Treat this as an implementation limit, not a target. Use fewer NICs unless measurements justify more.

## NVSHMEM 3.8 or later BLOCK contract

The contract covers the common IB path for IBRC, IBDevX, IBGDA, and GPUNetIO. It does not override provider-native UCX or libfabric selection.

Enable explicit selection with:

```bash
export NVSHMEM_ENABLE_NIC_PE_MAPPING=1
```

For any recommendation assigning more than one port to a PE, also set:

```bash
export NVSHMEM_ENABLE_MULTI_PORT=1
```

### `NVSHMEM_HCA_PE_MAPPING`

Expand `hca:port:count` into E logical slots.

When `E <= N`, preserve the historical one-slot behavior:

```text
PE P receives slot P mod E
```

This includes shared-NIC cases when fewer slots than PEs exist.

When `E > N`, distribute contiguous BLOCK ranges with any remainder assigned to the lowest local PEs:

```text
base  = floor(E / N)
rem   = E mod N
start = P * base + min(P, rem)
count = base + (1 if P < rem else 0)
slots = [start, start + count)
```

Example: five expanded slots across two PEs produce:

```text
PE 0 -> slots 0,1,2
PE 1 -> slots 3,4
```

Preserve the sequence exactly: group the closest intended ports for PE 0 first, followed by those for PE 1, and so on. If a port is intentionally shared, repeat it in the expanded sequence and explain the duplicate.

### `NVSHMEM_HCA_LIST`

`HCA_LIST` selects eligible HCA ports, but their effective order is the transport's canonical device order rather than an arbitrary exact PE sequence.

For explicit multi-port assignment when `E >= N`:

1. Compute `usable = E - (E mod N)`.
2. Drop the trailing `E - usable` canonical slots when the remainder is nonzero.
3. Warn about every dropped port.
4. Give each PE the contiguous block of `usable / N` slots.

Example: five canonical eligible ports across two PEs use the first four; each PE receives two, and the fifth is excluded as a trailing slot.

When `E < N`, do not reduce the list to zero. Preserve one-slot modulo behavior, allowing an HCA port to be shared by several PEs.

Use `HCA_LIST` only when the user wants set-based filtering and accepts this canonical ordering plus trailing-slot rule. Use `HCA_PE_MAPPING` for exact topology assignments and uneven remainder distribution.

### Multi-Port Disabled

When multi-port is disabled, preserve a single selected slot per PE even if more candidates are available. Never present a multi-slot BLOCK table without `NVSHMEM_ENABLE_MULTI_PORT=1`.

## Export Patterns

### 3.8 or Later Exact One Port Per PE

```bash
export NVSHMEM_ENABLE_NIC_PE_MAPPING=1
export NVSHMEM_HCA_PE_MAPPING='<PE-ordered hca:port:1 entries>'
```

### 3.8 or Later Exact Multiple Ports Per PE

```bash
export NVSHMEM_ENABLE_NIC_PE_MAPPING=1
export NVSHMEM_ENABLE_MULTI_PORT=1
export NVSHMEM_HCA_PE_MAPPING='<PE-blocked hca:port:1 entries>'
```

### 3.8 or Later Eligible Set

```bash
export NVSHMEM_ENABLE_NIC_PE_MAPPING=1
export NVSHMEM_ENABLE_MULTI_PORT=1
export NVSHMEM_HCA_LIST='<eligible hca:port entries>'
```

Before using the eligible-set pattern, show the canonical sequence and any trailing removal. Omit `NVSHMEM_ENABLE_MULTI_PORT` only when the resulting interpretation is intentionally one port per PE.

Do not place `NVSHMEM_HCA_LIST` and `NVSHMEM_HCA_PE_MAPPING` in the same export block. Do not quote multiple conditional alternatives as though they should all be exported together.

## Evidence Matrix

| Evidence | What it establishes | Applicability |
| --- | --- | --- |
| [Official environment-variable reference](https://docs.nvidia.com/nvshmem/latest/api/gen/env.html) | Live names, descriptions, and defaults for documented runtime settings | Consult through `$nvshmem-docs` only to resolve uncertainty; the latest route alone does not prove an older edition |
| [NVSHMEM 2.11.0 release notes](https://docs.nvidia.com/nvshmem/release-notes-install-guide/prior-releases/release-2110.html) | Official introduction of IBGDA multi-port support | Feature gate for IBGDA multi-port |
| [NVSHMEM 3.6.5 release notes](https://docs.nvidia.com/nvshmem/release-notes-install-guide/prior-releases/release-3605.html) | Official introduction of libfabric multi-NIC support | Feature gate for libfabric multi-NIC |
| [NVSHMEM 3.7.0 release notes](https://docs.nvidia.com/nvshmem/release-notes-install-guide/release-notes/release-3700.html) | Official 3.7 release context, including GPUNetIO introduction | 3.7 feature context |
| NVSHMEM public tag `v3.7.0-0`, commit `d72cf23` | Common selector uses local-PE modulo for explicit mapping; IBRC/IBDevX consume one selected device; automatic IBGDA, GPUNetIO, libfabric, and UCX boundaries | Bundled source-backed fallback for <=3.7 when live documentation is unavailable |
| Bundled **NVSHMEM 3.8 or later BLOCK contract** | HCA-list trailing removal, expanded-mapping remainder distribution, preserved one-slot cases, common-IB transport scope, and explicit mapping gate | NVSHMEM 3.8 or later, within the contract's recorded transport scope |

The bundled contract wins if a pre-release implementation snapshot differs, including for GPUNetIO contract coverage. Do not invoke live documentation merely to revalidate behavior that the contract resolves.
