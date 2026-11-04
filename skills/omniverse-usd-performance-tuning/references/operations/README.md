<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- AUTO-GENERATED FROM references/operations/operations.json -->
<!-- Source data lives in references/operations/operations.json. -->

# Operation Index

Catalog of all Usd Optimize operations known to this workflow. Each row
carries the routing fields (loss class, risk, confirmation, pipelines) from the
catalog. Use this to find operations by category, loss class, or argument count;
use upstream `usd-optimize` or the prebuilt Usd Optimize package for operation
behavior, parameters, defaults, and implementation gotchas.

The package resolution rule is centralized once in
[`usd-optimize` upstream handoff](../upstreams/usd-optimize.md): derive the
upstream operation guide from the operation key as
`docs/operations/<operation-key>.rst`, then resolve it under
the selected Usd Optimize package root. Do not duplicate package URLs, root fallbacks, or
upstream parameter/default tables here. Before executing any operation, consume
`<output_path>/setup-preflight.json` and confirm the op appears in
`usdOptimize.operationsAvailable`.

**Companion docs:**
- [Scoped operations](../operations.md) — operation selection, validation, physical tolerances and execution.
- [`operations.json`](operations.json) — machine-readable catalog (same data as below), including the per-op `curation` block (generated `status` + authored `wired_into`; `rationale` only on overrides).
- [`usd-optimize` upstream handoff](../upstreams/usd-optimize.md) — central upstream operation-guide and prebuilt package resolution.

**Loss class.** `lossless` reorganizes / dedups / regenerates derived data only.
`bounded-loss` removes or modifies authored content (the agent should confirm
with the user before running). `analysis-only` is read-only (`context.analysisMode = 1`).

## Geometry
| Operation | Key | Args | Loss | Risk | Confirm | Pipelines |
|---|---|---|---|---|---|---|
| Dice Meshes | `diceMeshes` | 22 | bounded-loss | medium | yes | — |
| Fit Primitives | `fitPrimitives` | 22 | bounded-loss | medium | yes | — |
| Split Meshes | `splitMeshes` | 16 | lossless | low | no | — |
| Primitives to Meshes | `primitivesToMeshes` | 13 | lossless | low | no | — |
| Shrinkwrap | `shrinkwrap` | 13 | bounded-loss | high | yes | — |
| De-duplicate Geometry | `deduplicateGeometry` | 12 | lossless | low | no | `safe-cleanup`, `memory-reduction`, `mesh-count-reduction` |
| Mesh Cleanup | `meshCleanup` | 11 | lossless | low | yes | `mesh-count-reduction`, `data-quality-baseline` |
| Decimate Meshes | `decimateMeshes` | 8 | bounded-loss | medium | yes | `mesh-count-reduction` |
| Generate Normals | `generateNormals` | 6 | lossless | low | no | `data-quality-baseline` |
| Merge Vertices | `mergeVertices` | 5 | lossless | low | no | — |
| Subdivide Meshes | `subdivideMeshes` | 5 | lossless | low | no | — |
| Remesh Meshes | `remeshMeshes` | 4 | bounded-loss | high | yes | — |
| Remove Small Geometry | `removeSmallGeometry` | 4 | bounded-loss | medium | yes | `mesh-count-reduction` |
| Triangulate Meshes | `triangulateMeshes` | 2 | lossless | low | no | — |
| Manifold Meshes | `manifoldMeshes` | 1 | bounded-loss | medium | yes | — |
| Sparse Meshes | `sparseMeshes` | 0 | bounded-loss | medium | yes | — |

## Hierarchy
| Operation | Key | Args | Loss | Risk | Confirm | Pipelines |
|---|---|---|---|---|---|---|
| Remove Prims | `removePrims` | 8 | bounded-loss | high | yes | — |
| De-duplicate Hierarchies | `deduplicateHierarchies` | 4 | lossless | medium | yes | `memory-reduction`, `mesh-count-reduction`, `instancing` |
| Prune Leaves | `pruneLeaves` | 4 | lossless | low | no | `safe-cleanup`, `memory-reduction`, `load-time-reduction` |
| Flatten Hierarchy | `flattenHierarchy` | 2 | lossless | medium | no | — |
| Organize Prototypes | `organizePrototypes` | 2 | lossless | low | no | — |
| Delete Prims | `deletePrims` | 1 | bounded-loss | high | yes | — |
| Delete Hidden Prims | `deleteHiddenPrims` | 0 | bounded-loss | medium | yes | — |
| Optimize Skeleton Roots | `optimizeSkelRoots` | 0 | lossless | low | no | — |
| Remove Untyped Prims | `removeUntypedPrims` | 0 | bounded-loss | low | yes | — |

## Materials
| Operation | Key | Args | Loss | Risk | Confirm | Pipelines |
|---|---|---|---|---|---|---|
| Optimize Materials | `optimizeMaterials` | 4 | lossless | low | no | `safe-cleanup`, `memory-reduction`, `load-time-reduction` |
| Move Materials | `moveMaterials` | 2 | lossless | medium | yes | — |

## Uv
| Operation | Key | Args | Loss | Risk | Confirm | Pipelines |
|---|---|---|---|---|---|---|
| generateAtlasUVs | `generateAtlasUVs` | 7 | lossless | medium | no | — |
| Generate Projection UVs | `generateProjectionUVs` | 7 | lossless | low | no | — |
| Remove Unused UVs | `removeUnusedUVs` | 3 | lossless | low | no | — |

## Metadata
| Operation | Key | Args | Loss | Risk | Confirm | Pipelines |
|---|---|---|---|---|---|---|
| Optimize Primvars | `optimizePrimvars` | 6 | lossless | low | no | — |
| Optimize Time Samples | `optimizeTimeSamples` | 6 | lossless | low | no | `safe-cleanup`, `load-time-reduction` |
| Edit Stage Metrics | `editStageMetrics` | 5 | lossless | low | no | — |
| Remove Attributes | `removeAttributes` | 3 | bounded-loss | medium | yes | — |
| Compute Extents | `computeExtents` | 1 | lossless | low | no | `safe-cleanup`, `load-time-reduction`, `data-quality-baseline` |

## Transform
| Operation | Key | Args | Loss | Risk | Confirm | Pipelines |
|---|---|---|---|---|---|---|
| Merge Static Meshes | `merge` | 16 | bounded-loss | high | yes | `mesh-count-reduction` |
| Box Clip | `boxClip` | 13 | bounded-loss | high | yes | — |
| Compute Pivot | `pivot` | 4 | lossless | low | no | — |

## Analysis
| Operation | Key | Args | Loss | Risk | Confirm | Pipelines |
|---|---|---|---|---|---|---|
| Find Occluded Meshes | `findOccludedMeshes` | 7 | analysis-only | medium | yes | — |
| Find Overlapping Meshes | `findOverlappingMeshes` | 5 | analysis-only | low | no | — |
| Find Coinciding Geometry | `findCoincidingGeometry` | 4 | analysis-only | low | no | — |
| Count Vertices | `countVertices` | 3 | analysis-only | low | no | — |
| Find Flat Hierarchies | `findFlatHierarchies` | 3 | analysis-only | low | no | — |
| Print Stats | `printStats` | 3 | analysis-only | low | no | — |
| RTX Mesh Count | `rtxMeshCount` | 1 | analysis-only | low | no | — |

## Utility
| Operation | Key | Args | Loss | Risk | Confirm | Pipelines |
|---|---|---|---|---|---|---|
| Generate Scene | `generateScene` | 12 | lossless | low | no | — |
| Utility Function | `utilityFunction` | 2 | lossless | low | no | — |
| Python Script | `pythonScript` | 1 | bounded-loss | high | yes | — |

> **Security warning — `pythonScript` executes arbitrary Python code** with the privileges of the running process. Never agent-initiated (specialty escape-hatch; see `CLASSIFICATION.md`): require a user-supplied or reviewed script and explicit confirmation before execution (`operation-safety.md`).

## Summary

Total operations: **48**
- geometry: 16
- hierarchy: 9
- materials: 2
- uv: 3
- metadata: 5
- transform: 3
- analysis: 7
- utility: 3

## Catalog currency

Use the live `operationsAvailable` list from setup-preflight to establish available operations.
The catalog records supported operation names; release snapshots used by development tests do
not establish what the current runtime provides.

The `Args` column renders `args_count` from the catalog. The observed argument
counts behind those values are recorded per runtime in
[`registry-args.json`](registry-args.json), read from the live operation
registry with UI group headers excluded. The two are compared in the test
suite, so a catalog value that disagrees with the recorded registry fails.

When the pinned install version moves, refresh both artifacts against the new
runtime and check in the results, so the catalog's availability examples and
argument counts stay representative.
