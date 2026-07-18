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
`.agents/operations/<operation-key>.md`, then resolve it under the selected
Usd Optimize package root. Do not duplicate package URLs, root fallbacks, or
upstream parameter/default tables here. Before executing any operation, consume
`<output_path>/setup-preflight.json` and confirm the op appears in
`usdOptimize.operationsAvailable`.

**Companion docs:**
- [Execution reference](EXECUTION.md) — docs-class wrapper/API invocation shape, batch orchestration, and validator import variants.
- [Classification rubric](CLASSIFICATION.md) — curation tiers and the canonical-over-specialty selection rule.
- [`pipelines.md`](../usd-optimize-run-operations/references/pipelines.md) — curated multi-op chains organized by bottleneck.
- [`operations.json`](operations.json) — machine-readable catalog (same data as below), including the per-op `curation` block (generated `status` + authored `wired_into`; `rationale` only on overrides).
- [`usd-optimize` upstream handoff](../upstreams/usd-optimize.md) — central upstream operation-guide and prebuilt package resolution.

**Loss class.** `lossless` reorganizes / dedups / regenerates derived data only.
`bounded-loss` removes or modifies authored content (the agent should confirm
with the user before running). `analysis-only` is read-only (`context.analysisMode = 1`).


## Geometry
| Operation | Key | Args | Loss | Risk | Confirm | Pipelines |
|---|---|---|---|---|---|---|
| Dice Meshes | `diceMeshes` | 22 | bounded-loss | medium | yes | — |
| Fit Primitives | `fitPrimitives` | 20 | bounded-loss | high | yes | — |
| Split Meshes | `splitMeshes` | 16 | lossless | low | no | — |
| Primitives to Meshes | `primitivesToMeshes` | 13 | lossless | low | no | — |
| Mesh Cleanup | `meshCleanup` | 11 | lossless | low | yes | `mesh-count-reduction`, `data-quality-baseline` |
| De-duplicate Geometry | `deduplicateGeometry` | 9 | lossless | low | no | `safe-cleanup`, `memory-reduction`, `mesh-count-reduction` |
| Decimate Meshes | `decimateMeshes` | 8 | bounded-loss | medium | yes | `mesh-count-reduction` |
| Shrinkwrap | `shrinkwrap` | 7 | bounded-loss | high | yes | — |
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
| Prune Leaves | `pruneLeaves` | 3 | lossless | low | no | `safe-cleanup`, `memory-reduction`, `load-time-reduction` |
| Flatten Hierarchy | `flattenHierarchy` | 2 | lossless | medium | no | — |
| Organize Prototypes | `organizePrototypes` | 2 | lossless | low | no | — |
| Delete Prims | `deletePrims` | 1 | bounded-loss | high | yes | — |
| De-duplicate Hierarchies | `deduplicateHierarchies` | 0 | lossless | medium | yes | `memory-reduction`, `mesh-count-reduction`, `instancing` |
| Delete Hidden Prims | `deleteHiddenPrims` | 0 | bounded-loss | medium | yes | — |
| Optimize Skeleton Roots | `optimizeSkelRoots` | 0 | lossless | low | no | — |
| Remove Untyped Prims | `removeUntypedPrims` | 0 | bounded-loss | low | yes | — |

## Materials
| Operation | Key | Args | Loss | Risk | Confirm | Pipelines |
|---|---|---|---|---|---|---|
| Optimize Materials | `optimizeMaterials` | 4 | lossless | low | no | `safe-cleanup`, `memory-reduction`, `load-time-reduction` |

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
| Edit Stage Metrics | `editStageMetrics` | 4 | lossless | low | no | — |
| Remove Attributes | `removeAttributes` | 3 | bounded-loss | medium | yes | — |
| Compute Extents | `computeExtents` | 1 | lossless | low | no | `safe-cleanup`, `load-time-reduction`, `data-quality-baseline` |

## Transform
| Operation | Key | Args | Loss | Risk | Confirm | Pipelines |
|---|---|---|---|---|---|---|
| Merge Static Meshes | `merge` | 14 | bounded-loss | high | yes | — |
| Box Clip | `boxClip` | 11 | bounded-loss | high | yes | — |
| Compute Pivot | `pivot` | 4 | lossless | low | no | — |

## Analysis
| Operation | Key | Args | Loss | Risk | Confirm | Pipelines |
|---|---|---|---|---|---|---|
| Find Occluded Meshes | `findOccludedMeshes` | 7 | analysis-only | medium | yes | — |
| Find Coinciding Geometry | `findCoincidingGeometry` | 4 | analysis-only | low | no | — |
| Find Overlapping Meshes | `findOverlappingMeshes` | 4 | analysis-only | low | no | — |
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

## Summary

Total operations: **47**
- geometry: 16
- hierarchy: 9
- materials: 1
- uv: 3
- metadata: 5
- transform: 3
- analysis: 7
- utility: 3

## Catalog currency

The checked-in probe snapshot (`probe-snapshots/usd-optimize-1.0.4.json`)
reflects usd-optimize 1.0.4, captured live from the GitHub release package.
It is not authoritative at runtime:
the live `operationsAvailable` list from the session's setup-preflight always
wins. When the pinned install version moves, refresh the snapshot (re-run the
setup probe against the new runtime and check in the emitted JSON) so the
catalog's availability examples stay representative.
