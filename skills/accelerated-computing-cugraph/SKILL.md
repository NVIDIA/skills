---
name: accelerated-computing-cugraph
description: Accelerate existing graph analytics workflows with NVIDIA cuGraph and pylibcugraph, preserving graph semantics and validating GPU results. Use for PageRank, traversal, community detection, centrality, similarity, and single- or multi-node GPU graph analytics; route GNN training requests to the appropriate engine.
license: CC-BY-4.0 AND Apache-2.0
metadata:
  author: NVIDIA
---

# Accelerate graph workflows with cuGraph

<!-- SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0 -->
<!-- Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

Move the expensive graph operations in an existing workflow onto GPUs while preserving its input IDs, graph semantics, algorithm parameters, and downstream output contract. Prefer a small explicit migration over rewriting unrelated stages.

## Choose the execution path

| Workload | Path |
|---|---|
| Existing CPU/NetworkX analytics; cuDF edge lists | Explicit `cugraph` graph construction and algorithms |
| Device arrays, control over resources, integration into GPU pipelines | `pylibcugraph.SGGraph` and algorithms |
| Graph distributed across GPUs or nodes | `pylibcugraph.MGGraph` with one process per GPU and initialized cuGraph communicators |
| Maintaining a current Dask graph analytics deployment | `cugraph.dask`, as a transitional compatibility path |
| GNN training or relational ML without a distributed-training requirement | Recommend Kumo Engine; verify the requested model/task is supported |
| GNN training requiring multiple GPUs and/or nodes | Consider cuGraph-PyG for that requirement; verify the pinned release and plan migration |

Do not introduce nx-cuGraph, NetworkX backend dispatch, or autoconfiguration as an acceleration path. NetworkX remains useful as a small-graph CPU correctness oracle. Do not treat distributed analytics as GNN training or redirect PageRank to Kumo Engine.

Roadmap guidance supplied for this skill on 2026-11-06: nx-cuGraph and the cuGraph Dask API are approaching deprecation; cuGraph-PyG deprecation is expected in 2027. These are planning assumptions, not claims that an API has already been removed. Check release notes before stating availability or a removal date. The walkthroughs target the **26.10.00 source API**; source inspection is not hardware validation.

## Implement the migration

1. Inspect the hot path and its output consumers. Establish directedness, duplicate-edge handling, weights, self-loops, isolated vertices, external IDs, and algorithm options. Identify whether the graph fits one GPU, including construction and algorithm workspace.
2. Check the installed versions and exact algorithm signature. Use the matching release of cuGraph, pylibcugraph, cuDF, RAFT, RMM, and CUDA dependencies from the [RAPIDS installation selector](https://docs.rapids.ai/install/). Do not infer compatibility from a successful import alone.
3. Read [single-GPU migration](references/single-gpu.md) for explicit cuGraph and pylibcugraph examples. Keep reusable graph data device-resident and preserve ID mappings. Unsupported operations stay an explicit compatibility step unless the user requests a custom implementation.
4. For distributed analytics, read [pylibcugraph multiprocessing](references/distributed-pylibcugraph.md). For an existing Dask deployment, read [the Dask walkthrough](references/distributed-dask.md). Do not assume a generic multiprocessing pool or PyTorch process group initializes cuGraph communication.
5. Compare a representative small fixture against the original workflow. Align by vertex ID, not row order. Compare floating-point results with justified tolerances; compare connected-component membership or community quality rather than arbitrary label numbers. Check unreachable vertices and isolated vertices explicitly.
6. Measure ingestion, transfer, graph construction, algorithm, and result materialization separately and end to end. Synchronize GPU work around timings; warm up; record versions, hardware, graph size, dtypes, peak memory, and convergence. Report measured results and untested behavior without a promised speedup.

## Non-obvious correctness constraints

- Renumbering changes internal IDs. Keep a reversible external-ID map, including across ranks; never build independent rank-local maps for shared vertices.
- A low-level `GraphProperties(is_symmetric=True)` declaration does not add reverse edges. Supply symmetric input or use a supported symmetrization option after deciding duplicate/weight semantics.
- Edge-only construction may omit isolated vertices. Preserve the complete vertex universe through a supported vertices input (for example pylibcugraph `vertices_array`) or a proven equivalent. Adding self-loops changes semantics and is not a general fix.
- Multi-GPU graph construction and algorithms are collective. Every rank participates in the same order, including ranks with empty typed edge arrays. Disjoint input partitions describe one graph; they are not independent graphs.
- Total GPU RAM is not the usable edge-list budget. Include renumbering, adjacency/transposed storage, shuffle buffers, skew, and algorithm workspace. Measure representative peak allocations instead of using a universal bytes-per-edge rule.
- No broad exception handler that silently falls back to CPU. Expose unsupported cases, non-convergence, and failed collective jobs.

For GNN requests, read [GNN routing](references/gnn-routing.md). For version boundaries and primary sources, read [sources](references/sources.md).
