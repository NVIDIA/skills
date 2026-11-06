# Single-GPU migration

<!-- SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0 -->
<!-- Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

## Translate the workload, not just the function name

| Existing operation | Explicit cuGraph starting point | Check before substitution |
|---|---|---|
| PageRank | `cugraph.pagerank` | Damping, weights, personalization, dangling handling, tolerance/convergence |
| Unweighted shortest paths | `cugraph.bfs` | Directedness, source, unreachable sentinel, predecessor output |
| Weighted shortest paths | `cugraph.sssp` | Weight domain and unreachable values; do not substitute BFS |
| Components | `cugraph.weakly_connected_components` or `strongly_connected_components` | Weak vs strong and release/SG/MG support |
| Communities | `cugraph.louvain` or `leiden` | Undirected input, weights, resolution, randomness, modularity |
| Betweenness | `cugraph.betweenness_centrality` | Sampling, normalization, endpoint and weight support |
| Candidate-pair similarity | `cugraph.jaccard` | Candidate pairs and scorer semantics; not Adamic–Adar |

Check the release-specific API; this table is not a guarantee of multi-GPU support. Preserve an unsupported operation explicitly rather than silently changing its mathematical meaning.

## cuGraph: DataFrame-oriented migration

This example is a directed weighted graph with no isolated vertices. For an existing NetworkX workflow, extract edges and node IDs once; do not add a backend dependency. For large inputs, load edge partitions directly instead of first constructing a Python graph.

```python
import cudf
import cugraph

edges = cudf.DataFrame({
    "src": [10, 20, 30, 30],
    "dst": [20, 30, 10, 20],
    "weight": [1.0, 2.0, 1.0, 3.0],
})
edges["weight"] = edges["weight"].astype("float32")
graph = cugraph.Graph(directed=True)
graph.from_cudf_edgelist(
    edges, source="src", destination="dst", edge_attr="weight",
    renumber=True, store_transposed=True,
)
scores = cugraph.pagerank(graph, alpha=0.85, tol=1e-6, max_iter=500)
# High-level PageRank returns external vertex IDs; order is not an ID map.
# Materialize once, only if downstream consumers require host dictionaries.
result = scores.to_pandas().set_index("vertex")["pagerank"].to_dict()
```

For string/composite keys, cuGraph renumbering can handle supported high-level inputs. For low-level calls, encode keys to a shared integer domain and retain the reverse map. Compare the GPU result with the baseline joined by external vertex ID. Match the baseline's duplicate-edge aggregation before building the graph.

## pylibcugraph: device-array integration

```python
import cupy as cp
import pylibcugraph as plc

src = cp.asarray([0, 1, 2, 2], dtype=cp.int32)
dst = cp.asarray([1, 2, 0, 1], dtype=cp.int32)
weights = cp.asarray([1, 2, 1, 3], dtype=cp.float32)
vertices = cp.arange(4, dtype=cp.int32)  # includes isolated vertex 3
handle = plc.ResourceHandle()
graph = plc.SGGraph(
    resource_handle=handle,
    graph_properties=plc.GraphProperties(is_symmetric=False, is_multigraph=False),
    src_or_offset_array=src, dst_or_index_array=dst,
    weight_array=weights, vertices_array=vertices,
    store_transposed=True, renumber=True, do_expensive_check=True,
)
ids, scores = plc.pagerank(
    resource_handle=handle, graph=graph,
    precomputed_vertex_out_weight_vertices=None,
    precomputed_vertex_out_weight_sums=None,
    initial_guess_vertices=None, initial_guess_values=None,
    alpha=0.85, epsilon=1e-6, max_iterations=500,
    do_expensive_check=True, fail_on_nonconvergence=True,
)
# Keep handle, graph, and input owners alive until device work completes.
cp.cuda.runtime.deviceSynchronize()
```

The returned IDs identify scores; do not assume positional alignment with input rows. SGGraph uses `src_or_offset_array` / `dst_or_index_array`; MGGraph uses `src_array` / `dst_array`. The 26.10 MG constructor has no `renumber` or `num_edges` argument. Do not copy obsolete examples that supply them.

For a CPU oracle, force NetworkX's CPU implementation (no configured GPU backend) and compare small fixtures containing a dangling node, an isolated node, non-contiguous IDs, and weighted edges. NetworkX and cuGraph convergence tolerances are not necessarily the same stopping rule. Validate scores at an agreed output tolerance rather than claiming bitwise equality.
