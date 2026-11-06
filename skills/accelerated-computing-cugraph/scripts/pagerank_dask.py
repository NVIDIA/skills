# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Transitional cuGraph Dask PageRank example targeting the 26.10 API."""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV glob readable by all workers")
    parser.add_argument("--output", required=True, help="Fresh Parquet output directory")
    parser.add_argument("--scheduler", help="Existing Dask scheduler address")
    parser.add_argument("--workers", type=int, required=True, help="Expected GPU worker count")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    import cugraph
    import cugraph.dask as dcg
    import cugraph.dask.comms.comms as comms
    import dask_cudf
    from dask_cuda import LocalCUDACluster
    from distributed import Client, wait

    cluster = None
    client = None
    initialized = False
    try:
        if args.scheduler:
            client = Client(args.scheduler)
        else:
            cluster = LocalCUDACluster(n_workers=args.workers)
            client = Client(cluster)
        client.wait_for_workers(args.workers, timeout=180)
        if len(client.scheduler_info()["workers"]) != args.workers:
            raise RuntimeError("Use a dedicated fixed-size cluster with exactly --workers GPUs")
        comms.initialize(p2p=True)
        initialized = True
        edges = dask_cudf.read_csv(
            args.input, dtype={"src": "int64", "dst": "int64", "weight": "float32"}
        ).repartition(npartitions=args.workers).persist()
        wait(edges)
        graph = cugraph.Graph(directed=True)
        graph.from_dask_cudf_edgelist(
            edges, source="src", destination="dst", edge_attr="weight",
            renumber=True, store_transposed=True,
        )
        result = dcg.pagerank(graph, alpha=0.85, tol=1e-6, max_iter=500)
        # Keep production results distributed; compute only tiny fixtures for comparison.
        result.to_parquet(args.output, write_index=False)
    finally:
        if initialized:
            comms.destroy()
        if client is not None:
            client.close()
        if cluster is not None:
            cluster.close()


if __name__ == "__main__":
    main()
