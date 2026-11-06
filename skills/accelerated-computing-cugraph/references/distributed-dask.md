# Transitional Dask multi-GPU and multi-node analytics

<!-- SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0 -->
<!-- Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

Use this path to maintain existing `cugraph.dask` workflows while planning migration to [pylibcugraph with explicit communication](distributed-pylibcugraph.md). The planned deprecation concerns the cuGraph Dask API; do not generalize it to all Dask applications. These commands target 26.10 and require runtime validation on the target cluster.

## Prepare

Install mutually compatible 26.10 cuGraph/cuDF/Dask-cuDF/Dask-CUDA/RAFT/RMM, distributed, UCXX, NCCL and CUDA dependencies using the RAPIDS installation selector. Verify versions match on scheduler, driver and workers. Use a dedicated fixed-size allocation with one worker process per GPU and no adaptive scaling during graph collectives. All workers must see the input/output paths and have network connectivity to each other, not only to the scheduler. Choose TCP/UCXX and RDMA options according to the pinned Dask-CUDA release and cluster; do not copy transport flags from unrelated releases.

The input contract matches the direct walkthrough: unique directed edges, numeric IDs, finite nonnegative weights, no isolates in the edge-only example. Preserve duplicates/isolates deliberately when adapting real workloads.

## Single node

From the skill directory:

```bash
python scripts/smoke_fixture.py create /tmp/cugraph-dask-input --ranks 2
CUDA_VISIBLE_DEVICES=0,1 python scripts/pagerank_dask.py \
  --workers 2 --input '/tmp/cugraph-dask-input/edges-*.csv' \
  --output /tmp/cugraph-dask-output
```

[pagerank_dask.py](../scripts/pagerank_dask.py) creates `LocalCUDACluster`, waits for the expected workers, initializes cuGraph communications with `p2p=True`, persists an edge DataFrame with one partition per worker, constructs a distributed graph with `store_transposed=True`, and runs `cugraph.dask.pagerank`. It writes results as distributed Parquet before destroying communicators and closing its client/cluster. It does not shut down an externally managed cluster.

## Two nodes, two GPUs each

Create four shards in shared storage. Replace the example hostname and paths with your allocation. On node 0, start a scheduler in a separate job step/process:

```bash
dask scheduler --host gpu-node-0 --port 8786
```

On **each** GPU node, start two workers (this command creates one worker per visible GPU):

```bash
CUDA_VISIBLE_DEVICES=0,1 dask cuda worker tcp://gpu-node-0:8786
```

From a driver with the same environment:

```bash
python scripts/smoke_fixture.py create /shared/cugraph-dask/input --ranks 4
python scripts/pagerank_dask.py --scheduler tcp://gpu-node-0:8786 \
  --workers 4 --input '/shared/cugraph-dask/input/edges-*.csv' \
  --output /shared/cugraph-dask/output
```

Keep the scheduler and workers running until the driver finishes. Stop their allocated processes through your scheduler when done. Do not expose an unauthenticated Dask scheduler to an untrusted network. Use the cluster's normal security configuration.

For this tiny fixture only, read the Parquet output on one GPU and export a CSV for the common checker:

```python
from pathlib import Path
import cudf
out = Path('/tmp/cugraph-dask-check')
out.mkdir(exist_ok=False)
cudf.read_parquet('/shared/cugraph-dask/output').to_csv(
    str(out / 'part-0.csv'), index=False,
)
```

```bash
python scripts/smoke_fixture.py check /tmp/cugraph-dask-check
```

The single-node result can be checked the same way using `/tmp/cugraph-dask-output`. Do not gather a production result to the driver just to inspect it; compute distributed summary checks and write partitioned outputs.

## Migration checkpoints

Keep edge/vertex schemas, ID mapping, graph properties, convergence parameters, and validation fixtures stable while replacing the execution layer. Dask owns scheduling and communicator setup in the old path. In the direct path you own rank/device assignment, disjoint input shards, NCCL UID exchange, handle lifetimes, collective ordering, and coordinated failure termination. Compare SG, Dask MG, and direct MG by vertex ID before measuring speed or scaling.
