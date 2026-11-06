# Distributed PageRank without a Dask scheduler

<!-- SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0 -->
<!-- Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

Use [pagerank_mg.py](../scripts/pagerank_mg.py) for the 26.10 API walkthrough. It supports `torch.multiprocessing.spawn` on one node and `torchrun` on multiple nodes. This is a source-checked example awaiting GPU execution validation, not a production launcher.

## Environment and communication

Use Linux NVIDIA GPU nodes with mutually compatible CUDA/driver, cuGraph 26.10, pylibcugraph 26.10, cuDF, CuPy, RMM, pylibraft, raft-dask, UCXX, and PyTorch. Resolve packages together using the RAPIDS installation selector, then record the solved environment. Use the same versions on every node. The 26.10 communicator helper imports `raft_dask.common.nccl` and `inject_comms_on_handle_coll_only`; removing the Dask scheduler does **not** eliminate that package dependency.

There are two distinct communication layers:

1. PyTorch Gloo exchanges a single cuGraph NCCL unique ID, validates rank readiness, and provides control barriers. It does not perform graph computation.
2. `cugraph_comms_init(rank, world_size, uid, device)` creates the cuGraph NCCL/RAFT communicator and subcommunicators. Wrap `cugraph_comms_get_raft_handle().getHandle()` in `pylibcugraph.ResourceHandle`. An uninitialized `ResourceHandle()` is insufficient for MGGraph.

Assign one process to each GPU. The supplied script assumes every process on a node sees the same ordered `CUDA_VISIBLE_DEVICES` list, so `LOCAL_RANK` selects a visible device. If a scheduler masks each process to only one GPU, adapt the device index to zero rather than using a global or local rank as a physical device ordinal. Keep global ranks unique across nodes.

Use trusted allocated nodes: object broadcast uses pickle. Configure the cluster's allowed interfaces, reachable rendezvous port, and NCCL transport according to its networking policy. Test NCCL connectivity on the allocation before diagnosing graph code. Gloo timeout bounds control operations; it does not bound every CUDA/NCCL collective. Use a scheduler walltime or job supervisor to terminate the entire job on a stalled rank.

## Input and output contract

For this runnable example, provide one CSV per global rank named `edges-<rank>.csv`, with `src,dst,weight`. IDs are int64; weights are finite nonnegative float32; edges are unique directed pairs; all vertices appear as an endpoint. Every logical edge occurs on exactly one rank. The constructor redistributes these arbitrary disjoint edge partitions into its internal graph layout. Do not pre-slice independent subgraphs and run SG PageRank on each.

The example checks schema, nulls, weights, and local file access before graph collectives. It assumes global deduplication and consistent encoding have already been done. For production inputs:

- Assign shards using a manifest with disjoint coverage; load directly into each GPU process.
- Preserve a globally consistent mapping for string/composite IDs.
- For isolates, supply disjoint `vertices_array` partitions using the release's MGGraph contract; edge-only CSVs cannot encode them. Verify this adaptation against an SG/CPU oracle before deployment.
- Empty ranks still participate, using empty arrays with the same dtypes. Lists of arrays and `num_arrays=1` avoid ambiguous array truth-value coercion in the 26.10 wrapper.
- The 26.10 MGGraph expensive-check branch calls `len(edge_id_array[i])` when optional edge IDs are omitted. This example disables that **constructor** check and keeps the PageRank expensive check enabled. Recheck this workaround on upgrades; do not disable validation broadly.

Outputs are local result partitions, not copies of the full result. The script writes `part-<global-rank>.csv`; join by returned vertex ID, never by input partition or row position. Use a new output directory per run. The example converts output for small walkthroughs; use a GPU/distributed output format for large production results.

## One node, two GPUs

Run from the skill directory, with a matching RAPIDS environment active:

```bash
python scripts/smoke_fixture.py create /tmp/cugraph-smoke-2 --ranks 2
CUDA_VISIBLE_DEVICES=0,1 python scripts/pagerank_mg.py \
  --spawn 2 --port 29501 \
  --input '/tmp/cugraph-smoke-2/edges-{rank}.csv' \
  --output /tmp/cugraph-result-2
python scripts/smoke_fixture.py check /tmp/cugraph-result-2
```

`spawn` creates fresh CUDA processes; do not use `fork` after CUDA initialization. An equivalent single-node launch is:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  scripts/pagerank_mg.py --input '/tmp/cugraph-smoke-2/edges-{rank}.csv' \
  --output /tmp/cugraph-result-torchrun
```

## Two nodes, two GPUs each

Create the four input shards once on shared storage. Synchronize code and the environment to both nodes. Replace `/shared/cugraph-demo` with an allocated shared path and `gpu-node-0` with a hostname reachable from both nodes.

```bash
python scripts/smoke_fixture.py create /shared/cugraph-demo/input --ranks 4
```

On node 0:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=2 --nproc-per-node=2 \
  --node-rank=0 --master-addr=gpu-node-0 --master-port=29500 \
  scripts/pagerank_mg.py \
  --input '/shared/cugraph-demo/input/edges-{rank}.csv' \
  --output /shared/cugraph-demo/output
```

On node 1, execute the same command with `--node-rank=1`. Both commands must be running concurrently. After both succeed:

```bash
python scripts/smoke_fixture.py check /shared/cugraph-demo/output
```

Without shared storage, stage the relevant shards to identical local input paths and collect every output part before validation. Do not let every node independently assign global ranks starting at zero.

## Python multiprocessing alternative

The collective graph code does not require PyTorch tensors. A pure-Python launcher can use `multiprocessing.get_context("spawn")`, distribute one NCCL UID through a queue/pipe on a single node, and pass global rank, world size, UID, and local device to each worker. Replace the Gloo readiness exchange and barriers with explicit parent supervision. A local multiprocessing queue does **not** connect separate machines; multi-node launch additionally needs a trusted rendezvous service or scheduler/MPI coordination. Prefer the supplied torchrun path until that control plane is implemented and tested; do not pretend a Python Pool alone is multi-node execution.

## Validate and diagnose

The fixture is one directed 12-vertex cycle crossing partition boundaries. Its exact PageRank is 1/12 per vertex. The checker rejects missing/duplicate IDs, nonfinite values, wrong scores, and wrong total mass. Repeat with one GPU, two GPUs, and two nodes; also test more ranks than vertices to exercise empty input shards where the release permits them.

This fixture does not establish production correctness. Add a nonuniform weighted graph with a dangling vertex, non-contiguous IDs, isolates via the vertices input, and an explicit CPU reference. Compare SG and MG results after joining IDs, within tolerances. Inject an unreadable shard to verify the readiness failure propagates to all ranks. Then test job-level failure termination under the scheduler before larger runs.

If a job hangs: first check that all ranks joined the same job, see distinct GPUs, broadcast the same cuGraph UID, and enter collectives in the same order. Check network/transport errors and whether a rank failed on I/O or OOM. Abort the whole job after the configured deadline rather than restarting one rank inside an active communicator. Resource cleanup is best effort after a failed collective; the supervisor owns whole-job termination.
