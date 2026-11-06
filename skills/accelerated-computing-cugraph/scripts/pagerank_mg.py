# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""26.10 API example: torchrun across nodes or torch multiprocessing on one node.

Input: one CSV per global rank, src,dst,weight; unique directed edges and no
isolated vertices. Output: one CSV per rank. See the accompanying walkthrough.
GPU runtime validation is required before production use.
"""
import argparse
from datetime import timedelta
import os
from pathlib import Path


def run(args):
    import cupy as cp
    import cudf
    import pylibcugraph as plc
    from pylibcugraph.comms import (
        cugraph_comms_create_unique_id,
        cugraph_comms_get_raft_handle,
        cugraph_comms_init,
        cugraph_comms_shutdown,
    )
    import torch
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank >= cp.cuda.runtime.getDeviceCount():
        raise ValueError("LOCAL_RANK must index a visible GPU; see device mapping guidance")
    cp.cuda.Device(local_rank).use()
    torch.cuda.set_device(local_rank)
    # Gloo is only the control plane for UID exchange and readiness checks.
    dist.init_process_group("gloo", timeout=timedelta(seconds=args.timeout))
    initialized = False
    graph = handle = raft_handle = ids = scores = None
    try:
        # All ranks report local I/O failures before entering cuGraph collectives.
        error = None
        edges = None
        try:
            edges = cudf.read_csv(
                args.input.format(rank=rank),
                dtype={"src": "int64", "dst": "int64", "weight": "float32"},
            )
            if set(edges.columns) != {"src", "dst", "weight"}:
                raise ValueError("CSV must contain exactly src,dst,weight")
            if bool(edges.isnull().any().any()):
                raise ValueError("Null endpoints or weights are not supported by this example")
            if not bool(cp.isfinite(edges["weight"].values).all()) or bool(
                (edges["weight"] < 0).any()
            ):
                raise ValueError("Weights must be finite and nonnegative")
            Path(args.output).mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            error = f"rank {rank}: {exc}"
        errors = [None] * world_size
        dist.all_gather_object(errors, error)
        if any(errors):
            raise RuntimeError(f"Input/output preparation failed: {errors}")

        uid = [cugraph_comms_create_unique_id() if rank == 0 else None]
        dist.broadcast_object_list(uid, src=0)
        cugraph_comms_init(rank, world_size, uid[0], device=local_rank)
        initialized = True
        raft_handle = cugraph_comms_get_raft_handle()
        handle = plc.ResourceHandle(raft_handle.getHandle())
        # Lists of arrays match num_arrays=1 and avoid array truth-value coercion.
        src, dst, weight = (edges[c].values for c in ("src", "dst", "weight"))
        graph = plc.MGGraph(
            resource_handle=handle,
            graph_properties=plc.GraphProperties(is_symmetric=False, is_multigraph=False),
            src_array=[src], dst_array=[dst], weight_array=[weight],
            num_arrays=1, store_transposed=True,
            # 26.10's expensive MG constructor check calls len(None) for
            # omitted edge IDs. Inputs are checked above; see version notes.
            do_expensive_check=False,
        )
        ids, scores = plc.pagerank(
            resource_handle=handle, graph=graph,
            precomputed_vertex_out_weight_vertices=None,
            precomputed_vertex_out_weight_sums=None,
            initial_guess_vertices=None, initial_guess_values=None,
            alpha=0.85, epsilon=1e-6, max_iterations=500,
            do_expensive_check=True, fail_on_nonconvergence=True,
        )
        cp.cuda.runtime.deviceSynchronize()
        result = cudf.DataFrame({"vertex": ids, "pagerank": scores})
        result.to_csv(str(Path(args.output) / f"part-{rank}.csv"), index=False)
        del result
        dist.barrier()
    finally:
        # Release graph/resources while their underlying RAFT/NCCL state is alive.
        ids = scores = graph = handle = raft_handle = None
        if initialized:
            cugraph_comms_shutdown()
        dist.destroy_process_group()


def spawned_worker(local_rank, args):
    os.environ.update(
        RANK=str(local_rank), WORLD_SIZE=str(args.spawn), LOCAL_RANK=str(local_rank),
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(args.port),
    )
    run(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV pattern containing {rank}")
    parser.add_argument("--output", required=True, help="Fresh output directory")
    parser.add_argument("--spawn", type=int, help="Spawn this many local GPU processes")
    parser.add_argument("--port", type=int, default=29501)
    parser.add_argument("--timeout", type=int, default=180, help="Gloo control timeout in seconds")
    args = parser.parse_args()
    if "{rank}" not in args.input:
        parser.error("--input must contain {rank}; do not duplicate the whole graph on every rank")
    if args.timeout <= 0 or (args.spawn is not None and args.spawn <= 0):
        parser.error("--timeout and --spawn must be positive")
    if args.spawn:
        if "RANK" in os.environ:
            parser.error("Use either --spawn or torchrun, not both")
        import torch.multiprocessing as mp
        mp.spawn(spawned_worker, args=(args,), nprocs=args.spawn, join=True)
    else:
        run(args)


if __name__ == "__main__":
    main()
