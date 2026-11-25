#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal NCCL all-reduce probe — runs IN the container as a cheap 2-node/2-GPU
job to prove the cluster's NCCL rendezvous works BEFORE a real multi-node run
burns GPU-hours hanging on the first collective.

It reads the same rendezvous env the multi-node templates export (WORLD_SIZE =
NODE COUNT, NUM_GPU_PER_NODE, NODE_RANK, plus torchrun's LOCAL_RANK and
MASTER_ADDR/MASTER_PORT), computes the true GLOBAL world-size/rank, does one
all-reduce, and prints NCCL_PROBE_OK. Launchers that invoke this probe under
``torchrun`` must preserve the TAO values in ``TAO_NODE_COUNT``,
``TAO_GPUS_PER_NODE``, and ``TAO_NODE_RANK`` because torchrun overwrites the
standard ``WORLD_SIZE`` environment variable with the global process count.
If NCCL is misconfigured (e.g. a cluster-specific
intra-node P2P hang) it HANGS on all_reduce — the orchestrating skill wraps this
with a timeout and, on timeout, sets the cluster's NCCL knob (NCCL_P2P_DISABLE=1,
NCCL_SOCKET_IFNAME, ...) and re-probes, caching the working env per cluster.

`--dry-run` prints the computed rendezvous config WITHOUT importing torch or
touching a GPU — the node-count->global-rank math (the easy thing to get wrong
given WORLD_SIZE is TAO's node count, not the global rank count) is testable
offline. torch is imported lazily so this file loads on a CPU host.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def rendezvous_config(env: dict | None = None) -> dict:
    """Compute the GLOBAL torch.distributed config from the TAO rendezvous env.

    WORLD_SIZE in the env is TAO's NODE COUNT; the true global world size is
    node_count * gpus_per_node, and the global rank is
    node_rank * gpus_per_node + local_rank.
    """
    e = os.environ if env is None else env
    # torchrun rewrites WORLD_SIZE to the global process count.  Preserve TAO's
    # node-count convention through explicit aliases when the probe is launched
    # beneath torchrun, while retaining the template variables as a fallback.
    node_count = int(e.get("TAO_NODE_COUNT", e.get("WORLD_SIZE", "1")))
    gpus_per_node = int(
        e.get("TAO_GPUS_PER_NODE", e.get("NUM_GPU_PER_NODE", "1"))
    )
    node_rank = int(e.get("TAO_NODE_RANK", e.get("NODE_RANK", "0")))
    local_rank = int(e.get("LOCAL_RANK", "0"))
    return {
        "global_world_size": node_count * gpus_per_node,
        "global_rank": node_rank * gpus_per_node + local_rank,
        "local_rank": local_rank,
        "node_count": node_count,
        "gpus_per_node": gpus_per_node,
        "master_addr": e.get("MASTER_ADDR", ""),
        "master_port": e.get("MASTER_PORT", "29500"),
    }


def _run(cfg: dict) -> int:
    import torch                       # lazy — only when actually probing on a GPU node
    import torch.distributed as dist

    torch.cuda.set_device(cfg["local_rank"])
    dist.init_process_group(
        backend="nccl", init_method="env://",
        world_size=cfg["global_world_size"], rank=cfg["global_rank"],
    )
    t = torch.ones(1, device=f"cuda:{cfg['local_rank']}")
    dist.all_reduce(t)                 # HANGS here if NCCL rendezvous is broken
    ok = int(t.item()) == cfg["global_world_size"]
    dist.barrier()
    if cfg["global_rank"] == 0:
        print("NCCL_PROBE_OK" if ok else
              f"NCCL_PROBE_BAD sum={int(t.item())} expected={cfg['global_world_size']}")
    dist.destroy_process_group()
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="print the computed rendezvous config and exit (no torch/GPU)")
    args = p.parse_args(argv)
    cfg = rendezvous_config()
    if args.dry_run:
        print(json.dumps(cfg))
        return 0
    return _run(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
