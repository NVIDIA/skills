#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic, network-free Run:ai MCP fixture for GPU utilization evals.

Two clusters share one server:

- LIVE_CLUSTER replays responses recorded on 2026-10-08 from a live Run:ai
  cluster through the Run:ai MCP server (identifiers sanitized). Its 16 GPUs
  are simulated: every allocated GPU reports 100% utilization, one GPU is busy
  outside Run:ai accounting, and the projects only hold GPUs from 10:24 UTC
  that day, so long windows have sparse series.
- LAB_CLUSTER is synthetic. Twelve projects hold GPUs across three GPU pools.
  Each utilization pattern repeats every four samples, so a series mean is
  exact whenever numberOfSamples is a multiple of four. Allocations reconcile
  per workload, project, department, node pool, and cluster.
- OLD_CLUSTER is synthetic and runs version 2.20, which rejects
  gpuMemoryUtilization in org-unit metrics, as the real backend does.

As on the real backend, get_cluster_metrics ignores nodePoolName for
allocation, capacity, and utilization metrics and returns cluster-wide series.
"""

from datetime import datetime, timedelta, timezone
import json
import math
import sys


LIVE_CLUSTER = "10000000-0000-4000-8000-0000000000a1"
LAB_CLUSTER = "30000000-0000-4000-8000-0000000000d1"
OLD_CLUSTER = "40000000-0000-4000-8000-0000000000e1"
NOW = {LIVE_CLUSTER: "2026-10-08T10:55:30Z", LAB_CLUSTER: "2026-10-12T09:00:00Z",
       OLD_CLUSTER: "2026-10-12T09:00:00Z"}
CLUSTER_VERSION = {LIVE_CLUSTER: "2.24", LAB_CLUSTER: "2.24", OLD_CLUSTER: "2.20"}
MEMORY_METRIC_MIN_VERSION = (2, 21)

NODE_POOLS = [
    {"clusterId": LIVE_CLUSTER, "isDefault": True, "networkTopologyName": None,
     "nodePoolId": "100000", "nodePoolName": "default", "phase": "Ready"},
    {"clusterId": LAB_CLUSTER, "isDefault": True, "networkTopologyName": None,
     "nodePoolId": "300001", "nodePoolName": "a100-train", "phase": "Ready"},
    {"clusterId": LAB_CLUSTER, "isDefault": False, "networkTopologyName": None,
     "nodePoolId": "300002", "nodePoolName": "h100-train", "phase": "Ready"},
    {"clusterId": LAB_CLUSTER, "isDefault": False, "networkTopologyName": None,
     "nodePoolId": "300003", "nodePoolName": "l40s-infer", "phase": "Ready"},
    {"clusterId": LAB_CLUSTER, "isDefault": False, "networkTopologyName": None,
     "nodePoolId": "300004", "nodePoolName": "cpu-pool", "phase": "Ready"},
    {"clusterId": OLD_CLUSTER, "isDefault": True, "networkTopologyName": None,
     "nodePoolId": "400001", "nodePoolName": "default", "phase": "Ready"},
]
POOL_IDS = {(pool["clusterId"], pool["nodePoolName"]): pool["nodePoolId"] for pool in NODE_POOLS}

POOL_HARDWARE = {
    LIVE_CLUSTER: {"default": ("Tesla-V100", 1, 16, 5)},
    LAB_CLUSTER: {"a100-train": ("NVIDIA-A100-SXM4-80GB", 8, 64, 8),
                  "h100-train": ("NVIDIA-H100-80GB-HBM3", 4, 32, 4),
                  "l40s-infer": ("NVIDIA-L40S", 4, 16, 4),
                  "cpu-pool": (None, 0, 0, 3)},
    OLD_CLUSTER: {"default": ("NVIDIA-A10", 2, 8, 3)},
}

# GPUs busy outside Run:ai accounting, in fully busy GPU equivalents. On the
# live cluster two 0.5-GPU pods outside any project share one device.
OUTSIDE_BUSY_GPUS = {(LIVE_CLUSTER, "default"): 1.0}

# The live cluster only recorded metrics inside these intervals.
LIVE_RECORDED = [("2026-10-07T10:00:00Z", "2026-10-07T20:00:00Z"),
                 ("2026-10-08T07:30:00Z", NOW[LIVE_CLUSTER])]

DEPARTMENTS = [
    {"clusterId": LIVE_CLUSTER, "id": "4500000", "name": "default", "gpuQuota": 6},
    {"clusterId": LIVE_CLUSTER, "id": "4500002", "name": "d1", "gpuQuota": 6},
    {"clusterId": LAB_CLUSTER, "id": "4600010", "name": "research", "gpuQuota": 48},
    {"clusterId": LAB_CLUSTER, "id": "4600011", "name": "platform", "gpuQuota": 24},
    {"clusterId": LAB_CLUSTER, "id": "4600012", "name": "applied", "gpuQuota": 4},
    {"clusterId": OLD_CLUSTER, "id": "4700010", "name": "legacy", "gpuQuota": 8},
]
DEPARTMENT_BY_ID = {dept["id"]: dept for dept in DEPARTMENTS}


def project(cluster_id, project_id, name, department, pool, deserved, limit, gpus,
            util=(0,), memory=(0,), *, since=None, missing=(), spike=None, every=1):
    """spike=(value, until) overrides utilization before `until`; every=k keeps every k-th sample
    of the project's own series, as sparse recording does on a live cluster."""
    dept = next(item for item in DEPARTMENTS
                if item["clusterId"] == cluster_id and item["name"] == department)
    return {"clusterId": cluster_id, "id": project_id, "name": name, "departmentId": dept["id"],
            "department": department, "pool": pool, "deserved": deserved, "limit": limit,
            "gpus": gpus, "util": list(util), "memory": list(memory), "since": since,
            "missing": set(missing), "spike": spike, "every": every}


LIVE_SINCE = "2026-10-08T10:24:43Z"

PROJECTS = [
    project(LIVE_CLUSTER, "4500004", "hc-team-a", "default", "default", 8, 16, 8, [100], [100], since=LIVE_SINCE),
    project(LIVE_CLUSTER, "4500005", "hc-team-b", "default", "default", 6, 16, 2, [100], [100], since=LIVE_SINCE),
    project(LIVE_CLUSTER, "4500006", "hc-team-c", "default", "default", 2, 4, 2, [100], [100], since=LIVE_SINCE),
    project(LIVE_CLUSTER, "4500001", "test", "default", "default", 1, None, 0),
    project(LIVE_CLUSTER, "4500003", "test2", "d1", "default", 1, None, 0),
    project(LAB_CLUSTER, "4600101", "vision-train", "research", "a100-train", 8, 24, 16, [2, 1, 3, 6], [21, 19, 20, 20]),
    project(LAB_CLUSTER, "4600102", "nlp-research", "research", "h100-train", 16, 24, 16, [8, 10, 28, 78], [35, 38, 52, 83]),
    project(LAB_CLUSTER, "4600103", "speech-asr", "research", "a100-train", 8, 16, 8, [52, 60, 64, 80], [58, 61, 63, 70]),
    project(LAB_CLUSTER, "4600104", "robotics-sim", "research", "h100-train", 8, 8, 8, [45, 46, 48, 49], [50, 50, 51, 53]),
    project(LAB_CLUSTER, "4600105", "genomics", "research", "a100-train", 4, 8, 2, [40], [45],
            missing={"gpuUtilization"}),
    project(LAB_CLUSTER, "4600106", "sandbox", "research", "a100-train", 2, 4, 0),
    project(LAB_CLUSTER, "4600201", "recsys", "platform", "a100-train", 6, 8, 6, [60, 70, 70, 88], [64, 66, 66, 72]),
    project(LAB_CLUSTER, "4600202", "llm-serving", "platform", "l40s-infer", 4, 4, 4, [1, 2, 1, 4], [86]),
    project(LAB_CLUSTER, "4600203", "chat-api", "platform", "l40s-infer", 4, 6, 4, [3, 5, 5, 35], [60, 60, 61, 62]),
    project(LAB_CLUSTER, "4600204", "ads-ranking", "platform", "h100-train", 4, 4, 3, [40, 50, 60, 70], [44, 50, 55, 60],
            missing={"gpuMemoryUtilization"}),
    project(LAB_CLUSTER, "4600205", "batch-etl", "platform", "a100-train", 2, 2, 1.5, [1, 2, 2, 3], [12]),
    project(LAB_CLUSTER, "4600206", "data-eng", "platform", "cpu-pool", 0, 0, 0),
    project(LAB_CLUSTER, "4600301", "doc-ocr", "applied", "a100-train", 1, 2, 1, [6, 8, 8, 14], [30]),
    project(LAB_CLUSTER, "4600302", "kb-search", "applied", "l40s-infer", 1, 1, 0.5, [0, 1, 1, 2], [25]),
    project(OLD_CLUSTER, "4700101", "legacy-cv", "legacy", "default", 4, 8, 4, [1, 1, 2, 4], [70]),
    project(OLD_CLUSTER, "4700102", "legacy-nlp", "legacy", "default", 2, 4, 2, [60, 70, 65, 85], [55]),
    project(OLD_CLUSTER, "4700103", "legacy-asr", "legacy", "default", 2, 2, 2, [0], [0],
            spike=(95, "2026-10-05T12:00:00Z"), every=4),
]
PROJECT_BY_ID = {item["id"]: item for item in PROJECTS}


def parse_time(value):
    return datetime.strptime(value.replace("+00:00", "Z"), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def stamp(value):
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


def workload(index, name, phase, kind, project_name, allocated, *, requested=None, idle=0.0,
             created="2026-10-05T08:00:00Z", updated=None, util=None):
    """Build one upstream-shaped workload row plus private fixture fields."""
    owner = next(item for item in PROJECTS if item["name"] == project_name)
    prefix = {LIVE_CLUSTER: "a", LAB_CLUSTER: "d", OLD_CLUSTER: "e"}[owner["clusterId"]]
    running = phase == "Running"
    return {
        "id": f"00000000-0000-4000-8000-0000000{prefix}{index:04d}",
        "name": name,
        "type": kind,
        "phase": phase,
        "projectId": owner["id"],
        "projectName": project_name,
        "clusterId": owner["clusterId"],
        "createdAt": created,
        "phaseUpdatedAt": updated or created,
        "requestedNodePools": [owner["pool"]],
        "currentNodePools": [owner["pool"]] if running else [],
        "priority": 125 if kind == "Inference" else 75 if kind == "Workspace" else 50,
        "preemptible": kind == "Training" or kind == "Distributed",
        "idleAllocatedGpus": idle if running else None,
        "allocatedGpu": allocated,
        "requestedGpu": allocated if requested is None else requested,
        "_departmentId": owner["departmentId"],
        "_department": owner["department"],
        "_util": util if util is not None else owner["util"],
    }


WORKLOADS = [
    workload(1, "hc-train-a1", "Running", "Training", "hc-team-a", 4, created=LIVE_SINCE),
    workload(2, "hc-train-a2", "Running", "Training", "hc-team-a", 4, created=LIVE_SINCE),
    workload(3, "hc-train-b2", "Running", "Training", "hc-team-b", 2, created=LIVE_SINCE),
    workload(4, "hc-train-c1", "Running", "Training", "hc-team-c", 2, created=LIVE_SINCE),
    workload(5, "hc-cpu-a3", "Running", "Training", "hc-team-a", 0, created="2026-10-08T10:24:51Z", util=[0]),
    workload(6, "hc-cpu-b3", "Running", "Training", "hc-team-b", 0, created="2026-10-08T10:24:52Z", util=[0]),
    workload(7, "hc-wait-a4", "Pending", "Training", "hc-team-a", 0, requested=4, created="2026-10-08T10:32:59Z"),
    workload(8, "hc-train-b1", "Pending", "Training", "hc-team-b", 0, requested=4, created="2026-10-08T10:24:40Z"),
    workload(9, "hc-pinned-c2", "Pending", "Training", "hc-team-c", 0, requested=1, created="2026-10-08T10:32:59Z"),
    workload(10, "hc-crash-b4", "Failed", "Training", "hc-team-b", 0, requested=2,
             created="2026-10-08T10:40:00Z", updated="2026-10-08T10:43:38Z"),
    workload(11, "vit-notebook-a", "Running", "Workspace", "vision-train", 4, idle=4.0,
             created="2026-10-02T08:00:00Z", util=[0]),
    workload(12, "vit-notebook-b", "Running", "Workspace", "vision-train", 4, idle=4.0,
             created="2026-10-03T14:00:00Z", util=[0]),
    workload(13, "vit-pretrain-3", "Running", "Training", "vision-train", 8, idle=6.0,
             created="2026-10-04T09:00:00Z", util=[4, 2, 6, 12]),
    workload(14, "vit-sweep-9", "Pending", "Training", "vision-train", 0, requested=4,
             created="2026-10-12T08:10:00Z"),
    workload(15, "llm-sft-9", "Running", "Distributed", "nlp-research", 16, created="2026-10-01T06:00:00Z"),
    workload(16, "tok-train-7", "Failed", "Training", "nlp-research", 0, requested=8,
             created="2026-10-12T06:55:00Z", updated="2026-10-12T07:40:00Z"),
    workload(17, "asr-train-2", "Running", "Training", "speech-asr", 8, created="2026-10-03T10:00:00Z"),
    workload(18, "sim-rl-4", "Running", "Training", "robotics-sim", 8, created="2026-10-02T12:00:00Z"),
    workload(19, "gwas-1", "Running", "Training", "genomics", 2, created="2026-10-04T16:00:00Z"),
    workload(20, "recsys-train", "Running", "Training", "recsys", 6, created="2026-10-01T09:30:00Z"),
    workload(21, "llama-70b-api", "Running", "Inference", "llm-serving", 4, idle=4.0,
             created="2026-09-28T11:00:00Z"),
    workload(22, "chat-api-v2", "Running", "Inference", "chat-api", 4, idle=2.0, created="2026-09-30T15:00:00Z"),
    workload(23, "ads-ctr-1", "Running", "Training", "ads-ranking", 3, created="2026-10-02T07:00:00Z"),
    workload(24, "etl-embed-1", "Running", "Training", "batch-etl", 0.5, idle=0.5, created="2026-10-01T02:00:00Z"),
    workload(25, "etl-embed-2", "Running", "Training", "batch-etl", 0.5, idle=0.5, created="2026-10-01T02:00:00Z"),
    workload(26, "etl-embed-3", "Running", "Training", "batch-etl", 0.5, idle=0.5, created="2026-10-01T02:00:00Z"),
    workload(27, "ocr-batch", "Running", "Training", "doc-ocr", 1, idle=1.0, created="2026-10-03T08:00:00Z"),
    workload(28, "kb-embed", "Running", "Inference", "kb-search", 0.5, idle=0.5, created="2026-10-02T08:00:00Z"),
    workload(29, "spark-etl", "Running", "Training", "data-eng", 0, created="2026-10-06T08:00:00Z", util=[0]),
    workload(30, "cv-notebook", "Running", "Workspace", "legacy-cv", 4, idle=4.0, created="2026-10-01T08:00:00Z"),
    workload(31, "nlp-train", "Running", "Training", "legacy-nlp", 2, created="2026-10-02T08:00:00Z"),
    workload(32, "asr-train", "Running", "Training", "legacy-asr", 2, idle=2.0, created="2026-10-01T08:00:00Z"),
]

STATUS_SIGNALS = {
    "hc-crash-b4": {"phaseMessage": "Container exited with code 1.", "phaseReason": "PodFailed",
                    "k8sPhase": "Failed",
                    "events": [{"createdAt": "2026-10-08T10:43:38Z", "type": "Warning",
                                "reason": "BackOff", "message": "Back-off restarting failed container"}]},
    "tok-train-7": {"phaseMessage": "Container exited with code 137 (OOMKilled).", "phaseReason": "PodFailed",
                    "k8sPhase": "Failed",
                    "events": [{"createdAt": "2026-10-12T07:40:00Z", "type": "Warning",
                                "reason": "OOMKilling", "message": "Memory cgroup out of memory: killed process 4711 (python)"}]},
    "vit-sweep-9": {"phaseMessage": "Not enough resources in node pool a100-train: requested 4 GPUs, quota exceeded.",
                    "phaseReason": "Unschedulable", "k8sPhase": "Pending"},
    "hc-wait-a4": {"phaseMessage": "Not enough resources in node pool default: requested 4 GPUs, 4 GPUs free.",
                   "phaseReason": "Unschedulable", "k8sPhase": "Pending"},
}

ORG_METRICS = ["gpuAllocation", "cpuAllocation", "cpuMemoryAllocation", "gpuUtilization",
               "gpuMemoryUtilization", "cpuUtilization", "cpuMemoryUtilization"]
ORG_DEFAULT_METRICS = ["gpuAllocation", "gpuUtilization", "gpuMemoryUtilization"]
CLUSTER_METRICS = ["allocatedGpu", "unallocatedGpu", "totalGpu", "gpuUtilization",
                   "gpuMemoryUtilization", "cpuUtilization", "cpuMemoryUtilization",
                   "avgWorkloadWaitTime", "workloadsCount"]
CLUSTER_DEFAULT_METRICS = ["allocatedGpu", "unallocatedGpu", "gpuUtilization"]
WINDOWS = {"last_1h": timedelta(hours=1), "last_24h": timedelta(hours=24),
           "last_7d": timedelta(days=7), "last_30d": timedelta(days=30)}
MAX_SAMPLES = 1000
MAX_POINTS = 5000

# Metrics the clusters never served; requesting them sets partialData. The
# cluster metrics endpoint has no unallocatedGpu query, so it is always absent.
UNAVAILABLE_CLUSTER_METRICS = {
    LIVE_CLUSTER: {"unallocatedGpu", "avgWorkloadWaitTime", "workloadsCount", "cpuMemoryUtilization"},
    LAB_CLUSTER: {"unallocatedGpu", "avgWorkloadWaitTime", "workloadsCount", "cpuMemoryUtilization"},
    OLD_CLUSTER: {"unallocatedGpu", "avgWorkloadWaitTime", "workloadsCount", "cpuMemoryUtilization"},
}
UNAVAILABLE_ORG_METRICS = {"cpuUtilization", "cpuMemoryUtilization", "cpuAllocation", "cpuMemoryAllocation"}
CPU_UTILIZATION = {LIVE_CLUSTER: [11.3], LAB_CLUSTER: [38.2, 37.9, 38.6, 38.1], OLD_CLUSTER: [22.0]}


class ToolError(Exception):
    """An error the real MCP server would return as an isError tool result."""


def require_known(args, allowed):
    unknown = sorted(set(args) - set(allowed))
    if unknown:
        raise ToolError(f"validation: unexpected argument(s): {', '.join(unknown)}")


def require_cluster(args, required=True):
    """Return the clusterId, failing like the server for missing or unknown clusters."""
    cluster_id = args.get("clusterId")
    if not cluster_id:
        if required:
            raise ToolError("validation: clusterId is required")
        return None
    if cluster_id not in NOW:
        raise ToolError(f"not_found: cluster {cluster_id} not found")
    return cluster_id


def require_pool(cluster_id, pool):
    """Fail like the server when a node pool name does not exist in the cluster(s)."""
    if not pool:
        return
    clusters = [cluster_id] if cluster_id else list(NOW)
    if not any((cluster, pool) in POOL_IDS for cluster in clusters):
        raise ToolError(f"not_found: node pool {pool} not found")


def page(items, args, key):
    limit = int(args.get("limit") or 50)
    offset = int(args.get("offset") or 0)
    if limit < 1 or limit > 200 or offset < 0:
        raise ToolError("validation: limit must be 1-200 and offset non-negative")
    result = {key: items[offset:offset + limit], "hasMore": offset + limit < len(items)}
    if result["hasMore"]:
        result["nextOffset"] = offset + limit
    return result


def list_node_pools(args):
    require_known(args, {"clusterId", "limit", "offset"})
    cluster_id = require_cluster(args, required=False)
    pools = [pool for pool in NODE_POOLS if not cluster_id or pool["clusterId"] == cluster_id]
    return page(pools, args, "nodePools")


def get_cluster_physical_inventory(args):
    require_known(args, {"clusterId", "nodePoolName"})
    cluster_id = require_cluster(args)
    pool = args.get("nodePoolName")
    require_pool(cluster_id, pool)
    rows = []
    for name, (model, gpu_nodes, gpus, _) in POOL_HARDWARE[cluster_id].items():
        if pool and name != pool:
            continue
        allocated = sum(item["gpus"] for item in PROJECTS
                        if item["clusterId"] == cluster_id and item["pool"] == name)
        rows.append({"nodePool": name, "gpuNodes": gpu_nodes, "gpusTotal": gpus,
                     "gpusAllocatable": gpus, "gpusAllocated": math.ceil(allocated),
                     "gpusFree": gpus - math.ceil(allocated), "gpuModel": model})
    keys = ["gpuNodes", "gpusTotal", "gpusAllocatable", "gpusAllocated", "gpusFree"]
    return {
        "clusterId": cluster_id,
        "nodePool": pool,
        "asOf": NOW[cluster_id],
        "totals": {key: sum(row[key] for row in rows) for key in keys},
        "byNodePool": [{key: value for key, value in row.items() if key != "gpuModel"} for row in rows],
        "byGpuModel": [{"gpuModel": row["gpuModel"], "nodes": row["gpuNodes"], "gpusTotal": row["gpusTotal"]}
                       for row in rows if row["gpuModel"]],
    }


def sample_count(args):
    samples = int(args.get("numberOfSamples") or 0) or 20
    if samples < 1 or samples > MAX_SAMPLES:
        raise ToolError(f"validation: numberOfSamples must be between 0 and {MAX_SAMPLES}")
    return samples


def parse_range(start, end):
    try:
        start, end = parse_time(start), parse_time(end)
    except ValueError as error:
        raise ToolError("validation: start and end must be RFC3339 timestamps") from error
    if end <= start:
        raise ToolError("validation: end must be after start")
    return start, end


def spaced(start, end, samples):
    span = (end - start).total_seconds()
    return [start + timedelta(seconds=round(span * index / (samples - 1))) if samples > 1 else end
            for index in range(samples)]


def sample_times(cluster_id, args, default_window):
    samples = sample_count(args)
    if bool(args.get("start")) != bool(args.get("end")):
        raise ToolError("validation: start and end must be provided together")
    if args.get("start"):
        start, end = parse_range(args["start"], args["end"])
    else:
        window = args.get("window") or default_window
        if window not in WINDOWS:
            raise ToolError("validation: window must be one of: last_1h, last_24h, last_7d, last_30d")
        end = parse_time(NOW[cluster_id])
        start = end - WINDOWS[window]
    return samples, spaced(start, end, samples)


def recorded(cluster_id, moment):
    if cluster_id != LIVE_CLUSTER:
        return True
    return any(parse_time(low) <= moment <= parse_time(high) for low, high in LIVE_RECORDED)


def holds_gpus(item, moment):
    return item["gpus"] > 0 and (item["since"] is None or moment >= parse_time(item["since"]))


def cycle(values, index):
    return values[index % len(values)]


def utilization(item, source, index, moment):
    spike = item["spike"]
    if source == "util" and spike and moment < parse_time(spike[1]):
        return spike[0]
    return cycle(item[source], index)


def org_point(members, metric, index, moment):
    """Aggregate one org-unit sample: allocation is summed, utilization is allocation-weighted."""
    holders = [item for item in members if holds_gpus(item, moment)]
    if not holders or not recorded(holders[0]["clusterId"], moment):
        return None
    allocated = sum(item["gpus"] for item in holders)
    if metric == "gpuAllocation":
        return allocated
    source = "util" if metric == "gpuUtilization" else "memory"
    return round(sum(item["gpus"] * utilization(item, source, index, moment) for item in holders) / allocated, 2)


def resolve_org_unit(args):
    org_type = args.get("orgType")
    if org_type not in ("project", "department"):
        raise ToolError("validation: orgType must be one of: project, department")
    org_id, org_name = args.get("orgId"), args.get("orgName")
    if bool(org_id) == bool(org_name):
        raise ToolError("validation: exactly one of orgId or orgName is required")
    cluster_id = args.get("clusterId")
    if cluster_id and cluster_id not in NOW:
        raise ToolError(f"not_found: cluster {cluster_id} not found")
    if org_type == "project":
        units = [{"id": item["id"], "name": item["name"], "clusterId": item["clusterId"]} for item in PROJECTS]
    else:
        units = DEPARTMENTS
    if org_id:
        matches = [unit for unit in units if unit["id"] == org_id]
    else:
        matches = [unit for unit in units
                   if unit["name"] == org_name and (not cluster_id or unit["clusterId"] == cluster_id)]
    if not matches:
        raise ToolError(f"not_found: {org_type} {org_id or org_name} not found")
    if len(matches) > 1:
        raise ToolError(f"ambiguous_org_unit: {org_type} name {org_name} matches several clusters; pass clusterId")
    return org_type, matches[0]


def get_org_unit_metrics(args):
    require_known(args, {"orgType", "orgId", "orgName", "clusterId", "metricTypes", "window",
                         "start", "end", "numberOfSamples", "nodePoolName"})
    org_type, unit = resolve_org_unit(args)
    cluster_id = unit["clusterId"]
    requested = args.get("metricTypes") or ORG_DEFAULT_METRICS
    for name in requested:
        if name not in ORG_METRICS:
            raise ToolError(f"validation: unknown metric type \"{name}\"")
    version = CLUSTER_VERSION[cluster_id]
    if "gpuMemoryUtilization" in requested and tuple(map(int, version.split("."))) < MEMORY_METRIC_MIN_VERSION:
        raise ToolError(f"validation: GPU_MEMORY_UTILIZATION metric is not supported for cluster version {version}")
    pool = args.get("nodePoolName")
    require_pool(cluster_id, pool)
    samples, times = sample_times(cluster_id, args, "last_24h")
    members = [item for item in PROJECTS if item["clusterId"] == cluster_id
               and (item["id"] == unit["id"] if org_type == "project" else item["departmentId"] == unit["id"])
               and (not pool or item["pool"] == pool)]
    result = {
        "orgType": org_type,
        "orgId": unit["id"],
        "orgName": unit["name"],
        "nodePool": pool,
        "timeRange": {"from": stamp(times[0]), "to": stamp(times[-1]), "numberOfSamples": samples},
        "seriesTruncated": False,
        "partialData": False,
    }
    if samples * len(requested) > MAX_POINTS:
        result["seriesTruncated"] = True
        return result
    missing = set(UNAVAILABLE_ORG_METRICS)
    if org_type == "project":
        missing |= PROJECT_BY_ID[unit["id"]]["missing"]
    series = {}
    for name in requested:
        if name in missing:
            result["partialData"] = True
            continue
        every = PROJECT_BY_ID[unit["id"]]["every"] if org_type == "project" else 1
        points = []
        for index, moment in enumerate(times):
            value = org_point(members, name, index, moment) if index % every == 0 else None
            if value is not None:
                points.append({"t": stamp(moment), "v": value})
        if points:
            series[name] = points
    result["series"] = series
    return result


def cluster_point(cluster_id, pools, metric, index, moment):
    """Utilization averages over every GPU in scope, allocated or not, as the backend does."""
    if not recorded(cluster_id, moment):
        return None
    total = sum(POOL_HARDWARE[cluster_id][pool][2] for pool in pools)
    holders = [item for item in PROJECTS
               if item["clusterId"] == cluster_id and item["pool"] in pools and holds_gpus(item, moment)]
    allocated = sum(item["gpus"] for item in holders)
    if metric == "totalGpu":
        return total
    if metric == "allocatedGpu":
        return allocated
    if metric == "unallocatedGpu":
        return total - allocated
    if metric == "cpuUtilization":
        return cycle(CPU_UTILIZATION[cluster_id], index)
    if not total:
        return None
    source = "util" if metric == "gpuUtilization" else "memory"
    outside = sum(OUTSIDE_BUSY_GPUS.get((cluster_id, pool), 0) for pool in pools) * 100
    busy = sum(item["gpus"] * utilization(item, source, index, moment) for item in holders) + outside
    return round(busy / total, 2)


def get_cluster_metrics(args):
    require_known(args, {"clusterId", "nodePoolName", "metricTypes", "window", "start", "end",
                         "numberOfSamples"})
    cluster_id = require_cluster(args)
    pool = args.get("nodePoolName")
    requested = args.get("metricTypes") or CLUSTER_DEFAULT_METRICS
    for name in requested:
        if name not in CLUSTER_METRICS:
            raise ToolError(f"validation: unknown metric type \"{name}\"")
    samples, times = sample_times(cluster_id, args, "last_1h")
    result = {
        "clusterId": cluster_id,
        "nodePool": pool,
        "timeRange": {"from": stamp(times[0]), "to": stamp(times[-1]), "numberOfSamples": samples},
        "seriesTruncated": False,
        "partialData": False,
    }
    if samples * len(requested) > MAX_POINTS:
        result["seriesTruncated"] = True
        return result
    pools = list(POOL_HARDWARE[cluster_id])
    series = {}
    for name in requested:
        if name in UNAVAILABLE_CLUSTER_METRICS[cluster_id]:
            result["partialData"] = True
            continue
        points = []
        for index, moment in enumerate(times):
            value = cluster_point(cluster_id, pools, name, index, moment)
            if value is not None:
                points.append({"t": stamp(moment), "v": value})
        if points:
            series[name] = points
    result["series"] = series
    return result


def find_project(args, cluster_id):
    """Resolve projectId/projectName for list_project_resources, like the server."""
    if args.get("projectId") and args.get("projectName"):
        raise ToolError("validation: projectId and projectName are mutually exclusive")
    if args.get("projectId"):
        return [item for item in PROJECTS if item["id"] == args["projectId"]]
    if args.get("projectName"):
        matches = [item for item in PROJECTS if item["name"] == args["projectName"]
                   and (not cluster_id or item["clusterId"] == cluster_id)]
        if not matches:
            raise ToolError(f"not_found: project {args['projectName']} not found")
        return matches
    return None


def list_project_resources(args):
    require_known(args, {"clusterId", "projectId", "projectName", "departmentId", "departmentName",
                         "nodePoolId", "nodePoolName", "limit", "offset"})
    cluster_id = require_cluster(args, required=False)
    for left, right in (("departmentId", "departmentName"), ("nodePoolId", "nodePoolName")):
        if args.get(left) and args.get(right):
            raise ToolError(f"validation: {left} and {right} are mutually exclusive")
    rows = [item for item in PROJECTS if not cluster_id or item["clusterId"] == cluster_id]
    chosen = find_project(args, cluster_id)
    if chosen is not None:
        rows = [item for item in rows if item in chosen]
    if args.get("departmentId"):
        rows = [item for item in rows if item["departmentId"] == args["departmentId"]]
    if args.get("departmentName"):
        if not any(dept["name"] == args["departmentName"] for dept in DEPARTMENTS):
            raise ToolError(f"not_found: department {args['departmentName']} not found")
        rows = [item for item in rows if item["department"] == args["departmentName"]]
    if args.get("nodePoolName"):
        require_pool(cluster_id, args["nodePoolName"])
        rows = [item for item in rows if item["pool"] == args["nodePoolName"]]
    if args.get("nodePoolId"):
        rows = [item for item in rows if POOL_IDS[(item["clusterId"], item["pool"])] == args["nodePoolId"]]
    cells = []
    for item in rows:
        if not item["deserved"] and not item["gpus"]:
            continue
        cells.append({
            "projectId": item["id"], "projectName": item["name"], "departmentId": item["departmentId"],
            "nodePoolId": POOL_IDS[(item["clusterId"], item["pool"])], "nodePoolName": item["pool"],
            "rank": "MediumLow",
            "gpu": {"deserved": item["deserved"], "limit": item["limit"], "overQuotaWeight": None,
                    "allocated": item["gpus"] or None},
            "cpuCores": {"deserved": None, "limit": None, "overQuotaWeight": None,
                         "allocated": round(item["gpus"] * 0.1, 2) or None},
            "memoryGib": {"deserved": None, "limit": None, "overQuotaWeight": None,
                          "allocated": round(item["gpus"] * 0.09, 3) or None},
        })
    return page(cells, args, "resources")


def list_projects(args):
    require_known(args, {"clusterId", "departmentId", "departmentName", "limit", "offset"})
    cluster_id = require_cluster(args, required=False)
    rows = [item for item in PROJECTS if not cluster_id or item["clusterId"] == cluster_id]
    if args.get("departmentId"):
        rows = [item for item in rows if item["departmentId"] == args["departmentId"]]
    if args.get("departmentName"):
        rows = [item for item in rows if item["department"] == args["departmentName"]]
    projects = [{"id": item["id"], "name": item["name"], "clusterId": item["clusterId"],
                 "departmentId": item["departmentId"], "phase": "Ready"} for item in rows]
    result = page(projects, args, "projects")
    result["totalCount"] = len(projects)
    return result


def list_departments(args):
    require_known(args, {"clusterId", "limit", "offset"})
    cluster_id = require_cluster(args, required=False)
    rows = []
    for dept in DEPARTMENTS:
        if cluster_id and dept["clusterId"] != cluster_id:
            continue
        children = [item for item in PROJECTS if item["departmentId"] == dept["id"]]
        gpus = sum(item["gpus"] for item in children)
        entity = {"id": dept["id"], "name": dept["name"], "clusterId": dept["clusterId"],
                  "projectCount": len(children), "gpuQuota": dept["gpuQuota"]}
        if gpus:
            entity["allocated"] = {"gpu": gpus, "cpuCores": round(gpus * 0.1, 2),
                                   "memoryGib": round(gpus * 0.09, 3)}
        rows.append(entity)
    result = page(rows, args, "departments")
    result["totalCount"] = len(rows)
    return result


def scoped_workloads(cluster_id):
    return [row for row in WORKLOADS if not cluster_id or row["clusterId"] == cluster_id]


def get_workloads_summary(args):
    """Return phase counts; a node-pool scope only sees workloads placed in that pool."""
    require_known(args, {"clusterId", "orgType", "orgName", "nodePoolName", "gpuIdleOnly"})
    cluster_id = require_cluster(args, required=False)
    rows = scoped_workloads(cluster_id)
    scope = {"clusterId": cluster_id} if cluster_id else {}
    org_type, org_name = args.get("orgType"), args.get("orgName")
    if bool(org_type) != bool(org_name):
        raise ToolError("validation: orgType and orgName must be set together")
    if org_type:
        org_type, unit = resolve_org_unit({"orgType": org_type, "orgName": org_name, "clusterId": cluster_id})
        key = "projectId" if org_type == "project" else "_departmentId"
        rows = [row for row in rows if row[key] == unit["id"]]
        scope.update({"orgType": org_type, "orgName": org_name, "orgUnitId": unit["id"]})
    pool = args.get("nodePoolName")
    if pool:
        require_pool(cluster_id, pool)
        rows = [row for row in rows if pool in row["currentNodePools"]]
        scope["nodePool"] = pool
    if args.get("gpuIdleOnly"):
        rows = [row for row in rows if row["phase"] == "Running" and row["idleAllocatedGpus"]]
        scope["gpuIdleOnly"] = True
    counts = {"running": 0, "pending": 0, "failed": 0, "completed": 0, "stopped": 0, "other": 0}
    fold = {"Running": "running", "Pending": "pending", "Failed": "failed", "Completed": "completed",
            "Stopped": "stopped"}
    by_phase = {}
    for row in rows:
        by_phase[row["phase"]] = by_phase.get(row["phase"], 0) + 1
        counts[fold.get(row["phase"], "other")] += 1
    as_of = NOW[cluster_id] if cluster_id else NOW[LAB_CLUSTER]
    return {"scope": scope, "asOf": as_of, "total": len(rows), "counts": counts, "byPhase": by_phase}


DETAIL_FIELDS = {"createdAt", "requestedNodePools", "currentNodePools", "priority", "preemptible",
                 "idleAllocatedGpus"}
SORT_KEYS = {"name": "name", "type": "type", "phase": "phase", "projectName": "projectName",
             "projectId": "projectId", "createdAt": "createdAt", "phaseUpdatedAt": "phaseUpdatedAt",
             "priority": "priority", "allocatedGPU": "allocatedGpu", "allocatedGpu": "allocatedGpu",
             "idleAllocatedGpus": "idleAllocatedGpus"}
FILTER_FIELDS = {"type": "type", "name": "name", "phase": "phase", "projectName": "projectName",
                 "projectId": "projectId", "departmentId": "_departmentId", "departmentName": "_department",
                 "allocatedGPU": "allocatedGpu", "idleAllocatedGpus": "idleAllocatedGpus",
                 "createdAt": "createdAt", "phaseUpdatedAt": "phaseUpdatedAt", "priority": "priority"}
NUMERIC_FIELDS = {"allocatedGpu", "idleAllocatedGpus", "priority"}


def compare(actual, operator, expected, numeric):
    if numeric:
        actual, expected = float(actual or 0), float(expected)
    operations = {
        "==": lambda: actual == expected, "!=": lambda: actual != expected,
        ">": lambda: actual > expected, ">=": lambda: actual >= expected,
        "<": lambda: actual < expected, "<=": lambda: actual <= expected,
        "=@": lambda: str(expected) in str(actual), "!@": lambda: str(expected) not in str(actual),
        "=^": lambda: str(actual).startswith(str(expected)),
        "=$": lambda: str(actual).endswith(str(expected)),
    }
    if operator not in operations:
        raise ToolError(f"validation: unsupported operator {operator}")
    return operations[operator]()


def public_row(row, include_details):
    """Map a fixture row to the compact or detailed list_workloads item."""
    item = {}
    for key, value in row.items():
        if key.startswith("_"):
            continue
        if key in DETAIL_FIELDS and not include_details:
            continue
        if key in ("requestedNodePools", "currentNodePools") and not value:
            continue
        if key == "idleAllocatedGpus" and value is None:
            continue
        item[key] = value
    return item


def list_workloads(args):
    require_known(args, {"clusterId", "name", "projectId", "phase", "type", "nodePoolName",
                         "createdAfter", "phaseUpdatedAfter", "projectName", "departmentId",
                         "departmentName", "allocatedGpu", "gpuIdleOnly", "filterBy", "includeDetails",
                         "sortBy", "sortOrder", "limit", "offset"})
    cluster_id = require_cluster(args, required=False)
    if args.get("gpuIdleOnly") and args.get("phase") not in (None, "", "Running"):
        raise ToolError("validation: phase must be Running when gpuIdleOnly is true")
    rows = scoped_workloads(cluster_id)
    for field in ("phase", "type", "name", "projectId"):
        if args.get(field):
            rows = [row for row in rows if row[field] == args[field]]
    if args.get("nodePoolName"):
        require_pool(cluster_id, args["nodePoolName"])
        rows = [row for row in rows if args["nodePoolName"] in row["currentNodePools"]]
    if args.get("createdAfter"):
        rows = [row for row in rows if parse_time(row["createdAt"]) >= parse_time(args["createdAfter"])]
    if args.get("phaseUpdatedAfter"):
        rows = [row for row in rows
                if parse_time(row["phaseUpdatedAt"]) >= parse_time(args["phaseUpdatedAfter"])]
    if args.get("gpuIdleOnly"):
        rows = [row for row in rows if row["phase"] == "Running" and row["idleAllocatedGpus"]]
    comparisons = [(name, args[name]) for name in ("projectName", "departmentId", "departmentName")
                   if args.get(name)]
    if args.get("allocatedGpu"):
        comparisons.append(("allocatedGPU", args["allocatedGpu"]))
    comparisons += [(item.get("field"), item) for item in args.get("filterBy") or []]
    for field, condition in comparisons:
        if field not in FILTER_FIELDS:
            raise ToolError(f"validation: unsupported filter field {field}")
        key = FILTER_FIELDS[field]
        rows = [row for row in rows
                if compare(row[key], condition["operator"], condition["value"], key in NUMERIC_FIELDS)]
    sort_by = args.get("sortBy") or "phaseUpdatedAt"
    if sort_by not in SORT_KEYS:
        raise ToolError(f"validation: sortBy must be one of {sorted(SORT_KEYS)}")
    order = args.get("sortOrder") or "desc"
    if order not in ("asc", "desc"):
        raise ToolError("validation: sortOrder must be asc or desc")
    key = SORT_KEYS[sort_by]
    rows = sorted(rows, key=lambda row: (row[key] if row[key] is not None else 0, row["name"]),
                  reverse=order == "desc")
    result = page(rows, args, "workloads")
    result["workloads"] = [public_row(row, bool(args.get("includeDetails"))) for row in result["workloads"]]
    return result


def find_workload(workload_id):
    row = next((item for item in WORKLOADS if item["id"] == workload_id), None)
    if row is None:
        raise ToolError(f"not_found: workload {workload_id} not found")
    return row


def get_workload_status(args):
    require_known(args, {"workloadId"})
    row = find_workload(args.get("workloadId"))
    result = {"id": row["id"], "name": row["name"], "clusterId": row["clusterId"], "type": row["type"],
              "phase": row["phase"], "projectName": row["projectName"], "departmentName": row["_department"],
              "createdAt": row["createdAt"], "phaseUpdatedAt": row["phaseUpdatedAt"],
              "conditions": [], "events": []}
    if row["phase"] == "Running":
        result["k8sPhase"] = "Running"
        result["conditions"] = [{"type": "Available", "status": "True", "reason": "Running",
                                 "lastTransitionTime": row["phaseUpdatedAt"]}]
    result.update(STATUS_SIGNALS.get(row["name"], {}))
    return result


def get_workload_metrics(args):
    """Return the fixed resource measurement set; values are strings, as upstream sends them."""
    require_known(args, {"workloadId", "start", "end", "numberOfSamples"})
    row = find_workload(args.get("workloadId"))
    cluster_id = row["clusterId"]
    end = args.get("end") or NOW[cluster_id]
    start = args.get("start") or stamp(parse_time(end) - timedelta(hours=1))
    times = spaced(*parse_range(start, end), sample_count(args))
    created = parse_time(row["createdAt"])
    running = row["phase"] == "Running"
    gpus = row["allocatedGpu"]
    kinds = {
        "GPU_UTILIZATION": lambda index: cycle(row["_util"], index),
        "GPU_MEMORY_USAGE_BYTES": lambda index: int(gpus * 80 * 2**30 * 0.2) if gpus else 0,
        "CPU_USAGE_CORES": lambda index: 0.4 if gpus else 0.1,
        "CPU_MEMORY_USAGE_BYTES": lambda index: 6 * 2**30,
    }
    measurements = []
    for kind, value in kinds.items():
        values = [{"value": str(value(index)), "timestamp": stamp(moment)}
                  for index, moment in enumerate(times)
                  if running and moment >= created and recorded(cluster_id, moment)]
        measurements.append({"type": kind, "values": values})
    result = {"workloadId": row["id"], "type": row["type"], "measurements": measurements}
    if not running:
        result["warnings"] = [f"workload is {row['phase']}; no resource samples in the window"]
    return result


def schema(properties, required=None):
    return {"type": "object", "properties": properties, "required": required or [],
            "additionalProperties": False}


COMPARISON = {"type": "object", "properties": {"operator": {"type": "string"}, "value": {"type": "string"}},
              "required": ["operator", "value"], "additionalProperties": False}
GPU_COMPARISON = {"type": "object", "properties": {"operator": {"type": "string"}, "value": {"type": "number"}},
                  "required": ["operator", "value"], "additionalProperties": False}

TOOLS = {
    "list_node_pools": {
        "description": "List the node pools in a cluster, or across every cluster when clusterId is omitted. Returns each pool's identity, cluster, whether it is the cluster default, and its lifecycle phase. Does NOT return quota, metrics, or GPU capacity. Read-only.",
        "inputSchema": schema({"clusterId": {"type": "string"}, "limit": {"type": "integer"},
                               "offset": {"type": "integer"}}),
        "handler": list_node_pools,
    },
    "list_projects": {
        "description": "Enumerate Run:ai project entities, optionally filtered by cluster or department. Returns id, name, clusterId, departmentId, and phase, plus an authoritative totalCount. Does NOT return quota, allocation, or utilization. Read-only.",
        "inputSchema": schema({"clusterId": {"type": "string"}, "departmentId": {"type": "string"},
                               "departmentName": {"type": "string"}, "limit": {"type": "integer"},
                               "offset": {"type": "integer"}}),
        "handler": list_projects,
    },
    "list_departments": {
        "description": "Enumerate Run:ai department entities with an authoritative totalCount. Each entity carries allocated (current point-in-time gpu/cpu/memory; absent when nothing is allocated), gpuQuota (-1 is unlimited), and projectCount. For allocation or utilization over time use get_org_unit_metrics. Read-only.",
        "inputSchema": schema({"clusterId": {"type": "string"}, "limit": {"type": "integer"},
                               "offset": {"type": "integer"}}),
        "handler": list_departments,
    },
    "list_project_resources": {
        "description": "List configured GPU/CPU/memory quotas and current allocation per project per node pool. Returns (project x node-pool) cells with deserved, limit, overQuotaWeight, and live allocated values (null when nothing is allocated). A project appears only where it has quota or allocation, so do NOT use this to enumerate or count projects. Do NOT call for time-series usage (use get_org_unit_metrics or get_cluster_metrics). Read-only.",
        "inputSchema": schema({"clusterId": {"type": "string"}, "projectId": {"type": "string"},
                               "projectName": {"type": "string"}, "departmentId": {"type": "string"},
                               "departmentName": {"type": "string"}, "nodePoolId": {"type": "string"},
                               "nodePoolName": {"type": "string"}, "limit": {"type": "integer"},
                               "offset": {"type": "integer"}}),
        "handler": list_project_resources,
    },
    "get_org_unit_metrics": {
        "description": "Get a project's or department's resource consumption over time \u2014 allocation and utilization trends for GPU, CPU, and memory. Use when the user asks how an org unit has been using resources over a period (e.g. \"how has my team's GPU utilization trended this week?\"). Returns time-series points per requested metric. Use get_workload_metrics for one workload and list_workloads for per-workload allocated or requested GPU-count filters. Resolve an ambiguous org-unit name with cluster context. This aggregate tool cannot identify individual workloads above a utilization threshold. Do NOT call for configured quota (use list_project_resources / list_department_resources), for a single current snapshot, for cluster-wide metrics (use get_cluster_metrics), or for per-workload metrics.",
        "inputSchema": schema({
            "orgType": {"type": "string", "description": "Which kind of org unit to query. One of: project, department."},
            "orgId": {"type": "string", "description": "Project or department ID. Mutually exclusive with orgName; one of the two is required."},
            "orgName": {"type": "string", "description": "Project or department name. Mutually exclusive with orgId; resolved to an ID. Returns ambiguous_org_unit if the name matches multiple."},
            "clusterId": {"type": "string", "description": "Cluster ID. Used to disambiguate orgName resolution; if omitted, name resolution spans all accessible clusters."},
            "metricTypes": {"type": "array", "items": {"type": "string"}, "description": "Metrics to return; allocation/utilization for gpu, cpu, cpuMemory, plus gpuMemoryUtilization. Default: gpu trio."},
            "window": {"type": "string", "description": "Convenience time range ending now. One of: last_1h, last_24h, last_7d, last_30d. Defaults to last_24h. Ignored when start and end are both set."},
            "start": {"type": "string", "description": "Explicit range start (RFC3339). Overrides window; must be paired with end."},
            "end": {"type": "string", "description": "Explicit range end (RFC3339). Overrides window; must be paired with start."},
            "numberOfSamples": {"type": "integer", "description": "Number of evenly spaced points across the range. Defaults to 20, capped at 1000."},
            "nodePoolName": {"type": "string", "description": "Scope the series to a single node pool by name. If omitted, spans the whole org unit."},
        }, ["orgType"]),
        "handler": get_org_unit_metrics,
    },
    "get_cluster_metrics": {
        "description": "Get a cluster's resource capacity and consumption over time \u2014 free vs allocated GPUs, GPU/CPU utilization, and scheduling pressure. Use when the user asks about cluster-wide capacity or saturation (e.g. \"how many GPUs are free this week?\", \"is the cluster GPU-saturated?\"). Scope to one node pool with nodePoolName. Use get_workload_metrics for one workload and list_workloads for per-workload allocated or requested GPU-count filters. This aggregate tool cannot identify individual workloads above a utilization threshold. Do NOT call for a single project's/department's consumption (use get_org_unit_metrics), for configured quota (use list_*_resources), or for per-workload metrics.",
        "inputSchema": schema({
            "clusterId": {"type": "string", "description": "Cluster ID to query."},
            "nodePoolName": {"type": "string", "description": "Scope the series to a single node pool. If omitted, spans the whole cluster."},
            "metricTypes": {"type": "array", "items": {"type": "string"}, "description": "Capacity (allocatedGpu, unallocatedGpu, totalGpu), utilization (gpu/gpuMemory/cpu/cpuMemory), scheduling (avgWorkloadWaitTime, workloadsCount)."},
            "window": {"type": "string", "description": "Convenience time range ending now. One of: last_1h, last_24h, last_7d, last_30d. Defaults to last_1h. Ignored when start and end are both set."},
            "start": {"type": "string", "description": "Explicit range start (RFC3339). Overrides window; must be paired with end."},
            "end": {"type": "string", "description": "Explicit range end (RFC3339). Overrides window; must be paired with start."},
            "numberOfSamples": {"type": "integer", "description": "Number of evenly spaced points across the range. Defaults to 20, capped at 1000."},
        }, ["clusterId"]),
        "handler": get_cluster_metrics,
    },
    "get_cluster_physical_inventory": {
        "description": "Get a cluster's current hardware capacity per node pool and per GPU model: GPU nodes, GPUs total, allocatable, allocated, and free (fully idle devices). Point-in-time. Do NOT call for quota or utilization over time. Read-only.",
        "inputSchema": schema({"clusterId": {"type": "string"}, "nodePoolName": {"type": "string"}},
                              ["clusterId"]),
        "handler": get_cluster_physical_inventory,
    },
    "get_workloads_summary": {
        "description": "Counts of workloads bucketed by phase for a cluster, a project or department (orgType + orgName), a node pool, or the whole platform. Use gpuIdleOnly to count Running workloads whose idleAllocatedGpus value is non-zero. Point-in-time snapshot. Do NOT call when the user needs workload names (use list_workloads) or for GPU utilization (use get_org_unit_metrics or get_cluster_metrics). Read-only.",
        "inputSchema": schema({"clusterId": {"type": "string"}, "orgType": {"type": "string"},
                               "orgName": {"type": "string"}, "nodePoolName": {"type": "string"},
                               "gpuIdleOnly": {"type": "boolean"}}),
        "handler": get_workloads_summary,
    },
    "list_workloads": {
        "description": "List workloads with a compact response (identity, type, phase, project, cluster, last phase update, requested and allocated GPUs). Set includeDetails for creation time, node pools, priority, preemptibility, and idleAllocatedGpus. Use gpuIdleOnly for Running workloads whose idleAllocatedGpus value is non-zero (a current snapshot). Filter by cluster, name, project, phase, type, node pool, departmentId/departmentName/projectName comparisons, allocatedGpu comparison, or typed filterBy (fields: type, name, phase, projectName, projectId, departmentId, departmentName, allocatedGPU, idleAllocatedGpus, createdAt, phaseUpdatedAt, priority). sortBy: name, type, phase, projectName, projectId, createdAt, phaseUpdatedAt, priority, allocatedGPU, idleAllocatedGpus. GPU utilization is metrics: use get_workload_metrics, get_org_unit_metrics, or get_cluster_metrics. Read-only.",
        "inputSchema": schema({
            "clusterId": {"type": "string"}, "name": {"type": "string"}, "projectId": {"type": "string"},
            "phase": {"type": "string"}, "type": {"type": "string"}, "nodePoolName": {"type": "string"},
            "createdAfter": {"type": "string"}, "phaseUpdatedAfter": {"type": "string"},
            "projectName": COMPARISON, "departmentId": COMPARISON, "departmentName": COMPARISON,
            "allocatedGpu": GPU_COMPARISON, "gpuIdleOnly": {"type": "boolean"},
            "filterBy": {"type": "array", "items": {
                "type": "object",
                "properties": {"field": {"type": "string"}, "operator": {"type": "string"},
                               "value": {"type": "string"}},
                "required": ["field", "operator", "value"], "additionalProperties": False}},
            "includeDetails": {"type": "boolean"}, "sortBy": {"type": "string"},
            "sortOrder": {"type": "string"}, "limit": {"type": "integer"}, "offset": {"type": "integer"},
        }),
        "handler": list_workloads,
    },
    "get_workload_status": {
        "description": "Get one workload's current phase, phaseMessage, phaseReason, k8sPhase, conditions, and recent events. Read-only.",
        "inputSchema": schema({"workloadId": {"type": "string"}}, ["workloadId"]),
        "handler": get_workload_status,
    },
    "get_workload_metrics": {
        "description": "Get how one workload's resource usage changed over time: GPU_UTILIZATION (percent), GPU_MEMORY_USAGE_BYTES, CPU_USAGE_CORES, CPU_MEMORY_USAGE_BYTES, as timestamped string values. start/end (RFC3339) default to the last hour; numberOfSamples defaults to 20, max 1000. Do not compare utilization across workloads by default; only for a complete candidate set of 10 or fewer. Read-only.",
        "inputSchema": schema({"workloadId": {"type": "string"}, "start": {"type": "string"},
                               "end": {"type": "string"}, "numberOfSamples": {"type": "integer"}},
                              ["workloadId"]),
        "handler": get_workload_metrics,
    },
}


def send(payload):
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def send_error(request_id, code, message):
    send({"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}})


def tool_result(request_id, result, *, is_error=False):
    text = result if isinstance(result, str) else json.dumps(result, sort_keys=True)
    payload = {"content": [{"type": "text", "text": text}]}
    if is_error:
        payload["isError"] = True
    send({"jsonrpc": "2.0", "id": request_id, "result": payload})


def tool_list():
    return [
        {"name": name, "description": tool["description"], "inputSchema": tool["inputSchema"],
         "annotations": {"readOnlyHint": True, "destructiveHint": False}}
        for name, tool in TOOLS.items()
    ]


def main():
    """Serve newline-delimited MCP JSON-RPC requests until stdin closes."""
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            send_error(None, -32700, "parse error")
            continue
        if not isinstance(request, dict):
            send_error(None, -32600, "invalid request")
            continue
        method, request_id = request.get("method"), request.get("id")
        if method == "initialize":
            send({"jsonrpc": "2.0", "id": request_id, "result": {
                "protocolVersion": "2024-11-05", "capabilities": {"tools": {}},
                "serverInfo": {"name": "runai-utilization-overview-eval", "version": "1.0.0"}}})
        elif method == "notifications/initialized":
            continue
        elif method == "tools/list":
            send({"jsonrpc": "2.0", "id": request_id, "result": {"tools": tool_list()}})
        elif method == "tools/call":
            params = request.get("params") or {}
            tool = TOOLS.get(params.get("name"))
            if tool is None:
                send_error(request_id, -32601, f"unknown tool {params.get('name')}")
                continue
            try:
                tool_result(request_id, tool["handler"](params.get("arguments") or {}))
            except ToolError as error:
                tool_result(request_id, str(error), is_error=True)
            except Exception as error:  # noqa: BLE001 - the JSON-RPC loop must survive handler bugs.
                print(f"fixture tool {params.get('name')} failed: {type(error).__name__}: {error}",
                      file=sys.stderr)
                tool_result(request_id, f"internal: {type(error).__name__}: {error}", is_error=True)
        elif request_id is not None:
            send_error(request_id, -32601, f"unknown method {method}")


if __name__ == "__main__":
    main()
