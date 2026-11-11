#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic, network-free Run:ai MCP fixture for cluster health evals.

Two clusters share one server:

- LIVE_CLUSTER replays responses recorded on 2026-10-07 from a live Run:ai
  cluster through the Run:ai MCP server (identifiers sanitized). It is idle and
  healthy.
- LAB_CLUSTER is synthetic. It follows the current MCP response shapes and
  holds one fault per node pool. Its totals reconcile per node, pool, and
  cluster.

As on the real backend, get_cluster_metrics ignores nodePoolName and returns
cluster-wide series.
"""

from datetime import datetime, timedelta, timezone
import json
import sys


LIVE_CLUSTER = "10000000-0000-4000-8000-0000000000a1"
LAB_CLUSTER = "20000000-0000-4000-8000-0000000000c1"
LAB_NOW = "2026-10-07T13:30:00Z"

NODE_POOLS = [
    {"clusterId": LIVE_CLUSTER, "isDefault": True, "networkTopologyName": None,
     "nodePoolId": "100000", "nodePoolName": "default", "phase": "Ready"},
    {"clusterId": LAB_CLUSTER, "isDefault": True, "networkTopologyName": None,
     "nodePoolId": "200001", "nodePoolName": "default", "phase": "Ready"},
    {"clusterId": LAB_CLUSTER, "isDefault": False, "networkTopologyName": None,
     "nodePoolId": "200002", "nodePoolName": "train-pool", "phase": "Ready"},
    {"clusterId": LAB_CLUSTER, "isDefault": False, "networkTopologyName": None,
     "nodePoolId": "200003", "nodePoolName": "infer-pool", "phase": "Ready"},
    {"clusterId": LAB_CLUSTER, "isDefault": False, "networkTopologyName": None,
     "nodePoolId": "200004", "nodePoolName": "edge-pool", "phase": "Unschedulable"},
    {"clusterId": LAB_CLUSTER, "isDefault": False, "networkTopologyName": None,
     "nodePoolId": "200005", "nodePoolName": "cpu-pool", "phase": "Ready"},
    {"clusterId": LAB_CLUSTER, "isDefault": False, "networkTopologyName": None,
     "nodePoolId": "200006", "nodePoolName": "archive-pool", "phase": "Empty"},
]

NODES_PER_POOL = {
    LIVE_CLUSTER: {"default": 5},
    LAB_CLUSTER: {"default": 4, "train-pool": 4, "infer-pool": 2, "edge-pool": 1,
                  "cpu-pool": 3, "archive-pool": 0},
}

UNHEALTHY_NODES = {
    LIVE_CLUSTER: [],
    LAB_CLUSTER: [
        {
            "name": "cpu-node-01",
            "nodePool": "cpu-pool",
            "status": "NotReady",
            "conditions": [{"type": "Ready", "reason": "KubeletNotReady",
                            "message": "container runtime network not ready: cni plugin not initialized"}],
            "taints": [{"key": "node.kubernetes.io/not-ready", "effect": "NoSchedule"}],
        },
        {
            "name": "node-edge-01",
            "nodePool": "edge-pool",
            "status": "Unknown",
            "gpus": {"model": "NVIDIA-L4", "count": 8},
            "conditions": [{"type": "Ready", "reason": "NodeStatusUnknown",
                            "message": "Kubelet stopped posting node status."}],
            "taints": [{"key": "node.kubernetes.io/unreachable", "effect": "NoExecute"},
                       {"key": "node.kubernetes.io/unreachable", "effect": "NoSchedule"}],
        },
        {
            "name": "node-train-03",
            "nodePool": "train-pool",
            "status": "NotReady",
            "gpus": {"model": "NVIDIA-A100-SXM4-80GB", "count": 8},
            "conditions": [{"type": "Ready", "reason": "KubeletNotReady",
                            "message": "PLEG is not healthy: pleg was last seen active 4m12s ago"}],
            "taints": [{"key": "node.kubernetes.io/not-ready", "effect": "NoSchedule"}],
        },
        {
            "name": "node-train-04",
            "nodePool": "train-pool",
            "status": "Unknown",
            "gpus": {"model": "NVIDIA-A100-SXM4-80GB", "count": 8},
            "conditions": [{"type": "Ready", "reason": "NodeStatusUnknown",
                            "message": "Kubelet stopped posting node status."},
                           {"type": "UnschedulableCordoned", "reason": "NodeCordoned",
                            "message": "Node cordoned by runai cordon for maintenance."}],
            "taints": [{"key": "runai", "value": "drain", "effect": "NoSchedule"},
                       {"key": "node.kubernetes.io/unreachable", "effect": "NoExecute"}],
        },
    ],
}

INVENTORY_POOLS = {
    LIVE_CLUSTER: [
        {"nodePool": "default", "gpuNodes": 1, "gpusTotal": 16, "gpusAllocatable": 16,
         "gpusAllocated": 0, "gpusFree": 16, "cpuCoresTotal": 40, "cpuCoresAllocated": 15.74,
         "memoryGiBTotal": 153.255, "memoryGiBAllocated": 28.771},
    ],
    LAB_CLUSTER: [
        {"nodePool": "default", "gpuNodes": 4, "gpusTotal": 32, "gpusAllocatable": 32,
         "gpusAllocated": 32, "gpusFree": 0, "cpuCoresTotal": 256, "cpuCoresAllocated": 188.5,
         "memoryGiBTotal": 2048, "memoryGiBAllocated": 1410.25},
        {"nodePool": "train-pool", "gpuNodes": 4, "gpusTotal": 32, "gpusAllocatable": 16,
         "gpusAllocated": 12, "gpusFree": 4, "cpuCoresTotal": 256, "cpuCoresAllocated": 101.0,
         "memoryGiBTotal": 2048, "memoryGiBAllocated": 612.5},
        {"nodePool": "infer-pool", "gpuNodes": 2, "gpusTotal": 16, "gpusAllocatable": 16,
         "gpusAllocated": 0, "gpusFree": 16, "cpuCoresTotal": 128, "cpuCoresAllocated": 6.4,
         "memoryGiBTotal": 1024, "memoryGiBAllocated": 24.0},
        {"nodePool": "edge-pool", "gpuNodes": 1, "gpusTotal": 8, "gpusAllocatable": 0,
         "gpusAllocated": 0, "gpusFree": 0, "cpuCoresTotal": 64, "cpuCoresAllocated": 0,
         "memoryGiBTotal": 512, "memoryGiBAllocated": 0},
        {"nodePool": "cpu-pool", "gpuNodes": 0, "gpusTotal": 0, "gpusAllocatable": 0,
         "gpusAllocated": 0, "gpusFree": 0, "cpuCoresTotal": 96, "cpuCoresAllocated": 41.2,
         "memoryGiBTotal": 384, "memoryGiBAllocated": 150.75},
    ],
}

GPU_MODELS = {
    LIVE_CLUSTER: [("default", "Tesla-V100", 1, 16)],
    LAB_CLUSTER: [
        ("default", "NVIDIA-H100-80GB-HBM3", 4, 32),
        ("train-pool", "NVIDIA-A100-SXM4-80GB", 4, 32),
        ("infer-pool", "NVIDIA-L40S", 2, 16),
        ("edge-pool", "NVIDIA-L4", 1, 8),
    ],
}

AS_OF = {
    LIVE_CLUSTER: {"health": "2026-10-07T13:27:13Z", "health_pool": "2026-10-07T13:27:21Z",
                   "inventory": "2026-10-07T13:27:11Z", "summary": "2026-10-07T13:27:12Z",
                   "metrics": "2026-10-07T13:27:18Z"},
    LAB_CLUSTER: {"health": LAB_NOW, "health_pool": LAB_NOW, "inventory": LAB_NOW,
                  "summary": LAB_NOW, "metrics": LAB_NOW},
}

PROJECTS = {
    "team-vision": ("4500101", "research"),
    "team-nlp": ("4500102", "research"),
    "inference-prod": ("4500103", "platform"),
    "search-svc": ("4500104", "platform"),
}


def workload(index, name, phase, kind, project, pools, requested, allocated, updated,
             *, current=None, pending_seconds=0, priority=50, preemptible=True, idle=0.0):
    """Build one upstream-shaped workload row plus private fixture fields."""
    project_id, department = PROJECTS[project]
    return {
        "id": f"00000000-0000-4000-8000-0000000c{index:04d}",
        "name": name,
        "type": kind,
        "phase": phase,
        "projectId": project_id,
        "projectName": project,
        "clusterId": LAB_CLUSTER,
        "createdAt": updated if phase == "Pending" else "2026-10-07T06:00:00Z",
        "phaseUpdatedAt": updated,
        "requestedNodePools": pools,
        "currentNodePools": pools if current is None else current,
        "priority": priority,
        "priorityClassName": {50: "train", 75: "build", 125: "inference"}[priority],
        "preemptible": preemptible,
        "idleAllocatedGpus": idle,
        "allocatedGpu": allocated,
        "requestedGpu": requested,
        "_department": department,
        "_pendingSeconds": pending_seconds,
    }


def pending(index, name, kind, project, pools, requested, minutes):
    """Build a Pending row whose wait ends at LAB_NOW."""
    updated = (parse_time(LAB_NOW) - timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return workload(index, name, "Pending", kind, project, pools, requested, 0, updated,
                    current=[], pending_seconds=minutes * 60,
                    priority=125 if kind == "Inference" else 50,
                    preemptible=kind != "Inference")


def parse_time(value):
    return datetime.strptime(value.replace("+00:00", "Z"), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


WORKLOADS = [
    workload(1, "vit-pretrain-12", "Running", "Training", "team-vision", ["default"], 8, 8, "2026-10-07T07:10:00Z", pending_seconds=420),
    workload(2, "llm-sft-7", "Running", "Distributed", "team-nlp", ["default"], 16, 16, "2026-10-07T08:02:00Z", pending_seconds=1380),
    workload(3, "chat-api", "Running", "Inference", "inference-prod", ["default"], 4, 4, "2026-10-06T22:00:00Z", priority=125, preemptible=False),
    workload(4, "rerank-api", "Running", "Inference", "inference-prod", ["default"], 2, 2, "2026-10-06T22:05:00Z", priority=125, preemptible=False),
    workload(5, "notebook-ana", "Running", "Workspace", "team-vision", ["default"], 1, 1, "2026-10-07T09:15:00Z", priority=75, preemptible=False, idle=1.0),
    workload(6, "eval-harness", "Running", "Training", "team-nlp", ["default"], 1, 1, "2026-10-07T11:40:00Z", pending_seconds=90),
    workload(7, "seg-train-3", "Running", "Training", "team-vision", ["train-pool"], 8, 8, "2026-10-07T10:30:00Z", pending_seconds=300),
    workload(8, "tok-train-1", "Running", "Training", "team-nlp", ["train-pool"], 4, 4, "2026-10-07T11:00:00Z", pending_seconds=60),
    pending(9, "embed-1", "Inference", "search-svc", ["infer-pool"], 1, 47),
    pending(10, "embed-2", "Inference", "search-svc", ["infer-pool"], 1, 41),
    pending(11, "batch-7", "Training", "team-nlp", ["default"], 8, 35),
    pending(12, "pinned-h200", "Training", "team-vision", ["train-pool"], 2, 30),
    pending(13, "svc-a", "Training", "team-vision", ["default"], 4, 22),
    pending(14, "eval-3", "Training", "team-nlp", ["default"], 2, 18),
    pending(15, "fresh-1", "Workspace", "team-vision", [], 1, 3),
    workload(16, "seg-train-2", "Failed", "Training", "team-vision", ["train-pool"], 8, 0, "2026-10-07T13:05:10Z", current=[], pending_seconds=240),
    workload(17, "tok-eval-4", "Failed", "Training", "team-nlp", ["train-pool"], 4, 0, "2026-10-07T13:06:02Z", current=[], pending_seconds=30),
    workload(18, "vit-sweep-1", "Completed", "Training", "team-vision", ["default"], 2, 0, "2026-10-07T10:12:00Z", current=[], pending_seconds=120),
    workload(19, "vit-sweep-2", "Completed", "Training", "team-vision", ["default"], 2, 0, "2026-10-07T10:48:00Z", current=[], pending_seconds=180),
    workload(20, "tok-build-9", "Completed", "Training", "team-nlp", ["default"], 1, 0, "2026-10-07T11:20:00Z", current=[], pending_seconds=20),
    workload(21, "seg-eval-5", "Completed", "Training", "team-vision", ["train-pool"], 4, 0, "2026-10-07T11:55:00Z", current=[], pending_seconds=40),
    workload(22, "data-prep-2", "Completed", "Training", "team-nlp", ["default"], 1, 0, "2026-10-07T12:20:00Z", current=[], pending_seconds=10),
]

ROW_BY_NAME = {row["name"]: row for row in WORKLOADS}


def capacity_status(name, gpus, nodes):
    """Status signals of a pending workload that the scheduler reports as not fitting."""
    row = ROW_BY_NAME[name]
    pool = row["requestedNodePools"][0]
    message = f"0/{nodes} nodes are available: {nodes} Insufficient nvidia.com/gpu."
    return {
        "phaseMessage": f"Not enough resources in node pool {pool}: requested {gpus} GPUs, 0 GPUs free.",
        "phaseReason": "Unschedulable",
        "k8sPhase": "Pending",
        "conditions": [{"type": "PodScheduled", "status": "False", "reason": "Unschedulable",
                        "message": message, "lastTransitionTime": row["phaseUpdatedAt"]}],
        "events": [{"createdAt": row["phaseUpdatedAt"], "type": "Warning",
                    "reason": "Unschedulable", "message": message}],
    }


STATUS_SIGNALS = {
    "embed-1": {"k8sPhase": "Pending", "conditions": [], "events": []},
    "embed-2": {"k8sPhase": "Pending", "conditions": [], "events": []},
    "batch-7": capacity_status("batch-7", 8, 4),
    "svc-a": capacity_status("svc-a", 4, 4),
    "eval-3": capacity_status("eval-3", 2, 4),
    # Recorded from a live pinned workload: once the scheduler retried, its recent
    # events named only the GPU shortage; the node-type pin is visible in the spec.
    "pinned-h200": {
        "k8sPhase": "Pending",
        "conditions": [{"type": "PodScheduled", "status": "False", "reason": "Unschedulable",
                        "message": "no nodes with enough resources were found: 4 node(s) didn't have enough resources: GPUs.",
                        "lastTransitionTime": "2026-10-07T13:20:00Z"}],
        "events": [{"createdAt": "2026-10-07T13:20:00Z", "type": "Warning", "reason": "Unschedulable",
                    "message": "no nodes with enough resources were found: 4 node(s) didn't have enough resources: GPUs."}],
    },
    "fresh-1": {"k8sPhase": "Pending", "conditions": [], "events": []},
    "seg-train-2": {
        "phaseMessage": "Pod evicted: node node-train-03 became NotReady.",
        "phaseReason": "PodFailed",
        "k8sPhase": "Failed",
        "conditions": [{"type": "Failed", "status": "True", "reason": "PodFailed",
                        "message": "Pod evicted: node node-train-03 became NotReady.",
                        "lastTransitionTime": "2026-10-07T13:05:10Z"}],
        "events": [{"createdAt": "2026-10-07T13:05:10Z", "type": "Warning", "reason": "Evicted",
                    "message": "The node was low on resource: node not ready."}],
    },
    "tok-eval-4": {
        "phaseMessage": "Pod evicted: node node-train-03 became NotReady.",
        "phaseReason": "PodFailed",
        "k8sPhase": "Failed",
        "conditions": [{"type": "Failed", "status": "True", "reason": "PodFailed",
                        "message": "Pod evicted: node node-train-03 became NotReady.",
                        "lastTransitionTime": "2026-10-07T13:06:02Z"}],
        "events": [{"createdAt": "2026-10-07T13:06:02Z", "type": "Warning", "reason": "Evicted",
                    "message": "The node was low on resource: node not ready."}],
    },
}

METRIC_NAMES = ["allocatedGpu", "unallocatedGpu", "totalGpu", "gpuUtilization",
                "gpuMemoryUtilization", "cpuUtilization", "cpuMemoryUtilization",
                "avgWorkloadWaitTime", "workloadsCount"]
DEFAULT_METRICS = ["allocatedGpu", "unallocatedGpu", "gpuUtilization"]
WINDOWS = {"last_1h": timedelta(hours=1), "last_24h": timedelta(hours=24),
           "last_7d": timedelta(days=7), "last_30d": timedelta(days=30)}

# unallocatedGpu, avgWorkloadWaitTime and workloadsCount came back absent with
# partialData=true on the live cluster; cpuMemoryUtilization was never recorded,
# so it is not replayed either.
UNAVAILABLE_METRICS = {"unallocatedGpu", "avgWorkloadWaitTime", "workloadsCount", "cpuMemoryUtilization"}

METRIC_VALUES = {
    (LIVE_CLUSTER, None): {"allocatedGpu": [0], "totalGpu": [16], "gpuUtilization": [0],
                           "gpuMemoryUtilization": [0],
                           "cpuUtilization": [11.271, 11.366, 11.232, 11.342]},
    (LAB_CLUSTER, None): {"allocatedGpu": [44], "totalGpu": [88],
                          "gpuUtilization": [39.6, 39.1, 40.1, 38.8],
                          "gpuMemoryUtilization": [35.2, 34.8, 35.9, 34.6],
                          "cpuUtilization": [41.3, 40.7, 42.0, 40.9]},
}


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
    if cluster_id not in NODES_PER_POOL:
        raise ToolError(f"not_found: cluster {cluster_id} not found")
    return cluster_id


def require_pool(cluster_id, pool):
    """Fail like the server when a node pool name does not exist in the cluster."""
    if pool and pool not in NODES_PER_POOL[cluster_id]:
        raise ToolError(f"not_found: node pool {pool} not found in cluster {cluster_id}")


def list_node_pools(args):
    require_known(args, {"clusterId", "limit", "offset"})
    cluster_id = require_cluster(args, required=False)
    pools = [pool for pool in NODE_POOLS if not cluster_id or pool["clusterId"] == cluster_id]
    limit = min(int(args.get("limit") or 50), 200)
    offset = int(args.get("offset") or 0)
    page = pools[offset:offset + limit]
    result = {"nodePools": page, "hasMore": offset + limit < len(pools)}
    if result["hasMore"]:
        result["nextOffset"] = offset + limit
    return result


def get_cluster_infrastructure_health(args):
    require_known(args, {"clusterId", "nodePoolName"})
    cluster_id = require_cluster(args)
    pool = args.get("nodePoolName")
    require_pool(cluster_id, pool)
    nodes = [node for node in UNHEALTHY_NODES[cluster_id] if not pool or node["nodePool"] == pool]
    by_pool = {}
    for node in nodes:
        entry = by_pool.setdefault(node["nodePool"], {"nodePool": node["nodePool"],
                                                      "nodesUnhealthy": 0, "gpusUnschedulable": 0})
        entry["nodesUnhealthy"] += 1
        entry["gpusUnschedulable"] += node.get("gpus", {}).get("count", 0)
    totals_scope = NODES_PER_POOL[cluster_id]
    return {
        "clusterId": cluster_id,
        "nodePool": pool,
        "asOf": AS_OF[cluster_id]["health_pool" if pool else "health"],
        "totals": {
            "nodesTotal": totals_scope.get(pool, 0) if pool else sum(totals_scope.values()),
            "nodesUnhealthy": len(nodes),
            "gpusUnschedulable": sum(entry["gpusUnschedulable"] for entry in by_pool.values()),
        },
        "byNodePool": [by_pool[name] for name in sorted(by_pool)],
        "unhealthyNodes": nodes,
    }


def get_cluster_physical_inventory(args):
    require_known(args, {"clusterId", "nodePoolName"})
    cluster_id = require_cluster(args)
    pool = args.get("nodePoolName")
    require_pool(cluster_id, pool)
    pools = [item for item in INVENTORY_POOLS[cluster_id] if not pool or item["nodePool"] == pool]
    keys = ["gpuNodes", "gpusTotal", "gpusAllocatable", "gpusAllocated", "gpusFree",
            "cpuCoresTotal", "cpuCoresAllocated", "memoryGiBTotal", "memoryGiBAllocated"]
    totals = {key: round(sum(item[key] for item in pools), 3) for key in keys}
    models = [
        {"gpuModel": model, "nodes": nodes, "gpusTotal": gpus}
        for model_pool, model, nodes, gpus in GPU_MODELS[cluster_id]
        if not pool or model_pool == pool
    ]
    return {
        "clusterId": cluster_id,
        "nodePool": pool,
        "asOf": AS_OF[cluster_id]["inventory"],
        "totals": totals,
        "byNodePool": [dict(item) for item in pools],
        "byGpuModel": models,
    }


def get_cluster_metrics(args):
    """Return cluster-wide series even for a nodePoolName, as the real backend does."""
    require_known(args, {"clusterId", "nodePoolName", "metricTypes", "window", "start",
                         "end", "numberOfSamples"})
    cluster_id = require_cluster(args)
    pool = args.get("nodePoolName")
    requested = args.get("metricTypes") or DEFAULT_METRICS
    unknown = [name for name in requested if name not in METRIC_NAMES]
    if unknown:
        raise ToolError(f"validation: unknown metricTypes {unknown}; one of {METRIC_NAMES}")
    samples = int(args.get("numberOfSamples") or 20)
    if samples < 1 or samples > 1000:
        raise ToolError("validation: numberOfSamples must be between 1 and 1000")
    if args.get("start") and args.get("end"):
        start, end = parse_time(args["start"]), parse_time(args["end"])
    else:
        window = args.get("window") or "last_1h"
        if window not in WINDOWS:
            raise ToolError(f"validation: window must be one of {sorted(WINDOWS)}")
        end = parse_time(AS_OF[cluster_id]["metrics"])
        start = end - WINDOWS[window]
    span = (end - start).total_seconds()
    times = [
        (start + timedelta(seconds=round(span * index / (samples - 1))) if samples > 1 else end)
        .strftime("%Y-%m-%dT%H:%M:%SZ")
        for index in range(samples)
    ]
    values = METRIC_VALUES[(cluster_id, None)]
    series = {}
    for name in requested:
        if name in UNAVAILABLE_METRICS or name not in values:
            continue
        cycle = values[name]
        series[name] = [{"t": stamp, "v": cycle[index % len(cycle)]} for index, stamp in enumerate(times)]
    return {
        "clusterId": cluster_id,
        "nodePool": pool,
        "timeRange": {"from": times[0], "to": times[-1], "numberOfSamples": samples},
        "series": series,
        "seriesTruncated": False,
        "partialData": any(name not in series for name in requested),
    }


def scoped_workloads(cluster_id):
    """Return workloads visible in one cluster (all clusters when None)."""
    return [row for row in WORKLOADS if not cluster_id or row["clusterId"] == cluster_id]


def get_workloads_summary(args):
    """Return phase counts; a node-pool scope only sees workloads placed in that pool."""
    require_known(args, {"clusterId", "orgType", "orgName", "nodePoolName", "gpuIdleOnly"})
    cluster_id = require_cluster(args, required=False)
    rows = scoped_workloads(cluster_id)
    scope = {}
    if cluster_id:
        scope["clusterId"] = cluster_id
    org_type, org_name = args.get("orgType"), args.get("orgName")
    if bool(org_type) != bool(org_name):
        raise ToolError("validation: orgType and orgName must be set together")
    if org_type:
        if org_type == "project":
            if org_name not in PROJECTS:
                raise ToolError(f"not_found: project {org_name} not found")
            rows = [row for row in rows if row["projectName"] == org_name]
            org_unit_id = PROJECTS[org_name][0]
        elif org_type == "department":
            departments = {dept for _, dept in PROJECTS.values()}
            if org_name not in departments:
                raise ToolError(f"not_found: department {org_name} not found")
            rows = [row for row in rows if row["_department"] == org_name]
            org_unit_id = {"research": "4500010", "platform": "4500011"}[org_name]
        else:
            raise ToolError("validation: orgType must be project or department")
        scope.update({"orgType": org_type, "orgName": org_name, "orgUnitId": org_unit_id})
    pool = args.get("nodePoolName")
    if pool:
        rows = [row for row in rows if pool in row["currentNodePools"]]
        scope["nodePool"] = pool
    if args.get("gpuIdleOnly"):
        rows = [row for row in rows if row["phase"] == "Running" and row["idleAllocatedGpus"]]
        scope["gpuIdleOnly"] = True
    buckets = {"running": 0, "pending": 0, "failed": 0, "completed": 0, "stopped": 0, "other": 0}
    fold = {"Running": "running", "Pending": "pending", "Initializing": "pending",
            "Creating": "pending", "Resuming": "pending", "Failed": "failed", "Degraded": "failed",
            "Completed": "completed", "Stopped": "stopped", "Stopping": "stopped"}
    by_phase = {}
    for row in rows:
        by_phase[row["phase"]] = by_phase.get(row["phase"], 0) + 1
        buckets[fold.get(row["phase"], "other")] += 1
    as_of = AS_OF[cluster_id]["summary"] if cluster_id else LAB_NOW
    return {"scope": scope, "asOf": as_of, "total": len(rows), "counts": buckets, "byPhase": by_phase}


DETAIL_FIELDS = {"createdAt", "requestedNodePools", "currentNodePools", "priority",
                 "priorityClassName", "preemptible", "idleAllocatedGpus"}
SORT_KEYS = {
    "name": "name", "type": "type", "phase": "phase", "projectName": "projectName",
    "projectId": "projectId", "createdAt": "createdAt", "phaseUpdatedAt": "phaseUpdatedAt",
    "priority": "priority", "allocatedGPU": "allocatedGpu", "allocatedGpu": "allocatedGpu",
    "totalPendingTimeSeconds": "_pendingSeconds", "idleAllocatedGpus": "idleAllocatedGpus",
}
FILTER_FIELDS = {
    "type": "type", "name": "name", "phase": "phase", "projectName": "projectName",
    "projectId": "projectId", "departmentName": "_department", "nodepool": "currentNodePools",
    "allocatedGPU": "allocatedGpu", "totalPendingTimeSeconds": "_pendingSeconds",
    "phaseUpdatedAt": "phaseUpdatedAt", "createdAt": "createdAt", "priority": "priority",
}
NUMERIC_FIELDS = {"allocatedGpu", "_pendingSeconds", "priority"}


def compare(actual, operator, expected, numeric):
    if isinstance(actual, list):
        if operator == "==":
            return expected in actual
        if operator == "!=":
            return expected not in actual
        raise ToolError(f"validation: operator {operator} is not supported for nodepool")
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
        item[key] = value
    return item


def list_workloads(args):
    require_known(args, {"clusterId", "phase", "type", "name", "projectId", "projectName",
                         "departmentName", "nodePoolName", "createdAfter", "phaseUpdatedAfter",
                         "totalPendingTimeSeconds", "filterBy", "includeDetails", "sortBy",
                         "sortOrder", "limit", "offset"})
    cluster_id = require_cluster(args, required=False)
    rows = scoped_workloads(cluster_id)
    for field in ("phase", "type", "name", "projectId"):
        if args.get(field):
            rows = [row for row in rows if row[field] == args[field]]
    if args.get("nodePoolName"):
        rows = [row for row in rows if args["nodePoolName"] in row["currentNodePools"]]
    if args.get("createdAfter"):
        rows = [row for row in rows if parse_time(row["createdAt"]) >= parse_time(args["createdAfter"])]
    if args.get("phaseUpdatedAfter"):
        rows = [row for row in rows
                if parse_time(row["phaseUpdatedAt"]) >= parse_time(args["phaseUpdatedAfter"])]
    comparisons = [(name, args[name]) for name in ("projectName", "departmentName", "totalPendingTimeSeconds")
                   if args.get(name)]
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
    rows = sorted(rows, key=lambda row: (row[SORT_KEYS[sort_by]], row["name"]), reverse=order == "desc")
    limit = int(args.get("limit") or 50)
    offset = int(args.get("offset") or 0)
    if limit < 1 or limit > 200 or offset < 0:
        raise ToolError("validation: limit must be 1-200 and offset non-negative")
    page = rows[offset:offset + limit]
    result = {"workloads": [public_row(row, bool(args.get("includeDetails"))) for row in page],
              "hasMore": offset + limit < len(rows)}
    if result["hasMore"]:
        result["nextOffset"] = offset + limit
    return result


def get_workload_status(args):
    require_known(args, {"workloadId"})
    workload_id = args.get("workloadId")
    row = next((item for item in WORKLOADS if item["id"] == workload_id), None)
    if row is None:
        raise ToolError(f"not_found: workload {workload_id} not found")
    result = {
        "id": row["id"],
        "name": row["name"],
        "clusterId": row["clusterId"],
        "type": row["type"],
        "phase": row["phase"],
        "projectName": row["projectName"],
        "departmentName": row["_department"],
        "createdAt": row["createdAt"],
        "phaseUpdatedAt": row["phaseUpdatedAt"],
        "conditions": [],
        "events": [],
    }
    if row["phase"] == "Running":
        result["k8sPhase"] = "Running"
        result["conditions"] = [{"type": "Available", "status": "True", "reason": "Running",
                                 "lastTransitionTime": row["phaseUpdatedAt"]}]
    if row["phase"] == "Completed":
        result["k8sPhase"] = "Succeeded"
    result.update(STATUS_SIGNALS.get(row["name"], {}))
    return result


SPEC_PLACEMENT = {"pinned-h200": {"nodeType": "gpu-h200"}}

# Older scheduler events that no longer appear among the recent status events.
# Each retry logs another shortage event, so at the default page size the
# affinity mismatch lands on the second history page.
HISTORY_EVENTS = {
    "pinned-h200": [{"createdAt": "2026-10-07T13:00:30Z", "type": "Warning", "reason": "Unschedulable",
                     "message": "0/4 nodes are available: 4 node(s) didn't match Pod's node affinity/selector."}]
    + [{"createdAt": f"2026-10-07T13:{minute:02d}:00Z", "type": "Warning", "reason": "Unschedulable",
        "message": "no nodes with enough resources were found: 4 node(s) didn't have enough resources: GPUs."}
       for minute in range(1, 20)],
}


def get_workload_history(args):
    require_known(args, {"workloadId", "limit", "offset"})
    workload_id = args.get("workloadId")
    row = next((item for item in WORKLOADS if item["id"] == workload_id), None)
    if row is None:
        raise ToolError(f"not_found: workload {workload_id} not found")
    limit = int(args.get("limit") or 20)
    offset = int(args.get("offset") or 0)
    if limit < 1 or limit > 50 or offset < 0:
        raise ToolError("validation: limit must be 1-50 and offset non-negative")
    events = STATUS_SIGNALS.get(row["name"], {}).get("events", []) + HISTORY_EVENTS.get(row["name"], [])
    records = [{"meta": {"creationTimestamp": event["createdAt"], "type": "Event"},
                "spec": {"event": event}} for event in events]
    records.append({"meta": {"creationTimestamp": row["createdAt"], "type": "PhaseUpdate"},
                    "spec": {"phaseUpdate": {"phase": "Pending", "phaseMessage": ""}}})
    records.sort(key=lambda record: record["meta"]["creationTimestamp"], reverse=True)
    result = {"workloadId": row["id"], "type": row["type"], "records": records[offset:offset + limit]}
    if offset + limit < len(records):
        result["next"] = offset + limit
    return result


def get_workload_spec(args):
    """Return the compact requested-spec summary, with scheduler messages for pending work."""
    require_known(args, {"workloadId", "includeFullSpec"})
    workload_id = args.get("workloadId")
    row = next((item for item in WORKLOADS if item["id"] == workload_id), None)
    if row is None:
        raise ToolError(f"not_found: workload {workload_id} not found")
    spec = {"image": "nvcr.io/nvidia/pytorch:24.08-py3",
            "compute": {"gpuDevicesRequest": row["requestedGpu"]},
            "nodePools": row["requestedNodePools"]}
    spec.update(SPEC_PLACEMENT.get(row["name"], {}))
    result = {"workloadId": row["id"], "name": row["name"], "type": row["type"],
              "projectName": row["projectName"], "departmentName": row["_department"], "spec": spec}
    if row["phase"] == "Running":
        result["allocatedResources"] = {"gpu": float(row["allocatedGpu"])}
    if row["phase"] == "Pending":
        events = STATUS_SIGNALS.get(row["name"], {}).get("events", [])
        pool = (row["requestedNodePools"] or [""])[0]
        result["pendingSchedulingMessages"] = [
            {"nodePool": pool, "phaseReason": "Unschedulable", "reason": event["reason"],
             "userMessage": event["message"]} for event in events]
    return result


def schema(properties, required=None):
    return {"type": "object", "properties": properties, "required": required or [],
            "additionalProperties": False}


COMPARISON = {"type": "object", "properties": {"operator": {"type": "string"}, "value": {"type": "string"}},
              "required": ["operator", "value"], "additionalProperties": False}

TOOLS = {
    "list_node_pools": {
        "description": "List the node pools in a cluster, or across every cluster when clusterId is omitted. Returns each pool's identity, cluster, whether it is the cluster default, its lifecycle phase (Creating, Updating, Deleting, Empty, Unschedulable, Ready), and network topology. Does NOT return quota, metrics, or GPU capacity. Read-only.",
        "inputSchema": schema({"clusterId": {"type": "string"}, "limit": {"type": "integer"},
                               "offset": {"type": "integer"}}),
        "handler": list_node_pools,
    },
    "get_cluster_infrastructure_health": {
        "description": "Get a cluster's current node-level infrastructure health: nodes that are NotReady or Unknown, with their k8s conditions, taints, and GPU counts, plus cluster and per-pool counts. Node-level only: it will NOT detect a failing GPU on an otherwise-Ready node. Point-in-time; cannot answer historical questions. Read-only.",
        "inputSchema": schema({"clusterId": {"type": "string"}, "nodePoolName": {"type": "string"}},
                              ["clusterId"]),
        "handler": get_cluster_infrastructure_health,
    },
    "get_cluster_physical_inventory": {
        "description": "Get a cluster's current hardware capacity per node pool and per GPU model: GPU nodes, GPUs total, allocatable (on Ready nodes), allocated, and free (fully idle devices), plus CPU cores and memory. Do NOT call for quota or utilization over time. Read-only.",
        "inputSchema": schema({"clusterId": {"type": "string"}, "nodePoolName": {"type": "string"}},
                              ["clusterId"]),
        "handler": get_cluster_physical_inventory,
    },
    # Like the real server, the description advertises nodePoolName scoping that the backend does not apply.
    "get_cluster_metrics": {
        "description": "Get a cluster's resource capacity and consumption over time: allocatedGpu, unallocatedGpu, totalGpu, gpuUtilization, gpuMemoryUtilization, cpuUtilization, cpuMemoryUtilization, avgWorkloadWaitTime, workloadsCount. Scope to one node pool with nodePoolName. window is one of last_1h, last_24h, last_7d, last_30d (default last_1h). partialData is true when some metrics failed; failed metrics are absent. Read-only.",
        "inputSchema": schema({
            "clusterId": {"type": "string"}, "nodePoolName": {"type": "string"},
            "metricTypes": {"type": ["array", "null"], "items": {"type": "string"}},
            "window": {"type": "string"}, "start": {"type": "string"}, "end": {"type": "string"},
            "numberOfSamples": {"type": "integer"},
        }, ["clusterId"]),
        "handler": get_cluster_metrics,
    },
    "get_workloads_summary": {
        "description": "Counts of workloads bucketed by phase (running, pending, failed, completed, stopped, other) for a cluster, a project or department, a node pool, or the whole platform. Pending workloads have no node-pool placement, so a nodePoolName scope does not count them. Do NOT call for why a workload is pending. Read-only.",
        "inputSchema": schema({"clusterId": {"type": "string"}, "orgType": {"type": "string"},
                               "orgName": {"type": "string"}, "nodePoolName": {"type": "string"},
                               "gpuIdleOnly": {"type": "boolean"}}),
        "handler": get_workloads_summary,
    },
    "list_workloads": {
        "description": "List workloads with a compact response (identity, type, phase, project, cluster, last phase update, requested and allocated GPUs). Set includeDetails for creation time, requested/current node pools, priority, preemptibility, and idle GPUs. Filter by cluster, phase, type, name, project, node pool (current placement only), or typed filterBy comparisons; sortBy accepts name, type, phase, projectName, projectId, createdAt, phaseUpdatedAt, priority, allocatedGPU, idleAllocatedGpus, totalPendingTimeSeconds. Read-only.",
        "inputSchema": schema({
            "clusterId": {"type": "string"}, "phase": {"type": "string"}, "type": {"type": "string"},
            "name": {"type": "string"}, "projectId": {"type": "string"},
            "projectName": COMPARISON, "departmentName": COMPARISON, "totalPendingTimeSeconds": COMPARISON,
            "nodePoolName": {"type": "string"}, "createdAfter": {"type": "string"},
            "phaseUpdatedAfter": {"type": "string"},
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
        "description": "Get one workload's current phase, phaseMessage, phaseReason, k8sPhase, conditions, and up to 5 recent events. For 'why' questions check phaseMessage first, then condition and event reasons. Read-only.",
        "inputSchema": schema({"workloadId": {"type": "string"}}, ["workloadId"]),
        "handler": get_workload_status,
    },
    "get_workload_history": {
        "description": "Call when the user asks about a workload's event history \u2014 what happened to a workload (job) over time and the lifecycle events logged along the way. Returns a chronological, paginated list of history records, newest first; each record is a phase update or an event (a lifecycle event with its reason and message). Pass the returned next value as offset to get older records. Read-only.",
        "inputSchema": schema({"workloadId": {"type": "string"}, "limit": {"type": "integer"},
                               "offset": {"type": "integer"}}, ["workloadId"]),
        "handler": get_workload_history,
    },
    "get_workload_spec": {
        "description": "Call when the user asks what a workload (job) requested or how it is configured \u2014 the resources it asked for, the image it runs, its environment variables, volumes, or node placement \u2014 or why it is pending scheduling. Returns a compact requested-spec summary by default (image, compute, storage, scaling, placement such as nodePools, nodeType, nodeAffinityRequired, podAffinity, tolerations, and scheduling); set includeFullSpec for the complete spec. Also returns allocated resources and, for pending workloads, the scheduler's pending messages. Read-only.",
        "inputSchema": schema({"workloadId": {"type": "string"}, "includeFullSpec": {"type": "boolean"}},
                              ["workloadId"]),
        "handler": get_workload_spec,
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
                "serverInfo": {"name": "runai-cluster-health-summary-eval", "version": "1.0.0"}}})
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
