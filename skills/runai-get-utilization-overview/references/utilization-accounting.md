# GPU utilization accounting

Why held, allocated, free, and utilized GPU numbers disagree, and how to read
them. Read this when the numbers in an answer do not reconcile or the user
asks why.

## Three different numbers

| Number | Where it comes from | Meaning |
| --- | --- | --- |
| Deserved (quota) | `deserved` under `gpu` in `list_project_resources`; `gpuQuota` in `list_departments` | GPUs guaranteed to the org unit. Not a measure of use. |
| Allocated (held) | `allocated` under `gpu`; `gpuAllocation`; `allocatedGpu` | GPUs the scheduler gave to running workloads, whether or not they compute. |
| Utilized | `gpuUtilization` | How busy the held GPUs are, in percent. |

This skill compares allocated with utilized. Deserved versus allocated
(over-quota borrowing, starvation, reclaim) belongs to
`runai-audit-assigned-quota`. A project can be over quota and idle, which
makes it a strong reclaim candidate, but the reclaim decision is a quota and
write question.

## Org-unit utilization is already per held GPU

`get_org_unit_metrics` reports a project's or department's `gpuUtilization`
weighted by the GPUs it holds: busy GPU time divided by held GPU time. A
project holding 8 GPUs at 3% is wasting about 7.8 GPUs, whatever the rest of
the cluster does.

## Cluster utilization is diluted

`get_cluster_metrics` reports `gpuUtilization` as the average over **every**
GPU in the cluster, including unallocated ones, which report 0%. A cluster
with 64 GPUs, 32 of them held and computing at 40%, reports 20%.

To get the utilization of the held GPUs:

```text
held utilization = gpuUtilization × totalGpu / allocatedGpu
idle held GPUs   = allocatedGpu × (1 − held utilization / 100)
```

The conversion assumes unallocated GPUs are idle.

## Node pools come from org-unit series

`get_cluster_metrics` accepts `nodePoolName`, but the current server applies
it only to workload counts: allocation, capacity, and utilization come back
cluster-wide for every pool. Org-unit metrics do honor `nodePoolName`, so a
pool's held GPUs and utilization are built from department series in that
pool: add the allocations and weight utilization by allocation at each
timestamp.

## Older clusters

Org-unit `gpuMemoryUtilization` needs cluster version 2.21 or later. On an
older cluster the whole metrics request is rejected while that metric is in
it; drop it and report memory utilization as unavailable.

## Utilization above allocation: GPUs outside Run:ai accounting

If `gpuUtilization × totalGpu / 100` is larger than `allocatedGpu`, more
devices are busy than Run:ai allocated. Pods that do not belong to a Run:ai
project (for example, fractional or system pods in other namespaces) use
devices that allocation and `gpusFree` do not count. Report it as GPUs busy
outside Run:ai accounting, cap the held utilization at 100%, and do not
attribute the extra work to any project.

## Fractional GPUs

A workload that requests part of a GPU holds that fraction: three 0.5-GPU
workloads hold 1.5 GPUs. Utilization is measured per physical device, and
several fractional workloads can share one device, so a fractional holder's
utilization is approximate.

## GPU memory is not compute

`gpuMemoryUtilization` shows memory held on the device, not work done. High
memory with near-zero compute means a model or dataset is loaded but nothing
runs: typical of idle notebooks and inference endpoints with no traffic. Such
a holder is idle for compute even though its memory is full.

## Bursty is not idle

Average utilization hides peaks. A holder at 31% average and 78% p95 runs hard
part of the time and idles between bursts (training with long data loading,
evaluation sweeps, daytime-only inference). Report both values. Reclaiming all
of its GPUs would starve the peaks; the waste is real but partial.

## `idleAllocatedGpus` is a snapshot

`list_workloads` returns `idleAllocatedGpus` for running workloads: how many
of the workload's allocated GPUs the cluster marks idle now, based on recent
GPU activity. It is a point-in-time value, refreshed periodically. It says
nothing about how long the workload has been idle; use
`get_workload_metrics` over the window for that, on request.

## Samples and GPU-hours

The MCP returns evenly spaced samples, not averages. The skill averages S
samples per window, so activity shorter than the step can be missed.
Idle GPU-hours are the sum over samples of `held × (1 − utilization / 100) ×
step`, and only cover the samples that have data.
