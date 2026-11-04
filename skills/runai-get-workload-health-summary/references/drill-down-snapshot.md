# Drill-down full health snapshot

Stage 4 of the workflow, after the `get_workload_status` gate. Follow these
rules when the user asks whether one workload is healthy, how it is doing, what
looks off, or explicitly requests a health snapshot.

## Sections

Return all five sections in order, followed by exactly one verdict.

1. **State:** phase, submit time, started time when available, last transition,
   recent five events, project, department, and kind.
2. **Spec:** GPU and memory ask, CPU, node pools, priority, preemptibility,
   image, mounts, runtime allocated resources, and actual placement.
3. **Pods:** MCP pod phase, requested/allocated resources, measurements, and
   node placement. Include desired-vs-ready, restart count, and readiness only
   when optional Kubernetes evidence was actually collected.
4. **Metrics:** request `start`, `end`, and `numberOfSamples`, then compute and
   label summaries from the returned time-series values. Report GPU utilization
   average and p95 and GPU-memory usage when those measurement types exist. For
   inference, summarize returned throughput/latency or NIM counters; do not
   invent percentiles, targets, request rates, or error rates that the returned
   samples and labels do not support.
5. **Policy/Notes:** effective policy when available, autoscaling caps,
   idle-timeout, preemption-related notes, missing evidence, and anything that
   needs follow-up.

## Policy lookup

For policy, use the numeric `projectId` returned by `list_workloads`. If the
drill-down began from a bare workload ID, resolve the status `projectName` with
`list_projects(clusterId)` and select an exact name match. Call
`get_workload_effective_policy(projectId, kind)` where `kind` is the lowercase
singular status type (`training`, `workspace`, `distributed`, or `inference`).
If the type is unsupported or project resolution is ambiguous, report policy
as unavailable rather than guessing.

## Verdict

End with exactly one verdict:

| Verdict | Meaning |
| --- | --- |
| `HEALTHY` | Running or completed as expected; MCP pod phases are healthy and no material warning is present. If optional readiness evidence was collected, it must also be healthy. |
| `DEGRADED` | Running but not fully healthy, near limit, restarting, low-use, or showing a readiness mismatch in evidence that was actually collected. |
| `FAILED` | Terminal failed state or failed pods. |
| `PENDING` | Waiting to start or initialize. |

Flag near-cap and off-nominal signals:

- GPU-memory usage above 90% of the workload's allocated GPU-memory limit
  (`gpuMemory` under `allocatedResources` in `get_workload_spec`). Label this
  denominator explicitly. If the limit is missing or cannot be converted to
  bytes, report absolute usage only and do not make a percentage-based verdict.
- Any Kubernetes-reported restart count above 0.
- Any Kubernetes-reported desired-vs-ready mismatch such as `3/4 Ready`.
- Inference p95 latency above two times its target when a target is available.
- GPU idle threshold crossed, using the threshold stated in the filter.

## Hand-off

When the drill-down verdict is not `HEALTHY` and the user asks for the cause,
route rather than guessing:

- Terminal, crashed, pod-start, OOM, image-pull, init, mount, node, or
  policy-terminated failures -> `runai-diagnose-failed-workload`.
- Slow or throttled inference -> `runai-diagnose-inference-latency`.
- Preempted workload -> `runai-diagnose-preempted-workload` when available.
- Degraded or idle workload cause questions -> the most specific workload
  diagnostic workflow available for that symptom. If none exists, state that
  this skill can report the degraded/idle health signals but cannot prove cause.
- Pending for capacity, placement, scheduling, quota, node selector, taint, or
  admission reasons -> `runai-diagnose-pending-workload` or the Pending/
  scheduling diagnostic workflow when available; otherwise state that this
  summary can report `PENDING` but not diagnose the scheduling cause.
- Cluster-wide scheduler, node-fleet, control-plane, or platform-health
  questions -> `runai-get-cluster-health-summary` when available; otherwise
  name the missing cluster-health workflow and stop.
- Fleet-wide utilization waste or team efficiency questions -> the utilization
  overview workflow, preferably `runai-get-utilization-overview`, when
  available; otherwise name the missing utilization workflow and stop.
- Historical run-to-run comparison -> a comparison workflow when available;
  otherwise name the missing comparison workflow and stop.

`HEALTHY` is a complete outcome. Do not hand off when the snapshot already
answers the user's question.

## Example

```text
Workload: training-example | Phase: Running (3h 12m) | Project: project-a | Kind: Distributed

State: Submitted 11:14 | Started 11:16 | Last event: ScalingReplicaSet 2m ago
Spec: 4 x h100, 256 GiB, 16 CPU | node-pool-a | priority=medium | preemptible=true
Pods: 4 Running | nodes node-gpu-17/node-gpu-18 | readiness/restarts unavailable from MCP
Metrics (1h, computed from returned samples): gpu_util avg 87% p95 94% | gpu_mem peak 78 GiB
Policy/Notes: idle timeout 4h; current idle 0m; no preemption events

Status: HEALTHY.
```
