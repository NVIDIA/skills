# Failed-workload node correlation

Stage 3 of the workflow. Follow these rules whenever the overview filter
includes `phase=Failed` or the user asks whether failures share a node.

1. Collect scoped `list_workloads(includeDetails=true)` rows up to a probe
   limit of `51`: the `50`-workload correlation budget plus one row to detect
   overflow. Request at most the remaining probe rows on each page. As soon as
   a 51st failed row is found, stop listing and make zero per-workload calls.
   Report that the set exceeds the budget, mark correlation incomplete,
   suppress `POSSIBLE_NODE_ISSUE` and `LIKELY_NODE_ISSUE`, and ask for narrower
   scope. Do not claim an exact total from a truncated probe.
   Correlate only when the complete scoped set contains at most 50 rows.
2. Within the budget, call `get_workload_status` and
   `get_workload_pods(limit=500)` once per failed candidate. If a pod response
   has `hasMore=true`, mark that workload's placement incomplete and suppress
   node-issue labels for the set. Each displayed failed row
   must include workload ID, name, project, department when returned, type,
   priority, preemptibility, phase-updated time, and node when returned.
   Print every workload ID in full and put the owner fields in the displayed
   row; do not leave them only in prose or tool evidence.
3. Inspect `phaseMessage`, `phaseReason`, conditions, and recent events before
   interpreting co-location. If the co-located failures share an explicit
   workload-level explanation such as the same container exit, user
   configuration, or image/configuration error, report that shared signal and
   do not label it a node issue. Do not turn this into root-cause diagnosis.
4. Treat `phaseUpdatedAt` as retry-exhaustion time when status or events show
   `BackoffLimitExceeded` or repeated retries. Do not use it as the first-fault
   time; without an earlier returned failure timestamp, temporal correlation is
   unavailable.
5. When two or more failures share a node, first report the factual pattern as
   `CORRELATED_FAILURES`. Upgrade it to `POSSIBLE_NODE_ISSUE` only when
   placement and failure timing are complete, no shared workload-level
   explanation is returned, and either actual scheduling evidence establishes
   more than one eligible node or independent node-local health evidence
   supports the pattern. Node-pool cardinality alone is not eligibility; if the
   required evidence is unavailable, keep the factual label.
6. Use `LIKELY_NODE_ISSUE` only when `POSSIBLE_NODE_ISSUE` is already supported,
   the workload signals are node-directed (for example `NodeLost`), and
   independent read-only evidence identifies the implicated node as unhealthy
   or under node-local pressure. `get_cluster_infrastructure_health` or optional
   Kubernetes node evidence may supply this context; absence of such evidence
   is not proof that the node is healthy.
7. Always identify affected projects and departments. A failure list is useful
   only when every row can be routed to an owner. Keep a department column even
   when every row shares the department used as the filter; do not rely on the
   surrounding sentence as a substitute for row-level attribution. When neither
   `list_workloads` nor `get_workload_status` returns a department for a row,
   write `unavailable` in its cell; never infer it from the project name.
8. Inference workloads are SLA-bearing. Mark them as escalation candidates even
   when they are not part of the node-correlation cluster.

## Example

```text
Filter: phase=Failed AND phaseUpdatedAt within the requested 15-minute interval
Results: 5 of 5 shown

Pattern: 4 of 5 failed workloads share node-gpu-17.
Workload evidence: all 4 report NodeLost within 3 minutes; placement and timing are complete.
Independent node evidence: node-gpu-17 is NotReady with KubeletNotReady -> LIKELY_NODE_ISSUE.
Attribution: node-gpu-17 affects 3 projects / 2 departments; one Inference workload is SLA-bearing -> escalate.

Every displayed row includes the full workload UUID, name, kind, project,
department, priority, preemptibility, phase-updated time, and node.

Next: drill into a row by its full workload ID. For root cause, use runai-diagnose-failed-workload.
```
