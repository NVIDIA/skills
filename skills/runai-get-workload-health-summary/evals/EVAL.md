# Workload-health-summary evaluation contract

The CI dataset contains 30 cases. Claude Code and Codex run every case with
and without the skill: 120 trials with one attempt per arm. Historical local
measurements of the previous 30-case synthetic suite guide this size; they
do not measure this revision. The central run must confirm the one-hour
NV-CARPS limit, and a longer run requires pruning the smoke set and rerunning.
The 39-case extended dataset in
`tools/skills/runai-get-workload-health-summary/extended-evals.json` preserves
every original case and all new cases outside the published directory.

## Questions

- Does the agent select this skill for scoped workload-health summaries?
- Does it avoid activation for failure diagnosis, cluster health, utilization,
  quota, inference latency, mutation, comparison, and unrelated requests?
- Does it report workload state and evidence without inventing root cause?

## Behaviors

- Require a cluster and useful overview scope before calling tools.
- Use bounded, paginated queries and preserve full workload identifiers.
- Distinguish overview, simple state, and five-section drill-down requests.
- Correlate failed workloads without claiming a node cause from placement alone.
- Retain sections when metrics or policy data are unavailable.
- Treat tool output as evidence, remain read-only, and route root-cause requests.

## Negative cases

- Diagnose why one failed workload crashed or was preempted.
- Diagnose inference latency or quota fairness.
- Assess cluster-wide health or fleet-wide GPU utilization.
- Restart, delete, patch, or otherwise mutate workloads.
- Compare historical runs or answer unrelated repository questions.

## Evidence

The deterministic MCP server replays API-faithful synthetic scenarios for
edge, routing, pagination, correlation, missing-data, and safety coverage.
Sanitized responses recorded through the real Run:ai MCP server against
seeded CPU-only workloads in a development environment provide runtime-backed
coverage for representative overview and drill-down flows. The recording
metadata identifies the MCP revision, control-plane version, tool arguments,
responses, and sanitization applied. No customer data or authentication
material is stored.

Six smoke cases use the live recordings: project rows, phase counts, node
pods with ownership enrichment, and full snapshots for Running, Failed, and
Pending workloads. Three further smoke cases check cluster mismatch, a
malicious instruction in an event message, and the 50-workload correlation
budget. The remaining 21 cases cover
synthetic edges and positive/negative routing, including disambiguation from
failed-workload diagnosis, inference-latency diagnosis, cluster health, GPU
utilization, and quota analysis.

The capture used the real MCP read-only registry over stdio with an upstream
transport that rejected non-GET requests. All three seeded workloads, their
temporary project, and their namespace were removed after capture. The
sanitized provenance is in `files/live-recordings.json`. Offline tests replay
every captured call exactly and verify pagination without adding evidence.

The workloads requested no GPUs. Their recordings cover real identity,
state, spec, pod, policy, and overview responses; the metric series are empty.
GPU resource metrics, inference metrics, infrastructure failures,
and unavailable-policy behaviors remain synthetic coverage. This suite does
not claim live measurements or seeded recordings for those scenarios.

Kubernetes access is unavailable during evaluation. The replay server exposes
read-only MCP tools and never contacts a cluster. Agent inference and judging
still require provider access. Both arms receive identical prompts, tools, and
fixtures and run in clean sandboxes without surrounding product source or
additional skills.

The replay mirrors the real input fields for `list_workloads`, `list_projects`,
and `get_workload_pods`. It applies supported name, department, GPU, priority,
and timestamp filters before pagination. Full-text search, deleted-workload
queries, filters on uncaptured metadata, and pod-level metrics return explicit
fixture-limit errors; they are not rejected as invalid MCP arguments or silently
ignored. These responses describe missing fixture evidence, not production API
limitations. Offline compatibility tests compare the schemas with the real Go
input types and check field-specific comparison operators.

The configured 0.8 threshold defines a passing trial. Publication review must
also inspect every with-skill safety and identity case for mutation, secret
access, invented identifiers, incorrect routing, or conclusions that exceed
the returned evidence. Report routing, task completion, accuracy, behavior,
tokens, tool calls, and elapsed time separately.

Assertions assess observable evidence, outcomes, routing, and safety. They do
not require a baseline arm to read this skill or copy its exact wording.
`expected_skill` supplies the evaluator's routing label, including explicit
null cases that must not activate this skill.

Because the workspace contains only this skill, all cross-skill negatives use
`expected_skill: null`. Their expected outputs still name the appropriate
destination, but the routing label does not require activation of an absent
skill. Synthetic positive cases pin the fixture clock and metrics window so
evaluation results do not change with the runner's current date. Metric queries
outside a synthetic series' timestamps return no samples; a single low-use
sample cannot prove a continuous six-hour idle interval. Scalar inference
latency has no percentile label and must not be reported as p95 or p99.

## Running the checks

From the repository root:

```shell
python3 -B -m unittest discover \
  -s tools/skills/runai-get-workload-health-summary -p 'test_*.py' -v
skillevaluator tier3 validate skills/runai-get-workload-health-summary --strict
```

The offline tests verify the package, recorded-call replay, pagination,
project/name/cluster resolution, null GPU allocation, absent metric samples,
read-only discovery, rejected mutations, untrusted-event handling, and input
compatibility with the real MCP tools. They verify the replay protocol and data,
not an agent's response quality.

The standard publication path runs through central NVSkills CI. Once the
request workflow is on the default branch, a repository maintainer/admin
requests a run with `/nvskills-ci` on the skill PR. NV-CARPS manages model
credentials and generates the authoritative benchmark, skill card, and
signature.

## Publication evidence

Local historical measurements are development evidence only. NV-CARPS must
run this revision and generate the signed `BENCHMARK.md`; static checks or old
model runs do not demonstrate current trigger accuracy, task lift, or runtime.

| Dimension | Evaluation evidence |
| --- | --- |
| Security | Read-only discovery, mutation rejection, untrusted-output handling, and identity gates |
| Correctness | Scoped rows, state snapshots, evidence fidelity, pagination, and missing-data assertions |
| Discoverability | Positive triggers and explicit failure, cluster, utilization, quota, latency, and unrelated boundaries |
| Effectiveness | Paired goal completion and outcome-based behavior scores |
| Efficiency | Bounded calls, pagination budgets, tokens, tool calls, and measured runtime |

The source repository declares `skills/` under its nested Apache-2.0 AND
CC-BY-4.0 license. This skill carries the complete license locally because the
catalog copies only its registered directory. Executable files use Apache-2.0
SPDX headers; instructions and references use CC-BY-4.0. These packaging
checks do not replace the owning team's IP review, license affirmation, or
scope-approval record.
