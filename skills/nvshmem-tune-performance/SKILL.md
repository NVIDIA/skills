---
name: nvshmem-tune-performance
description: Route NVSHMEM tuning to data collection, remote transport, NIC-to-PE mapping, or TMA. Do not use for unrelated CUDA, NCCL, or application tuning.
license: Apache-2.0
metadata:
  version: "1.0.0"
  author: NVIDIA NVSHMEM Team <nvshmem@nvidia.com>
  tags:
    - nvshmem
    - performance
    - tuning
---

# Tune NVSHMEM Performance

## Purpose

Guide the user to the smallest applicable NVSHMEM performance workflow. Offer optional baseline collection and three focused optimization services while preserving each specialist's inputs, execution boundary, and safety rules.

Do not reproduce the specialists' technical guidance in this router. Invoke:

| User goal | Specialist |
| --- | --- |
| Collect a baseline, run perftests, or package system and benchmark evidence | `$nvshmem-collect-performance-data` |
| Select an inter-node remote transport | `$nvshmem-select-remote-transport` |
| Select HCAs or optimize NIC-to-PE mapping | `$nvshmem-configure-nic-pe-mapping` |
| Assess, prepare, review, or debug NVSHMEM TMA use | `$nvshmem-enable-tma` |

## Requirements

- Make the four specialist skills listed above available; this router delegates all technical work to them.
- Require no API keys, cluster access, NVSHMEM installation, or benchmark tools to select a route. The selected specialist establishes any environment-specific prerequisites before acting.

## Inputs

### Required

- Obtain the user's NVSHMEM performance objective or selected service. When the request is underspecified, require an explicit selection among the workflow alternatives in Section 2 before invoking a specialist.

### Optional

- Reuse any supplied performance report, raw logs, exact NVSHMEM version and installation prefix, source or kernel path, fabric and provider, job scale, PE-to-GPU binding, GPU/NIC topology, selected transport, tuning constraints, and optimization goal.
- Reuse relevant results and unresolved conditions from an earlier specialist in the same workflow.
- Ask only for information needed to resolve the routing target. Let the selected specialist inspect the environment or request its own required inputs.

Prefer the current explicit user request, then explicit invocation arguments, then non-conflicting context and prior specialist output. Report conflicting values instead of choosing silently.

## Instructions

### 1. Resolve the Routing Target

Honor an explicitly named specialist or workflow. Otherwise classify the concrete objective:

- Route requests for performance baselines, benchmark curves, latency or bandwidth perftests, performance sanity checks, or packaged system evidence to performance data collection.
- Route requests about `NVSHMEM_REMOTE_TRANSPORT`, IBRC, IBDevX, IBGDA, GPUNetIO, UCX, libfabric, remote-network compatibility, or choosing an inter-node data path to remote transport selection.
- Route requests about HCA selection, NIC ports, GPU-to-NIC affinity, multi-NIC use, `NVSHMEM_HCA_*`, or PE-to-NIC assignment to NIC-to-PE mapping.
- Route requests about TMA, `NVSHMEM_TMA_POLICY`, CTA shared-memory registration, direct shared-memory transfers, or preparing a CUDA kernel for TMA to TMA enablement.

Invoke one specialist for a single concrete request. State which specialist is being invoked and why.

### 2. Resolve an Underspecified Performance Request

When the request states only a general intent to improve or tune NVSHMEM performance, do not select a specialist or run anything. Present these workflow alternatives and require an explicit selection:

1. **Baseline characterization workflow:** collect initial performance and system evidence before selecting an optimization domain.
2. **Targeted optimization workflow:** proceed without preliminary collection and select one of:
   - inter-node remote-transport selection;
   - topology-aware NIC-to-PE mapping;
   - TMA suitability assessment or enablement for eligible peer-GPU/NVLink kernel paths.

Permit performance-data collection as the terminal scope. Briefly distinguish the three optimization domains without recommending one in the absence of supporting evidence.

### 3. Sequence Explicit Multi-Stage Workflows

1. Invoke performance data collection first only when the user requests it, selects the baseline characterization workflow, or explicitly specifies a collection-first sequence. Let that skill obtain its required benchmark selection before any run.
2. Invoke remote transport selection before NIC-to-PE mapping when both are requested. Pass the selected transport, version, provider, topology evidence, and unresolved conditions into the mapping workflow.
3. Invoke TMA separately. Do not treat TMA as a remote-transport optimization or infer that inter-node benchmark evidence establishes TMA eligibility.
4. Invoke no unrequested specialist. Let the active specialist resolve its own prerequisites instead of filling them in.

After data collection completes, invoke an optimization specialist already selected by the user. If no optimization domain was selected, present the three targeted optimization alternatives again and require an explicit selection.

### 4. Preserve the Handoff

- Pass the original request, explicit constraints, exact paths, commands, logs, configuration values, source scope, raw artifacts, resolved facts, and relevant prior specialist output.
- Preserve the user's transport settings and tuning constraints during collection and later comparison.
- Return the specialist's result without replacing its confidence, unknowns, validation status, or requested follow-up.
- Keep benchmark execution, system inspection, source edits, and configuration changes inside the selected specialist's authorization and guardrails.

### 5. Handle an Unavailable Specialist

If a required skill is unavailable, name it, explain which requested service cannot proceed, and ask the user to make it available or choose another service. Do not imitate or reconstruct the missing skill.

## Limitations

- Do not run benchmarks, collectors, launchers, or target applications directly from this router.
- Do not edit CUDA source, generate transport exports, or synthesize NIC mappings directly from this router.
- Do not make data collection an implicit prerequisite merely because an optimization might benefit from later measurement.
- Do not promise a performance improvement or declare a regression without the evidence required by the selected specialist.
- Do not treat TMA as useful for remote-network traffic; route it only as a separate peer GPU/NVLink optimization workflow.
- Do not route unrelated NCCL, CUDA-only, operating-system, or general application-performance questions to these NVSHMEM skills.

## Examples

Route a request to collect evidence before choosing an optimization:

```text
User: Collect a baseline for our two-node NVSHMEM all-to-all before tuning it.
Action: Invoke $nvshmem-collect-performance-data and preserve the two-node and all-to-all constraints.
```

Route an explicit network-configuration request directly:

```text
User: Which NVSHMEM remote transport should we use with this InfiniBand cluster?
Action: Invoke $nvshmem-select-remote-transport because the request concerns an inter-node data path.
```

## Troubleshooting

| Error or condition | Cause | Solution |
| --- | --- | --- |
| Routing target is underspecified | The request states only a general intent to improve performance. | Present the baseline-characterization and targeted-optimization workflows; require an explicit selection. |
| A requested specialist is unavailable | Its skill is not installed or enabled. | Name the unavailable specialist and ask the user to enable it or choose another route. |
| Supplied settings conflict | The request and earlier context specify different topology, transport, or constraints. | Report the conflict and ask which value to preserve before handing off. |

<!-- ci trigger -->
