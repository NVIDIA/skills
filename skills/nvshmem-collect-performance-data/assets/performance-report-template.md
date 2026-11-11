# NVSHMEM Performance Data Collection

## Collection Summary

| Field | Value |
| --- | --- |
| Collection ID | `{{COLLECTION_ID}}` |
| UTC timestamp | `{{UTC_TIMESTAMP}}` |
| Status | `{{COMPLETE_PARTIAL_OR_INVALID}}` |
| Selected suite/tests | {{SELECTED_TESTS}} |
| Selection rationale | {{SELECTION_RATIONALE}} |
| Placements collected | {{PLACEMENTS_COLLECTED}} |
| Missing placements/tests | {{MISSING_ITEMS_OR_NONE}} |
| Range assessment | {{RANGE_ASSESSMENT}} |

## System and Software

| Item | Node 1 | Node 2 or N/A |
| --- | --- | --- |
| Hostname | {{NODE1_HOSTNAME}} | {{NODE2_HOSTNAME}} |
| OS and kernel | {{NODE1_OS_KERNEL}} | {{NODE2_OS_KERNEL}} |
| CPU | {{NODE1_CPU}} | {{NODE2_CPU}} |
| GPU model/count | {{NODE1_GPUS}} | {{NODE2_GPUS}} |
| NVIDIA driver | {{NODE1_DRIVER}} | {{NODE2_DRIVER}} |
| CUDA | {{NODE1_CUDA}} | {{NODE2_CUDA}} |
| NVSHMEM | {{NODE1_NVSHMEM}} | {{NODE2_NVSHMEM}} |
| NVSHMEM prefix/source | {{NODE1_NVSHMEM_PREFIX}} | {{NODE2_NVSHMEM_PREFIX}} |
| Launcher/bootstrap | {{NODE1_LAUNCHER}} | {{NODE2_LAUNCHER}} |
| Active NIC/RDMA ports | {{NODE1_NICS}} | {{NODE2_NICS}} |

Environment evidence files: `environment/{{NODE1_ENVIRONMENT_FILE}}`; `{{NODE2_ENVIRONMENT_PATH_OR_NA}}`

## Topology and Placement

### GPU/NIC Topology

```text
{{NVIDIA_SMI_TOPOLOGY}}
```

### PE Placement

| Placement | Node | PE | GPU index/UUID/PCI BDF | GPU link class | NIC/transport evidence |
| --- | --- | ---: | --- | --- | --- |
| {{PLACEMENT}} | {{HOSTNAME}} | {{PE}} | {{GPU_IDENTITY}} | {{LINK_CLASS}} | {{NIC_TRANSPORT}} |

### Launcher Preflight Evidence

| Placement | Expected | Observed rank/host/GPU binding | Result | Command | Stdout | Stderr | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| {{PLACEMENT}} | {{EXPECTED_PLACEMENT}} | {{OBSERVED_PREFLIGHT}} | {{PREFLIGHT_RESULT}} | `{{PREFLIGHT_COMMAND}}` | `{{PREFLIGHT_STDOUT}}` | `{{PREFLIGHT_STDERR}}` | `{{PREFLIGHT_STATUS}}` |

### NVSHMEM Initialization Evidence

| Placement | Result | Successfully initialized transports | PE/GPU/NIC selection | Command | Stdout | Stderr | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| {{PLACEMENT}} | {{PROBE_RESULT}} | {{INITIALIZED_TRANSPORTS}} | {{PROBE_DEVICE_SELECTION}} | `{{PROBE_COMMAND}}` | `{{PROBE_STDOUT}}` | `{{PROBE_STDERR}}` | `{{PROBE_STATUS}}` |

## Result Summary

| Placement | Test | Metric summary | Validation | Raw stdout | Raw stderr |
| --- | --- | --- | --- | --- | --- |
| {{PLACEMENT}} | `{{TEST}}` | {{METRIC_SUMMARY}} | {{VALIDATION}} | `{{RAW_STDOUT}}` | `{{RAW_STDERR}}` |

For bandwidth tests, report peak bandwidth and bandwidth at the largest measured message size. For latency tests, report latency at the smallest measured message size. Preserve each test's native units.

## Complete Results

### {{PLACEMENT}} — `{{TEST}}`

| Message size (bytes) or operation | Scope/configuration | Mean | Standard deviation | Minimum | Maximum | Repetitions | Unit |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| {{SIZE_OR_OPERATION}} | {{SCOPE_OR_CONFIG}} | {{MEAN}} | {{STDDEV_OR_NA}} | {{MIN_OR_NA}} | {{MAX_OR_NA}} | {{REPETITIONS}} | {{UNIT}} |

Repeat this section for every selected test and placement. Retain all valid rows rather than only the headline value.

## Commands and Configuration

### Relevant Environment

| Setting | Inherited | Requested override | Effective |
| --- | --- | --- | --- |
| `NVSHMEM_REMOTE_TRANSPORT` | {{INHERITED_REMOTE_TRANSPORT}} | {{REQUESTED_REMOTE_TRANSPORT_OR_NONE}} | {{EFFECTIVE_REMOTE_TRANSPORT}} |

```text
{{RELEVANT_ENVIRONMENT}}
```

### Exact Commands

```bash
{{EXACT_COMMANDS}}
```

Command record: `commands.md`

## Validation Findings

- {{VALIDATION_FINDING}}

## Anomalies and Missing Evidence

- {{ANOMALY_OR_NONE}}

## Assumptions and Comparability Limits

- {{ASSUMPTION_OR_LIMIT}}

## Artifact Index

- Environment evidence: `environment/`
- Launcher preflight and initialization evidence: placement directories
- Same-node results: `same-node/`
- Two-node results: `two-node/`
- Command record: `commands.md`

Remove paths for directories that were not collected. Keep paths relative so this report and its raw evidence can be submitted together.
