---
name: nvshmem-collect-performance-data
description: Collect and package NVSHMEM put/get bandwidth, latency, and other perftest results with system and topology evidence for performance sanity checks.
license: Apache-2.0
metadata:
  version: "1.0.0"
  author: NVIDIA NVSHMEM Team <nvshmem@nvidia.com>
  tags:
    - nvshmem
    - performance
    - perftest
---

# NVSHMEM Performance Data Collection

## Purpose

Collect reproducible NVSHMEM performance evidence, with the four device put/get bandwidth and latency tests as the recommended sanity suite. Preserve the system, topology, placement, initialization, transport, commands, raw output, and tuning configuration in a portable Markdown report without inventing system-specific acceptance thresholds.

## Requirements

- Use an allocated compute node with the target GPUs visible; never collect evidence or run benchmarks on a login node.
- Make the target NVSHMEM installation, selected perftest executable, and a known-good launcher/bootstrap path available. Direct target access is optional when a user can run the staged commands and return their complete output.
- Provide a writable output parent directory for the artifact package. The bundled collector itself is read-only and writes only to stdout.
- Require no API keys, privileged access, package installation, or system-configuration changes.

## Inputs

### Required

Resolve these values from the request, the target environment, or focused follow-up before benchmark execution:

- an explicit test, preset, or suite selection;
- the NVSHMEM installation prefix and perftest root;
- access to the participating compute nodes, or a user who can run and return the staged commands;
- a known-good launcher and bootstrap path;
- allocated nodes, total PEs, PEs per node, and local PE-to-GPU placement;
- the output parent directory for a new artifact package.

For an unspecified request, require only the suite choice first. Do not collect the remaining execution inputs until the user selects a suite. Do not ask again for values that are already explicit or safely observable.

### Optional

- exact message-size ranges, datatype, atomic operation, thread-group scope, CUDA graph, mmap, EGM, bidirectional, or other selected-test options;
- an exact `NVSHMEM_REMOTE_TRANSPORT` execution override;
- an installation or source tree containing the selected test's standard `.args` file;
- a caller-provided launcher command, binding policy, artifact-directory name, or reporting notes;
- an identical matched baseline for range assessment;
- raw stdout, stderr, exit statuses, hostname output, and environment evidence from user-run collection when direct execution is unavailable.

## Scripts

| Script | Use | Invocation |
| --- | --- | --- |
| `scripts/collect-performance-environment.sh` | Collect read-only NVSHMEM, GPU, network, RDMA, launcher, topology, and allowlisted environment evidence. It writes to stdout and can exit with status 2 after producing useful partial output. | Run on every participating compute node as `collect-performance-environment.sh [--prefix PATH]`; retain complete output even when it exits nonzero. |

## Instructions

### Resolve the Request

Classify the request before running a benchmark.

- Treat an exact executable, named family, or named preset suite as an explicit selection.
- Treat requests such as "run perftests," "check NVSHMEM performance," or "perform a sanity check" as unspecified.
- For an unspecified request, read [references/perftest-catalog.md](references/perftest-catalog.md), summarize the available families, recommend the default device sanity suite, and require the user to choose one of:
  1. the recommended four-test device sanity suite;
  2. a custom device suite;
  3. a host or feature-specific suite.
- Run nothing until the user explicitly selects a choice. After a custom category is chosen, ask only for the exact tests, placement, PE count, and special mode that materially affect the run.
- If the user already selected a test or suite, skip the menu and honor that scope.
- Use canonical executable names. Do not translate or accept shortened `_lat` names.

Assume the standard NVSHMEM perftest set exists. Do not inventory executables or enumerate every argument variant before selection. When a selected executable is missing at launch, report an installation or build problem instead of silently substituting another test.

### Default Device Sanity Suite

Use exactly these device point-to-point tests when the user selects the recommended suite:

```text
device/pt-to-pt/shmem_put_bw
device/pt-to-pt/shmem_put_latency
device/pt-to-pt/shmem_get_bw
device/pt-to-pt/shmem_get_latency
```

Run each test with exactly two PEs. Measure one same-node placement and, when a two-node allocation is already available, one two-node placement with one PE per node. Do not request or create a new allocation merely to add the two-node placement; record it as unavailable.

For a selected non-default test, read its entry in [references/perftest-catalog.md](references/perftest-catalog.md). Use two PEs for point-to-point tests unless the benchmark documents another requirement. Require an explicit node count, PE count, PEs per node, and GPU placement for collective, tile, or allocation tests.

### Prepare the Target

Resolve these values from the user, environment, or direct inspection:

- NVSHMEM installation prefix and perftest root;
- launcher and bootstrap path;
- allocated nodes and task/GPU capacity;
- local PE-to-GPU binding;
- selected same-node and two-node placements;
- output parent directory.

Resolve `PERF_ROOT` from an explicit path first, then `$NVSHMEM_PREFIX/bin/perftest`, then `$NVSHMEM_HOME/bin/perftest`. Do not search the filesystem for alternate installations. Check only the selected executable paths with `test -x`.

If the user supplies a remote-transport override, use that exact value for every probe and measured command. Otherwise do not ask for, select, or inject one; preserve the inherited setting or NVSHMEM default. Route transport-recommendation requests to `nvshmem-select-remote-transport` rather than invoking it automatically.

Prefer a caller-provided, known-good launcher. Otherwise use this order:

1. `srun --overlap` inside an active Slurm allocation;
2. `nvshmrun` when available and already configured;
3. `mpirun` only after a two-rank launcher and hostname preflight succeeds.

Never run a benchmark on a login node. For a same-node run, prove two ranks start on one hostname and use distinct GPUs. For a two-node run, prove one rank starts on each of two distinct hostnames. Do not call a run inter-node without distinct hostname evidence.

Before each placement probe, retain `00-preflight.command`, `00-preflight.stdout`, `00-preflight.stderr`, and `00-preflight.status` in its placement directory. Require every rank to report rank/local rank, hostname, and applicable GPU visibility or binding.

Create a new artifact directory without overwriting prior data:

```text
nvshmem-performance-<UTC timestamp>/
  environment/
  same-node/
  two-node/
  commands.md
  NVSHMEM_PERFORMANCE_REPORT.md
```

Use additional placement directories for custom suites when needed. Store stdout and stderr separately for every command and record its exit status.

### Collect System Evidence

Collect evidence before benchmarks on every participating compute node. Tell the user that the bundled collector is read-only, then run:

```text
run_script("scripts/collect-performance-environment.sh", "--prefix", "<resolved-prefix>")
```

Use the bundled [collector script](scripts/collect-performance-environment.sh) and pass the already resolved installation with `--prefix`. Prefer `run_script`; when it is unavailable, execute that same bundled file with `bash`. Redirect each node's complete output to `environment/<hostname>.txt`. Preserve partial output even when the collector exits with status 2.

If direct target access is unavailable, give the user the collector plus the selected launcher preflight and benchmark commands. Ask them to return the complete stdout, stderr, exit statuses, and hostname output. Do not treat login-node evidence as compute-node evidence.

### Run the Initialization Probe

Before measured tests, run a minimal device-side probe for every placement using the selected launcher and resolved executable:

```bash
NVSHMEM_INFO=1 NVSHMEM_DEBUG=INFO "$PERF_ROOT/device/pt-to-pt/shmem_put_latency" \
  -b 4 -e 4 -n 1 -w 1 -t 1 -s thread
```

Place the launcher before the executable in the actual command. Retain the command, stdout, stderr, and status as `01-init.*`, plus successfully initialized transport paths and PE/GPU/NIC selection. Treat runtime debug success messages—not the `NVSHMEM_INFO` default—as transport proof; mark absent proof incomplete. Mark the placement invalid or incomplete and stop on initialization failure, wrong placement, or mismatch with a supplied remote-transport override. Other initialized paths such as P2P or IBGDA may coexist and are not a mismatch.

Keep `NVSHMEM_INFO=1` and `NVSHMEM_DEBUG=INFO` probe-only. Do not select an HCA, disable P2P, or change a tuning variable merely to make the probe pass.

### Run Selected Tests

Use the selected test's standard adjacent `.args` configuration when available. Look beneath the installation's `share/src/perftest` tree or an explicitly supplied source tree for the matching relative `.args` file.

- For the default suite, use the first nonempty normal line for each test. For `shmem_put_bw`, deterministically use the first nonempty line without `--bidir` and record its complete scope, CTA-count, and threads-per-CTA configuration.
- This default-suite rule takes precedence over variant prompting. For custom or non-default suites, explain only relevant semantic variants and require a selection instead of running all variants.
- If no `.args` file is available, inspect the selected executable's `--help` and use its built-in defaults unless the user requested specific limits.
- Use `--repetitions 3` when supported. If native repetitions are unavailable, run the identical process three times and retain each raw result.
- Set `NVSHMEM_MACHINE_READABLE_OUTPUT=1`; when a remote-transport override was supplied, set it on every measured command too. Fall back to the human-readable table when the installed version does not emit machine-readable rows.
- Run benchmarks serially. Do not overlap tests or silently change message-size ranges, datatypes, atomic operations, scopes, CUDA graph mode, mmap mode, or bidirectionality.

Record the fully resolved command before execution. Preserve all other pre-existing `NVSHMEM_*`, CUDA, provider, launcher, and binding variables. Report inherited, requested, and effective remote-transport values; an explicit request overrides only inherited `NVSHMEM_REMOTE_TRANSPORT`.

### Validate Results

Validate every run before summarizing it:

- Require a zero launcher/test exit status.
- Confirm the intended node, PE, and GPU placement from hostname and per-rank output.
- Confirm the output identifies the selected test or expected metric.
- Require at least one nonempty message-size or operation row.
- Reject missing, nonnumeric, `NaN`, infinite, or nonpositive measured values.
- Preserve the output units; do not silently convert decimal GB/s to GiB/s.
- Confirm three timed repetitions or three separate runs when requested.
- Search stdout and stderr for initialization, CUDA, bootstrap, transport, timeout, and launch failures.

Mark a run `valid`, `invalid`, or `incomplete`. Do not discard partial evidence, average incompatible placements, or call a result expected merely because it is positive.

### Write the Handoff Report

Copy [assets/performance-report-template.md](assets/performance-report-template.md) to the artifact directory as `NVSHMEM_PERFORMANCE_REPORT.md`, replace every placeholder, and remove unused optional sections.

For bandwidth tests, report the peak bandwidth and the value at the largest measured message size, then retain the complete curve. For latency tests, report the smallest-message latency and retain the complete curve. Preserve native metrics for atomics, collectives, allocation, and other tests.

Use relative artifact paths in the report so the entire directory can be submitted as one package. Include:

- collection completeness and missing placements;
- selected tests and rationale;
- system, software, GPU, NIC, and topology evidence;
- launcher-preflight hostname/binding artifacts and runtime initialization proof;
- inherited, requested, effective, and successfully initialized transport paths plus PE/GPU/NIC evidence;
- summary and full result tables;
- exact commands and exit statuses;
- validation findings, anomalies, assumptions, and raw-log links.

If no matched baseline is supplied, write `Range assessment: not performed (no matched baseline supplied)`. Do not estimate expected ranges from nominal link speed. If a matched baseline is supplied, compare only identical benchmark, message size, placement, PE count, transport, and relevant options, and state any remaining comparability limits.

## Output Format

Return the artifact directory and a concise Markdown handoff. Copy the report template to `NVSHMEM_PERFORMANCE_REPORT.md` and include these fields in order:

| Field | Required content |
| --- | --- |
| Collection status | `complete`, `incomplete`, or `invalid`, plus unavailable placements and the reason. |
| Scope and placement | Selected tests, rationale, node count, total PEs, PEs per node, and PE-to-GPU binding. |
| Environment evidence | Relative paths to per-node collector output, NVSHMEM version, GPU/NIC topology, launcher, remote-transport configuration, and runtime initialization proof. |
| Results | A summary table, complete native-unit result curves, three-repetition evidence, and paths to stdout, stderr, commands, and exit statuses. |
| Validation and comparison | Per-run validity, anomalies, assumptions, and either a matched-baseline comparison or `Range assessment: not performed (no matched baseline supplied)`. |

Use relative paths so the package remains portable. Do not issue an expected or regressed verdict without a matched baseline.

## Examples

Collect the recommended suite after the user selects it:

```text
User: Run the recommended device sanity suite with two PEs on the allocated node.
Action: Use run_script("scripts/collect-performance-environment.sh", "--prefix", "<resolved-prefix>"), then collect the four device put/get bandwidth and latency tests with the resolved two-PE placement.
Output: Return the artifact-directory path and NVSHMEM_PERFORMANCE_REPORT.md with the same-node results.
```

Stage commands when direct compute-node access is unavailable:

```text
User: I need a two-node put-latency baseline but cannot grant you cluster access.
Action: Request the selected placement and output directory, then provide the collector, launcher preflight, initialization probe, and benchmark commands for the user to run.
Output: Package the returned stdout, stderr, exit statuses, and hostname evidence; mark missing data incomplete instead of estimating it.
```

## Troubleshooting

| Error or condition | Cause | Solution |
| --- | --- | --- |
| The collector exits with status 2. | One or more evidence sources are unavailable, but the script produced partial output. | Retain the complete output, record the missing evidence, and rerun on the allocated target compute node if possible. |
| Initialization probe fails or ranks use the wrong GPUs. | The launcher, bootstrap path, or PE-to-GPU binding is not valid for the placement. | Stop that placement, capture stdout and stderr, correct the known-good launcher or binding, and rerun the probe before measuring. |
| A selected perftest executable or `.args` file is missing. | The resolved installation or source tree is incomplete. | Report the installation or build problem; do not substitute a different test or invent command-line options. |
| A result cannot be compared to a baseline. | Benchmark, placement, PE count, transport, or options differ. | Preserve both datasets and report the comparability limit; do not label the result expected or regressed. |

## Limitations

- Treat system evidence as a snapshot of the participating compute nodes and active allocation. It cannot establish site-wide consistency, platform policy, or an untested network path.
- Assume the standard NVSHMEM perftest set is available; do not inventory it. Treat a selected executable missing at launch as an installation or build problem.
- Stage commands for the user when direct compute-node access is unavailable. Do not treat login-node evidence as compute-node evidence.
- Keep collection within the existing allocation and software environment; do not install dependencies, allocate nodes, or make driver, fabric, or system changes.
- Keep collection read-only and unprivileged. Collect only the allowlisted performance and launcher variables emitted by the bundled script; never expose unrelated environment variables or secrets.
- Never overwrite an existing artifact directory or raw log, replace a failed device test with a host analogue, or represent a same-node result as two-node evidence.
- Compare results only when the benchmark, message size, placement, PE count, transport, and relevant options match. Without a matched baseline, state that range assessment was not performed and do not issue an expected or regressed verdict.

<!-- ci trigger -->
