# DOCA Perftest: capabilities

What the tool measures, which backends and memory types it supports, and what it does not do. Source: [DOCA Perftest, DOCA 3.5](https://docs.nvidia.com/doca/archive/3-5-0/doca-perftest/index.html).

## Metrics

| Metric | Flag | Notes |
|---|---|---|
| Bandwidth | `-m bw` | Reported per device, node and test in multi-node mode. |
| Latency | `-m lat` | Mode matters: one-way uses local send-completion timing. |
| Message rate | (bandwidth/latency runs report it) | For small messages the rate, not the bandwidth, is the limit. |

## Verbs

- `-v write` (RDMA write) and `-v send` are the primary verbs.
- Atomic verbs are documented in the guide's "Atomic Verbs" section; check it for the exact supported set before relying on them.
- Multi-verb operations exist (one run, several verbs). Read the section before using them in a comparison.

## Transports

- `-c RC` (reliable connection) is the default in the guide's examples. Check the guide for the other transports this release supports.

## Drivers

| Driver | Flag | Requires | Behavior if missing |
|---|---|---|---|
| IBV | `-r ibv` (default) | libibverbs (DOCA-OFED or upstream rdma-core) | Always present. |
| DV | `-r dv` | DOCA Verbs package | Tool exits with an error; does not fall back. |

## Memory types

- `host` (default), `nullmr`, `device`.
- CUDA: `cuda_auto_detect` (recommended), `cuda_peermem`, `cuda_dmabuf`, `cuda_data_direct`. Needs CUDA and the matching kernel path.

## Engines

- Default CPU-driven engine.
- `--engine gpunetio`: GPU-initiated data path (CUDA kernels post WQEs, ring doorbells, poll CQ). Write verb only, unidirectional only, needs `-r dv`, CUDA memory and a Volta+ GPU.

## Traffic shapes

- Directions: unidirectional (default), bidirectional, dual unidirectional.
- Patterns (multi-node): one-to-one, one-to-many, many-to-one, all-to-all (including variable per-pair message sizes from a file), bisection, ring.
- Per-iteration sync (lock-step) and peer shifting exist for synchronized and fabric-level tests; read their limitations in the guide before using them.

## Orchestration

- Auto-launch of the remote server over passwordless SSH (CLI only).
- Multi-node runs use OpenMPI bundled with DOCA. A JSON scenario defines many tests in one file.
- Structured JSON output to stdout and a non-interactive mode exist for CI; performance thresholds can fail a run.
- Results file: explicit via CLI option, or the default path in the guide.

## Not supported or not in scope

- Not a replacement for DOCA Bench, which measures DOCA library APIs rather than raw RDMA verbs.
- Does not install drivers, firmware or DOCA. Use `doca-setup`.
- Does not measure application end-to-end throughput.
- GPU-initiated RDMA from the user's own code is covered by `doca-gpunetio`.

## Known limits to state with any number

- Results depend on core and NUMA placement, MTU, message size and queue-pair count. Report all of them.
- Client and server must run the same version.
