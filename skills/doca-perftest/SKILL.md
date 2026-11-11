---
license: Apache-2.0
name: doca-perftest
description: >
  Use this skill when the user is running DOCA Perftest (doca_perftest), the
  RDMA benchmark that measures bandwidth, message rate and latency for RDMA
  verbs, point-to-point or across a cluster, with an optional DOCA Verbs
  backend (-r dv) or GPUNetIO engine (--engine gpunetio). Trigger even when the
  user does not say "DOCA Perftest" — implicit phrasings include "how do I
  benchmark RDMA write bandwidth with DOCA", "run ib_write_bw but across 8
  nodes", "perftest says the device is busy", "my latency numbers differ between
  client and server", "launch the server over SSH", or "run an all-to-all RDMA
  test from a JSON scenario". Refuse and route elsewhere for DOCA install
  (doca-setup), DOCA Bench (doca-bench), and application-level throughput.
metadata:
  author: NVIDIA <doca-feedback@nvidia.com>
  kind: tool
  doca_version: "3.5"
  tags:
    - doca
    - rdma
    - benchmark
    - perftest
compatibility: >
  Requires DOCA 3.5 or later on Linux with a ConnectX or BlueField RDMA device,
  libibverbs (DOCA-OFED or upstream rdma-core), and OpenMPI for multi-node
  modes. The DOCA Verbs backend needs the doca-verbs package; the GPUNetIO
  engine needs doca-gpunetio, CUDA and a Volta+ GPU. Run benchmarks only on a
  trusted, non-shared fabric.
---

# DOCA Perftest

## When to load this skill

Load this skill when the user wants to — or the agent needs to — run
`doca_perftest` on a host with DOCA installed. Concretely:

- Measuring RDMA bandwidth, message rate, or latency between two hosts
  (point-to-point) with a verb such as write / read / send.
- Running the same benchmark across many nodes — JSON scenario or
  `--traffic_pattern` (pair-wise, all-to-all), orchestrated over
  OpenMPI / passwordless SSH.
- Comparing the IBV and DOCA Verbs backends (`-r ibv` vs `-r dv`) or
  driving the GPUNetIO engine (`--engine gpunetio`).
- Interpreting a run: client-vs-server number drift, "device busy",
  inconsistent latency across runs.

Do **not** load this skill for DOCA install (`doca-setup`), the doca-bench harness,
Flow pipeline tuning (`doca-flow-tune`), or application-level
throughput.

## Examples

- "Measure RDMA write bandwidth between two ConnectX-7 hosts with 64 KB messages." Point-to-point: run the server, then the client with `-n`.
- "I need latency across 16 nodes, all-to-all." Multi-node: JSON scenario or `--traffic_pattern`, with OpenMPI.
- "Compare the IBV and DOCA Verbs drivers on the same device." Same command, `-r ibv` vs `-r dv`.

## Instructions

Follow the loading order below. Open `TASKS.md` for the commands and `CAPABILITIES.md` for what the tool can measure and what it does not.

**Where to start:** DOCA Perftest is the DOCA 3.5 replacement for legacy `ib_write_bw` / `ib_send_lat` plus the orchestration scripts used to run them on many nodes. Start at `TASKS.md ## point-to-point` for a first run. Use `## multi-node` for scenarios and `## debug` when a run fails or the numbers look wrong.

If DOCA is not installed, route to `doca-setup/SKILL.md`. Client and server must run the same DOCA Perftest version; check with `doca-version/SKILL.md` before comparing numbers across hosts. If the question is about GPU-initiated RDMA in the user's own code, route to `doca-gpunetio/SKILL.md`.

## Example questions this skill answers well

- **"Run a point-to-point RDMA write bandwidth test."** Use `-d <dev> -n <server> -v write -m bw -s <size> -D <sec>`; the client coordinates the server side over SSH (requires SSH key exchange set up beforehand).
- **"Benchmark 16 nodes at once and get one report."** Use a JSON scenario or `--traffic_pattern`; the report aggregates per device, node and test.
- **"Latency differs between runs."** Check the latency mode, the core and NUMA placement, and whether both sides use the same message size and verb.

## Loading order

1. This file for scope and routing.
2. `TASKS.md` for the exact command for the job you are doing.
3. `CAPABILITIES.md` when you need to know whether a verb, driver, memory type or traffic pattern is supported.

Do not open every companion file.

## Scope boundaries

- Installs and driver packages: `doca-setup/SKILL.md`.
- Version matching across client, server and DOCA release: `doca-version/SKILL.md`.
- Any change to NIC or eswitch state, firmware, or host kernel settings (hugepages, IOMMU, PCIe rebind): follow `doca-hardware-safety/SKILL.md` first.
- Failure diagnosis: `doca-debug/SKILL.md` for the layer ladder, then the `TASKS.md ## debug` anchor here.
- Upstream CPU-initiated `ib_write_bw`: this tool replaces it on DOCA 3.5; do not mix the two in one comparison without saying so.
