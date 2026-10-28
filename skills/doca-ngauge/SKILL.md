---
license: Apache-2.0
name: doca-ngauge
description: >
  Use this skill when the operator is driving the DOCA ngauge tool to
  analyze, visualize, and debug network performance on a single node by
  probing NIC hardware counters and storing collected data in HDF5 format.
  Trigger even when the user does not say "DOCA ngauge" — implicit
  phrasings include "probe NIC hardware counters", "collect HDF5
  performance data", "plot ngauge output", "sample counters on
  BlueField/ConnectX", "measure network performance on a single node",
  "debug network performance with hardware counters", "run ngauge YAML
  config", "parse ngauge HDF5 output". Refuse and route elsewhere:
  DOCA install/bring-up to doca-setup, version questions to
  doca-version, hardware/firmware state changes to doca-hardware-safety,
  raw packet I/O to doca-eth, and host-DPU messaging to doca-comch.
metadata:
  author: NVIDIA <doca-feedback@nvidia.com>
  kind: tool
  doca_version: "3.5"
  tags:
    - doca
    - ngauge
    - counters
    - hdf5
    - performance
    - telemetry
compatibility: >
  Requires a host with ngauge installed (sudo apt-get install ngauge or
  sudo dnf install ngauge on x86 or Arm 64 hosts; pre-installed on the
  DPU) and a BlueField-3 or ConnectX-7 or above device with firmware
  version xx.43.1000 or higher. The fwctl driver must be installed on the
  host; on Linux kernel 6.15 and above it is available by default unless
  explicitly disabled during kernel configuration. ngauge cannot run
  simultaneously with other tools or services that also use fwctl.
---

# DOCA ngauge

## Scope boundaries

These requests are out of scope: say so, name the skill that owns
the request, and stop.

- raw Ethernet packet send/receive at line rate: `doca-eth/SKILL.md`.
- host-DPU control-plane messaging (Comch): `doca-comch/SKILL.md`.
- DOCA install and bring-up: `doca-setup/SKILL.md`.
- which DOCA version is installed, version compatibility: `doca-version/SKILL.md`.
- NIC or BlueField mode and firmware changes: `doca-hardware-safety/SKILL.md`.
- general DOCA orientation and documentation lookup: `doca-public-knowledge-map/SKILL.md`.

## Examples

- **"Probe NIC hardware counters on a BlueField-3 and save HDF5
  data."** — this is in scope. Configure the YAML file and run
  ngauge; do not attempt to change hardware or firmware state here.
- **"Plot the collected counter data from an ngauge output file."** —
  use the sample plugins under
  `/usr/share/doc/ngauge/examples/plugins`; dependencies may need
  separate installation.
- **"ngauge cannot start because fwctl is in use."** — check for
  other tools or services that also use fwctl; route install and
  driver conflict resolution to `doca-setup`.

## Quick facts

The authoritative source for the binary name, flags, paths, and
configuration schema is the public DOCA ngauge guide at
https://docs.nvidia.com/doca/archive/3-5-0/doca-ngauge/index.html.
Confirm details with the installed tool's `--help` and the sample
configuration files before quoting them.

- ngauge analyzes, visualizes, and debugs network performance on a
  single node by probing NIC hardware counters and storing collected
  data in HDF5 format with relevant metadata.
- Supported hardware: NVIDIA BlueField-3, NVIDIA ConnectX-7, and
  above; firmware version xx.43.1000 or higher.
- ngauge relies on the fwctl driver and cannot run simultaneously
  with other tools or services that also use fwctl.
- All configurations are defined in an input YAML file. Sample
  configuration files are under `/usr/share/doc/ngauge/examples/settings`.
- The device is specified by its PCIe address, for example
  `device: "0000:03:00.0"`.
- Output path and file prefix are mandatory. The output file is saved
  as `/path/to/output/directory/<prefix>_<DATE>_<TIME>.h5`; the exact
  file name is printed after each run.
- The `silent` option suppresses CLI progress indications when set to
  `true` (default `false`).
- Runtime parameters include `mode` (`repetitive` or `single`) and
  `period_us` (sampling period in microseconds; decimal or scientific
  notation accepted, e.g. `1e2` = 100 μs).
- Counter definitions require `id` (data ID); optional fields are
  `desc`, `unit`, `accumulating`, `normalizer` (`time`, a non-zero
  number, or a configured data ID with an `id/` prefix), `cutoff_min`,
  and `cutoff_max`.
- All supported performance counters are listed in the public guide's
  "Supported Data IDs" section.
- Run ngauge with:

  ```bash
  ngauge <configuration YAML file>
  ```

- To end a run early, use Ctrl+C (SIGINT). This is a normal and
  supported way to stop; all results collected up to that point are
  saved as usual.
- During a run, progress bars for each counter are displayed. Color
  coding: blue for low values, red for high values, gradient for
  intermediate values, solid gray for no changes. Data is updated once
  per second, showing mean, standard deviation (σ), minimum, and
  maximum observed during the last second.
- To reference the most recent results when the output directory is
  `/tmp` (the default), use:

  ```bash
  $(ls -1 /tmp/ngauge_data_*.h5 | tail -n1)
  ```

## Instructions

Follow Loading order below. Open the tasks companion for
configure / run / parse / debug. Open the capabilities companion for
what ngauge measures and reports. Do not open every companion file.

**Where to start:** This is a tool skill for invoking DOCA ngauge —
the single-node network performance analysis, visualization, and
debugging tool that probes NIC hardware counters and stores HDF5 data.
Open the tasks companion and start at `## configure` for the YAML
configuration decision, then `## run` for the execution flow. Open the
capabilities companion when the question is *what counters can be
measured, what metadata is stored, and what the progress display
implies*. If the user has not installed ngauge or fwctl, route to
`doca-setup/SKILL.md` first. If the user is asking about
hardware/firmware state changes, route to
`doca-hardware-safety/SKILL.md`. If the user is asking about version
compatibility, route to `doca-version/SKILL.md`.

**How to use this skill.** Read this file, then search the companion
file for the `## <name>` heading the question needs and read that whole
section. Answer completely from it, quoting only its symbols and
commands. Do not browse docs; use the entry points in
`doca-public-knowledge-map`. Give environment checks as commands for
the user's DOCA host. For out-of-scope requests, name the skill that
owns them and stop.

## Example questions this skill answers well

The CLASSES of ngauge questions this skill is built to answer, each
with one worked example. The class is the load-bearing piece; the
worked example is one instance.

- **"How do I configure ngauge to sample specific NIC hardware
  counters?"** — worked example: *"I need to measure RX bytes port 0
  and TX bytes port 0 on a BlueField-3 at a 100 μs period, save to
  /tmp with prefix ngauge_data_."* Answered by the YAML configuration
  model in the capabilities companion + the tasks companion step 2.
- **"How do I run ngauge and stop it early without losing data?"** —
  worked example: *"I started ngauge with a YAML config, want to stop
  after 10 seconds and still get HDF5 output."* Answered by the
  execution flow and Ctrl+C behavior in the tasks companion.
- **"How do I plot the collected HDF5 data?"** — worked example: *"I
  have /tmp/ngauge_data_2026-09-01_12-00-00.h5 and want to plot counter
  0x1020000100000000."* Answered by the parse/plot workflow in the
  tasks companion using `simple_plot.py` or `simple_text_plot.py`.
- **"What do the progress bar colors mean?"** — worked example:
  *"During a run, some bars are red, some blue, some gray; what does
  that indicate?"* Answered by the progress display semantics in the
  capabilities companion.
- **"Why can't ngauge run alongside another tool?"** — worked example:
  *"ngauge fails to start because fwctl is in use by another service."*
  Answered by the fwctl dependency and conflict section in the
  capabilities companion + route to `doca-setup`.
- **"What hardware and firmware are required?"** — worked example:
  *"Will ngauge work on ConnectX-6 or BlueField-2?"* Answered by the
  prerequisites in the capabilities companion.

## When to load this skill

Load this skill when the user is — or the agent needs to — drive
DOCA ngauge on a real host or BlueField Arm with ngauge installed.
Concretely:

- Probing NIC hardware counters on BlueField-3 or ConnectX-7 and
  above.
- Configuring the ngauge YAML file (device, output, params, counters).
- Running ngauge and interpreting the CLI progress bars and summary.
- Parsing or plotting the HDF5 output with the sample plugins.
- Diagnosing fwctl driver conflicts or missing prerequisites.
- Cross-checking counter data against expected network performance.

Do **not** load this skill for general DOCA orientation, raw packet
I/O, the comch control plane, DOCA install, version questions, or
hardware/firmware state changes. For those, route to
`doca-public-knowledge-map/SKILL.md`,
`doca-eth/SKILL.md`,
`doca-comch/SKILL.md`,
`doca-setup/SKILL.md`,
`doca-version/SKILL.md`, or
`doca-hardware-safety/SKILL.md`.

## What this skill provides

This is a **thin loader**. Substantive material lives in two companion
files:

- the capabilities companion — what ngauge measures and reports:
  supported hardware and firmware, the fwctl dependency and conflict,
  the YAML configuration model (device, output, params, counters),
  counter field semantics (`id`, `desc`, `unit`, `accumulating`,
  `normalizer`, `cutoff_min`, `cutoff_max`), HDF5 output format,
  progress display semantics, sample plugins, and the safety posture
  that ngauge itself is read-only probing while any step that changes
  hardware or firmware state routes to `doca-hardware-safety`.
- the tasks companion — step-by-step workflows for the in-scope task
  verbs: `configure` (YAML setup), `run` (execution and Ctrl+C),
  `parse` / `plot` (simple_plot.py and simple_text_plot.py), `debug`
  (fwctl conflict, missing dependencies), plus a `Command appendix`
  that honors the bundle's
  `doca-structured-tools-contract/SKILL.md`
  preamble.

The skill assumes a host or BlueField where ngauge and fwctl are
already installed and the operator has whatever privileges the public
DOCA ngauge guide requires for the chosen device.

## What this skill deliberately does not ship

This skill is **agent guidance**, not a samples or scripts bundle. To
keep the boundary clean, it deliberately does not contain — and pull
requests should not add:

- **Verbatim counter ID inventories, flag lists, or output column
  names.** The public DOCA ngauge guide on `docs.nvidia.com` and the
  installed `--help` on the user's version are the joint source of
  truth; copying them here pins the skill to one release and silently
  rots when ngauge evolves. The skill routes the agent at those sources
  instead.
- **Pre-baked example output.** Output is install-, version-, and
  device-specific. A captured example will mislead an operator on a
  different platform or state.
- **Wrappers, parsers, or scripts** in any language that consume
  ngauge output. The output format is documented; users who want to
  script against it should read the live guide and write the parser
  against their installed version.
- **A `samples/` or `reference/` subtree.** This is a thin loader for
  a documented DOCA artifact; substantive material lives on the public
  page and in `--help`.

## Loading order

1. Read this skill first to confirm the user's question is in scope
   (the user wants to probe NIC hardware counters, collect HDF5 data,
   or plot ngauge output, not learn the eth / comch / RDMA APIs).
2. **For what ngauge measures, the YAML configuration model, the
   read-only vs state-changing split, version availability, the fwctl
   conflict surface, observability, and safety posture, see the
   capabilities companion.**
3. **For the documented invocations and the configure → run →
   parse/plot workflow — `configure`, `run`, `parse`, `debug`, plus
   the `Command appendix` — see the tasks companion.**

## Related skills

- `doca-setup/SKILL.md` — env preparation,
  install verification, fwctl driver installation and conflict
  resolution, and the *I have no install yet* path. This skill assumes
  its preconditions are satisfied.
- `doca-version/SKILL.md` — canonical DOCA
  version-handling rules. The `## Version compatibility` section in
  the capabilities companion is a concise overlay that redirects here
  for the body.
- `doca-hardware-safety/SKILL.md` —
  any step that changes hardware or firmware state routes here. ngauge
  itself is read-only probing, but if a workflow includes firmware
  updates or device state changes, use this skill.
- `doca-public-knowledge-map/SKILL.md`
  — routing to the public DOCA ngauge guide and the rest of the public
  DOCA documentation set. The canonical URL is reached via
  `doca-public-knowledge-map ## DOCA tools`
  (`doca-public-knowledge-map/SKILL.md#doca-tools`).
- `doca-structured-tools-contract/SKILL.md`
  — the bundle's detect → prefer → fall back → report contract for
  structured helper tools. The Command appendix in the tasks companion
  honors this contract.
- `doca-debug/SKILL.md` — the cross-cutting
  debug ladder. ngauge slots in at the *runtime* layer as a
  single-node telemetry tool whose fwctl conflict surface escalates to
  the cross-cutting ladder when the cause is below DOCA.
- `doca-eth/SKILL.md` — the line-rate raw
  packet I/O surface. ngauge is for hardware counter telemetry, not
  packet I/O; use doca-eth when the application needs to operate on
  packets directly.
- `doca-comch/SKILL.md` — the host ↔ DPU
  control-plane primitive. ngauge is single-node performance telemetry,
  not a control-plane channel; pair them only when coordinating a
  measurement across host and DPU.
