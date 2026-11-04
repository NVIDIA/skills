---
license: Apache-2.0
name: doca-telemetry-traceback-application
description: >
  Use this skill when the operator is driving the DOCA Telemetry
  Traceback application to monitor telemetry data locally on a
  BlueField DPU or ConnectX NIC, detect long-term trends and
  threshold breaches with user-defined rules, and capture
  pre-event history plus active-event data to a log file. Trigger
  even when the user does not explicitly mention "DOCA Telemetry
  Traceback" — typical implicit phrasings include "monitor
  telemetry trends on the DPU without streaming all raw data",
  "log the samples before and after a threshold breach", "set up
  a rule that triggers when rx bytes/s exceeds 1M", "differential
  greater than rule on a data ID", "circular buffer of telemetry
  samples", or "traceback event log". Refuse and route elsewhere:
  the DOCA Telemetry Diag programming API and raw counter reads to
  doca-telemetry, DOCA install/bring-up to doca-setup, and version
  questions to doca-version.
metadata:
  author: NVIDIA <doca-feedback@nvidia.com>
  kind: tool
  doca_version: "3.5"
  tags:
    - doca
    - telemetry
    - traceback
    - monitoring
compatibility: >
  Requires DOCA SDK installed at /opt/mellanox/doca on Linux with a
  BlueField DPU or ConnectX NIC, plus root privileges to access
  hardware-level telemetry data. The application is distributed as
  source code and must be compiled before execution; the source is
  located at /opt/mellanox/doca/applications/telemetry/traceback/.
  Validation steps require a live DOCA install.
---

# DOCA Telemetry Traceback Application

## Scope boundaries

These requests are out of scope: say so, name the skill that owns
the request, and stop.

- the DOCA Telemetry Diag programming API and raw counter reads: `doca-telemetry/SKILL.md`.
- DOCA install and bring-up: `doca-setup/SKILL.md`.
- which DOCA version is installed, version consistency: `doca-version/SKILL.md`.
- general DOCA orientation and documentation lookup: `doca-public-knowledge-map/SKILL.md`.

## Examples

- **“Can I monitor telemetry trends on the DPU without streaming all raw data?”** — in scope. Configure rules and sample history depth; do not use the DOCA Telemetry Diag API directly here.
- **“The event log is empty even though a threshold was crossed.”** — check rule `comparison_type` and `sample_history_depth`; differential rules require depth samples before functioning.
- **“How do I build only this application?”** — use the Meson isolation flags; see the tasks companion.
- **“I'm writing my own C code against the DOCA Telemetry Diag library to read counters.”** — that part is
  `doca-telemetry`'s: the application uses Diag internally, but programming the Diag API is not this skill.
  Name `doca-telemetry` for it and do not write Diag API code here; handle every other part of the request
  as this skill describes (for example, `--force-ownership` still goes through `doca-hardware-safety`).

## Quick facts

The authoritative source for the binary name, flags, and paths is the public
DOCA Telemetry Traceback guide for the release; confirm them with the installed
tool's `--help` before quoting any of them. As that guide describes it:

- Standalone application on a BlueField DPU or ConnectX NIC; four components:
  DOCA Telemetry Diag instance, sample history circular buffer, rule
  collection, and events log file.
- Configuration JSON fields: `diag_sample_period_ns`, `sample_history_depth`,
  `data-ids` array, and `rules` array. Rule types are
  `incremental_greater_than` (`inc_gt`), `differential_greater_than`
  (`dt_gt`), `differential_less_than` (`dt_lt`), `greater_than` (`gt`),
  and `less_than` (`lt`).
- The circular buffer size is automatically doubled based on the requested
  depth to ensure equal historical and active event data.
- Requires sudo/root privileges; mandatory `-d` PCIe address and `-r` rules
  JSON; optional `--event-log` and `--force-ownership`.
- Unsupported data IDs are reported; adjust inputs accordingly.
- The application stops if the actual sample size or period does not match
  the requested parameters.
- `--force-ownership` changes diag domain ownership; route any use of it
  through `doca-hardware-safety/SKILL.md`.

## Instructions

Follow Loading order below. Open the tasks companion for
configure / build / run / debug. Open the capabilities companion for
what the application monitors and logs. Do not open every companion file.

**Where to start:** This is a tool skill for invoking the DOCA Telemetry
Traceback application — the local telemetry monitoring and traceback logging
service that processes telemetry data on the device and writes event logs
when user-defined rules trigger. Open the tasks companion and start at
`## configure` for the JSON configuration and rule definitions, then
`## run` for compile and execution. Open the capabilities companion when
the question is *what data can be monitored, what rules evaluate, and what
the event log contains*. If the user has not installed DOCA yet, route to
`doca-setup/SKILL.md` first. If the user asks about version, route to
`doca-version/SKILL.md`. If any step changes hardware or firmware state
(e.g., `--force-ownership`), route to
`doca-hardware-safety/SKILL.md`.

**How to use this skill.** Read this file, then search the companion file
for the `## <name>` heading the question needs and read that whole section.
Answer completely from it, quoting only its symbols and commands. Do not
browse docs; use the entry points in `doca-public-knowledge-map`. Give
environment checks as commands for the user's DOCA host. For out-of-scope
requests, name the skill that owns them and stop.

Compile and run commands are documented in the tasks companion; the
authoritative source is the public guide. The application is distributed as
source and must be compiled before execution. The source is located at
`/opt/mellanox/doca/applications/telemetry/traceback/`.

## Example questions this skill answers well

The CLASSES of telemetry-traceback questions this skill is built to answer,
each with one worked example. The class is the load-bearing piece; the
worked example is one instance.

- **“How do I set up local telemetry monitoring with traceback logging?”** —
  worked example: *“I want to monitor `port_0_rx_bytes` on the DPU and log
  history before and after it exceeds 1 MB/s”*. Answered by the JSON
  configuration and rule definitions in the tasks companion.
- **“What rule types are available and when do they activate/deactivate?”** —
  worked example: *“I need a rule that triggers each time a counter increases
  by a threshold, not just when it exceeds a fixed value”*. Answered by the
  rule definitions in the capabilities companion.
- **“Why is my differential rule not triggering?”** — worked example: *“I
  configured `differential_greater_than` but the event log stays empty”*.
  Answered by the depth requirement and delta logic in the capabilities
  companion.
- **“How do I compile and run only this application?”** — worked example: *“I
  want to build just the Telemetry Traceback app, not all DOCA applications”*.
  Answered by the Meson isolation flags in the tasks companion.
- **“What does the event log contain when multiple events overlap?”** —
  worked example: *“Two rules trigger at the same time; how are begin/end
  markers written?”*. Answered by the event log lifecycle in the capabilities
  companion.
- **“How do I know if my device supports the data IDs I configured?”** —
  worked example: *“The application reports unsupported IDs; what do I do?”*.
  Answered by the unsupported data ID handling in the tasks companion.

## When to load this skill

Load this skill when the user is — or the agent needs to — drive the DOCA
Telemetry Traceback application on a real host or BlueField Arm with DOCA
installed (or inside the public NGC DOCA container with the right device
passthrough). Concretely:

- Configuring JSON rules and data IDs for local telemetry monitoring.
- Compiling the application from source.
- Running the application with a PCIe address and rules file.
- Diagnosing an empty event log, unsupported data IDs, or sample size/period
  mismatch.
- Understanding rule activation/deactivation semantics.
- Cross-checking event log output against configured rules.

Do **not** load this skill for general DOCA orientation, the DOCA Telemetry
Diag programming API, raw telemetry collection, DOCA install, or version
questions. For those, route to
`doca-public-knowledge-map/SKILL.md`,
`doca-setup/SKILL.md`, or
`doca-version/SKILL.md`.

## What this skill provides

This is a **thin loader**. Substantive material lives in two companion files:

- the capabilities companion — what the application monitors and logs: the
  four components, configuration schema, rule definitions with
  activation/deactivation triggers, event log lifecycle, unsupported data ID
  behavior, version overlay that redirects to
  `doca-version/SKILL.md`, and the safety policy for
  `--force-ownership`.
- the tasks companion — step-by-step workflows for the in-scope task verbs:
  `configure` (JSON config, rule selection), `build` (Meson compile all or
  only this app), `run` (execution syntax, root, PCIe address), `debug`
  (empty log, unsupported IDs, sample mismatch), plus a `Command appendix`
  that honors the bundle's
  `doca-structured-tools-contract/SKILL.md`
  preamble.

The skill assumes a host or BlueField where DOCA is already installed (or the
public NGC DOCA container is running with the right device passthrough) and
the operator has root privileges as required by the public guide.

## What this skill deliberately does not ship

This skill is **agent guidance**, not a samples or scripts bundle. To keep the
boundary clean, it deliberately does not contain — and pull requests should
not add:

- **Verbatim flag inventories, subcommand names, or output column names.**
  The public DOCA Telemetry Traceback guide and the installed `--help` on the
  user's version are the joint source of truth; copying them here pins the
  skill to one release and silently rots when the application evolves. The
  skill routes the agent at those sources instead.
- **Pre-baked example output.** Output is install-, version-, and
  device-specific. A captured example will mislead an operator on a different
  platform / state.
- **Wrappers, parsers, or scripts** in any language that consume the
  application's output. The output format is documented; users who want to
  script against it should read the live guide and write the parser against
  their installed version.
- **A `samples/` or `reference/` subtree.** This is a thin loader for a
  documented DOCA artifact; substantive material lives on the public page and
  in `--help`.

## Loading order

1. Read this skill first to confirm the user's question is in scope (the user
   wants to monitor telemetry data locally and capture traceback event logs,
   not use the DOCA Telemetry Diag programming API directly).
2. **For what the application monitors, the configuration schema, rule
   definitions, event log lifecycle, unsupported data ID behavior, version
   availability, and safety posture, see the capabilities companion.**
3. **For the documented invocations and workflows — `configure`, `build`,
   `run`, `debug`, plus the `Command appendix` — see the tasks companion.**

## Related skills

- `doca-hardware-safety/SKILL.md` — the
  bundle's hardware/firmware state-change gate. The Telemetry Traceback
  application is read-only for telemetry data, but the `--force-ownership`
  flag changes diag domain ownership; route any use of that flag through this
  skill before execution.
- `doca-version/SKILL.md` — canonical DOCA
  version-handling rules. The `## Version compatibility` section in the
  capabilities companion is a concise overlay that redirects here for the
  body.
- `doca-setup/SKILL.md` — env preparation,
  install verification, and the *I have no install yet* path with the public
  NGC DOCA container. This skill assumes its preconditions are satisfied.
- `doca-public-knowledge-map/SKILL.md`
  — routing to the public DOCA Telemetry Traceback guide and the rest of the
  public DOCA documentation set. The canonical URL is reached via
  `doca-public-knowledge-map ## DOCA tools`
  (`doca-public-knowledge-map/SKILL.md#doca-tools`).
- `doca-structured-tools-contract/SKILL.md`
  — the bundle's detect → prefer → fall back → report contract for structured
  helper tools. The Command appendix in the tasks companion honors this
  contract.
