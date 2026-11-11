# DOCA Telemetry Traceback Application — Tasks

**Where to start:** The verbs that carry real workflow content are
`## configure`, `## build`, `## run`, and `## debug`. The `## modify`
verb carries a routing stub because the thing the agent modifies is the
JSON configuration and rule definitions, not the application source —
the Telemetry Traceback application is distributed as source code and
must be compiled, but source modification is out of scope for this
skill and routes to the DOCA Reference Applications page and development
tips. The `## test` verb is an iterative loop (configure → run → observe
event log → adjust, with every state-changing operation re-opening the
smoke), not a one-shot pass — see the eval-loop overlay in `## test`
below.

This file is loaded by [`SKILL.md`](SKILL.md) after
[`CAPABILITIES.md`](CAPABILITIES.md). It walks the agent through the six
task verbs every artifact in this bundle exposes
(`configure / build / modify / run / test / debug`), then explicitly
defers task verbs that do not belong here, and ends with the
`Command appendix` honoring the bundle's
`doca-structured-tools-contract/SKILL.md`
preamble.

## configure

The Telemetry Traceback application's *configuration* is the JSON file
documented in
[`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes) —
`diag_sample_period_ns`, `sample_history_depth`, `data-ids`, and
`rules` — layered on top of the precondition that DOCA is installed and
the application has been compiled. The application does not carry a
separate daemon contract of its own beyond the JSON file passed via
`-r`/`--rules`; if that flag is not provided, the application defaults
to `rules.json` in the working directory.

Steps the agent should walk the user through, in order:

1. **Confirm the DOCA install and version compatibility.** Walk
   `doca-version TASKS.md ## configure` (`doca-version/TASKS.md#configure`)
   to capture the installed DOCA version, then apply the Telemetry
   Traceback overlay in
   [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility):
   confirm the application source is present at
   `/opt/mellanox/doca/applications/telemetry/traceback/` and that the
   installed version supports the rule types and flags the user plans to
   use. If the artifact is not present on the install profile, route to
   `doca-setup ## install` (`doca-setup/TASKS.md`).
2. **Identify the device and the data IDs to monitor.** The application
   reads telemetry from a specific device via the PCIe address passed at
   run time, but the JSON configuration must name the data IDs to
   sample. Each `data_id` is a 16-character hexadecimal value in the
   format `0x[0-9a-fA-F]{16}`. Not all data IDs supported by the
   application are supported by every device; the application reports
   unsupported IDs and requires the administrator to adjust inputs. The
   agent should not commit to a data ID without confirming the device
   supports it — if unknown, run once with the candidate IDs and adjust
   based on the reported unsupported list.
3. **Choose the sample period and history depth.** `diag_sample_period_ns`
   is the sampling interval in nanoseconds and is passed directly to the
   diag instance. `sample_history_depth` is the number of historical
   records a rule evaluates; the application automatically provisions a
   circular buffer double this size to ensure equal historical and
   active event data. The agent must surface two depth constraints:
   differential rules (`dt_gt`, `dt_lt`) require at least
   `sample_history_depth` values in the buffer before they can function,
   and `gt`/`lt` rules require `sample_history_depth` consecutive samples
   on the correct side of the threshold.
4. **Define the `data-ids` array.** Each entry has a `name` (a friendly
   name used as the column header in the events log) and a `data_id`
   (the 16-character hexadecimal value). The `name` is arbitrary but
   should be stable because it appears in the log header.
5. **Define the `rules` array.** Each rule has a `name`, a `data_id`
   that must match an ID defined in the `data-ids` array, a
   `comparison_type`, and an integer `comparison_value`. The five rule
   types and their hysteresis are documented in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
   rule definitions. The agent should help the user select the rule type
   based on the metric's behavior:
   - `incremental_greater_than` (`inc_gt`) is designed for incrementally
     increasing values and triggers each time the value increases by the
     comparison threshold.
   - `differential_greater_than` (`dt_gt`) and
     `differential_less_than` (`dt_lt`) evaluate the delta between the
     newest sample and the sample at
     `[N-(sample_history_depth-1)]`.
   - `greater_than` (`gt`) and `less_than` (`lt`) are designed strictly
     for values that can fluctuate; they are not suitable for metrics
     that only increment, such as total bytes transferred.
6. **Plan the event log output.** The application writes event data to a
   log file. If `--event-log` is not provided, the application creates
   or truncates `events.log` in the working directory (the default named
   in the code flow). The agent should record the intended log path so
   the `## test` and `## debug` loops can inspect it.
7. **Validate the configuration before hardware allocation.** The
   application performs `validate_configuration()` to catch expected
   configuration errors before allocating hardware resources. The agent
   should pre-validate the same checks: `data_id` format, rule
   `data_id` matches an entry in `data-ids`, `comparison_type` is one of
   the five accepted values, `comparison_value` is an integer, and the
   sample period and depth are positive.
8. **Plan the rollback path.** Because configuration changes alter what
   is monitored and what is logged, the agent records the previous JSON
   file and event log path. If the user plans to use
   `--force-ownership`, that flag changes diag domain ownership and must
   be routed through `doca-hardware-safety/SKILL.md` before execution —
   see [`## Safety policy`](CAPABILITIES.md#safety-policy).

Example configuration, as documented in the public guide:

```json
{
  "diag_sample_period_ns": 100000,
  "sample_history_depth": 10000,
  "data-ids": [
    {
      "name": "port_0_rx_bytes",
      "data_id": "0x1020000100000000"
    }
  ],
  "rules": [
    {
      "name": "Rx Bytes/s > 1MeBs",
      "data_id": "0x1020000100000000",
      "comparison_type": "differential_greater_than",
      "comparison_value": 1048576
    }
  ]
}
```

## build

The Telemetry Traceback application is distributed as source code and
must be compiled prior to execution. The source is located at
`/opt/mellanox/doca/applications/telemetry/traceback/`. DOCA reference
applications are installed with full source code and build instructions;
for development and compilation tips, refer to the DOCA Reference
Applications page. For install-related questions, route to
`doca-setup/SKILL.md`.

**Compiling all applications.** By default, the DOCA Meson project is
configured to build every reference application simultaneously:

```bash
cd /opt/mellanox/doca/applications/
meson /tmp/build
ninja -C /tmp/build
```

The compiled binaries are generated in `/tmp/build/telemetry/`.

**Compiling only the current application.** To significantly reduce
build times, isolate the Telemetry Traceback application.

Option 1 — command line configuration:

```bash
cd /opt/mellanox/doca/applications/
meson /tmp/build -Denable_all_applications=false -Denable_telemetry=true -Denable_telemetry_traceback=true
ninja -C /tmp/build
```

Option 2 — configuration file. Edit
`/opt/mellanox/doca/applications/meson_options.txt` and set:

```
enable_all_applications = false
enable_telemetry = true
enable_telemetry_traceback = true
```

Then run the standard compilation commands:

```bash
cd /opt/mellanox/doca/applications/
meson /tmp/build
ninja -C /tmp/build
```

**Configuration rule.** If `enable_all_applications` is set to `false`,
the main `enable_telemetry` flag must be set to `true` for any
underlying telemetry application to compile successfully. The individual
applications default to `true`, but can be toggled on or off as needed
once the main telemetry flag is enabled.

## modify

The thing the agent modifies for the Telemetry Traceback application is
the JSON configuration and rule definitions, not the application source.
Configuration changes are covered in [`## configure`](#configure). If
the user wants to modify the application source code itself, that is out
of scope for this skill; the source is available at
`/opt/mellanox/doca/applications/telemetry/traceback/` and the user can
modify it as-is, but development and compilation tips live on the DOCA
Reference Applications page, not in this skill.

## run

The application is distributed as source code and must be compiled prior
to execution. The execution syntax is:

```bash
sudo doca_telemetry_traceback -d 03:00.0 --rules my_rules_file.json
```

Steps the agent should walk the user through, in order:

1. **Confirm the build completed.** The compiled binary is generated in
   `/tmp/build/telemetry/` when built with Meson. If the binary is not
   present, return to [`## build`](#build).
2. **Confirm root privileges.** The application strictly requires sudo
   (root privileges) to access hardware-level telemetry data on the
   device. The agent must state this requirement before any run command.
3. **Identify the target device PCIe address.** The mandatory `-d`
   parameter must exactly match the physical PCIe address of the target
   device (for example, `03:00.0`). A wrong address produces a device
   open failure.
4. **Prepare the rules JSON file.** The `-r`/`--rules` flag takes the
   path to the JSON configuration file. If not overridden via CLI, the
   application defaults to `rules.json` in the working directory.
5. **Optionally specify the event log path.** Use `--event-log` to set
   the output log file. If omitted, the application creates or truncates
   `events.log` in the working directory.
6. **Execute the application.** Example with an explicit event log:

   ```bash
   sudo doca_telemetry_traceback -d 03:00.0 --rules my_rules_file.json --event-log /tmp/traceback_events.log
   ```

7. **Monitor the run.** The application logs its configuration header
   immediately, then enters a continuous polling loop. It installs a
   SIGINT signal handler so the operator can terminate cleanly with
   CTRL+C. Device cleanup is handled automatically via C++ RAII
   `unique_ptr` objects when the SIGINT signal breaks the execution loop.
8. **Handle unsupported data IDs.** If the application reports
   unsupported data IDs, adjust the JSON inputs accordingly and re-run.
9. **Handle sample size/period mismatch.** The application calls
   `doca_telemetry_diag_get_sample_size()` and
   `doca_telemetry_diag_get_sample_period()` to confirm the hardware
   accepted the requested parameters. If the actual sample size or
   period does not match the requested parameters, the application
   explicitly stops. This is a deliberate safety check, not a bug; the
   agent must not recommend workarounds that bypass it. Adjust
   `diag_sample_period_ns` or the data IDs to match hardware
   capabilities.
10. **If `--force-ownership` is needed.** That flag forces ownership of
    the diag domain and only applies when the device firmware is in
    single domain mode. Route any use of it through
    `doca-hardware-safety/SKILL.md` before execution.

General DOCA flags (`-h`, `-v`, `-l`, `--sdk-log-level`,
`--log-filter`, `-j`) and program flags (`-d`, `-r`, `--event-log`,
`--force-ownership`) are listed in the
[`Command appendix`](#command-appendix).

## test

The Telemetry Traceback application does not ship a separate test
harness; the `## test` verb is an iterative verification loop built from
the run and debug verbs. The goal is to confirm that a configured rule
actually produces the expected event log output before the operator
relies on it for long-term monitoring.

The loop is:

1. **Configure a minimal rule set with a known threshold.** Choose a
   rule type appropriate to the metric being monitored. For an
   incrementing counter, use `incremental_greater_than` with a small
   `comparison_value` so the event triggers quickly. For a fluctuating
   metric, use `greater_than` or `less_than` with a threshold that will
   be crossed under normal load.
2. **Run the application** with `--event-log` pointing to a dedicated
   test log file, per [`## run`](#run).
3. **Generate the condition the rule monitors.** The application samples
   telemetry data; the operator must cause the threshold to be crossed
   (for example, by generating traffic on the monitored port).
4. **Observe the event log.** The log should contain the configuration
   header, an "event started" marker, the historical data dump, all
   active event data, and an "event completed" marker. If multiple
   events activate simultaneously, individual begin/end rows are
   inserted to track their respective lifecycles.
5. **If the log is empty or missing markers**, do not proceed to bulk
   use. Route to [`## debug`](#debug) and adjust the configuration.
6. **Terminate with SIGINT (CTRL+C)**, inspect the log, adjust the JSON
   configuration, and repeat until the event log matches the expected
   lifecycle.

Every state-changing operation in this loop (editing the JSON file,
changing the event log path, re-running with a different PCIe address)
re-opens the smoke from step 1. The test is not a one-shot pass.

## debug

Map the symptom to the right layer before any state-changing
intervention. The full error taxonomy is in
[`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy).

| Symptom | Likely layer | What to check |
| --- | --- | --- |
| Event log is empty even though a threshold was crossed | Rule configuration | Verify `comparison_type` and `sample_history_depth`. Differential rules require depth samples before functioning; `gt`/`lt` require consecutive samples on the correct side. Confirm the rule's `data_id` matches an entry in `data-ids`. |
| Application reports unsupported data IDs | Device capability | The specific device may not support all data IDs the application supports. Adjust the JSON inputs accordingly. |
| Application stops immediately after start | Sample size/period mismatch | The hardware did not accept the requested `diag_sample_period_ns` or sample size. The application explicitly stops. Adjust the sample period or data IDs to match hardware capabilities. |
| Application fails to access telemetry data | Privileges | The application strictly requires sudo (root privileges). |
| Application cannot find the device | PCIe addressing | Ensure the `-d` parameter exactly matches the physical PCIe address of the target device. |
| Diag domain ownership conflict | Firmware domain mode | `--force-ownership` forces ownership of the diag domain and only applies when the device firmware is in single domain mode. Route through `doca-hardware-safety/SKILL.md`. |

For deeper runtime diagnostics, increase log verbosity with the DOCA
logging flags (`-l`, `--sdk-log-level`, `--log-filter`) documented in
the public guide and the installed `--help`. For any issue encountered
with compilation, installation, or execution of the DOCA applications,
refer to the NVIDIA BlueField Platform Software Troubleshooting Guide.

## Deferred task verbs

The following task verbs do not belong in this skill and are explicitly
deferred:

- **`install`** — routes to `doca-setup/SKILL.md`. This skill assumes
  DOCA is already installed and the application source is present at
  `/opt/mellanox/doca/applications/telemetry/traceback/`.
- **`version`** — routes to `doca-version/SKILL.md`. Version-specific
  availability, changes, and compatibility questions are owned there.
- **`telemetry-diag-api`** — routes to the DOCA Telemetry Diag
  programming guide. This skill covers the Traceback application, not
  the underlying Telemetry Diag library API.
- **`raw-telemetry`** — routes to the DOCA Telemetry Diag programming
  guide. The Traceback application processes telemetry data locally for
  traceback logging; it is not a raw telemetry collection tool.
- **`hardware-state-change`** — routes to
  `doca-hardware-safety/SKILL.md`. The only hardware/firmware
  state-changing flag in this application is `--force-ownership`.
- **`source-modification`** — routes to the DOCA Reference Applications
  page and development tips. This skill covers configuration and
  operation, not modifying the application source code.

## Command appendix

The commands below are the fallback path when the bundle's structured
helper tools are not available; prefer the structured tools per
`doca-structured-tools-contract/SKILL.md`.

**Build all applications:**

```bash
cd /opt/mellanox/doca/applications/
meson /tmp/build
ninja -C /tmp/build
```

**Build only the Telemetry Traceback application (CLI flags):**

```bash
cd /opt/mellanox/doca/applications/
meson /tmp/build -Denable_all_applications=false -Denable_telemetry=true -Denable_telemetry_traceback=true
ninja -C /tmp/build
```

**Build only the Telemetry Traceback application (configuration file):**
Edit `/opt/mellanox/doca/applications/meson_options.txt` to set
`enable_all_applications = false`, `enable_telemetry = true`,
`enable_telemetry_traceback = true`, then:

```bash
cd /opt/mellanox/doca/applications/
meson /tmp/build
ninja -C /tmp/build
```

**Run the application:**

```bash
sudo doca_telemetry_traceback -d 03:00.0 --rules my_rules_file.json
```

**Run with an explicit event log:**

```bash
sudo doca_telemetry_traceback -d 03:00.0 --rules my_rules_file.json --event-log /tmp/traceback_events.log
```

**Print help:**

```bash
doca_telemetry_traceback --help
```

**Print version:**

```bash
doca_telemetry_traceback --version
```

**General flags:**

- `-h`, `--help` — print help synopsis and exit.
- `-v`, `--version` — print program version information and exit.
- `-l`, `--log-level` — set numeric log level (10 DISABLE, 20 CRITICAL,
  30 ERROR, 40 WARNING, 50 INFO, 60 DEBUG, 70 TRACE; TRACE requires
  compilation with TRACE support).
- `--sdk-log-level` — set SDK numeric log level using the same 10–70
  scale.
- `--log-filter` — filter logs from specific modules (comma-separated
  list).
- `-j`, `--json` — parse command-line flags from a specified input JSON
  file.

**Program flags:**

- `-d`, `--device` — mandatory; exact PCIe address of the device to read
  telemetry information from.
- `-r`, `--rules` — path to a JSON file containing the application
  configuration.
- `--event-log` — path to output log file for events information.
- `--force-ownership` — force ownership of the diag domain; only applies
  when the device firmware is in single domain mode. Route through
  `doca-hardware-safety/SKILL.md` before use.

## Cross-cutting

- **Root privileges** are mandatory for every run; the agent must state
  this before any execution command.
- **PCIe address** must exactly match the physical device; a wrong
  address fails at device open.
- **Unsupported data IDs** are reported by the application; adjust the
  JSON inputs and re-run.
- **Sample size/period mismatch** causes the application to stop
  deliberately; do not bypass this check.
- **`--force-ownership`** changes diag domain ownership and is the only
  hardware/firmware state-changing flag; route through
  `doca-hardware-safety/SKILL.md`.
- **Version questions** route to `doca-version/SKILL.md`; install
  questions route to `doca-setup/SKILL.md`.
- **The application is read-only for telemetry data** except for the
  `--force-ownership` flag; see
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy).
