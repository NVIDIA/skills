# DOCA Telemetry Traceback Application — Capabilities

**Where to start:** The Telemetry Traceback application is a single DOCA
artifact that can be deployed in more than one shape; the pattern overview
below names the recurring traceback-class questions. Pick the pattern first,
then drill into the H2 that owns the substance. For the *how* of executing
each pattern, jump to [TASKS.md](TASKS.md). For the host ↔ DPU control-plane
sibling whose channels coordinate the endpoints the relay carries traffic
between, see `doca-comch/CAPABILITIES.md`.

This file is loaded by [`SKILL.md`](SKILL.md). It documents *what the
application monitors and logs*, *which deployment shapes are documented*,
*what versions it ships in*, *the layered error and observability surfaces*,
and *the high-stakes safety policy that gates the data-path pieces*. For
step-by-step invocations and the smoke-before-bulk workflow, see
[`TASKS.md`](TASKS.md).

## Pattern overview

Every Telemetry Traceback question this skill teaches resolves into one of
SIX patterns. The patterns are CLASSES — they apply across every DOCA
install that ships the application, not just one platform.

| Traceback pattern | Class shape | Where the substance lives |
| --- | --- | --- |
| 1. Configure local telemetry monitoring with traceback logging | Define `diag_sample_period_ns`, `sample_history_depth`, `data-ids`, and `rules` in the JSON configuration file so the application samples device telemetry locally and writes event logs when rules trigger | [`## Capabilities and modes`](#capabilities-and-modes) configuration model + [TASKS.md ## configure](TASKS.md#configure) |
| 2. Select rule type and understand hysteresis | Choose among `incremental_greater_than`, `differential_greater_than`, `differential_less_than`, `greater_than`, and `less_than`; each has distinct activation and deactivation triggers | [`## Capabilities and modes`](#capabilities-and-modes) rule definitions + [TASKS.md ## configure](TASKS.md#configure) step 2 |
| 3. Build only this application | Isolate the Telemetry Traceback application from the full DOCA Meson build using `-Denable_all_applications=false -Denable_telemetry=true -Denable_telemetry_traceback=true` or by editing `meson_options.txt` | [TASKS.md ## build](TASKS.md#build) |
| 4. Run the application with correct PCIe address and root | Execute `doca_telemetry_traceback` with mandatory `-d` PCIe address and `-r` rules JSON, plus optional `--event-log` and `--force-ownership`; requires sudo/root privileges | [TASKS.md ## run](TASKS.md#run) |
| 5. Diagnose empty event log / unsupported IDs / sample mismatch | Map the symptom (no events logged, application reports unsupported data IDs, application stops due to sample size/period mismatch) to the right layer of the error taxonomy before any state-changing intervention | [`## Error taxonomy`](#error-taxonomy) + [TASKS.md ## debug](TASKS.md#debug) |
| 6. Handle overlapping events and log lifecycle | Understand how the events log captures history, event start/end markers, and simultaneous rule activations/deactivations | [`## Capabilities and modes`](#capabilities-and-modes) event log lifecycle + [`## Observability`](#observability) |

Two cross-cutting rules that apply to *every* pattern above:

- **The application is read-only for telemetry data, but `--force-ownership`
  changes diag domain ownership.** A misconfigured rule, an unsupported data
  ID, or a sample size/period mismatch produces a *silent* break — the
  operator observes an empty event log or an application stop whose root
  cause is the configuration, not the device. The agent must say so before
  any state-changing operation, and must route any use of
  `--force-ownership` through
  [`## Safety policy`](#safety-policy) before execution.
- **Differential rules require depth before functioning.** `dt_gt` and
  `dt_lt` require at least `sample_history_depth` values in the buffer before
  they can evaluate; `gt` and `lt` require `sample_history_depth` consecutive
  samples on the correct side of the threshold. An agent that diagnoses an
  empty event log without naming this depth requirement is missing the
  load-bearing constraint documented in
  [`## Capabilities and modes`](#capabilities-and-modes).

## Capabilities and modes

The DOCA Telemetry Traceback application is a documented DOCA artifact that
ships on installs which include it. The exact installed location, binary
name, packaging shape, and per-platform availability live in the public
DOCA Telemetry Traceback guide reachable via
`doca-public-knowledge-map ## DOCA tools` (`doca-public-knowledge-map/SKILL.md#doca-tools`)
and on the installed `--help`; the skill deliberately does not pin them —
see [`SKILL.md ## What this skill deliberately does not ship`](SKILL.md#what-this-skill-deliberately-does-not-ship).

**Use case — what the application is *for*.** The Telemetry Traceback
application provides a framework for monitoring telemetry data to identify
long-term trends locally on the device. Because funneling massive volumes of
raw telemetry data to a centralized data store can overwhelm network
resources at data-center scale, this application processes data directly on
the NVIDIA BlueField DPU or NVIDIA ConnectX NIC. It gathers telemetry data
into a circular buffer, retaining the last N samples from the device. This
historical data is continuously evaluated against user-defined rules to
detect specific events (anomalies or notable threshold breaches). When an
event triggers, the application writes the preceding historical data and all
subsequent active event data to a log file. The included rules are
purposefully simplistic, as this application is intended to serve as a
reference traceback model rather than a fully featured data processing
engine.

**Four-component architecture — the load-bearing concept.** Every Telemetry
Traceback deployment commits to this structure; omitting any component
produces an ambiguous answer the agent must flag back to the user.

| Component | What it does | Why the agent must name it |
| --- | --- | --- |
| DOCA Telemetry Diag instance | The underlying data provider, configured using parameters from the application's JSON configuration file (e.g., sample period, data IDs) alongside hardcoded application values | The application stops if the actual sample size or period does not match the requested parameters |
| Sample history circular buffer | A contiguous block of memory storing telemetry samples as `timestamp, data_id[0], ... data_id[N]`; buffer size is automatically doubled based on the requested depth to ensure equal historical and active event data | Differential rules require at least `sample_history_depth` values in the buffer before functioning |
| Rule collection | A set of logical monitors; each rule monitors a specific `data_id` against defined logic, triggering event start/end | Rule type determines activation and deactivation hysteresis |
| Events log file | Output destination for event data; logs configuration header, history, event start/end markers, and active event data | Overlapping events insert individual begin/end rows to track lifecycles |

**Configuration model.** The application retrieves its operational
parameters from a JSON configuration file. The fields are:

| Field | Type | Notes |
| --- | --- | --- |
| `diag_sample_period_ns` | Integer | Sampling interval in nanoseconds; passed directly to the diag instance via `doca_telemetry_diag_get_sample_period` |
| `sample_history_depth` | Integer | Number of historical records a rule evaluates; the application automatically provisions a circular buffer double this size |
| `data-ids` | Array | Collection of data ID definitions |
| `data-ids[N].name` | String | Friendly name for the `data_id`, used as column header in the events log |
| `data-ids[N].data_id` | String | 16-character hexadecimal value, format `0x[0-9a-fA-F]{16}` |
| `rules` | Array | Collection of rule definitions |
| `rules[N].name` | String | Friendly name used to identify the rule in the events log |
| `rules[N].data_id` | String | 16-character hexadecimal `data_id` this rule monitors; must match an ID defined in the `data-ids` array |
| `rules[N].comparison_type` | String | Defines the rule's operational logic; acceptable values: `incremental_greater_than` (`inc_gt`), `differential_greater_than` (`dt_gt`), `differential_less_than` (`dt_lt`), `greater_than` (`gt`), `less_than` (`lt`) |
| `rules[N].comparison_value` | Integer | Numerical threshold utilized when evaluating the rule |

**Rule definitions.** The application supports five distinct rule types.
Each rule establishes its own hysteresis logic, defining precisely when an
event activates and deactivates.

| Rule type | Activation trigger | Deactivation trigger |
| --- | --- | --- |
| `incremental_greater_than` (`inc_gt`) | Activates if the sampled value exceeds the calculated setpoint (Initial Setpoint = First Sampled Value + `comparison_value`) | Deactivates immediately once any new sampled value exceeds the current setpoint, and a new setpoint is calculated. Rapid data spikes exceeding the setpoint multiple times simultaneously will only activate the rule once |
| `differential_greater_than` (`dt_gt`) | Activates if the delta between the newest sample and the sample at `[N-(sample_history_depth-1)]` exceeds `comparison_value`; requires at least `sample_history_depth` values in the buffer before functioning | Deactivates only after `sample_history_depth` consecutive samples are acquired where each new delta remains below `comparison_value` |
| `differential_less_than` (`dt_lt`) | Operates using the identical historical delta logic as `dt_gt`, but activates when the delta is less than `comparison_value` | Same as `dt_gt` deactivation, but for deltas above `comparison_value` |
| `greater_than` (`gt`) | Activates when the sampled value explicitly exceeds `comparison_value` for `sample_history_depth` times consecutively; designed strictly for values that can fluctuate (increase and decrease), not suitable for metrics that only increment | Deactivates only after `sample_history_depth` consecutive samples fall below `comparison_value` |
| `less_than` (`lt`) | Operates using the identical fluctuation logic as `gt`, but activates when the sampled value remains below `comparison_value` for the required consecutive depth | Deactivates only after `sample_history_depth` consecutive samples rise above `comparison_value` |

**Event log lifecycle.** Upon execution, the application logs its
configuration to ensure absolute certainty regarding the parameters that
triggered an event. When an event starts, the log captures the available
history, an "event started" marker, all data collected during the event, and
an "event end" marker. If multiple events activate simultaneously, individual
begin/end rows are inserted to track their respective lifecycles. Data
logging pauses once all events conclude and resumes upon the next
activation.

**Unsupported data IDs.** It is not guaranteed that all data IDs supported
by the application are supported by your specific device. The application
will report any unsupported IDs, requiring administrators to adjust their
inputs accordingly.

**Sample size/period validation.** The application calls
`doca_telemetry_diag_get_sample_size()` and
`doca_telemetry_diag_get_sample_period()` to confirm the hardware accepted
the requested parameters. If the actual sample size or period does not match
the requested parameters, the application explicitly stops. While this could
be worked around, the codebase relies on these assumptions for simplicity and
requires them to be correct.

## Version compatibility

This skill documents the DOCA Telemetry Traceback application as described
in the DOCA 3.5.0 archive guide. Version-specific availability, changes, and
compatibility questions route to `doca-version/SKILL.md`. The application is
distributed as source code and must be compiled prior to execution; the
source is located at `/opt/mellanox/doca/applications/telemetry/traceback/`.
For any question about which DOCA release ships a given rule type, flag, or
behavior, consult `doca-version/SKILL.md` rather than assuming from this
skill.

## Error taxonomy

Map the symptom to the right layer before any state-changing intervention.

| Symptom | Likely layer | What to check |
| --- | --- | --- |
| Event log is empty even though a threshold was crossed | Rule configuration | Verify `comparison_type` and `sample_history_depth`; differential rules require depth samples before functioning; `gt`/`lt` require consecutive samples on the correct side |
| Application reports unsupported data IDs | Device capability | The specific device may not support all data IDs the application supports; adjust inputs accordingly |
| Application stops immediately after start | Sample size/period mismatch | The hardware did not accept the requested `diag_sample_period_ns` or sample size; the application explicitly stops |
| Application fails to access telemetry data | Privileges | The application strictly requires sudo (root privileges) to access hardware-level telemetry data |
| Application cannot find the device | PCIe addressing | Ensure the `-d` parameter exactly matches the physical PCIe address of the target device |
| Diag domain ownership conflict | Firmware domain mode | `--force-ownership` forces ownership of the diag domain; only applies when the device firmware is in single domain mode; route through `doca-hardware-safety/SKILL.md` |

## Observability

The application's primary observability surface is the events log file. The
log captures:

- **Configuration header** — written immediately upon execution via
  `log_app_start_header()`, ensuring absolute certainty regarding the
  parameters that triggered an event.
- **Historical data dump** — when the first rule activates, the application
  writes the data column headers and dumps the historical buffer to the
  events log.
- **Event start/end markers** — for all rule activations, an "event started"
  message is explicitly written; for deactivations, an "event completed"
  marker is written.
- **Active event data** — data received while any rule is active is logged
  strictly before the rules processing step for each sample, continuing
  until all rules deactivate.
- **Overlapping event tracking** — if multiple events activate
  simultaneously, individual begin/end rows are inserted to track their
  respective lifecycles.

The application also logs its configuration to ensure certainty regarding
the parameters that triggered an event. For runtime diagnostics, the
application supports standard DOCA logging flags (`-l`, `--sdk-log-level`,
`--log-filter`) as documented in the public guide and the installed
`--help`.

## Safety policy

The Telemetry Traceback application is **read-only for telemetry data** —
it samples device telemetry and writes event logs, but does not modify
hardware or firmware state. The one exception is the `--force-ownership`
flag, which forces ownership of the diag domain and only applies when the
device firmware is in single domain mode. Any use of `--force-ownership`
changes diag domain ownership and must be routed through
`doca-hardware-safety/SKILL.md` before execution.

The application strictly requires sudo (root privileges) to access
hardware-level telemetry data. The agent must state this requirement before
any run command. The application also stops if the actual sample size or
period does not match the requested parameters; this is a deliberate
safety check, not a bug, and the agent must not recommend workarounds that
bypass it.

## Public-source pointer

The source code for this application is located at:

```
/opt/mellanox/doca/applications/telemetry/traceback/
```

The authoritative public guide is the DOCA Telemetry Traceback application
documentation at:

```
https://docs.nvidia.com/doca/archive/3-5-0/doca-telemetry-traceback-application/index.html
```

For extended capability details of the underlying DOCA Telemetry Diag
library, refer to its official programming guide. For any issue encountered
with compilation, installation, or execution of the DOCA applications,
refer to the NVIDIA BlueField Platform Software Troubleshooting Guide.
