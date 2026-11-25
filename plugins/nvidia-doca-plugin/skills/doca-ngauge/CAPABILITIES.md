# DOCA ngauge — Capabilities

**Where to start:** ngauge is a single DOCA tool for analyzing,
visualizing, and debugging network performance on a single node by
probing NIC hardware counters and storing collected data in HDF5
format. Pick the pattern first, then drill into the H2 that owns the
substance. For the *how* of executing each pattern, jump to
[TASKS.md](TASKS.md). For the public guide that is the authoritative
source for binary names, flags, paths, and the configuration schema,
see [`## Public-source pointer`](#public-source-pointer).

This file is loaded by [`SKILL.md`](SKILL.md). It documents *what
ngauge measures and reports*, *which configuration axes are
documented*, *what versions and prerequisites it ships with*, *the
layered error and observability surfaces*, and *the safety policy that
gates any state-changing step*. For step-by-step invocations and the
configure → run → parse/plot workflow, see [`TASKS.md`](TASKS.md).

## Pattern overview

Every ngauge question this skill teaches resolves into one of SIX
patterns. The patterns are CLASSES — they apply across every DOCA
install that ships ngauge, not just one platform.

| ngauge pattern | Class shape | Where the substance lives |
| --- | --- | --- |
| 1. Configure a counter sampling run | Choose the device PCIe address, output path and prefix, runtime parameters (`mode`, `period_us`), and the counter list in the input YAML file | [`## Capabilities and modes`](#capabilities-and-modes) configuration model + [TASKS.md ## configure](TASKS.md#configure) |
| 2. Run and stop a measurement | Execute `ngauge <configuration YAML file>`; use Ctrl+C (SIGINT) to end early and still save all collected data | [TASKS.md ## run](TASKS.md#run) + [`## Safety policy`](#safety-policy) read-only posture |
| 3. Plot or parse HDF5 output | Use the sample plugins `simple_plot.py` or `simple_text_plot.py` against the generated `.h5` file and one or more counter IDs | [`## Capabilities and modes`](#capabilities-and-modes) sample plugins + [TASKS.md ## parse](TASKS.md) |
| 4. Diagnose fwctl conflict or missing prerequisites | Map the symptom (ngauge cannot start, fwctl in use, unsupported hardware or firmware) to the right layer of the error taxonomy before any state-changing intervention | [`## Error taxonomy`](#error-taxonomy) + [TASKS.md ## debug](TASKS.md#debug) |
| 5. Interpret progress display and measurement summary | Read per-counter progress bars, color coding, and the once-per-second mean / standard deviation / minimum / maximum updates | [`## Observability`](#observability) + [`## Capabilities and modes`](#capabilities-and-modes) progress display |
| 6. Route state-changing, install, or version questions | Send hardware or firmware state changes to `doca-hardware-safety`, installs to `doca-setup`, version questions to `doca-version` | [`## Safety policy`](#safety-policy) + [`SKILL.md ## Related skills`](SKILL.md#related-skills) |

Two cross-cutting rules that apply to *every* pattern above:

- **ngauge is read-only probing.** It probes NIC hardware counters and
  stores HDF5 data; it does not change hardware or firmware state. Any
  step that changes hardware or firmware state routes through
  `doca-hardware-safety/SKILL.md` before execution.
- **ngauge relies on the fwctl driver and cannot run simultaneously
  with other tools or services that also use fwctl.** The agent must
  check for fwctl conflicts before any run, and must route driver
  installation or conflict resolution to `doca-setup/SKILL.md`.

## Capabilities and modes

The DOCA ngauge tool is a documented DOCA artifact that ships on
installs which include it. The exact installed location, binary name,
packaging shape, and per-platform availability live in the public DOCA
ngauge guide reachable via
[`## Public-source pointer`](#public-source-pointer)
and on the installed `--help`; the skill deliberately does not pin
them — see
[`SKILL.md ## What this skill deliberately does not ship`](SKILL.md#what-this-skill-deliberately-does-not-ship).

**Use case — what ngauge is *for*.** ngauge analyzes, visualizes, and
debugs network performance on a single node. It probes NIC hardware
counters and stores the collected data in HDF5 format, along with
relevant metadata, for subsequent processing. The tool also provides
graphical progress updates and a measurement summary directly on the
CLI, offering real-time insights into the measurement process.

**Configuration model — the load-bearing concept.** Every ngauge run
commits to a point in this space; omitting any axis produces an
ambiguous answer the agent must flag back to the user.

| Axis | What it picks | Why the agent must name it |
| --- | --- | --- |
| Device | The PCIe address of the NIC to probe, for example `device: "0000:03:00.0"` | Without a device, ngauge has no target; the address must be the operator's actual device, not a placeholder |
| Output | The output directory path and file prefix, both mandatory; the file is saved as `/path/to/output/directory/<prefix>_<DATE>_<TIME>.h5` | The exact file name is printed after each run; omitting path or prefix makes the run invalid |
| Runtime parameters | `mode` (`repetitive` or `single`) and `period_us` (sampling period in microseconds; decimal or scientific notation accepted, e.g. `1e2` = 100 μs) | These control how long and how often ngauge samples; the agent must state them explicitly |
| Counters | A list of counter definitions; `id` (data ID) is mandatory, while `desc`, `unit`, `accumulating`, `normalizer`, `cutoff_min`, and `cutoff_max` are optional | The counter IDs determine what is measured; all supported performance counters are listed in the public guide's "Supported Data IDs" section |

**Counter field semantics.** Each counter entry requires an `id`
(data ID). Optional fields are:

- `desc` — description of the counter.
- `unit` — unit type.
- `accumulating` — whether the counter accumulates values.
- `normalizer` — if present, must be either `time`, a non-zero number,
  or one of the configured Data IDs with an `id/` prefix.
- `cutoff_min` — data below this value will not be recorded.
- `cutoff_max` — data above this value will not be recorded.

**Output and metadata.** The output file is saved in HDF5 format with
relevant metadata. The exact file name is printed after each run. If
the `silent` option is set to `true`, progress indications on the
command line are suppressed (default `false`).

**Progress display.** During a run, progress bars for each counter are
displayed. Color coding indicates value levels relative to other
values of the same counter: blue for low values, red for high values,
intermediate gradient colors for values between low and high, and
solid gray bars for counters whose values did not change during the
run. Data is updated once per second, showing the mean, standard
deviation (σ), minimum, and maximum values observed during the last
second.

**Sample plugins.** A sample plugin named `simple-plot` is provided
and installed under `/usr/share/doc/ngauge/examples/plugins`. It
demonstrates how to open the output HDF5 file generated by ngauge and
plot the data. An alternative plugin, `simple_text_plot.py`, generates
text-based plots directly in the terminal; its resolution is
significantly lower, but it supports zooming into specific regions and
is useful when graphical output is unavailable or network connectivity
to the server is limited. The sample plugins are provided as examples
and are not integral parts of the ngauge tool. Dependencies such as
NumPy, H5py, Matplotlib, plotext, and others may need to be installed
separately to run these plugins.

## Version compatibility

The canonical version-handling rules live in
`doca-version/SKILL.md`; this section is a concise overlay that
redirects there for the body.

- Supported hardware: NVIDIA BlueField-3, NVIDIA ConnectX-7, and
  above.
- Required firmware version: xx.43.1000 or higher.
- fwctl driver must be installed on the host. On hosts running Linux
  kernel version 6.15 and above, fwctl is available by default unless
  it was explicitly disabled during kernel configuration.
- ngauge is installed with `sudo apt-get install ngauge` or
  `sudo dnf install ngauge` on x86 or Arm 64 hosts. On the DPU, the
  ngauge package is pre-installed, so the above step is not necessary.
- This skill documents the DOCA 3.5.0 ngauge guide. For questions
  about other DOCA versions or version-specific behavior, route to
  `doca-version/SKILL.md`.

## Error taxonomy

Map the symptom to the right layer before any state-changing
intervention.

| Symptom | Layer | Route |
| --- | --- | --- |
| ngauge cannot start because fwctl is in use | fwctl driver conflict | Check for other tools or services that also use fwctl; route driver conflict resolution to `doca-setup/SKILL.md` |
| fwctl driver is missing | Prerequisite gap | Install fwctl via the OS package manager; on Linux kernel 6.15+ it is available by default unless explicitly disabled |
| Unsupported hardware or firmware below xx.43.1000 | Version / hardware gap | Confirm BlueField-3 or ConnectX-7 or above and firmware xx.43.1000 or higher; route version questions to `doca-version/SKILL.md` |
| Output path or prefix missing | Configuration error | Both output path and file prefix are mandatory in the YAML file |
| Invalid counter ID or normalizer | Configuration error | Counter `id` is mandatory; `normalizer`, if present, must be `time`, a non-zero number, or a configured Data ID with an `id/` prefix |
| Plugin fails to plot | Missing plugin dependencies | Dependencies such as NumPy, H5py, Matplotlib, plotext, and others may need to be installed separately |
| Run ended before full dataset collected | Normal operation | Ctrl+C (SIGINT) is a normal and supported way to end the run; all results collected up to that point are saved as usual |

## Observability

ngauge exposes the following observability surfaces:

- **CLI progress bars** for each counter, updated once per second,
  showing mean, standard deviation (σ), minimum, and maximum values
  observed during the last second.
- **Color coding** on progress bars: blue for low values, red for high
  values, gradient for intermediate values, solid gray for no changes.
- **Measurement summary** directly on the CLI, offering real-time
  insights into the measurement process.
- **Exact output file name** printed after each run.
- **HDF5 output file** with relevant metadata for subsequent
  processing.
- **Sample plotting plugins** under
  `/usr/share/doc/ngauge/examples/plugins` for graphical or text-based
  plots. The graphical plugin includes a zoom-in function; the lighter
  area around samples represents the worst-case measurement error.
- **Silent mode** via the `silent` option to suppress CLI progress
  indications when set to `true`.

To reference the most recent results when the output directory is
`/tmp` (the default), use the expression shown below:

```bash
$(ls -1 /tmp/ngauge_data_*.h5 | tail -n1)
```

## Safety policy

- **ngauge is read-only probing.** It does not change hardware or
  firmware state. Any step that changes hardware or firmware state
  routes through `doca-hardware-safety/SKILL.md` before execution.
- **fwctl conflict is a hard gate.** ngauge cannot run simultaneously
  with other tools or services that also use fwctl. The agent must
  check for conflicts before any run and must not recommend running
  ngauge alongside another fwctl user.
- **Ctrl+C is a supported stop path.** Ending a run early with SIGINT
  is normal and supported; all results collected up to that point are
  saved as usual. The agent must not treat an early stop as a failure.
- **Install and version questions route elsewhere.** Installs route to
  `doca-setup/SKILL.md`; version questions route to
  `doca-version/SKILL.md`. This skill assumes ngauge and fwctl are
  already installed and the operator has the privileges the public
  guide requires for the chosen device.

## Public-source pointer

The authoritative source for the binary name, flags, paths, and
configuration schema is the public DOCA ngauge guide at
https://docs.nvidia.com/doca/archive/3-5-0/doca-ngauge/index.html.
Confirm details with the installed tool's `--help` and the sample
configuration files under `/usr/share/doc/ngauge/examples/settings`
before quoting them. The skill deliberately does not pin verbatim
counter ID inventories, flag lists, or output column names; those live
in the public guide and the installed version.
