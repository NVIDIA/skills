# DOCA ngauge — Tasks

**Where to start:** The verbs that carry real workflow content are
`## configure`, `## run`, `## test`, and `## debug`. The other two
substantive verbs (`build`, `modify`) carry routing stubs because
ngauge is a shipped package / pre-installed binary, not a source
artifact the external operator compiles or patches — the *thing the
agent modifies* is the input YAML configuration and the invocation,
not the tool itself. The `## test` verb is an iterative loop
(configure → run → verify HDF5 → plot → admit), not a one-shot pass —
see the eval-loop overlay in `## test` below.

This file is loaded by [`SKILL.md`](SKILL.md) after
[`CAPABILITIES.md`](CAPABILITIES.md). It walks the agent through the six
task verbs every artifact in this bundle exposes
(`configure / build / modify / run / test / debug`), then explicitly
defers task verbs that do not belong here, and ends with the
`Command appendix` honoring the bundle's
`doca-structured-tools-contract/SKILL.md`
preamble.

For ngauge, the verbs that carry real workflow content are
`configure`, `run`, `test`, and `debug`. The other two verbs *exist as
anchors* because the agent's task-verb contract is uniform across
libraries, services, and tools — and each one carries a meaningful
**routing stub** that names where the user's question really belongs.

## configure

The ngauge *configuration* is the four-axis decision documented in
[`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes) —
device PCIe address × output path and prefix × runtime parameters ×
counter list — layered on top of the precondition that ngauge and the
fwctl driver are installed on the side the operator is about to invoke
the tool from. ngauge does not carry a separate config file or daemon
contract of its own beyond the input YAML file; the sample
configuration files under `/usr/share/doc/ngauge/examples/settings`
are the starting point.

Steps the agent should walk the user through, in order:

1. **Confirm the prerequisites.** The target device must be a
   BlueField-3 or ConnectX-7 or above with firmware version
   xx.43.1000 or higher. The fwctl driver must be installed on the
   host; on Linux kernel 6.15 and above it is available by default
   unless explicitly disabled during kernel configuration. If fwctl is
   missing or ngauge is not installed, route to
   `doca-setup ## install` (`doca-setup/TASKS.md#configure`) before
   continuing.
2. **Copy a sample configuration file.** Sample files are under
   `/usr/share/doc/ngauge/examples/settings`. Use the one that best
   fits the scenario (single/dual port, etc.). The agent should not
   invent a YAML file from scratch when a sample exists.
3. **Specify the device.** Set the PCIe address of the device to run
   on, for example `device: "0000:03:00.0"`. The address must be the
   operator's actual device, not a placeholder.
4. **Configure the output path and file prefix.** Both are mandatory.
   The output file is saved in the format
   `/path/to/output/directory/<prefix>_<DATE>_<TIME>.h5`; the exact
   file name is printed after each run. The `silent` option suppresses
   CLI progress indications when set to `true` (default `false`).
5. **Configure runtime parameters.** Set `mode` to `repetitive` or
   `single`, and set `period_us` to the sampling period in
   microseconds. Decimal or scientific notation is accepted; for
   example, `1e2` means 100 μs.
6. **Define the counters to measure.** Each counter entry requires an
   `id` (data ID). Optional fields are `desc`, `unit`, `accumulating`,
   `normalizer`, `cutoff_min`, and `cutoff_max`. The `normalizer`, if
   present, must be either `time`, a non-zero number, or one of the
   configured Data IDs with an `id/` prefix. All supported performance
   counters are listed in the public guide's "Supported Data IDs"
   section; the agent must confirm the chosen IDs against that list
   before committing.
7. **Validate the YAML against the sample and the public guide.** The
   agent should not quote counter IDs, flags, or paths from memory;
   confirm them with the installed tool's help and the sample
   configuration files before the run.

## build

No build workflow. ngauge is installed as a package
(`sudo apt-get install ngauge` or `sudo dnf install ngauge` on x86 or
Arm 64 hosts; pre-installed on the DPU). The public guide does not
document building ngauge from source. If the user asks to build ngauge
from source, route to `doca-setup/SKILL.md` for install verification
and to the public guide for any source-build instructions that may
exist there; do not invent build steps.

## modify

No modify workflow for the tool binary. The thing the agent modifies is
the input YAML configuration file and the invocation, not the installed
ngauge binary. If the user asks to patch, recompile, or otherwise
modify ngauge itself, route to `doca-setup/SKILL.md` and the public
guide; do not recommend editing the installed package.

## run

The run verb is the execution flow: invoke ngauge with a prepared YAML
configuration, observe the progress display, stop when appropriate, and
then parse or plot the resulting HDF5 output.

1. **Run ngauge.** Execute:

   ```bash
   ngauge <configuration YAML file>
   ```

   The output is saved as an HDF5 file (`.h5`) in the path specified
   in the configuration YAML. The exact file name is printed after
   each run.

2. **Observe the progress display.** During the run, progress bars for
   each counter are displayed. Color coding indicates value levels
   relative to other values of the same counter: blue for low values,
   red for high values, intermediate gradient colors for values
   between low and high, and solid gray bars for counters whose values
   did not change during the run. Data is updated once per second,
   showing the mean, standard deviation (σ), minimum, and maximum
   values observed during the last second.

3. **End the run.** To end a run before the full dataset is collected,
   use Ctrl+C (SIGINT). This is a normal and supported way to end the
   run, and all results collected up to that point are saved as usual.
   The agent must not treat an early stop as a failure.

4. **Parse or plot the output.** A sample plugin named `simple-plot` is
   installed under `/usr/share/doc/ngauge/examples/plugins`. To plot
   the data from an ngauge output file, use:

   ```bash
   /usr/share/doc/ngauge/examples/plugins/simple_plot.py <ngauge output .h5 file> <counter ID> [<counter ID> ...]
   ```

   Use the `-h` flag to view additional options, including plotting on
   a logarithmic scale and in the frequency domain. An alternative
   plugin, `simple_text_plot.py`, generates text-based plots directly
   in the terminal; its resolution is significantly lower, but it
   supports zooming into specific regions and is useful when graphical
   output is unavailable or network connectivity to the server is
   limited. The usage syntax is similar:

   ```bash
   /usr/share/doc/ngauge/examples/plugins/simple_text_plot.py <ngauge output .h5 file> <counter ID> [<counter ID> ...]
   ```

   Use the `-h` flag to view additional options, such as plotting in
   the frequency domain, specifying the start, end, or duration of the
   plotted interval, and more. The sample plugins are provided as
   examples and are not integral parts of the ngauge tool. Dependencies
   such as NumPy, H5py, Matplotlib, plotext, and others may need to be
   installed separately to run these plugins.

5. **Reference the most recent results.** If the output directory is
   `/tmp` (the default), the most recent results can be referenced
   without manually copying the file name by using the expression shown
   below:

   ```bash
   $(ls -1 /tmp/ngauge_data_*.h5 | tail -n1)
   ```

## test

The test verb is an iterative loop, not a one-shot pass. The agent
should walk the user through a minimal smoke test before admitting a
full measurement run:

1. **Configure a minimal YAML.** Use one counter, a short `period_us`,
   and a known output path and prefix. Confirm the device PCIe address
   and prerequisites.
2. **Run ngauge.** Execute the command from [`## run`](#run) and let it
   run briefly, then stop with Ctrl+C (SIGINT). Confirm that the exact
   output file name is printed and that the `.h5` file exists in the
   configured output path.
3. **Verify the output.** Plot the collected data with
   `simple_text_plot.py` (or `simple_plot.py` if graphical output is
   available) using the counter ID from the YAML. Confirm the plot
   renders and the counter values are visible.
4. **Admit the full run.** Only after the smoke test passes should the
   agent admit the full measurement configuration. If any step fails,
   route to [`## debug`](#debug) before retrying.

Because ngauge is read-only probing, the test loop does not change
hardware or firmware state. Any step that would change hardware or
firmware state routes through `doca-hardware-safety/SKILL.md` before
execution.

## debug

Map the symptom to the right layer before any state-changing
intervention. The error taxonomy in
[`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
is the starting point.

1. **fwctl conflict.** If ngauge cannot start because fwctl is in use,
   check for other tools or services that also use fwctl. ngauge cannot
   run simultaneously with other fwctl users. Route driver conflict
   resolution to `doca-setup/SKILL.md`.
2. **Missing fwctl driver.** If fwctl is missing, install it via the
   OS package manager. On hosts running Linux kernel 6.15 and above,
   fwctl is available by default unless explicitly disabled during
   kernel configuration. Route install steps to `doca-setup/SKILL.md`.
3. **Unsupported hardware or firmware.** Confirm the device is a
   BlueField-3 or ConnectX-7 or above and the firmware version is
   xx.43.1000 or higher. If not, route version and hardware questions
   to `doca-version/SKILL.md`.
4. **Configuration errors.** Both output path and file prefix are
   mandatory. Counter `id` is mandatory. The `normalizer`, if present,
   must be `time`, a non-zero number, or a configured Data ID with an
   `id/` prefix. Validate the YAML against the sample files under
   `/usr/share/doc/ngauge/examples/settings`.
5. **Plugin failures.** If a sample plugin fails to plot, dependencies
   such as NumPy, H5py, Matplotlib, plotext, and others may need to be
   installed separately. The sample plugins are examples and are not
   integral parts of ngauge.
6. **Early stop.** Ending a run with Ctrl+C (SIGINT) is normal and
   supported; all results collected up to that point are saved as
   usual. Do not treat an early stop as a failure.

## Deferred task verbs

The following task verbs are deferred to other skills and must not be
answered from this skill:

- **install** — ngauge and fwctl installation, package manager
  commands, and driver conflict resolution route to
  `doca-setup/SKILL.md`.
- **version** — DOCA version compatibility, firmware version checks,
  and version-specific behavior route to `doca-version/SKILL.md`.
- **hardware/firmware state changes** — any step that changes hardware
  or firmware state routes to `doca-hardware-safety/SKILL.md`.
- **packet I/O** — raw packet processing, line-rate I/O, and the eth
  API route to `doca-eth/SKILL.md`.
- **control plane** — host ↔ DPU control-plane primitives route to
  `doca-comch/SKILL.md`.
- **general DOCA orientation** — routing to the public documentation
  set and the knowledge map routes to
  `doca-public-knowledge-map/SKILL.md`.
- **cross-cutting debug** — when the cause is below DOCA or spans
  multiple components, route to `doca-debug/SKILL.md`.

## Command appendix

This appendix honors the bundle's
`doca-structured-tools-contract/SKILL.md`
preamble: detect the installed tool with its help before quoting flags,
prefer the installed version's help over this appendix, fall back to
the public guide, and report any discrepancy.

**Detect the installed tool**

```bash
ngauge --help
```

**Install fwctl (Deb-based)**

```bash
apt-cache search fwctl
sudo apt install <package-name>
```

**Install fwctl (RPM-based)**

```bash
dnf search fwctl
sudo dnf install <package-name>
```

**Install ngauge**

```bash
sudo apt-get install ngauge
```

or

```bash
sudo dnf install ngauge
```

**Run ngauge**

```bash
ngauge <configuration YAML file>
```

**Plot output (graphical)**

```bash
/usr/share/doc/ngauge/examples/plugins/simple_plot.py <ngauge output .h5 file> <counter ID> [<counter ID> ...]
```

**Plot output (text)**

```bash
/usr/share/doc/ngauge/examples/plugins/simple_text_plot.py <ngauge output .h5 file> <counter ID> [<counter ID> ...]
```

**Reference the most recent results in `/tmp`**

```bash
$(ls -1 /tmp/ngauge_data_*.h5 | tail -n1)
```

## Cross-cutting

- **Safety gate.** ngauge is read-only probing. Any step that changes
  hardware or firmware state routes through
  `doca-hardware-safety/SKILL.md` before execution.
- **Version gate.** Version questions, firmware version checks, and
  version-specific behavior route to `doca-version/SKILL.md`. The
  `## Version compatibility` section in
  [`CAPABILITIES.md`](CAPABILITIES.md#version-compatibility) is a
  concise overlay that redirects there for the body.
- **Install gate.** ngauge and fwctl installation, package manager
  commands, and driver conflict resolution route to
  `doca-setup/SKILL.md`. This skill assumes the preconditions are
  satisfied.
- **Debug ladder.** ngauge slots in at the *runtime* layer as a
  single-node telemetry tool. When the cause is below DOCA or spans
  multiple components, escalate to `doca-debug/SKILL.md`.
- **Public-source pointer.** The authoritative source for binary names,
  flags, paths, and the configuration schema is the public DOCA ngauge
  guide at
  https://docs.nvidia.com/doca/archive/3-5-0/doca-ngauge/index.html.
  Confirm details with the installed tool's help and the sample
  configuration files before quoting them.
