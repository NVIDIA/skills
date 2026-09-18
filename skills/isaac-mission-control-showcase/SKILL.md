---
name: isaac-mission-control-showcase
description: Run and validate an end-to-end Mission Control showcase with a locally installed Isaac Sim launched in its GUI window, driven through the isaac-sim-remote Python server, with Nova Carter SIL. Use for demos, showcase replays, Mission Control driving a simulated robot, or diagnosing the integrated small-warehouse scenario. Detect existing Isaac Sim installations without modifying them, automatically select a usable runtime without prompting whenever compatibility can be confirmed, and delegate requested installation or version changes to isaac-sim-installation. Defaults to a canonical Isaac 6.1 warehouse and deterministic circular route when the user does not specify another scenario.
license: CC-BY-4.0 AND Apache-2.0
metadata:
  author: "NVIDIA Isaac Team <info@nvidia.com>"
  version: 1.0.0
---

# Mission Control Showcase

Use the bundled runner for the default small-warehouse Nova Carter showcase.
It owns the integration and coordinates five upstream skills that it resolves
at runtime rather than vendoring. Resolve those dependencies once with
`python3 "$SKILL_DIR/scripts/doctor.py" dependencies --prepare` before the first run.

## Locate bundled resources

Before running any command, set `SKILL_DIR` to the absolute path of the
directory containing this loaded `SKILL.md`, using the skill location supplied
by the skill loader. Preserve that value in each shell invocation. Do not
derive it from the current working directory or assume a source repository
layout or a fixed installation directory.

All bundled paths in this document and its references, including `scripts/`,
`references/`, `shared/`, `assets/`, `config/`, and
`upstream-versions.lock.json`, are relative to `SKILL_DIR`. Read them beneath
that directory and prefix bundled script paths with `"$SKILL_DIR/"` when
following command examples in the references. Runtime output paths and
upstream skill paths reported by the dependency manifest retain their own
locations. Commands below can run from any working directory.

## Runtime selection

Always launch Isaac Sim from a local installation through this showcase
runner, with its GUI window on the host display. Watch the robot in that
window. Drive every stage operation through the `isaac-sim-remote` Python
server on `ISAAC_PYTHON_PORT`. There is no WebRTC, browser viewer, or
containerised Isaac Sim in this workflow.

Before preflight, inspect existing state read-only. Determine the installation
path and version, whether `isaac-sim.sh` and the bundled ROS 2 bridge
libraries are present, whether a display is
available, and whether an Isaac Sim is already running. Do not launch or
import Isaac Sim, start or stop a container, run an installer, or alter files.

Treat the installed version, warehouse URI, Nova Carter asset and sensor
graphs, and readiness probes as one compatibility set. Mere presence of Isaac
Sim is not proof of compatibility.

Apply this selection policy:

- If a 6.1.0 installation is present at `ISAAC_SIM_DIR`, report the detected
  version and continue without an unnecessary prompt.
- If a different version is detected and the launcher, ROS 2 bridge, and a
  verified warehouse URI are all present, use it automatically — no prompt —
  and report the choice:

  > Isaac Sim `<version>` detected at `<path>` and is usable (launcher
  > `<ready/status>`, ROS 2 bridge `<ready/status>`). Using this installation.

  Attempt the normal preflight and readiness contract; stop and report a
  mismatch rather than modifying that installation.
- Prompt before preflight only when the version cannot be determined, or none
  of launcher, ROS 2 bridge, or a verified warehouse URI can be confirmed —
  this is the sole runtime-selection case that requires user input, because no
  automatic choice can be verified safe:

  > Isaac Sim `<version>` was detected at `<path>`. Launcher:
  > `<ready/status>`. ROS 2 bridge: `<ready/status>`. Choose whether to use
  > the detected version or select or install another version.
- Docker is still required for the Mission Control cloud stack and Nova Carter
  SIL, but never for Isaac Sim itself.
- If nothing is installed, route to the resolved `isaac-sim-installation`
  skill rather than guessing a path. Resolve it through
  `python3 "$SKILL_DIR/scripts/doctor.py" dependencies`;
  never search for it and never vendor a copy.

Never upgrade, downgrade, overwrite, delete, repair, relabel, refresh, or
otherwise modify an existing Isaac Sim installation as showcase recovery.
Obtain explicit user direction before stopping an already-running Isaac Sim.

If the user asks to install, upgrade, downgrade, or otherwise obtain a
different Isaac Sim version, stop the showcase workflow and hand off to the
resolved `isaac-sim-installation` skill. Read its runtime `SKILL.md` at the path
the dependency manifest reports, and follow its own gates without bypassing any
of them. Do not reproduce or restate its installation logic here. The runner
requires a local standalone installation, so preselect that method. Stop after
the installed path is reported; do not launch Isaac Sim. Resume runtime
selection only after that handoff has completed and the user asks to continue.
When that skill runs a compatibility check, prefer a generous
`--timeout-seconds` such as `3600`; its 600-second default often expires on a
first run that is still populating shader and asset caches.

## Default workflow

1. Check whether the user already specified a different map, robot platform,
   simulator, or mission objective in the conversation; if not, use the
   canonical default without asking.
2. Apply [Runtime selection](#runtime-selection) without changing any detected
   installation. Resolve any user choice or installation handoff, then run the
   read-only preflight:

   ```bash
   bash "$SKILL_DIR/scripts/run.sh" --preflight
   ```

3. Ask the user to stop or relocate every reported running container that is
   not explicitly permitted. Allow unrelated workloads by exact container name
   with a repeated `--ignore-container NAME`; ignored containers still undergo
   the normal port-conflict checks. Do not stop an existing Isaac Sim
   automatically. Require a free `ISAAC_PYTHON_PORT`.
4. Start the stack and leave the robot idle:

   ```bash
   bash "$SKILL_DIR/scripts/run.sh"
   ```

5. Or run the deterministic closed route and wait for completion:

   ```bash
   bash "$SKILL_DIR/scripts/run.sh" --demo
   ```

6. Watch the robot in the Isaac Sim window that the runner opens. The runner
   loads the stage before Carter starts, so initial robot motion remains
   visible.
7. Stop only the recorded showcase resources:

   ```bash
   bash "$SKILL_DIR/scripts/run.sh" --stop
   ```

Every run writes `run-manifest.json` into its work directory before anything
starts and updates it at each transition, so an interrupted run can still be
described and stopped precisely. `run-result.json` is the machine-readable
acceptance artifact. Inspect a run, including one that was interrupted, with:

```bash
python3 "$SKILL_DIR/scripts/showcase.py" \
  run-status --work-dir <dir>
```

A stop targets only the containers that run recorded, is idempotent, and
preserves logs and evidence. Motion evidence is persisted when observed, so a
later stationary sample never overrides a confirmed observation. Acceptance
stays strict regardless: `COMPLETED` mission, robot online, healthy and `IDLE`,
motion confirmed, no OOM, required processes healthy.

See `references/troubleshooting.md` for the lifecycle states, recovery procedure, motion
verdicts, Isaac Sim exit handling, host memory policy, and the ROS 2 bridge
executable check.

The runner creates a fresh directory under `${TMPDIR:-/tmp}`, prints its path,
and writes all generated Mission Control configuration, maps, Isaac caches,
logs, and state there. It must not create runtime files in the caller's working
directory. If startup or demo execution fails after the Nova Carter container
is created, the runner saves its complete timestamped ROS 2 launch output to
`$SHOWCASE_WORK_DIR/logs/nova-carter-ros2.log`. It refreshes the same log after
stopping Nova Carter with `--stop`.

## Composition model

Two layers.

**Bundled and parent-owned** — everything in this package: `references/`,
`shared/`, `scripts/`, `assets/`, `config/`. The `references/` tree holds
integration adapters and routing contracts written *for this showcase*. They
are not copies of upstream skills.

**Runtime-resolved** — the five skills the showcase orchestrates. They are owned
by their own repositories, read and invoked from pinned checkouts **outside**
this package, and never copied in.

In the table below, adapter paths are relative to the corresponding reference
directory.

| Reference | Fronts | Adapter |
|---|---|---|
| `references/bring-up-cloud-stack/` | `bring-up-cloud-stack` | `scripts/run.py` |
| `references/change-map/` | `change-map` | `scripts/run.py` |
| `references/change-fleet-composition/` | `change-fleet-composition` | `scripts/run.py` |
| `references/isaac-sim-remote/` | `isaac-sim-remote` | `scripts/run.py` |
| `references/isaac-sim-installation/` | `isaac-sim-installation` | none, by design |

`upstream-versions.lock.json` is the single dependency declaration: the two
public GitHub repositories, their pinned refs and immutable commits, and each
skill's required entrypoints and resources.

Resolve once per host, then run:

```bash
python3 "$SKILL_DIR/scripts/doctor.py" dependencies --prepare \
  --env-file "$HOME/.mission-control-showcase/state/showcase-deps.env"
```

`scripts/doctor.py` runs both phases: `dependencies` resolves the upstream
skills and writes the manifest, `host` checks this machine. The dependency
phase must succeed first, because the host phase is handed the entrypoints it
resolved.

Read `references/workflow.md` for the stage router, and each reference's
`README.md` before using that stage.

## Using an upstream skill

The reference tells you where the upstream is; the upstream tells you how it
behaves.

1. Read its **runtime** `SKILL.md`, at the path the manifest records as
   `skills.<name>.skill_md`. Nothing in this package restates it.
2. Follow its own gates, including confirmation, licensing, and
   execution-approval gates.
3. Invoke it through the reference adapter, which runs it from its own skill
   directory so relative resources resolve there.
4. Never reimplement it, never vendor it, and never substitute another copy
   found on the machine.

Do not invoke `bring-up-cloud-stack` with `--with-sim`; that starts the generic
`mission-simulator`, not Nova Carter SIL. The showcase owns Nova SIL, Isaac Sim
launch and shutdown, scene restore, cross-domain readiness, mission submission,
and completion evidence.

## Missing dependencies

The runner performs a read-only freshness check and the adapters consume the
persisted manifest. Normal execution never clones, fetches, pulls, installs, or
updates anything.

With `MISSION_CONTROL_SHOWCASE_REQUIRE_PREFLIGHT=1`, a reference that cannot
load a ready manifest blocks rather than falling back to discovery. An adapter
exits `3` with `BLOCKED [<component>]: …` naming the skill, what was expected,
and the preflight command. Never work around a blocker by vendoring a skill.

## Canonical default

The default is:

- Isaac Sim: a local installation at `ISAAC_SIM_DIR` (default `~/isaacsim`),
  launched with its GUI window. Isaac Sim 6.1.0 is the last-known-good runtime.
  This is a default, not a compatibility ceiling.
- Warehouse:
  `https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1/Isaac/Environments/Simple_Warehouse/warehouse.usd`.
- Nova Carter SIL:
  `nvcr.io/nvidia/isaac/nova_carter_sil:release-3.2`.
- NVIDIA driver R590 or newer.
- A 64 GB-class host with at least 48 GB available before startup and 4 GB
  free swap.
- Nova Carter SIL is limited to 20 GB RAM plus up to 4 GB swap.
- Robot identity: `carter01`.
- Isaac Sim Python server on port `8226`.
- Websocket MQTT on port `9001`, matching the packaged Mission Control stack.
- ROS domain `77` for both Isaac Sim and Carter SIL, with normal host
  transport. The isolated domain prevents unrelated domain-0 DDS participants
  from exhausting Carter SIL memory while preserving Isaac-to-SIL discovery.

The exact warehouse URI is passed into Isaac Sim and verified after stage
load. Never discover the warehouse through `get_assets_root_path`, a local
cache, a source checkout, or a filesystem search.

The runner configures Mission Control and starts the cloud stack, then
launches Isaac Sim and restores the stage while the cloud initializes. Carter
SIL starts last, so the Isaac Sim window shows the robot's initial motion.

For a user-selected noncanonical Isaac version that is already installed:

1. Require a compatible installation and a canonical asset URI.
2. Pass both explicitly:

   ```bash
   ISAAC_SIM_DIR=<resolved-install-path> \
   WAREHOUSE_USD_URI=<resolved-uri> \
     bash "$SKILL_DIR/scripts/run.sh"
   ```

3. Do not infer a future URI by substituting a version number. A local
   installation carries no inspectable version tag, so the caller owns the
   asset contract.
4. If the requested version is not already installed, stop and hand off to
   `isaac-sim-installation`; do not install it inside this workflow.

## Robot identity

`carter01` is only the default. Set another identity with either:

```bash
bash "$SKILL_DIR/scripts/run.sh" \
  --robot-name my_robot --demo
```

or `ROBOT_NAME=my_robot`.

The runner applies the same value to Mission Control fleet configuration,
Mission Client `serial_number`, MQTT bridge client names, readiness polling,
and mission submission. Do not change only the mission payload.

## Deterministic route

The default replay is a closed route:

```json
[
  {"x": -5.1, "y": 1.4},
  {"x": -2.625, "y": 1.2},
  {"x": -0.2, "y": -2.075},
  {"x": -2.625, "y": -5.35},
  {"x": -5.1, "y": 1.4}
]
```

Submit it as a route-only mission with timeout `900` and solver
`NVIDIA_CUOPT`. Do not add start/end locations or iterations.

For custom coordinates, first bring up the stack without `--demo`, snap the
requested points through WPG `nearest_nodes`, validate with
`visualize_route`, and submit only the resulting routable points.

## Readiness contract

The runner must not submit a mission until all of these are observed:

- the canonical stage is active and the Isaac timeline is playing;
- ROS `/clock` advances;
- a live `/scan` `LaserScan` uses frame `front_2d_lidar` and has a timestamp
  within one second of simulation time;
- TF resolves the LiDAR frame into `map` at that exact sensor timestamp;
- `map -> base_link` resolves;
- the robot pose lies inside the active global costmap;
- Nav2 planner, controller, and navigator lifecycle nodes are active;
- Carter SIL has recorded no cgroup OOM event and its required map,
  localization, planning, control, navigation, and mission-client processes
  remain alive;
- Mission Control reports the configured robot online, `IDLE`, error-free,
  position-initialized, and near the reset pose.

Treat an initial exact-time LiDAR TF miss as a bounded startup condition.
Continue consuming newer scan samples until one resolves into `map`, or fail
at the readiness deadline with the latest measured timestamp. Never replace
this with a latest-TF lookup.

On failure, report the failed observation and measured values. Do not claim a
clock, TF, costmap, or permission root cause without those measurements.

The bundled `carter_warehouse_navigation.png` and companion YAML are the
single map resource. The runtime copies that pair into Mission Control's
generated config and Carter SIL's read-only `/maps` mount. Mission Control
uses the same PNG and metadata to generate the WPG resource under
`carter_warehouse_navigation.png`. The runner verifies the source, Mission
Control, and Carter hashes before startup, then verifies the active WPG map
identity, Carter launch option, and container-mounted bytes. Never recursively change permissions on a
source tree or apply a permission fix after services start.

## Modes and overrides

- `--preflight`: read-only installation, display, Docker, GPU, dependency,
  port, image, and asset checks.
- no flag: clean startup and full readiness validation, leaving the robot
  idle.
- `--demo` / `--replay`: startup, readiness, circular mission, canonical
  mission completion, and pose-motion evidence.
- `--stop`: stop the last recorded stack and SIGTERM the Isaac Sim process,
  without deleting containers, images, volumes, caches, or logs.
- `--dry-run`: print resolved commands without changing runtime state.
- `--work-dir PATH`: use a caller-selected new or empty runtime directory.
- `--no-pull`: require required images to already be local.
- `--ignore-container NAME`: allow one already-running container by exact
  name during preflight and startup; repeat the option for each allowed
  container. Port conflicts remain failures.

Supported environment overrides are `ISAAC_SIM_DIR`, `WAREHOUSE_USD_URI`,
`NOVA_CARTER_IMAGE`, `NOVA_CARTER_MEMORY_LIMIT`,
`NOVA_CARTER_MEMORY_SWAP_LIMIT`, `ROBOT_NAME`, `SHOWCASE_GPU_DEVICE`,
`SHOWCASE_ROS_DOMAIN_ID`, `SHOWCASE_ROS_LOCALHOST_ONLY`,
`ISAAC_PYTHON_PORT`, and `SHOWCASE_WORK_DIR`. Keep the default isolated domain
unless it conflicts with another local ROS deployment. Apply the same domain
and localhost setting to Isaac Sim and Carter SIL. Do not enable localhost-only
transport without verifying cross-process discovery between the host Isaac Sim
bridge and host-network Carter SIL.

## Acceptance

A successful `--demo` requires Mission Database to report the submitted
mission `COMPLETED`, the same robot to return to `IDLE` without errors, and
Mission Database poses to prove motion. Robot state, logs, a visible Isaac Sim
window, or a single screenshot alone are not completion.

Do not run `docker compose down -v`, remove containers, prune Docker state, or
delete the run directory as automatic recovery.
