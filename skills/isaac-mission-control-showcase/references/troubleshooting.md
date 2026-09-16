# Troubleshooting

- **Preflight reports running containers:** stop or relocate blocking
  containers, or enumerate unrelated workloads with repeated
  `--ignore-container NAME` options. Names match exactly; ignored containers
  do not bypass port-conflict checks, and the showcase never stops them.
- **Packaged dependency is missing:** confirm every dependency is installed
  either directly under `skills/<skill-name>` (npm layout) or below its
  collection at `skills/<collection>/<skill-name>` (repository layout). Do not
  clone source as a fallback.
- **Registry pull fails:** authenticate with the registry named by the selected
  image, then rerun. The runner pulls all runtime images in parallel and
  reports every failed image before starting services. `--no-pull` is valid
  only when every required image is already local.
- **Driver preflight fails:** install and load an NVIDIA R590-or-newer driver,
  then verify `nvidia-smi` before retrying. A newly installed driver may
  require a host reboot before its kernel module is active.
- **Memory preflight fails:** use a 64 GB-class host with at least 48 GB
  available RAM and 4 GB free swap. Carter receives a 20 GB memory limit and
  a 24 GB combined memory-plus-swap limit.
- **Warehouse URI fails:** keep the canonical 6.0 URI for the default. For a
  custom Isaac installation, resolve and provide a compatible canonical URI;
  never substitute a local cache.
- **Isaac Sim will not launch:** confirm `ISAAC_SIM_DIR/isaac-sim.sh` exists
  and is executable, and that `DISPLAY` or `WAYLAND_DISPLAY` is set. The GUI
  launch needs a real display; over SSH use an X or VNC session. Read the log
  path printed as `log=` and the pid recorded in `isaac-sim.pid`.
- **Map is unreadable:** inspect the two paths beneath the printed
  `showcase_work_dir`. Their directories should be `0755` and map files
  `0644`. Do not change permissions on the skill or a source checkout.
- **Robot identity mismatch:** confirm the same `--robot-name` appears in the
  generated defaults, Nova `serial_number`, MQTT topic, robot endpoint, and
  mission query.
- **Clock or LiDAR gate fails:** require `/scan` with frame
  `front_2d_lidar`, then use the measured clock/scan delta and failed
  timestamped transform from the runtime consistency probe. Do not infer a
  clock mismatch from a Nav2 rejection alone.
- **Robot outside costmap:** inspect the reported robot pose, costmap origin,
  dimensions, resolution, and calculated cell. Reset the Isaac scene before
  changing route coordinates.
- **Nav2 never activates:** each ROS CLI call is limited to five seconds and a
  full attempt to 30 seconds. Treat a probe that exceeds that bound as a
  failure; otherwise inspect the recorded Nova container logs and verify
  `/clock`, `/chassis/odom`, `map -> base_link`, and the runtime map YAML.
- **Mission returns to IDLE without completion:** query Mission Database by the
  printed mission ID. Only its `COMPLETED` state is canonical.

The runner records its state beneath the printed temporary directory.
`--stop` leaves those artifacts and stopped containers available for
diagnosis.

# Run lifecycle and recovery

Every run writes `run-manifest.json` into its work directory before any
container or Isaac Sim process starts, and updates it atomically at each
transition: `initializing`, `preflight`, `starting-cloud`, `starting-isaac`,
`starting-carter`, `ready`, `mission-running`, `accepted`, `failed`, `stopped`.

The manifest records the work directory, lifecycle state, Compose project, the
exact container names this run created, the Isaac Sim PID and any observed exit
status, selected ports and ROS domain, map and robot configuration, mission ID
once known, and evidence paths. `run-result.json` is the machine-readable
acceptance artifact written at the end of a run.

## Interrupted runs

`SIGINT` and `SIGTERM` are handled. The manifest is marked `failed` unless the
run had already recorded a verdict, since an interrupt during cleanup must not
overwrite an `accepted` result. The work directory is preserved, and the runner
prints the exact targeted stop command. Nothing is killed on the way out.

Recover with:

```bash
scripts/showcase.py run-status --work-dir <dir>     # what this run owns
scripts/run.sh --stop --work-dir <dir>              # stop only that run
```

A stop only touches containers recorded in that run's manifest, so a concurrent
run or an unrelated workload is never affected. Stopping is idempotent, and logs
and result evidence are preserved afterwards.

Resume is deliberately not offered for a partially started stack: the safe path
is a targeted stop followed by a fresh run. `run-status` reports exactly what
remains so that decision is informed rather than guessed.

## Motion evidence

Motion is recorded when observed, not recomputed later. A stationary sample
taken after a mission ends is not evidence the robot never moved, so
`run-status` reports one of four verdicts: `confirmed-historical`,
`confirmed-sample`, `none-observed` (a sample ran and saw nothing, with no
earlier evidence), or `no-evidence` (nothing has ever been observed).

Acceptance remains strict and is never weakened by a missing sample: the mission
must reach `COMPLETED`, the robot must be online, healthy and `IDLE`, motion
must be confirmed, no OOM may have occurred, and required Nova Carter and Nav2
processes must be healthy.

## Isaac Sim lifecycle

Isaac Sim and the stack stay alive after acceptance until an explicit targeted
stop, so GUI evidence is captured while Isaac Sim is known to be running. An
unexpected exit is recorded in the manifest with its status and reason; an
absent exit record means the exit was never observed, which is reported as such
rather than assumed to be healthy.

## Host memory policy

Swap is a fallback for scarce RAM, not a floor. A host whose available memory
already meets the requirement passes with zero swap. When available memory is
short and swap covers the gap, preflight warns that paging is likely. Only when
available memory plus swap is below the requirement does preflight fail. Swap is
never created, enabled, resized, or removed without explicit authorization.

## ROS 2 bridge executables

Preflight checks that every helper under
`<isaac>/exts/isaacsim.ros2.core/bin/` is executable. An archive that loses a
Unix executable bit installs a helper as `0664`; Isaac Sim then starts and ROS
startup dies with `PermissionError`. Preflight names the exact path and the
narrowest repair, `chmod +x <path>`. Apply that exact repair automatically,
without prompting: it is scoped to the single named helper, additive
(`+x` only, no other mode bits touched), and reported. Never broaden it to a
recursive or directory-wide permission change, and never apply it to any path
other than the one preflight named.
