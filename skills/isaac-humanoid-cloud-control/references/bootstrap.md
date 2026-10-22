# Fresh-machine bootstrap gate

## Simulation release authorization

This workflow controls only a virtual gantry inside Isaac Sim, never physical
robot hardware. Obtain explicit operator approval for the initial virtual gantry
release before invoking the helper. A broad demo request does not grant this
approval. If the attempt fails and the gantry is confirmed re-engaged, report
that result and obtain separate approval for at most one bounded retry. Keep the
gantry engaged while awaiting approval; never attempt a third release. The
helper must pass its live simulation identity interlock before any control
command; a simulation flag or the presence of a clock topic is insufficient.

Use this reference only when the requested demo must start from a fresh or
unverified host.

This browser-streamed workflow uses the Isaac Sim Standalone runtime, but the
public 6.0.1 runtime archive does not include the Git repository's
`tools/docker/web-viewer` build context. Retain the installation skill's method,
plan, and EULA approval gates. Treat a matching official source checkout as a
separate dependency and require installation authorization before acquiring it.

Begin with a read-only inventory. Let `preflight.py` discover conventional local
Isaac Sim, Isaac ROS, and official web-viewer installations before asking the
operator for paths. Use the bundled map asset so the first check is actionable:

```bash
SKILL_DIR=/absolute/path/to/isaac-humanoid-cloud-control
python3 "$SKILL_DIR/scripts/preflight.py" \
  --map "$SKILL_DIR/assets/carter_warehouse_navigation.yaml"
```

After discovery passes, obtain reusable export lines:

```bash
python3 "$SKILL_DIR/scripts/preflight.py" \
  --map "$SKILL_DIR/assets/carter_warehouse_navigation.yaml" \
  --emit-discovered-env
```

Use explicit `ISAAC_PATH`, `ISAAC_ROS_WS`, `WEB_VIEWER_CONTEXT`, or matching CLI
arguments only to override discovery. Use `--no-discover` only to diagnose a
specific requested path. If discovery fails, report the missing component; do
not ask the operator to locate a path the host can discover.

Resolve the upstream skill manifest first using the dependency workflow linked
from `SKILL.md`; read each owning runtime skill before invoking its adapter.

Resolve failures in this order and stop at each ownership or approval gate:

1. If Isaac Sim is absent, use `isaac-sim-installation` with Standalone binary
   preselected for the host streaming runtime. Require confirmation, EULA
   acknowledgement, and explicit approval before a download or install. Stop
   installation after warmup, then return here. On Linux require
   `exts/isaacsim.ros2.core/bin/isaacsim.ros2.core.check` to be executable; the
   6.0.1 archive can preserve mode `0664`, which makes ROS bridge startup fail.
2. If `isaac-ros` is absent, follow the CLI installation and Docker setup
   reference linked from `SKILL.md`. Treat this as a recoverable prerequisite:
   install the public release-4.6 CLI on a supported host, verify it, and resume.
   Do not stop merely to tell the operator to provision it manually. Honor an
   inspection-only/no-host-changes request and report actual installation errors.
3. Follow the same setup reference to initialize Docker mode when unconfigured,
   preserve existing workspace settings, add missing `realsense` and
   `gr00t_workflow` image keys, and activate with `--build-local` when needed.
   Keep simulation navigation-only and build the source-only Deploy extension
   through the release runbook. If a CUDA/package conflict remains, report its
   exact evidence; do not patch the CLI or substitute image keys to hide it.
4. If the official `IsaacSim/tools/docker/web-viewer` source is absent and the
   user authorized mission-package/app installation, shallow-clone the official
   repository at the exact runtime tag (`v6.0.1` here) with Git LFS smudging
   disabled. Verify the tag/commit and Dockerfile before building. Otherwise
   request an existing path; never use `main`, an unverified mirror, or a local
   GUI fallback.
5. Budget fresh-node time and disk for the full release stack. In this run the
   G1 cloud-control apt package resolved 488 packages, and the cloud pull
   included a roughly 4.2 GB cuOpt layer. A long first build or pull is not a
   readiness failure; require the semantic package, image, and health checks.
6. If cloud images or NGC credentials are absent, ask the operator to provision
   them before passing `--pull` to the bring-up-cloud-stack adapter. Never place credentials in skill
   files, command arguments, logs, or Compose configuration.

Continue to the ordered launch only after preflight has no blocking failures.

Fresh-node release-4.6 observations to classify rather than hide:

- The optional Foxglove bridge can repeatedly log `bad optional access` while
  the Isaac browser WebRTC viewer, ROS graph, and navigation remain healthy.
  Report the telemetry bridge as degraded; do not use that noise to override
  the independent runtime and browser-viewer gates.
- `initial_noise_publisher_node` can exit for `agile_velocity.yaml` when the
  graph has no `initial_noise` input. Require inference warmup, controller
  activation, fresh sensors, and guarded unsupported balance before deciding
  whether the selected controller group is usable.
- Mission Control may expand a single requested target into multiple WPG/cuOpt
  graph waypoints. Always monitor the stored mission tree and allow bounded
  final convergence while Nav2 status is executing, pose changes, physics is
  active, and the robot remains error-free.
