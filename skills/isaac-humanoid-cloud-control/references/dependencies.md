# Upstream dependency workflow

The dependency resolver and reference adapters use the same composition pattern
as isaac-mission-control-showcase. The lock file declares the same immutable
upstream commits; these are skill-script pins, not the Isaac Sim runtime version
or the separate official viewer source tag. Keep the release-4.6/Isaac Sim 6.0
runtime and viewer requirements in the main workflow.

## Resolve and prepare

Set `SKILL_DIR` from the loaded skill location. Run from any working directory:

```bash
python3 "$SKILL_DIR/scripts/doctor.py" dependencies
```

This verifies pinned commits, clean tracked files, runtime `SKILL.md` files,
entrypoints, and declared resources. It writes the dependency manifest at
`~/.humanoid-cloud-control/state/humanoid-deps.json`. Use `--check-only` for a
read-only inspection without writing state, or `--json` for structured output.
Exit codes are 0 for ready, 3 for blocked, and 2 for configuration/preparation
failure. Host readiness remains owned by `scripts/preflight.py`.

Resolution precedence is per-skill directory override (`path_env` in the lock),
per-upstream checkout override (`root_env`), then the managed checkout root.
There is no filesystem search or fallback to an installed copy. Overrides must
also match the pins. Relative overrides are normalized to absolute paths.

If dependencies are missing, report the exact blocker. After the user authorizes
preparing dependencies, run:

```bash
python3 "$SKILL_DIR/scripts/doctor.py" dependencies --prepare
```

Only this mode may fetch/create managed checkouts. It never updates an upstream
root supplied by environment override, and refuses dirty managed checkouts or
unexpected origins. Do not vendor upstream skill implementations into this skill.
Normal adapters never clone, fetch, install, or update dependencies.

Configuration:

- `HUMANOID_CLOUD_CONTROL_HOME`: state/cache home (default above).
- `HUMANOID_CLOUD_CONTROL_UPSTREAM_ROOT`: managed checkout root (default
  `~/.humanoid-cloud-control/upstreams`).
- `HUMANOID_CLOUD_CONTROL_DEPS_MANIFEST`: explicit manifest path. When using
  `--manifest`, export this same path for adapters, or generate `--env-file`
  and source it in subsequent shells.
- Per-upstream and per-skill override names are declared in
  `upstream-versions.lock.json`.

`--allow-commit-drift` is diagnostic only: adapters always refuse its manifest.
Re-resolve without that flag before execution.

## Use the owning skill

Read the runtime `SKILL.md` at `skills.<name>.skill_md` in the manifest and follow
its own confirmation, licensing, and execution gates. Then read the relevant
adapter contract and invoke it with the upstream's documented arguments:

| Skill | Contract | Invocation |
| --- | --- | --- |
| bring-up-cloud-stack | [Cloud adapter](bring-up-cloud-stack/README.md) | `python3 "$SKILL_DIR/references/bring-up-cloud-stack/scripts/run.py" <args>` |
| change-map | [Map adapter](change-map/README.md) | `python3 "$SKILL_DIR/references/change-map/scripts/run.py" <args>` |
| change-fleet-composition | [Fleet adapter](change-fleet-composition/README.md) | `python3 "$SKILL_DIR/references/change-fleet-composition/scripts/run.py" <args>` |
| isaac-sim-remote | [Scene adapter](isaac-sim-remote/README.md) | `python3 "$SKILL_DIR/references/isaac-sim-remote/scripts/run.py" <args>` |
| isaac-ros-activate | [Activation handoff and mode gate](isaac-ros-activate/README.md) | `python3 "$SKILL_DIR/references/isaac-ros-activate/scripts/run.py" --expected docker`; agent manages activation PTY. |
| submit-and-monitor-mission | [Verification handoff](submit-and-monitor-mission/README.md) | Agent reads the runtime skill and verifies the existing mission; no submission adapter. |
| isaac-sim-installation | [Installation handoff](isaac-sim-installation/README.md) | Agent reads the runtime skill; no automatic script adapter. |

Adapters load the persisted manifest, require verified dependencies, and run the
entrypoint from the owning skill's directory. Arguments and environment pass
through unchanged, as does the upstream exit code. Supply absolute paths for
parent-owned files, including scene inventory and `MC_WORKSPACE`. Missing or
stale dependencies produce `BLOCKED [<skill>]: ...` and exit 3 before execution.

The other instruction-only handoffs (`isaac-sim-headless-deployment` and
`occupancy-map`)
continue through their loaded skills. They have no cross-skill script invocation
defined by this package and are not entries in this script dependency manifest.
If unavailable, report them rather than substituting an implementation.

## Owning-skill responsibilities

Use the owning skills instead of reproducing their implementation:

- Use isaac-sim-installation only when Isaac Sim is absent or the user asks to
  install or change it. The standalone runtime is the supported host launch
  used by this workflow, but its 6.0.1 archive does not contain
  `tools/docker/web-viewer`. Retain the installation skill's confirmation and
  EULA gates, then obtain separate approval before checking out the official
  IsaacSim source at the exact runtime tag for the viewer build context. The
  installation skill stops before launch.
- Use isaac-sim-headless-deployment for headless launch conventions. Retain
  rendering because the browser livestream requires it.
- Use occupancy-map when the selected USD lacks a matching Nav2 map.
- Use change-map and change-fleet-composition to preview and apply Mission
  Control configuration. Do not hand-edit defaults.yaml.
- Use bring-up-cloud-stack for normal Mission Control startup. Do not enable
  its generic robot simulator profile; Isaac Sim and the G1 launch own
  simulation here.
- Read isaac-ros-activate from `skills.isaac-ros-activate.skill_md` in the
  manifest; use its mode-gate adapter, then enter and retain the Docker-mode
  Isaac ROS shell. Run all ros2, rosdep, and colcon commands in that shell.
- Use isaac-sim-remote for the localhost Python-server transport and live-stage
  execution contract. Use this skill's scene inventory and suggestion scripts
  for the humanoid-specific object catalog and navigation approach poses.
- Read submit-and-monitor-mission from the manifest after mission submission.
  Verify the existing mission ID; do not run its submission helper or create
  another mission. Adapt its runtime-log check to the active G1 launch instead
  of assuming Carter SIL.

If an owning skill or script is missing, stop and report the dependency. Do
not clone or reconstruct it silently. Only the explicit dependency preparation
phase may acquire the pinned upstream checkouts.

