# Isaac ROS activation handoff and mode-gate adapter

Read the upstream runtime `SKILL.md` at
`skills.isaac-ros-activate.skill_md` in the dependency manifest. Follow its
Docker-only mode gate and long-lived host PTY contract. Do not search the host
for an installed copy or require registration in the agent's skill catalog.

`doctor.py dependencies --prepare` obtains the owning public Isaac ROS CLI
repository at the immutable commit in `upstream-versions.lock.json`. The skill
exists on the pinned `main` commit, not on release-4.6. This checkout supplies
skill instructions and the mode-check helper only; it does not install or
upgrade the CLI, change its package release, or select its container image.
CLI installation remains on the release-4.6 bootstrap path.

Resolve the upstream skill's repo-relative mode-check example through this
adapter instead of copying that path into the caller's current directory:

```bash
python3 "$SKILL_DIR/references/isaac-ros-activate/scripts/run.py" --expected docker
```

The adapter runs the manifest's `check_mode` entrypoint from the owning skill
directory. It forwards arguments and exit status, never activates a container,
and exits 3 with `BLOCKED [isaac-ros-activate]: ...` for unavailable or stale
dependencies. Normal invocation does not fetch upstreams.

After the gate succeeds, the agent runs `isaac-ros activate` (or
`isaac-ros activate --build-local` for newly configured image layers) in a
long-lived host PTY, using the existing CLI setup reference. Use the chosen G1
workspace as `ISAAC_ROS_WS`, not the dependency checkout. Send later ROS commands
to the same activated shell and verify its activation state using the upstream
skill's documented status command or fallback.

If the CLI is missing or uninitialized, complete the parent skill's CLI/Docker
bootstrap and rerun the gate. Do not bypass an existing venv/baremetal mode.
