# bring-up-cloud-stack adapter

This parent-owned adapter contains no upstream skill implementation. Read the
runtime `SKILL.md` at `skills.bring-up-cloud-stack.skill_md` in the dependency manifest
before invoking it; the upstream owns its behavior and approval gates.

Before cloud startup, export the dedicated runtime `MC_WORKSPACE` and isolated
`COMPOSE_PROJECT_NAME` recorded by the cloud workspace procedure. Upstream
resources supply Compose, images, credentials configuration, and initialization;
the map/fleet skills write run-specific overrides into `MC_WORKSPACE`.
Never pass `--with-sim`: the G1 launch and Isaac Sim own simulation.

```bash
python3 "$SKILL_DIR/references/bring-up-cloud-stack/scripts/run.py" <upstream arguments>
```

The adapter loads `HUMANOID_CLOUD_CONTROL_DEPS_MANIFEST` (or the default state
manifest), requires pinned dependencies, and invokes the `up` entrypoint
from the upstream skill directory. Arguments, environment, stdout/stderr, and
upstream exit status pass through. Use absolute paths for parent-owned inputs.
It exits 3 with `BLOCKED [bring-up-cloud-stack]: ...` if dependencies are missing,
unpinned, or stale. It never searches for another copy or fetches anything.
