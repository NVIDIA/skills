# isaac-sim-remote adapter

This parent-owned adapter contains no upstream skill implementation. Read the
runtime `SKILL.md` at `skills.isaac-sim-remote.skill_md` in the dependency manifest
before invoking it; the upstream owns its behavior and approval gates.

Use only after the guarded unsupported-balance and readiness gates, to send
this skill's `scripts/scene_inventory.py` to `127.0.0.1:8226`. Pass its absolute
path with `--file`. Isaac Sim remains headless with rendering and the browser
stream enabled. The adapter does not launch Isaac Sim or release the gantry.

```bash
python3 "$SKILL_DIR/references/isaac-sim-remote/scripts/run.py" <upstream arguments>
```

The adapter loads `HUMANOID_CLOUD_CONTROL_DEPS_MANIFEST` (or the default state
manifest), requires pinned dependencies, and invokes the `send` entrypoint
from the upstream skill directory. Arguments, environment, stdout/stderr, and
upstream exit status pass through. Use absolute paths for parent-owned inputs.
It exits 3 with `BLOCKED [isaac-sim-remote]: ...` if dependencies are missing,
unpinned, or stale. It never searches for another copy or fetches anything.
