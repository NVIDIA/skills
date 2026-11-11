# change-fleet-composition adapter

This parent-owned adapter contains no upstream skill implementation. Read the
runtime `SKILL.md` at `skills.change-fleet-composition.skill_md` in the dependency manifest
before invoking it; the upstream owns its behavior and approval gates.

For requested fleet changes, preview and apply through the owning skill's gates.
Set absolute `MC_WORKSPACE` to the dedicated runtime workspace. Keep one identity across
Mission Control, the G1 launch, MQTT, mission payloads, and verification.
Do not enable the generic robot simulator.

```bash
python3 "$SKILL_DIR/references/change-fleet-composition/scripts/run.py" <upstream arguments>
```

The adapter loads `HUMANOID_CLOUD_CONTROL_DEPS_MANIFEST` (or the default state
manifest), requires pinned dependencies, and invokes the `set_fleet` entrypoint
from the upstream skill directory. Arguments, environment, stdout/stderr, and
upstream exit status pass through. Use absolute paths for parent-owned inputs.
It exits 3 with `BLOCKED [change-fleet-composition]: ...` if dependencies are missing,
unpinned, or stale. It never searches for another copy or fetches anything.
