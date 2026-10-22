# change-map adapter

This parent-owned adapter contains no upstream skill implementation. Read the
runtime `SKILL.md` at `skills.change-map.skill_md` in the dependency manifest
before invoking it; the upstream owns its behavior and approval gates.

For requested map changes, preview and apply only through the owning skill's
gates. Set absolute `MC_WORKSPACE` to the dedicated runtime workspace. Keep Mission Control
and the activated G1 launch on the same map PNG/YAML pair.

```bash
python3 "$SKILL_DIR/references/change-map/scripts/run.py" <upstream arguments>
```

The adapter loads `HUMANOID_CLOUD_CONTROL_DEPS_MANIFEST` (or the default state
manifest), requires pinned dependencies, and invokes the `set_map` entrypoint
from the upstream skill directory. Arguments, environment, stdout/stderr, and
upstream exit status pass through. Use absolute paths for parent-owned inputs.
It exits 3 with `BLOCKED [change-map]: ...` if dependencies are missing,
unpinned, or stale. It never searches for another copy or fetches anything.
