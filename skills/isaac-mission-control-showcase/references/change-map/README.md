# Run-local map configuration

Parent-owned integration adapter. This is **not** `change-map` and does not
document how it works. `change-map` is owned by its own repository; its runtime
`SKILL.md` is the authority for its behavior.

## When to use

After the workspace exists and before the cloud stack starts, to point
Mission Control at the showcase map.

## Showcase inputs and outputs

The bundled `carter_warehouse_navigation` PNG and its Nav2 YAML from this
package's `assets/`, applied into the run's workspace.

**Produces:** an edited `defaults.yaml` in the run workspace.

## Integration constraints

- The showcase verifies map hashes across source, Mission Control, and the
  Carter mount. The adapter does not weaken that check.
- The map identity must stay consistent with the warehouse USD and the route.

## Who owns the behavior

`change-map`, resolved at runtime from the checkout pinned in
`upstream-versions.lock.json`. Never read a copy of it from inside this
package; none is shipped.

## Finding it

Through the dependency manifest only:

```bash
python3 references/change-map/scripts/run.py <args...>
```

The adapter loads `$MISSION_CONTROL_SHOWCASE_DEPS_MANIFEST`, requires this
component to be `ready`, resolves its `set_map` entrypoint, and runs it with the
upstream skill directory as the working directory so relative resources
resolve there.

**Read the upstream runtime `SKILL.md` before invoking it.** The manifest
records its path as `skills.change-map.skill_md`. Follow its own gates; do not
assume this file describes them.

## Blockers and return behavior

The adapter exits `3` and prints `BLOCKED [change-map]: …` when the manifest is
missing, stale, or reports this component as not ready, when the entrypoint no
longer exists, or when the upstream `SKILL.md` cannot be read. It never falls
back to discovery, never substitutes another copy, and never fetches. Any other
exit code is the upstream's own, passed through unchanged.

## Retained by the parent

Map hash verification, the WPG resource check, the Carter `/maps` mount, and
warehouse/route consistency.
