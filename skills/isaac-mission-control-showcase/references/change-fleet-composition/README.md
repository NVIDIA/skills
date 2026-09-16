# Run-local robot identity

Parent-owned integration adapter. This is **not** `change-fleet-composition` and does not
document how it works. `change-fleet-composition` is owned by its own repository; its runtime
`SKILL.md` is the authority for its behavior.

## When to use

Before the cloud stack starts, to register the showcase robot identity.

## Showcase inputs and outputs

The resolved `ROBOT_NAME`, submitted as a complete fleet with `sim` false
so the generic simulator stays disabled.

**Produces:** an edited `robots:` block in the run workspace.

## Integration constraints

- The fleet is declarative: the parent submits the whole desired fleet, never
  a delta.
- `sim: false` keeps the generic simulator out of the run; Nova Carter SIL is
  the only robot source.

## Who owns the behavior

`change-fleet-composition`, resolved at runtime from the checkout pinned in
`upstream-versions.lock.json`. Never read a copy of it from inside this
package; none is shipped.

## Finding it

Through the dependency manifest only:

```bash
python3 references/change-fleet-composition/scripts/run.py <args...>
```

The adapter loads `$MISSION_CONTROL_SHOWCASE_DEPS_MANIFEST`, requires this
component to be `ready`, resolves its `set_fleet` entrypoint, and runs it with the
upstream skill directory as the working directory so relative resources
resolve there.

**Read the upstream runtime `SKILL.md` before invoking it.** The manifest
records its path as `skills.change-fleet-composition.skill_md`. Follow its own gates; do not
assume this file describes them.

## Blockers and return behavior

The adapter exits `3` and prints `BLOCKED [change-fleet-composition]: …` when the manifest is
missing, stale, or reports this component as not ready, when the entrypoint no
longer exists, or when the upstream `SKILL.md` cannot be read. It never falls
back to discovery, never substitutes another copy, and never fetches. Any other
exit code is the upstream's own, passed through unchanged.

## Retained by the parent

Robot identity propagation across Mission Control, the Mission Client, MQTT
bridge names, readiness polling, and mission submission.
