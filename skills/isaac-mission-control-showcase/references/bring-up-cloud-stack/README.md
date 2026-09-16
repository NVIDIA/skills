# Cloud stack bring-up

Parent-owned integration adapter. This is **not** `bring-up-cloud-stack` and does not
document how it works. `bring-up-cloud-stack` is owned by its own repository; its runtime
`SKILL.md` is the authority for its behavior.

## When to use

Before Isaac Sim launch, to start the Mission Control cloud services the
showcase drives.

## Showcase inputs and outputs

Uses the run's `MC_WORKSPACE` and the generated configuration the parent
wrote there.

**Produces:** a running cloud stack under the run's Compose project.

## Integration constraints

- Never pass `--with-sim`. It starts the generic `mission-simulator`, not
  Nova Carter SIL, and the showcase owns the robot.
- The Compose project name comes from the parent run manifest so a stop targets
  only this run's containers.

## Who owns the behavior

`bring-up-cloud-stack`, resolved at runtime from the checkout pinned in
`upstream-versions.lock.json`. Never read a copy of it from inside this
package; none is shipped.

## Finding it

Through the dependency manifest only:

```bash
python3 references/bring-up-cloud-stack/scripts/run.py <args...>
```

The adapter loads `$MISSION_CONTROL_SHOWCASE_DEPS_MANIFEST`, requires this
component to be `ready`, resolves its `up` entrypoint, and runs it with the
upstream skill directory as the working directory so relative resources
resolve there.

**Read the upstream runtime `SKILL.md` before invoking it.** The manifest
records its path as `skills.bring-up-cloud-stack.skill_md`. Follow its own gates; do not
assume this file describes them.

## Blockers and return behavior

The adapter exits `3` and prints `BLOCKED [bring-up-cloud-stack]: …` when the manifest is
missing, stale, or reports this component as not ready, when the entrypoint no
longer exists, or when the upstream `SKILL.md` cannot be read. It never falls
back to discovery, never substitutes another copy, and never fetches. Any other
exit code is the upstream's own, passed through unchanged.

## Retained by the parent

Container-conflict checks, port selection, the Compose project name, the run
manifest, readiness gating, and targeted stop.
