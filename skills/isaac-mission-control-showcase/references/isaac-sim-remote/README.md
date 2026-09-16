# Isaac Sim Python-server control

Parent-owned integration adapter. This is **not** `isaac-sim-remote` and does not
document how it works. `isaac-sim-remote` is owned by its own repository; its runtime
`SKILL.md` is the authority for its behavior.

## When to use

After Isaac Sim is running, for every stage operation: scene restore,
stage queries, and readiness probes.

## Showcase inputs and outputs

A script or snippet plus arguments, sent to the local Python server on
`ISAAC_PYTHON_PORT`.

**Produces:** the server's response, passed through unchanged.

## Integration constraints

- Loopback only, on the configured `ISAAC_PYTHON_PORT`. There is no WebRTC,
  browser viewer, or containerised Isaac Sim in this workflow.
- Isaac Sim runs from a local GUI installation; the adapter never launches or
  stops it.

## Who owns the behavior

`isaac-sim-remote`, resolved at runtime from the checkout pinned in
`upstream-versions.lock.json`. Never read a copy of it from inside this
package; none is shipped.

## Finding it

Through the dependency manifest only:

```bash
python3 references/isaac-sim-remote/scripts/run.py <args...>
```

The adapter loads `$MISSION_CONTROL_SHOWCASE_DEPS_MANIFEST`, requires this
component to be `ready`, resolves its `send` entrypoint, and runs it with the
upstream skill directory as the working directory so relative resources
resolve there.

**Read the upstream runtime `SKILL.md` before invoking it.** The manifest
records its path as `skills.isaac-sim-remote.skill_md`. Follow its own gates; do not
assume this file describes them.

## Blockers and return behavior

The adapter exits `3` and prints `BLOCKED [isaac-sim-remote]: …` when the manifest is
missing, stale, or reports this component as not ready, when the entrypoint no
longer exists, or when the upstream `SKILL.md` cannot be read. It never falls
back to discovery, never substitutes another copy, and never fetches. Any other
exit code is the upstream's own, passed through unchanged.

## Retained by the parent

Isaac Sim launch and shutdown, the GUI window, scene restore sequencing,
exact-time TF and scan readiness, and Isaac exit reporting.
