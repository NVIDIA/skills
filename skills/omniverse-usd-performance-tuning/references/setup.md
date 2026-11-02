# Runtime and output workspace

Use a verified standalone runtime for USD operations. Kit may supply remote access or
render profiling, but is not the operation executor for this workflow.

## Reuse or verify

Read `<output_path>/setup-preflight.json` when present. Reuse the recorded environment
when the interpreter, package identity, dependencies and required capabilities still match.
State the selected runtime once; do not present a runtime chooser or request confirmation
again just because a preflight file exists. Re-probe changed or missing capabilities.

For a new preflight, record:

- Interpreter and OpenUSD versions; optimizer package root, version evidence and fingerprint.
- Package sentinels: `python`, `usdpy`, `lib`, `extraLibs`, and the operation-doc layout
  resolved by [upstream guidance](upstreams/usd-optimize.md).
- Actual shared-library/import success and `UsdOptimizeCore.getInstance().getOperations()`.
  A Python executable or a nonempty install directory is insufficient evidence.
- Validator imports and applicable rule registration when validation is required.
- NumPy import/version evidence for composition checks. SciPy is needed only when running
  development tests that independently compare geometry; it is not a prerequisite for
  supported native optimization.
- Available operations in `usdOptimize.operationsAvailable`, input/output locations,
  resolver access, and any unavailable capability.

Use the existing [preflight schema](setup-usd-performance-tuning/scripts/setup-preflight.schema.json)
and [probe schema](setup-usd-performance-tuning/scripts/probe-snapshot.schema.json).
Package changelog/version strings may be fallback evidence; do not represent them as
binding-reported semantic versions. Do not mix unrelated USD Python and native library builds.

When a requested optimizer or operation is unavailable, record
`blocked_missing_usd_optimize` or `blocked_missing_usd_optimize_operation` and the observed
failure. Use [installation](install-standalone.md) when needed and authorized. Continue
independent read-only analysis when useful, without claiming the blocked work ran.
Use [remote access](remote-assets.md) before opening a remote stage.

## Workspace and artifacts

Use the user's selected `output_path`. If none was supplied, choose a fresh, collision-free
directory beside the input when writable and state it; ask only when no suitable authorized
location is available. Generated assets belong outside the installed skill directory.
Keep the source and inherited dependencies read-only. Record resolved source/dependency
fingerprints before work and verify them at completion.

Write runtime evidence to `setup-preflight.json`; retain scripts, configurations, profiles,
manifests, logs and outputs beneath the run workspace. Give batch target filenames a stable
identifier derived from the full source identity, not just a basename. Record changed-content
fingerprints separately; a path hash is a naming aid, not a valid cache key.

Keep large runtime artifacts on disk. Read compact structured artifacts first, then bounded
log excerpts around specific failures. Do not use live log streaming by default or paste
complete validator rows, binary USD data, or unbounded scene dumps into model context.
Record full evidence paths in the report. See [verification](verification.md) for measurement
and completion, and [operations](operations.md) for execution.
