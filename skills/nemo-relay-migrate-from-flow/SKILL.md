---
name: nemo-relay-migrate-from-flow
description: Use this skill when migrating applications, examples, integrations, documentation, manifests, or repository code from NeMo Flow to NeMo Relay across Python, Rust, Node.js, Go, C FFI, CLI, configuration, and observability surfaces.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation and Affiliates
---

# Migrate From NeMo Flow To NeMo Relay

Use this skill when a user has existing NeMo Flow code or documentation and
wants it converted to NeMo Relay. Treat the migration as a mechanical rename
plus language-specific validation, not a behavior rewrite.
Keep compatibility exceptions explicit before applying broad renames.

## Default Workflow

1. Inspect the working tree and identify touched surfaces: Rust, Python,
   Node.js, Go, C FFI, CLI/config, docs, or integrations.
2. Resolve `SKILL_DIR` to the absolute directory containing this `SKILL.md` and
   `TARGET_PATH` to the source repository or target project. Run the bundled
   helper in dry-run mode before editing:
   `python3 "$SKILL_DIR/scripts/migrate_from_nemo_flow.py" "$TARGET_PATH" --rename-paths`
3. Review the reported text edits and path renames with the user. Obtain explicit
   confirmation for the resolved target root, then rerun with `--write`,
   `--rename-paths`, and `--confirm-root "$TARGET_PATH"`.
4. Apply language-specific cleanup for package manager lockfiles, generated
   artifacts, and public API examples.
5. Search for remaining Flow names and verify the affected language surfaces.

## Mechanical Rename Map

- Brand and repository: `NeMo Flow` -> `NeMo Relay`,
  `NeMo-Flow` -> `NeMo-Relay`
- Python: `nemo-flow` -> `nemo-relay`, `nemo_flow` -> `nemo_relay`,
  `python/nemo_flow` -> `python/nemo_relay`
- Rust: `nemo-flow` -> `nemo-relay`, `nemo-flow-adaptive` ->
  `nemo-relay-adaptive`, `nemo_flow::` -> `nemo_relay::`
- Node.js: `nemo-flow-node` -> `nemo-relay-node`, including related entry
  points such as `/typed`, `/plugin`, `/adaptive`, and `/observability`
- Go: `github.com/NVIDIA/NeMo-Flow/go/nemo_flow` ->
  `github.com/NVIDIA/NeMo-Relay/go/nemo_relay`, package aliases
  `nemo_flow` -> `nemo_relay`, and source directories `go/nemo_flow` ->
  `go/nemo_relay`
- C FFI: `nemo_flow.h` -> `nemo_relay.h`, `nemo_flow_*` ->
  `nemo_relay_*`, `NemoFlow*` -> `NemoRelay*`, and `NEMO_FLOW_*` ->
  `NEMO_RELAY_*`
- CLI/config: `nemo-flow` -> `nemo-relay`, `.nemo-flow` -> `.nemo-relay`,
  `~/.config/nemo-flow` -> `~/.config/nemo-relay`, `NEMO_FLOW_*` ->
  `NEMO_RELAY_*`, and `x-nemo-flow-*` -> `x-nemo-relay-*`

Do not replace bare `flow`, `Flow`, or `FlowError`. Those can be domain words
or intentional compatibility names.

## Language Cleanup

- **Python**: update `pyproject.toml`, imports, type stubs, integration
  package paths, extras, and native module names. Regenerate or refresh lockfiles
  with the user's package workflow after source edits.
- **Rust**: update `Cargo.toml` crate names, workspace dependencies, package
  references, and `use nemo_relay::...` imports. Let Cargo regenerate
  `Cargo.lock` when dependencies changed.
- **Node.js**: update `package.json`, workspace names, package-lock entries,
  native addon artifact names, and imports from `nemo-relay-node`. Run the
  package manager to refresh locks.
- **Go**: update `go.mod`, import paths, package declarations, aliases, and any
  local directory layout under `go/nemo_relay`.
- **C FFI**: update header includes, exported symbol names, status and callback
  type names, macro constants, loader paths, and downstream bindings.
- **Docs and examples**: update badges, package install commands, repository
  links, hosted docs URLs, CLI commands, config paths, and integration names.

## Automation Helper

Use `$SKILL_DIR/scripts/migrate_from_nemo_flow.py` for first-pass edits. The
helper:

- runs as a dry run unless `--write` is passed
- skips common vendor, build, cache, and generated directories
- skips lockfiles unless `--include-lockfiles` is passed
- skips symbolic links and credential-bearing dotenv files
- requires the reviewed target root to be repeated with `--confirm-root` before
  writing, and refuses filesystem-root or home-directory writes
- anchors writes and renames to verified directory handles without following
  symbolic links, and refuses write mode on platforms that cannot provide those
  guarantees
- uses atomic no-replace path renames and exits nonzero when any requested
  mutation fails
- can report or perform path renames with `--rename-paths`
- rewrites only explicit NeMo Flow identifiers, package names, repository names,
  config paths, headers, environment variables, and FFI type prefixes

The helper does not classify arbitrary JSON, YAML, TOML, or INI files as secret.
Review every configuration file in the dry-run report. If any reported file is
unreviewed or credential-bearing, do not use `--write` on that root. Apply the
reviewed changes manually and leave secret-bearing files untouched without
reading or displaying their values.

Set shell-safe absolute paths before invoking the helper. Replace the example
values with the resolved skill directory and either the source repository or the
user's target project:

```bash
SKILL_DIR="/resolved/absolute/path/to/nemo-relay-migrate-from-flow"
TARGET_PATH="/resolved/absolute/path/to/target-project"

python3 "$SKILL_DIR/scripts/migrate_from_nemo_flow.py" "$TARGET_PATH" --rename-paths
python3 "$SKILL_DIR/scripts/migrate_from_nemo_flow.py" "$TARGET_PATH" \
  --write --rename-paths --confirm-root "$TARGET_PATH"
```

Use `--include-lockfiles` only when the user wants lockfiles edited directly;
otherwise regenerate them with Cargo, uv/pip, npm, or Go tooling.

## Verification

- Search for remaining explicit Flow identifiers:
  `rg -n "NeMo Flow|NeMo-Flow|nemo_flow|nemo-flow|NEMO_FLOW|NemoFlow|nemo_flow\\.h|nemo_flow_"`
- Run targeted tests for every affected language surface.
- For Rust changes, run `cargo test` or the repository's Rust test recipe.
- For Python changes, run the relevant import check and tests in the target
  environment.
- For Node.js changes, run package install, type checks, and package tests.
- For Go changes, run `go test ./...` from the updated module.
- For docs-only migrations, build or link-check docs if the site navigation,
  install commands, or API references changed.

## Related Skills

- `nemo-relay-get-started`
- `nemo-relay-instrument-calls`
- `nemo-relay-debug-runtime-integration`
