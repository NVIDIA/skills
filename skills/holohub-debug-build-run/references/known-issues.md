# Version-sensitive diagnostic priors

Verified 2026-07-31 at the official HoloHub tag `holoscan-sdk-4.5.0`
(`0a2f81ef978ccd83a676b1c3189cf5b201315a2b`) and rechecked 2026-09-02
against the published `holoscan-cli==4.6.0` command surface. Treat these as
hypotheses and recheck local help, metadata, source, and exact reproduction.

## Launcher and command identity

- The wrapper selects a Python command environment before parsing the verb.
  An active virtual environment can take priority; otherwise it may select or
  repair a wrapper-managed or container environment. Inspect
  `env-info --json`.
- Even help and dry-run invocations can bootstrap that environment.
- Root image provisioning and normal runtime can select different command
  environments. Compare outer and container `env-info --json`; rebuild an
  image whose committed command contract is stale.
- Never run `sudo ./holohub`; it can create foreign-owned environments and
  artifacts. Use `--as-root` only for an approved operation that requires it.

## Syntax and metadata

- Local help is the accepted-syntax authority. Generated templates, tutorials,
  and project READMEs can contain retired flags.
- In holoscan-cli 4.5.0 or newer, build, package, and sccache-enabled container
  dry runs do not create CLI-owned state. Wrapper setup, prompts, and other
  previews can still have side effects.
- `HOLOSCAN_CLI_BUILD_LOCAL=false` no longer selects local execution; 4.5.0
  parses common true and false spellings as booleans.
- `create` can update parent application registration. Review its preview and
  obtain any repository-required approval before running it.
- Current container setup uses repeatable `--extra-scripts`; older benchmark
  examples may use the invalid singular form.
- Pass dash-leading app arguments with equals, for example
  `--run-args="--flag value"`.
- Metadata commands are argv, not shell snippets. Only normal
  `run-container ... --` has shell semantics; a custom entrypoint receives
  argv.
- CLI Docker/build/configure overrides can replace mode values. Inspect and
  preserve required devices, mounts, dependencies, and environment.
- An application defines top-level `run` or `modes`, not both. Multiple modes
  require a default.
- Holoscan CLI 4.5.0 removes the `workflow` project type.

## Environment, tests, and output

- `status --json` build markers do not prove compilation or tests. Host changes
  do not update an already-built project image.
- `./holohub test` resolves its driver inside the image. A missing or stale
  image-side test script is distinct from an application test failure.
- In 4.5.0 or newer, an in-container test honors
  `HOLOSCAN_CLI_CTEST_SCRIPT` before importing an installed CLI package.
- In 4.6.0, the generated test command detects `xvfb-run`, warns and falls
  back to direct `ctest` when it is absent, and passes the active project root
  as `CTEST_SOURCE_DIRECTORY`. An unconditional `xvfb-run` call or a source
  directory under site-packages indicates stale or overridden image-side
  command behavior; `--no-xvfb` is an explicit bypass, not the first fix.
- The tested CTest driver recognizes APP/OP/PKG/EXT but not MODULE, so
  `test <module>` falls through to broader testing. Test declared operators and
  demos directly.
- `install --dev` changes the wrapper-selected Python environment immediately.
  Verify and uninstall with that same environment and exact build directory.
- `package <module>` honors the explicit module in 4.5.0 or newer and a real
  action fails if that module produces no CPack configuration.
- Host local-module overrides do not automatically reach a project container;
  set the override to the mounted container path in one workflow.
- Missing registry authentication can look like a generic pull error. Identify
  the registry boundary without requesting or printing credentials.
- Display forwarding is auto-detected. Deprecated display flags and exit zero
  do not prove correct pixels or recordings.
- Root operations can leave foreign-owned build, data, engine, or output files.
  Check ownership before editing source.

## Benchmark restoration

- Instrumented builds and the Python runner can patch source. Failed or killed
  runs can leave patched files or `*.bak` backups.
- A clean source diff does not remove instrumented binaries or cached CMake
  flags. Search for backups, rebuild normally, and rerun a finite smoke mode.
- Analyzer trimming can leave no retained samples for short runs. Record the
  retained count and every filter rule.
- Runner and analyzer short options differ; inspect both parsers in the
  benchmark container.

## Cache cleanup

- `clear-cache` has destructive-boundary guards, but clearing persisted
  artifacts remains destructive.
- Use the narrowest preview, review every path, and obtain approval. Repository
  guidance forbids removing build, data, or install trees without asking.
- A proved stale build from another image, branch, SDK, or user can justify
  `clear-cache --build`; it is not a first-line diagnostic.
