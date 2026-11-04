# Version-sensitive diagnostic priors

Use the CLI 5.0.0 evidence in the shared CLI contract. Treat these as
hypotheses and recheck local help, metadata, source, and exact reproduction.

## CLI environment and discovery

- `holoscan` resolves from `PATH`. A stale shell, a `uv tool` installation, or
  another virtual environment can run a different CLI than the committed pin;
  compare `version --json` and `env-info --json` (`cli.environment` and
  `cli.environment_prefix`) with `requirements.txt` or the module's pins.
- Inside a container, the CLI comes from the image. holoscan-applications'
  `Dockerfile` installs `requirements.txt` when the image is built, so an image
  reused with `--no-docker-build` after a pin change keeps the old CLI and test
  driver. Compare outer and container `env-info --json`.
- The root is discovered by walking up from the current directory; in
  holoscan-applications it is the directory containing
  `applications/<name>/metadata.json`. Compare with an explicit global
  `--project-root` or `HOLOSCAN_CLI_ROOT`, and read the root source in a dry
  run.
- Never prefix `holoscan` with `sudo`; it can create root-owned environments
  and artifacts. Use `--as-root` only for an approved operation that requires
  it.
- The `5.0.0` wheel declares `>=3.11,<3.15`; the older published `5.0.0a1`
  wheel declares `<3.14`. Recommend 3.12–3.14; preserve a requested alpha pin
  by selecting 3.12 or 3.13 only for that environment. Inspect package
  metadata before diagnosing a 3.14 failure; do not reject 3.14 for all CLI 5.x
  builds or bypass metadata. See "Python and CLI version" in the Holoscan CLI
  configuration reference.

## Base image and SDK selection

- `No default Holoscan SDK base image is configured` means no `--base-img`,
  `HOLOSCAN_CLI_BASE_IMAGE`, or configured base image. holoscan-applications'
  `Dockerfile` deliberately has no default; select a development image that
  matches the application's SDK, architecture, and dependencies rather than
  adding a default to shared files.
- Missing registry authentication can look like a generic pull error. Identify
  the registry boundary without requesting or printing credentials.
- Without `HOLOSCAN_SDK_ROOT` or `--local-sdk-root`, a `--local` command can
  select an SDK older than the project's minimum: the CLI does not compare
  versions. `Could not find a configuration file for package "holoscan" that
  is compatible with requested version "5.0"`, listing an older
  `holoscan-config.cmake`, is that mismatch. Select a compatible SDK root or
  the container path; do not lower the application's version requirement or
  change the host SDK.
- SDK 5 Debian packages do not include Python bindings, and SDK 5 images and
  builds use CUDA 13 only.
- The CLI reads `[tool.holoscan]` only at a standalone Module root. A table
  added at a holoscan-applications root has no effect; use the command option
  or its environment variable instead.

## Syntax and metadata

- Local help is the accepted-syntax authority. Older READMEs, tutorials, and
  templates can contain launcher commands or retired flags; translate them to
  the local `holoscan` help.
- `holoscan create` without `--template` uses the packaged Module template.
  Inside holoscan-applications it plans a `holoscan-<name>/` Module repository
  rather than an application; treat that as a workflow error, not a template
  bug. Creating from an application template fails with `cookiecutter is
  required` until the CLI's `create` extra is installed.
- When an application template generates into `applications/`, CLI 5.0.0
  appends a registration line in another format to
  `applications/CMakeLists.txt` if that file exists. holoscan-applications
  does not define its macro, so configure fails with an unknown-command error
  until that line is replaced with `add_holoscan_application(<name>)` where
  `CONTRIBUTING.md` says. Without that file, the creation preview still lists
  the edit, but the real creation skips it.
- `create --dryrun` rejects `--language python` alone but skips the packaged
  Module template's own checks. In 5.0.0, only the real creation rejects SDK
  minimums below 5.0.0, including a configured base SDK version that an
  explicit `--context holoscan_version=<version>` did not override.
- `HOLOSCAN_CLI_BUILD_LOCAL=false` no longer selects local execution; 5.0.0
  parses common true and false spellings as booleans.
  `HOLOSCAN_CLI_ALWAYS_BUILD=false` makes `run` skip image and project builds,
  so it can execute a stale image or executable, and makes other commands skip
  the image build. The repository CI driver forces it on.
- Pass dash-leading app arguments with equals, for example
  `--run-args="--flag value"`.
- Metadata commands are argv, not shell snippets. Only normal
  `run-container ... --` has shell semantics; a custom entrypoint receives
  argv.
- `--build-with` replaces the mode operator list. `--configure-args` and
  Docker argument layers append; inspect the effective preview and preserve
  required devices, mounts, dependencies, and environment. Dry runs hide
  `--docker-opts` and CMake option values.
- An application defines top-level `run` or `modes`, not both. Multiple modes
  require a default. `.github/scripts/validate_metadata.py` checks metadata
  against the installed CLI's schemas.
- Holoscan CLI 5.x does not expose the `workflow` project type.
- `holoscan build --benchmark` runs SDK 4 patch scripts from
  `benchmarks/holoscan_flow_benchmarking/` under the project root;
  holoscan-applications has none, so the command fails before building. SDK 5
  flow benchmarking uses explicit instrumentation from Holoscan Benchmarks.

## Tests and repository checks

- `status --json` build markers do not prove compilation or tests. Host changes
  do not update an already-built project image.
- `holoscan test` takes no mode and does not inherit mode CMake options. Pass
  them as one semicolon-separated `--cmake-options` value; a test run without
  them is configured differently from that mode's build.
- In container mode, `test` resolves its CTest driver inside the image unless
  `HOLOSCAN_CLI_CTEST_SCRIPT` names another. A missing or stale image-side
  driver is distinct from an application test failure.
- In 5.0.0, the CLI that composes the test command detects `xvfb-run`, warns
  and falls back to direct `ctest` when it is absent, and passes the active
  project root as `CTEST_SOURCE_DIRECTORY`; in container mode the host CLI
  composes it and only the CTest script is resolved inside the image. An
  unconditional `xvfb-run` means that CLI predates 5.0.0, and a source
  directory under site-packages points to a stale image-side
  `container.ctest`. Diagnose that boundary before changing dependencies.
- Every test run of the bundled driver empties and reuses `build-<project>`
  (`build-` for repository-wide tests). It scopes applications, operators,
  packages, and extensions, but not Modules, so a Module's `test` runs the
  repository-wide suite in `build-`.
- `--ctest-options=--no-tests=error` turns an empty selection into a failure.
  Hosted CI runs SDK-free tests with both language scopes disabled: a test
  registered inside a language scope never runs there, and a test outside it
  must not need the SDK, a GPU, or external data.
- `python .github/scripts/holoscan_ci.py` stops at the first failing lifecycle
  command and rejects invalid selectors, unsupported languages, and ambiguous
  or conflicting duplicate application records. It dry-runs `run <app> ci` only
  for applications with a `ci` mode and `test` for every application. With
  `--execute` it builds and runs instead, so every application's SDK and
  hardware prerequisites must be present.
- When a project declares both C++ and Python, an omitted `--language`
  selects Python and prints a notice on stderr. For a Module, 5.0.0 `build`
  and `test` also pass `HOLOSCAN_<MODULE>_BUILD_CPP` and
  `HOLOSCAN_<MODULE>_BUILD_PYTHON`, so `--language cpp` skips the Module's
  Python bindings.

## Lint

- `holoscan lint` runs pre-commit on all files from the root, and a path
  argument limits it to tracked and unignored files under that path. Hook
  environments are fetched on first use, so a blocked network fails before any
  check runs.
- holoscan-applications' metadata hook runs `python3` from `PATH` and needs
  `holoscan_cli` and `jsonschema`; outside the activated `requirements.txt`
  environment it fails with an import error.
- The copyright hook requires an Apache-2.0 SPDX identifier and the current
  UTC year in at least one notice of each changed file, comparing against
  `COPYRIGHT_BASE_REF` when it is set. It reports errors without rewriting
  notices.

## Containers and output

- Container commands mount the checkout at the container path that the
  preview's `-v` option shows, and each `--add-volume` path at
  `/workspace/volumes/<basename>`. Host paths passed
  through `--run-args` or `--cmake-options` do not exist in the container.
- `--forward-env` forwards a value, not the file or directory it names. Check
  both the container path and its mount when diagnosing overrides.
- Display forwarding is auto-detected; `--ssh-x11` and `--enable-x11` are
  deprecated. Exit zero does not prove correct pixels or recordings.
- Root operations can leave foreign-owned build, data, engine, or output files.
  Check ownership before editing source.

## Module install, package, and consumers

- `install --dev` changes the selected Python environment immediately; in a
  standalone Module, 5.0.0 runs it in the development container unless
  `--local` is passed, so a host consumer does not see it. Verify and
  uninstall with that same environment and exact build directory. It installs
  only hooks that a build staged as `holoscan_<snake_name>_dev.py`; the 5.0.0
  Module template stages none, so its preview can announce an install that the
  real action rejects.
- `package <module>` honors the explicit module in 5.0.0 and a real action
  fails if that module produces no CPack configuration. `WHEEL` needs a
  `pyproject.toml` that builds a wheel and the Python `build` package.
- In holoscan-applications, `application.dependencies.modules[]` does not
  provide a Module by itself: the CLI writes
  `external_operators_manifest.cmake`, but the repository's root CMake does
  not include it, and `HOLOSCAN_CLI_LOCAL_<NAME>` overrides only change that
  manifest. A missing module package or target there points to the consumption
  path, not to the module source.
- The CLI parses `application.dependencies.modules[]` on every build and run,
  including dry runs. An entry without `source.git_url` and a full 40-character
  `source.ref` fails with `missing source.git_url or source.ref`, although
  metadata validation passes because the schema does not check these entries.
  Hosted CI's `ci` dry run then fails too.
- A tree built in the development container records container paths, so run
  `cmake --install` for it inside that container rather than on the host.

## Debian consumer prerequisites

An SDK import or SDK container does not prove that `dpkg` knows a package named
`holoscan`. The 5.0.0 module template declares
`holoscan (>= <minimum-sdk-version>)`, the Debian alias of `holoscan-cuda-13`
on SDK 5.0 platforms. An SDK container can provide working libraries without
registering that package, as the SDK 4.6.0 CUDA 13 container did. For an
unconfigured module package, compare `dpkg-deb --info` with `dpkg-query` in
that consumer before changing producer code. Use the real declared SDK package
or report the format unverified; do not force dependencies or fabricate a
provider package.

## Benchmark restoration

- Flow-tracking instrumentation is a source change. A failed or killed run can
  leave tracing enabled or edits half-applied; compare with the recorded
  baseline.
- A clean source diff does not remove instrumented binaries or cached CMake
  flags. Rebuild normally and rerun a finite smoke mode.
- The flow runner refuses existing trial artifacts and fails on nonzero exits,
  timeouts, leftover processes, and empty logs, but it does not verify that the
  application honored the requested message count.
- Analyzer trimming can leave no retained samples for short runs. Record the
  retained count and every filter rule.
- Runner and analyzer short options differ; inspect both parsers.

## Cache cleanup

- `clear-cache` has destructive-boundary guards, but clearing persisted
  artifacts remains destructive. It acts on every project under the root.
  Without `--build`, `--data`, or `--install` it removes all three, including
  prepared datasets. `test --clear-cache` clears build and install folders.
- Use the narrowest preview, review every path, and obtain approval before
  removing build, data, or install trees.
- A proved stale build from another image, branch, SDK, or user can justify
  `clear-cache --build`; it is not a first-line diagnostic.
