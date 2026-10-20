# Holoscan CLI configuration

Use this reference for CLI 5.x configuration in a holoscan-applications
checkout or a standalone Holoscan Module repository. Inspect the selected
CLI's help, `version --json`, and `env-info --json` before applying
version-sensitive options. Preview mutating commands with the flags supported
by that command, then keep the same effect-bearing arguments for the real
invocation.

## Documentation and runtime authority

Use the CLI-owned [quick start](https://github.com/nvidia-holoscan/holoscan-cli/blob/main/README.md),
[command reference](https://github.com/nvidia-holoscan/holoscan-cli/blob/main/CLI_REFERENCE.md),
and [configuration guide](https://github.com/nvidia-holoscan/holoscan-cli/blob/main/CONFIGURATION.md).
Match version-sensitive details to the selected build's help and installed
package source. The CLI repository's `AGENTS.md` and `CONTRIBUTING.md` guide
contributions to that repository. holoscan-applications documents its
repository workflow in `README.md`, `CONTRIBUTING.md`, and `docs/ci.md`;
generated modules follow their own `README.md`, `DEVELOPER.md`, and
`AGENTS.md`.

## Python and CLI version

Recommend Python 3.12, 3.13, or 3.14. The `5.0.0` wheel declares
`>=3.11,<3.15` and is published on PyPI and NVIDIA's package index. The older
published `5.0.0a1` wheel declares `<3.14`; a newer release does not change
an existing wheel or a project's CLI pin. To keep an explicitly requested
older pin, select a compatible interpreter for that invocation. To move a
project to a newer CLI, update its pins together with authorization: the
`requirements.txt` pin in holoscan-applications, or a generated module's
requirements, development-group, and pre-commit pins. Python bindings in a
generated module need Python 3.12 or later. Do not use
`--ignore-requires-python` or add a project-wide Python pin by default. CLI
interpreter support does not establish SDK, CUDA, native extension, or module
compatibility; validate those independently.

In holoscan-applications, install `requirements.txt` into the activated
virtual environment that runs the CLI. It supplies `holoscan-cli`,
`pre-commit`, and `jsonschema`, and the metadata lint hook runs `python3`
from that environment. `holoscan create --template` also needs the CLI's
`create` extra (cookiecutter). `.github/requirements-ci.txt` adds the CMake
and Ninja used by hosted tests. A `uv tool` installation of the CLI has no pip
and none of these lint dependencies.

## Project configuration

- The CLI discovers the project root from nested directories: a
  holoscan-applications root contains `applications/<name>/metadata.json`, and
  a module root contains the module's `metadata.json`. For an explicit root,
  put `--project-root /path/to/root` before the subcommand;
  `HOLOSCAN_CLI_ROOT` is the environment form. Dry runs report the root's
  source.
- Read identity, SDK range, Dockerfile, and dependencies from metadata.
- A standalone Module keeps shared settings in `[tool.holoscan]` in its
  `pyproject.toml`: `cuda`, `ctest-script`, `forward-env`,
  `docker-build-args`, `docker-run-args`, `search-paths`, architecture-specific
  `base-images`, and the naming keys `container-prefix`, `repo-prefix`, and
  `workspace-name`. `search-paths` replaces the default project folders, and
  the CLI rejects unknown keys. The CLI reads this table only at a Module root
  and ignores it at a holoscan-applications root; there, use command options
  or their environment forms (`--base-img` or `HOLOSCAN_CLI_BASE_IMAGE`,
  `--cuda` or `HOLOSCAN_CLI_DEFAULT_CUDA_VERSION`, `--ctest-script` or
  `HOLOSCAN_CLI_CTEST_SCRIPT`).
- CUDA and CTest precedence is command, environment, Module project
  configuration, then defaults.
  Docker argument layers append in project, mode, environment, command order.
  `--build-with` replaces the mode operator list; `--configure-args` is
  repeatable and additive. Inspect `--dryrun --verbose` provenance.
- `--forward-env NAME` forwards a variable by name; it does not mount files
  or translate host paths. Store names, never secret values, in project config.
- CPU affinity/cgroup limits and supported SIPL assets are handled by the CLI;
  do not copy those mechanisms into a launcher.

## Base image and SDK selection

- Lifecycle commands use the development container unless `--local` is
  passed. SDK 5.x images and builds use CUDA 13 only.
- The container base image comes from `--base-img`, then
  `HOLOSCAN_CLI_BASE_IMAGE`, then, for a Module, `[tool.holoscan]`
  `base-images`. holoscan-applications sets none, and its `Dockerfile`
  intentionally has no default `BASE_IMAGE`, so a container command without
  one stops before building. Choose a development image that matches the application's
  SDK, architecture, and dependencies; the CLI does not check that match.
- Select a native SDK with `--local-sdk-root` or `HOLOSCAN_SDK_ROOT`; 4.x
  CUDA-qualified and 5.x architecture-only layouts are recognized. Otherwise
  a standalone Module first tries a `holoscan-sdk` build inside or beside the
  module, and every project then falls back to `HOLOSCAN_CLI_DEFAULT_HSDK_DIR`,
  an SDK under `/usr` (the SDK 5 Debian layout), then `/opt/nvidia/holoscan`.
  Dry runs name the selected SDK and its source. The CLI does not compare that
  SDK with the project's minimum version, so an SDK 4 installation at
  `/opt/nvidia/holoscan` fails at CMake configure for an SDK 5 project. SDK 5
  Debian packages do not include Python bindings. Do not change host SDK
  installations merely to repair discovery.
- Keep native and container build parents separate for build/package. Every
  `holoscan test` run of the bundled CTest driver empties and reuses its own
  test build directory, which the CLI build-parent override does not relocate.
  Inspect the selected driver and its cleanup scope before testing, especially
  when reusing a checkout.

## Module SDK selection

Choose the module's agreed SDK minimum explicitly with
`--context holoscan_version=<version>`; examples use 5.0.0. The packaged
template (`holoscan_cli/templates/module/cookiecutter.json` in the installed
CLI) defaults to 5.0.0 and rejects older minimums during generation; a
`create --dryrun` preview skips the template's checks (SDK minimum, reserved
slugs, naming, and license), which only the real creation enforces. A
configured base SDK
version overrides the default, then explicit context values take precedence,
as implemented by the installed `holoscan_cli/commands/create.py`. Use a CLI
4.x release to scaffold a module for the SDK 4.x API. Neither the default nor
the CLI version establishes the module's SDK contract.

The template implements operators in C++. `--language cpp` is the default;
`--language cpp python` adds Python bindings for those operators. Both the
preview and the real creation reject `--language python` alone, and the
template does not generate Python-implemented operators, editable-import
hooks, or wheels.
